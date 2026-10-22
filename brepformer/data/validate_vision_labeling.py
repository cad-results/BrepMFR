"""Measure whether the Claude vision pass improves or degrades face labels, on ground truth.

The geometric rules looked helpful until they were measured, and turned out to
overwrite 15% of correct mftrcad labels (100% of ``chamfer_2``). The vision pass
deserves exactly the same scepticism before it is run over 853 models: it is far
more expensive, and like the rules it produces confident-looking output that
pseudo-labels cannot contradict.

Two rates decide whether the pass is worth running, and they must be measured
together — either alone is misleading:

*Harm* — feed the **ground-truth** label in as the prediction. Every face is correct
by construction, so any DENY is the vision model corrupting a good label. A pass
that denies correct labels 20% of the time destroys more than it fixes.

*Benefit* — feed a **deliberately wrong** label for the same face. A DENY here is
correct, and the fix counts only if ``correct_class`` is the true class. Denying a
wrong label but replacing it with another wrong label is not an improvement, merely
churn, and reporting DENY-rate alone would hide that.

The pass is worth running only if benefit clearly exceeds harm, and specifically on
the rare classes — a model that is great on stock and blind to chamfer reproduces the
exact failure this corpus was built to fix, so results are broken out per class.

Usage:
    python -m brepformer.data.validate_vision_labeling --limit 25 --faces_per_model 6
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from brepformer.data.classes import (
    CLASS_NAMES, DEFEATURE_CLASS_NAMES, CLASS_TO_DEFEATURE,
)
from brepformer.reclassify_visual import (
    render_face_highlighted, query_claude_vision, _vision_client,
)
from brepformer.data.validate_geometric_rules import load_gt_face_labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", type=Path, default=Path("brepformer/data/mftrcad"))
    ap.add_argument("--limit", type=int, default=25, help="models to sample")
    ap.add_argument("--faces_per_model", type=int, default=6)
    ap.add_argument("--claude_model", default="claude-sonnet-5")
    ap.add_argument("--label_space", default="27", choices=["27", "5"],
                    help="27-class MFTRCAD taxonomy, or the 5-class defeature space. "
                         "Worth testing both: several 27-class distinctions "
                         "(chamfer vs chamfer_2, rectangular_passage vs _2) encode a "
                         "dataset convention rather than visible geometry, so no "
                         "amount of image quality can recover them, and measuring only "
                         "27-class would understate what vision can do at the "
                         "granularity that defeaturing actually needs.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rare_bias", action="store_true", default=True,
                    help="Prefer faces of rare classes when sampling, since aggregate "
                         "accuracy is dominated by stock and tells us little.")
    ap.add_argument("--out", type=Path, default=Path("/tmp/vision_val.json"))
    args = ap.parse_args()

    client, resolved = _vision_client(args.claude_model)
    print(f"vision backend {type(client).__name__} / {resolved}")

    names = CLASS_NAMES if args.label_space == "27" else DEFEATURE_CLASS_NAMES

    steps_dir = args.data_dir / "steps"
    labels_dir = args.data_dir / "labels"
    steps = sorted(p for p in steps_dir.iterdir()
                   if p.suffix.lower() in {".step", ".stp"})
    rng = random.Random(args.seed)
    rng.shuffle(steps)

    # class frequency, to bias sampling toward rare classes
    harm = collections.Counter()          # per class: correct label denied
    harm_tot = collections.Counter()
    benefit_fix = collections.Counter()   # wrong label -> denied AND fixed correctly
    benefit_deny = collections.Counter()  # wrong label -> denied (any suggestion)
    benefit_tot = collections.Counter()
    n_models = 0
    n_calls = 0
    records = []

    for step_path in steps:
        if n_models >= args.limit:
            break
        stem = step_path.stem
        # STEP stems already end in "_result", so the label is "<stem>.json";
        # the "_result.json" form exists for other corpora. Try both, and never
        # the "_rel" relation files.
        label_path = labels_dir / f"{stem}.json"
        if not label_path.exists():
            label_path = labels_dir / f"{stem}_result.json"
        if not label_path.exists():
            continue
        try:
            gt = load_gt_face_labels(label_path)
        except Exception:  # noqa: BLE001
            continue
        if not gt:
            continue
        if args.label_space == "5":
            gt = [CLASS_TO_DEFEATURE[c] if 0 <= c < len(CLASS_TO_DEFEATURE) else -1
                  for c in gt]

        idxs = [i for i, c in enumerate(gt) if c >= 0]
        if not idxs:
            continue
        if args.rare_bias:
            freq = collections.Counter(gt[i] for i in idxs)
            idxs.sort(key=lambda i: freq[gt[i]])
        else:
            rng.shuffle(idxs)
        idxs = idxs[:args.faces_per_model]

        used = False
        for i in idxs:
            true_c = gt[i]
            try:
                images = render_face_highlighted(str(step_path), i)
            except Exception as e:  # noqa: BLE001
                print(f"  render fail {stem}#{i}: {e}", file=sys.stderr)
                continue
            if not images:
                continue
            used = True

            # --- harm: present the true label ---
            harm_tot[true_c] += 1
            r = query_claude_vision(images, i, names[true_c], {},
                                    args.label_space, names, model=args.claude_model)
            n_calls += 1
            denied_correct = bool(r and r.get("verdict") == "DENY"
                                  and (r.get("confidence") or 0) >= 0.7)
            if denied_correct:
                harm[true_c] += 1

            # --- benefit: present a deliberately wrong label ---
            wrong_c = rng.choice([c for c in range(len(names)) if c != true_c])
            benefit_tot[true_c] += 1
            r2 = query_claude_vision(images, i, names[wrong_c], {},
                                     args.label_space, names, model=args.claude_model)
            n_calls += 1
            if r2 and r2.get("verdict") == "DENY" and (r2.get("confidence") or 0) >= 0.7:
                benefit_deny[true_c] += 1
                if r2.get("correct_class") == true_c:
                    benefit_fix[true_c] += 1

            records.append({
                "model": stem, "face": i, "true": true_c,
                "harm_denied": denied_correct,
                "shown_wrong": wrong_c,
                "wrong_denied": bool(r2 and r2.get("verdict") == "DENY"),
                "wrong_fixed_to": (r2 or {}).get("correct_class"),
            })

        if used:
            n_models += 1

    th, tht = sum(harm.values()), sum(harm_tot.values())
    bd, bf, bt = sum(benefit_deny.values()), sum(benefit_fix.values()), sum(benefit_tot.values())
    print(f"\n{n_models} models, {n_calls} vision calls\n")
    if not tht:
        raise SystemExit("no faces evaluated")

    print(f"HARM    correct label wrongly denied: {th}/{tht} = {100*th/tht:.1f}%")
    print(f"BENEFIT wrong label denied:           {bd}/{bt} = {100*bd/bt:.1f}%")
    print(f"BENEFIT wrong label fixed to truth:   {bf}/{bt} = {100*bf/bt:.1f}%")
    print(f"\nnet per reviewed face: {(bf - th)/max(tht,1):+.3f} labels improved\n")

    print(f"{'class':<26}{'n':>5}{'harm%':>8}{'fix%':>8}")
    for c in sorted(harm_tot, key=lambda c: -harm_tot[c]):
        n = harm_tot[c]
        print(f"{names[c]:<26}{n:>5}{100*harm[c]/n:>7.1f}%"
              f"{100*benefit_fix[c]/max(benefit_tot[c],1):>7.1f}%")

    args.out.write_text(json.dumps({
        "n_models": n_models, "n_calls": n_calls,
        "harm_rate": th / tht, "benefit_deny_rate": bd / max(bt, 1),
        "benefit_fix_rate": bf / max(bt, 1),
        "per_class": {names[c]: {
            "n": harm_tot[c], "harm": harm[c],
            "fix": benefit_fix[c], "deny": benefit_deny[c],
        } for c in harm_tot},
        "records": records,
    }, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
