"""Measure Claude vision as a *labeller* (not a verifier) against mftrcad ground truth.

The verify framing measured badly for a structural reason: shown a candidate label
and asked whether it is right, the model denied 91% of wrong labels and 74% of
correct ones, so the verdict was close to constant and carried almost no signal.
This harness removes the anchor — it asks for a class outright and compares to
ground truth.

This also produces the number that actually decides the pipeline: per-face accuracy,
directly comparable to the teacher model's. Vision is only worth spending on faces
where it beats the teacher, and the interesting comparison is on the faces the
teacher is *unsure* about, since those are the ones masked out of training entirely.

Reported per class, and separately for the rare classes, because aggregate accuracy
on this data is dominated by plain stock faces and would look fine even if every
feature were missed.

Usage:
    python -m brepformer.data.validate_vision_classify --limit 25 --label_space 5
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
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--faces_per_model", type=int, default=5)
    ap.add_argument("--claude_model", default="claude-sonnet-5")
    ap.add_argument("--label_space", default="5", choices=["27", "5"])
    ap.add_argument("--max_images", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("/tmp/vision_classify.json"))
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

    correct = collections.Counter()
    total = collections.Counter()
    predicted_as = collections.Counter()
    confusion = collections.Counter()
    n_models = 0
    n_calls = 0
    records = []

    for step_path in steps:
        if n_models >= args.limit:
            break
        stem = step_path.stem
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
        # rare classes first: aggregate accuracy here is dominated by stock
        freq = collections.Counter(gt[i] for i in idxs)
        idxs.sort(key=lambda i: freq[gt[i]])
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

            r = query_claude_vision(images, i, "", {}, args.label_space, names,
                                    model=args.claude_model, mode="classify",
                                    max_images=args.max_images)
            n_calls += 1
            pred = r.get("class") if r else None
            total[true_c] += 1
            if isinstance(pred, int) and 0 <= pred < len(names):
                predicted_as[pred] += 1
                confusion[(true_c, pred)] += 1
                if pred == true_c:
                    correct[true_c] += 1
            records.append({"model": stem, "face": i, "true": true_c,
                            "pred": pred,
                            "conf": (r or {}).get("confidence")})

        if used:
            n_models += 1

    tc, tt = sum(correct.values()), sum(total.values())
    print(f"\n{n_models} models, {n_calls} classify calls\n")
    if not tt:
        raise SystemExit("no faces evaluated")

    print(f"ACCURACY (rare-biased sample): {tc}/{tt} = {100*tc/tt:.1f}%")
    stock = 0 if args.label_space == "5" else 24
    rare_c = sum(v for k, v in correct.items() if k != stock)
    rare_t = sum(v for k, v in total.items() if k != stock)
    if rare_t:
        print(f"ACCURACY excluding stock/random: {rare_c}/{rare_t} = "
              f"{100*rare_c/rare_t:.1f}%")

    print(f"\n{'true class':<26}{'n':>5}{'recall':>9}")
    for c in sorted(total, key=lambda c: -total[c]):
        print(f"{names[c]:<26}{total[c]:>5}{100*correct[c]/total[c]:>8.1f}%")

    print("\nmost common confusions (true -> predicted):")
    for (t, p), n in confusion.most_common(10):
        if t == p:
            continue
        print(f"  {names[t]:<20} -> {names[p]:<20} {n}")

    args.out.write_text(json.dumps({
        "n_models": n_models, "n_calls": n_calls, "label_space": args.label_space,
        "accuracy": tc / tt, "accuracy_excl_stock": (rare_c / rare_t) if rare_t else None,
        "per_class": {names[c]: {"n": total[c], "correct": correct[c]} for c in total},
        "confusion": {f"{names[t]}->{names[p]}": n for (t, p), n in confusion.items()},
        "records": records,
    }, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
