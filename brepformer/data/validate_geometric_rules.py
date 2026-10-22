"""Measure whether geometric_rules.py's denials are trustworthy, using mftrcad ground truth.

Why this exists: on the pseudo-labelled corpus the rules rewrite roughly a third of
all faces (67/218, 188/596, 232/626 on the first three models). A rewrite that large
dominates the label quality of the whole corpus, and it was never checked against a
known-correct label anywhere. If the rules are right they are the cheapest accuracy
win available; if they are wrong they are actively poisoning the training set, and
they would do it invisibly because pseudo-labels have nothing to disagree with.

mftrcad has hand-authored per-face labels, so it can answer this directly.

The test: feed the *ground-truth* labels in as if they were predictions. Every face is
correct by construction, so any DENY the rules emit is a false denial, and its rate is
the rate at which the rules corrupt correct labels. CONFIRM/UNCERTAIN on a correct
label costs nothing.

Reported per class, because a rule that is fine on stock and catastrophic on chamfer
would look acceptable in aggregate while destroying exactly the rare classes this
corpus was built to rescue.

Usage:
    python -m brepformer.data.validate_geometric_rules --limit 60
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from brepformer.data.classes import CLASS_NAMES
from brepformer.geometric_rules import (
    apply_geometric_rules, VERDICT_DENY, VERDICT_CONFIRM, VERDICT_UNCERTAIN,
)


def load_gt_face_labels(label_path: Path, num_faces_hint: int | None = None):
    """Read per-face ground-truth class ids from an mftrcad *_result.json."""
    data = json.loads(label_path.read_text())
    if isinstance(data, list) and len(data) > 1:
        data = data[1]
    if not isinstance(data, dict):
        return None
    if "labels" in data and isinstance(data["labels"], list):
        return [int(x) for x in data["labels"]]
    if "cls" in data:
        cls = {int(k): int(v) for k, v in data["cls"].items()}
        n = (num_faces_hint or 0) or (max(cls) + 1 if cls else 0)
        return [cls.get(i, -1) for i in range(n)]
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", type=Path, default=Path("brepformer/data/mftrcad"))
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label_space", default="27", choices=["27", "5"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    steps_dir = args.data_dir / "steps"
    labels_dir = args.data_dir / "labels"

    steps = sorted(p for p in steps_dir.iterdir()
                   if p.suffix.lower() in {".step", ".stp"})
    random.Random(args.seed).shuffle(steps)

    # verdict counts per true class, plus what the rules wanted to change it to
    per_class = collections.defaultdict(collections.Counter)
    confusion = collections.Counter()
    # Which rule caused each false denial, per true class. An aggregate corruption rate
    # says a component is harmful; this says which of its rules to change.
    deny_rule = collections.defaultdict(collections.Counter)
    n_models = 0
    n_faces = 0
    skipped = 0

    for step_path in steps:
        if n_models >= args.limit:
            break

        stem = step_path.stem
        label_path = labels_dir / f"{stem}_result.json"
        if not label_path.exists():
            hits = list(labels_dir.glob(f"{stem}*.json"))
            hits = [h for h in hits if "_rel" not in h.name]
            if not hits:
                skipped += 1
                continue
            label_path = hits[0]

        try:
            gt = load_gt_face_labels(label_path)
            if not gt:
                skipped += 1
                continue
            verdicts = apply_geometric_rules(str(step_path), gt, args.label_space)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {stem}: {e}", file=sys.stderr)
            skipped += 1
            continue

        if len(verdicts) != len(gt):
            # Face ordering/count mismatch would make every comparison meaningless.
            skipped += 1
            continue

        for v in verdicts:
            i = v["face_idx"]
            true_c = gt[i]
            if true_c < 0:
                continue
            per_class[true_c][v["verdict"]] += 1
            n_faces += 1
            if v["verdict"] == VERDICT_DENY:
                confusion[(true_c, v.get("suggested"))] += 1
                deny_rule[true_c][v.get("rule") or "?"] += 1

        n_models += 1

    print(f"\nvalidated {n_models} models / {n_faces} labelled faces "
          f"({skipped} skipped)\n")
    if not n_faces:
        raise SystemExit("no faces validated — check paths / face-index alignment")

    tot_deny = sum(c[VERDICT_DENY] for c in per_class.values())
    print(f"OVERALL FALSE-DENIAL RATE: {tot_deny}/{n_faces} = "
          f"{100 * tot_deny / n_faces:.1f}% of correct labels would be overwritten\n")

    print(f"{'class':<26}{'faces':>8}{'deny':>8}{'deny%':>8}{'confirm':>9}{'uncert':>8}")
    rows = sorted(per_class.items(), key=lambda kv: -sum(kv[1].values()))
    for c, cnt in rows:
        tot = sum(cnt.values())
        d = cnt[VERDICT_DENY]
        name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)
        print(f"{name:<26}{tot:>8}{d:>8}{100 * d / tot:>7.1f}%"
              f"{cnt[VERDICT_CONFIRM]:>9}{cnt[VERDICT_UNCERTAIN]:>8}")

    print("\nmost common false rewrites (true -> suggested):")
    for (t, s), n in confusion.most_common(12):
        tn = CLASS_NAMES[t] if t is not None and t < len(CLASS_NAMES) else str(t)
        sn = CLASS_NAMES[s] if s is not None and s < len(CLASS_NAMES) else str(s)
        print(f"  {tn:<24} -> {sn:<24} {n}")

    print("\nfalse denials by rule (which rule to fix, not just that one is broken):")
    by_rule = collections.Counter()
    for c, cnt in deny_rule.items():
        for r, n in cnt.items():
            by_rule[r] += n
    for r, n in by_rule.most_common():
        worst = sorted(
            ((cnt[r], c) for c, cnt in deny_rule.items() if cnt[r]), reverse=True)[:3]
        detail = ", ".join(
            f"{CLASS_NAMES[c] if c < len(CLASS_NAMES) else c} {k}" for k, c in worst)
        print(f"  {r:<24} {n:>5}  ({detail})")

    if args.out:
        args.out.write_text(json.dumps({
            "n_models": n_models, "n_faces": n_faces,
            "false_denial_rate": tot_deny / n_faces,
            "per_class": {str(k): dict(v) for k, v in per_class.items()},
            "confusion": {f"{t}->{s}": n for (t, s), n in confusion.items()},
            "deny_by_rule": {str(k): dict(v) for k, v in deny_rule.items()},
        }, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
