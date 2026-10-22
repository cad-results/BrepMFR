#!/usr/bin/env python3
"""Score the dense multi-pass vision labeller against hand-labelled ground truth.

Nothing from ``vision_label.py`` may rewrite a training label until it has been measured
here, because the earlier, shallower version of the same idea looked plausible and was
worthless: as a verifier it denied 91% of wrong labels *and* 74% of correct ones, and as a
labeller it scored at or below chance on feature faces. Those numbers were only discovered
by running exactly this kind of check.

What this reports, and why each number is needed:

* **Per-class accuracy and confusion**, not just overall accuracy. Overall accuracy is
  dominated by `random` and `cut`, which are ~60% of all faces; a labeller that answers
  "random" for everything scores well on it while adding nothing. The classes that matter
  are `chamfer` (4% of faces) and `fillet`.
* **The teacher's accuracy on the same faces.** The vision pass is only worth running
  where it beats the model that already labels these faces for free. Being right 70% of
  the time is useless if the teacher is right 90% of the time on the same faces.
* **Accuracy as a function of the pass agreement**, since that is what gates whether a
  label gets used. If agreement=1.0 faces are much more accurate than agreement=0.67
  faces, the vote is a usable confidence signal and low-agreement faces can be masked;
  if accuracy is flat in agreement, the vote means nothing and the whole approach has to
  be treated as unreliable regardless of its mean accuracy.

Faces are sampled per class rather than uniformly, so a run of a few hundred API calls
says something about `chamfer` instead of spending 60% of the budget on `random`.

Usage:
    python -m brepformer.data.validate_dense_vision --per_class 12 --passes 3
    python -m brepformer.data.validate_dense_vision --per_class 30 --seg_dir \\
        brepformer/inference/defeature_ft     # also scores the teacher for comparison
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from brepformer.data.classes import CLASS_NAMES, DEFEATURE_CLASS_NAMES
from brepformer.vision_label import label_face


def sample_faces(steps: Path, labels: Path, per_class: int, seed: int,
                 max_models: int, n_classes: int = 5,
                 seg_dir: Optional[Path] = None,
                 teacher_conf_max: Optional[float] = None) -> List[tuple]:
    """(step, face, gt) samples, balanced across classes, spread over many models.

    Balanced because the rare classes are the point; spread over models because 12 chamfer
    faces from one part measure that part, not the class.

    ``teacher_conf_max`` restricts sampling to faces the teacher is *unsure* about, which is
    the population the vision pass is actually deployed on: `data/arbitrate_labels.py` only
    ever asks it about faces confidence masking was going to discard. Sampling uniformly
    instead measures vision against a teacher that is 92% accurate and answers a question
    that never comes up -- these faces are the ones where the teacher drops to ~55%.
    """
    rng = random.Random(seed)
    label_files = sorted(labels.glob("*.json"))
    rng.shuffle(label_files)
    per_class_hits: Dict[int, List[tuple]] = defaultdict(list)
    used_models = 0
    for lp in label_files:
        sp = next((steps / (lp.stem + e) for e in (".step", ".stp", ".STEP")
                   if (steps / (lp.stem + e)).exists()), None)
        if sp is None:
            continue
        try:
            cls = json.loads(lp.read_text())["cls"]
        except (OSError, ValueError, KeyError):
            continue
        if teacher_conf_max is not None:
            if seg_dir is None:
                raise SystemExit("--teacher_conf_max needs --seg_dir")
            tp = seg_dir / f"{lp.stem}.json"
            if not tp.exists():
                continue
            try:
                probs = json.loads(tp.read_text()).get("face_probs") or []
            except (OSError, ValueError):
                continue
            cls = {k: v for k, v in cls.items()
                   if int(k) < len(probs) and probs[int(k)]
                   and max(probs[int(k)]) < teacher_conf_max}
            if not cls:
                continue
        used_models += 1
        by_cls: Dict[int, List[int]] = defaultdict(list)
        for fi, c in cls.items():
            by_cls[int(c)].append(int(fi))
        # At most two faces per class per model, so no single part dominates a class.
        for c, faces in by_cls.items():
            if len(per_class_hits[c]) >= per_class:
                continue
            rng.shuffle(faces)
            for fi in faces[:2]:
                if len(per_class_hits[c]) < per_class:
                    per_class_hits[c].append((sp, fi, c))
        # In the 27-class space several classes are rare enough that no reachable number
        # of models fills their quota, so `max_models` -- not the quota -- is what ends the
        # scan there. Stopping only on a full quota would read the entire corpus.
        if (used_models >= max_models
                or all(len(per_class_hits[c]) >= per_class
                       for c in range(n_classes))):
            break
    out = [t for c in sorted(per_class_hits) for t in per_class_hits[c]]
    rng.shuffle(out)      # interleave classes so a partial run is still balanced
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=Path, default=Path("brepformer/data/defeature/steps"))
    ap.add_argument("--labels", type=Path,
                    default=Path("brepformer/data/defeature/labels"))
    ap.add_argument("--seg_dir", type=Path,
                    help="Teacher inference JSONs, to score the teacher on the same faces")
    ap.add_argument("--label_space", choices=["5", "27"], default="5",
                    help="Label space to validate. 27 needs --steps/--labels pointed at "
                         "mftrcad, whose labels are in the 27-class space.")
    ap.add_argument("--no_challenge", action="store_true",
                    help="Skip the objection/challenge round, to measure what it is worth")
    ap.add_argument("--min_agreement", type=float, default=None)
    ap.add_argument("--min_confidence", type=float, default=None)
    ap.add_argument("--per_class", type=int, default=12)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--max_models", type=int, default=200)
    ap.add_argument("--teacher_conf_max", type=float, default=None,
                    help="Only sample faces where the teacher's top-1 confidence is below "
                         "this (needs --seg_dir). This is the population the vision pass is "
                         "deployed on -- the faces masking discards -- and the only one on "
                         "which beating the teacher is a meaningful claim.")
    ap.add_argument("--sphere_step", type=int, default=15)
    ap.add_argument("--no_requests", action="store_true")
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path("brepformer/data/dense_vision_validation.json"))
    args = ap.parse_args()

    names = CLASS_NAMES if args.label_space == "27" else DEFEATURE_CLASS_NAMES
    kwargs = {}
    if args.min_agreement is not None:
        kwargs["min_agreement"] = args.min_agreement
    if args.min_confidence is not None:
        kwargs["min_confidence"] = args.min_confidence
    samples = sample_faces(args.steps, args.labels, args.per_class, args.seed,
                           args.max_models, n_classes=len(names),
                           seg_dir=args.seg_dir,
                           teacher_conf_max=args.teacher_conf_max)
    print(f"{len(samples)} faces sampled: "
          f"{dict(Counter(names[c] for _, _, c in samples))}", flush=True)

    records: List[Dict] = []
    calls = 0
    t0 = time.time()
    for i, (sp, fi, gt) in enumerate(samples, 1):
        try:
            r = label_face(str(sp), fi, args.label_space, names, args.model,
                           passes=args.passes,
                           allow_requests=not args.no_requests,
                           sphere_step=args.sphere_step,
                           challenge=not args.no_challenge, **kwargs)
        except Exception as e:  # noqa: BLE001 - keep going; one bad model is not a verdict
            print(f"  [{i}/{len(samples)}] {sp.stem} face {fi}: {type(e).__name__}: {e}",
                  flush=True)
            continue
        if not r:
            continue
        calls += r["n_calls"]
        teacher = teacher_conf = None
        if args.seg_dir:
            tp = args.seg_dir / f"{sp.stem}.json"
            if tp.exists():
                try:
                    tj = json.loads(tp.read_text())
                    preds = tj.get("face_preds") or []
                    probs = tj.get("face_probs") or []
                    if fi < len(preds):
                        teacher = int(preds[fi])
                    # The teacher's own confidence, because that -- not the class -- is what
                    # decides whether this face reaches training at all, and therefore which
                    # faces the vision pass is ever asked about. Measured on the defeature
                    # corpus the teacher is 99.2% correct where it reports 0.99+ and 45.5%
                    # where it reports under 0.5, so a comparison averaged over all
                    # confidences answers a question nobody asks.
                    if fi < len(probs) and probs[fi]:
                        teacher_conf = float(max(probs[fi]))
                except (OSError, ValueError):
                    pass
        records.append({"model": sp.stem, "face": fi, "gt": gt,
                        "pred": r["class"], "agreement": r["agreement"],
                        "confidence": r["confidence"], "teacher": teacher,
                        "teacher_conf": teacher_conf,
                        "votes": r["votes"], "masked": r.get("masked"),
                        "mask_reasons": r.get("mask_reasons"),
                        "final_label": r.get("final_label"),
                        "n_objections": len(r.get("objections") or []),
                        "challenge_class": (r.get("challenge") or {}).get("class"),
                        "rarity": r.get("rarity"), "n_passes": r.get("n_passes")})
        ok = "OK " if r["class"] == gt else "X  "
        print(f"  [{i}/{len(samples)}] {ok} {names[gt]:<8} -> {names[r['class']]:<8} "
              f"agree={r['agreement']:.2f}"
              + (" MASKED" if r.get("masked") else "")
              + (f" obj={len(r.get('objections') or [])}"
                 if r.get("objections") else "")
              + f" {sp.stem[:30]} f{fi}", flush=True)

    if not records:
        print("no faces scored")
        return

    n = len(records)
    correct = sum(r["pred"] == r["gt"] for r in records)
    print(f"\n=== {n} faces, {calls} API calls, {(time.time() - t0) / 60:.1f} min ===")
    print(f"overall accuracy: {correct}/{n} = {100 * correct / n:.1f}%")

    print("\nper class (GT):")
    print(f"  {'class':<9} {'n':>4} {'vision':>8} {'teacher':>8}   most common vision error")
    for c, name in enumerate(names):
        rows = [r for r in records if r["gt"] == c]
        if not rows:
            continue
        acc = 100 * sum(r["pred"] == c for r in rows) / len(rows)
        trows = [r for r in rows if r["teacher"] is not None]
        tacc = (f"{100 * sum(r['teacher'] == c for r in trows) / len(trows):7.1f}%"
                if trows else "      —")
        errs = Counter(names[r["pred"]] for r in rows if r["pred"] != c)
        top = ", ".join(f"{k} x{v}" for k, v in errs.most_common(2)) or "—"
        print(f"  {name:<9} {len(rows):>4} {acc:7.1f}% {tacc}   {top}")

    # Accuracy by agreement decides whether the vote can be used to mask: a flat curve
    # means the confidence carries no information and every label is equally suspect.
    print("\naccuracy by pass agreement:")
    for lo, hi, lbl in ((0.99, 1.01, "1.00 (unanimous)"), (0.6, 0.99, "0.67 (2 of 3)"),
                        (0.0, 0.6, "<0.6 (split)")):
        rows = [r for r in records if lo <= r["agreement"] < hi]
        if rows:
            acc = 100 * sum(r["pred"] == r["gt"] for r in rows) / len(rows)
            print(f"  {lbl:<18} n={len(rows):<4} {acc:.1f}%")

    # The masking report. Accuracy on the kept faces is the number that decides whether the
    # output is usable, because masked faces never reach the loss: a labeller at 70% overall
    # that reaches 92% on the 60% of faces it keeps is worth running, and one whose accuracy
    # does not rise when the doubtful faces are dropped is not -- the mask is then just
    # throwing away examples at random.
    kept = [r for r in records if not r.get("masked")]
    dropped = [r for r in records if r.get("masked")]
    if kept:
        kacc = 100 * sum(r["pred"] == r["gt"] for r in kept) / len(kept)
        print(f"\nmasking: {len(kept)}/{n} faces kept ({100 * len(kept) / n:.0f}%), "
              f"accuracy on kept faces {kacc:.1f}% "
              f"(vs {100 * correct / n:.1f}% unmasked)")
        if dropped:
            dacc = 100 * sum(r["pred"] == r["gt"] for r in dropped) / len(dropped)
            print(f"  accuracy on the {len(dropped)} masked faces: {dacc:.1f}% "
                  f"-- the mask is only doing work if this is well below the kept rate")
        print("  per class, kept / sampled and accuracy on kept:")
        for c, name in enumerate(names):
            rows = [r for r in records if r["gt"] == c]
            if not rows:
                continue
            k = [r for r in rows if not r.get("masked")]
            acc = f"{100 * sum(r['pred'] == c for r in k) / len(k):5.1f}%" if k else "    —"
            print(f"    {name:<22} {len(k):>3}/{len(rows):<3} {acc}")
        reasons = Counter(x for r in dropped for x in (r.get("mask_reasons") or []))
        if reasons:
            print("  why faces were masked:")
            for why, cnt in reasons.most_common(6):
                print(f"    x{cnt:<4} {why}")

    # Did the challenge round earn its extra API call? Only the faces where the check
    # objected can answer that, and only by comparing the answer before and after.
    ch = [r for r in records if r.get("n_objections")]
    if ch:
        moved = [r for r in ch if r.get("challenge_class") is not None
                 and r["challenge_class"] != r["pred"]]
        print(f"\nobjections raised on {len(ch)}/{n} faces; the challenge round changed "
              f"the verdict on {len(moved)}")
        cacc = 100 * sum(r["pred"] == r["gt"] for r in ch) / len(ch)
        rest = [r for r in records if not r.get("n_objections")]
        racc = (100 * sum(r["pred"] == r["gt"] for r in rest) / len(rest)) if rest else 0.0
        print(f"  final accuracy on objected faces {cacc:.1f}% vs {racc:.1f}% on the rest "
              f"-- objected faces should be the harder ones")

    # Common versus rare geometry. These two halves have different jobs and one accuracy
    # number hides both: a face whose geometry the corpus has labelled thousands of times must
    # be right, because those faces are the bulk of the loss and a systematic error there
    # teaches the student a wrong rule about the geometry it sees most often. A face the
    # corpus barely covers is allowed to be masked, and *should* be masked when unresolved --
    # what it must not be is confidently wrong.
    strata = [
        ("common (dominant row)", lambda r: (r.get("rarity") or {}).get("dominant_n", 0) > 0),
        ("novel geometry", lambda r: (r.get("rarity") or {}).get("novel")),
        ("rare classes live", lambda r: (r.get("rarity") or {}).get("rare_expected")),
        ("escalated (>3 passes)", lambda r: (r.get("n_passes") or 0) > 3),
    ]
    print("\nby corpus coverage:")
    print(f"  {'stratum':<24} {'n':>4} {'acc':>7} {'kept':>6} {'acc kept':>9} {'wrong kept':>11}")
    for name, pred in strata:
        rows = [r for r in records if pred(r)]
        if not rows:
            continue
        acc = 100 * sum(r["pred"] == r["gt"] for r in rows) / len(rows)
        k = [r for r in rows if not r.get("masked")]
        kacc = f"{100 * sum(r['pred'] == r['gt'] for r in k) / len(k):8.1f}%" if k else "       —"
        # Wrong *and kept* is the only kind of error that reaches the training set.
        bad = sum(r["pred"] != r["gt"] for r in k)
        print(f"  {name:<24} {len(rows):>4} {acc:6.1f}% {len(k):>6} {kacc} {bad:>11}")

    trows = [r for r in records if r["teacher"] is not None]
    if trows:
        tacc = 100 * sum(r["teacher"] == r["gt"] for r in trows) / len(trows)
        vacc = 100 * sum(r["pred"] == r["gt"] for r in trows) / len(trows)
        print(f"\non the {len(trows)} faces where both answered: "
              f"vision {vacc:.1f}%, teacher {tacc:.1f}%")
        # The comparison that actually licenses anything. Vision is only ever *used* on faces
        # the teacher was too unsure about to keep, and on those the teacher is far weaker
        # than its 92% average. Comparing over all confidences measures a population the
        # vision pass will never be asked to label.
        print("  split by the teacher's own confidence -- the axis masking uses:")
        print(f"    {'teacher conf':<14} {'n':>4} {'vision':>8} {'teacher':>8}")
        cr = [r for r in trows if r.get("teacher_conf") is not None]
        for lo, hi, lbl in ((0.0, 0.7, "< 0.70 (masked)"), (0.7, 0.9, "0.70-0.90"),
                            (0.9, 1.01, ">= 0.90 (kept)")):
            rows = [r for r in cr if lo <= r["teacher_conf"] < hi]
            if rows:
                v = 100 * sum(r["pred"] == r["gt"] for r in rows) / len(rows)
                t = 100 * sum(r["teacher"] == r["gt"] for r in rows) / len(rows)
                print(f"    {lbl:<14} {len(rows):>4} {v:7.1f}% {t:7.1f}%")
        if not cr:
            print("    (no face_probs in --seg_dir; re-run inference to get them)")

    args.out.write_text(json.dumps(
        {"records": records, "n": n, "accuracy": correct / n,
         "api_calls": calls, "passes": args.passes}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
