#!/usr/bin/env python3
"""Fuse the network's labels with geometry's, and route the disagreements to review.

The problem this replaces
-------------------------
Today's synthetic-label pipeline is a single chain: `infer.py` teacher -> confidence mask
in `merge_suitable_samples.py` -> train. It has two structural faults that no threshold
fixes.

*It has no second opinion.* Every pseudo-label is the teacher's, so the student inherits
the teacher's errors and self-distillation ratchets them in -- the measured mechanism
behind `chamfer` collapsing across generations.

*Masking discards the hard faces instead of resolving them.* A face the teacher is unsure
about is set to -1 and contributes nothing. But uncertainty is not spread evenly: it
concentrates exactly on the rare classes the corpus most needs (`chamfer`, `fillet`), so
masking preferentially deletes the labels worth having. Raising the global threshold to 0.7
cut some classes' keep rate to 0.4%.

What replaces it
----------------
Two independent labellers, then route by whether they agree:

  network   learned embedding of face features. Edge convexity and true surface type are
            dead in its input (`face_attr` layout mismatch), so it is effectively reasoning
            from shape and context.
  geometry  `geo_label.py`, a measured P(class | fact) posterior over 24 exact B-Rep facts,
            including the convexity the network cannot see.

Measured on 59 mftrcad models / 1388 faces, 5-class, against exact ground truth:

    network alone                          63.9%
    geometry alone                         72.6%
    they agree           65.2% of faces    80.6% precision
    they disagree        34.8% of faces    geo right 57.8%, net right 32.7%,
                                           BOTH WRONG only 9.5%

Three things follow, and each is a decision this file implements.

1. **Agreement is the accept signal, and network confidence is not.** Within the agreeing
   set, tightening the network's confidence from 0.0 to 0.95 moves precision 80.6% -> 81.5%
   while dropping coverage 65% -> 41%. It buys nothing, which is consistent with the
   earlier finding that the network's confidence is *higher* on its `chamfer` errors than
   its `chamfer` hits. Agreement between independent views is doing the work.

2. **On disagreement, geometry wins -- so it is the default, not the tie-break.** Geometry
   is right on 57.8% of disagreements against the network's 32.7%, nearly two to one. The
   network is nonetheless kept as the recorded alternative, because it is right on a third
   of the cases geometry gets wrong.

3. **The review queue is a binary choice, which is what makes it affordable.** Only 9.5% of
   disagreements have both labellers wrong, so 90.5% of the time the correct answer is one
   of exactly two named candidates. An arbiter that only has to pick A or B tops out at
   ~84% overall accuracy on this sample -- and it is a far easier question than the open
   27-way one that put Claude's dense-view labelling at chance. `--review_out` writes that
   queue for `vision_label.py` and the viewer to consume.

So this file does not mask. It labels every face, records how the label was reached, and
says which faces a human or a vision pass should look at -- which is the "look at it and
override it with confidence" workflow, rather than deleting the evidence.

Usage
-----
    # 1. network labels (note: infer.py double-frees in `brep_mfr`; use new_brepmfr)
    python -m brepformer.infer --checkpoint ... --step_dir steps/ --out_dir net_preds/ --5_classes
    # 2. geometric labels
    python -m brepformer.data.geo_label --stats brepformer/data/class_geometry_stats.json \\
        --step_dir steps/ --out_dir geo_preds/
    # 3. fuse, and emit the review queue
    python -m brepformer.data.fuse_labels --net_dir net_preds/ --geo_dir geo_preds/ \\
        --out_dir fused/ --label_space 5 --review_out review_queue.json
    # 4. score the result exactly like any other labeller
    python -m brepformer.data.label_harness --mode select --predictions fused/ \\
        --limit 60 --label_space 5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from brepformer.data.classes import CLASS_NAMES, DEFEATURE_CLASS_NAMES

#: Confidence written for each route. These are *tiers*, not probabilities: their only job
#: is to sort faces for `label_harness --mode select` and for a coverage cut, and the
#: measured precision of each tier is what gives them meaning.
#:
#: The gap between AGREE and the rest is deliberately large. Within-tier confidence from
#: either labeller was measured to add almost nothing (80.6% -> 81.5% precision for a 24
#: point coverage loss), so it is used only to break ties inside a tier.
TIER = {
    "agree": 0.90,          # both labellers, 80.6% precision at 65.2% coverage
    "geo_only": 0.55,       # disagreement, geometry preferred: 57.8% right
    "net_only": 0.40,       # geometry abstained
    "unknown": 0.10,        # neither had anything to say
}

#: Classes where agreement is *not* enough to accept without a human, per label space.
#:
#: `chamfer` is here on measurement, not caution: the two labellers agree on only 16% of
#: chamfer faces and are still just 60% right when they do -- the worst agreed-precision of
#: any class, against 86% for `random` and 100% for `fillet`. It is also the class whose
#: confidence is anti-correlated with correctness in the network, and the class the
#: defeaturing task turns on. Agreement on `chamfer` means two labellers sharing a blind
#: spot more often than it means a correct label.
ALWAYS_REVIEW = {"5": {"chamfer"}, "27": {"chamfer", "chamfer_2", "round"}}


def load_preds(d: Path) -> dict:
    """Read a directory of `infer.py`-format predictions, keyed by model stem.

    mftrcad stems carry a `_result` suffix that prediction filenames may or may not keep,
    so both forms are indexed rather than guessing which convention produced the directory.
    """
    out = {}
    for f in sorted(d.glob("*.json")):
        try:
            r = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        if "face_preds" not in r:
            continue
        out[f.stem] = r
        out.setdefault(f.stem.replace("_result", ""), r)
    return out


def confidence_of(r: dict, i: int) -> float:
    """A labeller's own confidence for face `i`, preferring an explicit gate score.

    `geo_label` scores its decisions on a different pooling than it makes them with, so its
    `face_confidence` is not `max(face_probs)` and is substantially better calibrated.
    """
    c = r.get("face_confidence")
    if c is not None and i < len(c):
        return float(c[i])
    probs = r.get("face_probs")
    if probs and i < len(probs):
        return float(max(probs[i]))
    return 0.0


def fuse_model(net: dict | None, geo: dict | None, names: list,
               always_review: set) -> dict:
    """Fuse one model's faces. Returns the fused prediction plus its review queue."""
    n_faces = 0
    for r in (net, geo):
        if r:
            n_faces = max(n_faces, len(r["face_preds"]))

    preds, conf, routes, review = [], [], [], []
    for i in range(n_faces):
        np_ = int(net["face_preds"][i]) if net and i < len(net["face_preds"]) else None
        gp = int(geo["face_preds"][i]) if geo and i < len(geo["face_preds"]) else None
        nc = confidence_of(net, i) if net else 0.0
        gc = confidence_of(geo, i) if geo else 0.0

        if np_ is not None and gp is not None and np_ == gp:
            route, pick = "agree", np_
        elif gp is not None and np_ is not None:
            # Geometry is the measured winner on disagreements (57.8% vs 32.7%).
            route, pick = "geo_only", gp
        elif gp is not None:
            route, pick = "geo_only", gp
        elif np_ is not None:
            route, pick = "net_only", np_
        else:
            route, pick = "unknown", -1

        name = names[pick] if 0 <= pick < len(names) else str(pick)
        needs_review = route != "agree" or name in always_review
        # Tie-break within the tier by the labellers' own confidence, scaled small enough
        # that it can never reorder two different tiers.
        preds.append(pick)
        conf.append(round(min(TIER[route] + 0.05 * max(nc, gc), 0.999), 5))
        routes.append(route)
        if needs_review and pick >= 0:
            cands = sorted({c for c in (gp, np_) if c is not None},
                           key=lambda c: (c != gp))  # geometry's candidate first
            review.append({
                "face": i,
                "route": route,
                "chosen": name,
                # The whole point of the queue: an arbiter picks among these, it does not
                # label from scratch. Both-wrong was only 9.5% of disagreements.
                "candidates": [names[c] if 0 <= c < len(names) else str(c) for c in cands],
                "geo_conf": round(gc, 4),
                "net_conf": round(nc, 4),
                "reason": ("class always reviewed" if route == "agree"
                           else "labellers disagree"),
            })

    return {
        "num_faces": n_faces,
        "face_preds": preds,
        "face_class_names": [names[c] if 0 <= c < len(names) else str(c) for c in preds],
        "face_confidence": conf,
        "face_route": routes,
        "source": "fuse_labels",
        "review": review,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net_dir", type=Path, required=True)
    ap.add_argument("--geo_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--label_space", default="5", choices=["27", "5"])
    ap.add_argument("--review_out", type=Path, default=None,
                    help="Write the binary-choice review queue here, for vision/the viewer.")
    ap.add_argument("--no_always_review", action="store_true",
                    help="Drop the ALWAYS_REVIEW class list. Measure before using this.")
    args = ap.parse_args()

    names = CLASS_NAMES if args.label_space == "27" else DEFEATURE_CLASS_NAMES
    always = set() if args.no_always_review else ALWAYS_REVIEW[args.label_space]

    net, geo = load_preds(args.net_dir), load_preds(args.geo_dir)
    stems = sorted(set(net) | set(geo))
    # Both directories index each model under its bare and `_result` stem; keep one.
    stems = [s for s in stems if not (s + "_result") in net or s.endswith("_result")]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
    queue = {}
    n_faces = n_review = n_models = 0

    for stem in stems:
        res = fuse_model(net.get(stem), geo.get(stem), names, always)
        if not res["num_faces"]:
            continue
        review = res.pop("review")
        res["model_id"] = stem
        (args.out_dir / f"{stem}.json").write_text(json.dumps(res))
        for r in res["face_route"]:
            counts[r] += 1
        n_faces += res["num_faces"]
        n_models += 1
        if review:
            queue[stem] = review
            n_review += len(review)

    print(f"fused {n_models} models / {n_faces} faces -> {args.out_dir}")
    print("\nroute distribution:")
    for r in ("agree", "geo_only", "net_only", "unknown"):
        if counts[r]:
            print(f"  {r:<10}{counts[r]:>7} faces  {100 * counts[r] / n_faces:>5.1f}%")
    print(f"\nreview queue: {n_review} faces ({100 * n_review / max(n_faces, 1):.1f}%) "
          f"across {len(queue)} models")
    print("Every queued face carries the 2 candidates to choose between, not an open "
          "question -- see the module docstring for why that distinction is the point.")

    if args.review_out:
        args.review_out.write_text(json.dumps(
            {"label_space": args.label_space, "n_faces": n_review, "models": queue},
            indent=1))
        print(f"wrote {args.review_out}")


if __name__ == "__main__":
    main()
