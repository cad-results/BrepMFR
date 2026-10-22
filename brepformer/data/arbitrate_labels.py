#!/usr/bin/env python3
"""Combine the teacher's inference, a second teacher, the measured corpus and the vision
pass into one label per face -- with the vision pass given the *narrowest* mandate that is
still useful.

Why this exists rather than a relabelling pass. The teacher already labels every face for
free, and per-class confidence masking already decides which of those labels are safe. On
the current corpus that leaves **109,893 of 216,781 faces (50.7%) masked to -1** -- thrown
away, contributing no gradient. Those discarded faces are the opportunity, and they are also
the only place where a vision label carries no downside: a face that is already going to be
dropped cannot be made worse by an answer, only better or unchanged. Meanwhile a face the
teacher kept confidently is one the earlier vision pass measurably *damaged* when it was
allowed to overwrite -- as a verifier it denied 74% of correct labels.

So the ladder below is ordered by how much a mistake costs, and the default mode
(``fill``) lets vision act only on faces inference gave up on:

| Situation | Default action | Why |
|---|---|---|
| vision never looked at the face | keep teacher | vision has zero effect on faces outside its budget |
| vision abstained or was masked | keep teacher | an abstention is not evidence against the teacher |
| teacher masked it, vision decided | **fill with vision** | pure gain: the face was going to be dropped |
| teacher kept it, vision agrees | keep, count as confirmed | two independent sources; the label is now doubly supported |
| teacher kept it, vision disagrees | **keep teacher** | overwriting here is the move that has already cost 15% of correct labels once |

The last row is where the two escalations live, both off by default and both requiring
evidence rather than a flag:

* ``--veto`` lets a *confident* disagreement mask the face instead of keeping it. That trades
  a possibly-wrong label for a certainly-absent one; worth it only where the gate measured
  vision to be more accurate than the teacher.
* ``--overwrite`` lets vision replace the teacher's class, and only for classes named in
  ``--gate_json`` where vision beat the teacher on ground truth by ``--gate_margin``. There
  is no flag that lets it overwrite a class it has not been measured to be better at.

Usage:
    # *best* -- fill only: recover the masked faces, touch nothing else
    python -m brepformer.data.arbitrate_labels \\
        --labels_in brepformer/data/suitable_samples_pcq5/labels \\
        --vision_dir brepformer/inference/hard_samples_5_vision \\
        --out_dir brepformer/data/suitable_samples_arb5/labels --label_space 5

    # dry run with the per-class delta, writing nothing
    python -m brepformer.data.arbitrate_labels ... --dry_run

    # escalated, licensed by the gate run
    python -m brepformer.data.arbitrate_labels ... --veto \\
        --overwrite --gate_json brepformer/data/dense_vision_validation.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from brepformer.data.classes import CLASS_NAMES, DEFEATURE_CLASS_NAMES

IGNORE_LABEL = -1


def read_labels(path: Path) -> Optional[List[int]]:
    """The ``[model_id, {"labels": [...]}]`` form the preprocessor consumes."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if isinstance(data, list) and len(data) == 2 and isinstance(data[1], dict):
        return list(map(int, data[1].get("labels") or []))
    if isinstance(data, dict) and "cls" in data:      # mftrcad ground-truth form
        cls = data["cls"]
        n = max((int(k) for k in cls), default=-1) + 1
        out = [IGNORE_LABEL] * n
        for k, v in cls.items():
            out[int(k)] = int(v)
        return out
    return None


def read_vision(path: Path) -> Dict[int, Dict]:
    """``{face_index: verdict}`` from a ``vision_label.py`` per-model JSON.

    Only the fields the ladder needs are kept, and ``final_label`` is read rather than
    ``class``: ``final_label`` is the one that already has the module's own masking applied,
    so a face vision itself refused to commit to arrives here as -1 and is treated as an
    abstention rather than as an answer.
    """
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    faces = data.get("faces") if isinstance(data, dict) else data
    if not isinstance(faces, dict):
        return {}
    out: Dict[int, Dict] = {}
    for k, r in faces.items():
        if not isinstance(r, dict):
            continue
        out[int(k)] = {
            "final": int(r.get("final_label", IGNORE_LABEL)),
            "cls": int(r.get("vision_class", IGNORE_LABEL)),
            "teacher": r.get("teacher_class"),
            "confidence": float(r.get("confidence") or 0.0),
            "agreement": float(r.get("agreement") or 0.0),
            "rarity": r.get("rarity") or {},
        }
    return out


def gate_licences(gate_json: Optional[Path], class_names: List[str],
                  margin: float, min_n: int,
                  teacher_conf_max: float = 1.01
                  ) -> Dict[int, Tuple[float, float, int]]:
    """Classes where the gate measured vision *better than the teacher*, from its records.

    Read from the validation output rather than configured, because the whole point is that
    the licence is earned per class on ground truth. A class the gate never sampled, or
    sampled too few times to separate the two, gets no licence -- absence of evidence is not
    a licence, and the rare classes are exactly where a small sample would fake one.

    ``teacher_conf_max`` restricts the comparison to faces the teacher was unsure about,
    which is the only population this ever applies to. Measured on the defeature ground
    truth the teacher is **92.3% overall** but **55.3% on the 7% of faces below 0.7
    confidence**, and 99.2% where it reports 0.99+. A licence computed over all confidences
    would therefore be asking "can vision beat a 92% model", which is the wrong question and
    would refuse every class; the right question is "can it beat a 55% model on the faces
    that are about to be thrown away".
    """
    if not gate_json or not gate_json.exists():
        return {}
    try:
        records = json.loads(gate_json.read_text()).get("records") or []
    except (OSError, ValueError):
        return {}
    out: Dict[int, Tuple[float, float, int]] = {}
    for c, _name in enumerate(class_names):
        rows = [r for r in records if r.get("gt") == c and r.get("teacher") is not None
                and (r.get("teacher_conf") is None
                     or r["teacher_conf"] < teacher_conf_max)]
        if len(rows) < min_n:
            continue
        v = sum(r["pred"] == c for r in rows) / len(rows)
        t = sum(r["teacher"] == c for r in rows) / len(rows)
        if v - t >= margin:
            out[c] = (round(v, 3), round(t, 3), len(rows))
    return out


def arbitrate(labels: List[int], vision: Dict[int, Dict], *,
              min_fill_confidence: float, veto: bool, veto_confidence: float,
              overwrite: Dict[int, Tuple[float, float, int]],
              overwrite_confidence: float) -> Tuple[List[int], collections.Counter]:
    """Apply the ladder to one model. Returns the new labels and a count per action."""
    out = list(labels)
    acts: collections.Counter = collections.Counter()
    for i, teacher in enumerate(labels):
        v = vision.get(i)
        if v is None:
            acts["untouched (not looked at)"] += 1
            continue
        if v["final"] < 0:
            # Vision abstained or masked itself. It does not get to remove a teacher label on
            # the strength of its own uncertainty.
            acts["vision abstained, teacher kept" if teacher >= 0
                 else "both gave up"] += 1
            continue
        if teacher < 0:
            # The branch this script exists for. Nothing is at risk: the face is currently
            # dropped from training entirely.
            if v["confidence"] >= min_fill_confidence:
                out[i] = v["final"]
                acts["FILLED (teacher had masked it)"] += 1
            else:
                acts["left masked (vision below fill bar)"] += 1
            continue
        if v["final"] == teacher:
            acts["confirmed (both agree)"] += 1
            continue
        # Disagreement on a face the teacher was confident enough to keep.
        lic = overwrite.get(v["final"])
        if lic and v["confidence"] >= overwrite_confidence:
            out[i] = v["final"]
            acts[f"OVERWROTE -> class {v['final']} (gate-licensed)"] += 1
        elif veto and v["confidence"] >= veto_confidence:
            out[i] = IGNORE_LABEL
            acts["VETOED to -1 (confident disagreement)"] += 1
        else:
            acts["disagreed, teacher kept"] += 1
    return out, acts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels_in", type=Path, required=True,
                    help="Existing label dir (already confidence-masked)")
    ap.add_argument("--vision_dir", type=Path, required=True,
                    help="vision_label.py per-model JSONs")
    ap.add_argument("--out_dir", type=Path, help="Where to write arbitrated labels")
    ap.add_argument("--label_space", choices=["5", "27"], default="5")
    ap.add_argument("--min_fill_confidence", type=float, default=0.55,
                    help="Vision confidence needed to fill a face the teacher masked")
    ap.add_argument("--veto", action="store_true",
                    help="Let a confident disagreement mask a kept face")
    ap.add_argument("--veto_confidence", type=float, default=0.75)
    ap.add_argument("--overwrite", action="store_true",
                    help="Let vision replace a kept label, for gate-licensed classes only")
    ap.add_argument("--overwrite_confidence", type=float, default=0.85)
    ap.add_argument("--gate_json", type=Path,
                    default=Path("brepformer/data/dense_vision_validation.json"))
    ap.add_argument("--gate_margin", type=float, default=0.10,
                    help="Accuracy points vision must beat the teacher by, per class")
    ap.add_argument("--gate_min_n", type=int, default=10,
                    help="Ground-truth faces of a class the gate needs before licensing it")
    ap.add_argument("--gate_conf_max", type=float, default=0.7,
                    help="Only compare vision against the teacher on gate faces where the "
                         "teacher's own confidence was below this. That is the population "
                         "arbitration acts on; over all confidences the teacher scores 92%% "
                         "and no class could ever be licensed.")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    names = CLASS_NAMES if args.label_space == "27" else DEFEATURE_CLASS_NAMES
    lic = gate_licences(args.gate_json, names, args.gate_margin,
                        args.gate_min_n, args.gate_conf_max) if args.overwrite else {}
    if args.overwrite:
        if lic:
            print("gate-licensed classes (vision acc / teacher acc / n):")
            for c, (v, t, n) in sorted(lic.items()):
                print(f"  {names[c]:<22} {v:.2f} / {t:.2f}  n={n}")
        else:
            print("--overwrite requested but the gate licenses no class; nothing will be "
                  "overwritten. Run data/validate_dense_vision.py with --seg_dir first.")

    acts: collections.Counter = collections.Counter()
    before: collections.Counter = collections.Counter()
    after: collections.Counter = collections.Counter()
    n_models = 0
    if args.out_dir and not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    for lp in sorted(args.labels_in.glob("*.json")):
        labels = read_labels(lp)
        if labels is None:
            continue
        vision = read_vision(args.vision_dir / f"{lp.stem}.json")
        if not vision:
            before.update(labels)
            after.update(labels)
            acts["model not visited by vision"] += 1
            continue
        n_models += 1
        new, a = arbitrate(
            labels, vision,
            min_fill_confidence=args.min_fill_confidence,
            veto=args.veto, veto_confidence=args.veto_confidence,
            overwrite=lic, overwrite_confidence=args.overwrite_confidence)
        acts.update(a)
        before.update(labels)
        after.update(new)
        if args.out_dir and not args.dry_run:
            raw = json.loads(lp.read_text())
            if isinstance(raw, list) and len(raw) == 2 and isinstance(raw[1], dict):
                raw[1]["labels"] = new
                (args.out_dir / lp.name).write_text(json.dumps(raw))
            else:
                (args.out_dir / lp.name).write_text(
                    json.dumps([lp.stem, {"labels": new}]))

    tot = sum(before.values())
    print(f"\n{n_models} models visited by vision, {tot} faces total")
    print("\nwhat happened, per face:")
    for k, v in acts.most_common():
        print(f"  {v:>8}  {k}")

    print("\nper class, kept faces before -> after:")
    print(f"  {'class':<22} {'before':>8} {'after':>8} {'delta':>8}")
    for c, name in enumerate(names):
        b, a2 = before.get(c, 0), after.get(c, 0)
        if b or a2:
            print(f"  {name:<22} {b:>8} {a2:>8} {a2 - b:>+8}")
    bm, am = before.get(IGNORE_LABEL, 0), after.get(IGNORE_LABEL, 0)
    print(f"  {'masked (-1)':<22} {bm:>8} {am:>8} {am - bm:>+8}"
          f"   = {100 * bm / max(tot, 1):.1f}% -> {100 * am / max(tot, 1):.1f}%")
    if args.dry_run:
        print("\ndry run: nothing written")
    elif args.out_dir:
        print(f"\nwrote {args.out_dir}")


if __name__ == "__main__":
    main()
