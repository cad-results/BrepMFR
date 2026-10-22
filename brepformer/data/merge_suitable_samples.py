"""Merge hand-triaged hard_samples into a mftrcad-compatible training corpus.

For each STEP file under ``--triage_root/{yes[,maybe]}/steps/``, this script:

1. Copies the STEP into ``--out_dir/steps/<model_id>.step``
2. Reads the corresponding 27-class inference label from
   ``--inference_dir/<model_id>.json`` (produced by ``brepformer.infer``)
3. Writes a mftrcad-format label JSON to
   ``--out_dir/labels/<model_id>_result.json`` containing:
     - ``labels``: per-face 27-class integer labels, with ``-1`` for faces the
       teacher model was not confident about (see ``--min_confidence``)
     - ``model_id``: string id
     - ``num_faces``: int
     - ``source``: "hard_samples_pseudo" (audit trail)

The mftrcad format expected by ``preprocess.py`` is ``[<model_id>, {...}]``
with a ``labels`` field for per-face class ids. We reproduce that.

Model IDs are the STEP file basename with spaces replaced by underscores
(so ``push_clamp.step`` → ``push_clamp``, and downloaded names like
``blum-hinge-cup-1__Assemblage1.step`` → ``blum-hinge-cup-1__Assemblage1``).

Masking mode matters, and the per-class mode carries an open question. A single global
``--min_confidence`` deletes whole classes rather than filtering evenly (measured: keep
rates from 81.8% down to 0.4% at 0.7), so prefer ``--per_class_top_frac``. But that mode
deliberately admits rare-class faces at ~0.60 mean confidence, and whether that helps or
teaches errors is **unvalidated** — trial6 is the experiment. Read "Phase 5" in
``brepformer/data/hard_samples/PLAN.md`` before interpreting a training result either
way; it records what a good *and* a bad outcome should be taken to mean, and why the
fallback is soft-label distillation rather than reverting to a global threshold.

Usage:
    python brepformer/data/merge_suitable_samples.py \\
        --triage_root brepformer/data/hard_samples \\
        --inference_dir brepformer/inference/hard_samples_27 \\
        --out_dir brepformer/data/suitable_samples \\
        --include yes maybe

    # per-class masking (preferred; see above)
    python brepformer/data/merge_suitable_samples.py \\
        --steps_dir brepformer/data/suitable_samples_scaled/steps \\
        --inference_dir brepformer/inference/hard_samples_27 \\
        --out_dir brepformer/data/suitable_samples_pcq \\
        --per_class_top_frac 0.5 --per_class_floor 0.5 --min_kept_frac 0.15
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from brepformer.data.classes import map_labels_to_defeature  # noqa: E402


def slugify_id(step_path: Path) -> str:
    return step_path.stem.replace(" ", "_")


#: Faces the teacher model is unsure about are written as this label. The face
#: segmentation loss is ``nn.CrossEntropyLoss(..., ignore_index=-1)`` and
#: ``face_labels`` already uses -1 for padding, so masked faces contribute no
#: gradient and need no model-side change. ``train_preprocessed.py``'s
#: inverse-frequency weighting also filters on ``0 <= c < num_face_classes``,
#: so masked faces are excluded from class statistics too.
IGNORE_LABEL = -1


def read_preds(inference_json: Path) -> tuple[list[int], list[float]] | None:
    """Return ``(labels, top1_confidences)`` from an inference JSON, or None.

    Confidences come back empty if the file has no ``face_probs``; callers that
    need them must check, because silently treating "no probabilities" as
    "confident" would train on raw argmax without saying so.
    """
    data = json.loads(inference_json.read_text())
    if not isinstance(data, dict):
        return None

    if "face_preds" in data:
        labels = list(map(int, data["face_preds"]))
    elif "predictions" in data and isinstance(data["predictions"], list):
        labels = list(map(int, data["predictions"]))
    else:
        return None

    probs = data.get("face_probs")
    if not probs:
        return labels, []
    if len(probs) != len(labels):
        # Fail loud rather than silently training on unmasked noisy labels.
        raise ValueError(
            f"{inference_json.name}: face_probs has length {len(probs)} vs "
            f"{len(labels)} labels. Re-run inference to regenerate probabilities."
        )
    return labels, [max(row) if row else 0.0 for row in probs]


def fit_per_class_thresholds(
    confs_by_class: dict[int, list[float]],
    top_frac: float,
    abs_floor: float = 0.0,
    min_faces: int = 0,
) -> dict[int, float]:
    """Per-class confidence cutoffs that keep the top ``top_frac`` of each class.

    Why this exists, and why a single global threshold was the wrong tool: the
    teacher's confidence scale is not comparable across classes, so one cutoff
    does not express one quality bar — it expresses "delete whichever classes this
    teacher happens to be shakiest on". Measured on the 853-model corpus at the old
    global 0.7, keep rates ranged from 81.8% (``rectangular_passage``) down to 0.4%
    (``rectangular_blind_slot``), 0.9% (``triangular_blind_step``) and 2.7%
    (``circular_blind_slot``); ``stock`` kept only 21.9% despite being the majority
    class on real parts, which biases the student toward over-segmenting features.
    Ranking *within* a class and keeping a fixed share equalises selectivity, so
    every class contributes gradient in proportion to how often it actually occurs.

    The direction of that damage flips with the teacher, which is the real reason
    not to rely on a global number: with the 27-class trial5 teacher, masking at 0.7
    *raised* chamfer's share (median confidence 0.751), whereas with the 5-class
    ``trial2_finetune_ss`` teacher chamfer was the weak class and got halved.

    ``abs_floor`` is the escape hatch for classes the teacher simply cannot do: if
    even a class's best faces sit near chance, admitting its top ``top_frac``
    imports labels that are mostly wrong. Faces below the floor stay masked even if
    that leaves the class under quota. ``min_faces`` pulls the other way, letting a
    small class keep its most confident faces so it is not erased by rounding.
    """
    if not 0.0 < top_frac <= 1.0:
        raise ValueError(f"top_frac must be in (0, 1], got {top_frac}")

    thresholds: dict[int, float] = {}
    for c, confs in confs_by_class.items():
        if not confs:
            continue
        ranked = sorted(confs, reverse=True)
        n_keep = max(int(round(len(ranked) * top_frac)), 1)
        if min_faces:
            n_keep = max(n_keep, min(min_faces, len(ranked)))
        # Cutoff is the confidence of the last face we intend to keep; using >=
        # at apply time means ties are kept, so a class can slightly exceed quota
        # rather than being cut mid-tie in an arbitrary order.
        thresholds[c] = max(ranked[n_keep - 1], abs_floor)
    return thresholds


def cap_dominant_classes(
    confs_by_class: dict[int, list[float]],
    thresholds: dict[int, float],
    max_frac: float,
) -> dict[int, float]:
    """Raise cutoffs so no class exceeds ``max_frac`` of all kept faces.

    Applied after the per-class quota because a class can be both well-predicted and
    overwhelming: ``Oring`` is 48.5% of kept faces on this corpus, so a student can
    score well while learning little about anything else. Trimming the dominant
    class's least confident faces keeps the corpus balanced without touching the loss,
    and unlike loss reweighting it also shrinks the epoch.

    Iterative because capping one class shrinks the total, which can push another
    class over the limit.
    """
    if not 0.0 < max_frac <= 1.0:
        raise ValueError(f"max_class_frac must be in (0, 1], got {max_frac}")
    out = dict(thresholds)
    ranked = {c: sorted(v, reverse=True) for c, v in confs_by_class.items()}

    for _ in range(len(out) + 1):
        kept = {c: sum(1 for x in ranked[c] if x >= out[c]) for c in out}
        total = sum(kept.values())
        if not total:
            break
        over = [c for c in out if kept[c] > max_frac * total]
        if not over:
            break
        for c in over:
            # Solve for the largest allowance consistent with the *other* classes'
            # current counts: n_c <= max_frac * (rest + n_c).
            rest = total - kept[c]
            allow = int(max_frac * rest / (1.0 - max_frac)) if max_frac < 1.0 else kept[c]
            allow = max(min(allow, len(ranked[c])), 1)
            out[c] = max(out[c], ranked[c][allow - 1])
    return out


def apply_thresholds(
    labels: list[int], confs: list[float], thresholds: dict[int, float],
    default: float = float("inf"),
) -> tuple[list[int], int]:
    """Mask faces below their predicted class's cutoff. Returns (labels, n_masked).

    ``default`` applies to classes absent from ``thresholds``; it is ``inf`` (mask
    everything) for the per-class path, where a missing class means the fitting pass
    never saw it and admitting it would be unfitted, and the global cutoff for
    single-threshold mode.
    """
    out: list[int] = []
    masked = 0
    for lab, conf in zip(labels, confs):
        if conf >= thresholds.get(lab, default):
            out.append(lab)
        else:
            out.append(IGNORE_LABEL)
            masked += 1
    return out, masked


def apply_thresholds_with_agreement(
    labels: list[int], confs: list[float], thresholds: dict[int, float],
    peer: tuple[list[int], list[float]] | None,
    agree_floor: float, coarse_self: list[int] | None = None,
    default: float = float("inf"),
) -> tuple[list[int], int, int]:
    """Mask as :func:`apply_thresholds`, but rescue faces a second teacher agrees with.

    Returns ``(labels, n_masked, n_rescued)``.

    Why a second teacher instead of a lower threshold: confidence is not calibrated
    across classes, so lowering the cutoff admits whichever class the teacher happens to
    be over-confident about -- which is how a global threshold deletes chamfer while
    keeping stock. Two *independently trained* teachers being wrong the same way is much
    rarer than one teacher being confidently wrong, so agreement is a stronger signal at
    a given confidence than confidence alone. That lets ``agree_floor`` sit below the
    fitted per-class cutoff and still keep precision, which is the point: it increases
    the number of usable faces rather than trading them.

    ``coarse_self`` is ``labels`` projected into the peer's label space; the comparison
    is made there, while the label that gets *written* is always the original ``labels``
    entry. Passing the peer in a different space without projecting would make
    "agreement" a comparison of unrelated integers.
    """
    out: list[int] = []
    masked = rescued = 0
    peer_labels, peer_confs = peer if peer else ([], [])
    have_peer = bool(peer_labels) and len(peer_labels) == len(labels)
    mine = coarse_self if coarse_self is not None else labels
    for i, (lab, conf) in enumerate(zip(labels, confs)):
        if conf >= thresholds.get(lab, default):
            out.append(lab)
            continue
        # Below its own cutoff: keep only if the other teacher independently predicted
        # the same class, and both are at least `agree_floor` confident.
        if (have_peer and peer_labels[i] == mine[i] and mine[i] >= 0
                and conf >= agree_floor
                and (not peer_confs or peer_confs[i] >= agree_floor)):
            out.append(lab)
            rescued += 1
            continue
        out.append(IGNORE_LABEL)
        masked += 1
    return out, masked, rescued


def load_peer_preds(
    peer_dir: Path | None, model_id: str, collapse: bool
) -> tuple[list[int], list[float]] | None:
    """Load the second teacher's predictions for one model, in the target label space."""
    if peer_dir is None:
        return None
    res = read_preds(peer_dir / f"{model_id}.json")
    if res is None:
        return None
    labels, confs = res
    if collapse:
        labels = map_labels_to_defeature(labels)
    return labels, confs


def load_face_labels(
    inference_json: Path, min_confidence: float = 0.0
) -> tuple[list[int], int] | None:
    """Extract per-face 27-class labels, masking faces below a global cutoff.

    Retained for the single-threshold mode (``--min_confidence``); the per-class
    path uses :func:`fit_per_class_thresholds` instead. See that function for why
    a global cutoff distorts the class distribution.
    """
    res = read_preds(inference_json)
    if res is None:
        return None
    labels, confs = res

    if min_confidence <= 0.0:
        return labels, 0
    if not confs:
        raise ValueError(
            f"{inference_json.name}: --min_confidence requested but face_probs is "
            "missing. Re-run inference to regenerate probabilities."
        )
    return apply_thresholds(labels, confs, {}, default=min_confidence)


def write_label_file(out_dir: Path, model_id: str, labels: list[int], source: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = [
        model_id,
        {
            "labels": labels,
            "num_faces": len(labels),
            "source": source,
        },
    ]
    target = out_dir / f"{model_id}_result.json"
    target.write_text(json.dumps(payload, separators=(",", ":")))
    return target


def collect_triaged_steps(triage_root: Path, include: list[str]) -> list[Path]:
    steps: list[Path] = []
    for bucket in include:
        d = triage_root / bucket / "steps"
        if not d.exists():
            continue
        steps.extend(sorted(p for p in d.iterdir() if p.suffix.lower() in {".step", ".stp"}))
    return steps


def collect_flat_steps(steps_dir: Path) -> list[Path]:
    """Collect STEP files from a flat directory, non-recursively.

    Why this exists: the triage buckets (``yes``/``maybe``) assume a human eyeballed
    every part, which was tractable at 5 files and is not at 1185. For the scaled
    corpus the quality gate is machine-applied instead, and it is a *stricter* filter
    than the eyeball was:

    * ``curate_corpus.py`` has already removed exact duplicates, capped parametric
      families and over-represented slugs, and dropped unusable face counts;
    * per-face confidence masking (below) then discards every face the teacher was
      unsure about, which no visual triage pass ever did.

    Non-recursive on purpose, mirroring ``infer.py --step_dir``: it must not descend
    into ``{yes,maybe,no}/`` and silently re-ingest hand-triaged files, nor into
    ``hard_samples_rejected/`` if someone points this at a parent directory.
    """
    if not steps_dir.exists():
        return []
    return sorted(
        p for p in steps_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".step", ".stp"}
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--triage_root", type=Path, default=Path("brepformer/data/hard_samples"))
    p.add_argument("--inference_dir", type=Path, default=Path("brepformer/inference/hard_samples_27"))
    p.add_argument("--out_dir", type=Path, default=Path("brepformer/data/suitable_samples"))
    p.add_argument("--include", nargs="+", default=["yes"], choices=["yes", "maybe"])
    p.add_argument("--steps_dir", type=Path, default=None,
                   help="Merge from this flat directory of STEP files instead of the "
                        "{yes,maybe}/steps triage buckets. Use for the curated "
                        "at-scale corpus, where geometric curation plus confidence "
                        "masking replaces per-file human triage.")
    p.add_argument("--source_tag", default="hard_samples_pseudo")
    p.add_argument("--min_confidence", type=float, default=0.7,
                   help="Mask faces whose top-1 predicted probability is below this "
                        "to -1 (ignore_index) so they contribute no loss. 0 disables "
                        "masking and trains on raw argmax. Default 0.7 keeps ~39%% of "
                        "faces on the current corpus.")
    p.add_argument("--per_class_top_frac", type=float, default=None,
                   help="Instead of one global cutoff, keep the most confident this "
                        "fraction of faces WITHIN EACH predicted class (e.g. 0.5 = top "
                        "half of every class). Equalises how hard each class is "
                        "filtered, so the teacher's weakest classes are thinned rather "
                        "than erased. Overrides --min_confidence. Requires two passes "
                        "over the inference JSONs to fit the cutoffs.")
    p.add_argument("--per_class_floor", type=float, default=0.0,
                   help="With --per_class_top_frac, never admit a face below this "
                        "confidence even if its class is under quota. Guards the "
                        "classes the teacher cannot do at all, where the top faces are "
                        "still near chance and would import mostly-wrong labels.")
    p.add_argument("--per_class_min_faces", type=int, default=0,
                   help="With --per_class_top_frac, let a small class keep at least "
                        "this many of its most confident faces so it is not erased by "
                        "rounding. Still subject to --per_class_floor.")
    p.add_argument("--max_class_frac", type=float, default=None,
                   help="Cap any single class at this fraction of all kept faces, "
                        "dropping its least confident faces first. Oring is 48.5%% of "
                        "kept faces on the current corpus; capping trims the dominant "
                        "class in the data rather than reweighting it in the loss.")
    p.add_argument("--label_space", choices=["27", "5"], default="27",
                   help="Class space of the written labels. '5' collapses the teacher's "
                        "27 MFTRCAD classes to the defeature taxonomy via "
                        "CLASS_TO_DEFEATURE, for training a model comparable to "
                        "trial1_ss1500. Thresholds are always fitted in 27-class space "
                        "first: the teacher's confidence is calibrated per fine class, "
                        "and collapsing before filtering would let a well-predicted "
                        "chamfer_2 face be judged against the pooled 'cut' cutoff.")
    p.add_argument("--agree_inference_dir", type=Path, default=None,
                   help="Second teacher's inference dir. Faces below their fitted "
                        "per-class cutoff are RESCUED (kept, not masked) when this "
                        "teacher independently predicted the same class and both are at "
                        "least --agree_floor confident. Two independently trained models "
                        "being wrong the same way is much rarer than one being "
                        "confidently wrong, so this raises the supervised face count "
                        "without lowering precision the way a smaller cutoff would.")
    p.add_argument("--agree_space", choices=["27", "5"], default="5",
                   help="Label space of --agree_inference_dir. Comparison happens in "
                        "the coarser of the two spaces; the written label is always in "
                        "--label_space.")
    p.add_argument("--agree_floor", type=float, default=0.4,
                   help="Minimum confidence, on BOTH teachers, for an agreement rescue. "
                        "Sits below the fitted per-class cutoffs on purpose -- that is "
                        "what makes it add faces.")
    p.add_argument("--min_kept_frac", type=float, default=0.25,
                   help="Skip a model entirely if masking leaves fewer than this "
                        "fraction of its faces labelled — a model that is mostly "
                        "'unknown' teaches almost nothing and still costs a forward pass.")
    args = p.parse_args()

    steps_out = args.out_dir / "steps"
    labels_out = args.out_dir / "labels"
    steps_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    if args.steps_dir is not None:
        triaged = collect_flat_steps(args.steps_dir)
        if not triaged:
            raise SystemExit(f"No STEP files found directly under {args.steps_dir}")
        print(f"source: flat dir {args.steps_dir} ({len(triaged)} STEP files)")
    else:
        triaged = collect_triaged_steps(args.triage_root, args.include)
        if not triaged:
            raise SystemExit(
                f"No STEP files found under {args.triage_root}/{{{','.join(args.include)}}}/steps"
            )
        print(f"source: triage buckets {args.include} ({len(triaged)} STEP files)")

    # ---- pass 1: fit per-class cutoffs over the whole corpus (if requested) ----
    thresholds: dict[int, float] | None = None
    if args.per_class_top_frac is not None:
        confs_by_class: dict[int, list[float]] = collections.defaultdict(list)
        for step_path in triaged:
            infer_json = args.inference_dir / f"{step_path.stem}.json"
            if not infer_json.exists():
                continue
            res = read_preds(infer_json)
            if not res or not res[0]:
                continue
            labels, confs = res
            if not confs:
                raise SystemExit(
                    f"{infer_json.name}: --per_class_top_frac needs face_probs, which "
                    "this inference JSON lacks. Re-run inference."
                )
            for lab, conf in zip(labels, confs):
                confs_by_class[lab].append(conf)
        if not confs_by_class:
            raise SystemExit("no predictions found to fit per-class thresholds")
        thresholds = fit_per_class_thresholds(
            confs_by_class, args.per_class_top_frac,
            abs_floor=args.per_class_floor, min_faces=args.per_class_min_faces,
        )
        if args.max_class_frac is not None:
            thresholds = cap_dominant_classes(
                confs_by_class, thresholds, args.max_class_frac)
        print(f"fitted per-class cutoffs (top {args.per_class_top_frac:.0%} per class, "
              f"floor {args.per_class_floor}):")
        for c in sorted(thresholds, key=lambda c: -len(confs_by_class[c])):
            n = len(confs_by_class[c])
            kept = sum(1 for x in confs_by_class[c] if x >= thresholds[c])
            print(f"  class {c:>3}  n={n:>7}  cutoff {thresholds[c]:.3f}  "
                  f"keep {kept:>7} ({100 * kept / n:>5.1f}%)")

    # ---- pass 2: apply and write ----
    ok = 0
    missing_labels = 0
    skipped_sparse = 0
    total_faces = 0
    total_masked = 0
    total_rescued = 0
    peer_missing = 0
    for step_path in triaged:
        model_id = slugify_id(step_path)
        # Inference JSON is named by original stem (not slugified).
        infer_json = args.inference_dir / f"{step_path.stem}.json"
        if not infer_json.exists():
            missing_labels += 1
            continue
        if thresholds is not None:
            res = read_preds(infer_json)
            if not res or not res[0]:
                missing_labels += 1
                continue
            if args.agree_inference_dir is not None:
                # Agreement is checked in the COARSER of the two label spaces: a
                # 5-class peer can only confirm the defeature family, not which of the
                # 27 classes it is, so both sides are collapsed before comparing.
                # Comparing a 27-class id against a 5-class id directly would make
                # "agreement" meaningless -- the ids do not denote the same thing.
                peer = load_peer_preds(
                    args.agree_inference_dir, step_path.stem,
                    collapse=(args.agree_space == "27"))
                coarse = (map_labels_to_defeature(res[0])
                          if args.agree_space == "5" else res[0])
                labels, masked, resc = apply_thresholds_with_agreement(
                    res[0], res[1], thresholds,
                    peer=(peer[0], peer[1]) if peer else None,
                    agree_floor=args.agree_floor,
                    coarse_self=coarse)
                total_rescued += resc
                if peer is None:
                    peer_missing += 1
            else:
                labels, masked = apply_thresholds(res[0], res[1], thresholds)
        else:
            result = load_face_labels(infer_json, args.min_confidence)
            if not result or not result[0]:
                missing_labels += 1
                continue
            labels, masked = result

        kept_frac = 1.0 - (masked / len(labels)) if labels else 0.0
        if kept_frac < args.min_kept_frac:
            skipped_sparse += 1
            continue

        total_faces += len(labels)
        total_masked += masked
        if args.label_space == "5":
            # -1 passes through map_labels_to_defeature unchanged, so the mask survives.
            labels = map_labels_to_defeature(labels)
        target_step = steps_out / f"{model_id}{step_path.suffix.lower()}"
        shutil.copyfile(step_path, target_step)
        write_label_file(labels_out, model_id, labels, args.source_tag)
        ok += 1

    print(f"merged: kept={ok} missing_labels={missing_labels} "
          f"skipped_too_sparse={skipped_sparse}")
    if args.agree_inference_dir is not None:
        print(f"  second-teacher agreement ({args.agree_inference_dir}, "
              f"{args.agree_space}-class, floor {args.agree_floor}): "
              f"{total_rescued} faces rescued from masking"
              + (f"; {peer_missing} models had no peer prediction"
                 if peer_missing else ""))
    if thresholds is not None and total_faces:
        print(f"  per-class masking (top {args.per_class_top_frac:.0%}): "
              f"{total_masked}/{total_faces} faces masked to -1 "
              f"({100 * total_masked / total_faces:.1f}%), "
              f"{total_faces - total_masked} supervised")
    elif args.min_confidence > 0 and total_faces:
        print(f"  confidence masking @ {args.min_confidence}: "
              f"{total_masked}/{total_faces} faces masked to -1 "
              f"({100 * total_masked / total_faces:.1f}%), "
              f"{total_faces - total_masked} supervised")
    else:
        print("  confidence masking DISABLED — training on raw argmax")
    print(f"  steps: {steps_out}")
    print(f"  labels: {labels_out}")


if __name__ == "__main__":
    main()
