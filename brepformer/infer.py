#!/usr/bin/env python3
"""End-to-end inference pipeline for BrepFormer.

Converts STEP files to graph representation, runs through a trained
BrepFormer model, and outputs per-face predictions.

Usage:
    # Single file
    python -m brepformer.infer --step model.step --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt --output preds.json

    # Batch mode
    python -m brepformer.infer --step_dir steps/ --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt --output_dir results/

    # Output .seg format (one label per line)
    python -m brepformer.infer --step model.step --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt --output preds.seg
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from brepformer.data.classes import (
    CLASS_NAMES, NUM_CLASSES,
    REAL_CLASS_NAMES, REAL_NUM_CLASSES, CLASS_TO_REAL_CLASS, map_labels_to_real,
    DEFEATURE_CLASS_NAMES, DEFEATURE_NUM_CLASSES, CLASS_TO_DEFEATURE, map_labels_to_defeature,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="BrepFormer inference on STEP files")

    # Input
    parser.add_argument("--step", type=str, default=None, help="Path to a single STEP file")
    parser.add_argument("--step_dir", type=str, default=None, help="Directory of STEP files for batch mode")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")

    # Output
    parser.add_argument("--output", type=str, default=None, help="Output file path (.json or .seg)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for batch mode")

    # Post-processing (mutually exclusive class remapping)
    class_group = parser.add_mutually_exclusive_group()
    class_group.add_argument("--8_classes", action="store_true", dest="eight_classes",
                             help="Remap 27 MFTRCAD classes to 8 grouped machining feature categories")
    class_group.add_argument("--5_classes", action="store_true", dest="five_classes",
                             help="Remap 27 MFTRCAD classes to 5 defeature categories (random, hole, chamfer, fillet, cut)")
    parser.add_argument("--coarse_decision", choices=["argmax", "balanced"], default="argmax",
                        help="How to pick a coarse label once the 27-class posterior has "
                             "been marginalized into coarse groups. 'argmax' takes the "
                             "largest marginal, which minimises 0-1 loss. Note the 27->5 "
                             "groups are wildly uneven -- cut absorbs 21 of 27 fine "
                             "classes, random and fillet one each -- so diffuse "
                             "probability mass lands on cut by construction and rare "
                             "classes are crushed. 'balanced' divides each marginal by "
                             "its group size before the argmax, which is the right rule "
                             "when the metric is macro-F1 rather than accuracy.")

    # Limit data
    parser.add_argument("--no_preprocessed", action="store_true",
                        help="Skip the preprocessed-pickle lookup entirely. Use for a "
                             "corpus whose model_ids cannot appear in mftrcad (e.g. "
                             "GrabCAD slugs), where the lookup can only ever miss.")
    parser.add_argument("--preprocessed_dir", type=str, default=None,
                        help="Preprocessed dir (with train/val/test pickle subdirs) to look "
                             "features up in. Without this the search is hard-coded to "
                             "`mftrcad_processed` paths, so inferring on any other corpus "
                             "with its own pickles -- defeature_processed, for instance -- "
                             "silently misses and falls back to re-running step_to_graph, "
                             "which is ~50x slower AND uses different features from the "
                             "ones the checkpoint trained on.")
    parser.add_argument("--limit_data_manifest", type=str, default=None,
                        help="Path to limit_data_manifest.json to filter STEP files in batch mode")

    return parser.parse_args()


def load_model(checkpoint_path: str, attr_layout: str = None):
    """Load trained BrepFormer model from checkpoint.

    ``attr_layout`` decides how ``face_attr``/``edge_attr`` columns are decoded, and the right
    answer depends on when the checkpoint was trained rather than on what we would prefer today.
    Checkpoints saved before the layout fix were trained on the *categorical* indexing — surface
    type as an "is it a plane" bit, edge length reading the convexity flag, edge convexity a
    constant — and their weights are adapted to exactly those inputs. Silently loading them
    under the corrected ``"onehot"`` decoding changes the inputs underneath trained weights and
    makes them look worse than they are.

    So a checkpoint whose saved config has no ``attr_layout`` field predates the fix and is
    loaded as ``"categorical"``. Pass ``attr_layout`` explicitly only to deliberately measure a
    checkpoint under the other decoding.
    """
    import pathlib
    from brepformer.configs.config import BrepClassifierConfig
    from brepformer.models.brep_classifier import BrepClassifier

    torch.serialization.add_safe_globals([pathlib.PosixPath, BrepClassifierConfig])

    if attr_layout is None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved = (ckpt.get("hyper_parameters") or {}).get("config")
        # Probe the instance __dict__, NOT getattr. Unpickling a dataclass rebuilds it against
        # the *current* class definition, so `getattr(saved, "attr_layout")` happily returns the
        # new field's default even for a config pickled before the field existed. Only the keys
        # actually restored into the instance dict distinguish the two eras.
        attr_layout = (vars(saved).get("attr_layout") if saved is not None else None) \
            or "categorical"
        del ckpt

    model = BrepClassifier.load_from_checkpoint(checkpoint_path)
    model.config.attr_layout = attr_layout
    model.encoder.node_feature.attr_layout = attr_layout
    model.encoder.attn_bias.attr_layout = attr_layout
    model.eval()
    return model


def prepare_batch(sample: dict, device: torch.device) -> dict:
    """Convert a single sample to a batched tensor dict for model input."""
    from brepformer.data.collator import BrepCollator

    # Wrap numpy arrays as tensors
    def _to_tensor(v):
        if isinstance(v, np.ndarray):
            return torch.from_numpy(v)
        if isinstance(v, torch.Tensor):
            return v
        return v

    tensor_sample = {
        "face_grid": _to_tensor(sample["face_grid"]),
        "face_attr": _to_tensor(sample["face_attr"]),
        "edge_index": _to_tensor(sample["edge_index"]),
        "edge_attr": _to_tensor(sample["edge_attr"]),
        "edge_grid": _to_tensor(sample["edge_grid"]),
        "spatial_pos": _to_tensor(sample["spatial_pos"]),
        "in_degree": _to_tensor(sample["in_degree"]),
        "label": _to_tensor(sample["label"]),
        "model_id": sample["model_id"],
        "num_faces": sample["num_faces"],
        "num_edges": sample["num_edges"],
    }
    if "d2_distance" in sample:
        tensor_sample["d2_distance"] = _to_tensor(sample["d2_distance"])
    if "angle_distance" in sample:
        tensor_sample["angle_distance"] = _to_tensor(sample["angle_distance"])

    collator = BrepCollator()
    batch = collator([tensor_sample])

    # Move tensors to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}
    return batch


def load_sample_from_preprocessed(step_path: str, preprocessed_dir: str = None) -> dict:
    """Load a preprocessed sample from pickle files (exact training format match).

    Searches preprocessed train/val/test splits for a matching model_id.
    This guarantees identical features to what the model saw during training.

    Args:
        step_path: Path to the STEP file (used to derive model_id).
        preprocessed_dir: Directory containing train/val/test subdirs with pickles.
                          If None, tries common locations relative to the STEP file.

    Returns:
        Sample dict with torch tensors, or None if not found.
    """
    import pickle

    model_id = Path(step_path).stem
    # Strip _result suffix if present (pickles use IDs without it)
    clean_id = model_id.replace("_result", "")

    # Find preprocessed dir
    if preprocessed_dir is None:
        step_dir = Path(step_path).parent
        candidates = [
            step_dir.parent.parent / "data" / "mftrcad_processed",
            step_dir.parent / "mftrcad_processed",
            Path("brepformer/data/mftrcad_processed"),
        ]
    else:
        candidates = [Path(preprocessed_dir)]

    for pdir in candidates:
        if not pdir.is_dir():
            continue
        index = _preprocessed_index(pdir)
        hit = index.get(clean_id) or index.get(model_id)
        if hit is None:
            continue
        with open(hit, "rb") as f:
            return pickle.load(f)
    return None


#: model_id -> pickle path, per preprocessed dir. Built once per process.
_INDEX_CACHE: dict[str, dict[str, str]] = {}


def _preprocessed_index(pdir: Path) -> dict[str, str]:
    """Map model_id -> pickle path for one preprocessed dir, built once.

    Why an index rather than a scan: pickle filenames are sequential
    (``00000.pkl``), so the ``model_id`` lives only *inside* the file and the
    lookup genuinely has to read them. The original code did that scan **per
    STEP file**, unpickling up to 28,555 files / 17 GB every time, and — because
    mftrcad ids look like ``20240116_231044_1009`` while hard-samples ids are
    GrabCAD slugs — the two id spaces are disjoint, so it always scanned
    everything and always returned None. That was ~90 s of the ~95 s spent per
    file, with the GPU idle: a full corpus pass took ~25 h instead of ~4.

    The index is memoised in-process and cached on disk, so the scan happens
    once ever rather than once per file. Behaviour is unchanged — the same
    ``model_id`` still resolves to the same pickle.

    Paths are stored **absolute**, and a cache holding relative ones is discarded. An
    earlier version stored whatever path it happened to walk, which made the on-disk cache
    valid only from the cwd that wrote it: read from anywhere else, the same relative path
    resolved to a *different corpus'* pickles, and the loader returned another model's
    features -- right shapes, wrong part, no error. That was observed: an index built for
    `defeature_processed` handed back a 27-face mftrcad sample for a 15-face defeature
    model. A silent feature/label mismatch is the worst failure mode this file has, so the
    cache is now self-validating rather than trusted.
    """
    key = str(pdir.resolve())
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key]

    import json
    import pickle as _pickle

    cache_file = pdir / "model_id_index.json"
    if cache_file.is_file():
        try:
            cached = json.loads(cache_file.read_text())
            # Every entry must be an absolute path that exists and lies inside this
            # preprocessed dir. Any other cache is from a different cwd or a different
            # corpus and would resolve to the wrong pickles.
            root = pdir.resolve()
            if cached and all(
                    Path(v).is_absolute() and Path(v).is_file()
                    and root in Path(v).resolve().parents
                    for v in cached.values()):
                _INDEX_CACHE[key] = cached
                return cached
            print(f"  discarding stale index cache {cache_file} "
                  f"(not absolute paths under {root})")
        except Exception:  # noqa: BLE001
            pass  # Corrupt or stale cache: rebuild below.

    index: dict[str, str] = {}
    for split in ["test", "val", "train"]:
        split_dir = pdir / split
        if not split_dir.is_dir():
            continue
        for pkl_file in sorted(split_dir.iterdir()):
            if pkl_file.suffix != ".pkl":
                continue
            try:
                with open(pkl_file, "rb") as f:
                    mid = _pickle.load(f)["model_id"]
            except Exception:  # noqa: BLE001
                continue
            index.setdefault(str(mid), str(pkl_file.resolve()))

    _INDEX_CACHE[key] = index
    try:
        cache_file.write_text(json.dumps(index))
    except Exception:  # noqa: BLE001
        pass  # A read-only data dir is fine; the in-process memo still helps.
    return index


def infer_single(model, step_path: str, device: torch.device,
                 use_preprocessed: bool = True,
                 preprocessed_dir: str = None) -> dict:
    """Run inference on a single STEP file.

    Tries to load from preprocessed pickle data first (exact training format).
    Falls back to step_to_graph conversion for unknown STEP files.

    Returns:
        Dictionary with model_id, num_faces, model_classes, face_preds, face_probs.
    """
    import traceback

    if not Path(step_path).exists():
        return {"error": f"STEP file not found: {step_path}"}

    # Try preprocessed data first (exact match with training). Skippable: for a
    # corpus whose ids cannot appear in mftrcad (e.g. GrabCAD slugs) the lookup
    # can only ever miss, so paying for it is pure waste.
    sample = (load_sample_from_preprocessed(step_path, preprocessed_dir)
              if use_preprocessed else None)

    if sample is None:
        # Fall back to step_to_graph conversion (for new STEP files)
        from brepformer.data.step_to_graph import step_to_preprocessed_sample
        try:
            sample = step_to_preprocessed_sample(step_path)
        except Exception as e:
            traceback.print_exc()
            return {"error": f"step_to_graph conversion failed for {step_path}: {e}"}

    if sample is None:
        return {"error": f"Failed to convert {step_path} (step_to_graph returned None)"}

    try:
        batch = prepare_batch(sample, device)
    except Exception as e:
        traceback.print_exc()
        return {"error": f"Failed to prepare batch for {step_path}: {e}"}

    num_faces = sample["num_faces"]
    model_id = sample["model_id"]

    try:
        with torch.no_grad():
            output = model(batch)
    except Exception as e:
        traceback.print_exc()
        return {"error": f"Model forward pass failed for {step_path}: {e}"}

    result = {"model_id": model_id, "num_faces": num_faces}

    if isinstance(output, dict):
        # Face segmentation mode
        model_logits = output["model_logits"][0]
        face_logits = output["face_logits"][0, :num_faces]  # (N, C)

        # Model-level predictions
        if model.config.multi_label:
            model_probs = torch.sigmoid(model_logits).cpu().numpy()
            model_classes = [int(i) for i in np.where(model_probs > 0.5)[0]]
        else:
            model_probs = torch.softmax(model_logits, dim=-1).cpu().numpy()
            model_classes = [int(model_logits.argmax().item())]

        # Face-level predictions
        face_probs = torch.softmax(face_logits, dim=-1).cpu().numpy()
        face_preds = face_logits.argmax(dim=-1).cpu().numpy().tolist()

        result["model_classes"] = model_classes
        result["model_class_names"] = [CLASS_NAMES[c] for c in model_classes]
        result["face_preds"] = face_preds
        result["face_class_names"] = [CLASS_NAMES[p] for p in face_preds]
        result["face_probs"] = face_probs.tolist()
    else:
        # Model-only mode
        logits = output[0]
        if model.config.multi_label:
            probs = torch.sigmoid(logits).cpu().numpy()
            model_classes = [int(i) for i in np.where(probs > 0.5)[0]]
        else:
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            model_classes = [int(logits.argmax().item())]

        result["model_classes"] = model_classes
        result["model_class_names"] = [CLASS_NAMES[c] for c in model_classes]

    return result


def save_result(result: dict, output_path: str):
    """Save inference result to file (.json or .seg)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".seg":
        # .seg format: one label per line
        if "face_preds" in result:
            with open(output_path, "w") as f:
                for label in result["face_preds"]:
                    f.write(f"{label}\n")
        else:
            print(f"Warning: No face predictions (model has no face segmentation head)")
    else:
        # JSON format
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

    print(f"Saved to {output_path}")


def _marginalize_probs(probs, fine_to_coarse, num_coarse: int) -> list:
    """Sum a fine-grained probability distribution into its coarse groups.

    ``probs`` is a list of per-face distributions over the fine classes and
    ``fine_to_coarse`` a sequence indexed by fine class (as ``CLASS_TO_DEFEATURE`` and
    ``CLASS_TO_REAL_CLASS`` both are). Each coarse class receives the total mass of the
    fine classes mapping onto it, giving the posterior over coarse labels implied by the
    fine posterior. Rows are renormalised as a guard against a distribution that is
    wider than the mapping.
    """
    out = []
    for row in probs:
        acc = [0.0] * num_coarse
        for fine, p in enumerate(row):
            if fine < len(fine_to_coarse):
                acc[fine_to_coarse[fine]] += float(p)
        total = sum(acc)
        out.append([a / total for a in acc] if total > 0 else acc)
    return out


def _coarse_argmax(coarse_probs, fine_to_coarse, num_coarse: int, rule: str) -> list:
    """Pick a coarse label per row from marginalized coarse probabilities.

    ``argmax`` is the plain Bayes rule for 0-1 loss. ``balanced`` first divides each
    coarse marginal by the number of fine classes feeding it, which cancels the
    structural advantage that a large group enjoys purely by absorbing diffuse mass --
    with 21 of 27 fine classes mapping to ``cut``, a uniform posterior already puts 78%
    on ``cut`` under the plain rule. Dividing by group size is the prior-correction that
    makes the decision optimal for balanced error / macro-F1 instead of accuracy.
    """
    if rule == "balanced":
        sizes = [0] * num_coarse
        for c in fine_to_coarse:
            sizes[c] += 1
        weights = [1.0 / s if s else 1.0 for s in sizes]
    else:
        weights = [1.0] * num_coarse
    return [int(max(range(num_coarse), key=lambda c: p[c] * weights[c])) for p in coarse_probs]


def _remap_result_to_real(result: dict, coarse_decision: str = "argmax") -> dict:
    """Remap a 27-class inference result to 8 real classes.

    Marginalizes then argmaxes, for the reason given in
    ``_remap_result_to_defeature``: the 27->8 collapse is many-to-one too.
    """
    if "face_preds" in result:
        probs = result.get("face_probs")
        if probs is not None:
            result["face_probs_27"] = probs
            n_real = len(REAL_CLASS_NAMES)
            coarse = _marginalize_probs(probs, CLASS_TO_REAL_CLASS, n_real)
            result["face_probs"] = coarse
            result["face_preds"] = _coarse_argmax(
                coarse, CLASS_TO_REAL_CLASS, n_real, coarse_decision)
        else:
            result["face_preds"] = map_labels_to_real(result["face_preds"])
        result["face_class_names"] = [REAL_CLASS_NAMES[p] for p in result["face_preds"]]
    if "model_classes" in result:
        # Deduplicate after remapping
        real_model = sorted(set(CLASS_TO_REAL_CLASS[c] for c in result["model_classes"]))
        result["model_classes"] = real_model
        result["model_class_names"] = [REAL_CLASS_NAMES[c] for c in real_model]
    return result


def _remap_result_to_defeature(result: dict, num_model_classes: int = 27,
                               coarse_decision: str = "argmax") -> dict:
    """Remap inference result to 5 defeature classes.

    If the model already outputs 5 classes (num_model_classes == 5), predictions
    are already in defeature space — only update display names, no remapping.
    For 27-class MFTRCAD models, apply CLASS_TO_DEFEATURE mapping.

    **Marginalize, then argmax.** The 27->5 collapse is many-to-one, so mapping the
    27-class argmax (``argmax-then-map``) is not the same as summing the probability
    mass of each defeature group and taking the largest (``marginalize-then-argmax``).
    Only the latter minimises 0-1 loss in the coarse space: five sibling classes each
    holding 0.15 of the mass beat a single rival holding 0.25, but argmax-then-map
    picks the rival. Measured over 334,175 faces, the two disagree on 35,626 (10.7%),
    overwhelmingly in favour of ``cut``. ``face_probs`` is marginalized alongside so
    downstream confidence gates read the same 5-wide space as ``face_preds``; the
    27-wide distribution is preserved as ``face_probs_27`` for auditing.
    """
    already_5class = (num_model_classes == DEFEATURE_NUM_CLASSES)

    if "face_preds" in result:
        probs = result.get("face_probs")
        if not already_5class and probs is not None:
            result["face_probs_27"] = probs
            coarse = _marginalize_probs(probs, CLASS_TO_DEFEATURE, DEFEATURE_NUM_CLASSES)
            result["face_probs"] = coarse
            result["face_preds"] = _coarse_argmax(
                coarse, CLASS_TO_DEFEATURE, DEFEATURE_NUM_CLASSES, coarse_decision)
        elif not already_5class:
            # No probabilities retained (e.g. --no_probs): fall back to argmax-then-map.
            result["face_preds"] = map_labels_to_defeature(result["face_preds"])
        result["face_class_names"] = [DEFEATURE_CLASS_NAMES[p] for p in result["face_preds"]]
    if "model_classes" in result:
        if not already_5class:
            defeature_model = sorted(set(CLASS_TO_DEFEATURE[c] for c in result["model_classes"]))
            result["model_classes"] = defeature_model
        result["model_class_names"] = [DEFEATURE_CLASS_NAMES[c] for c in result["model_classes"]]
    return result


def main():
    """Main inference function."""
    args = parse_args()

    if args.step is None and args.step_dir is None:
        print("Error: Provide either --step or --step_dir")
        sys.exit(1)

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    has_face_seg = model.config.face_segmentation
    print(f"Face segmentation: {'enabled' if has_face_seg else 'disabled'}")

    if args.step:
        # Single file mode
        print(f"\nProcessing {args.step}...")
        result = infer_single(model, args.step, device,
                              use_preprocessed=not args.no_preprocessed,
                              preprocessed_dir=args.preprocessed_dir)

        if "error" in result:
            print(f"Error: {result['error']}")
            sys.exit(1)

        # Remap classes if requested
        if args.eight_classes:
            result = _remap_result_to_real(result, args.coarse_decision)
        elif args.five_classes:
            result = _remap_result_to_defeature(result, model.config.num_face_classes,
                                                    args.coarse_decision)

        # Print summary
        if args.eight_classes:
            class_names = REAL_CLASS_NAMES
        elif args.five_classes:
            class_names = DEFEATURE_CLASS_NAMES
        else:
            class_names = CLASS_NAMES
        print(f"\nModel: {result['model_id']}")
        print(f"Faces: {result['num_faces']}")
        print(f"Model classes: {result.get('model_class_names', [])}")
        if "face_preds" in result:
            from collections import Counter
            counts = Counter(result["face_preds"])
            print(f"Face predictions ({len(result['face_preds'])} faces):")
            for cls_id, count in sorted(counts.items()):
                print(f"  {class_names[cls_id]}: {count}")

        if args.output:
            save_result(result, args.output)

    elif args.step_dir:
        # Batch mode
        step_dir = Path(args.step_dir)
        step_files = sorted(list(step_dir.glob("*.step")) + list(step_dir.glob("*.stp")))
        print(f"\nFound {len(step_files)} STEP files")

        if args.limit_data_manifest:
            import json as _json
            with open(args.limit_data_manifest) as _f:
                _manifest = _json.load(_f)
            allowed_ids = set(_manifest.get("model_ids", []))
            step_files = [f for f in step_files
                          if f.stem in allowed_ids or f.stem.replace("_result", "") in allowed_ids]
            print(f"Limit data: filtered to {len(step_files)} STEP files")

        output_dir = Path(args.output_dir) if args.output_dir else step_dir / "predictions"
        output_dir.mkdir(parents=True, exist_ok=True)

        all_results = []
        for step_file in step_files:
            _existing_seg = (Path(args.output_dir) if args.output_dir else step_dir / "predictions") / f"{step_file.stem}.seg"
            _existing_json = (Path(args.output_dir) if args.output_dir else step_dir / "predictions") / f"{step_file.stem}.json"
            if _existing_seg.exists() and _existing_json.exists():
                try:
                    with open(_existing_json) as _f:
                        all_results.append(json.load(_f))
                    print(f"  Skipping {step_file.name} (already processed)")
                    continue
                except Exception:
                    pass
            print(f"  Processing {step_file.name}...")
            result = infer_single(model, str(step_file), device,
                                  use_preprocessed=not args.no_preprocessed,
                                  preprocessed_dir=args.preprocessed_dir)
            all_results.append(result)

            if "error" not in result and args.eight_classes:
                result = _remap_result_to_real(result, args.coarse_decision)
            elif "error" not in result and args.five_classes:
                result = _remap_result_to_defeature(result, model.config.num_face_classes,
                                                    args.coarse_decision)

            if "error" not in result:
                # Save individual result
                out_path = output_dir / f"{step_file.stem}.json"
                save_result(result, str(out_path))

                # Also save .seg if face segmentation available
                if "face_preds" in result:
                    seg_path = output_dir / f"{step_file.stem}.seg"
                    save_result(result, str(seg_path))

        # Save combined results
        combined_path = output_dir / "all_predictions.json"
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nAll predictions saved to {combined_path}")


if __name__ == "__main__":
    main()
