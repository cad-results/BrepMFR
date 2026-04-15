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

    # Post-processing
    parser.add_argument("--real_classes", action="store_true",
                        help="Remap 27 MFTRCAD classes to 8 real machining feature categories")

    # Limit data
    parser.add_argument("--limit_data_manifest", type=str, default=None,
                        help="Path to limit_data_manifest.json to filter STEP files in batch mode")

    return parser.parse_args()


def load_model(checkpoint_path: str):
    """Load trained BrepFormer model from checkpoint."""
    import pathlib
    from brepformer.configs.config import BrepClassifierConfig
    from brepformer.models.brep_classifier import BrepClassifier

    torch.serialization.add_safe_globals([pathlib.PosixPath, BrepClassifierConfig])
    model = BrepClassifier.load_from_checkpoint(checkpoint_path)
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
        for split in ["test", "val", "train"]:
            split_dir = pdir / split
            if not split_dir.is_dir():
                continue
            for pkl_file in split_dir.iterdir():
                if not pkl_file.suffix == ".pkl":
                    continue
                with open(pkl_file, "rb") as f:
                    sample = pickle.load(f)
                if sample["model_id"] == clean_id or sample["model_id"] == model_id:
                    # Convert numpy to tensors for prepare_batch
                    return sample
    return None


def infer_single(model, step_path: str, device: torch.device) -> dict:
    """Run inference on a single STEP file.

    Tries to load from preprocessed pickle data first (exact training format).
    Falls back to step_to_graph conversion for unknown STEP files.

    Returns:
        Dictionary with model_id, num_faces, model_classes, face_preds, face_probs.
    """
    # Try preprocessed data first (exact match with training)
    sample = load_sample_from_preprocessed(step_path)

    if sample is None:
        # Fall back to step_to_graph conversion (for new STEP files)
        from brepformer.data.step_to_graph import step_to_preprocessed_sample
        sample = step_to_preprocessed_sample(step_path)

    if sample is None:
        return {"error": f"Failed to convert {step_path}"}

    batch = prepare_batch(sample, device)
    num_faces = sample["num_faces"]
    model_id = sample["model_id"]

    with torch.no_grad():
        output = model(batch)

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


def _remap_result_to_real(result: dict) -> dict:
    """Remap a 27-class inference result to 8 real classes."""
    if "face_preds" in result:
        result["face_preds"] = map_labels_to_real(result["face_preds"])
        result["face_class_names"] = [REAL_CLASS_NAMES[p] for p in result["face_preds"]]
    if "model_classes" in result:
        # Deduplicate after remapping
        real_model = sorted(set(CLASS_TO_REAL_CLASS[c] for c in result["model_classes"]))
        result["model_classes"] = real_model
        result["model_class_names"] = [REAL_CLASS_NAMES[c] for c in real_model]
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
        result = infer_single(model, args.step, device)

        if "error" in result:
            print(f"Error: {result['error']}")
            sys.exit(1)

        # Remap to real classes if requested
        if args.real_classes:
            result = _remap_result_to_real(result)

        # Print summary
        class_names = REAL_CLASS_NAMES if args.real_classes else CLASS_NAMES
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
            print(f"  Processing {step_file.name}...")
            result = infer_single(model, str(step_file), device)
            all_results.append(result)

            if "error" not in result and args.real_classes:
                result = _remap_result_to_real(result)

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
