#!/usr/bin/env python3
"""Batch STEP to graph JSON converter for pipe fitting dataset.

Scans class folders in ssdata1/, converts each STEP file to brepformer
graph JSON format, and builds labels.json and metadata.json.

Usage:
    python brepclassifier/convert_steps.py \
        --data_dir brepclassifier/data/ssdata1 \
        --num_workers 4
"""

import argparse
import json
import os
import re
import shutil
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Class folder mapping
CLASS_FOLDERS = {
    "1_elbow_wf": (0, "Elbow - Weld Fitting"),
    "2_elbow_pef": (1, "Elbow - Pipe End Fitting"),
    "3_elbow_sf": (2, "Elbow - Socket Fitting"),
    "4_tee_wf": (3, "Tee - Weld Fitting"),
    "5_tee_pef": (4, "Tee - Pipe End Fitting"),
    "6_tee_sf": (5, "Tee - Socket Fitting"),
    "8_elbow_misc": (6, "Elbow - Miscellaneous"),
    "9_tee_misc": (7, "Tee - Miscellaneous"),
}


def sanitize_filename(name: str) -> str:
    """Convert a filename to a safe model ID.

    Removes special characters, spaces, parentheses, etc.

    Args:
        name: Original filename (without extension).

    Returns:
        Sanitized model ID string.
    """
    # Remove file extensions
    name = re.sub(r'\.(step|stp|ipt)$', '', name, flags=re.IGNORECASE)
    # Replace special chars with underscore
    name = re.sub(r'[^a-zA-Z0-9_.-]', '_', name)
    # Collapse multiple underscores
    name = re.sub(r'_+', '_', name)
    # Strip leading/trailing underscores
    name = name.strip('_')
    # Limit length
    if len(name) > 100:
        name = name[:100]
    return name


def convert_single_file(args_tuple):
    """Convert a single STEP file to graph JSON.

    Args:
        args_tuple: (step_path, model_id, output_dir)

    Returns:
        (model_id, success, error_msg)
    """
    step_path, model_id, output_dir = args_tuple

    try:
        from brepclassifier.data.step_to_graph import step_to_graph

        result = step_to_graph(str(step_path), model_id)
        if result is None:
            return (model_id, False, "step_to_graph returned None")

        # Save JSON
        out_path = Path(output_dir) / f"{model_id}.json"
        with open(out_path, "w") as f:
            json.dump(result, f)

        return (model_id, True, None)

    except Exception as e:
        return (model_id, False, f"{type(e).__name__}: {str(e)}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert STEP files to graph JSON")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="brepclassifier/data/ssdata1",
        help="Directory containing class folders with STEP files",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers (default 1 for safety)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to convert (for testing)",
    )
    parser.add_argument(
        "--skip_copy",
        action="store_true",
        help="Skip copying STEP files to steps/ directory",
    )

    return parser.parse_args()


def main():
    """Main conversion function."""
    args = parse_args()
    data_dir = Path(args.data_dir)

    # Create output directories
    graphs_dir = data_dir / "graphs"
    steps_dir = data_dir / "steps"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    steps_dir.mkdir(parents=True, exist_ok=True)

    # Scan class folders and build file list
    labels = {}
    file_list = []
    class_counts = {}

    print("Scanning class folders...")
    for folder_name, (class_idx, class_name) in CLASS_FOLDERS.items():
        folder_path = data_dir / folder_name
        if not folder_path.exists():
            print(f"  Warning: {folder_path} not found, skipping")
            continue

        step_files = list(folder_path.glob("*.step")) + list(folder_path.glob("*.STEP")) + \
                     list(folder_path.glob("*.stp")) + list(folder_path.glob("*.STP"))

        class_counts[class_name] = len(step_files)
        print(f"  {folder_name}: {len(step_files)} files -> class {class_idx} ({class_name})")

        for step_file in step_files:
            model_id = sanitize_filename(step_file.stem)

            # Handle duplicate model IDs
            if model_id in labels:
                suffix = 1
                while f"{model_id}_{suffix}" in labels:
                    suffix += 1
                model_id = f"{model_id}_{suffix}"

            labels[model_id] = class_idx
            file_list.append((step_file, model_id, str(graphs_dir)))

    print(f"\nTotal: {len(file_list)} STEP files across {len(class_counts)} classes")

    # Apply limit if specified
    if args.limit is not None:
        file_list = file_list[:args.limit]
        print(f"Limited to {len(file_list)} files")

    # Convert files
    print("\nConverting STEP files to graph JSON...")
    successes = 0
    failures = 0
    errors = []

    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(convert_single_file, file_list),
                total=len(file_list),
            ))
    else:
        results = []
        for item in tqdm(file_list):
            results.append(convert_single_file(item))

    for model_id, success, error_msg in results:
        if success:
            successes += 1
        else:
            failures += 1
            errors.append((model_id, error_msg))

    print(f"\nConversion complete: {successes} succeeded, {failures} failed")

    if errors:
        print(f"\nFirst 10 errors:")
        for model_id, error_msg in errors[:10]:
            print(f"  {model_id}: {error_msg}")

    # Remove failed models from labels
    successful_ids = {mid for mid, success, _ in results if success}
    labels = {mid: cls for mid, cls in labels.items() if mid in successful_ids}

    # Copy STEP files to steps/ directory
    if not args.skip_copy:
        print("\nCopying STEP files to steps/...")
        for step_file, model_id, _ in tqdm(file_list):
            if model_id in successful_ids:
                dst = steps_dir / f"{model_id}.step"
                try:
                    shutil.copy2(step_file, dst)
                except Exception as e:
                    print(f"  Error copying {step_file}: {e}")

    # Save labels.json
    labels_path = data_dir / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"\nSaved labels for {len(labels)} models to {labels_path}")

    # Save metadata.json
    metadata = {
        "num_classes": 8,
        "num_samples": len(labels),
        "class_names": [name for _, (_, name) in sorted(CLASS_FOLDERS.items(), key=lambda x: x[1][0])],
        "class_counts": {
            str(idx): sum(1 for v in labels.values() if v == idx)
            for idx in range(8)
        },
        "class_folders": {name: idx for name, (idx, _) in CLASS_FOLDERS.items()},
        "conversion_stats": {
            "total_files": len(file_list),
            "successes": successes,
            "failures": failures,
        },
    }
    metadata_path = data_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    # Print class distribution
    print("\nClass distribution:")
    for idx in range(8):
        count = sum(1 for v in labels.values() if v == idx)
        name = metadata["class_names"][idx]
        print(f"  {idx}: {name:<30s} {count:>5d}")


if __name__ == "__main__":
    main()
