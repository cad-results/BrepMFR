#!/usr/bin/env python3
"""Prepare defeature dataset for BrepFormer training.

Copies STEP files and converts text labels from the navin_defeaturing dataset
into the format expected by the BrepFormer preprocessing pipeline.

Label remapping (7 original -> 5 target classes):
    0 -> 0  Random (Other)
    1 -> 1  Hole
    2 -> 1  Hole
    3 -> 2  Chamfer
    4 -> 3  Fillet
    5 -> 4  Cut
    6 -> 4  Cut

Usage:
    python brepformer/data/prepare_defeature.py \\
        --source /mnt/c/projects/data/navin_defeaturing \\
        --dest brepformer/data/defeature

    # Also convert STEP files to graph JSONs (requires pythonocc):
    python brepformer/data/prepare_defeature.py \\
        --source /mnt/c/projects/data/navin_defeaturing \\
        --dest brepformer/data/defeature \\
        --convert_graphs
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from collections import defaultdict

# Label remapping: original class -> defeature class
LABEL_REMAP = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 4}

DEFEATURE_CLASS_NAMES = ["random", "hole", "chamfer", "fillet", "cut"]
DEFEATURE_NUM_CLASSES = 5


def sanitize_filename(name: str) -> str:
    """Sanitize a filename by replacing problematic characters."""
    # Replace spaces, #, and other special chars with underscores
    name = re.sub(r'[#\s\(\)\[\]\{\}&$!@%^+=,;\'\"]+', '_', name)
    # Collapse multiple underscores
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    return name


def find_pairs(source_dir: Path):
    """Find all STEP + TXT label file pairs recursively.

    Returns list of (step_path, txt_path, relative_source_dir).
    """
    pairs = []
    for root, dirs, files in os.walk(source_dir):
        root_path = Path(root)
        # Build a map of basenames (without extension) to files
        step_files = {}
        txt_files = {}
        for f in files:
            stem = Path(f).stem
            ext = Path(f).suffix.lower()
            if ext == '.step':
                step_files[stem] = root_path / f
            elif ext == '.txt' and f != 'reformatter-labels.txt':
                txt_files[stem] = root_path / f

        # Match pairs
        for stem in step_files:
            if stem in txt_files:
                rel = root_path.relative_to(source_dir)
                pairs.append((step_files[stem], txt_files[stem], str(rel)))

    return pairs


def read_txt_labels(txt_path: Path) -> list:
    """Read per-face labels from a text file (one int per line)."""
    labels = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(int(line))
    return labels


def remap_labels(labels: list) -> list:
    """Remap original 7-class labels to 5-class defeature labels."""
    return [LABEL_REMAP.get(l, 0) for l in labels]


def labels_to_json(labels: list) -> dict:
    """Convert a list of per-face labels to the JSON format expected by preprocess.py.

    Returns {"cls": {"0": class_id, "1": class_id, ...}}
    """
    cls_dict = {str(i): label for i, label in enumerate(labels)}
    return {"cls": cls_dict}


def convert_step_to_graph(step_path: str) -> dict:
    """Convert a STEP file to graph JSON format using step_to_graph."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from brepformer.data.step_to_graph import step_to_graph

    data = step_to_graph(step_path)
    if data is None:
        return None

    return {
        "graph": {
            "edges": data["edge_index"].tolist(),
            "num_nodes": data["num_nodes"],
        },
        "graph_face_attr": data["face_attr"].tolist(),
        "graph_face_grid": data["face_grid"].tolist(),
        "graph_edge_attr": data["edge_attr"].tolist(),
        "graph_edge_grid": data["edge_grid"].tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare defeature dataset")
    parser.add_argument("--source", type=str, required=True,
                        help="Source directory (navin_defeaturing)")
    parser.add_argument("--dest", type=str, required=True,
                        help="Destination directory (brepformer/data/defeature)")
    parser.add_argument("--convert_graphs", action="store_true",
                        help="Also convert STEP files to graph JSONs (requires pythonocc)")
    args = parser.parse_args()

    source_dir = Path(args.source)
    dest_dir = Path(args.dest)

    if not source_dir.exists():
        print(f"Error: Source directory {source_dir} does not exist")
        sys.exit(1)

    # Create output directories
    steps_dir = dest_dir / "steps"
    labels_dir = dest_dir / "labels"
    steps_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    if args.convert_graphs:
        graphs_dir = dest_dir / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)

    # Find all STEP + TXT pairs
    pairs = find_pairs(source_dir)
    print(f"Found {len(pairs)} STEP+TXT pairs")

    # Handle duplicate filenames by tracking used names
    used_names = defaultdict(int)
    name_map = {}  # original_path -> sanitized_name

    # First pass: assign unique sanitized names
    for step_path, txt_path, rel_dir in pairs:
        base = sanitize_filename(step_path.stem)
        if not base:
            base = "unnamed"
        used_names[base] += 1

    # Reset and assign with dedup
    name_counts = defaultdict(int)
    assignments = []
    for step_path, txt_path, rel_dir in pairs:
        base = sanitize_filename(step_path.stem)
        if not base:
            base = "unnamed"

        if used_names[base] > 1:
            name_counts[base] += 1
            unique_name = f"{base}_{name_counts[base]}"
        else:
            unique_name = base

        assignments.append((step_path, txt_path, unique_name))

    # Second pass: copy files and convert labels
    stats = {"copied": 0, "skipped_empty": 0, "skipped_invalid": 0, "graphs_ok": 0, "graphs_fail": 0}
    label_distribution = defaultdict(int)

    from tqdm import tqdm

    for step_path, txt_path, unique_name in tqdm(assignments, desc="Copying"):
        # Read and remap labels
        try:
            original_labels = read_txt_labels(txt_path)
        except Exception as e:
            print(f"  Warning: Cannot read {txt_path}: {e}")
            stats["skipped_invalid"] += 1
            continue

        if not original_labels:
            stats["skipped_empty"] += 1
            continue

        remapped_labels = remap_labels(original_labels)

        # Count distribution
        for l in remapped_labels:
            label_distribution[l] += 1

        # Copy STEP file
        dest_step = steps_dir / f"{unique_name}.step"
        shutil.copy2(str(step_path), str(dest_step))

        # Write label JSON
        label_json = labels_to_json(remapped_labels)
        dest_label = labels_dir / f"{unique_name}.json"
        with open(dest_label, "w") as f:
            json.dump(label_json, f)

        stats["copied"] += 1

    print(f"\nCopied {stats['copied']} models")
    print(f"Skipped: {stats['skipped_empty']} empty, {stats['skipped_invalid']} invalid")

    print(f"\nLabel distribution (remapped {DEFEATURE_NUM_CLASSES} classes):")
    total_faces = sum(label_distribution.values())
    for cls_id in sorted(label_distribution.keys()):
        count = label_distribution[cls_id]
        name = DEFEATURE_CLASS_NAMES[cls_id] if cls_id < DEFEATURE_NUM_CLASSES else f"unknown_{cls_id}"
        pct = 100.0 * count / total_faces if total_faces > 0 else 0
        print(f"  {cls_id} ({name:>10s}): {count:>8d} faces ({pct:5.1f}%)")

    # Convert STEP files to graph JSONs
    if args.convert_graphs:
        print(f"\nConverting STEP files to graph JSONs...")
        graph_files = sorted(steps_dir.glob("*.step"))
        for step_file in tqdm(graph_files, desc="Converting"):
            model_id = step_file.stem
            try:
                graph_data = convert_step_to_graph(str(step_file))
                if graph_data is None:
                    stats["graphs_fail"] += 1
                    continue

                out = [model_id, graph_data]
                with open(graphs_dir / f"{model_id}.json", "w") as f:
                    json.dump(out, f)
                stats["graphs_ok"] += 1
            except Exception as e:
                print(f"  Warning: Failed to convert {model_id}: {e}")
                stats["graphs_fail"] += 1

        print(f"\nGraph conversion: {stats['graphs_ok']} ok, {stats['graphs_fail']} failed")

    # Write metadata
    metadata = {
        "dataset": "navin_defeaturing",
        "num_classes": DEFEATURE_NUM_CLASSES,
        "class_names": DEFEATURE_CLASS_NAMES,
        "label_remap": {str(k): v for k, v in LABEL_REMAP.items()},
        "num_models": stats["copied"],
        "total_faces": total_faces,
        "label_distribution": {
            DEFEATURE_CLASS_NAMES[k]: v
            for k, v in sorted(label_distribution.items())
            if k < DEFEATURE_NUM_CLASSES
        },
    }
    with open(dest_dir / "dataset_info.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved dataset info to {dest_dir / 'dataset_info.json'}")
    print("Done!")


if __name__ == "__main__":
    main()
