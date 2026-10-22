#!/usr/bin/env python3
"""Verification script for symmetry completion in defeaturing.

Runs 4 specific test cases + 10 random STEP files, producing:
  - Before/after defeatured STEP files (with and without symmetry)
  - .seg files for visualization
  - Summary report

For visual verification, use the defeature viewer:
    python -m brepformer.visualize_defeature \
        --step_dir output/originals \
        --defeatured_dir output/with_symmetry \
        --seg_dir output/segs

Or compare without symmetry:
    python -m brepformer.visualize_defeature \
        --step_dir output/originals \
        --defeatured_dir output/without_symmetry \
        --seg_dir output/segs
"""

import sys
import os
import json
import random
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brepformer.defeature_v2 import (
    defeature_step, load_seg_labels, _extract_face_geometry,
    _compute_bbox_diagonal, _find_candidate_axes, _find_angular_orbits,
    _complete_symmetry,
)
from brepformer.data.classes import CLASS_TO_DEFEATURE, NUM_CLASSES
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods


def load_shape_and_faces(step_path):
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        return None, []
    reader.TransferRoots()
    shape = reader.OneShape()
    faces = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        faces.append(topods.Face(exp.Current()))
        exp.Next()
    return shape, faces


def save_seg(preds, path):
    with open(path, "w") as f:
        for p in preds:
            f.write(f"{p}\n")


def run_test(step_path, preds, output_dir, model_id, verbose=True):
    """Run defeaturing with and without symmetry, return results."""
    orig_dir = output_dir / "originals"
    sym_dir = output_dir / "with_symmetry"
    nosym_dir = output_dir / "without_symmetry"
    seg_dir = output_dir / "segs"

    for d in [orig_dir, sym_dir, nosym_dir, seg_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Copy original
    orig_dest = orig_dir / f"{model_id}.step"
    shutil.copy2(step_path, orig_dest)

    # Save seg
    save_seg(preds, seg_dir / f"{model_id}.seg")

    # With symmetry
    out_sym = sym_dir / f"{model_id}_defeatured.step"
    result_sym = defeature_step(
        str(step_path), preds, str(out_sym),
        verbose=verbose, symmetry=True,
    )

    # Without symmetry
    out_nosym = nosym_dir / f"{model_id}_defeatured.step"
    result_nosym = defeature_step(
        str(step_path), preds, str(out_nosym),
        verbose=verbose, symmetry=False,
    )

    return result_sym, result_nosym


def main():
    output_dir = Path("brepformer/symmetry_verification")
    output_dir.mkdir(parents=True, exist_ok=True)

    step_dir = Path("brepformer/data/mftrcad/steps")
    label_dir = Path("brepformer/data/mftrcad/labels")
    inference_dir = Path("brepformer/data/inference")
    pipeline_dir = Path("brepformer/pipeline_output")

    results = []

    # ---------------------------------------------------------------
    # Part 1: 4 specific test cases from pipeline output
    # ---------------------------------------------------------------
    print("=" * 70)
    print("PART 1: Specific test cases with real model predictions")
    print("=" * 70)

    specific_cases = [
        ("1117T11_Iron_Pipe_Fitting_with_Flanged_Ends",
         inference_dir / "1117T11_Iron_Pipe_Fitting_with_Flanged_Ends.step",
         pipeline_dir / "1117T11_Iron_Pipe_Fitting_with_Flanged_Ends/preds.seg"),
        ("000700694_Substitute_2020-07_22",
         inference_dir / "000700694_Substitute_2020-07_22.step",
         pipeline_dir / "000700694_Substitute_2020-07_22/preds.seg"),
        ("02280839",
         inference_dir / "02280839.step",
         pipeline_dir / "02280839/preds.seg"),
        ("05-gland",
         inference_dir / "05-gland.stp",
         pipeline_dir / "05-gland/preds.seg"),
    ]

    for model_id, step_path, seg_path in specific_cases:
        if not step_path.exists() or not seg_path.exists():
            print(f"\n  SKIP: {model_id} (files missing)")
            continue

        preds = load_seg_labels(str(seg_path))
        print(f"\n{'─' * 60}")
        print(f"  {model_id} ({len(preds)} faces)")
        print(f"{'─' * 60}")

        r_sym, r_nosym = run_test(step_path, preds, output_dir, model_id)

        entry = {
            "model_id": model_id,
            "num_faces": len(preds),
            "with_symmetry": r_sym,
            "without_symmetry": r_nosym,
            "type": "real_predictions",
        }
        results.append(entry)

        print(f"  With symmetry:    status={r_sym['status']}, "
              f"removed={r_sym.get('removed', 0)}, "
              f"symmetry_added={r_sym.get('symmetry_added', 0)}")
        print(f"  Without symmetry: status={r_nosym['status']}, "
              f"removed={r_nosym.get('removed', 0)}")

    # ---------------------------------------------------------------
    # Part 2: 10 random STEP files with GT labels
    # ---------------------------------------------------------------
    print(f"\n\n{'=' * 70}")
    print("PART 2: 10 random STEP files with ground-truth labels")
    print("=" * 70)

    all_steps = sorted(os.listdir(step_dir))
    random.seed(2024)
    random.shuffle(all_steps)

    tested = 0
    for step_file in all_steps:
        if tested >= 10:
            break
        stem = step_file.replace(".step", "")
        label_file = label_dir / f"{stem}.json"
        if not label_file.exists():
            continue

        step_path = step_dir / step_file
        with open(label_file) as f:
            label_data = json.load(f)

        cls_dict = label_data["cls"]
        num_faces = len(cls_dict)
        if num_faces < 20:
            continue

        preds_27 = [cls_dict[str(i)] for i in range(num_faces)]
        preds_5 = [
            CLASS_TO_DEFEATURE[p] if 0 <= p < NUM_CLASSES else 0
            for p in preds_27
        ]
        n_features = sum(1 for p in preds_5 if p != 0)
        if n_features < 3:
            continue

        print(f"\n{'─' * 60}")
        print(f"  {stem} ({num_faces} faces, {n_features} features)")
        print(f"{'─' * 60}")

        r_sym, r_nosym = run_test(
            step_path, preds_5, output_dir, stem, verbose=True)

        entry = {
            "model_id": stem,
            "num_faces": num_faces,
            "num_features": n_features,
            "with_symmetry": r_sym,
            "without_symmetry": r_nosym,
            "type": "random_gt",
        }
        results.append(entry)

        print(f"  With symmetry:    status={r_sym['status']}, "
              f"removed={r_sym.get('removed', 0)}, "
              f"symmetry_added={r_sym.get('symmetry_added', 0)}")
        print(f"  Without symmetry: status={r_nosym['status']}, "
              f"removed={r_nosym.get('removed', 0)}")
        tested += 1

    # ---------------------------------------------------------------
    # Part 3: Synthetic orbit-drop tests
    # ---------------------------------------------------------------
    print(f"\n\n{'=' * 70}")
    print("PART 3: Synthetic tests — drop 1 face from rotational orbits")
    print("=" * 70)

    orbit_models = [
        ("20240116_231044_0_result.step", [22, 23, 24, 25]),
        ("20240116_231044_1007_result.step", [38, 39, 48, 49]),
        ("20240116_231044_100_result.step", [19, 25, 31, 37]),
        ("20240116_231044_1023_result.step", [31, 33, 35, 37]),
        ("20240116_231044_102_result.step", [15, 18, 21, 24]),
    ]

    for step_file, orbit in orbit_models:
        stem = step_file.replace(".step", "")
        step_path = step_dir / step_file
        label_file = label_dir / f"{stem}.json"

        shape, faces = load_shape_and_faces(str(step_path))
        if shape is None:
            continue
        nf = len(faces)

        with open(label_file) as f:
            label_data = json.load(f)
        cls_dict = label_data["cls"]
        preds_5 = [
            CLASS_TO_DEFEATURE[cls_dict[str(i)]] if 0 <= cls_dict[str(i)] < NUM_CLASSES else 0
            for i in range(nf)
        ]

        # Force-mark orbit and drop one
        test_preds = list(preds_5)
        for idx in orbit:
            test_preds[idx] = 3
        dropped = orbit[-1]
        test_preds[dropped] = 0

        print(f"\n{'─' * 60}")
        print(f"  {stem}: orbit {orbit}, dropped face {dropped}")
        print(f"{'─' * 60}")

        # Run with symmetry (should recover)
        syn_id = f"{stem}_dropped{dropped}"
        r_sym, r_nosym = run_test(
            step_path, test_preds, output_dir, syn_id, verbose=True)

        recovered = r_sym.get("symmetry_added", 0) > 0
        print(f"  Recovered: {recovered}")
        print(f"  With symmetry:    removed={r_sym.get('removed', 0)}, "
              f"symmetry_added={r_sym.get('symmetry_added', 0)}")
        print(f"  Without symmetry: removed={r_nosym.get('removed', 0)}")

        entry = {
            "model_id": syn_id,
            "num_faces": nf,
            "orbit": orbit,
            "dropped_face": dropped,
            "recovered": recovered,
            "with_symmetry": r_sym,
            "without_symmetry": r_nosym,
            "type": "synthetic_orbit_drop",
        }
        results.append(entry)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print(f"\n\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)

    total = len(results)
    sym_additions = sum(
        r["with_symmetry"].get("symmetry_added", 0) for r in results)
    synthetic = [r for r in results if r["type"] == "synthetic_orbit_drop"]
    recovered = sum(1 for r in synthetic if r.get("recovered"))

    print(f"  Total models tested: {total}")
    print(f"  Total symmetry additions: {sym_additions}")
    print(f"  Synthetic orbit-drop tests: {len(synthetic)}")
    print(f"  Recovered via symmetry: {recovered}/{len(synthetic)}")
    print()
    print(f"  Output directory: {output_dir}")
    print()
    print("  To visualize (requires GUI):")
    print(f"    python -m brepformer.visualize_defeature \\")
    print(f"        --step_dir {output_dir}/originals \\")
    print(f"        --defeatured_dir {output_dir}/with_symmetry \\")
    print(f"        --seg_dir {output_dir}/segs")
    print()
    print("  To compare WITHOUT symmetry:")
    print(f"    python -m brepformer.visualize_defeature \\")
    print(f"        --step_dir {output_dir}/originals \\")
    print(f"        --defeatured_dir {output_dir}/without_symmetry \\")
    print(f"        --seg_dir {output_dir}/segs")

    # Save report
    report_path = output_dir / "verification_report.json"

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        return obj

    with open(report_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"\n  Report saved: {report_path}")


if __name__ == "__main__":
    main()
