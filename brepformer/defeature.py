#!/usr/bin/env python3
"""Automatic defeaturing of STEP files using BrepFormer predictions.

Uses a trained BrepFormer model (5-class defeature or 27-class MFTRCAD) to
classify each face of a B-rep model, then removes detected manufacturing
features (holes, chamfers, fillets, cuts) using OpenCASCADE's
BRepAlgoAPI_Defeaturing, leaving only the base stock shape.

The defeaturing algorithm:
  1. Run inference to get per-face class predictions
  2. Collect all non-random faces (classes 1-4: hole, chamfer, fillet, cut)
  3. Attempt batch removal of all feature faces at once
  4. If batch fails, progressively add feature types (fillet -> chamfer ->
     hole -> cut) until failure, then try individual faces from failing types
  5. Validate and write the defeatured STEP file

Usage:
    # Single file (uses default defeature checkpoint)
    python -m brepformer.defeature --step model.step

    # Directory of files
    python -m brepformer.defeature --step_dir steps/

    # Custom checkpoint and output
    python -m brepformer.defeature --step model.step \
        --checkpoint results/my_model/best.ckpt \
        --output_dir my_output/

    # From pre-computed .seg predictions
    python -m brepformer.defeature --step model.step --seg preds.seg

    # Also save colored STEP for visual comparison
    python -m brepformer.defeature --step model.step --save_colored
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brepformer.data.classes import (
    DEFEATURE_CLASS_NAMES, DEFEATURE_NUM_CLASSES,
    DEFEATURE_CLASS_COLORS_HEX,
    CLASS_TO_DEFEATURE, NUM_CLASSES,
)

# Default checkpoint for the defeature model
DEFAULT_CHECKPOINT = "results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt"

# Feature classes to remove (everything except random=0)
FEATURE_CLASSES = {
    1: "hole",
    2: "chamfer",
    3: "fillet",
    4: "cut",
}

# Removal order: fillets first (smoothing edges are easiest to heal),
# then chamfers (similar), then holes (cylindrical fills), then cuts (largest)
REMOVAL_ORDER = [3, 2, 1, 4]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Automatic defeaturing of STEP files using BrepFormer predictions"
    )

    # Input
    parser.add_argument("--step", type=str, default=None,
                        help="Path to a single STEP file")
    parser.add_argument("--step_dir", type=str, default=None,
                        help="Directory of STEP files for batch mode")

    # Model
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help=f"Model checkpoint (default: {DEFAULT_CHECKPOINT})")

    # Pre-computed predictions
    parser.add_argument("--seg", type=str, default=None,
                        help="Path to .seg file with pre-computed predictions (single mode)")
    parser.add_argument("--seg_dir", type=str, default=None,
                        help="Directory of .seg prediction files (batch mode)")

    # Output
    parser.add_argument("--output_dir", type=str, default="brepformer/defeatured_output",
                        help="Output directory (default: brepformer/defeatured_output)")

    # Options
    parser.add_argument("--save_colored", action="store_true",
                        help="Also save a colored STEP showing predictions before defeaturing")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed defeaturing progress")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def load_seg_labels(seg_path: str) -> List[int]:
    """Load per-face labels from a .seg file (one label per line)."""
    labels = []
    with open(seg_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(int(line))
    return labels


def infer_defeature_predictions(model, device, step_path: str) -> Optional[List[int]]:
    """Run inference and return 5-class defeature predictions.

    Handles both 5-class defeature models and 27-class MFTRCAD models
    (the latter are remapped via CLASS_TO_DEFEATURE).
    """
    from brepformer.infer import infer_single

    result = infer_single(model, step_path, device)
    if "error" in result:
        return None

    face_preds = result.get("face_preds")
    if face_preds is None:
        return None

    # Remap 27-class predictions to 5 defeature classes
    num_model_classes = model.config.num_face_classes
    if num_model_classes > DEFEATURE_NUM_CLASSES:
        face_preds = [
            CLASS_TO_DEFEATURE[p] if 0 <= p < NUM_CLASSES else 0
            for p in face_preds
        ]

    return face_preds


# ---------------------------------------------------------------------------
# Core defeaturing engine
# ---------------------------------------------------------------------------
#
# BRepAlgoAPI_Defeaturing removes faces by EXTENDING adjacent faces to fill
# the gap — it doesn't just delete geometry. This is what makes it correct
# for every feature type:
#
#   Holes (circular, polygonal, blind, through):
#       Removes the cylindrical/planar wall faces AND bottom face (if blind).
#       The surrounding face (e.g. a flat plate) is extended across the hole.
#
#   Fillets:
#       Removes the fillet blend face. The two faces that were blended are
#       extended until they intersect, restoring the original sharp edge.
#
#   Chamfers:
#       Same as fillets — the chamfer face is removed and the adjacent faces
#       are extended to meet at a sharp edge.
#
#   Cuts (slots, pockets, passages, steps):
#       All faces forming the cut cavity are removed. Surrounding faces are
#       extended inward to fill the volume, restoring the stock shape.
#
# After defeaturing, ShapeUpgrade_UnifySameDomain merges adjacent faces that
# now lie on the same underlying surface (e.g. a plate face that was split
# by a hole is unified back into a single face).
# ---------------------------------------------------------------------------


def _try_defeaturing(shape, faces_to_remove: list) -> Optional[object]:
    """Attempt to remove faces from shape via BRepAlgoAPI_Defeaturing.

    This FILLS the gaps left by removed faces by extending adjacent geometry.
    It does not leave holes — the result is always a closed solid (or fails).

    Returns the defeatured shape on success, or None on failure.
    """
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Defeaturing

    if not faces_to_remove:
        return shape

    op = BRepAlgoAPI_Defeaturing()
    op.SetShape(shape)
    for face in faces_to_remove:
        op.AddFaceToRemove(face)
    op.SetRunParallel(True)
    op.Build()

    if op.IsDone() and not op.HasErrors():
        return op.Shape()
    return None


def _unify_faces(shape):
    """Merge adjacent faces on the same surface and edges on the same curve.

    After defeaturing, a face that was split by a feature (e.g. a plate with
    a hole) may be left as two separate faces on the same plane. This merges
    them into a single clean face. Also unifies edges/vertices similarly.
    """
    from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain

    unifier = ShapeUpgrade_UnifySameDomain(shape, True, True, True)
    unifier.Build()
    return unifier.Shape()


def _group_connected_faces(shape, faces: list) -> List[list]:
    """Group faces into connected components based on shared edges.

    Multi-face features (e.g. a blind hole = cylindrical wall + bottom disk,
    or a rectangular pocket = 4 walls + floor) must be removed as a unit.
    Trying to remove just one face of a multi-face feature will always fail
    because the kernel can't heal a partial feature.

    Uses TopTools edge-face ancestor map for efficient adjacency lookup.
    """
    from OCC.Core.TopExp import TopExp_Explorer, topexp
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCC.Core.TopoDS import topods
    from OCC.Core.TopTools import (
        TopTools_IndexedDataMapOfShapeListOfShape,
        TopTools_IndexedMapOfShape,
        TopTools_ListIteratorOfListOfShape,
    )

    if len(faces) <= 1:
        return [faces] if faces else []

    # Build a map: edge -> list of ancestor faces (for the whole shape)
    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

    # Build an indexed map of the feature faces for fast Contains() lookup
    feature_map = TopTools_IndexedMapOfShape()
    for f in faces:
        feature_map.Add(f)

    # Build adjacency: two feature faces are connected if they share an edge
    adjacency: Dict[int, set] = {i: set() for i in range(len(faces))}

    for i, face in enumerate(faces):
        edge_exp = TopExp_Explorer(face, TopAbs_EDGE)
        while edge_exp.More():
            edge = edge_exp.Current()
            # Look up all faces sharing this edge
            idx = edge_face_map.FindIndex(edge)
            if idx > 0:
                neighbors = edge_face_map.FindFromIndex(idx)
                it = TopTools_ListIteratorOfListOfShape(neighbors)
                while it.More():
                    neighbor_face = it.Value()
                    if feature_map.Contains(neighbor_face):
                        j = feature_map.FindIndex(neighbor_face) - 1  # 1-based
                        if j != i:
                            adjacency[i].add(j)
                    it.Next()
            edge_exp.Next()

    # BFS to find connected components
    visited = set()
    groups = []
    for start in range(len(faces)):
        if start in visited:
            continue
        component = []
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            component.append(faces[node])
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
        groups.append(component)

    return groups


def defeature_shape(shape, feature_faces_by_type: Dict[int, list],
                    verbose: bool = False) -> Tuple[object, int, int]:
    """Remove feature faces from a shape using progressive defeaturing.

    The algorithm ensures that features are FILLED (adjacent faces extended
    to close gaps), not just deleted. It handles:
      - Holes of all types (circular, polygonal, blind, through)
      - Fillets (blend faces removed, sharp edges restored)
      - Chamfers (angled faces removed, sharp edges restored)
      - Cuts (slots, pockets, passages — cavity faces removed, stock restored)

    Strategy:
      Phase 1 — Try removing ALL feature faces in a single call.
      Phase 2 — If that fails, progressively add feature types one at a time
                (fillet -> chamfer -> hole -> cut).
      Phase 3 — For types that fail as a group, find connected components
                (multi-face features like blind holes) and try each component
                as a unit. Single isolated faces are also tried individually.
      Phase 4 — Second pass on the modified shape: re-enumerate faces, match
                unremoved features by hash, and retry. Removing some features
                may unblock others (e.g. a fillet adjacent to a hole).
      Cleanup — Unify faces on the same surface (ShapeUpgrade_UnifySameDomain).

    All phase 1-3 attempts operate on the ORIGINAL shape with an accumulating
    face set, so face references never go stale.

    Args:
        shape: The original TopoDS_Shape.
        feature_faces_by_type: Dict mapping class_id -> list of TopoDS_Face.
        verbose: Print progress.

    Returns:
        (result_shape, num_removed, num_failed)
    """
    all_features = []
    for cls_id in REMOVAL_ORDER:
        all_features.extend(feature_faces_by_type.get(cls_id, []))

    total_features = len(all_features)
    if total_features == 0:
        return shape, 0, 0

    # --- Phase 1: Try all at once ---
    result = _try_defeaturing(shape, all_features)
    if result is not None:
        if verbose:
            print(f"    Phase 1: all {total_features} faces removed at once")
        result = _unify_faces(result)
        return result, total_features, 0

    if verbose:
        print(f"    Phase 1 failed ({total_features} faces), "
              f"trying progressive...")

    # --- Phase 2 + 3: Progressive type-by-type with connected components ---
    accumulated = []
    best_shape = shape
    best_count = 0

    for cls_id in REMOVAL_ORDER:
        type_faces = feature_faces_by_type.get(cls_id, [])
        if not type_faces:
            continue

        cls_name = DEFEATURE_CLASS_NAMES[cls_id]

        # Try adding all faces of this type at once
        candidate = accumulated + type_faces
        result = _try_defeaturing(shape, candidate)
        if result is not None:
            accumulated = candidate
            best_shape = result
            best_count = len(accumulated)
            if verbose:
                print(f"    Phase 2 + {cls_name}: all {len(type_faces)} faces "
                      f"(total: {best_count})")
            continue

        # Phase 3: Type group failed — try connected components.
        # Multi-face features (e.g. blind hole = cylinder + disk) must be
        # removed as a unit. Grouping by adjacency ensures we never try to
        # remove half a feature.
        groups = _group_connected_faces(shape, type_faces)
        if verbose:
            sizes = [len(g) for g in groups]
            print(f"    Phase 3 {cls_name}: {len(type_faces)} faces in "
                  f"{len(groups)} groups (sizes: {sizes})")

        added_from_type = 0
        for group in groups:
            candidate = accumulated + group
            result = _try_defeaturing(shape, candidate)
            if result is not None:
                accumulated = candidate
                best_shape = result
                best_count = len(accumulated)
                added_from_type += len(group)
                if verbose:
                    print(f"      + group of {len(group)} {cls_name} faces")
            elif verbose:
                print(f"      x group of {len(group)} {cls_name} faces failed")

        if verbose:
            print(f"      {cls_name}: {added_from_type}/{len(type_faces)} removed")

    # --- Phase 4: Iterative retry on modified shape ---
    # After some features are removed, the geometry changes. Features that
    # were previously impossible to remove may now succeed. Keep retrying
    # remaining faces on the modified shape until no more progress is made.
    if best_count > 0 and best_count < total_features:
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE
        from OCC.Core.TopoDS import topods
        from OCC.Core.TopTools import TopTools_IndexedMapOfShape

        iteration = 0
        max_iterations = 10  # safety limit
        while best_count < total_features and iteration < max_iterations:
            iteration += 1

            # Build map of faces we already accumulated for removal
            accumulated_map = TopTools_IndexedMapOfShape()
            for face in accumulated:
                accumulated_map.Add(face)

            # Build map of faces in the new (modified) shape
            new_face_map = TopTools_IndexedMapOfShape()
            exp = TopExp_Explorer(best_shape, TopAbs_FACE)
            while exp.More():
                new_face_map.Add(topods.Face(exp.Current()))
                exp.Next()

            # Collect unremoved feature faces that still exist in the new shape
            remaining = []
            for face in all_features:
                if not accumulated_map.Contains(face) and new_face_map.Contains(face):
                    remaining.append(face)

            if not remaining:
                break

            if verbose:
                print(f"    Phase 4 iter {iteration}: retrying {len(remaining)} "
                      f"remaining faces on modified shape...")

            progress_this_iter = 0

            # Try all remaining at once on the new shape
            result = _try_defeaturing(best_shape, remaining)
            if result is not None:
                best_shape = result
                best_count += len(remaining)
                progress_this_iter = len(remaining)
                if verbose:
                    print(f"      All {len(remaining)} removed")
            else:
                # Try connected groups on the new shape
                groups = _group_connected_faces(best_shape, remaining)
                for group in groups:
                    result = _try_defeaturing(best_shape, group)
                    if result is not None:
                        best_shape = result
                        best_count += len(group)
                        progress_this_iter += len(group)
                        # Update accumulated so next iteration knows
                        for f in group:
                            accumulated.append(f)
                        if verbose:
                            print(f"      + group of {len(group)} on iter {iteration}")
                    else:
                        # Try individual faces from the failed group
                        for face in group:
                            result = _try_defeaturing(best_shape, [face])
                            if result is not None:
                                best_shape = result
                                best_count += 1
                                progress_this_iter += 1
                                accumulated.append(face)
                                if verbose:
                                    print(f"      + 1 individual face on iter {iteration}")

            if progress_this_iter == 0:
                if verbose:
                    print(f"      No progress on iter {iteration}, stopping")
                break

    # --- Phase 5: Last-ditch fallback for total failures ---
    # When Phases 1-4 remove NOTHING (best_count == 0), the standard
    # progressive strategy has completely failed. This happens on heavily
    # filleted/interdependent models. Try alternative removal strategies:
    #   5a. All fillets batch on fresh shape (different face combination than P1)
    #   5b. Start from fillets-only shape, then iteratively add other types
    #   5c. Reverse removal order (cuts first, fillets last)
    #   5d. Single-face iterative on fresh shape with retry after each success
    if best_count == 0 and total_features > 0:
        if verbose:
            print(f"    Phase 5: last-ditch fallback (0/{total_features} removed)...")

        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE
        from OCC.Core.TopoDS import topods
        from OCC.Core.TopTools import TopTools_IndexedMapOfShape

        # 5a: Try just fillets on fresh shape
        fillet_faces = feature_faces_by_type.get(3, [])
        if fillet_faces:
            result = _try_defeaturing(shape, fillet_faces)
            if result is not None:
                if verbose:
                    print(f"      5a: all {len(fillet_faces)} fillets removed at once")
                best_shape = result
                best_count = len(fillet_faces)
                accumulated = list(fillet_faces)

                # Now iteratively try remaining types on the fillet-free shape
                for cls_id in [2, 1, 4]:  # chamfer, hole, cut
                    type_faces = feature_faces_by_type.get(cls_id, [])
                    if not type_faces:
                        continue
                    # Filter to faces still present in the modified shape
                    new_face_map = TopTools_IndexedMapOfShape()
                    exp = TopExp_Explorer(best_shape, TopAbs_FACE)
                    while exp.More():
                        new_face_map.Add(topods.Face(exp.Current()))
                        exp.Next()
                    present = [f for f in type_faces if new_face_map.Contains(f)]
                    if not present:
                        continue

                    result = _try_defeaturing(best_shape, present)
                    if result is not None:
                        best_shape = result
                        best_count += len(present)
                        accumulated.extend(present)
                        if verbose:
                            print(f"      5a+: all {len(present)} "
                                  f"{DEFEATURE_CLASS_NAMES[cls_id]} removed")
                    else:
                        groups = _group_connected_faces(best_shape, present)
                        for group in groups:
                            result = _try_defeaturing(best_shape, group)
                            if result is not None:
                                best_shape = result
                                best_count += len(group)
                                accumulated.extend(group)
                                if verbose:
                                    print(f"      5a+: group of {len(group)} "
                                          f"{DEFEATURE_CLASS_NAMES[cls_id]}")

        # 5b: If 5a didn't help (or no fillets), try reverse order
        if best_count == 0:
            reverse_order = [4, 1, 2, 3]  # cuts first, fillets last
            if verbose:
                print(f"      5b: trying reverse order (cut->hole->chamfer->fillet)...")

            for cls_id in reverse_order:
                type_faces = feature_faces_by_type.get(cls_id, [])
                if not type_faces:
                    continue
                # Try whole type
                candidate = accumulated + type_faces
                result = _try_defeaturing(shape, candidate)
                if result is not None:
                    accumulated = candidate
                    best_shape = result
                    best_count = len(accumulated)
                    if verbose:
                        print(f"      5b: all {len(type_faces)} "
                              f"{DEFEATURE_CLASS_NAMES[cls_id]}")
                    continue
                # Try groups
                groups = _group_connected_faces(shape, type_faces)
                for group in groups:
                    candidate = accumulated + group
                    result = _try_defeaturing(shape, candidate)
                    if result is not None:
                        accumulated = candidate
                        best_shape = result
                        best_count = len(accumulated)
                        if verbose:
                            print(f"      5b: group of {len(group)} "
                                  f"{DEFEATURE_CLASS_NAMES[cls_id]}")

        # 5c: Single-face iterative with retry after each success
        if best_count == 0:
            if verbose:
                print(f"      5c: single-face iterative with retry...")

            # Try each face individually, re-attempt all failed after each success
            remaining_faces = list(all_features)
            current_shape = shape
            removed_this_phase = 0
            max_rounds = 5

            for round_num in range(max_rounds):
                progress_this_round = 0
                still_remaining = []
                for face in remaining_faces:
                    # Check face still exists in current shape
                    face_map = TopTools_IndexedMapOfShape()
                    exp = TopExp_Explorer(current_shape, TopAbs_FACE)
                    while exp.More():
                        face_map.Add(topods.Face(exp.Current()))
                        exp.Next()
                    if not face_map.Contains(face):
                        # Face was removed as side-effect of another removal
                        removed_this_phase += 1
                        progress_this_round += 1
                        continue

                    result = _try_defeaturing(current_shape, [face])
                    if result is not None:
                        current_shape = result
                        removed_this_phase += 1
                        progress_this_round += 1
                        if verbose:
                            print(f"      5c round {round_num+1}: +1 face")
                    else:
                        still_remaining.append(face)

                remaining_faces = still_remaining
                if progress_this_round == 0 or not remaining_faces:
                    break

            if removed_this_phase > 0:
                best_shape = current_shape
                best_count = removed_this_phase
                if verbose:
                    print(f"      5c: {removed_this_phase} faces removed total")

        # Phase 5 iterative retry (same as Phase 4 loop) if 5a/5b/5c made partial progress
        if 0 < best_count < total_features:
            iteration = 0
            max_iterations = 10
            while best_count < total_features and iteration < max_iterations:
                iteration += 1
                new_face_map = TopTools_IndexedMapOfShape()
                exp = TopExp_Explorer(best_shape, TopAbs_FACE)
                while exp.More():
                    new_face_map.Add(topods.Face(exp.Current()))
                    exp.Next()

                remaining = [f for f in all_features
                             if new_face_map.Contains(f)
                             and f not in accumulated]
                if not remaining:
                    break

                progress = 0
                # Try all remaining
                result = _try_defeaturing(best_shape, remaining)
                if result is not None:
                    best_shape = result
                    best_count += len(remaining)
                    break

                groups = _group_connected_faces(best_shape, remaining)
                for group in groups:
                    result = _try_defeaturing(best_shape, group)
                    if result is not None:
                        best_shape = result
                        best_count += len(group)
                        accumulated.extend(group)
                        progress += len(group)
                    else:
                        for face in group:
                            result = _try_defeaturing(best_shape, [face])
                            if result is not None:
                                best_shape = result
                                best_count += 1
                                accumulated.append(face)
                                progress += 1
                if progress == 0:
                    break

        if verbose and best_count > 0:
            print(f"      Phase 5 total: {best_count}/{total_features} removed")

    # --- Cleanup: merge faces on the same surface ---
    best_shape = _unify_faces(best_shape)

    failed = total_features - best_count
    return best_shape, best_count, failed


def defeature_step(step_path: str, face_preds: List[int],
                   output_path: str, verbose: bool = False) -> dict:
    """Defeature a STEP file based on per-face predictions.

    Reads the STEP file, groups faces by predicted class, removes all
    non-random faces using the progressive defeaturing engine, validates
    the result, and writes the output.

    Args:
        step_path: Input STEP file path.
        face_preds: Per-face class predictions (0=random, 1=hole, 2=chamfer,
                    3=fillet, 4=cut).
        output_path: Output STEP file path.
        verbose: Print detailed progress.

    Returns:
        Dict with status, face counts, and defeaturing log.
    """
    from OCC.Core.STEPControl import (
        STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs,
    )
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.ShapeFix import ShapeFix_Shape

    # Read shape
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        return {"status": "error", "message": f"Failed to read {step_path}"}

    reader.TransferRoots()
    shape = reader.OneShape()

    # Enumerate faces in TopExp_Explorer order (matches inference)
    faces = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        faces.append(topods.Face(explorer.Current()))
        explorer.Next()

    num_faces = len(faces)
    if num_faces != len(face_preds):
        return {
            "status": "error",
            "message": (f"Face count mismatch: shape has {num_faces} faces, "
                        f"predictions have {len(face_preds)}"),
        }

    # Group faces by predicted feature type
    feature_faces_by_type: Dict[int, list] = {}
    keep_count = 0
    for i, (face, pred) in enumerate(zip(faces, face_preds)):
        if pred == 0:
            keep_count += 1
        elif pred in FEATURE_CLASSES:
            feature_faces_by_type.setdefault(pred, []).append(face)

    total_features = sum(len(v) for v in feature_faces_by_type.values())

    if total_features == 0:
        # Nothing to remove — write unchanged shape
        writer = STEPControl_Writer()
        writer.Transfer(shape, STEPControl_AsIs)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        writer.Write(str(output_path))
        return {
            "status": "no_features",
            "message": "No features detected, output is identical to input",
            "output": str(output_path),
            "num_faces": num_faces,
            "kept": keep_count,
            "removed": 0,
            "failed": 0,
        }

    if verbose:
        print(f"  Faces: {num_faces} total, {keep_count} random, "
              f"{total_features} features")
        for cls_id in REMOVAL_ORDER:
            n = len(feature_faces_by_type.get(cls_id, []))
            if n > 0:
                print(f"    {DEFEATURE_CLASS_NAMES[cls_id]}: {n}")

    # Run defeaturing
    result_shape, removed, failed = defeature_shape(
        shape, feature_faces_by_type, verbose=verbose,
    )

    # Shape healing pass
    fixer = ShapeFix_Shape(result_shape)
    fixer.Perform()
    result_shape = fixer.Shape()

    # Validate
    analyzer = BRepCheck_Analyzer(result_shape)
    is_valid = analyzer.IsValid()

    # Write output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = STEPControl_Writer()
    writer.Transfer(result_shape, STEPControl_AsIs)
    write_status = writer.Write(str(output_path))

    if write_status != IFSelect_RetDone:
        return {"status": "error", "message": f"Failed to write {output_path}"}

    return {
        "status": "success",
        "output": str(output_path),
        "num_faces": num_faces,
        "kept": keep_count,
        "removed": removed,
        "failed": failed,
        "valid": is_valid,
    }


# ---------------------------------------------------------------------------
# Colored STEP helper (for --save_colored)
# ---------------------------------------------------------------------------

def save_colored_step(step_path: str, preds: List[int], output_path: str):
    """Save a colored STEP showing predictions (for visual comparison)."""
    from brepformer.export_freecad import export_colored_step

    export_colored_step(
        step_path, preds, str(output_path),
        colors_hex=DEFEATURE_CLASS_COLORS_HEX,
        num_cls=DEFEATURE_NUM_CLASSES,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """Main defeaturing pipeline."""
    args = parse_args()

    if args.step is None and args.step_dir is None:
        print("Error: Provide either --step or --step_dir")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model once (shared across batch)
    model = None
    device = None
    if not args.seg and not args.seg_dir:
        import torch
        from brepformer.infer import load_model

        print(f"Loading model from {args.checkpoint}...")
        model = load_model(args.checkpoint)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"Device: {device}")

        if not model.config.face_segmentation:
            print("Error: Model has no face segmentation head — "
                  "cannot predict per-face features.")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Single file mode
    # ------------------------------------------------------------------
    if args.step:
        step_path = args.step
        model_id = Path(step_path).stem
        print(f"\nDefeaturing {step_path}...")

        # Get predictions
        seg_path = args.seg
        if seg_path is None and args.seg_dir:
            candidate = Path(args.seg_dir) / f"{model_id}.seg"
            if candidate.exists():
                seg_path = str(candidate)

        if seg_path:
            preds = load_seg_labels(seg_path)
        else:
            preds = infer_defeature_predictions(model, device, step_path)

        if preds is None:
            print("Failed to get predictions")
            sys.exit(1)

        # Summary
        counts = Counter(preds)
        print(f"  Predictions ({len(preds)} faces):")
        for cls_id in sorted(counts.keys()):
            name = (DEFEATURE_CLASS_NAMES[cls_id]
                    if cls_id < DEFEATURE_NUM_CLASSES else f"class_{cls_id}")
            action = "keep" if cls_id == 0 else "REMOVE"
            print(f"    {name}: {counts[cls_id]} [{action}]")

        # Defeature
        out_path = output_dir / f"{model_id}_defeatured.step"
        result = defeature_step(
            step_path, preds, str(out_path), verbose=args.verbose,
        )
        _print_result(result)

        if args.save_colored:
            colored_path = output_dir / f"{model_id}_colored.step"
            save_colored_step(step_path, preds, str(colored_path))

        # Save report
        report_path = output_dir / f"{model_id}_report.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Report: {report_path}")

    # ------------------------------------------------------------------
    # Batch mode
    # ------------------------------------------------------------------
    elif args.step_dir:
        step_dir = Path(args.step_dir)
        step_files = sorted(
            list(step_dir.glob("*.step"))
            + list(step_dir.glob("*.stp"))
            + list(step_dir.glob("*.STEP"))
        )
        print(f"\nFound {len(step_files)} STEP files")

        all_results = []
        success_count = 0
        no_feature_count = 0
        error_count = 0

        for idx, step_file in enumerate(step_files, 1):
            model_id = step_file.stem
            print(f"\n  [{idx}/{len(step_files)}] {step_file.name}...")

            # Get predictions
            seg_path = None
            if args.seg_dir:
                candidate = Path(args.seg_dir) / f"{model_id}.seg"
                if candidate.exists():
                    seg_path = str(candidate)

            if seg_path:
                preds = load_seg_labels(seg_path)
            else:
                preds = infer_defeature_predictions(
                    model, device, str(step_file),
                )

            if preds is None:
                print(f"    Skipping: failed to get predictions")
                error_count += 1
                continue

            out_path = output_dir / f"{model_id}_defeatured.step"
            result = defeature_step(
                str(step_file), preds, str(out_path), verbose=args.verbose,
            )
            result["model_id"] = model_id
            all_results.append(result)

            if result["status"] == "success":
                success_count += 1
                print(f"    Removed {result['removed']} features "
                      f"(failed: {result['failed']}, valid: {result['valid']})")
            elif result["status"] == "no_features":
                no_feature_count += 1
                print(f"    No features detected")
            else:
                error_count += 1
                print(f"    Error: {result.get('message', 'unknown')}")

            if args.save_colored:
                colored_path = output_dir / f"{model_id}_colored.step"
                save_colored_step(str(step_file), preds, str(colored_path))

        # Batch report
        report = {
            "total": len(step_files),
            "success": success_count,
            "no_features": no_feature_count,
            "errors": error_count,
            "results": all_results,
        }
        report_path = output_dir / "batch_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n{'=' * 60}")
        print(f"Defeaturing complete: {success_count} defeatured, "
              f"{no_feature_count} unchanged, {error_count} errors")
        print(f"Output: {output_dir}")
        print(f"Report: {report_path}")


def _print_result(result: dict):
    """Print a single defeaturing result summary."""
    status = result["status"]
    if status == "success":
        print(f"\n  Removed {result['removed']} feature faces "
              f"(failed: {result['failed']})")
        print(f"  Shape valid: {result['valid']}")
        print(f"  Output: {result['output']}")
    elif status == "no_features":
        print(f"\n  {result['message']}")
        print(f"  Output: {result['output']}")
    else:
        print(f"\n  Error: {result.get('message', 'unknown')}")


if __name__ == "__main__":
    main()
