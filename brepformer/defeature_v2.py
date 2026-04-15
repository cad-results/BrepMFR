#!/usr/bin/env python3
"""Modern automatic defeaturing of STEP files (v2) — pythonocc 7.9+ features.

Drop-in replacement for defeature.py with the same CLI interface. Key
improvements that increase the per-face removal success rate:

  Adaptive fuzzy tolerance
      When defeaturing fails at the default geometric tolerance, retries
      with progressively relaxed SetFuzzyValue() (1e-5 → 1e-3). This
      catches borderline geometry-healing failures — the single biggest
      source of "fillet cannot be removed" errors in v1.

  History-based face tracking
      Uses BRepTools_History (OCCT 7.4+) returned by the kernel instead of
      fragile IndexedMap.Contains() heuristics for mapping faces between
      phases. Precisely identifies removed, modified, and surviving faces.

  Pre-validation and auto-repair
      Validates input shapes with BRepCheck_Analyzer before defeaturing and
      auto-repairs tolerance issues, degenerate edges, and bad wires via
      ShapeFix_Shape + ShapeFix_ShapeTolerance.

  Enhanced healing pipeline
      Multi-stage post-defeaturing repair: tolerance harmonization →
      shape fix → conditional sewing (BRepBuilderAPI_Sewing for micro-gap
      closure) → face unification with angular/linear tolerance control
      (ShapeUpgrade_UnifySameDomain).

  Area-based ordering
      Connected-component groups and individual retry faces are sorted by
      surface area (smallest first). Smaller features have higher first-
      attempt success rates, and their removal often unblocks larger ones.

  Intermediate healing
      In the single-face iterative phase, a quick UnifySameDomain pass runs
      after each successful removal, improving geometry quality before the
      next attempt.

Requires: pythonocc-core >= 7.9 (OCCT >= 7.9)

Usage (identical to v1):
    python -m brepformer.defeature_v2 --step model.step
    python -m brepformer.defeature_v2 --step_dir steps/ --verbose
    python -m brepformer.defeature_v2 --step model.step --max_fuzzy 0.01
"""

import argparse
import gc
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brepformer.data.classes import (
    DEFEATURE_CLASS_NAMES, DEFEATURE_NUM_CLASSES,
    DEFEATURE_CLASS_COLORS_HEX,
    CLASS_TO_DEFEATURE, NUM_CLASSES,
)

DEFAULT_CHECKPOINT = "results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt"

FEATURE_CLASSES = {
    1: "hole",
    2: "chamfer",
    3: "fillet",
    4: "cut",
}

# Removal order: fillets first (smoothing edges easiest to heal),
# then chamfers (similar), then holes (cylindrical fills), then cuts (largest).
REMOVAL_ORDER = [3, 2, 1, 4]

# Fuzzy tolerance ladder for adaptive retry.
# 0 = OCCT default (~Precision::Confusion ≈ 1e-7).  Each step relaxes the
# geometric matching tolerance, allowing the kernel to close gaps that fail
# at tighter tolerance.
FUZZY_TOLERANCES = [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Modern defeaturing of STEP files using BrepFormer (v2)"
    )
    parser.add_argument("--step", type=str, default=None,
                        help="Path to a single STEP file")
    parser.add_argument("--step_dir", type=str, default=None,
                        help="Directory of STEP files for batch mode")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help=f"Model checkpoint (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument("--seg", type=str, default=None,
                        help="Path to .seg file with pre-computed predictions")
    parser.add_argument("--seg_dir", type=str, default=None,
                        help="Directory of .seg prediction files (batch mode)")
    parser.add_argument("--output_dir", type=str,
                        default="brepformer/defeatured_output",
                        help="Output directory")
    parser.add_argument("--save_colored", action="store_true",
                        help="Also save a colored STEP showing predictions")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed defeaturing progress")
    parser.add_argument("--max_fuzzy", type=float, default=1e-3,
                        help="Max fuzzy tolerance for adaptive retry (default: 1e-3)")
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
    """Run inference and return 5-class defeature predictions."""
    from brepformer.infer import infer_single

    result = infer_single(model, step_path, device)
    if "error" in result:
        return None

    face_preds = result.get("face_preds")
    if face_preds is None:
        return None

    num_model_classes = model.config.num_face_classes
    if num_model_classes > DEFEATURE_NUM_CLASSES:
        face_preds = [
            CLASS_TO_DEFEATURE[p] if 0 <= p < NUM_CLASSES else 0
            for p in face_preds
        ]

    return face_preds


# ---------------------------------------------------------------------------
# Shape analysis helpers (NEW in v2)
# ---------------------------------------------------------------------------

def _get_face_area(face) -> float:
    """Return the surface area of a TopoDS_Face."""
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop

    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    return props.Mass()


def _sort_faces_by_area(faces: list) -> list:
    """Sort faces by surface area, smallest first (easier to remove)."""
    return sorted(faces, key=_get_face_area)


def _sort_groups_by_total_area(groups: List[list]) -> List[list]:
    """Sort connected-component groups by total area, smallest first."""
    return sorted(groups, key=lambda g: sum(_get_face_area(f) for f in g))


def _validate_input_shape(shape, verbose: bool = False):
    """Validate and auto-repair input shape before defeaturing.

    Catches corrupted geometry, bad tolerances, and degenerate edges that
    would cause BRepAlgoAPI_Defeaturing to fail even on simple features.
    """
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.ShapeFix import ShapeFix_Shape, ShapeFix_ShapeTolerance

    analyzer = BRepCheck_Analyzer(shape)
    if analyzer.IsValid():
        return shape

    if verbose:
        print("  Input shape has validation issues — auto-repairing...")

    # Harmonize tolerances to a sane range
    tol_fixer = ShapeFix_ShapeTolerance()
    tol_fixer.LimitTolerance(shape, 1e-7, 1e-3)

    # General shape fix (wires, edges, faces)
    fixer = ShapeFix_Shape(shape)
    fixer.SetPrecision(1e-6)
    fixer.Perform()
    shape = fixer.Shape()

    if verbose:
        analyzer2 = BRepCheck_Analyzer(shape)
        print(f"  After repair: valid={analyzer2.IsValid()}")

    return shape


# ---------------------------------------------------------------------------
# Core defeaturing engine (ENHANCED in v2)
# ---------------------------------------------------------------------------

def _try_defeaturing(shape, faces_to_remove: list,
                     fuzzy_value: float = 0.0):
    """Attempt face removal with optional fuzzy tolerance.

    Returns (result_shape, history) on success, (None, None) on failure.
    The BRepTools_History tracks what happened to every face in the shape
    and is used for precise face mapping between phases.
    """
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Defeaturing

    if not faces_to_remove:
        return shape, None

    op = BRepAlgoAPI_Defeaturing()
    op.SetShape(shape)
    for face in faces_to_remove:
        op.AddFaceToRemove(face)
    if fuzzy_value > 0:
        op.SetFuzzyValue(fuzzy_value)
    op.SetRunParallel(True)
    op.Build()

    if op.IsDone() and not op.HasErrors():
        history = None
        try:
            history = op.History()
        except Exception:
            pass
        return op.Shape(), history
    return None, None


def _try_defeaturing_adaptive(shape, faces_to_remove: list,
                               verbose: bool = False,
                               max_fuzzy: float = 1e-3):
    """Try defeaturing with progressively relaxed tolerances.

    Returns (result_shape, history, fuzzy_used) on first success,
    or (None, None, 0.0) if all tolerances fail.
    """
    for tol in FUZZY_TOLERANCES:
        if tol > max_fuzzy:
            break
        result, history = _try_defeaturing(shape, faces_to_remove,
                                           fuzzy_value=tol)
        if result is not None:
            if verbose and tol > 0:
                print(f"        (succeeded at fuzzy={tol:.1e})")
            return result, history, tol
    return None, None, 0.0


def _heal_shape(shape, verbose: bool = False):
    """Multi-stage shape healing pipeline (v2).

    1. Tolerance harmonization — cap face/edge tolerances to a sane range
    2. General shape fix — wires, edges, faces, degenerate curves
    3. Conditional sewing — close micro-gaps (only if shape is invalid)
    4. Face unification — merge co-planar adjacent faces with angular and
       linear tolerance control (SetAngularTolerance / SetLinearTolerance,
       available in pythonocc >= 7.5)
    """
    from OCC.Core.ShapeFix import ShapeFix_Shape, ShapeFix_ShapeTolerance
    from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    from OCC.Core.BRepCheck import BRepCheck_Analyzer

    # 1. Harmonize tolerances
    tol_fixer = ShapeFix_ShapeTolerance()
    tol_fixer.LimitTolerance(shape, 1e-7, 1e-3)

    # 2. General shape fix
    fixer = ShapeFix_Shape(shape)
    fixer.SetPrecision(1e-6)
    fixer.Perform()
    shape = fixer.Shape()

    # 3. Conditional sewing (only if shape has issues after fix)
    analyzer = BRepCheck_Analyzer(shape)
    if not analyzer.IsValid():
        try:
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing

            sewer = BRepBuilderAPI_Sewing(1e-6)
            sewer.Add(shape)
            sewer.Perform()
            sewn = sewer.SewedShape()
            if sewn is not None:
                analyzer2 = BRepCheck_Analyzer(sewn)
                if analyzer2.IsValid():
                    shape = sewn
                    if verbose:
                        print("    Sewing repaired invalid shape")
        except Exception:
            pass

    # 4. Unify co-planar faces with tolerance control
    unifier = ShapeUpgrade_UnifySameDomain(shape, True, True, True)
    try:
        unifier.SetAngularTolerance(1e-3)   # ~0.06° — merge nearly-parallel faces
        unifier.SetLinearTolerance(1e-5)    # 10 µm linear gap tolerance
    except AttributeError:
        pass  # Older pythonocc builds — UnifySameDomain still works without these
    unifier.Build()
    shape = unifier.Shape()

    return shape


def _unify_faces(shape):
    """Quick face unification (used between single-face removals)."""
    from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain

    unifier = ShapeUpgrade_UnifySameDomain(shape, True, True, True)
    unifier.Build()
    return unifier.Shape()


def _group_connected_faces(shape, faces: list) -> List[list]:
    """Group faces into connected components by shared edges.

    Same algorithm as v1 (edge-adjacency BFS via TopTools ancestor map),
    but returns groups sorted by total surface area — smallest first —
    because smaller features are easier to remove and their success often
    unblocks adjacent larger features.
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

    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

    feature_map = TopTools_IndexedMapOfShape()
    for f in faces:
        feature_map.Add(f)

    adjacency: Dict[int, set] = {i: set() for i in range(len(faces))}

    for i, face in enumerate(faces):
        edge_exp = TopExp_Explorer(face, TopAbs_EDGE)
        while edge_exp.More():
            edge = edge_exp.Current()
            idx = edge_face_map.FindIndex(edge)
            if idx > 0:
                neighbors = edge_face_map.FindFromIndex(idx)
                it = TopTools_ListIteratorOfListOfShape(neighbors)
                while it.More():
                    neighbor = it.Value()
                    if feature_map.Contains(neighbor):
                        j = feature_map.FindIndex(neighbor) - 1
                        if j != i:
                            adjacency[i].add(j)
                    it.Next()
            edge_exp.Next()

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
            for nb in adjacency[node]:
                if nb not in visited:
                    queue.append(nb)
        groups.append(component)

    # Sort groups: smallest total area first
    return _sort_groups_by_total_area(groups)


def _collect_remaining_faces(all_features: list, accumulated: list,
                              result_shape, history,
                              verbose: bool = False) -> list:
    """Find feature faces that still need removal in the modified shape.

    v2 improvement: uses BRepTools_History from the last successful
    defeaturing operation for precise face mapping. Falls back to
    IndexedMap.Contains() if History is unavailable.
    """
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape

    accumulated_ids = set(id(f) for f in accumulated)
    remaining = []

    # Try history-based tracking first
    if history is not None:
        try:
            from OCC.Core.TopTools import TopTools_ListIteratorOfListOfShape

            for face in all_features:
                if id(face) in accumulated_ids:
                    continue

                # Face removed directly or as side-effect?
                if history.IsRemoved(face):
                    continue

                # Face modified (geometry reshaped by adjacent removal)?
                modified = history.Modified(face)
                if not modified.IsEmpty():
                    it = TopTools_ListIteratorOfListOfShape(modified)
                    while it.More():
                        remaining.append(topods.Face(it.Value()))
                        it.Next()
                    continue

                # Face survived unchanged
                remaining.append(face)

            if verbose and remaining:
                print(f"      (history: {len(remaining)} faces to retry)")
            return remaining
        except Exception:
            # History API unavailable — fall through to map check
            remaining = []

    # Fallback: IndexedMap containment check (same as v1)
    face_map = TopTools_IndexedMapOfShape()
    exp = TopExp_Explorer(result_shape, TopAbs_FACE)
    while exp.More():
        face_map.Add(topods.Face(exp.Current()))
        exp.Next()

    for face in all_features:
        if id(face) in accumulated_ids:
            continue
        if face_map.Contains(face):
            remaining.append(face)

    return remaining


def defeature_shape(shape, feature_faces_by_type: Dict[int, list],
                    verbose: bool = False,
                    max_fuzzy: float = 1e-3) -> Tuple[object, int, int]:
    """Progressive defeaturing with adaptive tolerance and history tracking.

    Phase 0 — Pre-validate and auto-repair input shape.
    Phase 1 — Try ALL features at once (adaptive tolerance).
    Phase 2 — Progressive type-by-type: fillet → chamfer → hole → cut (adaptive).
    Phase 3 — Connected components per failed type (area-sorted, adaptive).
    Phase 4 — Retry on modified shape (history-based tracking, adaptive).
    Phase 5 — Last-ditch fallbacks: alternate orders, single-face iterative
              with intermediate healing (adaptive).
    Cleanup — Multi-stage healing pipeline.
    """
    all_features = []
    for cls_id in REMOVAL_ORDER:
        all_features.extend(feature_faces_by_type.get(cls_id, []))

    total_features = len(all_features)
    if total_features == 0:
        return shape, 0, 0

    # --- Phase 0: Pre-validate ---
    shape = _validate_input_shape(shape, verbose=verbose)

    # --- Phase 1: Try all at once (adaptive tolerance) ---
    result, history, tol = _try_defeaturing_adaptive(
        shape, all_features, verbose=verbose, max_fuzzy=max_fuzzy,
    )
    if result is not None:
        if verbose:
            print(f"    Phase 1: all {total_features} faces removed at once")
        result = _heal_shape(result, verbose=verbose)
        return result, total_features, 0

    if verbose:
        print(f"    Phase 1 failed ({total_features} faces), "
              f"trying progressive...")

    # --- Phase 2 + 3: Progressive type-by-type + connected components ---
    accumulated = []
    best_shape = shape
    best_count = 0
    last_history = None

    for cls_id in REMOVAL_ORDER:
        type_faces = feature_faces_by_type.get(cls_id, [])
        if not type_faces:
            continue

        cls_name = DEFEATURE_CLASS_NAMES[cls_id]

        # Phase 2: entire type at once (adaptive)
        candidate = accumulated + type_faces
        result, history, tol = _try_defeaturing_adaptive(
            shape, candidate, verbose=verbose, max_fuzzy=max_fuzzy,
        )
        if result is not None:
            accumulated = candidate
            best_shape = result
            best_count = len(accumulated)
            last_history = history
            if verbose:
                print(f"    Phase 2 + {cls_name}: all {len(type_faces)} faces "
                      f"(total: {best_count})")
            continue

        # Phase 3: connected components, area-sorted (adaptive)
        groups = _group_connected_faces(shape, type_faces)
        if verbose:
            sizes = [len(g) for g in groups]
            print(f"    Phase 3 {cls_name}: {len(type_faces)} faces in "
                  f"{len(groups)} groups (sizes: {sizes})")

        added_from_type = 0
        for group in groups:
            candidate = accumulated + group
            result, history, tol = _try_defeaturing_adaptive(
                shape, candidate, verbose=verbose, max_fuzzy=max_fuzzy,
            )
            if result is not None:
                accumulated = candidate
                best_shape = result
                best_count = len(accumulated)
                last_history = history
                added_from_type += len(group)
                if verbose:
                    print(f"      + group of {len(group)} {cls_name} faces")
            elif verbose:
                print(f"      x group of {len(group)} {cls_name} faces failed")

        if verbose:
            print(f"      {cls_name}: {added_from_type}/{len(type_faces)} removed")

    # --- Phase 4: Iterative retry on modified shape (history-based) ---
    if best_count > 0 and best_count < total_features:
        iteration = 0
        max_iterations = 10
        while best_count < total_features and iteration < max_iterations:
            iteration += 1

            remaining = _collect_remaining_faces(
                all_features, accumulated, best_shape, last_history,
                verbose=verbose,
            )

            if not remaining:
                break

            if verbose:
                print(f"    Phase 4 iter {iteration}: retrying {len(remaining)} "
                      f"faces on modified shape...")

            progress = 0

            # Try all remaining at once (adaptive)
            result, history, tol = _try_defeaturing_adaptive(
                best_shape, remaining, verbose=verbose, max_fuzzy=max_fuzzy,
            )
            if result is not None:
                best_shape = result
                best_count += len(remaining)
                last_history = history
                progress = len(remaining)
                if verbose:
                    print(f"      All {len(remaining)} removed")
            else:
                # Try connected groups on modified shape (adaptive)
                groups = _group_connected_faces(best_shape, remaining)
                for group in groups:
                    result, history, tol = _try_defeaturing_adaptive(
                        best_shape, group, verbose=verbose, max_fuzzy=max_fuzzy,
                    )
                    if result is not None:
                        best_shape = result
                        best_count += len(group)
                        progress += len(group)
                        accumulated.extend(group)
                        last_history = history
                        if verbose:
                            print(f"      + group of {len(group)} on iter {iteration}")
                    else:
                        # Individual faces, sorted by area (adaptive)
                        for face in _sort_faces_by_area(group):
                            result, history, tol = _try_defeaturing_adaptive(
                                best_shape, [face], verbose=verbose,
                                max_fuzzy=max_fuzzy,
                            )
                            if result is not None:
                                best_shape = result
                                best_count += 1
                                progress += 1
                                accumulated.append(face)
                                last_history = history
                                if verbose:
                                    print(f"      + 1 face on iter {iteration}")

            if progress == 0:
                if verbose:
                    print(f"      No progress on iter {iteration}, stopping")
                break

    # --- Phase 5: Last-ditch fallback for total failures ---
    if best_count == 0 and total_features > 0:
        if verbose:
            print(f"    Phase 5: last-ditch fallback (0/{total_features})...")

        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE
        from OCC.Core.TopoDS import topods
        from OCC.Core.TopTools import TopTools_IndexedMapOfShape

        # 5a: Fillets only on fresh shape (adaptive)
        fillet_faces = feature_faces_by_type.get(3, [])
        if fillet_faces:
            result, history, tol = _try_defeaturing_adaptive(
                shape, fillet_faces, verbose=verbose, max_fuzzy=max_fuzzy,
            )
            if result is not None:
                if verbose:
                    print(f"      5a: all {len(fillet_faces)} fillets removed")
                best_shape = result
                best_count = len(fillet_faces)
                accumulated = list(fillet_faces)
                last_history = history

                # Try remaining types on the fillet-free shape
                for cls_id in [2, 1, 4]:
                    type_faces = feature_faces_by_type.get(cls_id, [])
                    if not type_faces:
                        continue
                    remaining = _collect_remaining_faces(
                        type_faces, [], best_shape, last_history,
                    )
                    if not remaining:
                        continue

                    result, history, tol = _try_defeaturing_adaptive(
                        best_shape, remaining, verbose=verbose,
                        max_fuzzy=max_fuzzy,
                    )
                    if result is not None:
                        best_shape = result
                        best_count += len(remaining)
                        accumulated.extend(remaining)
                        last_history = history
                        if verbose:
                            print(f"      5a+: all {len(remaining)} "
                                  f"{DEFEATURE_CLASS_NAMES[cls_id]} removed")
                    else:
                        groups = _group_connected_faces(best_shape, remaining)
                        for group in groups:
                            result, history, tol = _try_defeaturing_adaptive(
                                best_shape, group, verbose=verbose,
                                max_fuzzy=max_fuzzy,
                            )
                            if result is not None:
                                best_shape = result
                                best_count += len(group)
                                accumulated.extend(group)
                                last_history = history
                                if verbose:
                                    print(f"      5a+: group of {len(group)} "
                                          f"{DEFEATURE_CLASS_NAMES[cls_id]}")

        # 5b: Reverse order (adaptive)
        if best_count == 0:
            if verbose:
                print(f"      5b: reverse order (cut→hole→chamfer→fillet)...")

            for cls_id in [4, 1, 2, 3]:
                type_faces = feature_faces_by_type.get(cls_id, [])
                if not type_faces:
                    continue
                candidate = accumulated + type_faces
                result, history, tol = _try_defeaturing_adaptive(
                    shape, candidate, verbose=verbose, max_fuzzy=max_fuzzy,
                )
                if result is not None:
                    accumulated = candidate
                    best_shape = result
                    best_count = len(accumulated)
                    last_history = history
                    if verbose:
                        print(f"      5b: all {len(type_faces)} "
                              f"{DEFEATURE_CLASS_NAMES[cls_id]}")
                    continue
                groups = _group_connected_faces(shape, type_faces)
                for group in groups:
                    candidate = accumulated + group
                    result, history, tol = _try_defeaturing_adaptive(
                        shape, candidate, verbose=verbose, max_fuzzy=max_fuzzy,
                    )
                    if result is not None:
                        accumulated = candidate
                        best_shape = result
                        best_count = len(accumulated)
                        last_history = history
                        if verbose:
                            print(f"      5b: group of {len(group)} "
                                  f"{DEFEATURE_CLASS_NAMES[cls_id]}")

        # 5c: Single-face iterative with adaptive tolerance + intermediate healing
        if best_count == 0:
            if verbose:
                print(f"      5c: single-face iterative (area-sorted, "
                      f"adaptive tolerance, intermediate healing)...")

            remaining_faces = _sort_faces_by_area(all_features)
            current_shape = shape
            removed_this_phase = 0
            max_rounds = 5

            for round_num in range(max_rounds):
                progress_round = 0
                still_remaining = []
                for face in remaining_faces:
                    # Check face still exists in current shape
                    face_map = TopTools_IndexedMapOfShape()
                    exp = TopExp_Explorer(current_shape, TopAbs_FACE)
                    while exp.More():
                        face_map.Add(topods.Face(exp.Current()))
                        exp.Next()
                    if not face_map.Contains(face):
                        # Removed as side-effect of another removal
                        removed_this_phase += 1
                        progress_round += 1
                        continue

                    result, history, tol = _try_defeaturing_adaptive(
                        current_shape, [face], verbose=False,
                        max_fuzzy=max_fuzzy,
                    )
                    if result is not None:
                        # Intermediate healing: unify faces to improve
                        # geometry before the next removal attempt
                        current_shape = _unify_faces(result)
                        removed_this_phase += 1
                        progress_round += 1
                        if verbose:
                            print(f"      5c round {round_num+1}: +1 face"
                                  + (f" (fuzzy={tol:.1e})" if tol > 0 else ""))
                    else:
                        still_remaining.append(face)

                remaining_faces = still_remaining
                if progress_round == 0 or not remaining_faces:
                    break

            if removed_this_phase > 0:
                best_shape = current_shape
                best_count = removed_this_phase
                if verbose:
                    print(f"      5c: {removed_this_phase} faces removed total")

        # Phase 5 iterative retry (same logic as Phase 4) if partial progress
        if 0 < best_count < total_features:
            iteration = 0
            max_iterations = 10
            while best_count < total_features and iteration < max_iterations:
                iteration += 1

                face_map = TopTools_IndexedMapOfShape()
                exp = TopExp_Explorer(best_shape, TopAbs_FACE)
                while exp.More():
                    face_map.Add(topods.Face(exp.Current()))
                    exp.Next()

                remaining = [f for f in all_features
                             if face_map.Contains(f) and f not in accumulated]
                if not remaining:
                    break

                progress = 0
                result, history, tol = _try_defeaturing_adaptive(
                    best_shape, remaining, verbose=verbose, max_fuzzy=max_fuzzy,
                )
                if result is not None:
                    best_shape = result
                    best_count += len(remaining)
                    break

                groups = _group_connected_faces(best_shape, remaining)
                for group in groups:
                    result, history, tol = _try_defeaturing_adaptive(
                        best_shape, group, verbose=verbose, max_fuzzy=max_fuzzy,
                    )
                    if result is not None:
                        best_shape = result
                        best_count += len(group)
                        accumulated.extend(group)
                        progress += len(group)
                    else:
                        for face in _sort_faces_by_area(group):
                            result, history, tol = _try_defeaturing_adaptive(
                                best_shape, [face], verbose=verbose,
                                max_fuzzy=max_fuzzy,
                            )
                            if result is not None:
                                best_shape = result
                                best_count += 1
                                accumulated.append(face)
                                progress += 1
                if progress == 0:
                    break

        if verbose and best_count > 0:
            print(f"      Phase 5 total: {best_count}/{total_features} removed")

    # --- Cleanup: multi-stage healing ---
    best_shape = _heal_shape(best_shape, verbose=verbose)

    failed = total_features - best_count
    return best_shape, best_count, failed


def defeature_step(step_path: str, face_preds: List[int],
                   output_path: str, verbose: bool = False,
                   max_fuzzy: float = 1e-3) -> dict:
    """Defeature a STEP file based on per-face predictions (v2 engine)."""
    from OCC.Core.STEPControl import (
        STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs,
    )
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.BRepCheck import BRepCheck_Analyzer

    # Read shape
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        return {"status": "error", "message": f"Failed to read {step_path}"}

    reader.TransferRoots()
    shape = reader.OneShape()

    # Enumerate faces
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

    # Run defeaturing (v2 engine)
    t0 = time.time()
    result_shape, removed, failed = defeature_shape(
        shape, feature_faces_by_type, verbose=verbose, max_fuzzy=max_fuzzy,
    )
    elapsed = time.time() - t0

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
        "elapsed_s": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Colored STEP helper
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
# Subprocess isolation for batch processing
# ---------------------------------------------------------------------------

def _file_worker(result_queue, step_path, preds, output_path,
                 colored_path, verbose, max_fuzzy, save_colored):
    """Process a single STEP file in an isolated worker process.

    All OpenCASCADE objects are confined to this process and destroyed
    on exit, preventing stale C++ state from corrupting the next file.
    """
    try:
        result = defeature_step(step_path, preds, output_path,
                                verbose=verbose, max_fuzzy=max_fuzzy)
        # Send result BEFORE colored export — if the export triggers a
        # C++ crash, we still have the defeaturing result.
        result_queue.put(result)
        if save_colored and result["status"] in ("success", "no_features"):
            try:
                save_colored_step(step_path, preds, colored_path)
            except Exception as e:
                if verbose:
                    print(f"    Warning: colored export failed: {e}")
        gc.collect()
    except Exception as e:
        result_queue.put({"status": "error", "message": str(e)})


def _run_file_isolated(step_path, preds, output_path, colored_path,
                       verbose=False, max_fuzzy=1e-3, save_colored=False,
                       timeout=300):
    """Run defeaturing in a subprocess for crash isolation.

    OpenCASCADE can throw unrecoverable C++ exceptions
    (e.g. Standard_NullObject) during batch processing when internal
    object references become stale.  Running each file in its own
    process prevents these crashes from killing the entire batch and
    ensures memory is fully reclaimed between files.
    """
    result_queue = mp.Queue()
    p = mp.Process(
        target=_file_worker,
        args=(result_queue, step_path, preds, output_path,
              colored_path, verbose, max_fuzzy, save_colored),
    )
    p.start()
    p.join(timeout=timeout)

    if p.exitcode is None:
        # Process hung — kill it
        p.kill()
        p.join()
        return {"status": "error",
                "message": f"Timed out after {timeout}s"}

    # Check queue first — the worker sends defeaturing results before
    # attempting colored export, so we may have a valid result even if
    # the process crashed during the export or cleanup phase.
    try:
        return result_queue.get_nowait()
    except Exception:
        pass

    if p.exitcode != 0:
        return {"status": "error",
                "message": f"OCC crash (exit code {p.exitcode})"}

    return {"status": "error", "message": "No result from worker"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """Main defeaturing pipeline (v2)."""
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

    max_fuzzy = args.max_fuzzy

    # ------------------------------------------------------------------
    # Single file mode
    # ------------------------------------------------------------------
    if args.step:
        step_path = args.step
        model_id = Path(step_path).stem
        print(f"\nDefeaturing {step_path} (v2 engine)...")

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
            step_path, preds, str(out_path),
            verbose=args.verbose, max_fuzzy=max_fuzzy,
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
        print(f"\nFound {len(step_files)} STEP files (v2 engine)")

        all_results = []
        success_count = 0
        no_feature_count = 0
        error_count = 0
        total_removed = 0
        total_failed = 0

        for idx, step_file in enumerate(step_files, 1):
            model_id = step_file.stem
            print(f"\n  [{idx}/{len(step_files)}] {step_file.name}...")

            # Get predictions (lightweight, no OCC — safe in main process)
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
                gc.collect()

            if preds is None:
                print(f"    Skipping: failed to get predictions")
                error_count += 1
                continue

            out_path = str(output_dir / f"{model_id}_defeatured.step")
            colored_path = (str(output_dir / f"{model_id}_colored.step")
                            if args.save_colored else None)

            # Run in isolated subprocess — prevents OCC C++ crashes
            # (e.g. Standard_NullObject) from killing the batch, and
            # ensures all OCC memory is reclaimed between files.
            result = _run_file_isolated(
                str(step_file), preds, out_path, colored_path,
                verbose=args.verbose, max_fuzzy=max_fuzzy,
                save_colored=args.save_colored,
                timeout=300,
            )
            result["model_id"] = model_id
            all_results.append(result)

            if result["status"] == "success":
                success_count += 1
                total_removed += result["removed"]
                total_failed += result["failed"]
                print(f"    Removed {result['removed']} features "
                      f"(failed: {result['failed']}, valid: {result['valid']}, "
                      f"{result.get('elapsed_s', '?')}s)")
            elif result["status"] == "no_features":
                no_feature_count += 1
                print(f"    No features detected")
            else:
                error_count += 1
                print(f"    Error: {result.get('message', 'unknown')}")

        # Batch report
        report = {
            "total": len(step_files),
            "success": success_count,
            "no_features": no_feature_count,
            "errors": error_count,
            "total_removed": total_removed,
            "total_failed": total_failed,
            "results": all_results,
        }
        report_path = output_dir / "batch_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n{'=' * 60}")
        print(f"Defeaturing complete (v2): {success_count} defeatured, "
              f"{no_feature_count} unchanged, {error_count} errors")
        print(f"  Total: {total_removed} faces removed, {total_failed} failed")
        print(f"Output: {output_dir}")
        print(f"Report: {report_path}")


def _print_result(result: dict):
    """Print a single defeaturing result summary."""
    status = result["status"]
    if status == "success":
        print(f"\n  Removed {result['removed']} feature faces "
              f"(failed: {result['failed']})")
        print(f"  Shape valid: {result['valid']}")
        elapsed = result.get("elapsed_s")
        if elapsed is not None:
            print(f"  Time: {elapsed}s")
        print(f"  Output: {result['output']}")
    elif status == "no_features":
        print(f"\n  {result['message']}")
        print(f"  Output: {result['output']}")
    else:
        print(f"\n  Error: {result.get('message', 'unknown')}")


if __name__ == "__main__":
    main()
