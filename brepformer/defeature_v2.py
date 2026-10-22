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

# Best 5-class face-segmentation checkpoint measured on defeature_processed/test
# (157 real hand-labelled parts): face macro F1 0.8886, chamfer 0.874. Beats
# trial1_ss1500 ep57 (0.8564) and every arm of the 11-run trial6 sweep (best 0.8579).
# The `f1=` in the filename is the *model-level* metric and does not rank face
# segmentation - see SCRIPTS.md, "Checkpoint filenames rank the wrong metric".
DEFAULT_CHECKPOINT = "results/trial1_finetune_ss/best-epoch=32-val/f1=0.9354.ckpt"

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
    parser.add_argument("--primitives", action="store_true",
                        help="After defeaturing, fit geometric primitives to the result")
    parser.add_argument("--primitives_tol", type=float, default=0.1,
                        help="Geometric tolerance for primitive face grouping (default: 0.1)")
    parser.add_argument("--no_symmetry", action="store_true",
                        help="Disable symmetry completion (on by default)")
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
    import traceback
    from brepformer.infer import infer_single

    try:
        result = infer_single(model, step_path, device)
    except Exception as e:
        traceback.print_exc()
        print(f"Inference failed for {step_path}: {e}", file=sys.stderr)
        return None

    if "error" in result:
        print(f"Inference error for {step_path}: {result['error']}", file=sys.stderr)
        return None

    face_preds = result.get("face_preds")
    if face_preds is None:
        print(f"No face predictions returned for {step_path} "
              f"(model may lack face segmentation head)", file=sys.stderr)
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
# Symmetry completion — fill near-symmetric prediction gaps
# ---------------------------------------------------------------------------

import math as _math


def _extract_face_geometry(faces: list) -> List[dict]:
    """Extract lightweight geometry for symmetry analysis."""
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import (
        GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
        GeomAbs_Sphere, GeomAbs_Torus,
    )
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop

    infos = []
    for face in faces:
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        area = props.Mass()
        cm = props.CentreOfMass()
        centroid = (cm.X(), cm.Y(), cm.Z())

        sa = BRepAdaptor_Surface(face)
        stype = sa.GetType()

        info = {"area": area, "centroid": centroid, "axis": None, "center": None,
                "stype": stype}

        if stype == GeomAbs_Cylinder:
            cyl = sa.Cylinder()
            ax = cyl.Axis().Direction()
            loc = cyl.Location()
            info["axis"] = (ax.X(), ax.Y(), ax.Z())
            info["center"] = (loc.X(), loc.Y(), loc.Z())
        elif stype == GeomAbs_Cone:
            cone = sa.Cone()
            ax = cone.Axis().Direction()
            loc = cone.Location()
            info["axis"] = (ax.X(), ax.Y(), ax.Z())
            info["center"] = (loc.X(), loc.Y(), loc.Z())
        elif stype == GeomAbs_Torus:
            tor = sa.Torus()
            ax = tor.Axis().Direction()
            loc = tor.Location()
            info["axis"] = (ax.X(), ax.Y(), ax.Z())
            info["center"] = (loc.X(), loc.Y(), loc.Z())

        infos.append(info)
    return infos


def _compute_bbox_diagonal(shape) -> float:
    """Compute bounding box diagonal as characteristic length."""
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib

    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    return _math.sqrt(dx * dx + dy * dy + dz * dz)


def _vec_normalize(v: tuple) -> tuple:
    mag = _math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if mag < 1e-12:
        return (0.0, 0.0, 1.0)
    return (v[0] / mag, v[1] / mag, v[2] / mag)


def _vec_dot(a: tuple, b: tuple) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vec_cross(a: tuple, b: tuple) -> tuple:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _vec_sub(a: tuple, b: tuple) -> tuple:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_add(a: tuple, b: tuple) -> tuple:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vec_scale(v: tuple, s: float) -> tuple:
    return (v[0] * s, v[1] * s, v[2] * s)


def _vec_length(v: tuple) -> float:
    return _math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def _axes_collinear(center_a, axis_a, center_b, axis_b, tol=0.05):
    """Check if two directed axes are approximately collinear."""
    dot = abs(_vec_dot(axis_a, axis_b))
    if dot < 1.0 - tol:
        return False
    diff = _vec_sub(center_b, center_a)
    cross = _vec_cross(axis_a, diff)
    dist = _vec_length(cross)
    return dist < tol * 50


def _find_candidate_axes(face_infos: List[dict], bbox_diag: float
                         ) -> List[Tuple[tuple, tuple, float]]:
    """Find candidate rotational symmetry axes from curved-face axes."""
    axis_clusters: List[Tuple[tuple, tuple, float]] = []
    dist_tol = 0.01 * bbox_diag

    for info in face_infos:
        if info["axis"] is None:
            continue
        ax = _vec_normalize(info["axis"])
        center = info["center"]
        area = info["area"]

        merged = False
        for idx, (cl_center, cl_axis, cl_weight) in enumerate(axis_clusters):
            if _axes_collinear(cl_center, cl_axis, center, ax, tol=0.05):
                new_weight = cl_weight + area
                axis_clusters[idx] = (cl_center, cl_axis, new_weight)
                merged = True
                break
        if not merged:
            axis_clusters.append((center, ax, area))

    axis_clusters.sort(key=lambda x: x[2], reverse=True)

    if not axis_clusters:
        centroids = [info["centroid"] for info in face_infos]
        n = len(centroids)
        cx = sum(c[0] for c in centroids) / n
        cy = sum(c[1] for c in centroids) / n
        cz = sum(c[2] for c in centroids) / n
        model_center = (cx, cy, cz)
        axis_clusters = [
            (model_center, (0, 0, 1), 1.0),
            (model_center, (0, 1, 0), 1.0),
            (model_center, (1, 0, 0), 1.0),
        ]

    return axis_clusters[:3]


def _find_angular_orbits(
    face_infos: List[dict],
    face_preds: List[int],
    axis_point: tuple,
    axis_dir: tuple,
    bbox_diag: float,
) -> List[List[int]]:
    """Group face indices into angular equivalence orbits around an axis."""
    tol_axial = 0.01 * bbox_diag
    tol_radial = 0.01 * bbox_diag
    area_ratio_tol = 0.15

    ad = _vec_normalize(axis_dir)

    # Build perpendicular frame
    if abs(ad[0]) < 0.9:
        perp1 = _vec_normalize(_vec_cross(ad, (1, 0, 0)))
    else:
        perp1 = _vec_normalize(_vec_cross(ad, (0, 1, 0)))
    perp2 = _vec_cross(ad, perp1)

    # Compute cylindrical coordinates for each face
    fingerprints = []
    for i, info in enumerate(face_infos):
        c = info["centroid"]
        diff = _vec_sub(c, axis_point)
        axial_pos = _vec_dot(diff, ad)
        proj_on_axis = _vec_scale(ad, axial_pos)
        radial_vec = _vec_sub(diff, proj_on_axis)
        radial_dist = _vec_length(radial_vec)
        theta = _math.atan2(_vec_dot(radial_vec, perp2),
                            _vec_dot(radial_vec, perp1))
        fingerprints.append((axial_pos, radial_dist, theta, info["area"],
                             info["stype"]))

    # Group by (axial_pos, radial_dist, area, stype) — ignoring angle
    orbit_map: Dict[int, List[int]] = {}
    assigned = [False] * len(face_infos)

    for i in range(len(face_infos)):
        if assigned[i]:
            continue
        orbit = [i]
        assigned[i] = True
        ax_i, rad_i, _, area_i, stype_i = fingerprints[i]

        for j in range(i + 1, len(face_infos)):
            if assigned[j]:
                continue
            ax_j, rad_j, _, area_j, stype_j = fingerprints[j]
            if stype_i != stype_j:
                continue
            if abs(ax_i - ax_j) > tol_axial:
                continue
            if abs(rad_i - rad_j) > tol_radial:
                continue
            max_area = max(area_i, area_j, 1e-12)
            if abs(area_i - area_j) / max_area > area_ratio_tol:
                continue
            orbit.append(j)
            assigned[j] = True

        if len(orbit) >= 3:
            orbit_map[i] = orbit

    # Validate each orbit: check angular regularity (N-fold symmetry)
    valid_orbits = []
    for _, orbit in orbit_map.items():
        n = len(orbit)
        angles = sorted(fingerprints[i][2] for i in orbit)
        gaps = []
        for k in range(len(angles) - 1):
            gaps.append(angles[k + 1] - angles[k])
        wrap_gap = (2 * _math.pi) - (angles[-1] - angles[0])
        gaps.append(wrap_gap)

        expected_gap = 2 * _math.pi / n
        tol_angle = 0.12  # ~7 degrees
        regular = all(abs(g - expected_gap) < tol_angle for g in gaps)

        if regular:
            valid_orbits.append(orbit)

    return valid_orbits


def _complete_rotational_symmetry(
    face_infos: List[dict],
    face_preds: List[int],
    candidate_axes: List[Tuple[tuple, tuple, float]],
    bbox_diag: float,
    max_additions: int,
    verbose: bool = False,
) -> Tuple[List[int], int]:
    """Fill missed predictions using rotational symmetry."""
    preds = list(face_preds)
    total_added = 0

    for axis_point, axis_dir, _ in candidate_axes:
        if total_added >= max_additions:
            break

        orbits = _find_angular_orbits(
            face_infos, preds, axis_point, axis_dir, bbox_diag)

        for orbit in orbits:
            if total_added >= max_additions:
                break

            n = len(orbit)
            marked = [i for i in orbit if preds[i] != 0]
            unmarked = [i for i in orbit if preds[i] == 0]

            if not unmarked or not marked:
                continue
            if len(marked) / n < 0.65:
                continue

            # Check type agreement: majority label
            from collections import Counter as _Counter
            label_counts = _Counter(preds[i] for i in marked)
            majority_label, majority_count = label_counts.most_common(1)[0]
            if majority_count / len(marked) < 0.75:
                continue

            # Propagate
            for i in unmarked:
                if total_added >= max_additions:
                    break
                preds[i] = majority_label
                total_added += 1

            if verbose and unmarked:
                label_name = (DEFEATURE_CLASS_NAMES[majority_label]
                              if majority_label < len(DEFEATURE_CLASS_NAMES)
                              else "?")
                print(f"  Symmetry (rotational): orbit of {n}, "
                      f"filled {len(unmarked)} as {label_name}")

    return preds, total_added


def _find_candidate_mirror_planes(face_infos: List[dict]
                                  ) -> List[Tuple[tuple, tuple]]:
    """Find candidate mirror planes (point_on_plane, normal)."""
    centroids = [info["centroid"] for info in face_infos]
    n = len(centroids)
    cx = sum(c[0] for c in centroids) / n
    cy = sum(c[1] for c in centroids) / n
    cz = sum(c[2] for c in centroids) / n
    model_center = (cx, cy, cz)

    # Cardinal planes through model center
    planes = [
        (model_center, (1, 0, 0)),
        (model_center, (0, 1, 0)),
        (model_center, (0, 0, 1)),
    ]
    return planes


def _reflect_point(point: tuple, plane_point: tuple, plane_normal: tuple) -> tuple:
    """Reflect a point across a plane."""
    n = _vec_normalize(plane_normal)
    diff = _vec_sub(point, plane_point)
    dist = _vec_dot(diff, n)
    return _vec_sub(point, _vec_scale(n, 2 * dist))


def _complete_mirror_symmetry(
    face_infos: List[dict],
    face_preds: List[int],
    bbox_diag: float,
    max_additions: int,
    verbose: bool = False,
) -> Tuple[List[int], int]:
    """Fill missed predictions using mirror symmetry."""
    preds = list(face_preds)
    total_added = 0
    mirror_dist_tol = 0.05 * bbox_diag
    area_ratio_tol = 0.15
    min_match_ratio = 0.60
    min_pred_symmetry = 0.75

    planes = _find_candidate_mirror_planes(face_infos)

    for plane_point, plane_normal in planes:
        if total_added >= max_additions:
            break

        # Match face pairs across this plane (proper bipartite)
        n = len(face_infos)
        matched_pairs = []
        used = set()

        for i in range(n):
            if i in used:
                continue
            reflected = _reflect_point(
                face_infos[i]["centroid"], plane_point, plane_normal)

            best_j = -1
            best_dist = mirror_dist_tol

            for j in range(n):
                if j == i or j in used:
                    continue
                if face_infos[i]["stype"] != face_infos[j]["stype"]:
                    continue
                max_area = max(face_infos[i]["area"], face_infos[j]["area"], 1e-12)
                if abs(face_infos[i]["area"] - face_infos[j]["area"]) / max_area > area_ratio_tol:
                    continue
                dist = _vec_length(_vec_sub(reflected, face_infos[j]["centroid"]))
                if dist < best_dist:
                    best_dist = dist
                    best_j = j

            if best_j >= 0:
                matched_pairs.append((i, best_j))
                used.add(i)
                used.add(best_j)

        # Score: how many faces found a mirror partner
        match_ratio = len(matched_pairs) / max(n / 2, 1)
        if match_ratio < min_match_ratio:
            continue

        # Check prediction symmetry among feature-containing pairs
        both_marked = 0
        one_marked_pairs = []
        for i, j in matched_pairs:
            pi, pj = preds[i], preds[j]
            if pi != 0 and pj != 0:
                both_marked += 1
            elif pi != 0 and pj == 0:
                one_marked_pairs.append((i, j))
            elif pj != 0 and pi == 0:
                one_marked_pairs.append((j, i))

        feature_pairs = both_marked + len(one_marked_pairs)
        if feature_pairs == 0:
            continue
        pred_symmetry = both_marked / feature_pairs
        if pred_symmetry < min_pred_symmetry:
            continue
        if not one_marked_pairs:
            continue

        # Propagate: copy label from marked face to its unmarked counterpart
        for marked_idx, unmarked_idx in one_marked_pairs:
            if total_added >= max_additions:
                break
            preds[unmarked_idx] = preds[marked_idx]
            total_added += 1

        if verbose and one_marked_pairs:
            print(f"  Symmetry (mirror): {len(one_marked_pairs)} faces filled "
                  f"(match_ratio={match_ratio:.2f}, "
                  f"pred_symmetry={pred_symmetry:.2f})")

    return preds, total_added


def _complete_symmetry(
    shape,
    faces: list,
    face_preds: List[int],
    verbose: bool = False,
) -> List[int]:
    """Analyze shape geometry and fill near-symmetric prediction gaps.

    Called in defeature_step() after face enumeration, before grouping by type.
    Returns updated face_preds (or original if no changes warranted).
    """
    num_faces = len(faces)
    if num_faces < 6:
        return face_preds

    # No cap — allow symmetry to complete all detected patterns
    max_additions = num_faces

    face_infos = _extract_face_geometry(faces)
    bbox_diag = _compute_bbox_diagonal(shape)

    if bbox_diag < 1e-9:
        return face_preds

    # Rotational symmetry first (more constrained, higher confidence)
    candidate_axes = _find_candidate_axes(face_infos, bbox_diag)
    preds, rot_added = _complete_rotational_symmetry(
        face_infos, face_preds, candidate_axes, bbox_diag,
        max_additions, verbose)

    # Mirror symmetry (uses remaining budget)
    remaining_budget = max_additions - rot_added
    if remaining_budget > 0:
        preds, mir_added = _complete_mirror_symmetry(
            face_infos, preds, bbox_diag, remaining_budget, verbose)
    else:
        mir_added = 0

    total_added = rot_added + mir_added
    if verbose and total_added > 0:
        print(f"  Symmetry completion: {total_added} faces added "
              f"({rot_added} rotational, {mir_added} mirror)")

    return preds


# ---------------------------------------------------------------------------
# Geometric prediction expansion — catch missed features
# ---------------------------------------------------------------------------

def _expand_predictions_geometric(
    shape,
    faces: list,
    face_preds: List[int],
    verbose: bool = False,
) -> List[int]:
    """Propagate feature labels to unlabeled neighbors using geometric cues.

    The ML model frequently under-predicts: it labels some faces of a feature
    but misses others that are geometrically part of the same feature. This
    function catches those gaps by analyzing adjacency + surface geometry.

    Rules:
    1. Co-axial cylinder/cone propagation: if an unlabeled cylindrical face
       shares the same axis as a labeled hole/cut neighbor, label it.
    2. Small-face absorption: unlabeled faces that are small (below median area)
       and completely surrounded by feature faces get absorbed.
    3. Adjacent same-surface propagation: unlabeled planar faces adjacent to
       multiple labeled cut/hole faces with the same surface normal → label.
    4. Enclosed face propagation: unlabeled faces whose ALL edge-neighbors are
       features get labeled (they're interior to the feature volume).
    """
    from OCC.Core.TopExp import TopExp_Explorer, topexp
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCC.Core.TopoDS import topods
    from OCC.Core.TopTools import (
        TopTools_IndexedDataMapOfShapeListOfShape,
        TopTools_IndexedMapOfShape,
        TopTools_ListIteratorOfListOfShape,
    )
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop

    preds = list(face_preds)
    num_faces = len(faces)
    if num_faces < 6:
        return preds

    # Build adjacency map
    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

    face_index_map = TopTools_IndexedMapOfShape()
    for f in faces:
        face_index_map.Add(f)

    # Compute adjacency as face-index → set of face-indices
    adjacency: Dict[int, set] = {i: set() for i in range(num_faces)}
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
                    j_idx = face_index_map.FindIndex(neighbor)
                    if j_idx > 0:
                        j = j_idx - 1
                        if j != i:
                            adjacency[i].add(j)
                    it.Next()
            edge_exp.Next()

    # Extract geometry for all faces
    face_geom = []
    areas = []
    for face in faces:
        sa = BRepAdaptor_Surface(face)
        stype = sa.GetType()
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        area = props.Mass()
        areas.append(area)

        info = {"stype": stype, "area": area, "axis": None, "normal": None,
                "radius": None}

        if stype == GeomAbs_Cylinder:
            cyl = sa.Cylinder()
            ax = cyl.Axis().Direction()
            loc = cyl.Location()
            info["axis"] = (ax.X(), ax.Y(), ax.Z())
            info["center"] = (loc.X(), loc.Y(), loc.Z())
            info["radius"] = cyl.Radius()
        elif stype == GeomAbs_Cone:
            cone = sa.Cone()
            ax = cone.Axis().Direction()
            loc = cone.Location()
            info["axis"] = (ax.X(), ax.Y(), ax.Z())
            info["center"] = (loc.X(), loc.Y(), loc.Z())
        elif stype == GeomAbs_Plane:
            pln = sa.Plane()
            norm = pln.Axis().Direction()
            info["normal"] = (norm.X(), norm.Y(), norm.Z())

        face_geom.append(info)

    median_area = sorted(areas)[num_faces // 2] if num_faces > 0 else 0
    total_added = 0

    # Rule 1: Co-axial cylinder/cone propagation
    for i in range(num_faces):
        if preds[i] != 0:
            continue
        gi = face_geom[i]
        if gi["stype"] not in (GeomAbs_Cylinder, GeomAbs_Cone):
            continue
        if gi["axis"] is None:
            continue

        # Check if any neighbor is a labeled feature with same axis
        for j in adjacency[i]:
            if preds[j] == 0:
                continue
            gj = face_geom[j]
            if gj["axis"] is None:
                continue
            if _axes_collinear(gi.get("center", (0, 0, 0)), gi["axis"],
                               gj.get("center", (0, 0, 0)), gj["axis"],
                               tol=0.08):
                preds[i] = preds[j]
                total_added += 1
                break

    # Rule 2: Small-face absorption — small unlabeled faces surrounded by features
    for i in range(num_faces):
        if preds[i] != 0:
            continue
        if areas[i] > median_area:
            continue
        neighbors = adjacency[i]
        if not neighbors:
            continue
        labeled_neighbors = [j for j in neighbors if preds[j] != 0]
        if len(labeled_neighbors) >= len(neighbors) * 0.7:
            # Majority of neighbors are features — absorb with majority label
            from collections import Counter as _Ctr
            labels = _Ctr(preds[j] for j in labeled_neighbors)
            majority_label = labels.most_common(1)[0][0]
            preds[i] = majority_label
            total_added += 1

    # Rule 3: Adjacent planar propagation — unlabeled planar faces adjacent
    # to 2+ labeled cuts/holes with compatible normals
    for i in range(num_faces):
        if preds[i] != 0:
            continue
        gi = face_geom[i]
        if gi["stype"] != GeomAbs_Plane or gi["normal"] is None:
            continue
        neighbors = adjacency[i]
        feature_neighbors = [j for j in neighbors if preds[j] in (1, 4)]
        if len(feature_neighbors) < 2:
            continue
        # Check if this face's normal is parallel or anti-parallel to a
        # neighbor's feature face normal (part of the same planar cut)
        for j in feature_neighbors:
            gj = face_geom[j]
            if gj["normal"] is not None:
                dot = abs(_vec_dot(gi["normal"], gj["normal"]))
                if dot > 0.95:
                    preds[i] = preds[j]
                    total_added += 1
                    break

    # Rule 4: Enclosed face propagation — faces ALL of whose neighbors are features
    # Run iteratively until no more changes (cascading enclosure)
    for _ in range(3):
        changed = False
        for i in range(num_faces):
            if preds[i] != 0:
                continue
            neighbors = adjacency[i]
            if not neighbors:
                continue
            if all(preds[j] != 0 for j in neighbors):
                from collections import Counter as _Ctr2
                labels = _Ctr2(preds[j] for j in neighbors)
                preds[i] = labels.most_common(1)[0][0]
                total_added += 1
                changed = True
        if not changed:
            break

    if verbose and total_added > 0:
        print(f"  Geometric expansion: {total_added} faces added")

    return preds


# ---------------------------------------------------------------------------
# Core defeaturing engine (ENHANCED in v2)
# ---------------------------------------------------------------------------

def _count_faces(shape) -> int:
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    n = 0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        n += 1
        exp.Next()
    return n


def _try_defeaturing(shape, faces_to_remove: list,
                     fuzzy_value: float = 0.0):
    """Attempt face removal with optional fuzzy tolerance.

    Returns (result_shape, history) on success, (None, None) on failure.
    The BRepTools_History tracks what happened to every face in the shape
    and is used for precise face mapping between phases.

    A result is rejected as degenerate if it retains fewer than half
    the expected surviving faces (OCC sometimes reports success but
    produces a nearly-empty shape on invalid input geometry).
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
        result = op.Shape()
        orig_faces = _count_faces(shape)
        result_faces = _count_faces(result)

        # Reject if no faces were actually removed
        if result_faces >= orig_faces:
            return None, None

        # Reject if actual reduction is far too small compared to requested —
        # OCC can report success while only performing trivial topology changes
        # (e.g. when asked to remove all faces of a solid)
        actual_reduction = orig_faces - result_faces
        if len(faces_to_remove) > 2:
            min_reduction = max(1, len(faces_to_remove) // 10)
            if actual_reduction < min_reduction:
                return None, None

        # Reject degenerate results (nearly-empty shape)
        expected_min = max((orig_faces - len(faces_to_remove)) // 2, 4)
        if result_faces < expected_min:
            return None, None

        history = None
        try:
            history = op.History()
        except Exception:
            pass
        return result, history
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


def _remove_residual_faces(shape, verbose: bool = False,
                           max_fuzzy: float = 1e-3):
    """Remove residual small faces left over after defeaturing.

    After BRepAlgoAPI_Defeaturing removes declared features (e.g. pocket
    bottoms), the side walls of those features may remain as small faces
    if the ML model didn't label them. These prevent clean face unification.

    Strategy: sort faces by area and find the largest multiplicative gap in
    the area distribution. Faces below the gap are residual walls; faces
    above are the base geometry. Remove all faces below the gap.
    """
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop

    # Collect all faces with their areas
    face_areas = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods.Face(exp.Current())
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        face_areas.append((face, props.Mass()))
        exp.Next()

    if len(face_areas) <= 6:
        return shape

    # Sort by area
    face_areas.sort(key=lambda x: x[1])
    areas_only = [a for _, a in face_areas]

    # Find the first significant multiplicative gap (>= 10x) in the upper
    # portion of the distribution. This separates residual feature walls
    # (small faces) from the base geometry (large faces that form the stock).
    # We want the LOWEST such gap — the first transition from "small stuff"
    # to "real geometry" — not the largest gap (which might separate the
    # plate top from plate sides).
    best_gap_ratio = 1.0
    best_gap_idx = -1
    start_idx = max(1, int(len(areas_only) * 0.5))
    for i in range(start_idx, len(areas_only) - 1):
        if areas_only[i] > 0:
            ratio = areas_only[i + 1] / areas_only[i]
            if ratio >= 10.0:
                best_gap_ratio = ratio
                best_gap_idx = i
                break  # Take the first significant gap

    # Only proceed if there's a significant gap (> 5x jump in area)
    if best_gap_ratio < 5.0 or best_gap_idx < 0:
        return shape

    # Everything at or below best_gap_idx is a residual small face
    area_threshold = areas_only[best_gap_idx]
    small_faces = [f for f, a in face_areas if a <= area_threshold]
    remaining_count = len(face_areas) - len(small_faces)

    # Safety: must keep at least 4 faces (minimum for a closed solid)
    if remaining_count < 4:
        return shape

    if not small_faces:
        return shape

    if verbose:
        print(f"    Phase 6: {len(small_faces)} residual small faces "
              f"(threshold={area_threshold:.4f}, gap={best_gap_ratio:.1f}x)")

    # Try removing all at once
    result, _, _ = _try_defeaturing_adaptive(
        shape, small_faces, verbose=False, max_fuzzy=max_fuzzy)
    if result is not None:
        if verbose:
            print(f"      Removed all {len(small_faces)} residual faces at once")
        return result

    # Try in groups by connected components
    groups = _group_connected_faces(shape, small_faces)
    current = shape
    removed = 0
    for group in groups:
        result, _, _ = _try_defeaturing_adaptive(
            current, group, verbose=False, max_fuzzy=max_fuzzy)
        if result is not None:
            current = result
            removed += len(group)
        else:
            for face in _sort_faces_by_area(group):
                result, _, _ = _try_defeaturing_adaptive(
                    current, [face], verbose=False, max_fuzzy=max_fuzzy)
                if result is not None:
                    current = result
                    removed += 1

    if verbose and removed > 0:
        print(f"      Removed {removed}/{len(small_faces)} residual faces")

    return current


def _protect_structural_faces(shape, faces: list, face_preds: List[int],
                              verbose: bool = False) -> List[int]:
    """Reclassify structural faces back to 'random' (keep).

    Protects faces that define the primary 3D form from being defeatured.
    Uses area-based, geometry-based, and adjacency-based heuristics:

    1. Large-area dominance: faces >12% of total surface area
    2. Body cylinders/cones: large curved surfaces that form the main body
    3. Isolated feature faces: faces whose neighbors are mostly random (stock)
       — these are boundaries of protrusions, not machining cuts
    4. Connected protrusion groups: clusters of feature faces that together
       form a positive feature (boss, prism, rib) rather than a cut
    """
    from OCC.Core.TopExp import TopExp_Explorer, topexp
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
    from OCC.Core.TopoDS import topods
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Cone
    from OCC.Core.TopTools import (TopTools_IndexedDataMapOfShapeListOfShape,
                                   TopTools_ListIteratorOfListOfShape)

    if len(faces) < 6:
        return face_preds

    # Compute face areas
    areas = []
    for face in faces:
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        areas.append(props.Mass())

    total_area = sum(areas)
    if total_area <= 0:
        return face_preds

    # Build adjacency map
    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

    adjacency = [set() for _ in range(len(faces))]
    for i, face in enumerate(faces):
        edge_exp = TopExp_Explorer(face, TopAbs_EDGE)
        while edge_exp.More():
            edge = topods.Edge(edge_exp.Current())
            idx = edge_face_map.FindIndex(edge)
            if idx > 0:
                adj_list = edge_face_map.FindFromIndex(idx)
                it = TopTools_ListIteratorOfListOfShape(adj_list)
                while it.More():
                    adj_face = topods.Face(it.Value())
                    for k, f in enumerate(faces):
                        if f.IsSame(adj_face) and k != i:
                            adjacency[i].add(k)
                    it.Next()
            edge_exp.Next()

    # Collect feature face areas for comparison
    feature_areas = [areas[i] for i in range(len(face_preds))
                     if face_preds[i] != 0]
    if not feature_areas:
        return face_preds

    median_feature_area = sorted(feature_areas)[len(feature_areas) // 2]
    random_count = sum(1 for p in face_preds if p == 0)

    preds = list(face_preds)
    protected = set()

    # --- Pass 1: Area and geometry rules ---
    from OCC.Core.GeomAbs import GeomAbs_Plane

    # Compute area statistics for random vs feature faces
    random_total_area = sum(areas[i] for i in range(len(preds)) if preds[i] == 0)
    feature_total_area = total_area - random_total_area
    random_areas_list = sorted([areas[i] for i in range(len(preds)) if preds[i] == 0])
    median_random_area = (random_areas_list[len(random_areas_list) // 2]
                          if random_areas_list else 0)

    # Detect "inverted model" — the model labeled the primary body as features.
    # Signal: ALL random faces are small (none exceeds 5% of total area),
    # meaning every random prediction landed on tiny connecting faces, not
    # stock. If any random face is large, the model correctly found stock.
    max_random_area = max(random_areas_list) if random_areas_list else 0
    inverted_model = (max_random_area < total_area * 0.05
                      and median_random_area < median_feature_area * 0.15
                      and random_count >= 4)

    for i, (face, pred) in enumerate(zip(faces, preds)):
        if pred == 0:
            continue

        area_fraction = areas[i] / total_area
        sa = BRepAdaptor_Surface(face)
        stype = sa.GetType()

        protect = False

        # Rule 1: Large planar face (>15% total area) is structural stock.
        if area_fraction > 0.15 and stype == GeomAbs_Plane:
            protect = True

        # Rule 2: Large cylinder/cone (>15% total area) with few neighbors.
        # Body-defining cylinders (shafts) only touch a few faces at their
        # ends. Internal bores cut through the shape and touch many faces.
        # A large curved surface with <=6 neighbors is a body surface.
        if (area_fraction > 0.15
                and stype in (GeomAbs_Cylinder, GeomAbs_Cone)
                and pred in (2, 4)
                and len(adjacency[i]) <= 6):
            protect = True

        # Rule 3 (inverted model): When features dominate and random faces
        # are small, protect any feature face larger than the median random
        # face — the model has confused the primary body for features.
        if inverted_model and areas[i] > median_random_area * 2.0:
            protect = True

        if protect:
            protected.add(i)

    # --- Pass 2: Adjacency-based rules ---
    pass1_seeds = set(protected)  # Snapshot before Pass 2

    for i, (face, pred) in enumerate(zip(faces, preds)):
        if pred == 0 or i in protected:
            continue

        neighbors = adjacency[i]
        if not neighbors:
            continue

        # Count neighbor types (treating already-protected as random)
        feature_neighbors = sum(1 for n in neighbors
                                if preds[n] != 0 and n not in protected)
        random_neighbors = len(neighbors) - feature_neighbors

        sa_i = BRepAdaptor_Surface(faces[i])
        stype_i = sa_i.GetType()

        # Rule 3: Planar face entirely surrounded by random/protected faces.
        # This is a terminal face of a protrusion (boss top, rib cap) —
        # removing it destroys positive geometry. Only applies to planes;
        # cylinders/cones surrounded by random are legitimate holes/chamfers.
        if (feature_neighbors == 0 and random_neighbors >= 2
                and stype_i == GeomAbs_Plane):
            protected.add(i)
            continue

        # Rule 4: Large planar face (>5% total area) mostly surrounded by
        # random (>=80% random neighbors) — likely a protrusion wall or
        # stock face mislabeled as a feature.
        if (stype_i == GeomAbs_Plane
                and random_neighbors / len(neighbors) >= 0.80
                and areas[i] / total_area > 0.05):
            protected.add(i)

    # --- Pass 3: Propagate protection from safe seeds ---
    # Propagate from:
    #  - Pass 2 seeds (protrusion caps with all-random neighbors)
    #  - Pass 1 curved seeds (body cylinders/cones) — these are body surfaces
    #    whose neighbors are protrusion walls, not cut walls
    # Do NOT propagate from Pass 1 planar seeds (large stock faces) — cuts
    # naturally border stock planes and would cascade.
    pass2_seeds = protected - pass1_seeds

    # Include curved Pass-1 seeds as propagation sources
    propagation_seeds = set(pass2_seeds)
    for i in pass1_seeds:
        sa_i = BRepAdaptor_Surface(faces[i])
        if sa_i.GetType() in (GeomAbs_Cylinder, GeomAbs_Cone):
            propagation_seeds.add(i)

    if propagation_seeds:
        changed = True
        iterations = 0
        max_iterations = 2
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            for i in range(len(faces)):
                if preds[i] == 0 or i in protected:
                    continue
                neighbors = adjacency[i]
                if not neighbors:
                    continue
                # Must be planar
                sa_i = BRepAdaptor_Surface(faces[i])
                if sa_i.GetType() != GeomAbs_Plane:
                    continue
                # Must touch a propagation seed
                touches_seed = any(n in propagation_seeds for n in neighbors)
                if not touches_seed:
                    continue
                # Must also touch at least one original random face
                touches_random = any(face_preds[n] == 0 for n in neighbors)
                if not touches_random:
                    continue
                feature_neighbors = sum(1 for n in neighbors
                                        if preds[n] != 0 and n not in protected)
                if feature_neighbors <= 1:
                    protected.add(i)
                    propagation_seeds.add(i)
                    changed = True

    # Apply protection
    for i in protected:
        preds[i] = 0
        random_count += 1

    if verbose and protected:
        print(f"  Structural protection: {len(protected)} faces reclassified "
              f"as random (faces {sorted(protected)})")

    return preds


def _remove_enclosed_remnants(shape, verbose: bool = False,
                              max_fuzzy: float = 1e-3):
    """Remove small faces that became isolated after defeaturing.

    After features are removed, some originally-"random" faces may now be
    isolated remnants — small planar or cylindrical faces that were part of
    a feature's interior (pocket floors, hole walls) but weren't labeled by
    the model. These are identifiable because they're small relative to the
    remaining stock faces and form isolated clusters.

    Criteria for removal:
    - Face area is below 25% of the median remaining face area
    - OR face is a cylinder/cone (likely a hole wall remnant)
    - Only removes if the result is still valid
    """
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Cone
    from OCC.Core.BRepCheck import BRepCheck_Analyzer

    face_data = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods.Face(exp.Current())
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        area = props.Mass()
        sa = BRepAdaptor_Surface(face)
        stype = sa.GetType()
        face_data.append((face, area, stype))
        exp.Next()

    if len(face_data) <= 6:
        return shape

    areas = sorted(a for _, a, _ in face_data)
    median_area = areas[len(areas) // 2]
    threshold = median_area * 0.25

    # Identify candidate remnants: small faces or cylindrical/conical faces
    # that are much smaller than the stock
    candidates = []
    for face, area, stype in face_data:
        if area < threshold:
            candidates.append(face)
        elif stype in (GeomAbs_Cylinder, GeomAbs_Cone) and area < median_area * 0.4:
            candidates.append(face)

    if not candidates:
        return shape

    # Safety: don't remove more than 40% of remaining faces
    if len(candidates) > len(face_data) * 0.4:
        return shape

    # Try removing all candidates at once
    result, _, _ = _try_defeaturing_adaptive(
        shape, candidates, verbose=False, max_fuzzy=max_fuzzy)
    if result is not None:
        analyzer = BRepCheck_Analyzer(result)
        if analyzer.IsValid():
            if verbose:
                print(f"    Remnant cleanup: removed {len(candidates)} "
                      f"small/cylindrical faces")
            return result

    # Try individually
    current = shape
    removed = 0
    for face in _sort_faces_by_area(candidates):
        result, _, _ = _try_defeaturing_adaptive(
            current, [face], verbose=False, max_fuzzy=max_fuzzy)
        if result is not None:
            current = result
            removed += 1

    if verbose and removed > 0:
        print(f"    Remnant cleanup: removed {removed}/{len(candidates)} faces")

    return current


def _try_boolean_fill(shape, feature_faces: list,
                      verbose: bool = False, max_fuzzy: float = 1e-3):
    """Fill complex features using geometry-aware Boolean operations.

    Multiple strategies are tried for features that BRepAlgoAPI_Defeaturing
    cannot handle (pockets, through-cuts, complex holes):

    Strategy A: ThickSolid — remove feature faces from the shell and let OCC
                reconstruct the missing boundary via offset surfaces.
    Strategy B: Prism fill — extrude the boundary wire of the feature opening
                to create a fill solid, then fuse it with the shape.
    Strategy C: Sew + MakeSolid — for closed feature groups, sew faces into a
                shell, make a solid plug, and fuse.
    Strategy D: Per-group iterative — try each connected component independently.

    Returns the filled shape on success, None on failure.
    """
    from OCC.Core.BRepBuilderAPI import (
        BRepBuilderAPI_Sewing,
        BRepBuilderAPI_MakeSolid,
    )
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.TopoDS import topods
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_SHELL, TopAbs_FACE, TopAbs_SOLID, TopAbs_WIRE
    from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakeThickSolid
    from OCC.Core.TopTools import TopTools_ListOfShape

    if not feature_faces:
        return None

    orig_faces = _count_faces(shape)

    # Strategy A: ThickSolid (offset the shape by removing feature faces)
    # This tells OCC to "close" the shape by removing the specified faces and
    # building offset surfaces to fill the gap — works for pockets and blind cuts.
    try:
        faces_to_close = TopTools_ListOfShape()
        for face in feature_faces:
            faces_to_close.Append(face)

        for offset_dist in [0.0, 1e-6, 1e-5]:
            try:
                thick = BRepOffsetAPI_MakeThickSolid()
                thick.MakeThickSolidByJoin(shape, faces_to_close, offset_dist, 1e-3)
                thick.Build()
                if thick.IsDone():
                    result = thick.Shape()
                    unifier = ShapeUpgrade_UnifySameDomain(result, True, True, True)
                    unifier.Build()
                    unified = unifier.Shape()
                    result_faces = _count_faces(unified)
                    if result_faces < orig_faces:
                        analyzer = BRepCheck_Analyzer(unified)
                        if analyzer.IsValid():
                            if verbose:
                                print(f"      ThickSolid fill: {orig_faces} -> "
                                      f"{result_faces} faces")
                            return unified
            except Exception:
                continue
    except Exception:
        pass

    # Strategy B: Prism fill — find boundary wires of feature region,
    # extrude them to create fill bodies
    try:
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCC.Core.TopExp import topexp
        from OCC.Core.TopAbs import TopAbs_EDGE
        from OCC.Core.TopTools import (
            TopTools_IndexedDataMapOfShapeListOfShape,
            TopTools_IndexedMapOfShape,
            TopTools_ListIteratorOfListOfShape,
        )
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core.GeomAbs import GeomAbs_Plane
        from OCC.Core.gp import gp_Vec

        # Find boundary edges: edges shared between feature faces and non-feature faces
        feature_set = TopTools_IndexedMapOfShape()
        for f in feature_faces:
            feature_set.Add(f)

        edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
        topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

        # Find non-feature planar faces adjacent to feature faces
        # These define the "target level" for filling
        adjacent_stock_faces = []
        for i, face in enumerate(feature_faces):
            edge_exp = TopExp_Explorer(face, TopAbs_EDGE)
            while edge_exp.More():
                edge = edge_exp.Current()
                idx = edge_face_map.FindIndex(edge)
                if idx > 0:
                    neighbors = edge_face_map.FindFromIndex(idx)
                    it = TopTools_ListIteratorOfListOfShape(neighbors)
                    while it.More():
                        neighbor = topods.Face(it.Value())
                        if not feature_set.Contains(neighbor):
                            adjacent_stock_faces.append(neighbor)
                        it.Next()
                edge_exp.Next()

        # Find the dominant stock face normal (the surface we want to fill up to)
        if adjacent_stock_faces:
            normals_and_faces = []
            for sf in adjacent_stock_faces:
                sa = BRepAdaptor_Surface(sf)
                if sa.GetType() == GeomAbs_Plane:
                    pln = sa.Plane()
                    norm = pln.Axis().Direction()
                    normals_and_faces.append(((norm.X(), norm.Y(), norm.Z()), sf))

            if normals_and_faces:
                # Use the most common normal direction as fill direction
                from collections import Counter as _NCounter
                rounded_normals = []
                for n, _ in normals_and_faces:
                    rn = (round(n[0], 1), round(n[1], 1), round(n[2], 1))
                    rounded_normals.append(rn)
                most_common_dir = _NCounter(rounded_normals).most_common(1)[0][0]

                # Try to extrude each feature face along the fill direction
                fill_dir = gp_Vec(most_common_dir[0], most_common_dir[1],
                                  most_common_dir[2])

                # Compute needed extrusion height from bounding box
                from OCC.Core.Bnd import Bnd_Box
                from OCC.Core.BRepBndLib import brepbndlib
                bbox = Bnd_Box()
                for f in feature_faces:
                    brepbndlib.Add(f, bbox)
                xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
                diag = _math.sqrt((xmax-xmin)**2 + (ymax-ymin)**2 + (zmax-zmin)**2)
                extrude_height = diag * 0.5

                # Scale the fill direction
                fill_vec = gp_Vec(fill_dir.X() * extrude_height,
                                  fill_dir.Y() * extrude_height,
                                  fill_dir.Z() * extrude_height)

                # Try extruding the feature faces as a group and fusing
                for face in feature_faces:
                    try:
                        prism = BRepPrimAPI_MakePrism(face, fill_vec)
                        prism.Build()
                        if prism.IsDone():
                            prism_solid = prism.Shape()
                            for fuzzy in [0.0, 1e-4, 1e-3, 5e-3]:
                                if fuzzy > max_fuzzy and fuzzy > 0:
                                    break
                                try:
                                    fuse = BRepAlgoAPI_Fuse(shape, prism_solid)
                                    if fuzzy > 0:
                                        fuse.SetFuzzyValue(fuzzy)
                                    fuse.Build()
                                    if fuse.IsDone() and not fuse.HasErrors():
                                        result = fuse.Shape()
                                        unifier = ShapeUpgrade_UnifySameDomain(
                                            result, True, True, True)
                                        unifier.Build()
                                        unified = unifier.Shape()
                                        result_faces = _count_faces(unified)
                                        if result_faces < orig_faces:
                                            analyzer = BRepCheck_Analyzer(unified)
                                            if analyzer.IsValid():
                                                if verbose:
                                                    print(f"      Prism fill: "
                                                          f"{orig_faces} -> "
                                                          f"{result_faces} faces")
                                                return unified
                                except Exception:
                                    continue
                    except Exception:
                        continue
    except Exception:
        pass

    # Strategy C: Sew + MakeSolid for closed feature groups
    for sew_tol in [1e-6, 1e-4, 1e-3, 5e-3]:
        try:
            sewer = BRepBuilderAPI_Sewing(sew_tol)
            for face in feature_faces:
                sewer.Add(face)
            sewer.Perform()
            sewn = sewer.SewedShape()
            if sewn is None:
                continue

            solid = None
            solid_exp = TopExp_Explorer(sewn, TopAbs_SOLID)
            if solid_exp.More():
                solid = topods.Solid(solid_exp.Current())
            else:
                shell_exp = TopExp_Explorer(sewn, TopAbs_SHELL)
                if shell_exp.More():
                    shell = topods.Shell(shell_exp.Current())
                    try:
                        maker = BRepBuilderAPI_MakeSolid(shell)
                        if maker.IsDone():
                            solid = maker.Solid()
                    except Exception:
                        pass

            if solid is None:
                continue

            for fuzzy in [0.0, 1e-4, 1e-3, 5e-3]:
                if fuzzy > max_fuzzy and fuzzy > 0:
                    break
                try:
                    fuse = BRepAlgoAPI_Fuse(shape, solid)
                    if fuzzy > 0:
                        fuse.SetFuzzyValue(fuzzy)
                    fuse.Build()
                    if fuse.IsDone() and not fuse.HasErrors():
                        result = fuse.Shape()
                        unifier = ShapeUpgrade_UnifySameDomain(result, True, True, True)
                        unifier.Build()
                        unified = unifier.Shape()
                        result_faces = _count_faces(unified)
                        if result_faces < orig_faces:
                            analyzer = BRepCheck_Analyzer(unified)
                            if analyzer.IsValid():
                                if verbose:
                                    print(f"      Sew+Fuse fill: {orig_faces} -> "
                                          f"{result_faces} faces (tol={sew_tol})")
                                return unified
                except Exception:
                    continue
        except Exception:
            continue

    # Strategy D: Per-group iterative fill
    groups = _group_connected_faces(shape, feature_faces)
    if len(groups) > 1:
        current = shape
        any_progress = False
        for group in groups:
            # Try ThickSolid per group first
            try:
                faces_to_close = TopTools_ListOfShape()
                for face in group:
                    faces_to_close.Append(face)
                thick = BRepOffsetAPI_MakeThickSolid()
                thick.MakeThickSolidByJoin(current, faces_to_close, 0.0, 1e-3)
                thick.Build()
                if thick.IsDone():
                    result = thick.Shape()
                    unifier = ShapeUpgrade_UnifySameDomain(result, True, True, True)
                    unifier.Build()
                    unified = unifier.Shape()
                    if _count_faces(unified) < _count_faces(current):
                        analyzer = BRepCheck_Analyzer(unified)
                        if analyzer.IsValid():
                            current = unified
                            any_progress = True
                            continue
            except Exception:
                pass

            # Fallback: sew+fuse per group
            for sew_tol in [1e-4, 1e-3, 5e-3]:
                try:
                    sewer = BRepBuilderAPI_Sewing(sew_tol)
                    for face in group:
                        sewer.Add(face)
                    sewer.Perform()
                    sewn = sewer.SewedShape()
                    if sewn is None:
                        continue
                    solid = None
                    solid_exp = TopExp_Explorer(sewn, TopAbs_SOLID)
                    if solid_exp.More():
                        solid = topods.Solid(solid_exp.Current())
                    else:
                        shell_exp = TopExp_Explorer(sewn, TopAbs_SHELL)
                        if shell_exp.More():
                            shell = topods.Shell(shell_exp.Current())
                            try:
                                maker = BRepBuilderAPI_MakeSolid(shell)
                                if maker.IsDone():
                                    solid = maker.Solid()
                            except Exception:
                                pass
                    if solid is None:
                        continue
                    fuse = BRepAlgoAPI_Fuse(current, solid)
                    fuse.SetFuzzyValue(1e-3)
                    fuse.Build()
                    if fuse.IsDone() and not fuse.HasErrors():
                        result = fuse.Shape()
                        unifier = ShapeUpgrade_UnifySameDomain(result, True, True, True)
                        unifier.Build()
                        unified = unifier.Shape()
                        if _count_faces(unified) < _count_faces(current):
                            current = unified
                            any_progress = True
                            break
                except Exception:
                    continue

        if any_progress and _count_faces(current) < _count_faces(shape):
            return current

    return None


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

    from OCC.Core.BRepCheck import BRepCheck_Analyzer as _Analyzer
    shape_is_invalid = not _Analyzer(shape).IsValid()

    # --- Phase 1: Try all at once (adaptive tolerance) ---
    result, history, tol = _try_defeaturing_adaptive(
        shape, all_features, verbose=verbose, max_fuzzy=max_fuzzy,
    )
    if result is not None:
        if verbose:
            print(f"    Phase 1: all {total_features} faces removed at once")
        result = _remove_residual_faces(result, verbose=verbose,
                                        max_fuzzy=max_fuzzy)
        result = _heal_shape(result, verbose=verbose)
        return result, total_features, 0

    if shape_is_invalid:
        if verbose:
            print(f"    Phase 1 failed on invalid shape — "
                  f"defeaturing not possible, skipping remaining phases")
        return shape, 0, total_features

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
        failed_groups = []
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
            else:
                failed_groups.append(group)
                if verbose:
                    print(f"      x group of {len(group)} {cls_name} faces failed")

        # Phase 3b: Boolean fill for failed groups (complex cuts/pockets)
        if failed_groups:
            current_base = best_shape if best_count > 0 else shape
            for group in failed_groups:
                fill_result = _try_boolean_fill(
                    current_base, group, verbose=verbose, max_fuzzy=max_fuzzy)
                if fill_result is not None:
                    fill_fc = _count_faces(fill_result)
                    base_fc = _count_faces(current_base)
                    if fill_fc < base_fc:
                        current_base = fill_result
                        best_shape = current_base
                        best_count += len(group)
                        accumulated.extend(group)
                        added_from_type += len(group)
                        if verbose:
                            print(f"      + group of {len(group)} {cls_name} "
                                  f"faces (boolean fill)")

        if verbose:
            print(f"      {cls_name}: {added_from_type}/{len(type_faces)} removed")

    # --- Phase 3c: Count side-effect removals ---
    # Faces that failed in Phase 2/3 may have been removed as a side-effect
    # of adjacent feature removal. Count these as successes.
    if best_count > 0 and best_count < total_features:
        remaining = _collect_remaining_faces(
            all_features, accumulated, best_shape, last_history,
            verbose=False,
        )
        side_effect_removed = total_features - best_count - len(remaining)
        if side_effect_removed > 0:
            best_count += side_effect_removed
            if verbose:
                print(f"    Phase 3c: {side_effect_removed} faces removed as "
                      f"side-effect of adjacent features")

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

    # --- Phase 4b: Alternate order restart (cuts first) ---
    # When the standard order (fillet→chamfer→hole→cut) leaves failures,
    # try starting fresh with cuts first. Chamfers/fillets at cut edges often
    # get removed as side-effects of the cut removal.
    if best_count > 0 and best_count < total_features:
        remaining = _collect_remaining_faces(
            all_features, accumulated, best_shape, last_history, verbose=False)
        if remaining:
            # Try all remaining on the original shape (different context)
            result, history, tol = _try_defeaturing_adaptive(
                shape, all_features, verbose=False, max_fuzzy=max_fuzzy)
            if result is not None:
                best_shape = result
                best_count = total_features
                last_history = history
                if verbose:
                    print(f"    Phase 4b: all {total_features} removed on "
                          f"fresh attempt (alternate context)")
            else:
                # Try cuts-first order on fresh shape
                alt_order = [4, 1, 2, 3]  # cut → hole → chamfer → fillet
                alt_accumulated = []
                alt_shape = shape
                alt_history = None
                for cls_id in alt_order:
                    type_faces = feature_faces_by_type.get(cls_id, [])
                    if not type_faces:
                        continue
                    candidate = alt_accumulated + type_faces
                    r, h, t = _try_defeaturing_adaptive(
                        shape, candidate, verbose=False, max_fuzzy=max_fuzzy)
                    if r is not None:
                        alt_accumulated = candidate
                        alt_shape = r
                        alt_history = h

                # Check if alternate order did better
                alt_remaining = _collect_remaining_faces(
                    all_features, alt_accumulated, alt_shape, alt_history,
                    verbose=False) if alt_accumulated else all_features
                alt_removed = total_features - len(alt_remaining)
                if alt_removed > best_count:
                    best_shape = alt_shape
                    best_count = alt_removed
                    last_history = alt_history
                    accumulated = alt_accumulated
                    if verbose:
                        print(f"    Phase 4b: alternate order removed "
                              f"{alt_removed}/{total_features} (was {best_count})")

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

        # 5d: Boolean fill for stubborn features (complex cuts/pockets)
        # When BRepAlgoAPI_Defeaturing can't remove faces (complex topology),
        # try to construct a fill solid from the feature faces and fuse it
        # into the shape, effectively filling the cut back in.
        if best_count < total_features:
            remaining_for_fill = []
            if best_count > 0:
                face_map = TopTools_IndexedMapOfShape()
                exp = TopExp_Explorer(best_shape, TopAbs_FACE)
                while exp.More():
                    face_map.Add(topods.Face(exp.Current()))
                    exp.Next()
                remaining_for_fill = [f for f in all_features
                                      if face_map.Contains(f) and f not in accumulated]
            else:
                remaining_for_fill = list(all_features)

            if remaining_for_fill:
                fill_result = _try_boolean_fill(
                    best_shape if best_count > 0 else shape,
                    remaining_for_fill, verbose=verbose, max_fuzzy=max_fuzzy)
                if fill_result is not None:
                    fill_faces = _count_faces(fill_result)
                    base_faces = _count_faces(best_shape if best_count > 0 else shape)
                    if fill_faces < base_faces:
                        best_shape = fill_result
                        best_count += len(remaining_for_fill)
                        if verbose:
                            print(f"      5d: boolean fill removed "
                                  f"{len(remaining_for_fill)} faces")

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

    # --- Phase 6: Remove residual small faces ---
    # After removing declared features (e.g. cut bottoms), residual side-wall
    # faces from the same feature may remain if the model didn't label them.
    # These are small faces that share edges with the removal zone and would
    # prevent a clean result. Remove any face below an area threshold.
    if best_count > 0:
        best_shape = _remove_residual_faces(best_shape, verbose=verbose,
                                            max_fuzzy=max_fuzzy)

    # --- Phase 7: Remove enclosed remnants ---
    # Small faces originally labeled "random" that are now isolated after
    # adjacent features were removed (pocket floors, hole walls the model missed).
    if best_count > 0:
        best_shape = _remove_enclosed_remnants(best_shape, verbose=verbose,
                                               max_fuzzy=max_fuzzy)

    # --- Cleanup: multi-stage healing ---
    best_shape = _heal_shape(best_shape, verbose=verbose)

    failed = total_features - best_count
    return best_shape, best_count, failed


def defeature_step(step_path: str, face_preds: List[int],
                   output_path: str, verbose: bool = False,
                   max_fuzzy: float = 1e-3,
                   symmetry: bool = True) -> dict:
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

    # --- Symmetry completion: fill near-symmetric prediction gaps ---
    original_preds = list(face_preds)
    if symmetry:
        face_preds = _complete_symmetry(shape, faces, face_preds, verbose=verbose)
    symmetry_added = sum(1 for a, b in zip(original_preds, face_preds) if a != b)

    # --- Protect structural faces ---
    # Large faces that dominate the shape's surface area are likely structural
    # (body cylinders, main cones/prisms) even if predicted as features.
    # Reclassify them as "random" (keep) to avoid destroying the 3D form.
    face_preds = _protect_structural_faces(
        shape, faces, face_preds, verbose=verbose)

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
            "symmetry_added": symmetry_added,
        }

    # Cannot remove ALL faces — a solid needs base geometry to survive.
    # When predictions label everything as a feature (keep_count == 0),
    # the model has likely failed. Cap removal at 90% of faces — OCC
    # can't produce a valid solid by removing all its boundary faces.
    if keep_count == 0:
        if verbose:
            print(f"  WARNING: all {num_faces} faces predicted as features — "
                  f"no base geometry to keep. Capping removal.")
        # Keep the largest faces (most likely the base geometry the model
        # misclassified) by reclassifying them as "keep".
        all_feature_faces = []
        for cls_id in REMOVAL_ORDER:
            for face in feature_faces_by_type.get(cls_id, []):
                all_feature_faces.append((face, cls_id))
        # Sort by area descending — largest faces are likely base geometry
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.BRepGProp import brepgprop
        def _area(item):
            props = GProp_GProps()
            brepgprop.SurfaceProperties(item[0], props)
            return props.Mass()
        all_feature_faces.sort(key=_area, reverse=True)
        # Keep at least 10% of faces (by count) or at least 6 faces
        min_keep = max(num_faces // 10, 6)
        faces_to_keep = all_feature_faces[:min_keep]
        kept_set = set(id(f) for f, _ in faces_to_keep)
        # Rebuild feature_faces_by_type without the kept faces
        feature_faces_by_type = {}
        for cls_id in FEATURE_CLASSES:
            feature_faces_by_type[cls_id] = []
        for face, cls_id in all_feature_faces:
            if id(face) not in kept_set:
                feature_faces_by_type[cls_id].append(face)
        total_features = sum(len(v) for v in feature_faces_by_type.values())
        keep_count = min_keep
        if verbose:
            print(f"  Adjusted: keeping {min_keep} largest faces, "
                  f"removing {total_features}")

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
        "symmetry_added": symmetry_added,
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
                 colored_path, verbose, max_fuzzy, save_colored,
                 symmetry=True):
    """Process a single STEP file in an isolated worker process.

    All OpenCASCADE objects are confined to this process and destroyed
    on exit, preventing stale C++ state from corrupting the next file.
    """
    try:
        result = defeature_step(step_path, preds, output_path,
                                verbose=verbose, max_fuzzy=max_fuzzy,
                                symmetry=symmetry)
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
                       timeout=300, symmetry=True):
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
              colored_path, verbose, max_fuzzy, save_colored, symmetry),
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
            symmetry=not args.no_symmetry,
        )
        _print_result(result)

        if args.save_colored:
            colored_path = output_dir / f"{model_id}_colored.step"
            save_colored_step(step_path, preds, str(colored_path))

        # Primitive fitting (optional)
        if args.primitives and result["status"] in ("success", "no_features"):
            from brepformer.primitives import fit_primitives_from_step

            prim_path = output_dir / f"{model_id}_primitives.step"
            print(f"\nFitting primitives...")
            prim_report = fit_primitives_from_step(
                str(out_path), str(prim_path),
                verbose=args.verbose, tol=args.primitives_tol,
            )
            result["primitives"] = prim_report
            if prim_report["status"] == "success":
                print(f"  Primitives: {prim_report['num_primitives']} "
                      f"({', '.join(prim_report['primitive_types'])})")
                print(f"  Valid: {prim_report['valid']}")
                print(f"  Output: {prim_path}")
            else:
                print(f"  Primitives failed: "
                      f"{prim_report.get('message', 'unknown')}")

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

            existing_out = output_dir / f"{model_id}_defeatured.step"
            if existing_out.exists() and existing_out.stat().st_size > 0:
                print(f"    Skipping: output already exists")
                continue

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
                timeout=300, symmetry=not args.no_symmetry,
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
