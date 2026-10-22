#!/usr/bin/env python3
"""Geometric confirmation/denial rules for face class predictions.

Extracted and generalized from defeature_v2.py's structural protection and
geometric expansion logic. Works with both 27-class MFTRCAD and 5-class
defeature label spaces.

Each rule returns a verdict: CONFIRM, DENY (with suggested correction), or
UNCERTAIN. Rules are conservative — they only fire on high-confidence
geometric signals.

Usage:
    from brepformer.geometric_rules import apply_geometric_rules
    verdicts = apply_geometric_rules(step_path, face_preds, label_space="27")
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brepformer.data.classes import (
    CLASS_NAMES, NUM_CLASSES,
    DEFEATURE_CLASS_NAMES, DEFEATURE_NUM_CLASSES,
)

# Maps 27-class IDs to geometric expectations
# "surface_types": list of expected OCC GeomAbs types (None = any)
# "area_fraction_max": if face exceeds this fraction of total area, deny.
#                   Thresholds measured over 300 mftrcad models (per-face area / total
#                   area, by true class), not guessed:
#                     chamfer   n=53   med 0.001  p99 0.038  max 0.038  -> 0.10 inert
#                     round     n=33   med 0.001  p99 0.012  max 0.012  -> 0.05 inert
#                     chamfer_2 n=274  med 0.158  p90 0.472  max 0.861  -> 0.10 DENIES 62%
#                   So the cap was removed from chamfer_2 (and from 5-class chamfer,
#                   which absorbs chamfer_2). The two inert caps are kept as cheap guards;
#                   raising or adding a cap requires re-running the measurement in
#                   /tmp/measure_area.py first.
# "must_be_curved": whether face must be curved (cylinder/cone/torus/sphere)
# "must_be_planar": whether face must be planar
#
# A type constraint is only present when it was measured to deny <5% of that class's real
# faces (300 mftrcad models, per-face surface type by true class). It was absent from that
# check originally, and the constraints were assigned from each class's *name*, which is
# wrong in principle: a 27-class label names a **feature**, and a feature owns all of its
# faces. A circular-end pocket is 58% planar (its floor and walls) and 41% cylindrical
# (its ends); a blind hole is 52% planar (its bottom). Naming it "circular" says what the
# feature looks like, not what any one of its faces is. Measured mix:
#
#   class                  planar conical curved   old constraint denied
#   rectangular_passage_2      0%      0%   100%   must_be_planar   100%  -> inverted
#   circular_end_pocket       58%      1%    41%   must_be_curved    58%  -> dropped
#   blind_hole                52%      0%    48%   must_be_curved    52%  -> dropped
#   2sides_blind_step         48%      0%    52%   must_be_planar    52%  -> dropped
#   circular_blind_slot       46%      0%    54%   must_be_curved    46%  -> dropped
#   v_shaped_blind_slot       75%      0%    25%   must_be_planar    25%  -> dropped
#   Oring                      9%      1%    90%   must_be_curved     9%  -> dropped
#   through_hole               3%      0%    97%   must_be_curved     3%  -> kept
#   chamfer                   85%     13%     2%   must_be_planar     2%  -> kept
#   (every other class is >=99% single-typed; their constraints deny 0-1% and are kept)
#
# Re-run /tmp/measure_stype.py before adding a constraint to any class.
# "may_be_conical": exempt cones from must_be_planar (a chamfer on a circular edge
#                   is a cone, not a plane)
CLASS_27_GEOMETRY = {
    0: {"name": "chamfer", "must_be_planar": True, "may_be_conical": True,
        "area_fraction_max": 0.10},
    1: {"name": "through_hole", "must_be_curved": True},
    2: {"name": "triangular_passage", "must_be_planar": True},
    3: {"name": "rectangular_passage", "must_be_planar": True},
    4: {"name": "6sides_passage", "must_be_planar": True},
    5: {"name": "triangular_through_slot", "must_be_planar": True},
    6: {"name": "rectangular_through_slot", "must_be_planar": True},
    7: {"name": "circular_through_slot", "must_be_curved": True},
    8: {"name": "rectangular_through_step", "must_be_planar": True},
    9: {"name": "2sides_through_step", "must_be_planar": True},
    10: {"name": "slanted_through_step", "must_be_planar": True},
    11: {"name": "Oring"},  # 9% of real Oring faces are planar
    12: {"name": "blind_hole"},  # 52% planar: the hole bottom is a plane
    13: {"name": "triangular_pocket", "must_be_planar": True},
    14: {"name": "rectangular_pocket", "must_be_planar": True},
    15: {"name": "6sides_pocket", "must_be_planar": True},
    16: {"name": "circular_end_pocket"},  # 58% planar: floor and straight walls
    17: {"name": "rectangular_blind_slot", "must_be_planar": True},
    18: {"name": "v_shaped_blind_slot"},  # 25% curved
    19: {"name": "circular_blind_slot"},  # 46% planar
    20: {"name": "rectangular_blind_step", "must_be_planar": True},
    21: {"name": "2sides_blind_step"},  # 52% curved
    22: {"name": "triangular_blind_step", "must_be_planar": True},
    23: {"name": "round", "must_be_curved": True, "area_fraction_max": 0.05},
    24: {"name": "stock"},  # no constraints — stock can be anything
    25: {"name": "rectangular_passage_2", "must_be_curved": True},  # 100% curved,
        # not planar -- the original constraint was exactly inverted
    # No area cap: real chamfer_2 faces have a *median* area fraction of 0.158 and reach
    # 0.861, so any cap small enough to be meaningful denies the majority of them. This
    # class is not a small bevel face in mftrcad.
    26: {"name": "chamfer_2", "must_be_planar": True, "may_be_conical": True},
}

# 5-class geometric expectations
CLASS_5_GEOMETRY = {
    0: {"name": "random"},  # no constraints
    # No type constraint: `hole` unions through_hole (97% curved) with blind_hole (52%
    # planar -- the hole bottom), so the union is mixed and the constraint denied 23%.
    1: {"name": "hole"},
    # No area cap: this class absorbs 27-class chamfer_2 (see CLASS_TO_DEFEATURE), whose
    # median area fraction is 0.158, so a cap here inherits the same 62% false-denial rate.
    2: {"name": "chamfer", "must_be_planar": True, "may_be_conical": True},
    # 0.08 is inert on the only ground truth available: 27-class `round` is the sole class
    # mapping here and its measured maximum is 0.012.
    3: {"name": "fillet", "must_be_curved": True, "area_fraction_max": 0.08},
    # No type constraint: `cut` absorbs 21 of the 27 fine classes, including
    # rectangular_passage_2 (100% curved), Oring (90%) and circular_end_pocket (41%).
    # A union that wide has no characteristic surface type; the constraint denied 18%.
    4: {"name": "cut"},
}

VERDICT_CONFIRM = "CONFIRM"
VERDICT_DENY = "DENY"
VERDICT_UNCERTAIN = "UNCERTAIN"


def _load_step_faces(step_path: str):
    """Load STEP and return (shape, faces_list)."""
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP: {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()

    faces = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        faces.append(topods.Face(exp.Current()))
        exp.Next()

    return shape, faces


def _get_face_geometry(faces: list) -> List[Dict]:
    """Extract geometric properties for each face."""
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import (
        GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
        GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface,
    )
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop

    CURVED_TYPES = {GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus}

    results = []
    for face in faces:
        sa = BRepAdaptor_Surface(face)
        stype = sa.GetType()
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        area = props.Mass()

        info = {
            "stype": stype,
            "area": area,
            "is_planar": stype == GeomAbs_Plane,
            "is_curved": stype in CURVED_TYPES,
            "is_conical": stype == GeomAbs_Cone,
            "is_bspline": stype == GeomAbs_BSplineSurface,
            "axis": None,
            "axis_loc": None,
            "radius": None,
        }

        # An axis is a line, not just a direction: co-axiality needs a point on it too
        # (see _axes_coaxial). Storing only the direction is what made every pair of
        # parallel cylinders on a part read as co-axial.
        if stype == GeomAbs_Cylinder:
            cyl = sa.Cylinder()
            ax = cyl.Axis().Direction()
            loc = cyl.Axis().Location()
            info["axis"] = (ax.X(), ax.Y(), ax.Z())
            info["axis_loc"] = (loc.X(), loc.Y(), loc.Z())
            info["radius"] = cyl.Radius()
        elif stype == GeomAbs_Cone:
            cone = sa.Cone()
            ax = cone.Axis().Direction()
            loc = cone.Axis().Location()
            info["axis"] = (ax.X(), ax.Y(), ax.Z())
            info["axis_loc"] = (loc.X(), loc.Y(), loc.Z())
            info["radius"] = cone.RefRadius()
        elif stype == GeomAbs_Sphere:
            info["radius"] = sa.Sphere().Radius()
        elif stype == GeomAbs_Torus:
            tor = sa.Torus()
            ax = tor.Axis().Direction()
            loc = tor.Axis().Location()
            info["axis"] = (ax.X(), ax.Y(), ax.Z())
            info["axis_loc"] = (loc.X(), loc.Y(), loc.Z())
            info["radius"] = tor.MinorRadius()

        results.append(info)
    return results


def _axes_coaxial(gi: dict, gj: dict, rel_tol: float = 0.05) -> bool:
    """True when two faces share the *same axis line*, not merely a parallel direction.

    Two axes are collinear when their directions are parallel AND the offset between a
    point on each is itself parallel to that direction -- i.e. the perpendicular
    component of the offset vanishes. The original test checked only
    ``abs(dot(dir_i, dir_j)) > 0.95``, which every pair of parallel cylinders on a part
    satisfies regardless of where they sit. On a plate of through-holes that made each
    hole confirm every other: measured 12.6x under-count of distinct axis clusters (249
    found vs 3,130 real across 70 parts), worst case 334x, with one fixture plate
    collapsing 378 cylinder faces into a single "hole".

    The perpendicular offset is compared against a tolerance scaled by the larger of the
    two radii, so the test is size-invariant -- an absolute tolerance would be far too
    tight on a large casting and far too loose on a small pin.
    """
    ai, aj = gi.get("axis"), gj.get("axis")
    if ai is None or aj is None:
        return False
    if abs(sum(a * b for a, b in zip(ai, aj))) <= 0.95:
        return False  # directions not parallel

    pi, pj = gi.get("axis_loc"), gj.get("axis_loc")
    if pi is None or pj is None:
        return False  # cannot establish collinearity; refuse rather than assume

    offset = [b - a for a, b in zip(pi, pj)]
    along = sum(o * a for o, a in zip(offset, ai))
    perp_sq = sum((o - along * a) ** 2 for o, a in zip(offset, ai))

    radii = [r for r in (gi.get("radius"), gj.get("radius")) if r]
    scale = max(radii) if radii else 1.0
    tol = max(rel_tol * scale, 1e-7)
    return perp_sq <= tol * tol


def _build_adjacency(shape, faces: list) -> List[set]:
    """Build face adjacency from shared edges."""
    from OCC.Core.TopExp import TopExp_Explorer, topexp
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCC.Core.TopoDS import topods
    from OCC.Core.TopTools import (
        TopTools_IndexedDataMapOfShapeListOfShape,
        TopTools_ListIteratorOfListOfShape,
    )

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

    return adjacency


def apply_geometric_rules(
    step_path: str,
    face_preds: List[int],
    label_space: str = "27",
) -> List[Dict]:
    """Apply geometric rules to verify/deny each face prediction.

    Args:
        step_path: Path to the STEP file.
        face_preds: Per-face predicted class IDs.
        label_space: "27" for MFTRCAD classes, "5" for defeature classes.

    Returns:
        List of dicts, one per face:
          {
            "face_idx": int,
            "predicted": int,
            "verdict": "CONFIRM" | "DENY" | "UNCERTAIN",
            "reason": str,
            "suggested": int | None,  # only set if DENY
          }
    """
    geometry_map = CLASS_27_GEOMETRY if label_space == "27" else CLASS_5_GEOMETRY
    stock_class = 24 if label_space == "27" else 0

    shape, faces = _load_step_faces(step_path)
    if len(faces) != len(face_preds):
        raise ValueError(
            f"Face count mismatch: STEP has {len(faces)} faces, "
            f"predictions has {len(face_preds)}"
        )

    geom = _get_face_geometry(faces)
    adjacency = _build_adjacency(shape, faces)
    total_area = sum(g["area"] for g in geom)

    verdicts = []
    for i, pred in enumerate(face_preds):
        g = geom[i]
        area_frac = g["area"] / total_area if total_area > 0 else 0

        result = {
            "face_idx": i,
            "predicted": pred,
            "verdict": VERDICT_UNCERTAIN,
            "reason": "",
            "suggested": None,
            # Which rule produced the verdict. None means no rule fired, so the face
            # stayed UNCERTAIN by default. Needed to attribute a corruption rate to a
            # specific rule instead of matching on the human-readable reason string.
            "rule": None,
        }

        if pred not in geometry_map:
            result["rule"] = "unknown_class"
            result["verdict"] = VERDICT_UNCERTAIN
            result["reason"] = f"Unknown class {pred}"
            verdicts.append(result)
            continue

        rules = geometry_map[pred]

        # Rule 1: Surface type must match expectations
        if rules.get("must_be_curved") and not g["is_curved"] and not g["is_bspline"]:
            if g["is_planar"]:
                result["rule"] = "must_be_curved"
                result["verdict"] = VERDICT_DENY
                result["reason"] = (
                    f"Class '{rules['name']}' requires curved surface, "
                    f"but face is planar"
                )
                result["suggested"] = stock_class
                verdicts.append(result)
                continue

        if rules.get("must_be_planar") and not g["is_planar"]:
            # A chamfer is planar only when it breaks a straight edge. Chamfer a hole,
            # a boss or a shaft -- a circular edge -- and the resulting face is a cone.
            # Denying those unconditionally is what falsely denied 100% of chamfer_2
            # (53/53) against mftrcad ground truth, so conical faces are exempt for any
            # class flagged ``may_be_conical``.
            if rules.get("may_be_conical") and g["is_conical"]:
                pass
            elif g["is_curved"]:
                result["rule"] = "must_be_planar"
                result["verdict"] = VERDICT_DENY
                result["reason"] = (
                    f"Class '{rules['name']}' expects planar surface, "
                    f"but face is curved"
                )
                # Curved face mislabeled as planar feature → likely hole or stock
                if label_space == "5":
                    result["suggested"] = 1  # hole
                else:
                    result["suggested"] = 1  # through_hole
                verdicts.append(result)
                continue

        # Rule 2: Area fraction limits
        max_area = rules.get("area_fraction_max")
        if max_area and area_frac > max_area:
            result["rule"] = "area_fraction_max"
            result["verdict"] = VERDICT_DENY
            result["reason"] = (
                f"Face area is {area_frac:.1%} of total, exceeds "
                f"{max_area:.0%} limit for class '{rules['name']}'"
            )
            result["suggested"] = stock_class
            verdicts.append(result)
            continue

        # Rule 3: Large planar face (>15% total area) predicted as feature
        if (g["is_planar"] and area_frac > 0.15
                and pred != stock_class):
            result["rule"] = "large_planar_stock"
            result["verdict"] = VERDICT_DENY
            result["reason"] = (
                f"Large planar face ({area_frac:.1%} of total) is likely "
                f"structural stock, not '{rules['name']}'"
            )
            result["suggested"] = stock_class
            verdicts.append(result)
            continue

        # Rule 4: Hole/cylinder confirmation via co-axial neighbors
        if g["is_curved"] and g["axis"] is not None:
            hole_classes_27 = {1, 7, 11, 12, 16, 19}
            hole_classes_5 = {1}
            expected_holes = hole_classes_27 if label_space == "27" else hole_classes_5

            if pred in expected_holes:
                coaxial_neighbors = 0
                for j in adjacency[i]:
                    if _axes_coaxial(g, geom[j]):
                        coaxial_neighbors += 1
                if coaxial_neighbors >= 1:
                    result["rule"] = "coaxial_confirm"
                    result["verdict"] = VERDICT_CONFIRM
                    result["reason"] = (
                        f"Curved face with co-axial neighbor(s) confirms "
                        f"'{rules['name']}'"
                    )
                    verdicts.append(result)
                    continue

        # Rule 5: Stock confirmation — large face with few feature neighbors
        if pred == stock_class and area_frac > 0.05:
            feature_neighbors = sum(
                1 for j in adjacency[i] if face_preds[j] != stock_class
            )
            total_neighbors = len(adjacency[i])
            if total_neighbors > 0 and feature_neighbors / total_neighbors < 0.5:
                result["rule"] = "stock_confirm"
                result["verdict"] = VERDICT_CONFIRM
                result["reason"] = (
                    f"Large face ({area_frac:.1%}) with majority stock "
                    f"neighbors confirms stock"
                )
                verdicts.append(result)
                continue

        # Rule 6: Isolated feature face surrounded entirely by stock
        if pred != stock_class:
            neighbors = adjacency[i]
            if neighbors and all(face_preds[j] == stock_class for j in neighbors):
                if g["is_planar"] and area_frac < 0.03:
                    result["rule"] = "isolated_small_planar"
                    result["verdict"] = VERDICT_DENY
                    result["reason"] = (
                        f"Small planar face entirely surrounded by stock — "
                        f"likely mislabeled"
                    )
                    result["suggested"] = stock_class
                    verdicts.append(result)
                    continue

        verdicts.append(result)

    return verdicts


def summarize_verdicts(verdicts: List[Dict]) -> Dict:
    """Summarize geometric rule verdicts."""
    confirmed = sum(1 for v in verdicts if v["verdict"] == VERDICT_CONFIRM)
    denied = sum(1 for v in verdicts if v["verdict"] == VERDICT_DENY)
    uncertain = sum(1 for v in verdicts if v["verdict"] == VERDICT_UNCERTAIN)
    return {
        "total_faces": len(verdicts),
        "confirmed": confirmed,
        "denied": denied,
        "uncertain": uncertain,
        "denial_details": [
            {"face": v["face_idx"], "predicted": v["predicted"],
             "reason": v["reason"], "suggested": v["suggested"]}
            for v in verdicts if v["verdict"] == VERDICT_DENY
        ],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply geometric rules to verify face predictions"
    )
    parser.add_argument("--step", required=True, help="STEP file path")
    parser.add_argument("--seg", required=True,
                        help=".seg file or .json with face_preds")
    parser.add_argument("--label_space", default="27", choices=["27", "5"],
                        help="Label space: 27-class or 5-class")
    args = parser.parse_args()

    # Load predictions
    seg_path = Path(args.seg)
    if seg_path.suffix == ".json":
        import json
        with open(seg_path) as f:
            data = json.load(f)
        preds = data["face_preds"]
    else:
        with open(seg_path) as f:
            preds = [int(line.strip()) for line in f if line.strip()]

    verdicts = apply_geometric_rules(args.step, preds, args.label_space)
    summary = summarize_verdicts(verdicts)

    print(f"Total faces: {summary['total_faces']}")
    print(f"  Confirmed: {summary['confirmed']}")
    print(f"  Denied:    {summary['denied']}")
    print(f"  Uncertain: {summary['uncertain']}")

    if summary["denial_details"]:
        print("\nDenials:")
        for d in summary["denial_details"]:
            class_names = CLASS_NAMES if args.label_space == "27" else DEFEATURE_CLASS_NAMES
            pred_name = class_names[d["predicted"]] if d["predicted"] < len(class_names) else "?"
            sugg_name = class_names[d["suggested"]] if d["suggested"] is not None and d["suggested"] < len(class_names) else "?"
            print(f"  Face {d['face']:3d}: {pred_name} → {sugg_name} — {d['reason']}")
