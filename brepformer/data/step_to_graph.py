#!/usr/bin/env python3
"""STEP to BrepFormer graph converter using pythonOCC.

Converts STEP CAD files to the preprocessed sample format expected by
BrepFormer's PreprocessedDataset, enabling end-to-end inference on raw
STEP files.

Adapted from brepclassifier/data/step_to_graph.py.

Requires: pythonocc-core (conda install -c conda-forge pythonocc-core=7.9.0)
"""

import json
import logging
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopAbs import (
    TopAbs_FACE, TopAbs_EDGE, TopAbs_FORWARD, TopAbs_REVERSED,
)
from OCC.Core.TopoDS import topods
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
    GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface,
    GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse,
    GeomAbs_BSplineCurve,
)
from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_ListIteratorOfListOfShape
from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_SHELL, TopAbs_COMPOUND
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
from OCC.Core.BRepTools import breptools


# Surface type mapping (matching brepformer convention)
SURFACE_TYPE_MAP = {
    GeomAbs_Plane: 0,
    GeomAbs_Cylinder: 1,
    GeomAbs_Cone: 2,
    GeomAbs_Sphere: 3,
    GeomAbs_Torus: 4,
}

# Curve type mapping
CURVE_TYPE_MAP = {
    GeomAbs_Line: 0,
    GeomAbs_Circle: 1,
    GeomAbs_Ellipse: 2,
    GeomAbs_BSplineCurve: 3,
}


def read_step(step_path: str):
    """Read a STEP file and return the shape."""
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != 1:
        return None
    reader.TransferRoots()
    return reader.OneShape()


def get_faces(shape) -> list:
    """Extract all faces from a shape using TopExp_Explorer."""
    faces = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods.Face(explorer.Current())
        faces.append(face)
        explorer.Next()
    return faces


def build_face_adjacency(shape, faces: list) -> Tuple[List[List[int]], Dict]:
    """Build face adjacency graph via shared edges."""
    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

    # Use hash with collision resolution via IsSame
    face_hash_map = defaultdict(list)
    for i, face in enumerate(faces):
        face_hash_map[face.__hash__()].append(i)

    def _find_face_index(f):
        fh = f.__hash__()
        candidates = face_hash_map.get(fh, [])
        if len(candidates) == 1:
            return candidates[0]
        for idx in candidates:
            if faces[idx].IsSame(f):
                return idx
        return None

    edge_pairs = []
    edge_info = {}
    seen_pairs = set()

    for edge_idx in range(1, edge_face_map.Size() + 1):
        edge = edge_face_map.FindKey(edge_idx)
        face_list = edge_face_map.FindFromIndex(edge_idx)

        adj_faces = []
        it = TopTools_ListIteratorOfListOfShape(face_list)
        while it.More():
            f = topods.Face(it.Value())
            fidx = _find_face_index(f)
            if fidx is not None:
                adj_faces.append(fidx)
            it.Next()

        for i in range(len(adj_faces)):
            for j in range(i + 1, len(adj_faces)):
                fi, fj = adj_faces[i], adj_faces[j]
                pair = (min(fi, fj), max(fi, fj))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    edge_pairs.append([fi, fj])
                    edge_pairs.append([fj, fi])
                    edge_info[(fi, fj)] = [edge]
                    edge_info[(fj, fi)] = [edge]
                else:
                    if (fi, fj) in edge_info:
                        edge_info[(fi, fj)].append(edge)
                    if (fj, fi) in edge_info:
                        edge_info[(fj, fi)].append(edge)

    return edge_pairs, edge_info


def compute_face_uv_grid(face, nu: int = 10, nv: int = 10) -> np.ndarray:
    """Sample a 7-channel UV grid from a face: x, y, z, nx, ny, nz, mask."""
    grid = np.zeros((7, nu, nv), dtype=np.float32)

    surf_adaptor = BRepAdaptor_Surface(face)
    u_min = max(surf_adaptor.FirstUParameter(), -1e6)
    u_max = min(surf_adaptor.LastUParameter(), 1e6)
    v_min = max(surf_adaptor.FirstVParameter(), -1e6)
    v_max = min(surf_adaptor.LastVParameter(), 1e6)

    surface = BRep_Tool.Surface(face)
    if surface is None:
        return grid

    is_reversed = face.Orientation() == TopAbs_REVERSED

    for i in range(nu):
        for j in range(nv):
            u = u_min + (u_max - u_min) * i / max(nu - 1, 1)
            v = v_min + (v_max - v_min) * j / max(nv - 1, 1)
            try:
                pnt = gp_Pnt()
                surf_adaptor.D0(u, v, pnt)
                grid[0, i, j] = pnt.X()
                grid[1, i, j] = pnt.Y()
                grid[2, i, j] = pnt.Z()

                d1u = gp_Vec()
                d1v = gp_Vec()
                pnt2 = gp_Pnt()
                surf_adaptor.D1(u, v, pnt2, d1u, d1v)
                normal = d1u.Crossed(d1v)
                mag = normal.Magnitude()
                if mag > 1e-10:
                    normal.Divide(mag)
                    if is_reversed:
                        normal.Reverse()
                    grid[3, i, j] = normal.X()
                    grid[4, i, j] = normal.Y()
                    grid[5, i, j] = normal.Z()

                grid[6, i, j] = 1.0
            except Exception:
                grid[6, i, j] = 0.0

    return grid


def compute_face_attributes(face) -> np.ndarray:
    """Compute 14-dimensional face attributes."""
    attrs = np.zeros(14, dtype=np.float32)
    surf_adaptor = BRepAdaptor_Surface(face)

    surf_type = surf_adaptor.GetType()
    attrs[0] = SURFACE_TYPE_MAP.get(surf_type, 5)

    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    attrs[1] = props.Mass()

    centroid = props.CentreOfMass()
    attrs[2] = centroid.X()
    attrs[3] = centroid.Y()
    attrs[4] = centroid.Z()

    attrs[5] = 1.0 if surf_type == GeomAbs_BSplineSurface else 0.0

    try:
        from OCC.Core.TopExp import TopExp_Explorer as TE
        from OCC.Core.TopAbs import TopAbs_WIRE
        wire_exp = TE(face, TopAbs_WIRE)
        n_wires = 0
        while wire_exp.More():
            n_wires += 1
            wire_exp.Next()
        attrs[6] = float(n_wires)
    except Exception:
        attrs[6] = 1.0

    u_mid = (surf_adaptor.FirstUParameter() + surf_adaptor.LastUParameter()) / 2
    v_mid = (surf_adaptor.FirstVParameter() + surf_adaptor.LastVParameter()) / 2
    try:
        pnt = gp_Pnt()
        d1u = gp_Vec()
        d1v = gp_Vec()
        surf_adaptor.D1(u_mid, v_mid, pnt, d1u, d1v)
        normal = d1u.Crossed(d1v)
        mag = normal.Magnitude()
        if mag > 1e-10:
            normal.Divide(mag)
            if face.Orientation() == TopAbs_REVERSED:
                normal.Reverse()
            attrs[7] = normal.X()
            attrs[8] = normal.Y()
            attrs[9] = normal.Z()
    except Exception:
        pass

    try:
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRepBndLib import brepbndlib
        bbox = Bnd_Box()
        brepbndlib.Add(face, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        attrs[10] = xmax - xmin
        attrs[11] = ymax - ymin
        attrs[12] = zmax - zmin
    except Exception:
        pass

    attrs[13] = 1.0 if face.Orientation() == TopAbs_REVERSED else 0.0
    return attrs


def compute_edge_curve_grid(edge, n_pts: int = 10) -> np.ndarray:
    """Sample a 12-channel curve grid from an edge."""
    grid = np.zeros((12, n_pts), dtype=np.float32)
    try:
        curve_adaptor = BRepAdaptor_Curve(edge)
        t_min = max(curve_adaptor.FirstParameter(), -1e6)
        t_max = min(curve_adaptor.LastParameter(), 1e6)

        for i in range(n_pts):
            t = t_min + (t_max - t_min) * i / max(n_pts - 1, 1)
            try:
                pnt = gp_Pnt()
                tangent = gp_Vec()
                curve_adaptor.D1(t, pnt, tangent)
                grid[0, i] = pnt.X()
                grid[1, i] = pnt.Y()
                grid[2, i] = pnt.Z()
                mag = tangent.Magnitude()
                if mag > 1e-10:
                    tangent.Divide(mag)
                grid[3, i] = tangent.X()
                grid[4, i] = tangent.Y()
                grid[5, i] = tangent.Z()
            except Exception:
                pass
    except Exception:
        pass
    return grid


def _face_normal_at_point(face, x: float, y: float, z: float) -> Optional[np.ndarray]:
    """Get face normal at the closest point to (x, y, z)."""
    try:
        surface = BRep_Tool.Surface(face)
        if surface is None:
            return None
        sa_surface = ShapeAnalysis_Surface(surface)
        pnt = gp_Pnt(float(x), float(y), float(z))
        uv = sa_surface.ValueOfUV(pnt, 1e-3)
        surf_adaptor = BRepAdaptor_Surface(face)
        p = gp_Pnt()
        d1u = gp_Vec()
        d1v = gp_Vec()
        surf_adaptor.D1(uv.X(), uv.Y(), p, d1u, d1v)
        normal = d1u.Crossed(d1v)
        mag = normal.Magnitude()
        if mag > 1e-10:
            normal.Divide(mag)
            if face.Orientation() == TopAbs_REVERSED:
                normal.Reverse()
            return np.array([normal.X(), normal.Y(), normal.Z()])
    except Exception:
        pass
    return None


def compute_edge_attributes(edge, face1=None, face2=None) -> np.ndarray:
    """Compute 15-dimensional edge attributes."""
    attrs = np.zeros(15, dtype=np.float32)
    try:
        curve_adaptor = BRepAdaptor_Curve(edge)
        curve_type = curve_adaptor.GetType()
        attrs[0] = CURVE_TYPE_MAP.get(curve_type, 4)

        props = GProp_GProps()
        brepgprop.LinearProperties(edge, props)
        attrs[1] = props.Mass()

        t_min = curve_adaptor.FirstParameter()
        t_max = curve_adaptor.LastParameter()
        t_mid = (t_min + t_max) / 2
        pnt = gp_Pnt()
        tangent = gp_Vec()
        curve_adaptor.D1(t_mid, pnt, tangent)
        attrs[2] = pnt.X()
        attrs[3] = pnt.Y()
        attrs[4] = pnt.Z()
        mag = tangent.Magnitude()
        if mag > 1e-10:
            tangent.Divide(mag)
        attrs[5] = tangent.X()
        attrs[6] = tangent.Y()
        attrs[7] = tangent.Z()
    except Exception:
        pass

    if face1 is not None and face2 is not None:
        try:
            n1 = _face_normal_at_point(face1, attrs[2], attrs[3], attrs[4])
            n2 = _face_normal_at_point(face2, attrs[2], attrs[3], attrs[4])
            if n1 is not None and n2 is not None:
                dot = np.clip(np.dot(n1, n2), -1.0, 1.0)
                attrs[8] = np.arccos(dot)
                cross = np.cross(n1, n2)
                t_vec = np.array([attrs[5], attrs[6], attrs[7]])
                sign = np.dot(cross, t_vec)
                if abs(attrs[8]) < 0.01:
                    attrs[9] = 2
                elif sign > 0:
                    attrs[9] = 0
                else:
                    attrs[9] = 1
                attrs[10] = n1[0]
                attrs[11] = n1[1]
                attrs[12] = n1[2]
        except Exception:
            pass

    try:
        is_seam = BRep_Tool.IsClosed(edge, face1) if face1 is not None else False
        if not is_seam and face2 is not None:
            is_seam = BRep_Tool.IsClosed(edge, face2)
        attrs[13] = 1.0 if is_seam else 0.0
    except Exception:
        pass

    try:
        curve_adaptor = BRepAdaptor_Curve(edge)
        t_mid = (curve_adaptor.FirstParameter() + curve_adaptor.LastParameter()) / 2
        pnt2 = gp_Pnt()
        d1 = gp_Vec()
        d2 = gp_Vec()
        curve_adaptor.D2(t_mid, pnt2, d1, d2)
        d1_mag = d1.Magnitude()
        if d1_mag > 1e-10:
            cross_vec = d1.Crossed(d2)
            attrs[14] = cross_vec.Magnitude() / (d1_mag ** 3)
    except Exception:
        pass

    return attrs


def normalize_geometry(face_grid, face_attr, edge_grid, edge_attr):
    """Center and scale geometry to match MFTRCAD training format.

    Uses bounding-box center and half-max-extent as scale, matching the
    normalization used by the MFTRCAD dataset authors. This ensures inference
    features are consistent with training data.

    After normalization, converts face_attr and edge_attr to the MFTRCAD
    training layout:
      face_attr: [one_hot_type(6), 0, 0, 0, area, 0, cx, cy, cz]
      edge_attr: [0, convexity_onehot(2), length, flags(5), zeros(6)]
    """
    n_faces = face_grid.shape[0]
    if n_faces == 0:
        return face_grid, face_attr, edge_grid, edge_attr

    # Collect all valid xyz points from face UV-grids
    all_pts = []
    for i in range(n_faces):
        mask = face_grid[i, 6, :, :]
        valid = mask > 0.5
        if valid.any():
            pts = face_grid[i, 0:3, :, :][:, valid].T
            all_pts.append(pts)

    if not all_pts:
        return face_grid, face_attr, edge_grid, edge_attr

    all_pts = np.vstack(all_pts)

    # MFTRCAD normalization: bbox center + half-max-extent
    bbox_min = all_pts.min(axis=0)
    bbox_max = all_pts.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    extent = bbox_max - bbox_min
    scale = extent.max() / 2.0
    if scale < 1e-10:
        scale = 1.0

    # Normalize face UV-grid positions
    for i in range(n_faces):
        face_grid[i, 0, :, :] = (face_grid[i, 0, :, :] - center[0]) / scale
        face_grid[i, 1, :, :] = (face_grid[i, 1, :, :] - center[1]) / scale
        face_grid[i, 2, :, :] = (face_grid[i, 2, :, :] - center[2]) / scale

    # Normalize edge UV-grid positions
    for i in range(edge_grid.shape[0]):
        edge_grid[i, 0, :] = (edge_grid[i, 0, :] - center[0]) / scale
        edge_grid[i, 1, :] = (edge_grid[i, 1, :] - center[1]) / scale
        edge_grid[i, 2, :] = (edge_grid[i, 2, :] - center[2]) / scale

    # ---- Convert face_attr to MFTRCAD training format ----
    # Input layout: [type, area, cx, cy, cz, rational, n_loops, nx, ny, nz, bbx, bby, bbz, reversed]
    # Output layout: [one_hot_type(6), 0, 0, 0, area, 0, cx, cy, cz]
    n = face_attr.shape[0]
    new_face_attr = np.zeros((n, 14), dtype=np.float32)
    for i in range(n):
        surf_type = int(face_attr[i, 0])
        if 0 <= surf_type < 6:
            new_face_attr[i, surf_type] = 1.0
        area = face_attr[i, 1] / (scale ** 2)
        cx = (face_attr[i, 2] - center[0]) / scale
        cy = (face_attr[i, 3] - center[1]) / scale
        cz = (face_attr[i, 4] - center[2]) / scale
        new_face_attr[i, 9] = area
        new_face_attr[i, 11] = cx
        new_face_attr[i, 12] = cy
        new_face_attr[i, 13] = cz
    face_attr = new_face_attr

    # ---- Convert edge_attr to MFTRCAD training format ----
    # Input layout: [type, len, mx, my, mz, tx, ty, tz, dihedral, convexity, n1x, n1y, n1z, seam, curvature]
    # Output layout: [0, convex_flag, smooth_flag, length, concave_flag, circle_flag, 0, line_flag, other_type_flag, 0, 0, 0, 0, 0, 0]
    if edge_attr.shape[0] > 0:
        n_e = edge_attr.shape[0]
        new_edge_attr = np.zeros((n_e, 15), dtype=np.float32)
        for i in range(n_e):
            edge_type = int(edge_attr[i, 0])    # 0=Line, 1=Circle, 2=Ellipse, 3=BSpline
            length = edge_attr[i, 1] / scale
            convexity = int(edge_attr[i, 9])    # 0=convex, 1=concave, 2=smooth

            new_edge_attr[i, 3] = length

            # Convexity one-hot at indices 1, 2
            if convexity == 0:  # convex
                new_edge_attr[i, 1] = 1.0
            elif convexity == 2:  # smooth
                new_edge_attr[i, 2] = 1.0
            elif convexity == 1:  # concave
                new_edge_attr[i, 4] = 1.0

            # Edge type indicators
            if edge_type == 0:  # Line
                new_edge_attr[i, 7] = 1.0
            elif edge_type == 1:  # Circle
                new_edge_attr[i, 5] = 1.0
            elif edge_type >= 2:  # Other
                new_edge_attr[i, 8] = 1.0

        edge_attr = new_edge_attr

    return face_grid, face_attr, edge_grid, edge_attr


def step_to_graph(step_path: str) -> Optional[Dict]:
    """Convert a STEP file to raw graph data arrays.

    Args:
        step_path: Path to the STEP file.

    Returns:
        Dictionary with graph data arrays, or None on failure.
    """
    shape = read_step(step_path)
    if shape is None:
        return None

    faces = get_faces(shape)
    if len(faces) == 0:
        return None

    edge_pairs, edge_info = build_face_adjacency(shape, faces)

    face_attrs = []
    face_grids = []
    for face_i, face in enumerate(faces):
        try:
            face_attrs.append(compute_face_attributes(face))
            face_grids.append(compute_face_uv_grid(face))
        except Exception as e:
            logger.warning("Face %d attribute/grid computation failed: %s", face_i, e)
            face_attrs.append(np.zeros(14, dtype=np.float32))
            face_grids.append(np.zeros((7, 10, 10), dtype=np.float32))

    face_attr_array = np.stack(face_attrs)
    face_grid_array = np.stack(face_grids)

    if edge_pairs:
        src_list = [p[0] for p in edge_pairs]
        dst_list = [p[1] for p in edge_pairs]
        edge_index = np.array([src_list, dst_list], dtype=np.int64)

        edge_attrs_list = []
        edge_grids_list = []
        for pair in edge_pairs:
            fi, fj = pair
            shared_edges = edge_info.get((fi, fj), [])
            if shared_edges:
                f1 = faces[fi] if fi < len(faces) else None
                f2 = faces[fj] if fj < len(faces) else None
                try:
                    # Aggregate attributes from all shared edges (average)
                    all_attrs = []
                    for edge in shared_edges:
                        all_attrs.append(compute_edge_attributes(edge, f1, f2))
                    agg_attr = np.mean(np.stack(all_attrs), axis=0)
                    edge_attrs_list.append(agg_attr)

                    # Use the longest edge for the curve grid
                    best_edge = max(shared_edges, key=lambda e: agg_attr[1])
                    eg = compute_edge_curve_grid(best_edge)
                    n_pts = eg.shape[1]
                    # Sample face normals at each curve point
                    if f1 is not None:
                        for k in range(n_pts):
                            n1 = _face_normal_at_point(f1, eg[0, k], eg[1, k], eg[2, k])
                            if n1 is not None:
                                eg[6, k] = n1[0]
                                eg[7, k] = n1[1]
                                eg[8, k] = n1[2]
                    if f2 is not None:
                        for k in range(n_pts):
                            n2 = _face_normal_at_point(f2, eg[0, k], eg[1, k], eg[2, k])
                            if n2 is not None:
                                eg[9, k] = n2[0]
                                eg[10, k] = n2[1]
                                eg[11, k] = n2[2]
                    edge_grids_list.append(eg)
                except Exception as e:
                    logger.warning("Edge (%d,%d) computation failed: %s", fi, fj, e)
                    edge_attrs_list.append(np.zeros(15, dtype=np.float32))
                    edge_grids_list.append(np.zeros((12, 10), dtype=np.float32))
            else:
                edge_attrs_list.append(np.zeros(15, dtype=np.float32))
                edge_grids_list.append(np.zeros((12, 10), dtype=np.float32))

        edge_attr_array = np.stack(edge_attrs_list)
        edge_grid_array = np.stack(edge_grids_list)
    else:
        edge_index = np.array([[0], [0]], dtype=np.int64)
        edge_attr_array = np.zeros((1, 15), dtype=np.float32)
        edge_grid_array = np.zeros((1, 12, 10), dtype=np.float32)

    # Normalize
    face_grid_array, face_attr_array, edge_grid_array, edge_attr_array = normalize_geometry(
        face_grid_array, face_attr_array, edge_grid_array, edge_attr_array
    )

    return {
        "num_nodes": len(faces),
        "edge_index": edge_index,
        "face_attr": face_attr_array,
        "face_grid": face_grid_array,
        "edge_attr": edge_attr_array,
        "edge_grid": edge_grid_array,
    }


def step_to_preprocessed_sample(step_path: str, num_spatial: int = 64) -> Optional[Dict]:
    """Convert a STEP file to a preprocessed sample dict matching PreprocessedDataset format.

    Args:
        step_path: Path to the STEP file.
        num_spatial: Maximum spatial distance for shortest path computation.

    Returns:
        Dictionary matching PreprocessedDataset format, or None on failure.
    """
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent))
    from brepformer.data.preprocessing import precompute_graph_features

    data = step_to_graph(step_path)
    if data is None:
        return None

    # Extract face centroids and normals for descriptor computation
    face_centroids = data["face_attr"][:, 2:5]  # (N, 3)
    face_normals = data["face_attr"][:, 7:10]  # (N, 3)

    graph_features = precompute_graph_features(
        edge_index=data["edge_index"],
        num_nodes=data["num_nodes"],
        num_spatial=num_spatial,
        face_centroids=face_centroids,
        face_normals=face_normals,
    )

    model_id = Path(step_path).stem

    sample = {
        "model_id": model_id,
        "face_grid": data["face_grid"],
        "face_attr": data["face_attr"],
        "edge_index": data["edge_index"],
        "edge_attr": data["edge_attr"],
        "edge_grid": data["edge_grid"],
        "spatial_pos": graph_features["spatial_pos"],
        "in_degree": graph_features["in_degree"],
        "label": np.zeros(27, dtype=np.float32),  # dummy label
        "num_faces": data["num_nodes"],
        "num_edges": data["edge_index"].shape[1],
    }

    # Include D2 and angle descriptors so inference matches training features
    if "d2_distance" in graph_features:
        sample["d2_distance"] = graph_features["d2_distance"]
    if "angle_distance" in graph_features:
        sample["angle_distance"] = graph_features["angle_distance"]

    return sample


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m brepformer.data.step_to_graph <step_file>")
        sys.exit(1)

    step_path = sys.argv[1]
    data = step_to_graph(step_path)
    if data is None:
        print(f"Failed to convert {step_path}")
        sys.exit(1)

    print(f"Faces: {data['num_nodes']}")
    print(f"Edges: {data['edge_index'].shape[1]}")
    print(f"Face grid shape: {data['face_grid'].shape}")
    print(f"Edge grid shape: {data['edge_grid'].shape}")
