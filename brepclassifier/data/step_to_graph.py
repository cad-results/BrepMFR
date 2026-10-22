#!/usr/bin/env python3
"""STEP to brepformer graph JSON converter using pythonOCC.

Converts STEP CAD files to the brepformer graph JSON format:
["model_name", {
  "graph": {"edges": [[src...], [dst...]], "num_nodes": N},
  "graph_face_attr": (N, 14) float,
  "graph_face_grid": (N, 7, 10, 10) float,
  "graph_edge_attr": (E, 15) float,
  "graph_edge_grid": (E, 12, 10) float
}]

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
    """Read a STEP file and return the shape.

    Args:
        step_path: Path to the STEP file.

    Returns:
        TopoDS_Shape or None on failure.
    """
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != 1:  # IFSelect_RetDone
        return None
    reader.TransferRoots()
    shape = reader.OneShape()
    return shape


def get_faces(shape) -> list:
    """Extract all faces from a shape."""
    faces = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods.Face(explorer.Current())
        faces.append(face)
        explorer.Next()
    return faces


def get_edges(shape) -> list:
    """Extract all edges from a shape."""
    edges = []
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while explorer.More():
        edge = topods.Edge(explorer.Current())
        edges.append(edge)
        explorer.Next()
    return edges


def build_face_adjacency(shape, faces: list) -> Tuple[List[List[int]], Dict]:
    """Build face adjacency graph via shared edges.

    Args:
        shape: The TopoDS_Shape.
        faces: List of TopoDS_Face objects.

    Returns:
        edge_pairs: List of [src_face_idx, dst_face_idx] pairs (bidirectional).
        edge_info: Dict mapping (face_i, face_j) -> list of shared edges.
    """
    # Map edges to their parent faces
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

        # Collect face indices for this edge
        adj_faces = []
        it = TopTools_ListIteratorOfListOfShape(face_list)
        while it.More():
            f = topods.Face(it.Value())
            fidx = _find_face_index(f)
            if fidx is not None:
                adj_faces.append(fidx)
            it.Next()

        # Create adjacency edges between all pairs of adjacent faces
        for i in range(len(adj_faces)):
            for j in range(i + 1, len(adj_faces)):
                fi, fj = adj_faces[i], adj_faces[j]
                pair = (min(fi, fj), max(fi, fj))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    edge_pairs.append([fi, fj])
                    edge_pairs.append([fj, fi])  # bidirectional
                    edge_info[(fi, fj)] = [edge]
                    edge_info[(fj, fi)] = [edge]
                else:
                    # Additional shared edges
                    if (fi, fj) in edge_info:
                        edge_info[(fi, fj)].append(edge)
                    if (fj, fi) in edge_info:
                        edge_info[(fj, fi)].append(edge)

    return edge_pairs, edge_info


def compute_face_uv_grid(face, nu: int = 10, nv: int = 10) -> np.ndarray:
    """Sample a 7-channel UV grid from a face.

    Channels: x, y, z, nx, ny, nz, mask

    Args:
        face: TopoDS_Face
        nu: Number of U samples
        nv: Number of V samples

    Returns:
        grid: (7, nu, nv) float array
    """
    grid = np.zeros((7, nu, nv), dtype=np.float32)

    surf_adaptor = BRepAdaptor_Surface(face)
    u_min = surf_adaptor.FirstUParameter()
    u_max = surf_adaptor.LastUParameter()
    v_min = surf_adaptor.FirstVParameter()
    v_max = surf_adaptor.LastVParameter()

    # Clamp extreme parameter ranges
    u_min = max(u_min, -1e6)
    u_max = min(u_max, 1e6)
    v_min = max(v_min, -1e6)
    v_max = min(v_max, 1e6)

    # Get the surface for classification checking
    surface = BRep_Tool.Surface_s(face)
    if surface is None:
        return grid

    sa_surface = ShapeAnalysis_Surface(surface)

    is_reversed = face.Orientation() == TopAbs_REVERSED

    for i in range(nu):
        for j in range(nv):
            u = u_min + (u_max - u_min) * i / max(nu - 1, 1)
            v = v_min + (v_max - v_min) * j / max(nv - 1, 1)

            try:
                # Evaluate point
                pnt = gp_Pnt()
                surf_adaptor.D0(u, v, pnt)
                grid[0, i, j] = pnt.X()
                grid[1, i, j] = pnt.Y()
                grid[2, i, j] = pnt.Z()

                # Evaluate normal via D1
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

                # Mask: try to classify the point as inside the face
                grid[6, i, j] = 1.0

            except Exception:
                grid[6, i, j] = 0.0

    return grid


def compute_face_attributes(face) -> np.ndarray:
    """Compute 14-dimensional face attributes.

    Attributes:
    [0] surface_type (0-5)
    [1] area
    [2:5] centroid (x, y, z)
    [5] is_rational (0 or 1)
    [6] num_outer_loops
    [7:10] normal at centroid (nx, ny, nz)
    [10:13] bounding box extent (dx, dy, dz)
    [13] is_reversed (0 or 1)

    Args:
        face: TopoDS_Face

    Returns:
        attrs: (14,) float array
    """
    attrs = np.zeros(14, dtype=np.float32)

    surf_adaptor = BRepAdaptor_Surface(face)

    # Surface type
    surf_type = surf_adaptor.GetType()
    attrs[0] = SURFACE_TYPE_MAP.get(surf_type, 5)  # 5 = other

    # Area
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    attrs[1] = props.Mass()

    # Centroid
    centroid = props.CentreOfMass()
    attrs[2] = centroid.X()
    attrs[3] = centroid.Y()
    attrs[4] = centroid.Z()

    # Rationality
    attrs[5] = 1.0 if surf_type == GeomAbs_BSplineSurface else 0.0

    # Number of outer loops (wires)
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

    # Normal at center of parameter space
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

    # Bounding box extent
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

    # Is reversed
    attrs[13] = 1.0 if face.Orientation() == TopAbs_REVERSED else 0.0

    return attrs


def compute_edge_curve_grid(edge, n_pts: int = 10) -> np.ndarray:
    """Sample a 12-channel curve grid from an edge.

    Channels: x, y, z, tx, ty, tz, n1x, n1y, n1z, n2x, n2y, n2z
    where t = tangent, n1/n2 = normals of adjacent faces.

    For simplicity, n1 and n2 are set to zero (filled in compute_edge_attributes).

    Args:
        edge: TopoDS_Edge
        n_pts: Number of sample points

    Returns:
        grid: (12, n_pts) float array
    """
    grid = np.zeros((12, n_pts), dtype=np.float32)

    try:
        curve_adaptor = BRepAdaptor_Curve(edge)
        t_min = curve_adaptor.FirstParameter()
        t_max = curve_adaptor.LastParameter()

        # Clamp extreme ranges
        t_min = max(t_min, -1e6)
        t_max = min(t_max, 1e6)

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


def compute_edge_attributes(edge, face1=None, face2=None) -> np.ndarray:
    """Compute 15-dimensional edge attributes.

    Attributes:
    [0] curve_type (0-5)
    [1] length
    [2:5] midpoint (x, y, z)
    [5:8] tangent at midpoint (tx, ty, tz)
    [8] dihedral_angle
    [9] convexity (0=convex, 1=concave, 2=smooth)
    [10:13] face1 normal at edge midpoint
    [13] is_seam (0 or 1)
    [14] curvature at midpoint

    Args:
        edge: TopoDS_Edge
        face1: First adjacent TopoDS_Face (optional)
        face2: Second adjacent TopoDS_Face (optional)

    Returns:
        attrs: (15,) float array
    """
    attrs = np.zeros(15, dtype=np.float32)

    try:
        curve_adaptor = BRepAdaptor_Curve(edge)

        # Curve type
        curve_type = curve_adaptor.GetType()
        attrs[0] = CURVE_TYPE_MAP.get(curve_type, 4)  # 4 = other

        # Length
        props = GProp_GProps()
        brepgprop.LinearProperties(edge, props)
        attrs[1] = props.Mass()

        # Midpoint and tangent
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

    # Dihedral angle and convexity from adjacent faces
    if face1 is not None and face2 is not None:
        try:
            n1 = _face_normal_at_point(face1, attrs[2], attrs[3], attrs[4])
            n2 = _face_normal_at_point(face2, attrs[2], attrs[3], attrs[4])
            if n1 is not None and n2 is not None:
                dot = np.clip(np.dot(n1, n2), -1.0, 1.0)
                attrs[8] = np.arccos(dot)  # dihedral angle

                # Convexity
                cross = np.cross(n1, n2)
                t_vec = np.array([attrs[5], attrs[6], attrs[7]])
                sign = np.dot(cross, t_vec)
                if abs(attrs[8]) < 0.01:
                    attrs[9] = 2  # smooth
                elif sign > 0:
                    attrs[9] = 0  # convex
                else:
                    attrs[9] = 1  # concave

                attrs[10] = n1[0]
                attrs[11] = n1[1]
                attrs[12] = n1[2]
        except Exception:
            pass

    # Seam detection
    try:
        is_seam = BRep_Tool.IsClosed_s(edge, face1) if face1 is not None else False
        if not is_seam and face2 is not None:
            is_seam = BRep_Tool.IsClosed_s(edge, face2)
        attrs[13] = 1.0 if is_seam else 0.0
    except Exception:
        pass

    # Curvature at midpoint
    try:
        curve_adaptor = BRepAdaptor_Curve(edge)
        t_mid = (curve_adaptor.FirstParameter() + curve_adaptor.LastParameter()) / 2
        pnt2 = gp_Pnt()
        d1 = gp_Vec()
        d2 = gp_Vec()
        curve_adaptor.D2(t_mid, pnt2, d1, d2)
        d1_mag = d1.Magnitude()
        if d1_mag > 1e-10:
            # Curvature = |d1 x d2| / |d1|^3
            cross = d1.Crossed(d2)
            attrs[14] = cross.Magnitude() / (d1_mag ** 3)
    except Exception:
        pass

    return attrs


def _face_normal_at_point(face, x: float, y: float, z: float) -> Optional[np.ndarray]:
    """Get face normal at the closest point to (x, y, z)."""
    try:
        surface = BRep_Tool.Surface_s(face)
        if surface is None:
            return None

        sa_surface = ShapeAnalysis_Surface(surface)
        pnt = gp_Pnt(x, y, z)
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


def normalize_geometry(data: dict) -> dict:
    """Center and scale geometry to unit sphere.

    Normalizes face_grid xyz, face_attr centroid, edge_grid xyz,
    and edge_attr midpoint.

    Args:
        data: Dictionary with graph data arrays.

    Returns:
        Modified data dictionary.
    """
    face_grid = data["graph_face_grid"]  # (N, 7, 10, 10)
    n_faces = face_grid.shape[0]

    if n_faces == 0:
        return data

    # Collect all xyz points from face grids
    all_pts = []
    for i in range(n_faces):
        mask = face_grid[i, 6, :, :]
        valid = mask > 0.5
        if valid.any():
            pts = face_grid[i, 0:3, :, :][:, valid].T
            all_pts.append(pts)

    if not all_pts:
        return data

    all_pts = np.vstack(all_pts)

    # Compute center and scale
    center = all_pts.mean(axis=0)
    centered = all_pts - center
    scale = np.max(np.linalg.norm(centered, axis=1))
    if scale < 1e-10:
        scale = 1.0

    # Normalize face_grid xyz (channels 0-2)
    for i in range(n_faces):
        face_grid[i, 0, :, :] = (face_grid[i, 0, :, :] - center[0]) / scale
        face_grid[i, 1, :, :] = (face_grid[i, 1, :, :] - center[1]) / scale
        face_grid[i, 2, :, :] = (face_grid[i, 2, :, :] - center[2]) / scale

    # Normalize face_attr centroid (indices 2-4)
    face_attr = data["graph_face_attr"]
    face_attr[:, 2] = (face_attr[:, 2] - center[0]) / scale
    face_attr[:, 3] = (face_attr[:, 3] - center[1]) / scale
    face_attr[:, 4] = (face_attr[:, 4] - center[2]) / scale
    # Scale area
    face_attr[:, 1] = face_attr[:, 1] / (scale ** 2)
    # Scale bbox extent
    face_attr[:, 10:13] = face_attr[:, 10:13] / scale

    # Normalize edge_grid xyz (channels 0-2)
    edge_grid = data["graph_edge_grid"]
    for i in range(edge_grid.shape[0]):
        edge_grid[i, 0, :] = (edge_grid[i, 0, :] - center[0]) / scale
        edge_grid[i, 1, :] = (edge_grid[i, 1, :] - center[1]) / scale
        edge_grid[i, 2, :] = (edge_grid[i, 2, :] - center[2]) / scale

    # Normalize edge_attr midpoint (indices 2-4) and length
    edge_attr = data["graph_edge_attr"]
    if edge_attr.shape[0] > 0:
        edge_attr[:, 1] = edge_attr[:, 1] / scale
        edge_attr[:, 2] = (edge_attr[:, 2] - center[0]) / scale
        edge_attr[:, 3] = (edge_attr[:, 3] - center[1]) / scale
        edge_attr[:, 4] = (edge_attr[:, 4] - center[2]) / scale
        edge_attr[:, 14] = edge_attr[:, 14] / scale  # curvature has dimension 1/length

    data["graph_face_grid"] = face_grid
    data["graph_face_attr"] = face_attr
    data["graph_edge_grid"] = edge_grid
    data["graph_edge_attr"] = edge_attr

    return data


def step_to_graph(step_path: str, model_name: str) -> Optional[list]:
    """Convert a STEP file to brepformer graph JSON format.

    Args:
        step_path: Path to the STEP file.
        model_name: Name for the model.

    Returns:
        [model_name, data_dict] or None on failure.
    """
    # Read STEP file
    shape = read_step(step_path)
    if shape is None:
        return None

    # Extract faces and edges
    faces = get_faces(shape)
    if len(faces) == 0:
        return None

    # Build face adjacency
    edge_pairs, edge_info = build_face_adjacency(shape, faces)

    # Compute face attributes and grids
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

    face_attr_array = np.stack(face_attrs)  # (N, 14)
    face_grid_array = np.stack(face_grids)  # (N, 7, 10, 10)

    # Build edge index and compute edge attributes/grids
    if edge_pairs:
        src_list = [p[0] for p in edge_pairs]
        dst_list = [p[1] for p in edge_pairs]
        edge_index = [src_list, dst_list]

        edge_attrs = []
        edge_grids = []
        for pair in edge_pairs:
            fi, fj = pair
            # Get the shared edge(s) for this face pair
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
                    edge_attrs.append(agg_attr)

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
                    edge_grids.append(eg)
                except Exception as e:
                    logger.warning("Edge (%d,%d) computation failed: %s", fi, fj, e)
                    edge_attrs.append(np.zeros(15, dtype=np.float32))
                    edge_grids.append(np.zeros((12, 10), dtype=np.float32))
            else:
                edge_attrs.append(np.zeros(15, dtype=np.float32))
                edge_grids.append(np.zeros((12, 10), dtype=np.float32))

        edge_attr_array = np.stack(edge_attrs)
        edge_grid_array = np.stack(edge_grids)
    else:
        # No edges: create self-loops for single-face models
        edge_index = [[0], [0]]
        edge_attr_array = np.zeros((1, 15), dtype=np.float32)
        edge_grid_array = np.zeros((1, 12, 10), dtype=np.float32)

    # Assemble data
    data = {
        "graph": {
            "edges": edge_index,
            "num_nodes": len(faces),
        },
        "graph_face_attr": face_attr_array,
        "graph_face_grid": face_grid_array,
        "graph_edge_attr": edge_attr_array,
        "graph_edge_grid": edge_grid_array,
    }

    # Normalize geometry
    data = normalize_geometry(data)

    # Convert to lists for JSON serialization
    result = [model_name, {
        "graph": data["graph"],
        "graph_face_attr": data["graph_face_attr"].tolist(),
        "graph_face_grid": data["graph_face_grid"].tolist(),
        "graph_edge_attr": data["graph_edge_attr"].tolist(),
        "graph_edge_grid": data["graph_edge_grid"].tolist(),
    }]

    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python step_to_graph.py <step_file> [model_name]")
        sys.exit(1)

    step_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else Path(step_path).stem

    result = step_to_graph(step_path, model_name)
    if result is None:
        print(f"Failed to convert {step_path}")
        sys.exit(1)

    # Print summary
    data = result[1]
    print(f"Model: {result[0]}")
    print(f"Faces: {data['graph']['num_nodes']}")
    print(f"Edges: {len(data['graph']['edges'][0])}")
    print(f"Face grid shape: ({data['graph']['num_nodes']}, 7, 10, 10)")
    print(f"Edge grid shape: ({len(data['graph']['edges'][0])}, 12, 10)")

    # Save
    out_path = Path(step_path).with_suffix(".json")
    with open(out_path, "w") as f:
        json.dump(result, f)
    print(f"Saved to {out_path}")
