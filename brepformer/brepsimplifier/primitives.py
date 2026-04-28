#!/usr/bin/env python3
"""Primitive fitting for defeatured STEP files.

Takes a defeatured B-rep (output of defeature/defeature_v2) and approximates
it as a boolean union of geometric primitives: boxes, cylinders, cones,
spheres, and tori. Each face of the defeatured shape is analyzed for its
underlying surface type, grouped with neighbours that share compatible
geometry (same axis/radius/plane), and a best-fit primitive is constructed.

The result is a simplified CAD model built entirely from canonical primitives,
oriented and sized to match the defeatured input.

Usage:
    python -m brepformer.brepsimplifier.primitives --step defeatured.step
    python -m brepformer.brepsimplifier.primitives --step defeatured.step --output out.step
    python -m brepformer.brepsimplifier.primitives --step defeatured.step --verbose
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# OCC imports (deferred to functions to avoid top-level crash if OCC missing)
# ---------------------------------------------------------------------------


def _read_step(step_path: str):
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        return None
    reader.TransferRoots()
    return reader.OneShape()


def _write_step(shape, output_path: str) -> bool:
    from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCC.Core.IFSelect import IFSelect_RetDone

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    return writer.Write(str(output_path)) == IFSelect_RetDone


def _get_faces(shape) -> list:
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods

    faces = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        faces.append(topods.Face(exp.Current()))
        exp.Next()
    return faces


def _get_edges(shape) -> list:
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_EDGE
    from OCC.Core.TopoDS import topods

    edges = []
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        edges.append(topods.Edge(exp.Current()))
        exp.Next()
    return edges


# ---------------------------------------------------------------------------
# Face geometry analysis
# ---------------------------------------------------------------------------

# Surface type enum values from GeomAbs
SURF_PLANE = 0
SURF_CYLINDER = 1
SURF_CONE = 2
SURF_SPHERE = 3
SURF_TORUS = 4
SURF_OTHER = 99


def _analyze_face(face) -> dict:
    """Extract geometric parameters from a single face.

    Returns a dict with 'type' and type-specific parameters:
      plane:    normal (x,y,z), d (distance from origin), location
      cylinder: axis (x,y,z), center (x,y,z), radius
      cone:     axis (x,y,z), apex (x,y,z), semi_angle, ref_radius
      sphere:   center (x,y,z), radius
      torus:    axis (x,y,z), center (x,y,z), major_radius, minor_radius
    Also includes: area, bbox, centroid.
    """
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import (
        GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
        GeomAbs_Sphere, GeomAbs_Torus,
    )
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib

    sa = BRepAdaptor_Surface(face)
    stype = sa.GetType()

    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    area = props.Mass()
    centroid = props.CentreOfMass()

    bbox = Bnd_Box()
    brepbndlib.Add(face, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

    info = {
        "face": face,
        "area": area,
        "centroid": (centroid.X(), centroid.Y(), centroid.Z()),
        "bbox": (xmin, ymin, zmin, xmax, ymax, zmax),
    }

    if stype == GeomAbs_Plane:
        pln = sa.Plane()
        loc = pln.Location()
        ax = pln.Axis().Direction()
        a, b, c, d = pln.Coefficients()
        info["type"] = SURF_PLANE
        info["normal"] = (ax.X(), ax.Y(), ax.Z())
        info["location"] = (loc.X(), loc.Y(), loc.Z())
        info["coefficients"] = (a, b, c, d)

    elif stype == GeomAbs_Cylinder:
        cyl = sa.Cylinder()
        loc = cyl.Location()
        ax = cyl.Axis().Direction()
        info["type"] = SURF_CYLINDER
        info["axis"] = (ax.X(), ax.Y(), ax.Z())
        info["center"] = (loc.X(), loc.Y(), loc.Z())
        info["radius"] = cyl.Radius()

    elif stype == GeomAbs_Cone:
        cone = sa.Cone()
        loc = cone.Location()
        ax = cone.Axis().Direction()
        apex = cone.Apex()
        info["type"] = SURF_CONE
        info["axis"] = (ax.X(), ax.Y(), ax.Z())
        info["center"] = (loc.X(), loc.Y(), loc.Z())
        info["apex"] = (apex.X(), apex.Y(), apex.Z())
        info["semi_angle"] = cone.SemiAngle()
        info["ref_radius"] = cone.RefRadius()

    elif stype == GeomAbs_Sphere:
        sph = sa.Sphere()
        loc = sph.Location()
        info["type"] = SURF_SPHERE
        info["center"] = (loc.X(), loc.Y(), loc.Z())
        info["radius"] = sph.Radius()

    elif stype == GeomAbs_Torus:
        tor = sa.Torus()
        loc = tor.Location()
        ax = tor.Axis().Direction()
        info["type"] = SURF_TORUS
        info["axis"] = (ax.X(), ax.Y(), ax.Z())
        info["center"] = (loc.X(), loc.Y(), loc.Z())
        info["major_radius"] = tor.MajorRadius()
        info["minor_radius"] = tor.MinorRadius()

    else:
        info["type"] = SURF_OTHER

    return info


# ---------------------------------------------------------------------------
# Adjacency and grouping
# ---------------------------------------------------------------------------

def _build_face_adjacency(shape, faces: list) -> Dict[int, set]:
    """Build adjacency map: face index → set of neighboring face indices."""
    from OCC.Core.TopExp import TopExp_Explorer, topexp
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCC.Core.TopTools import (
        TopTools_IndexedDataMapOfShapeListOfShape,
        TopTools_IndexedMapOfShape,
        TopTools_ListIteratorOfListOfShape,
    )

    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

    face_map = TopTools_IndexedMapOfShape()
    for f in faces:
        face_map.Add(f)

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
                    if face_map.Contains(neighbor):
                        j = face_map.FindIndex(neighbor) - 1
                        if j != i and 0 <= j < len(faces):
                            adjacency[i].add(j)
                    it.Next()
            edge_exp.Next()

    return adjacency


def _vec_close(a: tuple, b: tuple, tol: float = 0.02) -> bool:
    """Check if two 3D vectors are nearly equal (or anti-parallel for normals)."""
    dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
    return abs(abs(dot) - 1.0) < tol


def _pt_close(a: tuple, b: tuple, tol: float) -> bool:
    """Check if two 3D points are within tolerance."""
    dx = a[0]-b[0]
    dy = a[1]-b[1]
    dz = a[2]-b[2]
    return math.sqrt(dx*dx + dy*dy + dz*dz) < tol


def _scalar_close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) < tol


def _curved_faces_compatible(info_a: dict, info_b: dict, tol: float) -> bool:
    """Check if two curved (non-planar) faces belong to the same primitive."""
    ta, tb = info_a["type"], info_b["type"]

    if ta == SURF_CYLINDER and tb == SURF_CYLINDER:
        return (_vec_close(info_a["axis"], info_b["axis"]) and
                _scalar_close(info_a["radius"], info_b["radius"], tol) and
                _axis_collinear(info_a["center"], info_a["axis"],
                                info_b["center"], info_b["axis"], tol))

    if ta == SURF_CONE and tb == SURF_CONE:
        return (_vec_close(info_a["axis"], info_b["axis"]) and
                _pt_close(info_a["apex"], info_b["apex"], tol))

    if ta == SURF_SPHERE and tb == SURF_SPHERE:
        return (_pt_close(info_a["center"], info_b["center"], tol) and
                _scalar_close(info_a["radius"], info_b["radius"], tol))

    if ta == SURF_TORUS and tb == SURF_TORUS:
        return (_vec_close(info_a["axis"], info_b["axis"]) and
                _pt_close(info_a["center"], info_b["center"], tol) and
                _scalar_close(info_a["major_radius"],
                              info_b["major_radius"], tol) and
                _scalar_close(info_a["minor_radius"],
                              info_b["minor_radius"], tol))

    return False


def _plane_caps_curved(plane_info: dict, curved_info: dict,
                       tol: float = 0.1) -> bool:
    """Check if a planar face is a cap of a curved primitive.

    For cylinders/cones: the plane normal must align with the axis AND
    the plane centroid must lie close to the axis line (within ~2x radius).
    This prevents large box faces that happen to be normal-aligned from
    being incorrectly claimed.
    """
    ct = curved_info["type"]
    if ct == SURF_CYLINDER:
        if not _vec_close(plane_info["normal"], curved_info["axis"]):
            return False
        # Check centroid proximity to cylinder axis
        pc = plane_info["centroid"]
        cc = curved_info["center"]
        ax = curved_info["axis"]
        dx = pc[0] - cc[0]
        dy = pc[1] - cc[1]
        dz = pc[2] - cc[2]
        proj = dx*ax[0] + dy*ax[1] + dz*ax[2]
        perp_sq = ((dx - proj*ax[0])**2 + (dy - proj*ax[1])**2 +
                   (dz - proj*ax[2])**2)
        r = curved_info["radius"]
        if perp_sq > (r * 1.5) ** 2:
            return False
        # A cap's area should be comparable to the circle area pi*r^2.
        # If the plane is much larger, it's a box face with a hole, not a cap.
        cap_area = math.pi * r * r
        return plane_info["area"] < cap_area * 2.0
    if ct == SURF_CONE:
        if not _vec_close(plane_info["normal"], curved_info["axis"]):
            return False
        pc = plane_info["centroid"]
        cc = curved_info["center"]
        ax = curved_info["axis"]
        dx = pc[0] - cc[0]
        dy = pc[1] - cc[1]
        dz = pc[2] - cc[2]
        proj = dx*ax[0] + dy*ax[1] + dz*ax[2]
        perp_sq = (dx - proj*ax[0])**2 + (dy - proj*ax[1])**2 + (dz - proj*ax[2])**2
        r = max(curved_info["ref_radius"], tol)
        return perp_sq < (r * 2.0) ** 2
    if ct == SURF_SPHERE:
        # A cap plane must be near the sphere center AND have area
        # comparable to a great circle cross-section (pi*r^2)
        r = curved_info["radius"]
        if not _pt_close(plane_info["centroid"], curved_info["center"],
                         r * 1.5):
            return False
        cap_area = math.pi * r * r
        return plane_info["area"] < cap_area * 2.0
    return False


def _axis_collinear(center_a: tuple, axis_a: tuple,
                    center_b: tuple, axis_b: tuple, tol: float) -> bool:
    """Check if two axes (point + direction) are approximately collinear."""
    dx = center_b[0] - center_a[0]
    dy = center_b[1] - center_a[1]
    dz = center_b[2] - center_a[2]
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    if dist < tol:
        return True
    ux, uy, uz = dx/dist, dy/dist, dz/dist
    dot = abs(ux*axis_a[0] + uy*axis_a[1] + uz*axis_a[2])
    return dot > (1.0 - 0.02)


def _group_faces_into_primitives(
    face_infos: List[dict],
    adjacency: Dict[int, set],
    tol: float = 0.1,
) -> List[List[int]]:
    """Group face indices into primitive candidates using a two-pass strategy.

    Pass 1 — Curved primitives: flood-fill from each non-planar face,
    pulling in adjacent curved faces with compatible geometry AND adjacent
    planar faces that act as caps (normal aligned with primitive axis).

    Pass 2 — Planar regions: remaining planar faces are grouped by pure
    edge-adjacency (no geometric constraint). A connected cluster of planes
    becomes a box candidate — all 6 sides of a box share edges.
    """
    n = len(face_infos)
    assigned = [False] * n
    groups: List[List[int]] = []

    # --- Pass 1: Curved primitives (cylinder, cone, sphere, torus) ---
    curved_indices = sorted(
        [i for i in range(n)
         if face_infos[i]["type"] not in (SURF_PLANE, SURF_OTHER)],
        key=lambda i: -face_infos[i]["area"],
    )

    for seed in curved_indices:
        if assigned[seed]:
            continue

        group = [seed]
        assigned[seed] = True
        # Only curved faces go in the expansion queue — planar caps are
        # added directly but don't propagate further. This prevents
        # intersection-patch planes from bridging into box faces.
        queue = [seed]

        while queue:
            current = queue.pop(0)
            for nb in adjacency.get(current, set()):
                if assigned[nb]:
                    continue
                fi_nb = face_infos[nb]
                nb_type = fi_nb["type"]

                if nb_type not in (SURF_PLANE, SURF_OTHER):
                    if _curved_faces_compatible(face_infos[seed], fi_nb, tol):
                        assigned[nb] = True
                        group.append(nb)
                        queue.append(nb)  # curved faces expand further
                elif nb_type == SURF_PLANE:
                    seed_info = face_infos[seed]
                    if _plane_caps_curved(fi_nb, seed_info, tol):
                        assigned[nb] = True
                        group.append(nb)
                        # Do NOT add to queue — caps don't propagate

        groups.append(group)

    # --- Pass 2: Planar faces grouped by adjacency (boxes) ---
    # Only propagate through PLANE faces. BSpline/OTHER faces don't form
    # meaningful box groups and act as bridges between unrelated regions.
    remaining_planes = [i for i in range(n)
                        if not assigned[i] and face_infos[i]["type"] == SURF_PLANE]
    remaining_other = [i for i in range(n)
                       if not assigned[i] and face_infos[i]["type"] != SURF_PLANE]

    visited_r = set()
    for seed in remaining_planes:
        if seed in visited_r:
            continue

        group = [seed]
        visited_r.add(seed)
        queue = [seed]

        while queue:
            current = queue.pop(0)
            for nb in adjacency.get(current, set()):
                if nb in visited_r or assigned[nb]:
                    continue
                if face_infos[nb]["type"] != SURF_PLANE:
                    continue
                visited_r.add(nb)
                group.append(nb)
                queue.append(nb)

        groups.append(group)

    # BSpline/OTHER faces: group by centroid proximity so nearby
    # freeform surfaces merge into single enclosing primitives.
    if remaining_other:
        total_area = sum(face_infos[i]["area"] for i in range(n))
        # Skip degenerate faces (no boundary points, no UV extent)
        valid_other = []
        for i in remaining_other:
            pts = _sample_face_boundary_points(face_infos[i]["face"], n_pts=6)
            if pts or face_infos[i]["area"] > total_area * 0.001:
                valid_other.append(i)

        # Group by centroid proximity: merge faces whose centroids are
        # within a characteristic length (cube root of total bbox volume)
        if valid_other:
            bb = _group_bbox(face_infos, list(range(n)))
            char_len = max(
                (bb[3]-bb[0]) * (bb[4]-bb[1]) * (bb[5]-bb[2]),
                1e-6,
            ) ** (1.0 / 3.0) * 0.3  # 30% of characteristic length
            char_len = max(char_len, 1.0)

            visited_o = set()
            for seed in valid_other:
                if seed in visited_o:
                    continue
                group = [seed]
                visited_o.add(seed)
                sc = face_infos[seed]["centroid"]
                for j in valid_other:
                    if j in visited_o:
                        continue
                    jc = face_infos[j]["centroid"]
                    dist = math.sqrt(
                        (sc[0]-jc[0])**2 + (sc[1]-jc[1])**2 +
                        (sc[2]-jc[2])**2)
                    if dist < char_len:
                        group.append(j)
                        visited_o.add(j)
                groups.append(group)

    return groups


# ---------------------------------------------------------------------------
# Primitive construction from face groups
# ---------------------------------------------------------------------------

def _group_bbox(face_infos: List[dict], indices: List[int]) -> Tuple:
    """Compute the bounding box of a group of faces."""
    xmin = ymin = zmin = float("inf")
    xmax = ymax = zmax = float("-inf")
    for i in indices:
        bb = face_infos[i]["bbox"]
        xmin = min(xmin, bb[0])
        ymin = min(ymin, bb[1])
        zmin = min(zmin, bb[2])
        xmax = max(xmax, bb[3])
        ymax = max(ymax, bb[4])
        zmax = max(zmax, bb[5])
    return xmin, ymin, zmin, xmax, ymax, zmax


def _dominant_type(face_infos: List[dict], indices: List[int]) -> int:
    """Return the representative surface type for a group.

    Curved surfaces (cylinder, cone, sphere, torus) take priority over
    planes, since a group with even one curved face was formed by the
    curved-primitive pass and should be constructed as that primitive.
    Only falls back to SURF_PLANE when no curved faces exist.
    """
    type_area: Dict[int, float] = defaultdict(float)
    for i in indices:
        type_area[face_infos[i]["type"]] += face_infos[i]["area"]

    # Prefer curved primitives over planes
    curved_types = {SURF_CYLINDER, SURF_CONE, SURF_SPHERE, SURF_TORUS}
    curved_present = {t: a for t, a in type_area.items() if t in curved_types}
    if curved_present:
        return max(curved_present, key=curved_present.get)

    return max(type_area, key=type_area.get)


def _make_box(xmin, ymin, zmin, xmax, ymax, zmax):
    """Create a box solid from bounding coordinates."""
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.gp import gp_Pnt

    dx = max(xmax - xmin, 1e-6)
    dy = max(ymax - ymin, 1e-6)
    dz = max(zmax - zmin, 1e-6)
    return BRepPrimAPI_MakeBox(gp_Pnt(xmin, ymin, zmin), dx, dy, dz).Shape()


def _simplify_points_dp(points: list, tolerance_frac: float = 0.05) -> list:
    """Douglas-Peucker-like simplification of a 3D polyline.

    Removes interior points whose removal doesn't change the bounding
    extent by more than *tolerance_frac* (fraction of the segment span).
    This reduces bounding volume for BSpline faces with wavy edges.
    """
    if len(points) <= 2:
        return list(points)

    # Compute overall bounding box of this edge's points
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    span_z = max(zs) - min(zs)
    overall_span = math.sqrt(span_x**2 + span_y**2 + span_z**2)

    if overall_span < 1e-10:
        return [points[0], points[-1]]

    tol = tolerance_frac * overall_span

    # Iterative simplification: remove points that don't change extent
    simplified = list(points)
    changed = True
    while changed and len(simplified) > 2:
        changed = False
        new_pts = [simplified[0]]
        i = 1
        while i < len(simplified) - 1:
            # Check if removing simplified[i] changes the bounding box
            # Look at the local triple: prev, current, next
            prev = new_pts[-1]
            curr = simplified[i]
            nxt = simplified[i + 1]

            # Bounding box with current point
            local_with = [prev, curr, nxt]
            lx_w = [p[0] for p in local_with]
            ly_w = [p[1] for p in local_with]
            lz_w = [p[2] for p in local_with]

            # Bounding box without current point
            local_without = [prev, nxt]
            lx_wo = [p[0] for p in local_without]
            ly_wo = [p[1] for p in local_without]
            lz_wo = [p[2] for p in local_without]

            # Change in extent
            dx = abs((max(lx_w) - min(lx_w)) - (max(lx_wo) - min(lx_wo)))
            dy = abs((max(ly_w) - min(ly_w)) - (max(ly_wo) - min(ly_wo)))
            dz = abs((max(lz_w) - min(lz_w)) - (max(lz_wo) - min(lz_wo)))
            extent_change = math.sqrt(dx**2 + dy**2 + dz**2)

            if extent_change < tol:
                # Safe to remove — skip this point
                changed = True
            else:
                new_pts.append(curr)
            i += 1
        new_pts.append(simplified[-1])
        simplified = new_pts

    return simplified


def _sample_face_boundary_points(face, n_pts: int = 20) -> list:
    """Sample points along the edges of a face for accurate extent measurement.

    Falls back to UV-grid sampling if the face has no edges (e.g. closed
    BSpline shells without trimming wires).

    For BSpline/freeform faces, applies Douglas-Peucker-like simplification
    to remove wavy detour points that inflate the bounding volume without
    contributing to the essential shape extent.
    """
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_EDGE
    from OCC.Core.TopoDS import topods
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
    from OCC.Core.GeomAbs import GeomAbs_BSplineCurve, GeomAbs_BezierCurve
    from OCC.Core.gp import gp_Pnt

    points = []
    has_freeform = False
    exp = TopExp_Explorer(face, TopAbs_EDGE)
    while exp.More():
        edge = topods.Edge(exp.Current())
        try:
            ca = BRepAdaptor_Curve(edge)
            u0 = ca.FirstParameter()
            u1 = ca.LastParameter()
            u0 = max(u0, -1e6)
            u1 = min(u1, 1e6)

            # Check if this edge is a freeform curve
            ctype = ca.GetType()
            is_freeform = ctype in (GeomAbs_BSplineCurve, GeomAbs_BezierCurve)
            if is_freeform:
                has_freeform = True

            edge_pts = []
            for k in range(n_pts):
                t = u0 + (u1 - u0) * k / max(n_pts - 1, 1)
                pt = gp_Pnt()
                ca.D0(t, pt)
                edge_pts.append((pt.X(), pt.Y(), pt.Z()))

            # Simplify freeform curves to remove wavy detours
            if is_freeform and len(edge_pts) > 2:
                edge_pts = _simplify_points_dp(edge_pts, tolerance_frac=0.05)

            points.extend(edge_pts)
        except Exception:
            pass
        exp.Next()

    if not points:
        # No edges — sample on UV grid (closed surfaces, untrimmed patches)
        try:
            sa = BRepAdaptor_Surface(face)
            u0, u1 = sa.FirstUParameter(), sa.LastUParameter()
            v0, v1 = sa.FirstVParameter(), sa.LastVParameter()
            # Clamp infinite UV domains
            u0 = max(u0, -1e4)
            u1 = min(u1, 1e4)
            v0 = max(v0, -1e4)
            v1 = min(v1, 1e4)
            if u1 - u0 > 1e-10 and v1 - v0 > 1e-10:
                n_uv = max(int(math.sqrt(n_pts)), 3)
                for iu in range(n_uv):
                    for iv in range(n_uv):
                        u = u0 + (u1 - u0) * iu / max(n_uv - 1, 1)
                        v = v0 + (v1 - v0) * iv / max(n_uv - 1, 1)
                        pt = gp_Pnt()
                        sa.D0(u, v, pt)
                        points.append((pt.X(), pt.Y(), pt.Z()))
        except Exception:
            pass

    return points


def _aabb_fallback_from_faces(face_infos: List[dict], indices: List[int]):
    """Build a tight axis-aligned bounding box from boundary points of faces."""
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.gp import gp_Pnt

    all_pts = []
    for i in indices:
        pts = _sample_face_boundary_points(face_infos[i]["face"], n_pts=12)
        all_pts.extend(pts)
        all_pts.append(face_infos[i]["centroid"])
    if not all_pts:
        return None
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    zs = [p[2] for p in all_pts]
    dx = max(max(xs) - min(xs), 1e-6)
    dy = max(max(ys) - min(ys), 1e-6)
    dz = max(max(zs) - min(zs), 1e-6)
    return BRepPrimAPI_MakeBox(gp_Pnt(min(xs), min(ys), min(zs)),
                               dx, dy, dz).Shape()


def _volume_sanity_check(
    volume: float,
    total_area: float,
    model_volume: float = 0.0,
    lenient: bool = False,
) -> bool:
    """Return True if the primitive volume is plausible.

    Two checks:
    1. Area-based: vol <= 0.1 * area^1.5 (sphere ratio is ~0.094).
    2. Model-based: if model volume is known, no single primitive should
       exceed 2x the entire model's volume (20x in lenient mode).
    """
    if total_area < 1e-6:
        return True
    area_threshold = 0.1 * total_area ** 1.5
    if not lenient and volume > area_threshold:
        return False
    vol_limit = 20.0 if lenient else 2.0
    if model_volume > 0 and volume > vol_limit * model_volume:
        return False
    return True


def _make_cylinder_from_faces(
    face_infos: List[dict], indices: List[int], model_volume: float = 0.0,
    lenient: bool = False,
):
    """Construct a cylinder primitive matching the exact geometry of the input faces."""
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCC.Core.gp import gp_Ax2, gp_Pnt, gp_Dir

    cyl_faces = [i for i in indices if face_infos[i]["type"] == SURF_CYLINDER]
    if not cyl_faces:
        return None

    ref = face_infos[cyl_faces[0]]
    axis = ref["axis"]
    center = ref["center"]
    radius = ref["radius"]

    if len(cyl_faces) > 1:
        total_area = sum(face_infos[i]["area"] for i in cyl_faces)
        radius = sum(face_infos[i]["radius"] * face_infos[i]["area"]
                     for i in cyl_faces) / max(total_area, 1e-12)

    # Only use boundary points from cylindrical faces (matching radius)
    # for height determination. Cap planes should NOT extend the cylinder
    # beyond the actual cylindrical surface extent.
    cyl_projections = []
    for i in cyl_faces:
        pts = _sample_face_boundary_points(face_infos[i]["face"], n_pts=12)
        for pt in pts:
            p = ((pt[0] - center[0]) * axis[0] +
                 (pt[1] - center[1]) * axis[1] +
                 (pt[2] - center[2]) * axis[2])
            cyl_projections.append(p)
        # Also use centroid of the cylinder face
        cx, cy, cz = face_infos[i]["centroid"]
        cyl_projections.append(
            (cx - center[0]) * axis[0] +
            (cy - center[1]) * axis[1] +
            (cz - center[2]) * axis[2])

    if not cyl_projections:
        return None

    h_min = min(cyl_projections)
    h_max = max(cyl_projections)
    height = max(h_max - h_min, 1e-6)

    origin = (
        center[0] + axis[0] * h_min,
        center[1] + axis[1] * h_min,
        center[2] + axis[2] * h_min,
    )

    cyl_vol = math.pi * radius * radius * height
    total_area = sum(face_infos[i]["area"] for i in indices)
    if not _volume_sanity_check(cyl_vol, total_area, model_volume, lenient=lenient):
        return None

    ax2 = gp_Ax2(gp_Pnt(*origin), gp_Dir(*axis))
    return BRepPrimAPI_MakeCylinder(ax2, radius, height).Shape()


def _make_cone_from_faces(
    face_infos: List[dict], indices: List[int], model_volume: float = 0.0,
    lenient: bool = False,
):
    """Construct a cone primitive matching the exact geometry of the input faces.

    Uses boundary-point sampling to find the actual extent of the cone along
    its axis, then computes radii at each end from the cone's geometric
    definition (ref_radius + h * tan(semi_angle)).

    Only cone faces (not cap planes) contribute to height determination,
    preventing single-face cones from spanning the entire model height.
    """
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCone
    from OCC.Core.gp import gp_Ax2, gp_Pnt, gp_Dir

    cone_faces = [i for i in indices if face_infos[i]["type"] == SURF_CONE]
    if not cone_faces:
        return None

    ref = face_infos[cone_faces[0]]
    axis = ref["axis"]
    center = ref["center"]
    semi_angle = ref["semi_angle"]  # may be negative

    # Only use boundary points from cone faces for height determination.
    # Cap planes should NOT extend the cone beyond the conical surface.
    cone_projections = []
    for i in cone_faces:
        pts = _sample_face_boundary_points(face_infos[i]["face"], n_pts=12)
        for pt in pts:
            p = ((pt[0] - center[0]) * axis[0] +
                 (pt[1] - center[1]) * axis[1] +
                 (pt[2] - center[2]) * axis[2])
            cone_projections.append(p)

    if not cone_projections:
        return None

    h_min = min(cone_projections)
    h_max = max(cone_projections)
    height = max(h_max - h_min, 1e-6)

    # Radius at a given height h along axis from center:
    # r(h) = ref_radius + h * tan(semi_angle)
    tan_sa = math.tan(semi_angle)
    r1 = max(ref["ref_radius"] + h_min * tan_sa, 0.0)
    r2 = max(ref["ref_radius"] + h_max * tan_sa, 0.0)

    origin = (
        center[0] + axis[0] * h_min,
        center[1] + axis[1] * h_min,
        center[2] + axis[2] * h_min,
    )

    cone_vol = (math.pi * height / 3.0) * (r1*r1 + r1*r2 + r2*r2)
    total_area = sum(face_infos[i]["area"] for i in indices)
    if not _volume_sanity_check(cone_vol, total_area, model_volume, lenient=lenient):
        return None

    ax2 = gp_Ax2(gp_Pnt(*origin), gp_Dir(*axis))
    try:
        return BRepPrimAPI_MakeCone(ax2, r1, r2, height).Shape()
    except Exception:
        # If r1 == r2, MakeCone may fail — fall back to cylinder
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
        r_avg = (r1 + r2) / 2.0
        return BRepPrimAPI_MakeCylinder(ax2, max(r_avg, 1e-6), height).Shape()


def _make_sphere_from_faces(face_infos: List[dict], indices: List[int]):
    """Construct a sphere primitive from a group of spherical faces."""
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
    from OCC.Core.gp import gp_Pnt

    sph_faces = [i for i in indices if face_infos[i]["type"] == SURF_SPHERE]
    if not sph_faces:
        return None

    ref = face_infos[sph_faces[0]]
    center = ref["center"]
    radius = ref["radius"]

    return BRepPrimAPI_MakeSphere(gp_Pnt(*center), radius).Shape()


def _make_torus_from_faces(
    face_infos: List[dict], indices: List[int], model_volume: float = 0.0,
):
    """Construct a torus primitive matching the actual angular extent of the faces.

    Samples boundary points from torus faces, projects them into the plane
    perpendicular to the torus axis, and computes the angular sweep.  If the
    faces span less than ~350° around the axis a partial torus sector is built
    via the ``BRepPrimAPI_MakeTorus(ax2, R1, R2, angle)`` overload.
    """
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeTorus
    from OCC.Core.gp import gp_Ax2, gp_Pnt, gp_Dir

    tor_faces = [i for i in indices if face_infos[i]["type"] == SURF_TORUS]
    if not tor_faces:
        return None

    ref = face_infos[tor_faces[0]]
    axis = ref["axis"]
    center = ref["center"]
    major_r = ref["major_radius"]
    minor_r = ref["minor_radius"]

    # Build a local right-handed frame: Z = axis, X = any perpendicular
    az = axis
    # Choose a seed vector not parallel to az
    if abs(az[0]) < 0.9:
        seed = (1.0, 0.0, 0.0)
    else:
        seed = (0.0, 1.0, 0.0)
    # ax = seed - (seed·az)*az  (perpendicular to az)
    dot_sa = seed[0]*az[0] + seed[1]*az[1] + seed[2]*az[2]
    ax_raw = (seed[0] - dot_sa*az[0],
              seed[1] - dot_sa*az[1],
              seed[2] - dot_sa*az[2])
    norm = math.sqrt(ax_raw[0]**2 + ax_raw[1]**2 + ax_raw[2]**2)
    loc_x = (ax_raw[0]/norm, ax_raw[1]/norm, ax_raw[2]/norm)
    # ay = az × ax
    loc_y = (az[1]*loc_x[2] - az[2]*loc_x[1],
             az[2]*loc_x[0] - az[0]*loc_x[2],
             az[0]*loc_x[1] - az[1]*loc_x[0])

    # Collect angles of boundary points projected onto the major-radius circle
    angles = []
    for i in tor_faces:
        pts = _sample_face_boundary_points(face_infos[i]["face"], n_pts=16)
        for pt in pts:
            dx = pt[0] - center[0]
            dy = pt[1] - center[1]
            dz = pt[2] - center[2]
            px = dx*loc_x[0] + dy*loc_x[1] + dz*loc_x[2]
            py = dx*loc_y[0] + dy*loc_y[1] + dz*loc_y[2]
            angles.append(math.atan2(py, px))

    if not angles:
        ax2 = gp_Ax2(gp_Pnt(*center), gp_Dir(*axis))
        return BRepPrimAPI_MakeTorus(ax2, major_r, minor_r).Shape()

    # Determine angular span: find the largest gap in the sorted angles,
    # then the sector is everything outside that gap.
    angles.sort()
    n = len(angles)
    max_gap = 0.0
    max_gap_idx = -1
    for k in range(n):
        gap = angles[(k + 1) % n] - angles[k]
        if k == n - 1:
            gap = angles[0] + 2*math.pi - angles[-1]
        if gap > max_gap:
            max_gap = gap
            max_gap_idx = k

    sweep = 2*math.pi - max_gap
    FULL_THRESHOLD = math.radians(350)
    PAD = math.radians(2)

    if sweep >= FULL_THRESHOLD:
        ax2 = gp_Ax2(gp_Pnt(*center), gp_Dir(*axis))
        return BRepPrimAPI_MakeTorus(ax2, major_r, minor_r).Shape()

    # Start angle = angle just after the largest gap
    start = angles[(max_gap_idx + 1) % n] - PAD
    sweep = min(sweep + 2*PAD, 2*math.pi)

    # Orient the ax2 XDirection toward the start angle
    sx = math.cos(start)*loc_x[0] + math.sin(start)*loc_y[0]
    sy = math.cos(start)*loc_x[1] + math.sin(start)*loc_y[1]
    sz = math.cos(start)*loc_x[2] + math.sin(start)*loc_y[2]

    ax2 = gp_Ax2(gp_Pnt(*center), gp_Dir(*axis), gp_Dir(sx, sy, sz))

    tor_vol = sweep / (2*math.pi) * 2*math.pi**2 * major_r * minor_r**2
    total_area = sum(face_infos[i]["area"] for i in indices)
    if not _volume_sanity_check(tor_vol, total_area, model_volume):
        return None

    return BRepPrimAPI_MakeTorus(ax2, major_r, minor_r, sweep).Shape()


def _tessellate_bspline_faces(
    face_infos: List[dict],
    indices: List[int],
    verbose: bool = False,
    model_volume: float = 0.0,
    nu: int = 6,
    nv: int = 6,
) -> Optional[object]:
    """Approximate BSpline/OTHER faces as a union of small rectangular prisms.

    Samples each face on a UV grid and creates a thin oriented box at each
    cell, with thickness derived from local curvature.  The result is a
    compound of many small boxes that together trace the surface shape.
    """
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.TopoDS import TopoDS_Compound
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.gp import gp_Pnt, gp_Vec
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface

    char_length = model_volume ** (1.0 / 3.0) if model_volume > 0 else 10.0
    min_thickness = char_length * 0.01

    prisms = []
    for idx in indices:
        face = face_infos[idx]["face"]
        try:
            sa = BRepAdaptor_Surface(face)
            u0 = sa.FirstUParameter()
            u1 = sa.LastUParameter()
            v0 = sa.FirstVParameter()
            v1 = sa.LastVParameter()
            u0 = max(u0, -1e4)
            u1 = min(u1, 1e4)
            v0 = max(v0, -1e4)
            v1 = min(v1, 1e4)
            if u1 - u0 < 1e-10 or v1 - v0 < 1e-10:
                continue
        except Exception:
            continue

        for iu in range(nu):
            for iv in range(nv):
                ua = u0 + (u1 - u0) * iu / nu
                ub = u0 + (u1 - u0) * (iu + 1) / nu
                va = v0 + (v1 - v0) * iv / nv
                vb = v0 + (v1 - v0) * (iv + 1) / nv

                try:
                    corners = []
                    for u in (ua, ub):
                        for v in (va, vb):
                            pt = gp_Pnt()
                            sa.D0(u, v, pt)
                            corners.append((pt.X(), pt.Y(), pt.Z()))

                    xs = [c[0] for c in corners]
                    ys = [c[1] for c in corners]
                    zs = [c[2] for c in corners]
                    dx = max(xs) - min(xs)
                    dy = max(ys) - min(ys)
                    dz = max(zs) - min(zs)
                    if dx < 1e-12 and dy < 1e-12 and dz < 1e-12:
                        continue

                    thickness = max(min_thickness, char_length * 0.005)
                    try:
                        um = (ua + ub) / 2
                        vm = (va + vb) / 2
                        pnt_m = gp_Pnt()
                        d1u = gp_Vec()
                        d1v = gp_Vec()
                        sa.D1(um, vm, pnt_m, d1u, d1v)
                        normal = d1u.Crossed(d1v)
                        mag = normal.Magnitude()
                        if mag > 1e-10:
                            normal.Divide(mag)
                            half_t = thickness / 2.0
                            nx, ny, nz = normal.X(), normal.Y(), normal.Z()
                            for sign in (-half_t, half_t):
                                corners.append((
                                    pnt_m.X() + nx * sign,
                                    pnt_m.Y() + ny * sign,
                                    pnt_m.Z() + nz * sign,
                                ))
                    except Exception:
                        cx = sum(xs) / len(xs)
                        cy = sum(ys) / len(ys)
                        cz = sum(zs) / len(zs)
                        half_t = thickness / 2.0
                        corners.append((cx, cy - half_t, cz))
                        corners.append((cx, cy + half_t, cz))

                    xs = [c[0] for c in corners]
                    ys = [c[1] for c in corners]
                    zs = [c[2] for c in corners]
                    xmin, xmax = min(xs), max(xs)
                    ymin, ymax = min(ys), max(ys)
                    zmin, zmax = min(zs), max(zs)

                    bx = max(xmax - xmin, min_thickness)
                    by = max(ymax - ymin, min_thickness)
                    bz = max(zmax - zmin, min_thickness)

                    box = BRepPrimAPI_MakeBox(
                        gp_Pnt(xmin, ymin, zmin), bx, by, bz).Shape()
                    prisms.append(box)
                except Exception:
                    continue

    if not prisms:
        return None

    if verbose:
        print(f"    BSpline→tessellated ({len(prisms)} prisms "
              f"from {len(indices)} faces)")

    if len(prisms) == 1:
        return prisms[0]

    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    for p in prisms:
        builder.Add(compound, p)
    return compound


def _make_enclosing_for_bspline(
    face_infos: List[dict],
    indices: List[int],
    verbose: bool = False,
    model_bbox: Tuple = None,
    model_volume: float = 0.0,
    total_model_area: float = 0.0,
) -> Optional[object]:
    """Fit a BSpline/OTHER face to a tight enclosing primitive.

    Strategy:
    1. Sample dense boundary + UV-grid interior points from all faces.
    2. Skip groups that span >40% of the model in 2+ dimensions
       (these are outer shells / blending surfaces, not detail features).
    3. Detect multi-lobe clusters and split if needed.
    4. Use PCA to find 3 principal axes.
    5. Compute convex hull volume estimate; if much smaller than OBB, tighten.
    6. Build an oriented bounding box (OBB) using the PCA axes.
    7. For elongated shapes with uniform radial spread, use a cylinder.
    8. Detect thin shells and fit a tighter prism.
    """
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.gp import gp_Ax2, gp_Pnt, gp_Dir, gp_Trsf
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepTools import breptools

    # Sample boundary points + UV-grid interior points for tighter fits
    all_pts = []
    for i in indices:
        # Boundary points (edges)
        pts = _sample_face_boundary_points(face_infos[i]["face"], n_pts=20)
        all_pts.extend(pts)
        all_pts.append(face_infos[i]["centroid"])

        # UV-grid surface sampling for interior points
        try:
            face = face_infos[i]["face"]
            sa = BRepAdaptor_Surface(face)
            u0, u1, v0, v1 = 0.0, 0.0, 0.0, 0.0
            breptools.UVBounds_s(face, u0, u1, v0, v1)
            # Clamp infinite domains
            u0 = max(u0, -1e4)
            u1 = min(u1, 1e4)
            v0 = max(v0, -1e4)
            v1 = min(v1, 1e4)
            if u1 - u0 > 1e-10 and v1 - v0 > 1e-10:
                # Sample 5x5 grid for interior points
                for iu in range(1, 4):  # Skip boundaries (already in edge samples)
                    for iv in range(1, 4):
                        u = u0 + (u1 - u0) * iu / 4.0
                        v = v0 + (v1 - v0) * iv / 4.0
                        pt = gp_Pnt()
                        sa.D0(u, v, pt)
                        all_pts.append((pt.X(), pt.Y(), pt.Z()))
        except Exception:
            pass

    if len(all_pts) < 3:
        return None

    # Check if this group spans most of the model (outer shell)
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    zs = [p[2] for p in all_pts]

    # Compute volume budget for this BSpline group based on area share
    total_area = sum(face_infos[i]["area"] for i in indices)
    volume_budget = 0.0
    if total_model_area > 1e-6 and model_volume > 0:
        area_share = total_area / total_model_area
        volume_budget = model_volume * area_share
        if verbose:
            print(f"    BSpline budget: area_share={area_share:.4f}, "
                  f"budget={volume_budget:.1f}")

    if model_bbox is not None:
        model_extents = [
            max(model_bbox[3] - model_bbox[0], 1e-6),
            max(model_bbox[4] - model_bbox[1], 1e-6),
            max(model_bbox[5] - model_bbox[2], 1e-6),
        ]
        group_extents = [
            max(xs) - min(xs),
            max(ys) - min(ys),
            max(zs) - min(zs),
        ]
        # Skip BSpline groups that span a large fraction of the model
        # in 2+ dimensions AND have no meaningful volume budget
        large_dims = sum(1 for k in range(3)
                         if group_extents[k] > 0.3 * model_extents[k])
        group_vol = (group_extents[0] * group_extents[1] *
                     group_extents[2])
        model_vol = (model_extents[0] * model_extents[1] *
                     model_extents[2])
        vol_ratio = group_vol / max(model_vol, 1e-6)
        if large_dims >= 2 or vol_ratio > 0.03:
            if verbose:
                print(f"    BSpline→SKIP (spans {large_dims}/3 model dims, "
                      f"vol_ratio={vol_ratio:.4f}, {len(indices)} faces)")
            return None  # skip — the analytic faces cover this volume

    # Check if this is a tiny detail feature relative to model size
    if model_volume > 0:
        char_length = model_volume ** (1.0/3.0)
        min_area = 0.01 * (char_length ** 2)  # 1% of characteristic area
        if total_area < min_area:
            if verbose:
                print(f"    BSpline→SKIP (tiny area {total_area:.1f} < {min_area:.1f}, "
                      f"{len(indices)} faces)")
            return None

    # Centroid
    n_pts = len(all_pts)
    cx = sum(p[0] for p in all_pts) / n_pts
    cy = sum(p[1] for p in all_pts) / n_pts
    cz = sum(p[2] for p in all_pts) / n_pts

    # Multi-lobe detection: check if point cloud has multiple distinct clusters
    # Use simple distance-based clustering (no sklearn/scipy)
    if len(all_pts) > 20:
        # Compute rough diameter
        dx_total = max(xs) - min(xs)
        dy_total = max(ys) - min(ys)
        dz_total = max(zs) - min(zs)
        diameter = math.sqrt(dx_total**2 + dy_total**2 + dz_total**2)
        cluster_threshold = diameter * 0.3  # Points closer than 30% of diameter belong together

        # Simple greedy clustering
        clusters = []
        for pt in all_pts:
            # Find closest cluster
            best_cluster = None
            best_dist = float('inf')
            for cluster in clusters:
                # Distance to cluster centroid
                cc = cluster['centroid']
                dist = math.sqrt((pt[0]-cc[0])**2 + (pt[1]-cc[1])**2 + (pt[2]-cc[2])**2)
                if dist < best_dist:
                    best_dist = dist
                    best_cluster = cluster

            if best_cluster is not None and best_dist < cluster_threshold:
                # Add to existing cluster
                best_cluster['points'].append(pt)
                # Update centroid incrementally
                n = len(best_cluster['points'])
                cc = best_cluster['centroid']
                best_cluster['centroid'] = (
                    (cc[0] * (n-1) + pt[0]) / n,
                    (cc[1] * (n-1) + pt[1]) / n,
                    (cc[2] * (n-1) + pt[2]) / n,
                )
            else:
                # Create new cluster
                clusters.append({
                    'points': [pt],
                    'centroid': pt
                })

        # If we found multiple significant clusters, this is a multi-lobe feature
        significant_clusters = [c for c in clusters if len(c['points']) >= 5]
        if len(significant_clusters) >= 2 and verbose:
            print(f"    BSpline→Multi-lobe detected ({len(significant_clusters)} clusters, "
                  f"using largest cluster only)")
            # Use the largest cluster only
            all_pts = max(significant_clusters, key=lambda c: len(c['points']))['points']
            # Recompute centroid
            n_pts = len(all_pts)
            cx = sum(p[0] for p in all_pts) / n_pts
            cy = sum(p[1] for p in all_pts) / n_pts
            cz = sum(p[2] for p in all_pts) / n_pts
            xs = [p[0] for p in all_pts]
            ys = [p[1] for p in all_pts]
            zs = [p[2] for p in all_pts]

    # PCA via covariance matrix
    cov = [[0.0]*3 for _ in range(3)]
    for p in all_pts:
        d = [p[0]-cx, p[1]-cy, p[2]-cz]
        for i in range(3):
            for j in range(3):
                cov[i][j] += d[i] * d[j]

    # Power iteration for eigenvectors (3 passes with deflation)
    axes = []
    for _ in range(3):
        v = [1.0, 0.5, 0.25]
        for _it in range(60):
            nv = [sum(cov[i][j]*v[j] for j in range(3)) for i in range(3)]
            mag = math.sqrt(sum(x**2 for x in nv))
            if mag < 1e-15:
                break
            v = [x/mag for x in nv]
        axes.append(tuple(v))
        # Deflate
        eigenval = sum(sum(cov[i][j]*v[j] for j in range(3))*v[i]
                       for i in range(3))
        for i in range(3):
            for j in range(3):
                cov[i][j] -= eigenval * v[i] * v[j]

    # Project points onto PCA axes for extents
    projs = [[], [], []]
    radials_0 = []  # radial distances from axis 0
    for p in all_pts:
        d = [p[0]-cx, p[1]-cy, p[2]-cz]
        for k in range(3):
            projs[k].append(sum(d[j]*axes[k][j] for j in range(3)))
        rx = d[0] - projs[0][-1]*axes[0][0]
        ry = d[1] - projs[0][-1]*axes[0][1]
        rz = d[2] - projs[0][-1]*axes[0][2]
        radials_0.append(math.sqrt(rx**2 + ry**2 + rz**2))

    extents = [max(p) - min(p) for p in projs]
    for k in range(3):
        extents[k] = max(extents[k], 1e-6)

    # Convex hull volume estimation (quick 2D projection method)
    # Project onto dominant plane (plane with max 2D extent) and estimate volume
    obb_vol = extents[0] * extents[1] * extents[2]
    convex_hull_vol_estimate = obb_vol  # Default: assume OBB volume

    if len(all_pts) >= 10:
        # Project onto XY plane of PCA frame
        pts_2d = [(projs[0][i], projs[1][i]) for i in range(len(all_pts))]

        # Quick convex hull via Graham scan
        def _cross_2d(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        # Sort points by x, then y
        pts_2d_sorted = sorted(pts_2d)
        if len(pts_2d_sorted) >= 3:
            # Build lower hull
            lower = []
            for p in pts_2d_sorted:
                while len(lower) >= 2 and _cross_2d(lower[-2], lower[-1], p) <= 0:
                    lower.pop()
                lower.append(p)
            # Build upper hull
            upper = []
            for p in reversed(pts_2d_sorted):
                while len(upper) >= 2 and _cross_2d(upper[-2], upper[-1], p) <= 0:
                    upper.pop()
                upper.append(p)
            # Combine
            hull_2d = lower[:-1] + upper[:-1]

            # Compute 2D hull area via shoelace formula
            if len(hull_2d) >= 3:
                area_2d = 0.0
                for i in range(len(hull_2d)):
                    j = (i + 1) % len(hull_2d)
                    area_2d += hull_2d[i][0] * hull_2d[j][1]
                    area_2d -= hull_2d[j][0] * hull_2d[i][1]
                area_2d = abs(area_2d) / 2.0

                # Estimate 3D volume: 2D hull area × extent in 3rd dimension
                convex_hull_vol_estimate = area_2d * extents[2]

                # If convex hull is significantly smaller, shrink extents proportionally
                if convex_hull_vol_estimate < 0.6 * obb_vol and convex_hull_vol_estimate > 1e-6:
                    shrink_factor = (convex_hull_vol_estimate / obb_vol) ** (1.0/3.0)
                    # Shrink all extents by the factor (more aggressive)
                    for k in range(3):
                        extents[k] *= max(shrink_factor, 0.4)
                    if verbose:
                        print(f"    BSpline→Tightened (hull {convex_hull_vol_estimate:.0f} "
                              f"vs OBB {obb_vol:.0f}, factor {shrink_factor:.2f})")
                    obb_vol = extents[0] * extents[1] * extents[2]

    # Check if cylinder is a better fit (elongated + radially uniform)
    h0 = extents[0]
    r_max = max(radials_0) if radials_0 else 1e-6
    aspect = h0 / max(2 * r_max, 1e-6)

    r_mean = sum(radials_0) / max(len(radials_0), 1)
    r_var = sum((r - r_mean)**2 for r in radials_0) / max(len(radials_0), 1)
    r_cv = math.sqrt(r_var) / max(r_mean, 1e-6) if r_mean > 1e-6 else 1.0

    # Estimate thickness: for thin shells, area >> volume^(2/3)
    raw_vol = extents[0] * extents[1] * extents[2]
    area_to_vol = total_area / max(raw_vol ** (2.0/3.0), 1e-6)
    is_thin_shell = area_to_vol > 6.0  # sphere has ratio ~4.8

    if is_thin_shell:
        # Thin shell: fit a tighter prism along the midplane
        # Find thinnest dimension
        thin_idx = min(range(3), key=lambda k: extents[k])

        # Estimate actual shell thickness from surface area and 2D extent
        # For a thin shell: area ≈ 2 × base_area, volume ≈ base_area × thickness
        base_area_est = total_area / 2.0
        thickness_est = obb_vol / max(base_area_est, 1e-6)

        # Clamp thickness to reasonable range (avoid degenerate or oversized)
        thickness_est = max(min(thickness_est, extents[thin_idx]), 0.5)

        # Instead of PCA thin-slab, fit midplane more tightly
        # Project points onto plane perpendicular to thin axis
        midplane_pts = []
        for i, p in enumerate(all_pts):
            # Get projection on the two thick axes
            coords = [projs[0][i], projs[1][i], projs[2][i]]
            # Remove thin component
            midplane_pt = [coords[k] if k != thin_idx else 0.0 for k in range(3)]
            midplane_pts.append(tuple(midplane_pt))

        # Find tight bounds in midplane
        thick_dims = [k for k in range(3) if k != thin_idx]
        for k in thick_dims:
            proj_vals = [midplane_pts[i][k] for i in range(len(midplane_pts))]
            extent_tight = max(proj_vals) - min(proj_vals)
            # Shrink extent by 10% to get tighter fit
            extents[k] = max(extent_tight * 0.95, extents[k] * 0.8)

        extents[thin_idx] = thickness_est

        if verbose:
            print(f"    BSpline→Thin shell prism (thickness={thickness_est:.2f})")

    # Compute minimum prism dimension: 1% of model characteristic length,
    # floor at 1.0 absolute. Prevents degenerate zero-thickness prisms.
    min_prism_dim = 1.0
    if model_volume > 0:
        char_length = model_volume ** (1.0 / 3.0)
        min_prism_dim = max(0.01 * char_length, 1.0)
    elif model_bbox is not None:
        mx = model_bbox[3] - model_bbox[0]
        my = model_bbox[4] - model_bbox[1]
        mz = model_bbox[5] - model_bbox[2]
        char_length = max(mx, my, mz)
        min_prism_dim = max(0.01 * char_length, 1.0)

    # Enforce minimum on all extents
    for k in range(3):
        extents[k] = max(extents[k], min_prism_dim)

    # Always use rectangular prisms for BSpline — they tile better
    # and give a more predictable simplification than cylinders.

    # Oriented bounding box from PCA axes
    mins = [min(p) for p in projs]
    ox = cx + sum(mins[k]*axes[k][0] for k in range(3))
    oy = cy + sum(mins[k]*axes[k][1] for k in range(3))
    oz = cz + sum(mins[k]*axes[k][2] for k in range(3))

    # Check orthogonality of PCA axes
    dots = [abs(sum(axes[0][j]*axes[1][j] for j in range(3))),
            abs(sum(axes[0][j]*axes[2][j] for j in range(3))),
            abs(sum(axes[1][j]*axes[2][j] for j in range(3)))]
    is_ortho = all(d < 0.15 for d in dots)

    # Volume check: keep 1.2x model volume hard limit, plus budget-based checks
    final_vol = extents[0] * extents[1] * extents[2]
    if model_volume > 0 and final_vol > 1.2 * model_volume:
        if verbose:
            print(f"    BSpline→SKIP (vol {final_vol:.0f} >> "
                  f"model {model_volume:.0f}, {len(indices)} faces)")
        return None

    # Budget-based shrinking: if primitive exceeds 3x its area-share budget,
    # shrink extents to fit within budget. If >10x, skip entirely.
    if volume_budget > 1e-6 and final_vol > 3.0 * volume_budget:
        if final_vol > 10.0 * volume_budget:
            if verbose:
                print(f"    BSpline→SKIP (vol {final_vol:.0f} >> "
                      f"10x budget {volume_budget:.0f}, {len(indices)} faces)")
            return None
        # Shrink extents uniformly to fit within 3x budget
        target_vol = 3.0 * volume_budget
        shrink = (target_vol / final_vol) ** (1.0 / 3.0)
        for k in range(3):
            extents[k] *= shrink
        # Re-enforce minimums after shrink
        for k in range(3):
            extents[k] = max(extents[k], min_prism_dim)
        if verbose:
            print(f"    BSpline→Prism shrunk (factor={shrink:.3f}, "
                  f"vol {final_vol:.0f}→{extents[0]*extents[1]*extents[2]:.0f}, "
                  f"budget={volume_budget:.0f})")
        # Recompute mins/projections for OBB origin with shrunken extents
        # Keep same PCA axes, just tighten toward centroid
        for k in range(3):
            mid = (min(projs[k]) + max(projs[k])) / 2.0
            half = extents[k] / 2.0
            projs[k] = [mid - half, mid + half]
        mins = [min(p) for p in projs]
        ox = cx + sum(mins[k]*axes[k][0] for k in range(3))
        oy = cy + sum(mins[k]*axes[k][1] for k in range(3))
        oz = cz + sum(mins[k]*axes[k][2] for k in range(3))

    if is_ortho:
        try:
            local_ax2 = gp_Ax2(
                gp_Pnt(ox, oy, oz),
                gp_Dir(*axes[2]),
                gp_Dir(*axes[0]),
            )
            trsf = gp_Trsf()
            trsf.SetTransformation(local_ax2)
            trsf.Invert()
            box = BRepPrimAPI_MakeBox(
                extents[0], extents[1], extents[2]).Shape()
            transformer = BRepBuilderAPI_Transform(box, trsf, True)
            transformer.Build()
            if transformer.IsDone():
                if verbose:
                    print(f"    BSpline→Prism ({extents[0]:.1f}x{extents[1]:.1f}"
                          f"x{extents[2]:.1f}, min={min_prism_dim:.1f}, "
                          f"{len(indices)} faces)")
                return transformer.Shape()
        except Exception:
            pass

    # Fallback: tight AABB with minimum prism dimension
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    zs = [p[2] for p in all_pts]
    dx = max(max(xs) - min(xs), min_prism_dim)
    dy = max(max(ys) - min(ys), min_prism_dim)
    dz = max(max(zs) - min(zs), min_prism_dim)

    # Volume check for fallback AABB (keep 1.2x limit + budget check)
    aabb_vol = dx * dy * dz
    if model_volume > 0 and aabb_vol > 1.2 * model_volume:
        if verbose:
            print(f"    BSpline→Box SKIP (vol {aabb_vol:.0f} >> "
                  f"model {model_volume:.0f}, {len(indices)} faces)")
        return None

    # Budget-based shrink for AABB fallback too
    if volume_budget > 1e-6 and aabb_vol > 3.0 * volume_budget:
        if aabb_vol > 10.0 * volume_budget:
            if verbose:
                print(f"    BSpline→Box SKIP (vol {aabb_vol:.0f} >> "
                      f"10x budget {volume_budget:.0f}, {len(indices)} faces)")
            return None
        # Shrink AABB toward centroid
        target_vol = 3.0 * volume_budget
        shrink = (target_vol / aabb_vol) ** (1.0 / 3.0)
        mid_x = (min(xs) + max(xs)) / 2.0
        mid_y = (min(ys) + max(ys)) / 2.0
        mid_z = (min(zs) + max(zs)) / 2.0
        dx = max(dx * shrink, min_prism_dim)
        dy = max(dy * shrink, min_prism_dim)
        dz = max(dz * shrink, min_prism_dim)
        bx = mid_x - dx / 2.0
        by = mid_y - dy / 2.0
        bz = mid_z - dz / 2.0
        if verbose:
            print(f"    BSpline→Prism shrunk ({dx:.1f}x{dy:.1f}x{dz:.1f}, "
                  f"{len(indices)} faces)")
        return BRepPrimAPI_MakeBox(gp_Pnt(bx, by, bz), dx, dy, dz).Shape()

    if verbose:
        print(f"    BSpline→Prism ({dx:.1f}x{dy:.1f}x{dz:.1f}, "
              f"min={min_prism_dim:.1f}, {len(indices)} faces)")
    return BRepPrimAPI_MakeBox(
        gp_Pnt(min(xs), min(ys), min(zs)), dx, dy, dz).Shape()


def _make_primitive_for_group(
    face_infos: List[dict],
    indices: List[int],
    verbose: bool = False,
    model_bbox: Tuple = None,
    model_volume: float = 0.0,
    total_model_area: float = 0.0,
    lenient: bool = False,
) -> Optional[object]:
    """Create a best-fit primitive for a face group.

    Dispatches to the appropriate constructor based on the dominant surface
    type in the group. Falls back to bounding box for groups with only
    BSpline/OTHER faces.

    When lenient=True (original-shape fallback), BSpline faces are
    tessellated into many small prisms and volume sanity limits are relaxed.
    """
    dtype = _dominant_type(face_infos, indices)

    if dtype == SURF_CYLINDER:
        prim = _make_cylinder_from_faces(face_infos, indices, model_volume)
        if prim is not None:
            if verbose:
                ref = [i for i in indices
                       if face_infos[i]["type"] == SURF_CYLINDER]
                r = face_infos[ref[0]]["radius"] if ref else "?"
                print(f"    Cylinder (r={r:.3f}, {len(indices)} faces)")
            return prim

    elif dtype == SURF_CONE:
        prim = _make_cone_from_faces(face_infos, indices, model_volume)
        if prim is not None:
            if verbose:
                print(f"    Cone ({len(indices)} faces)")
            return prim

    elif dtype == SURF_SPHERE:
        prim = _make_sphere_from_faces(face_infos, indices)
        if prim is not None:
            if verbose:
                ref = [i for i in indices
                       if face_infos[i]["type"] == SURF_SPHERE]
                r = face_infos[ref[0]]["radius"] if ref else "?"
                print(f"    Sphere (r={r:.3f}, {len(indices)} faces)")
            return prim

    elif dtype == SURF_TORUS:
        prim = _make_torus_from_faces(face_infos, indices, model_volume)
        if prim is not None:
            if verbose:
                print(f"    Torus ({len(indices)} faces)")
            return prim

    # Plane-dominant groups: oriented bounding box
    if dtype == SURF_PLANE:
        prim = _make_box_for_planar_group(
            face_infos, indices, verbose,
            model_bbox=model_bbox, model_volume=model_volume)
        if prim is not None:
            return prim

    # BSpline/OTHER groups
    if dtype == SURF_OTHER:
        if lenient:
            return _tessellate_bspline_faces(
                face_infos, indices, verbose,
                model_volume=model_volume)
        return _make_enclosing_for_bspline(
            face_infos, indices, verbose,
            model_bbox=model_bbox, model_volume=model_volume,
            total_model_area=total_model_area)

    # Ultimate fallback for any remaining type: tight boundary-point AABB
    all_pts = []
    for i in indices:
        pts = _sample_face_boundary_points(face_infos[i]["face"], n_pts=10)
        all_pts.extend(pts)
        all_pts.append(face_infos[i]["centroid"])
    if all_pts:
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        zs = [p[2] for p in all_pts]
        bb = (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))
    else:
        bb = _group_bbox(face_infos, indices)
    dx = bb[3]-bb[0]
    dy = bb[4]-bb[1]
    dz = bb[5]-bb[2]
    box_vol = max(dx, 1e-6) * max(dy, 1e-6) * max(dz, 1e-6)
    if model_volume > 0 and box_vol > 2.0 * model_volume:
        if verbose:
            print(f"    Box fallback SKIP (vol {box_vol:.0f} >> "
                  f"model {model_volume:.0f}, {len(indices)} faces)")
        return None
    if verbose:
        print(f"    Box fallback ({dx:.3f}x{dy:.3f}x{dz:.3f}, "
              f"{len(indices)} faces)")
    return _make_box(*bb)


def _make_box_for_planar_group(
    face_infos: List[dict], indices: List[int], verbose: bool = False,
    model_bbox: Tuple = None, model_volume: float = 0.0,
    lenient: bool = False,
):
    """Build an oriented box from a group of planar faces.

    Strategy: sample boundary points from all faces in the group, then
    find up to 3 orthogonal normal directions. Project all points onto
    each axis to get exact extents. If 3 orthogonal axes are found, build
    an oriented box via gp_Ax2. Otherwise fall back to AABB.

    Groups whose bounding box spans >50% of the model in 2+ dimensions
    are skipped (returns None) — they're likely cap faces from unrelated
    features incorrectly grouped together.
    """
    from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2, gp_Trsf
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform

    plane_faces = [i for i in indices if face_infos[i]["type"] == SURF_PLANE]

    # Collect unique normal directions from planar faces
    normals = []
    for i in plane_faces:
        n = face_infos[i]["normal"]
        # Canonicalize direction (pick consistent sign)
        if n[0] < -1e-6 or (abs(n[0]) < 1e-6 and n[1] < -1e-6) or \
           (abs(n[0]) < 1e-6 and abs(n[1]) < 1e-6 and n[2] < -1e-6):
            n = (-n[0], -n[1], -n[2])
        found = False
        for existing in normals:
            if _vec_close(n, existing):
                found = True
                break
        if not found:
            normals.append(n)

    # Sample boundary points from ALL faces in the group
    all_points = []
    for i in indices:
        pts = _sample_face_boundary_points(face_infos[i]["face"], n_pts=10)
        all_points.extend(pts)
        all_points.append(face_infos[i]["centroid"])

    # Skip plane groups that span too much of the model — likely
    # unrelated cap faces grouped together after BSpline removal.
    if model_bbox is not None and all_points:
        mx = model_bbox[3] - model_bbox[0]
        my = model_bbox[4] - model_bbox[1]
        mz = model_bbox[5] - model_bbox[2]
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        zs = [p[2] for p in all_points]
        gx, gy, gz = max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)
        model_exts = [mx, my, mz]
        group_exts = [gx, gy, gz]
        large_dims = sum(1 for k in range(3)
                         if model_exts[k] > 1e-6
                         and group_exts[k] > 0.5 * model_exts[k])
        box_vol = max(gx, 1e-6) * max(gy, 1e-6) * max(gz, 1e-6)
        if large_dims >= 2 and not lenient:
            if model_volume > 0 and box_vol > 2.0 * model_volume:
                if verbose:
                    print(f"    Plane group SKIP (spans {large_dims}/3 dims, "
                          f"box vol {box_vol:.0f} >> model vol {model_volume:.0f}, "
                          f"{len(indices)} faces)")
                return None

    if len(normals) >= 3 and len(all_points) > 0:
        # Check if first 3 normals are mutually orthogonal
        ax0, ax1, ax2 = normals[0], normals[1], normals[2]
        dot01 = abs(ax0[0]*ax1[0] + ax0[1]*ax1[1] + ax0[2]*ax1[2])
        dot02 = abs(ax0[0]*ax2[0] + ax0[1]*ax2[1] + ax0[2]*ax2[2])
        dot12 = abs(ax1[0]*ax2[0] + ax1[1]*ax2[1] + ax1[2]*ax2[2])

        if dot01 < 0.1 and dot02 < 0.1 and dot12 < 0.1:
            # Project all points onto each axis
            projs = [[], [], []]
            for pt in all_points:
                for k, ax in enumerate([ax0, ax1, ax2]):
                    projs[k].append(
                        pt[0]*ax[0] + pt[1]*ax[1] + pt[2]*ax[2])

            d0_min, d0_max = min(projs[0]), max(projs[0])
            d1_min, d1_max = min(projs[1]), max(projs[1])
            d2_min, d2_max = min(projs[2]), max(projs[2])

            dx = max(d0_max - d0_min, 1e-6)
            dy = max(d1_max - d1_min, 1e-6)
            dz = max(d2_max - d2_min, 1e-6)

            obb_vol = dx * dy * dz
            vol_limit = 20.0 if lenient else 3.0
            if model_volume > 0 and obb_vol > vol_limit * model_volume:
                if verbose:
                    print(f"    Box oriented SKIP (vol {obb_vol:.0f} >> "
                          f"model {model_volume:.0f}, {len(indices)} faces)")
                return None

            # Origin in world coords = sum of min-projections along each axis
            ox = d0_min*ax0[0] + d1_min*ax1[0] + d2_min*ax2[0]
            oy = d0_min*ax0[1] + d1_min*ax1[1] + d2_min*ax2[1]
            oz = d0_min*ax0[2] + d1_min*ax1[2] + d2_min*ax2[2]

            try:
                local_ax2 = gp_Ax2(
                    gp_Pnt(ox, oy, oz),
                    gp_Dir(*ax2),
                    gp_Dir(*ax0),
                )
                trsf = gp_Trsf()
                trsf.SetTransformation(local_ax2)
                trsf.Invert()

                box = BRepPrimAPI_MakeBox(dx, dy, dz).Shape()
                transformer = BRepBuilderAPI_Transform(box, trsf, True)
                transformer.Build()
                if transformer.IsDone():
                    if verbose:
                        print(f"    Box oriented "
                              f"({dx:.3f}x{dy:.3f}x{dz:.3f}, "
                              f"{len(indices)} faces)")
                    return transformer.Shape()
            except Exception:
                pass

    # Fallback: axis-aligned bounding box from sampled points
    if all_points:
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        zs = [p[2] for p in all_points]
        bb = (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))
    else:
        bb = _group_bbox(face_infos, indices)

    dx = bb[3]-bb[0]
    dy = bb[4]-bb[1]
    dz = bb[5]-bb[2]
    aabb_vol = max(dx, 1e-6) * max(dy, 1e-6) * max(dz, 1e-6)
    vol_limit = 20.0 if lenient else 3.0
    if model_volume > 0 and aabb_vol > vol_limit * model_volume:
        if verbose:
            print(f"    Box AABB SKIP (vol {aabb_vol:.0f} >> "
                  f"model {model_volume:.0f}, {len(indices)} faces)")
        return None

    if verbose:
        print(f"    Box AABB ({dx:.3f}x{dy:.3f}x{dz:.3f}, "
              f"{len(indices)} faces)")
    return _make_box(*bb)


# ---------------------------------------------------------------------------
# Boolean fusion
# ---------------------------------------------------------------------------

def _fuse_shapes(shapes: list, verbose: bool = False):
    """Fuse a list of shapes into a single compound/solid via boolean union."""
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.TopoDS import TopoDS_Compound
    from OCC.Core.BRep import BRep_Builder

    if not shapes:
        return None
    if len(shapes) == 1:
        return shapes[0]

    # Sort by volume (largest first) for more stable fuse order
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop

    def _volume(s):
        props = GProp_GProps()
        try:
            brepgprop.VolumeProperties(s, props)
            return abs(props.Mass())
        except Exception:
            return 0.0

    sorted_shapes = sorted(shapes, key=_volume, reverse=True)

    result = sorted_shapes[0]
    fused_count = 1

    for i, s in enumerate(sorted_shapes[1:], 1):
        try:
            fuser = BRepAlgoAPI_Fuse(result, s)
            fuser.SetRunParallel(True)
            fuser.Build()
            if fuser.IsDone():
                candidate = fuser.Shape()
                analyzer = BRepCheck_Analyzer(candidate)
                if analyzer.IsValid():
                    result = candidate
                    fused_count += 1
                    continue
                # Invalid but still usable — try healing
                result = candidate
                fused_count += 1
            else:
                if verbose:
                    print(f"    Fuse failed for primitive {i}, "
                          f"adding as compound member")
        except Exception as e:
            if verbose:
                print(f"    Fuse error for primitive {i}: {e}")

    if fused_count < len(sorted_shapes):
        # Some shapes couldn't be fused — assemble as compound
        if verbose:
            unfused = len(sorted_shapes) - fused_count
            print(f"    {unfused} primitives added as compound members")
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        builder.Add(compound, result)
        for s in sorted_shapes[fused_count:]:
            builder.Add(compound, s)
        return compound

    return result


def _heal_result(shape, verbose: bool = False):
    """Light healing pass on the final fused result."""
    from OCC.Core.ShapeFix import ShapeFix_Shape
    from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    from OCC.Core.BRepCheck import BRepCheck_Analyzer

    analyzer = BRepCheck_Analyzer(shape)
    if not analyzer.IsValid():
        fixer = ShapeFix_Shape(shape)
        fixer.SetPrecision(1e-6)
        fixer.Perform()
        shape = fixer.Shape()
        if verbose:
            analyzer2 = BRepCheck_Analyzer(shape)
            print(f"  Post-heal valid: {analyzer2.IsValid()}")

    try:
        unifier = ShapeUpgrade_UnifySameDomain(shape, True, True, True)
        unifier.Build()
        shape = unifier.Shape()
    except Exception:
        pass

    return shape


# ---------------------------------------------------------------------------
# Multi-solid support
# ---------------------------------------------------------------------------

def _get_solids(shape) -> list:
    """Extract individual solids from a shape (handles assemblies/compounds)."""
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_SOLID
    from OCC.Core.TopoDS import topods

    solids = []
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        solids.append(topods.Solid(exp.Current()))
        exp.Next()
    return solids


def _fit_single_solid(solid, verbose: bool, tol: float, lenient: bool = False):
    """Fit primitives to a single solid body. Returns (shapes_list, types_list)."""
    faces = _get_faces(solid)
    if not faces:
        return [], []

    face_infos = [_analyze_face(f) for f in faces]
    adjacency = _build_face_adjacency(solid, faces)
    groups = _group_faces_into_primitives(face_infos, adjacency, tol=tol)

    # Compute model bbox, volume, and total area for sanity checks
    mbbox = _group_bbox(face_infos, list(range(len(face_infos))))
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop
    gprops = GProp_GProps()
    try:
        brepgprop.VolumeProperties(solid, gprops)
        mvol = abs(gprops.Mass())
    except Exception:
        mvol = 0.0

    # Total model surface area for budget-based BSpline fitting
    total_area = sum(fi["area"] for fi in face_infos)

    type_names = {
        SURF_PLANE: "plane", SURF_CYLINDER: "cylinder", SURF_CONE: "cone",
        SURF_SPHERE: "sphere", SURF_TORUS: "torus", SURF_OTHER: "other",
    }

    primitives = []
    prim_types = []
    for group in groups:
        dtype = _dominant_type(face_infos, group)
        prim = _make_primitive_for_group(
            face_infos, group, verbose=verbose,
            model_bbox=mbbox, model_volume=mvol,
            total_model_area=total_area, lenient=lenient)
        if prim is not None:
            primitives.append(prim)
            prim_types.append(type_names.get(dtype, "box"))

    return primitives, prim_types


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def fit_primitives(
    shape,
    verbose: bool = False,
    tol: float = 0.1,
    lenient: bool = False,
) -> Tuple[object, dict]:
    """Decompose a B-rep shape into best-fit geometric primitives.

    For multi-solid assemblies, each solid is processed independently and
    the results are assembled as a compound (not fused), preserving
    the original structure and avoiding volume distortion from boolean
    operations between unrelated solids.

    Args:
        shape: Input TopoDS_Shape (single solid or assembly).
        verbose: Print progress details.
        tol: Geometric tolerance for grouping faces.
        lenient: Relax volume limits and tessellate BSplines (for
                 original-shape fallback when defeaturing fails).

    Returns:
        (result_shape, report_dict)
    """
    from OCC.Core.TopoDS import TopoDS_Compound
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop

    t0 = time.time()

    # Detect multi-solid assemblies
    solids = _get_solids(shape)
    n_solids = len(solids)

    faces = _get_faces(shape)
    n_faces = len(faces)

    type_names = {
        SURF_PLANE: "plane", SURF_CYLINDER: "cylinder", SURF_CONE: "cone",
        SURF_SPHERE: "sphere", SURF_TORUS: "torus", SURF_OTHER: "other",
    }

    if verbose:
        print(f"  Analyzing {n_faces} faces across {n_solids} solid(s)...")

    # Count surface types across entire shape
    face_infos_all = [_analyze_face(f) for f in faces]
    type_counts = defaultdict(int)
    for fi in face_infos_all:
        type_counts[fi["type"]] += 1
    if verbose:
        for t, c in sorted(type_counts.items()):
            print(f"    {type_names.get(t, 'unknown')}: {c}")

    # Process each solid independently
    all_result_shapes = []
    all_primitive_types = []

    if n_solids <= 1:
        # Single solid — process the whole shape directly
        face_infos = face_infos_all
        adjacency = _build_face_adjacency(shape, faces)
        groups = _group_faces_into_primitives(face_infos, adjacency, tol=tol)
        mbbox = _group_bbox(face_infos, list(range(len(face_infos))))

        gprops = GProp_GProps()
        try:
            brepgprop.VolumeProperties(shape, gprops)
            mvol = abs(gprops.Mass())
        except Exception:
            mvol = 0.0

        # Total model surface area for budget-based BSpline fitting
        total_area = sum(fi["area"] for fi in face_infos)

        if verbose:
            print(f"  Grouped into {len(groups)} primitive candidates")

        for group in groups:
            dtype = _dominant_type(face_infos, group)
            prim = _make_primitive_for_group(
                face_infos, group, verbose=verbose,
                model_bbox=mbbox, model_volume=mvol,
                total_model_area=total_area, lenient=lenient)
            if prim is not None:
                all_result_shapes.append(prim)
                all_primitive_types.append(type_names.get(dtype, "box"))
    else:
        # Multi-solid assembly — process each solid independently
        for si, solid in enumerate(solids):
            solid_faces = _get_faces(solid)
            if not solid_faces:
                continue
            if verbose and n_solids > 3:
                print(f"  Solid {si+1}/{n_solids} ({len(solid_faces)} faces)...")

            shapes, types = _fit_single_solid(solid, verbose=verbose, tol=tol,
                                              lenient=lenient)
            all_result_shapes.extend(shapes)
            all_primitive_types.extend(types)

    if not all_result_shapes:
        return shape, {"status": "no_primitives", "elapsed_s": 0}

    if verbose:
        print(f"  Constructed {len(all_primitive_types)} primitives total, "
              f"assembling...")

    # Assemble all primitives as compound — boolean fusion often destroys
    # volume when primitives overlap, so we preserve each primitive intact.
    if len(all_result_shapes) == 1:
        result = all_result_shapes[0]
    else:
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        for s in all_result_shapes:
            builder.Add(compound, s)
        result = compound

    # --- Volume calibration step ---
    # Compute original shape volume and total primitive volume.
    # If significantly different (ratio outside 0.8-1.3), apply a uniform
    # scale transform to match the target volume.
    from OCC.Core.gp import gp_Pnt, gp_Trsf
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform

    orig_props = GProp_GProps()
    try:
        brepgprop.VolumeProperties(shape, orig_props)
        original_vol = abs(orig_props.Mass())
    except Exception:
        original_vol = 0.0

    prim_props = GProp_GProps()
    try:
        brepgprop.VolumeProperties(result, prim_props)
        primitive_vol = abs(prim_props.Mass())
    except Exception:
        primitive_vol = 0.0

    volume_ratio = primitive_vol / max(original_vol, 1e-12)
    scale_applied = 1.0

    if original_vol > 1e-12 and primitive_vol > 1e-12:
        if volume_ratio < 0.8 or volume_ratio > 1.3:
            scale_factor = (original_vol / primitive_vol) ** (1.0 / 3.0)
            # Only scale if factor is within a safe range (0.2 to 2.0)
            # This handles cases where primitives are up to ~125x too
            # large or 8x too small (0.2^3=0.008, 2.0^3=8.0)
            if 0.2 <= scale_factor <= 2.0:
                # Get center of mass for scale center
                center = prim_props.CentreOfMass()
                trsf_scale = gp_Trsf()
                trsf_scale.SetScale(
                    gp_Pnt(center.X(), center.Y(), center.Z()),
                    scale_factor,
                )
                transformer = BRepBuilderAPI_Transform(result, trsf_scale, True)
                transformer.Build()
                if transformer.IsDone():
                    result = transformer.Shape()
                    scale_applied = scale_factor
                    # Recompute volume after scaling
                    scaled_props = GProp_GProps()
                    try:
                        brepgprop.VolumeProperties(result, scaled_props)
                        primitive_vol = abs(scaled_props.Mass())
                        volume_ratio = primitive_vol / max(original_vol, 1e-12)
                    except Exception:
                        pass
                    if verbose:
                        print(f"  Volume calibration: scale={scale_factor:.4f}, "
                              f"new ratio={volume_ratio:.4f}")
                else:
                    if verbose:
                        print(f"  Volume calibration: transform failed, "
                              f"keeping original")
            else:
                if verbose:
                    print(f"  Volume calibration: scale {scale_factor:.4f} "
                          f"outside safe range [0.5, 1.5], skipping")

    if verbose:
        print(f"  Volume: original={original_vol:.2f}, "
              f"primitives={primitive_vol:.2f}, ratio={volume_ratio:.4f}")

    # Heal
    result = _heal_result(result, verbose=verbose)

    elapsed = time.time() - t0

    # Validate
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    analyzer = BRepCheck_Analyzer(result)
    is_valid = analyzer.IsValid()

    report = {
        "status": "success",
        "input_faces": n_faces,
        "num_solids": n_solids,
        "face_types": {type_names.get(k, "unknown"): v
                       for k, v in type_counts.items()},
        "num_groups": len(all_primitive_types),
        "num_primitives": len(all_primitive_types),
        "primitive_types": all_primitive_types,
        "valid": is_valid,
        "elapsed_s": round(elapsed, 2),
        "volume_ratio": round(volume_ratio, 4),
        "scale_applied": round(scale_applied, 4),
    }

    if verbose:
        print(f"  Result: {len(all_primitive_types)} primitives, "
              f"valid={is_valid}, vol_ratio={volume_ratio:.4f}, "
              f"{elapsed:.2f}s")

    return result, report


def fit_primitives_from_step(
    step_path: str,
    output_path: str,
    verbose: bool = False,
    tol: float = 0.1,
    lenient: bool = False,
) -> dict:
    """End-to-end: read STEP → fit primitives → write STEP."""
    shape = _read_step(step_path)
    if shape is None:
        return {"status": "error", "message": f"Failed to read {step_path}"}

    result_shape, report = fit_primitives(shape, verbose=verbose, tol=tol,
                                          lenient=lenient)
    report["input"] = str(step_path)
    report["output"] = str(output_path)

    if report["status"] != "success":
        return report

    if not _write_step(result_shape, output_path):
        report["status"] = "write_error"
        report["message"] = f"Failed to write {output_path}"
        return report

    return report


# ---------------------------------------------------------------------------
# Full pipeline: defeature → fit primitives
# ---------------------------------------------------------------------------

def defeature_and_fit(
    step_path: str,
    output_dir: str,
    checkpoint: str = None,
    seg_path: str = None,
    verbose: bool = False,
    max_fuzzy: float = 1e-3,
    tol: float = 0.1,
    save_colored: bool = False,
    save_intermediate: bool = True,
) -> dict:
    """Run the full pipeline: inference → defeature → primitive fitting.

    Args:
        step_path: Input STEP file.
        output_dir: Directory for all outputs.
        checkpoint: Model checkpoint (None = default).
        seg_path: Pre-computed .seg file (skip inference if provided).
        verbose: Print detailed progress.
        max_fuzzy: Max fuzzy tolerance for defeaturing.
        tol: Tolerance for primitive grouping.
        save_colored: Save colored STEP showing predictions.
        save_intermediate: Save the defeatured intermediate STEP.

    Returns:
        Combined report dict.
    """
    from brepformer.defeature_v2 import (
        defeature_step, load_seg_labels,
        infer_defeature_predictions, save_colored_step,
        DEFAULT_CHECKPOINT,
    )

    model_id = Path(step_path).stem
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = checkpoint or DEFAULT_CHECKPOINT

    # --- Step 1: Get predictions ---
    if seg_path:
        preds = load_seg_labels(seg_path)
    else:
        import torch
        from brepformer.infer import load_model

        if not Path(step_path).exists():
            return {"status": "error",
                    "message": f"STEP file not found: {step_path}"}

        if verbose:
            print(f"Loading model from {ckpt}...")

        if not Path(ckpt).exists():
            return {"status": "error",
                    "message": f"Checkpoint not found: {ckpt}"}

        model = load_model(ckpt)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        preds = infer_defeature_predictions(model, device, step_path)

    if preds is None:
        return {"status": "error",
                "message": "Failed to get predictions (check stderr for details)"}

    # --- Step 2: Defeature ---
    defeatured_path = str(out_dir / f"{model_id}_defeatured.step")
    if verbose:
        print(f"\nDefeaturing {step_path}...")

    defeature_result = defeature_step(
        step_path, preds, defeatured_path,
        verbose=verbose, max_fuzzy=max_fuzzy,
    )

    if defeature_result["status"] not in ("success", "no_features"):
        return {"status": "error",
                "message": f"Defeaturing failed: {defeature_result.get('message')}",
                "defeature": defeature_result}

    if not save_intermediate:
        # Still need to read the defeatured shape for primitive fitting
        pass

    if save_colored:
        colored_path = str(out_dir / f"{model_id}_colored.step")
        try:
            save_colored_step(step_path, preds, colored_path)
        except Exception as e:
            if verbose:
                print(f"  Warning: colored export failed: {e}")

    # --- Step 3: Fit primitives on defeatured shape ---
    primitives_path = str(out_dir / f"{model_id}_primitives.step")

    prim_input = defeatured_path
    use_lenient = False
    defeatured_empty = _read_step(defeatured_path) is None
    nothing_removed = defeature_result.get("removed", 0) == 0
    if defeatured_empty or nothing_removed:
        if verbose:
            reason = "empty" if defeatured_empty else "no features removed"
            print(f"\n  Defeatured shape {reason} — falling back to original shape (lenient)")
        prim_input = step_path
        use_lenient = True

    if verbose:
        print(f"\nFitting primitives to {'original' if prim_input == step_path else 'defeatured'} shape"
              f"{' (lenient)' if use_lenient else ''}...")

    prim_report = fit_primitives_from_step(
        prim_input, primitives_path,
        verbose=verbose, tol=tol, lenient=use_lenient,
    )

    if not save_intermediate:
        # Clean up intermediate defeatured file
        try:
            Path(defeatured_path).unlink()
        except Exception:
            pass

    # --- Combined report ---
    report = {
        "status": prim_report.get("status", "error"),
        "model_id": model_id,
        "input": str(step_path),
        "defeatured_output": defeatured_path if save_intermediate else None,
        "primitives_output": primitives_path,
        "defeature": defeature_result,
        "primitives": prim_report,
    }

    report_path = out_dir / f"{model_id}_primitives_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit geometric primitives to a defeatured STEP file"
    )
    parser.add_argument("--step", type=str, required=True,
                        help="Input STEP file (defeatured or raw)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output STEP path (default: {stem}_primitives.step)")
    parser.add_argument("--pipeline", action="store_true",
                        help="Run full pipeline: defeature → primitives "
                             "(input is a raw STEP with features)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint for --pipeline mode")
    parser.add_argument("--seg", type=str, default=None,
                        help="Pre-computed .seg predictions for --pipeline mode")
    parser.add_argument("--output_dir", type=str,
                        default="brepformer/primitives_output",
                        help="Output directory for --pipeline mode")
    parser.add_argument("--tol", type=float, default=0.1,
                        help="Geometric tolerance for face grouping (default: 0.1)")
    parser.add_argument("--max_fuzzy", type=float, default=1e-3,
                        help="Max fuzzy tolerance for defeaturing (default: 1e-3)")
    parser.add_argument("--save_colored", action="store_true",
                        help="Save colored STEP showing predictions (--pipeline)")
    parser.add_argument("--save_intermediate", action="store_true",
                        help="Keep the intermediate defeatured STEP (--pipeline)")
    parser.add_argument("--lenient", action="store_true",
                        help="Relax volume limits and tessellate BSplines "
                             "(use when fitting originals without defeaturing)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.pipeline:
        # Full pipeline: raw STEP → defeature → primitives
        print(f"Running full pipeline on {args.step}...")
        report = defeature_and_fit(
            step_path=args.step,
            output_dir=args.output_dir,
            checkpoint=args.checkpoint,
            seg_path=args.seg,
            verbose=args.verbose,
            max_fuzzy=args.max_fuzzy,
            tol=args.tol,
            save_colored=args.save_colored,
            save_intermediate=args.save_intermediate,
        )
        print(f"\nStatus: {report['status']}")
        if report["status"] == "success":
            print(f"Primitives output: {report['primitives_output']}")
            pr = report["primitives"]
            print(f"  {pr['num_primitives']} primitives "
                  f"({', '.join(pr['primitive_types'])})")
            print(f"  Valid: {pr['valid']}")
            print(f"  Time: {pr['elapsed_s']}s")
        else:
            print(f"Error: {report.get('message', 'unknown')}")
    else:
        # Direct mode: defeatured STEP → primitives
        step_path = args.step
        output_path = args.output
        if output_path is None:
            stem = Path(step_path).stem
            output_path = str(Path(step_path).parent / f"{stem}_primitives.step")

        print(f"Fitting primitives to {step_path}...")
        report = fit_primitives_from_step(
            step_path, output_path,
            verbose=args.verbose, tol=args.tol,
            lenient=args.lenient,
        )

        print(f"\nStatus: {report['status']}")
        if report["status"] == "success":
            print(f"Output: {report['output']}")
            print(f"  Input faces: {report['input_faces']}")
            print(f"  Face types: {report['face_types']}")
            print(f"  Primitives: {report['num_primitives']} "
                  f"({', '.join(report['primitive_types'])})")
            print(f"  Valid: {report['valid']}")
            print(f"  Time: {report['elapsed_s']}s")
        else:
            print(f"Error: {report.get('message', 'unknown')}")


if __name__ == "__main__":
    main()
