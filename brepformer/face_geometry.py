#!/usr/bin/env python3
"""Exact per-face B-Rep geometry, for describing a face to a vision model in words.

The mesh statistics in ``vision_label.face_facts`` are proxies: they infer "is this
curved" from how much the triangle normals disagree. The B-Rep itself *states* the answer
-- this face is a cylinder of radius 3.0, that one is a cone with a 45-degree half-angle
-- and it also carries the topology, which is where the real discriminators live:

* A **fillet** meets its neighbours tangentially: the surface normals agree across the
  shared edge (edge continuity is smooth). A **chamfer** meets them at a sharp angle.
  That single fact separates the two classes the 5-class task keeps confusing, and it is
  invisible in any rendering because both look like a narrow band.
* A **hole** wall is a cylinder that closes on itself (periodic in u, ~360 degrees of
  wrap). A fillet is a cylinder too, but a partial one.
* A **cut** floor or wall is a plane whose edges are concave -- material was removed
  around it -- while ordinary stock faces sit on convex or mixed boundaries.

So this module reports what the geometry kernel knows, per face:
surface type and its parameters, area, wrap angle, concavity, neighbour count with their
surface types, and for each shared edge the normal deviation across it and whether it is
convex or concave.

Thresholds are deliberately NOT applied here. This returns measurements; what counts as
"tangential" or "narrow" is decided from the labelled data by
``data/measure_class_geometry.py``, so the numbers in the prompt come from the ground
truth rather than from a guess.

Usage:
    python -m brepformer.face_geometry --step model.step             # all faces
    python -m brepformer.face_geometry --step model.step --face 12   # one face
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: GeomAbs_SurfaceType enum index -> readable name, in OCC's declaration order.
SURFACE_TYPES = ["plane", "cylinder", "cone", "sphere", "torus", "bezier", "bspline",
                 "revolution", "extrusion", "offset", "other"]


def _aligned_shape(step_path: str):
    """Load and rotator-align a STEP, matching ``reclassify_visual._tessellate``.

    The alignment must match, because face indices are assigned by exploring the
    *aligned* shape; using the raw shape here would report geometry for the wrong face.
    """
    rotator_dir = Path(__file__).resolve().parent.parent / "rotator"
    if str(rotator_dir) not in sys.path:
        sys.path.insert(0, str(rotator_dir))
    from pipeline import load_step, align_shape

    shape = load_step(step_path)
    aligned, _R, _c, _m = align_shape(shape)
    return aligned


def _face_list(shape) -> List:
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods

    faces = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        faces.append(topods.Face(exp.Current()))
        exp.Next()
    return faces


def _surface_info(face) -> Dict:
    """Surface type and the parameters that matter per type, in model units."""
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface

    ad = BRepAdaptor_Surface(face, True)
    kind = SURFACE_TYPES[min(int(ad.GetType()), len(SURFACE_TYPES) - 1)]
    info: Dict = {"surface_type": kind}
    try:
        if kind == "cylinder":
            info["radius"] = round(ad.Cylinder().Radius(), 4)
        elif kind == "cone":
            info["half_angle_degrees"] = round(math.degrees(ad.Cone().SemiAngle()), 2)
            info["radius"] = round(ad.Cone().RefRadius(), 4)
        elif kind == "sphere":
            info["radius"] = round(ad.Sphere().Radius(), 4)
        elif kind == "torus":
            info["major_radius"] = round(ad.Torus().MajorRadius(), 4)
            info["minor_radius"] = round(ad.Torus().MinorRadius(), 4)
    except Exception:  # noqa: BLE001 - degenerate surfaces raise from the kernel
        pass

    # Parametric extent doubles as the wrap measurement for analytic surfaces: a full
    # bore spans 2*pi in u, a fillet a fraction of it. Read off the surface rather than
    # estimated from normals, so it is exact.
    try:
        u0, u1, v0, v1 = ad.FirstUParameter(), ad.LastUParameter(), \
            ad.FirstVParameter(), ad.LastVParameter()
        if kind in ("cylinder", "cone", "sphere", "torus", "revolution"):
            info["u_sweep_degrees"] = round(math.degrees(abs(u1 - u0)), 1)
        info["closed_in_u"] = bool(ad.IsUClosed())
        info["closed_in_v"] = bool(ad.IsVClosed())
    except Exception:  # noqa: BLE001
        pass
    return info


def _face_normal_at(face, pnt, reversed_: bool):
    """Outward unit normal of `face` at the surface point nearest `pnt`, or None."""
    import numpy as np
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCC.Core.GeomLProp import GeomLProp_SLProps

    try:
        surf = BRep_Tool.Surface(face)
        proj = GeomAPI_ProjectPointOnSurf(pnt, surf)
        if proj.NbPoints() < 1:
            return None
        u, v = proj.LowerDistanceParameters()
        props = GeomLProp_SLProps(surf, u, v, 1, 1e-6)
        if not props.IsNormalDefined():
            return None
        n = props.Normal()
        vec = np.array([n.X(), n.Y(), n.Z()], dtype=float)
    except Exception:  # noqa: BLE001
        return None
    mag = float(np.linalg.norm(vec))
    if mag <= 0:
        return None
    vec /= mag
    # The kernel's normal follows the surface's parameterisation, not the solid; a
    # REVERSED face points into the material unless flipped. Getting this wrong inverts
    # every convexity verdict, which would make chamfers look like cuts.
    return -vec if reversed_ else vec


def face_topology(shape, faces: Sequence, face_idx: int,
                  edge_face_map=None) -> Dict:
    """Neighbour and shared-edge geometry for one face.

    For each edge shared with another face, reports the angle between the two outward
    normals at the edge midpoint and whether the edge is convex or concave. Convexity is
    the sign of ``(n1 x n2) . edge_tangent``, the standard B-Rep test: it says whether the
    material forms a ridge or a valley there, which is what separates a boss from a
    pocket and a round from a cut.
    """
    import numpy as np
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
    from OCC.Core.TopExp import topexp, TopExp_Explorer
    from OCC.Core.TopoDS import topods
    from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

    if edge_face_map is None:
        edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
        topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

    face = faces[face_idx]
    rev_self = face.Orientation() == TopAbs_REVERSED

    edges: List[Dict] = []
    exp = TopExp_Explorer(face, TopAbs_EDGE)
    seen = set()
    while exp.More():
        edge = topods.Edge(exp.Current())
        exp.Next()
        # An edge is reached twice when a face's wire visits it in both orientations.
        # OCC 7.9 dropped TopoDS_Shape.HashCode, so identity comes from the ancestor
        # map's own index, which is the canonical id for the edge in this shape.
        if not edge_face_map.Contains(edge):
            continue
        key = edge_face_map.FindIndex(edge)
        if key in seen:
            continue
        seen.add(key)

        try:
            curve = BRepAdaptor_Curve(edge)
            t0, t1 = curve.FirstParameter(), curve.LastParameter()
            tmid = 0.5 * (t0 + t1)
            pnt = curve.Value(tmid)
            length = abs(t1 - t0)
            tan = curve.DN(tmid, 1)
            tangent = np.array([tan.X(), tan.Y(), tan.Z()], dtype=float)
            tn = float(np.linalg.norm(tangent))
            if tn <= 0:
                continue
            tangent /= tn
        except Exception:  # noqa: BLE001
            continue

        n_self = _face_normal_at(face, pnt, rev_self)
        if n_self is None:
            continue

        others = edge_face_map.FindFromKey(edge)
        it = others.begin() if hasattr(others, "begin") else None
        neigh_shapes = []
        try:
            # pythonocc exposes the list either as an iterable or via an iterator class.
            for sh in others:
                neigh_shapes.append(sh)
        except TypeError:
            from OCC.Core.TopTools import TopTools_ListIteratorOfListOfShape
            lit = TopTools_ListIteratorOfListOfShape(others)
            while lit.More():
                neigh_shapes.append(lit.Value())
                lit.Next()

        for sh in neigh_shapes:
            if sh.IsSame(face):
                continue
            nb = topods.Face(sh)
            nb_idx = next((j for j, f in enumerate(faces) if f.IsSame(nb)), None)
            n_nb = _face_normal_at(nb, pnt, nb.Orientation() == TopAbs_REVERSED)
            if n_nb is None:
                continue
            dev = math.degrees(math.acos(float(np.clip(n_self @ n_nb, -1.0, 1.0))))
            cross = np.cross(n_self, n_nb)
            sign = float(cross @ tangent)
            edges.append({
                "neighbour_face": nb_idx,
                "neighbour_surface": _surface_info(nb)["surface_type"],
                "normal_deviation_degrees": round(dev, 1),
                "convex": bool(sign > 0),
                "param_length": round(length, 4),
            })

    devs = [e["normal_deviation_degrees"] for e in edges]
    return {
        "n_neighbours": len(edges),
        "neighbour_surfaces": sorted({e["neighbour_surface"] for e in edges}),
        # Tangential continuity is the fillet signature: a blend is built to meet its
        # neighbours smoothly, so the normals barely change across the edge.
        "min_edge_normal_deviation_degrees": round(min(devs), 1) if devs else None,
        "max_edge_normal_deviation_degrees": round(max(devs), 1) if devs else None,
        "n_tangential_edges": sum(1 for d in devs if d < 10.0),
        "n_convex_edges": sum(1 for e in edges if e["convex"]),
        "n_concave_edges": sum(1 for e in edges if not e["convex"]),
        "edges": edges,
    }


def part_adjacency(shape, faces: Sequence, edge_face_map) -> Dict:
    """The whole part's attributed adjacency graph, built once, edge-first.

    Computing this per face (as ``face_topology`` does) evaluates every shared edge twice,
    once from each side, and the surface projections are the expensive part. Iterating the
    edge->faces map instead visits each edge once, which roughly halves the cost and, more
    importantly, produces the *graph* -- and the graph carries information no single face
    has.

    Returns ``{"per_face": {i: [edge records]}, "concave_component": {i: component id},
    "components": {cid: [face ids]}}``.
    """
    import numpy as np
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.TopAbs import TopAbs_REVERSED
    from OCC.Core.TopoDS import topods

    # An indexed map gives O(1) face -> index. Scanning ``faces`` with IsSame instead
    # (as the per-face path does) is O(n) per lookup, which on a 500-face part with 1,500
    # edges is over a million shape comparisons -- the graph would cost more than the
    # projections it is built from.
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape

    fmap = TopTools_IndexedMapOfShape()
    for f in faces:
        fmap.Add(f)

    def find(f):
        i = fmap.FindIndex(f)
        return i - 1 if i > 0 else None

    per_face: Dict[int, List[Dict]] = {i: [] for i in range(len(faces))}
    concave_pairs: List[tuple] = []

    # OCC 7.9's Python binding exposes Size(), not Extent(), on the indexed maps.
    for k in range(1, edge_face_map.Size() + 1):
        edge = topods.Edge(edge_face_map.FindKey(k))
        lst = edge_face_map.FindFromIndex(k)
        shapes = []
        try:
            for sh in lst:
                shapes.append(sh)
        except TypeError:
            from OCC.Core.TopTools import TopTools_ListIteratorOfListOfShape
            it = TopTools_ListIteratorOfListOfShape(lst)
            while it.More():
                shapes.append(it.Value())
                it.Next()
        # A seam or free edge has one adjacent face; a non-manifold edge has more than two.
        # Only genuine face-face contacts define adjacency.
        idxs = [j for j in (find(topods.Face(s)) for s in shapes) if j is not None]
        idxs = sorted(set(idxs))
        if len(idxs) < 2:
            continue
        try:
            curve = BRepAdaptor_Curve(edge)
            t0, t1 = curve.FirstParameter(), curve.LastParameter()
            tmid = 0.5 * (t0 + t1)
            pnt = curve.Value(tmid)
            tan = curve.DN(tmid, 1)
            tangent = np.array([tan.X(), tan.Y(), tan.Z()], dtype=float)
            tn = float(np.linalg.norm(tangent))
            if tn <= 0:
                continue
            tangent /= tn
            length = abs(t1 - t0)
        except Exception:  # noqa: BLE001
            continue

        normals = {}
        for j in idxs:
            f = faces[j]
            normals[j] = _face_normal_at(f, pnt, f.Orientation() == TopAbs_REVERSED)

        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                ja, jb = idxs[a], idxs[b]
                na, nb = normals[ja], normals[jb]
                if na is None or nb is None:
                    continue
                dev = math.degrees(math.acos(float(np.clip(na @ nb, -1.0, 1.0))))
                sign = float(np.cross(na, nb) @ tangent)
                convex = bool(sign > 0)
                per_face[ja].append({
                    "neighbour_face": jb,
                    "neighbour_surface": _surface_info(faces[jb])["surface_type"],
                    "normal_deviation_degrees": round(dev, 1),
                    "convex": convex, "param_length": round(length, 4)})
                per_face[jb].append({
                    "neighbour_face": ja,
                    "neighbour_surface": _surface_info(faces[ja])["surface_type"],
                    "normal_deviation_degrees": round(dev, 1),
                    "convex": convex, "param_length": round(length, 4)})
                # Joshi & Chang's AAG hint: material is removed *around* a concave edge, so
                # a machined feature is a set of faces mutually joined by concave edges --
                # the walls and floor of a pocket, the wall and seat of a counterbore. A
                # tangential edge is neither ridge nor valley and its convexity sign is
                # numerical noise, so it must not join components or every fillet would
                # weld the whole part into one.
                if not convex and dev >= 10.0:
                    concave_pairs.append((ja, jb, dev))

    def components(pairs) -> tuple:
        """Union-find over a set of face pairs -> (face -> component id, id -> members)."""
        parent = list(range(len(faces)))

        def root(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in pairs:
            ra, rb = root(a), root(b)
            if ra != rb:
                parent[ra] = rb
        groups: Dict[int, List[int]] = {}
        for i in range(len(faces)):
            groups.setdefault(root(i), []).append(i)
        # Singletons are not features; id 0 reads as "this face belongs to no concave
        # group" rather than as a one-face feature.
        comp_id, members_by_id, next_id = {}, {}, 1
        for _r, members in groups.items():
            cid = 0 if len(members) < 2 else next_id
            if cid:
                members_by_id[cid] = members
                next_id += 1
            for i in members:
                comp_id[i] = cid
        return comp_id, members_by_id

    loose_id, loose_members = components([(a, b) for a, b, _ in concave_pairs])
    # The plain AAG has a known failure that shows up immediately on real parts: on a bent
    # bracket the sheet-metal bend edges are concave too, so one component swallows most of
    # the part and its size stops meaning anything. Requiring a *sharp* valley (>=30 deg)
    # keeps pocket and counterbore corners while dropping bends and shallow blends. Which
    # of the two actually separates the classes is not decided here -- both are reported and
    # ``data/measure_class_geometry.py`` measures them against the labels.
    strict_id, strict_members = components(
        [(a, b) for a, b, dev in concave_pairs if dev >= 30.0])
    return {"per_face": per_face,
            "concave_component": loose_id, "components": loose_members,
            "strict_component": strict_id, "strict_components": strict_members}


def _summarise_edges(edges: List[Dict]) -> Dict:
    """The aggregate edge facts, shared by ``face_topology`` and ``part_adjacency``."""
    devs = [e["normal_deviation_degrees"] for e in edges]
    return {
        "n_neighbours": len(edges),
        "neighbour_surfaces": sorted({e["neighbour_surface"] for e in edges}),
        "min_edge_normal_deviation_degrees": round(min(devs), 1) if devs else None,
        "max_edge_normal_deviation_degrees": round(max(devs), 1) if devs else None,
        "n_tangential_edges": sum(1 for d in devs if d < 10.0),
        "n_convex_edges": sum(1 for e in edges if e["convex"]),
        "n_concave_edges": sum(1 for e in edges if not e["convex"]),
        "edges": edges,
    }


def _wire_facts(face) -> Dict:
    """Loop structure of one face: how many wires, and how many are inner.

    An inner wire is a hole punched through the face -- the top face of a plate with four
    bolt holes has four of them. This is the cheapest reliable way to tell that a face
    *bounds* a hole, and it separates a pocket floor (inner wire absent, boundary concave)
    from a plate face that a through hole passes through. It also catches the case a
    through hole is distinguished from a blind one by: the blind hole's bottom face has one
    outer wire and no inner wire, while a through hole has an inner wire at both ends.
    """
    from OCC.Core.BRepTools import breptools
    from OCC.Core.TopAbs import TopAbs_WIRE
    from OCC.Core.TopExp import TopExp_Explorer

    n = 0
    exp = TopExp_Explorer(face, TopAbs_WIRE)
    while exp.More():
        n += 1
        exp.Next()
    try:
        outer = breptools.OuterWire(face)
        has_outer = not outer.IsNull()
    except Exception:  # noqa: BLE001
        has_outer = True
    return {"n_wires": n,
            "n_inner_wires": max(0, n - (1 if has_outer else 0))}


def _curvature_at_mid(face) -> Dict:
    """Gaussian and mean curvature, and the two principal curvatures, at the face middle.

    The sign pair separates shapes that the surface type alone does not. A fillet blending
    a convex corner is a cylinder (K=0) but an internal fillet running around a pocket
    floor is a torus with **negative** Gaussian curvature -- a saddle -- while the convex
    outer round of the same radius has positive K. A bore wall and an external round are
    both K=0 cylinders that differ in the *sign* of the mean curvature once orientation is
    taken into account, which is a cheaper and more robust version of the concavity test.
    """
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomLProp import GeomLProp_SLProps
    from OCC.Core.TopAbs import TopAbs_REVERSED

    try:
        ad = BRepAdaptor_Surface(face, True)
        u = 0.5 * (ad.FirstUParameter() + ad.LastUParameter())
        v = 0.5 * (ad.FirstVParameter() + ad.LastVParameter())
        props = GeomLProp_SLProps(BRep_Tool.Surface(face), u, v, 2, 1e-6)
        if not props.IsCurvatureDefined():
            return {}
        k1, k2 = props.MinCurvature(), props.MaxCurvature()
        gauss, mean = props.GaussianCurvature(), props.MeanCurvature()
    except Exception:  # noqa: BLE001
        return {}
    # Curvature signs follow the parameterisation, like the normal does; a REVERSED face
    # must be flipped or a bore reads as a shaft.
    flip = -1.0 if face.Orientation() == TopAbs_REVERSED else 1.0
    return {"gaussian_curvature_sign": int(np.sign(round(gauss, 12)))
            if abs(gauss) > 1e-12 else 0,
            "mean_curvature_sign": int(np.sign(flip * mean)) if abs(mean) > 1e-12 else 0,
            "principal_curvatures": [round(flip * k1, 6), round(flip * k2, 6)],
            "is_saddle": bool(gauss < -1e-12)}


def _hull_facts(shape, faces: Sequence, centroids) -> Dict[int, Dict]:
    """How far each face sits inside the part's convex hull.

    A face on the hull is stock -- the outside of the raw billet, or an original body
    surface -- and a face well inside it is looking out from a cavity. This is the single
    cleanest separator for the `random` class, which is 27% of all faces and the class the
    others are most often confused with, and it needs no rules: it is a distance.
    """
    out: Dict[int, Dict] = {}
    try:
        from scipy.spatial import ConvexHull
        pts = []
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.TopAbs import TopAbs_VERTEX
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopoDS import topods
        exp = TopExp_Explorer(shape, TopAbs_VERTEX)
        seen = set()
        while exp.More():
            p = BRep_Tool.Pnt(topods.Vertex(exp.Current()))
            t = (round(p.X(), 6), round(p.Y(), 6), round(p.Z(), 6))
            if t not in seen:
                seen.add(t)
                pts.append(t)
            exp.Next()
        if len(pts) < 5:
            return out
        hull = ConvexHull(np.asarray(pts, dtype=float))
        eqs = hull.equations                      # A x + b <= 0 inside
        extent_scale = float(np.ptp(hull.points, axis=0).max()) or 1.0
        for i, c in centroids.items():
            if c is None:
                continue
            # Signed distance to the nearest hull plane: 0 on the hull, negative inside.
            d = float(np.max(eqs[:, :3] @ np.asarray(c) + eqs[:, 3]))
            out[i] = {"hull_depth_vs_extent": round(-d / extent_scale, 5),
                      "on_convex_hull": bool(-d / extent_scale < 0.005)}
    except Exception:  # noqa: BLE001 - scipy absent or degenerate hull
        return out
    return out


def _triangles_of_shape(shape):
    """(V, F) triangle soup of the whole part, for ray casting. Meshes if needed."""
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.TopoDS import topods

    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    x0, y0, z0, x1, y1, z1 = box.Get()
    diag = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2) or 1.0
    # Coarse on purpose. This mesh is only an occluder for ray casting -- it answers "is
    # something in the way", not "what shape is it" -- and the cost of every ray is linear
    # in the triangle count. A 1%-of-diagonal deflection produced meshes fine enough to
    # dominate the whole corpus measurement while changing no accessibility number
    # meaningfully.
    BRepMesh_IncrementalMesh(shape, diag * 0.05, False, 0.8, True)

    verts: List = []
    tris: List = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods.Face(exp.Current())
        exp.Next()
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is None:
            continue
        trsf = loc.Transformation()
        base = len(verts)
        for k in range(1, tri.NbNodes() + 1):
            p = tri.Node(k).Transformed(trsf)
            verts.append((p.X(), p.Y(), p.Z()))
        for k in range(1, tri.NbTriangles() + 1):
            a, b, c = tri.Triangle(k).Get()
            tris.append((base + a - 1, base + b - 1, base + c - 1))
    if not tris:
        return None, None
    return np.asarray(verts, dtype=float), np.asarray(tris, dtype=np.int64)


def _accessibility(V, F, centroids, normals, extent) -> Dict[int, Dict]:
    """Fraction of outward directions from which each face can be reached from outside.

    This is the measurement that says, numerically, what the rendering problem was: a bore
    wall can be reached from almost no direction because its own opening and far wall are in
    the way, while a stock face is open to nearly half the sphere. It is also a genuine
    manufacturing quantity -- tool accessibility -- and it separates the classes that differ
    by *enclosure* rather than by shape: a pocket floor from a flat stock face, a
    counterbore seat from a step, a blind hole bottom from an exposed boss top.

    Computed by ray casting the part's own triangle mesh with a vectorised Moeller-Trumbore
    test rather than an OCC ray-solid intersection, which is orders of magnitude slower.
    """
    if V is None or not centroids:
        return {}
    v0 = V[F[:, 0]]
    e1 = V[F[:, 1]] - v0
    e2 = V[F[:, 2]] - v0
    # Bounding sphere per triangle, for the prefilter below.
    tri_c = (v0 + V[F[:, 1]] + V[F[:, 2]]) / 3.0
    tri_r = np.maximum.reduce([
        np.linalg.norm(v0 - tri_c, axis=1),
        np.linalg.norm(V[F[:, 1]] - tri_c, axis=1),
        np.linalg.norm(V[F[:, 2]] - tri_c, axis=1)])
    # 26 directions: a hemisphere sample is enough to rank openness; more directions move
    # the number by a percent or two and cost linearly.
    dirs = []
    for az in range(0, 360, 45):
        for el in (-60, -30, 0, 30, 60):
            a, e = math.radians(az), math.radians(el)
            dirs.append((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a),
                         math.sin(e)))
    dirs += [(0, 0, 1), (0, 0, -1)]
    D = np.asarray(dirs, dtype=float)
    eps = extent * 1e-3

    out: Dict[int, Dict] = {}
    for i, c in centroids.items():
        n = normals.get(i)
        if c is None or n is None:
            continue
        c = np.asarray(c, dtype=float)
        # Only directions on the outward side of the face can reach it at all.
        cand = D[(D @ np.asarray(n)) > 0.1]
        if cand.size == 0:
            out[i] = {"accessibility": 0.0, "n_test_directions": 0}
            continue
        origin = c + np.asarray(n) * eps
        # Direction-at-a-time Moeller-Trumbore. Batching all directions into a
        # (triangles, directions, 3) grid was tried and is ~3x *slower*: the intermediate
        # cross product leaves cache and the einsums it needs are not BLAS shapes, which
        # costs more than the Python loop over a dozen directions saves.
        rel = tri_c - origin
        escaped = 0
        for d in cand:
            # Prefilter: a triangle can only be hit if the ray passes within its bounding
            # sphere. The perpendicular distance from the ray line is |d x rel| for a unit
            # d, which is far cheaper than the full intersection test, and on a real part it
            # discards well over 99% of triangles -- the exact test then runs on a handful
            # instead of tens of thousands. Without this, accessibility was by a wide margin
            # the slowest thing in the whole measurement (112 s on a single 283-face part).
            perp = np.linalg.norm(np.cross(d, rel), axis=1)
            along = rel @ d
            near = (perp <= tri_r) & (along > -tri_r)      # and not behind the origin
            if not near.any():
                escaped += 1
                continue
            idx = np.flatnonzero(near)
            a1, a2, av0 = e1[idx], e2[idx], v0[idx]
            tv = origin - av0
            qv = np.cross(tv, a1)
            pv = np.cross(d, a2)
            det = np.einsum("ij,ij->i", a1, pv)
            ok = np.abs(det) > 1e-12
            inv = np.zeros_like(det)
            inv[ok] = 1.0 / det[ok]
            u = np.einsum("ij,ij->i", tv, pv) * inv
            vparam = (qv @ d) * inv
            t = np.einsum("ij,ij->i", a2, qv) * inv
            hit = ok & (u >= 0) & (u <= 1) & (vparam >= 0) & (u + vparam <= 1) & (t > eps)
            if hit.any():
                continue
            escaped += 1
        out[i] = {"accessibility": round(escaped / len(cand), 3),
                  "n_test_directions": int(len(cand))}
    return out


def _axis_of(face):
    """(point on axis, unit axis direction, radius) for a face of revolution, else None."""
    import numpy as np
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface

    ad = BRepAdaptor_Surface(face, True)
    kind = SURFACE_TYPES[min(int(ad.GetType()), len(SURFACE_TYPES) - 1)]
    try:
        if kind == "cylinder":
            ax, r = ad.Cylinder().Axis(), ad.Cylinder().Radius()
        elif kind == "cone":
            ax, r = ad.Cone().Axis(), ad.Cone().RefRadius()
        else:
            return None
    except Exception:  # noqa: BLE001
        return None
    loc, d = ax.Location(), ax.Direction()
    return (np.array([loc.X(), loc.Y(), loc.Z()]),
            np.array([d.X(), d.Y(), d.Z()]), float(r))


def _same_axis(a, b, tol=1e-4) -> bool:
    """Do two faces of revolution lie on the same infinite axis with the same radius?"""
    import numpy as np

    if a is None or b is None:
        return False
    (p1, d1, r1), (p2, d2, r2) = a, b
    if abs(r1 - r2) > max(tol, 1e-3 * max(r1, r2)):
        return False
    if abs(abs(float(d1 @ d2)) - 1.0) > 1e-3:      # axes must be parallel
        return False
    off = p2 - p1
    return bool(np.linalg.norm(off - float(off @ d1) * d1) <= max(tol, 1e-3 * r1))


def _coaxial_sweep(shape, faces, idx, edge_face_map, topo_cache) -> Optional[float]:
    """Total sweep of the face plus every coaxial face reachable through tangent edges.

    This is what tells a hole from a fillet, and nothing else does. In the labelled
    corpus a hole wall is a cylinder swept exactly 180 degrees, because the kernel splits
    a full bore into two half-cylinders at the seam -- and a fillet is *also* a partial
    cylinder, typically 90. Looking at one face, the two are the same kind of object.
    Following the tangent seam and adding up the sweep recovers the whole bore: 360 for a
    hole, unchanged for a blend band, since a fillet's neighbours across its tangent edges
    are the faces it blends rather than more of itself.
    """
    mine = _axis_of(faces[idx])
    if mine is None:
        return None
    total = 0.0
    seen = set()
    stack = [idx]
    while stack:
        j = stack.pop()
        if j in seen or len(seen) > 12:      # a bore is 1-4 faces; cap runaway walks
            continue
        seen.add(j)
        total += float(_surface_info(faces[j]).get("u_sweep_degrees") or 0.0)
        if j not in topo_cache:
            topo_cache[j] = face_topology(shape, faces, j, edge_face_map)
        for e in topo_cache[j]["edges"]:
            k = e["neighbour_face"]
            if (k is None or k in seen
                    or e["normal_deviation_degrees"] >= 10.0
                    or not _same_axis(mine, _axis_of(faces[k]))):
                continue
            stack.append(k)
    return round(total, 1)


def _concave_surface(face) -> Optional[bool]:
    """For a face of revolution: is the material outside it (a bore) or inside (a shaft)?

    A bore wall's outward normal points back toward its own axis; a shaft's points away.
    Holes are always bores, which separates them from external rounds of the same radius.
    """
    import numpy as np
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.TopAbs import TopAbs_REVERSED

    ax = _axis_of(face)
    if ax is None:
        return None
    p_ax, d_ax, _r = ax
    ad = BRepAdaptor_Surface(face, True)
    try:
        u = 0.5 * (ad.FirstUParameter() + ad.LastUParameter())
        v = 0.5 * (ad.FirstVParameter() + ad.LastVParameter())
        pnt = ad.Value(u, v)
    except Exception:  # noqa: BLE001
        return None
    n = _face_normal_at(face, pnt, face.Orientation() == TopAbs_REVERSED)
    if n is None:
        return None
    off = np.array([pnt.X(), pnt.Y(), pnt.Z()]) - p_ax
    radial = off - float(off @ d_ax) * d_ax
    mag = float(np.linalg.norm(radial))
    if mag <= 1e-9:
        return None
    return bool(float(n @ (radial / mag)) < 0)


def face_geometry(step_path: str, face_indices: Optional[Sequence[int]] = None,
                  with_topology: bool = True, with_graph: bool = True,
                  with_accessibility: bool = True) -> Dict[int, Dict]:
    """Exact geometry for the requested faces (all of them if `face_indices` is None).

    Keyed by the same face index ``reclassify_visual._tessellate`` and the dataset labels
    use: ``TopExp_Explorer`` order over the aligned shape.

    ``with_graph`` and ``with_accessibility`` are whole-part computations, so they cost the
    same whether one face is requested or all of them. They are on by default because the
    facts they produce -- concave-component membership and openness to the outside -- are
    not derivable from a single face and are among the strongest available; turn them off
    for a fast pass over a very large corpus.
    """
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCC.Core.TopExp import topexp
    from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

    shape = _aligned_shape(step_path)
    faces = _face_list(shape)

    areas = []
    for f in faces:
        props = GProp_GProps()
        try:
            brepgprop.SurfaceProperties(f, props)
            areas.append(props.Mass())
        except Exception:  # noqa: BLE001
            areas.append(0.0)
    total_area = sum(areas) or 1.0

    edge_face_map = None
    if with_topology:
        edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
        topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

    wanted = range(len(faces)) if face_indices is None else [
        i for i in face_indices if 0 <= i < len(faces)]

    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib

    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    x0, y0, z0, x1, y1, z1 = box.Get()
    extent = max(x1 - x0, y1 - y0, z1 - z0) or 1.0

    # Whole-part computations, done once. The graph replaces the per-face topology walk
    # (same records, half the projections) and additionally yields the concave components.
    graph = None
    topo_cache: Dict[int, Dict] = {}
    if with_topology and with_graph:
        try:
            graph = part_adjacency(shape, faces, edge_face_map)
            for i, edges in graph["per_face"].items():
                topo_cache[i] = _summarise_edges(edges)
        except Exception:  # noqa: BLE001 - fall back to the per-face path
            graph = None
            topo_cache = {}

    centroids: Dict[int, Optional[tuple]] = {}
    face_normals: Dict[int, Optional[np.ndarray]] = {}
    if with_graph or with_accessibility:
        for i, f in enumerate(faces):
            props = GProp_GProps()
            try:
                brepgprop.SurfaceProperties(f, props)
                c = props.CentreOfMass()
                centroids[i] = (c.X(), c.Y(), c.Z())
            except Exception:  # noqa: BLE001
                centroids[i] = None
    hull = _hull_facts(shape, faces, centroids) if with_graph else {}

    access: Dict[int, Dict] = {}
    if with_accessibility:
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core.TopAbs import TopAbs_REVERSED
        for i in (wanted if face_indices is not None else range(len(faces))):
            c = centroids.get(i)
            if c is None:
                continue
            try:
                ad = BRepAdaptor_Surface(faces[i], True)
                u = 0.5 * (ad.FirstUParameter() + ad.LastUParameter())
                v = 0.5 * (ad.FirstVParameter() + ad.LastVParameter())
                pnt = ad.Value(u, v)
                # The mid-parameter point, not the centre of mass: the centroid of a bore
                # wall lies on its axis, in empty space, and rays cast from there escape in
                # every direction, which would report a hole as fully accessible.
                centroids[i] = (pnt.X(), pnt.Y(), pnt.Z())
                face_normals[i] = _face_normal_at(
                    faces[i], pnt, faces[i].Orientation() == TopAbs_REVERSED)
            except Exception:  # noqa: BLE001
                face_normals[i] = None
        try:
            V, F = _triangles_of_shape(shape)
            access = _accessibility(V, F, {i: centroids[i] for i in face_normals
                                           if centroids.get(i) is not None},
                                    face_normals, extent)
        except Exception:  # noqa: BLE001 - meshing failure must not lose the other facts
            access = {}

    out: Dict[int, Dict] = {}
    for i in wanted:
        rec = {"face": i, "n_faces_in_part": len(faces),
               "area": round(areas[i], 5),
               "area_fraction_of_part": round(areas[i] / total_area, 5),
               "part_extent": round(extent, 3)}
        rec.update(_surface_info(faces[i]))
        rec.update(_wire_facts(faces[i]))
        rec.update(_curvature_at_mid(faces[i]))
        if with_topology:
            if i not in topo_cache:
                topo_cache[i] = face_topology(shape, faces, i, edge_face_map)
            rec.update(topo_cache[i])
            rec["coaxial_group_sweep_degrees"] = _coaxial_sweep(
                shape, faces, i, edge_face_map, topo_cache)
        if graph is not None:
            for prefix, id_key, mem_key in (
                    ("concave_component", "concave_component", "components"),
                    ("strict_component", "strict_component", "strict_components")):
                cid = graph[id_key].get(i, 0)
                members = graph[mem_key].get(cid, [])
                # The id itself, not just what the component looks like: a 27-class label
                # names a *feature*, and a feature owns all of its faces, so anything that
                # wants to answer a question once per feature (rather than once per face)
                # needs to know which faces belong together. `measure_class_geometry` reads
                # an explicit CONDITIONS list, so this extra key is inert there.
                rec[f"{prefix}_id"] = cid
                rec[f"{prefix}_size"] = len(members)
                # What a component is *made of* is what names the feature: walls plus a
                # floor is a pocket, one cylinder plus one disc is a blind hole, a ring of
                # planes is a slot. Size alone cannot say which.
                rec[f"{prefix}_surfaces"] = "+".join(sorted(
                    {_surface_info(faces[j])["surface_type"] for j in members})) or None
                # Normalised, because "8 faces" means a pocket on a 400-face casting and
                # the whole part on a 10-face bracket.
                rec[f"{prefix}_area_fraction"] = round(
                    sum(areas[j] for j in members) / total_area, 5) if members else 0.0
        if i in hull:
            rec.update(hull[i])
        if i in access:
            rec.update(access[i])
        rec["concave_surface"] = _concave_surface(faces[i])
        r = rec.get("radius")
        if isinstance(r, (int, float)):
            # Absolute radius says nothing without scale: 2 mm is a bolt hole on a bracket
            # and a pinhole on an engine block. The ratio is what the class depends on.
            rec["radius_vs_part_extent"] = round(r / extent, 5)
        out[i] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", required=True)
    ap.add_argument("--face", type=int, action="append",
                    help="Restrict to this face (repeatable)")
    ap.add_argument("--no_topology", action="store_true")
    ap.add_argument("--brief", action="store_true",
                    help="Drop the per-edge detail from the output")
    args = ap.parse_args()

    g = face_geometry(args.step, args.face, with_topology=not args.no_topology)
    if args.brief:
        for rec in g.values():
            rec.pop("edges", None)
    print(json.dumps(g, indent=2))


if __name__ == "__main__":
    main()
