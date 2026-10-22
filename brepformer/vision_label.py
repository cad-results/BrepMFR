#!/usr/bin/env python3
"""Dense-view, multi-pass Claude vision labelling of individual B-Rep faces.

This is a stronger version of the vision pass in ``reclassify_visual.py``, built to
answer whether that pass measured badly because vision cannot do the task, or merely
because it was shown bad pictures. The earlier pass sent 6 views from a fixed 90-degree
camera set; a narrow chamfer is edge-on or occluded in most directions, so the model was
often asked to classify something it could not see.

Four changes, each aimed at that:

1. **Full-sphere candidate coverage at 15 degrees.** Every camera direction on a 15-degree
   spherical grid (~266 of them) is *considered* for every face.

2. **Analytic view ranking instead of rendering everything.** Rendering 266 images per
   face would cost days over the corpus. Instead each face's triangle normals and areas
   are projected onto every candidate direction, giving the exact foreshortened area the
   camera would see, in one vectorised operation. Only the best directions are rendered.
   This is both far faster *and* better than the old "render 6, sort by red pixels": the
   ranking now searches the whole sphere rather than picking the least-bad of six.

3. **The model can ask for more angles.** Analytic ranking cannot see occlusion by other
   geometry, and cannot know which cue a given face is ambiguous on. So the model may
   reply asking for specific azimuth/elevation pairs, which are rendered on demand and
   fed back. That loop is the "suggest more angles if needed" step.

4. **Multiple independent passes, then a vote.** Each pass sees a *different* selection
   of views (best-visibility, then maximally-separated directions, then a mixed set), so
   the passes are not near-duplicates of one another. Agreement across passes is the
   confidence signal; disagreement leaves the face masked rather than relabelled.

**No image is ever written to disk.** Every render stays as PNG bytes in memory and is
base64'd straight into the API call.

**This pass is measured, not trusted.** As a verifier the earlier version denied 91% of
wrong labels and 74% of correct ones; as a labeller it scored at or below chance on
feature faces. Run ``data/validate_vision_classify.py`` against mftrcad ground truth with
``--use_dense`` before letting any output here rewrite a label, and compare to the
teacher's accuracy on the same faces. Geometric *class guesses* are deliberately withheld
from the prompt (they corrupt 15% of correct labels, 100% of ``chamfer_2``); objective
measurements the model cannot see -- surface type, area share, neighbour count -- are
provided, since those are facts rather than hypotheses.

Usage:
    # label one face of one model, 3 passes, allowing 2 rounds of angle requests
    python -m brepformer.vision_label --step model.step --face 42 --label_space 5

    # batch: relabel the faces the teacher is least sure about
    python -m brepformer.vision_label \\
        --step_dir brepformer/data/suitable_samples_scaled/steps \\
        --seg_dir brepformer/inference/hard_samples_27 \\
        --out_dir brepformer/inference/hard_samples_27_vision \\
        --label_space 27 --faces_per_model 8 --passes 3
"""

from __future__ import annotations

import argparse
import base64
import collections
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from brepformer.data.classes import CLASS_NAMES, DEFEATURE_CLASS_NAMES
from brepformer.face_geometry import face_geometry
from brepformer.reclassify_visual import (
    _MESH_CACHE,
    _face_frame,
    _red_pixels,
    _render_to_png_bytes,
    _tessellate,
    _vision_client,
)

#: Faces whose label the pass declines to change stay at this value upstream. It is also
#: what a masked face is written as: ``merge_suitable_samples.IGNORE_LABEL`` is -1 and the
#: segmentation loss is ``CrossEntropyLoss(ignore_index=-1)``, so a masked face contributes
#: no gradient and is excluded from the class statistics.
IGNORE_LABEL = -1

#: The class value the model returns when it declines to answer. Same number as
#: ``IGNORE_LABEL`` by design -- an abstention and a mask mean the same thing downstream --
#: but named separately because they are different events: one is the model's judgement,
#: the other is this module's decision about the vote.
ABSTAIN = -1

#: A face is only allowed to supervise training if the passes agree this strongly and the
#: combined confidence clears this floor. Defaults chosen from the ground-truth validation
#: (unanimous faces are far more accurate than split ones); tune them there, not here.
DEFAULT_MIN_AGREEMENT = 0.66
DEFAULT_MIN_CONFIDENCE = 0.55


# ============================================================
# Candidate camera directions: the full sphere at a fixed step
# ============================================================

def sphere_directions(step_deg: int = 15) -> List[Tuple[str, Tuple[float, float, float],
                                                        Tuple[float, float, float]]]:
    """Every camera direction on a `step_deg` spherical grid, plus both poles.

    At 15 degrees this is 24 azimuths x 11 elevations + 2 poles = 266 directions. They
    are *candidates*: :func:`rank_directions` scores all of them analytically and only a
    handful are ever rendered.

    The up-vector is +Z except near the poles, where +Z is parallel to the view direction
    and VTK would produce a degenerate camera; there it switches to +Y.
    """
    out = []
    for elev in range(-90 + step_deg, 90, step_deg):
        er = math.radians(elev)
        cos_e, sin_e = math.cos(er), math.sin(er)
        for az in range(0, 360, step_deg):
            ar = math.radians(az)
            d = (math.sin(ar) * cos_e, math.cos(ar) * cos_e, sin_e)
            out.append((f"az{az:03d}_el{elev:+03d}", d, (0.0, 0.0, 1.0)))
    out.append(("top", (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)))
    out.append(("bottom", (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)))
    return out


def direction_from_angles(az_deg: float, el_deg: float):
    """Camera direction + up vector for an explicit azimuth/elevation request."""
    ar, er = math.radians(az_deg), math.radians(el_deg)
    cos_e = math.cos(er)
    d = (math.sin(ar) * cos_e, math.cos(ar) * cos_e, math.sin(er))
    up = (0.0, 0.0, 1.0) if abs(d[2]) < 0.95 else (0.0, 1.0, 0.0)
    return d, up


# ============================================================
# Analytic visibility: which directions actually show this face?
# ============================================================

def face_triangles(polydata, cell_face_ids: Sequence[int], face_idx: int):
    """Per-triangle (normal, area) arrays for one face, plus its vertex coordinates."""
    pts = polydata.GetPoints()
    normals, areas, verts = [], [], []
    for cid, fi in enumerate(cell_face_ids):
        if fi != face_idx:
            continue
        cell = polydata.GetCell(cid)
        pid = cell.GetPointIds()
        if pid.GetNumberOfIds() < 3:
            continue
        p = np.array([pts.GetPoint(pid.GetId(k)) for k in range(3)], dtype=float)
        n = np.cross(p[1] - p[0], p[2] - p[0])
        mag = float(np.linalg.norm(n))
        if mag <= 0:
            continue
        normals.append(n / mag)
        areas.append(mag / 2.0)          # cross-product magnitude is twice the area
        verts.append(p)
    if not normals:
        return None
    return np.asarray(normals), np.asarray(areas), np.concatenate(verts)


def rank_directions(polydata, cell_face_ids: Sequence[int], face_idx: int,
                    candidates) -> List[Tuple[float, str, tuple, tuple]]:
    """Score every candidate direction by the face area a camera there would see.

    ``d`` is the direction the camera sits in, not the direction it looks: the renderer
    places the camera at ``focal + d * dist``, so a triangle is visible when its outward
    normal points *toward* ``d``. The projected (foreshortened) area is therefore
    ``area * max(0, n . d)`` -- zero when the triangle faces away, maximal when the camera
    looks straight down the normal. Getting this sign backwards ranks the directions that
    see the *back* of the face, which renders as the far side of the solid and shows the
    face in none of the chosen views. Summing over the face's triangles gives its
    apparent size from that direction, which is exactly the quantity that decides whether
    the face is judgeable in the image.

    Vectorised over all directions at once: for 266 directions and a few hundred
    triangles this is one small matrix product, microseconds, versus ~8 seconds to
    render and measure them. Occlusion by *other* parts of the model is not modelled
    here -- that is what the red-pixel check on the actual renders is for.
    """
    tri = face_triangles(polydata, cell_face_ids, face_idx)
    if tri is None:
        return []
    normals, areas, _ = tri
    dirs = np.asarray([c[1] for c in candidates], dtype=float)   # (D, 3)
    cos = normals @ dirs.T                                       # (T, D)
    np.clip(cos, 0.0, None, out=cos)
    scores = areas @ cos                                         # (D,)
    ranked = [(float(s), c[0], c[1], c[2]) for s, c in zip(scores, candidates)]
    ranked.sort(key=lambda t: -t[0])
    return ranked


def spread_select(ranked, k: int, min_sep_deg: float = 40.0):
    """Pick `k` well-scoring directions that are also far apart from each other.

    Taking the top-k by score alone returns k near-identical cameras clustered around
    the face normal, which is k images of the same picture and tells the model nothing
    new. Requiring angular separation buys genuinely different viewpoints -- needed to
    tell a chamfer (flat band) from a fillet (curved band), which look alike head-on and
    differ in profile.
    """
    chosen: List[tuple] = []
    cos_min = math.cos(math.radians(min_sep_deg))
    for entry in ranked:
        if entry[0] <= 0:
            break
        d = np.asarray(entry[2])
        if all(float(d @ np.asarray(c[2])) < cos_min for c in chosen):
            chosen.append(entry)
        if len(chosen) >= k:
            break
    # If separation was too strict to fill the quota, top up by score.
    if len(chosen) < k:
        have = {c[1] for c in chosen}
        for entry in ranked:
            if entry[0] <= 0:
                break
            if entry[1] not in have:
                chosen.append(entry)
                have.add(entry[1])
            if len(chosen) >= k:
                break
    return chosen


# ============================================================
# Rendering (in memory only)
# ============================================================

def _colour_face(polydata, cell_face_ids, face_idx):
    import vtk

    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("Colors")
    for fi in cell_face_ids:
        colors.InsertNextTuple3(*((220, 50, 50) if fi == face_idx else (200, 200, 210)))
    polydata.GetCellData().SetScalars(colors)


def render_directions(polydata, cell_face_ids, face_idx, extent, entries,
                      image_size=(700, 700), zoom=True, min_red=40):
    """Render the given directions in memory; drop views where the face is not visible.

    ``min_red`` discards views whose highlighted-face pixel count is negligible: the
    analytic ranking says the face *faces* the camera, but another part of the model may
    be in front of it. Sending an image where the subject is hidden is worse than sending
    one fewer image, because the model will still answer.
    """
    focal, fscale = (_face_frame(polydata, cell_face_ids, face_idx, extent)
                     if zoom else (None, None))
    out = []
    for score, name, cam_dir, cam_up in entries:
        png = _render_to_png_bytes(polydata, cam_dir, cam_up, extent,
                                   size=image_size, focal=focal, scale=fscale)
        if not png:
            continue
        red = _red_pixels(png)
        if zoom and red < min_red:
            continue
        out.append({"name": name, "png": png, "red": red, "score": score,
                    "cam_dir": cam_dir, "cam_up": cam_up})
    return out


# ============================================================
# Seeing what an exterior camera cannot: sections, bore interiors, zooms
# ============================================================
#
# An exterior camera is structurally unable to see some of the faces being classified. A
# bore wall hides behind its own opening; a counterbore floor sits at the bottom of a
# shaft; a pocket fillet is occluded by the pocket walls from every direction. On those
# faces every one of the 266 candidate directions is bad, and the ranking can only pick the
# least bad -- which is how the earlier pass ended up asking the model to classify a face
# rendered as a 1-pixel sliver, or as nothing at all.
#
# Three constructions fix that, and none of them is available to a plain orbit:
#
#   * cutaway    -- clip the solid with a plane just in front of the face, deleting
#                   whatever was occluding it while leaving the face itself whole.
#   * section    -- clip through the face, exposing the true cross-section, which is how a
#                   draughtsman shows a bore: the profile of the hole becomes visible as a
#                   line, along with its depth and whether it breaks through.
#   * endoscope  -- put a perspective camera *inside* the feature, on its own axis, looking
#                   along it. For a hole this renders the inside of the tube directly.
#
# Plus a tight zoom, because a 1%-of-frame face carries almost no pixels to judge.


def _face_axis(normals: np.ndarray) -> Optional[np.ndarray]:
    """Axis of a surface of revolution, estimated from its triangle normals.

    Every normal of a cylinder or cone is perpendicular to the axis, so the axis is the
    direction of least variance in the normals -- the last right-singular vector. Estimated
    from the mesh rather than the kernel because this is used for framing the camera, which
    has to work for bsplines and swept faces that have no analytic axis at all.
    """
    if normals.shape[0] < 3:
        return None
    try:
        _u, s, vt = np.linalg.svd(normals - normals.mean(axis=0), full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if s[0] <= 1e-12:
        return None
    axis = vt[-1]
    n = float(np.linalg.norm(axis))
    return axis / n if n > 1e-12 else None


def _clip(polydata, origin, normal, keep_far: bool = True):
    """Half-space clip of the mesh, preserving the per-cell colours.

    ``keep_far`` keeps the half on the far side of the plane from ``normal``, i.e. cuts
    away the material between the camera and the plane.
    """
    import vtk

    plane = vtk.vtkPlane()
    plane.SetOrigin(float(origin[0]), float(origin[1]), float(origin[2]))
    plane.SetNormal(float(normal[0]), float(normal[1]), float(normal[2]))
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(polydata)
    clipper.SetClipFunction(plane)
    # vtkClipPolyData keeps f(x) > 0 by default; the camera side is f(x) > 0 when the plane
    # normal points at the camera, so keeping the far half means inverting it.
    clipper.SetInsideOut(bool(keep_far))
    clipper.GenerateClippedOutputOff()
    clipper.Update()
    return clipper.GetOutput()


def _render_view(polydata, cam_pos, focal, cam_up, extent,
                 size=(700, 700), scale=None, perspective=False,
                 view_angle=90.0) -> bytes:
    """Render from an explicit camera position, optionally in perspective.

    ``reclassify_visual._render_to_png_bytes`` only accepts a *direction* and always uses
    parallel projection, which cannot express a camera standing inside a bore: parallel
    projection has no viewpoint, so "inside" and "outside" render identically. Interior
    views need a real perspective frustum with a wide angle and a near clipping plane
    inside the feature.
    """
    import vtk

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.SetScalarModeToUseCellData()
    mapper.ScalarVisibilityOn()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetAmbient(0.45)
    prop.SetDiffuse(0.55)
    # A clipped solid is an open shell and an interior camera looks at the *back* of every
    # triangle it sees. With culling on (or one-sided lighting) those faces come out black
    # or missing, so the whole point of the view is lost.
    prop.BackfaceCullingOff()
    prop.FrontfaceCullingOff()

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1.0, 1.0, 1.0)
    renderer.SetTwoSidedLighting(True)

    ren_win = vtk.vtkRenderWindow()
    ren_win.SetOffScreenRendering(1)
    ren_win.AddRenderer(renderer)
    ren_win.SetSize(size[0], size[1])

    cam = renderer.GetActiveCamera()
    cam.SetPosition(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]))
    cam.SetFocalPoint(float(focal[0]), float(focal[1]), float(focal[2]))
    cam.SetViewUp(float(cam_up[0]), float(cam_up[1]), float(cam_up[2]))
    if perspective:
        cam.SetParallelProjection(False)
        cam.SetViewAngle(view_angle)
        cam.SetClippingRange(extent * 1e-4, extent * 20)
    else:
        cam.SetParallelProjection(True)
        cam.SetParallelScale(scale if scale is not None else extent * 0.6)
        cam.SetClippingRange(extent * 1e-3, extent * 40)
    ren_win.Render()

    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(ren_win)
    w2if.Update()
    writer = vtk.vtkPNGWriter()
    writer.WriteToMemoryOn()
    writer.SetInputConnection(w2if.GetOutputPort())
    writer.Write()
    png = bytes(memoryview(writer.GetResult()))
    ren_win.Finalize()
    return png


def special_views(polydata, cell_face_ids, face_idx, extent, best_dir, best_up,
                  facts: Dict, image_size=(700, 700)) -> List[Dict]:
    """Cutaway, section, endoscope and tight-zoom views of one face.

    Returned in the order they are worth looking at, each with a ``why`` string that goes
    into the prompt -- an unexplained cross-section is easy to misread as the part's real
    outer shape, which would be worse than not sending it.
    """
    normals, areas, verts = face_triangles(polydata, cell_face_ids, face_idx)
    if verts.size == 0:
        return []
    centroid = verts.reshape(-1, 3).mean(axis=0)
    span = float(np.linalg.norm(verts.reshape(-1, 3).ptp(axis=0))) or extent * 0.1
    d = np.asarray(best_dir, dtype=float)
    up = np.asarray(best_up, dtype=float)
    out: List[Dict] = []

    def add(name, png, why):
        if png and _red_pixels(png) > 0:
            out.append({"name": name, "png": png, "red": _red_pixels(png),
                        "score": 0.0, "why": why, "cam_dir": tuple(d),
                        "cam_up": tuple(up)})

    # 1. Tight zoom on the face alone. A face at 1% of the frame is a few dozen pixels;
    #    whether its edges are sharp or blended is simply not resolvable at that size.
    add("zoom_tight",
        _render_view(polydata, centroid + d * extent * 5, centroid, up, extent,
                     size=image_size, scale=max(span * 0.7, extent * 1e-3)),
        "close crop on the red face only, to judge its edges and curvature")

    # 2. Cutaway: delete everything between the camera and the face. This is the fix for an
    #    occluded face -- the face survives intact, its occluders do not.
    add("cutaway",
        _render_view(_clip(polydata, centroid + d * (span * 0.55), d, keep_far=True),
                     centroid + d * extent * 5, centroid, up, extent,
                     size=image_size, scale=max(span * 1.6, extent * 0.05)),
        "the solid is cut away in front of the red face, removing whatever hid it; "
        "the flat grey boundary is the cut, not a real face")

    # 3. Section through the face itself: shows depth and whether the feature breaks
    #    through, which distinguishes a through hole from a blind one and a pocket from a
    #    step -- none of which an exterior view can settle.
    side = np.cross(d, up)
    n_side = float(np.linalg.norm(side))
    if n_side > 1e-9:
        side /= n_side
        add("section",
            _render_view(_clip(polydata, centroid, side, keep_far=True),
                         centroid + d * extent * 5, centroid, up, extent,
                         size=image_size, scale=max(span * 2.2, extent * 0.1)),
            "cross-section cut through the red face: use it to read the feature's DEPTH "
            "and whether it breaks through the part")

    # 4. Endoscope: a perspective camera inside the feature, on its own axis. For a bore
    #    this renders the inside of the tube; nothing outside the part can show that.
    axis = _face_axis(normals)
    concave = facts.get("concave_surface")
    if axis is not None and concave is not False:
        radius = facts.get("radius")
        step = float(radius) * 1.5 if isinstance(radius, (int, float)) and radius > 0 \
            else span * 0.6
        a_up = np.cross(axis, d)
        if float(np.linalg.norm(a_up)) < 1e-6:
            a_up = np.cross(axis, np.array([0.0, 0.0, 1.0]))
        n_up = float(np.linalg.norm(a_up))
        if n_up > 1e-9:
            a_up /= n_up
            for sign in (1.0, -1.0):
                add(f"endoscope_{'+' if sign > 0 else '-'}axis",
                    _render_view(polydata, centroid + axis * (sign * step),
                                 centroid - axis * (sign * step), a_up, extent,
                                 size=image_size, perspective=True, view_angle=100.0),
                    "camera placed INSIDE the feature on its own axis, looking along it: "
                    "a tube of red means a bore wall, a red ring or disc means an end face")
                if len(out) >= 6:
                    break
    return out


# ============================================================
# Objective geometry facts (no class hypotheses)
# ============================================================

def _curvature_facts(normals: np.ndarray, areas: np.ndarray,
                     verts: np.ndarray) -> Tuple[Optional[float], Optional[bool]]:
    """(wrap angle in degrees around the face's own axis, is it concave).

    Both are None for a planar face, where neither means anything.

    The axis is the direction the normals are most nearly perpendicular to -- the
    smallest right-singular vector of the area-weighted normal matrix, which for any
    surface of revolution is its axis. Projecting the normals into the plane across that
    axis and measuring the angular span (as 360 minus the largest gap, so a span
    straddling 0 is not mistaken for a full sweep) gives how far around the surface goes.

    Concavity uses the sign of ``n . (triangle centroid - face centroid)``: on a bore the
    normals point inward toward the axis while the centroid offset points outward, giving
    a negative dot; on an external round they agree.
    """
    if len(normals) < 3:
        return None, None
    mean_n = normals.mean(axis=0)
    if float(np.linalg.norm(mean_n)) > 0.999:      # every normal identical: a plane
        return None, None

    w = normals * areas[:, None]
    try:
        axis = np.linalg.svd(w, full_matrices=False)[2][-1]
    except np.linalg.LinAlgError:
        return None, None

    perp = normals - np.outer(normals @ axis, axis)
    keep = np.linalg.norm(perp, axis=1) > 1e-6
    if keep.sum() < 3:
        return None, None
    perp = perp[keep] / np.linalg.norm(perp[keep], axis=1, keepdims=True)
    # Any orthonormal basis of the plane across the axis will do; angles are only used
    # for their spacing, not their absolute values.
    e1 = perp[0]
    e2 = np.cross(axis, e1)
    ang = np.sort(np.degrees(np.arctan2(perp @ e2, perp @ e1)) % 360.0)
    gaps = np.diff(np.concatenate([ang, ang[:1] + 360.0]))
    wrap = float(360.0 - gaps.max())

    tri_c = verts.reshape(-1, 3, 3).mean(axis=1)[keep]
    off = tri_c - verts.mean(axis=0)
    norm = np.linalg.norm(off, axis=1)
    ok = norm > 1e-9
    concave = None
    if ok.any():
        radial = float(np.sum((normals[keep][ok] * off[ok]).sum(axis=1)
                              / norm[ok] * areas[keep][ok]))
        concave = bool(radial < 0)
    return round(wrap, 1), concave


def face_facts(polydata, cell_face_ids, face_idx, extent, step_path) -> Dict:
    """Measurements the model cannot read off an image, with no class suggested.

    Deliberately excludes any rule-derived candidate class. The geometric rules were
    measured to overwrite 15% of correct mftrcad labels and 100% of ``chamfer_2``, and
    their errors are systematic, so naming their guess would anchor the model toward the
    same wrong answers on exactly the rare faces that matter. Area share, aspect ratio
    and surface type are facts, not guesses, and are the cues a human inspector would ask
    for -- a chamfer is *narrow*, which an aspect ratio states unambiguously and a zoomed
    image hides.
    """
    tri = face_triangles(polydata, cell_face_ids, face_idx)
    if tri is None:
        return {}
    normals, areas, verts = tri
    total = 0.0
    for cid, fi in enumerate(cell_face_ids):
        cell = polydata.GetCell(cid)
        pid = cell.GetPointIds()
        if pid.GetNumberOfIds() < 3:
            continue
        p = np.array([polydata.GetPoints().GetPoint(pid.GetId(k)) for k in range(3)])
        total += float(np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0]))) / 2.0

    lo, hi = verts.min(axis=0), verts.max(axis=0)
    dims = np.sort(hi - lo)[::-1]
    longest = float(dims[0]) or 1e-9
    # Curvature proxy: how much the triangle normals disagree. A planar face has one
    # normal; a cylinder's fan out. This separates chamfer from fillet, which is the
    # distinction the whole 5-class task keeps losing.
    mean_n = normals.mean(axis=0)
    mean_n /= (np.linalg.norm(mean_n) or 1.0)
    spread_deg = float(np.degrees(np.arccos(
        np.clip(normals @ mean_n, -1.0, 1.0))).max())

    # normal_spread alone cannot tell a hole from a fillet -- a bore wall and a rounded
    # edge are both "curved and narrow", and the first live test had holes confidently
    # called fillets on exactly that evidence. Two more facts separate them:
    #
    #  * wrap_angle: how far the surface turns around its own axis. A through-hole wall
    #    sweeps ~360 degrees; a fillet band sweeps 90 or less. This is the single
    #    strongest discriminator and neither is readable from an image, because a bore
    #    is half-hidden by its own far wall in every view.
    #  * concave: whether the surface curves away from the material (a bore, a pocket
    #    blend) or toward it (an external round). A hole is always concave.
    wrap_deg, concave = _curvature_facts(normals, areas, verts)

    return {
        "wrap_angle_around_own_axis_degrees": wrap_deg,
        "concave": concave,
        "area_fraction_of_part": round(float(areas.sum() / (total or 1.0)), 5),
        "face_size_vs_part": round(longest / (extent or 1e-9), 4),
        # Narrowness is length over WIDTH, not over the bounding box's thinnest
        # dimension: a planar face has zero thickness, so dividing by it produced
        # ratios like 1.2e11 for every flat face -- a number that cannot distinguish
        # a narrow chamfer from a whole side of the part, which is the one thing this
        # measurement exists to do.
        "length_vs_width": round(longest / (float(dims[1]) or 1e-9), 1),
        "thickness_vs_length": round(float(dims[2]) / longest, 4),
        "normal_spread_degrees": round(spread_deg, 1),
        "planar": bool(spread_deg < 5.0),
        "n_triangles": int(len(areas)),
    }


# ============================================================
# Prompting
# ============================================================

#: Per-class geometry distributions measured on the hand-labelled defeature corpus by
#: ``data/measure_class_geometry.py``. Loaded lazily and cached; absent file is not fatal,
#: the prompt just omits the measured block.
#: One stats file per label space. They cannot share one: the 27-class posteriors are
#: measured on mftrcad and the 5-class ones on the defeature corpus, and quoting 5-class
#: shares while asking for a 27-class answer would state, for example, "hole 87%" for a row
#: whose faces are really split across `through_hole`, `blind_hole` and `counterbore`.
_STATS_CACHE: Dict[str, Optional[Dict]] = {}
_STATS_DIR = Path(__file__).resolve().parent / "data"
STATS_PATHS = {"5": _STATS_DIR / "class_geometry_stats.json",
               "27": _STATS_DIR / "class_geometry_stats_27.json"}
STATS_PATH = STATS_PATHS["5"]      # kept for callers that predate the 27-class table


def _load_stats(label_space: str = "5") -> Optional[Dict]:
    if label_space not in _STATS_CACHE:
        path = STATS_PATHS.get(label_space)
        try:
            _STATS_CACHE[label_space] = json.loads(path.read_text())
        except (OSError, ValueError, AttributeError):
            _STATS_CACHE[label_space] = None
    return _STATS_CACHE[label_space]


#: A fact has to pay for its line in the prompt. Below this many bits of information gain
#: about the label it does not: `on_convex_hull` alone scores 0.05 bits over 115k labelled
#: faces -- knowing it moves the answer almost nowhere -- while `surface_type x
#: accessibility` scores 0.97 and outranks every fact that was hand-picked before measuring.
#: Quoting the flat ones anyway is not free: it spends prompt on rows that read as evidence
#: and are not, which is how the old hand-written rules did their damage.
MIN_INFO_GAIN_BITS = 0.15


def _info_gain(table: Dict) -> float:
    """Bits of information the fact carries about the label, over the faces it covers.

    H(class) - H(class | fact), both estimated from the table itself so the comparison is
    over the same face population; a fact measured on only the cylinders (``half_angle``) is
    not penalised for saying nothing about planes it never sees.
    """
    total = sum(b["n"] for b in table.values())
    if not total:
        return 0.0

    def H(ps) -> float:
        return -sum(p * math.log2(p) for p in ps if p > 0)

    marginal: Dict[str, float] = {}
    for b in table.values():
        for c, p in b["p"].items():
            marginal[c] = marginal.get(c, 0.0) + p * b["n"] / total
    conditional = sum(b["n"] / total * H(b["p"].values()) for b in table.values())
    return H(marginal.values()) - conditional


def _ranked_tables(post: Dict, label_space: str) -> List[tuple]:
    """(fact, row limit) for the tables worth quoting, most decisive first.

    Ranked by measurement rather than by hand. The joint tables win on their own merits --
    `cylinder` alone is 46% hole, `cylinder` swept a full 360 degrees is 87% hole -- so they
    do not need to be privileged in a hand-written order, and a new descriptor added to
    ``measure_class_geometry`` reaches the prompt exactly when it earns its place and not
    because someone believed in it.
    """
    scored = [(_info_gain(t), f, t) for f, t in post.get("facts", {}).items()]
    scored = [s for s in scored if s[0] >= MIN_INFO_GAIN_BITS]
    scored.sort(reverse=True)
    # The 27-class space spreads the same faces over five times as many labels, so more rows
    # are needed before the live set is covered, and more tables before the rare classes are
    # mentioned at all.
    scale = 2 if label_space == "27" else 1
    # 27 classes needs more rows per table but not proportionally more tables: at two rows
    # per table per class the block would run past 20k tokens on every one of the 3-5 calls a
    # face costs, and the tail tables are the weak ones by construction.
    cap = 11 if scale == 1 else 14
    return [(f, min(8, len(t)) * scale) for _, f, t in scored[:cap]]


def _stats_block(label_space: str) -> str:
    """The measured ground truth, as P(class | geometry) tables.

    These replace hand-written rules, which is not a stylistic choice. Rules written from
    a 40-model sample asserted "a cone means chamfer" and "chamfers have no tangential
    edges"; measured over all 1,561 labelled models, a cone is 46% chamfer and the median
    chamfer has *two* tangential edges. Both rules were confidently wrong, and the prompt
    would have stated them as fact -- the same way ``geometric_rules.py`` came to overwrite
    15% of correct labels.

    Posteriors rather than per-class profiles, because the model's actual question is "I
    am looking at a cone, what is it?" and not "what do chamfers look like?". With `random`
    and `cut` an order of magnitude larger than `chamfer`, those two questions have
    different answers, and only the first one is useful here.
    """
    stats = _load_stats(label_space)
    post = (stats or {}).get("posteriors")
    if not post:
        return ""
    meta = stats.get("meta", {})
    # With 27 classes most of the prior is a long tail of sub-1% labels; listing all of it
    # buries the handful of classes that are actually common.
    prior_items = sorted(((c, p) for c, p in post["prior"].items()
                          if c in stats["classes"]), key=lambda kv: -kv[1])
    if label_space == "27":
        prior_items = [kv for kv in prior_items if kv[1] >= 0.005]
    prior = ", ".join(f"{stats['classes'][c]['name']} {100 * p:.1f}%"
                      for c, p in prior_items)

    out = [f"""
MEASURED GROUND TRUTH from {meta.get('faces', '?')} hand-labelled faces across
{meta.get('models', '?')} models of this same dataset. Each row reads: given a face with
this geometry, here is how the labelled faces were actually classified. `n` is how many
faces the row is based on.

Base rate over all labelled faces: {prior}.
"""]
    for fact, limit in _ranked_tables(post, label_space):
        table = post["facts"].get(fact)
        if not table:
            continue
        out.append(f"  P(class | {fact}):  [{_info_gain(table):.2f} bits]")
        for label, e in list(table.items())[:limit]:
            share = ", ".join(f"{k} {int(100 * v)}%" for k, v in e["p"].items())
            out.append(f"    {label:<46} n={e['n']:<6} -> {share}")
        out.append("")
    out.append(
        "Those are base rates, not rules. Where a row is lopsided (85%+) it is close to\n"
        "decisive on its own. Where the top class is under ~60% the geometry genuinely\n"
        "does not settle it, and that is where the images have to do the work.\n")
    return "\n".join(out)


def _classify_prompt(label_space: str, class_names: List[str], facts: Dict,
                     views: List[Dict], allow_requests: bool,
                     previous: Optional[List[Dict]] = None,
                     rare: Optional[Dict] = None) -> str:
    # Constructed views must be labelled with what they are. An unexplained cross-section
    # reads as the part's real outer shape, and an endoscope frame reads as a solid tube;
    # either misreading is worse than not sending the image.
    view_lines = "\n".join(
        f"  image {i + 2}: {v['name']}"
        + (f" -- {v['why']}" if v.get("why") else
           " (azimuth/elevation are in the name)")
        + f" [red face = {v['red']} px]"
        for i, v in enumerate(views))
    facts_lines = "\n".join(f"  {k}: {v}" for k, v in facts.items() if v is not None)
    stats = _stats_block(label_space)

    request_block = ""
    if allow_requests:
        request_block = f"""
If the images genuinely do not let you decide, you may request up to 4 more camera
angles instead of answering. Prefer this over guessing. Reply with EXACTLY:
{{"need_views": [{{"azimuth": <0-359>, "elevation": <-90..90>}}, ...], "why": "<reason>"}}
Azimuth rotates around the part's Z axis; elevation is above (+) or below (-) the
equator. You may request this at most once."""

    # How well the corpus covers this face, stated plainly. Two faces with identical images
    # deserve different amounts of caution when one sits in a row of 3,000 labelled faces and
    # the other in a row of six, and the model cannot tell those apart from the tables alone:
    # every row it reads carries an `n`, but nothing says whether the row it landed in is the
    # big one. Note that this reports coverage, never the expected answer -- naming the
    # majority class here would anchor the reply on it and turn a measured base rate back into
    # the kind of rule that has already corrupted this dataset once.
    coverage = ""
    if rare:
        if rare.get("novel"):
            coverage = f"""
COVERAGE WARNING: the corpus has only {rare['joint_support']} labelled faces whose
measurements combine the way this one's do. The tables above are thin here and their
percentages are not reliable at this sample size, so the images and the definitions of the
classes have to carry this face. If they do not settle it, abstain -- on an unusual face that
is the expected answer, not a shortfall.
"""
        elif rare.get("rare_expected"):
            coverage = """
NOTE: the classes live for this geometry are all rare ones. Rare classes are under-labelled
precisely because they are hard, so weigh the images heavily and do not fall back on a common
class merely because it is common -- that error is what erases the rare classes entirely.
"""
        elif rare.get("dominant_n"):
            coverage = f"""
NOTE: this face's measurements fall in a row the corpus has {rare['dominant_n']} labelled
examples of, and that row is strongly one-sided. This is a well-understood geometry, so the
matching row is strong evidence; departing from it is allowed but needs a specific reason you
can name from the images, not a general impression.
"""

    prior = ""
    if previous:
        prior = ("\nEarlier passes on this same face answered: "
                 + "; ".join(f"class {p.get('class')} (conf {p.get('confidence')})"
                             for p in previous)
                 + ".\nJudge the images yourself; do not simply agree.")

    return f"""You are classifying ONE face of a B-Rep CAD model.

The face highlighted in RED is the subject. Image 1 shows the whole part unzoomed, for
context -- where on the part this face sits. The remaining views were chosen by computing
which camera directions see the largest area of this face, over a 15-degree grid covering
the whole sphere; some may instead be constructed cut-away, cross-section or interior
views, in which case the line below says which:
{view_lines}

Objective measurements of the red face (from the CAD geometry, not estimated from the
images):
{facts_lines}

Classify the RED face into exactly one of these {label_space} categories:
{chr(10).join(f"  {i}: {name}" for i, name in enumerate(class_names))}

{stats}{coverage}
What each measurement means, so you can look the red face up in the tables above:

- `coaxial_group_sweep_degrees`: the total angular sweep of this surface PLUS every
  coaxial surface joined to it tangentially. The kernel splits a full bore into two
  half-cylinders, so a hole wall alone reads as 180 while the group reads as 360. This is
  the fact you cannot see in any image -- a bore hides its own far wall from every camera
  -- and it is the one that most changes the answer.
- `n_tangential_edges`: edges where this face's normal matches its neighbour's, i.e.
  where it joins smoothly rather than at a visible crease. A blend has them by
  construction; a sharp-edged cut face does not.
- `concave_surface`: for a curved face, whether the material is outside it (a bore or
  pocket) or inside it (a shaft or external round).
- `n_convex_edges` / `n_concave_edges`: whether the boundary forms ridges or valleys,
  which says whether material was removed around this face.
- `half_angle_degrees`: a cone's taper. Around 45 degrees is the classic chamfer cut.
- `area_fraction_of_part`: share of the part's total surface area on this one face.
- `accessibility`: fraction of outward directions from which a straight line can reach this
  face from outside the solid, by ray-casting the part itself. 1.0 is an open outer surface;
  a bore wall reads ~0.5 because its own opening and far wall block most directions; a
  pocket floor or counterbore seat is lower still. This is *enclosure*, which no single view
  shows, and it is the second most informative measurement in the whole table.
- `n_inner_wires`: how many closed loops other than the outer boundary this face has, i.e.
  how many holes or pockets pass through it. A stock plate face pierced by two bores reads
  2; the bore walls themselves read 0.
- `mean_curvature_sign` / `is_saddle`: curvature at the face's mid-point, sign-corrected for
  orientation. An internal pocket fillet is a saddle; an external round of the identical
  radius is not, and in an image they look the same.
- `hull_depth_vs_extent` / `on_convex_hull`: how far the face sits below the part's convex
  hull, as a fraction of the part size. Zero means it is on the outer silhouette.
- `strict_component_size` / `strict_component_surfaces`: the face's group under the
  attributed-adjacency rule -- how many faces are joined to it across sharply concave edges,
  and which surface types they are. A machined feature is usually such a group (a bore's wall
  plus its floor), not one face alone.

How to decide:

1. Find the rows above that your measurements match, starting with the two-fact rows --
   they are much sharper than either fact alone. Read off which classes are live.
2. If one row is lopsided (85%+ for one class), that is your answer unless the images
   plainly contradict it.
3. If the live classes are still close, use the images to choose between them. That is
   what they are for: where on the part the face sits, whether it is inside a pocket or on
   the outer profile, what surrounds it, whether it repeats in a pattern.
4. Where an image and a measurement disagree, prefer the measurement. It comes from the
   CAD kernel and is exact; the face may be partly hidden in every view.
5. `random` means an ordinary body or stock surface rather than a machining feature, and
   it is the most common label -- do not invent a feature to explain an unremarkable face.
   But a large plain face can equally be the FLOOR of a pocket, so check whether its
   boundary is concave before defaulting to it.
6. ABSTAIN rather than guess. Answer `"class": -1` when any of these is true:
     - the red face is not clearly visible in any image AND the measurements do not settle
       it on their own;
     - the best-matching table rows leave two or more classes within ~15 points of each
       other and the images do not break the tie;
     - the face's measurements do not resemble any row above (an unusual bspline, a
       degenerate sliver), so the tables simply do not speak to it;
     - you would be picking a class mainly because it is the most common one.
   Abstaining is the correct answer, not a failure: an abstained face is dropped from the
   training set and costs it one example, while a confidently wrong one teaches it
   something false and costs far more. There is no quota — abstain as often as the
   evidence requires.
7. Otherwise state the class and a calibrated confidence: 0.9+ only when a lopsided table
   row and the images agree, ~0.6-0.75 when you are choosing between two live classes on
   image evidence, below 0.5 when you are close to abstaining.
{prior}{request_block}

Otherwise respond with EXACTLY this JSON and nothing else:
{{"class": <int, or -1 to abstain>, "confidence": 0.0-1.0,
 "runner_up": <int, the second most likely class, or -1 if none>,
 "reasoning": "<1 short sentence naming the measurement that decided it>"}}"""


# ============================================================
# Independent check on the answer: objections, not overrides
# ============================================================
#
# The vote across passes measures only self-consistency, and a model can be consistently
# wrong -- in the first live run two ground-truth holes came back as `fillet` from all three
# passes at 0.92 confidence. Agreement said the answer was certain; it was certain and
# wrong. So the answer is also checked against something that did not come from the model:
# the labelled corpus, and the handful of facts that are true by definition rather than by
# measurement.
#
# Crucially, an objection never rewrites the class. ``geometric_rules.py`` was allowed to
# rewrite and was measured to destroy 15% of correct labels and 100% of `chamfer_2`; the
# rules were confident and wrong in exactly the systematic way a rule is. Here an objection
# only buys the face one more look: it is stated back to the model, with more views, and the
# model may hold its answer or change it. A face where the objection stands and the model
# will not budge is a face to abstain on, not a face to overwrite.


def _matching_rows(label_space: str, facts: Dict) -> List[Dict]:
    """The measured posterior rows this face falls into, most specific first.

    Bins come from ``measure_class_geometry.CONDITIONS``, imported rather than restated, so
    the row a face is looked up in here is exactly the row it was counted in there. Copying
    the bin edges into this module would let the two drift apart and produce objections
    computed from a row the face does not belong to.
    """
    stats = _load_stats(label_space)
    post = (stats or {}).get("posteriors")
    if not post:
        return []
    try:
        from brepformer.data.measure_class_geometry import CONDITIONS, _bin_label
    except ImportError:
        return []
    edges = dict(CONDITIONS)

    def bin_of(key):
        v = facts.get(key)
        if v is None:
            return None
        e = edges.get(key)
        if e is None:
            return str(v)
        return _bin_label(float(v), e) if isinstance(v, (int, float)) else None

    rows = []
    for fact, table in post["facts"].items():
        if " x " in fact:
            a, b = fact.split(" x ", 1)
            la, lb = bin_of(a), bin_of(b)
            key = f"{facts.get(a)} & {b}={lb}" if la is not None and lb is not None else None
        else:
            lb = bin_of(fact)
            key = lb
        if key is None:
            continue
        entry = table.get(key)
        # Same information-gain floor as the quoted tables. A row from a fact that barely
        # moves the label is not evidence about this face either, and an objection raised
        # from one would be an objection from noise.
        if entry and _info_gain(table) >= MIN_INFO_GAIN_BITS:
            rows.append({"fact": fact, "row": key, "n": entry["n"], "p": entry["p"],
                         "joint": " x " in fact, "bits": round(_info_gain(table), 3)})
    # Joint rows first, then by sample size: the sharpest evidence the corpus has about
    # this particular face.
    rows.sort(key=lambda r: (not r["joint"], -r["n"]))
    return rows


#: A face whose geometry the corpus has seen fewer than this many times is being classified
#: with essentially no measured support: the posterior rows it falls into are estimated from
#: a handful of faces and can easily be wrong by a wide margin.
NOVEL_SUPPORT = 40
#: Above this row size, with the row this lopsided, the corpus is close to decisive and a
#: face landing in it is a *common* face -- the ones that must not be got wrong, because they
#: are the bulk of the loss and an error here teaches the student a wrong rule about the
#: geometry it sees most often.
DOMINANT_N = 500
DOMINANT_SHARE = 0.80


def rarity(label_space: str, class_names: List[str], facts: Dict) -> Dict:
    """How well the corpus covers this face, and what it expects -- before any model call.

    Drives how much effort the face gets. The two ends need opposite treatment and a single
    fixed budget serves neither:

    * A **rare or unseen** geometry is where the corpus cannot help and the model is on its
      own, so it gets more passes and the constructed views up front instead of only after
      an objection. What it must *not* get is a higher agreement bar: raising the bar on the
      faces the corpus under-covers is exactly the ratchet that drove `chamfer` and `cut`
      out of the pseudo-labels once already -- the rare classes fail a strict vote first, get
      masked, and vanish. Effort scales; thresholds do not.
    * A **common** geometry sitting in a large, lopsided row is where the answer is close to
      known in advance. There, a disagreeing answer is the thing that needs justifying, and
      the escalation is a challenge round rather than more passes.
    """
    rows = _matching_rows(label_space, facts)
    support = max((r["n"] for r in rows), default=0)
    joint = [r for r in rows if r["joint"]]
    # Support from the *specific* rows, since a face is only well covered if the corpus has
    # seen its combination of measurements, not merely each measurement separately.
    joint_support = max((r["n"] for r in joint), default=0)
    dominant = next((r for r in rows
                     if r["n"] >= DOMINANT_N
                     and max(r["p"].values(), default=0) >= DOMINANT_SHARE), None)
    expected = None
    if dominant:
        expected = max(dominant["p"], key=lambda k: dominant["p"][k])
    prior = ((_load_stats(label_space) or {}).get("posteriors") or {}).get("prior") or {}
    classes = ((_load_stats(label_space) or {}).get("classes") or {})
    by_name = {v["name"]: prior.get(k, 0.0) for k, v in classes.items()}
    # A face whose likely answers are all rare classes is a hard face even when its geometry
    # is common, because the rare classes are precisely the ones the corpus labels least
    # consistently and the student sees least often.
    top_classes = [c for r in rows[:3] for c, p in r["p"].items() if p >= 0.15]
    rare_expected = bool(top_classes) and all(by_name.get(c, 1.0) < 0.02
                                              for c in top_classes)
    return {"support": support, "joint_support": joint_support,
            "novel": joint_support < NOVEL_SUPPORT,
            "dominant_row": (f"{dominant['fact']} = {dominant['row']}"
                             if dominant else None),
            "dominant_n": dominant["n"] if dominant else 0,
            "dominant_share": (round(max(dominant["p"].values()), 3)
                               if dominant else 0.0),
            "expected": expected, "rare_expected": rare_expected,
            "hard": bool(dominant is None and (joint_support < NOVEL_SUPPORT
                                               or rare_expected))}


#: Facts that are true of a class *by construction*, not by measurement. Each entry is
#: (class-name substring, fact key, predicate on the value, why it contradicts). These are
#: deliberately few: only statements that follow from how the feature is modelled, because
#: anything merely *typical* is already covered -- and covered better -- by the posterior
#: rows, which carry a sample size.
_DEFINITIONAL: List[Tuple[str, str, str]] = [
    ("fillet", "n_tangential_edges",
     "a fillet or round is constructed to blend smoothly into what it joins, so it has at "
     "least one tangential edge; this face has none, which means it meets its neighbours "
     "at a visible crease"),
    ("round", "n_tangential_edges",
     "a round blends tangentially by construction; this face has no tangential edge"),
    ("hole", "concave_surface",
     "a hole is a bore, so its wall's outward normal points back at its own axis; this "
     "surface is convex, i.e. material is on the inside of it, which is a shaft or an "
     "external round, not a hole"),
]


def objections(label_space: str, class_names: List[str], cls: int, facts: Dict,
               views: List[Dict], rare: Optional[Dict] = None) -> List[str]:
    """Reasons to doubt ``cls`` for this face, drawn from outside the model's own reply."""
    if cls is None or cls < 0 or cls >= len(class_names):
        return []
    name = class_names[cls].lower()
    out: List[str] = []

    # The common-face guard. This face's geometry is one the corpus has seen hundreds of
    # times and labelled overwhelmingly one way, and the answer is not that way. Common faces
    # are most of the loss, so a mistake here is worth more than a mistake on a rare one, and
    # it is also the cheapest kind to catch: the disagreement is with a large sample, not with
    # a rule someone wrote.
    if rare and rare.get("expected") and rare["expected"] != class_names[cls]:
        out.append(
            f"This face's geometry ({rare['dominant_row']}) occurs {rare['dominant_n']} "
            f"times in the hand-labelled corpus and is `{rare['expected']}` in "
            f"{100 * rare['dominant_share']:.0f}% of them. You answered "
            f"`{class_names[cls]}`. That is not impossible, but it is the minority answer on "
            f"a large sample, so say specifically what you see that the {rare['dominant_n']} "
            f"faces in that row do not have -- and if you cannot point to something, the "
            f"majority answer is more likely right than your reading of the image.")

    for key, fact, why in _DEFINITIONAL:
        if key not in name:
            continue
        v = facts.get(fact)
        if fact == "n_tangential_edges" and isinstance(v, int) and v == 0:
            out.append(f"You answered `{class_names[cls]}`, but {why}.")
        if fact == "concave_surface" and v is False:
            out.append(f"You answered `{class_names[cls]}`, but {why}.")

    # The corpus objection: among faces measured to have this same geometry, how many were
    # actually this class? A share near zero on a large row is the strongest disagreement
    # available, and unlike a rule it comes with the number of faces behind it.
    for row in _matching_rows(label_space, facts)[:3]:
        share = row["p"].get(class_names[cls], 0.0)
        if row["n"] >= 150 and share < 0.03:
            live = ", ".join(f"{k} {int(100 * v)}%"
                             for k, v in list(row["p"].items())[:3])
            out.append(
                f"Of the {row['n']} hand-labelled faces with this same geometry "
                f"({row['fact']} = {row['row']}), only {100 * share:.1f}% were "
                f"`{class_names[cls]}`. The classes that actually occur there are: {live}. "
                f"Either a measurement of this face is unusual in a way the row misses, or "
                f"the answer is one of those instead.")
            break

    # Visibility: if the face never occupied a usable share of any frame, an answer that
    # leans on image evidence is not supported, whatever its stated confidence.
    best_red = max((v.get("red", 0) for v in views), default=0)
    if best_red < 1500:
        out.append(
            f"The red face covered at most {best_red} pixels in any view, so the images "
            f"could not have shown you its edges or surroundings. Decide from the "
            f"measurements alone, or abstain.")
    return out


def _challenge_prompt(class_names: List[str], cls: int, reasoning: str,
                      objs: List[str], views: List[Dict]) -> str:
    numbered = "\n".join(f"  {i + 1}. {o}" for i, o in enumerate(objs))
    extra = "\n".join(f"  image {i + 1}: {v['name']} -- {v.get('why', '')}"
                      for i, v in enumerate(views))
    return f"""You classified this red face as `{class_names[cls]}` ({cls}), reasoning:
"{reasoning}".

An independent check on that answer raised the following:
{numbered}

New views of the same face, constructed specifically to show what an ordinary exterior
camera cannot:
{extra}

These objections are not verdicts -- the check is mechanical and can be wrong, and one of
its predecessors was wrong often enough to corrupt labels when it was allowed to overrule.
Your job is to decide which is right.

Answer with EXACTLY this JSON and nothing else:
{{"class": <int, your final answer -- keep it or change it, or -1 to abstain>,
 "confidence": 0.0-1.0,
 "rebuttal": "<1 sentence: why the objection is wrong, or what you had missed>"}}"""


def _parse_reply(text: str) -> Optional[Dict]:
    """Pull the JSON verdict out of a reply, tolerating prose around or before it.

    A strict ``text[index("{"):rindex("}")]`` slice raised "substring not found" on ~30% of
    replies once the prompt grew: with a lot of evidence to weigh, the model narrates first
    and can be cut off by the token limit before it ever opens a brace. Those replies still
    contain the verdict in words, and discarding them silently shrinks the vote from three
    passes to one or two -- which looks like agreement rather than like missing data. So
    fall back to reading the fields out of the prose.
    """
    text = text.strip()
    for m in re.finditer(r"\{[^{}]*\}", text):
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            continue
        if isinstance(obj, dict) and ("class" in obj or "need_views" in obj):
            return obj
    # Truncated or prose-only reply: recover the two fields that matter.
    cls = re.search(r'"?class"?\s*[:=]\s*(\d+)', text)
    if cls:
        conf = re.search(r'"?confidence"?\s*[:=]\s*([0-9.]+)', text)
        return {"class": int(cls.group(1)),
                "confidence": float(conf.group(1)) if conf else 0.5,
                "reasoning": "recovered from unterminated reply", "truncated": True}
    return None


def _call_vision(client, model: str, images: List[bytes], prompt: str,
                 max_tokens: int = 1200) -> Optional[Dict]:
    content = []
    for png in images:
        if png:
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/png",
                "data": base64.b64encode(png).decode("utf-8")}})
    if not content:
        return None
    content.append({"type": "text", "text": prompt})
    try:
        resp = client.messages.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}])
        text = "".join(b.text for b in resp.content if getattr(b, "text", None))
        out = _parse_reply(text)
        if out is None:
            print(f"    unparseable reply ({resp.stop_reason}): "
                  f"{text[-160:]!r}", file=sys.stderr)
        return out
    except Exception as e:  # noqa: BLE001
        print(f"    vision API error: {type(e).__name__}: {e}", file=sys.stderr)
    return None


# ============================================================
# One face, several passes
# ============================================================

#: Each pass gets a different view *strategy*, so the passes are genuinely independent
#: evidence rather than three samples of the same picture. (label, n_views, min_sep_deg)
PASS_STRATEGIES = [
    ("best-visibility", 4, 0.0),     # the four directions that see the most face area
    ("wide-separation", 4, 60.0),    # spread around the sphere: profile + face-on
    ("mixed", 5, 30.0),              # compromise, plus one extra view
    # Reached only by the rarity escalation below, on faces whose geometry the corpus barely
    # covers. There the vote is all there is -- no measured row is going to settle it -- so
    # it is worth widening from three opinions to five, sampled differently enough that the
    # extra passes are not just the first three again.
    ("dense-orbit", 6, 20.0),
    ("far-separation", 5, 90.0),
]


def label_face(step_path: str, face_idx: int, label_space: str,
               class_names: List[str], model: str,
               passes: int = 3, allow_requests: bool = True,
               image_size: Tuple[int, int] = (700, 700),
               sphere_step: int = 15, verbose: bool = False,
               min_agreement: float = DEFAULT_MIN_AGREEMENT,
               min_confidence: float = DEFAULT_MIN_CONFIDENCE,
               challenge: bool = True) -> Optional[Dict]:
    """Label one face with `passes` independent vision passes, then vote.

    Returns ``{"class", "confidence", "agreement", "votes", "n_calls", "passes": [...]}``
    or None if the face could not be rendered at all.

    The returned ``confidence`` is deliberately *not* the model's stated confidence: a
    single pass reporting 0.9 has been measured to mean very little here. It is the share
    of passes agreeing on the winning class, scaled by their mean stated confidence, so a
    face that three passes label three different ways ends up near zero and stays masked.
    """
    polydata, cell_face_ids, extent, n_faces = _tessellate(step_path)
    if face_idx >= n_faces:
        return None
    _colour_face(polydata, cell_face_ids, face_idx)

    candidates = sphere_directions(sphere_step)
    ranked = rank_directions(polydata, cell_face_ids, face_idx, candidates)
    if not ranked or ranked[0][0] <= 0:
        return None

    # Mesh statistics first (they describe what the picture shows), then the kernel's own
    # answers on top, which override any proxy of the same quantity: an exact cylinder
    # radius beats a normal-spread estimate of "how curved is it".
    facts = face_facts(polydata, cell_face_ids, face_idx, extent, step_path)
    try:
        exact = face_geometry(step_path, [face_idx], with_topology=True).get(face_idx)
    except Exception as e:  # noqa: BLE001 - a kernel failure must not stop labelling
        if verbose:
            print(f"    exact geometry unavailable ({type(e).__name__}: {e})")
        exact = None
    if exact:
        exact.pop("edges", None)
        exact.pop("face", None)
        facts.update(exact)

    # How much the corpus knows about a face like this, decided before spending anything.
    # Effort is not uniform across faces because the difficulty is not: most faces are a
    # plane on the outside of the stock and the third pass never changes the answer, while a
    # geometry the corpus has seen four times gets no help from the measured rows at all.
    rare = rarity(label_space, class_names, facts)
    if rare["hard"]:
        passes = min(len(PASS_STRATEGIES), max(passes, passes + 2))

    client, resolved = _vision_client(model)

    # One whole-part context view, rendered once and reused by every pass: "where on the
    # part is this face" is the cue a zoomed crop destroys, and re-rendering it per pass
    # would be pure waste.
    ctx = render_directions(polydata, cell_face_ids, face_idx, extent,
                            [ranked[0]], image_size=image_size, zoom=False)
    ctx_png = [ctx[0]["png"]] if ctx else []

    results: List[Dict] = []
    all_views: List[Dict] = []
    n_calls = 0
    special_cache: Optional[List[Dict]] = None
    for pi in range(min(passes, len(PASS_STRATEGIES))):
        label, k, sep = PASS_STRATEGIES[pi]
        entries = spread_select(ranked, k, sep)
        views = render_directions(polydata, cell_face_ids, face_idx, extent, entries,
                                  image_size=image_size)
        # An occluded or tiny face gets the section / cutaway / interior views from the
        # start rather than only after an objection: waiting means the first passes answer
        # from images that do not show the face, and three such answers agreeing is exactly
        # the confident-and-wrong case. Rendered once and reused across passes.
        # ...and so does a face whose geometry the corpus cannot vouch for, even when it is
        # perfectly visible: with no measured row behind it, the images are the entire case,
        # so it should be seen from inside and in section too.
        if rare["hard"] or max((v["red"] for v in views), default=0) < 1500:
            if special_cache is None:
                special_cache = special_views(
                    polydata, cell_face_ids, face_idx, extent,
                    ranked[0][2], ranked[0][3], facts, image_size=image_size)
            views = views + special_cache
        if not views:
            continue
        all_views.extend(views)

        asked = False
        for attempt in range(2):        # answer, or one round of angle requests
            prompt = _classify_prompt(
                label_space, class_names, facts, views,
                allow_requests=allow_requests and not asked,
                previous=results or None, rare=rare)
            out = _call_vision(client, resolved, ctx_png + [v["png"] for v in views],
                               prompt)
            n_calls += 1
            if out is None:
                break
            if "need_views" in out and not asked:
                asked = True
                extra = []
                for req in (out.get("need_views") or [])[:4]:
                    try:
                        az, el = float(req["azimuth"]), float(req["elevation"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    d, up = direction_from_angles(az, el)
                    extra.append((0.0, f"req_az{az:.0f}_el{el:+.0f}", d, up))
                if verbose:
                    print(f"    pass {label}: model requested {len(extra)} more "
                          f"angles ({out.get('why', '')[:60]})")
                if not extra:
                    break
                views = views + render_directions(
                    polydata, cell_face_ids, face_idx, extent, extra,
                    image_size=image_size, min_red=1)
                continue
            if "class" in out:
                out["strategy"] = label
                out["n_views"] = len(views)
                results.append(out)
            break

    if not results:
        return None

    def tally(rows):
        v = collections.Counter()
        for r in rows:
            try:
                v[int(r["class"])] += 1
            except (TypeError, ValueError):
                continue
        return v

    votes = tally(results)
    if not votes:
        return None

    # An abstention is a vote for masking, not a missing vote. Dropping it would make one
    # confident pass out of three look unanimous, which is the failure mode the whole
    # multi-pass design exists to avoid.
    decided = {c: n for c, n in votes.items() if c >= 0}
    best = max(decided, key=lambda c: decided[c]) if decided else ABSTAIN
    agreement = votes[best] / sum(votes.values()) if best in votes else 0.0

    # The challenge round: check the leading answer against the corpus and against the facts
    # that hold by definition, and if anything objects, show the face again -- this time with
    # the section, cutaway and interior views -- and let it answer once more.
    challenged = None
    objs = (objections(label_space, class_names, best, facts, all_views, rare)
            if challenge and best >= 0 else [])
    if objs:
        sv = special_views(polydata, cell_face_ids, face_idx, extent,
                           ranked[0][2], ranked[0][3], facts, image_size=image_size)
        lead = next((r for r in results if int(r.get("class", -999)) == best), {})
        out = _call_vision(
            client, resolved,
            ctx_png + [v["png"] for v in sv],
            _challenge_prompt(class_names, best, str(lead.get("reasoning", ""))[:200],
                              objs, sv))
        n_calls += 1
        if verbose:
            print(f"    challenged ({len(objs)} objection(s), {len(sv)} special views)"
                  f" -> {out.get('class') if out else 'no reply'}")
        if out and "class" in out:
            out["strategy"] = "challenge"
            out["objections"] = objs
            out["n_views"] = len(sv)
            challenged = out
            results.append(out)
            votes = tally(results)
            decided = {c: n for c, n in votes.items() if c >= 0}
            # The challenge round saw strictly more evidence than any earlier pass, so it
            # breaks ties in its own favour rather than being outvoted by passes that could
            # not see inside the feature.
            try:
                cc = int(out["class"])
            except (TypeError, ValueError):
                cc = ABSTAIN
            if cc < 0:
                best = ABSTAIN
            elif decided:
                top = max(decided.values())
                best = cc if decided.get(cc, 0) >= top else max(
                    decided, key=lambda c: decided[c])
            agreement = (votes[best] / sum(votes.values())) if best in votes else 0.0

    agreeing = [r for r in results if int(r.get("class", -999)) == best]
    mean_conf = float(np.mean([float(r.get("confidence") or 0.0)
                               for r in agreeing])) if agreeing else 0.0
    confidence = round(agreement * mean_conf, 4)

    # Why a face is masked is recorded, not just that it was: "the passes disagreed" and
    # "the model abstained" and "an objection stood" are three different problems, and only
    # the first is fixed by more passes.
    reasons = []
    if best < 0:
        reasons.append("model abstained")
    if votes.get(ABSTAIN, 0):
        reasons.append(f"{votes[ABSTAIN]} of {sum(votes.values())} passes abstained")
    if agreement < min_agreement:
        reasons.append(f"agreement {agreement:.2f} < {min_agreement}")
    if confidence < min_confidence:
        reasons.append(f"confidence {confidence:.2f} < {min_confidence}")
    if objs and challenged is None:
        reasons.append("objection raised and the challenge round returned no answer")
    # The common-face guarantee. On a geometry the corpus has labelled hundreds of times and
    # labelled overwhelmingly one way, a minority answer is kept only if it survived the
    # challenge round *and* came back sure of itself. Otherwise the face is masked -- not
    # rewritten to the majority class, which is what `geometric_rules.py` did when it was
    # trusted to overrule and cost 15% of the correct labels it touched. Masking gives up one
    # training face; overwriting teaches the student a wrong rule about the geometry it sees
    # most often, which is the expensive mistake.
    if (rare.get("expected") and best >= 0
            and class_names[best] != rare["expected"]):
        held = (challenged is not None
                and int(challenged.get("class", -999)) == best
                and float(challenged.get("confidence") or 0.0) >= 0.85)
        if not held:
            reasons.append(
                f"answered `{class_names[best]}` on a geometry that is "
                f"`{rare['expected']}` in {100 * rare['dominant_share']:.0f}% of "
                f"{rare['dominant_n']} labelled faces, without a confident rebuttal")
    masked = bool(reasons)

    return {"class": best,
            "final_label": IGNORE_LABEL if masked else best,
            "masked": masked,
            "mask_reasons": reasons,
            "confidence": confidence,
            "agreement": round(agreement, 3),
            "mean_stated_confidence": round(mean_conf, 3),
            "votes": dict(votes),
            "objections": objs,
            "challenge": challenged,
            "rarity": rare,
            "n_passes": len(results),
            "n_calls": n_calls,
            "n_candidate_directions": len(candidates),
            "facts": facts,
            "passes": results}


# ============================================================
# Batch driver
# ============================================================

def _pick_faces(preds: List[int], probs: Optional[List[List[float]]],
                n: int) -> List[int]:
    """Lowest-confidence faces first: those are the ones masked out of training.

    Relabelling a face the teacher is already sure about cannot add supervision -- it is
    either already right, or wrong in a way one vision call will not fix. The faces below
    the mask cutoff are the only ones where a correct answer converts an unusable face
    into a usable one.
    """
    if not probs:
        return list(range(min(n, len(preds))))
    conf = [max(row) if row else 0.0 for row in probs]
    return sorted(range(len(conf)), key=lambda i: conf[i])[:n]


def _label_one_model(job: Dict) -> Dict:
    """Label the selected faces of one model. Runs in a worker process.

    One process per model rather than per face: the tessellation, the analytic ranking and
    the whole-part geometry (adjacency graph, convex hull, accessibility) are all per-model
    work that the faces of that model share, and ``_MESH_CACHE`` keeps the mesh alive across
    them. Processes rather than threads because the rendering is VTK and the ranking is
    numpy, so a thread pool would serialise on the interpreter for the CPU half of the work
    while the API half waits.
    """
    step = Path(job["step"])
    names = CLASS_NAMES if job["label_space"] == "27" else DEFEATURE_CLASS_NAMES
    faces: Dict[str, Dict] = {}
    calls = 0
    try:
        for fi in job["faces"]:
            r = label_face(str(step), fi, job["label_space"], names, job["model"],
                           passes=job["passes"],
                           allow_requests=job["allow_requests"],
                           image_size=tuple(job["image_size"]),
                           sphere_step=job["sphere_step"],
                           min_agreement=job["min_agreement"],
                           min_confidence=job["min_confidence"],
                           challenge=job["challenge"])
            if not r:
                continue
            calls += r["n_calls"]
            old = job["preds"].get(str(fi))
            faces[str(fi)] = {
                "vision_class": r["class"], "final_label": r["final_label"],
                "masked": r["masked"], "mask_reasons": r["mask_reasons"],
                "teacher_class": old, "confidence": r["confidence"],
                "agreement": r["agreement"], "votes": r["votes"],
                "n_objections": len(r["objections"]),
                "objections": r["objections"] or None,
                "challenge_class": (r["challenge"] or {}).get("class"),
                # Coverage travels with the verdict: `data/arbitrate_labels.py` treats a
                # verdict on a well-covered geometry differently from one on a geometry the
                # corpus has never seen, and it cannot recompute that without the facts.
                "rarity": r.get("rarity"), "n_passes": r.get("n_passes"),
                "n_calls": r["n_calls"]}
    except Exception as e:  # noqa: BLE001 - one bad model must not end the batch
        return {"model_id": step.stem, "error": f"{type(e).__name__}: {e}",
                "faces": faces, "calls": calls}
    finally:
        _MESH_CACHE.clear()
    return {"model_id": step.stem, "faces": faces, "calls": calls}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", help="Single STEP file")
    ap.add_argument("--step_dir", help="Directory of STEP files")
    ap.add_argument("--seg_dir", help="Existing inference JSONs, to pick faces and "
                                      "to compare against")
    ap.add_argument("--face", type=int, help="Single face index (with --step)")
    ap.add_argument("--out_dir", type=Path, help="Where to write label JSONs")
    ap.add_argument("--label_space", choices=["27", "5"], default="5")
    ap.add_argument("--passes", type=int, default=3,
                    help=f"Independent vision passes per face, each with a different "
                         f"view strategy (max {len(PASS_STRATEGIES)})")
    ap.add_argument("--faces_per_model", type=int, default=8)
    ap.add_argument("--sphere_step", type=int, default=15,
                    help="Degrees between candidate camera directions on the full "
                         "sphere. 15 gives 266 candidates, all scored analytically.")
    ap.add_argument("--image_size", type=int, nargs=2, default=[700, 700])
    ap.add_argument("--no_requests", action="store_true",
                    help="Disallow the model asking for extra angles")
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--limit", type=int, default=0, help="Cap models")
    ap.add_argument("--max_calls", type=int, default=400,
                    help="Hard cap on total API calls, so a batch cannot run away")
    ap.add_argument("--min_agreement", type=float, default=DEFAULT_MIN_AGREEMENT,
                    help="Faces whose passes agree less than this are masked to -1")
    ap.add_argument("--min_confidence", type=float, default=DEFAULT_MIN_CONFIDENCE,
                    help="Faces below this combined confidence are masked to -1")
    ap.add_argument("--no_challenge", action="store_true",
                    help="Skip the objection/challenge round")
    ap.add_argument("--workers", type=int, default=1,
                    help="Models labelled in parallel. Each face costs ~13 s of mostly "
                         "API latency, so a corpus-scale run is only feasible above 1; "
                         "8 turns ~36 h of serial work into ~4.5 h.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip models that already have an output JSON. A corpus run gets "
                         "interrupted; without this, restarting pays for it twice.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    names = CLASS_NAMES if args.label_space == "27" else DEFEATURE_CLASS_NAMES

    if args.step and args.face is not None:
        r = label_face(args.step, args.face, args.label_space, names, args.model,
                       passes=args.passes, allow_requests=not args.no_requests,
                       image_size=tuple(args.image_size),
                       sphere_step=args.sphere_step, verbose=True,
                       min_agreement=args.min_agreement,
                       min_confidence=args.min_confidence,
                       challenge=not args.no_challenge)
        print(json.dumps(r, indent=2))
        return

    if not args.step_dir:
        ap.error("pass --step with --face, or --step_dir for batch mode")

    steps = sorted(p for p in Path(args.step_dir).iterdir()
                   if p.suffix.lower() in (".step", ".stp"))
    if args.limit:
        steps = steps[:args.limit]
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for step in steps:
        if args.resume and args.out_dir and (args.out_dir / f"{step.stem}.json").exists():
            continue
        seg = None
        if args.seg_dir:
            sp = Path(args.seg_dir) / f"{step.stem}.json"
            if sp.exists():
                seg = json.loads(sp.read_text())
        preds = list(map(int, seg.get("face_preds", []))) if seg else []
        probs = seg.get("face_probs") if seg else None
        faces = _pick_faces(preds, probs, args.faces_per_model) if preds else []
        if not faces:
            continue
        jobs.append({"step": str(step), "faces": faces,
                     "preds": {str(i): preds[i] for i in faces if i < len(preds)},
                     "label_space": args.label_space, "model": args.model,
                     "passes": args.passes, "sphere_step": args.sphere_step,
                     "image_size": list(args.image_size),
                     "allow_requests": not args.no_requests,
                     "min_agreement": args.min_agreement,
                     "min_confidence": args.min_confidence,
                     "challenge": not args.no_challenge})

    print(f"{len(jobs)} models to label ({sum(len(j['faces']) for j in jobs)} faces), "
          f"{args.workers} worker(s)", flush=True)

    calls = 0
    total = changed = masked = 0
    t0 = time.time()

    def consume(res: Dict, si: int) -> None:
        nonlocal calls, total, changed, masked
        if res.get("error"):
            print(f"  [{si}/{len(jobs)}] {res['model_id']}: {res['error']}", flush=True)
        calls += res["calls"]
        for fi, rec in res["faces"].items():
            total += 1
            if rec["masked"]:
                masked += 1
            elif rec["teacher_class"] is not None \
                    and rec["vision_class"] != rec["teacher_class"]:
                changed += 1
        if args.out_dir and res["faces"]:
            (args.out_dir / f"{res['model_id']}.json").write_text(json.dumps(
                {"model_id": res["model_id"], "label_space": args.label_space,
                 "passes": args.passes, "min_agreement": args.min_agreement,
                 "min_confidence": args.min_confidence,
                 "faces": res["faces"]}, indent=2))
        rate = total / max(1e-9, (time.time() - t0) / 3600)
        print(f"  [{si}/{len(jobs)}] {res['model_id'][:38]}: "
              f"{len(res['faces'])} faces, {res['calls']} calls "
              f"({total} faces, {masked} masked, {calls} calls, {rate:.0f} faces/h)",
              flush=True)

    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            for si, res in enumerate(pool.imap_unordered(_label_one_model, jobs), 1):
                consume(res, si)
                if calls >= args.max_calls:
                    print(f"reached --max_calls {args.max_calls}, stopping")
                    pool.terminate()
                    break
    else:
        for si, job in enumerate(jobs, 1):
            consume(_label_one_model(job), si)
            if calls >= args.max_calls:
                print(f"reached --max_calls {args.max_calls}, stopping")
                break

    dt = (time.time() - t0) / 60
    if total:
        print(f"\n{total} faces labelled in {dt:.1f} min, {calls} API calls; "
              f"{masked} masked to {IGNORE_LABEL} ({100 * masked / total:.1f}%), "
              f"{changed} of the {total - masked} kept faces differ from the teacher "
              f"({100 * changed / max(1, total - masked):.1f}%)")
    else:
        print("no faces labelled")
    print("Nothing here rewrites a training label. Validate first:\n"
          "  python -m brepformer.data.validate_dense_vision --per_class 14 "
          f"--label_space {args.label_space}")


if __name__ == "__main__":
    main()
