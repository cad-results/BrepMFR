#!/usr/bin/env python3
"""Visual reclassification of face predictions using Rotator + Claude vision.

Renders dense multi-view images of each inferenced STEP model at tight angles
(30° / 15° increments), highlights individual faces, feeds them to Claude's
vision API for class verification, and cross-checks with geometric rules.

Pipeline:
  1. Load STEP + existing inference results (27-class or 5-class)
  2. Render aligned multi-view at 30° increments (12 views) using rotator
  3. Pick the faces most worth reviewing — lowest teacher confidence first, since
     those are the ones masked out of training entirely (see
     ``_select_faces_for_vision``)
  4. Render face-highlighted views and query Claude vision for class verification
  5. Output corrected inference JSON with a corrections log

Geometric rules are used only to *select* faces, never to rewrite labels: measured
against mftrcad ground truth they corrupt 15% of correct labels and 100% of
``chamfer_2``. See ``--geo_mode`` and ``data/validate_geometric_rules.py``.

Their suggested replacement class is also deliberately kept *out* of the vision
prompt. It would be a natural hypothesis to offer, but the rules' errors are
systematic (cylinder→``through_hole``, unplaceable→``stock``) and concentrated on
the rare classes, so offering the guess would anchor the vision model toward the
same wrong answers on precisely the faces that matter most.

Usage:
    # Single file
    python -m brepformer.reclassify_visual \
        --step model.step \
        --seg_json brepformer/inference/hard_samples_27/model.json \
        --output_dir brepformer/inference/hard_samples_27_corrected/

    # Batch mode
    python -m brepformer.reclassify_visual \
        --step_dir brepformer/data/hard_samples/ \
        --seg_dir brepformer/inference/hard_samples_27/ \
        --output_dir brepformer/inference/hard_samples_27_corrected/ \
        --label_space 27

    # Fine-angle pass for ambiguous faces
    python -m brepformer.reclassify_visual \
        --step model.step \
        --seg_json brepformer/inference/hard_samples_27/model.json \
        --output_dir brepformer/inference/hard_samples_27_corrected/ \
        --angle_step 15

Requires:
    - rotator conda env (pythonocc + vtk) for rendering
    - Claude credentials: either ANTHROPIC_API_KEY, or Bedrock via
      CLAUDE_CODE_USE_BEDROCK / AWS_BEARER_TOKEN_BEDROCK (see _vision_client)
    - brepformer environment for geometric rules
"""

import argparse
import base64
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brepformer.data.classes import CLASS_NAMES, DEFEATURE_CLASS_NAMES
from brepformer.geometric_rules import (
    apply_geometric_rules, summarize_verdicts,
    VERDICT_CONFIRM, VERDICT_DENY, VERDICT_UNCERTAIN,
)


# ============================================================
# Multi-angle view generation (extends rotator with finer angles)
# ============================================================

def generate_view_configs(angle_step_deg: int = 30):
    """Generate camera configurations for multi-angle rendering.

    Args:
        angle_step_deg: Angular increment in degrees (30 → 12 views, 15 → 24 views)

    Returns:
        List of (name, camera_direction, up_vector) tuples.
    """
    views = []
    up = (0, 0, 1)

    # Horizontal orbit at elevation 0°
    for i, azimuth in enumerate(range(0, 360, angle_step_deg)):
        rad = math.radians(azimuth)
        cam_dir = (math.sin(rad), math.cos(rad), 0)
        views.append((f"az{azimuth:03d}_el0", cam_dir, up))

    # Elevated views at +30° and +60°
    for elev in [30, 60]:
        elev_rad = math.radians(elev)
        cos_e = math.cos(elev_rad)
        sin_e = math.sin(elev_rad)
        for azimuth in range(0, 360, angle_step_deg * 2):
            rad = math.radians(azimuth)
            cam_dir = (math.sin(rad) * cos_e, math.cos(rad) * cos_e, sin_e)
            views.append((f"az{azimuth:03d}_el{elev}", cam_dir, up))

    # Top and bottom
    views.append(("top", (0, 0, 1), (0, -1, 0)))
    views.append(("bottom", (0, 0, -1), (0, 1, 0)))

    return views


#: step_path -> tessellated model, reused across every face of that model.
#: Only the most recent model is kept: the pass walks one model to completion
#: before moving on, so a size-1 cache gets the full benefit without holding
#: hundreds of megabytes of triangulation for models already finished.
_MESH_CACHE: dict = {}


def _tessellate(step_path: str):
    """Load, rotator-align and tessellate a STEP once, returning reusable geometry.

    Returns ``(polydata, cell_face_ids, extent, n_faces)`` where ``cell_face_ids``
    maps each triangle to its originating face index.

    Why this is split out: the highlight colour is the *only* thing that differs
    between faces of the same model, but the original code re-ran ``load_step`` +
    ``align_shape`` + ``BRepMesh_IncrementalMesh`` and rebuilt every triangle for
    each face. On a 600-face model reviewed at 50 faces that is 50 full
    tessellations of identical geometry — measured at ~5 s per face, roughly 60 h
    across 853 models, nearly all of it recomputing the same mesh. Building once
    per model and recolouring per face makes the per-face cost just the render.
    """
    cached = _MESH_CACHE.get(step_path)
    if cached is not None:
        return cached

    rotator_dir = Path(__file__).resolve().parent.parent / "rotator"
    if str(rotator_dir) not in sys.path:
        sys.path.insert(0, str(rotator_dir))
    from pipeline import load_step, align_shape

    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCC.Core.TopoDS import topods
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.BRep import BRep_Tool

    import vtk

    shape = load_step(step_path)
    aligned, R, center, method = align_shape(shape)
    BRepMesh_IncrementalMesh(aligned, 0.05).Perform()

    faces = []
    exp = TopExp_Explorer(aligned, TopAbs_FACE)
    while exp.More():
        faces.append(topods.Face(exp.Current()))
        exp.Next()

    all_points = vtk.vtkPoints()
    all_cells = vtk.vtkCellArray()
    cell_face_ids: List[int] = []

    vertex_offset = 0
    for fi, face in enumerate(faces):
        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, loc)
        if triangulation is None:
            continue

        trsf = loc.Transformation()
        n_nodes = triangulation.NbNodes()
        for i in range(1, n_nodes + 1):
            pnt = triangulation.Node(i)
            pnt.Transform(trsf)
            all_points.InsertNextPoint(pnt.X(), pnt.Y(), pnt.Z())

        reversed_ = face.Orientation() == TopAbs_REVERSED
        for i in range(1, triangulation.NbTriangles() + 1):
            n1, n2, n3 = triangulation.Triangle(i).Get()
            cell = vtk.vtkTriangle()
            a, b = (n2, n1) if reversed_ else (n1, n2)
            cell.GetPointIds().SetId(0, a - 1 + vertex_offset)
            cell.GetPointIds().SetId(1, b - 1 + vertex_offset)
            cell.GetPointIds().SetId(2, n3 - 1 + vertex_offset)
            all_cells.InsertNextCell(cell)
            cell_face_ids.append(fi)

        vertex_offset += n_nodes

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(all_points)
    polydata.SetPolys(all_cells)

    bounds = polydata.GetBounds()
    extent = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])

    entry = (polydata, cell_face_ids, extent, len(faces))
    _MESH_CACHE.clear()
    _MESH_CACHE[step_path] = entry
    return entry


def render_face_highlighted(step_path: str, face_idx: int,
                            angle_step: int = 30,
                            image_size: Tuple[int, int] = (600, 600)
                            ) -> List[bytes]:
    """Render views with a specific face highlighted in red, in-memory.

    Uses VTK to color one face distinctly against gray neighbors.
    Returns list of PNG image bytes (no files written to disk).
    """
    import vtk

    polydata, cell_face_ids, extent, n_faces = _tessellate(step_path)
    if face_idx >= n_faces:
        return []

    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("Colors")
    for fi in cell_face_ids:
        if fi == face_idx:
            colors.InsertNextTuple3(220, 50, 50)
        else:
            colors.InsertNextTuple3(200, 200, 210)
    polydata.GetCellData().SetScalars(colors)

    key_views = generate_view_configs(90)[:6]

    # One whole-model view for context (where is this face on the part?), then
    # views framed on the face itself so it is actually large enough to judge.
    #
    # Both halves matter. Zoom alone loses the "is this at an edge of the body or
    # in the middle of a pocket" cue that distinguishes a chamfer from a slot wall;
    # context alone gives a face too small to see at all.
    images: List[bytes] = []
    ctx = _render_to_png_bytes(polydata, *key_views[0][1:], extent, size=image_size)
    if ctx:
        images.append(ctx)

    focal, fscale = _face_frame(polydata, cell_face_ids, face_idx, extent)
    zoomed: List[Tuple[int, bytes]] = []
    for view_name, cam_dir, cam_up in key_views:
        png_bytes = _render_to_png_bytes(polydata, cam_dir, cam_up, extent,
                                         size=image_size, focal=focal, scale=fscale)
        if png_bytes:
            zoomed.append((_red_pixels(png_bytes), png_bytes))

    # Order by how much of the highlighted face is actually visible. The API call
    # only forwards the first few images, so taking views in fixed camera order
    # meant the face could be occluded or edge-on in every image sent — the model
    # was being asked to classify something it could not see, and unsurprisingly
    # fell back on "this is plain stock" for nearly every feature face.
    zoomed.sort(key=lambda t: -t[0])
    images.extend(b for _, b in zoomed)
    return images


def _face_frame(polydata, cell_face_ids: List[int], face_idx: int, extent: float):
    """Return (focal_point, parallel_scale) framing just this face, with context.

    The margin keeps neighbouring geometry in shot: a chamfer is only identifiable
    *relative* to the two faces it bridges, so a tight crop of the face alone would
    trade one unanswerable question for another.
    """
    import numpy as np

    pts = polydata.GetPoints()
    ids = vtk_cell_point_ids(polydata, cell_face_ids, face_idx)
    if not ids:
        return None, None

    coords = np.array([pts.GetPoint(i) for i in ids], dtype=float)
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    center = tuple((lo + hi) / 2.0)
    face_size = float(max(hi - lo))

    # Frame slightly wider than the face's own bounding box, so the face fills
    # most of the shot with its immediate neighbours still visible.
    #
    # The multiplier is deliberately below 1: `face_size` is the *largest* bbox
    # dimension, which for a thin annular chamfer is the ring diameter, not the
    # chamfer width. Scaling that up (an earlier attempt used 3x) framed the whole
    # part again and left the face at 0.16% of the frame — the cap below did all
    # the work and the zoom did none.
    scale = max(face_size * 0.8, extent * 0.005)
    scale = min(scale, extent * 0.6)
    return center, scale


def vtk_cell_point_ids(polydata, cell_face_ids: List[int], face_idx: int) -> List[int]:
    """Point ids belonging to the triangles of one face."""
    ids: set = set()
    for cid, fi in enumerate(cell_face_ids):
        if fi != face_idx:
            continue
        cell = polydata.GetCell(cid)
        pid = cell.GetPointIds()
        for k in range(pid.GetNumberOfIds()):
            ids.add(pid.GetId(k))
    return sorted(ids)


def _red_pixels(png_bytes: bytes) -> int:
    """Count highlight pixels, used to rank views by face visibility."""
    try:
        import io

        import numpy as np
        from PIL import Image

        a = np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB"), dtype=int)
        return int(((a[:, :, 0] > 150) & (a[:, :, 1] < 110)
                    & (a[:, :, 2] < 110)).sum())
    except Exception:  # noqa: BLE001
        return 0


def _render_to_png_bytes(polydata, cam_dir, cam_up, extent,
                         size=(600, 600), focal=None, scale=None) -> bytes:
    """Render a colored VTK polydata and return PNG bytes (no disk I/O).

    ``focal`` and ``scale`` override the whole-model framing so the camera can be
    zoomed onto one face. Without them a small feature renders as a few pixels:
    a real mftrcad chamfer measured 280 red pixels out of 360,000 (0.078% of the
    frame, a 1px-tall line) in its *most* visible view, which is not something any
    classifier could judge.
    """
    import vtk

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.SetScalarModeToUseCellData()
    mapper.ScalarVisibilityOn()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetAmbient(0.4)
    actor.GetProperty().SetDiffuse(0.6)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1.0, 1.0, 1.0)

    renWin = vtk.vtkRenderWindow()
    renWin.SetOffScreenRendering(1)
    renWin.AddRenderer(renderer)
    renWin.SetSize(size[0], size[1])

    fx, fy, fz = focal if focal is not None else (0.0, 0.0, 0.0)
    camera = renderer.GetActiveCamera()
    camera.SetParallelProjection(True)
    dist = extent * 5
    camera.SetPosition(fx + cam_dir[0] * dist,
                       fy + cam_dir[1] * dist,
                       fz + cam_dir[2] * dist)
    camera.SetFocalPoint(fx, fy, fz)
    camera.SetViewUp(cam_up[0], cam_up[1], cam_up[2])
    if focal is None:
        renderer.ResetCamera()
        camera.SetFocalPoint(fx, fy, fz)
    camera.SetParallelScale(scale if scale is not None else extent * 0.6)
    camera.SetClippingRange(extent * 0.01, extent * 20)
    renWin.Render()

    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(renWin)
    w2if.Update()

    writer = vtk.vtkPNGWriter()
    writer.WriteToMemoryOn()
    writer.SetInputConnection(w2if.GetOutputPort())
    writer.Write()

    png_data = bytes(memoryview(writer.GetResult()))
    renWin.Finalize()
    return png_data


# ============================================================
# Claude Vision API integration
# ============================================================

def _build_face_prompt(face_idx: int, predicted_class: str,
                       geom_info: Dict, label_space: str,
                       class_names: List[str]) -> str:
    """Build a structured prompt for Claude to verify a face classification."""
    prompt = f"""You are analyzing a B-Rep (boundary representation) CAD model face.

TASK: Verify whether Face #{face_idx} (highlighted in RED in the images) is correctly classified.

PREDICTED CLASS: "{predicted_class}"

GEOMETRIC CONTEXT:
- Surface type: {"planar" if geom_info.get("is_planar") else "curved" if geom_info.get("is_curved") else "BSpline/other"}
- Area fraction of total model: {geom_info.get("area_frac", 0):.1%}
{f'- Cylinder/cone axis detected' if geom_info.get("axis") else ''}
{f'- Radius: {geom_info.get("radius"):.3f}' if geom_info.get("radius") else ''}

AVAILABLE CLASSES ({label_space}-class):
{chr(10).join(f"  {i}: {name}" for i, name in enumerate(class_names))}

CLASSIFICATION GUIDELINES:
- "stock" / "random" = main body faces, structural surfaces, not machining features
- Holes are cylindrical through-passages or blind cavities
- Chamfers are small angled planar faces at edges (always small area)
- Fillets are smooth curved transitions between faces (always small area)
- Cuts/slots/pockets/passages are material-removal features with planar walls
- Steps are L-shaped material removal features

INSTRUCTIONS:
1. Look at the RED highlighted face from multiple angles
2. Consider its shape, size relative to the model, and relationship to neighbors
3. Decide if the predicted class is correct

Respond with EXACTLY this JSON format:
{{"verdict": "CONFIRM" or "DENY", "confidence": 0.0-1.0, "correct_class": <int or null>, "reasoning": "<1 sentence>"}}

If DENY, set correct_class to the class ID you believe is correct.
If CONFIRM, set correct_class to null."""
    return prompt


#: Bedrock model ids are not the plain API aliases: they need a cross-region
#: inference profile prefix ("us.") and a version suffix. Mapping the friendly
#: name keeps ``--claude_model claude-sonnet-5`` working on either backend.
_BEDROCK_MODEL_MAP = {
    "claude-sonnet-5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-sonnet-4-5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
}

#: (client, resolved_model), built once. Constructing a client per face would
#: re-do credential resolution tens of thousands of times.
_VISION_CLIENT: dict = {}


class VisionCredentialsError(RuntimeError):
    """No usable Claude credentials for the vision pass."""


def _vision_client(model: str):
    """Return (client, resolved_model_id) for the vision backend, or raise.

    Supports a direct ANTHROPIC_API_KEY and AWS Bedrock, preferring whichever is
    configured. Bedrock matters here because this machine authenticates that way
    (``CLAUDE_CODE_USE_BEDROCK`` + ``AWS_BEARER_TOKEN_BEDROCK``) and has no
    ``ANTHROPIC_API_KEY`` at all.

    Why this raises instead of returning None: the previous code returned None
    when the key was missing, and ``reclassify_single`` treats None as "the model
    had no opinion" and simply keeps the original label. A full 853-model run with
    no credentials would therefore print no error, write plausible-looking output
    for every file, and report only geometric corrections — silently producing a
    corpus with zero visual relabelling while looking like it worked. A missing
    credential is a setup error, not a per-face verdict, so it must be loud.
    """
    if _VISION_CLIENT:
        return _VISION_CLIENT["client"], _VISION_CLIENT["model"]

    import os

    try:
        import anthropic
    except ImportError as e:  # noqa: BLE001
        raise VisionCredentialsError(
            "the 'anthropic' package is required for the Claude vision pass; "
            "install it or pass --no_claude to run geometric rules only"
        ) from e

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    use_bedrock = bool(
        os.environ.get("CLAUDE_CODE_USE_BEDROCK")
        or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    )

    if api_key:
        client = anthropic.Anthropic(api_key=api_key)
        resolved = model
    elif use_bedrock:
        if not hasattr(anthropic, "AnthropicBedrock"):
            raise VisionCredentialsError(
                f"Bedrock is configured but anthropic {anthropic.__version__} has no "
                "AnthropicBedrock client; upgrade the anthropic package"
            )
        client = anthropic.AnthropicBedrock(
            aws_region=os.environ.get("AWS_REGION", "us-east-1")
        )
        resolved = _BEDROCK_MODEL_MAP.get(model, model)
    else:
        raise VisionCredentialsError(
            "no Claude credentials found: set ANTHROPIC_API_KEY, or configure "
            "Bedrock (CLAUDE_CODE_USE_BEDROCK / AWS_BEARER_TOKEN_BEDROCK), or "
            "pass --no_claude to run geometric rules only"
        )

    _VISION_CLIENT["client"] = client
    _VISION_CLIENT["model"] = resolved
    return client, resolved


def _build_classify_prompt(face_idx: int, label_space: str,
                           class_names: List[str]) -> str:
    """Ask for a label outright, without naming any proposed class.

    Why a separate prompt: the verify prompt shows a candidate label and asks
    whether it is correct, and measured against mftrcad ground truth that framing
    denies 91% of wrong labels *and* 74% of correct ones — a 17-point spread, i.e.
    the verdict is close to "always DENY" and carries almost no information. Naming
    a class to judge invites fault-finding. Asking for a classification cold removes
    the anchor, and the caller can compare the answer to the existing label itself.
    """
    return f"""You are analysing one face of a B-Rep CAD model.

The face outlined in RED is the subject. The first image shows the whole part for
context; the remaining images are zoomed to the subject face from several angles.

Classify the RED face into exactly one of these {label_space} categories:
{chr(10).join(f"  {i}: {name}" for i, name in enumerate(class_names))}

Guidance:
- Most faces of a typical part are plain body/stock surfaces, not machining features.
- A chamfer is a narrow flat angled band bridging two surfaces at an edge.
- A fillet is a narrow smoothly curved band bridging two surfaces at an edge.
- A hole is a cylindrical internal surface.
- Judge the face by its own shape and by what it connects to, not by the part's
  overall purpose.

Respond with EXACTLY this JSON and nothing else:
{{"class": <int>, "confidence": 0.0-1.0, "reasoning": "<1 short sentence>"}}"""


def query_claude_vision(
    image_bytes_list: List[bytes],
    face_idx: int,
    predicted_class: str,
    geom_info: Dict,
    label_space: str,
    class_names: List[str],
    model: str = "claude-sonnet-5",
    mode: str = "verify",
    max_images: int = 4,
) -> Optional[Dict]:
    """Query Claude vision API about a face's class.

    Args:
        image_bytes_list: List of PNG image bytes (in-memory, no disk I/O).
        mode: "verify" asks whether ``predicted_class`` is right and returns
            ``verdict``/``correct_class``. "classify" ignores ``predicted_class``
            and returns a ``class`` outright, avoiding the confirm/deny anchor.

    Returns the parsed JSON dict, or None on API failure.
    """
    client, model = _vision_client(model)

    image_content = []
    for png_bytes in image_bytes_list[:max_images]:
        if png_bytes:
            b64 = base64.b64encode(png_bytes).decode("utf-8")
            image_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": b64,
                },
            })

    if not image_content:
        return None

    if mode == "classify":
        prompt = _build_classify_prompt(face_idx, label_space, class_names)
    else:
        prompt = _build_face_prompt(
            face_idx, predicted_class, geom_info, label_space, class_names
        )

    image_content.append({"type": "text", "text": prompt})

    try:
        response = client.messages.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": image_content}],
        )
        text = response.content[0].text.strip()

        # Parse JSON from response
        if "{" in text:
            json_str = text[text.index("{"):text.rindex("}") + 1]
            return json.loads(json_str)
    except Exception as e:
        print(f"  Claude API error for face {face_idx}: {e}", file=sys.stderr)

    return None


# ============================================================
# Main reclassification pipeline
# ============================================================

def _select_faces_for_vision(
    geo_verdicts: List[Dict],
    face_preds: List[int],
    face_conf: Optional[List[float]],
    stock_class: int,
    mask_confidence: float,
    geo_mode: str = "advise",
) -> List[int]:
    """Choose which faces to spend vision calls on, best-value first.

    The budget is finite (``--max_claude_calls`` per model), so *which* faces get
    reviewed decides how much training signal this pass actually buys.

    Two deliberate departures from the original "geometrically uncertain, and not
    predicted stock" rule:

    1. **Order by teacher confidence, ascending.** Faces whose top-1 probability
       is below ``mask_confidence`` are the ones ``merge_suitable_samples.py``
       rewrites to -1, so today they contribute *no gradient at all*. Relabelling
       one converts a discarded face into a supervised face. Re-reviewing a face
       the teacher already called at 0.95 mostly re-confirms it. Same spend, far
       more supervision recovered.

    2. **Do not blanket-exclude stock predictions.** Excluding them was
       self-defeating for the problem that motivated this pass: the 5-class model
       never predicts chamfer, and a missed chamfer shows up precisely as a small
       face confidently-or-not called stock. If stock faces are never reviewed, a
       chamfer that the teacher called stock can never be recovered, so the dead
       class stays dead by construction. Stock faces are therefore admitted, but
       only when the teacher was unsure about them — a high-confidence stock face
       is both likely correct and enormously common, and reviewing those would
       consume the whole budget on the easiest calls in the corpus.

    Faces already corrected by a geometric DENY are skipped: those rules are
    high-confidence and deterministic, and re-litigating them with a vision model
    would let a weaker signal overwrite a stronger one.
    """
    candidates: List[int] = []
    for v in geo_verdicts:
        idx = v["face_idx"]
        if v["verdict"] == VERDICT_DENY and geo_mode == "rewrite":
            continue  # already corrected deterministically
        conf = face_conf[idx] if face_conf and idx < len(face_conf) else None
        unsure_teacher = conf is not None and conf < mask_confidence

        if face_preds[idx] == stock_class:
            # Only worth a call if the teacher itself was shaky (see docstring).
            if not unsure_teacher:
                continue
        elif v["verdict"] == VERDICT_CONFIRM and not unsure_teacher:
            # Geometry confirmed it and the teacher was confident: leave it.
            continue
        candidates.append(idx)

    if face_conf:
        candidates.sort(key=lambda i: face_conf[i] if i < len(face_conf) else 1.0)
    return candidates


def reclassify_single(
    step_path: str,
    seg_path: str,
    output_dir: Path,
    label_space: str = "27",
    angle_step: int = 30,
    use_claude: bool = True,
    claude_model: str = "claude-sonnet-5",
    max_claude_calls: int = 50,
    verbose: bool = False,
    mask_confidence: float = 0.7,
    geo_mode: str = "advise",
) -> Dict:
    """Reclassify a single model's face predictions.

    Returns result dict with corrected predictions and correction log.
    """
    class_names = CLASS_NAMES if label_space == "27" else DEFEATURE_CLASS_NAMES
    stock_class = 24 if label_space == "27" else 0
    name = Path(step_path).stem

    # Load predictions
    seg_p = Path(seg_path)
    face_conf: Optional[List[float]] = None
    if seg_p.suffix == ".json":
        with open(seg_p) as f:
            data = json.load(f)
        face_preds = data["face_preds"]
        probs = data.get("face_probs")
        if probs and len(probs) == len(face_preds):
            face_conf = [max(row) if row else 0.0 for row in probs]
    else:
        with open(seg_p) as f:
            face_preds = [int(line.strip()) for line in f if line.strip()]

    if verbose:
        print(f"Processing: {name} ({len(face_preds)} faces)")

    # Step 1: Geometric rules
    if geo_mode == "off":
        geo_verdicts = [{"face_idx": i, "verdict": VERDICT_UNCERTAIN, "reason": "",
                         "suggested": None} for i in range(len(face_preds))]
    else:
        geo_verdicts = apply_geometric_rules(step_path, face_preds, label_space)
    geo_summary = summarize_verdicts(geo_verdicts)

    if verbose:
        print(f"  Geometric rules: {geo_summary['confirmed']} confirmed, "
              f"{geo_summary['denied']} denied, {geo_summary['uncertain']} uncertain")

    corrected = list(face_preds)
    corrections = []

    # Geometric denials are NOT applied as rewrites by default. Measured against
    # mftrcad ground truth (validate_geometric_rules.py, 60 models / 1412 labelled
    # faces) they overwrite 15% of *correct* labels, and the damage is concentrated
    # exactly where this corpus is weakest:
    #
    #   chamfer_2              100% falsely denied -> through_hole
    #   rectangular_passage_2  100% falsely denied -> through_hole
    #   circular_end_pocket     59% falsely denied -> stock
    #   blind_hole              54% falsely denied -> stock
    #   stock                    0% falsely denied
    #
    # The rules are reliable on stock and destructive on rare curved features:
    # cylindrical faces get called through_hole and unplaceable faces get called
    # stock. Applying them would take chamfer from merely dead to actively
    # mislabelled — strictly worse than the teacher's argmax, and undetectable
    # without ground truth.
    #
    # What survives is their *diagnostic* value: a denial is a good signal that a
    # face is worth a vision call, and the suggestion is a useful hypothesis to put
    # in front of the vision model. So "advise" keeps the signal and drops the
    # rewrite. "rewrite" restores the old behaviour for comparison only.
    if geo_mode == "rewrite":
        for v in geo_verdicts:
            if v["verdict"] == VERDICT_DENY and v["suggested"] is not None:
                old_class = corrected[v["face_idx"]]
                corrected[v["face_idx"]] = v["suggested"]
                corrections.append({
                    "face_idx": v["face_idx"],
                    "old_class": old_class,
                    "new_class": v["suggested"],
                    "source": "geometric_rule",
                    "reason": v["reason"],
                })

    # Step 2: Claude vision for uncertain faces (if enabled)
    claude_corrections = 0
    if use_claude:
        uncertain_faces = _select_faces_for_vision(
            geo_verdicts, face_preds, face_conf, stock_class, mask_confidence,
            geo_mode,
        )

        if uncertain_faces and verbose:
            order = "confidence-ordered" if face_conf else "geometric-uncertain"
            print(f"  Querying Claude vision for "
                  f"{min(len(uncertain_faces), max_claude_calls)} of "
                  f"{len(uncertain_faces)} candidate faces ({order})")

        for face_idx in uncertain_faces[:max_claude_calls]:
            # Render face-highlighted views in-memory
            try:
                face_images = render_face_highlighted(
                    step_path, face_idx,
                    angle_step=angle_step
                )
            except Exception as e:
                if verbose:
                    print(f"    Face {face_idx}: render failed ({e})")
                continue

            if not face_images:
                continue

            pred_class_name = (
                class_names[corrected[face_idx]]
                if corrected[face_idx] < len(class_names) else "unknown"
            )

            # Get geometric info for prompt context
            geo_v = geo_verdicts[face_idx] if face_idx < len(geo_verdicts) else {}
            geom_info = {
                "is_planar": any(
                    v.get("reason", "").find("planar") >= 0
                    for v in [geo_v]
                ),
                "is_curved": any(
                    v.get("reason", "").find("curved") >= 0
                    for v in [geo_v]
                ),
            }

            claude_result = query_claude_vision(
                face_images, face_idx, pred_class_name,
                geom_info, label_space, class_names,
                model=claude_model,
            )

            if claude_result and claude_result.get("verdict") == "DENY":
                new_class = claude_result.get("correct_class")
                confidence = claude_result.get("confidence", 0)

                if (new_class is not None
                        and 0 <= new_class < len(class_names)
                        and confidence >= 0.7):
                    old_class = corrected[face_idx]
                    corrected[face_idx] = new_class
                    claude_corrections += 1
                    corrections.append({
                        "face_idx": face_idx,
                        "old_class": old_class,
                        "new_class": new_class,
                        "source": "claude_vision",
                        "confidence": confidence,
                        "reason": claude_result.get("reasoning", ""),
                    })

            # Rate limiting
            time.sleep(0.5)

    if verbose:
        print(f"  Total corrections: {len(corrections)} "
              f"(geometric: {len(corrections) - claude_corrections}, "
              f"claude: {claude_corrections})")

    # Write corrected output
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "face_preds": corrected,
        "original_preds": face_preds,
        "corrections": corrections,
        "summary": {
            "total_faces": len(face_preds),
            "total_corrections": len(corrections),
            "geometric_corrections": len(corrections) - claude_corrections,
            "claude_corrections": claude_corrections,
        },
    }

    out_path = output_dir / f"{name}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    if verbose:
        print(f"  Saved: {out_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Visual reclassification with Rotator + Claude vision"
    )
    parser.add_argument("--step", type=str, help="Single STEP file")
    parser.add_argument("--step_dir", type=str, help="Directory of STEP files")
    parser.add_argument("--seg_json", type=str,
                        help="Single .json inference file (with face_preds)")
    parser.add_argument("--seg_dir", type=str,
                        help="Directory of .json/.seg inference files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for corrected predictions")
    parser.add_argument("--label_space", default="27", choices=["27", "5"],
                        help="Label space (default: 27)")
    parser.add_argument("--angle_step", type=int, default=30,
                        help="View angle increment in degrees (default: 30)")
    parser.add_argument("--no_claude", action="store_true",
                        help="Skip Claude vision (geometric rules only)")
    parser.add_argument("--claude_model", default="claude-sonnet-5",
                        help="Claude model for vision queries")
    parser.add_argument("--max_claude_calls", type=int, default=50,
                        help="Max Claude API calls per model (default: 50)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max files in batch mode (0=all)")
    parser.add_argument("--geo_mode", default="advise",
                        choices=["advise", "rewrite", "off"],
                        help="How to use geometric_rules denials. 'advise' (default): "
                             "never rewrite a label, only flag the face for vision "
                             "review and pass the rule's guess as a hypothesis. "
                             "'rewrite': apply denials directly -- measured to corrupt "
                             "15%% of correct mftrcad labels and 100%% of chamfer_2, "
                             "kept only for comparison. 'off': ignore rules entirely.")
    parser.add_argument("--mask_confidence", type=float, default=0.7,
                        help="Teacher top-1 confidence below which a face counts as "
                             "unreliable. Must match merge_suitable_samples.py's "
                             "--min_confidence: those faces are the ones masked to -1, "
                             "so reviewing them is what converts discarded faces into "
                             "supervised ones. Vision calls are spent lowest-confidence "
                             "first (default: 0.7)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Resolve credentials before doing any rendering. A batch run is hours of
    # OCC/VTK work per hundred models; discovering a missing key at the first API
    # call is cheap, discovering it after the run is not.
    if not args.no_claude:
        client, resolved = _vision_client(args.claude_model)
        print(f"Vision backend: {type(client).__name__}, model {resolved}")

    if args.step and args.seg_json:
        reclassify_single(
            args.step, args.seg_json, output_dir,
            label_space=args.label_space,
            angle_step=args.angle_step,
            use_claude=not args.no_claude,
            claude_model=args.claude_model,
            max_claude_calls=args.max_claude_calls,
            verbose=args.verbose,
            mask_confidence=args.mask_confidence,
            geo_mode=args.geo_mode,
        )
    elif args.step_dir and args.seg_dir:
        step_dir = Path(args.step_dir)
        seg_dir = Path(args.seg_dir)

        step_files = sorted(
            list(step_dir.glob("*.step")) + list(step_dir.glob("*.stp"))
        )
        if args.limit > 0:
            step_files = step_files[:args.limit]

        print(f"Batch reclassification: {len(step_files)} STEP files")
        print(f"  Label space: {args.label_space}-class")
        print(f"  Angle step: {args.angle_step}°")
        print(f"  Claude vision: {'enabled' if not args.no_claude else 'disabled'}")

        successes = 0
        total_corrections = 0

        for step_file in step_files:
            # Find matching seg file
            stem = step_file.stem
            seg_file = None
            for ext in [".json", ".seg"]:
                candidate = seg_dir / f"{stem}{ext}"
                if candidate.exists():
                    seg_file = candidate
                    break

            if seg_file is None:
                if args.verbose:
                    print(f"  Skipping {stem}: no matching inference file")
                continue

            # Skip if already corrected
            out_file = output_dir / f"{stem}.json"
            if out_file.exists():
                if args.verbose:
                    print(f"  Skipping {stem}: already corrected")
                successes += 1
                continue

            try:
                result = reclassify_single(
                    str(step_file), str(seg_file), output_dir,
                    label_space=args.label_space,
                    angle_step=args.angle_step,
                    use_claude=not args.no_claude,
                    claude_model=args.claude_model,
                    max_claude_calls=args.max_claude_calls,
                    verbose=args.verbose,
                    mask_confidence=args.mask_confidence,
                    geo_mode=args.geo_mode,
                )
                successes += 1
                total_corrections += result["summary"]["total_corrections"]
            except Exception as e:
                print(f"  FAILED {stem}: {e}", file=sys.stderr)

        print(f"\nDone: {successes}/{len(step_files)} processed, "
              f"{total_corrections} total corrections")
    else:
        parser.error("Provide --step + --seg_json, or --step_dir + --seg_dir")


if __name__ == "__main__":
    main()
