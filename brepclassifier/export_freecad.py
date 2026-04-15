#!/usr/bin/env python3
"""Export colored STEP files for FreeCAD visualization.

Supports two coloring modes:

A) Whole-model mode (pipe classifier): All faces colored by the model's
   8-class prediction. Uses brepclassifier class colors.

B) Per-face mode (face segmentation): Each face colored by 27-class or
   8-real-class prediction. Uses brepformer class colors.

Usage:
    # Pipe mode: color all faces by pipe classifier prediction
    python brepclassifier/export_freecad.py \
        --step model.step \
        --pipe_checkpoint results/pipe_classifier/best-epoch=64-val/f1=0.5517.ckpt \
        --output colored.step

    # Pipe mode: from labels.json
    python brepclassifier/export_freecad.py \
        --step model.step \
        --pipe_labels_json brepclassifier/data/ssdata1/labels.json \
        --output colored.step

    # Face mode: per-face coloring from face segmentation checkpoint
    python brepclassifier/export_freecad.py \
        --step model.step \
        --face_checkpoint results/face_seg_heavy/best-epoch=46-val/f1=0.8608.ckpt \
        --output colored.step

    # Face mode: from .seg file
    python brepclassifier/export_freecad.py \
        --step model.step --seg preds.seg --output colored.step

    # Batch mode
    python brepclassifier/export_freecad.py \
        --step_dir steps/ --face_checkpoint results/face_seg_heavy/best.ckpt \
        --output_dir colored/
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
# Compatibility shim: older pytorch_lightning uses np.Inf, removed in NumPy 2.0
if not hasattr(np, 'Inf'):
    np.Inf = np.inf

# Pipe classifier colors (8 classes)
from brepclassifier.data.classes import (
    CLASS_NAMES as PIPE_CLASS_NAMES,
    CLASS_COLORS_HEX as PIPE_CLASS_COLORS_HEX,
    NUM_CLASSES as PIPE_NUM_CLASSES,
    UNLABELED_COLOR_HEX,
    hex_to_rgb01,
)

# BrepFormer face segmentation colors (27 / 8 real classes)
from brepformer.data.classes import (
    CLASS_COLORS_HEX as FACE_CLASS_COLORS_HEX,
    NUM_CLASSES as FACE_NUM_CLASSES,
    REAL_CLASS_COLORS_HEX, REAL_NUM_CLASSES,
    map_labels_to_real,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export colored STEP files for FreeCAD (pipe classifier or face segmentation)"
    )

    # Input
    parser.add_argument("--step", type=str, default=None, help="Path to STEP file")
    parser.add_argument("--step_dir", type=str, default=None,
                        help="Directory of STEP files for batch mode")

    # Pipe classifier label sources (whole-model coloring)
    parser.add_argument("--pipe_checkpoint", type=str, default=None,
                        help="PipeFittingClassifier checkpoint (colors all faces by model class)")
    parser.add_argument("--pipe_labels_json", type=str, default=None,
                        help="labels.json with model_id -> class_idx mapping")

    # Face segmentation label sources (per-face coloring)
    parser.add_argument("--face_checkpoint", type=str, default=None,
                        help="BrepFormer face seg checkpoint (per-face coloring)")
    parser.add_argument("--seg", type=str, default=None,
                        help="Path to .seg label file (one label per line)")
    parser.add_argument("--seg_dir", type=str, default=None,
                        help="Directory of .seg files (auto-matches by model_id)")
    parser.add_argument("--labels_json", type=str, default=None,
                        help="Path to face labels JSON file (e.g. model_result.json)")
    parser.add_argument("--labels_dir", type=str, default=None,
                        help="Directory of face label JSON files (for batch mode)")

    # Output
    parser.add_argument("--output", type=str, default=None, help="Output STEP file path")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for batch mode")

    # Post-processing
    parser.add_argument("--real_classes", action="store_true",
                        help="Remap 27-class face predictions to 8 real categories")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Label loading helpers
# ---------------------------------------------------------------------------

def load_seg_labels(seg_path: str):
    """Load labels from a .seg file (one label per line)."""
    labels = []
    with open(seg_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(int(line))
    return labels


def load_face_labels_from_json(json_path: str):
    """Load per-face labels from a _result.json file."""
    from brepformer.data.classes import NUM_CLASSES
    with open(json_path, "r") as f:
        data = json.load(f)
    if "cls" not in data:
        return None
    max_face = max(int(k) for k in data["cls"].keys())
    labels = [-1] * (max_face + 1)
    for face_id_str, class_id in data["cls"].items():
        face_id = int(face_id_str)
        if 0 <= class_id < NUM_CLASSES:
            labels[face_id] = class_id
    return labels


def load_pipe_labels_json(json_path: str):
    """Load labels.json mapping model_id -> class_idx."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return {str(k): int(v) for k, v in data.items()}


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def run_pipe_inference(step_path: str, checkpoint_path: str):
    """Run pipe fitting classifier to get whole-model class prediction."""
    import torch
    from brepclassifier.visualize_seg import load_classifier, infer_step_classifier

    model = load_classifier(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    pred = infer_step_classifier(model, step_path, device)
    return pred if pred != -1 else None


def run_face_inference(step_path: str, checkpoint_path: str):
    """Run BrepFormer face segmentation to get per-face labels."""
    import torch
    from brepformer.infer import load_model, infer_single

    model = load_model(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    result = infer_single(model, step_path, device)
    if "error" in result:
        print(f"  Inference error: {result['error']}")
        return None
    return result.get("face_preds", None)


# ---------------------------------------------------------------------------
# XCAF colored STEP export (copied from brepformer/export_freecad.py)
# ---------------------------------------------------------------------------

def export_colored_step(step_path: str, labels: list, output_path: str,
                        colors_hex: list = None, num_cls: int = None):
    """Write a colored STEP file using XCAF, with FreeCAD macro fallback.

    Attempts XCAF export first. If XCAF is broken (pythonocc build issue),
    falls back to generating a FreeCAD Python macro (.FCMacro) that applies
    colors when run inside FreeCAD.

    Args:
        step_path: Input STEP file path.
        labels: List of class IDs per face (TopExp_Explorer order).
        output_path: Output STEP file path.
        colors_hex: Hex color palette (defaults to FACE_CLASS_COLORS_HEX).
        num_cls: Number of classes (defaults to FACE_NUM_CLASSES).
    """
    if colors_hex is None:
        colors_hex = FACE_CLASS_COLORS_HEX
    if num_cls is None:
        num_cls = FACE_NUM_CLASSES

    # Try XCAF export in subprocess to catch C++ crashes gracefully
    ok = _try_xcaf_export(step_path, labels, output_path, colors_hex, num_cls)
    if ok:
        return True

    # Fallback: generate FreeCAD macro
    print("  XCAF export failed (pythonocc XCAF not available). Generating FreeCAD macro...")
    return _export_freecad_macro(step_path, labels, output_path, colors_hex, num_cls)


def _try_xcaf_export(step_path, labels, output_path, colors_hex, num_cls):
    """Try XCAF export in a subprocess to catch C++ crashes."""
    import subprocess
    import tempfile

    # Write labels to a temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "step_path": str(step_path),
            "labels": labels,
            "output_path": str(output_path),
            "colors_hex": colors_hex,
            "num_cls": num_cls,
        }, f)
        temp_path = f.name

    script = f'''
import json, sys
sys.path.insert(0, "{Path(__file__).resolve().parent.parent}")
with open("{temp_path}") as f:
    data = json.load(f)

from OCC.Core.STEPCAFControl import STEPCAFControl_Writer
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods
from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ColorSurf
from OCC.Core.TDocStd import TDocStd_Document
from OCC.Core.TCollection import TCollection_ExtendedString
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.STEPCAFControl import STEPCAFControl_Reader as XCAF_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TDF import TDF_LabelSequence
from brepclassifier.data.classes import hex_to_rgb01
UNLABELED = "#d0d0d0"

doc = TDocStd_Document(TCollection_ExtendedString("XDE"))
st = XCAFDoc_DocumentTool.ShapeTool(doc.Main())
ct = XCAFDoc_DocumentTool.ColorTool(doc.Main())
xr = XCAF_Reader()
xr.SetColorMode(True); xr.SetNameMode(True)
if xr.ReadFile(data["step_path"]) != IFSelect_RetDone:
    sys.exit(2)
xr.Transfer(doc)
tl = TDF_LabelSequence(); st.GetFreeShapes(tl)
faces = []
for i in range(tl.Length()):
    s = st.GetShape(tl.Value(i+1))
    e = TopExp_Explorer(s, TopAbs_FACE)
    while e.More():
        faces.append(topods.Face(e.Current())); e.Next()
for i,face in enumerate(faces):
    lbl = data["labels"][i] if i < len(data["labels"]) else -1
    if 0 <= lbl < data["num_cls"]:
        r,g,b = hex_to_rgb01(data["colors_hex"][lbl])
    else:
        r,g,b = hex_to_rgb01(UNLABELED)
    c = Quantity_Color(r,g,b, Quantity_TOC_RGB)
    fl = st.AddSubShape(tl.Value(1), face)
    if fl.IsNull(): fl = st.AddShape(face, False)
    ct.SetColor(fl, c, XCAFDoc_ColorSurf)
w = STEPCAFControl_Writer()
w.SetColorMode(True); w.SetNameMode(True); w.Transfer(doc)
from pathlib import Path as P
P(data["output_path"]).parent.mkdir(parents=True, exist_ok=True)
if w.Write(data["output_path"]) == IFSelect_RetDone:
    sys.exit(0)
else:
    sys.exit(3)
'''

    try:
        result = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True, text=True, timeout=120
        )
        Path(temp_path).unlink(missing_ok=True)
        if result.returncode == 0:
            print(f"Colored STEP saved to {output_path}")
            return True
        return False
    except Exception:
        Path(temp_path).unlink(missing_ok=True)
        return False


def _export_freecad_macro(step_path, labels, output_path, colors_hex, num_cls):
    """Generate a FreeCAD Python macro that applies face colors.

    The macro can be run inside FreeCAD to color the STEP file.
    Also saves a JSON sidecar with the color mapping.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build per-face RGB color list
    face_colors = []
    for i, lbl in enumerate(labels):
        if 0 <= lbl < num_cls:
            face_colors.append(list(hex_to_rgb01(colors_hex[lbl])))
        else:
            face_colors.append(list(hex_to_rgb01(UNLABELED_COLOR_HEX)))

    # Save JSON sidecar with labels and colors
    json_path = out_path.with_suffix('.colors.json')
    color_data = {
        "step_file": str(Path(step_path).resolve()),
        "labels": labels,
        "face_colors_rgb": face_colors,
        "colors_hex": colors_hex[:num_cls],
        "num_classes": num_cls,
    }
    with open(json_path, 'w') as f:
        json.dump(color_data, f, indent=2)

    # Generate FreeCAD macro
    macro_path = out_path.with_suffix('.FCMacro')
    abs_step = str(Path(step_path).resolve())
    macro_content = f'''# FreeCAD Macro: Apply per-face colors to STEP model
# Generated by brepclassifier/export_freecad.py
# Usage: Open FreeCAD, then Macro -> Execute this macro

import FreeCAD
import Part
import json

step_path = r"{abs_step}"
face_colors = {face_colors!r}

doc = FreeCAD.newDocument("ColoredModel")
shape = Part.Shape()
shape.read(step_path)
obj = doc.addObject("Part::Feature", "ColoredPart")
obj.Shape = shape

# Apply colors per face
if hasattr(obj.ViewObject, "DiffuseColor"):
    colors = []
    for i, face in enumerate(shape.Faces):
        if i < len(face_colors):
            r, g, b = face_colors[i]
            colors.append((r, g, b, 0.0))  # RGBA
        else:
            colors.append((0.816, 0.816, 0.816, 0.0))  # gray
    obj.ViewObject.DiffuseColor = colors

doc.recompute()
FreeCAD.Console.PrintMessage(f"Applied colors to {{len(shape.Faces)}} faces\\n")
'''

    with open(macro_path, 'w') as f:
        f.write(macro_content)

    print(f"FreeCAD macro saved to {macro_path}")
    print(f"Color data saved to {json_path}")
    print(f"  To use: Open FreeCAD -> Macro -> Execute '{macro_path.name}'")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _get_face_labels_for_step(step_path, args, face_model=None, face_device=None):
    """Resolve per-face labels for a single STEP file (face seg mode)."""
    labels = None
    model_id = Path(step_path).stem

    if args.seg:
        labels = load_seg_labels(args.seg)
    elif args.seg_dir:
        seg_file = Path(args.seg_dir) / f"{model_id}.seg"
        if seg_file.exists():
            labels = load_seg_labels(str(seg_file))
    if labels is None and args.labels_json:
        labels = load_face_labels_from_json(args.labels_json)
    if labels is None and args.labels_dir:
        label_file = Path(args.labels_dir) / f"{model_id}_result.json"
        if label_file.exists():
            labels = load_face_labels_from_json(str(label_file))
    if labels is None and face_model is not None:
        from brepformer.infer import infer_single
        result = infer_single(face_model, str(step_path), face_device)
        if "error" not in result:
            labels = result.get("face_preds", None)

    return labels


def _process_single(step_path, args, pipe_labels_map=None,
                    face_model=None, face_device=None):
    """Process a single STEP file and export colored version."""
    model_id = Path(step_path).stem
    is_pipe_mode = args.pipe_checkpoint or args.pipe_labels_json

    if is_pipe_mode:
        # Whole-model mode: all faces get the same color
        pipe_class = None
        if pipe_labels_map and model_id in pipe_labels_map:
            pipe_class = pipe_labels_map[model_id]
        elif args.pipe_checkpoint:
            pipe_class = run_pipe_inference(str(step_path), args.pipe_checkpoint)

        if pipe_class is None:
            print(f"  Skipping {model_id}: no pipe class available")
            return False

        # Count faces to create uniform label list
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE

        reader = STEPControl_Reader()
        status = reader.ReadFile(str(step_path))
        if status != IFSelect_RetDone:
            print(f"  Failed to read STEP: {step_path}")
            return False
        reader.TransferRoots()
        shape = reader.OneShape()
        num_faces = 0
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            num_faces += 1
            explorer.Next()

        labels = [pipe_class] * num_faces
        colors_hex = PIPE_CLASS_COLORS_HEX
        num_cls = PIPE_NUM_CLASSES
        print(f"  Pipe class: {pipe_class} ({PIPE_CLASS_NAMES[pipe_class]}), "
              f"{num_faces} faces")
    else:
        # Per-face mode
        labels = _get_face_labels_for_step(
            step_path, args, face_model=face_model, face_device=face_device
        )
        if labels is None:
            print(f"  Skipping {model_id}: no face labels available")
            return False

        if args.real_classes:
            labels = map_labels_to_real(labels)
            colors_hex = REAL_CLASS_COLORS_HEX
            num_cls = REAL_NUM_CLASSES
        else:
            colors_hex = FACE_CLASS_COLORS_HEX
            num_cls = FACE_NUM_CLASSES

        print(f"  Face labels: {len(labels)} faces")

    output = args.output or str(Path(step_path).with_suffix(".colored.step"))
    if args.output_dir:
        output = str(Path(args.output_dir) / f"{model_id}.colored.step")

    return export_colored_step(str(step_path), labels, output,
                               colors_hex=colors_hex, num_cls=num_cls)


def main():
    """Main export function."""
    args = parse_args()

    if args.step is None and args.step_dir is None:
        print("Error: Provide either --step or --step_dir")
        sys.exit(1)

    has_any_source = (args.pipe_checkpoint or args.pipe_labels_json or
                      args.face_checkpoint or args.seg or args.seg_dir or
                      args.labels_json or args.labels_dir)
    if not has_any_source:
        print("Error: Provide a label source: --pipe_checkpoint, --pipe_labels_json, "
              "--face_checkpoint, --seg, --seg_dir, --labels_json, or --labels_dir")
        sys.exit(1)

    # Load pipe labels map if provided
    pipe_labels_map = None
    if args.pipe_labels_json:
        pipe_labels_map = load_pipe_labels_json(args.pipe_labels_json)
        print(f"Loaded {len(pipe_labels_map)} pipe labels from {args.pipe_labels_json}")

    # Load face model once for batch inference
    face_model = None
    face_device = None
    if args.face_checkpoint:
        import torch
        from brepformer.infer import load_model
        print(f"Loading face seg model from {args.face_checkpoint}...")
        face_model = load_model(args.face_checkpoint)
        face_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        face_model = face_model.to(face_device)

    if args.step:
        # Single file mode
        print(f"Processing {args.step}...")
        _process_single(args.step, args, pipe_labels_map=pipe_labels_map,
                        face_model=face_model, face_device=face_device)
    elif args.step_dir:
        # Batch mode
        step_dir = Path(args.step_dir)
        step_files = sorted(
            list(step_dir.glob("*.step")) + list(step_dir.glob("*.stp"))
        )
        print(f"Found {len(step_files)} STEP files")

        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        success = 0
        for step_file in step_files:
            print(f"\nProcessing {step_file.name}...")
            ok = _process_single(
                str(step_file), args, pipe_labels_map=pipe_labels_map,
                face_model=face_model, face_device=face_device,
            )
            if ok:
                success += 1

        print(f"\nBatch export complete. {success}/{len(step_files)} files exported.")


if __name__ == "__main__":
    main()
