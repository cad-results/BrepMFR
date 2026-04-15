#!/usr/bin/env python3
"""Export colored STEP files for FreeCAD visualization.

Uses pythonOCC XCAF (STEPCAFControl_Writer + XCAFDoc_ColorSurf) to write
STEP AP214 files with per-face colors. FreeCAD reads these natively.

Usage:
    # From .seg file
    python -m brepformer.export_freecad --step model.step --seg preds.seg --output colored.step

    # From inference (run on the fly)
    python -m brepformer.export_freecad --step model.step --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt --output colored.step

    # From GT labels JSON
    python -m brepformer.export_freecad --step model.step --labels_json labels/model_result.json --output colored.step

    # Batch mode with GT labels
    python -m brepformer.export_freecad --step_dir steps/ --labels_dir labels/ --output_dir colored/
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brepformer.data.classes import (
    CLASS_COLORS_HEX, UNLABELED_COLOR_HEX, NUM_CLASSES,
    CLASS_NAMES, hex_to_rgb01, map_labels_to_real,
    REAL_CLASS_COLORS_HEX, REAL_NUM_CLASSES,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export colored STEP files for FreeCAD")

    # Input
    parser.add_argument("--step", type=str, default=None, help="Path to STEP file")
    parser.add_argument("--step_dir", type=str, default=None, help="Directory of STEP files for batch mode")

    # Label source (choose one)
    parser.add_argument("--seg", type=str, default=None, help="Path to .seg label file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint (runs inference)")
    parser.add_argument("--labels_json", type=str, default=None,
                        help="Path to labels JSON file (e.g. model_result.json)")
    parser.add_argument("--labels_dir", type=str, default=None,
                        help="Directory of label JSON files (for batch mode)")
    parser.add_argument("--seg_dir", type=str, default=None,
                        help="Directory of .seg prediction files (auto-matches by model_id)")

    # Output
    parser.add_argument("--output", type=str, default=None, help="Output STEP file path")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for batch mode")

    # Post-processing
    parser.add_argument("--real_classes", action="store_true",
                        help="Remap 27 MFTRCAD classes to 8 real machining feature categories")

    # Defeaturing
    parser.add_argument("--defeature", action="store_true",
                        help="Also output a defeatured STEP file (removes predicted features)")
    parser.add_argument("--defeature_output_dir", type=str, default="brepformer/defeatured_output",
                        help="Output directory for defeatured STEP files (default: brepformer/defeatured_output)")

    return parser.parse_args()


def load_seg_labels(seg_path: str):
    """Load labels from a .seg file (one label per line)."""
    labels = []
    with open(seg_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(int(line))
    return labels


def load_labels_from_json(json_path: str):
    """Load per-face labels from a _result.json file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    if "cls" not in data:
        return None

    # Find max face index
    max_face = max(int(k) for k in data["cls"].keys())
    labels = [-1] * (max_face + 1)
    for face_id_str, class_id in data["cls"].items():
        face_id = int(face_id_str)
        if 0 <= class_id < NUM_CLASSES:
            labels[face_id] = class_id

    return labels


def run_inference_for_labels(step_path: str, checkpoint_path: str):
    """Run BrepFormer inference to get per-face labels."""
    import torch
    from brepformer.infer import load_model, infer_single

    model = load_model(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    result = infer_single(model, step_path, device)
    if "error" in result:
        print(f"Inference error: {result['error']}")
        return None
    return result.get("face_preds", None)


def _build_step_color_psa(r, g, b, step_model):
    """Build the STEP color entity chain for a single face color.

    Creates: COLOUR_RGB → FILL_AREA_STYLE_COLOUR → FILL_AREA_STYLE →
    SURFACE_STYLE_FILL_AREA → SURFACE_SIDE_STYLE → SURFACE_STYLE_USAGE →
    PRESENTATION_STYLE_ASSIGNMENT.

    All entities are added to step_model.  Returns the PSA.
    """
    from OCC.Core.StepVisual import (
        StepVisual_ColourRgb,
        StepVisual_FillAreaStyleColour, StepVisual_FillAreaStyle,
        StepVisual_SurfaceStyleFillArea, StepVisual_SurfaceSideStyle,
        StepVisual_SurfaceStyleUsage, StepVisual_SurfaceSide,
        StepVisual_PresentationStyleAssignment,
        StepVisual_HArray1OfFillStyleSelect, StepVisual_FillStyleSelect,
        StepVisual_HArray1OfSurfaceStyleElementSelect,
        StepVisual_SurfaceStyleElementSelect,
        StepVisual_HArray1OfPresentationStyleSelect,
        StepVisual_PresentationStyleSelect,
    )
    from OCC.Core.TCollection import TCollection_HAsciiString

    col = StepVisual_ColourRgb()
    col.SetName(TCollection_HAsciiString(""))
    col.SetRed(r)
    col.SetGreen(g)
    col.SetBlue(b)
    step_model.AddEntity(col)

    fasc = StepVisual_FillAreaStyleColour()
    fasc.SetName(TCollection_HAsciiString(""))
    fasc.SetFillColour(col)
    step_model.AddEntity(fasc)

    fas = StepVisual_FillAreaStyle()
    fas.SetName(TCollection_HAsciiString(""))
    arr1 = StepVisual_HArray1OfFillStyleSelect(1, 1)
    sel1 = StepVisual_FillStyleSelect()
    sel1.SetValue(fasc)
    arr1.SetValue(1, sel1)
    fas.SetFillStyles(arr1)
    step_model.AddEntity(fas)

    ssfa = StepVisual_SurfaceStyleFillArea()
    ssfa.SetFillArea(fas)
    step_model.AddEntity(ssfa)

    sss = StepVisual_SurfaceSideStyle()
    sss.SetName(TCollection_HAsciiString(""))
    arr2 = StepVisual_HArray1OfSurfaceStyleElementSelect(1, 1)
    sel2 = StepVisual_SurfaceStyleElementSelect()
    sel2.SetValue(ssfa)
    arr2.SetValue(1, sel2)
    sss.SetStyles(arr2)
    step_model.AddEntity(sss)

    ssu = StepVisual_SurfaceStyleUsage()
    ssu.SetSide(StepVisual_SurfaceSide.StepVisual_ssBoth)
    ssu.SetStyle(sss)
    step_model.AddEntity(ssu)

    psa = StepVisual_PresentationStyleAssignment()
    arr3 = StepVisual_HArray1OfPresentationStyleSelect(1, 1)
    sel3 = StepVisual_PresentationStyleSelect()
    sel3.SetValue(ssu)
    arr3.SetValue(1, sel3)
    psa.SetStyles(arr3)
    step_model.AddEntity(psa)

    return psa


def export_colored_step(step_path: str, labels: list, output_path: str,
                        colors_hex: list = None, num_cls: int = None):
    """Write a colored STEP file with per-face colors.

    Uses STEPConstruct_Styles to attach color entities directly to
    the STEP model (no XCAF/TDocStd_Document required).

    Args:
        step_path: Input STEP file path.
        labels: List of class IDs per face (TopExp_Explorer order).
        output_path: Output STEP file path.
        colors_hex: Hex color palette (defaults to CLASS_COLORS_HEX).
        num_cls: Number of classes (defaults to NUM_CLASSES).
    """
    if colors_hex is None:
        colors_hex = CLASS_COLORS_HEX
    if num_cls is None:
        num_cls = NUM_CLASSES
    from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
    from OCC.Core.STEPConstruct import STEPConstruct_Styles
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods
    from OCC.Core.StepVisual import StepVisual_StyledItem
    from OCC.Core.StepData import StepData_StepModel
    from OCC.Core.IFSelect import IFSelect_RetDone

    # Read shape
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        print(f"Failed to read STEP file: {step_path}")
        return False

    reader.TransferRoots()
    shape = reader.OneShape()

    # Transfer shape to writer (creates the STEP entity model)
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)

    ws = writer.WS()
    step_model = StepData_StepModel.DownCast(ws.Model())
    styles = STEPConstruct_Styles(ws)
    styles.Init(ws)

    # Get faces in TopExp_Explorer order
    faces = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        faces.append(topods.Face(explorer.Current()))
        explorer.Next()

    # Build color PSAs (one per unique color) and assign to faces
    null_override = StepVisual_StyledItem()
    color_cache = {}

    for i, face in enumerate(faces):
        if i < len(labels):
            label = labels[i]
            if 0 <= label < num_cls:
                r, g, b = hex_to_rgb01(colors_hex[label])
            else:
                r, g, b = hex_to_rgb01(UNLABELED_COLOR_HEX)
        else:
            r, g, b = hex_to_rgb01(UNLABELED_COLOR_HEX)

        # Cache PSAs by color to avoid creating duplicate STEP entities
        key = (round(r, 4), round(g, 4), round(b, 4))
        if key not in color_cache:
            color_cache[key] = _build_step_color_psa(r, g, b, step_model)

        styles.AddStyle(face, color_cache[key], null_override)

    # Add styled items to the model
    for i in range(1, styles.NbStyles() + 1):
        step_model.AddEntity(styles.Style(i))

    # Write colored STEP
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer.Write(str(out_path))

    return True


def main():
    """Main export function."""
    args = parse_args()

    if args.step is None and args.step_dir is None:
        print("Error: Provide either --step or --step_dir")
        sys.exit(1)

    if args.step:
        # Single file mode
        print(f"Processing {args.step}...")

        # Determine label source
        labels = None
        if args.seg:
            labels = load_seg_labels(args.seg)
            print(f"  Loaded {len(labels)} labels from {args.seg}")
        elif args.seg_dir:
            model_id = Path(args.step).stem
            seg_file = Path(args.seg_dir) / f"{model_id}.seg"
            if seg_file.exists():
                labels = load_seg_labels(str(seg_file))
                print(f"  Loaded {len(labels)} labels from {seg_file}")
        if labels is None and args.labels_json:
            labels = load_labels_from_json(args.labels_json)
            if labels:
                print(f"  Loaded {len(labels)} labels from {args.labels_json}")
        if labels is None and args.labels_dir:
            model_id = Path(args.step).stem
            label_file = Path(args.labels_dir) / f"{model_id}_result.json"
            if label_file.exists():
                labels = load_labels_from_json(str(label_file))
                print(f"  Loaded labels from {label_file}")
        if labels is None and args.checkpoint:
            print(f"  Running inference with {args.checkpoint}...")
            labels = run_inference_for_labels(args.step, args.checkpoint)

        if labels is None:
            print("Error: No labels available. Provide --seg, --seg_dir, --labels_json, --labels_dir, or --checkpoint")
            sys.exit(1)

        if args.real_classes:
            labels = map_labels_to_real(labels)

        colors_hex = REAL_CLASS_COLORS_HEX if args.real_classes else CLASS_COLORS_HEX
        num_cls = REAL_NUM_CLASSES if args.real_classes else NUM_CLASSES
        output = args.output or str(Path(args.step).with_suffix(".colored.step"))
        export_colored_step(args.step, labels, output, colors_hex=colors_hex, num_cls=num_cls)

        # Optional defeaturing
        if args.defeature and labels is not None:
            from brepformer.defeature import defeature_step
            from brepformer.data.classes import CLASS_TO_DEFEATURE, DEFEATURE_NUM_CLASSES as _DN

            # Remap to defeature classes if needed
            defeature_labels = labels
            if num_cls > _DN:
                defeature_labels = [
                    CLASS_TO_DEFEATURE[l] if 0 <= l < NUM_CLASSES else 0
                    for l in labels
                ]
            defeature_dir = Path(args.defeature_output_dir)
            defeature_dir.mkdir(parents=True, exist_ok=True)
            model_id = Path(args.step).stem
            out_def = defeature_dir / f"{model_id}_defeatured.step"
            print(f"  Defeaturing -> {out_def}...")
            result = defeature_step(args.step, defeature_labels, str(out_def))
            if result["status"] == "success":
                print(f"  Defeatured: removed {result['removed']} faces "
                      f"(valid: {result['valid']})")
            else:
                print(f"  Defeature: {result.get('message', result['status'])}")

    elif args.step_dir:
        # Batch mode
        step_dir = Path(args.step_dir)
        step_files = sorted(list(step_dir.glob("*.step")) + list(step_dir.glob("*.stp")))
        print(f"Found {len(step_files)} STEP files")

        output_dir = Path(args.output_dir) if args.output_dir else step_dir / "colored"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load model once if using checkpoint
        model = None
        device = None
        if args.checkpoint:
            import torch
            from brepformer.infer import load_model
            model = load_model(args.checkpoint)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

        for step_file in step_files:
            print(f"\nProcessing {step_file.name}...")
            model_id = step_file.stem

            labels = None

            if args.seg_dir:
                seg_file = Path(args.seg_dir) / f"{model_id}.seg"
                if seg_file.exists():
                    labels = load_seg_labels(str(seg_file))

            if labels is None and args.labels_dir:
                label_file = Path(args.labels_dir) / f"{model_id}_result.json"
                if label_file.exists():
                    labels = load_labels_from_json(str(label_file))

            if labels is None and model is not None:
                from brepformer.infer import infer_single
                result = infer_single(model, str(step_file), device)
                if "error" not in result:
                    labels = result.get("face_preds", None)

            if labels is None:
                print(f"  Skipping {step_file.name}: no labels available")
                continue

            if args.real_classes:
                labels = map_labels_to_real(labels)

            colors_hex = REAL_CLASS_COLORS_HEX if args.real_classes else CLASS_COLORS_HEX
            num_cls = REAL_NUM_CLASSES if args.real_classes else NUM_CLASSES
            output = output_dir / f"{model_id}.colored.step"
            export_colored_step(str(step_file), labels, str(output),
                                colors_hex=colors_hex, num_cls=num_cls)

            # Optional defeaturing
            if args.defeature and labels is not None:
                from brepformer.defeature import defeature_step
                from brepformer.data.classes import (
                    CLASS_TO_DEFEATURE, DEFEATURE_NUM_CLASSES as _DN,
                )

                defeature_labels = labels
                if num_cls > _DN:
                    defeature_labels = [
                        CLASS_TO_DEFEATURE[l] if 0 <= l < NUM_CLASSES else 0
                        for l in labels
                    ]
                defeature_dir = Path(args.defeature_output_dir)
                defeature_dir.mkdir(parents=True, exist_ok=True)
                out_def = defeature_dir / f"{model_id}_defeatured.step"
                result = defeature_step(
                    str(step_file), defeature_labels, str(out_def),
                )
                if result["status"] == "success":
                    print(f"  Defeatured: removed {result['removed']} faces")

        print(f"\nBatch export complete. Files saved to {output_dir}")


if __name__ == "__main__":
    main()
