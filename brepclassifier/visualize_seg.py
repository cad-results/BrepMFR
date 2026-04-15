#!/usr/bin/env python3
"""Pipe fitting classification viewer.

Qt + pythonOCC 3D viewer for visualizing 8 pipe fitting classes on STEP models.
Whole-model coloring: all faces colored by the model-level class (GT or predicted).

Usage:
    # Single file with GT class
    python brepclassifier/visualize_seg.py --step model.step --gt_class 0

    # Batch mode with labels JSON
    python brepclassifier/visualize_seg.py \\
        --step_dir brepclassifier/data/ssdata1/steps/ \\
        --labels_json brepclassifier/data/ssdata1/labels.json

    # Run inference and display
    python brepclassifier/visualize_seg.py \\
        --step model.step \\
        --gt_class 2 \\
        --checkpoint results/pipe_classifier/best.ckpt

Keyboard Controls:
    D/RIGHT     Next model (batch mode)
    A/LEFT      Previous model (batch mode)
    T           Cycle display: GT -> Predicted -> Comparison
    I           Print model info to console
    S           Summary dialog
    ESC/Q       Quit
"""
from __future__ import annotations

import sys
import os
import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
# Compatibility shim: older pytorch_lightning uses np.Inf, removed in NumPy 2.0
if not hasattr(np, 'Inf'):
    np.Inf = np.inf

import torch

# Qt binding selection + pythonocc backend init
try:
    from PyQt5 import QtCore, QtWidgets
    _qt_backend = "pyqt5"
except ImportError:
    from PySide2 import QtCore, QtWidgets
    _qt_backend = "pyside2"

from OCC.Display.backend import load_backend
load_backend(_qt_backend)

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Display.qtDisplay import qtViewer3d

try:
    from OCC.Core.Aspect import Aspect_TOL_SOLID
    from OCC.Core.Prs3d import Prs3d_LineAspect
except Exception:
    Aspect_TOL_SOLID = None
    Prs3d_LineAspect = None
try:
    from OCC.Core.Graphic3d import Graphic3d_NOM_MATTE, Graphic3d_NOM_NEON
except Exception:
    Graphic3d_NOM_MATTE = None
    Graphic3d_NOM_NEON = None
try:
    from OCC.Core.Graphic3d import Graphic3d_TOSM_UNLIT
except Exception:
    Graphic3d_TOSM_UNLIT = None

from brepclassifier.data.classes import (
    CLASS_NAMES, CLASS_COLORS_HEX, NUM_CLASSES,
    UNLABELED_COLOR_HEX, HIGHLIGHT_COLOR_HEX, EDGE_COLOR_HEX,
    hex_to_rgb01, get_class_name,
)

# Face segmentation class definitions (27 / 8 real classes) for face_seg mode
from brepformer.data.classes import (
    CLASS_NAMES as FACE_CLASS_NAMES,
    CLASS_COLORS_HEX as FACE_CLASS_COLORS_HEX,
    NUM_CLASSES as FACE_NUM_CLASSES,
    REAL_CLASS_NAMES as FACE_REAL_CLASS_NAMES,
    REAL_CLASS_COLORS_HEX as FACE_REAL_CLASS_COLORS_HEX,
    REAL_NUM_CLASSES as FACE_REAL_NUM_CLASSES,
    get_class_name as get_face_class_name,
    get_real_class_name as get_face_real_class_name,
)


def rgb01_to_quantity(rgb: Tuple[float, float, float]) -> Quantity_Color:
    return Quantity_Color(rgb[0], rgb[1], rgb[2], Quantity_TOC_RGB)


def text_color_for_bg(color_hex: str) -> str:
    r, g, b = hex_to_rgb01(color_hex)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b)
    return "#000000" if luminance > 0.6 else "#ffffff"


def load_labels_json(json_path: str) -> Dict[str, int]:
    """Load labels.json mapping model_id -> class_idx."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return {str(k): int(v) for k, v in data.items()}


def infer_step_classifier(model, step_path: str, device) -> int:
    """Run inference on a single STEP file, returns predicted class index or -1."""
    from brepformer.data.step_to_graph import step_to_preprocessed_sample
    from brepformer.data.collator import BrepCollator

    sample = step_to_preprocessed_sample(step_path)
    if sample is None:
        return -1
    tensor_sample = {
        k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
        for k, v in sample.items()
    }
    batch = BrepCollator()([tensor_sample])
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    with torch.no_grad():
        logits = model(batch)
    return int(logits.argmax(dim=-1).item())


def load_classifier(checkpoint_path: str):
    """Load PipeFittingClassifier from checkpoint."""
    import pathlib
    from brepformer.configs.config import BrepClassifierConfig
    from brepclassifier.configs.config import PipeFittingConfig
    from brepclassifier.models.pipe_classifier import PipeFittingClassifier

    torch.serialization.add_safe_globals([
        pathlib.PosixPath, BrepClassifierConfig, PipeFittingConfig,
    ])
    model = PipeFittingClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


@dataclass
class FaceItem:
    face: object
    ais: object


class ClassifierViewer(QtWidgets.QMainWindow):
    """8-class pipe fitting viewer (whole-model classification + optional face seg)."""

    MODE_GT = "gt"
    MODE_PRED = "pred"
    MODE_COMPARE = "compare"
    MODE_FACE_SEG = "face_seg"

    def __init__(
        self,
        step_path: Optional[str] = None,
        gt_class: Optional[int] = None,
        labels_json: Optional[str] = None,
        checkpoint: Optional[str] = None,
        step_dir: Optional[str] = None,
        face_checkpoint: Optional[str] = None,
    ):
        super().__init__()
        self.setWindowTitle("Pipe Fitting Classifier Viewer")
        self.resize(1400, 900)
        self.setMinimumSize(1100, 700)

        self.class_colors_rgb = [hex_to_rgb01(c) for c in CLASS_COLORS_HEX]
        self.class_colors = [rgb01_to_quantity(c) for c in self.class_colors_rgb]
        self.unlabeled_color = rgb01_to_quantity(hex_to_rgb01(UNLABELED_COLOR_HEX))

        # Face seg colors (27-class)
        self.face_seg_colors = [
            rgb01_to_quantity(hex_to_rgb01(c)) for c in FACE_CLASS_COLORS_HEX
        ]

        self.face_items: List[FaceItem] = []
        self.gt_class: Optional[int] = gt_class
        self.pred_class: Optional[int] = None
        self.face_preds: Optional[List[int]] = None
        self.display_mode = self.MODE_GT
        self.step_path: Optional[str] = None
        self.checkpoint = checkpoint
        self.face_checkpoint = face_checkpoint
        self.model = None
        self.face_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Labels JSON: model_id -> class_idx
        self.labels_map: Dict[str, int] = {}
        if labels_json:
            self.labels_map = load_labels_json(labels_json)

        # Batch mode
        self.step_files: List[str] = []
        self.batch_index = 0

        if step_dir:
            step_dir_path = Path(step_dir)
            self.step_files = sorted(
                [str(f) for f in step_dir_path.glob("*.step")] +
                [str(f) for f in step_dir_path.glob("*.stp")]
            )

        self._build_ui()
        self._update_status()

        if step_path:
            self._gt_class_override = gt_class
            self.load_step(step_path)
            self._apply_colors()
        elif self.step_files:
            self._load_batch_item(0)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        root_layout = QtWidgets.QHBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        # 3D viewer
        self.viewer = qtViewer3d(central)
        self.viewer.InitDriver()
        self.display = self.viewer._display
        try:
            self.display.Context.SetAutomaticHilight(False)
        except Exception:
            pass
        self._configure_viewer_visuals()
        root_layout.addWidget(self.viewer, 1)

        # Side panel
        panel = QtWidgets.QWidget(central)
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(6)
        root_layout.addWidget(panel, 0)

        self.btn_import_step = QtWidgets.QPushButton("Import STEP")
        self.btn_import_step.clicked.connect(self.on_import_step)
        panel_layout.addWidget(self.btn_import_step)

        if self.checkpoint:
            self.btn_infer = QtWidgets.QPushButton("Run Inference")
            self.btn_infer.clicked.connect(self.on_run_inference)
            panel_layout.addWidget(self.btn_infer)

        if self.face_checkpoint:
            self.btn_face_infer = QtWidgets.QPushButton("Run Face Inference")
            self.btn_face_infer.clicked.connect(self.on_run_face_inference)
            panel_layout.addWidget(self.btn_face_infer)

        self.btn_toggle_mode = QtWidgets.QPushButton("Display: GT")
        self.btn_toggle_mode.clicked.connect(self.on_toggle_mode)
        panel_layout.addWidget(self.btn_toggle_mode)

        self.btn_summary = QtWidgets.QPushButton("Summary")
        self.btn_summary.clicked.connect(self.on_summary)
        panel_layout.addWidget(self.btn_summary)

        panel_layout.addSpacing(8)

        # Batch navigation
        nav_layout = QtWidgets.QHBoxLayout()
        self.btn_prev = QtWidgets.QPushButton("<< Prev")
        self.btn_prev.clicked.connect(self.on_prev_model)
        self.btn_next = QtWidgets.QPushButton("Next >>")
        self.btn_next.clicked.connect(self.on_next_model)
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        panel_layout.addLayout(nav_layout)

        # Status
        self.status_label = QtWidgets.QLabel("No STEP loaded")
        self.status_label.setWordWrap(True)
        panel_layout.addWidget(self.status_label)

        panel_layout.addSpacing(8)

        # Legend (8 classes, compact)
        legend_label = QtWidgets.QLabel("8 Pipe Fitting Classes")
        legend_label.setStyleSheet("font-weight: bold;")
        panel_layout.addWidget(legend_label)

        legend_widget = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(legend_widget)
        grid.setSpacing(3)
        for idx in range(NUM_CLASSES):
            item = QtWidgets.QLabel(f"{idx}: {CLASS_NAMES[idx]}")
            bg = CLASS_COLORS_HEX[idx]
            fg = text_color_for_bg(bg)
            item.setStyleSheet(
                f"background-color: {bg}; color: {fg}; padding: 2px 4px; "
                f"border-radius: 2px; font-size: 11px;"
            )
            grid.addWidget(item, idx, 0)
        panel_layout.addWidget(legend_widget)

        panel_layout.addStretch(1)
        self.setCentralWidget(central)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (QtCore.Qt.Key_Right, QtCore.Qt.Key_D):
            self.on_next_model()
        elif key in (QtCore.Qt.Key_Left, QtCore.Qt.Key_A):
            self.on_prev_model()
        elif key == QtCore.Qt.Key_T:
            self.on_toggle_mode()
        elif key == QtCore.Qt.Key_I:
            self._print_model_info()
        elif key == QtCore.Qt.Key_S:
            self.on_summary()
        elif key in (QtCore.Qt.Key_Escape, QtCore.Qt.Key_Q):
            self.close()
        else:
            super().keyPressEvent(event)

    def on_import_step(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open STEP", "", "STEP Files (*.stp *.step)"
        )
        if path:
            self._gt_class_override = None
            self.load_step(path)
            self._apply_colors()

    def on_run_inference(self) -> None:
        if not self.step_path or not self.checkpoint:
            return
        try:
            if self.model is None:
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
                self.model = load_classifier(self.checkpoint)
                self.model = self.model.to(self.device)
                QtWidgets.QApplication.restoreOverrideCursor()

            pred = infer_step_classifier(self.model, self.step_path, self.device)
            if pred == -1:
                QtWidgets.QMessageBox.warning(self, "Inference Error",
                                              "Failed to preprocess STEP file.")
                return
            self.pred_class = pred
            self.display_mode = self.MODE_PRED
            self.btn_toggle_mode.setText(f"Display: {self.display_mode.upper()}")
            self._apply_colors()
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.warning(self, "Inference Error", str(e))

    def on_run_face_inference(self) -> None:
        """Run BrepFormer face segmentation inference."""
        if not self.step_path or not self.face_checkpoint:
            return
        try:
            from brepformer.infer import load_model, infer_single

            if self.face_model is None:
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
                self.face_model = load_model(self.face_checkpoint)
                self.face_model = self.face_model.to(self.device)
                QtWidgets.QApplication.restoreOverrideCursor()

            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            result = infer_single(self.face_model, self.step_path, self.device)
            QtWidgets.QApplication.restoreOverrideCursor()

            if "face_preds" in result:
                self.face_preds = result["face_preds"]
                self.display_mode = self.MODE_FACE_SEG
                self.btn_toggle_mode.setText(f"Display: {self.display_mode.upper()}")
                self._apply_colors()
            elif "error" in result:
                QtWidgets.QMessageBox.warning(self, "Face Inference Error", result["error"])
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Face Inference Error",
                    "Model does not have face segmentation head."
                )
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.warning(self, "Face Inference Error", str(e))

    def on_toggle_mode(self) -> None:
        modes = [self.MODE_GT, self.MODE_PRED, self.MODE_COMPARE]
        if self.face_preds is not None:
            modes.append(self.MODE_FACE_SEG)
        try:
            idx = modes.index(self.display_mode)
        except ValueError:
            idx = -1
        self.display_mode = modes[(idx + 1) % len(modes)]
        self.btn_toggle_mode.setText(f"Display: {self.display_mode.upper()}")
        self._apply_colors()

    def on_summary(self) -> None:
        gt_name = get_class_name(self.gt_class) if self.gt_class is not None else "N/A"
        pred_name = get_class_name(self.pred_class) if self.pred_class is not None else "N/A"
        lines = [
            f"STEP: {os.path.basename(self.step_path or 'N/A')}",
            f"Faces: {len(self.face_items)}",
            "",
            f"GT class:   {self.gt_class} ({gt_name})",
            f"Pred class: {self.pred_class} ({pred_name})",
        ]
        if self.gt_class is not None and self.pred_class is not None:
            correct = "✓ Correct" if self.gt_class == self.pred_class else "✗ Incorrect"
            lines.append(f"Result: {correct}")
        if self.face_preds is not None:
            lines.append("")
            lines.append(f"Face seg: {len(self.face_preds)} predictions")
            from collections import Counter
            counts = Counter(self.face_preds)
            for cls_id, count in sorted(counts.items()):
                lines.append(f"  {get_face_class_name(cls_id)}: {count}")
        QtWidgets.QMessageBox.information(self, "Model Summary", "\n".join(lines))

    def on_prev_model(self) -> None:
        if self.step_files and self.batch_index > 0:
            self._load_batch_item(self.batch_index - 1)

    def on_next_model(self) -> None:
        if self.step_files and self.batch_index < len(self.step_files) - 1:
            self._load_batch_item(self.batch_index + 1)

    def load_step(self, path: str) -> None:
        reader = STEPControl_Reader()
        status = reader.ReadFile(path)
        if status != IFSelect_RetDone:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to read: {path}")
            return
        reader.TransferRoots()
        shape = reader.OneShape()
        self.step_path = path

        self.display.EraseAll()
        self.face_items.clear()
        self.pred_class = None
        self.face_preds = None

        # Look up GT class from labels_map
        if hasattr(self, '_gt_class_override') and self._gt_class_override is not None:
            self.gt_class = self._gt_class_override
        else:
            model_id = Path(path).stem
            self.gt_class = self.labels_map.get(model_id)

        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = topods.Face(explorer.Current())
            ais = self.display.DisplayShape(face, update=False, color=self.unlabeled_color)
            if isinstance(ais, list):
                ais = ais[0]
            self._apply_face_material(ais)
            try:
                self.display.Context.SetDisplayMode(ais, 1, False)
            except Exception:
                pass
            self.face_items.append(FaceItem(face=face, ais=ais))
            explorer.Next()

        if self.face_items:
            self.display.FitAll()
        self.display.Repaint()
        self._update_status()

    def _load_batch_item(self, index: int) -> None:
        self.batch_index = index
        self._gt_class_override = None
        self.load_step(self.step_files[index])
        self._apply_colors()

    def _get_active_color(self) -> Optional[Quantity_Color]:
        """Get the single color to apply to all faces based on display mode."""
        if self.display_mode == self.MODE_GT:
            if self.gt_class is not None and 0 <= self.gt_class < NUM_CLASSES:
                return self.class_colors[self.gt_class]
        elif self.display_mode == self.MODE_PRED:
            if self.pred_class is not None and 0 <= self.pred_class < NUM_CLASSES:
                return self.class_colors[self.pred_class]
        elif self.display_mode == self.MODE_COMPARE:
            if self.gt_class is not None and 0 <= self.gt_class < NUM_CLASSES:
                return self.class_colors[self.gt_class]
        return None

    def _apply_colors(self) -> None:
        if self.display_mode == self.MODE_FACE_SEG and self.face_preds is not None:
            # Per-face coloring from face segmentation
            for idx, item in enumerate(self.face_items):
                if idx < len(self.face_preds):
                    cls_id = self.face_preds[idx]
                    if 0 <= cls_id < FACE_NUM_CLASSES:
                        color = self.face_seg_colors[cls_id]
                    else:
                        color = self.unlabeled_color
                else:
                    color = self.unlabeled_color
                self._set_ais_color(item.ais, color, repaint=False)
        else:
            color = self._get_active_color() or self.unlabeled_color
            for item in self.face_items:
                self._set_ais_color(item.ais, color, repaint=False)
        self.display.Repaint()
        self._update_status()

    def _set_ais_color(self, ais, color: Quantity_Color, repaint: bool = True) -> None:
        if isinstance(ais, list):
            for item in ais:
                self._set_single_ais_color(item, color)
        else:
            self._set_single_ais_color(ais, color)
        if repaint:
            self.display.Repaint()

    def _set_single_ais_color(self, ais, color: Quantity_Color) -> None:
        try:
            ais.SetColor(color)
        except Exception:
            self.display.Context.SetColor(ais, color, False)
        self._apply_face_material(ais)
        self.display.Context.Redisplay(ais, False)

    def _apply_face_material(self, ais) -> None:
        applied = False
        if Graphic3d_NOM_NEON is not None:
            try:
                ais.SetMaterial(Graphic3d_NOM_NEON)
                applied = True
            except Exception:
                pass
        if not applied and Graphic3d_NOM_MATTE is not None:
            try:
                ais.SetMaterial(Graphic3d_NOM_MATTE)
            except Exception:
                pass
        self._apply_face_edges(ais)

    def _configure_viewer_visuals(self) -> None:
        if Graphic3d_TOSM_UNLIT is None:
            return
        try:
            self.display.View.SetShadingModel(Graphic3d_TOSM_UNLIT)
        except Exception:
            pass

    def _apply_face_edges(self, ais) -> None:
        if Prs3d_LineAspect is None or Aspect_TOL_SOLID is None:
            return
        try:
            drawer = ais.Attributes()
            drawer.SetFaceBoundaryDraw(True)
            line_aspect = Prs3d_LineAspect(
                rgb01_to_quantity(hex_to_rgb01(EDGE_COLOR_HEX)),
                Aspect_TOL_SOLID, 1.0,
            )
            drawer.SetFaceBoundaryAspect(line_aspect)
        except Exception:
            pass

    def _update_status(self) -> None:
        parts = []
        if self.step_path:
            parts.append(f"STEP: {os.path.basename(self.step_path)}")
        if self.step_files:
            parts.append(f"Model {self.batch_index + 1}/{len(self.step_files)}")
        parts.append(f"Faces: {len(self.face_items)}")
        parts.append(f"Mode: {self.display_mode.upper()}")

        gt_name = get_class_name(self.gt_class) if self.gt_class is not None else "N/A"
        pred_name = get_class_name(self.pred_class) if self.pred_class is not None else "N/A"
        parts.append(f"GT: {self.gt_class} ({gt_name})")
        parts.append(f"Pred: {self.pred_class} ({pred_name})")

        if self.display_mode == self.MODE_COMPARE and \
                self.gt_class is not None and self.pred_class is not None:
            indicator = "✓" if self.gt_class == self.pred_class else "✗"
            parts.append(f"Result: {indicator}")

        if self.face_preds is not None:
            parts.append(f"Face seg: {len(self.face_preds)} preds")

        self.status_label.setText("\n".join(parts))
        self.setWindowTitle(
            f"Pipe Fitting Viewer - {os.path.basename(self.step_path or '')}"
        )

    def _print_model_info(self) -> None:
        gt_name = get_class_name(self.gt_class) if self.gt_class is not None else "N/A"
        pred_name = get_class_name(self.pred_class) if self.pred_class is not None else "N/A"
        print(f"STEP: {self.step_path}")
        print(f"  Faces: {len(self.face_items)}")
        print(f"  GT:   {self.gt_class} ({gt_name})")
        print(f"  Pred: {self.pred_class} ({pred_name})")
        if self.gt_class is not None and self.pred_class is not None:
            print(f"  Correct: {self.gt_class == self.pred_class}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pipe fitting classification viewer (Qt + pythonOCC)"
    )
    parser.add_argument("--step", type=str, default=None,
                        help="Single STEP file to display")
    parser.add_argument("--gt_class", type=int, default=None,
                        help="Ground-truth class (0-7) for the given STEP file")
    parser.add_argument("--labels_json", type=str, default=None,
                        help="labels.json with model_id -> class_idx mapping")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="PipeFittingClassifier checkpoint (enables Run Inference button)")
    parser.add_argument("--face_checkpoint", type=str, default=None,
                        help="BrepFormer face seg checkpoint (enables Run Face Inference button)")
    parser.add_argument("--step_dir", type=str, default=None,
                        help="Directory of STEP files for batch browsing")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    window = ClassifierViewer(
        step_path=args.step,
        gt_class=args.gt_class,
        labels_json=args.labels_json,
        checkpoint=args.checkpoint,
        step_dir=args.step_dir,
        face_checkpoint=args.face_checkpoint,
    )
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
