#!/usr/bin/env python3
"""Per-face segmentation viewer for pipe fitting STEP files.

Qt + pythonOCC 3D viewer for visualizing per-face machining feature classes
on pipe fitting STEP models. Each face is colored individually by its
27-class or 8-real-class prediction from BrepFormer face segmentation.

Optionally shows whole-model pipe class in the status bar via --pipe_checkpoint.

Usage:
    # From .seg file
    python brepclassifier/visualize_face_seg.py --step model.step --seg preds.seg

    # Run face inference
    python brepclassifier/visualize_face_seg.py --step model.step \
        --checkpoint results/face_seg_heavy/best-epoch=46-val/f1=0.8608.ckpt

    # Browse directory with inference
    python brepclassifier/visualize_face_seg.py \
        --step_dir brepclassifier/data/ssdata1/steps/ \
        --checkpoint results/face_seg_heavy/best-epoch=46-val/f1=0.8608.ckpt

    # With 8 real class remapping
    python brepclassifier/visualize_face_seg.py --step model.step --seg preds.seg \
        --real_classes

    # With pipe class in status bar
    python brepclassifier/visualize_face_seg.py --step model.step \
        --checkpoint results/face_seg_heavy/best.ckpt \
        --pipe_checkpoint results/pipe_classifier/best.ckpt

Keyboard Controls:
    D/RIGHT     Next model (batch mode)
    A/LEFT      Previous model (batch mode)
    T           Cycle display: GT -> Predicted -> Comparison
    I           Print face info to console
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

# Face segmentation class definitions from brepformer
from brepformer.data.classes import (
    CLASS_NAMES, CLASS_COLORS_HEX, NUM_CLASSES,
    REAL_CLASS_NAMES, REAL_CLASS_COLORS_HEX, REAL_NUM_CLASSES,
    CLASS_TO_REAL_CLASS, map_labels_to_real,
    UNLABELED_COLOR_HEX, HIGHLIGHT_COLOR_HEX, EDGE_COLOR_HEX,
    hex_to_rgb01, get_class_name, get_real_class_name,
)


def rgb01_to_quantity(rgb: Tuple[float, float, float]) -> Quantity_Color:
    return Quantity_Color(rgb[0], rgb[1], rgb[2], Quantity_TOC_RGB)


def text_color_for_bg(color_hex: str) -> str:
    r, g, b = hex_to_rgb01(color_hex)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b)
    return "#000000" if luminance > 0.6 else "#ffffff"


def load_seg_labels(seg_path: str) -> List[int]:
    labels: List[int] = []
    with open(seg_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(int(line))
    return labels


def load_labels_from_json(json_path: str) -> Optional[List[int]]:
    """Load per-face labels from a _result.json file."""
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


@dataclass
class FaceItem:
    face: object
    ais: object


class FaceSegViewer(QtWidgets.QMainWindow):
    """Per-face segmentation viewer (27-class or 8 real-class modes)."""

    # Display modes
    MODE_GT = "gt"
    MODE_PRED = "pred"
    MODE_COMPARE = "compare"

    def __init__(
        self,
        step_path: Optional[str] = None,
        seg_path: Optional[str] = None,
        labels_json: Optional[str] = None,
        labels_dir: Optional[str] = None,
        checkpoint: Optional[str] = None,
        step_dir: Optional[str] = None,
        seg_dir: Optional[str] = None,
        real_classes: bool = False,
        pipe_checkpoint: Optional[str] = None,
    ):
        super().__init__()
        self.real_classes = real_classes

        # Select class palette based on mode
        if self.real_classes:
            self._class_names = REAL_CLASS_NAMES
            self._class_colors_hex = REAL_CLASS_COLORS_HEX
            self._num_classes = REAL_NUM_CLASSES
            self._get_class_name = get_real_class_name
        else:
            self._class_names = CLASS_NAMES
            self._class_colors_hex = CLASS_COLORS_HEX
            self._num_classes = NUM_CLASSES
            self._get_class_name = get_class_name

        title = "Face Seg Viewer (brepclassifier)"
        if self.real_classes:
            title += " [Real Classes]"
        self.setWindowTitle(title)
        self.resize(1600, 1000)
        self.setMinimumSize(1200, 800)

        self.class_colors_rgb = [hex_to_rgb01(c) for c in self._class_colors_hex]
        self.class_colors = [rgb01_to_quantity(c) for c in self.class_colors_rgb]
        self.unlabeled_color = rgb01_to_quantity(hex_to_rgb01(UNLABELED_COLOR_HEX))
        self.highlight_color = rgb01_to_quantity(hex_to_rgb01(HIGHLIGHT_COLOR_HEX))

        self.face_items: List[FaceItem] = []
        self.gt_labels: Optional[List[int]] = None
        self.pred_labels: Optional[List[int]] = None
        self.current_index: Optional[int] = None
        self.highlight_enabled = True
        self.display_mode = self.MODE_GT
        self.step_path: Optional[str] = None
        self.checkpoint = checkpoint
        self.labels_dir = labels_dir
        self.seg_dir = seg_dir
        self.pipe_checkpoint = pipe_checkpoint
        self.pipe_class: Optional[int] = None

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
            self.load_step(step_path)
            if seg_path:
                self._load_pred_labels(seg_path)
            if labels_json:
                self._load_gt_from_json(labels_json)
            if labels_dir:
                self._auto_load_gt_labels()
            self._auto_load_pred_labels()
            if self.pred_labels and not self.gt_labels:
                self.display_mode = self.MODE_PRED
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

        # Buttons
        self.btn_import_step = QtWidgets.QPushButton("Import STEP")
        self.btn_import_step.clicked.connect(self.on_import_step)
        panel_layout.addWidget(self.btn_import_step)

        self.btn_import_seg = QtWidgets.QPushButton("Import .seg")
        self.btn_import_seg.clicked.connect(self.on_import_seg)
        panel_layout.addWidget(self.btn_import_seg)

        if self.checkpoint:
            self.btn_infer = QtWidgets.QPushButton("Run Face Inference")
            self.btn_infer.clicked.connect(self.on_run_inference)
            panel_layout.addWidget(self.btn_infer)

        # Display mode toggle
        self.btn_toggle_mode = QtWidgets.QPushButton("Display: GT")
        self.btn_toggle_mode.clicked.connect(self.on_toggle_mode)
        panel_layout.addWidget(self.btn_toggle_mode)

        self.btn_summary = QtWidgets.QPushButton("Summary")
        self.btn_summary.clicked.connect(self.on_summary)
        panel_layout.addWidget(self.btn_summary)

        self.chk_highlight = QtWidgets.QCheckBox("Highlight current")
        self.chk_highlight.setChecked(True)
        self.chk_highlight.toggled.connect(self.on_toggle_highlight)
        panel_layout.addWidget(self.chk_highlight)

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

        # Face navigation
        face_nav = QtWidgets.QHBoxLayout()
        self.btn_prev_face = QtWidgets.QPushButton("< Face")
        self.btn_prev_face.clicked.connect(self.on_prev_face)
        self.btn_next_face = QtWidgets.QPushButton("Face >")
        self.btn_next_face.clicked.connect(self.on_next_face)
        face_nav.addWidget(self.btn_prev_face)
        face_nav.addWidget(self.btn_next_face)
        panel_layout.addLayout(face_nav)

        # Status
        self.status_label = QtWidgets.QLabel("No STEP loaded")
        self.status_label.setWordWrap(True)
        panel_layout.addWidget(self.status_label)

        self.info_label = QtWidgets.QLabel("")
        self.info_label.setWordWrap(True)
        panel_layout.addWidget(self.info_label)

        panel_layout.addSpacing(8)

        # Legend
        legend_title = "8 Real Classes" if self.real_classes else "27 MFTRCAD Classes"
        legend_label = QtWidgets.QLabel(legend_title)
        legend_label.setStyleSheet("font-weight: bold;")
        panel_layout.addWidget(legend_label)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(500)
        legend_widget = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(legend_widget)
        grid.setSpacing(3)
        for idx in range(self._num_classes):
            item = QtWidgets.QLabel(f"{idx}: {self._class_names[idx]}")
            bg = self._class_colors_hex[idx]
            fg = text_color_for_bg(bg)
            item.setStyleSheet(
                f"background-color: {bg}; color: {fg}; padding: 2px 4px; "
                f"border-radius: 2px; font-size: 11px;"
            )
            grid.addWidget(item, idx, 0)
        scroll.setWidget(legend_widget)
        panel_layout.addWidget(scroll)

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
            self._print_face_info()
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
            self.load_step(path)
            self._auto_load_gt_labels()
            self._auto_load_pred_labels()
            self._apply_colors()

    def on_import_seg(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open SEG", "", "SEG Files (*.seg)"
        )
        if path:
            self._load_pred_labels(path)
            self._apply_colors()

    def on_run_inference(self) -> None:
        if not self.step_path or not self.checkpoint:
            return
        try:
            import torch
            from brepformer.infer import load_model, infer_single

            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            model = load_model(self.checkpoint)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            result = infer_single(model, self.step_path, device)
            QtWidgets.QApplication.restoreOverrideCursor()

            if "face_preds" in result:
                self.pred_labels = result["face_preds"]
                self.display_mode = self.MODE_PRED
                self._apply_colors()
                self._update_status()
            elif "error" in result:
                QtWidgets.QMessageBox.warning(
                    self, "Inference Error", result["error"]
                )
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Inference Error",
                    "Model does not have face segmentation head."
                )
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.warning(self, "Inference Error", str(e))

    def on_toggle_mode(self) -> None:
        modes = [self.MODE_GT, self.MODE_PRED, self.MODE_COMPARE]
        idx = modes.index(self.display_mode)
        self.display_mode = modes[(idx + 1) % len(modes)]
        self.btn_toggle_mode.setText(f"Display: {self.display_mode.upper()}")
        self._apply_colors()

    def on_summary(self) -> None:
        if not self.face_items:
            return
        labels = self._active_labels()
        counts = {}
        unlabeled = 0
        for label in labels:
            if label is None or label == -1:
                unlabeled += 1
            else:
                name = self._get_class_name(label)
                counts[name] = counts.get(name, 0) + 1

        mode_label = "8 Real Classes" if self.real_classes else "27 MFTRCAD Classes"
        lines = [mode_label, f"Total faces: {len(labels)}", f"Unlabeled: {unlabeled}", ""]
        for name, count in sorted(counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {name}: {count}")

        if self.pipe_class is not None:
            from brepclassifier.data.classes import get_class_name as get_pipe_name
            lines.append("")
            lines.append(f"Pipe class: {self.pipe_class} ({get_pipe_name(self.pipe_class)})")

        QtWidgets.QMessageBox.information(self, "Summary", "\n".join(lines))

    def on_toggle_highlight(self, checked: bool) -> None:
        self.highlight_enabled = checked
        self._apply_colors()

    def on_prev_model(self) -> None:
        if self.step_files and self.batch_index > 0:
            self._load_batch_item(self.batch_index - 1)

    def on_next_model(self) -> None:
        if self.step_files and self.batch_index < len(self.step_files) - 1:
            self._load_batch_item(self.batch_index + 1)

    def on_prev_face(self) -> None:
        if self.current_index is not None and self.current_index > 0:
            self._set_current_face(self.current_index - 1)

    def on_next_face(self) -> None:
        if self.current_index is not None and self.current_index < len(self.face_items) - 1:
            self._set_current_face(self.current_index + 1)

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
        self.gt_labels = None
        self.pred_labels = None
        self.current_index = None
        self.pipe_class = None

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
            self._set_current_face(0)
        self.display.Repaint()
        self._update_status()

        # Run pipe classifier if available
        self._run_pipe_classification()

    def _run_pipe_classification(self) -> None:
        """Run pipe classifier in the background for status display."""
        if not self.pipe_checkpoint or not self.step_path:
            return
        try:
            import torch
            from brepclassifier.visualize_seg import load_classifier, infer_step_classifier

            model = load_classifier(self.pipe_checkpoint)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            pred = infer_step_classifier(model, self.step_path, device)
            if pred != -1:
                self.pipe_class = pred
                self._update_status()
        except Exception:
            pass

    def _load_pred_labels(self, seg_path: str) -> None:
        self.pred_labels = load_seg_labels(seg_path)

    def _load_gt_from_json(self, json_path: str) -> None:
        self.gt_labels = load_labels_from_json(json_path)

    def _auto_load_gt_labels(self) -> None:
        """Try to auto-detect GT labels from labels_dir."""
        if not self.labels_dir or not self.step_path:
            return
        model_id = Path(self.step_path).stem
        label_file = Path(self.labels_dir) / f"{model_id}_result.json"
        if label_file.exists():
            self.gt_labels = load_labels_from_json(str(label_file))

    def _auto_load_pred_labels(self) -> None:
        """Try to auto-detect predicted labels from seg_dir."""
        if not self.seg_dir or not self.step_path:
            return
        model_id = Path(self.step_path).stem
        seg_file = Path(self.seg_dir) / f"{model_id}.seg"
        if seg_file.exists():
            self.pred_labels = load_seg_labels(str(seg_file))

    def _load_batch_item(self, index: int) -> None:
        self.batch_index = index
        step_path = self.step_files[index]
        self.load_step(step_path)
        self._auto_load_gt_labels()
        self._auto_load_pred_labels()
        if self.pred_labels and not self.gt_labels:
            self.display_mode = self.MODE_PRED
        self._apply_colors()

    def _active_labels(self) -> List[Optional[int]]:
        """Get labels for current display mode (remapped to real classes if enabled)."""
        n = len(self.face_items)
        if self.display_mode == self.MODE_GT and self.gt_labels:
            raw = [self.gt_labels[i] if i < len(self.gt_labels) else None for i in range(n)]
        elif self.display_mode == self.MODE_PRED and self.pred_labels:
            raw = [self.pred_labels[i] if i < len(self.pred_labels) else None for i in range(n)]
        elif self.display_mode == self.MODE_COMPARE:
            if self.gt_labels:
                raw = [self.gt_labels[i] if i < len(self.gt_labels) else None for i in range(n)]
            elif self.pred_labels:
                raw = [self.pred_labels[i] if i < len(self.pred_labels) else None for i in range(n)]
            else:
                raw = [None] * n
        else:
            raw = [None] * n

        if self.real_classes:
            return [CLASS_TO_REAL_CLASS[l] if l is not None and 0 <= l < NUM_CLASSES else l for l in raw]
        return raw

    def _apply_colors(self) -> None:
        labels = self._active_labels()
        for idx in range(len(self.face_items)):
            label = labels[idx]
            if label is not None and 0 <= label < self._num_classes:
                color = self.class_colors[label]
            else:
                color = self.unlabeled_color
            self._set_face_color(idx, color, repaint=False)

        if self.current_index is not None and self.highlight_enabled:
            self._set_face_color(self.current_index, self.highlight_color, repaint=False)

        self.display.Repaint()
        self._update_status()

    def _set_current_face(self, index: int) -> None:
        if not self.face_items:
            return
        index = max(0, min(index, len(self.face_items) - 1))

        # Restore previous face color
        if self.current_index is not None:
            labels = self._active_labels()
            label = labels[self.current_index]
            if label is not None and 0 <= label < self._num_classes:
                color = self.class_colors[label]
            else:
                color = self.unlabeled_color
            self._set_face_color(self.current_index, color)

        self.current_index = index
        if self.highlight_enabled:
            self._set_face_color(index, self.highlight_color)

        self._update_status()

    def _set_face_color(self, index: int, color: Quantity_Color, repaint: bool = True) -> None:
        ais = self.face_items[index].ais
        if isinstance(ais, list):
            for item in ais:
                self._set_ais_color(item, color)
        else:
            self._set_ais_color(ais, color)
        if repaint:
            self.display.Repaint()

    def _set_ais_color(self, ais, color: Quantity_Color) -> None:
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
        if self.real_classes:
            parts.append("Classes: 8 real")
        if self.gt_labels:
            parts.append("GT: loaded")
        if self.pred_labels:
            parts.append("Pred: loaded")
        if self.pipe_class is not None:
            from brepclassifier.data.classes import get_class_name as get_pipe_name
            parts.append(f"Pipe: {self.pipe_class} ({get_pipe_name(self.pipe_class)})")
        self.status_label.setText("\n".join(parts))

        # Face info
        if self.current_index is not None:
            labels = self._active_labels()
            label = labels[self.current_index]
            label_text = self._get_class_name(label) if label is not None and label >= 0 else "Unlabeled"
            self.info_label.setText(
                f"Face {self.current_index + 1}/{len(self.face_items)}\n"
                f"Class: {label} ({label_text})"
            )
        else:
            self.info_label.setText("")

        title = "Face Seg Viewer (brepclassifier)"
        if self.real_classes:
            title += " [Real Classes]"
        title += f" - {os.path.basename(self.step_path or '')}"
        self.setWindowTitle(title)

    def _print_face_info(self) -> None:
        if self.current_index is None:
            return
        idx = self.current_index
        gt = self.gt_labels[idx] if self.gt_labels and idx < len(self.gt_labels) else None
        pred = self.pred_labels[idx] if self.pred_labels and idx < len(self.pred_labels) else None
        if self.real_classes:
            gt_r = CLASS_TO_REAL_CLASS[gt] if gt is not None and 0 <= gt < NUM_CLASSES else gt
            pred_r = CLASS_TO_REAL_CLASS[pred] if pred is not None and 0 <= pred < NUM_CLASSES else pred
            print(f"Face {idx}: GT={gt_r} ({get_real_class_name(gt_r) if gt_r is not None else 'N/A'}), "
                  f"Pred={pred_r} ({get_real_class_name(pred_r) if pred_r is not None else 'N/A'})")
        else:
            print(f"Face {idx}: GT={gt} ({get_class_name(gt) if gt is not None else 'N/A'}), "
                  f"Pred={pred} ({get_class_name(pred) if pred is not None else 'N/A'})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Per-face segmentation viewer for pipe fitting STEP files"
    )
    parser.add_argument("--step", type=str, default=None, help="Path to STEP file")
    parser.add_argument("--seg", type=str, default=None, help="Path to .seg label file")
    parser.add_argument("--labels_json", type=str, default=None,
                        help="Path to GT face labels JSON file")
    parser.add_argument("--labels_dir", type=str, default=None,
                        help="Directory of face label JSON files")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="BrepFormer face seg checkpoint for inference")
    parser.add_argument("--step_dir", type=str, default=None,
                        help="Directory of STEP files for batch browsing")
    parser.add_argument("--seg_dir", type=str, default=None,
                        help="Directory of .seg prediction files")
    parser.add_argument("--real_classes", action="store_true",
                        help="Remap 27 MFTRCAD classes to 8 real categories")
    parser.add_argument("--pipe_checkpoint", type=str, default=None,
                        help="PipeFittingClassifier checkpoint (shows pipe class in status)")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    window = FaceSegViewer(
        step_path=args.step,
        seg_path=args.seg,
        labels_json=args.labels_json,
        labels_dir=args.labels_dir,
        checkpoint=args.checkpoint,
        step_dir=args.step_dir,
        seg_dir=args.seg_dir,
        real_classes=args.real_classes,
        pipe_checkpoint=args.pipe_checkpoint,
    )
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
