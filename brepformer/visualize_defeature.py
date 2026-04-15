#!/usr/bin/env python3
"""BrepFormer defeature comparison viewer.

Qt + pythonOCC 3D viewer for comparing original STEP files (colored by
defeature class predictions or GT labels) side-by-side with their
defeatured counterparts.

Usage:
    # Single file comparison
    python -m brepformer.visualize_defeature --step original.step --defeatured original_defeatured.step

    # With .seg labels on the original
    python -m brepformer.visualize_defeature --step original.step --defeatured original_defeatured.step --seg preds.seg

    # With GT labels
    python -m brepformer.visualize_defeature --step original.step --defeatured original_defeatured.step --labels_json labels.json

    # Batch browsing
    python -m brepformer.visualize_defeature --step_dir steps/ --defeatured_dir defeatured/ --seg_dir segs/

    # With inference
    python -m brepformer.visualize_defeature --step_dir steps/ --defeatured_dir defeatured/ --checkpoint best.ckpt

Keyboard Controls:
    D/RIGHT     Next model (batch mode)
    A/LEFT      Previous model (batch mode)
    T           Toggle original coloring: GT <-> Predicted
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

from brepformer.data.classes import (
    DEFEATURE_CLASS_NAMES, DEFEATURE_CLASS_COLORS_HEX, DEFEATURE_NUM_CLASSES,
    get_defeature_class_name,
    UNLABELED_COLOR_HEX, HIGHLIGHT_COLOR_HEX, EDGE_COLOR_HEX,
    hex_to_rgb01, NUM_CLASSES,
)


# ---------------------------------------------------------------------------
# Utility functions (self-contained, copied from visualize_seg.py)
# ---------------------------------------------------------------------------

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


def load_labels_from_json(json_path: str, num_classes: int = DEFEATURE_NUM_CLASSES) -> Optional[List[int]]:
    """Load per-face labels from a _result.json file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    if "cls" not in data:
        return None
    max_face = max(int(k) for k in data["cls"].keys())
    labels = [-1] * (max_face + 1)
    for face_id_str, class_id in data["cls"].items():
        face_id = int(face_id_str)
        if 0 <= class_id < num_classes:
            labels[face_id] = class_id
    return labels


def _find_label_file(labels_dir: str, model_id: str) -> Optional[Path]:
    """Find the GT label JSON file for a model, trying multiple naming conventions."""
    d = Path(labels_dir)
    # Try: model_id.json (direct match)
    candidate = d / f"{model_id}.json"
    if candidate.exists() and "_rel" not in candidate.name:
        return candidate
    # Try: model_id_result.json
    candidate = d / f"{model_id}_result.json"
    if candidate.exists() and "_rel" not in candidate.name:
        return candidate
    # Try: strip _result suffix and re-add it
    stripped = model_id.replace("_result", "")
    if stripped != model_id:
        candidate = d / f"{stripped}_result.json"
        if candidate.exists() and "_rel" not in candidate.name:
            return candidate
    return None


def _find_seg_file(seg_dir: str, model_id: str) -> Optional[Path]:
    """Find the predicted .seg file for a model, trying multiple naming conventions."""
    d = Path(seg_dir)
    # Try: model_id.seg (direct match)
    candidate = d / f"{model_id}.seg"
    if candidate.exists():
        return candidate
    # Try: model_id_result.seg
    candidate = d / f"{model_id}_result.seg"
    if candidate.exists():
        return candidate
    # Try: strip _result suffix
    stripped = model_id.replace("_result", "")
    if stripped != model_id:
        candidate = d / f"{stripped}_result.seg"
        if candidate.exists():
            return candidate
        candidate = d / f"{stripped}.seg"
        if candidate.exists():
            return candidate
    return None


def _find_defeatured_file(defeatured_dir: str, model_id: str) -> Optional[Path]:
    """Find the defeatured STEP file for a model."""
    d = Path(defeatured_dir)
    # Try: model_id_defeatured.step
    candidate = d / f"{model_id}_defeatured.step"
    if candidate.exists():
        return candidate
    # Try: model_id_defeatured.stp
    candidate = d / f"{model_id}_defeatured.stp"
    if candidate.exists():
        return candidate
    # Try: model_id.step (same name in defeatured dir)
    candidate = d / f"{model_id}.step"
    if candidate.exists():
        return candidate
    candidate = d / f"{model_id}.stp"
    if candidate.exists():
        return candidate
    return None


def _find_report_file(report_dir: str, model_id: str) -> Optional[Path]:
    """Find the report JSON for a model."""
    d = Path(report_dir)
    candidate = d / f"{model_id}_report.json"
    if candidate.exists():
        return candidate
    return None


def _load_report(report_path: str) -> Optional[Dict]:
    """Load a defeature report JSON."""
    try:
        with open(report_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class FaceItem:
    face: object
    ais: object


# ---------------------------------------------------------------------------
# Main viewer class
# ---------------------------------------------------------------------------

class DefeatureViewer(QtWidgets.QMainWindow):
    """Defeature comparison viewer: original (colored) vs defeatured (gray)."""

    MODE_GT = "gt"
    MODE_PRED = "pred"

    def __init__(
        self,
        step_path: Optional[str] = None,
        defeatured_path: Optional[str] = None,
        seg_path: Optional[str] = None,
        labels_json: Optional[str] = None,
        labels_dir: Optional[str] = None,
        checkpoint: Optional[str] = None,
        step_dir: Optional[str] = None,
        defeatured_dir: Optional[str] = None,
        seg_dir: Optional[str] = None,
        report_dir: Optional[str] = None,
    ):
        super().__init__()
        self.setWindowTitle("BrepFormer Defeature Viewer")
        self.resize(1800, 1000)
        self.setMinimumSize(1200, 800)

        # Class palette: always 5 defeature classes
        self._class_names = DEFEATURE_CLASS_NAMES
        self._class_colors_hex = DEFEATURE_CLASS_COLORS_HEX
        self._num_classes = DEFEATURE_NUM_CLASSES
        self._get_class_name = get_defeature_class_name

        self.class_colors_rgb = [hex_to_rgb01(c) for c in self._class_colors_hex]
        self.class_colors = [rgb01_to_quantity(c) for c in self.class_colors_rgb]
        self.unlabeled_color = rgb01_to_quantity(hex_to_rgb01(UNLABELED_COLOR_HEX))
        self.highlight_color = rgb01_to_quantity(hex_to_rgb01(HIGHLIGHT_COLOR_HEX))
        self.defeatured_color = rgb01_to_quantity(hex_to_rgb01(UNLABELED_COLOR_HEX))

        # State
        self.face_items: List[FaceItem] = []
        self.defeatured_face_items: List[FaceItem] = []
        self.gt_labels: Optional[List[int]] = None
        self.pred_labels: Optional[List[int]] = None
        self.current_index: Optional[int] = None
        self.highlight_enabled = True
        self.display_mode = self.MODE_GT
        self.step_path: Optional[str] = None
        self.defeatured_path: Optional[str] = None
        self.checkpoint = checkpoint
        self.labels_dir = labels_dir
        self.seg_dir = seg_dir
        self.defeatured_dir = defeatured_dir
        self.report_dir = report_dir
        self.current_report: Optional[Dict] = None

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
            if defeatured_path:
                self._load_defeatured_step(defeatured_path)
            if seg_path:
                self._load_pred_labels(seg_path)
            if labels_json:
                self._load_gt_from_json(labels_json)
            if labels_dir:
                self._auto_load_gt_labels()
            self._auto_load_pred_labels()
            self._auto_load_report()
            if self.pred_labels and not self.gt_labels:
                self.display_mode = self.MODE_PRED
            self._sync_mode_button()
            self._apply_colors()
        elif self.step_files:
            self._load_batch_item(0)

    # -------------------------------------------------------------------
    # UI construction
    # -------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        root_layout = QtWidgets.QHBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        # Viewer area with splitter for side-by-side
        self.viewer_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # Left viewer: original (colored)
        self.original_container = QtWidgets.QWidget()
        oc_layout = QtWidgets.QVBoxLayout(self.original_container)
        oc_layout.setContentsMargins(0, 0, 0, 0)
        oc_layout.setSpacing(2)
        self.original_header = QtWidgets.QLabel("Original")
        self.original_header.setAlignment(QtCore.Qt.AlignCenter)
        self.original_header.setStyleSheet(
            "font-weight: bold; font-size: 13px; padding: 2px; background: #333; color: #fff;"
        )
        oc_layout.addWidget(self.original_header)
        self.viewer = qtViewer3d(self.original_container)
        self.viewer.InitDriver()
        self.display = self.viewer._display
        try:
            self.display.Context.SetAutomaticHilight(False)
        except Exception:
            pass
        self._configure_viewer_visuals_for(self.display)
        oc_layout.addWidget(self.viewer)
        self.viewer_splitter.addWidget(self.original_container)

        # Right viewer: defeatured (gray)
        self.defeatured_container = QtWidgets.QWidget()
        dc_layout = QtWidgets.QVBoxLayout(self.defeatured_container)
        dc_layout.setContentsMargins(0, 0, 0, 0)
        dc_layout.setSpacing(2)
        self.defeatured_header = QtWidgets.QLabel("Defeatured")
        self.defeatured_header.setAlignment(QtCore.Qt.AlignCenter)
        self.defeatured_header.setStyleSheet(
            "font-weight: bold; font-size: 13px; padding: 2px; background: #333; color: #fff;"
        )
        dc_layout.addWidget(self.defeatured_header)
        self.viewer2 = qtViewer3d(self.defeatured_container)
        self.viewer2.InitDriver()
        self.display2 = self.viewer2._display
        try:
            self.display2.Context.SetAutomaticHilight(False)
        except Exception:
            pass
        self._configure_viewer_visuals_for(self.display2)
        dc_layout.addWidget(self.viewer2)
        self.viewer_splitter.addWidget(self.defeatured_container)

        # Give both viewers equal space
        self.viewer_splitter.setSizes([900, 900])

        root_layout.addWidget(self.viewer_splitter, 1)

        # Side panel
        panel = QtWidgets.QWidget(central)
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(6)
        root_layout.addWidget(panel, 0)

        # Buttons
        self.btn_import_step = QtWidgets.QPushButton("Import Original STEP")
        self.btn_import_step.clicked.connect(self.on_import_step)
        panel_layout.addWidget(self.btn_import_step)

        self.btn_import_defeatured = QtWidgets.QPushButton("Import Defeatured STEP")
        self.btn_import_defeatured.clicked.connect(self.on_import_defeatured)
        panel_layout.addWidget(self.btn_import_defeatured)

        self.btn_import_seg = QtWidgets.QPushButton("Import .seg")
        self.btn_import_seg.clicked.connect(self.on_import_seg)
        panel_layout.addWidget(self.btn_import_seg)

        if self.checkpoint:
            self.btn_infer = QtWidgets.QPushButton("Run Inference")
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

        # Face navigation (original faces)
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

        # Report stats label
        self.report_label = QtWidgets.QLabel("")
        self.report_label.setWordWrap(True)
        self.report_label.setStyleSheet("font-weight: bold; color: #333;")
        self.report_label.hide()
        panel_layout.addWidget(self.report_label)

        panel_layout.addSpacing(8)

        # Legend
        legend_label = QtWidgets.QLabel("5 Defeature Classes")
        legend_label.setStyleSheet("font-weight: bold;")
        panel_layout.addWidget(legend_label)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(300)
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

    # -------------------------------------------------------------------
    # Keyboard handling
    # -------------------------------------------------------------------

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QtWidgets.QApplication.processEvents()
        try:
            self.display.View.MustBeResized()
        except Exception:
            pass
        try:
            self.display2.View.MustBeResized()
        except Exception:
            pass

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

    # -------------------------------------------------------------------
    # Button handlers
    # -------------------------------------------------------------------

    def on_import_step(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Original STEP", "", "STEP Files (*.stp *.step)"
        )
        if path:
            self.load_step(path)
            self._auto_load_gt_labels()
            self._auto_load_pred_labels()
            self._auto_load_defeatured()
            self._auto_load_report()
            self._apply_colors()

    def on_import_defeatured(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Defeatured STEP", "", "STEP Files (*.stp *.step)"
        )
        if path:
            self._load_defeatured_step(path)

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

            model = load_model(self.checkpoint)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            result = infer_single(model, self.step_path, device)
            if "face_preds" in result:
                self.pred_labels = result["face_preds"]
                self.display_mode = self.MODE_PRED
                self._sync_mode_button()
                self._apply_colors()
                self._update_status()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Inference Error", str(e))

    def on_toggle_mode(self) -> None:
        if self.display_mode == self.MODE_GT:
            self.display_mode = self.MODE_PRED
        else:
            self.display_mode = self.MODE_GT
        self._sync_mode_button()
        self._apply_colors()

    def _sync_mode_button(self) -> None:
        self.btn_toggle_mode.setText(f"Display: {self.display_mode.upper()}")

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

        lines = [
            "5 Defeature Classes",
            f"Original faces: {len(self.face_items)}",
            f"Defeatured faces: {len(self.defeatured_face_items)}",
            f"Unlabeled: {unlabeled}",
            "",
        ]

        for name, count in sorted(counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {name}: {count}")

        if self.current_report:
            lines.append("")
            lines.append("--- Defeature Report ---")
            status = self.current_report.get("status", "N/A")
            lines.append(f"  Status: {status}")
            if "removed" in self.current_report:
                lines.append(f"  Removed: {self.current_report['removed']}")
            if "failed" in self.current_report:
                lines.append(f"  Failed: {self.current_report['failed']}")
            if "valid" in self.current_report:
                lines.append(f"  Valid: {self.current_report['valid']}")

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

    # -------------------------------------------------------------------
    # STEP loading
    # -------------------------------------------------------------------

    def load_step(self, path: str) -> None:
        """Load original STEP file into the left viewer."""
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
        self.current_report = None

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
            try:
                self.display.View.MustBeResized()
            except Exception:
                pass
            self.display.FitAll()
            self._set_current_face(0)
        self.display.Repaint()
        self._update_status()

    def _load_defeatured_step(self, path: str) -> None:
        """Load defeatured STEP file into the right viewer."""
        reader = STEPControl_Reader()
        status = reader.ReadFile(path)
        if status != IFSelect_RetDone:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to read defeatured: {path}")
            return
        reader.TransferRoots()
        shape = reader.OneShape()
        self.defeatured_path = path

        self.display2.EraseAll()
        self.defeatured_face_items.clear()

        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = topods.Face(explorer.Current())
            ais = self.display2.DisplayShape(face, update=False, color=self.defeatured_color)
            if isinstance(ais, list):
                ais = ais[0]
            self._apply_face_material_for(ais, self.display2)
            try:
                self.display2.Context.SetDisplayMode(ais, 1, False)
            except Exception:
                pass
            self.defeatured_face_items.append(FaceItem(face=face, ais=ais))
            explorer.Next()

        if self.defeatured_face_items:
            try:
                self.display2.View.MustBeResized()
            except Exception:
                pass
            self.display2.FitAll()
        self.display2.Repaint()
        self._update_status()

    # -------------------------------------------------------------------
    # Label loading
    # -------------------------------------------------------------------

    def _load_pred_labels(self, seg_path: str) -> None:
        self.pred_labels = load_seg_labels(seg_path)

    def _load_gt_from_json(self, json_path: str) -> None:
        self.gt_labels = load_labels_from_json(json_path, self._num_classes)

    def _auto_load_gt_labels(self) -> None:
        """Try to auto-detect GT labels from labels_dir."""
        if not self.labels_dir or not self.step_path:
            return
        model_id = Path(self.step_path).stem
        label_file = _find_label_file(self.labels_dir, model_id)
        if label_file is not None:
            self.gt_labels = load_labels_from_json(str(label_file), self._num_classes)

    def _auto_load_pred_labels(self) -> None:
        """Try to auto-detect predicted labels from seg_dir."""
        if not self.seg_dir or not self.step_path:
            return
        model_id = Path(self.step_path).stem
        seg_file = _find_seg_file(self.seg_dir, model_id)
        if seg_file is not None:
            self.pred_labels = load_seg_labels(str(seg_file))

    def _auto_load_defeatured(self) -> None:
        """Try to auto-detect defeatured STEP from defeatured_dir."""
        if not self.defeatured_dir or not self.step_path:
            return
        model_id = Path(self.step_path).stem
        defeatured_file = _find_defeatured_file(self.defeatured_dir, model_id)
        if defeatured_file is not None:
            self._load_defeatured_step(str(defeatured_file))

    def _auto_load_report(self) -> None:
        """Try to auto-detect report JSON from report_dir or defeatured_dir."""
        if not self.step_path:
            return
        model_id = Path(self.step_path).stem
        # Try report_dir first, then fall back to defeatured_dir
        search_dirs = []
        if self.report_dir:
            search_dirs.append(self.report_dir)
        if self.defeatured_dir and self.defeatured_dir != self.report_dir:
            search_dirs.append(self.defeatured_dir)
        for d in search_dirs:
            report_file = _find_report_file(d, model_id)
            if report_file is not None:
                self.current_report = _load_report(str(report_file))
                break
        self._update_report_label()

    def _update_report_label(self) -> None:
        """Update the report stats label."""
        if self.current_report:
            status = self.current_report.get("status", "N/A")
            parts = [f"Report: {status}"]
            if "removed" in self.current_report:
                parts.append(f"removed={self.current_report['removed']}")
            if "failed" in self.current_report:
                parts.append(f"failed={self.current_report['failed']}")
            if "valid" in self.current_report:
                parts.append(f"valid={self.current_report['valid']}")
            self.report_label.setText(" | ".join(parts))
            self.report_label.show()
        else:
            self.report_label.hide()

    # -------------------------------------------------------------------
    # Batch navigation
    # -------------------------------------------------------------------

    def _load_batch_item(self, index: int) -> None:
        self.batch_index = index
        step_path = self.step_files[index]
        self.load_step(step_path)
        self._auto_load_gt_labels()
        self._auto_load_pred_labels()
        self._auto_load_defeatured()
        self._auto_load_report()
        if self.pred_labels and not self.gt_labels:
            self.display_mode = self.MODE_PRED
        elif self.gt_labels:
            if self.display_mode == self.MODE_PRED and not self.pred_labels:
                self.display_mode = self.MODE_GT
        self._sync_mode_button()
        self._apply_colors()

    # -------------------------------------------------------------------
    # Core display logic
    # -------------------------------------------------------------------

    def _active_labels(self) -> List[Optional[int]]:
        """Get labels for current display mode."""
        n = len(self.face_items)
        if self.display_mode == self.MODE_GT and self.gt_labels:
            return [self.gt_labels[i] if i < len(self.gt_labels) else None for i in range(n)]
        elif self.display_mode == self.MODE_PRED and self.pred_labels:
            return [self.pred_labels[i] if i < len(self.pred_labels) else None for i in range(n)]
        else:
            return [None] * n

    def _apply_colors(self) -> None:
        """Apply colors to original (left) viewer based on labels. Defeatured (right) stays gray."""
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

        try:
            self.display.View.MustBeResized()
        except Exception:
            pass
        self.display.Repaint()

        # Defeatured viewer: uniform gray on all faces
        for idx in range(len(self.defeatured_face_items)):
            self._set_defeatured_face_color(idx, self.defeatured_color, repaint=False)
        try:
            self.display2.View.MustBeResized()
        except Exception:
            pass
        self.display2.Repaint()

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
        if index >= len(self.face_items):
            return
        ais = self.face_items[index].ais
        if isinstance(ais, list):
            for item in ais:
                self._set_ais_color(item, color)
        else:
            self._set_ais_color(ais, color)
        if repaint:
            self.display.Repaint()

    def _set_defeatured_face_color(self, index: int, color: Quantity_Color, repaint: bool = True) -> None:
        if index >= len(self.defeatured_face_items):
            return
        ais = self.defeatured_face_items[index].ais
        if isinstance(ais, list):
            for item in ais:
                self._set_ais_color_for(item, color, self.display2)
        else:
            self._set_ais_color_for(ais, color, self.display2)
        if repaint:
            self.display2.Repaint()

    def _set_ais_color(self, ais, color: Quantity_Color) -> None:
        try:
            ais.SetColor(color)
        except Exception:
            self.display.Context.SetColor(ais, color, False)
        self._apply_face_material(ais)
        self.display.Context.Redisplay(ais, False)

    def _set_ais_color_for(self, ais, color: Quantity_Color, display) -> None:
        """Set color on an AIS object belonging to a specific display context."""
        try:
            ais.SetColor(color)
        except Exception:
            try:
                display.Context.SetColor(ais, color, False)
            except Exception:
                pass
        self._apply_face_material_for(ais, display)
        try:
            display.Context.Redisplay(ais, False)
        except Exception:
            pass

    # -------------------------------------------------------------------
    # Material and edge styling
    # -------------------------------------------------------------------

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

    def _apply_face_material_for(self, ais, display) -> None:
        """Apply material to an AIS object in a specific display context."""
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

    def _configure_viewer_visuals_for(self, display) -> None:
        if Graphic3d_TOSM_UNLIT is None:
            return
        try:
            display.View.SetShadingModel(Graphic3d_TOSM_UNLIT)
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

    # -------------------------------------------------------------------
    # Status display
    # -------------------------------------------------------------------

    def _update_status(self) -> None:
        parts = []
        if self.step_path:
            parts.append(f"Original: {os.path.basename(self.step_path)}")
        if self.defeatured_path:
            parts.append(f"Defeatured: {os.path.basename(self.defeatured_path)}")
        if self.step_files:
            parts.append(f"Model {self.batch_index + 1}/{len(self.step_files)}")
        parts.append(f"Original faces: {len(self.face_items)}")
        parts.append(f"Defeatured faces: {len(self.defeatured_face_items)}")
        parts.append(f"Mode: {self.display_mode.upper()}")
        if self.gt_labels:
            parts.append("GT: loaded")
        if self.pred_labels:
            parts.append("Pred: loaded")
        self.status_label.setText("\n".join(parts))

        # Face info
        if self.current_index is not None:
            idx = self.current_index
            labels = self._active_labels()
            label = labels[idx]
            label_text = self._get_class_name(label) if label is not None and label >= 0 else "Unlabeled"
            # Show both GT and Pred if available
            gt = self.gt_labels[idx] if self.gt_labels and idx < len(self.gt_labels) else None
            pred = self.pred_labels[idx] if self.pred_labels and idx < len(self.pred_labels) else None
            info_parts = [f"Face {idx + 1}/{len(self.face_items)}"]
            info_parts.append(f"Class: {label} ({label_text})")
            if gt is not None and pred is not None:
                gt_text = self._get_class_name(gt) if gt >= 0 else "Unlabeled"
                pred_text = self._get_class_name(pred) if pred >= 0 else "Unlabeled"
                match = "MATCH" if gt == pred else "MISMATCH"
                info_parts.append(f"GT: {gt} ({gt_text})")
                info_parts.append(f"Pred: {pred} ({pred_text})")
                info_parts.append(f"[{match}]")
            self.info_label.setText("\n".join(info_parts))
        else:
            self.info_label.setText("")

        title = "BrepFormer Defeature Viewer"
        title += f" - {os.path.basename(self.step_path or '')}"
        self.setWindowTitle(title)

    def _print_face_info(self) -> None:
        if self.current_index is None:
            return
        idx = self.current_index
        gt = self.gt_labels[idx] if self.gt_labels and idx < len(self.gt_labels) else None
        pred = self.pred_labels[idx] if self.pred_labels and idx < len(self.pred_labels) else None
        print(f"Face {idx}: GT={gt} ({self._get_class_name(gt) if gt is not None else 'N/A'}), "
              f"Pred={pred} ({self._get_class_name(pred) if pred is not None else 'N/A'})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="BrepFormer defeature comparison viewer")
    parser.add_argument("--step", type=str, default=None,
                        help="Path to original STEP file")
    parser.add_argument("--step_dir", type=str, default=None,
                        help="Directory of original STEP files for batch browsing")
    parser.add_argument("--defeatured", type=str, default=None,
                        help="Path to defeatured STEP file")
    parser.add_argument("--defeatured_dir", type=str, default=None,
                        help="Directory of defeatured files (auto-matches {model_id}_defeatured.step)")
    parser.add_argument("--seg", type=str, default=None,
                        help="Path to .seg file for coloring original faces")
    parser.add_argument("--seg_dir", type=str, default=None,
                        help="Directory of .seg files (auto-match by model_id)")
    parser.add_argument("--labels_json", type=str, default=None,
                        help="Path to GT labels JSON file")
    parser.add_argument("--labels_dir", type=str, default=None,
                        help="Directory of GT label JSON files")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint for live inference on original")
    parser.add_argument("--report_dir", type=str, default=None,
                        help="Directory of {model_id}_report.json files (defaults to defeatured_dir)")
    args = parser.parse_args()

    # Default report_dir to defeatured_dir if not specified
    report_dir = args.report_dir
    if report_dir is None:
        report_dir = args.defeatured_dir

    app = QtWidgets.QApplication(sys.argv)
    window = DefeatureViewer(
        step_path=args.step,
        defeatured_path=args.defeatured,
        seg_path=args.seg,
        labels_json=args.labels_json,
        labels_dir=args.labels_dir,
        checkpoint=args.checkpoint,
        step_dir=args.step_dir,
        defeatured_dir=args.defeatured_dir,
        seg_dir=args.seg_dir,
        report_dir=report_dir,
    )
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
