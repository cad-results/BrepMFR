#!/usr/bin/env python3
"""BrepFormer face segmentation viewer.

Qt + pythonOCC 3D viewer for visualizing 27 MFTRCAD machining feature classes
on STEP models. Supports loading from .seg files, GT label JSON, or live
inference with a trained BrepFormer checkpoint.

Usage:
    # View with .seg labels
    python -m brepformer.visualize_seg --step model.step --seg preds.seg

    # View with GT labels from dataset
    python -m brepformer.visualize_seg --step model.step --labels_json labels/model_result.json

    # Run inference and display
    python -m brepformer.visualize_seg --step model.step --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt

    # Browse a directory
    python -m brepformer.visualize_seg --step_dir steps/ --labels_dir labels/

    # Browse a directory with inference and checkpoint
    python -m brepformer.visualize_seg --step_dir brepformer/data/mftrcad/steps --labels_dir brepformer/data/mftrcad/labels --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt
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
    CLASS_NAMES, CLASS_COLORS_HEX, NUM_CLASSES,
    REAL_CLASS_NAMES, REAL_CLASS_COLORS_HEX, REAL_NUM_CLASSES,
    CLASS_TO_REAL_CLASS, map_labels_to_real,
    DEFEATURE_CLASS_NAMES, DEFEATURE_CLASS_COLORS_HEX, DEFEATURE_NUM_CLASSES,
    get_defeature_class_name,
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


def load_labels_from_json(json_path: str, num_classes: int = NUM_CLASSES) -> Optional[List[int]]:
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
    # Try: model_id.json (direct match, e.g. 20240116_231044_0_result.json)
    candidate = d / f"{model_id}.json"
    if candidate.exists() and "_rel" not in candidate.name:
        return candidate
    # Try: model_id_result.json (e.g. if model_id has no _result suffix)
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


@dataclass
class FaceItem:
    face: object
    ais: object


class SegViewer(QtWidgets.QMainWindow):
    """Face segmentation viewer (supports 27-class and 8 real-class modes)."""

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
    ):
        super().__init__()
        self.real_classes = real_classes

        # Auto-detect defeature mode from paths
        self.defeature_mode = False
        for p in (step_dir, labels_dir):
            if p and "defeature" in str(p).lower():
                self.defeature_mode = True
                break

        # Select class palette based on mode
        if self.defeature_mode:
            self._class_names = DEFEATURE_CLASS_NAMES
            self._class_colors_hex = DEFEATURE_CLASS_COLORS_HEX
            self._num_classes = DEFEATURE_NUM_CLASSES
            self._get_class_name = get_defeature_class_name
        elif self.real_classes:
            self._class_names = REAL_CLASS_NAMES
            self._class_colors_hex = REAL_CLASS_COLORS_HEX
            self._num_classes = REAL_NUM_CLASSES
            self._get_class_name = get_real_class_name
        else:
            self._class_names = CLASS_NAMES
            self._class_colors_hex = CLASS_COLORS_HEX
            self._num_classes = NUM_CLASSES
            self._get_class_name = get_class_name

        title = "BrepFormer Seg Viewer"
        if self.defeature_mode:
            title += " [Defeature 5-class]"
        elif self.real_classes:
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
        self.defeature_enabled = False
        self.defeature_output_dir = "brepformer/defeatured_output"
        self.labels_dir = labels_dir
        self.seg_dir = seg_dir

        # Compare mode: second viewer state
        self.compare_face_items: List[FaceItem] = []
        self._compare_viewer_initialized = False

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
            self._sync_mode_button()
            self._apply_colors()
        elif self.step_files:
            self._load_batch_item(0)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        root_layout = QtWidgets.QHBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        # Viewer area with splitter for compare mode
        self.viewer_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # Primary viewer container
        self.primary_container = QtWidgets.QWidget()
        pc_layout = QtWidgets.QVBoxLayout(self.primary_container)
        pc_layout.setContentsMargins(0, 0, 0, 0)
        pc_layout.setSpacing(2)
        self.primary_header = QtWidgets.QLabel("")
        self.primary_header.setAlignment(QtCore.Qt.AlignCenter)
        self.primary_header.setStyleSheet("font-weight: bold; font-size: 13px; padding: 2px; background: #333; color: #fff;")
        self.primary_header.hide()
        pc_layout.addWidget(self.primary_header)
        self.viewer = qtViewer3d(self.primary_container)
        self.viewer.InitDriver()
        self.display = self.viewer._display
        try:
            self.display.Context.SetAutomaticHilight(False)
        except Exception:
            pass
        self._configure_viewer_visuals()
        pc_layout.addWidget(self.viewer)
        self.viewer_splitter.addWidget(self.primary_container)

        # Compare viewer container (hidden by default)
        self.compare_container = QtWidgets.QWidget()
        cc_layout = QtWidgets.QVBoxLayout(self.compare_container)
        cc_layout.setContentsMargins(0, 0, 0, 0)
        cc_layout.setSpacing(2)
        self.compare_header = QtWidgets.QLabel("Predicted")
        self.compare_header.setAlignment(QtCore.Qt.AlignCenter)
        self.compare_header.setStyleSheet("font-weight: bold; font-size: 13px; padding: 2px; background: #333; color: #fff;")
        cc_layout.addWidget(self.compare_header)
        self.viewer2 = qtViewer3d(self.compare_container)
        self.viewer2.InitDriver()
        self.display2 = self.viewer2._display
        try:
            self.display2.Context.SetAutomaticHilight(False)
        except Exception:
            pass
        self._configure_viewer_visuals_for(self.display2)
        cc_layout.addWidget(self.viewer2)
        self.compare_container.hide()
        self.viewer_splitter.addWidget(self.compare_container)

        root_layout.addWidget(self.viewer_splitter, 1)

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

        # Compare accuracy label (shown in compare mode)
        self.compare_stats_label = QtWidgets.QLabel("")
        self.compare_stats_label.setWordWrap(True)
        self.compare_stats_label.setStyleSheet("font-weight: bold; color: #333;")
        self.compare_stats_label.hide()
        panel_layout.addWidget(self.compare_stats_label)

        panel_layout.addSpacing(8)

        # Legend
        if self.defeature_mode:
            legend_title = "5 Defeature Classes"
        elif self.real_classes:
            legend_title = "8 Real Classes"
        else:
            legend_title = "27 MFTRCAD Classes"
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
        elif key == QtCore.Qt.Key_F:
            self.on_defeature()
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

    def on_defeature(self) -> None:
        """Defeature the current model (F key) — removes predicted features."""
        if not getattr(self, "defeature_enabled", False):
            print("Defeaturing disabled. Relaunch with --defeature to enable.")
            return
        if not self.step_path:
            return
        labels = self.pred_labels
        if labels is None:
            QtWidgets.QMessageBox.warning(
                self, "Defeature", "No predictions available. Load a .seg "
                "file or run inference first.")
            return
        try:
            from brepformer.defeature import defeature_step
            from brepformer.data.classes import (
                CLASS_TO_DEFEATURE, NUM_CLASSES,
                DEFEATURE_NUM_CLASSES,
            )

            # Remap to 5 defeature classes if needed
            if self._num_classes > DEFEATURE_NUM_CLASSES:
                defeature_labels = [
                    CLASS_TO_DEFEATURE[l] if 0 <= l < NUM_CLASSES else 0
                    for l in labels
                ]
            else:
                defeature_labels = list(labels)

            out_dir = Path(getattr(self, "defeature_output_dir",
                                   "brepformer/defeatured_output"))
            out_dir.mkdir(parents=True, exist_ok=True)
            model_id = Path(self.step_path).stem
            out_path = out_dir / f"{model_id}_defeatured.step"

            print(f"Defeaturing {model_id}...")
            result = defeature_step(
                self.step_path, defeature_labels, str(out_path), verbose=True,
            )
            if result["status"] == "success":
                msg = (f"Defeatured: removed {result['removed']} faces "
                       f"(failed: {result['failed']}, valid: {result['valid']})\n"
                       f"Saved to {result['output']}")
            elif result["status"] == "no_features":
                msg = result["message"]
            else:
                msg = f"Error: {result.get('message', 'unknown')}"
            print(msg)
            QtWidgets.QMessageBox.information(self, "Defeature Result", msg)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Defeature Error", str(e))

    def on_toggle_mode(self) -> None:
        modes = [self.MODE_GT, self.MODE_PRED, self.MODE_COMPARE]
        idx = modes.index(self.display_mode)
        self.display_mode = modes[(idx + 1) % len(modes)]
        self._sync_mode_button()
        if self.display_mode == self.MODE_COMPARE:
            self._enter_compare_mode()
        else:
            self._exit_compare_mode()
        self._apply_colors()

    def _sync_mode_button(self) -> None:
        self.btn_toggle_mode.setText(f"Display: {self.display_mode.upper()}")

    def on_summary(self) -> None:
        if not self.face_items:
            return

        if self.display_mode == self.MODE_COMPARE:
            self._show_compare_summary()
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

        if self.defeature_mode:
            mode_label = "5 Defeature Classes"
        elif self.real_classes:
            mode_label = "8 Real Classes"
        else:
            mode_label = "27 MFTRCAD Classes"
        lines = [mode_label, f"Total faces: {len(labels)}", f"Unlabeled: {unlabeled}", ""]
        for name, count in sorted(counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {name}: {count}")

        QtWidgets.QMessageBox.information(self, "Summary", "\n".join(lines))

    def _show_compare_summary(self) -> None:
        n = len(self.face_items)
        if not self.gt_labels or not self.pred_labels:
            QtWidgets.QMessageBox.information(
                self, "Compare Summary", "Need both GT and Pred labels for comparison."
            )
            return

        correct = 0
        total = 0
        per_class_correct = {}
        per_class_total = {}
        for i in range(n):
            gt = self.gt_labels[i] if i < len(self.gt_labels) else -1
            pred = self.pred_labels[i] if i < len(self.pred_labels) else -1
            if gt < 0:
                continue
            total += 1
            name = self._get_class_name(gt)
            per_class_total[name] = per_class_total.get(name, 0) + 1
            if gt == pred:
                correct += 1
                per_class_correct[name] = per_class_correct.get(name, 0) + 1

        acc = correct / total * 100 if total > 0 else 0
        lines = [
            f"Face Accuracy: {correct}/{total} ({acc:.1f}%)",
            "",
            "Per-class accuracy:",
        ]
        for name in sorted(per_class_total.keys()):
            c = per_class_correct.get(name, 0)
            t = per_class_total[name]
            lines.append(f"  {name}: {c}/{t} ({c/t*100:.0f}%)")

        QtWidgets.QMessageBox.information(self, "Compare Summary", "\n".join(lines))

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

    def _load_batch_item(self, index: int) -> None:
        self.batch_index = index
        step_path = self.step_files[index]
        self.load_step(step_path)
        self._auto_load_gt_labels()
        self._auto_load_pred_labels()
        if self.pred_labels and not self.gt_labels:
            self.display_mode = self.MODE_PRED
        elif self.gt_labels:
            if self.display_mode == self.MODE_PRED and not self.pred_labels:
                self.display_mode = self.MODE_GT
        self._sync_mode_button()
        if self.display_mode == self.MODE_COMPARE:
            self._load_compare_step()
        self._apply_colors()

    # ---------------------------------------------------------------
    # Compare mode: dual viewer
    # ---------------------------------------------------------------

    def _enter_compare_mode(self) -> None:
        """Show the second viewer and load the current model into it."""
        self.primary_header.setText("Ground Truth")
        self.primary_header.show()
        self.compare_container.show()
        # Ensure the splitter gives both viewers equal space and display2
        # is fully realized before loading shapes into it.
        w = self.viewer_splitter.width()
        self.viewer_splitter.setSizes([w // 2, w // 2])
        QtWidgets.QApplication.processEvents()
        try:
            self.display2.View.MustBeResized()
        except Exception:
            pass
        self._load_compare_step()

    def _exit_compare_mode(self) -> None:
        """Hide the second viewer."""
        self.primary_header.hide()
        self.compare_container.hide()
        self.compare_stats_label.hide()
        # Clear compare viewer
        self.display2.EraseAll()
        self.compare_face_items.clear()
        self.display2.Repaint()

    def _load_compare_step(self) -> None:
        """Load the current STEP file into the compare viewer."""
        if not self.step_path:
            return
        reader = STEPControl_Reader()
        status = reader.ReadFile(self.step_path)
        if status != IFSelect_RetDone:
            return
        reader.TransferRoots()
        shape = reader.OneShape()

        self.display2.EraseAll()
        self.compare_face_items.clear()

        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = topods.Face(explorer.Current())
            ais = self.display2.DisplayShape(face, update=False, color=self.unlabeled_color)
            if isinstance(ais, list):
                ais = ais[0]
            self._apply_face_material_for(ais, self.display2)
            try:
                self.display2.Context.SetDisplayMode(ais, 1, False)
            except Exception:
                pass
            self.compare_face_items.append(FaceItem(face=face, ais=ais))
            explorer.Next()

        if self.compare_face_items:
            try:
                self.display2.View.MustBeResized()
            except Exception:
                pass
            self.display2.FitAll()
        self.display2.Repaint()

    def _apply_compare_colors(self) -> None:
        """Color primary viewer by GT and compare viewer by Pred."""
        n = len(self.face_items)
        n2 = len(self.compare_face_items)

        # Primary viewer: GT labels
        for idx in range(n):
            gt = self.gt_labels[idx] if self.gt_labels and idx < len(self.gt_labels) else None
            if gt is not None and 0 <= gt < self._num_classes:
                color = self.class_colors[gt]
            else:
                color = self.unlabeled_color
            self._set_face_color(idx, color, repaint=False)
        self.display.Repaint()

        # Compare viewer: Pred labels
        for idx in range(n2):
            pred = self.pred_labels[idx] if self.pred_labels and idx < len(self.pred_labels) else None
            if pred is not None and 0 <= pred < self._num_classes:
                color = self.class_colors[pred]
            else:
                color = self.unlabeled_color
            try:
                self._set_compare_face_color(idx, color, repaint=False)
            except Exception:
                pass
        self.display2.Repaint()
        # Force viewer2 to update after all colors are set
        try:
            self.display2.View.MustBeResized()
            self.display2.Repaint()
        except Exception:
            pass

        # Update compare stats
        self._update_compare_stats()

    def _set_compare_face_color(self, index: int, color: Quantity_Color, repaint: bool = True) -> None:
        if index >= len(self.compare_face_items):
            return
        ais = self.compare_face_items[index].ais
        if isinstance(ais, list):
            for item in ais:
                self._set_ais_color_for(item, color, self.display2)
        else:
            self._set_ais_color_for(ais, color, self.display2)
        if repaint:
            self.display2.Repaint()

    def _update_compare_stats(self) -> None:
        """Compute and display match/mismatch statistics."""
        n = len(self.face_items)
        if not self.gt_labels or not self.pred_labels:
            self.compare_stats_label.setText("Missing GT or Pred labels")
            self.compare_stats_label.show()
            return

        correct = 0
        total = 0
        for i in range(n):
            gt = self.gt_labels[i] if i < len(self.gt_labels) else -1
            pred = self.pred_labels[i] if i < len(self.pred_labels) else -1
            if gt < 0:
                continue
            total += 1
            if gt == pred:
                correct += 1

        acc = correct / total * 100 if total > 0 else 0
        self.compare_stats_label.setText(
            f"Match: {correct}/{total} ({acc:.1f}%)"
        )
        self.compare_stats_label.show()

    # ---------------------------------------------------------------
    # Core display logic
    # ---------------------------------------------------------------

    def _active_labels(self) -> List[Optional[int]]:
        """Get labels for current display mode (remapped to real classes if enabled)."""
        n = len(self.face_items)
        if self.display_mode == self.MODE_GT and self.gt_labels:
            raw = [self.gt_labels[i] if i < len(self.gt_labels) else None for i in range(n)]
        elif self.display_mode == self.MODE_PRED and self.pred_labels:
            raw = [self.pred_labels[i] if i < len(self.pred_labels) else None for i in range(n)]
        elif self.display_mode == self.MODE_COMPARE:
            # Primary viewer shows GT in compare mode
            if self.gt_labels:
                raw = [self.gt_labels[i] if i < len(self.gt_labels) else None for i in range(n)]
            else:
                raw = [None] * n
        else:
            raw = [None] * n

        if self.real_classes:
            return [CLASS_TO_REAL_CLASS[l] if l is not None and 0 <= l < NUM_CLASSES else l for l in raw]
        return raw

    def _apply_colors(self) -> None:
        if self.display_mode == self.MODE_COMPARE:
            self._apply_compare_colors()
            # Apply highlight on primary viewer
            if self.current_index is not None and self.highlight_enabled:
                self._set_face_color(self.current_index, self.highlight_color, repaint=False)
                self._set_compare_face_color(self.current_index, self.highlight_color, repaint=False)
                self.display.Repaint()
                self.display2.Repaint()
            self._update_status()
            return

        self.compare_stats_label.hide()
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

            # Also restore compare viewer face
            if self.display_mode == self.MODE_COMPARE and self.compare_face_items:
                pred = self.pred_labels[self.current_index] if self.pred_labels and self.current_index < len(self.pred_labels) else None
                if pred is not None and 0 <= pred < self._num_classes:
                    ccolor = self.class_colors[pred]
                else:
                    ccolor = self.unlabeled_color
                self._set_compare_face_color(self.current_index, ccolor)

        self.current_index = index
        if self.highlight_enabled:
            self._set_face_color(index, self.highlight_color)
            if self.display_mode == self.MODE_COMPARE and self.compare_face_items:
                self._set_compare_face_color(index, self.highlight_color)

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

    def _configure_viewer_visuals(self) -> None:
        self._configure_viewer_visuals_for(self.display)

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

    def _update_status(self) -> None:
        parts = []
        if self.step_path:
            parts.append(f"STEP: {os.path.basename(self.step_path)}")
        if self.step_files:
            parts.append(f"Model {self.batch_index + 1}/{len(self.step_files)}")
        parts.append(f"Faces: {len(self.face_items)}")
        parts.append(f"Mode: {self.display_mode.upper()}")
        if self.defeature_mode:
            parts.append("Classes: 5 defeature")
        elif self.real_classes:
            parts.append("Classes: 8 real")
        if self.gt_labels:
            parts.append("GT: loaded")
        if self.pred_labels:
            parts.append("Pred: loaded")
        self.status_label.setText("\n".join(parts))

        # Face info - show both GT and Pred in compare mode
        if self.current_index is not None:
            idx = self.current_index
            if self.display_mode == self.MODE_COMPARE:
                gt = self.gt_labels[idx] if self.gt_labels and idx < len(self.gt_labels) else None
                pred = self.pred_labels[idx] if self.pred_labels and idx < len(self.pred_labels) else None
                gt_text = self._get_class_name(gt) if gt is not None and gt >= 0 else "Unlabeled"
                pred_text = self._get_class_name(pred) if pred is not None and pred >= 0 else "Unlabeled"
                match = "MATCH" if gt == pred else "MISMATCH"
                self.info_label.setText(
                    f"Face {idx + 1}/{len(self.face_items)}\n"
                    f"GT: {gt} ({gt_text})\n"
                    f"Pred: {pred} ({pred_text})\n"
                    f"[{match}]"
                )
            else:
                labels = self._active_labels()
                label = labels[idx]
                label_text = self._get_class_name(label) if label is not None and label >= 0 else "Unlabeled"
                self.info_label.setText(
                    f"Face {idx + 1}/{len(self.face_items)}\n"
                    f"Class: {label} ({label_text})"
                )
        else:
            self.info_label.setText("")

        title = "BrepFormer Seg Viewer"
        if self.defeature_mode:
            title += " [Defeature]"
        elif self.real_classes:
            title += " [Real Classes]"
        title += f" - {os.path.basename(self.step_path or '')}"
        self.setWindowTitle(title)

    def _print_face_info(self) -> None:
        if self.current_index is None:
            return
        idx = self.current_index
        gt = self.gt_labels[idx] if self.gt_labels and idx < len(self.gt_labels) else None
        pred = self.pred_labels[idx] if self.pred_labels and idx < len(self.pred_labels) else None
        if self.real_classes and not self.defeature_mode:
            gt_r = CLASS_TO_REAL_CLASS[gt] if gt is not None and 0 <= gt < NUM_CLASSES else gt
            pred_r = CLASS_TO_REAL_CLASS[pred] if pred is not None and 0 <= pred < NUM_CLASSES else pred
            print(f"Face {idx}: GT={gt_r} ({get_real_class_name(gt_r) if gt_r is not None else 'N/A'}), "
                  f"Pred={pred_r} ({get_real_class_name(pred_r) if pred_r is not None else 'N/A'})")
        else:
            print(f"Face {idx}: GT={gt} ({self._get_class_name(gt) if gt is not None else 'N/A'}), "
                  f"Pred={pred} ({self._get_class_name(pred) if pred is not None else 'N/A'})")


def main() -> int:
    parser = argparse.ArgumentParser(description="BrepFormer face segmentation viewer")
    parser.add_argument("--step", type=str, default=None, help="Path to STEP file")
    parser.add_argument("--seg", type=str, default=None, help="Path to .seg label file")
    parser.add_argument("--labels_json", type=str, default=None,
                        help="Path to GT labels JSON file")
    parser.add_argument("--labels_dir", type=str, default=None,
                        help="Directory of label JSON files")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint for inference")
    parser.add_argument("--step_dir", type=str, default=None,
                        help="Directory of STEP files for batch browsing")
    parser.add_argument("--seg_dir", type=str, default=None,
                        help="Directory of .seg prediction files (auto-matches by model_id)")
    parser.add_argument("--real_classes", action="store_true",
                        help="Remap 27 MFTRCAD classes to 8 real machining feature categories")
    parser.add_argument("--defeature", action="store_true",
                        help="Enable defeaturing: press F to remove predicted features and save result")
    parser.add_argument("--defeature_output_dir", type=str, default="brepformer/defeatured_output",
                        help="Output directory for defeatured STEP files (default: brepformer/defeatured_output)")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    window = SegViewer(
        step_path=args.step,
        seg_path=args.seg,
        labels_json=args.labels_json,
        labels_dir=args.labels_dir,
        checkpoint=args.checkpoint,
        step_dir=args.step_dir,
        seg_dir=args.seg_dir,
        real_classes=args.real_classes,
    )
    window.defeature_enabled = args.defeature
    window.defeature_output_dir = args.defeature_output_dir
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
