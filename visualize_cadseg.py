#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize .seg face labels on a STEP model.

Face order follows the TopExp_Explorer (topological walk), matching the
ordering used in face_labeler.py and BRepNet feature extraction.
"""
from __future__ import annotations

import sys
import os
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Qt binding selection + pythonocc backend init
try:
    from PyQt5 import QtCore, QtWidgets
    _qt_backend = "qt-pyqt5"
except ImportError:  # pragma: no cover
    from PySide2 import QtCore, QtWidgets
    _qt_backend = "qt-pyside2"

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
except Exception:  # pragma: no cover - optional OCC build
    Aspect_TOL_SOLID = None
    Prs3d_LineAspect = None
try:
    from OCC.Core.Graphic3d import Graphic3d_NOM_MATTE, Graphic3d_NOM_NEON
except Exception:  # pragma: no cover - optional OCC build
    Graphic3d_NOM_MATTE = None
    Graphic3d_NOM_NEON = None
try:
    from OCC.Core.Graphic3d import Graphic3d_TOSM_UNLIT
except Exception:  # pragma: no cover - optional OCC build
    Graphic3d_TOSM_UNLIT = None

CLASS_NAMES = [
    "SOL",
    "Through hole",
    "blind hole",
    "chamfer",
    "Fillet",
    "Through cut",
    "Blind cut",
]

CLASS_COLORS_HEX = [
    "#1f77b4",  # blue
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#ff7f0e",  # orange
]

UNLABELED_COLOR_HEX = "#d0d0d0"
HIGHLIGHT_COLOR_HEX = "#FFD400"
EDGE_COLOR_HEX = "#2b2b2b"


def hex_to_rgb01(color_hex: str) -> Tuple[float, float, float]:
    color_hex = color_hex.lstrip("#")
    r = int(color_hex[0:2], 16) / 255.0
    g = int(color_hex[2:4], 16) / 255.0
    b = int(color_hex[4:6], 16) / 255.0
    return r, g, b


def rgb01_to_quantity(rgb: Tuple[float, float, float]) -> Quantity_Color:
    return Quantity_Color(rgb[0], rgb[1], rgb[2], Quantity_TOC_RGB)


def text_color_for_bg(color_hex: str) -> str:
    r, g, b = hex_to_rgb01(color_hex)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b)
    return "#000000" if luminance > 0.6 else "#ffffff"


def load_seg_labels(seg_path: str) -> List[int]:
    labels: List[int] = []
    with open(seg_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line == "":
                continue
            labels.append(int(line))
    return labels


@dataclass
class FaceItem:
    face: object
    ais: object


class SegViewer(QtWidgets.QMainWindow):
    def __init__(self, step_path: Optional[str] = None, seg_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("BRepMFR Seg Viewer")
        self.resize(1600, 1000)
        self.setMinimumSize(1200, 800)

        self.class_names = CLASS_NAMES
        self.class_colors_rgb = [hex_to_rgb01(c) for c in CLASS_COLORS_HEX]
        self.class_colors = [rgb01_to_quantity(c) for c in self.class_colors_rgb]
        self.unlabeled_color = rgb01_to_quantity(hex_to_rgb01(UNLABELED_COLOR_HEX))
        self.highlight_color = rgb01_to_quantity(hex_to_rgb01(HIGHLIGHT_COLOR_HEX))

        self.face_items: List[FaceItem] = []
        self.labels: List[Optional[int]] = []
        self.seg_labels: Optional[List[int]] = None
        self.current_index: Optional[int] = None
        self.highlight_enabled = True
        self.step_path: Optional[str] = None
        self.seg_path: Optional[str] = None

        self._build_ui()
        self.update_step_label()
        self.update_seg_label()
        self.update_info()

        if step_path:
            self.load_step(step_path)
        if seg_path:
            self.load_seg(seg_path)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        root_layout = QtWidgets.QHBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        self.viewer = qtViewer3d(central)
        self.viewer.InitDriver()
        self.display = self.viewer._display
        try:
            self.display.Context.SetAutomaticHilight(False)
        except Exception:
            pass
        self._configure_viewer_visuals()
        root_layout.addWidget(self.viewer, 1)

        panel = QtWidgets.QWidget(central)
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(6)
        root_layout.addWidget(panel, 0)

        self.btn_import_step = QtWidgets.QPushButton("Import STEP")
        self.btn_import_step.clicked.connect(self.on_import_step)
        panel_layout.addWidget(self.btn_import_step)

        self.btn_import_seg = QtWidgets.QPushButton("Import .seg")
        self.btn_import_seg.clicked.connect(self.on_import_seg)
        panel_layout.addWidget(self.btn_import_seg)

        self.btn_clear_all = QtWidgets.QPushButton("Clear All")
        self.btn_clear_all.clicked.connect(self.on_clear_all)
        panel_layout.addWidget(self.btn_clear_all)

        self.chk_highlight = QtWidgets.QCheckBox("Highlight current")
        self.chk_highlight.setChecked(True)
        self.chk_highlight.toggled.connect(self.on_toggle_highlight)
        panel_layout.addWidget(self.chk_highlight)

        self.btn_review = QtWidgets.QPushButton("Summary")
        self.btn_review.clicked.connect(self.on_review)
        panel_layout.addWidget(self.btn_review)

        panel_layout.addSpacing(8)

        nav_layout = QtWidgets.QHBoxLayout()
        self.btn_prev = QtWidgets.QPushButton("<< Prev")
        self.btn_prev.clicked.connect(self.on_prev)
        self.btn_next = QtWidgets.QPushButton("Next >>")
        self.btn_next.clicked.connect(self.on_next)
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        panel_layout.addLayout(nav_layout)

        self.step_label = QtWidgets.QLabel("STEP: (none)")
        self.step_label.setWordWrap(True)
        panel_layout.addWidget(self.step_label)

        self.seg_label = QtWidgets.QLabel("SEG: (none)")
        self.seg_label.setWordWrap(True)
        panel_layout.addWidget(self.seg_label)

        self.info_label = QtWidgets.QLabel("No STEP loaded")
        self.info_label.setWordWrap(True)
        panel_layout.addWidget(self.info_label)

        panel_layout.addSpacing(8)

        legend_label = QtWidgets.QLabel("Label Colors")
        legend_label.setStyleSheet("font-weight: bold;")
        panel_layout.addWidget(legend_label)

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(6)
        for idx, name in enumerate(self.class_names):
            item = QtWidgets.QLabel(f"{idx}: {name}")
            bg = CLASS_COLORS_HEX[idx]
            fg = text_color_for_bg(bg)
            item.setStyleSheet(
                f"background-color: {bg}; color: {fg}; padding: 3px 6px; border-radius: 2px;"
            )
            grid.addWidget(item, idx, 0)
        panel_layout.addLayout(grid)

        panel_layout.addStretch(1)
        self.setCentralWidget(central)

    def update_step_label(self) -> None:
        if self.step_path:
            name = os.path.basename(self.step_path)
            self.step_label.setText(f"STEP: {name}")
            self.setWindowTitle(f"BRepMFR Seg Viewer - {name}")
        else:
            self.step_label.setText("STEP: (none)")
            self.setWindowTitle("BRepMFR Seg Viewer")

    def update_seg_label(self) -> None:
        if self.seg_path:
            name = os.path.basename(self.seg_path)
            self.seg_label.setText(f"SEG: {name}")
        else:
            self.seg_label.setText("SEG: (none)")

    def keyPressEvent(self, event) -> None:  # pragma: no cover - UI only
        if event.key() in (QtCore.Qt.Key_Right, QtCore.Qt.Key_D):
            self.on_next()
            return
        if event.key() in (QtCore.Qt.Key_Left, QtCore.Qt.Key_A):
            self.on_prev()
            return
        super().keyPressEvent(event)

    def on_import_step(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open STEP", "", "STEP Files (*.stp *.step)"
        )
        if path:
            self.load_step(path)

    def on_import_seg(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open SEG", "", "SEG Files (*.seg)"
        )
        if path:
            self.load_seg(path)

    def on_clear_all(self) -> None:
        if not self.step_path and not self.seg_path and not self.face_items and self.seg_labels is None:
            return
        try:
            self.display.EraseAll()
            self.display.Repaint()
        except Exception:
            pass
        self.face_items.clear()
        self.labels.clear()
        self.seg_labels = None
        self.current_index = None
        self.step_path = None
        self.seg_path = None
        self.update_step_label()
        self.update_seg_label()
        self.update_info()

    def on_toggle_highlight(self, checked: bool) -> None:
        self.highlight_enabled = checked
        if self.current_index is None:
            return
        color = self.get_highlight_color(self.current_index) if checked else self.get_base_color(self.current_index)
        self.set_face_color(self.current_index, color)

    def on_review(self) -> None:
        if not self.face_items:
            QtWidgets.QMessageBox.information(self, "Summary", "No STEP loaded.")
            return
        counts = [0 for _ in self.class_names]
        unlabeled = 0
        for label in self.labels:
            if label is None:
                unlabeled += 1
            else:
                counts[label] += 1
        lines = [
            f"Total faces: {len(self.labels)}",
            f"Unlabeled: {unlabeled}",
            "",
        ]
        for idx, name in enumerate(self.class_names):
            lines.append(f"{idx} {name}: {counts[idx]}")
        QtWidgets.QMessageBox.information(self, "Summary", "\n".join(lines))

    def on_prev(self) -> None:
        if self.current_index is None:
            return
        if self.current_index <= 0:
            return
        self.set_current_index(self.current_index - 1)

    def on_next(self) -> None:
        if self.current_index is None:
            return
        if self.current_index >= len(self.face_items) - 1:
            return
        self.set_current_index(self.current_index + 1)

    def load_step(self, path: str) -> None:
        reader = STEPControl_Reader()
        status = reader.ReadFile(path)
        if status != IFSelect_RetDone:
            QtWidgets.QMessageBox.warning(
                self, "Load STEP", f"Failed to read STEP file: {path}"
            )
            return
        reader.TransferRoots()
        shape = reader.OneShape()
        self.step_path = path
        self.update_step_label()

        self.display.EraseAll()
        self.face_items.clear()
        self.labels.clear()
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
            try:
                self.display.Context.Redisplay(ais, False)
            except Exception:
                pass
            self.face_items.append(FaceItem(face=face, ais=ais))
            self.labels.append(None)
            explorer.Next()

        if not self.face_items:
            QtWidgets.QMessageBox.warning(
                self, "Load STEP", "No faces found in STEP file."
            )
            self.display.Repaint()
            return

        if self.seg_labels is not None:
            self.apply_seg_labels()

        self.display.FitAll()
        self.set_current_index(0)
        self.display.Repaint()

    def load_seg(self, path: str) -> None:
        try:
            labels = load_seg_labels(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load SEG", f"Failed to read .seg: {exc}")
            return
        self.seg_path = path
        self.seg_labels = labels
        self.update_seg_label()
        if self.face_items:
            self.apply_seg_labels()

    def apply_seg_labels(self) -> None:
        if not self.face_items:
            return
        num_faces = len(self.face_items)
        labels: List[Optional[int]] = [None for _ in range(num_faces)]
        invalid = 0
        for idx in range(num_faces):
            if self.seg_labels is None or idx >= len(self.seg_labels):
                continue
            label = self.seg_labels[idx]
            if 0 <= label < len(self.class_names):
                labels[idx] = label
            else:
                invalid += 1
        extra = 0 if self.seg_labels is None else max(0, len(self.seg_labels) - num_faces)
        missing = 0 if self.seg_labels is None else max(0, num_faces - len(self.seg_labels))
        self.labels = labels
        self.recolor_all_faces()
        if invalid or extra or missing:
            QtWidgets.QMessageBox.warning(
                self,
                "SEG Mismatch",
                "Loaded labels with mismatches:\n"
                f"- Missing labels: {missing}\n"
                f"- Extra labels: {extra}\n"
                f"- Invalid labels: {invalid}\n\n"
                "Missing/invalid labels are shown as unlabeled (gray).",
            )

    def update_info(self) -> None:
        if self.current_index is None:
            self.info_label.setText("No STEP loaded")
            return
        label = self.labels[self.current_index]
        label_text = "Unlabeled" if label is None else f"{label}: {self.class_names[label]}"
        self.info_label.setText(
            f"Face {self.current_index + 1}/{len(self.face_items)}\n"
            f"Label: {label_text}"
        )

    def get_base_color(self, index: int) -> Quantity_Color:
        label = self.labels[index]
        return self.unlabeled_color if label is None else self.class_colors[label]

    def get_highlight_color(self, index: int) -> Quantity_Color:
        return self.highlight_color

    def set_current_index(self, index: int) -> None:
        if not self.face_items:
            return
        index = max(0, min(index, len(self.face_items) - 1))
        if self.current_index is not None:
            self.set_face_color(self.current_index, self.get_base_color(self.current_index))
        self.current_index = index
        self.apply_current_highlight(self.current_index)
        self.update_info()

    def apply_current_highlight(self, index: int) -> None:
        if self.highlight_enabled:
            self.set_face_color(index, self.get_highlight_color(index))
        else:
            self.set_face_color(index, self.get_base_color(index))

    def recolor_all_faces(self) -> None:
        for idx in range(len(self.face_items)):
            self.set_face_color(idx, self.get_base_color(idx), repaint=False)
        if self.current_index is not None and self.highlight_enabled:
            self.set_face_color(
                self.current_index, self.get_highlight_color(self.current_index), repaint=False
            )
        self.display.Repaint()

    def set_face_color(self, index: int, color: Quantity_Color, repaint: bool = True) -> None:
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
                Aspect_TOL_SOLID,
                1.0,
            )
            drawer.SetFaceBoundaryAspect(line_aspect)
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize .seg labels on STEP models.")
    parser.add_argument("--step", type=str, default=None, help="Path to STEP file")
    parser.add_argument("--seg", type=str, default=None, help="Path to .seg label file")
    args = parser.parse_args()

    step_path = args.step if args.step and os.path.exists(args.step) else None
    seg_path = args.seg if args.seg and os.path.exists(args.seg) else None

    app = QtWidgets.QApplication(sys.argv)
    window = SegViewer(step_path=step_path, seg_path=seg_path)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
