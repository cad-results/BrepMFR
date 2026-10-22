"""Pipe fitting 8-class definitions: names, colors, and utilities.

Single source of truth for class names and color palettes used across
visualization, export, and analysis scripts for the pipe fitting classifier.
"""

from typing import Tuple


# 8 pipe fitting class names
CLASS_NAMES = [
    "Elbow - Weld Fitting",      # 0
    "Elbow - Pipe End Fitting",  # 1
    "Elbow - Socket Fitting",    # 2
    "Tee - Weld Fitting",        # 3
    "Tee - Pipe End Fitting",    # 4
    "Tee - Socket Fitting",      # 5
    "Elbow - Miscellaneous",     # 6
    "Tee - Miscellaneous",       # 7
]

NUM_CLASSES: int = 8

# Hex color palette: matplotlib tab10 (first 8 entries)
CLASS_COLORS_HEX = [
    "#1f77b4",  # 0  Blue
    "#ff7f0e",  # 1  Orange
    "#2ca02c",  # 2  Green
    "#d62728",  # 3  Red
    "#9467bd",  # 4  Purple
    "#8c564b",  # 5  Brown
    "#e377c2",  # 6  Pink
    "#7f7f7f",  # 7  Gray
]

UNLABELED_COLOR_HEX: str = "#d0d0d0"
HIGHLIGHT_COLOR_HEX: str = "#FFD400"
EDGE_COLOR_HEX: str = "#2b2b2b"


def hex_to_rgb01(hex_str: str) -> Tuple[float, float, float]:
    """Convert hex color string to (r, g, b) floats in [0, 1]."""
    hex_str = hex_str.lstrip("#")
    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0
    return (r, g, b)


def get_class_name(class_idx: int) -> str:
    """Get class name for a class index. Returns 'unknown' for invalid indices."""
    if 0 <= class_idx < NUM_CLASSES:
        return CLASS_NAMES[class_idx]
    return "unknown"
