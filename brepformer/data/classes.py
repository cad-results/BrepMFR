"""MFTRCAD 27-class definitions: names, colors, and utilities.

Single source of truth for class names and color palettes used across
visualization, export, and analysis scripts.
"""

from typing import List, Tuple


# 27 MFTRCAD machining feature class names
CLASS_NAMES: List[str] = [
    "chamfer",                    # 0
    "through_hole",               # 1
    "triangular_passage",         # 2
    "rectangular_passage",        # 3
    "6sides_passage",             # 4
    "triangular_through_slot",    # 5
    "rectangular_through_slot",   # 6
    "circular_through_slot",      # 7
    "rectangular_through_step",   # 8
    "2sides_through_step",        # 9
    "slanted_through_step",       # 10
    "Oring",                      # 11
    "blind_hole",                 # 12
    "triangular_pocket",          # 13
    "rectangular_pocket",         # 14
    "6sides_pocket",              # 15
    "circular_end_pocket",        # 16
    "rectangular_blind_slot",     # 17
    "v_shaped_blind_slot",        # 18
    "circular_blind_slot",        # 19
    "rectangular_blind_step",     # 20
    "2sides_blind_step",          # 21
    "triangular_blind_step",      # 22
    "round",                      # 23
    "stock",                      # 24
    "rectangular_passage_2",      # 25
    "chamfer_2",                  # 26
]

NUM_CLASSES: int = len(CLASS_NAMES)

# Hex color palette for 27 classes (tab20 + tab20b)
CLASS_COLORS_HEX: List[str] = [
    "#1f77b4",  # 0  blue
    "#aec7e8",  # 1  light blue
    "#ff7f0e",  # 2  orange
    "#ffbb78",  # 3  peach
    "#2ca02c",  # 4  green
    "#98df8a",  # 5  light green
    "#d62728",  # 6  red
    "#ff9896",  # 7  salmon
    "#9467bd",  # 8  purple
    "#c5b0d5",  # 9  lavender
    "#8c564b",  # 10 brown
    "#c49c94",  # 11 tan
    "#e377c2",  # 12 pink
    "#f7b6d2",  # 13 light pink
    "#7f7f7f",  # 14 gray
    "#c7c7c7",  # 15 silver
    "#bcbd22",  # 16 olive
    "#dbdb8d",  # 17 khaki
    "#17becf",  # 18 teal
    "#9edae5",  # 19 light teal
    "#393b79",  # 20 navy
    "#5254a3",  # 21 indigo
    "#6b6ecf",  # 22 slate blue
    "#9c9ede",  # 23 periwinkle
    "#637939",  # 24 dark olive
    "#8ca252",  # 25 moss green
    "#b5cf6b",  # 26 lime
]

UNLABELED_COLOR_HEX: str = "#d0d0d0"
HIGHLIGHT_COLOR_HEX: str = "#FFD400"
EDGE_COLOR_HEX: str = "#2b2b2b"

# Human-readable color names
COLOR_NAMES: List[str] = [
    "blue", "light blue", "orange", "peach", "green",
    "light green", "red", "salmon", "purple", "lavender",
    "brown", "tan", "pink", "light pink", "gray",
    "silver", "olive", "khaki", "teal", "light teal",
    "navy", "indigo", "slate blue", "periwinkle", "dark olive",
    "moss green", "lime",
]


def hex_to_rgb01(color_hex: str) -> Tuple[float, float, float]:
    """Convert hex color string to (r, g, b) floats in [0, 1]."""
    color_hex = color_hex.lstrip("#")
    r = int(color_hex[0:2], 16) / 255.0
    g = int(color_hex[2:4], 16) / 255.0
    b = int(color_hex[4:6], 16) / 255.0
    return (r, g, b)


def hex_to_rgb255(color_hex: str) -> Tuple[int, int, int]:
    """Convert hex color string to (r, g, b) ints in [0, 255]."""
    color_hex = color_hex.lstrip("#")
    return (
        int(color_hex[0:2], 16),
        int(color_hex[2:4], 16),
        int(color_hex[4:6], 16),
    )


def get_class_color_rgb01(class_id: int) -> Tuple[float, float, float]:
    """Get RGB [0,1] color for a class ID. Returns unlabeled color for invalid IDs."""
    if 0 <= class_id < NUM_CLASSES:
        return hex_to_rgb01(CLASS_COLORS_HEX[class_id])
    return hex_to_rgb01(UNLABELED_COLOR_HEX)


def get_class_name(class_id: int) -> str:
    """Get class name for a class ID. Returns 'unknown' for invalid IDs."""
    if 0 <= class_id < NUM_CLASSES:
        return CLASS_NAMES[class_id]
    return "unknown"


# ---------------------------------------------------------------------------
# 8 "real" (grouped) machining feature classes
# ---------------------------------------------------------------------------
# Maps the 27 MFTRCAD fine-grained classes into 8 higher-level categories.

REAL_CLASS_NAMES: List[str] = [
    "other_surfaces",  # 0
    "through_hole",    # 1
    "blind_hole",      # 2
    "chamfer",         # 3
    "fillet",          # 4
    "through_cut",     # 5
    "blind_cut",       # 6
    "extrude",         # 7
]

REAL_NUM_CLASSES: int = len(REAL_CLASS_NAMES)

# Distinct hex colors for the 8 real classes
REAL_CLASS_COLORS_HEX: List[str] = [
    "#7f7f7f",  # 0 other_surfaces — gray
    "#1f77b4",  # 1 through_hole   — blue
    "#aec7e8",  # 2 blind_hole     — light blue
    "#ff7f0e",  # 3 chamfer        — orange
    "#2ca02c",  # 4 fillet         — green
    "#d62728",  # 5 through_cut    — red
    "#9467bd",  # 6 blind_cut      — purple
    "#8c564b",  # 7 extrude        — brown
]

# Mapping: CLASS_TO_REAL_CLASS[i] gives the real class ID for 27-class ID i.
CLASS_TO_REAL_CLASS: List[int] = [
    3,   # 0  chamfer                -> chamfer
    1,   # 1  through_hole           -> through_hole
    5,   # 2  triangular_passage     -> through_cut
    5,   # 3  rectangular_passage    -> through_cut
    5,   # 4  6sides_passage         -> through_cut
    5,   # 5  triangular_through_slot -> through_cut
    5,   # 6  rectangular_through_slot -> through_cut
    5,   # 7  circular_through_slot  -> through_cut
    7,   # 8  rectangular_through_step -> extrude
    7,   # 9  2sides_through_step    -> extrude
    7,   # 10 slanted_through_step   -> extrude
    7,   # 11 Oring                  -> extrude
    2,   # 12 blind_hole             -> blind_hole
    6,   # 13 triangular_pocket      -> blind_cut
    6,   # 14 rectangular_pocket     -> blind_cut
    6,   # 15 6sides_pocket          -> blind_cut
    6,   # 16 circular_end_pocket    -> blind_cut
    6,   # 17 rectangular_blind_slot -> blind_cut
    6,   # 18 v_shaped_blind_slot    -> blind_cut
    6,   # 19 circular_blind_slot    -> blind_cut
    7,   # 20 rectangular_blind_step -> extrude
    7,   # 21 2sides_blind_step      -> extrude
    7,   # 22 triangular_blind_step  -> extrude
    4,   # 23 round                  -> fillet
    0,   # 24 stock                  -> other_surfaces
    0,   # 25 rectangular_passage_2  -> other_surfaces
    3,   # 26 chamfer_2              -> chamfer
]


def get_real_class_name(real_id: int) -> str:
    """Get real class name for a real class ID. Returns 'unknown' for invalid IDs."""
    if 0 <= real_id < REAL_NUM_CLASSES:
        return REAL_CLASS_NAMES[real_id]
    return "unknown"


def get_real_class_color_rgb01(real_id: int) -> Tuple[float, float, float]:
    """Get RGB [0,1] color for a real class ID. Returns unlabeled color for invalid IDs."""
    if 0 <= real_id < REAL_NUM_CLASSES:
        return hex_to_rgb01(REAL_CLASS_COLORS_HEX[real_id])
    return hex_to_rgb01(UNLABELED_COLOR_HEX)


def map_labels_to_real(labels: List[int]) -> List[int]:
    """Convert a list of 27-class labels to 8 real class labels.

    Labels outside [0, 26] (e.g. -1 for unlabeled) are passed through unchanged.
    """
    result = []
    for label in labels:
        if 0 <= label < NUM_CLASSES:
            result.append(CLASS_TO_REAL_CLASS[label])
        else:
            result.append(label)
    return result


# ---------------------------------------------------------------------------
# 5 defeature classes (navin_defeaturing dataset)
# ---------------------------------------------------------------------------
# Remapped from the original 7-class scheme:
#   0 -> 0 (Random/Other), 1 -> 1 (Hole), 2 -> 1 (Hole),
#   3 -> 2 (Chamfer), 4 -> 3 (Fillet), 5 -> 4 (Cut), 6 -> 4 (Cut)

DEFEATURE_CLASS_NAMES: List[str] = [
    "random",   # 0
    "hole",     # 1
    "chamfer",  # 2
    "fillet",   # 3
    "cut",      # 4
]

DEFEATURE_NUM_CLASSES: int = len(DEFEATURE_CLASS_NAMES)

DEFEATURE_CLASS_COLORS_HEX: List[str] = [
    "#7f7f7f",  # 0 random  — gray
    "#1f77b4",  # 1 hole    — blue
    "#ff7f0e",  # 2 chamfer — orange
    "#2ca02c",  # 3 fillet  — green
    "#d62728",  # 4 cut     — red
]

# Mapping from original 7-class labels to 5 defeature classes
DEFEATURE_LABEL_REMAP: List[int] = [0, 1, 1, 2, 3, 4, 4]


def get_defeature_class_name(class_id: int) -> str:
    """Get defeature class name. Returns 'unknown' for invalid IDs."""
    if 0 <= class_id < DEFEATURE_NUM_CLASSES:
        return DEFEATURE_CLASS_NAMES[class_id]
    return "unknown"


def get_defeature_class_color_rgb01(class_id: int) -> Tuple[float, float, float]:
    """Get RGB [0,1] color for a defeature class ID."""
    if 0 <= class_id < DEFEATURE_NUM_CLASSES:
        return hex_to_rgb01(DEFEATURE_CLASS_COLORS_HEX[class_id])
    return hex_to_rgb01(UNLABELED_COLOR_HEX)


# ---------------------------------------------------------------------------
# 27 MFTRCAD → 5 defeature class mapping (for fine-tuning)
# ---------------------------------------------------------------------------
# Maps each of the 27 MFTRCAD fine-grained classes to the 5 defeature categories.
# Used when fine-tuning a pre-trained MFTRCAD model on the defeature dataset,
# allowing warm-start initialization of classifier heads via weight averaging.

CLASS_TO_DEFEATURE: List[int] = [
    2,   # 0  chamfer                -> chamfer
    1,   # 1  through_hole           -> hole
    4,   # 2  triangular_passage     -> cut
    4,   # 3  rectangular_passage    -> cut
    4,   # 4  6sides_passage         -> cut
    4,   # 5  triangular_through_slot -> cut
    4,   # 6  rectangular_through_slot -> cut
    4,   # 7  circular_through_slot  -> cut
    4,   # 8  rectangular_through_step -> cut
    4,   # 9  2sides_through_step    -> cut
    4,   # 10 slanted_through_step   -> cut
    4,   # 11 Oring                  -> cut
    1,   # 12 blind_hole             -> hole
    4,   # 13 triangular_pocket      -> cut
    4,   # 14 rectangular_pocket     -> cut
    4,   # 15 6sides_pocket          -> cut
    4,   # 16 circular_end_pocket    -> cut
    4,   # 17 rectangular_blind_slot -> cut
    4,   # 18 v_shaped_blind_slot    -> cut
    4,   # 19 circular_blind_slot    -> cut
    4,   # 20 rectangular_blind_step -> cut
    4,   # 21 2sides_blind_step      -> cut
    4,   # 22 triangular_blind_step  -> cut
    3,   # 23 round                  -> fillet
    0,   # 24 stock                  -> random
    4,   # 25 rectangular_passage_2  -> cut
    2,   # 26 chamfer_2              -> chamfer
]


def map_labels_to_defeature(labels: List[int]) -> List[int]:
    """Convert a list of 27-class labels to 5 defeature class labels.

    Labels outside [0, 26] (e.g. -1 for unlabeled) are passed through unchanged.
    """
    result = []
    for label in labels:
        if 0 <= label < NUM_CLASSES:
            result.append(CLASS_TO_DEFEATURE[label])
        else:
            result.append(label)
    return result
