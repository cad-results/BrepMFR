#!/usr/bin/env python3
"""BrepMFR Viewer - Interactive 3D visualization for MTFRCAD B-rep models.

Browse preprocessed MTFRCAD data with ground-truth labels, view model
predictions sorted by accuracy, display per-class metrics, confusion
matrices, and t-SNE embeddings.

Modes:
    browse       -- Browse preprocessed data with GT labels
    predictions  -- Browse models sorted by prediction accuracy (Jaccard)
    analysis     -- Show analysis plots (metrics, confusion, embeddings)

Keyboard Controls:
    T/TAB       Cycle views: Plain -> GT -> Predicted -> Comparison
    D/RIGHT     Next model
    A/LEFT      Previous model
    1           Sort: worst accuracy first
    2           Sort: best accuracy first
    3           Sort: random shuffle
    M           Metrics chart popup
    N           Confusion matrix popup
    E           Embeddings plot popup
    I           Print model info to console
    L           Toggle class legend (3D swatches + sidebar panel)
    F           Toggle face labels (predicted feature name at each face)
    S           Screenshot
    R           Reset camera
    H           Help
    ESC/Q       Exit
"""

# Configure environment for WSL2/software rendering BEFORE importing Open3D
import os
os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
os.environ.setdefault('MESA_GL_VERSION_OVERRIDE', '3.3')
os.environ.setdefault('MESA_GLSL_VERSION_OVERRIDE', '330')
os.environ.setdefault('GALLIUM_DRIVER', 'llvmpipe')

import argparse
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("Error: open3d is required. Install with: pip install open3d")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ─── MTFRCAD class names (27 machining feature classes) ─────────────────────
CLASS_NAMES = [
    "chamfer", "through_hole", "triangular_passage", "rectangular_passage",
    "6sides_passage", "triangular_through_slot", "rectangular_through_slot",
    "circular_through_slot", "rectangular_through_step",
    "2sides_through_step", "slanted_through_step", "Oring",
    "blind_hole", "triangular_pocket", "rectangular_pocket",
    "6sides_pocket", "circular_end_pocket", "rectangular_blind_slot",
    "v_shaped_blind_slot", "circular_blind_slot", "rectangular_blind_step",
    "2sides_blind_step", "triangular_blind_step", "round",
    "stock", "rectangular_passage_2", "chamfer_2",
]

# Human-readable color names for the 27-class palette (tab20 + tab20b)
COLOR_NAMES = [
    "blue", "light blue", "orange", "peach", "green",
    "light green", "red", "salmon", "purple", "lavender",
    "brown", "tan", "pink", "light pink", "gray",
    "silver", "olive", "khaki", "teal", "light teal",
    "navy", "indigo", "slate blue", "periwinkle", "dark olive",
    "moss green", "lime",
]


class MTFRCADDataManager:
    """Data loading and indexing for MTFRCAD preprocessed data."""

    def __init__(self, processed_dir: str, data_dir: str = None,
                 analysis_dir: str = "analysis_results"):
        self.processed_dir = Path(processed_dir)
        self.data_dir = Path(data_dir) if data_dir else None
        self.analysis_dir = Path(analysis_dir)

        self.samples: List[dict] = []
        self.model_id_to_idx: Dict[str, int] = {}
        self.predictions: Optional[dict] = None
        self.per_class_metrics: Optional[list] = None
        self.embeddings_tsne: Optional[np.ndarray] = None
        self.embeddings_labels: Optional[np.ndarray] = None
        self.confusion_matrix: Optional[np.ndarray] = None
        self.per_class_confusion: Optional[np.ndarray] = None

    def load_split(self, split: str):
        """Load preprocessed pkl data for given split."""
        pkl_path = self.processed_dir / f"{split}.pkl"
        if not pkl_path.exists():
            print(f"Error: {pkl_path} not found")
            sys.exit(1)

        print(f"Loading {split} split from {pkl_path}...")
        with open(pkl_path, "rb") as f:
            self.samples = pickle.load(f)
        print(f"Loaded {len(self.samples)} samples")

        self.model_id_to_idx = {}
        for i, s in enumerate(self.samples):
            self.model_id_to_idx[s["model_id"]] = i

    def load_predictions(self) -> bool:
        """Load predictions.json from analysis_dir."""
        pred_path = self.analysis_dir / "predictions.json"
        if not pred_path.exists():
            print(f"Warning: {pred_path} not found (run analyze.py --mode predictions first)")
            return False

        with open(pred_path) as f:
            self.predictions = json.load(f)
        print(f"Loaded predictions for {len(self.predictions.get('predictions', []))} samples")
        return True

    def load_metrics(self) -> bool:
        """Load per_class_metrics.json."""
        metrics_path = self.analysis_dir / "per_class_metrics.json"
        if not metrics_path.exists():
            return False
        with open(metrics_path) as f:
            self.per_class_metrics = json.load(f)
        return True

    def load_embeddings(self) -> bool:
        """Load t-SNE embeddings and labels."""
        tsne_path = self.analysis_dir / "embeddings_tsne.npy"
        labels_path = self.analysis_dir / "labels.npy"
        if not tsne_path.exists() or not labels_path.exists():
            return False
        self.embeddings_tsne = np.load(tsne_path)
        self.embeddings_labels = np.load(labels_path)
        return True

    def load_confusion_matrices(self) -> bool:
        """Load or generate confusion matrix data."""
        ml_path = self.analysis_dir / "confusion_matrix_multilabel.npy"
        sl_path = self.analysis_dir / "confusion_matrix.npy"
        pc_path = self.analysis_dir / "per_class_confusion.npy"

        if ml_path.exists():
            self.confusion_matrix = np.load(ml_path)
        elif sl_path.exists():
            self.confusion_matrix = np.load(sl_path)

        if pc_path.exists():
            self.per_class_confusion = np.load(pc_path)

        # If no pre-computed matrix, generate from predictions
        if self.confusion_matrix is None and self.predictions is not None:
            self.confusion_matrix = self._generate_confusion_matrix()

        return self.confusion_matrix is not None

    def _generate_confusion_matrix(self) -> Optional[np.ndarray]:
        """Generate a multi-label co-occurrence/confusion matrix from predictions."""
        preds_list = self.predictions.get("predictions", [])
        if not preds_list:
            return None

        n_classes = len(CLASS_NAMES)
        cm = np.zeros((n_classes, n_classes), dtype=np.float64)

        for p in preds_list:
            gt_classes = set(p.get("target_classes", []))
            pred_classes = set(p.get("predicted_classes", []))

            for gt_c in gt_classes:
                for pred_c in pred_classes:
                    if 0 <= gt_c < n_classes and 0 <= pred_c < n_classes:
                        cm[gt_c, pred_c] += 1

        return cm

    def get_label_json(self, model_id: str) -> Optional[dict]:
        """Load per-face GT labels from the raw data labels/ directory."""
        if self.data_dir is None:
            return None
        label_path = self.data_dir / "labels" / f"{model_id}_result.json"
        if not label_path.exists():
            return None
        with open(label_path) as f:
            return json.load(f)

    def get_prediction_for_model(self, model_id: str) -> Optional[dict]:
        """Get prediction dict for a model_id."""
        if self.predictions is None:
            return None
        for p in self.predictions.get("predictions", []):
            if p["model_id"] == model_id:
                return p
        return None

    def get_sorted_indices(self, sort_mode: str) -> List[int]:
        """Get sample indices sorted by mode.

        Args:
            sort_mode: 'index', 'best', 'worst', 'random'

        Returns:
            List of indices into self.samples.
        """
        n = len(self.samples)
        if sort_mode == "index":
            return list(range(n))

        if sort_mode == "random":
            indices = list(range(n))
            random.shuffle(indices)
            return indices

        # For best/worst, use predictions if available
        if self.predictions is None:
            return list(range(n))

        pred_map = {}
        for p in self.predictions.get("predictions", []):
            pred_map[p["model_id"]] = p.get("jaccard_similarity", 0.0)

        scored = []
        for i, s in enumerate(self.samples):
            mid = s["model_id"]
            jaccard = pred_map.get(mid, -1.0)
            scored.append((i, jaccard))

        if sort_mode == "worst":
            scored.sort(key=lambda x: x[1])
        elif sort_mode == "best":
            scored.sort(key=lambda x: -x[1])

        return [idx for idx, _ in scored]


class BrepColorizer:
    """Color palette and coloring logic for 27 MTFRCAD classes."""

    def __init__(self):
        self.palette = self._build_palette(27)

    def _build_palette(self, n: int) -> np.ndarray:
        """Build a fixed 27-class color palette using tab20 + tab20b."""
        if not HAS_MATPLOTLIB:
            # Fallback: evenly spaced HSV colors
            colors = []
            for i in range(n):
                h = i / n
                import colorsys
                rgb = colorsys.hsv_to_rgb(h, 0.8, 0.9)
                colors.append(rgb)
            return np.array(colors)

        cmap1 = plt.colormaps.get_cmap("tab20")
        cmap2 = plt.colormaps.get_cmap("tab20b")
        colors1 = [cmap1(i / 20.0)[:3] for i in range(20)]
        colors2 = [cmap2(i / 20.0)[:3] for i in range(20)]
        all_colors = colors1 + colors2
        return np.array(all_colors[:n])

    def get_class_color(self, class_idx: int) -> np.ndarray:
        """Get RGB color for a class index."""
        return self.palette[class_idx % len(self.palette)]

    def color_faces_by_gt(self, face_grid: np.ndarray, label_json: dict) -> np.ndarray:
        """Color points by per-face GT class labels.

        Args:
            face_grid: (N, 7, 10, 10) face UV-grid data
            label_json: dict with 'cls' key mapping face_idx -> class_id

        Returns:
            (P, 3) RGB colors for each point
        """
        n_faces = face_grid.shape[0]
        cls_map = label_json.get("cls", {})
        colors = []
        for face_idx in range(n_faces):
            mask = face_grid[face_idx, 6, :, :]  # channel 6 = mask
            n_pts = int((mask > 0.5).sum())
            if n_pts == 0:
                n_pts = mask.size  # fallback: use all points

            class_id = cls_map.get(str(face_idx), 24)  # default to 'stock'
            color = self.get_class_color(int(class_id))
            colors.append(np.tile(color, (n_pts, 1)))

        if colors:
            return np.vstack(colors)
        return np.zeros((0, 3))

    def color_faces_by_prediction(self, face_grid: np.ndarray,
                                   pred_classes: List[int]) -> np.ndarray:
        """Color all faces by model-level predicted classes.

        Faces are colored by cycling through predicted classes.

        Args:
            face_grid: (N, 7, 10, 10) face UV-grid data
            pred_classes: list of predicted class indices

        Returns:
            (P, 3) RGB colors for each point
        """
        n_faces = face_grid.shape[0]
        if not pred_classes:
            pred_classes = [24]  # default to stock

        colors = []
        for face_idx in range(n_faces):
            mask = face_grid[face_idx, 6, :, :]
            n_pts = int((mask > 0.5).sum())
            if n_pts == 0:
                n_pts = mask.size

            cls_idx = pred_classes[face_idx % len(pred_classes)]
            color = self.get_class_color(cls_idx)
            colors.append(np.tile(color, (n_pts, 1)))

        if colors:
            return np.vstack(colors)
        return np.zeros((0, 3))

    def plain_color(self, n_points: int) -> np.ndarray:
        """Return uniform gray color for all points."""
        return np.full((n_points, 3), 0.6)


class BrepMFRViewer:
    """Main Open3D viewer for BrepMFR MTFRCAD data."""

    # View modes
    VIEW_PLAIN = 0
    VIEW_GT = 1
    VIEW_PREDICTED = 2
    VIEW_COMPARISON = 3
    VIEW_NAMES = ["Plain", "GT Labels", "Predicted", "Comparison"]

    def __init__(self, data_manager: MTFRCADDataManager, colorizer: BrepColorizer):
        self.dm = data_manager
        self.colorizer = colorizer

        self.vis: Optional[o3d.visualization.VisualizerWithKeyCallback] = None
        self.current_geometries: List = []

        # Navigation state
        self.sorted_indices: List[int] = []
        self.nav_pos: int = 0
        self.view_mode: int = self.VIEW_PLAIN
        self.sort_mode: str = "index"
        self.show_legend: bool = False
        self.show_labels: bool = False
        self._window_destroyed: bool = False

        # Matplotlib figure handles for legend/labels panels
        self._legend_fig = None
        self._labels_fig = None
        # Separate list for label marker geometries (3D spheres at centroids)
        self._label_geometries: List = []
        # Separate list for 3D legend geometries (colored cubes in scene)
        self._legend_geometries: List = []

    def _extract_points(self, face_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """Extract xyz points and normals from face_grid (N, 7, 10, 10).

        Channels: 0-2 = xyz, 3-5 = normals, 6 = mask.
        Uses mask channel to filter valid points.

        Returns:
            points: (P, 3)
            normals: (P, 3)
            n_faces: int
        """
        n_faces = face_grid.shape[0]
        all_pts = []
        all_normals = []

        for fi in range(n_faces):
            mask = face_grid[fi, 6, :, :]  # (10, 10)
            valid = mask > 0.5

            if valid.sum() == 0:
                # Use all points if mask is all zeros
                xyz = face_grid[fi, 0:3, :, :].reshape(3, -1).T  # (100, 3)
                nrm = face_grid[fi, 3:6, :, :].reshape(3, -1).T
            else:
                xyz = face_grid[fi, 0:3, :, :][:, valid].T  # (P, 3)
                nrm = face_grid[fi, 3:6, :, :][:, valid].T

            all_pts.append(xyz)
            all_normals.append(nrm)

        if all_pts:
            points = np.vstack(all_pts)
            normals = np.vstack(all_normals)
        else:
            points = np.zeros((0, 3))
            normals = np.zeros((0, 3))

        return points, normals, n_faces

    def _build_point_cloud(self, points: np.ndarray, normals: np.ndarray,
                            colors: np.ndarray) -> o3d.geometry.PointCloud:
        """Build an Open3D point cloud."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if normals.shape[0] == points.shape[0]:
            pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
        if colors.shape[0] == points.shape[0]:
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        return pcd

    def _get_current_sample(self) -> Optional[dict]:
        """Get current sample dict."""
        if not self.sorted_indices:
            return None
        idx = self.sorted_indices[self.nav_pos]
        return self.dm.samples[idx]

    def _get_current_model_id(self) -> str:
        sample = self._get_current_sample()
        return sample["model_id"] if sample else "unknown"

    def _refresh_display(self):
        """Rebuild and display the current model."""
        if self.vis is None:
            return

        # Remove old geometries (model + label markers)
        for geom in self.current_geometries:
            self.vis.remove_geometry(geom, reset_bounding_box=False)
        self.current_geometries.clear()
        for geom in self._label_geometries:
            self.vis.remove_geometry(geom, reset_bounding_box=False)
        self._label_geometries.clear()
        for geom in self._legend_geometries:
            self.vis.remove_geometry(geom, reset_bounding_box=False)
        self._legend_geometries.clear()

        sample = self._get_current_sample()
        if sample is None:
            return

        face_grid = sample["face_grid"]  # numpy (N, 7, 10, 10)
        model_id = sample["model_id"]
        points, normals, n_faces = self._extract_points(face_grid)

        if points.shape[0] == 0:
            print(f"  Warning: No points extracted for {model_id}")
            return

        if self.view_mode == self.VIEW_COMPARISON:
            self._show_comparison(face_grid, points, normals, model_id)
        else:
            colors = self._get_colors_for_mode(face_grid, points, model_id)
            pcd = self._build_point_cloud(points, normals, colors)
            self.vis.add_geometry(pcd, reset_bounding_box=(self.nav_pos == 0 and len(self.current_geometries) == 0))
            self.current_geometries.append(pcd)

        # Add face label markers if labels are enabled
        if self.show_labels:
            self._add_face_label_markers(face_grid, model_id)
            self._update_labels_figure(face_grid, model_id)

        # Show legend (3D in-scene cubes + matplotlib sidebar) for colored views
        if self.show_legend and self.view_mode != self.VIEW_PLAIN:
            self._add_3d_legend(face_grid, model_id)
            self._update_legend_figure(face_grid, model_id)
        elif self.show_legend and self.view_mode == self.VIEW_PLAIN:
            # Hide legend when switching to plain view
            if self._legend_fig is not None:
                plt.close(self._legend_fig)
                self._legend_fig = None

        # Reset bounding box for first display
        if len(self.current_geometries) > 0:
            self.vis.reset_view_point(True)

        self._print_status()

    def _get_colors_for_mode(self, face_grid: np.ndarray, points: np.ndarray,
                              model_id: str) -> np.ndarray:
        """Get point colors based on current view mode."""
        n_points = points.shape[0]

        if self.view_mode == self.VIEW_PLAIN:
            return self.colorizer.plain_color(n_points)

        elif self.view_mode == self.VIEW_GT:
            label_json = self.dm.get_label_json(model_id)
            if label_json is not None:
                return self.colorizer.color_faces_by_gt(face_grid, label_json)
            # Fallback: use multi-hot label from sample
            return self.colorizer.plain_color(n_points)

        elif self.view_mode == self.VIEW_PREDICTED:
            pred = self.dm.get_prediction_for_model(model_id)
            if pred is not None:
                pred_classes = pred.get("predicted_classes", [24])
                return self.colorizer.color_faces_by_prediction(face_grid, pred_classes)
            return self.colorizer.plain_color(n_points)

        return self.colorizer.plain_color(n_points)

    def _show_comparison(self, face_grid: np.ndarray, points: np.ndarray,
                          normals: np.ndarray, model_id: str):
        """Show GT and predicted side-by-side with offset."""
        # Compute offset based on model extent
        extent = points.max(axis=0) - points.min(axis=0)
        offset = np.array([extent[0] * 1.3, 0, 0])

        # GT (left)
        label_json = self.dm.get_label_json(model_id)
        if label_json is not None:
            gt_colors = self.colorizer.color_faces_by_gt(face_grid, label_json)
        else:
            gt_colors = self.colorizer.plain_color(points.shape[0])

        pcd_gt = self._build_point_cloud(points - offset / 2, normals, gt_colors)
        self.vis.add_geometry(pcd_gt, reset_bounding_box=False)
        self.current_geometries.append(pcd_gt)

        # Predicted (right)
        pred = self.dm.get_prediction_for_model(model_id)
        if pred is not None:
            pred_classes = pred.get("predicted_classes", [24])
            pred_colors = self.colorizer.color_faces_by_prediction(face_grid, pred_classes)
        else:
            pred_colors = self.colorizer.plain_color(points.shape[0])

        pcd_pred = self._build_point_cloud(points + offset / 2, normals, pred_colors)
        self.vis.add_geometry(pcd_pred, reset_bounding_box=False)
        self.current_geometries.append(pcd_pred)

    def _print_status(self):
        """Print current model info to console."""
        sample = self._get_current_sample()
        if sample is None:
            return

        model_id = sample["model_id"]
        n_faces = sample["num_faces"]
        view_name = self.VIEW_NAMES[self.view_mode]
        pos = self.nav_pos + 1
        total = len(self.sorted_indices)

        pred = self.dm.get_prediction_for_model(model_id)
        jaccard_str = ""
        if pred:
            jaccard = pred.get("jaccard_similarity", -1)
            if jaccard >= 0:
                jaccard_str = f" | Jaccard: {jaccard:.3f}"

        print(f"\r[{pos}/{total}] {model_id} | {n_faces} faces | {view_name} | sort={self.sort_mode}{jaccard_str}    ", end="", flush=True)

    def _print_model_info(self):
        """Print detailed model info."""
        sample = self._get_current_sample()
        if sample is None:
            return

        model_id = sample["model_id"]
        label = sample["label"]
        n_faces = sample["num_faces"]
        n_edges = sample["num_edges"]

        print(f"\n{'='*60}")
        print(f"Model: {model_id}")
        print(f"Faces: {n_faces}, Edges: {n_edges}")

        # GT classes from multi-hot label
        gt_classes = np.where(label > 0.5)[0].tolist()
        gt_names = [CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}" for c in gt_classes]
        print(f"GT classes ({len(gt_classes)}): {gt_classes}")
        print(f"GT names: {gt_names}")

        # Per-face GT if available
        label_json = self.dm.get_label_json(model_id)
        if label_json:
            cls_map = label_json.get("cls", {})
            face_classes = set(int(v) for v in cls_map.values())
            print(f"Per-face classes: {sorted(face_classes)}")

        # Prediction
        pred = self.dm.get_prediction_for_model(model_id)
        if pred:
            print(f"Predicted classes: {pred.get('predicted_classes', [])}")
            print(f"Jaccard similarity: {pred.get('jaccard_similarity', 'N/A')}")
            print(f"Correct: {pred.get('correct', 'N/A')}")
            print(f"Num pred: {pred.get('num_pred', 'N/A')}, Num target: {pred.get('num_target', 'N/A')}")

        print(f"{'='*60}")

    def _get_active_classes(self, face_grid: np.ndarray, model_id: str) -> List[int]:
        """Get the set of class indices active in the current view."""
        active = set()

        if self.view_mode == self.VIEW_GT or self.view_mode == self.VIEW_COMPARISON:
            label_json = self.dm.get_label_json(model_id)
            if label_json:
                cls_map = label_json.get("cls", {})
                for v in cls_map.values():
                    active.add(int(v))

        if self.view_mode == self.VIEW_PREDICTED or self.view_mode == self.VIEW_COMPARISON:
            pred = self.dm.get_prediction_for_model(model_id)
            if pred:
                for c in pred.get("predicted_classes", []):
                    active.add(int(c))

        if not active:
            # Fallback: show all classes
            active = set(range(len(CLASS_NAMES)))

        return sorted(active)

    def _get_face_labels(self, face_grid: np.ndarray, model_id: str) -> List[Tuple[int, str, str]]:
        """Get per-face label info: list of (face_idx, class_name, color_name)."""
        n_faces = face_grid.shape[0]
        face_labels = []

        if self.view_mode == self.VIEW_PREDICTED or self.view_mode == self.VIEW_COMPARISON:
            pred = self.dm.get_prediction_for_model(model_id)
            if pred:
                pred_classes = pred.get("predicted_classes", [24])
                for fi in range(n_faces):
                    cls_idx = pred_classes[fi % len(pred_classes)]
                    cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"class_{cls_idx}"
                    col_name = COLOR_NAMES[cls_idx] if cls_idx < len(COLOR_NAMES) else "unknown"
                    face_labels.append((fi, cls_name, col_name))
                return face_labels

        if self.view_mode == self.VIEW_GT:
            label_json = self.dm.get_label_json(model_id)
            if label_json:
                cls_map = label_json.get("cls", {})
                for fi in range(n_faces):
                    cls_idx = int(cls_map.get(str(fi), 24))
                    cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"class_{cls_idx}"
                    col_name = COLOR_NAMES[cls_idx] if cls_idx < len(COLOR_NAMES) else "unknown"
                    face_labels.append((fi, cls_name, col_name))
                return face_labels

        return face_labels

    def _compute_face_centroids(self, face_grid: np.ndarray) -> np.ndarray:
        """Compute the centroid of each face. Returns (N, 3) array."""
        n_faces = face_grid.shape[0]
        centroids = np.zeros((n_faces, 3))
        for fi in range(n_faces):
            mask = face_grid[fi, 6, :, :]
            valid = mask > 0.5
            if valid.sum() == 0:
                xyz = face_grid[fi, 0:3, :, :].reshape(3, -1).T
            else:
                xyz = face_grid[fi, 0:3, :, :][:, valid].T
            centroids[fi] = xyz.mean(axis=0)
        return centroids

    def _add_face_label_markers(self, face_grid: np.ndarray, model_id: str):
        """Add small colored spheres + leader lines at face centroids as label markers."""
        face_labels = self._get_face_labels(face_grid, model_id)
        if not face_labels:
            return

        centroids = self._compute_face_centroids(face_grid)
        # Determine a reasonable marker size based on model extent
        all_pts = centroids
        extent = all_pts.max(axis=0) - all_pts.min(axis=0)
        marker_radius = max(extent.max() * 0.015, 0.005)
        line_length = max(extent.max() * 0.06, 0.02)

        for fi, cls_name, col_name in face_labels:
            if fi >= len(centroids):
                continue
            centroid = centroids[fi]

            # Get class index to look up color
            pred = self.dm.get_prediction_for_model(model_id)
            if pred:
                pred_classes = pred.get("predicted_classes", [24])
                cls_idx = pred_classes[fi % len(pred_classes)]
            else:
                label_json = self.dm.get_label_json(model_id)
                if label_json:
                    cls_idx = int(label_json.get("cls", {}).get(str(fi), 24))
                else:
                    cls_idx = 24
            color = self.colorizer.get_class_color(cls_idx)

            # Sphere at centroid offset slightly upward
            tip = centroid + np.array([0, line_length, 0])
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=marker_radius)
            sphere.translate(tip)
            sphere.paint_uniform_color(color)
            sphere.compute_vertex_normals()
            self.vis.add_geometry(sphere, reset_bounding_box=False)
            self._label_geometries.append(sphere)

            # Leader line from centroid to sphere
            line_pts = np.array([centroid, tip])
            line = o3d.geometry.LineSet()
            line.points = o3d.utility.Vector3dVector(line_pts)
            line.lines = o3d.utility.Vector2iVector([[0, 1]])
            line.colors = o3d.utility.Vector3dVector([color])
            self.vis.add_geometry(line, reset_bounding_box=False)
            self._label_geometries.append(line)

    def _add_3d_legend(self, face_grid: np.ndarray, model_id: str):
        """Add colored cube legend entries in the 3D scene to the right of the model."""
        active_classes = self._get_active_classes(face_grid, model_id)
        if not active_classes:
            return

        # Get model bounding box from face centroids
        centroids = self._compute_face_centroids(face_grid)
        if len(centroids) == 0:
            return
        model_min = centroids.min(axis=0)
        model_max = centroids.max(axis=0)
        model_extent = model_max - model_min

        # Position legend column to the right of the model
        legend_x = model_max[0] + model_extent[0] * 0.5
        legend_top_y = model_max[1]
        legend_z = (model_min[2] + model_max[2]) / 2

        cube_size = max(model_extent.max() * 0.035, 0.012)
        spacing = cube_size * 2.2

        for i, cls_idx in enumerate(active_classes):
            color = self.colorizer.get_class_color(cls_idx)
            y = legend_top_y - i * spacing

            # Colored cube swatch
            cube = o3d.geometry.TriangleMesh.create_box(
                width=cube_size, height=cube_size, depth=cube_size * 0.3
            )
            cube.translate([legend_x, y - cube_size / 2, legend_z - cube_size * 0.15])
            cube.paint_uniform_color(color)
            cube.compute_vertex_normals()
            self.vis.add_geometry(cube, reset_bounding_box=False)
            self._legend_geometries.append(cube)

            # Small white dash next to cube as a visual separator
            dash = o3d.geometry.TriangleMesh.create_box(
                width=cube_size * 0.6, height=cube_size * 0.1, depth=cube_size * 0.1
            )
            dash.translate([legend_x + cube_size * 1.3, y - cube_size * 0.05, legend_z])
            dash.paint_uniform_color([0.8, 0.8, 0.8])
            dash.compute_vertex_normals()
            self.vis.add_geometry(dash, reset_bounding_box=False)
            self._legend_geometries.append(dash)

    def _update_legend_figure(self, face_grid: np.ndarray = None, model_id: str = None):
        """Create/update a matplotlib figure showing the class color legend."""
        if not HAS_MATPLOTLIB:
            print("\n  matplotlib not available for legend")
            return

        # Determine active classes
        if face_grid is not None and model_id is not None:
            active_classes = self._get_active_classes(face_grid, model_id)
        else:
            active_classes = list(range(len(CLASS_NAMES)))

        # Close old legend figure
        if self._legend_fig is not None:
            plt.close(self._legend_fig)
            self._legend_fig = None

        n = len(active_classes)
        fig_height = max(3.0, n * 0.45 + 1.2)
        fig, ax = plt.subplots(figsize=(5.5, fig_height))
        fig.canvas.manager.set_window_title("Class Legend")
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#1a1a1a')
        ax.set_xlim(0, 12)
        ax.set_ylim(0, n + 1.5)
        ax.axis('off')
        ax.set_title("Feature Color Legend", fontsize=13, fontweight='bold',
                      pad=10, color='white')

        for i, cls_idx in enumerate(active_classes):
            y = n - i + 0.3
            color = self.colorizer.get_class_color(cls_idx)
            color_name = COLOR_NAMES[cls_idx] if cls_idx < len(COLOR_NAMES) else "unknown"
            class_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"class_{cls_idx}"

            # Colored swatch (larger)
            ax.add_patch(plt.Rectangle((0.3, y - 0.35), 1.1, 0.7,
                                       facecolor=color, edgecolor='#555555',
                                       linewidth=0.8, joinstyle='round'))
            # Class name (bold) + color name
            ax.text(1.8, y, f"{class_name}", va='center', fontsize=10,
                    color='white', fontfamily='monospace', fontweight='bold')
            ax.text(1.8 + len(class_name) * 0.52 + 0.3, y, f"({color_name})",
                    va='center', fontsize=9, color='#aaaaaa', fontfamily='monospace')

        fig.tight_layout(pad=1.0)
        self._legend_fig = fig

        # Position legend window to the right of the Open3D viewer
        try:
            mng = fig.canvas.manager
            if hasattr(mng, 'window') and hasattr(mng.window, 'wm_geometry'):
                # TkAgg backend: position to right side of screen
                mng.window.wm_geometry("+940+50")
            elif hasattr(mng, 'window') and hasattr(mng.window, 'move'):
                # Qt backend
                mng.window.move(940, 50)
        except Exception:
            pass  # Fall back to default positioning

        plt.show(block=False)
        plt.pause(0.1)

    def _update_labels_figure(self, face_grid: np.ndarray, model_id: str):
        """Create/update a matplotlib figure showing per-face predicted labels."""
        if not HAS_MATPLOTLIB:
            return

        face_labels = self._get_face_labels(face_grid, model_id)
        if not face_labels:
            return

        # Close old labels figure
        if self._labels_fig is not None:
            plt.close(self._labels_fig)
            self._labels_fig = None

        n = len(face_labels)
        # Show in columns if many faces
        cols = max(1, (n + 29) // 30)
        rows_per_col = (n + cols - 1) // cols
        fig_width = cols * 4.0
        fig_height = max(2.5, rows_per_col * 0.35 + 1.2)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        fig.canvas.manager.set_window_title("Face Labels")
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#1a1a1a')
        ax.axis('off')
        ax.set_title(f"Predicted Face Labels — {model_id}",
                      fontsize=11, fontweight='bold', pad=10, color='white')
        ax.set_xlim(0, cols * 11)
        ax.set_ylim(0, rows_per_col + 1.5)

        for i, (fi, cls_name, col_name) in enumerate(face_labels):
            col = i // rows_per_col
            row = i % rows_per_col
            x_off = col * 11
            y = rows_per_col - row + 0.3

            # Get color for the swatch
            pred = self.dm.get_prediction_for_model(model_id)
            if pred:
                pred_classes = pred.get("predicted_classes", [24])
                cls_idx = pred_classes[fi % len(pred_classes)]
            else:
                label_json = self.dm.get_label_json(model_id)
                cls_idx = int(label_json.get("cls", {}).get(str(fi), 24)) if label_json else 24
            color = self.colorizer.get_class_color(cls_idx)

            ax.add_patch(plt.Rectangle((x_off + 0.2, y - 0.3), 0.7, 0.6,
                                       facecolor=color, edgecolor='#555555',
                                       linewidth=0.5, joinstyle='round'))
            ax.text(x_off + 1.2, y, f"Face {fi}: {cls_name}",
                    va='center', fontsize=9, color='white', fontfamily='monospace')

        fig.tight_layout(pad=1.0)
        self._labels_fig = fig

        # Position labels window below the legend
        try:
            mng = fig.canvas.manager
            if hasattr(mng, 'window') and hasattr(mng.window, 'wm_geometry'):
                mng.window.wm_geometry("+940+450")
            elif hasattr(mng, 'window') and hasattr(mng.window, 'move'):
                mng.window.move(940, 450)
        except Exception:
            pass

        plt.show(block=False)
        plt.pause(0.1)

    # ─── Analysis popup methods ────────────────────────────────────────────

    def _show_metrics_popup(self, block=False):
        """Show per-class metrics bar chart.

        Args:
            block: If True, don't call plt.show() (caller handles it).
                   If False, call plt.show(block=False) + plt.pause(0.5).
        """
        if not HAS_MATPLOTLIB:
            print("\n  matplotlib not available for metrics popup")
            return

        if not self.dm.load_metrics():
            print("\n  per_class_metrics.json not found")
            return

        metrics = self.dm.per_class_metrics
        classes = [m["class"] for m in metrics]
        f1s = [m["f1"] for m in metrics]
        precisions = [m["precision"] for m in metrics]
        recalls = [m["recall"] for m in metrics]
        names = [CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c) for c in classes]

        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(classes))
        width = 0.25

        ax.bar(x - width, precisions, width, label="Precision", alpha=0.8)
        ax.bar(x, recalls, width, label="Recall", alpha=0.8)
        ax.bar(x + width, f1s, width, label="F1", alpha=0.8)

        ax.set_xlabel("Class")
        ax.set_ylabel("Score")
        ax.set_title("Per-Class Metrics")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        if not block:
            plt.show(block=False)
            plt.pause(0.5)
        print("\n  Metrics popup shown")

    def _show_confusion_popup(self, block=False):
        """Show confusion matrix heatmap.

        Args:
            block: If True, don't call plt.show() (caller handles it).
                   If False, call plt.show(block=False) + plt.pause(0.5).
        """
        if not HAS_MATPLOTLIB:
            print("\n  matplotlib not available for confusion popup")
            return

        # Ensure predictions are loaded so matrix can be generated if needed
        if self.dm.predictions is None:
            self.dm.load_predictions()
        if not self.dm.load_confusion_matrices():
            print("\n  Confusion matrix data not found (need predictions.json with target_classes)")
            return

        cm = self.dm.confusion_matrix
        names = [CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i) for i in range(cm.shape[0])]

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(cm, cmap="Blues", aspect="auto")

        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(names, fontsize=6)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        ax.set_title("Multi-Label Co-occurrence Matrix")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        if not block:
            plt.show(block=False)
            plt.pause(0.5)
        print("\n  Confusion matrix popup shown")

    def _show_embeddings_popup(self, block=False):
        """Show t-SNE embedding scatter.

        Args:
            block: If True, don't call plt.show() (caller handles it).
                   If False, call plt.show(block=False) + plt.pause(0.5).
        """
        if not HAS_MATPLOTLIB:
            print("\n  matplotlib not available for embeddings popup")
            return

        if not self.dm.load_embeddings():
            print("\n  Embedding data not found")
            return

        tsne = self.dm.embeddings_tsne
        labels = self.dm.embeddings_labels

        fig, ax = plt.subplots(figsize=(10, 8))

        # For multi-label, color by dominant class
        if labels.ndim == 2:
            dominant = labels.argmax(axis=1)
        else:
            dominant = labels.astype(int)

        unique_classes = np.unique(dominant)
        for cls_id in unique_classes:
            mask = dominant == cls_id
            name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            color = self.colorizer.get_class_color(cls_id)
            ax.scatter(tsne[mask, 0], tsne[mask, 1], s=8, alpha=0.6,
                      color=color, label=name)

        # Highlight current model if we can find it
        # (embedding indices correspond to dataset order, but might be a subset)

        ax.set_title("t-SNE Embeddings (colored by dominant class)")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        if len(unique_classes) <= 15:
            ax.legend(fontsize=6, markerscale=2, loc="best")
        fig.tight_layout()
        if not block:
            plt.show(block=False)
            plt.pause(0.5)
        print("\n  Embeddings popup shown")

    # ─── Key callbacks ─────────────────────────────────────────────────────

    def _key_toggle_view(self, vis):
        self.view_mode = (self.view_mode + 1) % 4
        # Auto-show legend when entering a colored view mode
        if self.view_mode != self.VIEW_PLAIN and not self.show_legend:
            self.show_legend = True
        self._refresh_display()
        return False

    def _key_next_model(self, vis):
        if self.sorted_indices:
            self.nav_pos = (self.nav_pos + 1) % len(self.sorted_indices)
            self._refresh_display()
        return False

    def _key_prev_model(self, vis):
        if self.sorted_indices:
            self.nav_pos = (self.nav_pos - 1) % len(self.sorted_indices)
            self._refresh_display()
        return False

    def _key_sort_worst(self, vis):
        self.sort_mode = "worst"
        self.sorted_indices = self.dm.get_sorted_indices("worst")
        self.nav_pos = 0
        print(f"\n  Sort: worst accuracy first")
        self._refresh_display()
        return False

    def _key_sort_best(self, vis):
        self.sort_mode = "best"
        self.sorted_indices = self.dm.get_sorted_indices("best")
        self.nav_pos = 0
        print(f"\n  Sort: best accuracy first")
        self._refresh_display()
        return False

    def _key_sort_random(self, vis):
        self.sort_mode = "random"
        self.sorted_indices = self.dm.get_sorted_indices("random")
        self.nav_pos = 0
        print(f"\n  Sort: random shuffle")
        self._refresh_display()
        return False

    def _key_metrics(self, vis):
        self._show_metrics_popup()
        return False

    def _key_confusion(self, vis):
        self._show_confusion_popup()
        return False

    def _key_embeddings(self, vis):
        self._show_embeddings_popup()
        return False

    def _key_info(self, vis):
        self._print_model_info()
        return False

    def _key_legend(self, vis):
        self.show_legend = not self.show_legend
        if self.show_legend:
            self._refresh_display()
            print("\n  Legend shown")
        else:
            # Remove 3D legend geometries
            for geom in self._legend_geometries:
                self.vis.remove_geometry(geom, reset_bounding_box=False)
            self._legend_geometries.clear()
            if self._legend_fig is not None:
                plt.close(self._legend_fig)
                self._legend_fig = None
            print("\n  Legend hidden")
        return False

    def _key_toggle_labels(self, vis):
        self.show_labels = not self.show_labels
        if self.show_labels:
            print("\n  Face labels ON")
            self._refresh_display()
        else:
            # Remove label markers
            for geom in self._label_geometries:
                self.vis.remove_geometry(geom, reset_bounding_box=False)
            self._label_geometries.clear()
            if self._labels_fig is not None:
                plt.close(self._labels_fig)
                self._labels_fig = None
            print("\n  Face labels OFF")
        return False

    def _key_screenshot(self, vis):
        model_id = self._get_current_model_id()
        filename = f"screenshot_{model_id}_{self.VIEW_NAMES[self.view_mode]}.png"
        vis.capture_screen_image(filename)
        print(f"\n  Screenshot saved: {filename}")
        return False

    def _key_reset_camera(self, vis):
        vis.reset_view_point(True)
        return False

    def _key_help(self, vis):
        print("\n" + "=" * 60)
        print("BrepMFR Viewer - Keyboard Controls")
        print("=" * 60)
        print("  T/TAB       Cycle: Plain -> GT -> Predicted -> Comparison")
        print("  D/RIGHT     Next model")
        print("  A/LEFT      Previous model")
        print("  1           Sort: worst accuracy first")
        print("  2           Sort: best accuracy first")
        print("  3           Sort: random shuffle")
        print("  M           Metrics chart popup")
        print("  N           Confusion matrix popup")
        print("  E           Embeddings plot popup")
        print("  I           Print model info")
        print("  L           Toggle class legend (3D swatches + sidebar panel)")
        print("  F           Toggle face labels (predicted feature name at each face)")
        print("  S           Screenshot")
        print("  R           Reset camera")
        print("  H           This help")
        print("  ESC/Q       Exit")
        print("=" * 60)
        return False

    def _key_exit(self, vis):
        self._close_all_plots()
        vis.destroy_window()
        self._window_destroyed = True
        return True

    def _close_all_plots(self):
        """Close all matplotlib figures to prevent Tkinter errors on exit."""
        self._legend_fig = None
        self._labels_fig = None
        if HAS_MATPLOTLIB:
            plt.close('all')

    def _register_keys(self):
        """Register all keyboard callbacks."""
        v = self.vis

        # T/TAB to cycle views
        v.register_key_callback(ord('T'), self._key_toggle_view)
        v.register_key_callback(ord('t'), self._key_toggle_view)
        v.register_key_callback(9, self._key_toggle_view)  # TAB

        # D/RIGHT, A/LEFT for navigation
        v.register_key_callback(ord('D'), self._key_next_model)
        v.register_key_callback(ord('d'), self._key_next_model)
        v.register_key_callback(262, self._key_next_model)  # RIGHT arrow
        v.register_key_callback(ord('A'), self._key_prev_model)
        v.register_key_callback(ord('a'), self._key_prev_model)
        v.register_key_callback(263, self._key_prev_model)  # LEFT arrow

        # 1/2/3 for sort modes
        v.register_key_callback(ord('1'), self._key_sort_worst)
        v.register_key_callback(ord('2'), self._key_sort_best)
        v.register_key_callback(ord('3'), self._key_sort_random)

        # Analysis popups
        v.register_key_callback(ord('M'), self._key_metrics)
        v.register_key_callback(ord('m'), self._key_metrics)
        v.register_key_callback(ord('N'), self._key_confusion)
        v.register_key_callback(ord('n'), self._key_confusion)
        v.register_key_callback(ord('E'), self._key_embeddings)
        v.register_key_callback(ord('e'), self._key_embeddings)

        # Info/Legend/Labels
        v.register_key_callback(ord('I'), self._key_info)
        v.register_key_callback(ord('i'), self._key_info)
        v.register_key_callback(ord('L'), self._key_legend)
        v.register_key_callback(ord('l'), self._key_legend)
        v.register_key_callback(ord('F'), self._key_toggle_labels)
        v.register_key_callback(ord('f'), self._key_toggle_labels)

        # Screenshot/Reset/Help/Exit
        v.register_key_callback(ord('S'), self._key_screenshot)
        v.register_key_callback(ord('s'), self._key_screenshot)
        v.register_key_callback(ord('R'), self._key_reset_camera)
        v.register_key_callback(ord('r'), self._key_reset_camera)
        v.register_key_callback(ord('H'), self._key_help)
        v.register_key_callback(ord('h'), self._key_help)
        v.register_key_callback(256, self._key_exit)  # ESC
        v.register_key_callback(ord('Q'), self._key_exit)
        v.register_key_callback(ord('q'), self._key_exit)

    def run_browse(self, split: str, sort_mode: str, model_id: str = None):
        """Run the interactive browse viewer."""
        self.dm.load_split(split)
        self.dm.load_predictions()
        self.sort_mode = sort_mode

        # If specific model requested, jump to it
        if model_id:
            if model_id in self.dm.model_id_to_idx:
                self.sorted_indices = [self.dm.model_id_to_idx[model_id]]
                print(f"Viewing single model: {model_id}")
            else:
                print(f"Warning: model_id '{model_id}' not found in {split} split, showing all")
                self.sorted_indices = self.dm.get_sorted_indices(sort_mode)
        else:
            self.sorted_indices = self.dm.get_sorted_indices(sort_mode)

        self.nav_pos = 0
        self._launch_viewer()

    def run_predictions(self, sort_mode: str):
        """Run viewer focused on predictions (loads test split by default)."""
        if not self.dm.load_predictions():
            print("Error: predictions.json required for predictions mode")
            print("Run: python brepformer/analyze.py --mode predictions ...")
            sys.exit(1)

        # Try to determine which split the predictions came from
        for split in ["test", "val", "train"]:
            pkl_path = self.dm.processed_dir / f"{split}.pkl"
            if pkl_path.exists():
                self.dm.load_split(split)
                # Check if predictions match
                pred_ids = {p["model_id"] for p in self.dm.predictions.get("predictions", [])}
                sample_ids = {s["model_id"] for s in self.dm.samples}
                if pred_ids & sample_ids:
                    print(f"Matched predictions to {split} split ({len(pred_ids & sample_ids)} models)")
                    break

        self.sort_mode = sort_mode
        self.sorted_indices = self.dm.get_sorted_indices(sort_mode)
        self.nav_pos = 0
        self.view_mode = self.VIEW_PREDICTED
        self.show_legend = True  # Show legend by default in predictions mode
        self._launch_viewer()

    def run_analysis(self, show_metrics=False, show_confusion=False, show_embeddings=False):
        """Run analysis-only mode (matplotlib popups, no 3D viewer)."""
        if not HAS_MATPLOTLIB:
            print("Error: matplotlib required for analysis mode")
            sys.exit(1)

        shown = False
        if show_metrics:
            if self.dm.load_metrics():
                self._show_metrics_popup(block=True)
                shown = True
            else:
                print("per_class_metrics.json not found")

        if show_confusion:
            if self.dm.predictions is None:
                self.dm.load_predictions()
            if self.dm.load_confusion_matrices():
                self._show_confusion_popup(block=True)
                shown = True
            else:
                print("Confusion matrix data not found")

        if show_embeddings:
            if self.dm.load_embeddings():
                self._show_embeddings_popup(block=True)
                shown = True
            else:
                print("Embedding data not found")

        if not shown:
            print("No analysis data found. Run analyze.py first.")
            return

        print("\nClose plot windows to exit.")
        plt.show(block=True)

    def _launch_viewer(self):
        """Create the Open3D window and start the viewer loop."""
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        try:
            success = self.vis.create_window(
                window_name="BrepMFR Viewer", width=920, height=720,
                left=10, top=50
            )
            if not success:
                print("\nError: Failed to create visualization window.")
                print("Make sure a display server is available (WSLg, VcXsrv, etc.)")
                return
        except Exception as e:
            print(f"\nError creating window: {e}")
            return

        self._register_keys()

        # Set render options
        render_opt = self.vis.get_render_option()
        if render_opt:
            render_opt.background_color = np.array([0.1, 0.1, 0.1])
            render_opt.point_size = 5.0

        # Display initial model
        self._refresh_display()

        # Print help
        self._key_help(self.vis)

        # Run loop
        self.vis.run()
        if not self._window_destroyed:
            self._close_all_plots()
            self.vis.destroy_window()
        print("\nViewer closed.")


def run_analysis_only(args):
    """Run analysis mode without 3D viewer."""
    dm = MTFRCADDataManager(
        processed_dir=args.processed_dir,
        data_dir=args.data_dir,
        analysis_dir=args.analysis_dir,
    )
    colorizer = BrepColorizer()
    viewer = BrepMFRViewer(dm, colorizer)

    show_all = not args.metrics and not args.confusion and not args.embeddings
    viewer.run_analysis(
        show_metrics=args.metrics or show_all,
        show_confusion=args.confusion or show_all,
        show_embeddings=args.embeddings or show_all,
    )


def main():
    parser = argparse.ArgumentParser(
        description="BrepMFR Viewer - Interactive MTFRCAD B-rep visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  browse       Browse preprocessed data with GT labels and predictions
  predictions  Browse models sorted by prediction accuracy (Jaccard)
  analysis     Show analysis plots (metrics, confusion, embeddings)

Examples:
  python viewer.py --mode browse --split test --sort index
  python viewer.py --mode predictions --sort worst
  python viewer.py --mode analysis --metrics
  python viewer.py --mode analysis --confusion
  python viewer.py --mode analysis --embeddings
  python viewer.py --model_id 20240125_003844_7317
        """,
    )

    parser.add_argument("--mode", choices=["browse", "predictions", "analysis"],
                        default="browse", help="Viewer mode (default: browse)")
    parser.add_argument("--split", choices=["train", "val", "test"],
                        default="test", help="Data split (default: test)")
    parser.add_argument("--sort", choices=["index", "best", "worst", "random"],
                        default="index", help="Sort order (default: index)")
    parser.add_argument("--model_id", type=str, default=None,
                        help="View a specific model by ID")
    parser.add_argument("--data_dir", type=str, default="brepformer/data/mtfrcad",
                        help="Raw MTFRCAD data directory (for per-face labels)")
    parser.add_argument("--processed_dir", type=str,
                        default="brepformer/data/mtfrcad_processed",
                        help="Preprocessed data directory")
    parser.add_argument("--analysis_dir", type=str, default="analysis_results",
                        help="Analysis results directory")

    # Analysis-only flags
    parser.add_argument("--metrics", action="store_true",
                        help="Show per-class metrics chart (analysis mode)")
    parser.add_argument("--confusion", action="store_true",
                        help="Show confusion matrix heatmap (analysis mode)")
    parser.add_argument("--embeddings", action="store_true",
                        help="Show t-SNE embeddings scatter (analysis mode)")

    args = parser.parse_args()

    dm = MTFRCADDataManager(
        processed_dir=args.processed_dir,
        data_dir=args.data_dir,
        analysis_dir=args.analysis_dir,
    )
    colorizer = BrepColorizer()
    viewer = BrepMFRViewer(dm, colorizer)

    if args.mode == "analysis":
        run_analysis_only(args)
    elif args.mode == "predictions":
        viewer.run_predictions(sort_mode=args.sort)
    else:
        # browse mode
        viewer.run_browse(
            split=args.split,
            sort_mode=args.sort,
            model_id=args.model_id,
        )


if __name__ == "__main__":
    main()
