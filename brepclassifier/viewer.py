#!/usr/bin/env python3
"""Pipe Fitting Viewer - Interactive 3D visualization for pipe fitting classification.

Browse preprocessed pipe fitting data with ground-truth labels, view model
predictions, display per-class metrics, confusion matrices, and t-SNE embeddings.

Modes:
    browse       -- Browse preprocessed data with GT class coloring
    predictions  -- Browse models sorted by prediction correctness
    analysis     -- Show analysis plots (metrics, confusion, embeddings)

Keyboard Controls:
    T/TAB       Cycle views: Plain -> GT Class -> Predicted Class -> Comparison
    D/RIGHT     Next model
    A/LEFT      Previous model
    1           Sort: worst first (incorrect predictions)
    2           Sort: best first (correct predictions)
    3           Sort: random shuffle
    M           Metrics chart popup
    N           Confusion matrix popup
    E           Embeddings plot popup
    I           Print model info
    L           Toggle class legend
    S           Screenshot
    R           Reset camera
    H           Help
    ESC/Q       Exit
"""

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
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


CLASS_NAMES = [
    "Elbow - Weld Fitting",
    "Elbow - Pipe End Fitting",
    "Elbow - Socket Fitting",
    "Tee - Weld Fitting",
    "Tee - Pipe End Fitting",
    "Tee - Socket Fitting",
    "Elbow - Miscellaneous",
    "Tee - Miscellaneous",
]

CLASS_NAMES_SHORT = [
    "Elbow-WF", "Elbow-PEF", "Elbow-SF",
    "Tee-WF", "Tee-PEF", "Tee-SF",
    "Elbow-Misc", "Tee-Misc",
]


class PipeFittingDataManager:
    """Data loading and indexing for pipe fitting preprocessed data."""

    def __init__(self, processed_dir: str, analysis_dir: str = "analysis_results/pipe_classifier"):
        self.processed_dir = Path(processed_dir)
        self.analysis_dir = Path(analysis_dir)

        self.samples: List[dict] = []
        self.model_id_to_idx: Dict[str, int] = {}
        self.predictions: Optional[dict] = None
        self.per_class_metrics: Optional[list] = None
        self.embeddings_tsne: Optional[np.ndarray] = None
        self.embeddings_labels: Optional[np.ndarray] = None
        self.confusion_matrix: Optional[np.ndarray] = None

    def load_split(self, split: str):
        pkl_path = self.processed_dir / f"{split}.pkl"
        if not pkl_path.exists():
            print(f"Error: {pkl_path} not found")
            sys.exit(1)

        print(f"Loading {split} split from {pkl_path}...")
        with open(pkl_path, "rb") as f:
            self.samples = pickle.load(f)
        print(f"Loaded {len(self.samples)} samples")

        self.model_id_to_idx = {s["model_id"]: i for i, s in enumerate(self.samples)}

    def load_predictions(self) -> bool:
        pred_path = self.analysis_dir / "predictions.json"
        if not pred_path.exists():
            print(f"Warning: {pred_path} not found")
            return False
        with open(pred_path) as f:
            self.predictions = json.load(f)
        print(f"Loaded predictions for {len(self.predictions.get('predictions', []))} samples")
        return True

    def load_metrics(self) -> bool:
        metrics_path = self.analysis_dir / "per_class_metrics.json"
        if not metrics_path.exists():
            return False
        with open(metrics_path) as f:
            self.per_class_metrics = json.load(f)
        return True

    def load_embeddings(self) -> bool:
        tsne_path = self.analysis_dir / "embeddings_tsne.npy"
        labels_path = self.analysis_dir / "labels.npy"
        if not tsne_path.exists() or not labels_path.exists():
            return False
        self.embeddings_tsne = np.load(tsne_path)
        self.embeddings_labels = np.load(labels_path)
        return True

    def load_confusion_matrix(self) -> bool:
        cm_path = self.analysis_dir / "confusion_matrix.npy"
        if not cm_path.exists():
            return False
        self.confusion_matrix = np.load(cm_path)
        return True

    def get_prediction_for_model(self, model_id: str) -> Optional[dict]:
        if self.predictions is None:
            return None
        for p in self.predictions.get("predictions", []):
            if p["model_id"] == model_id:
                return p
        return None

    def get_sorted_indices(self, sort_mode: str) -> List[int]:
        n = len(self.samples)
        if sort_mode == "index":
            return list(range(n))
        if sort_mode == "random":
            indices = list(range(n))
            random.shuffle(indices)
            return indices

        if self.predictions is None:
            return list(range(n))

        pred_map = {}
        for p in self.predictions.get("predictions", []):
            pred_map[p["model_id"]] = p.get("correct", False)

        scored = []
        for i, s in enumerate(self.samples):
            correct = pred_map.get(s["model_id"], True)
            scored.append((i, 1 if correct else 0))

        if sort_mode == "worst":
            scored.sort(key=lambda x: x[1])
        elif sort_mode == "best":
            scored.sort(key=lambda x: -x[1])

        return [idx for idx, _ in scored]


class PipeFittingColorizer:
    """Color palette for 8 pipe fitting classes."""

    def __init__(self):
        self.palette = self._build_palette(8)

    def _build_palette(self, n: int) -> np.ndarray:
        if HAS_MATPLOTLIB:
            cmap = plt.colormaps.get_cmap("tab10")
            return np.array([cmap(i / 10.0)[:3] for i in range(n)])
        else:
            import colorsys
            return np.array([colorsys.hsv_to_rgb(i / n, 0.8, 0.9) for i in range(n)])

    def get_class_color(self, class_idx: int) -> np.ndarray:
        return self.palette[class_idx % len(self.palette)]

    def color_model_by_class(self, n_points: int, class_idx: int) -> np.ndarray:
        """Color entire model by a single class."""
        color = self.get_class_color(class_idx)
        return np.tile(color, (n_points, 1))

    def color_model_comparison(self, n_points: int, correct: bool) -> np.ndarray:
        """Green if correct, red if incorrect."""
        color = np.array([0.2, 0.8, 0.2]) if correct else np.array([0.8, 0.2, 0.2])
        return np.tile(color, (n_points, 1))

    def plain_color(self, n_points: int) -> np.ndarray:
        return np.full((n_points, 3), 0.6)


class PipeFittingViewer:
    """Open3D viewer for pipe fitting classification data."""

    VIEW_PLAIN = 0
    VIEW_GT = 1
    VIEW_PREDICTED = 2
    VIEW_COMPARISON = 3
    VIEW_NAMES = ["Plain", "GT Class", "Predicted Class", "Comparison"]

    def __init__(self, data_manager: PipeFittingDataManager, colorizer: PipeFittingColorizer):
        self.dm = data_manager
        self.colorizer = colorizer
        self.vis: Optional[o3d.visualization.VisualizerWithKeyCallback] = None
        self.current_geometries: List = []
        self.sorted_indices: List[int] = []
        self.nav_pos: int = 0
        self.view_mode: int = self.VIEW_PLAIN
        self.sort_mode: str = "index"
        self._window_destroyed: bool = False

    def _extract_points(self, face_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_faces = face_grid.shape[0]
        all_pts = []
        all_normals = []
        for fi in range(n_faces):
            mask = face_grid[fi, 6, :, :]
            valid = mask > 0.5
            if valid.sum() == 0:
                xyz = face_grid[fi, 0:3, :, :].reshape(3, -1).T
                nrm = face_grid[fi, 3:6, :, :].reshape(3, -1).T
            else:
                xyz = face_grid[fi, 0:3, :, :][:, valid].T
                nrm = face_grid[fi, 3:6, :, :][:, valid].T
            all_pts.append(xyz)
            all_normals.append(nrm)

        if all_pts:
            return np.vstack(all_pts), np.vstack(all_normals)
        return np.zeros((0, 3)), np.zeros((0, 3))

    def _build_point_cloud(self, points, normals, colors):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if normals.shape[0] == points.shape[0]:
            pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
        if colors.shape[0] == points.shape[0]:
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        return pcd

    def _get_current_sample(self):
        if not self.sorted_indices:
            return None
        return self.dm.samples[self.sorted_indices[self.nav_pos]]

    def _refresh_display(self):
        if self.vis is None:
            return

        for geom in self.current_geometries:
            self.vis.remove_geometry(geom, reset_bounding_box=False)
        self.current_geometries.clear()

        sample = self._get_current_sample()
        if sample is None:
            return

        face_grid = sample["face_grid"]
        model_id = sample["model_id"]
        gt_class = int(sample["label"])
        points, normals = self._extract_points(face_grid)

        if points.shape[0] == 0:
            return

        pred = self.dm.get_prediction_for_model(model_id)

        if self.view_mode == self.VIEW_COMPARISON:
            # Side by side: GT (left) vs Predicted (right)
            extent = points.max(axis=0) - points.min(axis=0)
            offset = np.array([extent[0] * 1.3, 0, 0])

            gt_colors = self.colorizer.color_model_by_class(points.shape[0], gt_class)
            pcd_gt = self._build_point_cloud(points - offset / 2, normals, gt_colors)
            self.vis.add_geometry(pcd_gt, reset_bounding_box=False)
            self.current_geometries.append(pcd_gt)

            pred_class = pred["predicted_class"] if pred else gt_class
            correct = pred["correct"] if pred else True
            pred_colors = self.colorizer.color_model_comparison(points.shape[0], correct)
            pcd_pred = self._build_point_cloud(points + offset / 2, normals, pred_colors)
            self.vis.add_geometry(pcd_pred, reset_bounding_box=False)
            self.current_geometries.append(pcd_pred)
        else:
            if self.view_mode == self.VIEW_PLAIN:
                colors = self.colorizer.plain_color(points.shape[0])
            elif self.view_mode == self.VIEW_GT:
                colors = self.colorizer.color_model_by_class(points.shape[0], gt_class)
            elif self.view_mode == self.VIEW_PREDICTED:
                pred_class = pred["predicted_class"] if pred else gt_class
                colors = self.colorizer.color_model_by_class(points.shape[0], pred_class)
            else:
                colors = self.colorizer.plain_color(points.shape[0])

            pcd = self._build_point_cloud(points, normals, colors)
            self.vis.add_geometry(pcd, reset_bounding_box=False)
            self.current_geometries.append(pcd)

        if self.current_geometries:
            self.vis.reset_view_point(True)

        self._print_status()

    def _print_status(self):
        sample = self._get_current_sample()
        if sample is None:
            return

        model_id = sample["model_id"]
        gt_class = int(sample["label"])
        gt_name = CLASS_NAMES_SHORT[gt_class] if gt_class < len(CLASS_NAMES_SHORT) else str(gt_class)
        n_faces = sample["num_faces"]
        view_name = self.VIEW_NAMES[self.view_mode]
        pos = self.nav_pos + 1
        total = len(self.sorted_indices)

        pred = self.dm.get_prediction_for_model(model_id)
        pred_str = ""
        if pred:
            pc = pred.get("predicted_class", -1)
            pred_name = CLASS_NAMES_SHORT[pc] if 0 <= pc < len(CLASS_NAMES_SHORT) else str(pc)
            correct = "OK" if pred.get("correct", False) else "WRONG"
            pred_str = f" | Pred: {pred_name} [{correct}]"

        print(f"\r[{pos}/{total}] {model_id} | {n_faces} faces | GT: {gt_name}{pred_str} | {view_name} | sort={self.sort_mode}    ", end="", flush=True)

    def _print_model_info(self):
        sample = self._get_current_sample()
        if sample is None:
            return

        model_id = sample["model_id"]
        gt_class = int(sample["label"])

        print(f"\n{'='*60}")
        print(f"Model: {model_id}")
        print(f"Faces: {sample['num_faces']}, Edges: {sample['num_edges']}")
        print(f"GT class: {gt_class} ({CLASS_NAMES[gt_class]})")

        pred = self.dm.get_prediction_for_model(model_id)
        if pred:
            print(f"Predicted: {pred.get('predicted_class')} ({pred.get('predicted_name', '')})")
            print(f"Confidence: {pred.get('confidence', 'N/A')}")
            print(f"Correct: {pred.get('correct', 'N/A')}")
        print(f"{'='*60}")

    def _show_class_legend(self):
        print(f"\n{'='*60}")
        print("CLASS LEGEND (8 pipe fitting classes)")
        print(f"{'='*60}")
        for i, name in enumerate(CLASS_NAMES):
            color = self.colorizer.get_class_color(i)
            r, g, b = int(color[0]*255), int(color[1]*255), int(color[2]*255)
            print(f"  {i}: {name:<30s}  RGB({r:3d},{g:3d},{b:3d})")
        print(f"{'='*60}")

    def _show_metrics_popup(self, block=False):
        if not HAS_MATPLOTLIB or not self.dm.load_metrics():
            print("\n  Metrics data not available")
            return

        metrics = self.dm.per_class_metrics
        classes = [m["class"] for m in metrics]
        f1s = [m["f1"] for m in metrics]
        names = [CLASS_NAMES_SHORT[c] if c < len(CLASS_NAMES_SHORT) else str(c) for c in classes]

        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(classes))
        width = 0.25
        ax.bar(x - width, [m["precision"] for m in metrics], width, label="Precision")
        ax.bar(x, [m["recall"] for m in metrics], width, label="Recall")
        ax.bar(x + width, f1s, width, label="F1")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_title("Per-Class Metrics")
        ax.legend()
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        if not block:
            plt.show(block=False)
            plt.pause(0.5)

    def _show_confusion_popup(self, block=False):
        if not HAS_MATPLOTLIB or not self.dm.load_confusion_matrix():
            print("\n  Confusion matrix not available")
            return

        cm = self.dm.confusion_matrix
        names = CLASS_NAMES_SHORT[:cm.shape[0]]

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticklabels(names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        if not block:
            plt.show(block=False)
            plt.pause(0.5)

    def _show_embeddings_popup(self, block=False):
        if not HAS_MATPLOTLIB or not self.dm.load_embeddings():
            print("\n  Embedding data not available")
            return

        tsne = self.dm.embeddings_tsne
        labels = self.dm.embeddings_labels

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.colormaps.get_cmap("tab10")(np.linspace(0, 1, 10))[:8]

        for cls_id in range(8):
            mask = labels == cls_id
            if mask.any():
                ax.scatter(tsne[mask, 0], tsne[mask, 1], s=10, alpha=0.7,
                          color=colors[cls_id], label=CLASS_NAMES_SHORT[cls_id])

        ax.set_title("t-SNE Embeddings")
        ax.legend(fontsize=8, markerscale=2)
        fig.tight_layout()
        if not block:
            plt.show(block=False)
            plt.pause(0.5)

    # ── Key callbacks ──
    def _key_toggle_view(self, vis):
        self.view_mode = (self.view_mode + 1) % 4
        self._refresh_display()
        return False

    def _key_next(self, vis):
        if self.sorted_indices:
            self.nav_pos = (self.nav_pos + 1) % len(self.sorted_indices)
            self._refresh_display()
        return False

    def _key_prev(self, vis):
        if self.sorted_indices:
            self.nav_pos = (self.nav_pos - 1) % len(self.sorted_indices)
            self._refresh_display()
        return False

    def _key_sort_worst(self, vis):
        self.sort_mode = "worst"
        self.sorted_indices = self.dm.get_sorted_indices("worst")
        self.nav_pos = 0
        self._refresh_display()
        return False

    def _key_sort_best(self, vis):
        self.sort_mode = "best"
        self.sorted_indices = self.dm.get_sorted_indices("best")
        self.nav_pos = 0
        self._refresh_display()
        return False

    def _key_sort_random(self, vis):
        self.sort_mode = "random"
        self.sorted_indices = self.dm.get_sorted_indices("random")
        self.nav_pos = 0
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
        self._show_class_legend()
        return False

    def _key_screenshot(self, vis):
        model_id = self._get_current_sample()["model_id"] if self._get_current_sample() else "unknown"
        filename = f"screenshot_{model_id}_{self.VIEW_NAMES[self.view_mode]}.png"
        vis.capture_screen_image(filename)
        print(f"\n  Screenshot: {filename}")
        return False

    def _key_reset(self, vis):
        vis.reset_view_point(True)
        return False

    def _key_help(self, vis):
        print(f"\n{'='*60}")
        print("Pipe Fitting Viewer - Keyboard Controls")
        print(f"{'='*60}")
        print("  T/TAB       Cycle: Plain -> GT -> Predicted -> Comparison")
        print("  D/RIGHT     Next model")
        print("  A/LEFT      Previous model")
        print("  1           Sort: worst first")
        print("  2           Sort: best first")
        print("  3           Sort: random")
        print("  M           Metrics popup")
        print("  N           Confusion matrix popup")
        print("  E           Embeddings popup")
        print("  I           Model info")
        print("  L           Class legend")
        print("  S           Screenshot")
        print("  R           Reset camera")
        print("  H           Help")
        print("  ESC/Q       Exit")
        print(f"{'='*60}")
        return False

    def _key_exit(self, vis):
        if HAS_MATPLOTLIB:
            plt.close('all')
        vis.destroy_window()
        self._window_destroyed = True
        return True

    def _register_keys(self):
        v = self.vis
        for key in [ord('T'), ord('t'), 9]:
            v.register_key_callback(key, self._key_toggle_view)
        for key in [ord('D'), ord('d'), 262]:
            v.register_key_callback(key, self._key_next)
        for key in [ord('A'), ord('a'), 263]:
            v.register_key_callback(key, self._key_prev)
        v.register_key_callback(ord('1'), self._key_sort_worst)
        v.register_key_callback(ord('2'), self._key_sort_best)
        v.register_key_callback(ord('3'), self._key_sort_random)
        for key in [ord('M'), ord('m')]:
            v.register_key_callback(key, self._key_metrics)
        for key in [ord('N'), ord('n')]:
            v.register_key_callback(key, self._key_confusion)
        for key in [ord('E'), ord('e')]:
            v.register_key_callback(key, self._key_embeddings)
        for key in [ord('I'), ord('i')]:
            v.register_key_callback(key, self._key_info)
        for key in [ord('L'), ord('l')]:
            v.register_key_callback(key, self._key_legend)
        for key in [ord('S'), ord('s')]:
            v.register_key_callback(key, self._key_screenshot)
        for key in [ord('R'), ord('r')]:
            v.register_key_callback(key, self._key_reset)
        for key in [ord('H'), ord('h')]:
            v.register_key_callback(key, self._key_help)
        v.register_key_callback(256, self._key_exit)
        for key in [ord('Q'), ord('q')]:
            v.register_key_callback(key, self._key_exit)

    def run_browse(self, split: str, sort_mode: str, model_id: str = None):
        self.dm.load_split(split)
        self.dm.load_predictions()
        self.sort_mode = sort_mode

        if model_id and model_id in self.dm.model_id_to_idx:
            self.sorted_indices = [self.dm.model_id_to_idx[model_id]]
        else:
            self.sorted_indices = self.dm.get_sorted_indices(sort_mode)

        self.nav_pos = 0
        self._launch_viewer()

    def run_predictions(self, sort_mode: str):
        if not self.dm.load_predictions():
            print("Error: predictions.json required")
            sys.exit(1)

        for split in ["test", "val", "train"]:
            pkl_path = self.dm.processed_dir / f"{split}.pkl"
            if pkl_path.exists():
                self.dm.load_split(split)
                pred_ids = {p["model_id"] for p in self.dm.predictions.get("predictions", [])}
                sample_ids = {s["model_id"] for s in self.dm.samples}
                if pred_ids & sample_ids:
                    break

        self.sort_mode = sort_mode
        self.sorted_indices = self.dm.get_sorted_indices(sort_mode)
        self.nav_pos = 0
        self.view_mode = self.VIEW_PREDICTED
        self._launch_viewer()

    def run_analysis(self, show_metrics=False, show_confusion=False, show_embeddings=False):
        if not HAS_MATPLOTLIB:
            print("Error: matplotlib required")
            sys.exit(1)

        shown = False
        if show_metrics:
            if self.dm.load_metrics():
                self._show_metrics_popup(block=True)
                shown = True
        if show_confusion:
            if self.dm.load_confusion_matrix():
                self._show_confusion_popup(block=True)
                shown = True
        if show_embeddings:
            if self.dm.load_embeddings():
                self._show_embeddings_popup(block=True)
                shown = True

        if shown:
            plt.show(block=True)
        else:
            print("No analysis data found.")

    def _launch_viewer(self):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        try:
            self.vis.create_window(window_name="Pipe Fitting Viewer", width=1280, height=720)
        except Exception as e:
            print(f"\nError creating window: {e}")
            return

        self._register_keys()

        render_opt = self.vis.get_render_option()
        if render_opt:
            render_opt.background_color = np.array([0.1, 0.1, 0.1])
            render_opt.point_size = 5.0

        self._refresh_display()
        self._key_help(self.vis)
        self.vis.run()

        if not self._window_destroyed:
            if HAS_MATPLOTLIB:
                plt.close('all')
            self.vis.destroy_window()
        print("\nViewer closed.")


def main():
    parser = argparse.ArgumentParser(description="Pipe Fitting Viewer")

    parser.add_argument("--mode", choices=["browse", "predictions", "analysis"],
                        default="browse")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--sort", choices=["index", "best", "worst", "random"],
                        default="index")
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--processed_dir", type=str,
                        default="brepclassifier/data/ssdata1_processed")
    parser.add_argument("--analysis_dir", type=str,
                        default="analysis_results/pipe_classifier")
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument("--confusion", action="store_true")
    parser.add_argument("--embeddings", action="store_true")

    args = parser.parse_args()

    dm = PipeFittingDataManager(
        processed_dir=args.processed_dir,
        analysis_dir=args.analysis_dir,
    )
    colorizer = PipeFittingColorizer()
    viewer = PipeFittingViewer(dm, colorizer)

    if args.mode == "analysis":
        show_all = not args.metrics and not args.confusion and not args.embeddings
        viewer.run_analysis(
            show_metrics=args.metrics or show_all,
            show_confusion=args.confusion or show_all,
            show_embeddings=args.embeddings or show_all,
        )
    elif args.mode == "predictions":
        viewer.run_predictions(sort_mode=args.sort)
    else:
        viewer.run_browse(split=args.split, sort_mode=args.sort, model_id=args.model_id)


if __name__ == "__main__":
    main()
