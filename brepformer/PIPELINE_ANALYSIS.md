# BrepFormer Pipeline Analysis

Thorough analysis of the BrepFormer training, preprocessing, and inference pipeline for potential issues affecting model quality and face segmentation accuracy.

---

## Critical Issues

### 1. Edge Attribute Index Mismatch in GraphAttnBias (CRITICAL) -- FIXED

**File:** `models/layers/embedding.py`, lines 414-417

In `GraphAttnBias.forward()`, edge attributes were extracted at indices that did not match the actual data layout.

The actual edge attribute layout from `data/step_to_graph.py` `compute_edge_attributes()`:

| Index | Attribute | Description |
|-------|-----------|-------------|
| 0 | Curve type | line/circle/ellipse/bspline |
| 1 | Edge length | arc length |
| 2 | Midpoint X | position, NOT angle |
| 3 | Midpoint Y | position, NOT convexity |
| 4 | Midpoint Z | position |
| 5-7 | Tangent X/Y/Z | normalized tangent vector |
| **8** | **Dihedral angle** | **angle between adjacent face normals** |
| **9** | **Convexity type** | **0=convex, 1=concave, 2=smooth** |
| 10-12 | Face1 normal | at edge midpoint |
| 13 | is_closed | flag |
| 14 | Curvature | at midpoint |

**Fix applied:** Changed `edge_ang = edge_attr[:, :, 2:3]` to `edge_attr[:, :, 8:9]` and `edge_convexity = edge_attr[:, :, 3]` to `edge_attr[:, :, 9]`.

---

### 2. Scatter-Add Double-Counting of Edges -- FIXED

**File:** `models/layers/embedding.py`, lines 457-458

Edge features were scattered in both forward and backward directions, but `build_face_adjacency()` already creates bidirectional edges. Each undirected edge contributed **4x** instead of 2x.

**Fix applied:** Removed the backward scatter line.

---

### 13. Dihedral Angle Computation Silently Fails (CRITICAL) -- FIXED

**File:** `data/step_to_graph.py`, lines 275, 138, 351

Two compounding bugs caused `compute_edge_attributes()` to produce **all-zero** dihedral angles, convexity values, and face normals for every edge:

**Bug A — Wrong pythonOCC API suffix:** `BRep_Tool.Surface_s(face)` and `BRep_Tool.IsClosed_s(edge, face)` do not exist in pythonOCC 7.9. The `_s` static-method suffix is from a different pythonOCC version. Every call threw `AttributeError`, caught silently by `except Exception: pass`, returning `None`.

**Bug B — numpy float32 type rejection:** After fixing Bug A, `_face_normal_at_point(face, attrs[2], attrs[3], attrs[4])` still failed because `attrs` is a `np.float32` array. The `gp_Pnt(x, y, z)` constructor in pythonOCC 7.9 accepts `float` and `np.float64` but rejects `np.float32`, throwing `TypeError` — again caught silently.

**Impact:** The most discriminative edge features — dihedral angle and convexity — were always zero during inference. Combined with Issue #1 (wrong indices in the model), the model had **no access** to edge convexity information whatsoever.

**Fixes applied:**
- Replaced `BRep_Tool.Surface_s` → `BRep_Tool.Surface` and `BRep_Tool.IsClosed_s` → `BRep_Tool.IsClosed` (3 occurrences).
- Added `float()` cast in `_face_normal_at_point`: `gp_Pnt(float(x), float(y), float(z))`.

**Verification:** After fix, 118/120 edges produce non-zero dihedral angles, all 3 convexity types (0=convex, 1=concave, 2=smooth) observed, and 120/120 edges have face normals.

---

## Significant Issues

### 3. Double FFN Invocation with Shared Weights -- FIXED

**File:** `models/layers/encoder_layer.py`, lines 103-113

The `GraphEncoderLayer` applied the FFN block twice using the **same** `self.norm2` and `self.ffn`.

**Fix applied:** Created separate `self.norm2`/`self.ffn1` and `self.norm3`/`self.ffn2` with independent weights.

---

### 4. Degenerate EdgeConv (Multi-hop Aggregation Broken) -- FIXED

**File:** `models/layers/blocks.py`, lines 312-336

The `EdgeConv` class concatenated each edge feature with itself; `edge_index` was never used.

**Fix applied:** Implemented scatter-mean neighbor aggregation — edge features are pooled to nodes via source endpoint, then gathered from destination node, concatenated with the original feature, and projected.

**Note:** `EdgeConv` is instantiated in `GraphAttnBias.__init__` but not yet called in `forward()`. The class is now correct if integrated into the multi-hop path encoding.

---

### 5. D2 and Angle Descriptors Are Degenerate One-Hot Vectors -- FIXED

**File:** `data/preprocessing.py`, lines 95-187

Both descriptors created one-hot histograms from single measurements (1/64 bins non-zero).

**Fix applied:** Two modes:
- **UV-grid mode** (when `face_grids` available): Samples 32 random point/normal pairs and builds proper distribution histograms.
- **Centroid-only mode** (fallback): Uses Gaussian-kernel smoothing around the measurement, spreading signal across neighboring bins.

`preprocess.py` now passes `face_grids` when `--compute_descriptors` is enabled.

---

### 6. RoPE Applied to Arbitrarily-Ordered Graph Nodes -- FIXED

**File:** `models/layers/attention.py`, line 190

RoPE encodes sequential position, but B-rep faces have no natural sequential ordering. The face order from `TopExp_Explorer` follows internal B-rep data structure order, which is arbitrary and inconsistent across models. The spatial position encoding (shortest paths) already handles structural relationships.

**Fix applied:** Added `use_rope` config flag (default `False`). RoPE is now disabled by default. Can be re-enabled with `--use_rope` for experimentation.

---

## Moderate Issues

### 7. Face Segmentation Loss Without Class Weighting -- FIXED

**File:** `models/brep_classifier.py`, line 72

No class weights in `CrossEntropyLoss`. With 27 MFTRCAD classes, severe class imbalance causes poor recall on rare features.

**Fix applied:**
- Added `face_class_weights` config option.
- `train_preprocessed.py` computes inverse-frequency weights from training data when `--face_segmentation` is enabled.
- `CrossEntropyLoss` receives the computed weights automatically.

---

### 8. Model Classification Loss May Conflict with Face Segmentation -- MITIGATED

**File:** `models/brep_classifier.py`, lines 253-266

```python
loss = self.model_cls_weight * model_loss + self.face_seg_weight * face_loss
```

The `--model_cls_weight` and `--face_seg_weight` CLI arguments already allow tuning the balance. For face-segmentation-focused training, use `--model_cls_weight 0.5 --face_seg_weight 2.0`.

**Status:** Configurable via existing parameters. No code change needed.

---

### 9. Normalization Train/Inference Inconsistency -- VERIFIED OK

- **Inference:** `data/step_to_graph.py` applies `normalize_geometry()` (center + scale to unit sphere)
- **Training:** `preprocess.py` loads graph JSON files without explicit normalization

**Verification:** MFTRCAD JSON graph files are **pre-normalized** (coordinates in [-1, 1], range ~2.0). No mismatch exists.

---

### 10. adam_beta1 = 0.99 (Unusual) -- FIXED

**File:** `configs/config.py`, line 54

Typical Adam uses `beta1=0.9`. The default of `0.99` gave 10x longer gradient memory, potentially slowing convergence.

**Fix applied:** Changed default from `0.99` to `0.9`.

---

### 11. Manual Metric Reset vs TorchMetrics Auto-Reset -- FIXED

**File:** `models/brep_classifier.py`, lines 474-491

`on_train_epoch_end()` and `on_validation_epoch_end()` manually reset metrics. PyTorch Lightning auto-resets torchmetrics objects when logged with `on_epoch=True`. The manual resets could race with PL's internal logging, potentially zeroing metrics before they are recorded.

**Fix applied:** Removed `on_train_epoch_end` and `on_validation_epoch_end` manual reset methods.

---

### 12. Warmup Schedule Interaction with ReduceLROnPlateau

**File:** `models/brep_classifier.py`, lines 442-472

Linear warmup directly modifies `pg["lr"]`. After warmup completes, `ReduceLROnPlateau` takes over. Since ReduceLROnPlateau reads the current `pg["lr"]` each step, the transition is smooth in practice.

**Status:** No change needed. Works correctly.

---

### 14. MFTRCAD JSON Edge Attributes Use Different Layout Than step_to_graph.py

The pre-existing MFTRCAD JSON graph files use a different 15-column edge attribute layout than `step_to_graph.py`:

| Index | MFTRCAD JSON | step_to_graph.py |
|-------|-------------|-----------------|
| 0 | curve_type (int) | curve_type (int) |
| 1 | is_rational (0/1) | edge length |
| 2 | is_seam (0/1) | midpoint X |
| 3 | **length** | midpoint Y |
| 4 | **convexity** | midpoint Z |
| 5-7 | tangent | tangent |
| 8 | dihedral angle | dihedral angle |
| 9-11 | face1 normal | convexity / face1 normal |
| 12-14 | reserved / curvature | face1_nz / is_closed / curvature |

Indices 0, 5-7, and 8 match between formats. Everything else differs. The model's `embedding.py` now reads indices matching `step_to_graph.py` (the richer, intended format).

**Impact:** Training on the existing MFTRCAD JSON pickle data will read incorrect values for edge length (gets is_rational) and convexity (gets midpoint Z). A retrain using data processed through `step_to_graph.py` from STEP files is required for the edge attribute fixes to take effect.

**Status:** Documented. No code change — the `step_to_graph.py` format is the canonical standard going forward.

---

## Information Flow Summary

```
STEP file
  |
  v
step_to_graph.py (extract faces, edges, UV-grids, attributes, normalize)
  |  [FIXED] BRep_Tool API suffix + np.float32 type cast (issue #13)
  v
preprocessing.py (shortest paths, in-degree, D2/angle descriptors)
  |  [FIXED] Multi-sample histograms replace one-hot (issue #5)
  v
collator.py (batch padding, attention masks)
  |
  v
BrepEncoder:
  +-> GraphNodeFeature (SurfaceEncoder CNN + attribute encodings + [CLS] token)
  +-> GraphAttnBias (spatial + D2 + angle + edge features -> attention bias)
  |     [FIXED] Edge attrs now read at correct indices (issue #1)
  |     [FIXED] Edge features scattered once, not twice (issue #2)
  +-> 8x GraphEncoderLayer (RMSNorm + Attention + 2x SwiGLU FFN)
  |     [FIXED] Two independent FFN modules per layer (issue #3)
  |     [FIXED] RoPE disabled by default (issue #6)
  |
  v
GraphPooling ([CLS] token for model-level)
  +-> NonLinearClassifier (4-layer MLP -> 27 model classes)
  +-> FaceSegmentationClassifier (3-layer MLP per face -> 27 face classes)
        [FIXED] Inverse-frequency class weights (issue #7)
```

---

## Fix Summary

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | Edge attribute index mismatch | Critical | **Fixed** |
| 2 | Scatter-add double-counting | Critical | **Fixed** |
| 13 | Dihedral angle silent failure (BRep_Tool API + np.float32) | Critical | **Fixed** |
| 3 | Double FFN shared weights | Significant | **Fixed** |
| 4 | Degenerate EdgeConv | Significant | **Fixed** |
| 5 | Degenerate D2/angle descriptors | Significant | **Fixed** |
| 6 | RoPE on arbitrary node order | Significant | **Fixed** (disabled by default) |
| 7 | No class weights in face loss | Moderate | **Fixed** |
| 8 | Model/face loss weight conflict | Moderate | Mitigated (configurable) |
| 9 | Normalization train/inference mismatch | Moderate | Verified OK |
| 10 | adam_beta1 = 0.99 | Moderate | **Fixed** (changed to 0.9) |
| 11 | Manual metric resets | Moderate | **Fixed** (removed) |
| 12 | Warmup + ReduceLROnPlateau interaction | Low | No change needed |
| 14 | JSON vs step_to_graph.py format mismatch | Info | Documented |

---

## Note on Retraining

Issues 1-7, 10, 11, and 13 affect model weights and/or training dynamics. The existing checkpoint was trained with incorrect edge indices, zero dihedral angles, double-counted edges, shared FFN weights, degenerate descriptors, RoPE noise, unweighted loss, slow beta1, and incorrect metric reporting. A full retrain from fresh initialization is required after fixes.

For maximum benefit, retrain using data processed through `step_to_graph.py` from STEP files (not the MFTRCAD JSON files, which use a different edge attribute layout — see issue #14).

Expected improvements after retrain:
- **Face segmentation accuracy:** Substantial, particularly for features depending on edge convexity and dihedral angle (pockets, slots, steps, holes).
- **Training stability:** Better convergence from corrected adam_beta1 and proper metric reporting.
- **Model capacity:** Increased from independent FFN weights and non-degenerate descriptors.
