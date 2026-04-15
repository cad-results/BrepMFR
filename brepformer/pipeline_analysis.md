# Pipeline Analysis: BrepFormer / BrepMFR Face Segmentation Issues

## Observed Symptom (Updated)

The model **trains correctly** and achieves **92.2% face accuracy** on the preprocessed
test set (2856 models, 72420 faces). Per-class accuracy ranges from 62% to 100%.

However, when running inference on STEP files via `infer.py`, predictions are
**dramatically wrong** — dominated by class 23 ("round") with near-zero agreement
with ground truth. Comparing the same model through both paths:

```
Model 20240116_231044_1009 (15 faces):
  Test (preprocessed):  [24,24,24,24,24,24,24,24,24,24,24,19,19,19,19]  ← matches GT
  Inference (STEP):     [24,23,23,7,24,24,23,23,24,23,23,23,23,23,23]   ← garbage
```

Out of 200 models compared: **0 out of 200 had matching predictions** between test
and inference pipelines. The root cause is **Issue #28**: the inference pipeline drops
D2 and angle descriptors that the model was trained with, causing severely degraded
attention patterns.

Earlier trial results (1-7% accuracy from comparing .seg files) reflect INFERENCE
predictions, not test accuracy. The model was learning all along — the evaluation
was broken.

### Previous (incorrect) analysis

The original analysis below assumed mode collapse during training. This was wrong —
training works correctly. The issues listed below are still valid engineering
concerns but are NOT the cause of the observed poor predictions. Issues #1-4 in
particular were misidentified as CRITICAL because the true root cause (#28) was not
yet discovered.

---

## 1. No Class Weighting in Either Pipeline (CRITICAL)

**Files:** `brepformer/models/brep_classifier.py:72-77`, `models/brepseg_model.py:146`

### Brepformer pipeline

The face segmentation loss uses `nn.CrossEntropyLoss(weight=face_weight_tensor,
ignore_index=-1)`, but `face_class_weights` defaults to `None` in config. The
`run_pipeline.py` **never passes** `--weighted_crossentropy` to `train_preprocessed.py`
(even when `--face_segmentation` is enabled), so the default pipeline path has NO class
weighting.

### BrepMFR main pipeline

`models/brepseg_model.py:146`:
```python
loss = CrossEntropyLoss(labels_onehot, node_seg)
# class_level_weight=None → defaults to 1.0
```

Both `class_level_weight` and `instance_level_weight` are None.

### Impact

With a **175x imbalance** between the largest and smallest classes:

```
class 24 (stock):              256,954 faces (35.1%)
class 23 (round):                4,088 faces ( 0.6%)
class  7 (circular_through_slot): 1,607 faces ( 0.2%)
```

The gradient signal is dominated by the majority class. But the observed collapse to
*minority* classes (not majority) suggests an even deeper problem: the model enters a
degenerate equilibrium where one class dominates predictions, and without class-weighted
gradients there is no strong corrective signal. Different random seeds land in different
degenerate basins.

### Potential fixes
- **Brepformer**: Pass `--weighted_crossentropy` in `run_pipeline.py` when
  `--face_segmentation` is set. The existing inverse-frequency logic at
  `train_preprocessed.py:188-212` is already implemented but never invoked.
- **BrepMFR**: Pass `class_level_weight` to the custom CrossEntropyLoss with
  inverse-frequency weights computed from the training set.
- Consider `sqrt(1/freq)` or effective-number-of-samples weighting
  (`(1 - beta) / (1 - beta^n)`) instead of raw inverse frequency, which over-corrects
  for extreme minorities.
- Focal loss (`alpha * (1-p)^gamma * CE`) naturally down-weights easy/majority examples.
- Label smoothing (0.05-0.1) prevents over-confidence on majority classes.

---

## 2. BrepMFR Custom CrossEntropyLoss is Numerically Unstable (CRITICAL)

**File:** `models/brepseg_model.py:36-68`

The NonLinearClassifier applies `F.softmax(x, dim=-1)` at line 44, then the custom
CrossEntropyLoss takes `torch.log(predict_prob + epsilon)` at line 67:

```python
# Classifier output (line 44):
x = F.softmax(x, dim=-1)  # returns probabilities in [0, 1]

# Loss computation (line 67):
ce = -label * torch.log(predict_prob + epsilon)  # epsilon=1e-12
```

This is the classic **softmax-then-log anti-pattern**:

1. **PyTorch's `nn.CrossEntropyLoss`** uses `log_softmax` internally, which leverages
   the log-sum-exp trick: `log(softmax(x)_i) = x_i - log(sum(exp(x)))`. This is
   numerically stable even when logits are very large or very small.

2. **The manual `softmax → log` path** first computes `exp(x_i) / sum(exp(x))`, which
   can underflow to 0 for minority-class logits. Then `log(0 + 1e-12) = -27.6`
   produces an extreme gradient. For majority classes with `softmax ≈ 1.0`,
   `log(1.0) = 0` produces near-zero gradient.

3. **Result**: Minority classes get extreme, noisy gradients while majority classes get
   near-zero gradients. Combined with no class weighting, this creates a toxic gradient
   landscape that drives training instability and mode collapse.

### Why this causes different trials to collapse to different classes

Different random initializations lead to different early softmax distributions. Once a
class gets a large logit early in training, the softmax saturates, `log(softmax)` goes
to 0 for that class, and the gradient signal for correcting other classes becomes noise.
The optimizer then reinforces whatever class it happened to favor initially.

### Potential fixes
- Replace the custom loss with `nn.CrossEntropyLoss(weight=..., ignore_index=-1)` which
  takes raw logits (remove the softmax from the classifier)
- Or use `F.log_softmax` + `F.nll_loss` which is mathematically equivalent but
  numerically stable
- The classifier should output raw logits, NOT softmax probabilities

---

## 3. AdamW beta1=0.99 Dilutes Minority-Class Gradients (HIGH)

**File:** `models/brepseg_model.py:306`

```python
optimizer = torch.optim.AdamW(self.parameters(), lr=0.002, betas=(0.99, 0.999))
```

Standard AdamW uses `beta1=0.9`. With `beta1=0.99`, the first-moment estimate averages
over ~100 gradient steps. On a dataset where minority classes appear in < 1% of faces:

- A rare class might contribute meaningful gradients every ~100 faces
- But `beta1=0.99` averages this over ~100 steps of majority-class gradients
- The rare-class gradient signal is diluted to near-zero in the momentum buffer
- The optimizer effectively "forgets" rare class information

### Why this matters for collapse

With beta1=0.99 + no class weighting + numerical instability:
1. Early training: random initialization favors some class
2. Momentum accumulates gradients from that class's predictions
3. Rare classes' corrective gradients get averaged away (1/100th contribution)
4. Model doubles down on the favored class → degenerate equilibrium

### Potential fixes
- Use standard `beta1=0.9` or even `beta1=0.8` for imbalanced problems
- Or use SGD with momentum, which has more predictable behavior on imbalanced data
- Consider separate optimizer groups: lower momentum for the classifier head, standard
  momentum for the encoder

---

## 4. Warmup / ReduceLROnPlateau Interaction Creates LR Trap (HIGH)

**File:** `models/brepseg_model.py:306-335`

The training uses two conflicting LR schedules:

1. **Manual warmup** (lines 318-335): Linearly scales LR from 0 to 0.002 over 5000
   steps, directly overwriting `param_groups["lr"]`
2. **ReduceLROnPlateau** (lines 309-314): Monitors `eval_loss`, reduces LR by 0.5x when
   loss plateaus for 5 epochs

**Conflict scenario:**
- Steps 0-5000: Warmup controls LR, scheduler is effectively bypassed
- After step 5000: Scheduler takes over, but `eval_loss` may already be at the
  degenerate equilibrium (predicting one class). The scheduler sees "plateaued" loss and
  immediately starts reducing LR.
- With the degenerate solution's loss as the baseline, the scheduler can only reduce LR
  further, making it impossible to escape the local minimum.
- `min_lr=0.000001` means the LR can drop 11 halvings (0.002 → 0.000001), effectively
  stopping learning.

### Potential fixes
- Use cosine annealing or one-cycle LR schedule instead of ReduceLROnPlateau
- Add a "burn-in" period where the scheduler isn't activated (e.g., 20 epochs)
- Monitor per-class accuracy alongside eval_loss; don't reduce LR if the model hasn't
  started learning minority classes yet

---

## 5. Only First Shared Edge Used Between Face Pairs (HIGH)

**Files:** `brepformer/data/step_to_graph.py:462-464`,
`brepclassifier/data/step_to_graph.py:652-654`

When two faces share multiple edges (common in complex machining features), the code
stores all shared edges in `edge_info[(fi, fj)]` as a list. But `step_to_graph()` only
uses the first edge:

```python
shared_edges = edge_info.get((fi, fj), [])
if shared_edges:
    edge = shared_edges[0]  # Only first edge used!
```

**Impact:** For a pocket feature with 3 edges between two faces, only the first edge's
geometry (curvature, tangent, length, dihedral angle) is used. The model loses 2/3 of
the edge information for that face pair. This is especially harmful for:
- Rectangular slots/pockets (4+ shared edges between parallel faces)
- V-shaped features (multiple edges at different angles)
- Complex passages with multiple edge segments

### Potential fixes
- Aggregate features from all shared edges: average, max, or concatenate
- Or create multi-edges in the graph (one graph edge per physical edge)
- At minimum, use the edge with the longest length or strongest curvature

---

## 6. Edge Curvature Not Scaled During Normalization (HIGH)

**Files:** `brepformer/data/step_to_graph.py:407-416`,
`brepclassifier/data/step_to_graph.py:583-594`

`normalize_geometry()` scales positions by `1/scale` and areas by `1/scale²`, but
**does not scale edge curvature** (edge_attr index 14). Curvature has dimension
`1/length` and should be scaled by `1/scale`:

```python
# Currently NOT done:
# edge_attr[:, 14] /= scale  # curvature = 1/radius, needs 1/scale correction
```

After normalization to unit sphere, the absolute curvature values are wrong by a factor
of `scale` (which varies per model depending on its physical size). This means:
- A small model's edges have artificially large curvature values
- A large model's edges have artificially small curvature values
- The model cannot learn scale-invariant curvature features

### Potential fixes
- Add `edge_attr[:, 14] /= scale` to `normalize_geometry()` after the scale is computed
- Or normalize curvature independently (e.g., divide by max curvature per model)

---

## 7. Face Hash Collision Risk in Adjacency Graph (HIGH)

**Files:** `brepformer/data/step_to_graph.py:88-91`,
`brepclassifier/data/step_to_graph.py:121-131`

The brepformer builds the face lookup using only `face.__hash__()`:
```python
face_hash_map[face.__hash__()] = i
```

The brepclassifier is slightly safer, using both `id(face)` and `face.__hash__()`.

OpenCASCADE shape hashing is **not guaranteed unique**. If two faces produce the same
hash, the later face overwrites the earlier one in `face_hash_map`, corrupting the
adjacency graph: the overwritten face becomes invisible (no edges connect to it), and
edges that should connect to it instead point to the wrong face.

**Impact:** Corrupted topology means spatial_pos (shortest paths), edge features, and
the entire attention bias operate on wrong connectivity. The transformer's structural
understanding of the B-rep is broken for affected models.

### Potential fixes
- Use `face.IsSame(other_face)` or `face.IsEqual(other_face)` instead of raw hash
- Build `defaultdict(list)` and resolve collisions by topological identity
- Add validation: `assert len(face_hash_map) == len(faces)` after building the map

---

## 8. Non-Stratified Data Splitting (HIGH)

**File:** `brepformer/preprocess.py:328-344`

Uses pure random permutation for train/val/test splitting:
```python
indices = np.random.permutation(num_files)
```

No stratification by class distribution. With 27 classes, many of which are rare, random
splitting can:
- Put all models with rare feature types into one split
- Create val/test sets with missing classes entirely
- Make validation metrics unreliable for tuning

**Contrast:** `brepclassifier/preprocess.py:170-208` uses `StratifiedShuffleSplit`.

### Potential fixes
- Use stratified splitting based on the dominant class or set of classes per model
- Verify all 27 classes appear in all three splits after splitting

---

## 9. Unused Face Attributes in Node Feature Encoder (HIGH)

**Files:** `brepformer/models/layers/embedding.py:240-269`,
`models/modules/layers/brep_encoder_layer.py:176-218`

Both pipelines' node feature encoders only use face_attr indices `[0, 1, 2:5, 5, 6]`:
- `[0]`: surface type (6 categories)
- `[1]`: area
- `[2:5]`: centroid (x, y, z)
- `[5]`: rationality (BSpline flag)
- `[6]`: num_loops / num_wires

**Completely ignored** (computed but never used):
- `[7:10]`: face normal at centroid
- `[10:13]`: bounding box extent
- `[13]`: is_reversed flag

**Impact:** The normal vector and bbox extent carry critical discriminative information:
- Stock faces tend to have axis-aligned normals and large bbox (outer shell)
- Machining feature faces tend to have varied normals and smaller bbox
- Without this, the model is blind to face orientation and relative scale

### Potential fixes
- Add encoders for normal (`face_attr[:, 7:10]`), bbox (`face_attr[:, 10:13]`), and
  is_reversed (`face_attr[:, 13]`) in the node feature embedding
- These features directly distinguish stock from machining features

---

## 10. `torch.cuda.empty_cache()` Every Training Step (MEDIUM-HIGH)

**Files:** `models/brepseg_model.py:126,159`,
`brepclassifier/models/pipe_classifier.py:191-192`

Both training_step and validation_step call `torch.cuda.empty_cache()`. This:
- Forces CUDA to release ALL cached memory and re-allocate every step
- **Slows training 10-30%** from repeated allocation/deallocation overhead
- Can increase memory fragmentation
- Provides zero benefit unless memory is critically constrained

**Combined with the other training issues**, this means training takes longer to reach
any given number of gradient steps, giving the degenerate solution more time to solidify.

### Potential fixes
- Remove per-step `empty_cache()` calls entirely
- If memory is truly constrained, call only at epoch boundaries

---

## 11. Dual-Loss Scale Mismatch (MEDIUM-HIGH)

**File:** `brepformer/models/brep_classifier.py:258-271`

```python
loss = self.model_cls_weight * model_loss + self.face_seg_weight * face_loss
```

- `model_loss` = BCEWithLogitsLoss (per-model, 27-dim multi-hot)
- `face_loss` = CrossEntropyLoss (per-face, hundreds of faces per batch)

Default weights both 1.0, but the loss magnitudes differ substantially. The model-level
loss can dominate early training, pulling the encoder toward whole-model patterns at the
expense of face-level discrimination.

### Potential fixes
- Log both losses separately to diagnose scale
- Tune weights based on observed magnitudes, or use adaptive multi-task weighting
- Train face segmentation standalone (set `model_cls_weight=0`)

---

## 12. Edge Grid Normal Channels Are Spatially Constant (MEDIUM)

**Files:** `brepformer/data/step_to_graph.py:470-482`,
`brepclassifier/data/step_to_graph.py:660-672`

Face normals along edges are sampled at a single midpoint and broadcast to all 10
sample positions:

```python
n1 = _face_normal_at_point(f1, eg[0, 5], eg[1, 5], eg[2, 5])
eg[6, :] = n1[0]  # constant across all 10 points
eg[7, :] = n1[1]
eg[8, :] = n1[2]
```

For curved surfaces, the normal varies along the edge. A through-hole edge has
smoothly rotating normals; a chamfer edge has linearly interpolating normals. This
variation is a key discriminator between feature types.

### Potential fixes
- Sample face normal at each of the 10 curve points independently

---

## 13. Surface Type Alone Cannot Distinguish Stock from Machining Features (MEDIUM)

**Files:** `brepformer/models/layers/embedding.py:241`,
`models/modules/layers/brep_encoder_layer.py:193`

The surface type embedding has only 6 categories: Plane(0), Cylinder(1), Cone(2),
Sphere(3), Torus(4), Other(5). Stock faces are typically Plane (type 0), but so are
rectangular_passage, rectangular_pocket, rectangular_through_slot, rectangular_blind_slot,
rectangular_through_step, rectangular_blind_step, and more.

Without additional face features (normal, bbox, relative area), the model has no
per-face signal to distinguish stock from planar machining features — it must rely
entirely on topological context from the attention mechanism.

### Potential fixes
- Add the unused face attributes (Issue #9) to provide per-face discrimination
- Add face area percentile (relative to all faces in the model) as a feature
- Stock faces tend to be the largest; this single feature would help enormously

---

## 14. BatchNorm Throughout the Pipeline Degrades Variable-Size Inference (MEDIUM)

**Files:** `brepformer/models/layers/blocks.py:131-178`,
`models/brepseg_model.py:17-18`, `models/modules/layers/brep_encoder_layer.py:154-173`

Both pipelines use BatchNorm1d extensively in classifiers and NonLinear modules. During
training, batch statistics are computed over all nodes in the batch. During inference
with eval mode, running statistics (accumulated from training) are used.

**Problem:** For variable-size graphs, the feature distributions shift between models
with 10 faces vs 100 faces. Running statistics are an average over all training batches,
which may be a poor fit for individual models at inference time. This causes attention
bias values and classifier logits to drift.

### Potential fixes
- Replace `BatchNorm1d` with `LayerNorm` in NonLinear, NonLinearClassifier, and
  FaceSegmentationClassifier
- Or use `GroupNorm` which is batch-size independent

---

## 15. Train-Inference Descriptor Mismatch (MEDIUM)

**File:** `brepformer/data/step_to_graph.py:512-553`

`step_to_preprocessed_sample()` (inference) computes graph features without
face_centroids or face_normals, so D2/angle descriptors are absent. If training used
`--compute_descriptors`, the attention bias modules for D2/angle receive zero tensors
at inference time instead of computed features.

### Potential fixes
- If descriptors were used during training, compute them at inference time
- Store a metadata flag indicating whether descriptors were used

---

## 16. Wire Count Includes Holes, Not Just Outer Loops (MEDIUM)

**Files:** `brepformer/data/step_to_graph.py:199-205`,
`brepclassifier/data/step_to_graph.py:286-296`

Face attribute index 6 counts ALL wires (outer boundary + holes) but doesn't
distinguish them. A face with 1 outer boundary + 2 holes reports 3, same as a face with
3 outer boundaries. This makes the "num_loops" feature ambiguous.

**Impact:** The model can't distinguish "face with holes" from "face with multiple
outer loops" — both get the same attribute value.

### Potential fixes
- Separate into two attributes: num_outer_wires and num_holes
- Or encode as (num_wires, num_holes) tuple

---

## 17. Seam Detection Only Checks One Face (MEDIUM)

**Files:** `brepformer/data/step_to_graph.py:351`,
`brepclassifier/data/step_to_graph.py:475`

Edge attribute index 13 (is_seam) is computed as:
```python
BRep_Tool.IsClosed(edge, face1)  # Only checks face1!
```

This ignores face2 entirely. A seam edge is one where both endpoints on a face map to
the same UV position — this is a per-face property, not per-edge. Checking only face1
gives incomplete seam information.

Additionally, the brepformer uses `IsClosed()` while brepclassifier uses `IsClosed_s()`
(static method suffix). These may behave differently.

### Potential fixes
- Check both faces: `is_seam = IsClosed(edge, face1) or IsClosed(edge, face2)`

---

## 18. Silent Exception Swallowing in step_to_graph.py (MEDIUM)

**Files:** `brepformer/data/step_to_graph.py:443-448`,
`brepclassifier/data/step_to_graph.py:631-636`

Face attribute and UV-grid computation catches all exceptions and silently replaces with
zeros:
```python
try:
    face_attrs.append(compute_face_attributes(face))
    face_grids.append(compute_face_uv_grid(face))
except Exception:
    face_attrs.append(np.zeros(14, dtype=np.float32))
    face_grids.append(np.zeros((7, 10, 10), dtype=np.float32))
```

Zero arrays are **indistinguishable from actual planar faces** with no curvature. The
model sees a flat face with area=0, centroid at origin, no normal — which looks like a
degenerate face but gets treated as real data with whatever label was assigned.

Similarly, `_face_normal_at_point` (line 272-295 / 498-523) catches exceptions and
returns None, causing edge grid normal channels to silently remain zero.

### Potential fixes
- Log warnings when exceptions occur, with face index and model ID
- Use a distinct sentinel value (e.g., NaN or a mask channel) for failed faces
- Count exception rate per model and flag models with many failures

---

## 19. BrepMFR Test Output Off-by-One and Hardcoded Path (LOW-MEDIUM)

**File:** `models/brepseg_model.py:226-245`

```python
end_index = max_n_node - np.sum((out_face_feature[i][:] == -1).astype(np.int))
pred_feature = out_face_feature[i][:end_index + 1]  # end_index+1 elements
...
for j in range(end_index):     # but only writes end_index elements
    feature_file.write(str(pred_feature[j]))
```

The last face's prediction is never written. Every .seg output is missing the final face.

Additionally, the output path is hardcoded to `/home/zhang/datasets_segmentation/2_val`
(line 238), which won't work outside the original author's environment.

### Potential fixes
- Use `range(end_index + 1)` or `range(len(pred_feature))`
- Make output path a command-line argument

---

## 20. Dataset Label Validation Without Enforcement (LOW-MEDIUM)

**File:** `data/dataset.py:85-86`

```python
if(torch.max(pyg_graph.label_feature) > 24 or torch.max(pyg_graph.label_feature) < 0):
    print(pyg_graph.data_id)
```

Out-of-range labels are printed but **not rejected**. The model continues training with
invalid labels. If any labels exceed num_classes, `F.one_hot(labels, num_classes)` will
crash at line 145 of brepseg_model.py. If labels are negative, they silently corrupt
one-hot encoding.

### Potential fixes
- Raise an exception or skip the sample when labels are out of range
- Filter invalid labels to -1 (ignored by loss) rather than passing them through

---

## 21. No Rotation Augmentation in Brepformer Pipeline (LOW-MEDIUM)

**File:** `brepformer/run_pipeline.py`, `brepformer/train_preprocessed.py`

The brepformer preprocesses data offline into JSON/numpy and trains from preprocessed
samples. There is no rotation augmentation at training time — the preprocessed
coordinates are used as-is.

The BrepMFR main pipeline (`data/dataset.py:53-56`) does apply `random_rotate=True`
during training, rotating UV-grids and edge grids.

**Impact:** The brepformer model learns orientation-specific features rather than
shape-intrinsic features. The same feature at different orientations gets different
embeddings.

### Potential fixes
- Apply random SO(3) rotation to face_grid, face_attr centroids/normals, edge_grid,
  and edge_attr midpoints/tangents/normals during training
- Or add rotation augmentation to the preprocessing step (generate multiple rotated
  copies per model)

---

## 22. Face ID Ordering Assumption Between Graph and Labels (LOW-MEDIUM)

**Files:** `brepformer/preprocess.py:245-252`, `brepformer/data/dataset.py:125-134`

The label files use face IDs from `_result.json`, while graph files use face indices
from `TopExp_Explorer` traversal order. These are assumed to match, but no validation
exists.

If the MFTRCAD labels were generated with a different OCC version or face enumeration
than `step_to_graph.py`, all face labels would be silently corrupted. However, the fact
that trial 2 achieves *some* accuracy (7.1% > random chance of 3.7%) suggests the
ordering is roughly correct but should still be validated.

### Potential fixes
- Validate a sample of models visually with `visualize_seg.py` in compare mode
- Add a consistency check: verify `len(face_labels) == num_faces_in_graph`

---

## 23. Multi-Label Default Creates Conflicting Gradients (LOW-MEDIUM)

**Files:** `brepformer/train_preprocessed.py:56`, `brepformer/train.py:71`

Both default to `--multi_label True`, using `BCEWithLogitsLoss` for model-level
classification alongside `CrossEntropyLoss` for face segmentation. The different loss
functions create conflicting gradient signals:
- BCE encourages the encoder to produce multi-hot-compatible representations
- CE encourages the encoder to produce single-class-discriminative representations

### Potential fixes
- Set `model_cls_weight=0` when training primarily for face segmentation
- Or switch model-level to single-label CE (not multi-label BCE)

---

## 24. Attention Mask Padding with Distance-0 Spatial Encoding (LOW)

**Files:** `brepformer/data/collator.py:81`,
`data/collator.py:42-49`

Both collators pad spatial_pos with 0 (brepformer) or +1 (BrepMFR, since it adds 1
before padding at line 43). The brepformer's padding means padded positions get the
distance-0 embedding (same as self-loop), potentially interfering with real self-loop
entries.

The BrepMFR collator also sets `spatial_pos = spatial_pos + 1` globally (line 43) to
reserve 0 as padding, which is cleaner but shifts all distances by 1.

### Potential fixes
- Initialize spatial_pos padding with `num_spatial` (max distance) instead of 0
- Ensure `padding_idx=0` in the spatial embedding handles this correctly

---

## 25. padding_idx=0 Zeroes Real Data Embeddings for Line/Convex/Plane (FIXED - was HIGH)

**Files:** `brepformer/models/layers/embedding.py`, `models/modules/layers/brep_encoder_layer.py`

Both pipelines used `padding_idx=0` on edge_type, edge_convexity, and (BrepMFR only)
face_type embeddings. Since the data uses 0-based type indices, this made:
- **Line edges (type 0)** → zero attention bias (most common edge for stock faces)
- **Convex edges (convexity 0)** → zero attention bias (most common for stock outer edges)
- **Plane faces (type 0, BrepMFR only)** → zero face type embedding (stock faces)

Stock face neighborhoods were systematically invisible to the attention mechanism,
while cylindrical features had full-strength signal. This directly biased the model
toward predicting cylindrical classes (round, through_hole) over planar ones (stock).

**Fixed:** All type/convexity indices shifted by +1 before embedding lookup, reserving
index 0 for actual padding.

---

## 26. Padding Edges Corrupt Face 0's Attention Bias via scatter_add (FIXED - was HIGH)

**File:** `brepformer/models/layers/embedding.py`

The collator zero-initializes `edge_index`, so padding edges have `[0, 0]`. After the
+1 CLS offset in GraphAttnBias, they target position `(1, 1)` — the self-attention of
face 0. Padding edges produce nonzero features after encoding (embeddings and NonLinear
have learned nonzero weights). If a small graph (10 edges) is batched with a large one
(200 edges), 190 padding edge features accumulate at face 0's self-attention, creating a
massive spurious bias.

**Fixed:** Padding edges are detected by checking `edge_grid.abs().sum() == 0` and
their features are zeroed before scatter_add.

---

## 27. init_params_global Corrupts padding_idx Embeddings (FIXED - was HIGH)

**File:** `brepformer/models/brep_encoder.py`

`BrepEncoder.__init__` calls `self.apply(init_params_global)` which runs
`nn.init.normal_` on ALL embedding weights, overwriting padding_idx=0 entries with
random nonzero values. Since padding_idx gradients are zeroed by PyTorch, these entries
never update — they stay as random constants forever. Every "padding" lookup returns a
random bias instead of zero.

The BrepMFR main model's `init_params` already handled this correctly (zeroing
padding_idx after init). The brepformer did not.

**Fixed:** Added post-init loop to re-zero all padding_idx entries in BrepEncoder.

---

## 28. Inference Pipeline Missing D2 and Angle Descriptors (CRITICAL - ROOT CAUSE)

**File:** `brepformer/data/step_to_graph.py:577-589`

`step_to_preprocessed_sample()` calls `precompute_graph_features()` which computes
d2_distance and angle_distance from face centroids and normals. However, the sample
dict only includes `spatial_pos` and `in_degree` — **d2_distance and angle_distance
are computed then silently dropped**.

```python
graph_features = precompute_graph_features(
    edge_index=..., num_nodes=..., num_spatial=...,
    face_centroids=face_centroids,  # Provided → d2 computed
    face_normals=face_normals,      # Provided → angle computed
)

sample = {
    "spatial_pos": graph_features["spatial_pos"],
    "in_degree": graph_features["in_degree"],
    # d2_distance: MISSING
    # angle_distance: MISSING
}
```

### Impact

The model's attention bias during inference is: `spatial + 0 + 0 + edge = bias`
instead of the training-time: `spatial + d2 + angle + edge = bias`.

**Comparing the same model (20240116_231044_1009) through both paths:**
- Test (preprocessed data, has d2+angle): `[24,24,24,...,19,19,19,19]` → perfect match
- Inference (STEP file, missing d2+angle): `[24,23,23,7,...,23,23,23]` → mostly wrong

Out of 200 models compared, **0 out of 200** had matching predictions between test
and inference. The model reports 92.2% test accuracy but inference gives near-random
predictions. This is why "the model has high accuracy yet fails to match ground truth."

The +1 embedding shift (Issue #25) made this worse: the model learned to rely MORE on
the attention bias features (since all embeddings now provide signal), so when d2/angle
were missing at inference, the degradation was more severe than before the fix.

### Fix

Added d2_distance and angle_distance to the sample dict:
```python
if "d2_distance" in graph_features:
    sample["d2_distance"] = graph_features["d2_distance"]
if "angle_distance" in graph_features:
    sample["angle_distance"] = graph_features["angle_distance"]
```

---

## 29. Class Weighting Auto-Enabled Causes Over-Prediction of Rare Classes (HIGH)

**File:** `brepformer/run_pipeline.py:244`

`run_pipeline.py` auto-enables `--weighted_crossentropy` whenever `--face_segmentation`
is set. With a 175x class imbalance, inverse-frequency weighting gives the "round"
class (0.6%) a weight ~58x larger than "stock" (35%). This creates a massive incentive
to predict rare classes, causing the model to over-predict "round" everywhere.

The BRepFormer paper (arXiv:2504.07378) uses standard cross-entropy **without** class
weighting and achieves 93.16% on MFTRCAD.

### Fix

Removed auto-enabling of `--weighted_crossentropy` from `run_pipeline.py`. The flag
can still be passed explicitly if needed.

---

## 30. BrepMFR CosineAnnealingWarmRestarts Destabilizes Training (HIGH)

**File:** `models/brepseg_model.py:282-284`

The CosineAnnealingWarmRestarts(T_0=20, T_mult=2) scheduler resets LR to maximum
every 20 epochs (then 60, then 140). This abrupt LR jump destabilizes partially-learned
features. Combined with the 5000-step warmup (which may not complete within T_0 for
small datasets), the scheduler creates chaotic LR dynamics.

The BRepFormer paper uses ReduceLROnPlateau with lr=0.001.

### Fix

Reverted to ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6) and lowered
base LR from 0.002 to 0.001 to match the paper.

---

## 31. prepare_batch() Also Drops D2/Angle Descriptors (CRITICAL)

**File:** `brepformer/infer.py:68-93`

Even after fixing `step_to_preprocessed_sample()` (Issue #28), `prepare_batch()`
hard-coded only 11 fields when wrapping the sample dict as tensors, silently
dropping `d2_distance` and `angle_distance` even when present in the sample.

```python
# BEFORE (broken): only these 11 keys were included
tensor_sample = {
    "face_grid": ..., "face_attr": ..., "edge_index": ...,
    "edge_attr": ..., "edge_grid": ..., "spatial_pos": ...,
    "in_degree": ..., "label": ..., "model_id": ...,
    "num_faces": ..., "num_edges": ...,
    # d2_distance: MISSING
    # angle_distance: MISSING
}
```

This meant d2/angle were dropped at TWO separate points in the inference pipeline:
once in `step_to_preprocessed_sample()` and again in `prepare_batch()`. Both needed
fixing.

### Fix

Added optional d2_distance and angle_distance to `prepare_batch()`:
```python
if "d2_distance" in sample:
    tensor_sample["d2_distance"] = _to_tensor(sample["d2_distance"])
if "angle_distance" in sample:
    tensor_sample["angle_distance"] = _to_tensor(sample["angle_distance"])
```

---

## 32. Training Data Format Mismatch with step_to_graph() (CRITICAL - OPEN)

**Files:** `brepformer/data/step_to_graph.py`, training graph JSONs in `mftrcad/graphs/`

The MFTRCAD dataset's graph JSON files use a **different feature format** than what
`step_to_graph()` produces. The model was trained on the dataset format, so inference
via `step_to_graph()` on unknown STEP files produces wrong features.

### Face attribute format mismatch

Training JSON format (14-dim):
```
[one_hot_type(6), 0, 0, 0, area, 0, centroid_x, centroid_y, centroid_z]
```

step_to_graph() format (14-dim):
```
[scalar_type, area, cx, cy, cz, rational, n_loops, nx, ny, nz, bbx, bby, bbz, reversed]
```

Correlation analysis across 201 faces from 5 models confirms:
- Train[0:6] = one-hot surface type (corr -0.94 with scalar type)
- Train[9] = area (corr 0.97 with inference area, ratio ~1.9x due to normalization)
- Train[11:14] = centroid (corr 0.98/0.99/0.92 with inference centroid)
- Train[6:8] = always zero (unused slots)
- Train[10] = always zero

### Edge attribute format mismatch

Training edge format (15-dim):
```
[0, convex_flag, smooth_flag, length, concave_flag, circle_flag, 0, line_flag, other_type_flag, 0, 0, 0, 0, 0, 0]
```

step_to_graph() format (15-dim):
```
[type, length, mx, my, mz, tx, ty, tz, dihedral, convexity, n1x, n1y, n1z, seam, curvature]
```

Confirmed: Train[3] = length (corr 0.98), Train[7] = line indicator (corr -0.84 with type).

### Normalization mismatch

Training data uses:
- **Center**: bounding box center `(min + max) / 2`
- **Scale**: half the max bounding box dimension `max(extent) / 2`

step_to_graph() uses:
- **Center**: mean of all UV-grid points
- **Scale**: max distance from center

Area ratio is constant within each model (std=0.0) but varies across models
(1.60-2.80), confirming different normalization scales. Extent ratios are
consistent across xyz (e.g., 1.32726 for model 0), confirming uniform scaling.

### Face UV-grid sampling mismatch (NOT FIXED)

After fixing normalization and attribute formats, face UV-grid positions and
trimming masks still differ between the dataset JSONs and `step_to_graph()`:
- xyz channels: max_diff up to 2.0
- mask channel: training has trimmed regions (0s), inference has all 1s
- Edge grids match **perfectly** (same points, tangents, normals)

This indicates `compute_face_uv_grid()` uses a different UV parameterization
and trimming algorithm than whatever tool created the MFTRCAD dataset. This
cannot be fixed by format conversion alone.

### Partial fix applied

`normalize_geometry()` was modified to:
1. Use bbox center + half-max-extent (matching training normalization)
2. Convert face_attr to one-hot type format with area at index 9 and centroid at 11-13
3. Convert edge_attr to one-hot flags format with length at index 3

After these fixes:
- Face attr: **exact match** (all 14 dims)
- Edge attr: **near-exact match** (minor convexity flag differences)
- Edge grid: **perfect match** (all 12 channels)
- Face grid: **still mismatched** (UV sampling difference)

### Measured impact

Testing on 50 models from the MFTRCAD test set:

| Inference path | Face Acc | F1 (macro) | F1 (weighted) | Precision | Recall | Perfect models |
|---------------|----------|-----------|---------------|-----------|--------|----------------|
| Preprocessed pickle (exact match) | **94.7%** | **86.4%** | **94.6%** | **86.8%** | **87.5%** | **24/50 (48%)** |
| step_to_graph (format-fixed) | 28.3% | 19.7% | 30.6% | 21.1% | 25.0% | 0/50 (0%) |

The ~66% accuracy gap is entirely due to the face UV-grid sampling mismatch.

### Workaround (current)

For dataset models, `infer.py` loads from preprocessed pickle files instead of
running `step_to_graph()`. This gives exact match with training (94.7% accuracy).
The fallback to `step_to_graph()` only triggers for STEP files not in the dataset.

### Fix for unknown STEP files (requires retraining)

To achieve full accuracy on arbitrary STEP files, repreprocess and retrain:

```bash
# Step 1: Convert all STEP files to graph JSONs using step_to_graph
python3 -c "
import sys, json, os
sys.path.insert(0, '.')
from brepformer.data.step_to_graph import step_to_graph
from pathlib import Path

steps_dir = Path('brepformer/data/mftrcad/steps')
out_dir = Path('brepformer/data/mftrcad_stg/graphs')
out_dir.mkdir(parents=True, exist_ok=True)

for step_file in sorted(steps_dir.glob('*.step')):
    data = step_to_graph(str(step_file))
    if data is None:
        continue
    model_id = step_file.stem
    out = [model_id, {
        'graph': {
            'edges': data['edge_index'].tolist(),
            'num_nodes': data['num_nodes'],
        },
        'graph_face_attr': data['face_attr'].tolist(),
        'graph_face_grid': data['face_grid'].tolist(),
        'graph_edge_attr': data['edge_attr'].tolist(),
        'graph_edge_grid': data['edge_grid'].tolist(),
    }]
    with open(out_dir / f'{model_id}.json', 'w') as f:
        json.dump(out, f)
    print(f'  {model_id}')
"

# Step 2: Copy labels directory
cp -r brepformer/data/mftrcad/labels brepformer/data/mftrcad_stg/labels

# Step 3: Preprocess with descriptors
python brepformer/preprocess.py \
    --data_dir brepformer/data/mftrcad_stg \
    --output_dir brepformer/data/mftrcad_stg_processed \
    --compute_descriptors

# Step 4: Retrain
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/mftrcad_stg_processed \
    --face_segmentation \
    --face_seg_weight 1.0 \
    --model_cls_weight 1.0 \
    --max_epochs 200 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --num_workers 4 \
    --output_dir results \
    --exp_name trial6_stg

# Step 5: Verify — step_to_graph inference should now match test accuracy
python brepformer/analyze.py \
    --checkpoint results/trial6_stg/best-*.ckpt \
    --data_dir brepformer/data/mftrcad_stg_processed \
    --step_dir brepformer/data/mftrcad/steps \
    --mode step_inference \
    --max_models 100 \
    --output_dir analysis_results/trial6_stg
```

After retraining, both the preprocessed and step_to_graph paths should produce
matching results since they use the same feature extraction code.

---

## Summary of Priorities

| Priority | Issue | Status |
|----------|-------|--------|
| **CRITICAL** | **#28 Inference missing d2/angle descriptors** | **FIXED - added to step_to_preprocessed_sample + prepare_batch** |
| **CRITICAL** | **#31 prepare_batch() also drops d2/angle** | **FIXED - added optional d2/angle to tensor dict** |
| **CRITICAL** | **#32 Training data format mismatch with step_to_graph** | **PARTIAL - normalization+attr format fixed; face UV-grid sampling still differs. Dataset models work via pickle fallback. Unknown STEP files need retraining (see #32 for steps).** |
| **HIGH** | **#29 Class weighting auto-enabled** | **FIXED - removed auto-enable** |
| **HIGH** | **#30 CosineAnnealing destabilizes training** | **FIXED - reverted to ReduceLROnPlateau, lr=0.001** |
| CRITICAL | #1 No class weighting | REVERTED - class weighting removed (paper doesn't use it) |
| CRITICAL | #2 Softmax→log instability | FIXED - replaced with nn.CrossEntropyLoss |
| HIGH | #3 AdamW beta1=0.99 | FIXED - changed to 0.9 |
| HIGH | #4 Warmup + ReduceLROnPlateau trap | REVERTED - ReduceLROnPlateau restored (matches paper) |
| HIGH | #5 Only first shared edge used | FIXED - aggregate all shared edges |
| HIGH | #6 Edge curvature not scaled | FIXED - divide by scale in normalize_geometry |
| HIGH | #7 Face hash collision | FIXED - IsSame collision resolution |
| HIGH | #8 Non-stratified splitting | FIXED - StratifiedShuffleSplit |
| HIGH | #9 Unused face attributes | FIXED - normal/bbox/reversed encoders added |
| HIGH | #25 padding_idx=0 zeroes Line/Convex/Plane | FIXED - +1 index shift |
| HIGH | #26 Padding edges corrupt face 0 bias | FIXED - mask padding before scatter_add |
| HIGH | #27 init_params_global corrupts padding_idx | FIXED - re-zero after init |
| MED-HIGH | #10 empty_cache() every step | FIXED - removed from train/val steps |
| MED-HIGH | #11 Dual-loss scale mismatch | FIXED - logs individual losses for tuning |
| MEDIUM | #12 Constant edge normals | FIXED - per-point normal sampling |
| MEDIUM | #13 Surface type insufficient | FIXED via #9 (normal/bbox provide discrimination) |
| MEDIUM | #14 BatchNorm at inference | FIXED - replaced with LayerNorm |
| MEDIUM | #15 Train-inference descriptor mismatch | FIXED via #28 (d2/angle now included) |
| MEDIUM | #16 Wire count includes holes | NOT FIXED - would require format change |
| MEDIUM | #17 Seam detection one face | FIXED - checks both faces |
| MEDIUM | #18 Silent exception swallowing | FIXED - logger.warning with context |
| LOW-MED | #19 Test output off-by-one + hardcoded path | NOT FIXED |
| LOW-MED | #20 Label validation without enforcement | NOT FIXED |
| LOW-MED | #21 No rotation augmentation (brepformer) | NOT FIXED |
| LOW-MED | #22 Face ID ordering assumption | NOT FIXED |
| LOW-MED | #23 Multi-label conflicting gradients | NOT FIXED |
| LOW | #24 Padding spatial_pos distance-0 | NOT FIXED (benign with attn_mask) |
