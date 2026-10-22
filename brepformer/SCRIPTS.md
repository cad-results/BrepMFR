# BrepFormer Scripts Reference

Complete reference for all scripts in the `brepformer/` directory with usage, CLI arguments, and examples. For an overview and quick start, see [README.md](README.md).

> Scripts marked ***recommended*** are the most capable variant when multiple scripts serve the same purpose.
> Examples marked ***best*** show the optimal flags for face segmentation quality.

---

## End-to-End Pipeline

### `run_pipeline.py` ***recommended — runs the full pipeline in one command***

Chains all stages — preprocess, train, test, analyze — with automatic checkpoint detection and sensible defaults. Use `--skip_*` flags to skip stages, or `--only` to run a single stage.

```bash
# *best* — full pipeline optimized for face segmentation quality
python brepformer/run_pipeline.py \
    --data_dir brepformer/data/mftrcad \
    --output_dir results/best_face_seg \
    --face_segmentation \
    --face_seg_weight 2.0 --model_cls_weight 0.5 \
    --compute_descriptors \
    --max_epochs 200 \
    --batch_size 32 --learning_rate 0.0016 --num_workers 4

# Full pipeline from raw data
python brepformer/run_pipeline.py \
    --data_dir brepformer/data/mftrcad \
    --output_dir results/full_run \
    --max_epochs 200

# Full pipeline with face segmentation
python brepformer/run_pipeline.py \
    --data_dir brepformer/data/mftrcad \
    --output_dir results/face_run \
    --face_segmentation \
    --face_seg_weight 2.0 --model_cls_weight 0.5

# With D2/angle descriptors for best quality
python brepformer/run_pipeline.py \
    --data_dir brepformer/data/mftrcad \
    --output_dir results/full_desc \
    --face_segmentation \
    --compute_descriptors

# Skip preprocessing (data already prepared)
python brepformer/run_pipeline.py \
    --data_dir brepformer/data/mftrcad \
    --output_dir results/full_run \
    --skip_preprocess

# Resume training from checkpoint
python brepformer/run_pipeline.py \
    --data_dir brepformer/data/mftrcad \
    --output_dir results/full_run \
    --skip_preprocess \
    --resume_from results/full_run/train/last.ckpt

# Only test + analyze an existing checkpoint
python brepformer/run_pipeline.py \
    --data_dir brepformer/data/mftrcad \
    --output_dir results/full_run \
    --only test \
    --checkpoint results/full_run/train/best-epoch=50-val/f1=0.8800.ckpt

# Quick sanity check
python brepformer/run_pipeline.py \
    --data_dir brepformer/data/sample \
    --output_dir /tmp/sanity \
    --fast_dev_run --num_workers 0 --batch_size 4 \
    --hidden_dim 64 --ffn_dim 128 --num_heads 8 --num_kv_heads 4 --num_layers 2
```

| Argument | Default | Description |
|----------|---------|-------------|
| **Stage control** | | |
| `--only` | None | Run only this stage: "preprocess", "train", "test", "analyze" |
| `--skip_preprocess` | False | Skip preprocessing stage |
| `--skip_train` | False | Skip training stage |
| `--skip_test` | False | Skip testing stage |
| `--skip_analyze` | False | Skip analysis stage |
| **Paths** | | |
| `--data_dir` | required | Raw data directory (graphs/ + labels/) or preprocessed dir |
| `--output_dir` | required | Root output directory |
| `--checkpoint` | auto | Checkpoint path (auto-detected from best training result) |
| `--resume_from` | None | Resume training from this checkpoint |
| **Preprocessing** | | |
| `--compute_descriptors` | False | Compute D2/angle multi-sample histograms (slower but improves quality) |
| `--split_ratio` | "0.8,0.1,0.1" | Train/val/test split ratios |
| `--seed` | 42 | Random seed |
| **Architecture** | | |
| `--hidden_dim` | 256 | Hidden dimension |
| `--ffn_dim` | 512 | FFN dimension |
| `--num_layers` | 8 | Transformer layers |
| `--num_heads` | 32 | Attention heads |
| `--num_kv_heads` | 8 | KV heads (GQA) |
| `--dropout` | 0.3 | Dropout probability |
| **Face segmentation** | | |
| `--face_segmentation` | False | Enable face-level segmentation head with automatic class weighting |
| `--face_seg_weight` | 1.0 | Loss weight for face segmentation |
| `--model_cls_weight` | 1.0 | Loss weight for model classification |
| **Training** | | |
| `--batch_size` | 32 | Batch size |
| `--learning_rate` | 0.002 | Learning rate |
| `--max_epochs` | 200 | Maximum epochs |
| `--num_workers` | 4 | Data loader workers |
| `--devices` | 1 | Number of GPUs |
| `--precision` | 32 | Training precision (16 or 32) |
| `--fast_dev_run` | False | Quick 1-batch sanity check |
| `--limit_data` | None | Limit total dataset to N samples; manifest auto-propagated to test/analyze |
| **Test/Analysis** | | |
| `--8_classes` | False | Remap 27 classes to 8 grouped machining feature categories |
| `--5_classes` | False | Remap 27 classes to 5 defeature categories (random, hole, chamfer, fillet, cut) |
| `--analyze_modes` | "all" | Comma-separated analysis modes |

**Output structure:**
```
output_dir/
├── preprocessed/        # train.pkl, val.pkl, test.pkl, metadata.json
├── train/               # checkpoints (best-*.ckpt, last.ckpt)
├── test_results.json    # model-level metrics on test split (Stage 3)
├── train_results.json   # same metrics evaluated on train split (Stage 3b)
├── face_preds.json      # per-face predictions on test split (if --face_segmentation)
├── train_face_preds.json# per-face predictions on train split (if --face_segmentation)
└── analysis/            # per_class_metrics.json, embeddings, confusion matrices
```

> **Stage 3b** runs automatically alongside Stage 3 whenever the test stage is active. It evaluates the checkpoint on the **training split** using the same metrics as the test evaluation, producing `train_results.json` (and `train_face_preds.json`). Comparing these files against `test_results.json` is the primary way to detect overfitting.

#### Defeature Dataset (5 classes)

```bash
# Full pipeline for the navin_defeaturing dataset (5 classes)
# Note: run_pipeline.py does not pass --num_classes to preprocess.py,
# so run the individual stages instead (see each section below).
```

### `inference_pipeline.sh` ***recommended — single STEP file end-to-end pipeline***

Takes any STEP file and runs the full pipeline: inference → FreeCAD export → analysis → segmentation viewer → defeature v2 → defeature viewer. All output is saved to `brepformer/pipeline_output/<filename>/`. Automatically switches conda environments from `brepmfr` (steps 1-4) to `new_brepmfr` (steps 5-6) for defeature v2 compatibility (pythonocc >= 7.9).

```bash
# *best* — full pipeline with analysis data
./brepformer/scripts/inference_pipeline.sh brepformer/data/defeature/steps/uh62382.step \
    --data_dir brepformer/data/defeature_processed

# Basic: just run on a STEP file (architecture analysis only)
./brepformer/scripts/inference_pipeline.sh brepformer/data/defeature/steps/uh62382.step

# Headless (no interactive viewers — good for batch/CI)
./brepformer/scripts/inference_pipeline.sh brepformer/data/defeature/steps/uh62382.step --skip_viewers

# Custom checkpoint
./brepformer/scripts/inference_pipeline.sh brepformer/data/defeature/steps/chicane.step \
    --checkpoint "results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt"

# Custom output directory
./brepformer/scripts/inference_pipeline.sh brepformer/data/defeature/steps/chicane.step \
    --output_dir /tmp/my_results/
```

| Argument | Default | Description |
|----------|---------|-------------|
| `<step_file>` | required | Path to a STEP (.step/.stp) file |
| `--checkpoint` | `results/trial1_ss/best-epoch=41-val/f1=0.9476.ckpt` | Model checkpoint |
| `--data_dir` | None | Preprocessed data directory for full analysis (`--mode all`) |
| `--skip_viewers` | false | Skip interactive viewer steps (4 and 6) |
| `--output_dir` | `brepformer/pipeline_output/<filename>/` | Override output directory |

**Pipeline steps:**

| Step | Module | Conda Env | Description |
|------|--------|-----------|-------------|
| 1 | `brepformer.infer` | brepmfr | Predict per-face labels → `preds.json`, `preds.seg` |
| 2 | `brepformer.export_freecad` | brepmfr | Export colored STEP → `<name>_colored.step` |
| 3 | `brepformer.analyze` | brepmfr | Model analysis → `analysis/` |
| 4 | `brepformer.visualize_seg` | brepmfr | Interactive 3D segmentation viewer |
| 5 | `brepformer.defeature_v2` | new_brepmfr | Remove features → `<name>_defeatured.step`, `<name>_report.json` |
| 6 | `brepformer.visualize_defeature` | new_brepmfr | Original vs defeatured comparison viewer |

**Output structure:**
```
brepformer/pipeline_output/<filename>/
├── preds.json                    # Full predictions (per-face probs, class names)
├── preds.seg                     # Per-face labels (one label per line)
├── <filename>_colored.step       # Colored STEP for FreeCAD
├── analysis/                     # Analysis outputs
│   ├── architecture.json         # Model architecture (always present)
│   ├── per_class_metrics.json    # Per-class P/R/F1 (if --data_dir)
│   ├── embeddings.npy            # Embeddings (if --data_dir)
│   ├── face_seg_metrics.json     # Face segmentation metrics (if --data_dir)
│   └── ...
├── <filename>_defeatured.step    # Defeatured STEP (features removed)
├── <filename>_colored.step       # Colored predictions overlay
└── <filename>_report.json        # Defeaturing report (removed/failed/valid)
```

**Environment notes:**
- The script expects `brepmfr` to be the active conda environment when invoked.
- Steps 5-6 auto-switch to `new_brepmfr` (requires pythonocc >= 7.9 for `defeature_v2`).
- If `--data_dir` is not provided, step 3 runs `--mode architecture` only (no dataset needed).
- If `--data_dir` is provided, step 3 runs `--mode all` (per_class, embeddings, confusion_matrix, face_segmentation, etc.).

---

## Data Acquisition & Labeling

How the GrabCAD corpus was obtained and turned into training labels. Read this before
re-running any of it: the pipeline is a chain of pseudo-labelling steps, and two of the
plausible-looking stages were **measured to make labels worse** and are deliberately not
used to rewrite anything.

### `data/harvest_grabcad_slugs.py`

Collects candidate model slugs from GrabCAD listing/search pages. Slugs only — no files.

```bash
# *best* — harvest slugs for STEP-bearing mechanical parts
python -m brepformer.data.harvest_grabcad_slugs \
    --out brepformer/data/grabcad_slugs.json --target 1000
```

### `data/grabcad_login.py`

Opens a browser so **you** can log in to GrabCAD by hand and saves the resulting session
cookies for the downloaders to reuse. Log in yourself in the window it opens; do not put
credentials on a command line or in a file — they would land in shell history, process
listings and these logs.

```bash
# *best* — one interactive login, reused by every later download
python -m brepformer.data.grabcad_login --state brepformer/data/grabcad_state.json
```

### `data/download_grabcad_playwright.py`

Downloads STEP files for harvested slugs by driving a real browser session. This is the
one that works; the two variants below are earlier attempts kept for reference.

```bash
# *best* — download STEPs using the saved session
python -m brepformer.data.download_grabcad_playwright \
    --slugs brepformer/data/grabcad_slugs.json \
    --state brepformer/data/grabcad_state.json \
    --out_dir brepformer/data/hard_samples --limit 200
```

### `data/download_grabcad.py` / `data/download_grabcad_api.py`

Earlier downloaders: a plain-HTTP scraper and a REST-API pass. Superseded by the
Playwright version, which is the only one that reliably reaches the file endpoint.

```bash
python -m brepformer.data.download_grabcad --slugs brepformer/data/grabcad_slugs.json \
    --out_dir brepformer/data/hard_samples
python -m brepformer.data.download_grabcad_api --slugs brepformer/data/grabcad_slugs.json \
    --out_dir brepformer/data/hard_samples
```

### `data/probe_grabcad_*.py`

Five single-purpose diagnostics, written to answer one question each when a download
stage broke. They are not part of the pipeline; run one only when the corresponding
stage fails.

```bash
python -m brepformer.data.probe_grabcad_search   # find an endpoint that yields slugs
python -m brepformer.data.probe_grabcad_dom      # dump download-related page DOM
python -m brepformer.data.probe_grabcad_files    # per-model file list from Angular scope
python -m brepformer.data.probe_grabcad_api      # is the community REST API reachable?
python -m brepformer.data.probe_grabcad_auth     # what makes the file endpoint return 200?
```

### `data/curate_corpus.py`

Filters a downloaded corpus down to parts worth training on (size, face count, parse
success), so the expensive graph conversion is not spent on assemblies and meshes.

```bash
# *best* — curate before graph conversion
python -m brepformer.data.curate_corpus \
    --steps_dir brepformer/data/hard_samples \
    --out_dir brepformer/data/hard_samples_curated
```

### `rotator/pipeline.py`

Aligns a STEP part so its dominant axes lie on the coordinate axes, then renders 6
orthographic views as PNGs. `rotator/environment.yml` describes a dedicated **`rotator`**
env, but **that env does not exist on this machine** — `new_brepmfr` already has both
pythonocc and VTK, so run it there. (The vision pass never needed the env either: it
imports `rotator/pipeline.py` by path, not by environment.) Two distinct uses:

- **Rendering** for any workflow that needs images of a part — this is what feeds the
  vision pass below.
- **Canonical alignment** as preprocessing. Worth knowing that the 14-dim face features
  in `step_to_graph.py` include centroid, normal and bounding box, which are all
  frame-dependent, so aligning first removes variance the model would otherwise have to
  learn around. `normalize_geometry()` already centres and scales, but does not rotate.

Alignment is **one rigid rotation of the whole shape**, derived from area-weighted
cylinder/cone/torus axes; it is not per-face. Non-90° elbows cannot be fully aligned and
report a 5°+ deviation — that is geometric, not a failure.

```bash
# *best* — batch render 6 views per part, flat output, for a vision pass
conda run -n new_brepmfr python rotator/pipeline.py \
    --step_dir brepformer/data/suitable_samples_scaled/steps \
    -o brepformer/data/suitable_samples_views/ --simple

# single part, with alignment debug plots and a 2x3 montage
conda run -n new_brepmfr python rotator/pipeline.py \
    --step brepformer/data/hard_samples/123-block-5__123_Block.step \
    -o /tmp/rot_check/ --visualize -v

# higher-resolution views (vision accuracy is sensitive to this)
conda run -n new_brepmfr python rotator/pipeline.py \
    --step_dir brepformer/data/suitable_samples_scaled/steps \
    -o brepformer/data/suitable_samples_views/ --simple --image_size 1024 1024
```

| Flag | Default | Description |
|---|---|---|
| `--step` / `--step_dir` | — | Single file or batch directory |
| `-o`, `--output` | `output/` | Output directory |
| `--limit` | 0 (all) | Cap batch mode |
| `--simple` | False | Six PNGs per model, no `views/` subdirectory |
| `--mesh` | False | Overlay triangular mesh edges |
| `--visualize` | False | Before/after alignment plot + montage |
| `--interactive` | False | VTK 3D window (needs a display) |
| `--image_size` | 800 800 | Rendered pixel dimensions |

### `reclassify_visual.py`

The rotator + Claude vision pass: renders face-highlighted multi-view images of the
lowest-confidence faces and asks Claude to classify them, emitting a corrected inference
JSON plus a corrections log.

> **Measured result: do not use this to rewrite labels.** Two harnesses below scored it
> against mftrcad ground truth. As a *verifier* it denied 91% of wrong labels but also
> 74% of correct ones — a near-constant verdict carrying almost no signal. As a
> *labeller* it came out at or below chance on feature faces. Since pseudo-labels cannot
> contradict it, running it over the corpus would produce confident-looking corruption.
> Re-measure with the validators before trusting any change here.

It is still useful for what it *selects*: `--no_claude` gives the face-highlighted
renders and the low-confidence face ranking without spending any API calls, which is the
fastest way to eyeball what the teacher is unsure about.

```bash
# *best* — inspect what the teacher is unsure about, no API calls, no label changes
python -m brepformer.reclassify_visual \
    --step_dir brepformer/data/suitable_samples_scaled/steps \
    --seg_dir brepformer/inference/hard_samples_27 \
    --output_dir brepformer/inference/hard_samples_27_review/ \
    --label_space 27 --no_claude --limit 20

# with the vision pass, capped — validate first (see the two harnesses below)
python -m brepformer.reclassify_visual \
    --step_dir brepformer/data/suitable_samples_scaled/steps \
    --seg_dir brepformer/inference/hard_samples_27 \
    --output_dir brepformer/inference/hard_samples_27_corrected/ \
    --label_space 27 --max_claude_calls 50 --geo_mode advise
```

| Argument | Default | Description |
|---|---|---|
| `--step` / `--step_dir` | — | Single STEP or batch directory |
| `--seg_json` / `--seg_dir` | — | Existing inference output to review |
| `--output_dir` | required | Corrected JSONs + corrections log |
| `--label_space` | 27 | Space of the input inference |
| `--angle_step` | 30 | Degrees between rendered views |
| `--no_claude` | False | Render and select only; make no API calls |
| `--claude_model` | claude-sonnet-5 | Vision model |
| `--max_claude_calls` | 50 | Hard cap on API calls |
| `--mask_confidence` | 0.7 | Faces below this are candidates for review |
| `--geo_mode` | advise | `advise` = rules only *select* faces. **Never set this to rewrite** — see below |
| `--limit` | 0 (all) | Cap models |

### `vision_label.py`

The rewritten dense labeller, and the one to use if any vision pass is used at all. It
differs from `reclassify_visual.py` in four ways, each of which fixed a measured failure:

1. **Full-sphere view search.** The old pass rendered a fixed ring of 6 views, and on a
   ground-truth model **2 of 6 faces came out with literally zero pixels of the
   highlighted face** — the model was being asked to classify something it could not see.
   This one scores all 266 directions (15° steps in azimuth and elevation, plus poles)
   *analytically* — projected face area is `Σ areaᵢ · max(0, nᵢ · d)`, one matrix product,
   2–5 ms — and only renders the winners. Same model: 6/6 faces visible, median best-view
   visibility 21,908 → 29,807 px, and slightly *faster* per face because 266 candidates
   are ranked by arithmetic instead of rendered.
2. **Exact B-Rep facts in the prompt**, from `face_geometry.py` — surface type, radius,
   sweep, per-edge normal deviation, convexity, coaxial group sweep. Some of these decide
   the class and are invisible in *any* image (see the hole-vs-fillet note below).
3. **Measured posteriors instead of hand-written rules** (see `data/measure_class_geometry.py`).
4. **Multi-pass self-consistency**: 3 passes with different view strategies
   (best-visibility / wide-separation / mixed), and the model may request up to 4 extra
   named angles per pass; confidence = pass agreement × mean stated confidence, so
   `--agreement`-based masking is possible downstream.

Nothing is written to disk during rendering — images go to the API as in-memory PNG bytes.

**Constructed views, for geometry no camera can reach.** Some faces are invisible from
*every* one of the 266 directions: a bore wall hides behind its own opening, a counterbore
seat sits at the bottom of a shaft, a pocket fillet is occluded by the pocket walls. For
those faces the ranking can only pick the least-bad view, which is how the earlier pass came
to send a 1-pixel sliver. Four constructions fix it, and none is reachable by orbiting:

| View | What it is | What it settles |
|---|---|---|
| `zoom_tight` | close crop framed on the face's own bounding box | whether the edges are sharp or blended — unresolvable when the face is 1% of the frame |
| `cutaway` | solid clipped by a plane just *in front* of the face, deleting its occluders while leaving the face whole | any face hidden behind other geometry |
| `section` | clipped *through* the face — a draughtsman's cross-section | feature **depth**, and whether it breaks through: through vs blind hole, pocket vs step |
| `endoscope_±axis` | **perspective** camera placed *inside* the feature, on its own axis, looking along it | a tube of red is a bore wall; a red ring or disc is an end face |

These are added automatically when the best exterior view shows under 1,500 px of the face,
and again during the challenge round below. Interior views need a real perspective frustum
(parallel projection has no viewpoint, so "inside" and "outside" render identically) and
two-sided lighting with culling off, since a clipped solid is an open shell whose triangles
are seen from behind.

**Independent verification, then one more look.** The vote across passes only measures
self-consistency, and a model can be consistently wrong — in the first live run two
ground-truth holes came back `fillet` from all three passes at 0.92 confidence. So the
leading answer is checked against something that did not come from the model:

* **the corpus** — the measured posterior rows this face falls into (bins imported from
  `measure_class_geometry.CONDITIONS`, never restated, so the row it is looked up in is the
  row it was counted in). If under 3% of ≥150 labelled faces with this same geometry were
  the chosen class, that is an objection carrying a sample size;
* **definitional facts** — the three statements true by *construction* rather than by
  frequency: a fillet blends tangentially, a round blends tangentially, a hole is a bore.
  Anything merely *typical* is left to the posterior rows, which is why this list is short;
* **visibility** — if the face never exceeded ~1,500 px in any frame, an answer resting on
  image evidence is not supported whatever its stated confidence.

An objection **never rewrites the class**. `geometric_rules.py` was allowed to rewrite and
destroyed 15% of correct labels and 100% of `chamfer_2`. Here an objection buys the face one
more look: it is stated back to the model together with the cutaway/section/endoscope views,
and the model may hold its answer or change it. Because that round saw strictly more
evidence than any earlier pass, it breaks ties in its own favour. A face where the objection
stands and the model will not move is a face to **mask**, not to overwrite.

**Which facts get quoted is measured, not chosen.** Each posterior table is scored by its
information gain about the label — `H(class) − H(class | fact)` in bits, over the faces that
table covers — and only tables above `MIN_INFO_GAIN_BITS` (0.15) reach the prompt, ranked. The
ranking overturned the hand-picked order that preceded it:

| Fact | Bits | Sharpest row |
|---|---|---|
| `surface_type × coaxial_group_sweep_degrees` | 1.06 | `cylinder & 340–361° → hole 88%` (n=26,802) |
| `surface_type × accessibility` | **0.97** | `torus & accessibility=… → fillet 87%` (n=1,578) |
| `coaxial_group_sweep_degrees` | 0.80 | `340–361 → hole 80%` (n=31,133) |
| `surface_type × n_tangential_edges` | 0.80 | `plane & 0 tangential → random 81%` |
| `surface_type × n_inner_wires` | **0.75** | `plane & n_inner_wires=… → random 91%` (n=1,774) |
| `surface_type × on_convex_hull` | **0.74** | `torus & on_hull → fillet 84%` (n=1,942) |
| `mean_curvature_sign` | **0.57** | `+1 → hole 57%` (n=45,018) — beats `surface_type` alone (0.51) |
| `on_convex_hull` alone | 0.05 | — dropped |
| `is_saddle` alone | 0.05 | — dropped |

So three of the new descriptors outrank most of what was there before, but **only in their
joint form**: `on_convex_hull`, `is_saddle` and `hull_depth_vs_extent` each carry ~0.05 bits
on their own and are dropped, because they only mean something conditioned on the surface
type. The same floor gates the per-face rows used for objections — an objection raised from a
near-uniform row would be an objection from noise. A descriptor added to
`measure_class_geometry.py` therefore reaches the prompt exactly when it earns its place.

**Effort scales with how rare the face is; thresholds do not.** `rarity()` runs before any API
call and asks how well the corpus covers *this* face — the sample size of the joint rows it
falls into, whether the classes live for it are all rare ones, and whether it sits in a large
one-sided row:

* **Novel or rare geometry** (under 40 labelled faces sharing its combination of
  measurements, or only rare classes live): passes go 3 → 5, using two extra view strategies
  reserved for this case (`dense-orbit`, `far-separation`), and the cutaway / section /
  endoscope views are rendered *up front* rather than only after an objection. The prompt
  says so: the tables are thin here, the images have to carry the face, abstain if they
  cannot. What such a face does **not** get is a higher agreement bar — raising the bar on
  under-covered faces is precisely the ratchet that drove `chamfer` and `cut` out of the
  pseudo-labels once already: the rare classes fail the strict vote first, get masked, and
  vanish.
* **Common geometry** (a row of ≥500 labelled faces that is ≥80% one class) gets the opposite
  treatment — the **common-face guard**. A minority answer there is an automatic objection
  quoting the row and its sample size, and it survives only if the challenge round *held* the
  answer at ≥0.85 confidence. Otherwise the face is masked. It is never rewritten to the
  majority class: masking costs one training face, while overwriting teaches the student a
  wrong rule about the geometry it sees most often, which is what `geometric_rules.py` did.

This is aimed at a measured failure: on ground truth the pre-guard labeller scored **14.3% on
`random`**, calling stock faces `cut` — and `plane & accessibility ≥ 0.999 → random 86%`
(n=14,720) is exactly the dominant row that now objects to it.

**Masking.** The output is per-face `final_label`, which is `-1` (the `ignore_index` the
segmentation loss and `merge_suitable_samples.py` already use) whenever any of: the model
abstained (`class: -1`, which the prompt asks for explicitly and treats as a correct answer
rather than a failure); any pass abstained; pass agreement below `--min_agreement`; combined
confidence below `--min_confidence`; or an objection was raised and the challenge round
returned nothing. `mask_reasons` records *which*, because "the passes disagreed", "the model
abstained" and "an objection stood" are different problems and only the first is fixed by
more passes. A sixth reason exists for common faces: answering against a large one-sided row
without a confident rebuttal (see the guard above).

**Both label spaces.** The posterior tables are per label space and cannot be shared:
`class_geometry_stats.json` is measured on the 5-class defeature corpus,
`class_geometry_stats_27.json` on mftrcad. Quoting 5-class shares while asking for a
27-class answer would state "hole 87%" for a row whose faces are really split across
`through_hole` / `blind_hole` / `counterbore`. The 27-class tables are in places *sharper*
than the 5-class ones — `cone & n_tangential_edges=0 → chamfer_2 98%` (n=6,534) — because
the finer labels cut along the same geometric lines.

> **Still gated on `data/validate_dense_vision.py`.** The predecessor looked plausible and
> measured at or below chance. Do not let this rewrite a training label until the
> validator says it beats the teacher on the same faces.

```bash
# *best* — one face, full detail, to see exactly what the model is shown and why
python -m brepformer.vision_label \
    --step brepformer/data/defeature/steps/0-00528-552-06.step \
    --face 12 --label_space 5 --passes 3 --verbose

# batch: relabel a directory, capped, resumable
python -m brepformer.vision_label \
    --step_dir brepformer/data/suitable_samples_scaled/steps \
    --seg_dir brepformer/inference/hard_samples_5_ft \
    --out_dir brepformer/inference/hard_samples_5_vision/ \
    --label_space 5 --passes 3 --faces_per_model 12 --max_calls 2000
```

| Argument | Default | Description |
|---|---|---|
| `--step` / `--face` | — | Single-face debug mode (repeat `--face`) |
| `--step_dir` / `--seg_dir` / `--out_dir` | — | Batch mode over an inference directory |
| `--label_space` | 5 | `5` or `27` |
| `--passes` | 3 | Independent passes voted per face |
| `--faces_per_model` | 12 | Lowest-confidence faces to relabel per model |
| `--sphere_step` | 15 | Degrees between candidate directions (266 at 15°) |
| `--image_size` | 768 | Render size |
| `--no_requests` | False | Forbid the model asking for extra angles |
| `--min_agreement` | 0.66 | Below this the face is masked to -1 |
| `--min_confidence` | 0.55 | Below this the face is masked to -1 |
| `--no_challenge` | False | Skip the objection round (to measure what it is worth) |
| `--max_calls` | 0 (none) | Hard cap on API calls |

### `face_geometry.py`

Exact per-face B-Rep geometry in words, for the prompt: surface type and its parameters,
area fraction, wrap angle, concavity, neighbour surface types, and per shared edge the
normal deviation and convexity. Deliberately applies **no thresholds** — it reports
measurements, and what counts as "tangential" is decided from the labelled data by
`data/measure_class_geometry.py`.

Beyond the per-face surface and edge facts it computes five whole-part descriptors, each
because a single face cannot supply it:

| Descriptor | What it measures | Confusion it addresses |
|---|---|---|
| `accessibility` | fraction of outward directions from which the face can be reached from outside the solid, by vectorised Möller–Trumbore ray casting against the part's own triangle mesh | separates classes that differ by **enclosure** rather than shape: pocket floor vs stock face, counterbore seat vs step, blind-hole bottom vs boss top. Also a real manufacturing quantity (tool accessibility). Measured 1.0 on stock faces and 0.47 on bore walls on the first test part |
| `on_convex_hull` / `hull_depth_vs_extent` | signed distance from the face to the part's convex hull (scipy `ConvexHull`, distance = max over hull half-space equations) | the cleanest separator for `random`/stock, which is 27% of faces and the class the others are most confused with. A face on the hull is original body surface; one well inside is looking out of a cavity |
| `concave_component_*` / `strict_component_*` | Joshi & Chang attributed-adjacency-graph components: maximal sets of faces joined by concave edges, with size, member surface types and area share | names the *feature* rather than the face — walls + floor is a pocket, cylinder + disc is a blind hole. Two variants because the plain AAG over-merges: on a bent bracket the bend edges are concave too and one component swallowed 12 of 15 faces, so a strict variant requires a ≥30° valley. **Which one separates the classes is not decided here** — both are reported and measured against the labels |
| `n_inner_wires` | inner loops of the face (`BRepTools.OuterWire` vs the wire count) | the cheapest reliable "this face bounds a hole"; a through hole has an inner wire at both ends, a blind hole's bottom has none |
| curvature signs | Gaussian/mean curvature and both principal curvatures at mid-uv (`GeomLProp_SLProps`), orientation-corrected | an internal fillet round a pocket floor is a saddle (K<0) while the convex outer round of the same radius has K>0; the mean-curvature sign is a cheaper, more robust concavity test than the axis projection |

The `accessibility` ray casting uses the mesh rather than an OCC ray–solid intersection,
which is orders of magnitude slower, and casts from the face's **mid-parameter point, not
its centre of mass** — the centroid of a bore wall lies on its axis in empty space, and rays
from there escape in every direction, reporting a hole as fully accessible.

It was also, by a wide margin, the slowest thing in the whole measurement: 112 s of a 136 s
run on one 283-face part. Two things were tried; only one worked.

* **Batching all directions** into a `(triangles, directions, 3)` grid was ~3× *slower* —
  the intermediate cross product leaves cache and the einsums it needs are not BLAS shapes,
  which costs more than the Python loop over a dozen directions saves. A comment in the
  source records this so it is not "optimised" again.
* **A triangle bounding-sphere prefilter** — reject any triangle whose perpendicular distance
  from the ray line exceeds its own circumradius (`|d × rel| > r`, plus a behind-the-origin
  test) before running the exact intersection — discards well over 99% of triangles and cut
  accessibility from **112 s to 4.7 s (~24×)** on the same part, with bit-identical output.

Verified by re-running `0-00528-552-06.step` and checking the accessibility vector is
unchanged to the digit; a ray-cast speedup that changes any value is a bug, not a speedup.

The one other fact worth calling out is `coaxial_group_sweep_degrees`. A hole wall and a fillet
are *both* partial cylinders, and looking at one face they are the same kind of object —
the kernel splits a full bore into two 180° half-cylinders, so "180°" describes both. This
walks the tangential seam across coaxial same-radius faces and sums the sweep, recovering
360° for the bore and leaving the blend band unchanged. It is the single strongest
hole-vs-fillet discriminator and it cannot be seen in any rendering.

```bash
# *best* — everything the kernel knows about one face
python -m brepformer.face_geometry --step model.step --face 12

# whole part, without the per-edge detail
python -m brepformer.face_geometry --step model.step --brief
```

### `geometric_rules.py`

Hand-written rules mapping face geometry to feature classes (cylinder → through_hole,
and so on).

> **Measured result: use only as advisory.** Scored against mftrcad ground truth these
> rules overwrite **15% of correct labels**, including **100% of `chamfer_2`**. Their
> errors are systematic and concentrated on exactly the rare classes the corpus was
> expanded to fix. `reclassify_visual.py` therefore uses them only to *pick* faces to
> look at, and deliberately keeps their suggested class out of the vision prompt so it
> cannot anchor the model toward the same wrong answers.

```bash
# *best* — advisory report only, no labels changed
python -m brepformer.geometric_rules \
    --seg_json brepformer/inference/hard_samples_27/some_model.json \
    --step brepformer/data/hard_samples/some_model.step --report
```

### `data/validate_geometric_rules.py`

Scores the rules against mftrcad ground truth: how often they deny a correct label
(harm) versus fix a wrong one (benefit), broken out per class. This is the script that
produced the 15% / 100%-of-`chamfer_2` numbers above.

```bash
# *best* — the measurement that decides whether rules may rewrite anything
python -m brepformer.data.validate_geometric_rules --limit 200
```

### `data/validate_vision_labeling.py`

Measures the vision pass as a **verifier**, on ground truth, using two rates that only
mean something together: *harm* (feed the true label; any DENY is corruption) and
*benefit* (feed a deliberately wrong label; a DENY counts only if the suggested class is
the true one). Reporting DENY-rate alone would hide replacing one wrong label with
another.

```bash
# *best* — harm vs benefit before spending any vision budget on the corpus
python -m brepformer.data.validate_vision_labeling --limit 25 --faces_per_model 6
```

### `data/validate_vision_classify.py`

Measures the vision pass as a **labeller** instead — asks for a class outright, with no
candidate to anchor on, and compares to ground truth. This removes the anchoring that
made the verify framing degenerate. It reports per-face accuracy directly comparable to
the teacher's, per class, and separately on the faces the teacher is unsure about, since
those are the ones masked out of training and the only ones worth paying to relabel.

```bash
# *best* — per-face accuracy vs the teacher, in the deployed 5-class space
python -m brepformer.data.validate_vision_classify --limit 25 --label_space 5

# 27-class space, more models
python -m brepformer.data.validate_vision_classify --limit 100 --label_space 27
```

### `data/measure_class_geometry.py`

Measures what the geometry of each label class actually looks like, over the whole
hand-labelled corpus, and writes `data/class_geometry_stats.json` — which
`vision_label._stats_block` renders straight into the prompt. This exists because
**every hand-written rule that has been checked against the corpus turned out to be
wrong**, and wrong in ways that read as obviously true:

| Rule written by hand from a 40-model sample | Measured on 115,766 faces |
|---|---|
| "a cone means chamfer" (89% on the sample) | 46% — `cut` and `random` cones are as common |
| "chamfers have 0 tangential edges" | median is **2** |
| "narrow + curved = fillet" | also describes a bore wall; turned 3 GT holes into confident fillets |

So the script reports `P(class | fact)` — the **posterior**, not the per-class marginal.
"65% of chamfers are cones" reads like an answer and is not one: the model is shown a cone
and needs `P(chamfer | cone)`, and since `random` and `cut` are an order of magnitude
larger classes, a fact characteristic of chamfers can still be dominated by them.
Percentiles rather than means, because a class whose radius spans three orders of
magnitude has a meaningless mean. It also emits three *joint* tables
(`surface_type × coaxial_sweep`, `× concavity`, `× tangential edges`), because that is
where the single-fact ambiguity actually resolves.

Bins with fewer than 30 faces (50 for joint tables) are dropped rather than stated, so the
prompt never asserts a share computed from a handful of faces.

```bash
# *best* — full 5-class corpus, feeds the prompt (115,766 faces / 1,561 models, ~4 min)
python -m brepformer.data.measure_class_geometry --workers 24

# the 27-class space, from mftrcad ground truth
python -m brepformer.data.measure_class_geometry --label_space 27 \
    --steps brepformer/data/mftrcad/steps --labels brepformer/data/mftrcad/labels \
    --out brepformer/data/class_geometry_stats_27.json --workers 24

# quick look while iterating
python -m brepformer.data.measure_class_geometry --limit 200
```

Selected measured posteriors now quoted in the prompt (5-class):

| Condition | Verdict |
|---|---|
| cylinder & coaxial sweep 340–361° | **hole 87%** |
| cylinder & sweep 50–100° | fillet 63%, cut 31% |
| cylinder & sweep 100–200° | **cut 68%** |
| cone & half-angle 40–50° | **chamfer 84%** |
| torus | fillet 74% |
| area fraction ≥ 0.1 of the part | random 87% |
| base rates | random 27%, cut 30%, hole 24%, fillet 15%, chamfer 4% |

### `data/arbitrate_labels.py`

How the vision pass is allowed to touch the corpus: as a **fourth voter with the narrowest
mandate**, not as a relabeller. The teacher already labels every face for free and per-class
confidence masking already decides which of those labels are safe — on the current corpus that
leaves **109,893 of 216,781 faces (50.7%) masked to `-1`**, contributing no gradient at all.
Those discarded faces are both the largest opportunity and the only place a vision label has
no downside: a face already destined to be dropped cannot be made worse by an answer.

The ladder is ordered by what a mistake costs:

| Situation | Default action | Why |
|---|---|---|
| vision never looked at the face | keep teacher | no effect outside its call budget |
| vision abstained / masked itself | keep teacher | an abstention is not evidence against the teacher |
| teacher masked it, vision decided | **fill with vision** | pure gain — the face was going to be dropped |
| teacher kept it, vision agrees | keep, count `confirmed` | two independent sources now support it |
| teacher kept it, vision disagrees | **keep teacher** | overwriting here already cost 15% of correct labels once |

Two escalations exist, both off by default and neither enabled by a flag alone:

* `--veto` lets a confident disagreement mask a kept face — trading a possibly-wrong label
  for a certainly-absent one.
* `--overwrite` lets vision replace a class, and **only** for classes where
  `--gate_json` (the `validate_dense_vision.py` output) measured vision beating the teacher by
  `--gate_margin` on at least `--gate_min_n` ground-truth faces. A class the gate never
  sampled gets no licence: absence of evidence is not a licence, and a small sample would fake
  one exactly on the rare classes. With no licences the script says so and overwrites nothing.

`read_vision` reads `final_label`, not `class`, so a face `vision_label.py` itself refused to
commit to arrives as an abstention rather than as an answer. Coverage (`rarity`) travels with
each verdict so this script can tell a verdict on well-measured geometry from one on geometry
the corpus has never seen.

```bash
# *best* — fill only: recover masked faces, touch nothing the teacher kept
python -m brepformer.data.arbitrate_labels \
    --labels_in brepformer/data/suitable_samples_pcq5/labels \
    --vision_dir brepformer/inference/hard_samples_5_vision \
    --out_dir brepformer/data/suitable_samples_arb5/labels --label_space 5

# dry run: the per-class delta and the per-face action histogram, writing nothing
python -m brepformer.data.arbitrate_labels ... --dry_run

# escalated, licensed by the gate
python -m brepformer.data.arbitrate_labels ... --veto --overwrite \
    --gate_json brepformer/data/dense_vision_validation.json
```

| Argument | Default | Description |
|---|---|---|
| `--labels_in` | — | Existing (already confidence-masked) label dir |
| `--vision_dir` | — | `vision_label.py` per-model JSONs |
| `--out_dir` | — | Where arbitrated labels are written |
| `--min_fill_confidence` | 0.55 | Vision confidence needed to fill a masked face |
| `--veto` / `--veto_confidence` | off / 0.75 | Let a confident disagreement mask a kept face |
| `--overwrite` / `--overwrite_confidence` | off / 0.85 | Replace a kept label, gate-licensed classes only |
| `--gate_json` | `data/dense_vision_validation.json` | Where the per-class licence is read from |
| `--gate_margin` / `--gate_min_n` | 0.10 / 10 | Accuracy points and sample size a licence requires |
| `--dry_run` | off | Report the delta, write nothing |

### `data/validate_dense_vision.py`

The gate. Scores `vision_label.py` against hand-labelled ground truth and reports the
three numbers that decide whether it may touch a training label:

* **Per-class accuracy**, not overall. Overall accuracy is dominated by `random` and `cut`
  (~60% of faces); a labeller that always answers `random` scores respectably and adds
  nothing. `chamfer` is 4% of faces and is the class that matters.
* **The teacher's accuracy on the same faces.** Being right 70% of the time is worthless
  if the model that labels those faces for free is right 90% of the time.
* **Accuracy as a function of pass agreement.** This is what decides whether the vote can
  be used to mask. If unanimous faces are much more accurate than split ones, agreement is
  a usable confidence and the rest can be dropped; if the curve is flat, the vote means
  nothing and no threshold makes the output safe.

* **Accuracy on the faces it keeps, versus the ones it masks.** Masked faces never reach the
  loss, so a labeller at 70% overall that reaches 92% on the 60% it keeps is worth running,
  and one whose accuracy does not rise when the doubtful faces are dropped is not — its mask
  is throwing away examples at random.
* **Accuracy by corpus coverage.** Common geometry (a large one-sided posterior row), novel
  geometry, faces where only rare classes are live, and faces the rarity check escalated to
  five passes, each with *wrong-and-kept* counts — the only errors that reach the training
  set. The two halves have different jobs and one accuracy number hides both: common faces
  must be right because they are the bulk of the loss, while rare ones are allowed to be
  masked and should be when unresolved.
* **What the challenge round is worth**: how many faces drew an objection, how many verdicts
  it changed, and accuracy on objected faces versus the rest.

Faces are sampled balanced per class and spread over models (≤2 faces per class per
model), so a few hundred calls say something about `chamfer` rather than spending 60% of
the budget on `random`.

Measured on the version *before* the rarity escalation and common-face guard: 69.1% overall
on 70 faces, `hole` 100%, `cut` 93%, `fillet` 86%, `chamfer` 57%, and **`random` 14.3%** —
stock faces called `cut`. Agreement was strongly informative (unanimous 77.2%, 2-of-3 30.0%,
split 0%), which is what licensed masking on it at all.

```bash
# *best* — the gate, ~70 faces balanced across 5 classes, with the teacher for comparison
python -m brepformer.data.validate_dense_vision --per_class 14 --passes 3 \
    --seg_dir brepformer/inference/defeature_ft

# 27-class space
python -m brepformer.data.validate_dense_vision --per_class 8 --passes 3 \
    --steps brepformer/data/mftrcad/steps --labels brepformer/data/mftrcad/labels \
    --label_space 27
```

## Data Preprocessing

### `copy data command`
cp -r /mnt/c/projects/data/mftrcad /home/user/BrepMFR/brepformer/data/mftrcad

#### Defeature Dataset — Data Preparation

The defeature dataset uses STEP files with per-face text labels (one class per line). `prepare_defeature.py` copies STEP files, remaps the 7-class labels to 5 classes, and writes JSON labels in the format expected by `preprocess.py`. Then `step_to_graph.py` converts STEP geometry to graph JSONs.

| Original Class | Remapped Class | Name |
|----------------|----------------|------|
| 0 | 0 | Random (Other) |
| 1 | 1 | Hole |
| 2 | 1 | Hole |
| 3 | 2 | Chamfer |
| 4 | 3 | Fillet |
| 5 | 4 | Cut |
| 6 | 4 | Cut |

```bash
# Step 1: Copy STEP files and convert text labels to JSON (fast, no pythonocc needed)
python brepformer/data/prepare_defeature.py \
    --source /mnt/c/projects/data/navin_defeaturing \
    --dest brepformer/data/defeature

# Step 2: Convert STEP files to graph JSONs (requires pythonocc, slow — ~1-5 sec/model)
python brepformer/data/prepare_defeature.py \
    --source /mnt/c/projects/data/navin_defeaturing \
    --dest brepformer/data/defeature \
    --convert_graphs

# Or run just the graph conversion on already-copied data:
python -c "
import sys, json
sys.path.insert(0, '.')
from brepformer.data.step_to_graph import step_to_graph
from pathlib import Path
from tqdm import tqdm

steps_dir = Path('brepformer/data/defeature/steps')
out_dir = Path('brepformer/data/defeature/graphs')
out_dir.mkdir(parents=True, exist_ok=True)

for step_file in tqdm(sorted(steps_dir.glob('*.step'))):
    model_id = step_file.stem
    if (out_dir / f'{model_id}.json').exists():
        continue
    try:
        data = step_to_graph(str(step_file))
        if data is None:
            continue
        out = [model_id, {
            'graph': {'edges': data['edge_index'].tolist(), 'num_nodes': data['num_nodes']},
            'graph_face_attr': data['face_attr'].tolist(),
            'graph_face_grid': data['face_grid'].tolist(),
            'graph_edge_attr': data['edge_attr'].tolist(),
            'graph_edge_grid': data['edge_grid'].tolist(),
        }]
        with open(out_dir / f'{model_id}.json', 'w') as f:
            json.dump(out, f)
    except Exception as e:
        print(f'  Failed: {model_id}: {e}')
"
```

**Output structure:**
```
brepformer/data/defeature/
├── steps/          # 1561 sanitized STEP files
├── labels/         # 1561 JSON label files (remapped 5-class)
├── graphs/         # graph JSONs from step_to_graph (after conversion)
└── dataset_info.json  # metadata: class distribution, remap table
```

### `preprocess.py` ***recommended***

Converts raw JSON graph data to pickle format for efficient multi-worker loading. Reads graph files from `graphs/` and per-face labels from `labels/`, computes shortest paths and in-degree features, and writes train/val/test splits to pickle files.

The output includes `face_labels` (per-face class IDs, 0-26 or -1 for missing) alongside the multi-hot model labels, enabling face segmentation training without re-preprocessing.

```bash
# *best* — with multi-sample descriptors for highest model quality
python brepformer/preprocess.py \
    --data_dir brepformer/data/mftrcad \
    --output_dir brepformer/data/mftrcad_processed \
    --compute_descriptors \
    --d2_bins 64 \
    --angle_bins 64

# Basic usage
python brepformer/preprocess.py \
    --data_dir brepformer/data/mftrcad \
    --output_dir brepformer/data/mftrcad_processed

# Full configuration
python brepformer/preprocess.py \
    --data_dir brepformer/data/mftrcad \
    --output_dir brepformer/data/mftrcad_processed \
    --num_classes 27 \
    --split_ratio "0.8,0.1,0.1" \
    --seed 42

# With D2/angle descriptors (slower, needed for descriptor-aware models)
python brepformer/preprocess.py \
    --data_dir brepformer/data/mftrcad \
    --output_dir brepformer/data/mftrcad_desc \
    --compute_descriptors \
    --d2_bins 64 \
    --angle_bins 64

# With external single-label file
python brepformer/preprocess.py \
    --data_dir brepformer/data/mftrcad \
    --output_dir brepformer/data/mftrcad_custom \
    --label_file path/to/my_labels.json \
    --num_classes 10
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | required | Input directory with `graphs/` and `labels/` subdirectories |
| `--output_dir` | required | Output directory for pickle files |
| `--label_file` | None | External labels JSON for single-label mode (optional) |
| `--num_classes` | 27 | Number of classes |
| `--compute_descriptors` | False | Compute D2/angle descriptors using multi-sample histograms from UV-grids (slower preprocessing) |
| `--num_spatial` | 64 | Max shortest-path distance for spatial position |
| `--d2_bins` | 64 | D2 descriptor histogram bins |
| `--angle_bins` | 64 | Angle descriptor histogram bins |
| `--split_ratio` | "0.8,0.1,0.1" | Train/val/test split ratios |
| `--seed` | 42 | Random seed for split reproducibility |

**Output:**
- `train.pkl`, `val.pkl`, `test.pkl` — serialized sample lists
- `metadata.json` — dataset statistics (num_samples, num_classes, split sizes)

#### Defeature Dataset — Preprocessing

```bash
# *best* — preprocess defeature data with descriptors (5 classes)
python brepformer/preprocess.py \
    --data_dir brepformer/data/defeature \
    --output_dir brepformer/data/defeature_processed \
    --num_classes 5 \
    --compute_descriptors \
    --d2_bins 64 \
    --angle_bins 64

# Basic preprocessing (faster, no descriptors)
python brepformer/preprocess.py \
    --data_dir brepformer/data/defeature \
    --output_dir brepformer/data/defeature_processed \
    --num_classes 5
```

#### Trials 6 / 6A — Pseudo-Labelled Corpus Preparation

Builds the two pcq (per-class-quantile) corpora that the trial6 sweep trains on. The
same 809 GrabCAD STEP files are merged twice — once with labels collapsed to the 5-class
defeature space, once left in the 27-class MFTRCAD space — so the two model families
train on identical geometry and differ only in label space. Thresholds are always fitted
in 27-class space, because the teacher's confidence is calibrated per fine class.

Run these in order; steps 1a/1b are independent of each other but both need step 0c.

```bash
# 0a. Merge pcq pseudo-labels in 5-class space (for trial6)
python brepformer/data/merge_suitable_samples.py \
    --steps_dir brepformer/data/suitable_samples_scaled/steps \
    --inference_dir brepformer/inference/hard_samples_27 \
    --out_dir brepformer/data/suitable_samples_pcq5 \
    --per_class_top_frac 0.5 --per_class_floor 0.5 --min_kept_frac 0.15 --label_space 5

# 0b. Merge the same corpus in 27-class space (for trial6A)
python brepformer/data/merge_suitable_samples.py \
    --steps_dir brepformer/data/suitable_samples_scaled/steps \
    --inference_dir brepformer/inference/hard_samples_27 \
    --out_dir brepformer/data/suitable_samples_pcq27 \
    --per_class_top_frac 0.5 --per_class_floor 0.5 --min_kept_frac 0.15 --label_space 27

# 0d. OPTIONAL — second-teacher agreement, to keep MORE faces at the same precision.
#     Faces below their fitted per-class cutoff are rescued when an independently
#     trained teacher predicted the same class. Two models being wrong the same way is
#     much rarer than one being confidently wrong, so this adds supervised faces
#     instead of trading precision for them the way a lower cutoff does.
#     Requires 5-class inference over the same STEPs (see "Trials 6 / 6A — Inference").
conda run -n new_brepmfr --no-capture-output python brepformer/data/merge_suitable_samples.py \
    --steps_dir brepformer/data/suitable_samples_scaled/steps \
    --inference_dir brepformer/inference/hard_samples_27 \
    --out_dir brepformer/data/suitable_samples_pcq5_agree \
    --per_class_top_frac 0.5 --per_class_floor 0.5 --min_kept_frac 0.15 \
    --label_space 5 \
    --agree_inference_dir brepformer/inference/hard_samples_5_ft \
    --agree_space 5 --agree_floor 0.4

# 0c. Build graphs ONCE (same STEP files, same ids) and share them via symlink
python -m brepformer.data.steps_to_graphs \
    --steps_dir brepformer/data/suitable_samples_pcq27/steps \
    --out_dir   brepformer/data/suitable_samples_pcq27/graphs \
    --tolerate_crashes
ln -sfn "$PWD/brepformer/data/suitable_samples_pcq27/graphs" \
        brepformer/data/suitable_samples_pcq5/graphs

# 1a. Preprocess pcq5 (settings MUST match defeature_processed exactly, or
#     combine_processed refuses to merge them)
python brepformer/preprocess.py \
    --data_dir brepformer/data/suitable_samples_pcq5 \
    --output_dir brepformer/data/suitable_samples_pcq5_processed \
    --num_classes 5 --compute_descriptors \
    --num_spatial 64 --d2_bins 64 --angle_bins 64 --seed 42

# 1b. Preprocess pcq27 (must match mftrcad_processed)
python brepformer/preprocess.py \
    --data_dir brepformer/data/suitable_samples_pcq27 \
    --output_dir brepformer/data/suitable_samples_pcq27_processed \
    --num_classes 27 --compute_descriptors \
    --num_spatial 64 --d2_bins 64 --angle_bins 64 --seed 42

# 2a. Combine for trial6: splits preserved, so the defeature test set stays
#     bit-identical and a metric change is attributable to the added data
python -m brepformer.data.combine_processed \
    --inputs brepformer/data/defeature_processed \
             brepformer/data/suitable_samples_pcq5_processed \
    --output_dir brepformer/data/trial6_processed

# 2b. Combine for trial6A
python -m brepformer.data.combine_processed \
    --inputs brepformer/data/mftrcad_processed \
             brepformer/data/suitable_samples_pcq27_processed \
    --output_dir brepformer/data/trial6a_processed
```

### `data/steps_to_graphs.py`

Batch STEP → graph JSON converter for any corpus of STEP files. Resumable (skips
existing outputs) with atomic writes, so a run killed part-way can simply be restarted.

```bash
# *best* — subprocess per file, so an OCC segfault costs one file not the whole run
python -m brepformer.data.steps_to_graphs \
    --steps_dir brepformer/data/suitable_samples_pcq27/steps \
    --out_dir   brepformer/data/suitable_samples_pcq27/graphs \
    --tolerate_crashes

# Faster, for a corpus known to be clean (~50ms/file less overhead)
python -m brepformer.data.steps_to_graphs \
    --steps_dir brepformer/data/aman_sample \
    --out_dir   brepformer/data/aman_sample/graphs
```

| Argument | Default | Description |
|---|---|---|
| `--steps_dir` | required | Directory of `.step`/`.stp` files |
| `--out_dir` | required | Where to write `<model_id>.json` |
| `--limit` | none | Convert at most N remaining files |
| `--tolerate_crashes` | False | Re-exec a subprocess per file; survives OCC segfaults |

### `data/combine_processed.py`

Merges several preprocessed corpora into one train/val/test tree by symlink, preserving
each corpus's own split. Re-running `preprocess.py` over the union would recompute
descriptors for all ~28.5k mftrcad models to add a few hundred; this is near-instant.
Incompatible descriptor settings are rejected up front rather than failing at collate
time mid-run.

```bash
# *best* — symlinks, splits preserved (test set stays bit-identical)
python -m brepformer.data.combine_processed \
    --inputs brepformer/data/defeature_processed \
             brepformer/data/suitable_samples_pcq5_processed \
    --output_dir brepformer/data/trial6_processed

# Copy instead of symlink (if the output must survive the inputs being deleted)
python -m brepformer.data.combine_processed \
    --inputs brepformer/data/mftrcad_processed \
             brepformer/data/suitable_samples_pcq27_processed \
    --output_dir brepformer/data/trial6a_processed --copy
```

| Argument | Default | Description |
|---|---|---|
| `--inputs` | required | Preprocessed dirs to combine; order sets sample numbering |
| `--output_dir` | required | Output tree (`train/`, `val/`, `test/`, `metadata.json`, `provenance.json`) |
| `--copy` | False | Copy pickles instead of symlinking |
| `--allow_incompatible` | False | Proceed despite mismatched preprocessing settings — almost always wrong |

---

## Training

### `train_preprocessed.py` ***recommended***

Training with preprocessed pickle data. Supports multi-worker data loading, optional face segmentation head (with automatic inverse-frequency class weighting), mixed precision, gradient accumulation, and checkpoint resumption.

```bash
# *best* — face segmentation with tuned weights and class balancing
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --face_segmentation \
    --face_seg_weight 1.0 \
    --model_cls_weight 1.0 \
    --face_seg_hidden_dim 512 \
    --max_faces 500
    --max_epochs 300 \
    --batch_size 32 \
    --learning_rate 0.0005 \
    --num_workers 4 \
    --precision "32" \
    --output_dir results \
    --exp_name trial3

# Basic training (model-level classification only)
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --max_epochs 100

# Full configuration
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --max_epochs 200 \
    --batch_size 32 \
    --learning_rate 0.0016 \
    --num_workers 4 \
    --hidden_dim 256 \
    --num_layers 8 \
    --num_heads 32 \
    --num_kv_heads 8 \
    --dropout 0.3 \
    --output_dir results \
    --exp_name trial2 \
    --devices 1 \
    --precision "32"

# With face segmentation head (multi-task training)
# Note: inverse-frequency class weights are computed automatically from
# the training data when --face_segmentation is enabled.
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --face_segmentation \
    --face_seg_weight 1.0 \
    --model_cls_weight 1.0 \
    --max_epochs 200 \
    --batch_size 32 \
    --learning_rate 0.0016 \
    --precision "32"
    --exp_name face_seg

# Face segmentation with tuned loss weights
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --face_segmentation \
    --face_seg_weight 2.0 \
    --model_cls_weight 0.5 \
    --face_seg_hidden_dim 512 \
    --face_seg_dropout 0.3 \
    --max_epochs 200 \
    --batch_size 32 \
    --learning_rate 0.0016 \
    --precision "32"
    --exp_name face_seg_heavy

# Resume from checkpoint
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --resume_from results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --max_epochs 200

# Fast development test (1 batch train + 1 batch val)
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --fast_dev_run
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | required | Preprocessed data directory |
| `--num_classes` | auto | Number of classes (auto-detected from metadata.json) |
| `--multi_label` | True | Multi-label classification |
| `--face_segmentation` | False | Enable face-level segmentation head |
| `--face_seg_weight` | 1.0 | Loss weight for face segmentation |
| `--model_cls_weight` | 1.0 | Loss weight for model classification |
| `--num_face_classes` | 27 | Number of face-level classes |
| `--face_seg_hidden_dim` | 512 | Hidden dimension for face segmentation MLP |
| `--face_seg_dropout` | 0.3 | Dropout for face segmentation MLP |
| `--use_rope` | False | Enable Rotary Position Embeddings (disabled by default for graph nodes) |
| `--hidden_dim` | 256 | Hidden dimension |
| `--ffn_dim` | 512 | FFN dimension |
| `--num_layers` | 8 | Transformer layers |
| `--num_heads` | 32 | Attention heads |
| `--num_kv_heads` | 8 | KV heads (GQA) |
| `--dropout` | 0.3 | Dropout probability |
| `--batch_size` | 32 | Batch size |
| `--learning_rate` | 0.002 | Learning rate |
| `--max_epochs` | 200 | Maximum epochs |
| `--warmup_steps` | 5000 | LR warmup steps. **Counts batches, not optimizer steps** (`current_step` is incremented in `training_step`), so with a small `--batch_size` this is reached far sooner than it looks — and while warmup is active it rewrites the LR every batch, overwriting any `ReduceLROnPlateau` reduction. |
| `--gradient_clip_val` | 1.0 | Gradient clipping |
| `--lr_monitor` | "train/loss_epoch" | Metric `ReduceLROnPlateau` watches. The default decays the LR on *training* loss, so it keeps cutting while the model overfits; prefer `val/loss` or `val/face_f1`. Mode (min/max) is derived from the name. |
| `--early_stop_monitor` | "val/loss" | Metric `EarlyStopping` watches. On a pseudo-labelled corpus `val/loss` mixes ground truth with teacher noise; `val/face_f1` stops on the deployed axis. |
| `--early_stop_patience` | 20 | EarlyStopping patience in epochs |
| `--num_workers` | 4 | Data loader workers |
| `--accumulate_grad_batches` | 1 | Gradient accumulation steps |
| `--output_dir` | "results" | Output directory |
| `--exp_name` | "brepformer" | Experiment name |
| `--seed` | 42 | Random seed |
| `--devices` | 1 | Number of GPUs |
| `--precision` | "32" | Training precision ("16" or "32") |
| `--fast_dev_run` | False | Quick test run |
| `--resume_from` | None | Resume an **interrupted** run: restores optimizer state, epoch counter and LR schedule. Wrong tool for continuing a finished run on new data — it starts at the old epoch and stops immediately against `--max_epochs`. |
| `--init_weights` | None | Warm start from a checkpoint's **weights only**, on a fresh optimizer and schedule. Layers whose name or shape mismatches (typically the heads, when the class count differs) stay at their initial values; the count of loaded vs skipped tensors is printed, and a large "other" skip count means the checkpoint is a different architecture. |
| `--limit_data` | None | Limit total dataset to N samples; saves manifest for test/analyze/infer |

**Callbacks:**
- `ModelCheckpoint`: saves top-3 `best-*` by `val/f1` — **model-level multi-label F1**, not per-face segmentation
- `ModelCheckpoint`: saves top-3 `bestface-*` by `val/face_f1` when `--face_segmentation` is set. Prefer these when comparing segmentation runs; the two metrics rank epochs differently, and the checkpoint that shipped as `defeature_v2.py`'s default ranks 13th of 26 on face F1
- `EarlyStopping`: `--early_stop_patience` epochs on `--early_stop_monitor`
- `LearningRateMonitor`: logs LR to TensorBoard

#### Defeature Dataset — Training

```bash
# *best* — face segmentation training on defeature data (5 classes)
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/defeature_processed \
    --face_segmentation \
    --num_face_classes 5 \
    --face_seg_weight 1.0 \
    --model_cls_weight 1.0 \
    --face_seg_hidden_dim 512 \
    --max_faces 500 \
    --max_epochs 200 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --num_workers 0 \
    --precision "32" \
    --output_dir results \
    --exp_name trial1_ss

# Quick sanity check
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/defeature_processed \
    --face_segmentation \
    --num_face_classes 5 \
    --fast_dev_run \
    --num_workers 0 \
    --batch_size 4

# Model-level classification only (no face segmentation)
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/defeature_processed \
    --max_epochs 200 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --output_dir results \
    --exp_name defeature_cls
```

#### Trials 6 / 6A — Hyperparameter Sweep

An 18-arm sweep that separates four questions earlier runs had confounded: whether the
pcq pseudo-labelled data helps, whether inverse-frequency class weighting helps, whether
rebalancing the loss toward the face head helps, and which initialisation wins. Every
treatment has a `*ctl*` twin trained without the pcq data, so a gain is attributable to
the data rather than to the settings that changed alongside it, and both existing
5-class checkpoints are used as warm starts rather than assuming either one.

Shared by every arm: `--max_faces 1000` (recovers 72% of pcq faces; at 500 the cap drops
58% of them), `--batch_size 2 --accumulate_grad_batches 16` for an effective batch of 32
matching both baselines, `--learning_rate 0.001`, `--seed 42`, and
`--lr_monitor val/face_f1 --early_stop_monitor val/face_f1 --early_stop_patience 25`.

**Why both monitors move off their defaults.** `ReduceLROnPlateau` shipped watching
`train/loss_epoch`, so it decayed the LR on *training* loss and kept cutting it while
the model overfit. `EarlyStopping` shipped watching `val/loss`, which on a
pseudo-labelled corpus mixes ground truth with teacher noise — a run could be stopped by
the noise rather than by the task. `val/face_f1` is the axis `defeature_v2.py` is
deployed on. Both are now flags with the historical values as defaults, so existing
recipes elsewhere in this file are unchanged; the sweep sets them identically for every
arm, so the comparison stays clean.

**Why the from-scratch arms matter.** `trial1_ss1500` was configured with
`warmup_steps=5000` but at `batch_size 32` on 1,248 samples it only ever reached ~2,262
batches, so it never left linear warmup: peak LR ~45% of nominal, and because the warmup
override rewrites `pg["lr"]` every batch it also erased every `ReduceLROnPlateau`
reduction. At `batch_size 2` warmup completes in ~5 epochs, so these are the first
correctly-scheduled from-scratch runs — which is why the earlier "fine-tuning beats
direct training" result needs re-testing rather than assuming.

Run the whole sweep through the runner (sequential, resumable, logs to
`results/sweep_logs/`, records wall-clock per arm in `status.json`):

```bash
# *best* — run the cheap 5-class family first (~1,900 models per arm)
python -m brepformer.sweeps.trial6_sweep --run --family 5

# then the 27-class family (~23,500 models per arm, ~10x the cost)
python -m brepformer.sweeps.trial6_sweep --run --family 27

# preview without executing
python -m brepformer.sweeps.trial6_sweep --run --family 5 --dry_run

# re-run arms already marked complete in status.json
python -m brepformer.sweeps.trial6_sweep --run --family 5 --rerun_completed
```

| Argument | Default | Description |
|---|---|---|
| `--emit` | — | Print the individual commands (the source of the two lists below) |
| `--run` | — | Execute the arms sequentially |
| `--family` | both | Restrict to `5` or `27` |
| `--dry_run` | False | Print commands without running |
| `--rerun_completed` | False | Ignore `status.json` and re-run finished arms |

##### 5-class arms (direct defeature task)

11 arms. Corpora: `trial6_processed` = defeature + pcq5 (~1,900 models); `defeature_processed` = defeature only (1,561).

| Arm | Corpus | Arm-specific flags | Purpose |
|---|---|---|---|
| `t6_base` | `trial6_processed` | _(none — baseline)_ | pcq data, baseline loss - the straight comparison to t6ctl_base |
| `t6_w` | `trial6_processed` | `--weighted_crossentropy` | pcq data + inverse-frequency class weights |
| `t6_f` | `trial6_processed` | `--face_seg_weight 2.0` `--model_cls_weight 0.5` | pcq data + loss shifted 4:1 toward face segmentation |
| `t6_wf` | `trial6_processed` | `--weighted_crossentropy` `--face_seg_weight 2.0` `--model_cls_weight 0.5` | pcq data + both loss changes (interaction cell) |
| `t6_init_def` | `trial6_processed` | `--init_weights trial1_ss1500/…` | continue the CURRENT defeature_v2 default (0.8564) on the pcq corpus |
| `t6_init_def_wf` | `trial6_processed` | `--init_weights trial1_ss1500/…` `--weighted_crossentropy` `--face_seg_weight 2.0` `--model_cls_weight 0.5` | same warm start + both loss changes |
| `t6_init_ft` | `trial6_processed` | `--init_weights trial1_finetune_ss/…` | continue the best-measured checkpoint (0.8886) on the pcq corpus |
| `t6_init_ft_wf` | `trial6_processed` | `--init_weights trial1_finetune_ss/…` `--weighted_crossentropy` `--face_seg_weight 2.0` `--model_cls_weight 0.5` | same warm start + both loss changes — the a-priori favourite |
| `t6ctl_base` | `defeature_processed` | _(none — baseline)_ | CONTROL: first properly-warmed from-scratch 5-class run; isolates the LR-schedule defect from the fine-tune advantage |
| `t6ctl_wf` | `defeature_processed` | `--weighted_crossentropy` `--face_seg_weight 2.0` `--model_cls_weight 0.5` | CONTROL: best loss settings without the pcq data |
| `t6ctl_init_ft` | `defeature_processed` | `--init_weights trial1_finetune_ss/…` | CONTROL: best warm start without the pcq data — separates 'the checkpoint was good' from 'the extra data helped' |

Every arm is the same command with those flags appended. Spelled out for `t6_base`:

```bash
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/trial6_processed \
    --num_classes 5 \
    --num_face_classes 5 \
    --max_faces 1000 \
    --batch_size 2 \
    --accumulate_grad_batches 16 \
    --learning_rate 0.001 \
    --max_epochs 200 \
    --seed 42 \
    --output_dir results \
    --face_segmentation \
    --lr_monitor val/face_f1 \
    --early_stop_monitor val/face_f1 \
    --early_stop_patience 25 \
    --exp_name t6_base
```

Print all 11 in full with `python -m brepformer.sweeps.trial6_sweep --emit --family 5`.

##### 27-class arms (collapsed 27→5 at evaluation)

7 arms. Corpora: `trial6a_processed` = mftrcad + pcq27 (~23,500 models); `mftrcad_processed` = mftrcad only.

| Arm | Corpus | Arm-specific flags | Purpose |
|---|---|---|---|
| `t6a_base` | `trial6a_processed` | _(none — baseline)_ | mftrcad + pcq27, baseline loss; collapsed 27->5 at eval |
| `t6a_w` | `trial6a_processed` | `--weighted_crossentropy` | mftrcad + pcq27 + class weights (27 classes are severely imbalanced) |
| `t6a_f` | `trial6a_processed` | `--face_seg_weight 2.0` `--model_cls_weight 0.5` | mftrcad + pcq27 + loss shifted toward face segmentation |
| `t6a_wf` | `trial6a_processed` | `--weighted_crossentropy` `--face_seg_weight 2.0` `--model_cls_weight 0.5` | mftrcad + pcq27 + both loss changes (interaction cell) |
| `t6a_init_t5` | `trial6a_processed` | `--init_weights trial5/…` | continue trial5 on the pcq-augmented 27-class corpus |
| `t6a_init_t5_wf` | `trial6a_processed` | `--init_weights trial5/…` `--weighted_crossentropy` `--face_seg_weight 2.0` `--model_cls_weight 0.5` | same warm start + both loss changes |
| `t6actl_base` | `mftrcad_processed` | _(none — baseline)_ | CONTROL: mftrcad only, same settings; isolates the pcq27 data |

Every arm is the same command with those flags appended. Spelled out for `t6a_base`:

```bash
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/trial6a_processed \
    --num_classes 27 \
    --num_face_classes 27 \
    --max_faces 1000 \
    --batch_size 2 \
    --accumulate_grad_batches 16 \
    --learning_rate 0.001 \
    --max_epochs 200 \
    --seed 42 \
    --output_dir results \
    --face_segmentation \
    --lr_monitor val/face_f1 \
    --early_stop_monitor val/face_f1 \
    --early_stop_patience 25 \
    --exp_name t6a_base
```

Print all 7 in full with `python -m brepformer.sweeps.trial6_sweep --emit --family 27`.

#### Limited Data Training (Subset Testing)

Use `--limit_data N` to train on a random subset of the full preprocessed dataset. This is useful for quick experiments and debugging without re-preprocessing. The flag:

1. Randomly selects N samples proportionally across train/val/test splits (seeded by `--seed`)
2. Saves a `limit_data_manifest.json` in the experiment output directory
3. The manifest is reused by test, analyze, and infer scripts via `--limit_data_manifest`

```bash
# Train on 1000 samples (defeature: ~800 train, ~100 val, ~100 test)
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/defeature_processed \
    --face_segmentation \
    --num_face_classes 5 \
    --face_seg_weight 1.0 \
    --model_cls_weight 1.0 \
    --max_faces 500 \
    --max_epochs 200 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --num_workers 0 \
    --output_dir results \
    --exp_name defeature_1k \
    --limit_data 1000

# Test on the same 1000-sample subset
python brepformer/test_preprocessed.py \
    --data_dir brepformer/data/defeature_processed \
    --checkpoint "results/trial1_ss1000/best-epoch=28-val/f1=0.9146.ckpt" \
    --batch_size 32 \
    --num_workers 0 \
    --output_file results/defeature_1k/test_results.json \
    --output_face_preds results/defeature_1k/face_preds.json \
    --limit_data_manifest results/defeature_1k/limit_data_manifest.json

# Analyze on the same subset
python brepformer/analyze.py \
    --checkpoint "results/trial1_ss1000/best-epoch=28-val/f1=0.9146.ckpt" \
    --data_dir brepformer/data/defeature_processed \
    --mode all \
    --output_dir results/defeature_1k/analysis \
    --limit_data_manifest results/defeature_1k/limit_data_manifest.json

# Batch inference on the same subset's STEP files
python -m brepformer.infer \
    --step_dir brepformer/data/defeature/steps/ \
    --checkpoint "results/trial1_ss1000/best-epoch=28-val/f1=0.9146.ckpt" \
    --output_dir results/defeature_1k/inference/ \
    --limit_data_manifest results/defeature_1k/limit_data_manifest.json

# Or use run_pipeline.py which auto-propagates the manifest
python brepformer/run_pipeline.py \
    --data_dir brepformer/data/defeature \
    --output_dir results/defeature_1k_pipeline \
    --face_segmentation \
    --num_face_classes 5 \
    --skip_preprocess \
    --limit_data 1000
```

**Manifest format** (`limit_data_manifest.json`):
```json
{
  "limit_data": 1000,
  "seed": 42,
  "data_dir": "brepformer/data/defeature_processed",
  "splits": {
    "train": ["00001.pkl", "00005.pkl", "..."],
    "val": ["00002.pkl", "..."],
    "test": ["00003.pkl", "..."]
  },
  "model_ids": ["model_a", "model_b", "..."]
}
```

| Script | Flag | Description |
|--------|------|-------------|
| `train_preprocessed.py` | `--limit_data N` | Create manifest and train on N-sample subset |
| `run_pipeline.py` | `--limit_data N` | Same as above; manifest auto-propagated to test/analyze |
| `test_preprocessed.py` | `--limit_data_manifest PATH` | Evaluate on the manifest's subset |
| `analyze.py` | `--limit_data_manifest PATH` | Analyze on the manifest's subset |
| `infer.py` | `--limit_data_manifest PATH` | Filter batch STEP files to the manifest's model_ids |

### `train.py` (legacy — use `train_preprocessed.py` instead)

Training with raw JSON data. Single-worker only, no face segmentation, no class weighting, no `--use_rope`. Kept for backward compatibility with raw JSON workflows.

```bash
python brepformer/train.py \
    --data_dir brepformer/data/mftrcad \
    --max_epochs 100 \
    --num_workers 0

# With external labels (single-label)
python brepformer/train.py \
    --data_dir brepformer/data/mftrcad \
    --label_file path/to/labels.json \
    --num_classes 10 \
    --no_multi_label
```

---

## Testing

### `test.py`

Evaluates a checkpoint on raw (non-preprocessed) data — the original test entry point.
Prefer `test_preprocessed.py` when a preprocessed corpus exists; it is much faster and is
what every result in README.md was produced with.

```bash
# *best* — only when no preprocessed corpus exists
python brepformer/test.py \
    --data_dir brepformer/data/mftrcad \
    --checkpoint "results/trial5/best-epoch=39-val/f1=0.8832.ckpt"
```

### `test_symmetry.py`

Checks whether predictions are stable under rigid transforms of the same part — rotate
or translate a STEP file and the per-face labels should not change. Non-trivial here
because the 14-dim face features include centroid, normal and bounding box, which *are*
frame-dependent, so a large disagreement is a real generalisation problem rather than a
bug in the test. Pairs with `rotator/pipeline.py`, which can canonicalise the frame.

```bash
# *best* — how much do predictions move when the part does?
python brepformer/test_symmetry.py \
    --step_dir brepformer/data/aman_sample \
    --checkpoint "results/trial1_finetune_ss/best-epoch=32-val/f1=0.9354.ckpt" \
    --5_classes
```

### `test_preprocessed.py` ***recommended***

Evaluate a trained model on any dataset split. When the loaded model has `face_segmentation=True`, face metrics (`test/face_acc`, `test/face_f1`) are computed and logged automatically. Use `--output_face_preds` to write per-face predictions with aggregate metrics. Use `--split train` (or `val`) to evaluate on splits other than test.

```bash
# *best* — full face segmentation evaluation with per-face predictions (test split)
python brepformer/test_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --checkpoint results/trial5/best-epoch=39-val/f1=0.8832.ckpt \
    --batch_size 32 \
    --num_workers 4 \
    --output_file preds/trial5/test_face_results.json \
    --output_face_preds preds/trial5/face_preds.json

# Evaluate on the training split (to check for overfitting)
python brepformer/test_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --checkpoint "results/trial5/best-epoch=39-val/f1=0.8832.ckpt" \
    --batch_size 32 \
    --num_workers 4 \
    --split train \
    --output_file brepformer/preds/trial5/train_face_results.json \
    --output_face_preds brepformer/preds/trial5/train_face_preds.json

# Evaluate on the validation split
python brepformer/test_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --checkpoint results/trial3/best-epoch=42-val/f1=0.8795.ckpt \
    --batch_size 32 \
    --num_workers 4 \
    --split val \
    --output_file preds/trial3/val_face_results.json

# Standard test (no face preds)
python brepformer/test_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --batch_size 32 \
    --num_workers 4 \
    --output_file test_results.json

# With per-face predictions and metrics (face segmentation models)
python brepformer/test_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --checkpoint results/face_seg_heavy/best-epoch=46-val/f1=0.8608.ckpt \
    --output_file test_face_results.json \
    --output_face_preds face_preds.json
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | required | Preprocessed data directory |
| `--checkpoint` | required | Model checkpoint path |
| `--batch_size` | 32 | Batch size |
| `--num_workers` | 4 | Data loader workers |
| `--split` | `"test"` | Dataset split to evaluate: `train`, `val`, or `test` |
| `--output_file` | None | Output JSON for model-level results |
| `--output_face_preds` | None | Output JSON for per-face predictions |
| `--8_classes` | False | Remap 27 MFTRCAD classes to 8 grouped machining feature categories |
| `--5_classes` | False | Remap 27 MFTRCAD classes to 5 defeature categories (random, hole, chamfer, fillet, cut) |
| `--limit_data_manifest` | None | Path to manifest JSON for reproducible dataset subsetting |

> **Note:** `--8_classes` and `--5_classes` are mutually exclusive. If neither is specified, the full 27-class predictions are used.

**`--output_face_preds` JSON structure:**
```json
{
  "metrics": {
    "face_accuracy": 0.85,
    "mean_iou": 0.72,
    "per_class_iou": [0.91, 0.78, 0.65, ...]
  },
  "predictions": [
    {
      "model_id": "00000001",
      "num_faces": 24,
      "face_preds": [24, 0, 0, 1, ...],
      "face_targets": [24, 0, 0, 1, ...],
      "face_probs": [[0.01, 0.02, ...], ...]
    }
  ]
}
```

#### Defeature Dataset — Testing

```bash
# *best* — full face segmentation evaluation on defeature test split
python brepformer/test_preprocessed.py \
    --data_dir brepformer/data/defeature_processed \
    --checkpoint results/trial1_ss/best-epoch=41-val/f1=0.9476.ckpt \
    --batch_size 4 \
    --num_workers 0 \
    --output_file brepformer/preds/trial1_ss/test_results.json \
    --output_face_preds brepformer/preds/trial1_ss/face_preds.json

# Train split evaluation (overfitting check)
python brepformer/test_preprocessed.py \
    --data_dir brepformer/data/defeature_processed \
    --checkpoint results/trial1_ss/best-epoch=41-val/f1=0.9476.ckpt \
    --batch_size 4 \
    --num_workers 0 \
    --split train \
    --output_file brepformer/preds/trial1_ss/train_results.json \
    --output_face_preds brepformer/preds/trial1_ss/train_face_preds.json

# Standard test
python brepformer/test_preprocessed.py \
    --data_dir brepformer/data/defeature_processed \
    --checkpoint results/defeature/best-*.ckpt \
    --batch_size 32 \
    --num_workers 4
```

---

## Inference

### `infer.py` ***recommended***

End-to-end inference on raw STEP files. Converts STEP geometry to a graph representation using pythonOCC, runs it through the trained model, and outputs per-face predictions. Requires `pythonocc-core`.

```bash
# *best* — batch inference with face segmentation model
python -m brepformer.infer \
    --step_dir brepformer/data/mftrcad/steps/ \
    --checkpoint "results/trial5/best-epoch=39-val/f1=0.8832.ckpt" \
    --output_dir inference/trial5/face_inference_results/

# Single file → JSON output (full predictions with probabilities)
python -m brepformer.infer \
    --step brepformer/data/mftrcad/steps/20240116_231044_0_result.step \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --output preds.json

# Single file → .seg output (one label per line)
python -m brepformer.infer \
    --step brepformer/data/mftrcad/steps/20240116_231044_0_result.step \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --output preds.seg

# Batch mode (writes per-model .json + .seg + combined all_predictions.json)
python -m brepformer.infer \
    --step_dir brepformer/data/mftrcad/steps/ \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --output_dir inference/trial2/

# Batch mode for face model (writes per-model .json + .seg + combined all_predictions.json)
python -m brepformer.infer \
    --step_dir brepformer/data/mftrcad/steps/ \
    --checkpoint results/face_seg_heavy/best-epoch=46-val/f1=0.8608.ckpt \
    --output_dir inference/face_seg_heavy/
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--step` | None | Path to a single STEP file |
| `--step_dir` | None | Directory of STEP files (batch mode) |
| `--checkpoint` | required | Model checkpoint path |
| `--output` | None | Output file path (.json or .seg) |
| `--output_dir` | None | Output directory (batch mode) |
| `--8_classes` | False | Remap 27 MFTRCAD classes to 8 grouped machining feature categories |
| `--5_classes` | False | Remap 27 MFTRCAD classes to 5 defeature categories (random, hole, chamfer, fillet, cut) |
| `--limit_data_manifest` | None | Path to manifest JSON to filter batch STEP files by model_id |

> **Note:** `--8_classes` and `--5_classes` are mutually exclusive. If neither is specified, the full 27-class predictions are used.

**Output formats:**
- **`.seg`**: One integer label per line. Line _i_ is the predicted class for face _i_ (TopExp_Explorer order). Directly loadable by `visualize_seg.py` and `export_freecad.py`.
- **`.json`**: Full predictions including model-level classes, per-face predictions, per-face class probabilities, and human-readable class names.

**Console output** includes a summary of predicted face counts per class:
```
Model: 00000001
Faces: 24
Model classes: ['chamfer', 'through_hole', 'stock']
Face predictions (24 faces):
  chamfer: 6
  through_hole: 4
  stock: 14
```

#### Trials 6 / 6A — Inference

`--5_classes` collapses a 27-class head via `CLASS_TO_DEFEATURE`, so the same command
works for both families and their outputs are directly comparable.

```bash
# *best* — 5-class arm on real parts
python brepformer/infer.py \
    --step_dir brepformer/data/aman_sample \
    --checkpoint "$(ls results/t6_wf/bestface-*.ckpt | tail -1)" --5_classes \
    --output_dir brepformer/inference/t6_wf

# 27-class arm, collapsed to 5 for comparison
python brepformer/infer.py \
    --step_dir brepformer/data/aman_sample \
    --checkpoint "$(ls results/t6a_base/bestface-*.ckpt | tail -1)" --5_classes \
    --output_dir brepformer/inference/t6a_base

# 27-class arm in its native space
python brepformer/infer.py \
    --step_dir brepformer/data/aman_sample \
    --checkpoint "$(ls results/t6a_base/bestface-*.ckpt | tail -1)" \
    --output_dir brepformer/inference/t6a_base_27
```

#### Defeature Dataset — Inference

```bash
# *best* — batch inference on defeature STEP files
python -m brepformer.infer \
    --step_dir brepformer/data/defeature/steps/ \
    --checkpoint results/trial1_ss/best-epoch=41-val/f1=0.9476.ckpt \
    --output_dir inference/trial1_ss/

# Single defeature STEP file
python -m brepformer.infer \
    --step brepformer/data/defeature/steps/chicane.step \
    --checkpoint "results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt" \
    --output preds.seg
```

---

## FreeCAD Export

### `export_freecad.py` ***recommended***

Writes colored STEP AP214 files using pythonOCC's XCAF framework (`STEPCAFControl_Writer` + `XCAFDoc_ColorSurf`). Each face is assigned a surface color matching its predicted or ground-truth class. FreeCAD reads these colors natively — no plugins required. Requires `pythonocc-core`.

```bash
# *best* — batch colored export from face segmentation inference results
python -m brepformer.export_freecad \
    --step_dir brepformer/data/mftrcad/steps/ \
    --seg_dir inference/trial5/face_inference_results/ \
    --output_dir brepformer/pipeline_output/colored/

# From a .seg label file
python -m brepformer.export_freecad \
    --step brepformer/data/mftrcad/steps/20240116_231044_0_result.step \
    --seg preds.seg \
    --output colored.step

# From live inference (runs model on the fly)
python -m brepformer.export_freecad \
    --step brepformer/data/sample/steps/20240116_231044_0_result.step \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --output colored.step

# From live inference for the face model
python -m brepformer.export_freecad \
    --step brepformer/data/sample/steps/20240116_231044_0_result.step \
    --checkpoint results/face_seg_heavy/best-epoch=46-val/f1=0.8608.ckpt \
    --output colored.step
    
# From ground-truth labels JSON
python -m brepformer.export_freecad \
    --step brepformer/data/mftrcad/steps/00000001.step \
    --labels_json brepformer/data/mftrcad/labels/00000001_result.json \
    --output colored_gt.step

# Auto-match GT labels by model_id from labels directory
python -m brepformer.export_freecad \
    --step brepformer/data/mftrcad/steps/00000001.step \
    --labels_dir brepformer/data/mftrcad/labels/ \
    --output colored_gt.step

# Batch mode with GT labels
python -m brepformer.export_freecad \
    --step_dir brepformer/data/mftrcad/steps/ \
    --labels_dir brepformer/data/mftrcad/labels/ \
    --output_dir brepformer/pipeline_output/colored/

# Batch mode with inference
python -m brepformer.export_freecad \
    --step_dir brepformer/data/mftrcad/steps/ \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --output_dir brepformer/pipeline_output/colored/

# From predicted .seg files in a directory
python -m brepformer.export_freecad \
    --step brepformer/data/sample/steps/20240116_231044_0_result.step \
    --seg_dir inference/trial2/ \
    --output colored.step

# Batch mode with predicted .seg files from face inference
python -m brepformer.export_freecad \
    --step_dir brepformer/data/sample/steps/ \
    --seg_dir inference/trial5/face_inference_results/ \
    --output_dir brepformer/pipeline_output/colored/
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--step` | None | Path to a single STEP file |
| `--step_dir` | None | Directory of STEP files (batch mode) |
| `--seg` | None | Path to .seg label file |
| `--seg_dir` | None | Directory of .seg prediction files (auto-matches by model_id) |
| `--checkpoint` | None | Model checkpoint (runs inference on the fly) |
| `--labels_json` | None | Path to GT labels JSON (`_result.json`) |
| `--labels_dir` | None | Directory of GT label JSON files (auto-matches by model_id) |
| `--output` | None | Output STEP file path (default: `{input}.colored.step`) |
| `--output_dir` | None | Output directory (batch mode, default: `{step_dir}/colored/`) |
| `--8_classes` | False | Remap 27 MFTRCAD classes to 8 grouped machining feature categories |
| `--5_classes` | False | Remap 27 MFTRCAD classes to 5 defeature categories (random, hole, chamfer, fillet, cut) |

> **Note:** `--8_classes` and `--5_classes` are mutually exclusive. If neither is specified, the full 27-class predictions are used.

**Label source priority** (first match wins): `--seg` > `--seg_dir` > `--labels_json` > `--labels_dir` > `--checkpoint`.

**FreeCAD workflow:** Open the colored `.step` file in FreeCAD (File → Open). Each face displays its predicted class color. See [viewer.md](viewer.md) for the 27-class color reference table.

#### Defeature Dataset — FreeCAD Export

```bash
# Batch colored export from defeature inference results
python -m brepformer.export_freecad \
    --step_dir brepformer/data/defeature/steps/ \
    --seg_dir inference/trial1_ss1500/ \
    --output_dir brepformer/pipeline_output/colored_defeature/

# Single model from GT labels
python -m brepformer.export_freecad \
    --step brepformer/data/defeature/steps/uh62382.step \
    --labels_json brepformer/data/defeature/labels/uh62382.json \
    --output colored_gt.step

# Live inference on a single defeature STEP file
python -m brepformer.export_freecad \
    --step brepformer/data/defeature/steps/uh62382.step \
    --checkpoint "results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt" \
    --output colored.step
```

---

## Automatic Defeaturing

Two defeaturing engines are available. Both have the same CLI interface and output format:

| Engine | Module | pythonocc | Key feature |
|---|---|---|---|
| **v2** (recommended) | `brepformer.defeature_v2` | >= 7.9 | Adaptive tolerance, history tracking, enhanced healing |
| v1 | `brepformer.defeature` | >= 7.5 | Original 5-phase progressive engine |

The `scripts/run_defeature.sh` wrapper selects the engine via `--engine v1|v2` (default: v2).

### `defeature_v2.py` ***recommended — modern engine with adaptive tolerance***

Drop-in replacement for `defeature.py` with six improvements that increase per-face removal success rate:

1. **Adaptive fuzzy tolerance** — retries with `SetFuzzyValue()` at 0 → 1e-5 → 5e-5 → 1e-4 → 5e-4 → 1e-3
2. **History-based face tracking** — uses `BRepTools_History` instead of `IndexedMap.Contains()` heuristics
3. **Pre-validation** — auto-repairs input shapes with `ShapeFix_Shape` + `ShapeFix_ShapeTolerance`
4. **Enhanced healing** — tolerance harmonization → shape fix → conditional sewing → face unification with angular/linear tolerance
5. **Area-based ordering** — removes smallest features first (higher success rate, unblocks larger features)
6. **Intermediate healing** — `UnifySameDomain` between single-face removals in Phase 5c

```bash
# *best* — v2 engine via script wrapper
./scripts/run_defeature.sh --step model.step --verbose

# Or directly
python -m brepformer.defeature_v2 --step model.step

# Batch mode
python -m brepformer.defeature_v2 --step_dir brepformer/data/defeature/steps/

# Higher tolerance for complex fillet models
python -m brepformer.defeature_v2 --step model.step --max_fuzzy 0.01 --verbose

# From pre-computed .seg predictions
python -m brepformer.defeature_v2 --step model.step --seg preds.seg

# Batch with pre-computed predictions + colored output
python -m brepformer.defeature_v2 \
    --step_dir brepformer/data/defeature/steps/ \
    --seg_dir inference/trial1_ss1500/ \
    --output_dir brepformer/defeatured_output/trial1_ss1500 \
    --max_fuzzy 0.01
    --save_colored --verbose
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--step` | None | Path to a single STEP file |
| `--step_dir` | None | Directory of STEP files (batch mode) |
| `--checkpoint` | `results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt` | Model checkpoint (must have face segmentation head) |
| `--seg` | None | Path to .seg file with pre-computed predictions (single mode) |
| `--seg_dir` | None | Directory of .seg prediction files (batch mode) |
| `--output_dir` | `brepformer/defeatured_output` | Output directory for defeatured STEP files |
| `--save_colored` | False | Also save a colored STEP showing predictions before defeaturing |
| `--verbose` | False | Print detailed per-phase defeaturing progress |
| `--max_fuzzy` | `1e-3` | Maximum fuzzy tolerance for adaptive retry (v2 only) |

**Phase algorithm (v2):**
```
Phase 0 — Pre-validate input shape (auto-repair if needed)
Phase 1 — Try ALL features at once (adaptive tolerance)
Phase 2 — Progressive type-by-type: fillet → chamfer → hole → cut (adaptive)
Phase 3 — Connected components per failed type (area-sorted, adaptive)
Phase 4 — Retry on modified shape (history-based face tracking, adaptive)
Phase 5 — Last-ditch: alternate orders, single-face iterative (adaptive, intermediate healing)
Cleanup — Multi-stage healing pipeline
```

#### Trials 6 / 6A — Segmentation + Defeaturing

```bash
# *best* — defeature real parts with a sweep checkpoint, keeping colored output
python brepformer/defeature_v2.py \
    --step_dir brepformer/data/aman_sample \
    --checkpoint "$(ls results/t6_wf/bestface-*.ckpt | tail -1)" \
    --output_dir brepformer/defeatured_output/t6_wf \
    --save_colored --primitives --verbose

# 27-class arm
python brepformer/defeature_v2.py \
    --step_dir brepformer/data/aman_sample \
    --checkpoint "$(ls results/t6a_base/bestface-*.ckpt | tail -1)" \
    --output_dir brepformer/defeatured_output/t6a_base \
    --save_colored --primitives --verbose

# reuse the .seg files already written by infer.py (skips re-inference)
python brepformer/defeature_v2.py \
    --step_dir brepformer/data/aman_sample \
    --seg_dir brepformer/inference/t6_wf \
    --output_dir brepformer/defeatured_output/t6_wf_from_seg \
    --save_colored
```

### `defeature.py` (v1 — original engine)

Original defeaturing pipeline with 5-phase progressive face removal. Same CLI as v2 except without `--max_fuzzy`.

```bash
# *best* -- defeature a single STEP file (uses default defeature checkpoint)
python -m brepformer.defeature \
    --step model.step

# Defeature a directory of STEP files
python -m brepformer.defeature \
    --step_dir brepformer/data/defeature/steps/

# Custom checkpoint and output directory
python -m brepformer.defeature \
    --step model.step \
    --checkpoint "results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt" \
    --output_dir my_defeatured/

# From pre-computed .seg predictions (skip inference)
python -m brepformer.defeature \
    --step model.step \
    --seg preds.seg

# Batch mode with pre-computed .seg files
python -m brepformer.defeature \
    --step_dir brepformer/data/defeature/steps/ \
    --seg_dir inference/trial1_ss1500/

# Also save colored STEP for visual comparison
python -m brepformer.defeature \
    --step model.step \
    --save_colored --verbose
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--step` | None | Path to a single STEP file |
| `--step_dir` | None | Directory of STEP files (batch mode) |
| `--checkpoint` | `results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt` | Model checkpoint (must have face segmentation head) |
| `--seg` | None | Path to .seg file with pre-computed predictions (single mode) |
| `--seg_dir` | None | Directory of .seg prediction files (batch mode) |
| `--output_dir` | `brepformer/defeatured_output` | Output directory for defeatured STEP files |
| `--save_colored` | False | Also save a colored STEP showing predictions before defeaturing |
| `--verbose` | False | Print detailed defeaturing progress |

**Output structure:**
```
brepformer/defeatured_output/
├── model_id_defeatured.step    # Defeatured STEP file
├── model_id_colored.step       # Colored predictions (if --save_colored)
├── model_id_report.json        # Per-model defeaturing report
└── batch_report.json           # Batch summary (batch mode only)
```

**Report JSON fields:**
| Field | Description |
|-------|-------------|
| `status` | `"success"`, `"no_features"`, or `"error"` |
| `num_faces` | Total faces in input model |
| `kept` | Faces classified as random (kept) |
| `removed` | Feature faces successfully removed |
| `failed` | Feature faces that could not be removed |
| `valid` | Whether the output shape passes BRepCheck validation |

**Integration with other scripts:**

Both `export_freecad.py` and `visualize_seg.py` support a `--defeature` flag that delegates to this module:

```bash
# Export colored STEP + defeatured STEP in one command
python -m brepformer.export_freecad \
    --step model.step \
    --checkpoint "results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt" \
    --defeature

# In the viewer, press F to defeature the current model
python -m brepformer.visualize_seg \
    --step model.step \
    --checkpoint "results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt" \
    --defeature
```

#### Defeature Dataset -- Full Defeaturing Pipeline

```bash
# Step 1: Run inference on defeature STEP files
python -m brepformer.infer \
    --step_dir brepformer/data/defeature/steps/ \
    --checkpoint "results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt" \
    --output_dir inference/trial1_ss1500/

# Step 2: Defeature using inference results (v2 engine via script)
./scripts/run_defeature.sh \
    --step_dir brepformer/data/defeature/steps/ \
    --seg_dir inference/trial1_ss1500/ \
    --output_dir brepformer/defeatured_output/ \
    --save_colored --verbose

# Or using v2 module directly
python -m brepformer.defeature_v2 \
    --step_dir brepformer/data/defeature/steps/ \
    --seg_dir inference/trial1_ss1500/ \
    --output_dir brepformer/defeatured_output/ \
    --save_colored --verbose

# Or combine inference + defeaturing in one command
python -m brepformer.defeature_v2 \
    --step_dir brepformer/data/defeature/steps/ \
    --checkpoint "results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt" \
    --output_dir brepformer/defeatured_output/

# Using v1 engine for comparison
./scripts/run_defeature.sh --engine v1 \
    --step_dir brepformer/data/defeature/steps/ \
    --seg_dir inference/trial1_ss1500/ \
    --output_dir brepformer/defeatured_output_v1/ \
    --verbose
```

---

## Primitive Fitting

### `brepsimplifier/test_data/compare_primitives.py` and `test_partfield/compare_primitives.py`

Two standalone comparisons of primitive-fitting backends on their own bundled test data.
They are experiment scratchpads rather than pipeline stages: each is run from inside its
own directory and writes its comparison there.

```bash
# *best* — compare fitted primitives on the brepsimplifier test set
python brepformer/brepsimplifier/test_data/compare_primitives.py

# the PartField variant, on its own test data
python brepformer/test_partfield/compare_primitives.py
```

### `brepsimplifier/primitives.py` (CLI)

Fit geometric primitives (box, cylinder, cone, sphere, torus) to a STEP file. BSpline/freeform surfaces are handled in two ways depending on mode: single enclosing prisms (standard) or UV-grid tessellation into many small rectangular prisms (lenient).

See [`brepformer/brepsimplifier/README.md`](brepsimplifier/README.md) and [`brepformer/brepsimplifier/SCRIPTS.md`](brepsimplifier/SCRIPTS.md) for full documentation.

```bash
# Direct mode: fit primitives to a defeatured STEP file
python -m brepformer.brepsimplifier.primitives \
    --step defeatured_model.step \
    --output output_primitives.step --verbose

# Direct + lenient: tessellate BSpline faces (for non-defeatured originals)
python -m brepformer.brepsimplifier.primitives \
    --step original_model.step \
    --output output_primitives.step --lenient --verbose

# Pipeline mode: defeature → primitives (lenient auto-activates on failure)
python -m brepformer.brepsimplifier.primitives \
    --step raw_model.step --pipeline \
    --output_dir pipeline_output/ --verbose
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--step` | required | Input STEP file |
| `--output` | `{stem}_primitives.step` | Output STEP path (direct mode) |
| `--pipeline` | false | Run defeature → primitives pipeline |
| `--checkpoint` | None | Model checkpoint (pipeline mode) |
| `--seg` | None | Pre-computed .seg predictions (pipeline mode) |
| `--output_dir` | `brepformer/primitives_output` | Output directory (pipeline mode) |
| `--tol` | 0.1 | Geometric tolerance for face grouping |
| `--max_fuzzy` | 1e-3 | Max fuzzy tolerance for defeaturing |
| `--lenient` | false | Tessellate BSpline faces into small prisms |
| `--save_colored` | false | Save colored predictions STEP (pipeline) |
| `--save_intermediate` | false | Keep defeatured STEP (pipeline) |
| `--verbose` | false | Detailed progress output |

### `scripts/run_primitives.sh`

Shell wrapper for `primitives.py`. Same arguments.

```bash
./brepformer/scripts/run_primitives.sh --step model.step --lenient --verbose
```

---

## Analysis

### `gen_script_index.py`

Regenerates the "Complete Script Index" appendix at the end of this file: one row per
runnable script, linking to the section that documents it, and an explicit mark for any
script with no recorded invocation. Run it after adding a script so nothing goes
untracked; `--check` fails instead of writing, for CI.

```bash
# *best* — refresh the index after adding or documenting a script
python -m brepformer.gen_script_index

# fail if the index is stale, without modifying anything
python -m brepformer.gen_script_index --check
```

### `analyze.py` ***recommended***

Comprehensive model and prediction analysis with multiple modes. Includes a `face_segmentation` mode for per-face metrics (per-class IoU, mean IoU, per-face accuracy, 27x27 confusion matrix).

```bash
# step inference analysis comparing preprocessed vs STEP conversion paths

# *best* 
python brepformer/analyze.py     
    --checkpoint "results/trial5/best-epoch=39-val/f1=0.8832.ckpt"     
    --data_dir brepformer/data/mftrcad_processed     --step_dir brepformer/data/mftrcad/steps     --mode all     
    --output_dir brepformer/analysis_results/trial5

# Full analysis including face segmentation metrics
python brepformer/analyze.py \
    --checkpoint "results/trial5/best-epoch=39-val/f1=0.8832.ckpt" \
    --data_dir brepformer/data/mftrcad_processed \
    --step_dir brepformer/data/mftrcad/steps \
    --mode all \
    --output_dir analysis_results

# Architecture analysis (no data needed)
python brepformer/analyze.py \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --mode architecture \
    --output_dir analysis_results

# Per-class model-level performance
python brepformer/analyze.py \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --data_dir brepformer/data/mftrcad_processed \
    --mode per_class \
    --output_dir analysis_results

# Embedding visualization (t-SNE)
python brepformer/analyze.py \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --data_dir brepformer/data/mftrcad_processed \
    --mode embeddings \
    --num_samples 1000 \
    --output_dir analysis_results

# Detailed per-sample predictions
python brepformer/analyze.py \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --data_dir brepformer/data/mftrcad_processed \
    --mode predictions \
    --output_dir analysis_results

# Confusion matrix
python brepformer/analyze.py \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --data_dir brepformer/data/mftrcad_processed \
    --mode confusion_matrix \
    --output_dir analysis_results

# Face segmentation metrics (requires face_segmentation model)
python brepformer/analyze.py \
    --checkpoint results/face_seg_heavy/best-epoch=46-val/f1=0.8608.ckpt \
    --data_dir brepformer/data/mftrcad_processed \
    --mode face_segmentation \
    --output_dir analysis_results

# Run all analyses
python brepformer/analyze.py \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --data_dir brepformer/data/mftrcad_processed \
    --mode all \
    --output_dir analysis_results
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | required | Model checkpoint |
| `--data_dir` | None | Data directory (required for most modes) |
| `--mode` | "architecture" | Analysis mode |
| `--output_dir` | "analysis_results" | Output directory |
| `--batch_size` | 32 | Batch size |
| `--num_workers` | 4 | Data loader workers |
| `--num_samples` | 1000 | Samples for embedding analysis |
| `--split` | "test" | Data split to analyze ("train", "val", "test") |
| `--8_classes` | False | Remap 27 MFTRCAD classes to 8 grouped machining feature categories |
| `--5_classes` | False | Remap 27 MFTRCAD classes to 5 defeature categories (random, hole, chamfer, fillet, cut) |
| `--step_dir` | None | Directory of STEP files (required for `step_inference` mode) |

> **Note:** `--8_classes` and `--5_classes` are mutually exclusive. If neither is specified, the full 27-class predictions are used.
| `--max_models` | 50 | Max models to evaluate in `step_inference` mode |
| `--limit_data_manifest` | None | Path to manifest JSON for reproducible dataset subsetting |

| Mode | Description | Output files |
|------|-------------|--------------|
| `architecture` | Model config, parameter counts, per-module breakdown, model size | `architecture.json` |
| `per_class` | Per-class precision, recall, F1, support; co-occurrence matrix | `per_class_metrics.json`, `co_occurrence_matrix.npy` |
| `embeddings` | Graph embeddings: raw (256D), PCA (50D), t-SNE (2D) | `embeddings.npy`, `embeddings_pca50.npy`, `embeddings_tsne.npy` |
| `predictions` | Per-sample predictions with probabilities, Jaccard similarity | `predictions.json` |
| `confusion_matrix` | Standard confusion matrix (single-label) or co-occurrence (multi-label) | `confusion_matrix.npy` or `co_occurrence_matrix.npy` |
| `face_segmentation` | Per-face accuracy, per-class IoU, mean IoU, 27x27 confusion matrix | `face_seg_metrics.json`, `face_seg_confusion.npy` |
| `step_inference` | Compares preprocessed pickle vs step_to_graph inference paths on test models. Reports face accuracy, F1, precision, recall for each path. Requires `--step_dir`. | `step_inference_results.json` |
| `all` | Runs all of the above (except `step_inference`) | All output files |

### `scan_checkpoints.py`

Ranks every saved checkpoint of one or more runs by **face-level** F1 on a held-out
split. Use it before trusting a `best-*.ckpt` filename: `ModelCheckpoint` monitors
`val/f1`, which is model-level multi-label F1, and the two orderings disagree — the
checkpoint shipped as `defeature_v2.py`'s default ranked 13th of 26 on face F1. The test
split is loaded and prepared once and reused across checkpoints, so scanning 30 of them
costs about as much as 2-3 `eval_per_class.py` runs.

```bash
# *best* — rank every 5-class checkpoint on the real hand-labelled test set
python -m brepformer.scan_checkpoints \
    --ckpt_dirs results/trial1_ss results/trial1_ss1000 results/trial1_ss1500 \
                results/trial1_finetune_ss results/trial2_finetune_ss results/trial5 \
    --data_dir brepformer/data/defeature_processed --split test --label_space 5 \
    --out brepformer/analysis_results/ckpt_scan_defeature_test.json

# rank the sweep arms
python -m brepformer.scan_checkpoints \
    --ckpt_dirs results/t6_base results/t6_w results/t6_f results/t6_wf \
                results/t6_init_def results/t6_init_ft \
                results/t6ctl_base results/t6ctl_wf \
    --data_dir brepformer/data/defeature_processed --split test --label_space 5 \
    --out brepformer/analysis_results/ckpt_scan_trial6.json

# 27-class arms in their native space (in-domain check)
python -m brepformer.scan_checkpoints \
    --ckpt_dirs results/t6a_base results/t6a_w results/t6a_f results/t6a_init_t5 \
    --data_dir brepformer/data/mftrcad_processed --split test --label_space 27 \
    --out brepformer/analysis_results/ckpt_scan_trial6a_mftrcad.json
```

| Argument | Default | Description |
|---|---|---|
| `--ckpt_dirs` | required | Run directories, searched recursively (the `/` in `val/f1` makes each checkpoint a *directory*) |
| `--data_dir` | `defeature_processed` | Preprocessed corpus to score against |
| `--split` | test | Split to score |
| `--limit` | none | Cap test models; leave unset — subsampling reorders near-ties |
| `--label_space` | 5 | `5` collapses a 27-class head's predictions; targets are left alone if the corpus is already 5-class |
| `--skip_last` | False | Exclude `last*.ckpt` (included by default, since the metric that pruned the others was the wrong one) |

### `eval_per_class.py`

Per-class precision/recall/F1 for a single checkpoint, plus what each weak class is
confused *with*, its mean face area, and the model's confidence when it predicts it.
`--focus <class_id>` breaks recall down by face-area decile — the number that predicts
whether a feature class transfers to real parts, since a class can score well overall and
still be dead in deployment if training only ever showed it at one scale.

```bash
# *best* — per-class face report on the real hand-labelled test set
python -m brepformer.eval_per_class \
    --checkpoint "results/trial1_finetune_ss/best-epoch=32-val/f1=0.9354.ckpt" \
    --data_dir brepformer/data/defeature_processed --split test --label_space 5

# a 27-class checkpoint, collapsed for comparison (predictions only; the
# defeature targets are already 5-class and are left alone)
python -m brepformer.eval_per_class \
    --checkpoint "results/trial5/best-epoch=39-val/f1=0.8832.ckpt" \
    --data_dir brepformer/data/defeature_processed --split test --label_space 5

# why is a class weak? recall vs face area, in deciles
python -m brepformer.eval_per_class \
    --checkpoint "results/trial5/best-epoch=39-val/f1=0.8832.ckpt" \
    --data_dir brepformer/data/mftrcad_processed --split test --focus 26
```

### `sweeps/collect_results.py`

Scores every sweep arm on the same held-out split and writes the markdown tables in
`RESULTS_TRIAL6.md`. Reads the arm list from `trial6_sweep.py`, so the settings column
cannot drift from what was actually executed.

```bash
# *best* — score all arms + both baselines, write the results doc
python -m brepformer.sweeps.collect_results \
    --out brepformer/RESULTS_TRIAL6.md \
    --json_out brepformer/analysis_results/trial6_sweep_results.json
```

#### Defeature Dataset — Analysis

```bash
# *best* — full analysis on defeature model
python brepformer/analyze.py \
    --checkpoint results/trial1_ss/best-epoch=41-val/f1=0.9476.ckpt \
    --data_dir brepformer/data/defeature_processed \
    --mode all \
    --output_dir brepformer/analysis_results/defeature

# Face segmentation metrics (5-class confusion matrix)
python brepformer/analyze.py \
    --checkpoint "results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt" \
    --data_dir brepformer/data/defeature_processed \
    --mode face_segmentation \
    --output_dir analysis_results/defeature

# Step inference comparison
python brepformer/analyze.py \
    --checkpoint "results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt" \
    --data_dir brepformer/data/defeature_processed \
    --step_dir brepformer/data/defeature/steps \
    --mode step_inference \
    --max_models 50 \
    --output_dir analysis_results/defeature
```

**`face_seg_metrics.json` structure:**
```json
{
  "face_accuracy": 0.85,
  "mean_iou": 0.72,
  "per_class": [
    {
      "class_id": 0,
      "class_name": "chamfer",
      "precision": 0.91,
      "recall": 0.88,
      "f1": 0.89,
      "iou": 0.81,
      "support": 3420
    }
  ]
}
```

---

## Visualization

### `visualize.py`

Generate matplotlib plots from analysis results.

```bash
# *best* — all plots from face segmentation analysis
python brepformer/visualize.py \
    --mode all \
    --input_dir analysis_results \
    --log_dir results/face_seg_best \
    --output_dir plots \
    --format png --dpi 150

# Embedding visualization
python brepformer/visualize.py \
    --mode embeddings \
    --input_dir analysis_results \
    --output_dir plots

# Per-class metrics
python brepformer/visualize.py \
    --mode metrics \
    --input_dir analysis_results \
    --output_dir plots

# Training curves
python brepformer/visualize.py \
    --mode training \
    --log_dir results/brepformer \
    --output_dir plots

# All visualizations
python brepformer/visualize.py \
    --mode all \
    --input_dir analysis_results \
    --log_dir results/brepformer \
    --output_dir plots \
    --format png \
    --dpi 150
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | "all" | Visualization mode ("embeddings", "metrics", "training", "all") |
| `--input_dir` | "analysis_results" | Analysis results directory |
| `--log_dir` | "results/brepformer" | TensorBoard logs directory |
| `--output_dir` | "plots" | Output directory for image files |
| `--format` | "png" | Image format ("png", "pdf", "svg") |
| `--dpi` | 150 | Image DPI |

### `visualize_seg.py`

Interactive Qt + pythonOCC 3D viewer for visualizing 27 MFTRCAD machining feature classes on STEP models. Displays actual B-rep geometry with per-face coloring. Requires `pythonocc-core` and `PyQt5`.

```bash
# *best* — browse dataset with face seg predictions vs ground truth
python -m brepformer.visualize_seg \
    --step_dir brepformer/data/mftrcad/steps/ \
    --labels_dir brepformer/data/mftrcad/labels/ \
    --seg_dir inference/trial5/face_inference_results/

# View with predicted labels from .seg file
python -m brepformer.visualize_seg \
    --step brepformer/data/defeature/steps/uh62382.step \
    --seg preds.seg

# View with ground-truth labels from JSON
python -m brepformer.visualize_seg \
    --step brepformer/data/defeature/steps/uh62382.step \
    --labels_json brepformer/data/mftrcad/labels/00000001_result.json

# Run inference and display
python -m brepformer.visualize_seg \
    --step brepformer/data/defeature/steps/uh62382.step \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt

# Browse a dataset directory with auto-loaded GT labels
python -m brepformer.visualize_seg \
    --step_dir brepformer/data/mftrcad/steps/ \
    --labels_dir brepformer/data/mftrcad/labels/

# Browse with live inference enabled
python -m brepformer.visualize_seg \
    --step_dir brepformer/data/mftrcad/steps/ \
    --labels_dir brepformer/data/mftrcad/labels/ \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt

# Browse with predicted .seg files from inference
python -m brepformer.visualize_seg \
    --step_dir brepformer/data/sample/steps/ \
    --seg_dir inference/trial2/

# Browse with predicted .seg files from face inference + GT labels
python -m brepformer.visualize_seg \
    --step_dir brepformer/data/sample/steps/ \
    --labels_dir brepformer/data/sample/labels/ \
    --seg_dir inference/trial5/face_inference_results/

# The same but for all data
python -m brepformer.visualize_seg \
    --step_dir brepformer/data/mftrcad/steps \
    --labels_dir brepformer/data/sample/labels/ \
    --seg_dir inference/trial5/face_inference_results/


# Browse with predicted .seg files from face inference + GT labels + 8 grouped classes
python -m brepformer.visualize_seg \
    --step_dir brepformer/data/sample/steps/ \
    --labels_dir brepformer/data/sample/labels/ \
    --seg_dir inference/trial5/face_inference_results/ \
    --8_classes

# Browse with predicted .seg files from face inference + GT labels + 8 grouped classes 
python -m brepformer.visualize_seg \
    --step_dir brepformer/data/defeature/steps/ \
    --labels_dir brepformer/data/defeature/labels/ \
    --seg_dir brepformer/preds/trial1_ss1500/ \
    --8_classes

# Browse with 5 defeature classes
python -m brepformer.visualize_seg \
    --step_dir brepformer/data/defeature/steps/ \
    --labels_dir brepformer/data/defeature/labels/ \
    --seg_dir brepformer/preds/trial1_ss1500/ \
    --5_classes
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--step` | None | Path to a single STEP file |
| `--seg` | None | Path to .seg label file (predicted labels) |
| `--seg_dir` | None | Directory of .seg prediction files (auto-matches by model_id) |
| `--labels_json` | None | Path to GT labels JSON file |
| `--labels_dir` | None | Directory of GT label JSON files (auto-matches by model_id) |
| `--checkpoint` | None | Model checkpoint (enables "Run Inference" button) |
| `--step_dir` | None | Directory of STEP files (enables batch prev/next navigation) |
| `--8_classes` | False | Remap 27 MFTRCAD classes to 8 grouped machining feature categories |
| `--5_classes` | False | Remap 27 MFTRCAD classes to 5 defeature categories (random, hole, chamfer, fillet, cut) |

> **Note:** `--8_classes` and `--5_classes` are mutually exclusive. If neither is specified, the full 27-class predictions are used.

See [viewer.md](viewer.md) for keyboard shortcuts, display modes, and the 27-class color reference.

#### Trials 6 / 6A — Visualization

```bash
# *best* — browse real parts with the sweep arm's predictions
python -m brepformer.visualize_seg \
    --step_dir brepformer/data/aman_sample \
    --seg_dir brepformer/inference/t6_wf \
    --5_classes

# compare a 27-class arm's collapsed predictions on the same parts
python -m brepformer.visualize_seg \
    --step_dir brepformer/data/aman_sample \
    --seg_dir brepformer/inference/t6a_base \
    --5_classes
```

#### Defeature Dataset — Visualization

```bash
# Browse defeature dataset with GT labels
python -m brepformer.visualize_seg \
    --step_dir brepformer/data/defeature/steps/ \
    --labels_dir brepformer/data/defeature/labels/

# Browse with predicted .seg files from inference
python -m brepformer.visualize_seg \
    --step_dir brepformer/data/defeature/steps/ \
    --labels_dir brepformer/data/defeature/labels/ \
    --seg_dir inference/trial1_ss/

# Single defeature model with live inference
python -m brepformer.visualize_seg \
    --step brepformer/data/defeature/steps/chicane.step \
    --checkpoint "results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt"
    --labels_dir brepformer/data/defeature/labels/ \
    --seg_dir inference/defeature/

# Plots from defeature analysis
python brepformer/visualize.py \
    --mode all \
    --input_dir analysis_results/defeature \
    --log_dir results/defeature \
    --output_dir plots/defeature
```

### `visualize_defeature.py`

Interactive Qt + pythonOCC 3D viewer for comparing original STEP models side-by-side with their defeatured counterparts. Shows the original model with per-face class coloring (left) next to the defeatured clean stock shape (right). Requires `pythonocc-core` and `PyQt5`.

```bash
# *best* — browse defeatured results with GT labels
python -m brepformer.visualize_defeature \
    --step_dir brepformer/data/defeature/steps/ \
    --defeatured_dir brepformer/defeatured_output/trial1_ss1500 \
    --labels_dir brepformer/data/defeature/labels/

# Single model comparison
python -m brepformer.visualize_defeature \
    --step brepformer/data/defeature/steps/uh62382.step \
    --defeatured brepformer/defeatured_output/uh62382_defeatured.step

# With predicted .seg labels on original
python -m brepformer.visualize_defeature \
    --step brepformer/data/defeature/steps/uh62382.step \
    --defeatured brepformer/defeatured_output/uh62382_defeatured.step \
    --seg inference/trial1_ss1500/uh62382.seg

# Browse with predictions coloring the original
python -m brepformer.visualize_defeature \
    --step_dir brepformer/data/defeature/steps/ \
    --defeatured_dir brepformer/defeatured_output/ \
    --seg_dir inference/trial1_ss1500/

# With live inference on original model
python -m brepformer.visualize_defeature \
    --step brepformer/data/defeature/steps/uh62382.step \
    --defeatured brepformer/defeatured_output/uh62382_defeatured.step \
    --checkpoint "results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt"

# Browse with defeaturing reports (shows removed/failed stats)
python -m brepformer.visualize_defeature \
    --step_dir brepformer/data/defeature/steps/ \
    --defeatured_dir brepformer/defeatured_output/ \
    --labels_dir brepformer/data/defeature/labels/ \
    --report_dir brepformer/defeatured_output/
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--step` | None | Path to a single original STEP file |
| `--step_dir` | None | Directory of original STEP files (batch navigation) |
| `--defeatured` | None | Path to a single defeatured STEP file |
| `--defeatured_dir` | None | Directory of defeatured STEP files (auto-matches `{model_id}_defeatured.step`) |
| `--seg` | None | Path to .seg label file for coloring original faces |
| `--seg_dir` | None | Directory of .seg prediction files (auto-matches by model_id) |
| `--labels_json` | None | Path to GT labels JSON file |
| `--labels_dir` | None | Directory of GT label JSON files (auto-matches by model_id) |
| `--checkpoint` | None | Model checkpoint (enables live inference on original) |
| `--report_dir` | None | Directory of `{model_id}_report.json` defeaturing reports |

**Keyboard controls:** Same as `visualize_seg.py` — D/RIGHT (next model), A/LEFT (prev model), T (toggle GT/Pred on original), I (face info), S (summary with defeaturing stats), Q/ESC (quit).

**Display layout:**
- **Left viewer**: Original STEP model with per-face class coloring (5-class defeature palette)
- **Right viewer**: Defeatured STEP model in uniform gray (all features removed)

---

## TensorBoard

### Trials 6 / 6A — TensorBoard

All sweep arms log under `results/<arm>/version_*`, so one server compares them. Face
metrics (`val/face_acc`, `val/face_f1`) are the ones to read; `val/f1` is model-level.

```bash
# *best* — compare every trial6/6A arm side by side
python brepformer/tensorboard_server.py --log_dir results --compare --face_segmentation \
    --experiments t6_base t6_w t6_f t6_wf t6_init_def t6_init_ft t6ctl_base t6ctl_wf \
                  t6a_base t6a_w t6a_f t6a_init_t5 t6actl_base

# just the 5-class family against the two pre-sweep baselines
python brepformer/tensorboard_server.py --log_dir results --compare --face_segmentation \
    --experiments t6_base t6_wf t6ctl_base trial1_ss1500 trial1_finetune_ss

# plain server over everything in results/
python brepformer/tensorboard_server.py --log_dir results --port 6006
```

### `tensorboard_server.py`

Launch TensorBoard for training monitoring. When training with `--face_segmentation`, additional face-level metrics (`face_acc`, `face_f1`) appear alongside model-level metrics.

```bash
# *best* — compare all experiments (face seg metrics visible when trained with --face_segmentation)
python brepformer/tensorboard_server.py --log_dir results --compare --face_segmentation

# Basic usage
python brepformer/tensorboard_server.py --log_dir results

# Custom port
python brepformer/tensorboard_server.py --log_dir results --port 8080

# Remote access (bind to all interfaces)
python brepformer/tensorboard_server.py --log_dir results --host 0.0.0.0

# Compare multiple experiments
python brepformer/tensorboard_server.py --log_dir results --compare
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--log_dir` | "results" | Log directory |
| `--port` | 6006 | TensorBoard port |
| `--host` | "localhost" | Host to bind |
| `--compare` | False | Experiment comparison mode |
| `--reload_interval` | 30 | Reload interval (seconds) |

**Alternative: Direct TensorBoard**
```bash
tensorboard --logdir results --port 6006
```

---

## Model Export

### `export_model.py`

Export trained models for deployment.

```bash
# *best* — export face segmentation model as state dict
python brepformer/export_model.py \
    --checkpoint results/trial5/best-epoch=39-val/f1=0.8832.ckpt \
    --format state_dict \
    --output_path face_seg_model.pt

# Export state dict (PyTorch)
python brepformer/export_model.py \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --format state_dict \
    --output_path model_weights.pt

# Export TorchScript
python brepformer/export_model.py \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --format torchscript \
    --output_path model_scripted.pt \
    --max_faces 100 \
    --max_edges 200

# Export ONNX
python brepformer/export_model.py \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --format onnx \
    --output_path model.onnx
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | required | Model checkpoint |
| `--format` | "state_dict" | Export format ("state_dict", "torchscript", "onnx") |
| `--output_path` | required | Output file path |
| `--max_faces` | 100 | Max faces for tracing |
| `--max_edges` | 200 | Max edges for tracing |

---

## STEP to Graph Conversion

### `data/step_to_graph.py`

Converts STEP CAD files to BrepFormer graph format using pythonOCC. Used internally by `infer.py` and `export_freecad.py`. Can also be run standalone for debugging.

```bash
# Convert a STEP file and print summary
python -m brepformer.data.step_to_graph brepformer/data/defeature/steps/uh62382.step
```

**Key functions:**
- `step_to_graph(step_path)` — returns raw graph data dict (face_grid, face_attr, edge_index, edge_attr, edge_grid)
- `step_to_preprocessed_sample(step_path)` — returns a dict matching `PreprocessedDataset` format (with spatial_pos, in_degree computed)

---

## Class Definitions

### `data/classes.py`

Single source of truth for the 27 MFTRCAD machining feature class definitions. All visualization, export, and analysis scripts import from here.

**Contents:**
- `CLASS_NAMES` — list of 27 class name strings
- `CLASS_COLORS_HEX` — list of 27 hex color strings (tab20 + tab20b palette)
- `NUM_CLASSES` — integer 27
- `COLOR_NAMES` — human-readable color names
- `UNLABELED_COLOR_HEX`, `HIGHLIGHT_COLOR_HEX`, `EDGE_COLOR_HEX` — special colors
- `hex_to_rgb01()`, `hex_to_rgb255()`, `get_class_color_rgb01()`, `get_class_name()` — utility functions

```python
from brepformer.data.classes import CLASS_NAMES, CLASS_COLORS_HEX, NUM_CLASSES

print(CLASS_NAMES[0])       # "chamfer"
print(CLASS_COLORS_HEX[0])  # "#1f77b4"
print(NUM_CLASSES)           # 27
```

**8 grouped classes (`--8_classes`):**

The `--8_classes` flag (available on `infer.py`, `visualize_seg.py`, `export_freecad.py`, `test_preprocessed.py`, `analyze.py`) remaps the 27 fine-grained classes into 8 higher-level machining feature categories:

| Real ID | Name | 27-class IDs |
|---------|------|-------------|
| 0 | other_surfaces | stock(24), rectangular_passage_2(25) |
| 1 | through_hole | through_hole(1) |
| 2 | blind_hole | blind_hole(12) |
| 3 | chamfer | chamfer(0), chamfer_2(26) |
| 4 | fillet | round(23) |
| 5 | through_cut | triangular_passage(2), rectangular_passage(3), 6sides_passage(4), triangular_through_slot(5), rectangular_through_slot(6), circular_through_slot(7) |
| 6 | blind_cut | triangular_pocket(13), rectangular_pocket(14), 6sides_pocket(15), circular_end_pocket(16), rectangular_blind_slot(17), v_shaped_blind_slot(18), circular_blind_slot(19) |
| 7 | extrude | rectangular_through_step(8), 2sides_through_step(9), slanted_through_step(10), Oring(11), rectangular_blind_step(20), 2sides_blind_step(21), triangular_blind_step(22) |

```python
from brepformer.data.classes import REAL_CLASS_NAMES, REAL_NUM_CLASSES, map_labels_to_real

print(REAL_CLASS_NAMES)     # ['other_surfaces', 'through_hole', ..., 'extrude']
print(REAL_NUM_CLASSES)     # 8
print(map_labels_to_real([0, 1, 12, 24]))  # [3, 1, 2, 0]
```

**Defeature classes (5 categories):**

The defeature dataset (navin_defeaturing) uses 5 machining feature classes, remapped from the original 7-class scheme:

| Class ID | Name | Color | Original IDs |
|----------|------|-------|-------------|
| 0 | random | gray (#7f7f7f) | 0 |
| 1 | hole | blue (#1f77b4) | 1, 2 |
| 2 | chamfer | orange (#ff7f0e) | 3 |
| 3 | fillet | green (#2ca02c) | 4 |
| 4 | cut | red (#d62728) | 5, 6 |

```python
from brepformer.data.classes import DEFEATURE_CLASS_NAMES, DEFEATURE_NUM_CLASSES

print(DEFEATURE_CLASS_NAMES)  # ['random', 'hole', 'chamfer', 'fillet', 'cut']
print(DEFEATURE_NUM_CLASSES)  # 5
```

**Dataset statistics:**
- 1561 models, 115,766 total faces
- Class distribution: random 27.3%, hole 23.6%, chamfer 3.8%, fillet 14.9%, cut 30.5%

---

## Fine-Tuning Pipeline (MFTRCAD 27-class -> Defeature 5-class)

### Overview

Fine-tune a pre-trained MFTRCAD model on the defeature dataset by:
1. Loading encoder weights from a pre-trained 27-class checkpoint (trial5)
2. Replacing classifier heads with new 5-class heads, warm-started by averaging the 27-class weights according to the `CLASS_TO_DEFEATURE` mapping
3. Training with differential learning rates (lower for encoder, higher for new heads)
4. Optionally freezing the encoder for initial epochs

### `fine_tune.py` ***recommended***

Dedicated fine-tuning script that handles weight transfer, head remapping, and differential learning rates automatically.

#### MFTRCAD -> Defeature Class Mapping

| MFTRCAD 27-Class | ID | Defeature 5-Class | ID |
|---|---|---|---|
| chamfer, chamfer_2 | 0, 26 | chamfer | 2 |
| through_hole, blind_hole | 1, 12 | hole | 1 |
| round | 23 | fillet | 3 |
| stock | 24 | random | 0 |
| All passages, slots, steps, pockets, Oring, passage_2 | 2-11, 13-22, 25 | cut | 4 |

```python
from brepformer.data.classes import CLASS_TO_DEFEATURE, map_labels_to_defeature

print(CLASS_TO_DEFEATURE)  # [2, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 0, 4, 2]
print(map_labels_to_defeature([0, 1, 12, 23, 24]))  # [2, 1, 1, 3, 0]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--pretrained` | required | Path to pre-trained 27-class checkpoint |
| `--data_dir` | required | Preprocessed defeature data directory |
| `--num_classes` | 5 | Number of target classes |
| `--num_face_classes` | 5 | Number of target face classes |
| `--freeze_encoder_epochs` | 0 | Freeze encoder for this many initial epochs (0 = never) |
| `--encoder_lr_factor` | 0.1 | LR multiplier for encoder params (relative to head LR) |
| `--no_warm_start` | False | Don't warm-start heads from 27-class weights |
| `--face_segmentation` | True | Enable face segmentation (default for fine-tuning) |
| `--face_seg_weight` | 1.0 | Loss weight for face segmentation |
| `--model_cls_weight` | 1.0 | Loss weight for model classification |
| `--weighted_crossentropy` | False | Use inverse-frequency class weights for face seg |
| `--learning_rate` | 0.0005 | Learning rate for classifier heads |
| `--max_epochs` | 100 | Maximum fine-tuning epochs |
| `--warmup_steps` | 1000 | Warmup steps (lower than pre-training) |
| `--batch_size` | auto | Batch size (auto-detected from GPU memory) |
| `--num_workers` | 0 | Data loader workers |
| `--output_dir` | "results" | Output directory |
| `--exp_name` | "finetune" | Experiment name |

### Complete Fine-Tuning Pipeline

The full pipeline from raw defeature data to evaluated fine-tuned model:

```bash
# ========================================================
# Step 0: Ensure defeature data is prepared
# (skip if brepformer/data/defeature/ already has graphs/ and labels/)
# ========================================================
python brepformer/data/prepare_defeature.py \
    --source /mnt/c/projects/data/navin_defeaturing \
    --dest brepformer/data/defeature \
    --convert_graphs

# ========================================================
# Step 1: Preprocess defeature data (5 classes, with descriptors)
# ========================================================
python brepformer/preprocess.py \
    --data_dir brepformer/data/defeature \
    --output_dir brepformer/data/defeature_processed \
    --num_classes 5 \
    --compute_descriptors \
    --d2_bins 64 \
    --angle_bins 64

# ========================================================
# Step 2: Fine-tune from trial5 (best MFTRCAD weights)
# ========================================================
python brepformer/fine_tune.py \
    --pretrained "results/trial5/best-epoch=39-val/f1=0.8832.ckpt" \
    --data_dir brepformer/data/defeature_processed \
    --face_segmentation \
    --num_classes 5 \
    --num_face_classes 5 \
    --face_seg_weight 1.0 \
    --model_cls_weight 1.0 \
    --learning_rate 0.0005 \
    --encoder_lr_factor 0.1 \
    --max_epochs 100 \
    --warmup_steps 1000 \
    --num_workers 0 \
    --precision 32 \
    --output_dir results \
    --exp_name trial1_finetune_ss

# ========================================================
# Step 3: Test the fine-tuned model
# ========================================================
python brepformer/test_preprocessed.py \
    --data_dir brepformer/data/defeature_processed \
    --checkpoint "results/trial1_finetune_ss/best-epoch=32-val/f1=0.9354.ckpt" \
    --batch_size 1 \
    --num_workers 0 \
    --output_file brepformer/preds/trial1_finetune_ss/test_results.json \
    --output_face_preds brepformer/preds/trial1_finetune_ss/face_preds.json

# ========================================================
# Step 4: Analyze the fine-tuned model
# ========================================================
python brepformer/analyze.py \
    --checkpoint "results/trial1_finetune_ss/best-epoch=32-val/f1=0.9354.ckpt" \
    --data_dir brepformer/data/defeature_processed \
    --mode all \
    --output_dir brepformer/analysis_results/trial1_finetune_ss

# ========================================================
# Step 5: Inference on STEP files
# ========================================================
python -m brepformer.infer \
    --step_dir brepformer/data/defeature/steps/ \
    --checkpoint "results/trial1_ss1500/last.ckpt" \
    --output_dir inference/trial1_1500_ss/

# ========================================================
# Step 6: Visualize results
# ========================================================
python -m brepformer.visualize_seg \
    --step_dir brepformer/data/defeature/steps/ \
    --labels_dir brepformer/data/defeature/labels/ \
    --seg_dir inference/trial1_finetune_ss/
```

#### Fine-Tuning Variants

```bash
# *best* — fine-tune with frozen encoder for 10 epochs, then unfreeze
python brepformer/fine_tune.py \
    --pretrained "results/trial5/best-epoch=39-val/f1=0.8832.ckpt" \
    --data_dir brepformer/data/defeature_processed \
    --face_segmentation \
    --num_classes 5 \
    --num_face_classes 5 \
    --freeze_encoder_epochs 10 \
    --encoder_lr_factor 0.1 \
    --learning_rate 0.0005 \
    --max_epochs 100 \
    --output_dir results \
    --exp_name finetune_frozen

# With class-weighted loss (for imbalanced defeature classes)
python brepformer/fine_tune.py \
    --pretrained "results/trial5/best-epoch=39-val/f1=0.8832.ckpt" \
    --data_dir brepformer/data/defeature_processed \
    --face_segmentation \
    --num_classes 5 \
    --num_face_classes 5 \
    --weighted_crossentropy \
    --learning_rate 0.0005 \
    --output_dir results \
    --exp_name finetune_weighted

# Without warm-start (random init for heads)
python brepformer/fine_tune.py \
    --pretrained "results/trial5/best-epoch=39-val/f1=0.8832.ckpt" \
    --data_dir brepformer/data/defeature_processed \
    --face_segmentation \
    --num_classes 5 \
    --num_face_classes 5 \
    --no_warm_start \
    --output_dir results \
    --exp_name finetune_cold

# Quick sanity check
python brepformer/fine_tune.py \
    --pretrained "results/trial5/best-epoch=39-val/f1=0.8832.ckpt" \
    --data_dir brepformer/data/defeature_processed \
    --face_segmentation \
    --num_classes 5 \
    --num_face_classes 5 \
    --fast_dev_run \
    --num_workers 0 \
    --batch_size 4
```

---

## Flexible Defeaturing Commands

Both `defeature.py` and `defeature_v2.py` accept the same core CLI. Below are flexible Python commands for common workflows with variable input/output paths. The `scripts/run_defeature.sh` wrapper adds `--engine v1|v2` selection (default: v2).

### Variable Input/Output Paths

```bash
# Defeature a single file to a custom output directory
python -m brepformer.defeature_v2 \
    --step /path/to/my_model.step \
    --output_dir /path/to/output/

# Defeature a custom directory of STEP files
python -m brepformer.defeature_v2 \
    --step_dir /path/to/my_steps/ \
    --output_dir /path/to/my_output/ \
    --verbose

# v1 engine on a custom directory
python -m brepformer.defeature \
    --step_dir /path/to/my_steps/ \
    --output_dir /path/to/my_output/
```

### Using Pre-computed Predictions

```bash
# Single file with custom .seg file
python -m brepformer.defeature_v2 \
    --step /path/to/model.step \
    --seg /path/to/predictions.seg \
    --output_dir /path/to/output/

# Batch: match .seg files from one dir to STEP files from another
python -m brepformer.defeature_v2 \
    --step_dir /path/to/step_files/ \
    --seg_dir /path/to/seg_predictions/ \
    --output_dir /path/to/defeatured/ \
    --save_colored --verbose
```

### Custom Checkpoint

```bash
# Use a different model checkpoint
python -m brepformer.defeature_v2 \
    --step /path/to/model.step \
    --checkpoint /path/to/my_checkpoint.ckpt \
    --output_dir /path/to/output/

# Batch with custom checkpoint
python -m brepformer.defeature_v2 \
    --step_dir /path/to/step_files/ \
    --checkpoint results/my_experiment/best-epoch=50-val/f1=0.95.ckpt \
    --output_dir /path/to/defeatured/ \
    --verbose
```

### v2 Tolerance Control

```bash
# Higher tolerance for complex/filleted models
python -m brepformer.defeature_v2 \
    --step model.step \
    --max_fuzzy 0.01 \
    --verbose

# Lower tolerance for precision-critical models
python -m brepformer.defeature_v2 \
    --step model.step \
    --max_fuzzy 1e-4 \
    --verbose
```

### Shell Wrapper with Variable Paths

```bash
# Via run_defeature.sh with custom paths
./brepformer/scripts/run_defeature.sh \
    --step /path/to/model.step \
    --output_dir /path/to/output/ \
    --save_colored --verbose

# v1 engine via wrapper
./brepformer/scripts/run_defeature.sh --engine v1 \
    --step_dir /path/to/step_files/ \
    --output_dir /path/to/v1_output/ \
    --verbose

# v2 with high tolerance via wrapper
./brepformer/scripts/run_defeature.sh \
    --step model.step \
    --max_fuzzy 0.01 --verbose
```

### Combined Inference + Defeaturing (Variable Paths)

```bash
# Step 1: Run inference on custom STEP files
python -m brepformer.infer \
    --step_dir /path/to/my_steps/ \
    --checkpoint results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt \
    --output_dir /path/to/my_inference/

# Step 2: Defeature using those predictions
python -m brepformer.defeature_v2 \
    --step_dir /path/to/my_steps/ \
    --seg_dir /path/to/my_inference/ \
    --output_dir /path/to/my_defeatured/ \
    --save_colored --verbose

# Or combine inference + defeaturing in one command
python -m brepformer.defeature_v2 \
    --step_dir /path/to/my_steps/ \
    --checkpoint results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt \
    --output_dir /path/to/my_defeatured/ \
    --verbose
```

## Further Improvements (Trials 6 / 6A)

### Runbook — the exact sequence, in dependency order

Everything below is re-runnable as written from the repo root. Each stage is idempotent
*except* `preprocess.py`, which is the one thing that cannot be resumed (see the warning
in stage 3). `--run` consults `results/sweep_logs/status.json` and skips arms that
already returned 0, so an interrupted sweep is restarted by re-issuing the same command.

Prefix every command with the env, since none of this works in `base`:

```bash
cd BrepMFR
conda activate new_brepmfr     # or prefix each: conda run -n new_brepmfr --no-capture-output …
```

| # | Stage | Cost | State as of 2026-08-31 |
|---|---|---|---|
| 1 | Merge pcq labels, 5- and 27-class | ~2 min | done — 809 labels each |
| 2 | Build graphs once, symlink to pcq5 | ~25 min | done — 807 of 809 (2 OCC crashes) |
| 3 | Preprocess pcq5 and pcq27 | ~2-3 h each, parallel | **running** |
| 4 | Combine into `trial6_processed` / `trial6a_processed` | seconds | pending |
| 5 | Sweep, 5-class family (11 arms) | ~1 h/arm | 3 of 11 done (the `t6ctl_*` controls) |
| 6 | Sweep, 27-class family (7 arms) | ~10 h/arm | pending |
| 7 | Rank checkpoints, per-class eval, write results | ~20 min | pending |

```bash
# ── 1. Merge pseudo-labels in both label spaces ───────────────────────────────
# Same 809 STEPs twice; thresholds always fitted in 27-class space because the
# teacher's confidence is calibrated per fine class.
python brepformer/data/merge_suitable_samples.py \
    --steps_dir brepformer/data/suitable_samples_scaled/steps \
    --inference_dir brepformer/inference/hard_samples_27 \
    --out_dir brepformer/data/suitable_samples_pcq5 \
    --per_class_top_frac 0.5 --per_class_floor 0.5 --min_kept_frac 0.15 --label_space 5

python brepformer/data/merge_suitable_samples.py \
    --steps_dir brepformer/data/suitable_samples_scaled/steps \
    --inference_dir brepformer/inference/hard_samples_27 \
    --out_dir brepformer/data/suitable_samples_pcq27 \
    --per_class_top_frac 0.5 --per_class_floor 0.5 --min_kept_frac 0.15 --label_space 27

# ── 2. Graphs once, shared by symlink ─────────────────────────────────────────
# Resumable (skips existing outputs, atomic writes), so just re-run it if killed.
python -m brepformer.data.steps_to_graphs \
    --steps_dir brepformer/data/suitable_samples_pcq27/steps \
    --out_dir   brepformer/data/suitable_samples_pcq27/graphs \
    --tolerate_crashes
ln -sfn "$PWD/brepformer/data/suitable_samples_pcq27/graphs" \
        brepformer/data/suitable_samples_pcq5/graphs

# ── 3. Preprocess — NOT RESUMABLE ─────────────────────────────────────────────
# preprocess.py restarts numbering at 00000.pkl and overwrites, and writes
# metadata.json only on success. A killed run therefore leaves a directory that
# looks populated but is unusable, and re-running redoes all of it. The two are
# independent and both CPU-bound, so run them concurrently; settings MUST match
# the corpus each will be merged with or combine_processed refuses the merge.
python brepformer/preprocess.py \
    --data_dir brepformer/data/suitable_samples_pcq5 \
    --output_dir brepformer/data/suitable_samples_pcq5_processed \
    --num_classes 5 --compute_descriptors \
    --num_spatial 64 --d2_bins 64 --angle_bins 64 --seed 42 \
    > results/sweep_logs/preprocess_pcq5.log 2>&1 &

python brepformer/preprocess.py \
    --data_dir brepformer/data/suitable_samples_pcq27 \
    --output_dir brepformer/data/suitable_samples_pcq27_processed \
    --num_classes 27 --compute_descriptors \
    --num_spatial 64 --d2_bins 64 --angle_bins 64 --seed 42 \
    > results/sweep_logs/preprocess_pcq27.log 2>&1 &
wait

# Verify before proceeding. Counting pickles is NOT enough: a full-looking corpus
# whose face_labels are all -1 trains without error and contributes nothing, which
# is exactly the failure that cost a full preprocessing run on 2026-08-31 (the
# pseudo-label writer emits [model_id, {"labels": …}] and preprocess.py used to read
# only the ground-truth "cls" dict). Always assert supervised faces exist.
for d in brepformer/data/suitable_samples_pcq{5,27}_processed; do
    echo "$d  train=$(ls $d/train 2>/dev/null | wc -l)  meta=$([ -f $d/metadata.json ] && echo ok || echo MISSING)"
done

python - <<'PY'
import glob, pickle
from collections import Counter
for d in ("brepformer/data/suitable_samples_pcq5_processed",
          "brepformer/data/suitable_samples_pcq27_processed"):
    kept = masked = 0
    files = sorted(glob.glob(d + "/train/*.pkl"))
    for f in files[:50]:
        c = Counter(pickle.load(open(f, "rb"))["face_labels"].tolist())
        kept += sum(v for k, v in c.items() if k >= 0)
        masked += c.get(-1, 0)
    verdict = "OK" if kept else "*** ALL MASKED — DO NOT TRAIN ON THIS ***"
    print(f"{d}: {len(files)} pkl, {kept} supervised / {kept+masked} faces  {verdict}")
PY
# Expect roughly 49% supervised at --per_class_top_frac 0.5
# (106,888 of 216,781 faces over the 809-model corpus).

# ── 4. Combine — splits preserved, so the defeature test set stays bit-identical
#      and any metric change is attributable to the added data.
python -m brepformer.data.combine_processed \
    --inputs brepformer/data/defeature_processed \
             brepformer/data/suitable_samples_pcq5_processed \
    --output_dir brepformer/data/trial6_processed

python -m brepformer.data.combine_processed \
    --inputs brepformer/data/mftrcad_processed \
             brepformer/data/suitable_samples_pcq27_processed \
    --output_dir brepformer/data/trial6a_processed

# ── 5/6. Sweep, cheap family first. Resumable via status.json. ────────────────
python -m brepformer.sweeps.trial6_sweep --run --family 5
python -m brepformer.sweeps.trial6_sweep --run --family 27

# ── 7. Score everything on the real hand-labelled test set ────────────────────
# collect_results reads the arm list from trial6_sweep.py, so its settings column
# cannot drift from what was executed — run it first, then drill in.
python -m brepformer.sweeps.collect_results \
    --out brepformer/RESULTS_TRIAL6.md \
    --json_out brepformer/analysis_results/trial6_sweep_results.json

# Rank the arms head-to-head (one prepared test split reused across checkpoints).
python -m brepformer.scan_checkpoints \
    --ckpt_dirs results/t6_base results/t6_w results/t6_f results/t6_wf \
                results/t6_init_def results/t6_init_def_wf \
                results/t6_init_ft results/t6_init_ft_wf \
                results/t6ctl_base results/t6ctl_wf results/t6ctl_init_ft \
    --data_dir brepformer/data/defeature_processed --split test --label_space 5 \
    --out brepformer/analysis_results/ckpt_scan_trial6.json

# 27-class arms, collapsed to 5 so they are comparable on the same test set.
python -m brepformer.scan_checkpoints \
    --ckpt_dirs results/t6a_base results/t6a_w results/t6a_f results/t6a_wf \
                results/t6a_init_t5 results/t6a_init_t5_wf results/t6actl_base \
    --data_dir brepformer/data/defeature_processed --split test --label_space 5 \
    --out brepformer/analysis_results/ckpt_scan_trial6a.json

# Then the per-class breakdown for whichever arm won. Point --checkpoint at the
# arm's bestface-* file (per-face F1), NOT best-* (model-level multi-label F1).
python -m brepformer.eval_per_class \
    --checkpoint "results/<winning_arm>/bestface-....ckpt" \
    --data_dir brepformer/data/defeature_processed --split test --label_space 5

# ── 8. TensorBoard, for the training-dynamics questions the tables can't answer ─
# Point it at results/ (not one run) so every arm is overlaid on the same axes; the
# Lightning TensorBoardLogger writes results/<exp_name>/version_N/ and TensorBoard
# picks up new runs while the sweep is still going, so it can be left running.
tensorboard --logdir results --port 6006 --bind_all
# -> http://localhost:6006
#
# What to actually look for, given this sweep's questions:
# - val/face_f1 per arm: the axis every checkpoint is selected on. Compare each
#   t6_* against its t6ctl_* twin — that difference IS the pcq data's contribution.
# - train/face_f1 vs val/face_f1 divergence: the pcq arms train on ~49% supervised
#   faces, so an arm can look healthy on train while the added corpus does nothing.
# - lr-* : confirms warmup actually completed. The whole reason the from-scratch arms
#   are being re-run is that trial1_ss1500 never exceeded 45% of its nominal LR
#   (warmup_steps=5000 at batch 32 never finished), so verify the curve plateaus.
# - Filter the run list with the regex box: `t6_` for treatments, `t6ctl` for
#   controls, `t6a` for the 27-class family.
```

### All 18 arms, spelled out

Every arm is one `train_preprocessed.py` call: the shared block, the corpus, the
label-space size, then the arm's own flags. `--emit` prints these fully expanded
(`python -m brepformer.sweeps.trial6_sweep --emit --family 5`); `trial6_sweep.py`'s
`SWEEP` table is the single source of truth, so if a flag here ever disagrees with the
emitted form, the emitted form is right.

```bash
COMMON="--max_faces 1000 --batch_size 2 --accumulate_grad_batches 16 \
        --learning_rate 0.001 --max_epochs 200 --seed 42 --output_dir results \
        --face_segmentation --lr_monitor val/face_f1 \
        --early_stop_monitor val/face_f1 --early_stop_patience 25"
C5="--num_classes 5 --num_face_classes 5"
C27="--num_classes 27 --num_face_classes 27"
D5=brepformer/data/trial6_processed          # defeature + pcq5   (~1,900 models)
D5C=brepformer/data/defeature_processed      # defeature only     (1,561)
D27=brepformer/data/trial6a_processed        # mftrcad + pcq27    (~23,500)
D27C=brepformer/data/mftrcad_processed       # mftrcad only
WF="--weighted_crossentropy --face_seg_weight 2.0 --model_cls_weight 0.5"
CKPT_DEF=results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt        # defeature_v2's shipped default
CKPT_FT=results/trial1_finetune_ss/best-epoch=32-val/f1=0.9354.ckpt    # best measured, macro 0.8886
CKPT_T5=results/trial5/best-epoch=39-val/f1=0.8832.ckpt                # best 27-class
T=brepformer/train_preprocessed.py

# ---- 5-class family, 2x2 factorial over (class weighting x loss balance) ----
python $T --data_dir $D5  $C5 $COMMON --exp_name t6_base
python $T --data_dir $D5  $C5 $COMMON --weighted_crossentropy --exp_name t6_w
python $T --data_dir $D5  $C5 $COMMON --face_seg_weight 2.0 --model_cls_weight 0.5 --exp_name t6_f
python $T --data_dir $D5  $C5 $COMMON $WF --exp_name t6_wf

# ---- 5-class warm starts, from BOTH existing checkpoints, at the two loss corners.
#      Neither checkpoint is assumed: the one that ships and the one that measures
#      best are different models.
python $T --data_dir $D5  $C5 $COMMON --init_weights $CKPT_DEF --exp_name t6_init_def
python $T --data_dir $D5  $C5 $COMMON --init_weights $CKPT_DEF $WF --exp_name t6_init_def_wf
python $T --data_dir $D5  $C5 $COMMON --init_weights $CKPT_FT  --exp_name t6_init_ft
python $T --data_dir $D5  $C5 $COMMON --init_weights $CKPT_FT  $WF --exp_name t6_init_ft_wf   # a-priori favourite

# ---- 5-class CONTROLS: identical settings, no pcq data. These are what make a gain
#      attributable to the data rather than to the hyperparameters. All three are DONE.
python $T --data_dir $D5C $C5 $COMMON --exp_name t6ctl_base                                   # done: val face_acc 0.273
python $T --data_dir $D5C $C5 $COMMON $WF --exp_name t6ctl_wf                                 # done: val face_acc 0.273
python $T --data_dir $D5C $C5 $COMMON --init_weights $CKPT_FT --exp_name t6ctl_init_ft         # done: val face_acc 0.842

# ---- 27-class family, collapsed 27->5 at evaluation ----
python $T --data_dir $D27  $C27 $COMMON --exp_name t6a_base
python $T --data_dir $D27  $C27 $COMMON --weighted_crossentropy --exp_name t6a_w
python $T --data_dir $D27  $C27 $COMMON --face_seg_weight 2.0 --model_cls_weight 0.5 --exp_name t6a_f
python $T --data_dir $D27  $C27 $COMMON $WF --exp_name t6a_wf
python $T --data_dir $D27  $C27 $COMMON --init_weights $CKPT_T5 --exp_name t6a_init_t5
python $T --data_dir $D27  $C27 $COMMON --init_weights $CKPT_T5 $WF --exp_name t6a_init_t5_wf
python $T --data_dir $D27C $C27 $COMMON --exp_name t6actl_base                                # CONTROL: isolates pcq27
```

Prefer `trial6_sweep --run` over pasting these: it runs them sequentially, records
wall-clock per arm, and will not re-run a finished arm. Use the explicit forms to re-run
one arm in isolation, or to change a single flag without editing the sweep table.

### Where each stage is documented in full

The per-flag reference for every command above lives next to its script, so nothing is
duplicated:

| Stage | Section |
|---|---|
| Corpus preparation (merge, graphs, preprocess, combine) | [Trials 6 / 6A — Pseudo-Labelled Corpus Preparation](#trials-6--6a--pseudo-labelled-corpus-preparation) |
| Training (all 18 arms) | [Trials 6 / 6A — Hyperparameter Sweep](#trials-6--6a--hyperparameter-sweep) |
| Checkpoint ranking + per-class eval | [`scan_checkpoints.py`](#scan_checkpointspy), [`eval_per_class.py`](#eval_per_classpy), [`sweeps/collect_results.py`](#sweepscollect_resultspy) |
| Inference | [Trials 6 / 6A — Inference](#trials-6--6a--inference) |
| Segmentation + defeaturing | [Trials 6 / 6A — Segmentation + Defeaturing](#trials-6--6a--segmentation--defeaturing) |
| Visualization | [Trials 6 / 6A — Visualization](#trials-6--6a--visualization) |
| Training-curve comparison | [Trials 6 / 6A — TensorBoard](#trials-6--6a--tensorboard) |

Two things to know when reading the results:

1. **Target to beat: macro 0.8886** on `defeature_processed/test`
   (random .820 | hole .954 | chamfer .874 | fillet .887 | cut .908), set by
   `trial1_finetune_ss/best-epoch=32`. The older 0.8564 figure was
   `trial1_ss1500/best-epoch=57`, which is `defeature_v2.py`'s shipped default but only
   13th of 26 on face F1.
2. **Compare `bestface-*` checkpoints, not `best-*`.** The `val/f1` in a `best-*`
   filename is model-level multi-label F1, not per-face segmentation; the two orderings
   disagree. Results land in `brepformer/RESULTS_TRIAL6.md`.

<!-- BEGIN GENERATED SCRIPT INDEX -->

## Complete Script Index

Every runnable script in the repo (54 of 67 Python files; the rest are imported libraries, listed at the bottom). Generated by `python -m brepformer.gen_script_index` — re-run it after adding a script so nothing goes untracked. A blank **Documented** cell means the script has no recorded invocation above and needs one.

### Data acquisition

| Script | Documented in | What it does |
|---|---|---|
| `brepformer/data/curate_corpus.py` | [`data/curate_corpus.py`](#datacurate_corpuspy) | Curate the hard_samples corpus for training quality, not file count. |
| `brepformer/data/download_grabcad.py` | [`data/download_grabcad.py` / `data/download_grabcad_api.py`](#datadownload_grabcadpy-datadownload_grabcad_apipy) | Download STEP files referenced by GrabCAD library URLs in a Markdown file. |
| `brepformer/data/grabcad_login.py` | [`data/grabcad_login.py`](#datagrabcad_loginpy) | Open a headed Chromium on the shared GrabCAD profile and wait for real login. |

### Data preparation

| Script | Documented in | What it does |
|---|---|---|
| `brepformer/data/combine_processed.py` | [`data/combine_processed.py`](#datacombine_processedpy) | Combine several preprocessed corpora into one train/val/test tree for training. |
| `brepformer/data/merge_suitable_samples.py` | [`vision_label.py`](#vision_labelpy) | Merge hand-triaged hard_samples into a mftrcad-compatible training corpus. |
| `brepformer/data/prepare_defeature.py` | [Defeature Dataset — Data Preparation](#defeature-dataset-data-preparation) | Prepare defeature dataset for BrepFormer training. |
| `brepformer/data/step_to_graph.py` | [`data/step_to_graph.py`](#datastep_to_graphpy) | STEP to BrepFormer graph converter using pythonOCC. |
| `brepformer/data/steps_to_graphs.py` | [`data/steps_to_graphs.py`](#datasteps_to_graphspy) | Batch-convert a directory of STEP files to the graph JSONs ``preprocess.py`` reads. |
| `brepformer/preprocess.py` | [Note: run_pipeline.py does not pass --num_classes to preprocess.py,](#note-run_pipelinepy-does-not-pass---num_classes-to-preprocesspy) | Preprocess MFTRCAD data for BrepFormer training. |

### Training

| Script | Documented in | What it does |
|---|---|---|
| `brepformer/fine_tune.py` | [`fine_tune.py` ***recommended***](#fine_tunepy-recommended) | Fine-tune a pre-trained MFTRCAD (27-class) BrepFormer on the defeature (5-class) dataset. |
| `brepformer/sweeps/trial6_sweep.py` | [*best* — run the cheap 5-class family first (~1,900 models per arm)](#best-run-the-cheap-5-class-family-first-1900-models-per-arm) | The trial6 / trial6a hyperparameter sweep: one definition, three consumers. |
| `brepformer/train.py` | [`train.py` (legacy — use `train_preprocessed.py` instead)](#trainpy-legacy-use-train_preprocessedpy-instead) | Training script for BrepFormer B-rep classification. |
| `brepformer/train_preprocessed.py` | [`train_preprocessed.py` ***recommended***](#train_preprocessedpy-recommended) | Training script for BrepFormer using preprocessed data. |

### Evaluation & analysis

| Script | Documented in | What it does |
|---|---|---|
| `brepformer/analyze.py` | [`analyze.py` ***recommended***](#analyzepy-recommended) | Analysis script for BrepFormer models and predictions. |
| `brepformer/data/label_harness.py` | [Label harness](#label-harness--measuring-any-labeller-before-trusting-it-2026-09-01) | Score any label-producing component against mftrcad ground truth: `harm` (correct labels destroyed), `benefit` (wrong labels recovered), `select` (risk–coverage / auto-accept budget). Supersedes the two `validate_*` scripts below. |
| `brepformer/data/geo_label.py` | [Rebuilding synthetic labels](#rebuilding-synthetic-labels-a-second-opinion-instead-of-a-mask-2026-09-01) | Label faces from exact B-Rep geometry via the measured `class_geometry_stats*.json` posterior. Needs no GPU and no checkpoint; **beats the trained network 72.6% vs 63.9%, and 86% vs 12% on `chamfer`**. |
| `brepformer/data/fuse_labels.py` | [Rebuilding synthetic labels](#rebuilding-synthetic-labels-a-second-opinion-instead-of-a-mask-2026-09-01) | Fuse network + geometry labels; accept where they agree, prefer geometry where they do not, and emit a **binary-choice** review queue for the 35% that disagree. Replaces confidence masking. |
| `brepformer/data/edge_prune_probe.py` | [Learn the prune](#the-fix-for-the-feature-partition-learn-the-prune-2026-09-02) | Train a binary "same feature?" edge classifier and take components of the pruned graph. Raises the feature-partition ceiling **56.3% → 88.0%**; shows dihedral angle and convexity are individually at chance (AUC 0.47 / 0.53), which is why the hand-thresholded concave rule fails. |
| `brepformer/data/supervision_census.py` | [Family-5 sweep result](#family-5-sweep-result-initialization-dwarfs-every-knob-the-sweep-tested-2026-09-02) | Count supervised faces per split before training. Catches all-`-1` corpora, and mismatched val supervision density that makes two arms' face metrics look comparable when they are not. Prints the majority-class **collapse baseline**. |
| `brepformer/so3_probe.py` | [SO(3) robustness](#so3-robustness-not-a-catastrophe-but-a-heavy-tail--and-no-augmentation-exists-2026-09-02) | Score a checkpoint under uniform SO(3) and 24-element octahedral rotations of an already-preprocessed sample. The FoV-Net catastrophe does **not** reproduce (67-69% worst-case retention, not 36%), but `hole` falls **0.971 → 0.120** in the worst frame — which is *axis-aligned*, so report the worst frame not the mean. Also established that **no geometric augmentation exists in the training path at all**. |
| `brepformer/scan_checkpoints.py --device cpu` | [Removing the val-set confound](#removing-the-val-set-confound-no-sweep-arm-beat-its-own-initialization-2026-09-02) | Re-score every arm of a sweep on **one** held-out split, off-GPU, including the `--init_weights` checkpoints as baselines. Showed no family-5 arm beat the weights it was seeded from. |
| `brepformer/data/repair_descriptors.py` | [Bug 2: the D2 channel](#bug-2-the-d2-channel-was-empty-because-of-a-max_dist-and-it-repairs-without-re-preprocessing-2026-09-04) | Recompute `d2_distance` in already-preprocessed pickles from the stored `face_grid` — no STEP, no OCC. The channel was empty because `max_dist` was estimated from the wrong centroid column and came out at 1e-6, so **99.2% of face pairs had an all-zero histogram**; repair takes the mean empty-pair fraction 0.892 → 0.000. `--check_only` looks without writing. |
| `train_preprocessed.py --rotate_aug so3\|octahedral` | [Rotation augmentation](#rotation-augmentation-now-exists-2026-09-04) | The first geometric augmentation in this pipeline. Train split only (rotating val would change what the early-stop metric means), per-worker RNG streams, reuses `so3_probe.rotate_sample`. |
| `scan_checkpoints.py --dump_probs` + `label_harness.py --gt_pkl_dir` | [Auto-accept on the common split](#auto-accept-measured-on-the-same-157-parts-as-every-checkpoint-f1-2026-09-04) | Put the risk–coverage curve on the same 157 real parts as every checkpoint F1. **80% of faces auto-accept at a certified 95% precision** (mftrcad said 11.5%), and `random` — not `chamfer` — is the badly-calibrated class. |
| `brepformer/data/vet_fact.py` | [Diagnostics that killed the feature-clustering plan](#diagnostics-that-killed-the-feature-clustering-plan-2026-09-01) | Measure how many bits of the label a *candidate* geometric fact carries, on exact mftrcad GT, before paying to re-measure 12,000 models. Showed the existing `strict_component_size` binning throws away **+0.62 bits** of side-count signal. |
| `brepformer/data/validate_geometric_rules.py` | [`data/validate_geometric_rules.py`](#datavalidate_geometric_rulespy) | Measure whether geometric_rules.py's denials are trustworthy, using mftrcad ground truth. |
| `brepformer/data/validate_vision_classify.py` | [`data/validate_vision_classify.py`](#datavalidate_vision_classifypy) | Measure Claude vision as a *labeller* (not a verifier) against mftrcad ground truth. |
| `brepformer/data/validate_vision_labeling.py` | [`data/validate_vision_labeling.py`](#datavalidate_vision_labelingpy) | Measure whether the Claude vision pass improves or degrades face labels, on ground truth. |
| `brepformer/eval_per_class.py` | [`eval_per_class.py`](#eval_per_classpy) | Per-class face-segmentation report for a trained checkpoint on preprocessed data. |
| `brepformer/gen_script_index.py` | [`gen_script_index.py`](#gen_script_indexpy) | Regenerate the "Complete Script Index" appendix at the end of SCRIPTS.md. |
| `brepformer/scan_checkpoints.py` | [`scan_checkpoints.py`](#scan_checkpointspy) | Rank every saved checkpoint of a run by FACE-level F1 on a held-out split. |
| `brepformer/sweeps/collect_results.py` | [`sweeps/collect_results.py`](#sweepscollect_resultspy) | Score every sweep arm on the same held-out real-part split and emit markdown tables. |
| `brepformer/test.py` | [`test.py`](#testpy) | Test script for BrepFormer B-rep classification. |
| `brepformer/test_preprocessed.py` | [`test_preprocessed.py` ***recommended***](#test_preprocessedpy-recommended) | Test script for BrepFormer using preprocessed data. |
| `brepformer/test_symmetry.py` | [`test_symmetry.py`](#test_symmetrypy) | Verification script for symmetry completion in defeaturing. |

### Inference & labeling

| Script | Documented in | What it does |
|---|---|---|
| `brepformer/geometric_rules.py` | [`geometric_rules.py`](#geometric_rulespy) | Geometric confirmation/denial rules for face class predictions. |
| `brepformer/infer.py` | [`infer.py` ***recommended***](#inferpy-recommended) | End-to-end inference pipeline for BrepFormer. |
| `brepformer/reclassify_visual.py` | [`reclassify_visual.py`](#reclassify_visualpy) | Visual reclassification of face predictions using Rotator + Claude vision. |

### Defeaturing

| Script | Documented in | What it does |
|---|---|---|
| `brepformer/brepsimplifier/primitives.py` | [`brepsimplifier/primitives.py` (CLI)](#brepsimplifierprimitivespy-cli) | Primitive fitting for defeatured STEP files. |
| `brepformer/brepsimplifier/test_data/compare_primitives.py` | [`brepsimplifier/test_data/compare_primitives.py` and `test_partfield/compare_primitives.py`](#brepsimplifiertest_datacompare_primitivespy-and-test_partfieldcompare_primitivespy) | Compare BrepMFR primitive fitting vs PartField's statistical approach. |
| `brepformer/defeature.py` | [`defeature.py` (v1 — original engine)](#defeaturepy-v1-original-engine) | Automatic defeaturing of STEP files using BrepFormer predictions. |
| `brepformer/defeature_v2.py` | [`defeature_v2.py` ***recommended — modern engine with adaptive tolerance***](#defeature_v2py-recommended-modern-engine-with-adaptive-tolerance) | Modern automatic defeaturing of STEP files (v2) — pythonocc 7.9+ features. |
| `brepformer/primitives.py` | [`brepsimplifier/primitives.py` (CLI)](#brepsimplifierprimitivespy-cli) | Re-export from brepformer.brepsimplifier.primitives. |
| `brepformer/run_pipeline.py` | [`run_pipeline.py` ***recommended — runs the full pipeline in one command***](#run_pipelinepy-recommended-runs-the-full-pipeline-in-one-command) | End-to-end BrepFormer pipeline: preprocess -> train -> test -> analyze. |
| `brepformer/test_partfield/compare_primitives.py` | [`brepsimplifier/test_data/compare_primitives.py` and `test_partfield/compare_primitives.py`](#brepsimplifiertest_datacompare_primitivespy-and-test_partfieldcompare_primitivespy) | Compare BrepMFR primitive fitting vs PartField's statistical approach. |

### Visualization & export

| Script | Documented in | What it does |
|---|---|---|
| `brepformer/export_freecad.py` | [`export_freecad.py` ***recommended***](#export_freecadpy-recommended) | Export colored STEP files for FreeCAD visualization. |
| `brepformer/export_model.py` | [`export_model.py`](#export_modelpy) | Export BrepFormer model for inference or deployment. |
| `brepformer/tensorboard_server.py` | [`tensorboard_server.py`](#tensorboard_serverpy) | TensorBoard server launcher for BrepFormer training visualization. |
| `brepformer/visualize.py` | [`visualize.py`](#visualizepy) | Visualization script for BrepFormer analysis results. |
| `brepformer/visualize_defeature.py` | [`visualize_defeature.py`](#visualize_defeaturepy) | BrepFormer defeature comparison viewer. |
| `brepformer/visualize_seg.py` | [`visualize_seg.py`](#visualize_segpy) | BrepFormer face segmentation viewer. |
| `rotator/pipeline.py` | [`rotator/pipeline.py`](#rotatorpipelinepy) | BREP Alignment and Multi-View Rendering Pipeline for Pipe Fittings. |

### Other

| Script | Documented in | What it does |
|---|---|---|
| `brepformer/data/arbitrate_labels.py` | [`data/arbitrate_labels.py`](#dataarbitrate_labelspy) | Combine the teacher's inference, a second teacher, the measured corpus and the vision pass into one label per face -- with the vision pass given the *narrowest* mandate that is still useful. |
| `brepformer/data/download_grabcad_api.py` | [`data/download_grabcad.py` / `data/download_grabcad_api.py`](#datadownload_grabcadpy-datadownload_grabcad_apipy) | GrabCAD downloader driven by the community REST API (Phase 2, second pass). |
| `brepformer/data/download_grabcad_playwright.py` | [`data/download_grabcad_playwright.py`](#datadownload_grabcad_playwrightpy) | Playwright-based GrabCAD downloader for STEP training data. |
| `brepformer/data/harvest_grabcad_slugs.py` | [`data/harvest_grabcad_slugs.py`](#dataharvest_grabcad_slugspy) | Harvest STEP-bearing GrabCAD slugs to grow the corpus toward 1000 parts. |
| `brepformer/data/measure_class_geometry.py` | [`data/measure_class_geometry.py`](#datameasure_class_geometrypy) | Measure exact B-Rep geometry per label class over the hand-labelled defeature corpus. |
| `brepformer/data/probe_grabcad_api.py` | [`data/probe_grabcad_*.py`](#dataprobe_grabcad_py) | Diagnostic: can we hit GrabCAD's community REST API from the browser context? |
| `brepformer/data/probe_grabcad_auth.py` | [`data/probe_grabcad_*.py`](#dataprobe_grabcad_py) | Diagnostic: what makes GrabCAD's file-download endpoint return 200 instead of 401? |
| `brepformer/data/probe_grabcad_dom.py` | [`data/probe_grabcad_*.py`](#dataprobe_grabcad_py) | Diagnostic: dump the download-related DOM of a few GrabCAD model pages. |
| `brepformer/data/probe_grabcad_files.py` | [`data/probe_grabcad_*.py`](#dataprobe_grabcad_py) | Diagnostic: pull GrabCAD's per-model file list out of the Angular scope. |
| `brepformer/data/probe_grabcad_search.py` | [`data/probe_grabcad_*.py`](#dataprobe_grabcad_py) | Diagnostic: find a GrabCAD search/listing endpoint that yields model slugs. |
| `brepformer/data/validate_dense_vision.py` | [`data/validate_dense_vision.py`](#datavalidate_dense_visionpy) | Score the dense multi-pass vision labeller against hand-labelled ground truth. |
| `brepformer/face_geometry.py` | [`face_geometry.py`](#face_geometrypy) | Exact per-face B-Rep geometry, for describing a face to a vision model in words. |
| `brepformer/vision_label.py` | [`vision_label.py`](#vision_labelpy) | Dense-view, multi-pass Claude vision labelling of individual B-Rep faces. |

### Imported libraries (not invoked directly)

| Module | What it provides |
|---|---|
| `brepformer/configs/config.py` | Configuration dataclass for BrepFormer model. |
| `brepformer/data/classes.py` | MFTRCAD 27-class definitions: names, colors, and utilities. |
| `brepformer/data/collator.py` | Batch collation for BrepFormer. |
| `brepformer/data/dataset.py` | Dataset for MFTRCAD B-rep data. |
| `brepformer/data/preprocessed_dataset.py` | Dataset for preprocessed MFTRCAD data. |
| `brepformer/data/preprocessing.py` | Preprocessing utilities for BrepFormer. |
| `brepformer/models/brep_classifier.py` | BrepClassifier: Full model for B-rep classification. |
| `brepformer/models/brep_encoder.py` | BrepEncoder: Stacked Transformer encoder for B-rep graphs. |
| `brepformer/models/layers/attention.py` | Multi-head Attention with Rotary Position Embeddings (RoPE). |
| `brepformer/models/layers/blocks.py` | Basic building blocks for BrepFormer. |
| `brepformer/models/layers/embedding.py` | Embedding modules for BrepFormer. |
| `brepformer/models/layers/encoder_layer.py` | Graph Encoder Layer for BrepFormer. |
| `brepformer/models/pooling.py` | Graph pooling strategies for BrepFormer. |

<!-- END GENERATED SCRIPT INDEX -->

<!-- (appended after the generated index; the generator only rewrites the block above) -->

## Correctness fixes, 2026-09-01

Six defects found by reading the live tree against the data actually in it. They split
cleanly into two groups, and the split is the whole reason this section exists: three are
**inference-side**, need no retraining, and are already applied; three are
**representation-side**, require re-preprocessing and a re-run, and are deliberately NOT
applied yet because the trial6 sweep is mid-flight.

Every claim below was measured, not inferred. Commands to reproduce are inline.

### Applied — no retraining needed

**1. `infer.py` collapsed the label space with argmax-then-map.**
`_remap_result_to_defeature` mapped the 27-class argmax down to 5 classes. The collapse
is many-to-one, so that is not the same as summing each group's probability mass and
taking the largest, and only the latter minimises 0-1 loss in the coarse space. Five
siblings at 0.15 each beat one rival at 0.25; argmax-then-map picks the rival. The two
rules disagree on 35,626 of 334,175 faces (10.7%). `face_probs` also stayed 27-wide
while `face_preds` became 5-class, so any downstream confidence gate was reading a
different label space from the predictions it gated.

Now marginalizes first, rewrites `face_probs` into the coarse space, and keeps the fine
distribution as `face_probs_27`. Same fix applied to the 27->8 `_remap_result_to_real`.

Measuring the fix also surfaced the more important structural fact: **the 27->5 groups
are wildly uneven** — `cut` absorbs 21 of the 27 fine classes, `random` and `fillet` one
each. A completely uniform (i.e. uninformative) 27-class posterior therefore puts 77.8%
of its mass on `cut` and the plain argmax confidently returns `cut`. Correct for
accuracy, ruinous for macro-F1. Hence `--coarse_decision`:

```bash
# Bayes-optimal for accuracy; diffuse mass lands on `cut` by construction.
python -m brepformer.infer --step part.step --checkpoint CKPT --5_classes \
    --coarse_decision argmax    # default

# Divides each marginal by its group size first. Use this when the number you report
# is macro-F1, and to stop `cut` winning on ignorance alone.
python -m brepformer.infer --step part.step --checkpoint CKPT --5_classes \
    --coarse_decision balanced

# The asymmetry, verified directly:
python - <<'EOF'
from brepformer.infer import _marginalize_probs, _coarse_argmax
from brepformer.data.classes import CLASS_TO_DEFEATURE, DEFEATURE_NUM_CLASSES
c = _marginalize_probs([[1/27]*27], CLASS_TO_DEFEATURE, DEFEATURE_NUM_CLASSES)
print([round(p,3) for p in c[0]])                                   # [.037 .074 .074 .037 .778]
print(_coarse_argmax(c, CLASS_TO_DEFEATURE, 5, 'argmax'))           # [4]  -> cut
print(_coarse_argmax(c, CLASS_TO_DEFEATURE, 5, 'balanced'))         # [0]
EOF
```

**2. `geometric_rules.py` co-axiality tested direction only.**
The test was `abs(dot(axis_i, axis_j)) > 0.95`, which every pair of parallel cylinders on
a part satisfies no matter where they sit — the axis *location* was never even extracted,
only the direction. An axis is a line, not a direction. `_axes_coaxial` now also requires
the offset between a point on each axis to be parallel to that axis (perpendicular
component below a radius-scaled tolerance, so the test is size-invariant).

```bash
# On the fixture plate this was measured against: 740 axial faces, one axis cluster
# before, 189 after. The plate is 300x400 mm with M8 holes on a 25 mm pattern -- about
# 12x16 = 192 holes -- so 189 is essentially exact.
python - <<'EOF'
from brepformer.geometric_rules import _load_step_faces, _get_face_geometry, _axes_coaxial
p = ("brepformer/data/hard_samples/cmm-fixture-plate-300-mm-x-400-mm-x-12-mm-"
     "m8-thread-holes-25-mm-pattern-1__CMM_Fixture_Plate_300_mm_x_400_mm_x_12_mm_"
     "M8_Thread_Holes_25_mm_Pattern.step")
shape, faces = _load_step_faces(p)
geom = _get_face_geometry(faces)
axial = [i for i, g in enumerate(geom) if g["axis"] is not None]
print(len(faces), "faces,", len(axial), "with an axis")
EOF
```

**3. `must_be_planar` denied every conical chamfer.**
A chamfer is planar only when it breaks a *straight* edge. Chamfer a hole, a boss or a
shaft — a circular edge — and the face is a cone. Classes 0/26 (`chamfer`, `chamfer_2`)
and 5-class 2 (`chamfer`) now carry `may_be_conical`, which exempts cones from the
planarity gate. This is what produced the recorded 100% chamfer_2 corruption; the
previously-recorded diagnosis (area caps) was wrong about the mechanism for the first
53/53, and right about the residual — see below.

**Rules are now tagged.** Every verdict carries a `rule` field naming the rule that
produced it, so a corruption rate can be attributed to a specific rule instead of matched
out of a human-readable string. `validate_geometric_rules.py` prints the breakdown:

```bash
python -m brepformer.data.validate_geometric_rules --limit 60 --label_space 27 \
    --out /tmp/geo.json
```

Feeding ground truth in as predictions, so every DENY is provably false, 60 models /
1412 faces, before -> after these fixes:

| metric | before | after |
|---|---|---|
| overall false-denial rate | 15.0% | **13.8%** |
| `chamfer_2` falsely denied | 100% (53/53) | **71.7% (38/53)** |

And the new per-rule attribution, which is the actionable part:

```
must_be_planar        82  (rectangular_passage_2 70, 2sides_blind_step 11, v_shaped_blind_slot 1)
must_be_curved        69  (circular_end_pocket 51, blind_hole 13, circular_blind_slot 4)
area_fraction_max     38  (chamfer_2 38)
large_planar_stock     6  (rectangular_through_step 2, triangular_blind_step 1, ...)
```

All 38 residual `chamfer_2` denials are `area_fraction_max`. That **refutes the claim
that the area caps are negligible**: they are negligible corpus-wide (0.34% of faces) and
decisive for this one class. Corpus-wide rates hide per-class catastrophes, which is the
same lesson that produced this file's "selector, not corrector" rule.

This attribution then drove two further fixes — the area caps and the type constraints
were both **measured** rather than guessed, taking the false-denial rate from 13.8% to
**0.6%**. See ["Label harness"](#label-harness--measuring-any-labeller-before-trusting-it-2026-09-01)
below for the measurement scripts, the per-class distributions and the final numbers.
The rules still run as `--geo_mode advise`: `benefit` mode shows they recover 0 of 374
corrupted labels, so they select faces for review and never rewrite one.

### NOT applied — these need re-preprocessing and a re-run

Do not apply these while a sweep is running. `trial6_sweep.py` launches a fresh
`train_preprocessed.py` per arm, so editing the model mid-sweep would silently give later
arms a different architecture from earlier ones and destroy the comparison.

**4. The whole face-attribute pathway reads the wrong columns.**
`normalize_geometry` (`data/step_to_graph.py:618`) emits the native MFTRCAD layout —
verified identical to the raw `mftrcad/graphs/*.json`, so this is the dataset's format and
the preprocessing is right:

```
face_attr[14] = [plane, cylinder, cone, sphere, torus, other, 0, 0, 0, area, 0, cx, cy, cz]
edge_attr[15] = [0, convex, smooth, length, concave, circle, 0, line, other_type, 0 x6]
```

`GraphNodeFeature.forward` (`models/layers/embedding.py:243`) instead reads an
index-based layout that no data in this repo has ever had. What each encoder actually
receives:

| encoder | intended | actually gets |
|---|---|---|
| `face_type_encoder` | type index 0-5 | col 0 = the *plane* bit, so **6 surface types collapse to plane/not-plane** |
| `face_area_encoder` | area | col 1 = the cylinder bit |
| `face_centroid_encoder` | (cx,cy,cz) | cols 2:5 = cone/sphere/torus bits, ~96% zero |
| `face_rationality_encoder` | 0/1 | col 5, always 0 |
| `face_loop_encoder` | loop count | col 6, always 0 |
| `face_normal_encoder` | (nx,ny,nz) | cols 7:10 = `[0, 0, area]` |
| `face_bbox_encoder` | bbox extent | cols 10:13 = `[0, cx, cy]` |
| `face_reversed_encoder` | 0/1 | col 13 = `cz`, truncated to int then clamped to 0..1 |
| `edge_len_encoder` | length | col 1 = the convex flag (real length is col 3) |
| `edge_conv_encoder` | convexity | col 9, **always 0** — convexity is entirely dead |

Losing convexity matters most: it is the single most diagnostic signal separating a
fillet from a chamfer or a pocket wall. Surface type collapsing to a binary is the
second. The model still reaches 0.94 because `face_grid` — 7-channel UV samples carrying
positions and normals — feeds a CNN that recovers most of the geometry.

```bash
# Reproduce: the columns read as centroids are all zero, and the layout is native.
python - <<'EOF'
import pickle, glob, numpy as np, json
np.set_printoptions(precision=3, suppress=True, linewidth=200)
d = pickle.load(open(sorted(glob.glob('brepformer/data/mftrcad_processed/train/*.pkl'))[0],'rb'))
fa = np.asarray(d['face_attr']); print(fa[:4]); print('cols 2:5 ->', fa[:4,2:5].ravel())
_, g = json.load(open('brepformer/data/mftrcad/graphs/20240116_231044_0_result.json'))
print('raw graph identical:', np.allclose(np.asarray(g['graph_face_attr']), fa))
EOF
```

The fix keeps every module shape identical (so checkpoints still load): argmax cols 0:6
for the type index, area from col 9, centroid from cols 11:14, one-hot indices for edge
type and convexity, real length from col 3; feed explicit zeros to the four encoders
whose inputs the MFTRCAD format simply does not carry. Semantics of the learned weights
change, so warm-start arms must be re-run, not fine-tuned.

**5. The D2 descriptor channel carries no geometry.**
Same root cause, second consumer. `preprocess.py:346` and `step_to_graph.py:652` both
read `face_attr[:, 2:5]` as centroids. Inference additionally reads `[0, 0, area]` as
normals and omits `face_grids=`, so it takes the Gaussian-kernel branch while training
took the UV-histogram branch — a live train/inference skew on top of the column bug.

```bash
# 51% of face pairs (mftrcad) and 78% (trial6) have all D2 mass in bin 0; the rest are
# uniform 1/64 across all bins. Either way: zero information.
python - <<'EOF'
import pickle, glob, numpy as np
for c in ['mftrcad_processed','trial6_processed']:
    degen = tot = 0
    for f in sorted(glob.glob(f'brepformer/data/{c}/train/*.pkl'))[:20]:
        d = pickle.load(open(f,'rb')); n = int(d['num_faces'])
        sub = np.asarray(d['d2_distance'])[:n,:n,:]; m = sub.sum(-1)
        degen += int((sub[...,0] >= m*0.999).sum()); tot += m.size
    print(c, 'all D2 mass in bin 0: %.1f%%' % (100*degen/tot))
EOF
```

Fixing it requires re-preprocessing (hours for mftrcad's 28.5k models) and retraining.
**It is not a trial6 confound**: the defect is identical across every arm, treatment and
control, so the A/B still measures the pcq data's contribution — over a uniformly
degraded representation. Every trial number to date is a floor.

**6. Two smaller model-side issues.**
- `GraphAttnBias.edge_conv_layer` (`embedding.py:369`) is constructed and never called.
  The documented multi-hop edge aggregation does not happen, and the parameters sit dead
  in every checkpoint.
- `MultiheadAttention` is built with `max_seq_len=512` while training runs
  `--max_faces 1000`. Latent only because `use_rope` defaults to False; enabling RoPE
  without raising `max_seq_len` would crash in `reshape_for_broadcast`.

Also checked and NOT a bug: the fully-masked-row guard in `attention.py:217` does fire,
because `GraphAttnBias` writes `-inf` into both rows and columns of padded pairs even
though `attn_mask` alone only masks columns. And `edge_path` is `None` throughout — no
pickle carries it — so the hop-decay loop is dead code rather than a wrong computation.

---

## Label harness — measuring any labeller before trusting it, 2026-09-01

`brepformer/data/label_harness.py` generalises `validate_geometric_rules.py` into the
single place a label-producing component gets scored against mftrcad's exact per-face
ground truth. It answers three different questions and needs all three, because each one
alone is gameable.

| mode | question | how a bad component escapes the *other* modes |
|---|---|---|
| `harm` | how many **correct** labels does it destroy? | — (this is the decisive one) |
| `benefit` | how many **wrong** labels does it fix? | a do-nothing component scores 0% harm, which looks perfect |
| `select` | at what confidence can predictions be **auto-accepted**? | a component can be accurate on average yet be confidently wrong on exactly the rare classes |

### Run it

```bash
# harm: feed ground truth in as the component's input, so every change is provably false.
conda run -n brep_mfr python -m brepformer.data.label_harness \
    --mode harm --component geometric_rules --limit 60 --label_space 27

# benefit: corrupt the truth with *plausible* confusions first, then see what comes back.
conda run -n brep_mfr python -m brepformer.data.label_harness \
    --mode benefit --limit 60 --noise confusable --noise_rate 0.3

# select: risk-coverage on real predictions. Needs infer.py .json output (face_probs),
# NOT .seg, and the --label_space must match the checkpoint's head width.
conda run -n new_brepmfr python -m brepformer.data.label_harness \
    --mode select --predictions /tmp/mftrcad_select_preds --limit 60 --label_space 5
```

Add `--out results.json` for machine-readable output, `--show_all` to list classes with
nothing to report, `--seed` to change the model sample (fixed by default so two runs are
comparable).

### Generating predictions on ground-truth data for `--mode select`

The existing `brepformer/inference/*` outputs are on GrabCAD hard samples, which have no
ground truth, so `select` needs a fresh inference run over mftrcad. Sample the same
seeded model list the harness uses, so the two overlap:

```bash
python - <<'EOF'
import random, pathlib
d = pathlib.Path("brepformer/data/mftrcad/steps")
steps = sorted(p for p in d.iterdir() if p.suffix.lower() in {".step", ".stp"})
random.Random(0).shuffle(steps)                     # same seed as --seed 0
out = pathlib.Path("/tmp/mftrcad_select_sample"); out.mkdir(exist_ok=True)
for s in steps[:60]:
    t = out / s.name
    if not t.exists(): t.symlink_to(s.resolve())
EOF

# Use the *preprocessed* path: it skips OCC entirely, which is both ~50x faster and
# avoids the crash below.
conda run -n new_brepmfr python brepformer/infer.py \
    --step_dir /tmp/mftrcad_select_sample \
    --checkpoint brepformer/brepsimplifier/model_checkpoint/f1=0.9415.ckpt \
    --output_dir /tmp/mftrcad_select_preds \
    --preprocessed_dir brepformer/data/mftrcad_processed
```

**Two traps hit while doing this, both silent:**

1. `infer.py` in the `brep_mfr` env (py3.9, pythonocc 7.9.0) dies with
   `free(): double free detected in tcache 2` before processing a single file, with and
   without `--no_preprocessed` and with `CUDA_VISIBLE_DEVICES=""`. Run inference in
   **`new_brepmfr`** (py3.11, pythonocc 7.9.3), which works. The harness itself runs in
   either env.
2. `f1=0.9415.ckpt` is a **5-class** checkpoint, and without `--5_classes` `infer.py`
   prints 27-class *names* over a 5-class posterior. Scored against 27-class ground truth
   that read as **7.6% accuracy with `stock` at 0%** — a units error that looks exactly
   like a broken model. The harness now aborts when the posterior width disagrees with
   `--label_space`, and collapses mftrcad's 27-class truth through `CLASS_TO_DEFEATURE`
   when `--label_space 5`.

### Measured: harm driven from 15.0% to 0.6%, and it still fixes nothing

`--mode harm` on the same seeded sample after each fix. The harness found every step of
this; none of it was visible on pseudo-labels.

| state | false-denial rate | dominant cause |
|---|---|---|
| as shipped | 15.0% | direction-only co-axiality; cones denied as non-planar |
| + co-axiality and cone fixes | 13.8% | `area_fraction_max` on `chamfer_2` (72%) |
| + area caps measured, not guessed | 11.3% | `must_be_planar` on `rectangular_passage_2` (100%) |
| + type constraints measured, not guessed | **0.6%** (16/2544, 100 models) | `large_planar_stock`, 13 of the 16 |
| same, `--label_space 5` | **0.4%** | `large_planar_stock`, 2 of 2 |

Two measurement scripts drove the last two rows; re-run them before adding any threshold:

```bash
# Per-class area-fraction distribution -> sets area_fraction_max.
conda run -n brep_mfr python /tmp/measure_area.py     # 300 models, ~2 min
# Per-class surface-type mix -> decides must_be_planar / must_be_curved.
conda run -n brep_mfr python /tmp/measure_stype.py    # 300 models, ~2 min
```

What they showed: `chamfer_2` has a **median** area fraction of 0.158 and a max of 0.861,
so the 0.10 cap denied 62% of it; `rectangular_passage_2` is **0% planar / 100% curved**,
so `must_be_planar` was exactly inverted; `circular_end_pocket` is 58% planar and
`blind_hole` 52% planar, because a label names a *feature* and a feature owns its floor
and walls as well as its characteristic surface. Constraints are now present only where
measured to deny <5%. The 5-class `hole` and `cut` lost theirs entirely — `cut` unions 21
fine classes and has no characteristic type at all.

The residual `large_planar_stock` denials are a genuine heuristic tension (a real
`rectangular_through_step` face can legitimately exceed 15% of total area), not a
misencoded assumption, so they are left in place at 0.6%.

**`--mode benefit --limit 100 --noise_rate 0.3` — still 0 of 374 fixed.**

```
corrupted faces fixed:     0/374 =  0.0%
corrupted faces churned: 117/374 = 31.3%   (changed, still wrong)
clean faces broken:       13/2170 =  0.6%
NET: -13 labels (REGRESSION)
```

Zero recovery, and it is structural rather than tunable: every DENY suggests `stock` or
`through_hole` and nothing else, so the rules have no vocabulary in which to restore a
`chamfer`↔`chamfer_2` confusion. This settles `--geo_mode`. Even now that the rules are
nearly harmless, rewrite mode is a strict regression — they are a **detector with a 0.6%
false-alarm rate, not a corrector**, and should only ever flag a face for review.

### Measured: confidence is unusable on `chamfer` — the auto-accept budget

`--mode select --label_space 5`, 59 mftrcad models / 1388 faces, `f1=0.9415.ckpt`.
(This is a defeature-trained checkpoint scored on mftrcad collapsed 27→5, so the 63.9%
is a cross-corpus number and not a claim about the model; the *ranking* is the point.)

```
AUGRC 0.2310   (0.3610 = confidence uninformative, 0 = every error ranked last)

>=99% precision: accept  1.2% of faces
>=95% precision: accept 11.5% of faces
>=90% precision: accept 17.7% of faces

class      faces   acc   mean conf   conf when WRONG
cut          791   58%     0.893        0.858
random       484   82%     0.902        0.874
chamfer       64   12%     0.892        0.898   <-- higher when wrong
hole          37   32%     0.785        0.811   <-- higher when wrong
fillet        12   83%     0.955        0.927
```

For `chamfer` and `hole`, mean confidence is **higher on the errors than on the hits**.
No global threshold can auto-accept those two classes — raising the bar admits their
errors preferentially. That is the concrete argument for a per-class accept threshold
(and mandatory human review on `chamfer`) rather than one global confidence cut, and the
number to re-measure against every new checkpoint.

---

## Rebuilding synthetic labels: a second opinion instead of a mask, 2026-09-01

The pseudo-label pipeline was a single chain — `infer.py` teacher → per-class-quantile
confidence mask in `merge_suitable_samples.py` → train — with two faults no threshold can
fix. There is **no second opinion**, so the student inherits the teacher's errors and
self-distillation ratchets them in. And **masking deletes the hard faces** rather than
resolving them, which falls hardest on the rare classes the corpus most needs (`chamfer`
keep rate 0.4% at a 0.7 global cut).

Two new components replace that with two independent labellers and a routed disagreement.

| script | what it is |
| --- | --- |
| `brepformer/data/geo_label.py` | Labels faces from exact B-Rep geometry using the *already measured* `class_geometry_stats*.json` posterior over 24 facts. Emits `infer.py`-format JSON so the harness scores it unchanged. |
| `brepformer/data/fuse_labels.py` | Fuses network + geometry labels, and writes a **binary-choice** review queue for the faces they disagree on. |

### Headline: geometry beats the trained network on real ground truth

All numbers below: 59 mftrcad models / 1388 faces, exact GT, 5-class, `--mode select`.
`base AUGRC` is each labeller's own error rate — what uninformative confidence would score
— so **compare each AUGRC to its own base, never across label spaces.**

| labeller | accuracy | `chamfer` | AUGRC | base | auto-accept @≥95% |
| --- | --- | --- | --- | --- | --- |
| network (`f1=0.9415.ckpt`) | 63.9% | **12%** | 0.2310 | 0.3610 | 11.5% |
| geometry (`geo_label`) | **72.6%** | **86%** | 0.1804 | 0.2738 | 3.8% |
| fused (`fuse_labels`) | **72.6%** | **86%** | **0.1697** | 0.2738 | **12.0%** |

A measured posterior over exact geometry beats the trained transformer by 8.7 points
overall and by **7× on `chamfer`**, the class the whole defeaturing task turns on. That is
less a statement about geometry than about the model: `face_attr` reads an index layout the
data never had, so edge convexity — the single best fillet-vs-chamfer discriminator — is
identically zero in the network's input and a first-class fact here.

### Why fusion, and why the review queue is binary

```
agree     65.2% of faces   80.6% precision
disagree  34.8% of faces   geo right 57.8%   net right 32.7%   BOTH WRONG only 9.5%
```

Three decisions follow, each measured rather than assumed:

- **Agreement is the accept signal; network confidence is not.** Inside the agreeing set,
  tightening network confidence 0.0 → 0.95 moves precision 80.6% → 81.5% while coverage
  falls 65% → 41%. It buys nothing — consistent with confidence being *anti-correlated*
  with correctness on `chamfer`.
- **Geometry is the default on disagreement, not the tie-break** — right nearly 2:1.
- **Both labellers are wrong on only 9.5% of disagreements**, so 90.5% of the time the
  right answer is one of exactly two named candidates. An arbiter picking A-or-B would
  reach ~84% overall on this sample. This is the reframe that makes the Claude imaging
  pass viable: the open 27-way question measured **at chance** on the topological factor,
  while "is this face a chamfer or a fillet" is a question a picture can actually answer.

```bash
# 1. network labels — infer.py double-frees in `brep_mfr`; use new_brepmfr
conda run -n new_brepmfr python -m brepformer.infer --checkpoint <ckpt> \
    --step_dir steps/ --out_dir net_preds/ --5_classes
# 2. geometric labels (~3 s/model, CPU, no GPU and no checkpoint needed)
conda run -n new_brepmfr python -m brepformer.data.geo_label \
    --stats brepformer/data/class_geometry_stats.json \
    --step_dir steps/ --out_dir geo_preds/
# 3. fuse + emit the binary review queue
conda run -n new_brepmfr python -m brepformer.data.fuse_labels \
    --net_dir net_preds/ --geo_dir geo_preds/ --out_dir fused/ \
    --label_space 5 --review_out review_queue.json
# 4. score it like any other labeller
conda run -n new_brepmfr python -m brepformer.data.label_harness --mode select \
    --predictions fused/ --limit 60 --label_space 5
```

### Three design choices that measurement reversed

Each was written the "obviously right" way first and was wrong. `geo_label` keeps the
diagnostics that overturned them so none has to be taken on trust.

**1. Pool evidence by summing, not averaging** (`--sweep_alpha`). Facts are correlated, so
the first implementation averaged the log-likelihood ratios to stop `k` correlated facts
voting `k` times. But averaging also rescales every *informative* fact by `k/(k+1)` when an
uninformative one is added, so it can only dilute — it cost over half the macro-recall and
made fact selection halt after three facts:

```
macro-recall, all 24 facts:   alpha=0 (sum) 0.307    alpha=0.5 0.242    alpha=1 (mean) 0.128
```

Correlation is a *calibration* problem, not a ranking problem. Handle it downstream.

**2. Use every fact; do not select a subset** (`--select_facts`, 120 models, split by
model). Greedy forward selection won on the half it was chosen on and **lost** on the other:

```
all 24 facts               0.3121 held-out    <- generalises best
greedy-selected 7          0.2877 held-out    (0.3092 on the selection half)
9 hand-picked              0.2520 held-out
```

The split must be **by model, not by face** — faces of one part share a component, an
accessibility field and a bounding box, so a face-level split leaks the answer.

**3. Score the decision with different pooling than you make it with** (`GATE_ALPHA`).
Summed evidence saturates confidence: at `alpha=0` the 5-class table's top-decile precision
(69.6%) came out *below* its own overall accuracy (72.6%) — most-confident faces were its
worst. Since the gate never changes a label, decoupling is free:

```
5-class    decide 0, gate 0     AUGRC 0.2345      decide 0, gate 1  AUGRC 0.1851  <- dominates
           decide 1, gate 1     AUGRC 0.2039
27-class   decide 0, gate 0.5   AUGRC 0.2169  <- best
```

`label_harness --mode select` now honours a `face_confidence` field when a labeller
supplies one (falling back to `max(face_probs)`), and accepts predictions that carry a
confidence but no full posterior — with a weaker id-range label-space guard in that case.

### The 27-class table is the stronger one, and says where geometry's limit is

```
27-class geometry: accuracy 49.1%   AUGRC 0.2182 (base 0.5086)   ≥95% precision: 29.6% of faces
  chamfer_2  53 faces  100%  conf 1.000      triangular_passage  82 faces  0%
  round      12 faces   75%                  rectangular_pocket  34 faces 18%
```

Built on 12,000 models rather than 1,561, it ranks its errors far better relative to its
own baseline and yields a **29.6%** auto-accept budget. `chamfer_2` is called perfectly on
all 53 faces. The 0% classes are all polygonal (`triangular_*`, `6sides_*`): no fact in the
table counts the sides of a feature, so one wall of a hex pocket is indistinguishable from
one wall of a rectangular one. That is a missing fact, not a limit of the approach — and it
is a *feature-level* property, which is the same category error that put the face-centred
vision crops at chance.

### Reusable diagnostics

```bash
# how to pool: alpha x temperature against accuracy AND AUGRC (they disagree)
python -m brepformer.data.geo_label --sweep_calibration --label_space 5 \
    --stats brepformer/data/class_geometry_stats.json --limit 60
# which facts carry the decision, with the overfit kept visible
python -m brepformer.data.geo_label --select_facts --limit 120 --alpha 0.0
```

### Diagnostics that killed the feature-clustering plan, 2026-09-01

`geo_label` scores 0% on every polygonal 27-class label, so the plan was to cluster faces into
AAG concave components, classify the *feature*, and propagate. Measured first. It loses:
31.8% vs 49.1% per-face, 25.5% component purity, 57.3% voting ceiling; with oracle `stock`
removal 51.1% purity, 63.7% ceiling, and voting still loses 22.5% vs 35.4%. Recorded in
README ("Correction: clustering into AAG components does *not* fix it"). Prerequisite added to
`face_geometry.py` — each per-face record now carries `{prefix}_id`, the component id, which
nothing previously exposed (`measure_class_geometry` reads an explicit `CONDITIONS` list so the
extra key is inert there).

```bash
# leakage check on the 27-class numbers before trusting them: eval models that fed the table
# vs those that did not.  Held-out scored HIGHER (49.9% vs 46.1%) -> no memorisation advantage.
conda run -n new_brepmfr python /tmp/clean27.py

# component purity: are concave components features at all?
conda run -n new_brepmfr python /tmp/component_purity.py     # purity, ceiling, vote vs per-face
conda run -n new_brepmfr python /tmp/purity_by_size.py       # 100% @1 face -> 0% @7+
conda run -n new_brepmfr python /tmp/purity_nostock.py       # oracle stock removal

# does an integer side-count fact carry signal the table is throwing away?  +0.62 bits, yes.
conda run -n new_brepmfr python brepformer/data/vet_fact.py
```

`brepformer/data/vet_fact.py` (kept in the repo; the purity scripts above were one-shot and are
not) is the one worth re-running: it measures `I(class ; n_planes, n_cyls)` over
a component against `H(class)` on exact mftrcad GT, and compares it to the same signal passed
through the coarse `strict_component_size` bin the table currently uses. Use it as the template
for vetting any new fact **before** paying to re-measure 12,000 models with
`measure_class_geometry.py` — it costs one pass over 60 models instead of a full re-measure.

### Family-5 sweep result: initialization dwarfs every knob the sweep tested, 2026-09-02

8 of 9 arms complete (`t6_init_def_wf` still running). Read the **`bestface-*` checkpoint
filename, not `best-epoch-*`** — the latter ranks `val/f1`, which is the *model-level*
classification metric and is high for models whose face segmentation is dead.

| arm | data | init | best **face_f1** | best val/f1 (model-level) |
|---|---|---|---|---|
| `t6ctl_base` | defeature GT | scratch | 0.1001 | 0.7064 |
| `t6ctl_wf` | defeature GT | scratch | 0.0887 | 0.7058 |
| `t6ctl_init_ft` | defeature GT | trial1_finetune_ss | **0.8637** | 0.9298 |
| `t6_base` | trial6 masked | scratch | 0.1255 | 0.7947 |
| `t6_w` | trial6 masked | scratch | 0.1982 | 0.6897 |
| `t6_f` | trial6 masked | scratch | 0.1255 | 0.7950 |
| `t6_wf` | trial6 masked | scratch | 0.1942 | 0.7947 |
| `t6_init_def` | trial6 masked | trial1_ss1500 | 0.6958 | 0.8534 |
| `t6_init_def_wf` | trial6 masked | trial1_ss1500 | 0.7065 | 0.8506 |

**Every from-scratch arm collapsed to a single class.** `t6_base` held `val/face_acc` at
*exactly* 0.457 for all 65 epochs while `train/face_acc` climbed 0.366 → 0.510; 0.457 is the
per-model majority-class share of the val split (majority label `cut` = 46.4% of 22,110
supervised val faces), and face macro F1 0.1255 ≈ 1/5 classes correct is the arithmetic
signature of a constant prediction. `bestface-*` checkpoints stop being written at epoch 02.

So the sweep's designed variables — masking strategy, `--weighted_crossentropy`,
`--face_seg_weight` — were measured between two collapsed models and mean nothing there.
`--weighted_crossentropy` is the only knob with a visible effect and it merely moves collapse
from 0.126 to 0.198. **Always pass `--init_weights`.**

**Correction, 2026-09-02:** the first version of this table read `t6_init_def_wf` at 0.7065 and
concluded weighting was worth ~+1.1 pt among the arms that learned. That was a **partial run** —
the arm was at epoch 6. By epoch 31 it reached **face_f1 0.8177** and is still training. So
init + weighting is worth roughly **+12 points** over init alone (0.6958 → 0.8177), and it very
nearly closes the gap to the defeature-GT arm (0.8637) despite training on masked pseudo-labels.
Never read a metric off a live run without checking the epoch.

**The one comparison everyone will want to make is invalid as run.** `t6ctl_init_ft` 0.8637 vs
`t6_init_def` 0.6958 crosses two different validation sets — `defeature_processed` val is
**99.6% supervised** (12,521 of 12,570 faces), `trial6_processed` val is **63.0%** (22,110 of
35,070). To compare the corpora, both arms must validate on the same held-out GT split.

```bash
# the only metric worth reading out of a sweep directory
for a in results/t6*; do ls $a/bestface-*/ 2>/dev/null | grep -o "face_f1=[0-9.]*"; done

# supervision census -- run this on any new *_processed dir BEFORE training on it
python brepformer/data/supervision_census.py brepformer/data/trial6_processed
```

### The fix for the feature partition: learn the prune, 2026-09-02

The negative result above said the AAG concave partition is unfixable by better pooling —
ceiling 56-57%, below per-face accuracy. `brepformer/data/edge_prune_probe.py` tests the
FeatureFox route (arXiv:2604.26770, verified: *FeatureFox: Sample-Efficient Panoptic Graph
Segmentation for Machining Feature Recognition in B-Rep 3D-CAD Models*, Fuchs, Kacan, Haag,
Lohse, 29 Apr 2026): instead of taking components of the raw adjacency graph, train a binary
edge classifier for "same feature?", prune what it rejects, then take components.

```bash
conda run -n new_brepmfr python brepformer/data/edge_prune_probe.py --limit 400
```

```
400 models -> 11944 train edges / 11938 test edges, 43.0% same-feature in train
held-out edge AUC = 0.945   (dihedral angle alone = 0.471, convexity alone = 0.531)

partition                           pure %  ceiling %  components
existing strict concave (baseline)    24.8       56.3         659
learned edge prune                    83.5       88.0        1786
oracle same-label prune              100.0      100.0        1977
```

**Ceiling rises 56.3% → 88.0%**, purity 24.8% → 83.5%, from 11 cheap attributes
`face_geometry` already emits and seconds of CPU. It improves with scale (at `--limit 60`:
AUC 0.895, ceiling 83.3%), and the baseline stays flat, so this is not a sample-size artifact.

**Why the hand-built rule never had a chance:** dihedral angle alone scores AUC **0.471** and
convexity alone **0.531** — both at chance on this task. Those are exactly the two quantities
`part_adjacency` thresholds to build concave components. The signal is real but lives only in
the *combination*, which is why a learned prune finds it and a hand threshold cannot.

Notes on method, since both are easy to get wrong: the split is **by model, never by edge**
(edges of one part share faces and a bbox), and edge features are encoded **symmetrically**
(`min`/`max`, not iteration order) because the pair is undirected. Ground truth is
same-*semantic-label*, not same-instance — mftrcad has no instance ids — so the ceiling is
optimistic for instance segmentation but exact for the per-face labelling task we run.

### Correction: we were reporting AURC as AUGRC, and no auto-accept budget was certified, 2026-09-02

`compute_augrc` divided the running error count by `k`, the *accepted* count, which is **AURC**
— the incumbent metric (Jaeger et al., ICLR 2023, arXiv:2211.15259). Generalised risk divides
by `n`. AUGRC (Traub et al., NeurIPS 2024, **arXiv:2407.01032**) exists precisely because
AURC's `1/k` denominator gives high-confidence failures a harmonic weight, which breaks
monotonicity and **amplifies the influence of label noise** — which this corpus has. Fixed:
`compute_risk_coverage` now returns both, verified against the closed form
`(1-AUROC_f)·acc·(1-acc) + ½(1-acc)²` to within 0.004.

Each has its **own** uninformative baseline and they are different numbers. A random ranking has
`AUROC_f = 0.5`, so `augrc_base = (1-acc)/2` — half the error rate — while `aurc_base` is the
error rate. The previously documented baselines (0.27 for 5 classes, 0.51 for 27) were the AURC
ones.

Re-scored, 59 mftrcad models / 1388 faces, 5-class:

| labeller | acc | AUGRC (base) | AURC (base) | ≥95% observed | ≥95% certified |
|---|---|---|---|---|---|
| network | 63.9% | 0.1451 (0.1805) | 0.2310 (0.3610) | 11.5% | **none** |
| geometry | 72.6% | **0.1069** (0.1369) | 0.1804 (0.2738) | 3.8% | **none** |
| fused | 72.6% | 0.1070 (0.1369) | **0.1697** (0.2738) | **12.0%** | **none** |

**This reverses a published claim.** "Fusion improves the ranking, AUGRC 0.1804 → 0.1697" was
an AURC artifact: on AUGRC, fusion and geometry are a dead tie (0.1070 vs 0.1069, geometry
marginally ahead). Fusion's real win is the *operating point* — a 12.0% auto-accept budget at
≥95% versus geometry's 3.8%, and it is the only labeller with any certified budget at all
(11.7% at ≥90%). Fuse for the budget, not for the ranking.

**And the auto-accept budgets we reported are not statistically supported.** `--mode select`
now prints a Clopper-Pearson lower bound alongside the empirical cut
(`binom_precision_lower_bound`, `DELTA = 0.05`, with a `delta/log2(m)` multiplicity correction
per Geifman & El-Yaniv arXiv:1705.08500 — scanning every `k` and keeping the best is selecting
on the data the bound is computed from). Exact binomial rather than normal/Hoeffding because
for a 0/1 loss the exact inversion is both valid and strictly tighter (RCPS, JACM 2021).
**At ≥95%, no labeller certifies any coverage** — "3.8% at 95%" was the precision *on this
sample*, computed from ~53 faces. To claim a 95% gate we need more exactly-labelled eval faces;
at ≥90% the fused labeller certifies 11.7%.

**And the ceiling converts into accuracy** (`--classify`). A component-level classifier over the
pruned partition, with feature-level descriptors no per-face fact can express (integer plane /
cylinder / cone counts = the side count, plus an occluded-planar-face count as a bottom-face and
hence through-vs-blind proxy), propagating one label to every face:

```bash
conda run -n new_brepmfr python brepformer/data/edge_prune_probe.py --limit 400 --classify
```

```
per-face accuracy from a component-level classifier (same held-out models):
  learned edge prune                62.8%
  existing strict concave           34.9%
```

**62.8% vs 34.9% on identical held-out models** — the partition is the whole difference. For
scale, `geo_label`'s per-face 27-class accuracy is 49.1%, but that was measured on a different
60-model sample with a table built from 12,000 models, so treat +13.7 as indicative rather than
a matched comparison; 62.8 vs 34.9 is the controlled number. Hard voting over the existing
partition scored 31.8%, so the classifier itself is worth only ~3 points on the bad partition
and ~28 on the good one.

### Removing the val-set confound: no sweep arm beat its own initialization, 2026-09-02

The headline family-5 comparison (`t6ctl_init_ft` 0.8637 vs `t6_init_def` 0.6958 face macro F1)
was **invalid as run**, because each arm was scored on its own training corpus's validation
split and those splits differ in supervision density — `defeature_processed` val is 99.6%
supervised, `trial6_processed` val is 63.0% (`brepformer/data/supervision_census.py`). Scoring
every arm on **one** held-out set removes the confound. The natural choice is
`defeature_processed/test`: 157 real hand-labelled parts, 5-class, and the split the deployed
`defeature_v2.py` is judged on.

This is a forward pass per model, no training, so it runs on CPU and leaves the GPU to the
trainer. `--device` was added to `scan_checkpoints.py` for exactly this:

```bash
conda run -n new_brepmfr python -m brepformer.scan_checkpoints \
  --ckpt_dirs \
    "results/trial1_finetune_ss/best-epoch=32-val" \
    "results/trial1_ss1500/best-epoch=57-val" \
    "results/t6ctl_init_ft/bestface-epoch=00-val" \
    "results/t6ctl_init_ft/bestface-epoch=01-val" \
    "results/t6ctl_init_ft/bestface-epoch=02-val" \
    "results/t6_init_def/bestface-epoch=00-val" \
    "results/t6_init_def/bestface-epoch=22-val" \
    "results/t6_init_def/bestface-epoch=25-val" \
    "results/t6_init_def_wf/bestface-epoch=25-val" \
    "results/t6_init_def_wf/bestface-epoch=29-val" \
  --data_dir brepformer/data/defeature_processed --split test --label_space 5 \
  --device cpu --out results/common_eval/common_val_defeature_test.json
```

Only checkpoints whose `bestface-*` directory still holds a `.ckpt` are listed — lightning's
top-k pruning deletes the file but leaves the directory, so most `bestface-epoch=NN-val/` dirs
in these runs are empty. The two `trial1_*` entries are the **initializations** the arms were
launched from, and including them is the whole point.

```
rank    macro  weight  minCls    random    hole  chamfer  fillet     cut  run/epoch
1      0.8886  0.8951  0.8197     0.820   0.954    0.874   0.887   0.908  trial1_finetune_ss/best-epoch=32-val   <- INIT
2      0.8860  0.8959  0.8192     0.819   0.955    0.857   0.887   0.911  t6ctl_init_ft/bestface-epoch=01-val
3      0.8819  0.8892  0.8041     0.804   0.950    0.865   0.886   0.905  t6ctl_init_ft/bestface-epoch=00-val
4      0.8808  0.8905  0.8134     0.813   0.947    0.853   0.884   0.906  t6ctl_init_ft/bestface-epoch=02-val
5      0.8564  0.8619  0.7776     0.778   0.976    0.836   0.839   0.853  trial1_ss1500/best-epoch=57-val        <- INIT
6      0.8454  0.8556  0.7765     0.777   0.971    0.810   0.821   0.849  t6_init_def_wf/bestface-epoch=29-val
7      0.8339  0.8484  0.7651     0.765   0.964    0.778   0.819   0.844  t6_init_def_wf/bestface-epoch=25-val
8      0.7669  0.7689  0.6389     0.639   0.889    0.821   0.694   0.792  t6_init_def/bestface-epoch=00-val
9      0.6807  0.6878  0.4098     0.410   0.893    0.666   0.713   0.722  t6_init_def/bestface-epoch=22-val
10     0.6525  0.6330  0.4437     0.444   0.890    0.646   0.713   0.571  t6_init_def/bestface-epoch=25-val
```

**Four things change, and the first one voids the sweep's headline.**

**1. Not one arm beat the checkpoint it started from.** `t6ctl_init_ft` peaks at 0.8860 against
its init's **0.8886**; `t6_init_def_wf` peaks at 0.8454 against its init's **0.8564**. The
family-5 sweep, nine arms and roughly 29 GPU-hours, produced no checkpoint better than the two
weights files it was seeded with. Every "improvement" in the sweep was measured against that
arm's own validation split, where the init was never scored as a baseline. **Always score the
initialization on the same split as the arms** — it is one forward pass and it is the only
number that says whether training helped.

**2. The corpus gap is ~1 point, not 17.** The unconfounded question is retention against each
arm's own init: defeature exact GT retains **99.7%** (0.8886 → 0.8860), trial6 masked
pseudo-labels retain **98.7%** (0.8564 → 0.8454). The apparent 0.8637-vs-0.6958 chasm was two
different inits (0.8886 vs 0.8564, a 3.2-point gap on its own) plus two different val sets. So
the masked pseudo-labels are close to harmless — and also close to useless, because neither
corpus produced a gain. This is roughly the "good trial6" reading pre-committed in
`project_trial6_open_question`, but for an unwelcome reason: nothing moved either way.

**3. Loss weighting is still the largest knob, and smaller than advertised.** `t6_init_def_wf`
0.8454 vs `t6_init_def` 0.7669 = **+7.9 points** on the common set, against the +12 measured on
trial6's own val. Real, reproducible, still not enough to clear the init.

**4. `t6_init_def` gets monotonically worse the longer it trains** — 0.7669 at epoch 0, 0.6807
at 22, 0.6525 at 25, and `random` collapses from 0.639 to 0.410 — while its *own* val face F1
stayed in the 0.64-0.70 band. That is overfitting to the masked-label distribution, invisible
from inside the run. The `_wf` variant does not show it. Any future arm on masked labels needs
a real-GT split in the early-stopping monitor, not just its own val.

**Scope limit this result puts on "geometry beats the network".** The best checkpoint here
scores **0.874 face F1 on `chamfer`** over 157 real parts. The 12% chamfer recall behind the
`geo_label` comparison was measured on 59 **mftrcad** models remapped 27→5, a different
distribution. Both numbers stand; "the network cannot do chamfer" does not — it is "the network
cannot do chamfer *on remapped mftrcad*". Re-run `label_harness --mode select` against
`defeature_processed/test` before repeating the claim.

### SO(3) robustness: not a catastrophe, but a heavy tail — and no augmentation exists, 2026-09-02

A separate question from the val-set confound above, and worth keeping separate: that one asks
*which corpus/knob helped*, this one asks *does the model depend on the world frame*. FoV-Net
reports UV-Net falling from **98.95 to 35.44** per-face accuracy on MFCAD++ under random test
rotation, so it is a live candidate cause of the synthetic-to-real gap — real GrabCAD parts
arrive in whatever frame their author used.

`brepformer/so3_probe.py` rotates the frame-dependent slots of an already-preprocessed sample
rather than re-running OCC on a rotated STEP: `face_grid[:, 0:3]` points and `[:, 3:6]` normals,
`edge_grid[:, 0:3]` points and `[:, 3:6]` tangents, `face_attr[:, 11:14]` centroid. Everything
else is a rotation invariant and is deliberately untouched — post-`normalize_geometry`
`edge_attr` is length plus convexity/curve-type one-hots, `spatial_pos`/`in_degree` are
topological, and `d2_distance`/`angle_distance` are pairwise shape descriptors. So a drop cannot
be blamed on a corrupted input; the only thing that changed is the frame. `--renormalize`
(default) recomputes bbox centre and scale afterwards, because the normaliser divides by half
the max *axis-aligned* extent, which is itself frame-dependent.

Two frame families are drawn: uniform SO(3) (Shoemake quaternion) and the 24 axis-aligned
90-degree rotations, the latter as the control.

```bash
conda run -n new_brepmfr python -m brepformer.so3_probe \
  --ckpts "results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt" \
          "results/trial1_finetune_ss/best-epoch=32-val/f1=0.9354.ckpt" \
          "results/t6_init_def_wf/bestface-epoch=29-val/face_f1=0.8177.ckpt" \
  --data_dir brepformer/data/defeature_processed --split test --label_space 5 \
  --n_rotations 6 --n_octahedral 3 --device cpu \
  --out results/common_eval/so3_probe_defeature_test.json
```

```
checkpoint                            ident    mean   worst   worst frame  SO3 ret  oct ret
trial1_ss1500/best-epoch=57-val      0.8564  0.7534  0.5898         so3-1    86.9%    90.1%
  biggest per-class losses there: hole 0.976->0.498, chamfer 0.836->0.456, random 0.778->0.539
trial1_finetune_ss/best-epoch=32-val 0.8886  0.8696  0.8470         so3-5    97.1%    99.5%
  biggest per-class losses there: chamfer 0.874->0.800, random 0.820->0.776, cut 0.908->0.874
t6_init_def_wf/bestface-epoch=29-val 0.8454  0.7386  0.5770  octahedral-2   88.2%    85.7%
  biggest per-class losses there: hole 0.971->0.120, random 0.777->0.538, cut 0.849->0.725
```

**The FoV-Net catastrophe does not reproduce.** Worst-case retention is 67–69% of macro F1, not
36%, so rotation is not the dominant cause of our synthetic-to-real gap. Don't reprioritise the
`face_attr` layout fix behind it.

**But the tail is heavy, and it is not about SO(3) specifically.** `t6_init_def_wf`'s worst frame
is `octahedral-2`, an *axis-aligned 90-degree* rotation, at 0.5770 — worse than any of the six
random SO(3) draws. Mean over frames drops to 0.7386 from 0.8454. So the failure mode is
"specific frames", not "non-axis-aligned frames", and averaging over draws hides it: report the
**worst** frame, not the mean. (`octahedral-1` scored *identical* to identity because the draw
returned the identity matrix — a legitimate draw, and a useful sanity check that the rotation
plumbing is exact.)

**`hole` is the fragile class: 0.971 → 0.120.** That is mechanically explainable. A hole is a
cylinder, and what separates it from any other cylindrical face is its *axis direction* — which
after `normalize_geometry` is not in `face_attr` at all (one-hot type, area, centroid, nothing
else). So the model can only recover hole-ness from the UV-grid normals, i.e. from the frame.
This is the same root cause as the layout bug: the discriminative attribute was never handed to
the network, so it substitutes a frame-dependent proxy.

**And the augmentation I previously credited the pipeline with does not exist.** Neither
`brepformer/data/preprocessed_dataset.py` nor `brepformer/train_preprocessed.py` contains any
rotation, flip, or augmentation code — the only `rotat` hit under `brepformer/data/` is the
docstring line in `preprocessing.py` calling D2 a rotation-invariant signature. So this is not
"we augment with 90-degree rotations and need SO(3) too"; **there is no geometric augmentation
at all.** That makes it the cheapest available fix: it needs no re-preprocessing, because
rotating a loaded sample is a tensor op that `so3_probe.rotate_sample` already implements. Wire
that into the training dataset's `__getitem__` behind a flag and the fragile classes above get a
free shot at improvement.

**Note the ranking this reinforces:** `trial1_finetune_ss` ep32 is simultaneously the best
checkpoint on the common split (0.8886) and by far the most frame-invariant (97.1% SO(3), worst
frame 0.8470, a 4.7% loss). It is the production checkpoint on both counts.

#### Round 2, after `t6_init_def_wf` ran to completion and a 10th arm appeared, 2026-09-03

The round-1 table above scored `t6_init_def_wf` at epoch 29 (0.8177 on its own val). The arm then
ran to **968.8 min / epoch 67**, peaking at 0.8448 on its own val — so its round-1 entry was a
partial run and had to be re-scored. Lightning's top-k had meanwhile deleted the ep25/ep29 files,
which is why the round-1 numbers are only recoverable from
`results/common_eval/common_val_defeature_test.json`. **Re-score after a run ends; a mid-run
checkpoint may not survive to be re-scored later.**

The sweep driver also started a 10th arm on its own, `t6_init_ft` — trial6 masked labels from the
*better* init (`trial1_finetune_ss`, 0.8886) rather than from `trial1_ss1500` (0.8564). That is
the arm that could actually have beaten the incumbent, and it is the informative one.

```bash
conda run -n new_brepmfr python -m brepformer.scan_checkpoints \
  --ckpt_dirs "results/trial1_finetune_ss/best-epoch=32-val" \
              "results/trial1_ss1500/best-epoch=57-val" \
              "results/t6_init_def_wf/bestface-epoch=61-val" \
              "results/t6_init_def_wf/bestface-epoch=66-val" \
              "results/t6_init_def_wf/bestface-epoch=67-val" \
              "results/t6_init_ft/bestface-epoch=44-val" \
              "results/t6_init_ft/bestface-epoch=50-val" \
              "results/t6_init_ft/bestface-epoch=52-val" \
  --data_dir brepformer/data/defeature_processed --split test --label_space 5 \
  --device cpu --out results/common_eval/common_val_round2.json
```

```
rank    macro  weight  minCls    random    hole  chamfer  fillet     cut  run/epoch
1      0.8886  0.8951  0.8197     0.820   0.954    0.874   0.887   0.908  trial1_finetune_ss ep32   <- INIT (0.8886)
2      0.8579  0.8662  0.7899     0.790   0.966    0.830   0.841   0.862  t6_init_def_wf ep67
3      0.8564  0.8619  0.7776     0.778   0.976    0.836   0.839   0.853  trial1_ss1500 ep57        <- INIT (0.8564)
4      0.8559  0.8672  0.7924     0.792   0.966    0.816   0.840   0.865  t6_init_def_wf ep61
5      0.8543  0.8629  0.7905     0.790   0.965    0.822   0.837   0.857  t6_init_def_wf ep66
6      0.8426  0.8546  0.7592     0.759   0.944    0.812   0.830   0.868  t6_init_ft ep52  (still training)
7      0.8425  0.8541  0.7560     0.756   0.945    0.816   0.827   0.868  t6_init_ft ep50
8      0.8380  0.8520  0.7540     0.754   0.945    0.798   0.828   0.866  t6_init_ft ep44
```

**Amendment to the round-1 claim.** "No arm beat its own initialization" is now too strong for one
arm: `t6_init_def_wf` ep67 reaches 0.8579 against its init's 0.8564, i.e. **+0.0015**. Call that a
tie, not a win — it is one part in 570 on 157 models, and its own two neighbouring epochs (0.8559,
0.8543) straddle the init. The substantive claim is unchanged: **the best sweep checkpoint still
loses to the incumbent `trial1_finetune_ss` ep32 by 3.1 points**, and 968 GPU-minutes bought a tie.

**The 10th arm is the clean result, and it is negative.** `t6_init_ft` starts from 0.8886 and lands
at **0.8426** — it *destroys* 4.6 points. Its own val face F1 has been flat at ~0.821 for eight-plus
epochs, so this is near its ceiling, not an unfinished run (treat the exact figure as provisional
until it exits).

**Read the two init arms together and the mechanism is unmistakable.** They begin 3.2 points apart
(0.8886 vs 0.8564) and converge to within 1.5 points of each other (0.8426 vs 0.8579) — one pulled
*up* 0.2 points, the other dragged *down* 4.6. Both land at ~0.85. **The trial6 masked labels, not
the initialization, set the ceiling, and that ceiling is about 0.86 on real parts.** Training on
them cannot lift a checkpoint above their own quality; it regresses a better one toward them. This
is why the fix has to be label quality (`fuse_labels.py`, the learned edge prune, the `face_attr`
layout) and not another sweep over training knobs.

#### Final: 11 arms, ~72 GPU-hours, nothing beat the incumbent, 2026-09-04

The sweep finished. Two more arms appeared beyond the nine originally emitted: `t6_init_ft`
(876 min) and `t6_init_ft_wf` (790 min), both on trial6 masked labels from the **better** init
(`trial1_finetune_ss`, 0.8886). All 11 arms rc=0. Final scoring of every surviving checkpoint on
`defeature_processed/test`:

```
rank    macro  weight  minCls    random    hole  chamfer  fillet     cut  run/epoch
1      0.8886  0.8951  0.8197     0.820   0.954    0.874   0.887   0.908  trial1_finetune_ss ep32  <- INCUMBENT / init
2      0.8579  0.8662  0.7899     0.790   0.966    0.830   0.841   0.862  t6_init_def_wf ep67
3      0.8564  0.8619  0.7776     0.778   0.976    0.836   0.839   0.853  trial1_ss1500 ep57       <- init
4      0.8444  0.8555  0.7601     0.760   0.944    0.818   0.831   0.869  t6_init_ft ep70
5      0.8426  0.8546  0.7592     0.759   0.944    0.812   0.830   0.868  t6_init_ft ep52
6      0.8392  0.8517  0.7767     0.777   0.941    0.787   0.840   0.852  t6_init_ft_wf ep29
7      0.8380  0.8520  0.7540     0.754   0.945    0.798   0.828   0.866  t6_init_ft ep44
8      0.8376  0.8430  0.7445     0.745   0.954    0.798   0.855   0.835  t6_init_ft_wf ep47
9      0.8217  0.8212  0.7171     0.717   0.934    0.794   0.855   0.808  t6_init_ft_wf ep49
```

**Bottom line: 11 arms and roughly 72 GPU-hours produced no checkpoint better than the weights
file that already existed.** `trial1_finetune_ss` ep32 remains the best model on real parts by
3.1 points.

**The ceiling claim is now confirmed across all four initialized arms.** They land at 0.8579,
0.8444, 0.8392, 0.8376 — a **2-point spread**, despite their two inits being **3.2 points apart**
(0.8886 vs 0.8564) and despite half of them using loss weighting. Two arms were pulled *up* to
~0.85 and two were dragged *down* to ~0.84. Nothing about the training configuration escapes that
band, because the band is the quality of the trial6 masked labels themselves.

**And loss weighting reverses sign depending on the init**, which retroactively explains the
+7.9-point figure recorded earlier. From the weaker init it helps (`t6_init_def` 0.7669 →
`t6_init_def_wf` 0.8579). From the better init it *hurts* (`t6_init_ft` 0.8444 →
`t6_init_ft_wf` 0.8392). Both arms end up in the same band either way, so weighting was never
"worth +7.9 points" — it was pulling a lagging arm up to a ceiling the other arm reached without
it. **A knob measured from a single initialization tells you nothing about the knob.**

The practical consequences, in order:
1. `defeature_v2.py` should point at `trial1_finetune_ss/best-epoch=32-val/f1=0.9354.ckpt`. That
   was already the finding from the 26-checkpoint scan; 11 more arms did not change it.
2. Stop sweeping training knobs on trial6 labels. The ceiling is the labels.
3. Spend the GPU on label quality and on the input defects instead — the `face_attr`/`edge_attr`
   layout mismatch, the dead D2 descriptor, `fuse_labels.py`, the learned edge prune.

### Bug 4 fixed: the attribute layout the model reads never existed on disk, 2026-09-04

**No re-preprocessing is required — I had assumed otherwise and was wrong.** Everything the
encoder wants is already in the pickles, just at different column indices.

Verified column-by-column across all three corpora before touching any code
(`/tmp/chk_layout.py`, nonzero-fraction and min/max per column of `face_attr` and `edge_attr` in
`mftrcad_processed`, `defeature_processed`, `trial6_processed`). What is on disk, in every case:

```
face_attr[14] = [ one_hot_type(0:6), 0, 0, 0, area(9), 0, cx(11), cy(12), cz(13) ]
edge_attr[15] = [ 0, convex(1), smooth(2), length(3), concave(4), circle(5), 0,
                  line(7), other_type(8), 0, 0, 0, 0, 0, 0 ]
```

What `embedding.py` read instead — the layout `compute_face_attributes` produces *before*
`normalize_geometry`, which therefore never reaches a pickle:

| the model thought it read | what it actually got |
|---|---|
| `face_type` = col 0 | the **"is it a plane"** bit — surface type collapsed to binary |
| `face_area` = col 1 | the "is it a cylinder" bit — **area destroyed** |
| `face_centroid` = cols 2:5 | cone/sphere/torus bits, **identically zero on most parts** |
| `face_normal` = cols 7:10 | `[0, 0, area]` |
| `face_bbox` = cols 10:13 | `[0, cx, cy]` — a shifted centroid |
| `edge_type` = col 0 | always 0, so **one constant edge type** |
| `edge_len` = col 1 | the **convex flag**, not the length (length is col 3) |
| `edge_convexity` = col 9 | always 0, so **convexity is a constant embedding** |

Convexity is the fillet-vs-chamfer discriminator, and BRepNet's ablation prices its removal at
−4.6 IoU, its single largest feature loss. Every trial number in this repo was produced without
it, and without real surface type, area, centroid, edge length or edge type.

**The fix** is a layout-aware decode in `GraphNodeFeature.forward` and `GraphAttnBias.forward`,
selected by a new `BrepClassifierConfig.attr_layout` (`"onehot"` default, `"categorical"` legacy)
and a `--attr_layout` training flag. Every parameter shape is unchanged, so existing checkpoints
still load. Attributes that `normalize_geometry` genuinely drops — loops, normals, bbox,
orientation, the raw dihedral angle — are fed **zeros** rather than reinterpreted from unrelated
columns: the encoder contributes a constant bias, which is honest about the absence and keeps the
checkpoint contract.

**Two traps worth recording, both hit in practice.**

1. **Old checkpoints must keep the old decoding.** Their weights are adapted to the broken
   inputs. Measured on 12 models / 516 faces, the incumbent `trial1_finetune_ss` ep32 scores
   **82.95% under `categorical` and 32.75% under `onehot`.** So flipping the default without
   guarding it would have silently destroyed every published number. `infer.load_model` now
   auto-selects, and everything scored earlier in this file re-scores identically.
2. **`getattr` cannot detect a pre-fix config.** Unpickling a dataclass rebuilds it against the
   *current* class definition, so `getattr(saved_config, "attr_layout")` cheerfully returns the
   new field's default even for a config pickled before the field existed — the first version of
   the detection silently chose `onehot` for old checkpoints. Probe `vars(saved).get(...)`, the
   instance `__dict__`, which only holds keys that were really restored.

The A/B is the same configuration as `t6ctl_init_ft` with only the layout changed:

```bash
conda run -n new_brepmfr python brepformer/train_preprocessed.py \
  --data_dir brepformer/data/defeature_processed \
  --num_classes 5 --num_face_classes 5 --max_faces 1000 \
  --batch_size 2 --accumulate_grad_batches 16 --learning_rate 0.001 \
  --max_epochs 200 --seed 42 --output_dir results --face_segmentation \
  --lr_monitor val/face_f1 --early_stop_monitor val/face_f1 --early_stop_patience 25 \
  --init_weights "results/trial1_finetune_ss/best-epoch=32-val/f1=0.9354.ckpt" \
  --attr_layout onehot --exp_name fix4_onehot_def
```

All 243 tensors load, 0 skipped. **Expect it to start low and have to climb**: the attribute
encoders are being asked to interpret inputs whose meaning changed, so epoch 0 posts face F1
**0.3266**, not ~0.88. The bar it must clear is the incumbent's **0.8886** on
`defeature_processed/test`; anything less means the transformer body was more tuned to the broken
inputs than the correct attributes are worth. Run in progress.

### Rotation augmentation now exists, 2026-09-04

The SO(3) probe found the pipeline had **no geometric augmentation at all** — no rotation, no
flip, nothing — while the corpus is axis-aligned. Added as a train-split-only flag rather than a
preprocessing pass, so it costs no disk and can be A/B'd against every run already recorded here.

```bash
# --rotate_aug none | octahedral | so3   (default none, i.e. historical behaviour)
conda run -n new_brepmfr python brepformer/train_preprocessed.py \
  --data_dir brepformer/data/defeature_processed \
  --num_classes 5 --num_face_classes 5 --max_faces 1000 \
  --batch_size 2 --accumulate_grad_batches 16 --learning_rate 0.001 \
  --max_epochs 200 --seed 42 --output_dir results --face_segmentation \
  --lr_monitor val/face_f1 --early_stop_monitor val/face_f1 --early_stop_patience 25 \
  --init_weights "results/trial1_finetune_ss/best-epoch=32-val/f1=0.9354.ckpt" \
  --rotate_aug so3 --exp_name aug_so3_def
```

Implementation notes, all of which are places it would have been easy to get this wrong:

- **Reuses `so3_probe.rotate_sample`**, so the frames the model trains on are generated by
  exactly the code that measures robustness. One definition of "what carries the frame", not two.
- **`rotate_sample` no longer deepcopies.** It shallow-copies the dict and copies only
  `face_grid`, `edge_grid`, `face_attr` — the three arrays it rewrites. A deepcopy would
  duplicate `d2_distance`, (F+1, F+1, 64) float32 = **256 MB for a 1000-face part**, on every
  `__getitem__`, which would have made augmentation slower than the forward pass. Behaviour in
  the probe is unchanged.
- **Per-worker RNG streams**, seeded `[rotate_seed, worker_id]`. Without this every dataloader
  worker draws the same frame sequence and the augmentation is `num_workers`× less diverse than
  it looks — a classic silent one.
- **Val and test are never rotated.** Rotating val changes what `val/face_f1` means and would
  break comparability with all 11 sweep arms above; robustness is measured after the fact by
  `so3_probe.py`.
- **Renormalisation is on**, so each frame is rescaled the way preprocessing a rotated STEP
  would have rescaled it (the normaliser divides by half the max *axis-aligned* bbox extent,
  which is frame-dependent). Verified: under octahedral the area column is unchanged (axis
  permutation leaves an axis-aligned bbox alone) and under SO(3) it moves 1.3675 → 1.0693.

Verified before use (`/tmp/chk_aug.py`) that for both modes `face_labels` is unchanged, surface
normals stay unit length, the rotation-invariant slots (`spatial_pos`, `in_degree`, `edge_attr`,
`d2_distance`) are bit-identical, and two visits to the same index give **different** frames.

Also repointed `defeature.py` / `defeature_v2.py` `DEFAULT_CHECKPOINT` from
`trial1_ss1500` ep57 (face macro F1 0.8564) to `trial1_finetune_ss` ep32 (**0.8886**) — the
sweep's conclusion, applied.

### Auto-accept, measured on the same 157 parts as every checkpoint F1, 2026-09-04

The auto-accept numbers recorded earlier were on remapped mftrcad; the checkpoint F1 numbers were
on `defeature_processed/test`. Quoting both invites combining two measurements that were never
taken on the same parts. Two small additions put them on one footing:

- `scan_checkpoints.py --dump_probs DIR` — writes per-model `face_preds`/`face_probs` JSONs for
  the pickles it is already scoring. For a 27-class head read at `--label_space 5` it **sums**
  the 27-class mass into each defeature class rather than remapping the argmax: mass can split
  across several 27-class ids that collapse to one defeature class, and the split understates the
  confidence the model actually has — which is exactly the quantity a risk-coverage curve ranks
  on. The F1 path is left as argmax-then-remap so every number above stays comparable.
- `label_harness.py --gt_pkl_dir` — takes select-mode ground truth from a preprocessed split's
  `face_labels` instead of mftrcad `labels/`.

```bash
conda run -n new_brepmfr python -m brepformer.scan_checkpoints \
  --ckpt_dirs "results/trial1_finetune_ss/best-epoch=32-val" \
  --data_dir brepformer/data/defeature_processed --split test --label_space 5 \
  --device cpu --dump_probs results/probs/trial1_finetune_ss_ep32 \
  --out results/common_eval/incumbent_probs.json          # reproduces macro 0.8886 exactly

conda run -n new_brepmfr python -m brepformer.data.label_harness \
  --mode select --label_space 5 --limit 0 \
  --predictions results/probs/trial1_finetune_ss_ep32 \
  --gt_pkl_dir brepformer/data/defeature_processed/test \
  --out results/common_eval/select_incumbent_defeature_test.json
```

**Result — confidence ranks errors well, so review-and-override beats masking.** 157 models,
13,445 supervised faces, overall accuracy 89.6%. **AUGRC 0.0208** against 0.0521 for
uninformative confidence (AURC 0.0285 vs 0.1043) — the ranking carries about 60% of the available
signal.

| target precision | accept (observed) | accept (95%-certified) | threshold |
|---|---|---|---|
| ≥99% | 22.3% | 16.6% | 0.998 |
| ≥98% | 39.1% | 30.4% | 0.994 |
| **≥95%** | **82.6%** | **80.0%** | 0.891 |
| ≥90% | 98.9% | 97.1% | 0.504 |

So **80% of faces can be auto-accepted at a certified 95% precision**, leaving 20% for review —
a far better operating point than a flat 0.7 confidence mask, which accepts everything above one
threshold regardless of class and (per the masking finding) is class-selective in a way that
ratchets `chamfer` and `cut` away.

| true class | faces | acc | mean conf | **conf when wrong** |
|---|---|---|---|---|
| `cut` | 5425 | 89% | 0.936 | 0.785 |
| `hole` | 3032 | 98% | 0.977 | 0.737 |
| `random` | 3011 | **80%** | 0.899 | **0.816** |
| `fillet` | 1568 | 92% | 0.928 | 0.749 |
| `chamfer` | 409 | 94% | 0.958 | 0.801 |

**`random` is the bottleneck and it is the worst-calibrated class**: lowest accuracy (80%) *and*
the smallest gap between mean confidence and confidence-when-wrong (0.899 vs 0.816, a 0.083
margin, versus 0.240 for `hole`). Confident errors on `random` are what caps the ≥99% band at
22%. That argues for **per-class thresholds** — `hole` and `fillet` errors are cheaply
separable, `random` errors are not — and it says the review queue should be seeded with
faces *predicted* `random`, not with the globally least-confident faces.

Note also that `chamfer` reaches 94% accuracy here. "The network cannot do chamfer" was a
remapped-mftrcad result and does not hold on real parts.

### Bugs 5 and 6, 2026-09-04

**Bug 6 — the RoPE frequency table was smaller than the data.** `max_seq_len` defaulted to 512
while every run uses `--max_faces 1000`, and a B-rep sequence is `num_faces + 1` (the extra slot
is the graph token). `--use_rope` would therefore have died on the first part with >511 faces,
with an opaque `view()` shape error deep in the attention stack, because
`reshape_for_broadcast` slices `freqs_cis[:seq_len]` and silently gets a shorter table.
Latent only because `use_rope` defaults False. Two changes: `train_preprocessed.py` now sets
`max_seq_len=max(512, args.max_faces + 1)`, and `reshape_for_broadcast` raises a message that
names the number to raise. No parameter shapes move when `use_rope=False`, so checkpoints are
unaffected (verified: the incumbent still loads, `attr_layout=categorical`, `max_seq_len=512`).

**Bug 5 — `GraphAttnBias.edge_conv_layer` is dead and stays.** Constructed, never called in
`forward`; its parameters are allocated, written into every checkpoint and never receive a
gradient. **Not removed**, deliberately: `load_from_checkpoint` is strict, so deleting it makes
every checkpoint in `results/` unloadable in exchange for a few thousand idle parameters. Marked
in place with what to do instead — wire it into `forward` if multi-hop edge aggregation is ever
wanted, since the module is already the right shape.

Remaining: bug 2, the dead D2 descriptor. `preprocess.py:346` computes pairwise distances between
one-hot type bits rather than centroids, so the channel is empty (51%/78% of face pairs put all
mass in bin 0, the rest uniform). This one *does* need re-preprocessing, so it is the only one of
the four that costs corpus-hours rather than an edit.

### Bug 2: the D2 channel was empty because of a *max_dist*, and it repairs without re-preprocessing, 2026-09-04

I had this filed as the expensive one — "needs re-preprocessing, hours for mftrcad's 28.5k
models". It does not. The mechanism is narrower than the diagnosis assumed.

`compute_d2_descriptor` builds its histogram from **point clouds read out of `face_grid`**, which
every pickle already stores and which were always correct. The face centroids it is also handed
are used for exactly one thing: estimating `max_dist`, the upper edge of the histogram range.
Those centroids were read from `face_attr[:, 2:5]` — the cone/sphere/torus one-hot bits, zero on
most parts — so:

```
face_attr[:, 2:5]   (what was read)   -> max_dist = 1e-06
face_attr[:, 11:14] (the real column) -> max_dist = 2.17053
```

With a 1e-6 range every real distance falls outside the histogram and `np.histogram` returns all
zeros. Measured on `defeature_processed/test/00000.pkl`, **99.2% of face pairs have an entirely
empty D2 histogram**; with the right column, 0% are empty and each pair gets ~7 populated bins.
Over 8 test files the mean empty-pair fraction is **0.892 → 0.000**.

`angle_distance` was never affected: its range is a fixed `[0, pi]` and its normals come from the
grid.

**Three call sites fixed** — all read `face_attr[:, 2:5]`:
`preprocess.py:346`, `data/dataset.py:247`, `data/step_to_graph.py:652`. The last also read
normals from `face_attr[:, 7:10]`, which post-normalisation is `[0, 0, area]`; it now takes them
from the grid and passes `face_grids` through, as `preprocess.py` already did.

**`brepformer/data/repair_descriptors.py`** rewrites `d2_distance` in existing pickles from the
stored grid — no STEP, no OCC. Temp-file + atomic rename, so an interrupted run leaves no
truncated pickle.

```bash
# Look before writing: reports the empty-pair fraction per split.
python -m brepformer.data.repair_descriptors \
  --dirs brepformer/data/defeature_processed --check_only

python -m brepformer.data.repair_descriptors \
  --dirs brepformer/data/defeature_processed brepformer/data/trial6_processed
```

**Not run yet, deliberately, for two reasons.** `fix4_onehot_def` is training off
`defeature_processed/train` right now and reads the pickles lazily every epoch, so rewriting them
mid-run would change the inputs under a live experiment. And the D2 repair is a *second*
independent variable: run it after the layout A/B resolves, then train a third arm, or the two
fixes cannot be attributed separately. Sequence: fix4 finishes → score it → repair → one more arm.

---

## The layout A/B is NEGATIVE: fixing the decoding lost 1.9 points, 2026-09-05

`fix4_onehot_def` finished — early-stopped at epoch 106, best val face F1 **0.8714 at ep81**, having
climbed from 0.3266 at epoch 0. On val it *beat* its categorical twin `t6ctl_init_ft` (0.8637), which
is the number it would have been reported on if the bar had not been fixed in advance. On
`defeature_processed/test`, the 157 real hand-labelled parts every checkpoint in this file is scored
on:

```
rank    macro  weight  minCls      random      hole   chamfer    fillet       cut  run/epoch
1      0.8886  0.8951  0.8197       0.820     0.954     0.874     0.887     0.908  trial1_finetune_ss/best-epoch=32-val
5      0.8699  0.8809  0.8133       0.813     0.950     0.845     0.850     0.891  fix4_onehot_def/best-epoch=106-val
11     0.8660  0.8776  0.8095       0.809     0.949     0.837     0.847     0.886  fix4_onehot_def/bestface-epoch=81-val
```

The pre-registered bar was 0.8886 and the pre-registered reading was written down before the run
finished: *"anything less means the transformer body was more tuned to the broken inputs than the
correct attributes are worth."* **0.8699 < 0.8886, so that is the reading.** The incumbent stays in
production; `defeature_v2.py` does not change.

**Where the 1.9 points went is the informative part, and it is exactly the wrong place.** `random`
(0.813 vs 0.820), `hole` (0.950 vs 0.954) and `cut` (0.891 vs 0.908) are within run-to-run noise.
The whole deficit is **`chamfer` (0.845 vs 0.874) and `fillet` (0.850 vs 0.887)** — the two classes
that edge *convexity* discriminates, and convexity was precisely one of the columns the broken
layout misread (it read `edge_attr[3]`, the convex flag, as edge length). Handing the model the
correct convexity made it worse at the only two classes convexity is for.

**What this does and does not prove.** The arm warm-started from the categorical incumbent and then
flipped the decoding, so it tests *"re-decode and fine-tune broken-layout weights"*, not *"train the
fixed model"*. 106 epochs was not enough to re-adapt embeddings whose weights are matched to the old
column semantics, and the classes that depend most on a re-interpreted column are the ones that
suffered. A clean test needs `onehot` from an init that never saw the broken layout — and the only
such init is from-scratch, which collapses (see "Training: Initialization Dwarfs Every Other Knob").
So the honest state is: **the correct layout has no available init under which it pays for itself**,
and that is a fact about this corpus's size, not about which decoding is true. `attr_layout` stays a
flag with `categorical` as the pin for every reproduction; it does not become the default.

## D2 repair completed on the 27-class side, 2026-09-05

Ran to completion on `mftrcad_processed` + `trial6a_processed` (~1.5 h, CPU only, alongside
training). Mean empty-pair fraction per split:

| corpus / split | files repaired | before | after |
|---|---|---|---|
| `mftrcad_processed/train` | 22,844 | 0.44 | 0.004 |
| `mftrcad_processed/val` | 2,855 | 0.4926 | 0.0040 |
| `mftrcad_processed/test` | 2,856 | 0.4853 | 0.0039 |
| `trial6a_processed/{train,val,test}` | 645 / 81 / 81 | 0.4367 / 0.3536 / 0.4118 | 0.0002 / 0.0003 / 0.0003 |

Note the 27-class corpus was **less** broken than the 5-class one (0.44–0.49 empty vs 0.892 on
`defeature_processed`): `max_dist` came from `face_attr[:, 2:5]`, the cone/sphere/torus one-hot bits,
so a corpus with more curved features accidentally got a non-degenerate range more often. The
residual 0.4% is genuine — coincident faces at zero separation.

**A trap this hit, and it would have been silent.** `trial6_processed` and `trial6a_processed` are
**symlink farms** into `defeature_processed` and `mftrcad_processed`. `os.replace` onto a symlink
path replaces the *symlink* with a regular file, so the first version of `repair_descriptors.py`
would have de-linked the corpus, left the 17 GB of real pickles still broken, and grown the disk by
another 17 GB — while reporting a perfect repair. It now resolves the path before writing and
dedupes by resolved target, so naming both corpora in one `--dirs` repairs each real file once and
prints `N already handled via another path`.

The 27-class arms therefore train on repaired descriptors from their first epoch. That is not a
confound: no family-27 arm had ever been run, so there is no earlier number for the D2 fix to
contaminate. `defeature_processed` is still unrepaired, deliberately — repairing it and re-running
the 5-class arms is the clean attribution of the D2 fix alone.

## Family-27 sweep launched, 2026-09-05

`t6a_init_t5` running (11,734 steps/epoch, ~37 min/epoch, `--early_stop_patience 25`), then
`t6a_init_t5_wf`. Both warm-start from `trial5`, the only 27-class checkpoint that loads — and also
the teacher that produced the pcq27 labels, which is why `t6actl_base` exists as the control. It is
from-scratch, so it needs `--include_scratch`.

One thing to know before re-running the driver: `python -m brepformer.sweeps.trial6_sweep` must
itself run **inside** `new_brepmfr`. `brepformer/models/__init__.py` imports torch at package
import, so the driver dies with `ModuleNotFoundError: No module named 'torch'` before it ever
launches a subprocess — even though every subprocess it would launch is correctly wrapped in
`conda run`.

---

## Per-class auto-accept: the global threshold's 95% is pooled and three of five classes are under it, 2026-09-05

`brepformer/data/accept_queue.py` — the operational half of `label_harness --mode select`. That says
*whether* confidence ranks errors last; this decides per face **accept or review**, writes a
per-model review queue, and attaches a certified precision to every face it does not queue. Nothing
is masked: an unreviewed face still gets a label, with a number on how much to trust it.

Calibrated on the incumbent's posteriors over the 157 real hand-labelled parts / 13,445 faces:

```
predicted class       faces    acc  threshold   accept  observed  certified
cut                    5252    92%      0.904    83.5%     96.0%      95.0%
hole                   3182    93%      0.917    91.1%     96.2%      95.0%
random                 2845    84%      0.944    58.2%     96.6%      95.0%
fillet                 1692    85%      0.976    51.1%     97.2%      95.0%
chamfer                 474    81%       none     0.0%         -          -

per-class:  accept 72.9% of 13445 faces at certified >=95%
global:     accept 80.0% at threshold 0.917
difference: -7.1 points of coverage at the same certified precision
```

**Per-class loses on raw coverage, and that is the wrong way to read it.** The 7.1 points cost the
Bonferroni split (delta 0.05 → 0.01 per class) plus the weaker bound each class gets from its own
smaller sample. What the global cut buys with them is visible only when you ask what its single
number is a guarantee *about*:

```
precision per class INSIDE the global accepted set (threshold 0.917):
predicted class       accepted  observed  certified  vs target
cut                       4316     96.3%      95.5%         OK
hole                      2898     96.2%      95.2%         OK
random                    1853     95.6%      94.2%      UNDER
fillet                    1301     93.6%      91.7%      UNDER
chamfer                    393     88.5%      83.9%      UNDER
```

The global 95% is a **pooled** guarantee, carried by `cut` and `hole`, which are 55% of the faces.
Three of five classes sit under target inside the accepted set, and the worst is `chamfer` at 88.5%
observed / 83.9% certified — i.e. the global cut silently auto-accepts roughly one wrong chamfer in
nine, in the class where a wrong auto-accepted label is most damaging, because `chamfer`/`fillet`
confusion is this model's characteristic failure. So the extra coverage is coverage it was not
entitled to. Per-class thresholds trade 7.1 points for a guarantee that holds *per class*, which is
the promise a review workflow can actually act on.

**The actionable rule that falls out: every face predicted `chamfer` gets looked at.** No threshold
certifies 95% for it at any confidence — 474 faces, 3.5% of the corpus, and the cheapest 3.5% of
review anyone will ever buy. This also corrects the earlier note that `chamfer` was well calibrated
and `random` was the bottleneck: that was measured per **true** class, and a threshold can only key
on the **predicted** class. By predicted class `chamfer` is the worst (81%), not the best.

```bash
# 1. Posteriors for exactly one checkpoint. --dump_probs keys files by model_id, so pointing
# --ckpt_dirs at a run with several checkpoints leaves you the LAST one's dumps, silently.
mkdir -p '/tmp/ck32/best-epoch=32-val' && ln -s \
  "$PWD/results/trial1_finetune_ss/best-epoch=32-val/f1=0.9354.ckpt" \
  '/tmp/ck32/best-epoch=32-val/f1=0.9354.ckpt'
python -m brepformer.scan_checkpoints --ckpt_dirs /tmp/ck32 \
  --data_dir brepformer/data/defeature_processed --split test --label_space 5 \
  --dump_probs results/probs_incumbent --device cpu

# 2. Calibrate (also prints the global comparison and the inside-the-accepted-set breakdown).
python -m brepformer.data.accept_queue calibrate \
  --probs_dir results/probs_incumbent \
  --gt_pkl_dir brepformer/data/defeature_processed/test --label_space 5 \
  --target_precision 0.95 --out results/accept_thresholds_5class.json

# 3. Apply to an unlabelled corpus -> per-model accept/review + queue_index.json.
python -m brepformer.data.accept_queue apply --probs_dir /tmp/probs_newparts \
  --thresholds results/accept_thresholds_5class.json --out_dir /tmp/queue
```

`apply` orders each model's review list least-confident-first, and orders the models themselves by
review volume, so a reviewer with limited time opens the parts that yield the most decisions per
part. On the calibration corpus it reproduces 72.9% exactly, which is the sanity check that the
saved thresholds mean what the calibration said.

**A latent bug fixed to make this possible.** `binom_precision_lower_bound(n_ok, n, delta)` accepted
a `delta` and then **ignored it** — `_delta_eff(n)` read the module-level 0.05 unconditionally. Only
`mode_select` called it, and only at the default, so no recorded number changes; but a per-class
search must spend a smaller delta per class, and cannot if the argument does nothing.


---

## The 27-class arm beats its own init — the first one that does, 2026-09-06

Every arm of the family-5 sweep failed to beat the checkpoint it started from, which is why "score
the init as a baseline" became a standing rule. `trial5` had no `bestface-*` checkpoints at all — it
predates face-F1 monitoring — so its face segmentation had never been measured, and
`t6a_init_t5`'s 0.8592 could not be called a gain against anything. Measuring it changes the
reading of the arm.

Both checkpoints, same 250 held-out `trial6a_processed/val` models, 27-class, one CPU pass:

```
rank    macro  weight  minCls  run/epoch
1      0.8832  0.9320  0.5897  t6a_init_t5 bestface-epoch=30      <- the arm
2      0.8655  0.9300  0.5952  trial5 best-epoch=39               <- the init it started from
```

**+1.8 points of macro F1 over the init, against +0.2 weighted** — and that gap between the two
metrics is the whole story. Per class:

```
  +0.268 circular_blind_slot (n=19):        0.621 -> 0.889
  +0.129 rectangular_through_slot (n=40):   0.731 -> 0.860
  +0.088 rectangular_through_step (n=99):   0.652 -> 0.740
  +0.046 triangular_blind_step (n=125):     0.760 -> 0.806
  ...
  -0.024 rectangular_pocket (n=365):        0.889 -> 0.865
  -0.027 rectangular_passage (n=423):       0.891 -> 0.864
  -0.118 circular_through_slot (n=12):      0.880 -> 0.762
```

The arm gains on the **rare** classes and gives back a little on the **common** ones. That is the
signature of the pcq27 data being enriched in features mftrcad under-represents, and it is why
macro moves nine times as far as weighted. It is also the only shape of result for which the
`t6actl_base` control matters — without it, "the extra data helped" and "27 more epochs helped"
are the same number.

**Two caveats, and the second one is not small.** `circular_through_slot` has 12 faces, so its
-0.118 is one or two faces and means nothing. And **`t6a_init_t5`'s checkpoint was selected on this
very split** — it is the early-stop monitor — while `trial5` never saw it for selection. So +1.8 is
an upper bound on the real gain, and the comparison has to be repeated on `trial6a_processed/test`
once the arm stops. Do not quote the val number as the result.

```bash
# --dump_probs and --ckpt_dirs both key on directories, so isolate the exact checkpoints first.
# Without this, a run dir with several .ckpt files scores all of them and the dumps overwrite.
rm -rf /tmp/init_ab && mkdir -p /tmp/init_ab/trial5-ep39 /tmp/init_ab/t6a-best
ln -s "$PWD/results/trial5/best-epoch=39-val/f1=0.8832.ckpt" \
      /tmp/init_ab/trial5-ep39/
ln -s "$PWD/results/t6a_init_t5/bestface-epoch=30-val/face_f1=0.8592.ckpt" \
      /tmp/init_ab/t6a-best/

# --device cpu deliberately: the GPU is training the next arm, and 250 models is a few minutes.
python -m brepformer.scan_checkpoints \
  --ckpt_dirs /tmp/init_ab/trial5-ep39 /tmp/init_ab/t6a-best \
  --data_dir brepformer/data/trial6a_processed --split val --label_space 27 \
  --limit 250 --device cpu --out /tmp/init_ab.json
```

Note `--label_space 27`: passing `5` here would collapse the posterior into defeature classes and
compare two 27-class models on the wrong axis, without erroring.


---

# Runnable scripts — the current queue, in order (2026-09-04)

Everything above this line is a record of what was run and what it showed. This last section is
the opposite: **only the commands worth running now**, in the order they should run, with the
optimizations from 2026-09-04 already applied. If a command elsewhere in this file disagrees with
one here, the one here is newer.

Environment for every command below: `conda activate new_brepmfr` (py3.11). The py3.9 `brep_mfr`
env double-frees in `infer.py`. `conda run` buffers stdout, so for anything long either activate
the env first or write a script file and run it.

## Why the sweep commands changed shape

The eleven family-5 arms finished (~72 GPU-hours) and **nothing beat the checkpoint the warm
starts were seeded from**. Three consequences are now enforced in
`brepformer/sweeps/trial6_sweep.py` rather than left as advice, which is why the invocations below
are shorter than the ones in "Further Improvements (Trials 6 / 6A)":

1. **From-scratch arms are opt-in** (`select()`). All six from-scratch family-5 arms collapsed to
   a constant face prediction — macro F1 0.089–0.198 against a 1/5 = 0.2 floor. Five of the seven
   family-27 arms are also from-scratch, on ~12x the samples per epoch. So `--run --family 27` now
   costs **2 arms, not 7**; `--include_scratch` brings the other five back.
2. **`--attr_layout` is always pinned** into the emitted command, never left to the model default.
   That default flipped on 2026-09-04 and a checkpoint scores ~50 points worse decoded under the
   wrong layout (82.95% categorical vs 32.75% onehot on the same 12 models), so an unpinned re-run
   would silently not reproduce the recorded number.
3. **Re-sweeps get suffixed names** (`suffix()`): `--attr_layout onehot --rotate_aug so3` writes to
   `results/t6a_init_t5__onehot_rotso3`, keyed the same way in `sweep_logs/status.json`. Without
   this, a re-swept arm reports "already complete, skipped" against the *original* arm's status
   entry and overwrites the results directory the tables cite.

`--emit` still prints all seven arms — this file documents what the sweep *is*, not only what is
worth re-running.

## 1. The 27-class arms (only two are worth running)

```bash
python -m brepformer.sweeps.trial6_sweep --run --family 27 --dry_run   # look first
python -m brepformer.sweeps.trial6_sweep --run --family 27
```

That is `t6a_init_t5` and `t6a_init_t5_wf`, both warm-started from `trial5`
(`results/trial5/best-epoch=39-val/f1=0.8832.ckpt`), the only 27-class checkpoint that loads. Note
that trial5 is also the teacher that produced the pcq27 labels, which is exactly why
`t6actl_base` exists as a control — but it is from-scratch, so it needs the flag:

```bash
python -m brepformer.sweeps.trial6_sweep --run --family 27 --include_scratch   # all 7, tens of GPU-hours
python -m brepformer.sweeps.trial6_sweep --run --family 27 --only t6actl_base --include_scratch
```

## 2. Re-sweep the warm starts on the *fixed* model — DEPRIORITIZED 2026-09-05

The layout A/B resolved **negative**: `fix4_onehot_def` scored 0.8699 on test against the
incumbent's 0.8886, losing entirely on `chamfer` and `fillet` (see "The layout A/B is NEGATIVE"
above). So `--attr_layout onehot` is no longer expected to pay, and a full re-sweep under it is not
worth the GPU-hours until there is an init that never saw the broken layout. Kept here because the
*rotation* arms are still open questions and are orthogonal — but run them under `categorical`, the
pin that reproduces, unless the specific question is the layout again.

Old weights were trained under the broken decoding, so an `onehot` re-sweep must start from scratch
or from a onehot-trained checkpoint (`results/fix4_onehot_def/...` is now one), not from
`CKPT_FINETUNE`.

```bash
# STILL OPEN: SO(3) rotation augmentation on the train split, under the pin that reproduces.
python -m brepformer.sweeps.trial6_sweep --run --family 5 --rotate_aug so3

# The cheap control for the rotation question: 24 axis-aligned 90-degree frames instead of
# uniform SO(3). If octahedral recovers `hole` and so3 does not, the loss is interpolation of
# the grid, not the frame.
python -m brepformer.sweeps.trial6_sweep --run --family 5 --rotate_aug octahedral

# The layout re-sweep, if it is ever worth it again -> results/t6_init_*__onehot
python -m brepformer.sweeps.trial6_sweep --run --family 5 --attr_layout onehot
```

`--rotate_aug` is train-split only by construction — `train_preprocessed.py` deliberately does not
pass it to val, because rotating val changes what `val/face_f1` measures and the early-stop
monitor has to stay comparable to every run recorded above.

## 3. Repair the D2 channel — 27-class side DONE, 5-class side still open

`d2_distance` was all-zero for 99.2% of face pairs on `defeature_processed` because `max_dist` came
out at 1e-6. The point clouds come from `face_grid`, which is on disk and was always correct, so
this repairs in place — no STEP, no OCC, no re-preprocessing.

`mftrcad_processed` and `trial6a_processed` were repaired on 2026-09-05 (0.44–0.49 → 0.004 empty
pairs; table above), so the family-27 arms already train on fixed descriptors. **The 5-class side is
deliberately still broken** — repairing it and re-running one 5-class arm is the clean attribution
of the D2 fix on its own, against a 0.8886 baseline that was measured on the unrepaired corpus.

```bash
python -m brepformer.data.repair_descriptors \
  --dirs brepformer/data/defeature_processed --check_only          # reports empty-pair fraction
python -m brepformer.data.repair_descriptors \
  --dirs brepformer/data/defeature_processed brepformer/data/trial6_processed
```

Do **not** run this while a job reads those pickles — the datasets load lazily every epoch, so a
rewrite mid-run changes the inputs under a live experiment. Naming both corpora in one `--dirs` is
correct and cheap: the second is a symlink farm into the first, and the script dedupes by resolved
target rather than repairing each file twice or replacing the symlinks with copies.

## 4. Score and compare checkpoints

The bar is the incumbent `trial1_finetune_ss` ep32 at **face macro F1 0.8886** on
`defeature_processed/test` (157 real hand-labelled parts, 13,445 faces). The `f1=` in a checkpoint
filename is the *model-level* metric and does not rank face segmentation.

```bash
# --ckpt_dirs takes RUN directories, not checkpoint paths: every surviving checkpoint under
# each is scored, and the split is loaded and collated once for all of them.
python -m brepformer.scan_checkpoints \
  --ckpt_dirs results/fix4_onehot_def results/trial1_finetune_ss \
  --data_dir brepformer/data/defeature_processed --split test --label_space 5

# Per-face posteriors for the risk-coverage / auto-accept analysis below. Pass exactly ONE
# run dir with one checkpoint in it, or the dumps overwrite each other by model_id.
python -m brepformer.scan_checkpoints \
  --ckpt_dirs results/trial1_finetune_ss \
  --data_dir brepformer/data/defeature_processed --split test --label_space 5 \
  --dump_probs /tmp/probs_incumbent
```

There is no `--attr_layout` here on purpose: `infer.load_model` reads the layout out of the
checkpoint's own saved config (`vars(saved).get("attr_layout")`), so a pre-fix checkpoint decodes
as `categorical` and a post-fix one as `onehot` without being told which it is. `getattr` cannot
do this — unpickling a dataclass rebuilds it against the *current* class definition, so a
newly-added field silently returns its default for old pickles.

Watch `chamfer`/`fillet` (edge convexity discriminates them, and convexity was one of the columns
the broken layout misread) and `hole` (0.971 → 0.120 in the worst rotated frame, for the same
reason).

## 5. Auto-accept budget — the number the review workflow runs on

80% of faces auto-accept at a **certified 95% precision** on real parts, against 11.5% measured on
mftrcad. `random` is the miscalibrated class (80% acc, 0.083 mean margin — the smallest), not
`chamfer` (94%, 0.157). So seed the review queue with faces *predicted* `random`, and set
thresholds per class.

No separate script: `label_harness --mode select` already is the risk-coverage analysis, and
`--gt_pkl_dir` points it at a preprocessed split's exact `face_labels` instead of another model's
pseudo-labels. It prints accuracy, AUGRC and AURC each against its own uninformative baseline, the
coverage/precision table, the observed *and* Clopper-Pearson-certified budget at four target
precisions, and a per-class breakdown.

```bash
python -m brepformer.data.label_harness --mode select \
  --predictions results/probs_incumbent \
  --gt_pkl_dir brepformer/data/defeature_processed/test --label_space 5 --limit 0
```

**`--limit 0` is not optional.** `--limit` defaults to **60 models**, so without it this reports a
60-model subsample (89.3% accuracy, AUGRC 0.0193) that looks like the full-split number and is not
labelled as a subsample anywhere in the output. With `--limit 0`: 157 models / 13,445 faces,
accuracy 89.6%, AUGRC 0.0208 against an uninformative 0.0521. The same applies to
`--mode harm`/`--mode benefit`.

The per-class table is keyed on the **true** label. A threshold can only key on the *predicted*
label, which is what `accept_queue.py` does — and the two disagree about which class is worst
(`chamfer` is the best by true class, the worst by predicted). For the accept/review decision use:

```bash
python -m brepformer.data.accept_queue calibrate --probs_dir results/probs_incumbent \
  --gt_pkl_dir brepformer/data/defeature_processed/test --label_space 5 \
  --target_precision 0.95 --out results/accept_thresholds_5class.json

python -m brepformer.data.accept_queue apply --probs_dir /tmp/probs_newparts \
  --thresholds results/accept_thresholds_5class.json --out_dir /tmp/queue
```

Certified per class: accept 72.9% of faces, review 27.1%, and **every predicted `chamfer` face is
reviewed** because none certifies at any confidence. The global cut accepts 80.0% but three of five
classes are under target inside that set — see "Per-class auto-accept" above for why the lower
number is the right one.

## 6. Measure any labeller before trusting it

`--gt_pkl_dir` reads exact `face_labels` straight out of a preprocessed split, so a heuristic can
be scored against ground truth instead of against another model's pseudo-labels.

```bash
python -m brepformer.data.label_harness --mode harm --geo_mode advise \
  --gt_pkl_dir brepformer/data/defeature_processed/test --label_space 5 --limit 0
```

`--limit 0` for the whole split: the default is **60 models**, and a subsample is not labelled as
one in the output (`--mode select` reports 89.3% / AUGRC 0.0193 on 60 models vs 89.6% / 0.0208 on
all 157).

Note the module path: the file is `brepformer/data/label_harness.py`, so
`python -m brepformer.label_harness` exits 2.

## 7. Rotation sensitivity of a trained checkpoint

```bash
python -m brepformer.so3_probe \
  --ckpts 'results/trial1_finetune_ss/best-epoch=32-val/f1=0.9354.ckpt' \
  --data_dir brepformer/data/defeature_processed --split test --label_space 5 \
  --n_rotations 8 --n_octahedral 4
```

`--n_rotations` draws uniform SO(3) frames (Shoemake quaternion); `--n_octahedral` draws from the
24 axis-aligned 90-degree rotations, which is the control that separates "the model needs the
canonical frame" from "grid interpolation lost the geometry". Renormalisation is on by default —
`normalize_geometry` scales by half the max *axis-aligned* bbox extent, so it is itself
frame-dependent; `--no_renormalize` measures the model without that correction.

`rotate_sample` no longer deepcopies: it copies only `face_grid`, `edge_grid` and `face_attr` and
shares the rest. A deepcopy duplicated `d2_distance` — (F+1, F+1, 64) float32, 256 MB for a
1000-face part — on every call, which is what made this unusable as a training-time augmentation.
Nothing downstream mutates the shared arrays; do not write into the returned sample in place.

## 8. TensorBoard

Already serving all runs on port 6006 (`--logdir results`); starting a
second one exits 255. Progress bars are `\r`-delimited, so `grep` cannot recover a metric history
from a training log and top-k pruning deletes all but the last three checkpoint files — read the
event files instead:

```bash
python - <<'PY'
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
ea = EventAccumulator(glob.glob("results/fix4_onehot_def/version_0/events*")[0],
                      size_guidance={'scalars': 0}); ea.Reload()
for t in ('val/face_f1', 'train/face_f1'):
    print(t, [round(e.value, 4) for e in ea.Scalars(t)])
PY
```

## The init comparison survives on `test`, and it is bigger there: +4.7 macro, 2026-09-08

`t6a_init_t5` finished: 100 epochs, 62 h wall, early-stopped, best 27-class val face F1 **0.8676 at
ep74**. It did *not* plateau at ep21 as it appeared to on 2026-09-06 — it climbed for another 44
epochs. Do not read a plateau off two consecutive val checkpoints on this corpus.

Item 1 of the outstanding queue, run on the full `test` split with no `--limit` (2923 labelled
models), CPU so it could not OOM the trainer that owns the GPU:

```bash
# One checkpoint per directory: --dump_probs and the ranking are both keyed per directory, and a
# run dir with several checkpoints scores all of them.
rm -rf /tmp/init_ab_test && mkdir -p /tmp/init_ab_test/arm-ep74 /tmp/init_ab_test/init-trial5
ln -s "$PWD/results/t6a_init_t5/bestface-epoch=74-val/face_f1=0.8676.ckpt" /tmp/init_ab_test/arm-ep74/
ln -s "$PWD/results/trial5/best-epoch=39-val/f1=0.8832.ckpt"               /tmp/init_ab_test/init-trial5/

conda run -n new_brepmfr --no-capture-output python -m brepformer.scan_checkpoints \
    --ckpt_dirs /tmp/init_ab_test/arm-ep74 /tmp/init_ab_test/init-trial5 \
    --data_dir brepformer/data/trial6a_processed --split test \
    --label_space 27 --device cpu --out /tmp/init_ab_test/scores.json
```

`--label_space 27` is required. At `--label_space 5` this silently marginalizes two 27-class heads
into the defeature space and compares them on an axis neither was trained for.

|                | init `trial5` | arm ep74 | delta |
|----------------|--------------|----------|-------|
| macro F1       | 0.8062 | **0.8529** | **+0.0467** |
| weighted F1    | 0.8721 | **0.9068** | +0.0346 |
| min-class F1   | 0.5234 | **0.6528** | +0.1294 |

23 of 27 classes improve. The four regressions and the three largest gains:

| class | n | init | arm | delta |
|---|---|---|---|---|
| `rectangular_passage` | 4908 | 0.892 | 0.760 | **-0.132** |
| `6sides_passage` | 5289 | 0.896 | 0.815 | **-0.081** |
| `circular_blind_slot` | 485 | 0.864 | 0.854 | -0.009 |
| `rectangular_blind_slot` | 580 | 0.655 | 0.653 | -0.002 |
| `Oring` | 5532 | 0.654 | **0.956** | +0.302 |
| `through_hole` | 1216 | 0.523 | **0.809** | +0.286 |
| `rectangular_blind_step` | 2242 | 0.613 | **0.833** | +0.220 |

> **RETRACTED 2026-09-17 by the corpus control. The +4.7 is a test-split contamination artifact.**
> Every number in the two tables above is scored on `trial6a_processed/test`, which is 2856 MFTRCAD
> ground-truth models **plus 81 pcq27 models carrying the arm's own teacher's pseudo-labels**. On the
> clean MFTRCAD-only split the +4.7 becomes +0.7, and the control gains the same +0.7 without ever
> seeing the pcq27 data. See "The corpus control retires the +4.7" below. The class-level readings
> that follow — the two `*_passage` regressions, the `Oring`/`through_hole` gains, the rare-vs-common
> split — are all computed on the contaminated split and should be re-read there, not here.

The val measurement (+1.8 macro, +0.2 weighted) **understated** this, which is the opposite of the
expected direction for a number measured on the selection split. So the val figure was not selection
bias; it was a 250-model subsample. Weighted F1 moving +3.5 rather than +0.2 also kills the earlier
"gains are confined to rare classes" reading: `Oring` (n=5532) and `stock` (n=25834) both improve,
and the two real regressions are *common* classes, not rare ones.

`rectangular_passage` and `6sides_passage` cost precision to `6sides_passage` recall (0.918 recall at
0.734 precision): the arm over-predicts `6sides_passage` and eats `rectangular_passage`. That is a
confusion between two classes distinguished only by side count — a candidate for the next arm, not a
reason to prefer the init, which loses 4.7 macro overall.

### The common classes gain too, on models neither checkpoint has seen

The earlier "gains are confined to rare classes" reading came from the 250-model val subsample and is
wrong. Splitting the 27 classes at the median support (n = 1493 faces):

| group | classes | mean delta | up / down |
|---|---|---|---|
| common, n >= 1493 | 14 (72,380 faces) | **+0.0350** | 12 / 2 |
| rare, n < 1493 | 13 (10,870 faces) | **+0.0593** | 11 / 2 |

Rare classes gain more per class, but the common half gains too, and it carries 87% of the faces —
which is why weighted F1 moves +3.5 and not +0.2. The specific classes worth naming:

| class | n | init | arm | delta |
|---|---|---|---|---|
| `stock` | 25834 | 0.963 | **0.983** | +0.020 |
| `Oring` | 5532 | 0.654 | **0.956** | +0.302 |
| `circular_end_pocket` | 4403 | 0.879 | **0.928** | +0.049 |
| `rectangular_passage_2` | 3566 | 0.912 | **0.945** | +0.033 |
| `chamfer_2` | 2814 | 0.970 | **0.983** | +0.013 |
| `chamfer` | 1167 | 0.828 | **0.912** | +0.084 |
| `round` (the 27-class fillet) | 891 | 0.909 | **0.952** | +0.043 |

`stock` is the single biggest class at 25,834 faces and it still improves; `chamfer` gains +8.4 and
`round` +4.3, which matters because chamfer/fillet confusion is this model's characteristic failure
and the class where a wrong auto-accepted label is most damaging (see the accept-queue section).

All of these are measured on `trial6a_processed/test`: 2923 models used for neither training nor
checkpoint selection, so the gains are on unseen models, not fitted ones. The honest limit is that
`test` is the same distribution as `train` (held-out mftrcad + pcq27 models, not a different domain),
so this establishes generalization across models, not across corpora. Cross-corpus generalization
would need scoring against `defeature_processed` or the Fusion 360 set.

Still outstanding: this compares arm-vs-init, not data-vs-epochs. `t6actl_base` (queue item 2) is
what attributes the gain to the pcq27 corpus rather than to 62 h of additional training.


## A transient CUDA fault idled the GPU for four days, 2026-09-13

`t6a_init_t5_wf` died at **epoch 11** with `RuntimeError: CUDA error: unknown error`, raised from
inside `torchmetrics`' `_unique_impl` during `training_step`. Ten epochs had already completed on
identical batch shapes and the GPU tests clean now, so this is the generic WSL2 symptom of the
device dropping out — not a fault in the model or the data.

The crash is not the expensive part. It was the **last** arm in the driver's `subprocess.call` loop,
so the loop exited, and the GPU sat at 0% / 71 MiB for four days. Two changes so this cannot repeat:

**`--retries` (default 2), resuming from `last.ckpt`.** Retries append to the log instead of
truncating it, so the original traceback survives. Each retry passes `--resume_from`, because a plain
relaunch would restart from `--init_weights` at epoch 0 and quietly report a different experiment
under the same arm name. An arm that dies *before* writing `last.ckpt` is deliberately **not**
retried: with no checkpoint the retry is a fresh start, and a first-epoch crash is far more likely to
be a config error than a dropped device.

**A crashed arm is picked up rather than restarted.** If `status.json` records a non-zero rc for the
arm and `last.ckpt` exists, the first attempt already carries `--resume_from`. Gated on the recorded
rc, so a *completed* arm is never silently continued.

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader   # 0% + no process = idle
conda run -n new_brepmfr python -c "import torch;x=torch.randn(2000,2000,device='cuda');print((x@x).sum().item())"
tr '\r' '\n' < results/sweep_logs/<arm>.log | grep -nE "RuntimeError|Traceback" | tail
```

Check `nvidia-smi` whenever a "progress?" answer would otherwise be "still training" — an idle GPU
and a running arm look identical in `results/`, which is what hid this for four days.

## The 27-class control was the wrong control, 2026-09-13

`t6actl_base` was queued as the arm that attributes `t6a_init_t5`'s +4.7 macro to the pcq27 corpus.
It cannot: it is `scratch=True`, so it removes the pcq27 data **and** discards `--init_weights`. If it
collapses — and every from-scratch arm in this program has — the result is uninterpretable, because a
collapse is explained equally well by "the pcq27 data was necessary" and by "no 27-class model trains
from scratch here". It confounds exactly the variable it was built to isolate.

Added `t6actl_init_t5`: `--init_weights` held at `CKPT_TRIAL5`, corpus switched from
`trial6a_processed` (mftrcad + pcq27) to `mftrcad_processed` (mftrcad only), every other flag
byte-identical to `t6a_init_t5`. One variable moves, so its gap to `t6a_init_t5` on
`trial6a_processed/test` **is** the pcq27 contribution.

```bash
conda run -n new_brepmfr --no-capture-output python -m brepformer.sweeps.trial6_sweep \
    --run --family 27 --only t6actl_init_t5 t6a_init_t5_wf
```

Arms execute in `SWEEP` order, not the order given on the command line, so `t6a_init_t5_wf` (resumed
from its ep11 crash) runs first and `t6actl_init_t5` second. `t6actl_base` stays in `SWEEP` as a
floor, still behind `--include_scratch`, but it is no longer the control anything depends on.

## Cross-corpus: the 27-class gain ANTI-transfers to real parts, 2026-09-14

Queued ahead of the labelling work at the user's request, and it is the most consequential number in
this file. Everything the 27-class program has measured is same-distribution — held-out mftrcad +
pcq27. The deployed use case is real designer parts, and the only real hand-labelled evidence is the
157 parts in `defeature_processed/test`. Scoring the 27-class winner there, marginalized to 5 classes,
asks whether +4.7 macro on synthetic data means anything for the thing that actually ships.

It does not. It is worse than nothing:

| checkpoint | head | macro | weighted | min-class | random | hole | chamfer | fillet | cut |
|---|---|---|---|---|---|---|---|---|---|
| `trial1_finetune_ss` ep32 (deployed) | 5 | **0.8886** | 0.8951 | 0.8197 | 0.820 | 0.954 | 0.874 | 0.887 | 0.908 |
| `trial5` ep39 (the 27-class init) | 27 | 0.5977 | 0.5980 | 0.3686 | 0.369 | 0.729 | 0.704 | 0.520 | 0.666 |
| `t6a_init_t5` ep74 (the +4.7 arm) | 27 | **0.4287** | 0.4164 | 0.1701 | 0.170 | 0.276 | 0.614 | 0.487 | 0.596 |

The arm is **17 points worse than its own init** on real parts while being 4.7 better on synthetic
ones. The training that produced the synthetic gain actively destroyed real-part transfer, worst on
`hole` (0.729 → 0.276) and `random` (0.369 → 0.170) — i.e. on `stock`, the class that improved most on
synthetic data (0.963 → 0.983). The model learned the mftrcad/pcq27 notion of a stock face harder, and
that notion does not survive contact with a real part.

```bash
# --collapse marginalize is required. The default argmax_remap is bug 1's pattern, kept as the
# default only so the 27-class-at-label_space-5 numbers already in this file stay comparable; it
# understates a 27-class head and must not be used for a new claim.
rm -rf /tmp/xcorpus
for d in arm27-ep74 init27-trial5 incumbent5; do mkdir -p /tmp/xcorpus/$d; done
ln -s "$PWD/results/t6a_init_t5/bestface-epoch=74-val/face_f1=0.8676.ckpt" /tmp/xcorpus/arm27-ep74/
ln -s "$PWD/results/trial5/best-epoch=39-val/f1=0.8832.ckpt"              /tmp/xcorpus/init27-trial5/
ln -s "$PWD/results/trial1_finetune_ss/best-epoch=32-val/f1=0.9354.ckpt"   /tmp/xcorpus/incumbent5/

conda run -n new_brepmfr --no-capture-output python -m brepformer.scan_checkpoints \
    --ckpt_dirs /tmp/xcorpus/arm27-ep74 /tmp/xcorpus/init27-trial5 /tmp/xcorpus/incumbent5 \
    --data_dir brepformer/data/defeature_processed --split test \
    --label_space 5 --collapse marginalize --device cpu --out /tmp/xcorpus/scores.json
```

**What this does and does not license.** The 27-class models were never trained on defeature labels
and the 27→5 mapping is lossy, so the *level* (0.43 vs 0.89) is partly a label-space penalty and is
not a fair head-to-head. The *direction* is not: `trial5` and `t6a_init_t5` are the same architecture
scored the same way through the same mapping, and more synthetic training moved that comparison 17
points the wrong way. Any future 27-class arm must report this number alongside its synthetic macro,
or a synthetic gain will keep being read as progress. Also worth stating plainly: the +4.7 result is
still correct on its own terms and is still the first arm to beat its init — it just does not
generalize off-corpus.

## Bug 7: the entire multi-hop edge pathway is dead, in three independent places, 2026-09-14

Auditing bug 5 for its own arm turned up that the unused `edge_conv_layer` is not one dead module but
the visible end of a whole missing mechanism — Graphormer's multi-hop *edge* encoding, which is the
paper's central device for getting edge features into attention:

1. **`edge_path` is never produced.** `compute_shortest_paths` in `preprocessing.py` ends
   `return spatial_pos, None`, docstringed "(optional, not implemented)". No pickle in any corpus
   carries an `edge_path` key — verified on `defeature_processed/test`.
2. **So section 5 of `GraphAttnBias.forward` never executes.** It is guarded by
   `if edge_path is not None`, which is never true, so `edge_dis_encoder` is a second dead module and
   `config.multi_hop_max_dist = 16` is a knob wired to nothing.
3. **And if `edge_path` were supplied, section 5 would still carry no edge information.** It embeds
   `torch.full_like(hop_edges, hop + 1)` — the hop *number* — and never gathers the edge features at
   the indices `hop_edges` actually holds. That makes it a learned per-hop constant applied wherever a
   path of that length exists, which is a function of `spatial_pos` alone and therefore redundant with
   `spatial_pos_encoder` two sections above it.
4. **`edge_conv_layer` (bug 5)** is the aggregation step that would consume the gathered features.

So today edge features reach attention through exactly one route: the direct-neighbour `scatter_add`
in section 4. Two faces that do not share an edge exchange only shortest-path distance, D2 and angle
descriptors — no edge geometry at all. For fillet/chamfer, where the discriminator is the convexity of
edges along a short path between faces, that is the pathway most likely to matter, and BRepNet's
ablation prices convexity removal at -4.6 IoU as its single largest feature loss.

**One correction to the bug-5 entry.** It was recorded as needing an architecture change that would
break `load_from_checkpoint`. It does not: `edge_conv_layer` and `edge_dis_encoder` are already
constructed, already saved in every checkpoint, and already carry (untrained) weights. Wiring them in
adds **no** parameters, so every existing checkpoint stays loadable and a warm start works normally —
the two modules simply begin from the random init that has been sitting in the checkpoints all along.
The expensive part is preprocessing, not the model.

**Cost, which is the real constraint.** `edge_path` is `(N+1, N+1, max_dist)`. At `max_faces 1000` and
`multi_hop_max_dist 16` that is 1001^2 x 16 x int64 = **128 MB per sample**, on top of the 256 MB the
D2 tensor already costs, and the collator pads to the batch maximum. Any implementation must cap
`multi_hop_max_dist` low (4 is the useful range for fillet/chamfer adjacency) and store int16/int32,
or the arm will OOM before it trains. Path-tracking BFS also makes preprocessing meaningfully slower
than the current distance-only BFS, over 28.5k models.

### The arms this needs, and the control that makes them readable

`fix7_edgepath` cannot be compared against the recorded 0.8886, because it needs a re-preprocessed
corpus and re-preprocessing also picks up the D2 repair. Two changes, one number, again. So the D2
arm (queue item 3) must land **first** and become the baseline:

```bash
# 0. PREREQUISITE: repair D2 on the 5-class side and train the D2 arm. Its score is the baseline.
#    (queue item 3 — do not skip; without it fix7 measures D2 + edge_path together)

# 1. Re-preprocess with edge paths, capped and narrow.
python brepformer/preprocess.py --data_dir <steps> --output_dir brepformer/data/defeature_ep4 \
    --emit_edge_path --multi_hop_max_dist 4      # flags do not exist yet; see below

# 2. TREATMENT: edge_path supplied and section 5 gathering real edge features.
python brepformer/train_preprocessed.py --data_dir brepformer/data/defeature_ep4 \
    --num_classes 5 --num_face_classes 5 --max_faces 1000 \
    --batch_size 2 --accumulate_grad_batches 16 --learning_rate 0.001 \
    --max_epochs 200 --seed 42 --output_dir results --face_segmentation \
    --lr_monitor val/face_f1 --early_stop_monitor val/face_f1 --early_stop_patience 25 \
    --attr_layout categorical --multi_hop_max_dist 4 \
    --init_weights "results/trial1_finetune_ss/best-epoch=32-val/f1=0.9354.ckpt" \
    --exp_name fix7_edgepath

# 3. CONTROL: the SAME re-preprocessed corpus, edge_path withheld. One variable.
python brepformer/train_preprocessed.py --data_dir brepformer/data/defeature_ep4 \
    ... identical ... --no_edge_path --exp_name fix7ctl_noedgepath
```

The control matters for the reason `t6actl_base` did not work: the treatment corpus is a *new*
preprocessing run, so `fix7_edgepath` vs the D2 arm differs in the corpus as well as the mechanism.
`fix7ctl_noedgepath` reads the same pickles and ignores the `edge_path` key, so the treatment-control
gap is the mechanism alone. Do not substitute the D2 arm for this control.

Three flags have to be built before any of this runs: `preprocess.py --emit_edge_path`, a
path-tracking variant of `compute_shortest_paths`, and `train_preprocessed.py --no_edge_path`.
Section 5 of `GraphAttnBias.forward` must be rewritten to gather `edge_feat` at `hop_edges` and pass
it through `edge_conv_layer`; leaving it as a hop-number embedding would produce an arm that measures
nothing, since that term is already redundant with `spatial_pos_encoder`.

## Bug 8: the UV validity mask is a dead channel on real parts and a live one in pretraining, 2026-09-15

Found by auditing `face_grid`, not by reading model code. `face_grid` is 7 channels — xyz, unit
normal, and **mask**. Channel 6 is a validity gate: a face is a *trimmed* region of a surface, so a
grid over its UV parameter box samples points that lie outside the face (inside a through-hole, off
the corner of a triangular pocket floor, past the taper of a narrowing fillet), and the mask says
which samples are real.

`step_to_graph.compute_face_uv_grid` never computed it. It gridded the UV box and wrote `mask = 1`
wherever the surface evaluated. So on every STEP-derived corpus here the channel is **identically 1**
— a dead input — and every off-face sample is handed the underlying surface's xyz and normal and
declared valid. MFTRCAD's published graphs are not like that:

```bash
conda run -n new_brepmfr --no-capture-output python -m brepformer.data.uv_mask_audit disk \
    --pkl_dirs brepformer/data/defeature_processed/test brepformer/data/mftrcad_processed/test \
    --graph_dirs brepformer/data/mftrcad/graphs --limit 150 \
    --out results/uv_mask_audit_disk.json
```

```
pkl defeature_processed/test     11544 faces  masked mean 0.000 median 0.000 p90 0.000  any   0.0%  >half   0.0%
                                 ^^ channel 6 is the constant 1 here: DEAD input channel
pkl mftrcad_processed/test        3798 faces  masked mean 0.445 median 0.360 p90 0.640  any 100.0%  >half  21.6%
json mftrcad/graphs              3970 faces  masked mean 0.445 median 0.360 p90 0.650  any 100.0%  >half  21.6%
```

The SurfaceEncoder learns channel 6 as a validity gate on the pretraining corpus and is handed a
constant on the real parts it is deployed on. That is an input-channel distribution shift sitting
exactly on the transfer this program depends on.

### The convention is fitted, not guessed

The obvious fix — classify each UV sample against the trim loops — gets the *direction* right and the
*level* wrong. Run against MFTRCAD's own STEP files and compared to their published masks on the same
faces, an unpadded trim test gives 0.196 against their 0.452 (correlation 0.896: same quantity, wrong
level). Sweeping a UV padding applied before sampling settles it:

```bash
conda run -n new_brepmfr --no-capture-output python -m brepformer.data.uv_mask_audit fit \
    --steps_dir brepformer/data/mftrcad/steps --graphs_dir brepformer/data/mftrcad/graphs \
    --limit 15 --pad_sweep 0 0.1 0.15 0.2 0.3 --out results/uv_pad_fit.json
```

```
matched 15 models, 410 faces; MFTRCAD published masked mean 0.452

   pad    masked  mean|diff|    corr
  0.00     0.196       0.256   0.896
  0.10     0.463       0.038   0.913   <-- MFTRCAD
  0.15     0.678       0.227   0.843
  0.20     0.680       0.228   0.839
  0.30     0.687       0.238   0.790
```

MFTRCAD pads the face UV box by **10% of its extent on each side**, then masks `TopAbs_OUT`. 0.10
reproduces their masks at mean absolute per-face error 0.038; every other value is off by an order of
magnitude more. Scored per face, not per corpus — a corpus mean is easy to hit by accident. The other
hypothesis, gridding the *underlying surface's* full parameter range (a whole cylinder's 0..2π), is
wrong: it gives 0.866.

That means the mismatch has two halves, not one. Our grids are taken over a UV domain **10% narrower
on each side** than the pretraining corpus's, so even the xyz and normal channels are sampled at
different parameter locations — and then everything is called valid.

### What it costs the real corpus

```bash
conda run -n new_brepmfr --no-capture-output python -m brepformer.data.uv_mask_audit steps \
    --steps_dir brepformer/data/defeature/steps --limit 12 --out results/uv_mask_audit_steps.json
```

622 faces over 12 real parts: **28.5% of faces are fed at least one off-face sample as valid, and
10.0% of faces are more than half off-face** (mean 10.6% of grid points, median 0 — concentrated, not
diffuse). The faces it concentrates on are the trimmed ones: hole walls, pocket floors, tapered
fillets. `hole` is the class that collapsed hardest cross-corpus (0.729 → 0.276).

### One claim that did not survive being measured

`normalize_geometry` derives the bbox centre and scale from `mask > 0.5`, so this looked like it
should also shift every normalized coordinate and the area scaling. It does not. Measured on the same
12 parts, the scale is **identical on 12/12** and the centre moves by at most 0.6% of the half-extent:
the part bbox is set by the extreme faces, whose UV boxes are tight, and the overshoot is interior.
Claim withdrawn — bug 8 is an input-channel and sampling-domain bug, not a normalization bug.

### The fix, and the arms it needs

`--mftrcad_uv` on `steps_to_graphs.py` and `prepare_defeature.py` (both halves under one flag: padding
without masking would be strictly worse than doing nothing, since it samples further off-face and
still calls it valid). Off by default, because flipping it changes every pickle and would make new
numbers incomparable with the tables above.

```bash
# 1. Re-derive the real-part graphs under MFTRCAD's convention, into a NEW tree.
conda run -n new_brepmfr --no-capture-output python -m brepformer.data.steps_to_graphs \
    --steps_dir brepformer/data/defeature/steps \
    --out_dir brepformer/data/defeature_uv/graphs --mftrcad_uv --tolerate_crashes

# 2. Preprocess it with the same labels as the incumbent corpus.
cp -r brepformer/data/defeature/labels brepformer/data/defeature_uv/labels
conda run -n new_brepmfr --no-capture-output python -m brepformer.preprocess \
    --data_dir brepformer/data/defeature_uv \
    --output_dir brepformer/data/defeature_uv_processed --num_classes 5

# 3. THE CHEAP READ, no training: score the existing checkpoints on the fixed test split.
#    If the mismatch is real, the mftrcad-pretrained checkpoints should move.
conda run -n new_brepmfr --no-capture-output python -m brepformer.scan_checkpoints \
    --ckpt_dirs results/trial5 results/t6a_init_t5 results/trial1_finetune_ss \
    --data_dir brepformer/data/defeature_uv_processed --split test \
    --label_space 5 --collapse marginalize --out results/uv_fix_crosscorpus.json
```

Step 3 is the whole point and it costs one forward pass per model. **The asymmetry is the
diagnostic**, and it has to be stated before the run or it is unfalsifiable:

- `trial5` and `t6a_init_t5` were trained *only* on MFTRCAD-convention grids. If bug 8 is what broke
  them on real parts, they should **improve** on the fixed split.
- `trial1_finetune_ss` was fine-tuned on the broken-mask corpus, so it adapted to `mask = 1`. It
  should **degrade**. That is not a failure of the fix; it is the fix removing an input the incumbent
  learned to rely on.
- If all three move the same way, the mask is not the mechanism and this is a data-quality
  improvement with no bearing on the anti-transfer.

Only if the asymmetry shows up is a re-finetune (`fix8_uv_def`, plus `fix8ctl_legacy_def` on the
unfixed corpus at identical hyperparameters) worth GPU time. This deliberately does **not** touch
`mftrcad_processed`: that corpus already has the right convention, which is the point.

### The corpus control retires the +4.7, and finds a contaminated test split, 2026-09-17

`t6actl_init_t5` finished (rc=0, 2444.9 min): same init as the arm (`trial5` ep39), same schedule,
**mftrcad only — no pcq27 data**. It is the control [the control rule](#a-control-must-move-one-variable)
demanded, and it does not confirm the arm. It refutes it.

Scored first on `trial6a_processed/test`, which is what every earlier 27-class number used:

| checkpoint | trained on | macro | weighted | min-class |
|---|---|---|---|---|
| `t6a_init_t5_wf/bestface-ep85` | mftrcad + pcq27 | 0.8476 | 0.9051 | 0.6292 |
| `t6a_init_t5/bestface-ep74` (recorded) | mftrcad + pcq27 | 0.8529 | 0.9068 | 0.6528 |
| **`t6actl_init_t5/bestface-ep61`** | **mftrcad only** | **0.7841** | 0.8652 | 0.4557 |
| `trial5/best-ep39` (the shared init) | mftrcad | 0.8062 | 0.8721 | 0.5234 |

Read naively that is a *stronger* result than before: the control does not merely fail to reach the
arm, it lands 2.2 points *below* the shared init, so the pcq27 corpus looks worth +6.9. That reading
is wrong, and the reason is the split itself.

**`trial6a_processed/test` is not a ground-truth split.** It is a symlink farm: 2856 links into
`mftrcad_processed/test` (exact synthetic labels) **plus 81 links into
`suitable_samples_pcq27_processed/test`** — GrabCAD parts whose labels are pseudo-labels produced by
the teacher the arm was trained to imitate. 2.8% of the models, and enough to swing a 27-class macro
because those 81 models carry 21 of the 27 classes at tiny support.

```bash
# What the split is actually made of -- resolve the symlinks, do not trust the file count.
python - <<'EOF'
import os, pathlib, collections
c = collections.Counter()
for p in pathlib.Path('brepformer/data/trial6a_processed/test').glob('*.pkl'):
    c['mftrcad' if 'mftrcad_processed' in os.path.realpath(p) else 'pcq27'] += 1
print(c)   # Counter({'mftrcad': 2856, 'pcq27': 81})
EOF
```

Scoring the same four checkpoints on the **uncontaminated** `mftrcad_processed/test`:

| checkpoint | trained on | macro | weighted | min-class | vs init |
|---|---|---|---|---|---|
| **`t6actl_init_t5/bestface-ep61`** | **mftrcad only** | **0.8781** | 0.9280 | 0.6492 | **+0.0070** |
| `t6a_init_t5/bestface-ep74` | mftrcad + pcq27 | 0.8779 | 0.9269 | 0.6528 | +0.0068 |
| `t6a_init_t5_wf/bestface-ep85` | mftrcad + pcq27 | 0.8753 | 0.9246 | 0.6297 | +0.0042 |
| `trial5/best-ep39` (init) | mftrcad | 0.8711 | 0.9233 | 0.6556 | — |

**The control ties the arm to within 0.0002 macro.** On ground truth the pcq27 corpus is worth
+0.7 points, extra epochs on mftrcad alone are worth the same +0.7, and 60 hours of GPU bought a
difference of two ten-thousandths. The +4.7 does not survive removing 81 models from the test set.

And the 81 models on their own say exactly where it came from:

| checkpoint | macro on the 81 pcq27 models |
|---|---|
| `t6a_init_t5/bestface-ep74` | **0.5132** |
| `t6a_init_t5_wf/bestface-ep85` | 0.4859 |
| `trial5/best-ep39` (init) | 0.3351 |
| `t6actl_init_t5/bestface-ep61` (control) | **0.2145** |

The arm beats its init by **+18 points** there and the control *loses* 12. So the pcq27 arm learned
one thing well: to reproduce its own teacher's labels on GrabCAD-shaped parts. That is not the same as
being right — the same arm scores 0.4287 against the 157 **hand-labelled** real parts versus its init's
0.5977 (see "the 27-class gain ANTI-transfers"). It agrees with the teacher more and with the truth
less.

Three things follow, and they are process changes rather than another arm:

1. **Never score on `trial6a_processed` or `trial6_processed` again.** Both are pcq-merged symlink
   farms; both leak pseudo-labels into test. Score on `mftrcad_processed/test` (synthetic GT) and
   `defeature_processed/test` (157 real hand-labelled parts), and report both.
2. **A pseudo-label corpus must never contribute to its own test split.** `merge_suitable_samples.py`
   splits the merged pool 0.8/0.1/0.1 like any other corpus, which is correct for training data and
   silently self-serving for evaluation.
3. **The control was worth its 41 hours precisely because it came back negative.** Without it the
   contamination was invisible: the arm's +4.7 and the control's absence were the same number, exactly
   as the control rule predicted, and the failure mode it caught was not the one it was built for.

```bash
# Reproduce. The clean split is the whole point -- do not substitute trial6a_processed.
rm -rf /tmp/ctl_ab && mkdir -p /tmp/ctl_ab/{init-trial5,arm-ep74,armwf-ep85,ctl-ep61}
ln -s "$PWD/results/trial5/best-epoch=39-val/f1=0.8832.ckpt"                  /tmp/ctl_ab/init-trial5/
ln -s "$PWD/results/t6a_init_t5/bestface-epoch=74-val/face_f1=0.8676.ckpt"    /tmp/ctl_ab/arm-ep74/
ln -s "$PWD/results/t6a_init_t5_wf/bestface-epoch=85-val/face_f1=0.8621.ckpt" /tmp/ctl_ab/armwf-ep85/
ln -s "$PWD/results/t6actl_init_t5/bestface-epoch=61-val/face_f1=0.8873.ckpt" /tmp/ctl_ab/ctl-ep61/

conda run -n new_brepmfr --no-capture-output python -m brepformer.scan_checkpoints \
  --ckpt_dirs /tmp/ctl_ab/init-trial5 /tmp/ctl_ab/arm-ep74 /tmp/ctl_ab/armwf-ep85 /tmp/ctl_ab/ctl-ep61 \
  --data_dir brepformer/data/mftrcad_processed --split test \
  --label_space 27 --out results/scan_ctl_clean_mftrcad_27.json

# The 81 contaminating models, isolated by resolving the symlinks.
python - <<'EOF'
import os, pathlib
out = pathlib.Path('/tmp/pcq27_test'); out.mkdir(exist_ok=True)
for p in sorted(pathlib.Path('brepformer/data/trial6a_processed/test').glob('*.pkl')):
    r = os.path.realpath(p)
    if 'mftrcad_processed' not in r and not (out / p.name).exists():
        (out / p.name).symlink_to(r)
EOF
mkdir -p /tmp/pcq27_only && ln -sfn /tmp/pcq27_test /tmp/pcq27_only/test
conda run -n new_brepmfr --no-capture-output python -m brepformer.scan_checkpoints \
  --ckpt_dirs /tmp/ctl_ab/init-trial5 /tmp/ctl_ab/arm-ep74 /tmp/ctl_ab/armwf-ep85 /tmp/ctl_ab/ctl-ep61 \
  --data_dir /tmp/pcq27_only --split test --label_space 27 \
  --out results/scan_pcq27subset_27.json
```

### Bug 8: the recursive fix ladder, 2026-09-17

A fix is not "done" when the code changes; it is done when a measurement moves, or when every facet
has been separately tested and the ones that do not move are named. Bug 8 has **two mechanisms** (mask
semantics, and a 10%-narrower UV sampling domain) and a **third quantity that moves with them** (the
D2/angle attention-bias descriptors, which are computed from `mask > 0.5` — see
`preprocessing.py:129` and `:207`). A single before/after would confound all three. Hence a ladder:
each rung is cheaper than the one below it, and each rung either localises the effect or kills the
hypothesis before the expensive rung is paid for.

There are also **three independent candidate fixes**, not one. They attack the same mismatch from
different sides, so if the first fails to move anything the hypothesis is not yet dead:

| | Approach | What it changes | Cost | When it is the right one |
|---|---|---|---|---|
| **A** | `steps_to_graphs --mftrcad_uv` | Real parts re-derived under MFTRCAD's convention | 1561 STEP conversions + preprocess (~1 h, no GPU) | The channel carries real signal and the pretrained encoder already knows how to use it |
| **B** | Re-derive MFTRCAD's 28,820 STEPs under the *legacy* convention | Pretraining corpus made to match the real parts | ~20 h CPU | Same hypothesis from the other side; the tie-breaker if A is ambiguous |
| **C** | `--mask_policy const\|zero` | Channel 6 deleted on **both** corpora, in the dataset | Zero — no preprocessing at all | The channel carries little signal but large shift. Also the cheapest falsification: if C alone recovers the transfer, the mask *was* the mechanism |

C is wired into `train_preprocessed.py`, `test_preprocessed.py` and `scan_checkpoints.py` (the last one
patches the pickle directly, because it bypasses `PreprocessedDataset` and calls `prepare_batch`).
`const` and `zero` are information-identical and differ only in what the first BatchNorm sees, which is
why both exist — a constant channel is not automatically inert downstream of normalisation.

**L0 — assert the fix is in the data.** Never score before this. A flag that silently failed to thread
(mine did, twice: the `--single` worker and the `--tolerate_crashes` re-exec both dropped it) produces
a corpus that looks converted and is not.

```bash
python -m brepformer.data.uv_mask_audit disk \
  --pkl_dirs brepformer/data/defeature_uv_processed/test \
             brepformer/data/defeature_processed/test --limit 200
# PASS: defeature_uv_processed masked mean ~0.4 and "any masked" ~100%;
#       defeature_processed still 0.000. If the new corpus reads 0.000, the flag did not thread.
```

**L1 — the free read: no training, no GPU required.** Score existing checkpoints on the legacy and
fixed splits. The prediction is stated *before* running it, and it is asymmetric, which is what makes
it falsifiable rather than a story fitted afterwards:

- `trial5`, `t6a_init_t5`, `t6actl_init_t5` — trained only on MFTRCAD's convention — should **improve**
  on the fixed split.
- `trial1_finetune_ss` — fine-tuned on broken masks — should **degrade**. That is the fix removing an
  input the incumbent learned to lean on, not a failure.
- All three moving the same direction ⇒ the mask is not the mechanism, and bug 8 is a data-quality
  improvement with no bearing on the anti-transfer. Say so and stop climbing.

```bash
for CK in results/trial5 results/t6a_init_t5 results/t6actl_init_t5 results/trial1_finetune_ss; do
  for D in defeature_processed defeature_uv_processed; do
    python -m brepformer.scan_checkpoints --ckpt_dirs "$CK" \
      --data_dir "brepformer/data/$D" --split test \
      --label_space 5 --collapse marginalize --device cpu \
      --out "results/bug8_l1_$(basename $CK)_$D.json"
  done
done
# Candidate C on the same rung, and it needs no new corpus at all:
python -m brepformer.scan_checkpoints --ckpt_dirs results/trial1_finetune_ss \
  --data_dir brepformer/data/defeature_processed --split test \
  --label_space 5 --collapse marginalize --mask_policy zero --device cpu
```

`--collapse marginalize` is not optional for the 27-class checkpoints: argmax-then-remap scores a
27-class head at 5 classes by throwing away the probability mass that the remap would have merged
(bug 1).

**L0 RESULT, 2026-09-17: PASS.** `defeature_uv_processed` (1561 models re-derived, 0 failures) now
reads masked mean **0.400** (median 0.360, 100% of faces masked somewhere, 11.4% over half) against
MFTRCAD's **0.445** (median 0.360, 100%, 22.1%) — same quantity, same median, same shape. The legacy
`defeature_processed` still reads 0.000, so the two corpora are genuinely different and the flag
threaded. Recorded in `results/bug8_l0_disk.json`.

**L1 RESULT, 2026-09-17: the asymmetry predicted in advance is exactly what happened.** Four
checkpoints, same models, 5-class face F1 through `--collapse marginalize`, legacy split vs fixed split:

| checkpoint | trained on masks | legacy | fixed | delta |
|---|---|---|---|---|
| `t6a_init_t5` ep74 (the anti-transferring arm) | MFTRCAD only | 0.4307 | **0.5064** | **+0.0757** |
| `t6actl_init_t5` ep61 (control) | MFTRCAD only | 0.6017 | **0.6304** | **+0.0287** |
| `trial5` ep39 (init) | MFTRCAD only | 0.5884 | **0.6069** | **+0.0185** |
| `trial1_finetune_ss` ep32 (deployed incumbent) | **broken (`mask=1`)** | 0.8853 | 0.8700 | **-0.0153** |

Every checkpoint that only ever saw MFTRCAD's convention improves; the one fine-tuned on broken masks
degrades. That is the pre-stated asymmetry, and it is not a small effect for the arm that
anti-transferred worst — the gap between `t6a_init_t5` and its own init narrows from 0.158 to 0.101, so
**bug 8 accounts for roughly a third of the anti-transfer** on its own, before any retraining.

Per class, the fix is doing what the mechanism predicts. `hole` gains most on the worst-hit checkpoint
(**+0.096**), which is the class the anti-transfer destroyed (0.729 → 0.276) and the class whose faces
overshoot their UV box most: a cylindrical hole wall is a full-period surface trimmed to a partial arc.
`random` +0.216 and `fillet` +0.073 follow. The one consistent *loss* is `chamfer`, down 0.018–0.042 on
all four checkpoints including the ones that gain overall — chamfers are narrow bands where a 10×10 grid
has few samples to spare, so masking costs them coverage. That is a real cost of the fix, not noise, and
it is the thing to watch in L3.

Because L1 came back confirmed, L2 and L3 were launched: L2 on CPU (both attribution corpora in
parallel), L3 on the GPU (`fix8_uv_def` then `fix8ctl_legacy_def`, sequentially, one GPU).

**L2 — facet decomposition, only if L1 moves.** Two more corpora, each isolating one facet, so a
confirmed L1 effect can be attributed rather than just observed:

```bash
# Corpus C -- fixed mask, LEGACY descriptors. Isolates "the encoder sees a real mask" from
# "the attention bias is computed over different points".
python -m brepformer.preprocess --data_dir brepformer/data/defeature_uv \
  --output_dir brepformer/data/defeature_uv_legacydesc_processed \
  --num_classes 5 --compute_descriptors --descriptors_ignore_mask \
  --num_spatial 64 --d2_bins 64 --angle_bins 64 --split_ratio 0.8,0.1,0.1 --seed 42

# Corpus D -- trim masking over the UNPADDED box. Isolates masking from the sampling domain.
# --uv_pad 0 is a deliberate setting, not a disabled one.
python -m brepformer.data.steps_to_graphs --steps_dir brepformer/data/defeature/steps \
  --out_dir brepformer/data/defeature_uv0 --mftrcad_uv --uv_pad 0.0 --tolerate_crashes
```

Read L2 as: if corpus D recovers most of L1's effect, the *mask* is the mechanism and the sampling
domain is a rounding error. If only the full pad does, the shifted xyz/normal sample locations matter
too, and B becomes worth its 20 CPU-hours. If corpus C kills the effect, what moved was the attention
bias, not the encoder input — a different bug wearing bug 8's clothes.

**L3 — retrain, one variable.** Same init, same hyperparameters as the 0.8886 incumbent, one corpus
apart. Both arms, or the corpus and the extra epochs are the same number again (see
[the control rule](#a-control-must-move-one-variable)).

```bash
for TAG in fix8_uv_def:defeature_uv_processed fix8ctl_legacy_def:defeature_processed; do
  python brepformer/train_preprocessed.py \
    --data_dir "brepformer/data/${TAG#*:}" \
    --num_classes 5 --num_face_classes 5 --max_faces 1000 \
    --batch_size 2 --accumulate_grad_batches 16 --learning_rate 0.001 \
    --max_epochs 200 --seed 42 --output_dir results --face_segmentation \
    --lr_monitor val/face_f1 --early_stop_monitor val/face_f1 --early_stop_patience 25 \
    --attr_layout categorical \
    --init_weights "results/trial1_finetune_ss/best-epoch=32-val/f1=0.9354.ckpt" \
    --exp_name "${TAG%%:*}"
done
```

**L4 — if no facet moves at L1 or L2.** Do not keep re-running A with different paddings; that is
sweeping, not diagnosing. Switch approach: C at train time (`--mask_policy const` on both arms of an
L3-shaped pair) tests whether *removing* the channel helps even when repairing it did not, and B tests
the same hypothesis from the pretraining side. If A, B and C all come back flat, bug 8 is real as a
data defect and false as an explanation for the anti-transfer — record that, and the 28.5% of faces
being fed off-face geometry is still worth fixing on its own merits.

## 9. What is still outstanding, in dependency order (updated 2026-09-15)

Not everything is done. One job is running, four are blocked behind it, and two are open questions
with no scheduled arm. Each entry says what it is waiting on, so nothing here has to be re-derived.

**DONE — `t6a_init_t5`.** 100 epochs, 62 h, early-stopped; best 27-class val face F1 **0.8676 at
ep74**. It kept improving for 44 epochs after the ep21-30 stretch that looked like a plateau.

**RUNNING — `t6a_init_t5_wf`, resumed at epoch 11 after a four-day stall.** It crashed on 2026-09-09
with a transient `CUDA error: unknown error` and nothing restarted it; the driver now retries and
resumes (see "A transient CUDA fault idled the GPU for four days" above). Best val face F1 so far
0.8491 at ep10, to be compared against `t6a_init_t5`'s 0.8676 on `test`, not val.

**QUEUED behind it — `t6actl_init_t5`, the control that actually attributes the +4.7.** Replaces
`t6actl_base`, which changed the corpus *and* the initialization and so could not isolate either.

```bash
tr '\r' '\n' < results/sweep_logs/t6a_init_t5_wf.log | grep -oE 'Epoch [0-9]+' | tail -1
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader   # idle GPU looks like progress
cat results/sweep_logs/status.json                                         # rc + minutes + attempts
```

**1. DONE 2026-09-08, RETRACTED 2026-09-17 — the init comparison on `test`.** Arm **0.8529** macro vs
init **0.8062**, +4.7 points, 23 of 27 classes up — measured on `trial6a_processed/test`, which
contains 81 pcq27 pseudo-labelled models. On the clean `mftrcad_processed/test` the arm gains +0.0068
and the control gains +0.0070. See "The corpus control retires the +4.7". The original figures below
are kept because the class-level analysis is still valid *on that split*; they are not evidence about
the corpus. The val figure understated it, so it was a subsample effect, not
selection bias. Full table and the two real regressions (`rectangular_passage`, `6sides_passage`) are
in "The init comparison survives on `test`" above. Repeat the same command for `t6a_init_t5_wf` when
it stops, adding its `bestface-*` directory as a third `--ckpt_dirs` entry.

**2. DONE 2026-09-17, NEGATIVE — the corpus-attribution control, `t6actl_init_t5`.** rc=0, 2444.9 min.
On clean GT it ties the arm to 0.0002 macro (0.8781 vs 0.8779) without ever seeing pcq27, and it
exposed the contaminated test split. The pcq27 corpus buys nothing measurable on ground truth. Full
result in "The corpus control retires the +4.7"; the original rationale is kept below because it is
still the right rationale.

**2b. NEW, from that control — stop pseudo-labels reaching test splits.** `merge_suitable_samples.py`
splits the merged pool 0.8/0.1/0.1, so every pcq corpus puts ~10% of its pseudo-labelled models into
its own `test`. Route pseudo-labelled models to `train` only, and until that lands, evaluate on
`mftrcad_processed/test` and `defeature_processed/test` — never on `trial6*_processed/test`.

**2c. Original rationale for item 2, unchanged:** the corpus-attribution control. Without it, "the
pcq27 data helped" and "62 more hours of training helped" are the same number, and the arm's +4.7
macro is only interesting if it is attributable to the corpus. `t6actl_base` was the wrong instrument
for this — it is from-scratch, so it drops the init as well as the data — and is demoted to a floor.

```bash
python -m brepformer.sweeps.trial6_sweep --run --family 27 --only t6actl_init_t5
# floor only, expect collapse, do not read as a rival:
python -m brepformer.sweeps.trial6_sweep --run --family 27 --only t6actl_base --include_scratch
```

**3. BLOCKED on the GPU — the D2 attribution arm.** The 27-class corpora are repaired; the 5-class
ones deliberately are not, so this is the one clean measurement of the D2 fix on its own, against a
0.8886 baseline measured on the unrepaired corpus. Repair first, then re-run exactly the incumbent's
configuration and change nothing else.

```bash
python -m brepformer.data.repair_descriptors \
  --dirs brepformer/data/defeature_processed brepformer/data/trial6_processed --check_only
python -m brepformer.data.repair_descriptors \
  --dirs brepformer/data/defeature_processed brepformer/data/trial6_processed

python brepformer/train_preprocessed.py \
  --data_dir brepformer/data/defeature_processed \
  --num_classes 5 --num_face_classes 5 --max_faces 1000 \
  --batch_size 2 --accumulate_grad_batches 16 --learning_rate 0.001 \
  --max_epochs 200 --seed 42 --output_dir results --face_segmentation \
  --lr_monitor val/face_f1 --early_stop_monitor val/face_f1 --early_stop_patience 25 \
  --attr_layout categorical \
  --init_weights "results/trial1_finetune_ss/best-epoch=32-val/f1=0.9354.ckpt" \
  --exp_name fix2_d2repair_def
```

Note `--attr_layout categorical`: the layout A/B was negative, so this arm must hold the layout at
the incumbent's setting or it measures two things at once again.

**4. BLOCKED on the GPU — the rotation arms, still genuinely open.** `--rotate_aug` exists and is
verified label- and invariant-preserving, but no arm has ever trained with it, so its effect on face
F1 is unmeasured. Run `octahedral` as well as `so3`: if the axis-aligned frames recover `hole` and
uniform SO(3) does not, the loss is grid interpolation rather than the frame, which are different
problems with different fixes.

```bash
python -m brepformer.sweeps.trial6_sweep --run --family 5 --rotate_aug so3
python -m brepformer.sweeps.trial6_sweep --run --family 5 --rotate_aug octahedral
```

**4b. DONE 2026-09-14 — the cross-corpus check, run ahead of the labelling work.** The 27-class arm
scores **0.4287** macro on the 157 real hand-labelled parts against its init's 0.5977 and the deployed
5-class model's 0.8886. The synthetic gain anti-transfers. See "Cross-corpus: the 27-class gain
ANTI-transfers to real parts" above. **Every future 27-class arm reports this number too**, or a
synthetic gain gets read as progress again:

```bash
python -m brepformer.scan_checkpoints --ckpt_dirs <one-ckpt-dir> \
  --data_dir brepformer/data/defeature_processed --split test \
  --label_space 5 --collapse marginalize --device cpu
```

**4c. NOW THE PRIORITY, and it needs no GPU — bug 8, the UV mask convention mismatch.** Jumps the
queue for two reasons: the first read is a re-preprocess plus a scoring pass, so it runs *beside* the
training arms instead of waiting for them, and it is the only candidate mechanism yet found for 4b's
anti-transfer. `face_grid` channel 6 is identically 1 on every STEP-derived corpus and 44.5% masked on
the pretraining corpus, and MFTRCAD's convention is now fitted exactly (pad the face UV box 10% per
side, then mask `TopAbs_OUT`; mean absolute per-face error 0.038 against their own masks). The fix is
built behind `--mftrcad_uv`; the asymmetric prediction that makes the scoring pass falsifiable is
stated in "Bug 8" above, along with the commands, and the L0-L4 ladder plus the three independent
candidate fixes are in "Bug 8: the recursive fix ladder". Only a confirmed asymmetry buys the arms
`fix8_uv_def` / `fix8ctl_legacy_def`.

```bash
python -m brepformer.data.uv_mask_audit disk \
  --pkl_dirs brepformer/data/defeature_processed/test brepformer/data/mftrcad_processed/test \
  --graph_dirs brepformer/data/mftrcad/graphs --limit 150
python -m brepformer.data.uv_mask_audit fit --limit 15 --pad_sweep 0 0.1 0.15 0.2 0.3
python -m brepformer.data.uv_mask_audit steps --steps_dir brepformer/data/defeature/steps --limit 12
```

**5. AFTER bug 8 — bug 7, the dead multi-hop edge pathway.** `edge_path` is never produced,
section 5 of `GraphAttnBias.forward` therefore never runs, and it would carry no edge information even
if it did; `edge_conv_layer` (the old bug 5) is the aggregation step at the end of it. This is the
largest remaining architectural gap and the one most likely to move fillet/chamfer, because today two
faces that do not share an edge exchange no edge geometry at all. Needs three new flags, a
path-tracking BFS, a rewritten section 5, the D2 arm as its baseline, and its own
`fix7ctl_noedgepath` control on the same re-preprocessed corpus. Full design, cost analysis and
commands in "Bug 7: the entire multi-hop edge pathway is dead" above.

**6. BLOCKED on the 5-class work above — the labelling workflow.** `accept_queue calibrate
--label_space 27` and 27-class fusion were queued against a 27-class model. The cross-corpus result
demotes them: a model that scores 0.43 on real parts cannot calibrate an accept threshold for real
parts. Re-point both at whichever 5-class checkpoint wins items 3/5, and keep
`--gt_pkl_dir brepformer/data/defeature_processed/test` — the 157 real parts are the only ground
truth an accept guarantee can be built on.

**7. Bug 5 is now folded into item 5.** It was filed as "wire `edge_conv_layer` into `forward`, an
architecture change that breaks `load_from_checkpoint`". Both halves were wrong: it is one step of the
bug-7 mechanism, not a standalone fix, and it adds **no parameters** — `edge_conv_layer` and
`edge_dis_encoder` are already constructed and already saved in every checkpoint, so wiring them in
keeps every checkpoint loadable and lets a warm start work normally.

**8. Deliberately NOT doing — the `onehot` re-sweep.** The layout A/B was negative (0.8699 vs
0.8886). The flag stays, `categorical` stays the pin, and reopening this needs a bigger corpus or an
init that never saw the broken layout, not another sweep arm.
