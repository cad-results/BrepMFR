# BrepSimplifier Scripts Reference

All scripts live under `brepformer/brepsimplifier/` and share the `new_brepmfr` conda environment (pythonocc-core >= 7.9).

---

## Core Primitive Fitting

### `primitives.py` (CLI)

Fit geometric primitives (box, cylinder, cone, sphere, torus) to a defeatured STEP file. Reads exact surface parameters from the OCC kernel — no statistical estimation. BSpline and other freeform surfaces are integrated into rectangular prisms with configurable minimum dimensions.

```bash
# Direct mode: fit primitives to an already-defeatured STEP file
python -m brepformer.brepsimplifier.primitives \
    --step brepformer/brepsimplifier/test_data/stepfiles/4756040110.stp \
    --output 4756040110_primitives.step \
    --tol 0.1 \
    --verbose

# Direct mode with lenient BSpline tessellation (for original/non-defeatured shapes)
python -m brepformer.brepsimplifier.primitives \
    --step /mnt/c/projects/data/steps/g-MEC-GRUNDFOSPUMP-P410_LOD400.step \
    --output pump_primitives.step \
    --lenient --verbose

# Pipeline mode: defeature first, then fit primitives
python -m brepformer.brepsimplifier.primitives \
    --step brepformer/brepsimplifier/test_data/stepfiles/Á111Van.stp \
    --pipeline \
    --output_dir brepformer/brepsimplifier/pipeline_output \
    --checkpoint results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt \
    --tol 0.1 \
    --verbose

# Using a real test file in pipeline mode
# (lenient mode activates automatically if defeaturing fails)
python -m brepformer.brepsimplifier.primitives \
    --step brepformer/brepsimplifier/test_data/stepfiles/g-MEC-GRUNDFOSPUMP-P410_LOD400.stp \
    --pipeline \
    --output_dir brepformer/primitives_output/ \
    --verbose

# From pre-computed .seg predictions (skip inference)
python -m brepformer.brepsimplifier.primitives \
    --step <your_model>.step \
    --pipeline \
    --seg predictions.seg \
    --output_dir pipeline_output/
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--step` | required | Input STEP file (defeatured or raw) |
| `--output` | `{stem}_primitives.step` | Output STEP path (direct mode) |
| `--pipeline` | false | Run full defeature → primitives pipeline |
| `--checkpoint` | None | Model checkpoint (pipeline mode) |
| `--seg` | None | Pre-computed .seg file (pipeline mode, skips inference) |
| `--output_dir` | `brepformer/primitives_output` | Output directory (pipeline mode) |
| `--tol` | 0.1 | Geometric tolerance for face grouping |
| `--max_fuzzy` | 1e-3 | Max fuzzy tolerance for defeaturing |
| `--save_colored` | false | Save colored STEP with predictions (pipeline) |
| `--save_intermediate` | false | Keep intermediate defeatured STEP (pipeline) |
| `--lenient` | false | Tessellate BSpline faces into small prisms (use for originals without defeaturing) |
| `--verbose` | false | Detailed progress output |

---

### `run_simplifier.sh`

Shell wrapper for `primitives.py` with the same arguments. Sets project root and runs `python -m brepformer.brepsimplifier.primitives`.

```bash
# Direct mode on a real test file
./brepformer/brepsimplifier/run_simplifier.sh \
    --step brepformer/brepsimplifier/test_data/stepfiles/Á111Van.stp \
    --verbose

# Pipeline mode
./brepformer/brepsimplifier/run_simplifier.sh \
    --step <your_model>.step \
    --pipeline \
    --verbose

# With lenient mode for BSpline-heavy models
./brepformer/brepsimplifier/run_simplifier.sh \
    --step <your_model>.step \
    --lenient --verbose

# With all options
./brepformer/brepsimplifier/run_simplifier.sh \
    --step <your_model>.step \
    --output <output>.step \
    --tol 0.15 \
    --verbose
```

---

## Visualization

### `view_step.sh`

Qt + pythonOCC STEP viewer with WSL2 software rendering support. Wraps `brepformer.visualize_defeature`. Auto-configures display for WSLg, VcXsrv, and native X11.

```bash
# View a single STEP file (e.g. a test model)
./brepformer/brepsimplifier/view_step.sh \
    brepformer/brepsimplifier/test_data/stepfiles/4756040110.stp

./brepformer/brepsimplifier/view_step.sh \
    brepformer/brepsimplifier/pipeline_output/Á111Van_primitives.step

# Compare original vs. defeatured
./brepformer/brepsimplifier/view_step.sh \
    --original brepformer/brepsimplifier/test_data/stepfiles/g-MEC-GRUNDFOSPUMP-P410_LOD400.stp \
    --defeatured brepformer/primitives_output/g-MEC-GRUNDFOSPUMP-P410_LOD400_defeatured.step

# Compare original vs. primitive output
./brepformer/brepsimplifier/view_step.sh \
    --original brepformer/brepsimplifier/test_data/stepfiles/Á111Van.stp \
    --defeatured brepformer/brepsimplifier/pipeline_output/Á111Van_primitives.step

# Browse all STEP files in a results directory
./brepformer/brepsimplifier/view_step.sh \
    --dir brepformer/brepsimplifier/test_data/results/

# View any arbitrary STEP file
./brepformer/brepsimplifier/view_step.sh <your_model>.step
```

| Argument | Default | Description |
|----------|---------|-------------|
| `<file.step>` | — | Single file to view |
| `--original`, `-o` | — | Original STEP file (comparison mode) |
| `--defeatured`, `-d` | — | Defeatured STEP file (comparison mode) |
| `--primitives`, `--prim`, `-p` | — | Primitives STEP file (for reference) |
| `--dir` | — | Directory of STEP files to browse |

Display is auto-configured for WSL2 (WSLg, VcXsrv) with Mesa software rendering (`LIBGL_ALWAYS_SOFTWARE=1`, `GALLIUM_DRIVER=llvmpipe`). Falls back to host IP via `/etc/resolv.conf` nameserver detection if standard DISPLAY values fail.

---

## Evaluation & Comparison

### `test_data/compare_primitives.py`

Runs BrepMFR primitive fitting on all STEP files in `test_data/stepfiles/` and evaluates quality vs. PartField's statistical methods.

```bash
# Run the full evaluation (processes all 4 test models)
python brepformer/brepsimplifier/test_data/compare_primitives.py

# Typical output (abbreviated):
#   Model                           Faces  Prims   VolRat  Valid   Time
#   4756040110                        714    298   1.0000      Y   1.4s
#   Guide_Roller_Assembly            1076    548   1.0000      Y   1.8s
#   g-MEC-GRUNDFOSPUMP-P410_LOD400    216     44   1.0000      Y   0.5s
#   Á111Van                           406    227   1.0000      Y   0.7s
```

Reports:
- **Volume accuracy** — volume ratio between fitted primitives and original model (calibrated to 1.0)
- **Primitive count and type distribution** — boxes, cylinders, cones, spheres, tori
- **Surface type fidelity** — are cylinders preserved as cylinders, not approximated as boxes?
- **B-rep validity** — does the output pass `BRepCheck_Analyzer`?
- **Timing** — wall-clock time per model
- **PartField limitations** — documents why statistical methods (PCA, Powell optimization, bounding-box heuristics) produce inferior results

Output: `test_data/results/comparison_report.json`

---

## Test Data

### `test_data/stepfiles/`

Four industrial STEP files copied from PartField for advanced testing:

| File | Faces | Solids | Description |
|------|-------|--------|-------------|
| `4756040110.stp` | 714 | 10 | Multi-solid mechanical assembly |
| `Guide_Roller_Assembly.stp` | 1076 | 15 | Roller bearing assembly with many cones |
| `g-MEC-GRUNDFOSPUMP-P410_LOD400.stp` | 216 | 1 | Industrial pump (many BSpline surfaces) |
| `Á111Van.stp` | 406 | 28 | Vehicle component with diverse primitives |

### `test_data/results/`

Generated primitive STEP files and `comparison_report.json` from the evaluation script.

| File | Description |
|------|-------------|
| `4756040110_primitives.step` | 298 primitives (164 cylinders, 55 planes, 47 tori, 23 cones, 8 spheres) |
| `Guide_Roller_Assembly_primitives.step` | 548 primitives (347 cones, 178 cylinders, 13 planes, 10 tori) |
| `g-MEC-GRUNDFOSPUMP-P410_LOD400_primitives.step` | 44 primitives (30 cylinders, 13 planes, 1 cone) |
| `Á111Van_primitives.step` | 227 primitives (121 cylinders, 55 cones, 25 planes, 20 tori, 4 spheres) |
| `comparison_report.json` | Full evaluation report with per-model metrics and PartField limitations |

---

## Other Entry Points

### `brepformer/primitives.py` (re-export wrapper)

Thin re-export of `brepformer.brepsimplifier.primitives` for backward compatibility with `defeature_v2.py`:

```python
from brepformer.primitives import fit_primitives, fit_primitives_from_step
```

### `brepformer/scripts/run_primitives.sh`

Original shell entry point (runs `python -m brepformer.primitives`). Equivalent to `run_simplifier.sh`.

### `brepformer/defeature_v2.py --primitives`

The defeature script has `--primitives` and `--primitives_tol` flags that run primitive fitting after defeaturing in single-file mode:

```bash
python -m brepformer.defeature_v2 \
    --step brepformer/brepsimplifier/test_data/stepfiles/4756040110.stp \
    --primitives \
    --primitives_tol 0.1 \
    --verbose

# Or with a placeholder
python -m brepformer.defeature_v2 \
    --step <your_model>.step \
    --primitives \
    --primitives_tol 0.1 \
    --verbose
```

---

## Python API

```python
from brepformer.brepsimplifier import fit_primitives, fit_primitives_from_step, defeature_and_fit

# Direct: shape → primitives (returns shape + report dict)
result_shape, report = fit_primitives(shape, verbose=True, tol=0.1)
# report = {
#   "status": "success",
#   "num_primitives": 298,
#   "primitive_types": ["cylinder", "plane", "torus", ...],
#   "volume_ratio": 1.0,
#   "scale_applied": 0.923,
#   "valid": True,
#   "elapsed_s": 1.38,
# }

# Direct with lenient mode (BSpline faces → many small prisms)
result_shape, report = fit_primitives(shape, verbose=True, tol=0.1, lenient=True)

# File-to-file: STEP → primitives → STEP
report = fit_primitives_from_step(
    "brepformer/brepsimplifier/test_data/stepfiles/4756040110.stp",
    "output_primitives.step",
    tol=0.1,
)

# File-to-file with lenient mode
report = fit_primitives_from_step(
    "original_with_bsplines.step",
    "output_primitives.step",
    tol=0.1,
    lenient=True,
)

# Full pipeline: raw STEP → defeature → primitives
# (lenient activates automatically when defeaturing fails)
report = defeature_and_fit(
    "<your_model>.step",
    "output_dir/",
    checkpoint="results/trial1_ss/best-*.ckpt",
    verbose=True,
)
```

---

## Environment Setup

All scripts require the `new_brepmfr` conda environment with pythonocc-core >= 7.9:

```bash
conda activate new_brepmfr

# Verify OCC is available
python -c "from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox; print('OK')"
```

The viewer scripts additionally require a working X11 display (WSLg, VcXsrv, or native). Mesa software rendering is used automatically on WSL2.
