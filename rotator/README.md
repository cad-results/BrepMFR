# Rotator - BREP Pipe Fitting Alignment & 6-View Rendering

Aligns STEP-format pipe fittings (elbows, tees) so their pipe axes are
parallel to the coordinate axes, then renders 6 orthographic views.

## Environment

Uses the `rotator` conda environment. Key dependencies are listed in
`environment.yml`:

| Package | Purpose |
|---|---|
| pythonocc-core 7.9.0 | OpenCASCADE Python bindings - STEP loading, BREP topology, transforms |
| vtk 9.4.2 | Offscreen 6-view rendering (orthographic PNG export) |
| numpy | Array math |
| scipy | Rotation conversions (matrix <-> axis-angle) |
| matplotlib + pillow | Debug visualizations and comparison grids |

Activate before running:

```bash
conda activate rotator
# or prefix commands:
conda run -n rotator python pipeline.py ...
```

## Pipeline

`pipeline.py` runs the full process: **Load -> Align -> Tessellate -> Render**.

### Usage

```bash
# Single file
python pipeline.py --step data/ssdata1/steps/0-90deg_LR_Inch_Elbow.step -o output/

# Batch (first 20 files)
python pipeline.py --step_dir data/ssdata1/steps/ -o output/ --limit 20

# Process a specific class folder
python pipeline.py --step_dir data/ssdata1/4_tee_wf/ -o output/tee_wf/

# Simple output: just 6 PNGs per model, no subdirectories
python pipeline.py --step data/ssdata1/steps/some_file.step -o output/ --simple

# Show triangular mesh edges in rendered views
python pipeline.py --step data/ssdata1/steps/some_file.step -o output/ --mesh

# With debug visualizations at each step
python pipeline.py --step data/ssdata1/steps/some_file.step -o output/ --visualize

# Interactive 3D viewer (requires display / WSLg)
python pipeline.py --step data/ssdata1/steps/some_file.step -o output/ --interactive

# Custom image size
python pipeline.py --step data/ssdata1/steps/some_file.step -o output/ --image_size 1024 1024

# Verbose logging
python pipeline.py --step data/ssdata1/steps/some_file.step -o output/ -v
```

### CLI flags

| Flag | Description |
|---|---|
| `--step PATH` | Process a single STEP file |
| `--step_dir DIR` | Batch process all `.step`/`.stp` files in a directory |
| `-o`, `--output DIR` | Output directory (default: `output/`) |
| `--limit N` | Cap batch mode to first N files (0 = all) |
| `--mesh` | Show internal triangular mesh edges in rendered views (hidden by default) |
| `--simple` | Simple output: just 6 view PNGs per model, no `views/` subdirectory or extras |
| `--visualize` | Save before/after alignment plots and 6-view grid image |
| `--interactive` | Open a VTK 3D window after alignment (needs a display) |
| `--image_size W H` | Rendered image dimensions in pixels (default: 800 800) |
| `-v`, `--verbose` | Debug-level logging |

### Output structure

**Default:**
```
output/<model_name>/
  views/
    <name>_front.png      # +Y camera, looking toward origin
    <name>_back.png       # -Y camera
    <name>_right.png      # +X camera
    <name>_left.png       # -X camera
    <name>_top.png        # +Z camera
    <name>_bottom.png     # -Z camera
  debug/                  # only with --visualize
    <name>_alignment.png  # side-by-side original vs aligned with axis arrows
  <name>_grid.png         # only with --visualize; 2x3 montage of all 6 views
```

**With `--simple`:**
```
output/<model_name>/
  <name>_front.png
  <name>_back.png
  <name>_right.png
  <name>_left.png
  <name>_top.png
  <name>_bottom.png
```

By default, rendered views show the clean solid surface without internal
triangulation lines. Pass `--mesh` to overlay the triangular mesh edges.

## How alignment works

The alignment is **one global rotation per STEP file**. It does not operate
per-face or per-component -- the entire shape is rotated as a rigid body so
that the dominant pipe directions end up on the coordinate axes.

### Step 1: Extract pipe axis directions

Every face in the BREP is inspected. Depending on its parametric surface type,
the pipe axis direction is extracted differently:

| Surface type | How the axis is found | Weight |
|---|---|---|
| **Cylinder** | `surf.Cylinder().Axis().Direction()` -- exact from parametric definition | 1.0 x area |
| **Cone** | `surf.Cone().Axis().Direction()` -- exact | 0.5 x area |
| **Torus** | Tangent to the major circle at U-parameter bounds; gives the pipe opening direction at each end of the toroidal bend | 0.8 x area |
| **BSpline** | Sample surface normals on a grid, run PCA; if the normals lie in a plane (smallest eigenvalue ratio < 0.05), the surface is cylindrical and the plane normal is the axis | 0.7 x area |
| **SurfaceOfRevolution** | `surf.AxeOfRevolution().Direction()` | 0.6 x area |

The weighting by surface area ensures that the main pipe bodies dominate over
small features like bolt holes or chamfers.

### Step 2: Cluster into unique directions

Detected axes are clustered with a 15-degree angular threshold.
Anti-parallel directions (e.g. +X and -X) are treated as the same axis.
Within each cluster, directions are averaged weighted by area, producing
one representative unit vector per unique pipe direction.

Clusters are sorted by total area so the most prominent axis comes first.

### Step 3: Build rotation matrix

Given the top two cluster directions `d1`, `d2`:

1. Gram-Schmidt: orthogonalize `d2` against `d1`
2. Cross product: `d3 = d1 x d2_orth`
3. Assemble `R = [d1; d2_orth; d3]` (rows are the source frame axes)
4. Ensure `det(R) = +1` (proper rotation, no reflection)

This maps `d1 -> X`, `d2_orth -> Y`, `d3 -> Z`.

**Fallback cases:**
- 1 unique axis: that axis maps to Z; X and Y are chosen arbitrarily orthogonal
- 0 axes detected: PCA on sampled surface points; principal axes -> XYZ
- Degenerate / empty shape: identity (no rotation)

### Step 4: Apply transform

1. Translate center of mass to the origin
2. Apply rotation via axis-angle (converted from `R` using `scipy.spatial.transform.Rotation`)
3. Both steps use `BRepBuilderAPI_Transform` so the full BREP topology is preserved

### Step 5: Verify

After alignment, pipe axes are re-extracted from the transformed shape and
compared against the nearest coordinate axis. The maximum angular deviation
is reported:

- **PERFECT**: < 1 degree
- **GOOD**: < 5 degrees (typical for BSpline-approximated cylinders)
- **CHECK**: >= 5 degrees (usually non-90-degree elbows where perfect axis alignment is geometrically impossible)

### Why non-90-degree elbows show larger deviations

A 45-degree elbow has pipe openings at 45 degrees to each other. After
alignment, one pipe axis maps exactly to a coordinate axis, but the other
is inherently at 45 degrees from the nearest axis. This is mathematically
unavoidable and the alignment is still optimal -- the algorithm aligns the
first (most prominent) axis perfectly and places the second as close to the
next coordinate axis as orthogonalization allows.

## Pipeline steps in code

```
process_single()                   # Entry point per file
  load_step(path)                  # STEPControl_Reader -> TopoDS_Shape
  align_shape(shape)               # Full alignment pipeline:
    get_shape_center(shape)        #   GProp center of mass
    compute_alignment_rotation()   #   Axis detection + clustering + R
      extract_pipe_axes(shape)     #     Cylinder/Cone/Torus/BSpline/Revolution
      cluster_axes(axes)           #     Group by direction, weight by area
    apply_transform(shape, R, c)   #   BRepBuilderAPI_Transform
    verify_alignment(aligned)      #   Re-detect axes, measure deviation
  tessellate_shape(aligned)        #   BRepMesh -> vertices + triangles
  render_six_views(verts, tris)    #   VTK offscreen orthographic rendering
    make_vtk_polydata()            #     Build VTK mesh with normals
    render_single_view() x 6      #     One PNG per camera direction
```

## 6 rendered views

All views use orthographic (parallel) projection centered on the origin.

| View | Camera position | Up vector | What it shows |
|---|---|---|---|
| Front | +Y | +Z | Pipe profile or opening along Y axis |
| Back | -Y | +Z | Opposite side of front |
| Right | +X | +Z | Pipe profile or opening along X axis |
| Left | -X | +Z | Opposite side of right |
| Top | +Z | -Y | Plan view; T-shape for tees, L-shape for elbows |
| Bottom | -Z | +Y | Underside plan view |

## Reference: BrepMFR utilities

The `BrepMFR/brepformer/` folder contains related tools that may be useful
for extended workflows:

- `visualize_seg.py` -- Qt + pythonOCC interactive viewer for per-face
  segmentation labels (27 MFTRCAD classes or 8 real classes)
- `export_freecad.py` -- Exports STEP files with per-face colors using
  XCAF (STEPCAFControl_Writer + XCAFDoc_ColorSurf) for FreeCAD viewing
- `data/step_to_graph.py` -- Converts STEP to graph tensors; contains
  `normalize_geometry()` (center + scale to unit sphere) and
  `compute_face_attributes()` (14-dim per-face feature vector including
  surface type, area, centroid, normal, bounding box)
