#!/usr/bin/env python3
"""BREP Alignment and Multi-View Rendering Pipeline for Pipe Fittings.

Aligns STEP-format pipe fittings (elbows, tees) so that cylindrical pipe
axes are parallel to coordinate axes, then renders 6 orthographic views.

Alignment algorithm:
  1. Extract pipe axis directions from BREP topology (exact):
     - Cylinder/Cone: axis read directly from parametric surface
     - Torus: pipe opening directions computed from tangent at U bounds
     - BSpline: cylinder detection via PCA on sampled surface normals
  2. Cluster axes by direction (anti-parallel treated as equivalent)
  3. Build orthonormal frame from the two most prominent pipe directions
  4. Compute rotation mapping pipe axes -> coordinate axes
  5. Falls back to PCA when no analytic features detected

Usage:
    conda run -n rotator python pipeline.py --step data/steps/some_file.step -o output/
    conda run -n rotator python pipeline.py --step_dir data/steps/ -o output/ --limit 5
    conda run -n rotator python pipeline.py --step data/steps/some_file.step -o output/ --visualize
    conda run -n rotator python pipeline.py --step data/steps/some_file.step -o output/ --simple
    conda run -n rotator python pipeline.py --step data/steps/some_file.step -o output/ --mesh
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCC.Core.TopoDS import topods
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import (
    GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Torus,
    GeomAbs_BSplineSurface, GeomAbs_SurfaceOfRevolution,
)
import math
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.gp import gp_Trsf, gp_Vec, gp_Ax1, gp_Dir, gp_Pnt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRep import BRep_Tool
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# STEP Loading
# ============================================================

def load_step(path: str):
    """Load a STEP file and return the TopoDS_Shape."""
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP file: {path}")
    reader.TransferRoots()
    return reader.OneShape()


# ============================================================
# Cylinder Axis Extraction (exact from BREP topology)
# ============================================================

def _get_face_area(face) -> float:
    """Get surface area of a face."""
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    return abs(props.Mass())


def _extract_torus_pipe_directions(face) -> List[Tuple[np.ndarray, float]]:
    """Extract pipe opening directions from a toroidal surface.

    For a toroidal bend (e.g. elbow), the tangent direction at each end of
    the U parameter range gives the pipe opening direction.

    Returns:
        List of (direction, weighted_area) for each pipe opening.
    """
    surf = BRepAdaptor_Surface(face)
    torus = surf.Torus()
    area = _get_face_area(face)
    if area < 1e-10:
        return []

    # Torus coordinate system
    ax3 = torus.Position()
    x_dir = np.array([ax3.XDirection().X(), ax3.XDirection().Y(), ax3.XDirection().Z()])
    y_dir = np.array([ax3.YDirection().X(), ax3.YDirection().Y(), ax3.YDirection().Z()])

    # U parameter bounds define the angular extent of the bend
    u_min = surf.FirstUParameter()
    u_max = surf.LastUParameter()

    results = []
    # Tangent at angle u in torus frame: (-sin(u), cos(u), 0)
    # In global: tangent = -sin(u) * x_dir + cos(u) * y_dir
    for u in [u_min, u_max]:
        tangent = -math.sin(u) * x_dir + math.cos(u) * y_dir
        norm = np.linalg.norm(tangent)
        if norm > 1e-10:
            tangent /= norm
            results.append((tangent, area * 0.8))

    return results


def _detect_bspline_cylinder_axis(face, n_samples: int = 8) -> Optional[Tuple[np.ndarray, float]]:
    """Detect if a BSpline surface is approximately cylindrical via normal PCA.

    Samples surface normals and checks if they lie in a plane (as they would
    for a cylinder). The normal to that plane is the cylinder axis direction.

    Returns:
        (axis_direction, area) if cylindrical, else None.
    """
    surf = BRepAdaptor_Surface(face)
    u_min = max(surf.FirstUParameter(), -1e4)
    u_max = min(surf.LastUParameter(), 1e4)
    v_min = max(surf.FirstVParameter(), -1e4)
    v_max = min(surf.LastVParameter(), 1e4)

    normals = []
    pnt = gp_Pnt()
    from OCC.Core.gp import gp_Vec
    d1u = gp_Vec()
    d1v = gp_Vec()

    for i in range(n_samples):
        for j in range(n_samples):
            u = u_min + (u_max - u_min) * i / max(n_samples - 1, 1)
            v = v_min + (v_max - v_min) * j / max(n_samples - 1, 1)
            try:
                surf.D1(u, v, pnt, d1u, d1v)
                n_vec = d1u.Crossed(d1v)
                mag = n_vec.Magnitude()
                if mag > 1e-10:
                    normals.append([n_vec.X() / mag, n_vec.Y() / mag, n_vec.Z() / mag])
            except Exception:
                pass

    if len(normals) < 6:
        return None

    normals_arr = np.array(normals)

    # PCA of normals: for a cylinder, normals lie in a plane,
    # so the smallest eigenvalue should be near zero
    center = normals_arr.mean(axis=0)
    centered = normals_arr - center
    cov = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Ratio test: smallest eigenvalue should be much smaller than the others
    sorted_evals = np.sort(eigenvalues)
    if sorted_evals[1] < 1e-10:
        return None
    ratio = sorted_evals[0] / sorted_evals[1]

    if ratio < 0.05:  # strong planar distribution of normals -> cylinder
        axis = eigenvectors[:, 0]  # eigenvector for smallest eigenvalue
        norm = np.linalg.norm(axis)
        if norm > 1e-10:
            axis /= norm
            area = _get_face_area(face)
            if area > 1e-10:
                return (axis, area * 0.7)  # slightly lower confidence than analytic

    return None


def extract_pipe_axes(shape) -> List[Tuple[np.ndarray, float]]:
    """Extract pipe axis directions from all relevant surface types.

    Handles: Cylinder (exact), Cone (exact), Torus (pipe tangent directions),
    and BSpline (cylinder detection via normal PCA).

    Returns:
        List of (unit_direction, weighted_area) tuples.
    """
    axes = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods.Face(explorer.Current())
        try:
            surf = BRepAdaptor_Surface(face)
            surf_type = surf.GetType()

            if surf_type == GeomAbs_Cylinder:
                cyl = surf.Cylinder()
                ax_dir = cyl.Axis().Direction()
                direction = np.array([ax_dir.X(), ax_dir.Y(), ax_dir.Z()])
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction /= norm
                    area = _get_face_area(face)
                    if area > 1e-10:
                        axes.append((direction, area))

            elif surf_type == GeomAbs_Cone:
                cone = surf.Cone()
                ax_dir = cone.Axis().Direction()
                direction = np.array([ax_dir.X(), ax_dir.Y(), ax_dir.Z()])
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction /= norm
                    area = _get_face_area(face)
                    if area > 1e-10:
                        axes.append((direction, area * 0.5))

            elif surf_type == GeomAbs_Torus:
                torus_dirs = _extract_torus_pipe_directions(face)
                axes.extend(torus_dirs)

            elif surf_type == GeomAbs_BSplineSurface:
                result = _detect_bspline_cylinder_axis(face)
                if result is not None:
                    axes.append(result)

            elif surf_type == GeomAbs_SurfaceOfRevolution:
                # Revolution surface has an axis - extract it
                try:
                    rev_surf = surf.AxeOfRevolution()
                    ax_dir = rev_surf.Direction()
                    direction = np.array([ax_dir.X(), ax_dir.Y(), ax_dir.Z()])
                    norm = np.linalg.norm(direction)
                    if norm > 1e-10:
                        direction /= norm
                        area = _get_face_area(face)
                        if area > 1e-10:
                            axes.append((direction, area * 0.6))
                except Exception:
                    pass

        except Exception:
            pass
        explorer.Next()
    return axes


def cluster_axes(axes: List[Tuple[np.ndarray, float]],
                 angle_threshold_deg: float = 15.0) -> List[Dict]:
    """Cluster cylinder axes, treating anti-parallel as equivalent.

    Returns:
        List of dicts {'direction': unit_vec, 'total_area': float},
        sorted by total_area descending.
    """
    if not axes:
        return []

    cos_threshold = np.cos(np.radians(angle_threshold_deg))
    clusters = []

    for direction, area in axes:
        matched = False
        for cluster in clusters:
            ref = cluster['ref_direction']
            cos_angle = abs(np.dot(direction, ref))
            if cos_angle > cos_threshold:
                sign = 1.0 if np.dot(direction, ref) > 0 else -1.0
                cluster['weighted_sum'] += direction * sign * area
                cluster['total_area'] += area
                matched = True
                break
        if not matched:
            clusters.append({
                'ref_direction': direction.copy(),
                'weighted_sum': direction * area,
                'total_area': area,
            })

    result = []
    for c in clusters:
        d = c['weighted_sum']
        norm = np.linalg.norm(d)
        if norm > 1e-10:
            d /= norm
        result.append({'direction': d, 'total_area': c['total_area']})

    result.sort(key=lambda x: -x['total_area'])
    return result


# ============================================================
# PCA Fallback
# ============================================================

def get_surface_points(shape, n_samples: int = 12) -> np.ndarray:
    """Sample points from all BREP faces for PCA."""
    points = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods.Face(explorer.Current())
        try:
            surf = BRepAdaptor_Surface(face)
            u_min = max(surf.FirstUParameter(), -1e4)
            u_max = min(surf.LastUParameter(), 1e4)
            v_min = max(surf.FirstVParameter(), -1e4)
            v_max = min(surf.LastVParameter(), 1e4)
            for i in range(n_samples):
                for j in range(n_samples):
                    u = u_min + (u_max - u_min) * i / max(n_samples - 1, 1)
                    v = v_min + (v_max - v_min) * j / max(n_samples - 1, 1)
                    try:
                        pnt = surf.Value(u, v)
                        points.append([pnt.X(), pnt.Y(), pnt.Z()])
                    except Exception:
                        pass
        except Exception:
            pass
        explorer.Next()
    return np.array(points) if points else np.zeros((0, 3))


def pca_axes(points: np.ndarray) -> np.ndarray:
    """Compute principal axes via eigendecomposition of covariance matrix.

    Returns:
        3x3 matrix where rows are principal axes (descending variance).
    """
    center = points.mean(axis=0)
    centered = points - center
    cov = centered.T @ centered / max(len(centered), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = eigenvalues.argsort()[::-1]
    axes = eigenvectors[:, idx].T
    if np.linalg.det(axes) < 0:
        axes[2] *= -1
    return axes


# ============================================================
# Alignment
# ============================================================

def compute_alignment_rotation(shape) -> Tuple[np.ndarray, str]:
    """Compute rotation matrix to align pipe axes with coordinate axes.

    Returns:
        (R, method) where R is a 3x3 rotation matrix and method is a string
        describing which alignment strategy was used.
    """
    axes = extract_pipe_axes(shape)
    clusters = cluster_axes(axes)

    logger.info(f"  Pipe axis detections: {len(axes)}, unique axes: {len(clusters)}")
    for i, c in enumerate(clusters[:5]):
        d = c['direction']
        logger.info(f"    Axis {i}: dir=({d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f}), "
                     f"area={c['total_area']:.2f}")

    if len(clusters) >= 2:
        d1 = clusters[0]['direction']
        d2 = clusters[1]['direction']

        # Gram-Schmidt orthogonalization
        d2_orth = d2 - np.dot(d2, d1) * d1
        norm_d2 = np.linalg.norm(d2_orth)

        if norm_d2 > 0.1:  # axes sufficiently non-parallel
            d2_orth /= norm_d2
            d3 = np.cross(d1, d2_orth)
            d3 /= np.linalg.norm(d3)

            # R maps: d1->X, d2_orth->Y, d3->Z
            R = np.array([d1, d2_orth, d3])
            if np.linalg.det(R) < 0:
                R[2] *= -1
            return R, "cylinder_2axis"

        logger.info("  Pipe axes nearly parallel, trying with 1 axis")

    if len(clusters) >= 1:
        d1 = clusters[0]['direction']
        # Build orthonormal frame with d1 as Z
        if abs(d1[0]) < 0.9:
            d2 = np.cross(d1, [1, 0, 0])
        else:
            d2 = np.cross(d1, [0, 1, 0])
        d2 /= np.linalg.norm(d2)
        d3 = np.cross(d1, d2)
        d3 /= np.linalg.norm(d3)
        # Map: d2->X, d3->Y, d1->Z
        R = np.array([d2, d3, d1])
        if np.linalg.det(R) < 0:
            R[1] *= -1
        return R, "cylinder_1axis"

    # PCA fallback
    logger.info("  No pipe axes detected, using PCA alignment")
    points = get_surface_points(shape)
    if len(points) < 3:
        return np.eye(3), "identity"
    R = pca_axes(points)
    return R, "pca"


def verify_alignment(aligned_shape) -> Dict:
    """Verify that cylinder axes in the aligned shape are parallel to coordinate axes.

    Returns dict with alignment quality metrics.
    """
    axes = extract_pipe_axes(aligned_shape)
    clusters = cluster_axes(axes, angle_threshold_deg=20.0)

    coord_axes = np.eye(3)
    max_deviations = []

    for c in clusters:
        d = c['direction']
        # Find closest coordinate axis
        dots = np.abs(coord_axes @ d)
        best_axis_idx = np.argmax(dots)
        alignment_cos = dots[best_axis_idx]
        deviation_deg = np.degrees(np.arccos(min(alignment_cos, 1.0)))
        max_deviations.append(deviation_deg)

    return {
        'num_axes': len(clusters),
        'max_deviation_deg': max(max_deviations) if max_deviations else 0.0,
        'deviations_deg': max_deviations,
        'perfect': all(d < 1.0 for d in max_deviations) if max_deviations else True,
    }


def get_shape_center(shape) -> np.ndarray:
    """Get center of mass of the shape."""
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    if abs(props.Mass()) < 1e-10:
        brepgprop.SurfaceProperties(shape, props)
    c = props.CentreOfMass()
    return np.array([c.X(), c.Y(), c.Z()])


def apply_transform(shape, rotation: np.ndarray, center: np.ndarray):
    """Apply centering + rotation to a shape.

    Args:
        shape: TopoDS_Shape
        rotation: 3x3 rotation matrix
        center: 3D point to translate to origin first

    Returns:
        Transformed TopoDS_Shape
    """
    # Translate to origin
    trsf_t = gp_Trsf()
    trsf_t.SetTranslation(gp_Vec(-float(center[0]), -float(center[1]), -float(center[2])))
    shape_centered = BRepBuilderAPI_Transform(shape, trsf_t, True).Shape()

    # Apply rotation via axis-angle
    scipy_rot = ScipyRotation.from_matrix(rotation)
    rotvec = scipy_rot.as_rotvec()
    angle = np.linalg.norm(rotvec)

    if angle > 1e-10:
        axis = rotvec / angle
        trsf_r = gp_Trsf()
        trsf_r.SetRotation(
            gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(float(axis[0]), float(axis[1]), float(axis[2]))),
            float(angle)
        )
        shape_aligned = BRepBuilderAPI_Transform(shape_centered, trsf_r, True).Shape()
    else:
        shape_aligned = shape_centered

    return shape_aligned


def align_shape(shape):
    """Full alignment pipeline: center + rotate to align pipe axes with coord axes.

    Returns:
        (aligned_shape, rotation_matrix, center, method_name)
    """
    center = get_shape_center(shape)
    logger.info(f"  Center of mass: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")

    R, method = compute_alignment_rotation(shape)
    logger.info(f"  Alignment method: {method}")

    aligned = apply_transform(shape, R, center)

    # Verify
    quality = verify_alignment(aligned)
    if quality['num_axes'] > 0:
        logger.info(f"  Alignment quality: max deviation = {quality['max_deviation_deg']:.2f} deg "
                     f"({'PERFECT' if quality['perfect'] else 'GOOD' if quality['max_deviation_deg'] < 5 else 'CHECK'})")
    else:
        logger.info("  Alignment quality: no pipe axes to verify")

    return aligned, R, center, method


# ============================================================
# Tessellation
# ============================================================

def tessellate_shape(shape, deflection: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Tessellate a shape into vertices and triangle indices.

    Args:
        shape: TopoDS_Shape
        deflection: Mesh deflection tolerance (smaller = finer mesh)

    Returns:
        (vertices [Nx3], triangles [Mx3]) as numpy arrays
    """
    mesh = BRepMesh_IncrementalMesh(shape, deflection)
    mesh.Perform()

    all_vertices = []
    all_triangles = []
    vertex_offset = 0

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods.Face(explorer.Current())
        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, loc)

        if triangulation is not None:
            trsf = loc.Transformation()
            n_nodes = triangulation.NbNodes()

            for i in range(1, n_nodes + 1):
                pnt = triangulation.Node(i)
                pnt.Transform(trsf)
                all_vertices.append([pnt.X(), pnt.Y(), pnt.Z()])

            n_tri = triangulation.NbTriangles()
            for i in range(1, n_tri + 1):
                tri = triangulation.Triangle(i)
                n1, n2, n3 = tri.Get()
                if face.Orientation() == TopAbs_REVERSED:
                    all_triangles.append([n2 - 1 + vertex_offset,
                                          n1 - 1 + vertex_offset,
                                          n3 - 1 + vertex_offset])
                else:
                    all_triangles.append([n1 - 1 + vertex_offset,
                                          n2 - 1 + vertex_offset,
                                          n3 - 1 + vertex_offset])

            vertex_offset += n_nodes
        explorer.Next()

    if not all_vertices:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    return np.array(all_vertices, dtype=np.float64), np.array(all_triangles, dtype=int)


def get_bounding_extent(shape) -> float:
    """Get the maximum extent of the shape's bounding box."""
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    return max(xmax - xmin, ymax - ymin, zmax - zmin)


# ============================================================
# 6-View Rendering (VTK offscreen)
# ============================================================

# Camera configs: (name, camera_direction, up_vector)
VIEWS = [
    ("front",  ( 0,  1,  0), (0, 0, 1)),   # +Y looking at origin
    ("back",   ( 0, -1,  0), (0, 0, 1)),   # -Y looking at origin
    ("right",  ( 1,  0,  0), (0, 0, 1)),   # +X looking at origin
    ("left",   (-1,  0,  0), (0, 0, 1)),   # -X looking at origin
    ("top",    ( 0,  0,  1), (0, -1, 0)),  # +Z looking at origin
    ("bottom", ( 0,  0, -1), (0,  1, 0)),  # -Z looking at origin
]


def make_vtk_polydata(vertices: np.ndarray, triangles: np.ndarray):
    """Create a VTK polydata object from vertices and triangles."""
    import vtk

    points = vtk.vtkPoints()
    for v in vertices:
        points.InsertNextPoint(float(v[0]), float(v[1]), float(v[2]))

    cells = vtk.vtkCellArray()
    for t in triangles:
        tri = vtk.vtkTriangle()
        tri.GetPointIds().SetId(0, int(t[0]))
        tri.GetPointIds().SetId(1, int(t[1]))
        tri.GetPointIds().SetId(2, int(t[2]))
        cells.InsertNextCell(tri)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)

    # Compute normals for smooth shading
    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputData(polydata)
    normals_filter.ComputePointNormalsOn()
    normals_filter.SplittingOff()
    normals_filter.Update()

    return normals_filter.GetOutput()


def render_single_view(polydata, cam_dir, cam_up, extent, output_path,
                       size=(800, 800), show_mesh=False):
    """Render one orthographic view and save as PNG."""
    import vtk

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.65, 0.70, 0.82)
    actor.GetProperty().SetAmbient(0.3)
    actor.GetProperty().SetDiffuse(0.6)
    actor.GetProperty().SetSpecular(0.2)
    actor.GetProperty().SetSpecularPower(20)

    if show_mesh:
        actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetEdgeColor(0.2, 0.2, 0.25)
        actor.GetProperty().SetLineWidth(0.5)
    else:
        actor.GetProperty().EdgeVisibilityOff()

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1.0, 1.0, 1.0)

    # Render window (offscreen) - must exist before camera reset
    renWin = vtk.vtkRenderWindow()
    renWin.SetOffScreenRendering(1)
    renWin.AddRenderer(renderer)
    renWin.SetSize(size[0], size[1])

    # Camera setup: orthographic projection
    camera = renderer.GetActiveCamera()
    camera.SetParallelProjection(True)
    dist = extent * 5
    camera.SetPosition(cam_dir[0] * dist, cam_dir[1] * dist, cam_dir[2] * dist)
    camera.SetFocalPoint(0, 0, 0)
    camera.SetViewUp(cam_up[0], cam_up[1], cam_up[2])

    # Let VTK compute proper clipping and zoom, then apply our scale
    renderer.ResetCamera()
    camera.SetParallelScale(extent * 0.6)
    camera.SetClippingRange(extent * 0.01, extent * 20)

    renWin.Render()

    # Save
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(renWin)
    w2if.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputConnection(w2if.GetOutputPort())
    writer.Write()

    renWin.Finalize()


def render_six_views(vertices: np.ndarray, triangles: np.ndarray,
                     output_dir: Path, name: str,
                     size: Tuple[int, int] = (800, 800),
                     show_mesh: bool = False):
    """Render all 6 orthographic views and save as PNGs.

    Returns:
        List of saved file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    polydata = make_vtk_polydata(vertices, triangles)

    # Compute extent from vertices
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    extent = np.max(vmax - vmin)

    saved = []
    for view_name, cam_dir, cam_up in VIEWS:
        out_path = output_dir / f"{name}_{view_name}.png"
        render_single_view(polydata, cam_dir, cam_up, extent, out_path, size,
                           show_mesh=show_mesh)
        saved.append(out_path)
        logger.debug(f"    Saved {out_path}")

    return saved


# ============================================================
# Visualization helpers (optional, for --visualize)
# ============================================================

def save_comparison_grid(image_paths: List[Path], output_path: Path):
    """Create a 2x3 grid of the 6 view images."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(output_path.stem.replace("_grid", ""), fontsize=14, fontweight='bold')

    view_labels = ["Front (+Y)", "Back (-Y)", "Right (+X)", "Left (-X)", "Top (+Z)", "Bottom (-Z)"]
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    for idx, (path, label, (r, c)) in enumerate(zip(image_paths, view_labels, positions)):
        if path.exists():
            img = Image.open(path)
            axes[r][c].imshow(img)
        axes[r][c].set_title(label, fontsize=12)
        axes[r][c].axis('off')

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Grid saved: {output_path}")


def visualize_axes_on_shape(shape_orig, shape_aligned, axes_clusters,
                            rotation, center, output_path: Path):
    """Save a visualization showing the detected axes and alignment result."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(16, 7))

    # Original with detected axes
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Original + Detected Pipe Axes", fontsize=11)

    verts_orig, tris_orig = tessellate_shape(shape_orig, deflection=0.2)
    if len(verts_orig) > 0:
        ax1.plot_trisurf(verts_orig[:, 0], verts_orig[:, 1], verts_orig[:, 2],
                         triangles=tris_orig, color='lightsteelblue', edgecolor='gray',
                         linewidth=0.1, alpha=0.4)

        # Draw detected axes
        extent = np.max(verts_orig.max(axis=0) - verts_orig.min(axis=0))
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        for i, c in enumerate(axes_clusters[:3]):
            d = c['direction']
            scale = extent * 0.7
            ax1.quiver(center[0], center[1], center[2],
                       d[0]*scale, d[1]*scale, d[2]*scale,
                       color=colors[i % len(colors)], linewidth=3,
                       arrow_length_ratio=0.1,
                       label=f"Axis {i} (area={c['total_area']:.1f})")
        ax1.legend(fontsize=8)
        _set_equal_axes(ax1, verts_orig)

    # Aligned shape
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("Aligned (axes -> XYZ)", fontsize=11)

    verts_aligned, tris_aligned = tessellate_shape(shape_aligned, deflection=0.2)
    if len(verts_aligned) > 0:
        ax2.plot_trisurf(verts_aligned[:, 0], verts_aligned[:, 1], verts_aligned[:, 2],
                         triangles=tris_aligned, color='lightsteelblue', edgecolor='gray',
                         linewidth=0.1, alpha=0.4)

        # Draw coordinate axes
        ext = np.max(verts_aligned.max(axis=0) - verts_aligned.min(axis=0))
        for axis_vec, color, label in [([1,0,0], 'red', 'X'),
                                        ([0,1,0], 'green', 'Y'),
                                        ([0,0,1], 'blue', 'Z')]:
            ax2.quiver(0, 0, 0, axis_vec[0]*ext*0.6, axis_vec[1]*ext*0.6,
                       axis_vec[2]*ext*0.6, color=color, linewidth=2,
                       arrow_length_ratio=0.1, label=label)
        ax2.legend(fontsize=8)
        _set_equal_axes(ax2, verts_aligned)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Axis visualization saved: {output_path}")


def _set_equal_axes(ax, vertices):
    """Set equal aspect ratio for a 3D matplotlib axes."""
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = (vmin + vmax) / 2
    half_range = np.max(vmax - vmin) / 2 * 1.2
    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(center[2] - half_range, center[2] + half_range)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def visualize_vtk_interactive(vertices, triangles, title="Shape"):
    """Show an interactive VTK 3D window (requires display)."""
    import vtk

    polydata = make_vtk_polydata(vertices, triangles)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.65, 0.70, 0.82)
    actor.GetProperty().SetAmbient(0.3)
    actor.GetProperty().SetDiffuse(0.6)
    actor.GetProperty().SetSpecular(0.2)
    actor.GetProperty().EdgeVisibilityOn()
    actor.GetProperty().SetEdgeColor(0.2, 0.2, 0.25)

    # Coordinate axes
    axes_actor = vtk.vtkAxesActor()
    ext = np.max(vertices.max(axis=0) - vertices.min(axis=0))
    axes_actor.SetTotalLength(ext * 0.5, ext * 0.5, ext * 0.5)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.AddActor(axes_actor)
    renderer.SetBackground(0.95, 0.95, 0.98)

    renWin = vtk.vtkRenderWindow()
    renWin.SetWindowName(title)
    renWin.AddRenderer(renderer)
    renWin.SetSize(1000, 800)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)

    renderer.ResetCamera()
    renWin.Render()
    iren.Start()


# ============================================================
# Main Pipeline
# ============================================================

def process_single(step_path: str, output_dir: Path,
                   visualize: bool = False, interactive: bool = False,
                   image_size: Tuple[int, int] = (800, 800),
                   show_mesh: bool = False, simple: bool = False) -> bool:
    """Process a single STEP file: load -> align -> render 6 views.

    Returns True on success.
    """
    name = Path(step_path).stem
    model_output = output_dir / name
    logger.info(f"Processing: {Path(step_path).name}")

    # 1. Load
    try:
        shape = load_step(step_path)
    except Exception as e:
        logger.error(f"  Failed to load: {e}")
        return False

    # 2. Align
    try:
        axes_raw = extract_pipe_axes(shape)
        clusters = cluster_axes(axes_raw)
        center = get_shape_center(shape)
        aligned_shape, R, center, method = align_shape(shape)
    except Exception as e:
        logger.error(f"  Failed to align: {e}")
        return False

    # 3. Visualize alignment (optional, skip in simple mode)
    if visualize and not simple:
        viz_dir = model_output / "debug"
        viz_dir.mkdir(parents=True, exist_ok=True)
        try:
            visualize_axes_on_shape(shape, aligned_shape, clusters, R, center,
                                    viz_dir / f"{name}_alignment.png")
        except Exception as e:
            logger.warning(f"  Visualization failed: {e}")

    # 4. Tessellate
    try:
        vertices, triangles = tessellate_shape(aligned_shape)
        logger.info(f"  Tessellated: {len(vertices)} vertices, {len(triangles)} triangles")
    except Exception as e:
        logger.error(f"  Tessellation failed: {e}")
        return False

    if len(vertices) == 0 or len(triangles) == 0:
        logger.error("  Empty mesh after tessellation")
        return False

    # 5. Interactive visualization (optional, skip in simple mode)
    if interactive and not simple:
        try:
            visualize_vtk_interactive(vertices, triangles,
                                       title=f"Aligned: {name} ({method})")
        except Exception as e:
            logger.warning(f"  Interactive visualization failed: {e}")

    # 6. Render 6 views
    try:
        if simple:
            views_dir = model_output
        else:
            views_dir = model_output / "views"
        saved = render_six_views(vertices, triangles, views_dir, name,
                                 size=image_size, show_mesh=show_mesh)
        logger.info(f"  Rendered {len(saved)} views -> {views_dir}")
    except Exception as e:
        logger.error(f"  Rendering failed: {e}")
        return False

    # 7. Create comparison grid (optional, skip in simple mode)
    if visualize and not simple:
        try:
            save_comparison_grid(saved, model_output / f"{name}_grid.png")
        except Exception as e:
            logger.warning(f"  Grid creation failed: {e}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Align pipe fittings and render 6 orthographic views",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python pipeline.py --step data/ssdata1/steps/some_file.step -o output/
  python pipeline.py --step_dir data/ssdata1/steps/ -o output/ --limit 5
  python pipeline.py --step_dir data/ssdata1/1_elbow_wf/ -o output/elbows/ --visualize
  python pipeline.py --step data/ssdata1/steps/some_file.step -o output/ --simple
  python pipeline.py --step data/ssdata1/steps/some_file.step -o output/ --mesh
""")
    parser.add_argument("--step", type=str, help="Single STEP file path")
    parser.add_argument("--step_dir", type=str, help="Directory of STEP files for batch mode")
    parser.add_argument("-o", "--output", type=str, default="output",
                        help="Output directory (default: output/)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max files to process in batch mode (0=all)")
    parser.add_argument("--visualize", action="store_true",
                        help="Save alignment debug visualizations at each step")
    parser.add_argument("--interactive", action="store_true",
                        help="Show interactive 3D viewer after alignment (requires display)")
    parser.add_argument("--mesh", action="store_true",
                        help="Show internal triangular mesh edges in rendered views")
    parser.add_argument("--simple", action="store_true",
                        help="Simple output: just 6 view PNGs per model, no subdirectories or extras")
    parser.add_argument("--image_size", type=int, nargs=2, default=[800, 800],
                        metavar=("W", "H"), help="Output image size (default: 800 800)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_size = tuple(args.image_size)

    if args.step:
        success = process_single(args.step, output_dir,
                                  visualize=args.visualize,
                                  interactive=args.interactive,
                                  image_size=image_size,
                                  show_mesh=args.mesh,
                                  simple=args.simple)
        sys.exit(0 if success else 1)

    elif args.step_dir:
        step_dir = Path(args.step_dir)
        files = sorted(list(step_dir.glob("*.step")) + list(step_dir.glob("*.stp")))
        if args.limit > 0:
            files = files[:args.limit]
        logger.info(f"Batch mode: {len(files)} files from {step_dir}")

        successes = 0
        failures = 0
        for f in files:
            ok = process_single(str(f), output_dir,
                                 visualize=args.visualize,
                                 interactive=args.interactive,
                                 image_size=image_size,
                                 show_mesh=args.mesh,
                                 simple=args.simple)
            if ok:
                successes += 1
            else:
                failures += 1

        logger.info(f"\nDone: {successes} succeeded, {failures} failed out of {len(files)}")
        sys.exit(0 if failures == 0 else 1)

    else:
        parser.error("Provide --step or --step_dir")


if __name__ == "__main__":
    main()
