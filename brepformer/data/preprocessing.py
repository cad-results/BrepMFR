"""Preprocessing utilities for BrepFormer.

Functions for computing graph structure features:
- Shortest paths between nodes
- In-degree computation
- D2 shape descriptor (distance histogram)
- Angle descriptor (angle histogram)
"""

from collections import deque
from typing import Tuple, Optional

import numpy as np
import torch


def compute_shortest_paths(
    edge_index: np.ndarray,
    num_nodes: int,
    max_dist: int = 64,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Compute shortest path distances between all pairs of nodes using BFS.

    Args:
        edge_index: Edge indices of shape (2, num_edges).
        num_nodes: Number of nodes in the graph.
        max_dist: Maximum distance to compute (for clipping).

    Returns:
        Tuple of:
        - spatial_pos: Shortest path distances of shape (num_nodes + 1, num_nodes + 1).
                       Position 0 is reserved for virtual [CLS] token.
        - edge_path: Edge indices along shortest paths (optional, not implemented).
    """
    # Build adjacency list
    adj = [[] for _ in range(num_nodes)]
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        adj[src].append(dst)
        adj[dst].append(src)  # Undirected

    # Initialize distance matrix
    # +1 for virtual [CLS] token at position 0
    spatial_pos = np.full((num_nodes + 1, num_nodes + 1), max_dist, dtype=np.int64)

    # [CLS] token connected to all nodes with distance 1
    spatial_pos[0, :] = 1
    spatial_pos[:, 0] = 1
    spatial_pos[0, 0] = 0

    # BFS from each node
    for start in range(num_nodes):
        dist = np.full(num_nodes, -1, dtype=np.int64)
        dist[start] = 0
        queue = deque([start])

        while queue:
            node = queue.popleft()
            if dist[node] >= max_dist:
                continue

            for neighbor in adj[node]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[node] + 1
                    queue.append(neighbor)

        # Fill spatial_pos matrix (offset by 1 for [CLS] token)
        for end in range(num_nodes):
            if dist[end] >= 0:
                spatial_pos[start + 1, end + 1] = min(dist[end], max_dist)

    return spatial_pos, None


def compute_in_degree(
    edge_index: np.ndarray,
    num_nodes: int,
) -> np.ndarray:
    """Compute in-degree of each node.

    Args:
        edge_index: Edge indices of shape (2, num_edges).
        num_nodes: Number of nodes in the graph.

    Returns:
        In-degree of each node of shape (num_nodes,).
    """
    in_degree = np.zeros(num_nodes, dtype=np.int64)
    for i in range(edge_index.shape[1]):
        dst = edge_index[1, i]
        in_degree[dst] += 1
    return in_degree


def compute_d2_descriptor(
    face_centroids: np.ndarray,
    num_bins: int = 64,
    max_dist: Optional[float] = None,
    face_grids: Optional[np.ndarray] = None,
    num_samples: int = 32,
) -> np.ndarray:
    """Compute D2 shape descriptor (pairwise distance histogram).

    D2 descriptor measures the distribution of distances between pairs of points
    on the surface, providing a rotation-invariant shape signature.

    When face_grids is provided, samples random point pairs from the UV grids
    of both faces to build a proper distribution histogram. Otherwise, uses
    Gaussian-kernel smoothing around the centroid distance.

    Args:
        face_centroids: Centroid coordinates of shape (num_faces, 3).
        num_bins: Number of histogram bins.
        max_dist: Maximum distance for histogram (auto-computed if None).
        face_grids: Optional UV-grid tensor of shape (num_faces, 7, H, W).
        num_samples: Number of random point pairs to sample per face pair.

    Returns:
        D2 distance descriptors of shape (num_faces + 1, num_faces + 1, num_bins).
        Position 0 is for virtual [CLS] token.
    """
    num_faces = len(face_centroids)

    # Extract point clouds from UV-grids if available
    face_points = None
    if face_grids is not None:
        face_points = []
        for i in range(num_faces):
            mask = face_grids[i, 6, :, :] > 0.5  # validity mask
            pts = face_grids[i, 0:3, :, :][:, mask].T  # (n_valid, 3)
            face_points.append(pts if len(pts) > 0 else face_centroids[i:i+1])

    # Compute pairwise centroid distances for max_dist estimation
    diff = face_centroids[:, None, :] - face_centroids[None, :, :]
    centroid_distances = np.sqrt((diff ** 2).sum(axis=-1))

    if max_dist is None:
        max_dist = centroid_distances.max() + 1e-6

    d2_desc = np.zeros((num_faces + 1, num_faces + 1, num_bins), dtype=np.float32)
    bin_centers = (np.arange(num_bins) + 0.5) / num_bins * max_dist

    if face_points is not None:
        # Multi-sample histogram from UV-grid points
        rng = np.random.RandomState(42)
        for i in range(num_faces):
            for j in range(num_faces):
                pts_i = face_points[i]
                pts_j = face_points[j]
                idx_i = rng.randint(0, len(pts_i), size=num_samples)
                idx_j = rng.randint(0, len(pts_j), size=num_samples)
                dists = np.sqrt(((pts_i[idx_i] - pts_j[idx_j]) ** 2).sum(axis=-1))
                hist, _ = np.histogram(dists, bins=num_bins, range=(0, max_dist))
                total = hist.sum()
                if total > 0:
                    d2_desc[i + 1, j + 1, :] = hist.astype(np.float32) / total
    else:
        # Gaussian-kernel smoothing around centroid distance
        sigma = max_dist / num_bins * 2.0
        for i in range(num_faces):
            for j in range(num_faces):
                d = centroid_distances[i, j]
                kernel = np.exp(-0.5 * ((bin_centers - d) / sigma) ** 2)
                total = kernel.sum()
                if total > 0:
                    d2_desc[i + 1, j + 1, :] = kernel / total

    # [CLS] token gets uniform distribution
    d2_desc[0, :, :] = 1.0 / num_bins
    d2_desc[:, 0, :] = 1.0 / num_bins

    return d2_desc


def compute_angle_descriptor(
    face_normals: np.ndarray,
    num_bins: int = 64,
    face_grids: Optional[np.ndarray] = None,
    num_samples: int = 32,
) -> np.ndarray:
    """Compute angle descriptor (pairwise normal angle histogram).

    Measures the distribution of angles between face normal vectors,
    providing orientation relationship information.

    When face_grids is provided, samples random normal pairs from the UV grids
    to build a distribution. Otherwise, uses Gaussian-kernel smoothing around
    the centroid normal angle.

    Args:
        face_normals: Normal vectors of shape (num_faces, 3).
        num_bins: Number of histogram bins.
        face_grids: Optional UV-grid tensor of shape (num_faces, 7, H, W).
        num_samples: Number of random normal pairs to sample per face pair.

    Returns:
        Angle descriptors of shape (num_faces + 1, num_faces + 1, num_bins).
        Position 0 is for virtual [CLS] token.
    """
    num_faces = len(face_normals)

    # Extract per-point normals from UV-grids if available
    face_normal_pts = None
    if face_grids is not None:
        face_normal_pts = []
        for i in range(num_faces):
            mask = face_grids[i, 6, :, :] > 0.5
            normals = face_grids[i, 3:6, :, :][:, mask].T  # (n_valid, 3)
            nrm = np.linalg.norm(normals, axis=-1, keepdims=True)
            normals = normals / (nrm + 1e-8)
            face_normal_pts.append(normals if len(normals) > 0 else face_normals[i:i+1])

    # Normalize mean normals
    norms = np.linalg.norm(face_normals, axis=-1, keepdims=True)
    face_normals_n = face_normals / (norms + 1e-8)

    # Compute pairwise mean-normal angles
    dot_products = np.einsum("ik,jk->ij", face_normals_n, face_normals_n)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    mean_angles = np.arccos(dot_products)  # (num_faces, num_faces)

    angle_desc = np.zeros((num_faces + 1, num_faces + 1, num_bins), dtype=np.float32)
    bin_centers = (np.arange(num_bins) + 0.5) / num_bins * np.pi

    if face_normal_pts is not None:
        # Multi-sample histogram from UV-grid normals
        rng = np.random.RandomState(42)
        for i in range(num_faces):
            for j in range(num_faces):
                ni = face_normal_pts[i]
                nj = face_normal_pts[j]
                idx_i = rng.randint(0, len(ni), size=num_samples)
                idx_j = rng.randint(0, len(nj), size=num_samples)
                dots = np.clip((ni[idx_i] * nj[idx_j]).sum(axis=-1), -1.0, 1.0)
                angs = np.arccos(dots)
                hist, _ = np.histogram(angs, bins=num_bins, range=(0, np.pi))
                total = hist.sum()
                if total > 0:
                    angle_desc[i + 1, j + 1, :] = hist.astype(np.float32) / total
    else:
        # Gaussian-kernel smoothing around mean-normal angle
        sigma = np.pi / num_bins * 2.0
        for i in range(num_faces):
            for j in range(num_faces):
                a = mean_angles[i, j]
                kernel = np.exp(-0.5 * ((bin_centers - a) / sigma) ** 2)
                total = kernel.sum()
                if total > 0:
                    angle_desc[i + 1, j + 1, :] = kernel / total

    # [CLS] token gets uniform distribution
    angle_desc[0, :, :] = 1.0 / num_bins
    angle_desc[:, 0, :] = 1.0 / num_bins

    return angle_desc


def precompute_graph_features(
    edge_index: np.ndarray,
    num_nodes: int,
    face_centroids: Optional[np.ndarray] = None,
    face_normals: Optional[np.ndarray] = None,
    face_grids: Optional[np.ndarray] = None,
    num_spatial: int = 64,
    d2_bins: int = 64,
    angle_bins: int = 64,
) -> dict:
    """Precompute all graph structure features.

    Args:
        edge_index: Edge indices of shape (2, num_edges).
        num_nodes: Number of nodes (faces).
        face_centroids: Face centroid coordinates.
        face_normals: Face normal vectors.
        face_grids: Optional UV-grid tensor of shape (num_faces, 7, H, W)
                    for multi-sample D2/angle descriptors.
        num_spatial: Maximum spatial distance.
        d2_bins: Number of D2 histogram bins.
        angle_bins: Number of angle histogram bins.

    Returns:
        Dictionary with precomputed features:
        - spatial_pos: Shortest path distances
        - in_degree: Node in-degrees
        - d2_distance: D2 shape descriptors (optional)
        - angle_distance: Angle descriptors (optional)
    """
    result = {}

    # Shortest paths
    spatial_pos, edge_path = compute_shortest_paths(
        edge_index, num_nodes, max_dist=num_spatial
    )
    result["spatial_pos"] = spatial_pos
    if edge_path is not None:
        result["edge_path"] = edge_path

    # In-degree
    result["in_degree"] = compute_in_degree(edge_index, num_nodes)

    # D2 descriptor
    if face_centroids is not None:
        result["d2_distance"] = compute_d2_descriptor(
            face_centroids, num_bins=d2_bins, face_grids=face_grids,
        )

    # Angle descriptor
    if face_normals is not None:
        result["angle_distance"] = compute_angle_descriptor(
            face_normals, num_bins=angle_bins, face_grids=face_grids,
        )

    return result
