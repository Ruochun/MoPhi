"""Solver-independent connected contact-patch partitioning of triangle meshes."""

from collections import deque

import numpy as np


def partition_mesh_contact_patches(vertices, triangle_indices, radius, max_normal_angle_deg=45.0):
    """Return deterministic contiguous int32 patch IDs without changing topology.

    Grow edge-connected regions from ascending face seeds. Each added triangle
    must fit inside the seed-centroid radius and face within the normal-angle
    bound of the seed. Oversized seed triangles remain singleton patches: this
    routine never remeshes. Boundaries/nonmanifold edges are not crossed.
    Compute once on a reference shape and preserve these IDs during deformation;
    size/normal bounds are guaranteed only on that reference geometry.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(triangle_indices)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Require finite (N, 3) vertices")
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or len(faces) == 0
        or faces.dtype.kind not in "iu"
        or np.any(faces < 0)
        or np.any(faces >= len(vertices))
    ):
        raise ValueError("Require nonempty valid integer triangle indices")
    if not np.isfinite([radius, max_normal_angle_deg]).all() or radius <= 0 or not 0 <= max_normal_angle_deg < 90:
        raise ValueError("Require positive radius and normal angle in [0, 90) degrees")
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths == 0):
        raise ValueError("Cannot partition degenerate triangles")
    normals /= lengths[:, None]
    centers = triangles.mean(axis=1)
    edges = {}
    for face, tri in enumerate(faces):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edges.setdefault(tuple(sorted((int(a), int(b)))), []).append(face)
    adjacency = [[] for _ in faces]
    for pair in edges.values():
        if len(pair) == 2:
            a, b = pair
            adjacency[a].append(b)
            adjacency[b].append(a)
    ids = np.full(len(faces), -1, dtype=np.int32)
    threshold = np.cos(np.radians(max_normal_angle_deg))
    patch = 0
    for seed in range(len(faces)):
        if ids[seed] >= 0:
            continue
        ids[seed] = patch
        # A seed exceeding the radius must not absorb other faces.
        if np.max(np.linalg.norm(triangles[seed] - centers[seed], axis=1)) <= radius:
            queue = deque([seed])
            visited = {seed}
            while queue:
                face = queue.popleft()
                for neighbor in sorted(adjacency[face]):
                    if neighbor in visited or ids[neighbor] >= 0:
                        continue
                    visited.add(neighbor)
                    if normals[neighbor] @ normals[seed] < threshold - 1e-12:
                        continue
                    if np.max(np.linalg.norm(triangles[neighbor] - centers[seed], axis=1)) > radius:
                        continue
                    ids[neighbor] = patch
                    queue.append(neighbor)
        patch += 1
    return ids
