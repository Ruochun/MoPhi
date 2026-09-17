"""Deterministic volume sampling between opposing triangle-mesh surfaces."""

import numpy as np


def _validated_geometry(vertices, triangles):
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices must be a finite (N, 3) array")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
        raise ValueError("triangles must be a non-empty (M, 3) array")
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError("triangles contain an out-of-range vertex index")
    triangle_vertices = vertices[triangles]
    doubled_areas = np.linalg.norm(
        np.cross(triangle_vertices[:, 1] - triangle_vertices[:, 0], triangle_vertices[:, 2] - triangle_vertices[:, 0]),
        axis=1,
    )
    if np.any(doubled_areas <= 1.0e-12):
        raise ValueError("triangles contain degenerate geometry")
    return vertices, triangles


def _minimum_triangle_distance(point, triangle_vertices):
    """Return the distance from one point to a collection of triangles."""
    a = triangle_vertices[:, 0]
    b = triangle_vertices[:, 1]
    c = triangle_vertices[:, 2]
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    closest = np.empty_like(a)
    assigned = np.zeros(len(a), dtype=bool)

    mask = (d1 <= 0.0) & (d2 <= 0.0)
    closest[mask] = a[mask]
    assigned |= mask

    bp = point - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    mask = (~assigned) & (d3 >= 0.0) & (d4 <= d3)
    closest[mask] = b[mask]
    assigned |= mask

    vc = d1 * d4 - d3 * d2
    mask = (~assigned) & (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    interpolation = d1[mask] / (d1[mask] - d3[mask])
    closest[mask] = a[mask] + interpolation[:, None] * ab[mask]
    assigned |= mask

    cp = point - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    mask = (~assigned) & (d6 >= 0.0) & (d5 <= d6)
    closest[mask] = c[mask]
    assigned |= mask

    vb = d5 * d2 - d1 * d6
    mask = (~assigned) & (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    interpolation = d2[mask] / (d2[mask] - d6[mask])
    closest[mask] = a[mask] + interpolation[:, None] * ac[mask]
    assigned |= mask

    va = d3 * d6 - d5 * d4
    mask = (~assigned) & (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    interpolation = (d4[mask] - d3[mask]) / ((d4[mask] - d3[mask]) + (d5[mask] - d6[mask]))
    closest[mask] = b[mask] + interpolation[:, None] * (c[mask] - b[mask])
    assigned |= mask

    mask = ~assigned
    denominator = 1.0 / (va[mask] + vb[mask] + vc[mask])
    v = vb[mask] * denominator
    w = vc[mask] * denominator
    closest[mask] = a[mask] + v[:, None] * ab[mask] + w[:, None] * ac[mask]
    return float(np.sqrt(np.min(np.einsum("ij,ij->i", point - closest, point - closest))))


def sample_grid_between_mesh_surfaces(
    vertices,
    triangles,
    spacing: float,
    bounds_min,
    bounds_max,
    *,
    ray_axis: int = 0,
    clearance: float = 0.0,
):
    """Sample grid points bracketed by an open mesh along one coordinate axis.

    The explicit bounds close otherwise-open ends. A point is retained when the
    mesh has an intersection on both sides along ``ray_axis`` and its Euclidean
    distance from every triangle is at least ``clearance``.

    ``ray_axis`` is expressed in the input mesh's coordinate frame: 0, 1, and 2
    select its X, Y, and Z axes. Choose an axis that crosses two opposing mesh
    walls; the other two axes are limited only by ``bounds_min`` and
    ``bounds_max``. The returned points remain in that same frame and use the
    same units as the vertices, spacing, bounds, and clearance.
    """
    vertices, triangles = _validated_geometry(vertices, triangles)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing must be finite and positive")
    if ray_axis not in (0, 1, 2):
        raise ValueError("ray_axis must be 0, 1, or 2")
    if not np.isfinite(clearance) or clearance < 0.0:
        raise ValueError("clearance must be finite and non-negative")
    bounds_min = np.asarray(bounds_min, dtype=np.float64)
    bounds_max = np.asarray(bounds_max, dtype=np.float64)
    if bounds_min.shape != (3,) or bounds_max.shape != (3,) or not np.isfinite((bounds_min, bounds_max)).all():
        raise ValueError("bounds_min and bounds_max must be finite three-vectors")
    if np.any(bounds_max <= bounds_min) or np.any(bounds_max - bounds_min < 2.0 * clearance):
        raise ValueError("bounds must enclose a positive region after applying clearance")

    # Project onto the plane normal to ray_axis, then reconstruct every mesh
    # crossing along the omitted coordinate for each candidate grid point.
    axes = [axis for axis in range(3) if axis != ray_axis]
    triangle_vertices = vertices[triangles]
    projected = triangle_vertices[:, :, axes]
    edge_0 = projected[:, 1] - projected[:, 0]
    edge_1 = projected[:, 2] - projected[:, 0]
    determinant = edge_0[:, 0] * edge_1[:, 1] - edge_0[:, 1] * edge_1[:, 0]
    usable = np.abs(determinant) > 1.0e-14

    grid_axes = [
        np.arange(bounds_min[axis] + clearance, bounds_max[axis] - clearance + 1.0e-12, spacing) for axis in range(3)
    ]
    candidates = np.stack(np.meshgrid(*grid_axes, indexing="ij"), axis=-1).reshape((-1, 3))
    accepted = []
    for point in candidates:
        offset = point[axes] - projected[:, 0]
        barycentric_1 = np.zeros(len(triangles), dtype=np.float64)
        barycentric_2 = np.zeros(len(triangles), dtype=np.float64)
        barycentric_1[usable] = (
            offset[usable, 0] * edge_1[usable, 1] - offset[usable, 1] * edge_1[usable, 0]
        ) / determinant[usable]
        barycentric_2[usable] = (
            edge_0[usable, 0] * offset[usable, 1] - edge_0[usable, 1] * offset[usable, 0]
        ) / determinant[usable]
        intersects = usable & (barycentric_1 >= -1.0e-10) & (barycentric_2 >= -1.0e-10)
        intersects &= barycentric_1 + barycentric_2 <= 1.0 + 1.0e-10
        crossing = (
            triangle_vertices[:, 0, ray_axis]
            + barycentric_1 * (triangle_vertices[:, 1, ray_axis] - triangle_vertices[:, 0, ray_axis])
            + barycentric_2 * (triangle_vertices[:, 2, ray_axis] - triangle_vertices[:, 0, ray_axis])
        )
        if not np.any(intersects & (crossing < point[ray_axis])):
            continue
        if not np.any(intersects & (crossing > point[ray_axis])):
            continue
        if clearance and _minimum_triangle_distance(point, triangle_vertices) < clearance:
            continue
        accepted.append(point)
    return np.asarray(accepted, dtype=np.float64).reshape((-1, 3))


__all__ = ["sample_grid_between_mesh_surfaces"]
