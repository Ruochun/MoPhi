"""Visualization utilities shared by MoPhi visualizer backends and demos."""

from __future__ import annotations

import numpy as np


def _get_newton_viewer(visualizer):
    """Return a Newton viewer from either a raw viewer or MoPhi OpenGL wrapper."""
    return getattr(visualizer, "_viewer", visualizer)


def _set_newton_sky_colors(visualizer, upper, lower) -> None:
    viewer = _get_newton_viewer(visualizer)
    if hasattr(viewer, "renderer"):
        viewer.renderer.sky_upper = tuple(upper)
        viewer.renderer.sky_lower = tuple(lower)
        viewer.renderer.background_color = tuple(upper)


def _log_box_instances(visualizer, name: str, centers, half_extents, colors) -> None:
    """Log non-physical decorative boxes through Newton's instanced renderer."""
    import newton
    import warp as wp

    viewer = _get_newton_viewer(visualizer)
    if not hasattr(viewer, "log_geo") or not hasattr(viewer, "log_instances"):
        return

    mesh_name = f"{name}_box_mesh"
    viewer.log_geo(mesh_name, newton.GeoType.BOX, (1.0, 1.0, 1.0), 0.0, True, hidden=True)
    centers_np = np.asarray(centers, dtype=np.float32)
    half_extents_np = np.asarray(half_extents, dtype=np.float32)
    colors_np = np.asarray(colors, dtype=np.float32)
    xforms = [wp.transform(wp.vec3(*center), wp.quat_identity()) for center in centers_np]
    viewer.log_instances(
        name,
        mesh_name,
        wp.array(xforms, dtype=wp.transform),
        wp.array(half_extents_np, dtype=wp.vec3),
        wp.array(colors_np, dtype=wp.vec3),
        None,
    )


def log_underwater_environment(
    visualizer,
    *,
    domain_min=(-2.0, -2.0, 0.0),
    domain_max=(2.0, 4.0, 3.0),
    name_prefix: str = "underwater_environment",
) -> None:
    """Log a decorative sandbed, rocks, vegetation, and underwater sky."""
    import warp as wp

    domain_min_np = np.asarray(domain_min, dtype=np.float32)
    domain_max_np = np.asarray(domain_max, dtype=np.float32)
    center = 0.5 * (domain_min_np + domain_max_np)
    size = domain_max_np - domain_min_np
    sandbed_half_height = 0.04
    sandbed_center = np.array([center[0], center[1], -sandbed_half_height], dtype=np.float32)
    _set_newton_sky_colors(visualizer, (0.015, 0.16, 0.24), (0.01, 0.06, 0.09))
    _log_box_instances(
        visualizer,
        f"{name_prefix}_sandbed",
        [sandbed_center],
        [[0.5 * size[0], 0.5 * size[1], sandbed_half_height]],
        [[0.58, 0.48, 0.28]],
    )

    rng = np.random.default_rng(7)
    rock_positions = np.column_stack(
        (
            rng.uniform(domain_min_np[0] + 0.25, domain_max_np[0] - 0.25, 30),
            rng.uniform(domain_min_np[1] + 0.25, domain_max_np[1] - 0.25, 30),
            rng.uniform(0.06, 0.14, 30),
        )
    ).astype(np.float32)
    rock_radii = rng.uniform(0.07, 0.20, 30).astype(np.float32)
    rock_colors = rng.uniform((0.18, 0.20, 0.16), (0.42, 0.38, 0.27), (30, 3)).astype(np.float32)
    visualizer.log_points(
        f"{name_prefix}_rocks",
        wp.array(rock_positions, dtype=wp.vec3),
        radii=wp.array(rock_radii, dtype=wp.float32),
        colors=wp.array(rock_colors, dtype=wp.vec3),
    )

    plant_roots = np.column_stack(
        (
            rng.uniform(domain_min_np[0] + 0.2, domain_max_np[0] - 0.2, 45),
            rng.uniform(domain_min_np[1] + 0.2, domain_max_np[1] - 0.2, 45),
            np.full(45, 0.04),
        )
    ).astype(np.float32)
    plant_heights = rng.uniform(0.25, 0.75, 45).astype(np.float32)
    plant_tips = plant_roots.copy()
    plant_tips[:, 0] += rng.uniform(-0.12, 0.12, 45)
    plant_tips[:, 2] += plant_heights
    plant_colors = rng.uniform((0.03, 0.28, 0.12), (0.10, 0.62, 0.25), (45, 3)).astype(np.float32)
    visualizer.log_lines(
        f"{name_prefix}_vegetation",
        wp.array(plant_roots, dtype=wp.vec3),
        wp.array(plant_tips, dtype=wp.vec3),
        wp.array(plant_colors, dtype=wp.vec3),
    )


def log_warehouse_environment(
    visualizer,
    *,
    aisle_length: float = 8.0,
    aisle_half_width: float = 1.7,
    name_prefix: str = "warehouse_environment",
) -> None:
    """Log decorative warehouse racks, pallets, crates, walls, and floor grid."""
    import warp as wp

    _set_newton_sky_colors(visualizer, (0.12, 0.14, 0.18), (0.06, 0.07, 0.08))
    box_centers = []
    box_extents = []
    box_colors = []
    for side in (-1.0, 1.0):
        x = side * (aisle_half_width + 0.35)
        for y in np.linspace(0.5, aisle_length - 0.5, 5):
            for z in (0.45, 1.35, 2.25):
                box_centers.append((x, float(y), z))
                box_extents.append((0.45, 0.65, 0.06))
                box_colors.append((0.16, 0.24, 0.34))
            for z in (0.45, 1.35):
                box_centers.append((x, float(y), z + 0.30))
                box_extents.append((0.32, 0.42, 0.24))
                box_colors.append((0.56, 0.31, 0.12))
        for y in (0.0, aisle_length):
            box_centers.append((x, y, 1.35))
            box_extents.append((0.07, 0.07, 1.35))
            box_colors.append((0.12, 0.18, 0.26))
    # Keep a far wall for depth while leaving the negative-Y entrance open to
    # the default camera.
    box_centers.append((0.0, aisle_length + 0.8, 1.5))
    box_extents.append((4.0, 0.08, 1.5))
    box_colors.append((0.32, 0.34, 0.36))
    _log_box_instances(visualizer, f"{name_prefix}_fixtures", box_centers, box_extents, box_colors)

    grid_x = np.arange(-3.5, 3.51, 0.5, dtype=np.float32)
    grid_y = np.arange(-1.0, aisle_length + 0.01, 0.5, dtype=np.float32)
    starts = []
    ends = []
    for x in grid_x:
        starts.append((x, -1.0, 0.006))
        ends.append((x, aisle_length, 0.006))
    for y in grid_y:
        starts.append((-3.5, y, 0.006))
        ends.append((3.5, y, 0.006))
    colors = np.tile(np.array([[0.24, 0.26, 0.28]], dtype=np.float32), (len(starts), 1))
    visualizer.log_lines(
        f"{name_prefix}_floor_grid",
        wp.array(starts, dtype=wp.vec3),
        wp.array(ends, dtype=wp.vec3),
        wp.array(colors, dtype=wp.vec3),
    )


def log_orientation_and_scale_reference(
    visualizer,
    *,
    axis_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    axis_length: float = 0.5,
    scale_bar_center: tuple[float, float, float] = (0.8, 0.0, 0.0),
    reference_length: float = 1.0,
    marker_radius: float = 0.04,
    name_prefix: str = "world_reference",
) -> None:
    """Log persistent XYZ axes and a metric scale bar into a visualizer.

    The helper accepts either a MoPhi visualizer or a compatible Newton viewer
    exposing ``log_lines`` and ``log_points``. When ``log_arrows`` is
    unavailable, axis arrows fall back to colored line segments. X, Y, and Z
    use the conventional red, green, and blue colors; the yellow bar spans
    ``reference_length`` metres along world X.
    """
    if axis_length <= 0.0:
        raise ValueError(f"axis_length must be positive, got {axis_length}.")
    if reference_length <= 0.0:
        raise ValueError(f"reference_length must be positive, got {reference_length}.")
    if marker_radius <= 0.0:
        raise ValueError(f"marker_radius must be positive, got {marker_radius}.")

    import warp as wp

    axis_origin_np = np.asarray(axis_origin, dtype=np.float32)
    axis_starts = np.repeat(axis_origin_np[np.newaxis, :], 3, axis=0)
    axis_ends = axis_starts + np.eye(3, dtype=np.float32) * float(axis_length)
    axis_colors = np.eye(3, dtype=np.float32)
    log_axes = visualizer.log_arrows if hasattr(visualizer, "log_arrows") else visualizer.log_lines
    log_axes(
        f"{name_prefix}_axes",
        wp.array(axis_starts, dtype=wp.vec3),
        wp.array(axis_ends, dtype=wp.vec3),
        wp.array(axis_colors, dtype=wp.vec3),
    )

    scale_center_np = np.asarray(scale_bar_center, dtype=np.float32)
    half_length = 0.5 * float(reference_length)
    scale_start = scale_center_np + np.array([-half_length, 0.0, 0.0], dtype=np.float32)
    scale_end = scale_center_np + np.array([half_length, 0.0, 0.0], dtype=np.float32)
    scale_color = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)
    visualizer.log_lines(
        f"{name_prefix}_scale_bar",
        wp.array([scale_start], dtype=wp.vec3),
        wp.array([scale_end], dtype=wp.vec3),
        wp.array(scale_color, dtype=wp.vec3),
    )
    visualizer.log_points(
        f"{name_prefix}_scale_markers",
        wp.array([scale_start, scale_end], dtype=wp.vec3),
        radii=wp.array([marker_radius, marker_radius], dtype=wp.float32),
        colors=wp.array(np.repeat(scale_color, 2, axis=0), dtype=wp.vec3),
    )


def rotate_batch(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a set of vectors by a batch of quaternions (vectorised Rodrigues).

    Uses the double-cross-product form of Rodrigues' rotation formula::

        v' = v + 2w (q × v) + 2 (q × (q × v))

    where ``q`` is the vector part ``[qx, qy, qz]`` and ``w = qw`` is the
    scalar part of the xyzw quaternion.

    Args:
        q_xyzw: ``(N, 4)`` float array of unit xyzw quaternions.
        v:      ``(K, 3)`` float array of vectors to rotate.

    Returns:
        ``(N, K, 3)`` float32 array of rotated vectors — one ``(K, 3)``
        block per quaternion.
    """
    q_vec = q_xyzw[:, :3].astype(np.float64)  # (N, 3)
    w = q_xyzw[:, 3].astype(np.float64)  # (N,)

    q_b = q_vec[:, np.newaxis, :]  # (N, 1, 3)
    w_b = w[:, np.newaxis, np.newaxis]  # (N, 1, 1)
    v_b = v[np.newaxis, :, :].astype(np.float64)  # (1, K, 3)

    qxv = np.cross(q_b, v_b)  # (N, K, 3)
    qxqxv = np.cross(q_b, qxv)  # (N, K, 3)
    return (v_b + 2.0 * w_b * qxv + 2.0 * qxqxv).astype(np.float32)
