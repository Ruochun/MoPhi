"""Visualization utilities shared by MoPhi visualizer backends and demos."""

from __future__ import annotations

import numpy as np


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
