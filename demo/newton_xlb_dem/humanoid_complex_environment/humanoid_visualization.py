"""Visualization mechanics shared by humanoid experiment variants."""

import warp as wp


@wp.kernel
def transform_contact_proxy_lines(
    body_q: wp.array(dtype=wp.transform),
    body_indices: wp.array(dtype=wp.int32),
    local_starts: wp.array(dtype=wp.vec3),
    local_ends: wp.array(dtype=wp.vec3),
    world_starts: wp.array(dtype=wp.vec3),
    world_ends: wp.array(dtype=wp.vec3),
):
    line_idx = wp.tid()
    body_q_i = body_q[body_indices[line_idx]]
    world_starts[line_idx] = wp.transform_point(body_q_i, local_starts[line_idx])
    world_ends[line_idx] = wp.transform_point(body_q_i, local_ends[line_idx])


def configure_scene_lighting(viewer, sky_upper, sky_lower, light_color) -> None:
    """Apply a scene-selected sky gradient and directional-light colour."""
    if not hasattr(viewer, "renderer"):
        return
    viewer.renderer.sky_upper = tuple(sky_upper)
    viewer.renderer.sky_lower = tuple(sky_lower)
    viewer.renderer.background_color = tuple(sky_upper)
    viewer.renderer._light_color = tuple(light_color)
