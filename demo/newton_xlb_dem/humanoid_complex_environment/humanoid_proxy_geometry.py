"""Robot contact-proxy geometry preparation for humanoid experiments."""

import numpy as np

from mophi.couplers.newton_deme import NewtonDEMEMeshOwnerSpec, combine_triangle_meshes


def transform_points(points: np.ndarray, transform) -> np.ndarray:
    """Apply a Warp-style XYZW transform to NumPy points."""
    position = np.asarray(transform[:3], dtype=np.float32)
    quaternion_xyz = np.asarray(transform[3:6], dtype=np.float32)
    quaternion_w = float(transform[6])
    cross = np.cross(quaternion_xyz, points)
    return points + 2.0 * np.cross(quaternion_xyz, cross + quaternion_w * points) + position


def make_scaled_box_mesh_proxies(visual_meshes, base_vertices, base_indices, padding):
    """Scale one origin-centred box OBJ to bound each robot visual mesh."""
    base_bounds_min = base_vertices.min(axis=0)
    base_bounds_max = base_vertices.max(axis=0)
    base_center = 0.5 * (base_bounds_min + base_bounds_max)
    base_size = base_bounds_max - base_bounds_min
    if not np.all(np.isfinite(base_vertices)) or np.any(base_size <= 0.0):
        raise ValueError("Contact proxy mesh has invalid bounds.")
    if len(base_indices) % 3 != 0 or np.any(base_indices < 0) or np.any(base_indices >= len(base_vertices)):
        raise ValueError("Contact proxy mesh has invalid triangle indices.")
    proxies = []
    for mesh_info in visual_meshes:
        bounds_min = mesh_info["scaled_vertices"].min(axis=0) - padding
        bounds_max = mesh_info["scaled_vertices"].max(axis=0) + padding
        proxy_center = 0.5 * (bounds_min + bounds_max)
        proxy_size = bounds_max - bounds_min
        vertices = (base_vertices - base_center) * (proxy_size / base_size) + proxy_center
        proxies.append(
            {
                "robot_instance": mesh_info["robot_instance"],
                "body_index": mesh_info["body_index"],
                "body_label": mesh_info["body_label"],
                "shape_label": mesh_info["shape_label"],
                "half_extents": 0.5 * proxy_size,
                "body_local_vertices": transform_points(vertices, mesh_info["shape_transform"]),
                "triangle_indices": base_indices,
            }
        )
    return proxies


def make_deme_mesh_owner_specs(contact_mesh_proxies, robot_families):
    """Combine proxies per Newton body into DEME mesh-owner specifications."""
    proxies_by_body = {}
    for proxy in contact_mesh_proxies:
        key = (proxy["robot_instance"], proxy["body_index"])
        proxies_by_body.setdefault(key, []).append(proxy)
    specs = []
    for (robot_instance, body_index), body_proxies in sorted(proxies_by_body.items()):
        vertices, triangle_indices = combine_triangle_meshes(
            [(proxy["body_local_vertices"], proxy["triangle_indices"]) for proxy in body_proxies]
        )
        specs.append(
            NewtonDEMEMeshOwnerSpec(
                name=f"robot_{robot_instance}_body_{body_index}",
                newton_body_index=body_index,
                family=robot_families[robot_instance],
                mass=1.0,
                principal_moi=(1.0, 1.0, 1.0),
                vertices=vertices,
                triangle_indices=triangle_indices,
            )
        )
    return specs
