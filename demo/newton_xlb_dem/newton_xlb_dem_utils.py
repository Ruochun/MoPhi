"""demo/newton_xlb_dem/newton_xlb_dem_utils.py

Demo-specific utility functions for the Newton + XLB + DEME three-way co-simulation demo.

This module is intentionally scoped to the newton_xlb_dem demo and should NOT be used
as reusable infrastructure for other robots or scenarios.  Reusable helpers that work
across multiple demos belong in the mophi package (e.g. mophi.xlb_build_streamlines).

Contents
--------
Policy helpers
  quat_rotate_inverse     -- TorchScript quaternion inverse rotation
  compute_obs             -- ANYmal C 48-D walking-policy observation

Robot geometry helpers (pre-finalize, from Newton ModelBuilder)
  scan_and_enlarge_foot_spheres       -- scan SPHERE shapes, double radii, record info
  build_foot_tip_descriptors          -- build per-foot contact-proxy descriptor list
  print_foot_tip_descriptors          -- startup print for foot-tip descriptors
  collect_visual_body_part_descriptors -- enumerate non-colliding visual mesh shapes
  print_visual_body_part_descriptors  -- startup print for visual body-part shapes

Per-frame pose helper
  compute_foot_tip_poses  -- world-space foot sphere centres + shank orientations

XLB LBM obstacle helpers
  xlb_world_to_grid_idx          -- map world position to interior LBM grid cell
  xlb_prescribed_robot_box_grid  -- compute robot AABB in grid space
  xlb_update_robot_box_gpu       -- update bc_mask / missing_mask via Warp kernels
  (compatibility wrappers around mophi.couplers.newton_xlb)
"""

import numpy as np
import torch
import warp as wp
from newton import GeoType, ShapeFlags
from mophi.couplers.newton_xlb import update_box_boundary_gpu, world_to_grid_index

# ─────────────────────────────────────────────────────────────────────────────
# Policy helpers
# ─────────────────────────────────────────────────────────────────────────────


@torch.jit.script
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse of a quaternion (last dimension is [x,y,z,w])."""
    q_w = q[..., 3]
    q_vec = q[..., :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    if q_vec.dim() == 2:
        c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    else:
        c = q_vec * torch.einsum("...i,...i->...", q_vec, v).unsqueeze(-1) * 2.0
    return a - b + c


def compute_obs(actions, joint_q_t, joint_qd_t, joint_pos_initial, indices, gravity_vec, command):
    """Compute the 48-D observation vector required by the ANYmal C walking policy.

    Mirrors newton/examples/robot/example_robot_anymal_c_walk.py::compute_obs().
    The observation concatenates: base linear velocity (body frame), base angular
    velocity (body frame), projected gravity, velocity command, joint position
    error (lab order), joint velocity (lab order), previous actions.

    Args:
        actions:          1×12 float32 GPU tensor — previous policy actions.
        joint_q_t:        1-D float32 GPU torch tensor — Newton state joint_q (zero-copy
                          from wp.to_torch(coupler.newton_state_0.joint_q)).
        joint_qd_t:       1-D float32 GPU torch tensor — Newton state joint_qd (zero-copy
                          from wp.to_torch(coupler.newton_state_0.joint_qd)).
        joint_pos_initial: 1×12 float32 GPU tensor — initial joint positions (clone of
                           joint_q[7:] at t=0).
        indices:          1-D int64 GPU tensor — lab-to-mujoco reordering indices.
        gravity_vec:      1×3 float32 GPU tensor — [0, 0, -1].
        command:          1×3 float32 GPU tensor — velocity command [vx, vy, yaw].

    Returns:
        obs: 1×48 float32 GPU tensor — policy observation vector.
    """
    root_quat_w = joint_q_t[3:7].to(dtype=torch.float32).unsqueeze(0)
    root_lin_vel_w = joint_qd_t[:3].to(dtype=torch.float32).unsqueeze(0)
    root_ang_vel_w = joint_qd_t[3:6].to(dtype=torch.float32).unsqueeze(0)
    joint_pos_current = joint_q_t[7:].to(dtype=torch.float32).unsqueeze(0)
    joint_vel_current = joint_qd_t[6:].to(dtype=torch.float32).unsqueeze(0)
    vel_b = quat_rotate_inverse(root_quat_w, root_lin_vel_w)
    a_vel_b = quat_rotate_inverse(root_quat_w, root_ang_vel_w)
    grav = quat_rotate_inverse(root_quat_w, gravity_vec)
    joint_pos_rel = joint_pos_current - joint_pos_initial
    joint_vel_rel = joint_vel_current
    rearranged_joint_pos_rel = torch.index_select(joint_pos_rel, 1, indices)
    rearranged_joint_vel_rel = torch.index_select(joint_vel_rel, 1, indices)
    obs = torch.cat([vel_b, a_vel_b, grav, command, rearranged_joint_pos_rel, rearranged_joint_vel_rel, actions], dim=1)
    return obs


# ─────────────────────────────────────────────────────────────────────────────
# Robot geometry helpers (operate on Newton ModelBuilder, called before finalize)
# ─────────────────────────────────────────────────────────────────────────────


def scan_and_enlarge_foot_spheres(builder):
    """Scan builder shapes for GeoType.SPHERE, double each radius, and record per-sphere info.

    Side-effect: modifies builder.shape_scale in-place (doubles all sphere radii).
    Doubling the URDF's small sphere radii prevents the robot from stumbling on the
    flat ground plane.

    Returns:
        builder_foot_spheres     : dict mapping builder body-index →
                                   {"local_offset": [x, y, z], "sphere_radius": float}
        builder_body_name_to_idx : dict mapping short body label → builder body index
    """
    builder_foot_spheres: dict = {}
    for i in range(len(builder.shape_type)):
        if builder.shape_type[i] == GeoType.SPHERE:
            r = builder.shape_scale[i][0]
            builder.shape_scale[i] = (r * 2.0, 0.0, 0.0)
            # shape_transform[i].p is the sphere-centre offset in the body's local frame.
            local_p = builder.shape_transform[i].p
            builder_foot_spheres[builder.shape_body[i]] = {
                "local_offset": [float(local_p[0]), float(local_p[1]), float(local_p[2])],
                "sphere_radius": float(r * 2.0),
            }
    builder_body_name_to_idx = {lbl.split("/")[-1]: i for i, lbl in enumerate(builder.body_label)}
    return builder_foot_spheres, builder_body_name_to_idx


def build_foot_tip_descriptors(builder_body_name_to_idx, builder_foot_spheres, foot_shank_names):
    """Build the static list of foot-tip contact proxy descriptors.

    ANYmal C has a GeoType.SPHERE collision shape at the distal end of each SHANK link.
    These spheres are the ground-contact proxies and will serve as coupling surfaces for
    DEM particles and XLB fluid boundaries in future co-sim work.

    Args:
        builder_body_name_to_idx : dict mapping short body label → builder body index.
        builder_foot_spheres     : dict as returned by scan_and_enlarge_foot_spheres().
        foot_shank_names         : sequence of shank body names, e.g.
                                   ["LF_SHANK", "RF_SHANK", "LH_SHANK", "RH_SHANK"].

    Returns:
        foot_tip_descriptors  : list of dicts with keys:
            "label"         : str   — short body name, e.g. "LF_SHANK"
            "body_idx"      : int   — row index into the body_q / body_qd arrays
            "local_offset"  : list  — sphere centre offset in the shank's body frame [m]
            "sphere_radius" : float — sphere radius [m]  (constant throughout simulation)
        foot_tip_sphere_radii : list[float] — per-foot radii in foot_shank_names order.
    """
    foot_tip_descriptors = []
    for shank_name in foot_shank_names:
        b_idx = builder_body_name_to_idx.get(shank_name)
        if b_idx is None:
            raise RuntimeError(f"[FootTip] Shank body '{shank_name}' not found in builder body labels")
        sphere_info = builder_foot_spheres.get(b_idx)
        if sphere_info is None:
            raise RuntimeError(f"[FootTip] No sphere shape found attached to shank body '{shank_name}'")
        foot_tip_descriptors.append(
            {
                "label": shank_name,
                "body_idx": b_idx,
                "local_offset": sphere_info["local_offset"],
                "sphere_radius": sphere_info["sphere_radius"],
            }
        )
    foot_tip_sphere_radii = [d["sphere_radius"] for d in foot_tip_descriptors]
    return foot_tip_descriptors, foot_tip_sphere_radii


def print_foot_tip_descriptors(foot_tip_descriptors):
    """Print a startup summary of the foot-tip contact proxy geometry."""
    print("[FootTip] Leg-tip contact proxies (sphere geometry):")
    for d in foot_tip_descriptors:
        lo = d["local_offset"]
        print(
            f"         {d['label']}: body_idx={d['body_idx']}, "
            f"local_offset=[{lo[0]:.4f}, {lo[1]:.4f}, {lo[2]:.4f}] m, "
            f"sphere_radius={d['sphere_radius']:.4f} m"
        )
    print()


def collect_visual_body_part_descriptors(builder):
    """Enumerate all visual (non-colliding, visible) shapes from a Newton ModelBuilder.

    Newton's add_urdf() loads both collision shapes (from <collision> tags) and
    visual shapes (from <visual> tags) because load_visual_shapes=True by default.
    Visual shapes carry the actual 3-D geometry of each robot link — typically
    triangle-mesh representations loaded from STL or DAE files referenced in the
    URDF.  They are not used for contact detection (ShapeFlags.COLLIDE_SHAPES is
    not set), but they are what Newton's renderer displays and what can guide a
    higher-fidelity XLB wall boundary in future work.

    Returns:
        list of dicts, one per visual shape:
            "body_idx"    : int            — row index into body_q / body_qd arrays
            "body_name"   : str            — short body name, e.g. "base"
            "shape_label" : str            — full shape label from the builder
            "geo_type"    : GeoType        — geometry kind (usually GeoType.MESH)
            "local_xform" : wp.transform   — shape pose in the body's local frame
            "scale"       : list[float]    — [sx, sy, sz] scale factors [m]
            "mesh"        : newton.Mesh | None — triangle mesh with .vertices (N×3
                                                float32) and .indices (M int32 =
                                                3×num_tris); None for non-mesh shapes
    """
    descriptors: list[dict] = []
    for i in range(len(builder.shape_type)):
        flags = builder.shape_flags[i]
        # Keep only shapes that are visible but do not participate in collision.
        if not (flags & ShapeFlags.VISIBLE):
            continue
        if flags & ShapeFlags.COLLIDE_SHAPES:
            continue
        b_idx = builder.shape_body[i]
        if b_idx < 0:
            continue  # world-attached shape (e.g. a stray ground shape)
        body_name = builder.body_label[b_idx].split("/")[-1]
        geo_type = GeoType(builder.shape_type[i])
        scale = builder.shape_scale[i]
        descriptors.append(
            {
                "body_idx": b_idx,
                "body_name": body_name,
                "shape_label": builder.shape_label[i],
                "geo_type": geo_type,
                "local_xform": builder.shape_transform[i],
                "scale": [float(scale[0]), float(scale[1]), float(scale[2])],
                "mesh": builder.shape_source[i],
            }
        )
    return descriptors


def print_visual_body_part_descriptors(descriptors):
    """Print a startup summary of robot visual body-part shapes.

    Demonstrates that handles to every robot body part (and its visual mesh) are
    available.  These handles are the foundation for a higher-fidelity XLB robot
    representation beyond the current AABB box.
    """
    n_bodies = len({d["body_idx"] for d in descriptors})
    print(
        f"[BodyParts] Robot visual shapes from URDF <visual> tags: "
        f"{len(descriptors)} shape(s) across {n_bodies} body(-ies)."
    )
    for d in descriptors:
        mesh_info = ""
        if d["mesh"] is not None:
            nv = len(d["mesh"].vertices)
            nt = len(d["mesh"].indices) // 3
            mesh_info = f", mesh: {nv} verts / {nt} tris"
        print(
            f"            body_idx={d['body_idx']} [{d['body_name']}]  "
            f"geo={d['geo_type'].name}  "
            f"scale=[{d['scale'][0]:.3f},{d['scale'][1]:.3f},{d['scale'][2]:.3f}]" + mesh_info
        )
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame pose helper
# ─────────────────────────────────────────────────────────────────────────────


def compute_foot_tip_poses(body_q_np, foot_tip_descriptors):
    """Compute world-space sphere centres and shank orientations for all foot tips.

    Newton represents each foot tip as a sphere shape attached to the SHANK body.
    The contact proxy is fully described by:
      • sphere centre in world space: body_pos + rotate(body_quat, local_offset)
      • sphere radius [m]  (constant, stored in foot_tip_descriptors)

    The rotation is applied via the Rodrigues formula for rotate(q, o):
      rotate(q, o) = o + 2*qw*(qv × o) + 2*(qv × (qv × o))
    where qv = (qx, qy, qz) and qw is the scalar part.

    Args:
        body_q_np           : (N, 7) float32 numpy array — Newton body transforms
                              [px, py, pz, qx, qy, qz, qw], or None.
        foot_tip_descriptors: list of dicts as returned by build_foot_tip_descriptors().

    Returns:
        foot_tip_positions : list of [px, py, pz] world-space sphere centres [m].
                             Empty list when body_q_np is None or empty.
        foot_tip_rotations : list of [qx, qy, qz, qw] shank orientations.
                             Empty list when body_q_np is None or empty.
    """
    foot_tip_positions = []
    foot_tip_rotations = []
    if body_q_np is None or len(body_q_np) == 0:
        return foot_tip_positions, foot_tip_rotations
    for d in foot_tip_descriptors:
        t = body_q_np[d["body_idx"]]  # [px, py, pz, qx, qy, qz, qw]
        px, py, pz = float(t[0]), float(t[1]), float(t[2])
        qx, qy, qz, qw = float(t[3]), float(t[4]), float(t[5]), float(t[6])
        ox, oy, oz = d["local_offset"]
        # First cross product: qv × o
        cx = qy * oz - qz * oy
        cy = qz * ox - qx * oz
        cz = qx * oy - qy * ox
        # Second cross product: qv × (qv × o)
        ccx = qy * cz - qz * cy
        ccy = qz * cx - qx * cz
        ccz = qx * cy - qy * cx
        foot_tip_positions.append(
            [
                px + ox + 2.0 * (qw * cx + ccx),
                py + oy + 2.0 * (qw * cy + ccy),
                pz + oz + 2.0 * (qw * cz + ccz),
            ]
        )
        foot_tip_rotations.append([qx, qy, qz, qw])
    return foot_tip_positions, foot_tip_rotations


# ─────────────────────────────────────────────────────────────────────────────
# XLB LBM obstacle helpers
# ─────────────────────────────────────────────────────────────────────────────
def xlb_world_to_grid_idx(world_pos, domain_min, domain_max, grid_dims):
    """Map a world-space 3-D point to the nearest interior LBM grid cell index.

    Returns a 3-element int numpy array clamped to [1, N−2] so the obstacle
    never overlaps the wall / inlet / outlet cells at grid indices 0 and N−1.

    Args:
        world_pos  : array-like length-3 — world-space position [m].
        domain_min : array-like length-3 — LBM domain min corner [m].
        domain_max : array-like length-3 — LBM domain max corner [m].
        grid_dims  : (NX, NY, NZ) tuple — LBM grid dimensions.
    """
    return world_to_grid_index(world_pos, domain_min, domain_max, grid_dims)


def xlb_prescribed_robot_box_grid(
    base_pos, domain_min, domain_max, grid_dims, half_ext_x, half_ext_y, below_base, above_base
):
    """Return (gc_min, gc_max) integer grid arrays for the prescribed robot AABB.

    The box is centred on base_pos (world-space [x, y, z]) with fixed half-extents
    that enclose the full ANYmal C geometry (torso + leg reach) with margin.
    Returns (None, None) when the box is entirely outside the LBM domain.

    Args:
        base_pos   : array-like length-3 | None — Newton base body world position [m].
        domain_min : array-like length-3 — LBM domain min corner [m].
        domain_max : array-like length-3 — LBM domain max corner [m].
        grid_dims  : (NX, NY, NZ) tuple.
        half_ext_x : float — box half-extent in x (walking direction) [m].
        half_ext_y : float — box half-extent in y (lateral direction) [m].
        below_base : float — box extent below the base centre [m].
        above_base : float — box extent above the base centre [m].
    """
    if base_pos is None:
        return None, None
    world_min = np.array(
        [
            base_pos[0] - half_ext_x,
            base_pos[1] - half_ext_y,
            # z_min is clamped to the lower domain boundary so the box remains inside
            # the fluid domain and does not extend below the ground region.
            max(0.0, base_pos[2] - below_base),
        ]
    )
    world_max = np.array(
        [
            base_pos[0] + half_ext_x,
            base_pos[1] + half_ext_y,
            base_pos[2] + above_base,
        ]
    )
    if np.any(world_max <= domain_min) or np.any(world_min >= domain_max):
        return None, None
    gc_min = xlb_world_to_grid_idx(np.maximum(world_min, domain_min), domain_min, domain_max, grid_dims)
    gc_max = xlb_world_to_grid_idx(np.minimum(world_max, domain_max), domain_min, domain_max, grid_dims)
    if np.any(gc_max < gc_min):
        return None, None
    return gc_min, gc_max


def xlb_update_robot_box_gpu(
    bc_mask, missing_mask, old_gc_min, old_gc_max, new_gc_min, new_gc_max, robot_bc_id, vel_c_wp, q
):
    """Update bc_mask and missing_mask in-place on the GPU for the moving robot AABB.

    Clears the old robot box region and stamps the new one using Warp GPU kernels,
    with no CPU round-trip and no GPU reallocation.  The robot box region is always
    in the LBM interior (guaranteed by xlb_world_to_grid_idx clamping), so the
    fluid base value for those cells is always 0 / False — clearing simply writes
    those defaults without needing a stored base-mask copy.

    Pass new_gc_min = None to clear the robot obstacle entirely.

    Args:
        bc_mask      : wp.array4d(uint8) — device-resident bc_mask (1, NX, NY, NZ).
        missing_mask : wp.array4d(bool)  — device-resident missing_mask (Q, NX, NY, NZ).
        old_gc_min   : int[3] numpy array | None — previous box min corner.
        old_gc_max   : int[3] numpy array | None — previous box max corner.
        new_gc_min   : int[3] numpy array | None — new box min corner (None = clear only).
        new_gc_max   : int[3] numpy array | None — new box max corner.
        robot_bc_id  : int  — HalfwayBounceBackBC ID for the robot obstacle.
        vel_c_wp     : wp.array2d(int32) (Q, 3) — D3Q19 velocity stencil on device.
        q            : int  — number of lattice velocity directions (19 for D3Q19).

    Returns:
        (new_gc_min, new_gc_max) — the updated box corners (same as inputs).
        Callers should store these to pass as old_gc_min/max on the next call.
    """
    return update_box_boundary_gpu(
        bc_mask, missing_mask, old_gc_min, old_gc_max, new_gc_min, new_gc_max, robot_bc_id, vel_c_wp, q
    )
