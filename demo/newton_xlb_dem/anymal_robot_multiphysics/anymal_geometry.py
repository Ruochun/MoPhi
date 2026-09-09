"""ANYmal geometry and pose helpers shared by its demo variants.

Contents
--------
Robot geometry helpers (pre-finalize, from Newton ModelBuilder)
  scan_and_enlarge_foot_spheres       -- scan SPHERE shapes, double radii, record info
  build_foot_tip_descriptors          -- build per-foot contact-proxy descriptor list
  print_foot_tip_descriptors          -- startup print for foot-tip descriptors
  collect_visual_body_part_descriptors -- enumerate non-colliding visual mesh shapes
  print_visual_body_part_descriptors  -- startup print for visual body-part shapes

Per-frame pose helper
  compute_foot_tip_poses  -- world-space foot sphere centres + shank orientations

"""

import numpy as np
import warp as wp
from newton import GeoType, ShapeFlags

# ─────────────────────────────────────────────────────────────────────────────
# Robot geometry helpers
# ─────────────────────────────────────────────────────────────────────────────


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
