"""One downward-facing articulated hand attempting to massage DEME clumps.

The default backend uses DEME-managed ellipsoidal clumps and kinematic copies
of Newton's Allegro convex contact meshes. Mesh-clump contact is disabled while
the clumps settle, then enabled before Newton begins driving the hand.
"""

import json
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np
import warp as wp

import mophi
import newton
from flexible_hand_common import (
    configure_replicated_controls as _configure_replicated_controls_shared,
    update_finger_targets as _update_finger_targets,
    update_hand_root_descend_lift as _update_hand_root_descend_lift,
)
from mophi.couplers.newton_deme import (
    NewtonDEMEParticleExchange,
    NewtonDEMEOwnerMap,
    NewtonDEMEOwnerPoseExchange,
    extract_newton_shape_triangle_meshes,
    write_wavefront_mesh,
)
from mophi.utils.package_provider import load_package_provider
from newton import JointTargetMode

DEME = load_package_provider("deme")

# =============================================================================
# Simulation configuration
# All user-tunable constants are defined here.
# =============================================================================

# -- Timing -------------------------------------------------------------------
RENDER_FPS = 25
NEWTON_DT = 1.0 / 1000.0
DEME_DT = 5e-5
SIM_DURATION_SECONDS = 5.0

# -- Hand replication ---------------------------------------------------------
HAND_COUNT = 1
HAND_SPACING = (1.4, 1.4, 0.0)
HAND_BASE_HEIGHT = 0.85
HAND_DOWNWARD_ROTATION_RPY = (np.pi, 0.0, 0.0)
HAND_ASSET_NAME = "wonik_allegro"
HAND_ASSET_RELATIVE_PATH = Path("usd") / "allegro_left_hand_with_cube.usda"
HAND_ASSET_IGNORE_PATHS = [".*Dummy", ".*CollisionPlane", ".*DexCube", ".*root_joint"]
ENABLE_HAND_SELF_COLLISIONS = False

# -- Prescribed descend-grasp-lift motion -------------------------------------
# Move the kinematic hand root into the clump pile before lifting it back upward,
# while maintaining the varied finger motions.
HAND_DESCEND_START_SECONDS = 0.05
HAND_DESCEND_DURATION_SECONDS = 1.5
HAND_DESCEND_DEPTH = 0.175
HAND_LIFT_START_SECONDS = 3.0
HAND_LIFT_DURATION_SECONDS = 1.5
HAND_LIFT_HEIGHT = 0.65

# -- Granular backend ---------------------------------------------------------
DEME_SETTLE_TIME = 1.0
DEME_CLUMP_SCALE = 0.025
DEME_HCP_SAMPLE_CENTER = (0.0, 0.12, 0.35)
DEME_HCP_SAMPLE_HALF_EXTENTS = (0.28, 0.28, 0.22)
DEME_HCP_SAMPLE_SPACING = 0.12
DEME_INITIAL_VELOCITY_RANDOM_SEED = 20260615
DEME_INITIAL_VELOCITY_MIN = (-0.08, -0.08, -0.12)
DEME_INITIAL_VELOCITY_MAX = (0.08, 0.08, 0.02)
DEME_PARTICLE_DENSITY = 1800.0
DEME_PARTICLE_FAMILY = 0
DEME_HAND_PROXY_SLEEP_FAMILY = 20
DEME_HAND_PROXY_ACTIVE_FAMILY = 21
DEME_DOMAIN_Z_MAX = 3.0
DEME_MATERIAL_YOUNGS_MODULUS = 2.0e5
DEME_MATERIAL_POISSON_RATIO = 0.3
DEME_MATERIAL_RESTITUTION = 0.05
DEME_MATERIAL_FRICTION = 0.9
DEME_CLUMP_COLOR = (0.76, 0.60, 0.42)
DEME_HAND_PROXY_DIRECTORY_NAME = "deme_hand_contact_meshes"
TRAY_FRICTION = 0.9
GRANULAR_LIFT_THRESHOLD = 0.08
GRANULAR_DISPLACEMENT_THRESHOLD = 0.03

# -- Static shallow tray -------------------------------------------------------
TRAY_FLOOR_HALF_EXTENTS = (0.38, 0.38, 0.025)
TRAY_FLOOR_CENTER = (0.0, 0.12, 0.025)
TRAY_WALL_THICKNESS = 0.025
TRAY_WALL_HEIGHT = 0.10
TRAY_COLOR = (0.22, 0.32, 0.38)

# -- Finger drives ------------------------------------------------------------
FINGER_DRIVE_STIFFNESS = 150.0
FINGER_DRIVE_DAMPING = 5.0
FINGER_JOINT_ARMATURE = 1.0e-2
FINGER_MOTION_FREQUENCY = 0.55
HAND_PHASE_SPREAD_CYCLES = 1.0
# Profiles repeat across the hand array: (name, closure scale, motion scale, phase offset).
GRASP_PROFILES = (
    ("deep_scoop", 1.45, 0.22, 0.00),
    ("pulsing_grasp", 1.15, 0.55, 0.13),
    ("gentle_grasp", 0.82, 0.25, 0.27),
    ("open_rake", 0.35, 0.38, 0.41),
)
# Each entry is: (joint-label suffix, center [rad], amplitude [rad], phase [cycles]).
FINGER_MOTION_SPECS = (
    ("index_joint_0", 0.30, 0.28, 0.00),
    ("index_joint_1", 0.45, 0.38, 0.04),
    ("index_joint_2", 0.45, 0.38, 0.08),
    ("index_joint_3", 0.45, 0.38, 0.12),
    ("middle_joint_0", 0.30, 0.28, 0.18),
    ("middle_joint_1", 0.45, 0.38, 0.22),
    ("middle_joint_2", 0.45, 0.38, 0.26),
    ("middle_joint_3", 0.45, 0.38, 0.30),
    ("ring_joint_0", 0.30, 0.28, 0.36),
    ("ring_joint_1", 0.45, 0.38, 0.40),
    ("ring_joint_2", 0.45, 0.38, 0.44),
    ("ring_joint_3", 0.45, 0.38, 0.48),
    ("thumb_joint_0", 0.55, 0.30, 0.58),
    ("thumb_joint_1", 0.45, 0.38, 0.62),
    ("thumb_joint_2", 0.45, 0.38, 0.66),
    ("thumb_joint_3", 0.45, 0.38, 0.70),
)

# -- Contact-ready model settings ---------------------------------------------
SHAPE_CONTACT_STIFFNESS = 1.0e3
SHAPE_CONTACT_DAMPING = 1.0e2
SHAPE_CONTACT_MARGIN = 0.003
SHAPE_CONTACT_GAP = 0.008
MAX_CONTACTS_PER_HAND = 900
MAX_CONSTRAINTS_PER_HAND = 2400
SOLVER_ITERATIONS = 12
SOLVER_LINE_SEARCH_ITERATIONS = 8

# -- Visualization ------------------------------------------------------------
CAMERA_POSITION = (1.4, -2.0, 1.6)
CAMERA_PITCH = -25.0
CAMERA_YAW = 145.0
VIEWER_WORLD_SPACING = (0.0, 0.0, 0.0)
REFERENCE_AXIS_ORIGIN = (-0.9, -0.9, 0.02)
REFERENCE_SCALE_BAR_CENTER = (-0.3, -0.9, 0.06)
SHOW_LABORATORY_ENVIRONMENT = True
RENDER_SETTLING_PHASE = False
LABORATORY_ROOM_CENTER = (0.0, 0.0, 0.0)
LABORATORY_ROOM_HALF_EXTENTS = (2.0, 2.0, 1.8)

# -- Output -------------------------------------------------------------------
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "output" / Path(__file__).stem
SAVE_MOVIE = True
MOVIE_OUTPUT_PATH = OUTPUT_DIRECTORY / "demo_flexible_hand_deme.mp4"
MOVIE_FPS = RENDER_FPS
SAVE_FINAL_STATE = True
FINAL_STATE_OUTPUT_PATH = OUTPUT_DIRECTORY / "final_state.npz"
RUN_METADATA_OUTPUT_PATH = OUTPUT_DIRECTORY / "run_metadata.json"

# -- Derived constants --------------------------------------------------------
FRAME_DT = 1.0 / RENDER_FPS
NUM_FRAMES = int(round(SIM_DURATION_SECONDS / FRAME_DT))
SIM_SUBSTEPS = int(round(FRAME_DT / NEWTON_DT))
DEME_SUBSTEPS = int(round(NEWTON_DT / DEME_DT))
SETTLING_TIME_EPSILON = 1.0e-9
DEME_CLUMP_SPHERE_RADII = np.array([1.00, 0.88, 0.64, 0.88, 0.64], dtype=np.float32) * DEME_CLUMP_SCALE
DEME_CLUMP_SPHERE_OFFSETS = (
    np.array(
        [[0.0, 0.0, 0.00], [0.0, 0.0, 0.86], [0.0, 0.0, 1.44], [0.0, 0.0, -0.86], [0.0, 0.0, -1.44]],
        dtype=np.float32,
    )
    * DEME_CLUMP_SCALE
)
if not np.isclose(SIM_SUBSTEPS * NEWTON_DT, FRAME_DT, rtol=0.0, atol=1.0e-12):
    raise ValueError("FRAME_DT must be an integer multiple of NEWTON_DT.")
if not np.isclose(DEME_SUBSTEPS * DEME_DT, NEWTON_DT, rtol=0.0, atol=1.0e-12):
    raise ValueError("NEWTON_DT must be an integer multiple of DEME_DT.")
if DEME_SETTLE_TIME < 0.0:
    raise ValueError("DEME_SETTLE_TIME must be non-negative.")
if DEME_HCP_SAMPLE_SPACING <= 0.0:
    raise ValueError("DEME_HCP_SAMPLE_SPACING must be positive.")
if np.any(np.asarray(DEME_HCP_SAMPLE_HALF_EXTENTS) < 0.0):
    raise ValueError("DEME_HCP_SAMPLE_HALF_EXTENTS must be non-negative.")
if np.any(np.asarray(DEME_INITIAL_VELOCITY_MAX) < np.asarray(DEME_INITIAL_VELOCITY_MIN)):
    raise ValueError("Every DEME_INITIAL_VELOCITY_MAX component must be >= DEME_INITIAL_VELOCITY_MIN.")


def _replicated_world_offsets() -> np.ndarray:
    """Return the centered physical XY offsets used by ``ModelBuilder.replicate``."""
    side_length = int(np.ceil(np.sqrt(HAND_COUNT)))
    offsets = []
    for hand_idx in range(HAND_COUNT):
        offsets.append(
            (
                (hand_idx // side_length) * HAND_SPACING[0],
                (hand_idx % side_length) * HAND_SPACING[1],
                0.0,
            )
        )
    offsets = np.asarray(offsets, dtype=np.float32)
    offsets[:, :2] -= 0.5 * (offsets[:, :2].min(axis=0) + offsets[:, :2].max(axis=0))
    return offsets


def _format_clump_position_bounds(label: str, positions: np.ndarray) -> str:
    """Return compact min/max diagnostics for DEME clump positions."""
    if len(positions) == 0:
        return f"{label}: no clumps"
    pos_min = positions.min(axis=0)
    pos_max = positions.max(axis=0)
    return (
        f"{label}: min=({pos_min[0]:.3f}, {pos_min[1]:.3f}, {pos_min[2]:.3f}), "
        f"max=({pos_max[0]:.3f}, {pos_max[1]:.3f}, {pos_max[2]:.3f})"
    )


def _export_hand_contact_mesh_proxies(hand: newton.ModelBuilder, directory: Path) -> list[tuple[int, Path]]:
    """Bake each Newton Allegro convex contact mesh into body-local OBJ coordinates for DEME."""
    directory.mkdir(parents=True, exist_ok=True)
    proxies = []
    shape_indices = [
        shape_idx
        for shape_idx, shape_type in enumerate(hand.shape_type)
        if shape_type == newton.GeoType.CONVEX_MESH
        and hand.shape_flags[shape_idx] & newton.ShapeFlags.COLLIDE_SHAPES
        and int(hand.shape_body[shape_idx]) >= 0
    ]
    for mesh in extract_newton_shape_triangle_meshes(hand, shape_indices):
        obj_path = directory / f"hand_body_{mesh.body_index:02d}_shape_{mesh.shape_index:02d}.obj"
        write_wavefront_mesh(obj_path, mesh.body_local_vertices, mesh.triangle_indices)
        proxies.append((mesh.body_index, obj_path))
    return proxies


def _build_deme_granular_system(hand: newton.ModelBuilder, model):
    """Create DEME clumps, a shared ground plane, and kinematic hand mesh proxies."""
    deme_solver = DEME.DEMSolver()
    deme_solver.UseFrictionalHertzianModel()
    deme_solver.SetVerbosity("INFO")
    wall_material = deme_solver.LoadMaterial(
        {
            "E": DEME_MATERIAL_YOUNGS_MODULUS,
            "nu": DEME_MATERIAL_POISSON_RATIO,
            "CoR": DEME_MATERIAL_RESTITUTION,
            "mu": DEME_MATERIAL_FRICTION,
        }
    )
    particle_material = deme_solver.LoadMaterial(
        {
            "E": DEME_MATERIAL_YOUNGS_MODULUS,
            "nu": DEME_MATERIAL_POISSON_RATIO,
            "CoR": DEME_MATERIAL_RESTITUTION,
            "mu": DEME_MATERIAL_FRICTION,
        }
    )
    deme_solver.SetMaterialPropertyPair("CoR", wall_material, particle_material, DEME_MATERIAL_RESTITUTION)
    deme_solver.SetMaterialPropertyPair("mu", wall_material, particle_material, DEME_MATERIAL_FRICTION)

    world_offsets = _replicated_world_offsets()
    floor_height = TRAY_FLOOR_CENTER[2] + TRAY_FLOOR_HALF_EXTENTS[2]
    deme_solver.AddBCPlane([0.0, 0.0, floor_height], [0.0, 0.0, 1.0], wall_material)
    print(f"DEME's floor is at z={floor_height}")
    x_low = world_offsets[:, 0].min() - 1.5 * HAND_SPACING[0]
    x_high = world_offsets[:, 0].max() + 1.5 * HAND_SPACING[0]
    y_low = world_offsets[:, 1].min() - 1.5 * HAND_SPACING[1]
    y_high = world_offsets[:, 1].max() + 1.5 * HAND_SPACING[1]
    z_low = 0.0 - 1.5 * HAND_SPACING[0]
    deme_solver.InstructBoxDomainDimension([x_low, x_high], [y_low, y_high], [z_low, DEME_DOMAIN_Z_MAX])

    template_mass = DEME_PARTICLE_DENSITY * (4.0 / 3.0 * np.pi * 2.0 * 1.0 * 1.0)
    template_moi = [
        1.0 / 5.0 * template_mass * (1.0**2 + 2.0**2),
        1.0 / 5.0 * template_mass * (1.0**2 + 2.0**2),
        1.0 / 5.0 * template_mass * (1.0**2 + 1.0**2),
    ]
    particle_template = deme_solver.LoadClumpType(
        template_mass,
        template_moi,
        DEME.GetDEMEDataFile("clumps/ellipsoid_2_1_1.csv"),
        particle_material,
    )
    particle_template.Scale(DEME_CLUMP_SCALE)
    hcp_sampler = DEME.HCPSampler(DEME_HCP_SAMPLE_SPACING)
    sample_center_offset = np.asarray(DEME_HCP_SAMPLE_CENTER, dtype=np.float32)
    clump_positions = []
    for world_offset in world_offsets:
        sample_center = (world_offset + sample_center_offset).tolist()
        clump_positions.extend(hcp_sampler.SampleBox(sample_center, DEME_HCP_SAMPLE_HALF_EXTENTS))
    clumps = deme_solver.AddClumps(particle_template, clump_positions)
    clumps.SetFamily(DEME_PARTICLE_FAMILY)
    velocity_rng = np.random.default_rng(DEME_INITIAL_VELOCITY_RANDOM_SEED)
    initial_velocities = velocity_rng.uniform(
        low=np.asarray(DEME_INITIAL_VELOCITY_MIN, dtype=np.float32),
        high=np.asarray(DEME_INITIAL_VELOCITY_MAX, dtype=np.float32),
        size=(len(clump_positions), 3),
    ).astype(np.float32)
    clumps.SetVel(initial_velocities.tolist())
    clump_tracker = deme_solver.Track(clumps)

    proxy_directory = OUTPUT_DIRECTORY / DEME_HAND_PROXY_DIRECTORY_NAME
    local_proxy_meshes = _export_hand_contact_mesh_proxies(hand, proxy_directory)
    if not local_proxy_meshes:
        mophi.fatal("No Newton convex hand contact meshes were found for DEME proxy creation.")
    initial_state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, initial_state)
    initial_body_q = initial_state.body_q.numpy()
    hand_proxy_trackers = []
    for hand_idx in range(HAND_COUNT):
        body_offset = hand_idx * hand.body_count
        for local_body_idx, obj_path in local_proxy_meshes:
            body_idx = body_offset + local_body_idx
            mesh_owner = deme_solver.AddWavefrontMeshObject(str(obj_path), wall_material, False)
            mesh_owner.SetInitPos(initial_body_q[body_idx, :3].tolist())
            mesh_owner.SetInitQuat(initial_body_q[body_idx, 3:7].tolist())
            mesh_owner.SetFamily(DEME_HAND_PROXY_SLEEP_FAMILY)
            hand_proxy_trackers.append((body_idx, deme_solver.Track(mesh_owner)))
    deme_solver.SetFamilyFixed(DEME_HAND_PROXY_SLEEP_FAMILY)
    deme_solver.SetFamilyFixed(DEME_HAND_PROXY_ACTIVE_FAMILY)
    deme_solver.DisableContactBetweenFamilies(DEME_PARTICLE_FAMILY, DEME_HAND_PROXY_SLEEP_FAMILY)
    deme_solver.SetGravitationalAcceleration([0.0, 0.0, -9.81])
    deme_solver.SetInitTimeStep(DEME_DT)
    deme_solver.SetErrorOutAvgContacts(150)
    deme_solver.SetErrorOutVelocity(150)
    deme_solver.SetInitBinNumTarget(1000)
    # deme_solver.DisableAdaptiveBinSize()
    deme_solver.Initialize()
    print(f"[DEME] Initialized {len(clump_positions)} ellipsoidal clumps and one shared ground plane.")
    print(f"[DEME] Hand mesh contact proxies loaded: {len(hand_proxy_trackers)}.")
    return deme_solver, clump_tracker, hand_proxy_trackers, len(clump_positions)


def _add_static_tray(builder: newton.ModelBuilder) -> list[int]:
    """Add one non-moving shallow tray and return its shape indices."""
    hx, hy, hz = TRAY_FLOOR_HALF_EXTENTS
    wall_half_height = 0.5 * TRAY_WALL_HEIGHT
    wall_z = TRAY_WALL_HEIGHT * 0.5
    wall_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        ke=SHAPE_CONTACT_STIFFNESS,
        kd=SHAPE_CONTACT_DAMPING,
        mu=TRAY_FRICTION,
        margin=SHAPE_CONTACT_MARGIN,
        gap=SHAPE_CONTACT_GAP,
    )
    shape_indices = [
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(*TRAY_FLOOR_CENTER), wp.quat_identity()),
            hx=hx,
            hy=hy,
            hz=hz,
            cfg=wall_cfg,
            label="tray_floor",
        )
    ]
    for label, center, extents in (
        ("tray_wall_left", (-hx, TRAY_FLOOR_CENTER[1], wall_z), (TRAY_WALL_THICKNESS, hy, wall_half_height)),
        ("tray_wall_right", (hx, TRAY_FLOOR_CENTER[1], wall_z), (TRAY_WALL_THICKNESS, hy, wall_half_height)),
        ("tray_wall_near", (0.0, TRAY_FLOOR_CENTER[1] - hy, wall_z), (hx, TRAY_WALL_THICKNESS, wall_half_height)),
        ("tray_wall_far", (0.0, TRAY_FLOOR_CENTER[1] + hy, wall_z), (hx, TRAY_WALL_THICKNESS, wall_half_height)),
    ):
        shape_indices.append(
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(wp.vec3(*center), wp.quat_identity()),
                hx=extents[0],
                hy=extents[1],
                hz=extents[2],
                cfg=wall_cfg,
                label=label,
            )
        )
    return shape_indices


def _configure_hand_template():
    """Load one downward-facing hand and tray."""
    hand = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(hand)
    hand.default_shape_cfg.ke = SHAPE_CONTACT_STIFFNESS
    hand.default_shape_cfg.kd = SHAPE_CONTACT_DAMPING
    hand.default_shape_cfg.margin = SHAPE_CONTACT_MARGIN
    hand.default_shape_cfg.gap = SHAPE_CONTACT_GAP

    asset_directory = mophi.download_newton_asset(HAND_ASSET_NAME, [HAND_ASSET_RELATIVE_PATH])
    asset_path = asset_directory / HAND_ASSET_RELATIVE_PATH
    hand_rotation = wp.quat_rpy(*HAND_DOWNWARD_ROTATION_RPY)
    hand.add_usd(
        str(asset_path),
        xform=wp.transform(wp.vec3(0.0, 0.0, HAND_BASE_HEIGHT), hand_rotation),
        floating=True,
        enable_self_collisions=ENABLE_HAND_SELF_COLLISIONS,
        ignore_paths=HAND_ASSET_IGNORE_PATHS,
        hide_collision_shapes=True,
    )
    root_joint_indices = [
        joint_idx
        for joint_idx, (joint_type, parent_idx) in enumerate(zip(hand.joint_type, hand.joint_parent))
        if joint_type == newton.JointType.FREE and parent_idx == -1
    ]
    if len(root_joint_indices) != 1:
        mophi.fatal(f"Expected one free Allegro root joint, found {len(root_joint_indices)}.")
    root_joint_idx = root_joint_indices[0]
    root_body_idx = int(hand.joint_child[root_joint_idx])
    hand.body_flags[root_body_idx] = int(newton.BodyFlags.KINEMATIC)
    root_q_start = int(hand.joint_q_start[root_joint_idx])
    root_qd_start = int(hand.joint_qd_start[root_joint_idx])

    finger_dof_indices = []
    centers = []
    amplitudes = []
    phases = []
    for label_suffix, center, amplitude, phase_cycles in FINGER_MOTION_SPECS:
        matches = [joint_idx for joint_idx, label in enumerate(hand.joint_label) if label.endswith(f"/{label_suffix}")]
        if len(matches) != 1:
            mophi.fatal(f"Expected one Allegro joint ending with '{label_suffix}', found {len(matches)}.")
        joint_idx = matches[0]
        q_idx = int(hand.joint_q_start[joint_idx])
        dof_idx = int(hand.joint_qd_start[joint_idx])
        finger_dof_indices.append(dof_idx)
        centers.append(center)
        amplitudes.append(amplitude)
        phases.append(phase_cycles)
        hand.joint_q[q_idx] = center
        hand.joint_target_pos[dof_idx] = center
        hand.joint_target_ke[dof_idx] = FINGER_DRIVE_STIFFNESS
        hand.joint_target_kd[dof_idx] = FINGER_DRIVE_DAMPING
        hand.joint_target_mode[dof_idx] = int(JointTargetMode.POSITION)
        hand.joint_armature[dof_idx] = FINGER_JOINT_ARMATURE

    tray_shape_indices = _add_static_tray(hand)
    return (
        hand,
        np.asarray(finger_dof_indices, dtype=np.int32),
        np.asarray(centers, dtype=np.float32),
        np.asarray(amplitudes, dtype=np.float32),
        np.asarray(phases, dtype=np.float32),
        root_q_start,
        root_qd_start,
        tray_shape_indices,
        asset_path,
    )


def _configure_replicated_controls(
    hand: newton.ModelBuilder,
    local_finger_dof_indices: np.ndarray,
    local_centers: np.ndarray,
    local_amplitudes: np.ndarray,
    local_phases: np.ndarray,
    local_root_q_start: int,
    local_root_qd_start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    return _configure_replicated_controls_shared(
        hand,
        local_finger_dof_indices,
        local_centers,
        local_amplitudes,
        local_phases,
        local_root_q_start,
        local_root_qd_start,
        HAND_COUNT,
        GRASP_PROFILES,
        HAND_PHASE_SPREAD_CYCLES,
    )


def _write_run_metadata(
    model,
    asset_path: Path,
    completed_frames: int,
    elapsed_wall_seconds: float,
    controlled_dof_count: int,
    clump_count: int,
    lifted_clump_count: int,
    displaced_clump_count: int,
    deme_hand_proxy_count: int,
    profile_names: list[str],
) -> None:
    completed_sim_seconds = completed_frames * FRAME_DT
    metadata = {
        "demo": "flexible_hand_deme",
        "baseline": "Newton example_robot_allegro_hand.py",
        "asset_path": str(asset_path),
        "hand_count": HAND_COUNT,
        "granular_backend": "DEME ellipsoidal clumps",
        "clump_count": clump_count,
        "lifted_clump_count": lifted_clump_count,
        "displaced_clump_count": displaced_clump_count,
        "deme_hand_proxy_count": deme_hand_proxy_count,
        "deme_hand_proxy_geometry": "Newton convex contact meshes",
        "deme_settle_time": DEME_SETTLE_TIME,
        "deme_hcp_sample_spacing": DEME_HCP_SAMPLE_SPACING,
        "deme_initial_velocity_random_seed": DEME_INITIAL_VELOCITY_RANDOM_SEED,
        "grasp_profiles": sorted(set(profile_names)),
        "hand_descend_start_seconds": HAND_DESCEND_START_SECONDS,
        "hand_descend_duration_seconds": HAND_DESCEND_DURATION_SECONDS,
        "hand_descend_depth": HAND_DESCEND_DEPTH,
        "hand_lift_start_seconds": HAND_LIFT_START_SECONDS,
        "hand_lift_duration_seconds": HAND_LIFT_DURATION_SECONDS,
        "hand_lift_height": HAND_LIFT_HEIGHT,
        "controlled_finger_dofs": controlled_dof_count,
        "bodies": model.body_count,
        "joints": model.joint_count,
        "show_laboratory_environment": SHOW_LABORATORY_ENVIRONMENT,
        "render_settling_phase": RENDER_SETTLING_PHASE,
        "completed_frames": completed_frames,
        "completed_sim_seconds": completed_sim_seconds,
        "elapsed_wall_seconds": elapsed_wall_seconds,
        "newton_dt": NEWTON_DT,
        "deme_dt": DEME_DT,
        "render_fps": RENDER_FPS,
    }
    RUN_METADATA_OUTPUT_PATH.write_text(json.dumps(metadata, indent=4) + "\n", encoding="utf-8")


def main() -> None:
    if not hasattr(mophi, "NewtonXLBDEMCoupler"):
        mophi.fatal("mophi.NewtonXLBDEMCoupler is unavailable. Build with -DMOPHI_BUILD_NEWTON_XLB_DEM=ON.")
    mophi.check_newton_warp_mujoco_versions(
        mophi.REQUIRED_NEWTON_VERSION,
        mophi.REQUIRED_WARP_VERSION,
        mophi.REQUIRED_MUJOCO_VERSION,
    )
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    wp.init()

    (
        hand,
        local_finger_dof_indices,
        local_centers,
        local_amplitudes,
        local_phases,
        local_root_q_start,
        local_root_qd_start,
        local_tray_shape_indices,
        asset_path,
    ) = _configure_hand_template()
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.replicate(hand, HAND_COUNT, spacing=HAND_SPACING)
    target_indices, centers, amplitudes, phases, root_q_starts, root_qd_starts, profile_names = (
        _configure_replicated_controls(
            hand,
            local_finger_dof_indices,
            local_centers,
            local_amplitudes,
            local_phases,
            local_root_q_start,
            local_root_qd_start,
        )
    )
    builder.add_ground_plane()
    model = builder.finalize()
    model.set_gravity((0.0, 0.0, -9.81))

    solver = newton.solvers.SolverMuJoCo(
        model,
        solver="newton",
        integrator="implicitfast",
        njmax=MAX_CONSTRAINTS_PER_HAND,
        nconmax=MAX_CONTACTS_PER_HAND,
        impratio=10.0,
        cone="elliptic",
        iterations=SOLVER_ITERATIONS,
        ls_iterations=SOLVER_LINE_SEARCH_ITERATIONS,
        use_mujoco_contacts=False,
    )
    deme_solver, deme_clump_tracker, deme_hand_proxy_trackers, clump_count = _build_deme_granular_system(hand, model)
    coupler = mophi.NewtonXLBDEMCoupler()
    coupler.initialize(model, solver, None, deme_solver, NEWTON_DT)

    device = wp.get_device()
    hand_proxy_pose_exchange = NewtonDEMEOwnerPoseExchange()
    hand_proxy_pose_exchange.initialize(
        model,
        deme_solver,
        NewtonDEMEOwnerMap(
            [body_idx for body_idx, _ in deme_hand_proxy_trackers],
            [int(tracker.GetOwnerID()) for _, tracker in deme_hand_proxy_trackers],
        ),
        device,
    )
    clump_pose_exchange = NewtonDEMEParticleExchange()
    clump_pose_exchange.initialize(
        deme_solver,
        int(deme_clump_tracker.GetOwnerID()),
        clump_count,
        device,
    )
    target_indices_wp = wp.array(target_indices, dtype=wp.int32, device=device)
    centers_wp = wp.array(centers, dtype=wp.float32, device=device)
    amplitudes_wp = wp.array(amplitudes, dtype=wp.float32, device=device)
    phases_wp = wp.array(phases, dtype=wp.float32, device=device)
    root_q_starts_wp = wp.array(root_q_starts, dtype=wp.int32, device=device)
    root_qd_starts_wp = wp.array(root_qd_starts, dtype=wp.int32, device=device)
    initial_root_heights_wp = wp.array(
        coupler.newton_state_0.joint_q.numpy()[root_q_starts + 2],
        dtype=wp.float32,
        device=device,
    )
    controlled_dof_count = len(target_indices)

    vis = mophi.create_opengl_visualizer_or_fatal(model)
    vis.set_world_offsets(VIEWER_WORLD_SPACING)
    vis.set_camera(wp.vec3(*CAMERA_POSITION), CAMERA_PITCH, CAMERA_YAW)
    if SHOW_LABORATORY_ENVIRONMENT:
        mophi.log_laboratory_environment(
            vis,
            room_center=LABORATORY_ROOM_CENTER,
            room_half_extents=LABORATORY_ROOM_HALF_EXTENTS,
        )
    shape_colors = {}
    for hand_idx in range(HAND_COUNT):
        shape_offset = hand_idx * hand.shape_count
        for local_shape_idx in local_tray_shape_indices:
            shape_colors[shape_offset + local_shape_idx] = TRAY_COLOR
    vis.update_shape_colors(shape_colors)
    mophi.log_orientation_and_scale_reference(
        vis,
        axis_origin=REFERENCE_AXIS_ORIGIN,
        scale_bar_center=REFERENCE_SCALE_BAR_CENTER,
    )

    deme_clump_colors_wp = wp.array(
        np.tile(np.asarray(DEME_CLUMP_COLOR, dtype=np.float32), (clump_count, 1)),
        dtype=wp.vec3,
        device=device,
    )
    movie_writer = imageio.get_writer(MOVIE_OUTPUT_PATH, fps=MOVIE_FPS) if SAVE_MOVIE else None
    completed_frames = 0
    final_body_q = None
    final_joint_q = None
    start_wall_time = time.perf_counter()
    print(f"[Hands] Simulating {HAND_COUNT} downward-facing Allegro hands.")
    print(
        f"[Granular stage] Added {clump_count} DEME ellipsoidal clumps and "
        f"{len(deme_hand_proxy_trackers)} kinematic hand mesh proxies."
    )
    print(f"[Granular stage] Advancing DEME with {DEME_SUBSTEPS} substeps per Newton step.")
    print(
        f"[Root motion] Descending {HAND_DESCEND_DEPTH:.2f} m, then lifting "
        f"{HAND_LIFT_HEIGHT:.2f} m after {HAND_LIFT_START_SECONDS:.2f} s."
    )

    try:
        print(f"[DEME] Settling clumps for {DEME_SETTLE_TIME:.2f} s with hand contact disabled.")
        if RENDER_SETTLING_PHASE:
            settling_time = 0.0
            settling_frame_count = 0
            while settling_time < DEME_SETTLE_TIME - SETTLING_TIME_EPSILON:
                if not vis.is_running():
                    print(f"[Viewer] Window closed during settling after {settling_frame_count} frame(s).")
                    break
                settling_dt = min(FRAME_DT, DEME_SETTLE_TIME - settling_time)
                deme_solver.DoDynamics(settling_dt)
                settling_time += settling_dt
                settling_frame_count += 1

                vis.begin_frame(settling_time - DEME_SETTLE_TIME)
                vis.log_state(coupler.newton_state_0)
                settling_clump_positions, settling_clump_orientations = clump_pose_exchange.read_deme_particle_poses()
                vis.log_clumps(
                    "deme_granular_material",
                    settling_clump_positions,
                    settling_clump_orientations,
                    DEME_CLUMP_SPHERE_RADII,
                    DEME_CLUMP_SPHERE_OFFSETS,
                    colors=deme_clump_colors_wp,
                )
                vis.end_frame()
                if movie_writer is not None:
                    movie_writer.append_data(vis.get_frame().numpy())
        else:
            deme_solver.DoDynamics(DEME_SETTLE_TIME)

        settled_clump_positions_wp, _ = clump_pose_exchange.read_deme_particle_poses()
        settled_clump_positions = settled_clump_positions_wp.numpy()
        print(_format_clump_position_bounds("[DEME] Bounds after settling", settled_clump_positions))
        initial_clump_positions = settled_clump_positions.copy()
        hand_proxy_pose_exchange.set_deme_owner_pose_from_newton(coupler.newton_state_0)
        deme_solver.ChangeFamily(DEME_HAND_PROXY_SLEEP_FAMILY, DEME_HAND_PROXY_ACTIVE_FAMILY)
        print("[DEME] Settling complete; hand mesh contact enabled.")

        for frame in range(NUM_FRAMES):
            if not vis.is_running():
                break
            sim_time = frame * FRAME_DT
            wp.launch(
                _update_finger_targets,
                dim=controlled_dof_count,
                inputs=[
                    target_indices_wp,
                    centers_wp,
                    amplitudes_wp,
                    phases_wp,
                    model.joint_limit_lower,
                    model.joint_limit_upper,
                    sim_time,
                    FINGER_MOTION_FREQUENCY,
                ],
                outputs=[coupler.newton_control.joint_target_pos],
                device=device,
            )
            for substep in range(SIM_SUBSTEPS):
                substep_time = sim_time + substep * NEWTON_DT
                wp.launch(
                    _update_hand_root_descend_lift,
                    dim=HAND_COUNT,
                    inputs=[
                        root_q_starts_wp,
                        root_qd_starts_wp,
                        initial_root_heights_wp,
                        substep_time,
                        HAND_DESCEND_START_SECONDS,
                        HAND_DESCEND_DURATION_SECONDS,
                        HAND_DESCEND_DEPTH,
                        HAND_LIFT_START_SECONDS,
                        HAND_LIFT_DURATION_SECONDS,
                        HAND_LIFT_HEIGHT,
                    ],
                    outputs=[
                        coupler.newton_state_0.joint_q,
                        coupler.newton_state_0.joint_qd,
                    ],
                    device=device,
                )
                newton.eval_fk(
                    model,
                    coupler.newton_state_0.joint_q,
                    coupler.newton_state_0.joint_qd,
                    coupler.newton_state_0,
                )
                coupler.step_newton()
                hand_proxy_pose_exchange.set_deme_owner_pose_from_newton(coupler.newton_state_0)
                for _ in range(DEME_SUBSTEPS):
                    coupler.step_deme()

            vis.begin_frame(sim_time)
            vis.log_state(coupler.newton_state_0)
            current_clump_positions, current_clump_orientations = clump_pose_exchange.read_deme_particle_poses()
            if frame == 0:
                print(
                    _format_clump_position_bounds(
                        "[DEME] Bounds at first rendered frame", current_clump_positions.numpy()
                    )
                )
            vis.log_clumps(
                "deme_granular_material",
                current_clump_positions,
                current_clump_orientations,
                DEME_CLUMP_SPHERE_RADII,
                DEME_CLUMP_SPHERE_OFFSETS,
                colors=deme_clump_colors_wp,
            )
            vis.end_frame()
            completed_frames = frame + 1
            if movie_writer is not None:
                movie_writer.append_data(vis.get_frame().numpy())
    finally:
        elapsed_wall_seconds = time.perf_counter() - start_wall_time
        final_body_q = coupler.newton_state_0.body_q.numpy()
        final_clump_positions_wp, _ = clump_pose_exchange.read_deme_particle_poses()
        final_clump_positions = final_clump_positions_wp.numpy()
        if SAVE_FINAL_STATE:
            final_joint_q = coupler.newton_state_0.joint_q.numpy()
        if movie_writer is not None:
            movie_writer.close()
        vis.close()
        hand_proxy_pose_exchange.finalize()
        clump_pose_exchange.finalize()
        coupler.finalize()

    if SAVE_FINAL_STATE:
        np.savez_compressed(
            FINAL_STATE_OUTPUT_PATH,
            body_q=final_body_q,
            joint_q=final_joint_q,
            clump_positions=final_clump_positions,
        )
    lifted_clump_count = int(
        np.count_nonzero(final_clump_positions[:, 2] > initial_clump_positions[:, 2] + GRANULAR_LIFT_THRESHOLD)
    )
    clump_displacements = np.linalg.norm(final_clump_positions - initial_clump_positions, axis=1)
    displaced_clump_count = int(np.count_nonzero(clump_displacements > GRANULAR_DISPLACEMENT_THRESHOLD))
    _write_run_metadata(
        model,
        asset_path,
        completed_frames,
        elapsed_wall_seconds,
        controlled_dof_count,
        clump_count,
        lifted_clump_count,
        displaced_clump_count,
        len(deme_hand_proxy_trackers),
        profile_names,
    )
    completed_sim_seconds = completed_frames * FRAME_DT
    realtime_factor = completed_sim_seconds / elapsed_wall_seconds if elapsed_wall_seconds > 0.0 else 0.0
    print(f"[Performance] Simulated {completed_sim_seconds:.2f} s in {elapsed_wall_seconds:.2f} wall-clock s.")
    print(f"[Performance] Aggregate realtime factor: {realtime_factor:.2f}x.")
    print(f"[DEME clumps] Lifted above threshold at final frame: {lifted_clump_count}.")
    print(f"[DEME clumps] Displaced beyond threshold at final frame: {displaced_clump_count}.")
    print(f"[Output] Saved run metadata to '{RUN_METADATA_OUTPUT_PATH}'.")


if __name__ == "__main__":
    main()
