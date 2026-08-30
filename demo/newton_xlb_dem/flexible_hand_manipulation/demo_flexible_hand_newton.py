"""Many downward-facing articulated hands attempting to grasp loose Newton cubes.

This demo keeps the scalable 64-hand Newton-only path. For the smaller
one-hand DEME clump version, run ``demo_flexible_hand_deme.py``.
"""

import json
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np
import warp as wp

import mophi
import newton
from newton import JointTargetMode

# =============================================================================
# Simulation configuration
# All user-tunable constants are defined here.
# =============================================================================

# -- Timing -------------------------------------------------------------------
RENDER_FPS = 25
NEWTON_DT = 1.0 / 200.0
SIM_DURATION_SECONDS = 5.0

# -- Hand replication ---------------------------------------------------------
HAND_COUNT = 64
HAND_SPACING = (1.4, 1.4, 0.0)
HAND_BASE_HEIGHT = 0.75
HAND_DOWNWARD_ROTATION_RPY = (np.pi, 0.0, 0.0)
HAND_ASSET_NAME = "wonik_allegro"
HAND_ASSET_RELATIVE_PATH = Path("usd") / "allegro_left_hand_with_cube.usda"
HAND_ASSET_IGNORE_PATHS = [".*Dummy", ".*CollisionPlane", ".*DexCube", ".*root_joint"]
ENABLE_HAND_SELF_COLLISIONS = False

# -- Prescribed descend-grasp-lift motion -------------------------------------
# Move the kinematic hand root into the cube bed before lifting it back upward,
# while maintaining the varied finger motions.
HAND_DESCEND_START_SECONDS = 0.25
HAND_DESCEND_DURATION_SECONDS = 1.0
HAND_DESCEND_DEPTH = 0.12
HAND_LIFT_START_SECONDS = 3.0
HAND_LIFT_DURATION_SECONDS = 1.5
HAND_LIFT_HEIGHT = 0.65

# -- Newton cube bed ----------------------------------------------------------
CUBE_GRID_X = (-0.065, 0.065)
CUBE_GRID_Y = (0.075, 0.205)
CUBE_LAYER_Z = (0.075, 0.155)
CUBE_HALF_EXTENT = 0.035
CUBE_DENSITY = 450.0
CUBE_FRICTION = 0.9
CUBE_CONTACT_STIFFNESS = 5.0e2  # Increase by 1 order of mag will cause cubes to fly around
CUBE_CONTACT_DAMPING = 3.0e2
GRANULAR_LIFT_THRESHOLD = 0.08
GRANULAR_DISPLACEMENT_THRESHOLD = 0.03
CUBE_COLORS = (
    (0.96, 0.62, 0.12),
    (0.18, 0.68, 0.92),
    (0.32, 0.78, 0.42),
    (0.92, 0.30, 0.52),
)

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
CAMERA_POSITION = (8.0, -9.5, 6.5)
CAMERA_PITCH = -30.0
CAMERA_YAW = 140.0
VIEWER_WORLD_SPACING = (0.0, 0.0, 0.0)
REFERENCE_AXIS_ORIGIN = (-5.4, -5.4, 0.02)
REFERENCE_SCALE_BAR_CENTER = (-4.4, -5.4, 0.06)
SHOW_LABORATORY_ENVIRONMENT = True
LABORATORY_ROOM_CENTER = (0.0, 0.0, 0.0)
LABORATORY_ROOM_HALF_EXTENTS = (6.0, 6.0, 1.8)

# -- Output -------------------------------------------------------------------
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "output" / Path(__file__).stem
SAVE_MOVIE = True
MOVIE_OUTPUT_PATH = OUTPUT_DIRECTORY / "demo_flexible_hand_newton.mp4"
MOVIE_FPS = RENDER_FPS
SAVE_FINAL_STATE = True
FINAL_STATE_OUTPUT_PATH = OUTPUT_DIRECTORY / "final_state.npz"
RUN_METADATA_OUTPUT_PATH = OUTPUT_DIRECTORY / "run_metadata.json"

# -- Derived constants --------------------------------------------------------
FRAME_DT = 1.0 / RENDER_FPS
NUM_FRAMES = int(round(SIM_DURATION_SECONDS / FRAME_DT))
SIM_SUBSTEPS = int(round(FRAME_DT / NEWTON_DT))
CUBES_PER_HAND = len(CUBE_GRID_X) * len(CUBE_GRID_Y) * len(CUBE_LAYER_Z)
if not np.isclose(SIM_SUBSTEPS * NEWTON_DT, FRAME_DT, rtol=0.0, atol=1.0e-12):
    raise ValueError("FRAME_DT must be an integer multiple of NEWTON_DT.")


@wp.kernel
def _update_finger_targets(
    target_indices: wp.array(dtype=wp.int32),
    centers: wp.array(dtype=wp.float32),
    amplitudes: wp.array(dtype=wp.float32),
    phase_cycles: wp.array(dtype=wp.float32),
    lower_limits: wp.array(dtype=wp.float32),
    upper_limits: wp.array(dtype=wp.float32),
    sim_time: float,
    frequency: float,
    targets: wp.array(dtype=wp.float32),
):
    target_idx = wp.tid()
    target_dof_idx = target_indices[target_idx]
    phase = 2.0 * wp.pi * (frequency * sim_time + phase_cycles[target_idx])
    target = centers[target_idx] + amplitudes[target_idx] * wp.sin(phase)
    targets[target_dof_idx] = wp.clamp(target, lower_limits[target_dof_idx], upper_limits[target_dof_idx])


@wp.kernel
def _update_hand_root_descend_lift(
    root_q_starts: wp.array(dtype=wp.int32),
    root_qd_starts: wp.array(dtype=wp.int32),
    initial_root_heights: wp.array(dtype=wp.float32),
    sim_time: float,
    descend_start: float,
    descend_duration: float,
    descend_depth: float,
    lift_start: float,
    lift_duration: float,
    lift_height: float,
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
):
    hand_idx = wp.tid()
    q_start = root_q_starts[hand_idx]
    qd_start = root_qd_starts[hand_idx]
    descend_fraction = wp.clamp((sim_time - descend_start) / descend_duration, 0.0, 1.0)
    descend_smooth = descend_fraction * descend_fraction * (3.0 - 2.0 * descend_fraction)
    descend_velocity = 6.0 * descend_fraction * (1.0 - descend_fraction) * descend_depth / descend_duration
    if sim_time <= descend_start or sim_time >= descend_start + descend_duration:
        descend_velocity = 0.0

    lift_fraction = wp.clamp((sim_time - lift_start) / lift_duration, 0.0, 1.0)
    lift_smooth = lift_fraction * lift_fraction * (3.0 - 2.0 * lift_fraction)
    lift_velocity = 6.0 * lift_fraction * (1.0 - lift_fraction) * lift_height / lift_duration
    if sim_time <= lift_start or sim_time >= lift_start + lift_duration:
        lift_velocity = 0.0

    joint_q[q_start + 2] = initial_root_heights[hand_idx] - descend_depth * descend_smooth + lift_height * lift_smooth
    joint_qd[qd_start + 0] = 0.0
    joint_qd[qd_start + 1] = 0.0
    joint_qd[qd_start + 2] = -descend_velocity + lift_velocity
    joint_qd[qd_start + 3] = 0.0
    joint_qd[qd_start + 4] = 0.0
    joint_qd[qd_start + 5] = 0.0


def _add_static_tray(builder: newton.ModelBuilder) -> list[int]:
    """Add one non-moving shallow tray and return its shape indices."""
    hx, hy, hz = TRAY_FLOOR_HALF_EXTENTS
    wall_half_height = 0.5 * TRAY_WALL_HEIGHT
    wall_z = TRAY_WALL_HEIGHT * 0.5
    wall_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        ke=SHAPE_CONTACT_STIFFNESS,
        kd=SHAPE_CONTACT_DAMPING,
        mu=CUBE_FRICTION,
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


def _add_newton_cube_bed(builder: newton.ModelBuilder) -> tuple[list[int], list[int]]:
    """Add loose Newton cubes."""
    cube_cfg = newton.ModelBuilder.ShapeConfig(
        density=CUBE_DENSITY,
        ke=CUBE_CONTACT_STIFFNESS,
        kd=CUBE_CONTACT_DAMPING,
        mu=CUBE_FRICTION,
        margin=SHAPE_CONTACT_MARGIN,
        gap=SHAPE_CONTACT_GAP,
    )
    body_indices = []
    shape_indices = []
    cube_idx = 0
    for z in CUBE_LAYER_Z:
        for y in CUBE_GRID_Y:
            for x in CUBE_GRID_X:
                body_idx = builder.add_body(
                    xform=wp.transform(wp.vec3(x, y, z), wp.quat_identity()),
                    label=f"granular_cube_{cube_idx}",
                )
                shape_idx = builder.add_shape_box(
                    body=body_idx,
                    hx=CUBE_HALF_EXTENT,
                    hy=CUBE_HALF_EXTENT,
                    hz=CUBE_HALF_EXTENT,
                    cfg=cube_cfg,
                    label=f"granular_cube_shape_{cube_idx}",
                )
                body_indices.append(body_idx)
                shape_indices.append(shape_idx)
                cube_idx += 1
    return body_indices, shape_indices


def _configure_hand_template():
    """Load one downward-facing hand, tray, and loose Newton cube bed."""
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
    cube_body_indices, cube_shape_indices = _add_newton_cube_bed(hand)
    return (
        hand,
        np.asarray(finger_dof_indices, dtype=np.int32),
        np.asarray(centers, dtype=np.float32),
        np.asarray(amplitudes, dtype=np.float32),
        np.asarray(phases, dtype=np.float32),
        root_q_start,
        root_qd_start,
        cube_body_indices,
        cube_shape_indices,
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
    """Expand controls and vary the grasp behavior across replicated hands."""
    target_indices = []
    centers = []
    amplitudes = []
    phases = []
    root_q_starts = []
    root_qd_starts = []
    profile_names = []
    for hand_idx in range(HAND_COUNT):
        profile_name, closure_scale, motion_scale, profile_phase = GRASP_PROFILES[hand_idx % len(GRASP_PROFILES)]
        profile_names.append(profile_name)
        world_dof_offset = hand_idx * hand.joint_dof_count
        world_coord_offset = hand_idx * hand.joint_coord_count
        hand_phase = hand_idx * (HAND_PHASE_SPREAD_CYCLES / HAND_COUNT) + profile_phase
        root_q_starts.append(world_coord_offset + local_root_q_start)
        root_qd_starts.append(world_dof_offset + local_root_qd_start)
        for local_idx, center, amplitude, phase in zip(
            local_finger_dof_indices,
            local_centers * closure_scale,
            local_amplitudes * motion_scale,
            local_phases + hand_phase,
        ):
            target_indices.append(world_dof_offset + int(local_idx))
            centers.append(center)
            amplitudes.append(amplitude)
            phases.append(phase)
    return (
        np.asarray(target_indices, dtype=np.int32),
        np.asarray(centers, dtype=np.float32),
        np.asarray(amplitudes, dtype=np.float32),
        np.asarray(phases, dtype=np.float32),
        np.asarray(root_q_starts, dtype=np.int32),
        np.asarray(root_qd_starts, dtype=np.int32),
        profile_names,
    )


def _write_run_metadata(
    model,
    asset_path: Path,
    completed_frames: int,
    elapsed_wall_seconds: float,
    controlled_dof_count: int,
    cube_count: int,
    lifted_cube_count: int,
    displaced_cube_count: int,
    profile_names: list[str],
) -> None:
    completed_sim_seconds = completed_frames * FRAME_DT
    metadata = {
        "demo": "flexible_hand_newton",
        "baseline": "Newton example_robot_allegro_hand.py",
        "asset_path": str(asset_path),
        "hand_count": HAND_COUNT,
        "granular_backend": "Newton cubes",
        "cubes_per_hand": CUBES_PER_HAND,
        "cube_count": cube_count,
        "lifted_cube_count": lifted_cube_count,
        "displaced_cube_count": displaced_cube_count,
        "grasp_profiles": sorted(set(profile_names)),
        "hand_descend_start_seconds": HAND_DESCEND_START_SECONDS,
        "hand_descend_duration_seconds": HAND_DESCEND_DURATION_SECONDS,
        "hand_descend_depth": HAND_DESCEND_DEPTH,
        "hand_lift_start_seconds": HAND_LIFT_START_SECONDS,
        "hand_lift_duration_seconds": HAND_LIFT_DURATION_SECONDS,
        "hand_lift_height": HAND_LIFT_HEIGHT,
        "cube_contact_stiffness": CUBE_CONTACT_STIFFNESS,
        "cube_contact_damping": CUBE_CONTACT_DAMPING,
        "controlled_finger_dofs": controlled_dof_count,
        "bodies": model.body_count,
        "joints": model.joint_count,
        "show_laboratory_environment": SHOW_LABORATORY_ENVIRONMENT,
        "completed_frames": completed_frames,
        "completed_sim_seconds": completed_sim_seconds,
        "elapsed_wall_seconds": elapsed_wall_seconds,
        "newton_dt": NEWTON_DT,
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
        local_cube_body_indices,
        local_cube_shape_indices,
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
    cube_count = HAND_COUNT * CUBES_PER_HAND
    coupler = mophi.NewtonXLBDEMCoupler()
    coupler.initialize(model, solver, None, None, NEWTON_DT)

    device = wp.get_device()
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
        cube_color = CUBE_COLORS[hand_idx % len(CUBE_COLORS)]
        for local_shape_idx in local_cube_shape_indices:
            shape_colors[shape_offset + local_shape_idx] = cube_color
        for local_shape_idx in local_tray_shape_indices:
            shape_colors[shape_offset + local_shape_idx] = TRAY_COLOR
    vis.update_shape_colors(shape_colors)
    mophi.log_orientation_and_scale_reference(
        vis,
        axis_origin=REFERENCE_AXIS_ORIGIN,
        scale_bar_center=REFERENCE_SCALE_BAR_CENTER,
    )

    cube_body_indices = np.asarray(
        [
            hand_idx * hand.body_count + local_body_idx
            for hand_idx in range(HAND_COUNT)
            for local_body_idx in local_cube_body_indices
        ],
        dtype=np.int32,
    )
    initial_cube_positions = coupler.newton_state_0.body_q.numpy()[cube_body_indices, :3].copy()
    movie_writer = imageio.get_writer(MOVIE_OUTPUT_PATH, fps=MOVIE_FPS) if SAVE_MOVIE else None
    completed_frames = 0
    final_body_q = None
    final_joint_q = None
    start_wall_time = time.perf_counter()
    print(f"[Hands] Simulating {HAND_COUNT} downward-facing Allegro hands.")
    print(f"[Granular stage] Added {cube_count} loose Newton cubes in shallow trays.")
    print(
        f"[Root motion] Descending {HAND_DESCEND_DEPTH:.2f} m, then lifting "
        f"{HAND_LIFT_HEIGHT:.2f} m after {HAND_LIFT_START_SECONDS:.2f} s."
    )

    try:
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

            vis.begin_frame(sim_time)
            vis.log_state(coupler.newton_state_0)
            vis.end_frame()
            completed_frames = frame + 1
            if movie_writer is not None:
                movie_writer.append_data(vis.get_frame().numpy())
    finally:
        elapsed_wall_seconds = time.perf_counter() - start_wall_time
        final_body_q = coupler.newton_state_0.body_q.numpy()
        final_cube_positions = final_body_q[cube_body_indices, :3].copy()
        if SAVE_FINAL_STATE:
            final_joint_q = coupler.newton_state_0.joint_q.numpy()
        if movie_writer is not None:
            movie_writer.close()
        vis.close()
        coupler.finalize()

    if SAVE_FINAL_STATE:
        np.savez_compressed(
            FINAL_STATE_OUTPUT_PATH,
            body_q=final_body_q,
            joint_q=final_joint_q,
            cube_positions=final_cube_positions,
        )
    lifted_cube_count = int(
        np.count_nonzero(final_cube_positions[:, 2] > initial_cube_positions[:, 2] + GRANULAR_LIFT_THRESHOLD)
    )
    cube_displacements = np.linalg.norm(final_cube_positions - initial_cube_positions, axis=1)
    displaced_cube_count = int(np.count_nonzero(cube_displacements > GRANULAR_DISPLACEMENT_THRESHOLD))
    _write_run_metadata(
        model,
        asset_path,
        completed_frames,
        elapsed_wall_seconds,
        controlled_dof_count,
        cube_count,
        lifted_cube_count,
        displaced_cube_count,
        profile_names,
    )
    completed_sim_seconds = completed_frames * FRAME_DT
    realtime_factor = completed_sim_seconds / elapsed_wall_seconds if elapsed_wall_seconds > 0.0 else 0.0
    print(f"[Performance] Simulated {completed_sim_seconds:.2f} s in {elapsed_wall_seconds:.2f} wall-clock s.")
    print(f"[Performance] Aggregate realtime factor: {realtime_factor:.2f}x.")
    print(f"[Cubes] Lifted above threshold at final frame: {lifted_cube_count}.")
    print(f"[Cubes] Displaced beyond threshold at final frame: {displaced_cube_count}.")
    print(f"[Output] Saved run metadata to '{RUN_METADATA_OUTPUT_PATH}'.")


if __name__ == "__main__":
    main()
