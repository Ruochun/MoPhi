"""Many independently animated articulated-hand instances using Newton.

This demo adapts Newton 1.0.0's public ``example_robot_allegro_hand.py``
baseline for MoPhi. It replicates a contact-capable Allegro hand model and a
dynamic cup into many isolated simulation worlds, then drives every finger
joint with an individually configurable grasping motion.

The fingers are articulated rigid links rather than deformable solids. Four
repeating grasp profiles produce varied cup-contact outcomes, including
retained cups and drops.
"""

import json
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np
import warp as wp

import mophi
import newton
import newton.utils
from newton import JointTargetMode
from pxr import Usd

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
HAND_SPACING = (0.5, 0.5, 0.0)
HAND_BASE_HEIGHT = 0.5
HAND_ASSET_NAME = "wonik_allegro"
HAND_ASSET_RELATIVE_PATH = Path("usd") / "allegro_left_hand_with_cube.usda"
HAND_ASSET_IGNORE_PATHS = [".*Dummy", ".*CollisionPlane", ".*DexCube"]
ENABLE_HAND_SELF_COLLISIONS = False

# -- Cup workload -------------------------------------------------------------
CUP_ASSET_NAME = "manipulation_objects/cup"
CUP_ASSET_RELATIVE_PATH = Path("model.usda")
CUP_MESH_PRIM_PATH = "/root/Model/Model"
CUP_SCALE_PRIM_PATH = "/root/Model"
CUP_BOWL_SCALE = 1.3
# Begin clear of the fingers so contact develops during simulation instead of from initial penetration.
CUP_INITIAL_POSITION = np.array([0.0, -0.14, 1.18], dtype=np.float32)
CUP_INITIAL_ORIENTATION_XYZW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
CUP_COLORS = (
    (0.10, 0.65, 0.95),
    (0.95, 0.25, 0.55),
    (0.15, 0.80, 0.40),
    (0.95, 0.70, 0.10),
)
CUP_DENSITY = 180.0
CUP_FRICTION = 1.2
CUP_SDF_MAX_RESOLUTION = 32
CUP_DROP_DISTANCE = 0.25
CUP_HANDLE_RADIUS = 0.007
CUP_HANDLE_OUTER_X = 0.075
CUP_HANDLE_CONNECTOR_CENTER_X = 0.057
CUP_HANDLE_CONNECTOR_HALF_LENGTH = 0.018
CUP_HANDLE_VERTICAL_HALF_LENGTH = 0.026
CUP_HANDLE_CONNECTOR_Z = 0.027
# Profiles repeat across the hand array: (name, closure scale, motion scale, cup offset, cup tilt).
GRASP_PROFILES = (
    ("overclosed_drop", 1.55, 0.12, (0.000, 0.000, 0.000), 0.00),
    ("pulsing_drop", 1.30, 0.45, (0.006, 0.000, 0.006), 0.08),
    ("loose_hold", 0.85, 0.25, (-0.008, 0.008, 0.012), -0.12),
    ("open_drop", 0.20, 0.08, (0.015, -0.160, 0.025), 0.18),
)

# -- Finger drives ------------------------------------------------------------
FINGER_DRIVE_STIFFNESS = 150.0
FINGER_DRIVE_DAMPING = 5.0
FINGER_JOINT_ARMATURE = 1.0e-2
FINGER_MOTION_FREQUENCY = 0.65
HAND_PHASE_SPREAD_CYCLES = 1.0
# Each entry is: (joint-label suffix, center [rad], amplitude [rad], phase [cycles]).
# This block independently controls every joint in every articulated finger.
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
SHAPE_CONTACT_MARGIN = 0.005
SHAPE_CONTACT_GAP = 0.015
MAX_CONTACTS_PER_HAND = 450
MAX_CONSTRAINTS_PER_HAND = 1800
SOLVER_ITERATIONS = 10
SOLVER_LINE_SEARCH_ITERATIONS = 5

# -- Visualization ------------------------------------------------------------
CAMERA_POSITION = (4.8, -5.5, 4.3)
CAMERA_PITCH = -32.0
CAMERA_YAW = 140.0
REFERENCE_AXIS_ORIGIN = (-0.6, -0.6, 0.02)
REFERENCE_SCALE_BAR_CENTER = (0.4, -0.6, 0.06)
SHOW_LABORATORY_ENVIRONMENT = True
LABORATORY_ROOM_CENTER = (0.0, 0.0, 0.0)
LABORATORY_ROOM_HALF_EXTENTS = (6.0, 6.0, 1.8)

# -- Output -------------------------------------------------------------------
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "output" / Path(__file__).stem
SAVE_MOVIE = True
MOVIE_OUTPUT_PATH = OUTPUT_DIRECTORY / "demo_many_flexible_hands.mp4"
MOVIE_FPS = RENDER_FPS
SAVE_FINAL_STATE = True
FINAL_STATE_OUTPUT_PATH = OUTPUT_DIRECTORY / "final_state.npz"
RUN_METADATA_OUTPUT_PATH = OUTPUT_DIRECTORY / "run_metadata.json"

# -- Derived constants --------------------------------------------------------
FRAME_DT = 1.0 / RENDER_FPS
NUM_FRAMES = int(round(SIM_DURATION_SECONDS / FRAME_DT))
SIM_SUBSTEPS = int(round(FRAME_DT / NEWTON_DT))
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


def _load_cup_mesh() -> tuple[newton.Mesh, Path]:
    """Load Newton's public cup asset and bake its authored scale into the mesh."""
    cup_asset_directory = Path(newton.utils.download_asset(CUP_ASSET_NAME))
    cup_asset_path = cup_asset_directory / CUP_ASSET_RELATIVE_PATH
    cup_stage = Usd.Stage.Open(str(cup_asset_path))
    cup_mesh = newton.usd.get_mesh(
        cup_stage.GetPrimAtPath(CUP_MESH_PRIM_PATH),
        load_normals=True,
        face_varying_normal_conversion="vertex_splitting",
    )
    cup_scale = np.asarray(newton.usd.get_scale(cup_stage.GetPrimAtPath(CUP_SCALE_PRIM_PATH)), dtype=np.float32)
    if not np.allclose(cup_scale, 1.0):
        cup_mesh = cup_mesh.copy(vertices=cup_mesh.vertices * cup_scale, recompute_inertia=True)
    cup_mesh = cup_mesh.copy(vertices=cup_mesh.vertices * CUP_BOWL_SCALE, recompute_inertia=True)
    cup_mesh.build_sdf(max_resolution=CUP_SDF_MAX_RESOLUTION, margin=SHAPE_CONTACT_GAP)
    return cup_mesh, cup_asset_path


def _configure_hand_template():
    """Load one Allegro hand and add one dynamic, contact-capable cup."""
    hand = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(hand)
    hand.default_shape_cfg.ke = SHAPE_CONTACT_STIFFNESS
    hand.default_shape_cfg.kd = SHAPE_CONTACT_DAMPING
    hand.default_shape_cfg.margin = SHAPE_CONTACT_MARGIN
    hand.default_shape_cfg.gap = SHAPE_CONTACT_GAP

    asset_directory = Path(newton.utils.download_asset(HAND_ASSET_NAME))
    asset_path = asset_directory / HAND_ASSET_RELATIVE_PATH
    hand.add_usd(
        str(asset_path),
        xform=wp.transform(wp.vec3(0.0, 0.0, HAND_BASE_HEIGHT)),
        enable_self_collisions=ENABLE_HAND_SELF_COLLISIONS,
        ignore_paths=HAND_ASSET_IGNORE_PATHS,
        hide_collision_shapes=True,
    )

    finger_dof_indices = []
    centers = []
    amplitudes = []
    phases = []

    for label_suffix, center, amplitude, phase_cycles in FINGER_MOTION_SPECS:
        matches = [joint_idx for joint_idx, label in enumerate(hand.joint_label) if label.endswith(f"/{label_suffix}")]
        if len(matches) != 1:
            mophi.fatal(f"Expected one Allegro joint ending with '{label_suffix}', found {len(matches)}.")
        joint_idx = matches[0]
        dof_idx = int(hand.joint_qd_start[joint_idx])
        finger_dof_indices.append(dof_idx)
        centers.append(center)
        amplitudes.append(amplitude)
        phases.append(phase_cycles)

        hand.joint_q[dof_idx] = center
        hand.joint_target_pos[dof_idx] = center
        hand.joint_target_ke[dof_idx] = FINGER_DRIVE_STIFFNESS
        hand.joint_target_kd[dof_idx] = FINGER_DRIVE_DAMPING
        hand.joint_target_mode[dof_idx] = int(JointTargetMode.POSITION)
        hand.joint_armature[dof_idx] = FINGER_JOINT_ARMATURE

    cup_mesh, cup_asset_path = _load_cup_mesh()
    cup_body_idx = hand.add_body(
        xform=wp.transform(wp.vec3(*CUP_INITIAL_POSITION), wp.quat(*CUP_INITIAL_ORIENTATION_XYZW)),
        label="cup",
    )
    cup_shape_cfg = newton.ModelBuilder.ShapeConfig(
        density=CUP_DENSITY,
        ke=SHAPE_CONTACT_STIFFNESS,
        kd=SHAPE_CONTACT_DAMPING,
        mu=CUP_FRICTION,
        margin=SHAPE_CONTACT_MARGIN,
        gap=SHAPE_CONTACT_GAP,
    )
    cup_shape_indices = [hand.add_shape_mesh(body=cup_body_idx, mesh=cup_mesh, cfg=cup_shape_cfg)]
    connector_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * wp.pi)
    for connector_idx, connector_z in enumerate((-CUP_HANDLE_CONNECTOR_Z, CUP_HANDLE_CONNECTOR_Z)):
        cup_shape_indices.append(
            hand.add_shape_capsule(
                body=cup_body_idx,
                xform=wp.transform(
                    wp.vec3(CUP_HANDLE_CONNECTOR_CENTER_X, 0.0, connector_z),
                    connector_rotation,
                ),
                radius=CUP_HANDLE_RADIUS,
                half_height=CUP_HANDLE_CONNECTOR_HALF_LENGTH,
                cfg=cup_shape_cfg,
                label=f"cup_handle_connector_{connector_idx}",
            )
        )
    cup_shape_indices.append(
        hand.add_shape_capsule(
            body=cup_body_idx,
            xform=wp.transform(wp.vec3(CUP_HANDLE_OUTER_X, 0.0, 0.0), wp.quat_identity()),
            radius=CUP_HANDLE_RADIUS,
            half_height=CUP_HANDLE_VERTICAL_HALF_LENGTH,
            cfg=cup_shape_cfg,
            label="cup_handle_outer_grip",
        )
    )
    cup_joint_idx = hand.joint_label.index("cup_free_joint")
    cup_q_start = int(hand.joint_q_start[cup_joint_idx])

    return (
        hand,
        np.asarray(finger_dof_indices, dtype=np.int32),
        np.asarray(centers, dtype=np.float32),
        np.asarray(amplitudes, dtype=np.float32),
        np.asarray(phases, dtype=np.float32),
        cup_body_idx,
        cup_q_start,
        cup_shape_indices,
        asset_path,
        cup_asset_path,
    )


def _configure_replicated_workload(
    builder: newton.ModelBuilder,
    hand: newton.ModelBuilder,
    local_finger_dof_indices: np.ndarray,
    local_centers: np.ndarray,
    local_amplitudes: np.ndarray,
    local_phases: np.ndarray,
    cup_q_start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Apply varied grasp profiles and cup poses to the replicated worlds."""
    target_indices = []
    centers = []
    amplitudes = []
    phases = []
    profile_names = []

    for hand_idx in range(HAND_COUNT):
        profile_name, closure_scale, motion_scale, cup_offset, cup_tilt = GRASP_PROFILES[hand_idx % len(GRASP_PROFILES)]
        profile_names.append(profile_name)
        world_dof_offset = hand_idx * hand.joint_dof_count
        world_coord_offset = hand_idx * hand.joint_coord_count
        hand_phase = hand_idx * (HAND_PHASE_SPREAD_CYCLES / HAND_COUNT)
        profile_centers = local_centers * closure_scale

        for local_idx, center, amplitude, phase in zip(
            local_finger_dof_indices,
            profile_centers,
            local_amplitudes * motion_scale,
            local_phases + hand_phase,
        ):
            target_indices.append(world_dof_offset + int(local_idx))
            centers.append(center)
            amplitudes.append(amplitude)
            phases.append(phase)
            builder.joint_q[world_coord_offset + int(local_idx)] = float(center)

        world_cup_q_start = world_coord_offset + cup_q_start
        cup_offset_np = np.asarray(cup_offset, dtype=np.float32)
        builder.joint_q[world_cup_q_start : world_cup_q_start + 3] = (
            np.asarray(builder.joint_q[world_cup_q_start : world_cup_q_start + 3]) + cup_offset_np
        ).tolist()
        builder.joint_q[world_cup_q_start + 3 : world_cup_q_start + 7] = wp.quat_rpy(
            cup_tilt,
            0.5 * cup_tilt,
            0.0,
        )

    return (
        np.asarray(target_indices, dtype=np.int32),
        np.asarray(centers, dtype=np.float32),
        np.asarray(amplitudes, dtype=np.float32),
        np.asarray(phases, dtype=np.float32),
        profile_names,
    )


def _write_run_metadata(
    model,
    asset_path: Path,
    completed_frames: int,
    elapsed_wall_seconds: float,
    controlled_dof_count: int,
    cup_asset_path: Path,
    profile_names: list[str],
    dropped_cup_mask: np.ndarray,
) -> None:
    completed_sim_seconds = completed_frames * FRAME_DT
    dropped_cup_count = int(np.count_nonzero(dropped_cup_mask))
    profile_outcomes = {}
    for profile_name in sorted(set(profile_names)):
        profile_mask = np.asarray([name == profile_name for name in profile_names], dtype=bool)
        profile_dropped = int(np.count_nonzero(dropped_cup_mask & profile_mask))
        profile_outcomes[profile_name] = {
            "retained": int(np.count_nonzero(profile_mask)) - profile_dropped,
            "dropped": profile_dropped,
        }
    metadata = {
        "demo": "many_flexible_hands",
        "baseline": "Newton example_robot_allegro_hand.py",
        "asset_path": str(asset_path),
        "cup_asset_path": str(cup_asset_path),
        "hand_count": HAND_COUNT,
        "cup_count": HAND_COUNT,
        "grasp_profiles": sorted(set(profile_names)),
        "grasp_profile_outcomes": profile_outcomes,
        "dropped_cup_count": dropped_cup_count,
        "retained_cup_count": HAND_COUNT - dropped_cup_count,
        "bodies": model.body_count,
        "joints": model.joint_count,
        "controlled_finger_dofs": controlled_dof_count,
        "controlled_finger_dofs_per_hand": controlled_dof_count // HAND_COUNT,
        "contacts_enabled": True,
        "contact_objects_added": True,
        "show_laboratory_environment": SHOW_LABORATORY_ENVIRONMENT,
        "solver_iterations": SOLVER_ITERATIONS,
        "solver_line_search_iterations": SOLVER_LINE_SEARCH_ITERATIONS,
        "completed_frames": completed_frames,
        "completed_sim_seconds": completed_sim_seconds,
        "elapsed_wall_seconds": elapsed_wall_seconds,
        "simulated_seconds_per_wall_second": (
            completed_sim_seconds / elapsed_wall_seconds if elapsed_wall_seconds > 0.0 else None
        ),
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
        cup_body_idx,
        cup_q_start,
        local_cup_shape_indices,
        asset_path,
        cup_asset_path,
    ) = _configure_hand_template()
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.replicate(hand, HAND_COUNT, spacing=HAND_SPACING)
    target_indices, centers, amplitudes, phases, profile_names = _configure_replicated_workload(
        builder,
        hand,
        local_finger_dof_indices,
        local_centers,
        local_amplitudes,
        local_phases,
        cup_q_start,
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

    coupler = mophi.NewtonXLBDEMCoupler()
    coupler.initialize(model, solver, None, None, NEWTON_DT)

    controlled_dof_count = len(centers)
    if controlled_dof_count != HAND_COUNT * len(FINGER_MOTION_SPECS):
        mophi.fatal(
            f"Expected {HAND_COUNT * len(FINGER_MOTION_SPECS)} controlled finger DOFs, found {controlled_dof_count}."
        )
    device = wp.get_device()
    target_indices_wp = wp.array(target_indices, dtype=wp.int32, device=device)
    centers_wp = wp.array(centers, dtype=wp.float32, device=device)
    amplitudes_wp = wp.array(amplitudes, dtype=wp.float32, device=device)
    phases_wp = wp.array(phases, dtype=wp.float32, device=device)

    vis = mophi.create_opengl_visualizer_or_fatal(model)
    vis.set_camera(wp.vec3(*CAMERA_POSITION), CAMERA_PITCH, CAMERA_YAW)
    if SHOW_LABORATORY_ENVIRONMENT:
        mophi.log_laboratory_environment(
            vis,
            room_center=LABORATORY_ROOM_CENTER,
            room_half_extents=LABORATORY_ROOM_HALF_EXTENTS,
        )
    cup_shape_colors = {}
    for hand_idx in range(HAND_COUNT):
        cup_color = CUP_COLORS[hand_idx % len(CUP_COLORS)]
        shape_offset = hand_idx * hand.shape_count
        for local_shape_idx in local_cup_shape_indices:
            cup_shape_colors[shape_offset + local_shape_idx] = cup_color
    vis.update_shape_colors(cup_shape_colors)
    mophi.log_orientation_and_scale_reference(
        vis,
        axis_origin=REFERENCE_AXIS_ORIGIN,
        scale_bar_center=REFERENCE_SCALE_BAR_CENTER,
    )

    movie_writer = imageio.get_writer(MOVIE_OUTPUT_PATH, fps=MOVIE_FPS) if SAVE_MOVIE else None
    completed_frames = 0
    final_body_q = None
    final_joint_q = None
    cup_body_indices = np.arange(HAND_COUNT, dtype=np.int32) * hand.body_count + cup_body_idx
    initial_cup_heights = coupler.newton_state_0.body_q.numpy()[cup_body_indices, 2].copy()
    start_wall_time = time.perf_counter()
    print(
        f"[Hands] Simulating {HAND_COUNT} Allegro hands with {controlled_dof_count} independently driven finger DOFs."
    )
    print(f"[Contact] Added {HAND_COUNT} dynamic SDF-mesh cups across " f"{len(GRASP_PROFILES)} varied grasp profiles.")

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
            for _ in range(SIM_SUBSTEPS):
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
        )
    final_cup_heights = final_body_q[cup_body_indices, 2]
    dropped_cup_mask = final_cup_heights < initial_cup_heights - CUP_DROP_DISTANCE
    dropped_cup_count = int(np.count_nonzero(dropped_cup_mask))
    _write_run_metadata(
        model,
        asset_path,
        completed_frames,
        elapsed_wall_seconds,
        controlled_dof_count,
        cup_asset_path,
        profile_names,
        dropped_cup_mask,
    )
    completed_sim_seconds = completed_frames * FRAME_DT
    realtime_factor = completed_sim_seconds / elapsed_wall_seconds if elapsed_wall_seconds > 0.0 else 0.0
    print(f"[Performance] Simulated {completed_sim_seconds:.2f} s in {elapsed_wall_seconds:.2f} wall-clock s.")
    print(f"[Performance] Aggregate {HAND_COUNT}-hand realtime factor: {realtime_factor:.2f}x.")
    print(f"[Cups] Retained: {HAND_COUNT - dropped_cup_count}; dropped: {dropped_cup_count}.")
    print(f"[Output] Saved run metadata to '{RUN_METADATA_OUTPUT_PATH}'.")


if __name__ == "__main__":
    main()
