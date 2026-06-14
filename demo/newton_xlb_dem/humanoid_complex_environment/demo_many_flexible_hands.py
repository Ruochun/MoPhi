"""Many independently animated articulated-hand instances using Newton.

This demo adapts Newton 1.0.0's public ``example_robot_allegro_hand.py``
baseline for MoPhi. It replicates a contact-capable Allegro hand model into
many isolated simulation worlds and drives every finger joint with an
individually configurable periodic target.

The fingers are articulated rigid links rather than deformable solids. This
first stage focuses on scalable finger actuation; grasped objects and contact
workloads can be added later without replacing the hand model.
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
MAX_CONTACTS_PER_HAND = 300
MAX_CONSTRAINTS_PER_HAND = 200
SOLVER_ITERATIONS = 10
SOLVER_LINE_SEARCH_ITERATIONS = 5

# -- Visualization ------------------------------------------------------------
CAMERA_POSITION = (4.8, -5.5, 4.3)
CAMERA_PITCH = -32.0
CAMERA_YAW = 140.0
REFERENCE_AXIS_ORIGIN = (-0.6, -0.6, 0.02)
REFERENCE_SCALE_BAR_CENTER = (0.4, -0.6, 0.06)

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
    phase = 2.0 * wp.pi * (frequency * sim_time + phase_cycles[target_idx])
    target = centers[target_idx] + amplitudes[target_idx] * wp.sin(phase)
    targets[target_idx] = wp.clamp(target, lower_limits[target_idx], upper_limits[target_idx])


def _configure_hand_template() -> tuple[newton.ModelBuilder, np.ndarray, np.ndarray, np.ndarray, Path]:
    """Load one Allegro hand and return its per-DOF motion arrays."""
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

    dof_count = hand.joint_dof_count
    centers = np.zeros(dof_count, dtype=np.float32)
    amplitudes = np.zeros(dof_count, dtype=np.float32)
    phases = np.zeros(dof_count, dtype=np.float32)
    configured_dofs = set()

    for label_suffix, center, amplitude, phase_cycles in FINGER_MOTION_SPECS:
        matches = [joint_idx for joint_idx, label in enumerate(hand.joint_label) if label.endswith(f"/{label_suffix}")]
        if len(matches) != 1:
            mophi.fatal(f"Expected one Allegro joint ending with '{label_suffix}', found {len(matches)}.")
        joint_idx = matches[0]
        dof_idx = int(hand.joint_qd_start[joint_idx])
        centers[dof_idx] = center
        amplitudes[dof_idx] = amplitude
        phases[dof_idx] = phase_cycles
        configured_dofs.add(dof_idx)

        hand.joint_q[dof_idx] = center
        hand.joint_target_pos[dof_idx] = center
        hand.joint_target_ke[dof_idx] = FINGER_DRIVE_STIFFNESS
        hand.joint_target_kd[dof_idx] = FINGER_DRIVE_DAMPING
        hand.joint_target_mode[dof_idx] = int(JointTargetMode.POSITION)
        hand.joint_armature[dof_idx] = FINGER_JOINT_ARMATURE

    if configured_dofs != set(range(dof_count)):
        missing = sorted(set(range(dof_count)) - configured_dofs)
        mophi.fatal(f"Finger motion does not configure every Allegro DOF. Missing indices: {missing}")

    return hand, centers, amplitudes, phases, asset_path


def _replicate_motion_arrays(
    local_centers: np.ndarray,
    local_amplitudes: np.ndarray,
    local_phases: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand one hand's controls and phase-shift each replicated hand."""
    centers = np.tile(local_centers, HAND_COUNT)
    amplitudes = np.tile(local_amplitudes, HAND_COUNT)
    hand_phase_offsets = np.arange(HAND_COUNT, dtype=np.float32) * (HAND_PHASE_SPREAD_CYCLES / HAND_COUNT)
    phases = np.tile(local_phases, HAND_COUNT) + np.repeat(hand_phase_offsets, len(local_phases))
    return centers, amplitudes, phases


def _write_run_metadata(
    model,
    asset_path: Path,
    completed_frames: int,
    elapsed_wall_seconds: float,
    controlled_dof_count: int,
) -> None:
    completed_sim_seconds = completed_frames * FRAME_DT
    metadata = {
        "demo": "many_flexible_hands",
        "baseline": "Newton example_robot_allegro_hand.py",
        "asset_path": str(asset_path),
        "hand_count": HAND_COUNT,
        "bodies": model.body_count,
        "joints": model.joint_count,
        "controlled_finger_dofs": controlled_dof_count,
        "controlled_finger_dofs_per_hand": controlled_dof_count // HAND_COUNT,
        "contacts_enabled": True,
        "contact_objects_added": False,
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

    hand, local_centers, local_amplitudes, local_phases, asset_path = _configure_hand_template()
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.replicate(hand, HAND_COUNT, spacing=HAND_SPACING)
    model = builder.finalize()
    model.set_gravity((0.0, 0.0, 0.0))

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

    centers, amplitudes, phases = _replicate_motion_arrays(local_centers, local_amplitudes, local_phases)
    controlled_dof_count = len(centers)
    if controlled_dof_count != model.joint_dof_count:
        mophi.fatal(
            f"Expected {controlled_dof_count} replicated hand DOFs, but Newton model has {model.joint_dof_count}."
        )
    device = wp.get_device()
    centers_wp = wp.array(centers, dtype=wp.float32, device=device)
    amplitudes_wp = wp.array(amplitudes, dtype=wp.float32, device=device)
    phases_wp = wp.array(phases, dtype=wp.float32, device=device)

    vis = mophi.create_opengl_visualizer_or_fatal(model)
    vis.set_camera(wp.vec3(*CAMERA_POSITION), CAMERA_PITCH, CAMERA_YAW)
    mophi.log_orientation_and_scale_reference(
        vis,
        axis_origin=REFERENCE_AXIS_ORIGIN,
        scale_bar_center=REFERENCE_SCALE_BAR_CENTER,
    )

    movie_writer = imageio.get_writer(MOVIE_OUTPUT_PATH, fps=MOVIE_FPS) if SAVE_MOVIE else None
    completed_frames = 0
    final_body_q = None
    final_joint_q = None
    start_wall_time = time.perf_counter()
    print(
        f"[Hands] Simulating {HAND_COUNT} Allegro hands with {controlled_dof_count} independently driven finger DOFs."
    )
    print(
        "[Contact] Hand collision geometry is retained; "
        "external grasp/contact objects are deferred to a later stage."
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
        if SAVE_FINAL_STATE:
            final_body_q = coupler.newton_state_0.body_q.numpy()
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
    _write_run_metadata(model, asset_path, completed_frames, elapsed_wall_seconds, controlled_dof_count)
    completed_sim_seconds = completed_frames * FRAME_DT
    realtime_factor = completed_sim_seconds / elapsed_wall_seconds if elapsed_wall_seconds > 0.0 else 0.0
    print(f"[Performance] Simulated {completed_sim_seconds:.2f} s in {elapsed_wall_seconds:.2f} wall-clock s.")
    print(f"[Performance] Aggregate {HAND_COUNT}-hand realtime factor: {realtime_factor:.2f}x.")
    print(f"[Output] Saved run metadata to '{RUN_METADATA_OUTPUT_PATH}'.")


if __name__ == "__main__":
    main()
