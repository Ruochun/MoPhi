"""Unitree G1 swimming idea demo using Newton and XLB.

XLB simulates flow around prescribed axis-aligned boxes that follow selected
robot links. Newton advances the articulated robot while analytic buoyancy,
drag, and a scripted swimming stroke keep it moving underwater. Fluid forces
are intentionally not extracted from XLB in this first-stage demonstration.
"""

import json
import math
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
import warp as wp
import yaml

import mophi
import newton
import newton.examples
import newton.utils
import xlb
from newton import JointTargetMode
from xlb.compute_backend import ComputeBackend
from xlb.grid import grid_factory
from xlb.operator.boundary_condition import HalfwayBounceBackBC, ZouHeBC, ExtrapolationOutflowBC
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.precision_policy import Precision, PrecisionPolicy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import newton_xlb_dem_utils as demo_utils  # noqa: E402

# =============================================================================
# Simulation configuration
# All user-tunable constants are defined here.
# =============================================================================

# -- Timing -------------------------------------------------------------------
RENDER_FPS = 25
NEWTON_DT = 1.0 / 200.0
SIM_DURATION_SECONDS = 5.0
XLB_STEPS_PER_FRAME = 2
XLB_WARMUP_STEPS = 100
XLB_VIS_INTERVAL = 5

# -- Robot and swimming motion ------------------------------------------------
ROBOT_NAME = "g1_29dof"
ROBOT_START_Y_CLEARANCE = 0.8
ROBOT_START_HEIGHT = 1.5
# The G1 asset faces +X by default; rotate it +90 degrees around Z to face +Y.
ROBOT_START_ORIENTATION_XYZW = np.array([0.0, 0.0, 0.7071068, 0.7071068], dtype=np.float32)
SWIM_STROKE_FREQUENCY = 0.65
SWIM_FORWARD_FORCE = 35.0
SWIM_FORWARD_FORCE_BODY_LABEL = "pelvis"

# -- Prescribed swimming motion -----------------------------------------------
# Each entry is: (joint-name pattern, pose offset [rad], amplitude [rad], phase [cycles]).
# Edit this block to prototype different strokes without changing the simulation loop.
PRESCRIBED_SWIM_MOTION = (
    ("left_shoulder_pitch_joint", -0.25, 1.05, 0.00),
    ("right_shoulder_pitch_joint", -0.25, 1.05, 0.50),
    ("left_shoulder_roll_joint", 0.30, 0.35, 0.10),
    ("right_shoulder_roll_joint", -0.30, -0.35, 0.60),
    ("left_elbow_joint", 0.65, 0.55, 0.20),
    ("right_elbow_joint", 0.65, 0.55, 0.70),
    ("left_hip_pitch_joint", -0.20, 0.55, 0.50),
    ("right_hip_pitch_joint", -0.20, 0.55, 0.00),
    ("left_knee_joint", 0.45, 0.40, 0.65),
    ("right_knee_joint", 0.45, 0.40, 0.15),
)

# -- Analytic underwater forces -----------------------------------------------
BUOYANCY_GRAVITY = 9.81
BUOYANCY_SCALE = 1.0
LINEAR_DRAG_COEFFICIENT = 22.0
ANGULAR_DRAG_COEFFICIENT = 4.0
DEPTH_HOLD_STIFFNESS = 80.0
DEPTH_HOLD_DAMPING = 40.0

# -- XLB fluid ----------------------------------------------------------------
XLB_GRID_DIMS = (64, 96, 64)
XLB_DOMAIN_MIN = np.array([-2.0, -4.0, 0.0], dtype=np.float64)
XLB_DOMAIN_MAX = np.array([2.0, 4.0, 3.0], dtype=np.float64)
XLB_FLOW_DIRECTION_Y = 1
XLB_INLET_SPEED = 0.015
XLB_KINEMATIC_VISCOSITY = 0.02
XLB_OMEGA = 1.0 / (3.0 * XLB_KINEMATIC_VISCOSITY + 0.5)
XLB_STREAMLINE_STEP_SIZE = 0.12
XLB_OBSTACLE_SPECS = (
    ("pelvis", np.array([0.20, 0.14, 0.16])),
    ("torso", np.array([0.24, 0.16, 0.24])),
    ("left_shoulder", np.array([0.10, 0.10, 0.22])),
    ("right_shoulder", np.array([0.10, 0.10, 0.22])),
    ("left_elbow", np.array([0.08, 0.08, 0.20])),
    ("right_elbow", np.array([0.08, 0.08, 0.20])),
    ("left_hip", np.array([0.11, 0.11, 0.24])),
    ("right_hip", np.array([0.11, 0.11, 0.24])),
    ("left_knee", np.array([0.09, 0.09, 0.22])),
    ("right_knee", np.array([0.09, 0.09, 0.22])),
)

# -- Visualization ------------------------------------------------------------
CAMERA_POSITION = (7.0, -8.0, 3.2)
CAMERA_PITCH = -8.0
CAMERA_YAW = 140.0
REFERENCE_AXIS_ORIGIN = (1.2, -3.6, 0.1)
REFERENCE_SCALE_BAR_CENTER = (0.2, -3.6, 0.14)
SHOW_UNDERWATER_ENVIRONMENT = True

# -- Output -------------------------------------------------------------------
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "output" / Path(__file__).stem
SAVE_MOVIE = True
MOVIE_OUTPUT_PATH = OUTPUT_DIRECTORY / "demo_humanoid_swimming.mp4"
MOVIE_FPS = RENDER_FPS
RUN_METADATA_OUTPUT_PATH = OUTPUT_DIRECTORY / "run_metadata.json"

# -- Derived constants --------------------------------------------------------
ROBOT_START_POSITION = np.array(
    [0.0, XLB_DOMAIN_MIN[1] + ROBOT_START_Y_CLEARANCE, ROBOT_START_HEIGHT],
    dtype=np.float32,
)
XLB_STREAMLINE_STEPS = int(np.ceil((XLB_DOMAIN_MAX[1] - XLB_DOMAIN_MIN[1]) / XLB_STREAMLINE_STEP_SIZE)) + 2
FRAME_DT = 1.0 / RENDER_FPS
NUM_FRAMES = int(round(SIM_DURATION_SECONDS / FRAME_DT))
SIM_SUBSTEPS = int(round(FRAME_DT / NEWTON_DT))
if not np.isclose(SIM_SUBSTEPS * NEWTON_DT, FRAME_DT, atol=1.0e-12):
    raise ValueError("FRAME_DT must be an integer multiple of NEWTON_DT.")


def _find_body_index(body_labels: list[str], pattern: str) -> int:
    matches = [idx for idx, label in enumerate(body_labels) if pattern.lower() in label.lower()]
    if not matches:
        mophi.fatal(f"Could not find a G1 body matching '{pattern}'. Available labels:\n{body_labels}")
    return matches[0]


def _build_prescribed_swim_motion(builder) -> list[tuple[int, float, float, float]]:
    """Resolve the editable prescribed-motion table to Newton target indices."""
    resolved_motion = []
    for pattern, offset, amplitude, phase_cycles in PRESCRIBED_SWIM_MOTION:
        matches = [idx for idx, label in enumerate(builder.joint_label) if pattern.lower() in label.lower()]
        if len(matches) != 1:
            mophi.fatal(f"Expected one G1 joint matching '{pattern}', found {len(matches)}.")
        resolved_motion.append((int(builder.joint_qd_start[matches[0]]), offset, amplitude, phase_cycles))
    return resolved_motion


def _prescribed_swim_targets(
    initial_targets: np.ndarray,
    resolved_motion: list[tuple[int, float, float, float]],
    sim_time: float,
) -> np.ndarray:
    """Build one target-pose sample from the prescribed swimming-motion block."""
    targets = initial_targets.copy()
    stroke_phase = 2.0 * math.pi * SWIM_STROKE_FREQUENCY * sim_time
    for target_idx, offset, amplitude, phase_cycles in resolved_motion:
        targets[target_idx] += offset + amplitude * math.sin(stroke_phase + 2.0 * math.pi * phase_cycles)
    return targets


def _build_newton_robot(robot_config, config, asset_directory: Path):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(armature=0.1, limit_ke=1.0e2, limit_kd=1.0)
    builder.default_shape_cfg.ke = 5.0e4
    builder.default_shape_cfg.kd = 5.0e2
    builder.default_shape_cfg.kf = 1.0e3
    builder.default_shape_cfg.mu = 0.5
    builder.add_usd(
        newton.examples.get_asset(str(asset_directory / robot_config.asset_path)),
        xform=wp.transform(wp.vec3(*ROBOT_START_POSITION), wp.quat(*ROBOT_START_ORIENTATION_XYZW)),
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        joint_ordering="dfs",
        hide_collision_shapes=True,
    )
    builder.approximate_meshes("convex_hull")
    builder.joint_q[:3] = ROBOT_START_POSITION.tolist()
    builder.joint_q[3:7] = ROBOT_START_ORIENTATION_XYZW.tolist()
    builder.joint_q[7 : 7 + len(config["mjw_joint_pos"])] = config["mjw_joint_pos"]
    for joint_idx in range(len(config["mjw_joint_stiffness"])):
        dof_idx = joint_idx + 6
        builder.joint_target_ke[dof_idx] = config["mjw_joint_stiffness"][joint_idx]
        builder.joint_target_kd[dof_idx] = config["mjw_joint_damping"][joint_idx]
        builder.joint_armature[dof_idx] = config["mjw_joint_armature"][joint_idx]
        builder.joint_target_mode[dof_idx] = int(JointTargetMode.POSITION)
        builder.joint_target_pos[dof_idx] = config["mjw_joint_pos"][joint_idx]
    return builder


def _make_box_grid(body_position: np.ndarray, half_extents: np.ndarray):
    return demo_utils.xlb_prescribed_robot_box_grid(
        body_position,
        XLB_DOMAIN_MIN,
        XLB_DOMAIN_MAX,
        XLB_GRID_DIMS,
        float(half_extents[0]),
        float(half_extents[1]),
        float(half_extents[2]),
        float(half_extents[2]),
    )


def _update_xlb_boxes(bc_mask, missing_mask, old_boxes, new_boxes, box_bcs, vel_c_wp, q) -> None:
    """Clear every old box before stamping new boxes so overlaps remain solid."""
    for old_box in old_boxes:
        demo_utils.xlb_update_robot_box_gpu(
            bc_mask,
            missing_mask,
            old_box[0],
            old_box[1],
            None,
            None,
            0,
            vel_c_wp,
            q,
        )
    for new_box, box_bc in zip(new_boxes, box_bcs):
        demo_utils.xlb_update_robot_box_gpu(
            bc_mask,
            missing_mask,
            None,
            None,
            new_box[0],
            new_box[1],
            box_bc.id,
            vel_c_wp,
            q,
        )


def _setup_xlb(initial_boxes):
    velocity_set = xlb.velocity_set.D3Q19(
        precision_policy=PrecisionPolicy.FP32FP32,
        compute_backend=ComputeBackend.WARP,
    )
    xlb.init(velocity_set, ComputeBackend.WARP, PrecisionPolicy.FP32FP32)
    grid = grid_factory(XLB_GRID_DIMS, compute_backend=ComputeBackend.WARP)
    faces = grid.bounding_box_indices(remove_edges=True)
    inlet = ZouHeBC(
        bc_type="velocity",
        prescribed_value=np.array([0.0, XLB_FLOW_DIRECTION_Y * XLB_INLET_SPEED, 0.0]),
        indices=faces["front"],
    )
    outlet = ExtrapolationOutflowBC(indices=faces["back"])
    box_bcs = []
    placeholder_boxes = []
    for box_idx in range(len(initial_boxes)):
        placeholder = np.array([2 + box_idx, 2, 2], dtype=np.int32)
        placeholder_boxes.append((placeholder, placeholder.copy()))
        box_bcs.append(HalfwayBounceBackBC(indices=[[int(placeholder[0])], [2], [2]]))
    stepper = IncompressibleNavierStokesStepper(
        grid=grid,
        boundary_conditions=[inlet, outlet, *box_bcs],
        collision_type="BGK",
    )
    f0, f1, bc_mask, missing_mask = stepper.prepare_fields()
    c_np = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
            [1, 1, 0],
            [-1, -1, 0],
            [1, -1, 0],
            [-1, 1, 0],
            [1, 0, 1],
            [-1, 0, -1],
            [1, 0, -1],
            [-1, 0, 1],
            [0, 1, 1],
            [0, -1, -1],
            [0, 1, -1],
            [0, -1, 1],
        ],
        dtype=np.int32,
    )
    vel_c_wp = wp.array(c_np, dtype=wp.int32, device=bc_mask.device)
    _update_xlb_boxes(
        bc_mask,
        missing_mask,
        placeholder_boxes,
        initial_boxes,
        box_bcs,
        vel_c_wp,
        velocity_set.q,
    )
    for timestep in range(XLB_WARMUP_STEPS):
        f0, f1 = stepper(f0, f1, bc_mask, missing_mask, XLB_OMEGA, timestep)
        f0, f1 = f1, f0
    macro = Macroscopic(velocity_set, PrecisionPolicy.FP32FP32, ComputeBackend.WARP)
    rho = grid.create_field(cardinality=1, dtype=Precision.FP32)
    velocity = grid.create_field(cardinality=3, dtype=Precision.FP32)
    return stepper, f0, f1, bc_mask, missing_mask, box_bcs, vel_c_wp, macro, rho, velocity


def _underwater_forces(
    body_q: np.ndarray,
    body_qd: np.ndarray,
    body_mass: np.ndarray,
    propulsion_idx: int,
    target_depth: float,
) -> np.ndarray:
    """Return a stable net flotation-pack wrench applied at the pelvis."""
    forces = np.zeros((len(body_q), 6), dtype=np.float32)
    forces[propulsion_idx, 2] += float(body_mass.sum()) * BUOYANCY_GRAVITY * BUOYANCY_SCALE
    forces[propulsion_idx, :3] -= LINEAR_DRAG_COEFFICIENT * body_qd[propulsion_idx, :3]
    forces[propulsion_idx, 3:] -= ANGULAR_DRAG_COEFFICIENT * body_qd[propulsion_idx, 3:]
    depth_error = target_depth - body_q[propulsion_idx, 2]
    forces[propulsion_idx, 2] += DEPTH_HOLD_STIFFNESS * depth_error
    forces[propulsion_idx, 2] -= DEPTH_HOLD_DAMPING * body_qd[propulsion_idx, 2]
    forces[propulsion_idx, 1] += SWIM_FORWARD_FORCE
    return forces


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

    from newton.examples.robot import example_robot_policy

    robot_config = example_robot_policy.ROBOT_CONFIGS[ROBOT_NAME]
    asset_directory = Path(newton.utils.download_asset(robot_config.asset_dir))
    with (asset_directory / robot_config.yaml_path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    builder = _build_newton_robot(robot_config, config, asset_directory)
    body_labels = list(builder.body_label)
    obstacle_body_indices = [_find_body_index(body_labels, pattern) for pattern, _ in XLB_OBSTACLE_SPECS]
    propulsion_idx = _find_body_index(body_labels, SWIM_FORWARD_FORCE_BODY_LABEL)
    prescribed_swim_motion = _build_prescribed_swim_motion(builder)
    model = builder.finalize()
    model.set_gravity((0.0, 0.0, -9.81))
    initial_state = model.state()
    newton.eval_fk(model, initial_state.joint_q, initial_state.joint_qd, initial_state)
    initial_body_q = initial_state.body_q.numpy()
    target_depth = float(initial_body_q[propulsion_idx, 2])
    initial_boxes = [
        _make_box_grid(initial_body_q[body_idx, :3], half_extents)
        for body_idx, (_, half_extents) in zip(obstacle_body_indices, XLB_OBSTACLE_SPECS)
    ]
    if any(gc_min is None for gc_min, _ in initial_boxes):
        mophi.fatal("An initial swimming-robot XLB box lies outside the fluid domain.")

    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=False, solver="newton", nconmax=1024, njmax=2048)
    stepper, f0, f1, bc_mask, missing_mask, box_bcs, vel_c_wp, macro, rho, velocity = _setup_xlb(initial_boxes)

    coupler = mophi.NewtonXLBDEMCoupler()
    coupler.initialize(model, solver, stepper, None, NEWTON_DT)
    coupler.set_xlb_masks(bc_mask, missing_mask)
    initial_targets = coupler.newton_control.joint_target_pos.numpy().copy()
    body_mass = model.body_mass.numpy()

    vis = mophi.create_opengl_visualizer_or_fatal(model)
    vis.set_camera(wp.vec3(*CAMERA_POSITION), CAMERA_PITCH, CAMERA_YAW)
    mophi.log_orientation_and_scale_reference(
        vis,
        axis_origin=REFERENCE_AXIS_ORIGIN,
        scale_bar_center=REFERENCE_SCALE_BAR_CENTER,
    )
    if SHOW_UNDERWATER_ENVIRONMENT:
        mophi.log_underwater_environment(
            vis,
            domain_min=XLB_DOMAIN_MIN,
            domain_max=XLB_DOMAIN_MAX,
        )
    seed_points = mophi.xlb_make_y_plane_seeds(
        XLB_DOMAIN_MIN,
        XLB_DOMAIN_MAX,
        flow_direction=XLB_FLOW_DIRECTION_Y,
    )
    old_boxes = initial_boxes
    xlb_timestep = XLB_WARMUP_STEPS
    movie_writer = imageio.get_writer(MOVIE_OUTPUT_PATH, fps=MOVIE_FPS) if SAVE_MOVIE else None
    completed_frames = 0

    try:
        for frame in range(NUM_FRAMES):
            if not vis.is_running():
                break
            sim_time = frame * FRAME_DT
            target = _prescribed_swim_targets(initial_targets, prescribed_swim_motion, sim_time)
            wp.copy(coupler.newton_control.joint_target_pos, wp.array(target, dtype=wp.float32, device=wp.get_device()))

            for _ in range(SIM_SUBSTEPS):
                body_q = coupler.newton_state_0.body_q.numpy()
                body_qd = coupler.newton_state_0.body_qd.numpy()
                forces = _underwater_forces(body_q, body_qd, body_mass, propulsion_idx, target_depth)
                coupler.set_newton_body_forces(wp.array(forces, dtype=wp.spatial_vector, device=wp.get_device()))
                coupler.step_newton()

            body_q = coupler.newton_state_0.body_q.numpy()
            new_boxes = [
                _make_box_grid(body_q[body_idx, :3], half_extents)
                for body_idx, (_, half_extents) in zip(obstacle_body_indices, XLB_OBSTACLE_SPECS)
            ]
            _update_xlb_boxes(bc_mask, missing_mask, old_boxes, new_boxes, box_bcs, vel_c_wp, 19)
            old_boxes = new_boxes
            for _ in range(XLB_STEPS_PER_FRAME):
                f0, f1 = stepper(f0, f1, bc_mask, missing_mask, XLB_OMEGA, xlb_timestep)
                f0, f1 = f1, f0
                xlb_timestep += 1

            vis.begin_frame(sim_time)
            vis.log_state(coupler.newton_state_0)
            if frame % XLB_VIS_INTERVAL == 0:
                rho, velocity = macro(f0, rho, velocity)
                velocity_np = velocity.numpy().transpose(1, 2, 3, 0).astype(np.float32)
                points, speeds, directions = mophi.xlb_build_streamlines(
                    velocity_np,
                    XLB_DOMAIN_MIN,
                    XLB_DOMAIN_MAX,
                    seed_points,
                    n_steps=XLB_STREAMLINE_STEPS,
                    step_size=XLB_STREAMLINE_STEP_SIZE,
                )
                stream_pos, stream_radii, stream_colors = mophi.xlb_make_streamline_warp_arrays(
                    points, speeds, directions
                )
            vis.log_points("xlb_flow", stream_pos, radii=stream_radii, colors=stream_colors)
            vis.end_frame()
            completed_frames = frame + 1
            if movie_writer is not None:
                movie_writer.append_data(vis.get_frame().numpy())
    finally:
        if movie_writer is not None:
            movie_writer.close()
        vis.close()
        coupler.finalize()

    metadata = {
        "demo": "humanoid_swimming",
        "completed_frames": completed_frames,
        "requested_duration_seconds": SIM_DURATION_SECONDS,
        "newton_dt": NEWTON_DT,
        "buoyancy_scale": BUOYANCY_SCALE,
        "target_depth": target_depth,
        "depth_hold_stiffness": DEPTH_HOLD_STIFFNESS,
        "depth_hold_damping": DEPTH_HOLD_DAMPING,
        "xlb_grid_dims": XLB_GRID_DIMS,
        "xlb_domain_min": XLB_DOMAIN_MIN.tolist(),
        "xlb_domain_max": XLB_DOMAIN_MAX.tolist(),
        "xlb_flow_direction_y": XLB_FLOW_DIRECTION_Y,
        "xlb_obstacle_count": len(XLB_OBSTACLE_SPECS),
        "robot_start_position": ROBOT_START_POSITION.tolist(),
        "robot_start_orientation_xyzw": ROBOT_START_ORIENTATION_XYZW.tolist(),
        "prescribed_swim_joint_count": len(PRESCRIBED_SWIM_MOTION),
        "prescribed_swim_frequency": SWIM_STROKE_FREQUENCY,
        "show_underwater_environment": SHOW_UNDERWATER_ENVIRONMENT,
        "fluid_force_model": "analytic buoyancy and drag; no XLB-to-Newton force feedback",
    }
    RUN_METADATA_OUTPUT_PATH.write_text(json.dumps(metadata, indent=4) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
