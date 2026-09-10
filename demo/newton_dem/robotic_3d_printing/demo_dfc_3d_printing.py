"""Drive a Newton-owned robotic print head from a parsed G-code path.

This is the motion-control foundation for a future Newton/DEME DFC concrete
printing co-simulation. The current stage intentionally creates no DEME solver
or material: MoPhi parses the toolpath, Newton IK converts its Cartesian
positions to ABB IRB6700 joint targets, and Newton advances the printing mechanism.
"""

import json
import math
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import warp as wp

import mophi
import newton
import newton.ik as ik
from mophi.utils import parse_gcode

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared_utils.abb_irb6700_printer_utils import build_abb_irb6700_printer  # noqa: E402

# ── User configuration ───────────────────────────────────────────────────────
NEWTON_DT = 1.0 / 500.0
NEWTON_GRAVITY = (0.0, 0.0, 0.0)
RENDER_FPS = 50
WARMUP_SECONDS = 0.5
POST_PATH_HOLD_SECONDS = 0.5
RENDER_SETTLING_PHASE = True
GCODE_TIME_SCALE = 1.0
GCODE_RAPID_FEED_RATE = 0.25
GCODE_TO_WORLD_SCALE = (1.0, 1.0, 1.0)

JOINT_TARGET_STIFFNESS = 500.0
JOINT_TARGET_DAMPING = 50.0
AUGER_TARGET_STIFFNESS = 100.0
AUGER_TARGET_DAMPING = 10.0
IK_ITERATIONS = 24
IK_DAMPING = 0.1

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PRINTER_ASSET_DIRECTORY = Path(__file__).resolve().parent / "data" / "chrono_irb6700_printer"
GCODE_PATH = Path(__file__).resolve().parent / "data" / "dfc_single_bead.gcode"
BUILD_PLATE_CENTER = (2.52, -0.57, 0.04)
BUILD_PLATE_HALF_EXTENTS = (0.70, 0.70, 0.04)

USE_OMNIVERSE_VISUALIZATION = False
SAVE_MOVIE = True
CAMERA_POSITION = (4.7, -4.0, 2.7)
REFERENCE_AXIS_ORIGIN = (1.35, -1.25, 0.02)
REFERENCE_SCALE_BAR_CENTER = (2.45, -1.55, 0.05)
PATH_VISUAL_SAMPLES = 80
PATH_VISUAL_COLOR = (0.15, 0.80, 0.95)
PATH_VISUAL_WIDTH = 0.006
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "output" / Path(__file__).stem
MOVIE_OUTPUT_PATH = OUTPUT_DIRECTORY / "dfc_3d_printing_gcode.mp4"
USD_OUTPUT_PATH = OUTPUT_DIRECTORY / "dfc_3d_printing_gcode.usdc"
METADATA_OUTPUT_PATH = OUTPUT_DIRECTORY / "run_metadata.json"
MOVIE_FPS = RENDER_FPS

# ── Derived constants ────────────────────────────────────────────────────────
FRAME_DT = 1.0 / RENDER_FPS
SIM_SUBSTEPS = int(round(FRAME_DT / NEWTON_DT))
WARMUP_STEPS = int(round(WARMUP_SECONDS / NEWTON_DT))


def _quat_rotate_xyzw(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return vector + 2.0 * np.cross(
        quaternion[:3],
        np.cross(quaternion[:3], vector) + quaternion[3] * vector,
    )


def _quat_multiply_xyzw(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = lhs
    rx, ry, rz, rw = rhs
    return np.asarray(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dtype=np.float32,
    )


def _gcode_offset(program, program_time: float) -> np.ndarray:
    """Return the configured world-axis offset from the G-code path origin."""
    path_origin = np.asarray(program.moves[0].start, dtype=np.float32)
    path_position = np.asarray(program.position_at(program_time), dtype=np.float32)
    return (path_position - path_origin) * np.asarray(GCODE_TO_WORLD_SCALE, dtype=np.float32)


if not np.isclose(SIM_SUBSTEPS * NEWTON_DT, FRAME_DT, rtol=0.0, atol=1.0e-12):
    raise ValueError("FRAME_DT must be an integer multiple of NEWTON_DT")
if GCODE_TIME_SCALE <= 0.0:
    raise ValueError("GCODE_TIME_SCALE must be positive")
if PATH_VISUAL_SAMPLES < 2:
    raise ValueError("PATH_VISUAL_SAMPLES must be at least two")
if not GCODE_PATH.is_file():
    mophi.fatal(f"Required G-code path is missing: {GCODE_PATH}")
if not PRINTER_ASSET_DIRECTORY.is_dir():
    mophi.fatal(f"Required ABB printer assets are missing: {PRINTER_ASSET_DIRECTORY}")

mophi.check_newton_warp_mujoco_versions(
    mophi.REQUIRED_NEWTON_VERSION,
    mophi.REQUIRED_WARP_VERSION,
    mophi.REQUIRED_MUJOCO_VERSION,
)

program = parse_gcode(GCODE_PATH, rapid_feed_rate=GCODE_RAPID_FEED_RATE)
if not program.moves:
    mophi.fatal(f"G-code path contains no supported motion: {GCODE_PATH}")
path_seconds = program.duration * GCODE_TIME_SCALE
motion_frames = int(math.ceil(path_seconds / FRAME_DT))
hold_frames = int(round(POST_PATH_HOLD_SECONDS / FRAME_DT))

print("=== MoPhi G-code-driven DFC 3D-printing motion stage ===")
print(f"[G-code] Loaded {len(program.moves)} moves ({program.duration:.3f} s nominal).")
wp.init()
device = wp.get_device()

print("[Newton] Reconstructing the ABB IRB6700 and auger print head ...")
try:
    printer = build_abb_irb6700_printer(
        PRINTER_ASSET_DIRECTORY,
        JOINT_TARGET_STIFFNESS,
        JOINT_TARGET_DAMPING,
        AUGER_TARGET_STIFFNESS,
        AUGER_TARGET_DAMPING,
    )
except (FileNotFoundError, ValueError) as exc:
    mophi.fatal(str(exc))
builder = printer.builder
nozzle_body_idx = printer.nozzle_body_index
nozzle_position_local = np.asarray(printer.nozzle_position_local, dtype=np.float32)
nozzle_rotation_local = np.asarray(printer.nozzle_rotation_local, dtype=np.float32)

builder.add_ground_plane()
plate_cfg = newton.ModelBuilder.ShapeConfig(has_particle_collision=True)
builder.add_shape_box(
    -1,
    xform=wp.transform(wp.vec3(*BUILD_PLATE_CENTER)),
    hx=BUILD_PLATE_HALF_EXTENTS[0],
    hy=BUILD_PLATE_HALF_EXTENTS[1],
    hz=BUILD_PLATE_HALF_EXTENTS[2],
    cfg=plate_cfg,
    label="build_plate",
)

model = builder.finalize()
model.set_gravity(NEWTON_GRAVITY)
solver = newton.solvers.SolverMuJoCo(model, disable_contacts=True)
state_0 = model.state()
state_1 = model.state()
control = model.control()
newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
wp.copy(control.joint_target_pos, model.joint_q)

initial_nozzle_body_transform = state_0.body_q.numpy()[nozzle_body_idx]
initial_nozzle_body_rotation = initial_nozzle_body_transform[3:7].astype(np.float32)
initial_nozzle_position = initial_nozzle_body_transform[:3] + _quat_rotate_xyzw(
    initial_nozzle_body_rotation,
    nozzle_position_local,
)
initial_nozzle_rotation = _quat_multiply_xyzw(initial_nozzle_body_rotation, nozzle_rotation_local)

position_objective = ik.IKObjectivePosition(
    link_index=nozzle_body_idx,
    link_offset=wp.vec3(*nozzle_position_local),
    target_positions=wp.array([initial_nozzle_position], dtype=wp.vec3, device=device),
)
rotation_objective = ik.IKObjectiveRotation(
    link_index=nozzle_body_idx,
    link_offset_rotation=wp.quat(*nozzle_rotation_local),
    target_rotations=wp.array([initial_nozzle_rotation], dtype=wp.vec4, device=device),
)
joint_limit_objective = ik.IKObjectiveJointLimit(
    joint_limit_lower=model.joint_limit_lower,
    joint_limit_upper=model.joint_limit_upper,
    weight=10.0,
)
ik_joint_q = wp.clone(model.joint_q.reshape((1, model.joint_coord_count)))
ik_solver = ik.IKSolver(
    model=model,
    n_problems=1,
    objectives=[position_objective, rotation_objective, joint_limit_objective],
    lambda_initial=IK_DAMPING,
    jacobian_mode=ik.IKJacobianType.ANALYTIC,
)

OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
if USE_OMNIVERSE_VISUALIZATION:
    vis = mophi.OmniverseVisualizer(output_path=str(USD_OUTPUT_PATH), fps=RENDER_FPS)
else:
    vis = mophi.OpenGLVisualizer(model)
camera_to_nozzle = initial_nozzle_position - np.asarray(CAMERA_POSITION)
camera_yaw = np.degrees(np.arctan2(camera_to_nozzle[1], camera_to_nozzle[0]))
camera_pitch = np.degrees(np.arctan2(camera_to_nozzle[2], np.hypot(camera_to_nozzle[0], camera_to_nozzle[1])))
vis.set_camera(pos=wp.vec3(*CAMERA_POSITION), pitch=camera_pitch, yaw=camera_yaw)
mophi.log_orientation_and_scale_reference(
    vis,
    axis_origin=REFERENCE_AXIS_ORIGIN,
    scale_bar_center=REFERENCE_SCALE_BAR_CENTER,
)

path_times = np.linspace(0.0, program.duration, PATH_VISUAL_SAMPLES)
path_points = np.asarray([initial_nozzle_position + _gcode_offset(program, time) for time in path_times])
path_starts = wp.array(path_points[:-1], dtype=wp.vec3, device=device)
path_ends = wp.array(path_points[1:], dtype=wp.vec3, device=device)
path_colors = wp.full(len(path_points) - 1, PATH_VISUAL_COLOR, dtype=wp.vec3, device=device)

movie_writer = None
if SAVE_MOVIE and not USE_OMNIVERSE_VISUALIZATION:
    movie_writer = imageio.get_writer(MOVIE_OUTPUT_PATH, fps=MOVIE_FPS)


def _set_gcode_target(program_time: float) -> None:
    target_position = initial_nozzle_position + _gcode_offset(program, program_time)
    position_objective.set_target_position(0, wp.vec3(*target_position))
    ik_solver.step(ik_joint_q, ik_joint_q, iterations=IK_ITERATIONS)
    wp.copy(control.joint_target_pos, ik_joint_q.flatten())


def _step_newton() -> None:
    global state_0, state_1
    state_0.clear_forces()
    solver.step(state_0, state_1, control, None, NEWTON_DT)
    state_0, state_1 = state_1, state_0


def _render(frame_time: float) -> None:
    vis.begin_frame(frame_time)
    vis.log_state(state_0)
    vis.log_lines(
        "gcode_toolpath",
        path_starts,
        path_ends,
        path_colors,
        width=PATH_VISUAL_WIDTH,
    )
    vis.end_frame()
    if movie_writer is not None:
        movie_writer.append_data(vis.get_frame().numpy())


sim_time = 0.0
frames_completed = 0
settling_frames_completed = 0
try:
    _set_gcode_target(0.0)
    for warmup_step in range(WARMUP_STEPS):
        _step_newton()
        sim_time += NEWTON_DT
        settling_frame_due = (warmup_step + 1) % SIM_SUBSTEPS == 0 or warmup_step + 1 == WARMUP_STEPS
        if RENDER_SETTLING_PHASE and settling_frame_due:
            if not vis.is_running():
                break
            _render(sim_time)
            settling_frames_completed += 1
            frames_completed += 1

    for frame in range(motion_frames + hold_frames):
        if not vis.is_running():
            break
        path_time = min((frame + 1) * FRAME_DT / GCODE_TIME_SCALE, program.duration)
        _set_gcode_target(path_time)
        for _ in range(SIM_SUBSTEPS):
            _step_newton()
        sim_time += FRAME_DT
        _render(sim_time)
        frames_completed += 1
finally:
    if movie_writer is not None:
        movie_writer.close()
    vis.close()

final_nozzle_body_transform = state_0.body_q.numpy()[nozzle_body_idx]
final_nozzle_position = final_nozzle_body_transform[:3] + _quat_rotate_xyzw(
    final_nozzle_body_transform[3:7],
    nozzle_position_local,
)
requested_final_nozzle_position = initial_nozzle_position + _gcode_offset(program, program.duration)
final_nozzle_error = float(np.linalg.norm(final_nozzle_position - requested_final_nozzle_position))

METADATA_OUTPUT_PATH.write_text(
    json.dumps(
        {
            "phase": "gcode_driven_newton_motion",
            "printer_model": "chrono_reference_abb_irb6700_with_auger",
            "printer_asset_directory": str(PRINTER_ASSET_DIRECTORY),
            "gcode_path": str(GCODE_PATH),
            "gcode_move_count": len(program.moves),
            "gcode_nominal_duration": program.duration,
            "gcode_time_scale": GCODE_TIME_SCALE,
            "newton_dt": NEWTON_DT,
            "render_fps": RENDER_FPS,
            "ik_iterations": IK_ITERATIONS,
            "frames_completed": frames_completed,
            "settling_frames_completed": settling_frames_completed,
            "render_settling_phase": RENDER_SETTLING_PHASE,
            "requested_final_nozzle_position": requested_final_nozzle_position.tolist(),
            "actual_final_nozzle_position": final_nozzle_position.tolist(),
            "final_nozzle_position_error": final_nozzle_error,
            "deme_enabled": False,
            "dfc_enabled": False,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(f"Completed {frames_completed} frames with {final_nozzle_error:.6f} m final nozzle error.")
print(f"Output: {OUTPUT_DIRECTORY}")
