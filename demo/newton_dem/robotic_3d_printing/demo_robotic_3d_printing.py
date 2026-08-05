"""Newton machine stage for a robotic DEME/DFC 3D-printing demonstration.

The UR10, pedestal, print head, and build station are managed by Newton.  The
future DEME stage can emit DFC material at ``NOZZLE_OUTLET_LOCAL`` and couple
contact loads back to the Newton end effector without changing this machine
assembly.
"""

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import warp as wp

import mophi
import mujoco
import newton
import newton.utils
from newton import JointTargetMode
from newton.selection import ArticulationView

# ── User configuration ────────────────────────────────────────────────────────────────────────
WORLD_COUNT = 1
NEWTON_DT = 1.0 / 500.0
RENDER_FPS = 50
MOTION_SEGMENT_SECONDS = 0.8
WARMUP_SECONDS = 0.5
PRINT_CYCLES = 2

# These adjacent targets define one complete machine trajectory: hold a stable
# printing posture, make a slight base-joint XY sweep, then return to center.
APPROACH_Q = np.array([0.50 * np.pi, 0.80, -0.90, -1.47, -0.50 * np.pi, 0.0], dtype=np.float32)
PRINT_PATH_Q = np.array(
    [
        [0.49 * np.pi, 0.80, -0.90, -1.47, -0.50 * np.pi, 0.0],
        [0.51 * np.pi, 0.80, -0.90, -1.47, -0.50 * np.pi, 0.0],
        [0.49 * np.pi, 0.80, -0.90, -1.47, -0.50 * np.pi, 0.0],
        [0.50 * np.pi, 0.80, -0.90, -1.47, -0.50 * np.pi, 0.0],
    ],
    dtype=np.float32,
)
RETRACT_Q = np.array([0.50 * np.pi, 0.80, -0.90, -1.47, -0.50 * np.pi, 0.0], dtype=np.float32)
JOINT_TARGET_STIFFNESS = 500.0
JOINT_TARGET_DAMPING = 50.0

PEDESTAL_HEIGHT = 1.2
PEDESTAL_RADIUS = 0.09
BUILD_PLATE_CENTER = (0.0, 0.82, 0.06)
BUILD_PLATE_HALF_EXTENTS = (0.55, 0.55, 0.06)
BUILD_FRAME_POST_HALF_EXTENTS = (0.035, 0.035, 0.36)
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FUNNEL_OBJ_PATH = REPOSITORY_ROOT / "data" / "mesh" / "funnel.obj"
FUNNEL_MESH_SCALE = 0.001
FUNNEL_MESH_COLOR = (0.18, 0.20, 0.23)
FUNNEL_MESH_LOCAL_ROTATION_AXIS = (0.0, 1.0, 0.0)
FUNNEL_MESH_LOCAL_ROTATION_ANGLE = -0.5 * np.pi
# End-effector-local outlet used as the future DFC material injection point.
NOZZLE_OUTLET_LOCAL = (0.55, 0.0, 0.0)

USE_OMNIVERSE_VISUALIZATION = False
SAVE_MOVIE = True
CAMERA_POSITION = (3.1, -3.0, 2.5)
CAMERA_PITCH = -24.0
CAMERA_YAW = 136.0
REFERENCE_AXIS_ORIGIN = (0.8, -0.55, 0.02)
REFERENCE_SCALE_BAR_CENTER = (0.05, -0.55, 0.05)
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "output" / Path(__file__).stem
MOVIE_OUTPUT_PATH = OUTPUT_DIRECTORY / "robotic_3d_printing.mp4"
USD_OUTPUT_PATH = OUTPUT_DIRECTORY / "robotic_3d_printing.usdc"
METADATA_OUTPUT_PATH = OUTPUT_DIRECTORY / "run_metadata.json"
MOVIE_FPS = RENDER_FPS

# ── Derived constants ──────────────────────────────────────────────────────────────────────
FRAME_DT = 1.0 / RENDER_FPS
SIM_SUBSTEPS = int(round(FRAME_DT / NEWTON_DT))
MOTION_WAYPOINTS = np.vstack((APPROACH_Q, PRINT_PATH_Q, RETRACT_Q, APPROACH_Q))
MOTION_SEGMENT_COUNT = len(MOTION_WAYPOINTS) - 1
NUM_FRAMES = int(round(PRINT_CYCLES * MOTION_SEGMENT_COUNT * MOTION_SEGMENT_SECONDS / FRAME_DT))
WARMUP_STEPS = int(round(WARMUP_SECONDS / NEWTON_DT))


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _motion_target(sim_time: float) -> np.ndarray:
    """Interpolate the cyclic, low-amplitude XY sweep program."""
    cycle_duration = MOTION_SEGMENT_COUNT * MOTION_SEGMENT_SECONDS
    cycle_time = sim_time % cycle_duration
    segment = min(int(cycle_time / MOTION_SEGMENT_SECONDS), MOTION_SEGMENT_COUNT - 1)
    phase = _smoothstep((cycle_time - segment * MOTION_SEGMENT_SECONDS) / MOTION_SEGMENT_SECONDS)
    return (1.0 - phase) * MOTION_WAYPOINTS[segment] + phase * MOTION_WAYPOINTS[segment + 1]


if not np.isclose(SIM_SUBSTEPS * NEWTON_DT, FRAME_DT, rtol=0.0, atol=1.0e-12):
    raise ValueError("FRAME_DT must be an integer multiple of NEWTON_DT")
if not hasattr(mophi, "NewtonXLBDEMCoupler"):
    mophi.fatal("mophi.NewtonXLBDEMCoupler is unavailable. Rebuild with -DMOPHI_BUILD_NEWTON_XLB_DEM=ON.")

mophi.check_newton_warp_mujoco_versions(
    mophi.REQUIRED_NEWTON_VERSION,
    mophi.REQUIRED_WARP_VERSION,
    mophi.REQUIRED_MUJOCO_VERSION,
)

print("=== MoPhi robotic 3D-printing machine (Newton stage) ===")
wp.init()
device = wp.get_device()

print("[Newton] Downloading the Universal Robots UR10 asset ...")
asset_path = newton.utils.download_asset("universal_robots_ur10")
asset_file = str(asset_path / "usd" / "ur10_instanceable.usda")

robot_builder = newton.ModelBuilder()
newton.solvers.SolverMuJoCo.register_custom_attributes(robot_builder)
robot_builder.add_usd(
    asset_file,
    xform=wp.transform(wp.vec3(0.0, 0.0, PEDESTAL_HEIGHT)),
    collapse_fixed_joints=False,
    enable_self_collisions=False,
    hide_collision_shapes=True,
)
robot_builder.add_shape_cylinder(
    -1,
    xform=wp.transform(wp.vec3(0.0, 0.0, PEDESTAL_HEIGHT / 2.0)),
    half_height=PEDESTAL_HEIGHT / 2.0,
    radius=PEDESTAL_RADIUS,
    label="robot_pedestal",
)

ee_link_body_idx = robot_builder.body_label.index("/ur10/ee_link")
if not FUNNEL_OBJ_PATH.is_file():
    mophi.fatal(f"Required print-head mesh is missing: {FUNNEL_OBJ_PATH}")

head_cfg = newton.ModelBuilder.ShapeConfig(has_shape_collision=True, has_particle_collision=True)
funnel_surface = mophi.load_obj(str(FUNNEL_OBJ_PATH), scale=FUNNEL_MESH_SCALE)
funnel_mesh = newton.Mesh(
    funnel_surface.vertices,
    funnel_surface.indices,
    compute_inertia=False,
    is_solid=False,
    color=FUNNEL_MESH_COLOR,
)
robot_builder.add_shape_mesh(
    ee_link_body_idx,
    xform=wp.transform(
        wp.vec3(*NOZZLE_OUTLET_LOCAL),
        wp.quat_from_axis_angle(
            wp.vec3(*FUNNEL_MESH_LOCAL_ROTATION_AXIS),
            FUNNEL_MESH_LOCAL_ROTATION_ANGLE,
        ),
    ),
    mesh=funnel_mesh,
    cfg=head_cfg,
    label="funnel_print_head",
)

for joint_index in range(len(robot_builder.joint_target_ke)):
    robot_builder.joint_target_ke[joint_index] = JOINT_TARGET_STIFFNESS
    robot_builder.joint_target_kd[joint_index] = JOINT_TARGET_DAMPING
    robot_builder.joint_target_mode[joint_index] = int(JointTargetMode.POSITION)

builder = newton.ModelBuilder()
builder.replicate(robot_builder, WORLD_COUNT, spacing=(2.0, 2.0, 0.0))
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
for x_sign in (-1.0, 1.0):
    for y_sign in (-1.0, 1.0):
        builder.add_shape_box(
            -1,
            xform=wp.transform(
                wp.vec3(
                    BUILD_PLATE_CENTER[0] + x_sign * BUILD_PLATE_HALF_EXTENTS[0],
                    BUILD_PLATE_CENTER[1] + y_sign * BUILD_PLATE_HALF_EXTENTS[1],
                    BUILD_FRAME_POST_HALF_EXTENTS[2],
                )
            ),
            hx=BUILD_FRAME_POST_HALF_EXTENTS[0],
            hy=BUILD_FRAME_POST_HALF_EXTENTS[1],
            hz=BUILD_FRAME_POST_HALF_EXTENTS[2],
            label=f"build_station_post_{int(x_sign):+d}_{int(y_sign):+d}",
        )

newton_model = builder.finalize()
newton_solver = newton.solvers.SolverMuJoCo(newton_model, disable_contacts=True)

OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
if USE_OMNIVERSE_VISUALIZATION:
    vis = mophi.OmniverseVisualizer(output_path=str(USD_OUTPUT_PATH), fps=RENDER_FPS)
else:
    vis = mophi.OpenGLVisualizer(newton_model)
vis.set_camera(pos=wp.vec3(*CAMERA_POSITION), pitch=CAMERA_PITCH, yaw=CAMERA_YAW)
mophi.log_orientation_and_scale_reference(
    vis,
    axis_origin=REFERENCE_AXIS_ORIGIN,
    scale_bar_center=REFERENCE_SCALE_BAR_CENTER,
)

coupler = mophi.NewtonXLBDEMCoupler()
coupler.initialize(
    newton_model=newton_model,
    newton_solver=newton_solver,
    xlb_simulation=None,
    deme_solver=None,
    sim_dt=NEWTON_DT,
)
arm_view = ArticulationView(
    newton_model,
    "*ur10*",
    exclude_joint_types=[newton.JointType.FREE, newton.JointType.DISTANCE],
)
if arm_view.count != WORLD_COUNT:
    mophi.fatal(f"Expected {WORLD_COUNT} UR10 articulation, found {arm_view.count}.")

joint_values = arm_view.get_attribute("joint_q", coupler.newton_state_0).numpy()
controlled_dofs = min(arm_view.joint_dof_count, len(APPROACH_Q))


def _set_joint_target(target: np.ndarray, set_state: bool = False) -> None:
    joint_values[:, 0, :controlled_dofs] = target[:controlled_dofs]
    target_wp = wp.array(joint_values, dtype=wp.float32, device=device)
    arm_view.set_attribute("joint_target_pos", coupler.newton_control, target_wp)
    if set_state:
        arm_view.set_attribute("joint_q", coupler.newton_state_0, target_wp)


_set_joint_target(APPROACH_Q, set_state=True)
for _ in range(WARMUP_STEPS):
    coupler.step_newton()

movie_writer = None
if SAVE_MOVIE and not USE_OMNIVERSE_VISUALIZATION:
    movie_writer = imageio.get_writer(MOVIE_OUTPUT_PATH, fps=MOVIE_FPS)

sim_time = 0.0
frames_completed = 0
try:
    for frame in range(NUM_FRAMES):
        if not vis.is_running():
            break
        _set_joint_target(_motion_target(sim_time))
        for _ in range(SIM_SUBSTEPS):
            coupler.step_newton()
        sim_time += FRAME_DT

        vis.begin_frame(sim_time)
        vis.log_state(coupler.newton_state_0)
        vis.end_frame()
        if movie_writer is not None:
            movie_writer.append_data(vis.get_frame().numpy())
        frames_completed = frame + 1
finally:
    if movie_writer is not None:
        movie_writer.close()
    vis.close()
    coupler.finalize()

METADATA_OUTPUT_PATH.write_text(
    json.dumps(
        {
            "phase": "newton_machine_only",
            "robot_asset": asset_file,
            "print_head_mesh": str(FUNNEL_OBJ_PATH),
            "print_head_mesh_scale": FUNNEL_MESH_SCALE,
            "frames_completed": frames_completed,
            "newton_dt": NEWTON_DT,
            "render_fps": RENDER_FPS,
            "nozzle_outlet_local": NOZZLE_OUTLET_LOCAL,
            "deme_dfc_material_enabled": False,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(f"Completed {frames_completed} frames. Output: {OUTPUT_DIRECTORY}")
