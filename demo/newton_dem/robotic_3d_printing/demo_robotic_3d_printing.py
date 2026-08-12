"""Newton/DEME co-simulation stage for robotic granular 3D printing.

Newton owns the UR10 and its funnel print head. DEME owns a matching kinematic
funnel proxy and the granular material, and feeds proxy contact loads directly
from DEME CUDA storage into Newton's device-resident body-force array.
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
from mophi.utils import load_package_provider

DEME = load_package_provider("deme")

# ── User configuration ────────────────────────────────────────────────────────────────────────
WORLD_COUNT = 1
NEWTON_DT = 1.0 / 500.0
DEME_DT = 1.0 / 6000.0
NEWTON_GRAVITY = (0.0, 0.0, 0.0)
RENDER_FPS = 50
PRINT_MOTION_SECONDS = 5.0
WARMUP_SECONDS = 0.5
RENDER_SETTLING_PHASE = True

# These adjacent targets define the complete one-way printing trajectory. Only
# the base joint changes, producing a small, nearly straight lateral trail.
PRINT_START_Q = np.array([0.50 * np.pi, 0.80, -0.90, -1.47, -0.50 * np.pi, 0.0], dtype=np.float32)
PRINT_END_Q = np.array([0.40 * np.pi, 0.80, -0.90, -1.47, -0.50 * np.pi, 0.0], dtype=np.float32)
JOINT_TARGET_STIFFNESS = 500.0
JOINT_TARGET_DAMPING = 50.0

PEDESTAL_HEIGHT = 1.2
PEDESTAL_RADIUS = 0.09
BUILD_PLATE_CENTER = (0.0, 0.82, 0.06)
BUILD_PLATE_HALF_EXTENTS = (0.55, 0.55, 0.06)
BUILD_FRAME_POST_HALF_EXTENTS = (0.035, 0.035, 0.36)
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FUNNEL_OBJ_PATH = REPOSITORY_ROOT / "data" / "mesh" / "funnel.obj"
COHESION_FORCE_MODEL_PATH = (
    REPOSITORY_ROOT / "data" / "force_models" / "ForceModelWithCohesion.cu"
)
FUNNEL_MESH_SCALE = 0.001
FUNNEL_MESH_COLOR = (0.18, 0.20, 0.23)
FUNNEL_MESH_LOCAL_ROTATION_AXIS = (0.0, 1.0, 0.0)
FUNNEL_MESH_LOCAL_ROTATION_ANGLE = -0.5 * np.pi
# The wrist orientation maps its local +X direction downward. Extending this
# mount offset lowers the funnel while preserving the arm's lateral trajectory.
NOZZLE_OUTLET_LOCAL = (0.65, 0.0, 0.0)

DEME_PARTICLE_RADIUS = 0.002
DEME_PARTICLE_VISUAL_RADIUS = 0.002
DEME_PARTICLE_DENSITY = 1100.0
DEME_PARTICLE_COLOR = (0.90, 0.35, 0.08)
DEME_PARTICLE_FAMILY = 0
DEME_FUNNEL_FAMILY = 10
# Newton always drives the DEME funnel pose. Enable this reverse path only
# when DEME particle loads should influence the Newton-controlled arm.
ENABLE_DEME_TO_NEWTON_FORCE_FEEDBACK = False
DEME_DOMAIN_X = (-0.75, 0.75)
DEME_DOMAIN_Y = (0.25, 1.55)
# Keep the box bottom below the explicit build-plate plane so the two
# analytical boundaries are not coincident.
DEME_DOMAIN_Z = (0.0, 1.20)
# This funnel-local box stays within the bowl's inner volume. Its points are
# transformed to world space through the initial Newton-derived funnel pose.
DEME_PARTICLE_SAMPLE_BOX_CENTER_LOCAL = (0.0, 0.0, 0.10)
DEME_PARTICLE_SAMPLE_BOX_HALF_EXTENTS = (0.045, 0.045, 0.040)
DEME_PARTICLE_GRID_SPACING = DEME_PARTICLE_RADIUS * 2.1
# Keep cohesion disabled while isolating the custom force-model kernel from
# subsequent material and domain changes.
DEME_COHESION = 350.0
DEME_WALL_MATERIAL = {
    "E": 5.0e5,
    "nu": 0.3,
    "CoR": 0.25,
    "mu": 0.45,
    "Crr": 0.0,
    "Cohesion": DEME_COHESION,
}
DEME_PARTICLE_MATERIAL = {
    "E": 5.0e5,
    "nu": 0.3,
    "CoR": 0.25,
    "mu": 0.45,
    "Crr": 0.0,
    "Cohesion": DEME_COHESION,
}

USE_OMNIVERSE_VISUALIZATION = False
SAVE_MOVIE = True
# The view direction is derived below to keep the funnel bowl in focus.
CAMERA_POSITION = (3.1, 0.82, 2.2)
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
DEME_SUBSTEPS = int(round(NEWTON_DT / DEME_DT))
NUM_FRAMES = int(round(PRINT_MOTION_SECONDS / FRAME_DT))
WARMUP_STEPS = int(round(WARMUP_SECONDS / NEWTON_DT))


@wp.kernel
def _write_deme_proxy_load(
    contact_acceleration: wp.array(dtype=wp.vec3),
    angular_acceleration_global: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=wp.float32),
    moi_local: wp.array(dtype=wp.vec3),
    orientation: wp.array(dtype=wp.quat),
    body_index: int,
    body_forces: wp.array(dtype=wp.spatial_vector),
):
    """Convert DEME proxy accelerations to a Newton world-space wrench on-device."""
    force = mass[0] * contact_acceleration[0]
    q = orientation[0]
    alpha_local = wp.quat_rotate_inv(q, angular_acceleration_global[0])
    torque_local = wp.cw_mul(moi_local[0], alpha_local)
    torque_global = wp.quat_rotate(q, torque_local)
    # TODO(wrench reference point): DEME reports this torque about its funnel
    # owner frame, but Newton applies the wrench at the end-effector body. Add
    # (p_funnel - p_end_effector) x force before enabling feedback for studies
    # where the arm's rotational response must be physically accurate.
    body_forces[body_index] = wp.spatial_vector(force, torque_global)


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _motion_target(sim_time: float) -> np.ndarray:
    """Interpolate once from the starting posture to the lateral endpoint."""
    phase = _smoothstep(sim_time / PRINT_MOTION_SECONDS)
    return (1.0 - phase) * PRINT_START_Q + phase * PRINT_END_Q


def _quat_multiply_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compose xyzw quaternions without introducing another runtime dependency."""
    av, aw = a[:3], a[3]
    bv, bw = b[:3], b[3]
    return np.append(aw * bv + bw * av + np.cross(av, bv), aw * bw - np.dot(av, bv))


def _quat_rotate_xyzw(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return vector + 2.0 * np.cross(q[:3], np.cross(q[:3], vector) + q[3] * vector)


local_axis = np.asarray(FUNNEL_MESH_LOCAL_ROTATION_AXIS, dtype=np.float32)
local_axis /= np.linalg.norm(local_axis)
local_half_angle = 0.5 * FUNNEL_MESH_LOCAL_ROTATION_ANGLE
funnel_local_q = np.append(local_axis * np.sin(local_half_angle), np.cos(local_half_angle)).astype(np.float32)
funnel_local_position = np.asarray(NOZZLE_OUTLET_LOCAL, dtype=np.float32)


if not np.isclose(SIM_SUBSTEPS * NEWTON_DT, FRAME_DT, rtol=0.0, atol=1.0e-12):
    raise ValueError("FRAME_DT must be an integer multiple of NEWTON_DT")
if not np.isclose(DEME_SUBSTEPS * DEME_DT, NEWTON_DT, rtol=0.0, atol=1.0e-12):
    raise ValueError("NEWTON_DT must be an integer multiple of DEME_DT")
if not hasattr(mophi, "NewtonXLBDEMCoupler"):
    mophi.fatal("mophi.NewtonXLBDEMCoupler is unavailable. Rebuild with -DMOPHI_BUILD_NEWTON_XLB_DEM=ON.")

mophi.check_newton_warp_mujoco_versions(
    mophi.REQUIRED_NEWTON_VERSION,
    mophi.REQUIRED_WARP_VERSION,
    mophi.REQUIRED_MUJOCO_VERSION,
)

print("=== MoPhi robotic 3D-printing Newton/DEME co-simulation ===")
wp.init()
device = wp.get_device()
if not device.is_cuda:
    mophi.fatal("This demo requires a CUDA Warp device for direct DEME/Newton pointer exchange.")

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
if len(robot_builder.joint_q) < len(PRINT_START_Q):
    mophi.fatal(f"UR10 exposes only {len(robot_builder.joint_q)} joint coordinates; expected {len(PRINT_START_Q)}.")
robot_builder.joint_q[: len(PRINT_START_Q)] = PRINT_START_Q.tolist()

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
newton_model.set_gravity(NEWTON_GRAVITY)
newton_solver = newton.solvers.SolverMuJoCo(newton_model, disable_contacts=True)
initial_newton_state = newton_model.state()
newton.eval_fk(newton_model, newton_model.joint_q, newton_model.joint_qd, initial_newton_state)
initial_ee_transform = initial_newton_state.body_q.numpy()[ee_link_body_idx]
initial_ee_position = initial_ee_transform[:3]
initial_ee_q = initial_ee_transform[3:7]
initial_funnel_position = initial_ee_position + _quat_rotate_xyzw(initial_ee_q, funnel_local_position)
initial_funnel_q = _quat_multiply_xyzw(initial_ee_q, funnel_local_q)
initial_funnel_focus = initial_funnel_position + _quat_rotate_xyzw(
    initial_funnel_q,
    np.asarray(DEME_PARTICLE_SAMPLE_BOX_CENTER_LOCAL),
)
camera_to_funnel = initial_funnel_focus - np.asarray(CAMERA_POSITION)
camera_yaw = np.degrees(np.arctan2(camera_to_funnel[1], camera_to_funnel[0]))
camera_pitch = np.degrees(
    np.arctan2(camera_to_funnel[2], np.hypot(camera_to_funnel[0], camera_to_funnel[1]))
)

print("[DEME] Creating granular material and the matching funnel contact proxy ...")
if not COHESION_FORCE_MODEL_PATH.is_file():
    mophi.fatal(f"Required DEME cohesion force model is missing: {COHESION_FORCE_MODEL_PATH}")
deme_solver = DEME.DEMSolver([device.ordinal])
if device.ordinal not in deme_solver.GetGPUDeviceIDs():
    mophi.fatal(
        f"DEME workers {deme_solver.GetGPUDeviceIDs()} do not share Warp CUDA device {device.ordinal}."
    )
cohesion_force_model = deme_solver.ReadContactForceModel(str(COHESION_FORCE_MODEL_PATH))
cohesion_force_model.SetMustHaveMatProp({"E", "nu", "CoR", "mu", "Crr", "Cohesion"})
cohesion_force_model.SetMustPairwiseMatProp({"CoR", "mu", "Crr", "Cohesion"})
cohesion_force_model.SetPerContactWildcards(
    {"delta_time", "delta_tan_x", "delta_tan_y", "delta_tan_z"}
)
# deme_solver.SetVerbosity("ERROR")
wall_material = deme_solver.LoadMaterial(DEME_WALL_MATERIAL)
particle_material = deme_solver.LoadMaterial(DEME_PARTICLE_MATERIAL)
deme_solver.SetMaterialPropertyPair("CoR", wall_material, particle_material, DEME_PARTICLE_MATERIAL["CoR"])
deme_solver.SetMaterialPropertyPair("mu", wall_material, particle_material, DEME_PARTICLE_MATERIAL["mu"])
deme_solver.SetMaterialPropertyPair("Crr", wall_material, particle_material, DEME_PARTICLE_MATERIAL["Crr"])
deme_solver.SetMaterialPropertyPair("Cohesion", particle_material, particle_material, DEME_COHESION)
deme_solver.SetMaterialPropertyPair("Cohesion", wall_material, particle_material, DEME_COHESION)
particle_mass = DEME_PARTICLE_DENSITY * (4.0 / 3.0) * np.pi * DEME_PARTICLE_RADIUS**3
particle_template = deme_solver.LoadSphereType(particle_mass, DEME_PARTICLE_RADIUS, particle_material)

particle_sampler = DEME.GridSampler(DEME_PARTICLE_GRID_SPACING)
particle_positions_local = particle_sampler.SampleBox(
    DEME_PARTICLE_SAMPLE_BOX_CENTER_LOCAL,
    DEME_PARTICLE_SAMPLE_BOX_HALF_EXTENTS,
)
particle_positions = [
    (initial_funnel_position + _quat_rotate_xyzw(initial_funnel_q, np.asarray(position))).tolist()
    for position in particle_positions_local
]
deme_particle_count = len(particle_positions)
if deme_particle_count == 0:
    mophi.fatal("The configured DEME particle sampling box produced no particles.")
particles = deme_solver.AddClumps(particle_template, particle_positions)
particles.SetFamily(DEME_PARTICLE_FAMILY)
particle_tracker = deme_solver.Track(particles)

deme_funnel = deme_solver.AddWavefrontMeshObject(str(FUNNEL_OBJ_PATH), wall_material)
deme_funnel.Scale(FUNNEL_MESH_SCALE)
deme_funnel.SetInitPos(initial_funnel_position.tolist())
deme_funnel.SetInitQuat(initial_funnel_q.tolist())
deme_funnel.SetFamily(DEME_FUNNEL_FAMILY)
deme_solver.SetFamilyFixed(DEME_FUNNEL_FAMILY)
funnel_tracker = deme_solver.Track(deme_funnel)

# These explicit XYZ ranges define the six analytical enclosure walls. The
# separate build-plate plane below remains the printing surface inside the box.
deme_solver.InstructBoxDomainDimension(DEME_DOMAIN_X, DEME_DOMAIN_Y, DEME_DOMAIN_Z)
# deme_solver.InstructBoxDomainBoundingBC("all", wall_material)
build_plate_top = BUILD_PLATE_CENTER[2] + BUILD_PLATE_HALF_EXTENTS[2]
deme_solver.AddBCPlane(
    [BUILD_PLATE_CENTER[0], BUILD_PLATE_CENTER[1], build_plate_top],
    [0.0, 0.0, 1.0],
    wall_material,
)
deme_solver.SetGravitationalAcceleration([0.0, 0.0, -9.81])
deme_solver.SetInitTimeStep(DEME_DT)
deme_solver.SetErrorOutAvgContacts(1000)
# deme_solver.DisableAdaptiveBinSize()
deme_solver.Initialize()
print(f"[DEME] Initialized {deme_particle_count} spheres and a {deme_funnel.GetNumTriangles()}-triangle funnel.")

OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
if USE_OMNIVERSE_VISUALIZATION:
    vis = mophi.OmniverseVisualizer(output_path=str(USD_OUTPUT_PATH), fps=RENDER_FPS)
else:
    vis = mophi.OpenGLVisualizer(newton_model)
vis.set_camera(pos=wp.vec3(*CAMERA_POSITION), pitch=camera_pitch, yaw=camera_yaw)
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
    deme_solver=deme_solver,
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
controlled_dofs = min(arm_view.joint_dof_count, len(PRINT_START_Q))


def _set_joint_target(target: np.ndarray, set_state: bool = False) -> None:
    joint_values[:, 0, :controlled_dofs] = target[:controlled_dofs]
    target_wp = wp.array(joint_values, dtype=wp.float32, device=device)
    arm_view.set_attribute("joint_target_pos", coupler.newton_control, target_wp)
    if set_state:
        arm_view.set_attribute("joint_q", coupler.newton_state_0, target_wp)


def _sync_newton_funnel_pose_to_deme() -> None:
    """Drive DEME's fixed funnel proxy from the corresponding Newton body."""
    # TODO(deme3 device pose setters): deme3 3.0.1 provides device-pointer getters,
    # but Tracker.SetPos/SetOriQ remain host-only. This is the sole host-staged
    # physics exchange in this demo. Replace this block when APIs equivalent to
    # the prospective calls below become available:
    # funnel_tracker.SetPositionsFromDevice(newton_body_q.ptr, 1, device.ordinal)
    # funnel_tracker.SetOrientationQuaternionsFromDevice(newton_body_q.ptr, 1, device.ordinal)
    ee_transform = coupler.get_newton_body_q_array().numpy()[ee_link_body_idx]
    ee_position = ee_transform[:3]
    ee_q = ee_transform[3:7]
    funnel_position = ee_position + _quat_rotate_xyzw(ee_q, funnel_local_position)
    funnel_q = _quat_multiply_xyzw(ee_q, funnel_local_q)
    funnel_tracker.SetPos(funnel_position.tolist())
    funnel_tracker.SetOriQ(funnel_q.tolist())


deme_contact_acceleration = wp.empty(1, dtype=wp.vec3, device=device)
deme_angular_acceleration = wp.empty(1, dtype=wp.vec3, device=device)
deme_funnel_mass = wp.empty(1, dtype=wp.float32, device=device)
deme_funnel_moi = wp.empty(1, dtype=wp.vec3, device=device)
deme_funnel_orientation = wp.empty(1, dtype=wp.quat, device=device)
newton_external_body_forces = wp.zeros(newton_model.body_count, dtype=wp.spatial_vector, device=device)
deme_particle_positions = wp.empty(deme_particle_count, dtype=wp.vec3, device=device)
deme_particle_radii = wp.full(deme_particle_count, DEME_PARTICLE_VISUAL_RADIUS, dtype=wp.float32, device=device)
deme_particle_colors = wp.full(deme_particle_count, DEME_PARTICLE_COLOR, dtype=wp.vec3, device=device)
if ENABLE_DEME_TO_NEWTON_FORCE_FEEDBACK:
    coupler.set_newton_body_forces(newton_external_body_forces)


def _update_deme_funnel_load_on_device() -> None:
    """Copy DEME contact response into Newton's registered wrench buffer."""
    funnel_tracker.ContactAccelerationsToDevice(deme_contact_acceleration.ptr, 1, device.ordinal)
    funnel_tracker.ContactAngularAccelerationsGlobalToDevice(deme_angular_acceleration.ptr, 1, device.ordinal)
    funnel_tracker.MassesToDevice(deme_funnel_mass.ptr, 1, device.ordinal)
    funnel_tracker.MOIsToDevice(deme_funnel_moi.ptr, 1, device.ordinal)
    funnel_tracker.OrientationQuaternionsToDevice(deme_funnel_orientation.ptr, 1, device.ordinal)
    wp.launch(
        _write_deme_proxy_load,
        dim=1,
        inputs=[
            deme_contact_acceleration,
            deme_angular_acceleration,
            deme_funnel_mass,
            deme_funnel_moi,
            deme_funnel_orientation,
            ee_link_body_idx,
            newton_external_body_forces,
        ],
        device=device,
    )


movie_writer = None
if SAVE_MOVIE and not USE_OMNIVERSE_VISUALIZATION:
    movie_writer = imageio.get_writer(MOVIE_OUTPUT_PATH, fps=MOVIE_FPS)

sim_time = 0.0
frames_completed = 0
settling_frames_completed = 0


def _render_scene(frame_time: float) -> None:
    """Render the current coupled Newton/DEME state at the configured cadence."""
    particle_tracker.PositionsToDevice(deme_particle_positions.ptr, deme_particle_count, device.ordinal)
    vis.begin_frame(frame_time)
    vis.log_state(coupler.newton_state_0)
    vis.log_points(
        "deme_print_material",
        deme_particle_positions,
        radii=deme_particle_radii,
        colors=deme_particle_colors,
    )
    vis.end_frame()
    if movie_writer is not None:
        movie_writer.append_data(vis.get_frame().numpy())


try:
    _set_joint_target(PRINT_START_Q, set_state=True)
    _sync_newton_funnel_pose_to_deme()
    for warmup_step in range(WARMUP_STEPS):
        coupler.step_newton()
        _sync_newton_funnel_pose_to_deme()
        for _ in range(DEME_SUBSTEPS):
            coupler.step_deme()
        if ENABLE_DEME_TO_NEWTON_FORCE_FEEDBACK:
            _update_deme_funnel_load_on_device()
        sim_time += NEWTON_DT

        settling_frame_due = (warmup_step + 1) % SIM_SUBSTEPS == 0 or warmup_step + 1 == WARMUP_STEPS
        if RENDER_SETTLING_PHASE and settling_frame_due:
            if not vis.is_running():
                break
            _render_scene(sim_time)
            settling_frames_completed += 1
            frames_completed += 1

    motion_time = 0.0
    for frame in range(NUM_FRAMES):
        if not vis.is_running():
            break
        _set_joint_target(_motion_target(motion_time))
        for _ in range(SIM_SUBSTEPS):
            coupler.step_newton()
            _sync_newton_funnel_pose_to_deme()
            for _ in range(DEME_SUBSTEPS):
                coupler.step_deme()
            if ENABLE_DEME_TO_NEWTON_FORCE_FEEDBACK:
                _update_deme_funnel_load_on_device()
        sim_time += FRAME_DT
        motion_time += FRAME_DT
        _render_scene(sim_time)
        frames_completed += 1
finally:
    if movie_writer is not None:
        movie_writer.close()
    vis.close()
    coupler.finalize()

METADATA_OUTPUT_PATH.write_text(
    json.dumps(
        {
            "phase": "newton_deme_cosimulation",
            "robot_asset": asset_file,
            "print_head_mesh": str(FUNNEL_OBJ_PATH),
            "print_head_mesh_scale": FUNNEL_MESH_SCALE,
            "frames_completed": frames_completed,
            "settling_frames_completed": settling_frames_completed,
            "render_settling_phase": RENDER_SETTLING_PHASE,
            "newton_dt": NEWTON_DT,
            "newton_gravity": NEWTON_GRAVITY,
            "deme_dt": DEME_DT,
            "render_fps": RENDER_FPS,
            "nozzle_outlet_local": NOZZLE_OUTLET_LOCAL,
            "deme_particle_count": deme_particle_count,
            "deme_contact_model": "ForceModelWithCohesion.cu",
            "deme_cohesion": DEME_COHESION,
            "deme_funnel_proxy_enabled": True,
            "newton_to_deme_pose_coupling_enabled": True,
            "deme_to_newton_force_feedback_enabled": ENABLE_DEME_TO_NEWTON_FORCE_FEEDBACK,
            "device_force_exchange": ENABLE_DEME_TO_NEWTON_FORCE_FEEDBACK,
            "device_pose_exchange": False,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(f"Completed {frames_completed} frames. Output: {OUTPUT_DIRECTORY}")
