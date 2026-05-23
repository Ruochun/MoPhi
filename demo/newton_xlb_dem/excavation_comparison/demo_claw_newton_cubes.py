"""demo/newton_xlb_dem/excavation_comparison/demo_claw_newton_cubes.py

Newton-only comparison demo for UR10 excavator plowing with a coarse cube terrain.

This script mirrors the control policy used by ``demo_claw_newton.py`` in the same
folder (same UR10 arm, same plow mesh, same ready-to-plow → plow trajectory), but
replaces DEME granular material with a coarse Newton collision representation:
an array of static cubes occupying roughly the same workspace volume.

The goal is to provide an A/B comparison where terrain representation differs while
the robot policy stays the same.

Run:
    python demo/newton_xlb_dem/excavation_comparison/demo_claw_newton_cubes.py
or:
    python -m demo.newton_xlb_dem.excavation_comparison.demo_claw_newton_cubes
"""

import os

import numpy as np

import mophi

if not hasattr(mophi, "NewtonXLBDEMCoupler"):
    mophi.fatal("mophi.NewtonXLBDEMCoupler is not available.\n" "Re-build MoPhi with -DMOPHI_BUILD_NEWTON_XLB_DEM=ON.")

import mujoco
import newton
import warp as wp
from newton import JointTargetMode
from newton.selection import ArticulationView

mophi.check_newton_warp_mujoco_versions(
    mophi.REQUIRED_NEWTON_VERSION,
    mophi.REQUIRED_WARP_VERSION,
    mophi.REQUIRED_MUJOCO_VERSION,
)

# ── World and timing ────────────────────────────────────────────────────────
WORLD_COUNT = 1
RENDER_FPS = 50
NEWTON_DT = 1.0 / 500.0
FRAME_DT = 1.0 / RENDER_FPS
SIM_SUBSTEPS = int(round(FRAME_DT / NEWTON_DT))
if not np.isclose(SIM_SUBSTEPS * NEWTON_DT, FRAME_DT, rtol=0.0, atol=1.0e-12):
    raise ValueError(
        "FRAME_DT must be an integer multiple of NEWTON_DT. "
        f"Got FRAME_DT={FRAME_DT:.12g}, NEWTON_DT={NEWTON_DT:.12g}."
    )

PAUSE_AFTER_FIRST_FRAME = False
NUM_FRAMES = 1 if PAUSE_AFTER_FIRST_FRAME else 250
NEWTON_WARMUP_FRAMES = 50

# ── Geometry and control policy ─────────────────────────────────────────────
PEDESTAL_HEIGHT = 1.2
OBJ_CM_TO_M = 0.01
READY_TO_PLOW_Q = np.array([0.5 * np.pi, -1.35, 1.20, -0.20, -1.0 * np.pi, -1.0], dtype=np.float32)
PLOW_TARGET_Q = np.array([0.5 * np.pi, -0.45, 0.65, -0.45, -1.0 * np.pi, -1.0], dtype=np.float32)
PLOW_DURATION = 1.6
WRIST_SCOOP_START_FRACTION = 0.75
WRIST_SCOOP_INWARD_DELTA = -1.8
_WRIST_3_DOF_INDEX = 5
_MIN_DURATION_EPSILON = 1.0e-12

# ── Coarse Newton cube terrain (static obstacle field) ─────────────────────
# Coarse approximation of the granular pile footprint/height:
# centered near the same y-offset used by the DEME pile in demo_claw_newton.py.
_CUBE_HALF = 0.09
_CUBE_GRID_X = (-0.54, -0.18, 0.18, 0.54)
_CUBE_GRID_Y = (0.66, 1.02, 1.38, 1.74)
_CUBE_GRID_Z = (0.12, 0.33)

# ── Setup ───────────────────────────────────────────────────────────────────
print("=== MoPhi UR10 claw demo (Newton coarse-cube comparison) ===\n")
wp.init()
device = wp.get_device()

_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_DEMO_DIR)))
EXCAVATOR_OBJ_PATH = os.path.join(_REPO_ROOT, "data", "mesh", "excavator.obj")

print("[Newton] Downloading UR10 robot assets ...")
asset_path = newton.utils.download_asset("universal_robots_ur10")
asset_file = str(asset_path / "usd" / "ur10_instanceable.usda")
print(f"[Newton] Assets ready: {asset_path}\n")

ur10_sub = newton.ModelBuilder()
newton.solvers.SolverMuJoCo.register_custom_attributes(ur10_sub)
ur10_sub.add_usd(
    asset_file,
    xform=wp.transform(wp.vec3(0.0, 0.0, PEDESTAL_HEIGHT)),
    collapse_fixed_joints=False,
    enable_self_collisions=False,
    hide_collision_shapes=True,
)
ur10_sub.add_shape_cylinder(
    -1,
    xform=wp.transform(wp.vec3(0.0, 0.0, PEDESTAL_HEIGHT / 2.0)),
    half_height=PEDESTAL_HEIGHT / 2.0,
    radius=0.08,
)

# Excavator plow mesh (visual mesh and collision shape attached to ee_link).
PLOW_LOCAL_ROT = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(np.pi))
ee_link_body_idx = ur10_sub.body_label.index("/ur10/ee_link")
_plow_surface = mophi.load_obj(EXCAVATOR_OBJ_PATH, scale=OBJ_CM_TO_M)
plow_mesh = newton.Mesh(
    _plow_surface.vertices,
    _plow_surface.indices,
    compute_inertia=False,
    is_solid=False,
    color=(0.55, 0.45, 0.35),
)
ur10_sub.add_shape_mesh(
    ee_link_body_idx,
    mesh=plow_mesh,
    xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), PLOW_LOCAL_ROT),
)

# Proxy collision box attached to the plow link to make tool–cube interaction
# robust even for open surface mesh regions.
ur10_sub.add_shape_box(
    ee_link_body_idx,
    xform=wp.transform(wp.vec3(0.0, 0.06, -0.09), wp.quat_identity()),
    hx=0.10,
    hy=0.07,
    hz=0.05,
)

for i in range(len(ur10_sub.joint_target_ke)):
    ur10_sub.joint_target_ke[i] = 500
    ur10_sub.joint_target_kd[i] = 50
    ur10_sub.joint_target_mode[i] = int(JointTargetMode.POSITION)

builder = newton.ModelBuilder()
builder.replicate(ur10_sub, WORLD_COUNT, spacing=(2.0, 2.0, 0.0))
builder.add_ground_plane()

cube_count = 0
for z in _CUBE_GRID_Z:
    for y in _CUBE_GRID_Y:
        for x in _CUBE_GRID_X:
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(float(x), float(y), float(z)), wp.quat_identity()),
                hx=_CUBE_HALF,
                hy=_CUBE_HALF,
                hz=_CUBE_HALF,
            )
            cube_count += 1

newton_model = builder.finalize()
newton_solver = newton.solvers.SolverMuJoCo(
    newton_model,
    disable_contacts=False,
)

print(f"[Newton] Model built: {newton_model.body_count} bodies, {newton_model.joint_count} joints.")
print(f"[Terrain] Added {cube_count} coarse Newton cube obstacle(s).\n")

vis = mophi.OpenGLVisualizer(newton_model)
vis.set_camera(
    pos=wp.vec3(3.0, -3.0, 2.5),
    pitch=-25.0,
    yaw=135.0,
)

coupler = mophi.NewtonXLBDEMCoupler()
coupler.set_verbosity(mophi.VERBOSITY_INFO)
coupler.initialize(
    newton_model=newton_model,
    newton_solver=newton_solver,
    xlb_simulation=None,
    deme_solver=None,
    sim_dt=NEWTON_DT,
)

articulation_view = ArticulationView(
    newton_model,
    "*ur10*",
    exclude_joint_types=[newton.JointType.FREE, newton.JointType.DISTANCE],
)
assert articulation_view.count == WORLD_COUNT, f"Expected {WORLD_COUNT} articulation(s), found {articulation_view.count}"

dof_count = articulation_view.joint_dof_count
joint_q_target_np = articulation_view.get_attribute("joint_q", coupler.newton_state_0).numpy()
num_ready_dofs = min(dof_count, len(READY_TO_PLOW_Q))
joint_q_target_np[:, 0, :num_ready_dofs] = READY_TO_PLOW_Q[:num_ready_dofs]
joint_q_target_wp = wp.array(joint_q_target_np, dtype=wp.float32, device=device)
articulation_view.set_attribute("joint_q", coupler.newton_state_0, joint_q_target_wp)
articulation_view.set_attribute("joint_target_pos", coupler.newton_control, joint_q_target_wp)

print(f"[Control] Warm-up: {NEWTON_WARMUP_FRAMES} frame(s) × {SIM_SUBSTEPS} substeps ...")
for _ in range(NEWTON_WARMUP_FRAMES * SIM_SUBSTEPS):
    coupler.step_newton()
print("[Control] Warm-up complete.\n")

sim_time = 0.0
_sim_duration = max(NUM_FRAMES * FRAME_DT, _MIN_DURATION_EPSILON)
_scoop_start_time = max(WRIST_SCOOP_START_FRACTION * _sim_duration, PLOW_DURATION)
_scoop_phase_dt = max(_sim_duration - _scoop_start_time, _MIN_DURATION_EPSILON)
_min_ee_height = float("inf")

print(
    f"Running up to {NUM_FRAMES} frame(s) "
    f"(frame_dt={FRAME_DT * 1000:.1f} ms, "
    f"{SIM_SUBSTEPS} Newton substeps × {NEWTON_DT * 1000:.3f} ms) ...\n"
)

for frame in range(NUM_FRAMES):
    if not vis.is_running():
        print(f"\n[Viewer] Window closed by user after frame {frame}.")
        break

    _plow_interp = np.clip(sim_time / PLOW_DURATION, 0.0, 1.0)
    _plow_q = (1.0 - _plow_interp) * READY_TO_PLOW_Q + _plow_interp * PLOW_TARGET_Q
    _joint_q_cmd = _plow_q.copy()
    if num_ready_dofs > _WRIST_3_DOF_INDEX and sim_time >= _scoop_start_time:
        _scoop_interp = np.clip((sim_time - _scoop_start_time) / _scoop_phase_dt, 0.0, 1.0)
        _wrist_3_scoop_target = PLOW_TARGET_Q[_WRIST_3_DOF_INDEX] + WRIST_SCOOP_INWARD_DELTA
        _joint_q_cmd[_WRIST_3_DOF_INDEX] = (1.0 - _scoop_interp) * _plow_q[
            _WRIST_3_DOF_INDEX
        ] + _scoop_interp * _wrist_3_scoop_target

    joint_q_target_np[:, 0, :num_ready_dofs] = _joint_q_cmd[:num_ready_dofs]
    articulation_view.set_attribute(
        "joint_target_pos",
        coupler.newton_control,
        wp.array(joint_q_target_np, dtype=wp.float32, device=device),
    )

    for _ in range(SIM_SUBSTEPS):
        coupler.step_newton()

    body_q_np = coupler.newton_state_0.body_q.numpy()
    _min_ee_height = min(_min_ee_height, float(body_q_np[ee_link_body_idx, 2]))
    sim_time += FRAME_DT

    vis.begin_frame(sim_time)
    vis.log_state(coupler.newton_state_0)
    vis.end_frame()

    if (frame + 1) % 50 == 0 or frame == 0:
        print(f"  frame {frame + 1:>4}/{NUM_FRAMES}  sim_time={sim_time:.3f} s")

print()

if PAUSE_AFTER_FIRST_FRAME and vis.is_running():
    print("[Debug] First frame rendered. Close the viewer window to continue.")
    while vis.is_running():
        vis.begin_frame(sim_time)
        vis.log_state(coupler.newton_state_0)
        vis.end_frame()

vis.close()
coupler.finalize()

print("Demo completed successfully.")
print(f"  Newton version : {newton.__version__}")
print(f"  Warp  version  : {wp.__version__}")
print(f"  Terrain        : Newton coarse-cube field ({cube_count} cubes)")
print(f"  Min ee_link z  : {_min_ee_height:.4f} m")
