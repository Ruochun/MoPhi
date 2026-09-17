"""demo/newton_xlb_dem/anymal_robot_multiphysics/demo_anymal_robot_multiphysics.py

Three-way co-simulation demo: Newton (ANYmal C walking robot) + XLB (LBM fluid) + DEME (particles).

This demo exercises mophi.NewtonXLBDEMCoupler, which manages all three solvers:

  • Newton   — drives the ANYmal C quadruped walking robot using MuJoCo-based
               physics (SolverMuJoCo) controlled by a pre-trained reinforcement
               learning walking policy.  The robot walks forward on a flat ground
               plane, and its spatial representation (body transforms) is extracted
               at every step.
  • XLB      — a D3Q19 incompressible Navier-Stokes LBM fluid with an explicit
               lattice-to-SI mapping. The fluid takes several smaller physical
               steps inside every Newton step. The robot is represented by a
               prescribed axis-aligned box whose device masks follow its base.
  • DEME     — a real discrete-element solver (pip install "deme[cuda12]>=3.0.14,<4").  A
               deme.DEMSolver is created in Python, populated with particles and
               contact proxies, stepped every substep, and its live particle
               positions are visualized each frame.

This three-way demo requires all three Python solvers (Newton, XLB, and DEME).
If any required module is missing, the script exits early with an actionable
install message.

The robot setup and walking policy exactly follow Newton's
``newton/examples/robot/example_robot_anymal_c_walk.py``.  The only difference
is that this demo uses a flat ground plane (no procedural terrain) and wraps
all physics inside the MoPhi three-way co-simulation coupler.

Prerequisites
-------------
  • Build MoPhi with -DMOPHI_BUILD_NEWTON_XLB_DEM=ON (compiles
    NewtonXLBDEMCoupler and builds the mophi_core Python extension module).
  • pip install --upgrade newton warp-lang mujoco==3.6.0 torch
  • pip install "xlb[cuda]"                (XLB LBM solver)
  • pip install "deme[cuda12]>=3.0.14,<4"                       (DEME discrete-element solver)

Running
-------
From the MoPhi build tree (after ``cmake --build``):

    python demo/newton_xlb_dem/anymal_robot_multiphysics/demo_anymal_robot_multiphysics.py

Or from the repository root after installing the mophi package:

    python -m demo.newton_xlb_dem.anymal_robot_multiphysics.demo_anymal_robot_multiphysics
"""

import sys
from importlib.util import find_spec
from pathlib import Path

import numpy as np

# ─── Import MoPhi ─────────────────────────────────────────────────────────
if find_spec("mophi") is None:
    sys.exit(
        "ERROR: Could not import the 'mophi' package.\n"
        "       Build MoPhi with -DMOPHI_BUILD_NEWTON_XLB_DEM=ON and ensure that\n"
        "       the build directory (python/) is on PYTHONPATH."
    )
import mophi
from mophi.couplers.newton_deme import (
    NewtonDEMEContactAccelerationCoupler,
    NewtonDEMEOwnerMap,
)
from mophi.couplers.newton_xlb import (
    NewtonBodyBoxBoundary,
    NewtonXLBHalfwayBounceBackWrench,
    NewtonXLBWrenchExchange,
    XLBPhysicalScaling,
)
from mophi.couplers.xlb_deme import XLBDEMEParticleExchange

if not hasattr(mophi, "NewtonXLBDEMCoupler"):
    mophi.fatal(
        "ERROR: mophi.NewtonXLBDEMCoupler is not available.\n"
        "       Re-build MoPhi with -DMOPHI_BUILD_NEWTON_XLB_DEM=ON."
    )

# ─── Import Newton + Warp + MuJoCo + PyTorch ───────────────────────────────
if find_spec("torch") is None:
    mophi.fatal("PyTorch is required for this demo.\nInstall with:  pip install torch")
if find_spec("newton") is None or find_spec("warp") is None or find_spec("mujoco") is None:
    mophi.fatal(
        "Newton, Warp, and MuJoCo are required for this demo.\n"
        "Install with:  pip install --upgrade newton warp-lang mujoco==3.6.0"
    )
import torch
import newton
import warp as wp
import mujoco

mophi.check_newton_warp_mujoco_versions(
    mophi.REQUIRED_NEWTON_VERSION,
    mophi.REQUIRED_WARP_VERSION,
    mophi.REQUIRED_MUJOCO_VERSION,
)

# ─── Import demo-specific utilities ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
import anymal_geometry as geometry_utils
from anymal_deme_scene import build_anymal_deme_scene, prepare_anymal_deme_population  # noqa: E402
from anymal_newton_scene import build_anymal_newton_scene  # noqa: E402
from anymal_policy import compute_observation, evaluate_joint_targets  # noqa: E402

# ─── Import XLB + DEME (required for this three-way demo) ──────────────────
if find_spec("xlb") is None:
    mophi.fatal('XLB is required for this demo.\nInstall with:  pip install "xlb[cuda]"')
import xlb
from anymal_xlb_scene import build_anymal_xlb_scene  # noqa: E402
from mophi.utils.package_provider import load_package_provider

DEME = load_package_provider("deme")

print("=== MoPhi Newton (ANYmal C) + XLB + DEME three-way co-simulation demo ===\n")


# ─── Joint-index remapping ─────────────────────────────────────────────────
# The ANYmal C RL policy was trained with legs ordered [LF, RF, LH, RH] × [HAA, HFE, KFE]
# ("lab" convention), while MuJoCo/Newton uses a different internal ordering.
# These index arrays reorder the 12 joint outputs so they apply to the correct actuators.
# Mirrors newton/examples/robot/example_robot_anymal_c_walk.py.
lab_to_mujoco = [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11]
mujoco_to_lab = [0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11]
POLICY_ACTION_SCALE = 0.5

# Newton robot construction choices retained visibly by this experiment.
ANYMAL_BASE_HEIGHT = 0.62
ANYMAL_BASE_YAW = float(np.pi * 0.5)
ANYMAL_JOINT_ARMATURE = 0.06
ANYMAL_JOINT_LIMIT_STIFFNESS = 1.0e3
ANYMAL_JOINT_LIMIT_DAMPING = 1.0e1
ANYMAL_SHAPE_STIFFNESS = 5.0e4
ANYMAL_SHAPE_DAMPING = 5.0e2
ANYMAL_SHAPE_FRICTION_STIFFNESS = 1.0e3
ANYMAL_SHAPE_FRICTION = 0.75
ANYMAL_TARGET_STIFFNESS = 150.0
ANYMAL_TARGET_DAMPING = 5.0
ANYMAL_SOLVER_ITERATIONS = 50
ANYMAL_MAXIMUM_JOINTS = 50
ANYMAL_MAXIMUM_CONTACTS = 100
FOOT_SHANK_NAMES = ["LF_SHANK", "RF_SHANK", "LH_SHANK", "RH_SHANK"]
INITIAL_JOINT_POSITIONS = {
    "RH_HAA": 0.0,
    "RH_HFE": -0.4,
    "RH_KFE": 0.8,
    "LH_HAA": 0.0,
    "LH_HFE": -0.4,
    "LH_KFE": 0.8,
    "RF_HAA": 0.0,
    "RF_HFE": 0.4,
    "RF_KFE": -0.8,
    "LF_HAA": 0.0,
    "LF_HFE": 0.4,
    "LF_KFE": -0.8,
}

# ─── Initialize Warp ───────────────────────────────────────────────────────
wp.init()
torch_device = wp.device_to_torch(wp.get_device())

# ─── Load the ANYmal C robot model ─────────────────────────────────────────
# newton.utils.download_asset("anybotics_anymal_c") downloads the ANYmal C URDF
# and pre-trained RL walking policy from the Newton Assets repository.
# The robot starts above the flat ground plane with a stable initial base pose.
print("[Newton] Downloading ANYmal C robot assets ...")
asset_path = mophi.download_newton_asset(
    "anybotics_anymal_c",
    ["urdf/anymal.urdf", "rl_policies/anymal_walking_policy_physx.pt"],
)
urdf_path = str(asset_path / "urdf" / "anymal.urdf")
policy_path = str(asset_path / "rl_policies" / "anymal_walking_policy_physx.pt")
print(f"[Newton] Assets ready: {asset_path}\n")

print("[Newton] Building ANYmal C model ...")
_newton_scene = build_anymal_newton_scene(
    urdf_path,
    geometry_utils,
    INITIAL_JOINT_POSITIONS,
    FOOT_SHANK_NAMES,
    ANYMAL_BASE_HEIGHT,
    ANYMAL_BASE_YAW,
    ANYMAL_JOINT_ARMATURE,
    ANYMAL_JOINT_LIMIT_STIFFNESS,
    ANYMAL_JOINT_LIMIT_DAMPING,
    ANYMAL_SHAPE_STIFFNESS,
    ANYMAL_SHAPE_DAMPING,
    ANYMAL_SHAPE_FRICTION_STIFFNESS,
    ANYMAL_SHAPE_FRICTION,
    ANYMAL_TARGET_STIFFNESS,
    ANYMAL_TARGET_DAMPING,
    ANYMAL_SOLVER_ITERATIONS,
    ANYMAL_MAXIMUM_JOINTS,
    ANYMAL_MAXIMUM_CONTACTS,
)
builder = _newton_scene.builder
newton_model = _newton_scene.model
newton_solver = _newton_scene.solver
builder_foot_spheres = _newton_scene.foot_spheres
builder_body_name_to_idx = _newton_scene.body_name_to_index
body_part_visual_descriptors = _newton_scene.visual_descriptors
foot_tip_descriptors = _newton_scene.foot_descriptors
foot_tip_sphere_radii = _newton_scene.foot_radii
foot_tip_body_indices = _newton_scene.foot_body_indices

print(f"[Newton] ANYmal C model built: {newton_model.body_count} bodies, " f"{newton_model.joint_count} joints.\n")

# ─── Identify foot-tip contact proxies ───────────────────────────────────
# ANYmal C has a GeoType.SPHERE collision shape at the distal end of each SHANK
# link.  These spheres are the ground-contact proxies and will serve as coupling
# surfaces for DEM particles and XLB fluid boundaries in future co-sim work.
# See anymal_geometry.build_foot_tip_descriptors for the descriptor schema.
# Visual meshes are nice, but for the production code we don't print them
# geometry_utils.print_foot_tip_descriptors(foot_tip_descriptors)
# geometry_utils.print_visual_body_part_descriptors(body_part_visual_descriptors)

# ─── Visualization backend selection ─────────────────────────────────────────
# Set USE_OMNIVERSE_VISUALIZATION = True to write each frame to a USD scene file
# (requires:  pip install usd-core).
# Set USE_OMNIVERSE_VISUALIZATION = False (default) to use Newton's real-time
# OpenGL window via mophi.OpenGLVisualizer.
# The simulation loop body (begin_frame / log_state / log_points / end_frame)
# is identical for both backends; only the constructor and close() differ.
USE_OMNIVERSE_VISUALIZATION = False
REFERENCE_AXIS_ORIGIN = (0.8, -0.9, 0.02)
REFERENCE_SCALE_BAR_CENTER = (0.0, -0.9, 0.06)
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "output" / Path(__file__).stem
OMNIVERSE_OUTPUT_PATH = OUTPUT_DIRECTORY / "demo_anymal_robot_multiphysics.usdc"

vis = None
_vis_available = False

OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
if USE_OMNIVERSE_VISUALIZATION:
    vis = mophi.OmniverseVisualizer(output_path=str(OMNIVERSE_OUTPUT_PATH), fps=50.0)
    _vis_available = vis.pxr_available
    if _vis_available:
        print("[USD] pxr (OpenUSD) available — Omniverse USD export enabled.\n")
        # Register visual mesh geometry so the USD scene shows the full robot shape
        # (body, legs, etc.) instead of only body-origin markers.  The mesh prims are
        # static children of the animated body xforms, so no per-frame cost is incurred.
        vis.set_mesh_shapes(body_part_visual_descriptors)
    else:
        print(
            "[USD] pxr (OpenUSD) is not installed — Omniverse export disabled.\n"
            "      Install with:  pip install usd-core\n"
            "      The simulation will run without visualization."
        )
else:
    vis = mophi.create_opengl_visualizer_or_fatal(newton_model)
    _vis_available = True
    print("[Viewer] MoPhi OpenGL visualization window opened.\n")

if _vis_available:
    mophi.log_orientation_and_scale_reference(
        vis,
        axis_origin=REFERENCE_AXIS_ORIGIN,
        scale_bar_center=REFERENCE_SCALE_BAR_CENTER,
    )

# ─── XLB real LBM simulation setup ───────────────────────────────────────────
# Configures a real 3-D incompressible Navier-Stokes LBM simulation using XLB.
# Physical scenario (world coordinates):
#   • Inlet at the upstream y-face, blowing in the negative-y direction.
#   • Ground wall at the domain floor, using no-slip halfway bounce-back.
#   • Outlet at the downstream y-face, using extrapolation outflow.
#   • All other faces are left open (no explicit BC — acceptable for a low-Re demo).
#
# LBM physical mapping uses dx = world-domain length / lattice resolution and
# dt = _XLB_DT. Velocities map as u_lattice = u_SI * dt / dx, while kinematic
# viscosity maps as nu_lattice = nu_SI * dt / dx^2.
_XLB_SCALE = 4
_XLB_NX, _XLB_NY, _XLB_NZ = 32 * _XLB_SCALE, 32 * _XLB_SCALE, 16 * _XLB_SCALE
_XLB_DOMAIN_MIN = np.array([-2.0, -1.0, 0.0], dtype=np.float64)
_XLB_DOMAIN_MAX = np.array([2.0, 3.0, 2.0], dtype=np.float64)
_XLB_DT = 1.0e-3  # physical seconds represented by one lattice step
_XLB_PHYSICAL_INLET_SPEED = 0.625  # m/s
_XLB_PHYSICAL_KINEMATIC_VISCOSITY = 9.765625e-3  # m^2/s; deliberately viscous
_XLB_PHYSICAL_DENSITY = 1.0  # kg/m^3; keeps this intentionally approximate feedback gentle
# This closes the XLB -> Newton device path by treating each stationary-wall
# bounce-back as a momentum exchange. The mask moves but the BC does not use
# Newton wall velocity, so this is an ad-hoc demonstration, not a validated
# moving-boundary hydrodynamic force model.
# TODO: Remove this path when true moving-boundary treatment and force feedback are available.
ENABLE_XLB_AD_HOC_FORCE_FEEDBACK = True
# These limits are numerical tricks, not physics. Stationary halfway bounce-back
# produces artificial impulses when its voxel mask moves, so the two-way PoC
# deliberately weakens and caps that invalid reaction until a moving-wall BC exists.
# TODO: Remove these limits when true moving-boundary treatment and force feedback are available.
XLB_NONPHYSICAL_WRENCH_FEEDBACK_GAIN = 0.01
XLB_NONPHYSICAL_MAX_FORCE_TO_WEIGHT_RATIO = 0.1
XLB_NONPHYSICAL_MAX_TORQUE = 5.0  # N m
_XLB_SCALING = XLBPhysicalScaling.from_domain(
    _XLB_DOMAIN_MIN,
    _XLB_DOMAIN_MAX,
    (_XLB_NX, _XLB_NY, _XLB_NZ),
    _XLB_DT,
    _XLB_PHYSICAL_KINEMATIC_VISCOSITY,
    _XLB_PHYSICAL_DENSITY,
)
_XLB_CELL_SIZE = _XLB_SCALING.cell_size
_XLB_INLET_SPEED = float(_XLB_SCALING.velocity_to_lattice(_XLB_PHYSICAL_INLET_SPEED))
_XLB_OMEGA = _XLB_SCALING.omega
_XLB_WARMUP_SECONDS = 0.6
_XLB_WARMUP_STEPS = int(round(_XLB_WARMUP_SECONDS / _XLB_DT))
_XLB_VIS_INTERVAL = 5  # refresh streamline visualisation every N Newton frames
# This cap is intentionally based on robot weight only to bound the nonphysical
# mask-change impulse. It is not a fluid-derived force scale.
# TODO: Remove this cap when true moving-boundary treatment and force feedback are available.
_XLB_NONPHYSICAL_MAX_FORCE = (
    XLB_NONPHYSICAL_MAX_FORCE_TO_WEIGHT_RATIO * float(np.sum(newton_model.body_mass.numpy())) * 9.81
)

# State variables for the XLB solver (populated during setup below).
xlb_simulation = None  # passed to coupler.initialize() as a reference handle
_xlb_stepper = None
_xlb_runtime = None
_xlb_u_np = None  # velocity in (NX, NY, NZ, 3) layout; updated each vis interval

# ─── XLB robot-obstacle: constants, helpers, and per-frame state ──────────────
# The robot is represented as a prescribed axis-aligned bounding box (AABB) in
# the LBM grid, sized to enclose the ANYmal C body.  Box position follows
# joint_q[:3] (Newton base body world-space position) — no full body-transform
# query is required.
#
# The reusable NewtonBodyBoxBoundary reads Newton body_q and rewrites XLB's
# device masks directly before each group of smaller XLB physical substeps.
# The stepper is built once and never changed.
#
# missing_mask correctness (pull-streaming halfway bounce-back):
#   missing_mask[l, x, y, z] = True iff node (x,y,z) is a solid robot cell AND
#   its pull-from source (x - c[l,0], y - c[l,1], z - c[l,2]) lies outside the
#   solid box (i.e., is a fluid cell or out of domain).  Interior solid cells
#   where every pull source is also solid remain False and do not interact with
#   the fluid, which is physically correct.

# Fixed half-extents [m] of the prescribed robot box around the base body centre.
# Chosen to conservatively enclose the robot geometry with margin.
_XLB_ROBOT_HALF_EXT_X = 0.55  # ±0.55 m in x (walking direction)
_XLB_ROBOT_HALF_EXT_Y = 0.45  # ±0.45 m in y (lateral direction)
_XLB_ROBOT_BELOW_BASE = 0.62  # m below base centre (base is at ~0.62 m when standing)
_XLB_ROBOT_ABOVE_BASE = 0.28  # m above base centre (to top of torso)

# Per-frame state (populated by the XLB setup block below).
_xlb_robot_bc_id = None  # HalfwayBounceBackBC ID assigned to the robot obstacle
_xlb_vel_c_wp = None  # D3Q19 velocity stencil as a device-resident Warp array (q, 3) int32


print("[XLB] Setting up real LBM simulation ...")
# Guard against XLB-internal runtime/setup failures with a clear fatal message.
# This demo requires XLB; setup errors must terminate rather than silently degrade.
try:
    _xlb_scene = build_anymal_xlb_scene(
        (_XLB_NX, _XLB_NY, _XLB_NZ),
        _XLB_INLET_SPEED,
        _XLB_OMEGA,
        np.array([-_XLB_ROBOT_HALF_EXT_X, -_XLB_ROBOT_HALF_EXT_Y, 0.0]),
        np.array([_XLB_ROBOT_HALF_EXT_X, _XLB_ROBOT_HALF_EXT_Y, 0.62 + _XLB_ROBOT_ABOVE_BASE]),
        _XLB_DOMAIN_MIN,
        _XLB_DOMAIN_MAX,
    )
    _xlb_backend = _xlb_scene.backend
    _xlb_precision = _xlb_scene.precision
    _xlb_vel_set = _xlb_scene.velocity_set
    _xlb_grid = _xlb_scene.grid
    _xlb_inlet_idx = _xlb_scene.inlet_indices
    _xlb_outlet_idx = _xlb_scene.outlet_indices
    _xlb_wall_idx = _xlb_scene.wall_indices
    _robot_indices = _xlb_scene.robot_indices
    _xlb_inlet_bc = _xlb_scene.inlet_boundary
    _xlb_wall_bc = _xlb_scene.wall_boundary
    _xlb_outlet_bc = _xlb_scene.outlet_boundary
    _xlb_robot_bc = _xlb_scene.robot_boundary
    _robot_init_gc_min = _xlb_scene.robot_grid_min
    _robot_init_gc_max = _xlb_scene.robot_grid_max
    _xlb_stepper = _xlb_scene.stepper
    _xlb_runtime = _xlb_scene.runtime
    _xlb_bc_mask, _xlb_missing_mask = _xlb_runtime.bc_mask, _xlb_runtime.missing_mask

    # Store robot BC ID and D3Q19 velocity stencil for analytical mask updates.
    _xlb_robot_bc_id = _xlb_robot_bc.id
    _xlb_vel_c_np = _xlb_scene.velocity_stencil_numpy
    _xlb_vel_c_wp = _xlb_scene.velocity_stencil_warp

    # Warm-up: advance the LBM toward an initial near-steady state with the
    # robot obstacle already in place at its initial position.
    print(f"[XLB] Running {_XLB_WARMUP_STEPS} warm-up steps (ω = {_XLB_OMEGA:.4f}) ...")
    _xlb_scene.warm_up(_XLB_WARMUP_STEPS)

    # Extract macro state; XLB stores fields as (Q/D, NX, NY, NZ) — transpose
    # to (NX, NY, NZ, 3) for the streamline helper.
    _xlb_u_np = _xlb_scene.sample_velocity_numpy()

    xlb_simulation = _xlb_runtime
    print(
        f"[XLB] LBM simulation ready: {_XLB_NX}×{_XLB_NY}×{_XLB_NZ} grid "
        f"(dx={_XLB_CELL_SIZE:.5f} m, dt={_XLB_DT:.4g} s, "
        f"inlet={_XLB_PHYSICAL_INLET_SPEED:.3f} m/s, "
        f"nu={_XLB_PHYSICAL_KINEMATIC_VISCOSITY:.6g} m²/s, "
        f"robot BC id={_xlb_robot_bc_id}, "
        f"initial box {_robot_init_gc_min}–{_robot_init_gc_max}).\n"
    )
except Exception as exc:
    mophi.fatal(
        f"Failed to set up XLB simulation: {exc}\n"
        "Ensure compatible XLB dependencies are installed and the environment supports the selected backend."
    )


# ─── XLB flow-field visualisation helpers ─────────────────────────────────────

# Build initial streamline visualisation from the LBM macro state.
# If XLB is available use the real macro state; otherwise fall back to an
# analytically-constructed approximation (laminar channel flow in −y direction).
if _xlb_u_np is not None:
    _xlb_sl_u = _xlb_u_np
else:
    # Analytic fallback: smooth channel-like profile, flow in −y direction.
    _xlb_sl_u = np.zeros((_XLB_NX, _XLB_NY, _XLB_NZ, 3), dtype=np.float32)
    _xlb_sl_z = np.linspace(0.0, 2.0, _XLB_NZ)
    _xlb_sl_profile = 4.0 * (_xlb_sl_z / 2.0) * (1.0 - _xlb_sl_z / 2.0)
    _xlb_sl_u[:, :, :, 1] = -0.8 * _xlb_sl_profile[np.newaxis, np.newaxis, :]  # −y

# Seed streamlines on a regular y-plane grid just inside the inlet face so
# flow features remain easy to inspect during visualization.
_xlb_seed_pts = mophi.xlb_make_y_plane_seeds(_XLB_DOMAIN_MIN, _XLB_DOMAIN_MAX)
_xlb_streamline_pts, _xlb_streamline_spd, _xlb_streamline_dirs = mophi.xlb_build_streamlines(
    _xlb_sl_u, _XLB_DOMAIN_MIN, _XLB_DOMAIN_MAX, _xlb_seed_pts
)
print(f"[XLB] Generated {len(_xlb_streamline_pts)} streamline sample point(s) for flow visualisation.\n")

if _vis_available:
    _xlb_streamline_pos_wp, _xlb_streamline_radii_wp, _xlb_streamline_colors_wp = mophi.xlb_make_streamline_warp_arrays(
        _xlb_streamline_pts, _xlb_streamline_spd, _xlb_streamline_dirs
    )
    print(f"[Viewer] XLB flow streamlines registered ({len(_xlb_streamline_pts)} point(s)).\n")

# ─── Build the DEME solver ────────────────────────────────────────────────
# Representative radius types [m] for a simple polydisperse particle set.
_DEM_RADIUS_TYPES = [0.030, 0.045, 0.060, 0.040]
_DEM_SAMPLE_CENTER = [0.0, 2.75, 0.25]
_DEM_SAMPLE_HALF_DIMENSIONS = [1.5, 1.75, 0.15]
_DEM_RANDOM_SEED = 42
_DEME_MATERIAL = {"E": 1e5, "nu": 0.3, "mu": 0.3, "CoR": 0.2}
_DEME_GRAVITY = [0.0, 0.0, -9.81]
_DEME_AVERAGE_CONTACT_LIMIT = 500
_FIXED_FAM = 10

# Use DEME's Poisson-disk sampler
# The Poisson-disk sampler enforces a minimum centre-to-centre spacing based on
# particle size so initial spheres do not overlap.
_deme_population = prepare_anymal_deme_population(
    DEME,
    _DEM_RADIUS_TYPES,
    _DEM_SAMPLE_CENTER,
    _DEM_SAMPLE_HALF_DIMENSIONS,
    _DEM_RANDOM_SEED,
)
_sampled_positions = _deme_population.sampled_positions
_NUM_DEM_SPHERES = len(_sampled_positions)
_dem_sphere_positions_np = _deme_population.positions_numpy
_radius_indices = _deme_population.radius_indices
_dem_sphere_radii_np = _deme_population.radii_numpy

# Velocity in -y direction [m/s] — opposite to robot's forward (+y) direction,
# so the spheres move towards the robot's face.
_DEM_SPHERE_INIT_VELOCITY_Y = [0.0, -1.2, 0.0]  # m/s
_DEM_SPHERE_COLOR = [0.8, 0.4, 0.1]  # orange — DEM particle colour
# Explicit Newton simulation step for this demo.
SIM_DT = 1.0 / 200.0

# ── DEME → Newton force feedback ─────────────────────────────────────────
# Toggle whether DEME contact feedback affects Newton robot motion.
# Off reproduces legacy behavior (DEME particles do not push back on the robot).
ENABLE_DEME_FORCE_FEEDBACK = True
# Scale applied to DEME-derived contact wrench before injecting into Newton.
# Values >1 amplify interaction so stepping on rolling particles can destabilize gait.
DEME_FORCE_FEEDBACK_SCALE = 10.0

if _vis_available:
    # Radii and colours are constant throughout the simulation; allocate once.
    # The position array (_dem_sphere_pos_wp) is rebuilt from _dem_sphere_positions_np
    # every frame inside the simulation loop after positions are updated.
    _dem_sphere_radii_wp = wp.array(_dem_sphere_radii_np, dtype=wp.float32)
    _dem_sphere_colors_wp = wp.array(
        np.tile(_DEM_SPHERE_COLOR, (_NUM_DEM_SPHERES, 1)).astype(np.float32),
        dtype=wp.vec3,
    )
    print(f"[Viewer] {_NUM_DEM_SPHERES} DEM sphere(s) registered for visualisation.\n")

print("[DEME] Creating deme.DEMSolver ...")
_deme_scene = build_anymal_deme_scene(
    DEME,
    foot_tip_sphere_radii,
    _DEM_RADIUS_TYPES,
    _deme_population,
    _DEME_MATERIAL,
    _DEME_GRAVITY,
    _DEME_AVERAGE_CONTACT_LIMIT,
    _FIXED_FAM,
    _DEM_SPHERE_INIT_VELOCITY_Y,
    SIM_DT,
)
deme_solver = _deme_scene.solver
wall_mat = _deme_scene.material
shank_trackers = _deme_scene.shank_trackers
particle_templates = _deme_scene.particle_templates
used_types = _deme_scene.used_particle_templates
particles = _deme_scene.particles
particles_tracker = _deme_scene.particles_tracker

# ─── Initialize the coupler ───────────────────────────────────────────────
coupler = mophi.NewtonXLBDEMCoupler()
coupler.set_verbosity(mophi.VERBOSITY_INFO)

print("[Coupler] Initializing NewtonXLBDEMCoupler ...")
coupler.initialize(
    newton_model=newton_model,
    newton_solver=newton_solver,
    xlb_simulation=xlb_simulation,
    deme_solver=deme_solver,
    sim_dt=SIM_DT,
)
print("[Coupler] NewtonXLBDEMCoupler initialized.\n")

# ── Bind XLB device-resident mask arrays to the coupler ──────────────────────
# After initialization, hand the coupler the device handles for XLB's bc_mask
# and missing_mask. MoPhi's Newton-body boundary helper writes them in-place
# from Newton's device transforms, with no CPU round-trip.
if _xlb_stepper is not None:
    coupler.set_xlb_masks(_xlb_bc_mask, _xlb_missing_mask)
    print("[Coupler] XLB bc_mask and missing_mask device handles bound.\n")

    _newton_xlb_boundary = NewtonBodyBoxBoundary()
    _newton_xlb_boundary.initialize(
        _xlb_bc_mask,
        _xlb_missing_mask,
        _xlb_vel_c_wp,
        body_index=0,
        domain_min=_XLB_DOMAIN_MIN,
        domain_max=_XLB_DOMAIN_MAX,
        lower_extents=(_XLB_ROBOT_HALF_EXT_X, _XLB_ROBOT_HALF_EXT_Y, _XLB_ROBOT_BELOW_BASE),
        upper_extents=(_XLB_ROBOT_HALF_EXT_X, _XLB_ROBOT_HALF_EXT_Y, _XLB_ROBOT_ABOVE_BASE),
        boundary_id=_xlb_robot_bc_id,
        initial_grid_min=_robot_init_gc_min,
        initial_grid_max=_robot_init_gc_max,
    )

# ─── Load the ANYmal C walking policy ────────────────────────────────────
print("[Policy] Loading ANYmal C walking policy ...")
policy = torch.jit.load(policy_path, map_location=torch_device)

# Initial joint positions — zero-copy view from the Warp array (GPU torch tensor),
# then clone to snapshot the initial state before the simulation advances.
# joint_q[0:7] = free-joint pose (position + quaternion); joint_q[7:] = revolute DOFs.
joint_pos_initial = wp.to_torch(coupler.newton_state_0.joint_q)[7:].to(dtype=torch.float32).unsqueeze(0).clone()

act = torch.zeros(1, 12, device=torch_device, dtype=torch.float32)
lab_to_mujoco_indices = torch.tensor(lab_to_mujoco, device=torch_device)
mujoco_to_lab_indices = torch.tensor(mujoco_to_lab, device=torch_device)
gravity_vec = torch.tensor([[0.0, 0.0, -1.0]], device=torch_device, dtype=torch.float32)
command = torch.zeros((1, 3), device=torch_device, dtype=torch.float32)
command[0, 0] = 1.0  # walk forward (x-direction)

print("[Policy] ANYmal C walking policy loaded.\n")

# ─── GPU-only runtime exchange setup ───────────────────────────────────────
_deme_owner_ids = [int(tracker.GetOwnerID()) for tracker in shank_trackers]
_deme_contact_exchange = NewtonDEMEContactAccelerationCoupler()
_deme_contact_exchange.initialize(
    newton_model,
    deme_solver,
    NewtonDEMEOwnerMap(foot_tip_body_indices, _deme_owner_ids),
    [descriptor["local_offset"] for descriptor in foot_tip_descriptors],
    coupler.newton_state_0.body_q.device,
    force_scale=DEME_FORCE_FEEDBACK_SCALE,
)
_xlb_deme_particles = XLBDEMEParticleExchange()
_xlb_deme_particles.initialize(
    deme_solver,
    int(particles_tracker.GetOwnerID()),
    _NUM_DEM_SPHERES,
    coupler.newton_state_0.body_q.device,
)
_xlb_wrench_reducer = NewtonXLBHalfwayBounceBackWrench()
_xlb_wrench_reducer.initialize(
    body_index=0,
    boundary_id=_xlb_robot_bc_id,
    domain_min=_XLB_DOMAIN_MIN,
    cell_size=_XLB_CELL_SIZE,
    physical_density=_XLB_PHYSICAL_DENSITY,
    xlb_dt=_XLB_DT,
    lattice_velocities=_xlb_vel_c_wp,
    opposite_indices=np.asarray(_xlb_vel_set.opp_indices, dtype=np.int32),
    device=coupler.newton_state_0.body_q.device,
)
_xlb_wrench_exchange = NewtonXLBWrenchExchange()
_xlb_wrench_exchange.initialize(newton_model, [0], coupler.newton_state_0.body_q.device)
_combined_body_forces = wp.zeros(
    int(newton_model.body_count), dtype=wp.spatial_vector, device=coupler.newton_state_0.body_q.device
)
coupler.set_newton_body_forces(_combined_body_forces)
if ENABLE_XLB_AD_HOC_FORCE_FEEDBACK:
    print(
        "[Coupling] Ad-hoc XLB halfway-bounce-back wrench feedback enabled. "
        "This demonstrates two-way GPU exchange but is not a validated moving-wall force model.\n"
        f"[Coupling] NONPHYSICAL numerical moderation: gain={XLB_NONPHYSICAL_WRENCH_FEEDBACK_GAIN:g}, "
        f"force cap={_XLB_NONPHYSICAL_MAX_FORCE:.3f} N, torque cap={XLB_NONPHYSICAL_MAX_TORQUE:.3f} N m.\n"
    )
if ENABLE_DEME_FORCE_FEEDBACK:
    print(f"[Coupling] DEME force feedback enabled (scale={DEME_FORCE_FEEDBACK_SCALE:.1f}).\n")
else:
    print("[Coupling] DEME force feedback disabled.\n")

# ─── Co-simulation loop ───────────────────────────────────────────────────
# The ANYmal C walking policy runs once per frame.
# Each frame advances SIM_SUBSTEPS × SIM_DT seconds of physics.
SIM_SUBSTEPS = 4  # physics substeps per policy frame
FRAME_DT = SIM_DT * SIM_SUBSTEPS  # policy control rate
NUM_FRAMES = 250  # ≈ 5 s at 50 Hz (or until the viewer is closed)
XLB_SUBSTEPS_PER_NEWTON_STEP = int(round(SIM_DT / _XLB_DT))
if not np.isclose(XLB_SUBSTEPS_PER_NEWTON_STEP * _XLB_DT, SIM_DT, rtol=0.0, atol=1.0e-12):
    raise ValueError("Newton dt must be an integer multiple of the physical XLB dt.")
sim_time = 0.0

# ─── Movie recording settings ────────────────────────────────────────────
# Set SAVE_MOVIE = True to record the rendered simulation frames to a video file.
# Requires: pip install imageio imageio-ffmpeg
# When SAVE_MOVIE=True and imageio is missing, this demo exits with an
# actionable install message instead of silently disabling recording.
SAVE_MOVIE = True
MOVIE_OUTPUT_PATH = OUTPUT_DIRECTORY / "demo_anymal_robot_multiphysics.mp4"
MOVIE_FPS = 50  # frames per second for the output video

_movie_writer = None
if SAVE_MOVIE and _vis_available and not USE_OMNIVERSE_VISUALIZATION:
    if find_spec("imageio") is None:
        mophi.fatal("imageio is required when SAVE_MOVIE=True.\n" "Install with:  pip install imageio imageio-ffmpeg")
    import imageio

    _movie_writer = imageio.get_writer(MOVIE_OUTPUT_PATH, fps=MOVIE_FPS)
    print(f"[Movie] Recording simulation to '{MOVIE_OUTPUT_PATH}' at {MOVIE_FPS} fps.\n")

print(
    f"Running up to {NUM_FRAMES} policy frame(s) "
    f"(frame_dt={FRAME_DT * 1000:.1f} ms, {SIM_SUBSTEPS} Newton steps × {SIM_DT * 1000:.1f} ms, "
    f"{XLB_SUBSTEPS_PER_NEWTON_STEP} XLB steps per Newton step × {_XLB_DT * 1000:.1f} ms) ...\n"
)
# print(f"{'Frame':>5}  {'Base X [m]':>12}  {'Base Y [m]':>12}  {'Base Z [m]':>12}  Representation")
# print("-" * 72)

# Pre-allocate the zero buffer used to prepend free-joint DOFs each frame.
free_joint_zeros = torch.zeros(6, device=torch_device, dtype=torch.float32)

# Set the initial camera position once before the simulation loop.
# Newton's ViewerGL handles mouse and keyboard camera controls (orbit, pan, zoom)
# natively; calling set_camera() only once here lets the user freely adjust the
# view angle during the simulation without the camera being reset every frame.
if _vis_available and not USE_OMNIVERSE_VISUALIZATION:
    _init_base_pos = wp.to_torch(coupler.newton_state_0.joint_q)[:3].cpu().numpy()
    vis.set_camera(
        pos=wp.vec3(_init_base_pos[0], _init_base_pos[1] - 6.0, _init_base_pos[2] + 2.0),
        # pos=wp.vec3(_init_base_pos[0] + 6., _init_base_pos[1], _init_base_pos[2] + 2.0),
        pitch=-10.0,
        yaw=90.0,
        # yaw=180.0,
    )

for frame in range(NUM_FRAMES):
    # Stop early if the OpenGL viewer window has been closed by the user.
    # For OmniverseVisualizer, vis.is_running() always returns True.
    if _vis_available and not vis.is_running():
        print(f"\n[Viewer] Window closed by user after frame {frame}.")
        break

    # ── Policy inference: compute observation and get joint targets ──────────
    # Zero-copy conversion of the Warp GPU arrays to PyTorch tensors.  This
    # avoids the former torch.tensor(state.joint_q[...], device=...) pattern
    # which triggered a synchronous GPU→CPU copy via Warp's __getitem__.
    joint_q_t = wp.to_torch(coupler.newton_state_0.joint_q)
    joint_qd_t = wp.to_torch(coupler.newton_state_0.joint_qd)
    obs = compute_observation(
        act,
        joint_q_t,
        joint_qd_t,
        joint_pos_initial,
        lab_to_mujoco_indices,
        gravity_vec,
        command,
    )
    act, target_with_zeros = evaluate_joint_targets(
        policy, obs, joint_pos_initial, mujoco_to_lab_indices, free_joint_zeros, POLICY_ACTION_SCALE
    )
    target_wp = wp.from_torch(target_with_zeros, dtype=wp.float32, requires_grad=False)
    wp.copy(coupler.newton_control.joint_target_pos, target_wp)

    _deme_contact_exchange.set_deme_owner_pose_from_newton(coupler.newton_state_0)

    # ── Physics substeps ──────────────────────────────────────────────────────
    for _ in range(SIM_SUBSTEPS):
        _combined_body_forces.zero_()
        if ENABLE_DEME_FORCE_FEEDBACK:
            wp.copy(_combined_body_forces, _deme_contact_exchange.newton_body_forces)
        if ENABLE_XLB_AD_HOC_FORCE_FEEDBACK:
            # This gain and clamp are explicitly a nonphysical numerical trick:
            # they keep the moving-mask PoC runnable, but do not correct its
            # stationary-wall momentum exchange or make the wrench scientific.
            # TODO: Remove this treatment when true moving-boundary treatment and force feedback are available.
            _xlb_wrench_exchange.write_numerically_limited_xlb_wrenches_to_newton(
                _xlb_wrench_reducer.forces,
                _xlb_wrench_reducer.torques,
                _combined_body_forces,
                feedback_gain=XLB_NONPHYSICAL_WRENCH_FEEDBACK_GAIN,
                maximum_force=_XLB_NONPHYSICAL_MAX_FORCE,
                maximum_torque=XLB_NONPHYSICAL_MAX_TORQUE,
                clear_destination=False,
            )

        coupler.step_newton()
        _deme_contact_exchange.step_deme()

        if ENABLE_DEME_FORCE_FEEDBACK:
            _deme_contact_exchange.write_deme_contact_accelerations_to_newton()

        # XLB has its own smaller physical time step. Hold the newly advanced
        # Newton body transform fixed while the fluid catches up to this time.
        _newton_xlb_boundary.update_from_newton(coupler.newton_state_0.body_q)
        for _ in range(XLB_SUBSTEPS_PER_NEWTON_STEP):
            coupler.step_xlb()
        if ENABLE_XLB_AD_HOC_FORCE_FEEDBACK:
            _xlb_wrench_reducer.reduce(_xlb_runtime.f0, _xlb_bc_mask, _xlb_missing_mask, coupler.newton_state_0.body_q)

    sim_time += FRAME_DT

    # Visualization sampling is deliberately separate from physical XLB cadence.
    if _vis_available and frame % _XLB_VIS_INTERVAL == 0:
        _xlb_u_np = _xlb_scene.sample_velocity_numpy()
        _xlb_streamline_pts, _xlb_streamline_spd, _xlb_streamline_dirs = mophi.xlb_build_streamlines(
            _xlb_u_np, _XLB_DOMAIN_MIN, _XLB_DOMAIN_MAX, _xlb_seed_pts
        )
        _xlb_streamline_pos_wp, _xlb_streamline_radii_wp, _xlb_streamline_colors_wp = (
            mophi.xlb_make_streamline_warp_arrays(_xlb_streamline_pts, _xlb_streamline_spd, _xlb_streamline_dirs)
        )

    # ── Visualization ─────────────────────────────────────────────────────────
    if _vis_available:
        _dem_sphere_pos_wp, _ = _xlb_deme_particles.read_deme_particle_state()

        vis.begin_frame(sim_time)
        vis.log_state(coupler.newton_state_0)
        # Render live DEME particle positions.
        vis.log_points(
            "dem_particles",
            _dem_sphere_pos_wp,
            radii=_dem_sphere_radii_wp,
            colors=_dem_sphere_colors_wp,
        )
        # Render XLB flow field as arrow glyphs — each arrow has a large head sphere
        # at the current streamline position and tapering shaft spheres trailing behind
        # along the −velocity direction, making the flow direction immediately legible.
        # Glyphs are sampled from the live LBM macro state (updated every
        # _XLB_VIS_INTERVAL frames); the cool-blue palette contrasts with the orange
        # DEM particles and does not obstruct the robot geometry or ground plane.
        vis.log_points(
            "xlb_streamlines",
            _xlb_streamline_pos_wp,
            radii=_xlb_streamline_radii_wp,
            colors=_xlb_streamline_colors_wp,
        )
        vis.end_frame()

        # Movie recording is only supported with the OpenGL backend (get_frame()
        # returns None for OmniverseVisualizer).
        if _movie_writer is not None:
            _movie_writer.append_data(vis.get_frame().numpy())

    # ── Console status (periodic) ─────────────────────────────────────────────
    # if (frame + 1) % 25 == 0 or frame == 0:
    #     if all_transforms:
    #         base = all_transforms[0]
    #         px, py, pz = base[0], base[1], base[2]
    #         qx, qy, qz, qw = base[3], base[4], base[5], base[6]
    #         print(
    #             f"{frame + 1:>5}  {px:>12.4f}  {py:>12.4f}  {pz:>12.4f}  "
    #             f"[{newton_model.body_count} bodies, "
    #             f"base quat=({qx:.3f},{qy:.3f},{qz:.3f},{qw:.3f})]"
    #         )
    #     else:
    #         print(f"{frame + 1:>5}  (Newton not available — no transforms)")

print()

# ─── Finalize ─────────────────────────────────────────────────────────────
if _movie_writer is not None:
    _movie_writer.close()
    print(f"[Movie] Saved simulation recording to '{MOVIE_OUTPUT_PATH}'.\n")

if vis is not None:
    vis.close()

print("[Coupler] Finalizing NewtonXLBDEMCoupler ...")
_deme_contact_exchange.finalize()
_xlb_deme_particles.finalize()
_xlb_wrench_reducer.finalize()
_xlb_wrench_exchange.finalize()
if _xlb_stepper is not None:
    _newton_xlb_boundary.finalize()
    _xlb_scene.finalize()
coupler.finalize()
print("[Coupler] NewtonXLBDEMCoupler finalized.\n")

print("Demo completed successfully.")
print(f"  Newton version : {newton.__version__}")
print(f"  Warp   version : {wp.__version__}")
print(f"  XLB    version : {xlb.__version__}")
print(
    "\nRuntime exchange summary:\n"
    "  Newton body poses, XLB boundary masks and ad-hoc reaction wrench, DEME contact feedback, "
    "and DEME particle state were exchanged through shared-device arrays without host staging."
)
