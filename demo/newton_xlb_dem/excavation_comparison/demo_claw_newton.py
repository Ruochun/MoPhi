"""demo/newton_xlb_dem/excavation_comparison/demo_claw_newton.py

Newton UR10 robot arm + DEME granular terrain co-simulation demo with plowing.

This demo simulates a UR10 excavator arm plowing into a granular terrain pile
managed by DEME.  It reproduces Newton's ``example_robot_ur10`` example inside
MoPhi's ``NewtonXLBDEMCoupler`` framework.

The robot is mounted on a cylindrical pedestal above the ground plane.  Six
revolute joints are position-controlled.  The arm starts in a ready-to-plow
configuration above the pile and its joints are gradually driven toward a
final plowing configuration that pushes the end-effector downward.

An excavator plow mesh (``data/mesh/excavator.obj``) serves dual roles:

  1. Visualization in Newton — rigidly attached to the UR10 end-effector link
     (``ee_link``).  The OBJ file uses centimetre units; it is scaled by 0.01
     so it is correctly sized in metres.  A 180° local rotation about X makes
     the bowl face downward.

  2. Contact proxy in DEME — the same OBJ is loaded into DEME as a kinematic
     mesh body.  Its position and orientation are updated every Newton substep
     from the arm's computed end-effector pose, so DEME always sees the plow
     where the arm actually is.

Simulation phases
-----------------
  Phase 0 — Newton warm-up: arm drives to the ready-to-plow joint configuration.
  Phase 1 — DEME settling: granular terrain settles under gravity for
    ``DEME_SETTLE_TIME`` seconds.  The plow mesh is present in DEME but
    contact with terrain particles is disabled.
  Phase 2 — Active plowing: contact is enabled, Newton joint targets ramp from
    the ready-to-plow pose toward ``PLOW_TARGET_Q`` over ``PLOW_DURATION``
    seconds, driving the plow downward into the pile.  The DEME plow position
    is synchronised with Newton's ee_link every substep.

Future phases will add:
  • XLB (lattice-Boltzmann) fluid flow around the moving arm.

The demo closely mirrors Newton's ``example_robot_ur10`` setup (single arm,
SolverMuJoCo, position control) while wrapping all physics inside the MoPhi
``NewtonXLBDEMCoupler``.

Prerequisites
-------------
  • Build MoPhi with -DMOPHI_BUILD_NEWTON_XLB_DEM=ON (compiles
    NewtonXLBDEMCoupler and builds the mophi_core Python extension module).
  • pip install --upgrade newton warp-lang mujoco==3.6.0
  • pip install deme3              (DEME discrete-element solver — required)

Running
-------
From the MoPhi build tree (after ``cmake --build``):

    python demo/newton_xlb_dem/excavation_comparison/demo_claw_newton.py

Or from the repository root after installing the mophi package:

    python -m demo.newton_xlb_dem.excavation_comparison.demo_claw_newton
"""

import os
import time
from pathlib import Path

import numpy as np

# Required: build MoPhi with -DMOPHI_BUILD_NEWTON_XLB_DEM=ON and ensure
# python/ is on PYTHONPATH (the build tree puts the .so next to the package).
import mophi
from mophi.utils.package_provider import load_package_provider

DEME = load_package_provider("deme")

if not hasattr(mophi, "NewtonXLBDEMCoupler"):
    mophi.fatal("mophi.NewtonXLBDEMCoupler is not available.\n" "Re-build MoPhi with -DMOPHI_BUILD_NEWTON_XLB_DEM=ON.")

# Required: pip install --upgrade newton warp-lang mujoco==3.6.0
import newton
import warp as wp
import mujoco
from newton import JointTargetMode
from newton.selection import ArticulationView

mophi.check_newton_warp_mujoco_versions(
    mophi.REQUIRED_NEWTON_VERSION,
    mophi.REQUIRED_WARP_VERSION,
    mophi.REQUIRED_MUJOCO_VERSION,
)


def _sync_deme_plow_pose_from_newton(plow_tracker, body_q_np: np.ndarray, ee_link_body_idx: int) -> list[float]:
    """Sync DEME plow tracker pose from Newton ee_link transform and return ee position."""
    _ee_pos = body_q_np[ee_link_body_idx, 0:3].tolist()
    _ee_quat_np = body_q_np[ee_link_body_idx, 3:7]
    _plow_world_quat = mophi.quat_mul(_ee_quat_np, _PLOW_LOCAL_ROT_NP).tolist()
    plow_tracker.SetPos(_ee_pos)
    plow_tracker.SetOriQ(_plow_world_quat)
    return _ee_pos


def _rotate_vector_by_quat_xyzw(quat_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate a 3D vector by a quaternion in [x, y, z, w] convention."""
    q_xyz = quat_xyzw[:3]
    q_w = quat_xyzw[3]
    t = 2.0 * np.cross(q_xyz, vec)
    return vec + q_w * t + np.cross(q_xyz, t)


# ══════════════════════════════════════════════════════════════════════════════
# Simulation configuration
# All user-tunable constants are defined here.  Change values in this section;
# do not scatter magic numbers through the setup code below.
# ══════════════════════════════════════════════════════════════════════════════

# ── World ─────────────────────────────────────────────────────────────────
WORLD_COUNT = 1  # number of UR10 arms in the simulation

# ── Timing ────────────────────────────────────────────────────────────────
# Physics step sizes are explicit; rendering cadence is configured
# independently; collaboration loop counts are derived from these.
RENDER_FPS = 50
NEWTON_DT = 1.0e-3
DEME_DT = 1.0e-4

# Debug mode: render one frame then keep the viewer alive for visual
# inspection until the user closes it.
PAUSE_AFTER_FIRST_FRAME = False
NUM_FRAMES = 1 if PAUSE_AFTER_FIRST_FRAME else 250  # default run ≈ 5 s at 50 Hz

# ── Scene geometry ────────────────────────────────────────────────────────
# Height of the cylindrical pedestal that mounts the arm above the ground.
PEDESTAL_HEIGHT = 1.2  # [m]
# The excavator OBJ is authored in centimetres; this factor converts to metres.
OBJ_CM_TO_M = 0.01

# ── Arm joint targets — change both when re-tuning the arm trajectory ─────
# READY_TO_PLOW_Q is the initial pose the arm holds during Newton warm-up.
# PLOW_TARGET_Q is the end pose reached at PLOW_DURATION; both are in the
# UR10 joint order: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3.
# These two arrays are closely coupled: always update them together.
READY_TO_PLOW_Q = np.array(
    [0.5 * np.pi, -1.35, 1.20, -0.60, -1.0 * np.pi, -1.0],
    dtype=np.float32,
)
PLOW_TARGET_Q = np.array(
    # A downward cut: shoulder_lift increases (less negative) and elbow decreases
    # relative to READY_TO_PLOW_Q.  Keep wrist_2 fixed to preserve tool orientation.
    [0.5 * np.pi, -0.45, 0.65, -0.45, -1.0 * np.pi, -1.0],
    dtype=np.float32,
)

# ── Phase control ─────────────────────────────────────────────────────────
# Number of Newton-only warm-up render-frames so the arm settles to
# READY_TO_PLOW_Q before DEME settling begins.
NEWTON_WARMUP_FRAMES = 50  # ≈ 1 s at RENDER_FPS

# Time for the granular terrain to settle under gravity before plow contact
# is activated.  Increase if particles are still visibly moving at plow activation.
DEME_SETTLE_TIME = 1.2  # [s]

# Wall-clock (simulation) time over which to linearly interpolate from
# READY_TO_PLOW_Q to PLOW_TARGET_Q.  Longer → slower, gentler plowing.
PLOW_DURATION = 1.6  # [s]

# Show DEME settling-phase rendering by default.  Set to False to run settling
# as a single non-rendered blocking call (faster startup / no settling frames).
RENDER_SETTLING_PHASE = True

# Late-stage scoop motion for wrist_3 (index 5) near the end of the run.
# Fraction of total simulation duration; start is clamped to ≥ PLOW_DURATION.
WRIST_SCOOP_START_FRACTION = 0.75
WRIST_SCOOP_INWARD_DELTA = -1.8  # inward wrist_3 delta added to PLOW_TARGET_Q[wrist_3] [rad]
_WRIST_3_DOF_INDEX = 5  # UR10 joint order: wrist_3 is the last (0-based index 5)

# ── DEME → Newton force feedback ─────────────────────────────────────────
# When True, the net contact force reported by DEME for the plow mesh is fed
# back into Newton as an external body force on the ee_link each substep.
# Newton's position controller must then generate additional joint torques to
# maintain tracking under the granular load, so the arm's dynamics realistically
# reflect the resistance of the terrain.  The initial and final joint targets are
# unchanged; only the effort required to reach them is affected.
ENABLE_DEME_FORCE_FEEDBACK = True

# ── Visualization & output ────────────────────────────────────────────────
# Set USE_OMNIVERSE_VISUALIZATION = True to write each frame to a USD file
# (requires: pip install usd-core).  False uses Newton's real-time OpenGL window.
# The simulation loop body is identical for both backends.
USE_OMNIVERSE_VISUALIZATION = False
REFERENCE_AXIS_ORIGIN = (0.8, -0.9, 0.02)
REFERENCE_SCALE_BAR_CENTER = (0.0, -0.9, 0.06)

# Set SAVE_MOVIE = True to record the rendered simulation to a video file
# (requires: pip install imageio imageio-ffmpeg).
SAVE_MOVIE = True
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "output" / Path(__file__).stem
MOVIE_OUTPUT_PATH = OUTPUT_DIRECTORY / "demo_claw_newton.mp4"
MOVIE_FPS = RENDER_FPS
OMNIVERSE_OUTPUT_PATH = OUTPUT_DIRECTORY / "demo_claw_newton.usdc"

# ── Derived timing constants ──────────────────────────────────────────────
FRAME_DT = 1.0 / RENDER_FPS

SIM_SUBSTEPS = int(round(FRAME_DT / NEWTON_DT))
if not np.isclose(SIM_SUBSTEPS * NEWTON_DT, FRAME_DT, rtol=0.0, atol=1.0e-12):
    raise ValueError(
        "FRAME_DT must be an integer multiple of NEWTON_DT. "
        f"Got FRAME_DT={FRAME_DT:.12g}, NEWTON_DT={NEWTON_DT:.12g}."
    )

DEME_SUBSTEPS = int(round(NEWTON_DT / DEME_DT))
if not np.isclose(DEME_SUBSTEPS * DEME_DT, NEWTON_DT, rtol=0.0, atol=1.0e-12):
    raise ValueError(
        "NEWTON_DT must be an integer multiple of DEME_DT. " f"Got NEWTON_DT={NEWTON_DT:.12g}, DEME_DT={DEME_DT:.12g}."
    )

SIM_FPS = RENDER_FPS
SIM_DT = NEWTON_DT

# Small tolerances for time-integration bounds to avoid an extra loop iteration
# from floating-point roundoff.
_SETTLING_TIME_EPSILON = 1.0e-12
_MIN_DURATION_EPSILON = 1.0e-12

# ─── DEME granular terrain constants ──────────────────────────────────────
# Physics parameters follow DEMdemo_Plow.cpp (projectchrono/DEM-Engine).
# Ellipsoid template with anisotropic semi-axes; scaling sets the physical particle size.
_DEM_PI = 3.1415927
_DEM_TERRAIN_SCALING = 0.03  # metres per template unit

# Unscaled ellipsoid template mass and MOI.
# ellipsoid_2_1_1.csv encodes an elongated ellipsoid in template coordinates.
# The MOI terms follow the standard solid-ellipsoid formula using the two
# orthogonal semi-axes for each principal axis.
# These values replicate DEMdemo_Plow.cpp exactly.
_DEM_TEMPLATE_MASS = 2600.0 * (4.0 / 3.0 * _DEM_PI * 2.0 * 1.0 * 1.0)
_DEM_TEMPLATE_MOI = [
    1.0 / 5.0 * _DEM_TEMPLATE_MASS * (1.0**2 + 2.0**2),  # I_x = (m/5)(sy²+sz²) = (m/5)(1+4)
    1.0 / 5.0 * _DEM_TEMPLATE_MASS * (1.0**2 + 2.0**2),  # I_y = (m/5)(sx²+sz²) = (m/5)(1+4)
    1.0 / 5.0 * _DEM_TEMPLATE_MASS * (1.0**2 + 1.0**2),  # I_z = (m/5)(sx²+sy²) = (m/5)(1+1)
]

# Terrain pile geometry (world-space, z-up).
# Configured to keep a compact pile beneath the arm workspace.
_DEM_WORLD_HS = 1.1  # domain half-size in x and y [m]
_DEM_BOWL_BOT = 0.0  # domain bottom
_DEM_FILL_HW = 1.0  # pile fill half-width in x and y [m]
_DEM_FILL_BOT = _DEM_BOWL_BOT + 3.0 * _DEM_TERRAIN_SCALING  # first layer bottom
_DEM_FILL_H = 5.0  # total pile height [m]
_DEM_LAYER_STEP = 4.5 * _DEM_TERRAIN_SCALING  # vertical spacing between fill layers

# DEME family IDs for terrain particles and the excavator plow mesh.
# Terrain particles are placed in family 0 (DEME default).  Plow families
# start at 10 to leave families 1–9 free for future use (e.g., walls, probes).
# The plow starts in the sleep family (fixed, contact disabled) during the
# terrain settling phase, then switches to the active family once contact is
# enabled for plowing.
_DEME_TERRAIN_FAMILY = 0  # default family for terrain particles
_DEME_PLOW_SLEEP_FAMILY = 10  # fixed, contact with terrain disabled during settling
_DEME_PLOW_ACTIVE_FAMILY = 11  # fixed, contact with terrain enabled during plowing

# Plow local rotation as numpy [x, y, z, w] — mirrors PLOW_LOCAL_ROT
# (defined below via wp.quat_from_axis_angle about +X by 180°).
# Used to compose the ee_link world quaternion with the plow's local frame when
# updating the DEME mesh orientation from Newton's arm state each step.
_PLOW_LOCAL_ROT_NP = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

# ─── Initialize Warp ───────────────────────────────────────────────────────
print("=== MoPhi UR10 Claw Arm co-simulation demo ===\n")
wp.init()
device = wp.get_device()

# ─── Paths ────────────────────────────────────────────────────────────────
# Resolve the excavator OBJ mesh relative to the repository root so the demo
# can be run from any working directory.
_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_DEMO_DIR)))
EXCAVATOR_OBJ_PATH = os.path.join(_REPO_ROOT, "data", "mesh", "excavator.obj")

# ─── Load the UR10 robot model ─────────────────────────────────────────────
# newton.utils.download_asset("universal_robots_ur10") downloads the UR10 USD
# from the Newton Assets repository.  The arm is mounted on a cylindrical
# pedestal so its base link clears the ground plane.
print("[Newton] Downloading UR10 robot assets ...")
asset_path = mophi.download_newton_asset("universal_robots_ur10", ["usd/ur10_instanceable.usda"])
asset_file = str(asset_path / "usd" / "ur10_instanceable.usda")
print(f"[Newton] Assets ready: {asset_path}\n")

print("[Newton] Building UR10 model ...")

# Build a single-arm sub-builder, then replicate it WORLD_COUNT times.
# This matches Newton's example pattern and lets ArticulationView find the
# articulation by the "*ur10*" glob pattern.
ur10_sub = newton.ModelBuilder()
newton.solvers.SolverMuJoCo.register_custom_attributes(ur10_sub)

ur10_sub.add_usd(
    asset_file,
    xform=wp.transform(wp.vec3(0.0, 0.0, PEDESTAL_HEIGHT)),
    collapse_fixed_joints=False,
    enable_self_collisions=False,
    hide_collision_shapes=True,
)
# Cylindrical pedestal attached to the world frame (fixed base).
ur10_sub.add_shape_cylinder(
    -1,
    xform=wp.transform(wp.vec3(0.0, 0.0, PEDESTAL_HEIGHT / 2.0)),
    half_height=PEDESTAL_HEIGHT / 2.0,
    radius=0.08,
)

# ─── Attach the excavator plow mesh to the UR10 end-effector ─────────────
# The plow OBJ is authored in centimetres; OBJ_CM_TO_M converts it to metres.
# The OBJ coordinate origin is near the mounting bracket of the plow, and the
# blade/scoop extends in the local negative-Z direction.
# Apply a 180° local rotation so the bowl opens downward for plowing.
PLOW_LOCAL_ROT = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(np.pi))

ee_link_body_idx = ur10_sub.body_label.index("/ur10/ee_link")
assert ur10_sub.body_label[ee_link_body_idx] == "/ur10/ee_link", f"Unexpected ee_link body index: {ee_link_body_idx}"

print(f"[Plow] Loading excavator mesh from {EXCAVATOR_OBJ_PATH} ...")
_plow_surface = mophi.load_obj(EXCAVATOR_OBJ_PATH, scale=OBJ_CM_TO_M)
plow_mesh = newton.Mesh(
    _plow_surface.vertices,
    _plow_surface.indices,
    compute_inertia=False,  # plow mass is negligible for this demo phase
    is_solid=False,  # treat as a surface shell (open plow geometry)
    color=(0.55, 0.45, 0.35),  # earthy brown — excavator steel colour
)
ur10_sub.add_shape_mesh(
    ee_link_body_idx,
    mesh=plow_mesh,
    xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), PLOW_LOCAL_ROT),
)
print(
    f"[Plow] Excavator plow attached to body {ee_link_body_idx} "
    f"({ur10_sub.body_label[ee_link_body_idx]})  "
    f"[{_plow_surface.num_vertices} vertices, {_plow_surface.num_faces} triangles]\n"
)

# Position control with stiff gains matching Newton's UR10 example.
for i in range(len(ur10_sub.joint_target_ke)):
    ur10_sub.joint_target_ke[i] = 500
    ur10_sub.joint_target_kd[i] = 50
    ur10_sub.joint_target_mode[i] = int(JointTargetMode.POSITION)
# Arm motion is therefore commanded through joint_target_pos values, not by
# directly applying torques in this demo.

# Replicate into a single-world builder and add the ground plane.
builder = newton.ModelBuilder()
builder.replicate(ur10_sub, WORLD_COUNT, spacing=(2.0, 2.0, 0.0))
builder.add_ground_plane()

newton_model = builder.finalize()

# SolverMuJoCo with contacts disabled — the arm doesn't need to collide with
# the ground during this first phase; DEME granular contacts will be added later.
newton_solver = newton.solvers.SolverMuJoCo(
    newton_model,
    disable_contacts=True,
)

print(f"[Newton] UR10 model built: {newton_model.body_count} bodies, {newton_model.joint_count} joints.\n")

# ─── Visualization backend selection ─────────────────────────────────────────
# USE_OMNIVERSE_VISUALIZATION is defined in the configuration block above.
# The simulation loop body (begin_frame / log_state / end_frame)
# is identical for both backends; only the constructor and close() differ.
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
if USE_OMNIVERSE_VISUALIZATION:
    vis = mophi.OmniverseVisualizer(output_path=str(OMNIVERSE_OUTPUT_PATH), fps=float(SIM_FPS))
    if not vis.pxr_available:
        mophi.fatal(
            "pxr (OpenUSD) is not installed — Omniverse USD export unavailable.\n" "Install with:  pip install usd-core"
        )
    print("[USD] pxr (OpenUSD) available — Omniverse USD export enabled.\n")
else:
    vis = mophi.OpenGLVisualizer(newton_model)
    print("[Viewer] MoPhi OpenGL visualization window opened.\n")

# ─── Build the DEME granular terrain ──────────────────────────────────────
# Creates a pile of ellipsoidal particles following the approach of
# DEM-Engine's DEMdemo_Plow.cpp demo.  Particles are sampled layer-by-layer
# with a Poisson-disk sampler and deposited near the ground plane.
# No interaction with the Newton excavator in this phase — DEME runs
# independently.  Particle–excavator coupling will be added in a future phase.
print("[DEME] Building granular terrain (ellipsoidal particles) ...")
deme_solver = DEME.DEMSolver()
deme_solver.UseFrictionalHertzianModel()
deme_solver.SetVerbosity("ERROR")

mat_walls = deme_solver.LoadMaterial({"E": 1e6, "nu": 0.3, "CoR": 0.3, "mu": 0.5})
mat_particles = deme_solver.LoadMaterial({"E": 1e6, "nu": 0.3, "CoR": 0.3, "mu": 0.5})
# Mixed contact properties between wall and particle materials.
deme_solver.SetMaterialPropertyPair("CoR", mat_walls, mat_particles, 0.3)
deme_solver.SetMaterialPropertyPair("mu", mat_walls, mat_particles, 0.5)

# Ellipsoid template (semi-axes 2:1:1) scaled to physical particle size.
# The CSV file encodes the clump geometry relative to unit sphere radii;
# Scale() resizes the template to a reasonable physical particle size.
particle_template = deme_solver.LoadClumpType(
    _DEM_TEMPLATE_MASS,
    _DEM_TEMPLATE_MOI,
    DEME.GetDEMEDataFile("clumps/ellipsoid_2_1_1.csv"),
    mat_particles,
)
particle_template.Scale(_DEM_TERRAIN_SCALING)

# Box domain: closed on bottom and four sides, open at the top so
# particles can be launched upward by the excavator without being trapped.
_DEM_BOX_POS_OFF_X, _DEM_BOX_POS_OFF_Y = 0.0, 1.2
deme_solver.InstructBoxDomainDimension(
    [-_DEM_WORLD_HS + _DEM_BOX_POS_OFF_X, _DEM_WORLD_HS + _DEM_BOX_POS_OFF_X],
    [-_DEM_WORLD_HS + _DEM_BOX_POS_OFF_Y, _DEM_WORLD_HS + _DEM_BOX_POS_OFF_Y],
    [_DEM_BOWL_BOT, _DEM_WORLD_HS * 10.0],
)
deme_solver.InstructBoxDomainBoundingBC("top_open", mat_walls)

# Sample particle positions layer by layer (mirrors DEMdemo_Plow.cpp).
# PDSampler guarantees spacing proportional to configured particle size, so
# no two initial particles overlap.
sampler = DEME.PDSampler(4.0 * _DEM_TERRAIN_SCALING)
pile_positions = []
layer_z = 0.0
while layer_z < _DEM_FILL_H:
    center = [_DEM_BOX_POS_OFF_X, _DEM_BOX_POS_OFF_Y, _DEM_FILL_BOT + layer_z]
    half_ext = [_DEM_FILL_HW, _DEM_FILL_HW, 0.0]
    layer_pts = sampler.SampleBox(center, half_ext)
    pile_positions.extend(layer_pts)
    layer_z += _DEM_LAYER_STEP

_dem_num_terrain_particles = len(pile_positions)
num_layers = int(_DEM_FILL_H / _DEM_LAYER_STEP) + 1
print(f"[DEME] Sampled {_dem_num_terrain_particles} terrain particle(s) " f"in {num_layers} layer(s).")

the_pile = deme_solver.AddClumps(particle_template, pile_positions)
the_pile.SetFamily(0)
_dem_terrain_tracker = deme_solver.Track(the_pile)

deme_solver.SetGravitationalAcceleration([0.0, 0.0, -9.81])
deme_solver.SetInitTimeStep(DEME_DT)
print(f"[DEME] Running at step size {DEME_DT}.\n")
deme_solver.SetErrorOutAvgContacts(100)

# pyDEME (older version) may have weird bin size adaptation mechanism, disable it
deme_solver.DisableAdaptiveBinSize()

# ── Excavator plow mesh: contact proxy for the plowing phase ──────────
# The same OBJ used by Newton for visualisation is loaded into DEME as a
# kinematic mesh body.  Contact with terrain particles is disabled during
# settling (sleep family) and enabled only once the arm reaches the pile.
#
# Family assignment (defined before Initialize()):
#   _DEME_PLOW_SLEEP_FAMILY — fixed, no contact with terrain family 0.
#   _DEME_PLOW_ACTIVE_FAMILY — fixed, contact with terrain enabled (default).
# The mesh switches from sleep to active via ChangeFamily() after settling.
# Even though both families are "fixed", the tracker allows externally
# updating the mesh pose each step (see pyDEME_ConePenetration.py pattern).
print(f"[DEME] Loading plow mesh from {EXCAVATOR_OBJ_PATH} ...")
_plow_deme_obj = deme_solver.AddWavefrontMeshObject(EXCAVATOR_OBJ_PATH, mat_walls)
# Scale from centimetres to metres, matching Newton's plow mesh.
_plow_deme_obj.Scale(OBJ_CM_TO_M)
# Place above the terrain pile during settling (contact disabled,
# so exact position is not critical here).
_plow_deme_obj.SetInitPos([_DEM_BOX_POS_OFF_X, _DEM_BOX_POS_OFF_Y, 2.0])
# Identity quaternion — will be corrected from Newton state after settling.
_plow_deme_obj.SetInitQuat([0.0, 0.0, 0.0, 1.0])
_plow_deme_obj.SetFamily(_DEME_PLOW_SLEEP_FAMILY)
# Both families are fixed so gravity cannot move the mesh;
# pose is driven externally via tracker.SetPos / SetOriQ each step.
deme_solver.SetFamilyFixed(_DEME_PLOW_SLEEP_FAMILY)
deme_solver.SetFamilyFixed(_DEME_PLOW_ACTIVE_FAMILY)
# Disable contact between terrain (family 0) and sleep family
# so particles fall freely without hitting the plow during settling.
deme_solver.DisableContactBetweenFamilies(_DEME_TERRAIN_FAMILY, _DEME_PLOW_SLEEP_FAMILY)
_plow_deme_tracker = deme_solver.Track(_plow_deme_obj)
print(
    f"[DEME] Plow mesh loaded "
    f"({_plow_deme_obj.GetNumTriangles()} triangles). "
    f"Contact disabled until settling completes.\n"
)

deme_solver.Initialize()

print(f"[DEME] Granular terrain initialized " f"({_dem_num_terrain_particles} ellipsoidal particle(s)).\n")

# ─── Initialize the coupler ────────────────────────────────────────────────
# XLB is passed as None; DEME terrain solver is connected when available
# (no Newton–DEME force coupling yet — particles run independently).
coupler = mophi.NewtonXLBDEMCoupler()
coupler.set_verbosity(mophi.VERBOSITY_INFO)

print("[Coupler] Initializing NewtonXLBDEMCoupler ...")
coupler.initialize(
    newton_model=newton_model,
    newton_solver=newton_solver,
    xlb_simulation=None,
    deme_solver=deme_solver,
    sim_dt=SIM_DT,
)
print("[Coupler] NewtonXLBDEMCoupler initialized.\n")

# ─── Fixed joint target setup (after coupler.initialize()) ───────────────
# ArticulationView provides structured access to the UR10 articulation's DOFs.
# The "*ur10*" glob matches all UR10 instances in the model.
# Control path summary:
#   1) Write desired joint positions into coupler.newton_control via
#      articulation_view.set_attribute("joint_target_pos", ...).
#   2) coupler.step_newton() advances one Newton step using those targets.
#   3) We update the target every frame by interpolating READY_TO_PLOW_Q →
#      PLOW_TARGET_Q, so shoulder and elbow are actively commanded throughout.
articulation_view = ArticulationView(
    newton_model,
    "*ur10*",
    exclude_joint_types=[newton.JointType.FREE, newton.JointType.DISTANCE],
)
assert (
    articulation_view.count == WORLD_COUNT
), f"Expected {WORLD_COUNT} UR10 articulation(s), found {articulation_view.count}"

dof_count = articulation_view.joint_dof_count

# Initialize to the ready-to-plow direction defined in the configuration block.
# Keep any extra DOFs at their current initialized values.
joint_q_target_np = articulation_view.get_attribute("joint_q", coupler.newton_state_0).numpy()
num_ready_dofs = min(dof_count, len(READY_TO_PLOW_Q))
joint_q_target_np[:, 0, :num_ready_dofs] = READY_TO_PLOW_Q[:num_ready_dofs]

joint_q_target_wp = wp.array(joint_q_target_np, dtype=wp.float32, device=device)
articulation_view.set_attribute("joint_q", coupler.newton_state_0, joint_q_target_wp)
articulation_view.set_attribute("joint_target_pos", coupler.newton_control, joint_q_target_wp)

print(f"[Control] Applied fixed ready-to-plow joint target for {num_ready_dofs} DOFs.\n")

# ─── Newton arm warm-up (Phase 0) ─────────────────────────────────────────
# Run Newton alone for NEWTON_WARMUP_FRAMES render-frames so the stiff
# position controller drives the arm to its ready-to-plow configuration.
# After this phase, coupler.newton_state_0.body_q contains the correct
# end-effector world transform that is used to place the DEME plow mesh.
print(f"[Newton] Running warm-up ({NEWTON_WARMUP_FRAMES} frames × {SIM_SUBSTEPS} substeps) ...")
for _ in range(NEWTON_WARMUP_FRAMES * SIM_SUBSTEPS):
    coupler.step_newton()
print("[Newton] Warm-up complete.\n")

# ─── DEME terrain visualisation arrays ────────────────────────────────────
# Clump template geometry mirrors the ellipsoid_2_1_1.csv data fed to the DEME
# solver (5 component spheres; template-unit values scaled by _DEM_TERRAIN_SCALING).
# CSV columns are x, y, z, r (offset from clump centre in local frame, then radius).
# These arrays mirror the template's component-sphere layout so visualized clumps
# match DEME particle geometry.
_DEM_CLUMP_SPHERE_RADII = np.array([1.00, 0.88, 0.64, 0.88, 0.64], dtype=np.float32) * _DEM_TERRAIN_SCALING
_DEM_CLUMP_SPHERE_OFFSETS = (
    np.array(
        [[0.0, 0.0, 0.00], [0.0, 0.0, 0.86], [0.0, 0.0, 1.44], [0.0, 0.0, -0.86], [0.0, 0.0, -1.44]],
        dtype=np.float32,
    )
    * _DEM_TERRAIN_SCALING
)

_dem_terrain_colors_wp = wp.array(
    np.tile([0.76, 0.60, 0.42], (_dem_num_terrain_particles, 1)).astype(np.float32),
    dtype=wp.vec3,
)
print(f"[Viewer] {_dem_num_terrain_particles} DEME terrain particle(s) registered for visualisation.\n")

# ─── DEME terrain settling (Phase 1) ──────────────────────────────────────
# Let the granular pile settle under gravity for DEME_SETTLE_TIME seconds
# before introducing plow contact.  The plow mesh is present in DEME but
# contact with terrain particles is disabled (sleep family).
# DoDynamicsThenSync() is a blocking call that advances DEME internally and
# returns with a synchronized state — safe to call tracker.SetPos() after it.
if not USE_OMNIVERSE_VISUALIZATION:
    vis.set_camera(
        pos=wp.vec3(3.0, -3.0, 2.5),
        pitch=-25.0,
        yaw=135.0,
    )

print(f"[DEME] Settling terrain for {DEME_SETTLE_TIME:.1f} s ...")
if RENDER_SETTLING_PHASE and not USE_OMNIVERSE_VISUALIZATION:
    _settling_time = 0.0
    _settling_frame_count = 0
    while _settling_time < DEME_SETTLE_TIME - _SETTLING_TIME_EPSILON:
        if not vis.is_running():
            print(f"\n[Viewer] Window closed by user during settling after {_settling_frame_count} frame(s).")
            break
        _settling_dt = min(FRAME_DT, DEME_SETTLE_TIME - _settling_time)
        deme_solver.DoDynamicsThenSync(_settling_dt)
        _settling_time += _settling_dt
        _settling_frame_count += 1

        vis.begin_frame(_settling_time)
        vis.log_state(coupler.newton_state_0)
        _terrain_pos_np = np.array(_dem_terrain_tracker.Positions(), dtype=np.float32)
        _terrain_quat_np = np.array(_dem_terrain_tracker.OrientationQuaternions(), dtype=np.float32)
        _dem_terrain_pos_wp = wp.array(_terrain_pos_np, dtype=wp.vec3)
        _dem_terrain_orient_wp = wp.array(_terrain_quat_np, dtype=wp.vec4)
        vis.log_clumps(
            "dem_terrain",
            _dem_terrain_pos_wp,
            _dem_terrain_orient_wp,
            _DEM_CLUMP_SPHERE_RADII,
            _DEM_CLUMP_SPHERE_OFFSETS,
            colors=_dem_terrain_colors_wp,
        )
        vis.end_frame()
else:
    deme_solver.DoDynamicsThenSync(DEME_SETTLE_TIME)
print("[DEME] Settling complete.\n")

# After settling, teleport the DEME plow to match the Newton arm's current
# end-effector pose, then activate contact.
#
# Newton body_q layout (per Warp transform): [px, py, pz, qx, qy, qz, qw].
# The plow is attached to the ee_link body with local rotation _PLOW_LOCAL_ROT_NP,
# so the DEME mesh world orientation = ee_link_world_quat * _PLOW_LOCAL_ROT_NP.
_body_q_np = coupler.newton_state_0.body_q.numpy()
_ee_pos = _sync_deme_plow_pose_from_newton(_plow_deme_tracker, _body_q_np, ee_link_body_idx)
# Switch from sleep family to active family; contact with terrain is
# enabled for _DEME_PLOW_ACTIVE_FAMILY by default (never disabled against
# _DEME_TERRAIN_FAMILY).
deme_solver.ChangeFamily(_DEME_PLOW_SLEEP_FAMILY, _DEME_PLOW_ACTIVE_FAMILY)
_ee_pos_str = ", ".join(f"{v:.3f}" for v in _ee_pos)
print(f"[DEME] Plow contact activated.  Plow placed at ({_ee_pos_str}).\n")

# ─── Movie recording setup ────────────────────────────────────────────────
# SAVE_MOVIE, MOVIE_OUTPUT_PATH, and MOVIE_FPS are set in the configuration block.
_movie_writer = None
if SAVE_MOVIE and not USE_OMNIVERSE_VISUALIZATION:
    import imageio  # pip install imageio imageio-ffmpeg

    _movie_writer = imageio.get_writer(MOVIE_OUTPUT_PATH, fps=MOVIE_FPS)
    print(f"[Movie] Recording simulation to '{MOVIE_OUTPUT_PATH}' at {MOVIE_FPS} fps.\n")

# ─── Co-simulation loop (Phase 2: active plowing) ─────────────────────────
# Each frame:
#   1. Ramp Newton joint targets from ready-to-plow toward PLOW_TARGET_Q.
#   2. For each Newton substep:
#        a. Pass the DEME contact force from the previous substep to Newton (if enabled).
#        b. Advance Newton one substep.
#        c. Sync DEME plow pose from Newton's ee_link world transform.
#        d. Advance DEME (DEME_SUBSTEPS micro-steps per Newton substep).
#        e. Query the updated DEME contact force for use in the next substep.
#   3. Visualize Newton arm state and DEME terrain particles.
print(
    f"Running up to {NUM_FRAMES} frame(s) "
    f"(frame_dt={FRAME_DT * 1000:.1f} ms, "
    f"{SIM_SUBSTEPS} Newton substeps × {NEWTON_DT * 1000:.3f} ms, "
    f"{DEME_SUBSTEPS} DEME micro-steps per Newton substep × {DEME_DT * 1000:.3f} ms) ...\n"
)

# Point the camera toward the arm base and terrain pile with a slight downward
# angle so both robot motion and near-ground particle behavior remain visible.
if not USE_OMNIVERSE_VISUALIZATION:
    vis.set_camera(
        pos=wp.vec3(3.0, -3.0, 2.5),
        pitch=-25.0,
        yaw=135.0,
    )

# Static world-orientation and one-metre scale reference.
mophi.log_orientation_and_scale_reference(
    vis,
    axis_origin=REFERENCE_AXIS_ORIGIN,
    scale_bar_center=REFERENCE_SCALE_BAR_CENTER,
)

sim_time = 0.0
_sim_duration = max(NUM_FRAMES * FRAME_DT, _MIN_DURATION_EPSILON)
# Keep scoop phase at/after the end of base plow interpolation so wrist_3
# overlay does not compete with unfinished READY_TO_PLOW_Q -> PLOW_TARGET_Q motion.
_scoop_start_time = max(WRIST_SCOOP_START_FRACTION * _sim_duration, PLOW_DURATION)
_scoop_phase_dt = max(_sim_duration - _scoop_start_time, _MIN_DURATION_EPSILON)

# ─── DEME → Newton force feedback setup ───────────────────────────────────────
# A reusable (body_count, 6) float32 scratch buffer for building the external-force
# wp.array passed to the coupler each substep.  Only the ee_link entry is written;
# all other entries stay zero.  The DEME force from the previous substep is kept in
# _deme_contact_force_np and re-used as the injection for the current substep (one
# substep lag — acceptable for this co-simulation cadence).
_ext_forces_np = np.zeros((newton_model.body_count, 6), dtype=np.float32)
_deme_contact_force_np = np.zeros(3, dtype=np.float32)  # net world-space [Fx, Fy, Fz]
_deme_contact_torque_np = np.zeros(3, dtype=np.float32)  # net world-space [Tx, Ty, Tz]
# These arrays are only consumed inside ENABLE_DEME_FORCE_FEEDBACK branches.
# Keeping them always initialized avoids conditional local-name coupling.
# Force feedback uses a one-Newton-substep lag when enabled: the DEME contact
# force queried at the end of substep N is injected into Newton at substep N+1.
# At NEWTON_DT = 2 ms this lag is at most one substep (2 ms), well within the
# coupling bandwidth of the position-controlled arm at RENDER_FPS = 50 Hz.
if ENABLE_DEME_FORCE_FEEDBACK:
    _plow_mass = float(_plow_deme_tracker.Mass())
    _plow_moi_local_np = np.asarray(_plow_deme_tracker.MOI(), dtype=np.float32)

for frame in range(NUM_FRAMES):
    # Stop early if the OpenGL viewer window has been closed by the user.
    # For OmniverseVisualizer, vis.is_running() always returns True.
    if not vis.is_running():
        print(f"\n[Viewer] Window closed by user after frame {frame}.")
        break

    # ── Ramp joint targets toward plowing configuration ────────────────────
    # Linear interpolation from READY_TO_PLOW_Q to PLOW_TARGET_Q over
    # PLOW_DURATION seconds.  This drives shoulder_lift and elbow continuously
    # every frame; after PLOW_DURATION the arm holds the final pose.
    #
    # Near the end of the full simulation, apply an extra inward rotation on
    # wrist_3 to scoop particles toward the arm's central rod.
    _plow_interp = np.clip(sim_time / PLOW_DURATION, 0.0, 1.0)
    _plow_q = (1.0 - _plow_interp) * READY_TO_PLOW_Q + _plow_interp * PLOW_TARGET_Q
    _joint_q_cmd = _plow_q.copy()
    if num_ready_dofs > _WRIST_3_DOF_INDEX:
        if sim_time >= _scoop_start_time:
            _scoop_interp = np.clip((sim_time - _scoop_start_time) / _scoop_phase_dt, 0.0, 1.0)
            _wrist_3_scoop_target = PLOW_TARGET_Q[_WRIST_3_DOF_INDEX] + WRIST_SCOOP_INWARD_DELTA
            _joint_q_cmd[_WRIST_3_DOF_INDEX] = (1.0 - _scoop_interp) * _plow_q[
                _WRIST_3_DOF_INDEX
            ] + _scoop_interp * _wrist_3_scoop_target

    joint_q_target_np[:, 0, :num_ready_dofs] = _joint_q_cmd[:num_ready_dofs]
    joint_q_target_wp = wp.array(joint_q_target_np, dtype=wp.float32, device=device)
    articulation_view.set_attribute("joint_target_pos", coupler.newton_control, joint_q_target_wp)

    # ── Collaboration loop: counts are derived from explicit dt constants ──
    for _ in range(SIM_SUBSTEPS):
        # If force feedback is enabled, inject the DEME contact force (from the
        # previous DEME micro-step cycle) into Newton before this substep so the
        # integrator sees the granular resistance alongside joint-actuator torques.
        if ENABLE_DEME_FORCE_FEEDBACK:
            _ext_forces_np[ee_link_body_idx, :3] = _deme_contact_force_np
            _ext_forces_np[ee_link_body_idx, 3:6] = _deme_contact_torque_np
            coupler.set_newton_body_forces(wp.array(_ext_forces_np, dtype=wp.spatial_vector))

        # Advance Newton one substep with the current joint targets.
        coupler.step_newton()

        # Synchronise DEME plow pose with Newton's end-effector world transform.
        # The plow mesh is kinematic (SetFamilyFixed), so DEME will not move it
        # on its own; we must update it explicitly every Newton substep.
        # World orientation = ee_link_world_quat * _PLOW_LOCAL_ROT_NP (180° about X).
        _body_q_np = coupler.newton_state_0.body_q.numpy()
        _sync_deme_plow_pose_from_newton(_plow_deme_tracker, _body_q_np, ee_link_body_idx)

        # ── DEME micro-step loop (runs at explicit DEME_DT) ──
        for _ in range(DEME_SUBSTEPS):
            coupler.step_deme()

        # Query contact-induced acceleration on the tracked plow owner after DEME
        # micro-steps complete, then convert to equivalent net wrench:
        #   force_world = mass * ContactAcc()
        #   torque_local_principal = MOI_principal * ContactAngAccLocal()
        #   torque_world = rotate(local_principal_torque, OriQ()).
        if ENABLE_DEME_FORCE_FEEDBACK:
            _contact_acc_world_np = np.asarray(_plow_deme_tracker.ContactAcc(), dtype=np.float32)
            _deme_contact_force_np[:] = _plow_mass * _contact_acc_world_np

            _contact_ang_acc_local_np = np.asarray(_plow_deme_tracker.ContactAngAccLocal(), dtype=np.float32)
            _contact_torque_local_np = _plow_moi_local_np * _contact_ang_acc_local_np
            _plow_ori_q_np = np.asarray(_plow_deme_tracker.OriQ(), dtype=np.float32)
            _deme_contact_torque_np[:] = _rotate_vector_by_quat_xyzw(_plow_ori_q_np, _contact_torque_local_np)

    sim_time += FRAME_DT

    # ── Visualization ──────────────────────────────────────────────────────
    vis.begin_frame(sim_time)
    vis.log_state(coupler.newton_state_0)

    # Render live DEME terrain particles as clumps (overlapping-sphere assemblies).
    # Each particle's template geometry matches the ellipsoid_2_1_1.csv clump
    # fed to the DEME solver; live positions and orientations come from the tracker.
    _terrain_pos_np = np.array(_dem_terrain_tracker.Positions(), dtype=np.float32)
    _terrain_quat_np = np.array(_dem_terrain_tracker.OrientationQuaternions(), dtype=np.float32)
    _dem_terrain_pos_wp = wp.array(_terrain_pos_np, dtype=wp.vec3)
    _dem_terrain_orient_wp = wp.array(_terrain_quat_np, dtype=wp.vec4)
    vis.log_clumps(
        "dem_terrain",
        _dem_terrain_pos_wp,
        _dem_terrain_orient_wp,
        _DEM_CLUMP_SPHERE_RADII,
        _DEM_CLUMP_SPHERE_OFFSETS,
        colors=_dem_terrain_colors_wp,
    )

    vis.end_frame()

    if _movie_writer is not None:
        _movie_writer.append_data(vis.get_frame().numpy())

    # ── Console status (periodic) ──────────────────────────────────────────
    if (frame + 1) % 50 == 0 or frame == 0:
        print(f"  frame {frame + 1:>4}/{NUM_FRAMES}  sim_time={sim_time:.3f} s")

print()

# ─── Debug pause after first rendered frame ──────────────────────────────────
if PAUSE_AFTER_FIRST_FRAME and not USE_OMNIVERSE_VISUALIZATION and vis.is_running():
    print("[Debug] First frame rendered. Inspect the initial configuration; close the viewer window to continue.")
    while vis.is_running():
        vis.begin_frame(sim_time)
        vis.log_state(coupler.newton_state_0)

        _terrain_pos_np = np.array(_dem_terrain_tracker.Positions(), dtype=np.float32)
        _terrain_quat_np = np.array(_dem_terrain_tracker.OrientationQuaternions(), dtype=np.float32)
        _dem_terrain_pos_wp = wp.array(_terrain_pos_np, dtype=wp.vec3)
        _dem_terrain_orient_wp = wp.array(_terrain_quat_np, dtype=wp.vec4)
        vis.log_clumps(
            "dem_terrain",
            _dem_terrain_pos_wp,
            _dem_terrain_orient_wp,
            _DEM_CLUMP_SPHERE_RADII,
            _DEM_CLUMP_SPHERE_OFFSETS,
            colors=_dem_terrain_colors_wp,
        )

        vis.end_frame()
        time.sleep(5.0)

# ─── Finalize ─────────────────────────────────────────────────────────────
if _movie_writer is not None:
    _movie_writer.close()
    print(f"[Movie] Saved simulation recording to '{MOVIE_OUTPUT_PATH}'.\n")

vis.close()

print("[Coupler] Finalizing NewtonXLBDEMCoupler ...")
coupler.finalize()
print("[Coupler] NewtonXLBDEMCoupler finalized.\n")

print("Demo completed successfully.")
print(f"  Newton version : {newton.__version__}")
print(f"  Warp  version  : {wp.__version__}")
print(f"  DEME           : installed ({_dem_num_terrain_particles} terrain particles)")
