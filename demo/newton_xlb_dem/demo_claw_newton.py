"""demo/newton_xlb_dem/demo_claw_newton.py

Newton UR10 robot arm co-simulation demo — groundwork phase for claw plowing.

This is the first step toward a full claw-plowing-granular-material co-simulation.
It reproduces Newton's ``example_robot_ur10`` example inside MoPhi's
``NewtonXLBDEMCoupler`` framework, demonstrating a single UR10 6-DOF industrial
robot arm executing sinusoidal joint trajectories.

The robot is mounted on a cylindrical pedestal above the ground plane.  All six
revolute joints sweep continuously through their full range using pre-computed
sinusoidal target trajectories, exercising the arm's full reach envelope — the
motion that will later be used to plow granular material.

An excavator plow mesh (``data/mesh/excavator.obj``) is rigidly attached to the
UR10's end-effector link (``ee_link``).  The OBJ file uses centimetre units; the
mesh is scaled by 0.01 when loaded so it is correctly sized in metres.  The plow
sweeps through space as the arm moves, providing a visual preview of the plowing
trajectory that DEME particles will later interact with.

Future phases will add:
  • DEME (discrete-element) granular particles displaced by the arm's end-effector.
  • XLB (lattice-Boltzmann) fluid flow around the moving arm.

The demo closely mirrors Newton's ``example_robot_ur10`` setup (single arm,
SolverMuJoCo, per-substep sinusoidal joint-target updates) while wrapping all
physics inside the MoPhi ``NewtonXLBDEMCoupler``.

Prerequisites
-------------
  • Build MoPhi with -DMOPHI_BUILD_NEWTON_XLB_DEM=ON (compiles
    NewtonXLBDEMCoupler and builds the mophi_core Python extension module).
  • pip install newton warp-lang   (Newton rigid-body physics + Warp GPU runtime)

Running
-------
From the MoPhi build tree (after ``cmake --build``):

    python demo/newton_xlb_dem/demo_claw_newton.py

Or from the repository root after installing the mophi package:

    python -m demo.newton_xlb_dem.demo_claw_newton
"""

import os
import sys

import numpy as np

# ─── Import MoPhi ─────────────────────────────────────────────────────────
try:
    import mophi
except ImportError as exc:
    sys.exit(
        "ERROR: Could not import the 'mophi' package.\n"
        "       Build MoPhi with -DMOPHI_BUILD_NEWTON_XLB_DEM=ON and ensure that\n"
        "       the build directory (python/) is on PYTHONPATH.\n"
        f"       ({exc})"
    )

if not hasattr(mophi, "NewtonXLBDEMCoupler"):
    sys.exit(
        "ERROR: mophi.NewtonXLBDEMCoupler is not available.\n"
        "       Re-build MoPhi with -DMOPHI_BUILD_NEWTON_XLB_DEM=ON."
    )

# ─── Import Newton + Warp ──────────────────────────────────────────────────
try:
    import newton
    import warp as wp
    from newton import JointTargetMode
    from newton.selection import ArticulationView

    _newton_available = True
except ImportError:
    _newton_available = False
    print(
        "ERROR: Newton (or warp) is not installed.\n"
        "       Install with:  pip install newton warp-lang"
    )
    sys.exit(1)

# ─── Warp kernel: per-substep sinusoidal joint-target update ──────────────
# Mirrors the trajectory kernel from Newton's example_robot_ur10.py.
# Each substep the simulation-time parameter `t` advances by `dt`, and the
# corresponding joint target is linearly interpolated from the pre-computed
# trajectory table.  `dim` is set to `world_count` (1 for a single arm).


# ─── OBJ mesh loader ──────────────────────────────────────────────────────
# Pure-Python parser for Wavefront OBJ files.  Handles the v//vn and v/vt/vn
# face formats used by the excavator plow mesh (all faces are triangles).
def _load_obj_mesh(path: str, scale: float = 1.0):
    """Parse a Wavefront OBJ file and return (vertices, indices) as numpy arrays.

    Args:
        path:  Path to the .obj file.
        scale: Uniform scale factor applied to all vertex coordinates.  Use
               0.01 to convert centimetre OBJ coordinates to metres.

    Returns:
        vertices:  float32 ndarray of shape (N, 3) — vertex positions [m].
        indices:   int32 ndarray of shape (F*3,) — flat triangle index list.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    vertices = []
    indices = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]) * scale, float(parts[2]) * scale, float(parts[3]) * scale])
            elif line.startswith("f "):
                # Each token is v, v/vt, v//vn, or v/vt/vn — take the vertex index only.
                parts = line.split()[1:]
                face_verts = [int(p.split("/")[0]) - 1 for p in parts]  # OBJ uses 1-based indices
                if len(face_verts) == 3:
                    indices.extend(face_verts)
                elif len(face_verts) == 4:
                    # Fan-triangulate quads.
                    indices.extend([face_verts[0], face_verts[1], face_verts[2]])
                    indices.extend([face_verts[0], face_verts[2], face_verts[3]])
    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.int32)


@wp.kernel
def update_joint_target_trajectory_kernel(
    joint_target_trajectory: wp.array3d[wp.float32],
    time: wp.array[wp.float32],
    dt: wp.float32,
    # output
    joint_target: wp.array3d[wp.float32],
):
    world_idx = wp.tid()
    t = time[world_idx]
    t = wp.mod(t + dt, float(joint_target_trajectory.shape[0] - 1))
    step = int(t)
    time[world_idx] = t

    num_dofs = joint_target.shape[2]
    for dof in range(num_dofs):
        # Offset dof index by world_idx so each arm (if replicated) sweeps a
        # different phase; for world_count=1 this has no effect.
        di = (dof + world_idx) % num_dofs
        joint_target[world_idx, 0, dof] = wp.lerp(
            joint_target_trajectory[step, world_idx, di],
            joint_target_trajectory[step + 1, world_idx, di],
            wp.frac(t),
        )


# ─── Simulation timing ────────────────────────────────────────────────────
# Match Newton's UR10 example: 50 Hz policy rate, 10 substeps per frame,
# MuJoCo solver at 1/500 s time step.
WORLD_COUNT = 1  # single UR10 arm
SIM_FPS = 50
SIM_SUBSTEPS = 10
FRAME_DT = 1.0 / SIM_FPS
SIM_DT = FRAME_DT / SIM_SUBSTEPS  # 1/500 s per substep
NUM_FRAMES = 250  # ≈ 5 s at 50 Hz (or until the viewer is closed)
CONTROL_SPEED = 50.0  # trajectory parameter speed [trajectory-steps / sim-second]

# ─── Trajectory constants ─────────────────────────────────────────────────
# Joints whose reported limit magnitude exceeds this threshold (6 rad ≈ 343°,
# i.e., nearly two full turns) are considered effectively unbounded and are
# assigned a ±π working range instead.
UNBOUNDED_JOINT_LIMIT_THRESHOLD = 6.0  # radians

# Number of trajectory samples per radian of joint range.  With CONTROL_SPEED=50,
# advancing at SIM_DT × CONTROL_SPEED = 0.1 steps/substep, a table built at
# 50 samples/rad covers ~0.02 rad per substep — fine enough for smooth motion.
TRAJECTORY_SAMPLES_PER_RADIAN = 50

# ─── Initialize Warp ───────────────────────────────────────────────────────
print("=== MoPhi UR10 Claw Arm co-simulation demo ===\n")
wp.init()
device = wp.get_device()

# ─── Paths ────────────────────────────────────────────────────────────────
# Resolve the excavator OBJ mesh relative to the repository root so the demo
# can be run from any working directory.
_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_DEMO_DIR))
EXCAVATOR_OBJ_PATH = os.path.join(_REPO_ROOT, "data", "mesh", "excavator.obj")

# ─── Load the UR10 robot model ─────────────────────────────────────────────
# newton.utils.download_asset("universal_robots_ur10") downloads the UR10 USD
# from the Newton Assets repository.  The arm is mounted on a cylindrical
# pedestal at height=1.2 m so its base link clears the ground plane.
print("[Newton] Downloading UR10 robot assets ...")
asset_path = newton.utils.download_asset("universal_robots_ur10")
asset_file = str(asset_path / "usd" / "ur10_instanceable.usda")
print(f"[Newton] Assets ready: {asset_path}\n")

print("[Newton] Building UR10 model ...")
pedestal_height = 1.2

# Build a single-arm sub-builder, then replicate it WORLD_COUNT times.
# This matches Newton's example pattern and lets ArticulationView find the
# articulation by the "*ur10*" glob pattern.
ur10_sub = newton.ModelBuilder()
newton.solvers.SolverMuJoCo.register_custom_attributes(ur10_sub)

ur10_sub.add_usd(
    asset_file,
    xform=wp.transform(wp.vec3(0.0, 0.0, pedestal_height)),
    collapse_fixed_joints=False,
    enable_self_collisions=False,
    hide_collision_shapes=True,
)
# Cylindrical pedestal: body index -1 attaches to the world (fixed).
ur10_sub.add_shape_cylinder(
    -1,
    xform=wp.transform(wp.vec3(0.0, 0.0, pedestal_height / 2.0)),
    half_height=pedestal_height / 2.0,
    radius=0.08,
)

# ─── Attach the excavator plow mesh to the UR10 end-effector ─────────────
# The plow OBJ is authored in centimetres (bounding box ≈ 60 × 86 × 57 cm).
# Scaling by OBJ_CM_TO_M converts it to metres.
# The OBJ coordinate origin is near the mounting bracket of the plow (Z ≈ 0),
# with the blade/scoop extending in the −Z direction (Z ≈ −0.54 m after scale).
# Attaching to the ee_link (body index 7) with identity local transform places
# the plow mount at the end-effector flange and lets the blade sweep freely as
# the arm moves through its sinusoidal trajectory.
OBJ_CM_TO_M = 0.01

ee_link_body_idx = ur10_sub.body_label.index("/ur10/ee_link")
assert ur10_sub.body_label[ee_link_body_idx] == "/ur10/ee_link", (
    f"Unexpected ee_link body index: {ee_link_body_idx}"
)

if os.path.isfile(EXCAVATOR_OBJ_PATH):
    print(f"[Plow] Loading excavator mesh from {EXCAVATOR_OBJ_PATH} ...")
    _plow_verts, _plow_indices = _load_obj_mesh(EXCAVATOR_OBJ_PATH, scale=OBJ_CM_TO_M)
    plow_mesh = newton.Mesh(
        _plow_verts,
        _plow_indices,
        compute_inertia=False,  # plow mass is negligible for this demo phase
        is_solid=False,         # treat as a surface shell (open plow geometry)
        color=(0.55, 0.45, 0.35),  # earthy brown — excavator steel colour
    )
    ur10_sub.add_shape_mesh(
        ee_link_body_idx,
        mesh=plow_mesh,
    )
    print(
        f"[Plow] Excavator plow attached to body {ee_link_body_idx} "
        f"({ur10_sub.body_label[ee_link_body_idx]})  "
        f"[{len(_plow_verts)} vertices, {len(_plow_indices) // 3} triangles]\n"
    )
else:
    print(f"[Plow] WARNING: excavator OBJ not found at {EXCAVATOR_OBJ_PATH} — plow will be omitted.\n")

# Position control with stiff gains matching Newton's UR10 example.
for i in range(len(ur10_sub.joint_target_ke)):
    ur10_sub.joint_target_ke[i] = 500
    ur10_sub.joint_target_kd[i] = 50
    ur10_sub.joint_target_mode[i] = int(JointTargetMode.POSITION)

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
# Set USE_OMNIVERSE_VISUALIZATION = True to write each frame to a USD scene file
# (requires:  pip install usd-core).
# Set USE_OMNIVERSE_VISUALIZATION = False (default) to use Newton's real-time
# OpenGL window via mophi.OpenGLVisualizer.
# The simulation loop body (begin_frame / log_state / end_frame)
# is identical for both backends; only the constructor and close() differ.
USE_OMNIVERSE_VISUALIZATION = False

vis = None
_vis_available = False

if USE_OMNIVERSE_VISUALIZATION:
    vis = mophi.OmniverseVisualizer(output_path="demo_claw_newton.usdc", fps=float(SIM_FPS))
    _vis_available = vis.pxr_available
    if _vis_available:
        print("[USD] pxr (OpenUSD) available — Omniverse USD export enabled.\n")
    else:
        print(
            "[USD] pxr (OpenUSD) is not installed — Omniverse export disabled.\n"
            "      Install with:  pip install usd-core\n"
            "      The simulation will run without visualization."
        )
else:
    try:
        vis = mophi.OpenGLVisualizer(newton_model)
        _vis_available = True
        print("[Viewer] MoPhi OpenGL visualization window opened.\n")
    except Exception as exc:
        vis = None
        _vis_available = False
        print(
            f"[Viewer] Could not open MoPhi OpenGL viewer ({exc}).\n"
            "         The simulation will run without visualization."
        )

# ─── Initialize the coupler ────────────────────────────────────────────────
# XLB and DEME are passed as None — they will be connected in future phases.
coupler = mophi.NewtonXLBDEMCoupler()
coupler.set_verbosity(mophi.VERBOSITY_INFO)

print("[Coupler] Initializing NewtonXLBDEMCoupler ...")
coupler.initialize(
    newton_model=newton_model,
    newton_solver=newton_solver,
    xlb_simulation=None,
    deme_solver=None,
    sim_dt=SIM_DT,
)
print("[Coupler] NewtonXLBDEMCoupler initialized.\n")

# ─── Joint trajectory setup (after coupler.initialize()) ─────────────────
# ArticulationView provides structured access to the UR10 articulation's DOFs.
# The "*ur10*" glob matches all UR10 instances in the model; for WORLD_COUNT=1
# this gives view.count = 1.
articulation_view = ArticulationView(
    newton_model,
    "*ur10*",
    exclude_joint_types=[newton.JointType.FREE, newton.JointType.DISTANCE],
)
assert articulation_view.count == WORLD_COUNT, (
    f"Expected {WORLD_COUNT} UR10 articulation(s), found {articulation_view.count}"
)

dof_count = articulation_view.joint_dof_count

# Read joint limits and initial joint positions from the post-FK state that
# coupler.initialize() created (eval_fk has already been called on newton_state_0).
dof_lower = articulation_view.get_attribute("joint_limit_lower", newton_model)[0, 0].numpy()
dof_upper = articulation_view.get_attribute("joint_limit_upper", newton_model)[0, 0].numpy()
joint_q_init = articulation_view.get_attribute("joint_q", coupler.newton_state_0).numpy().squeeze(axis=1)

# Build a sinusoidal trajectory table for every DOF, phase-shifted so that each
# joint starts at its current position.  This mirrors Newton's UR10 example.
joint_target_trajectory = np.zeros((0, WORLD_COUNT, dof_count), dtype=np.float32)

for i in range(dof_count):
    lower = dof_lower[i]
    upper = dof_upper[i]
    if not np.isfinite(lower) or abs(lower) > UNBOUNDED_JOINT_LIMIT_THRESHOLD:
        # Unbounded joint: assume full ±π range.
        lower = -float(np.pi)
        upper = float(np.pi)
    limit_range = upper - lower
    normalized = (joint_q_init[:, i] - lower) / limit_range * 2.0 - 1.0
    phase_shift = np.zeros(WORLD_COUNT)
    mask = np.abs(normalized) < 1.0
    phase_shift[mask] = np.arcsin(normalized[mask])

    traj = np.sin(np.linspace(phase_shift, 2.0 * np.pi + phase_shift, int(limit_range * TRAJECTORY_SAMPLES_PER_RADIAN)))
    traj = traj * (upper - lower) * 0.5 + 0.5 * (upper + lower)

    target_trajectory = np.tile(joint_q_init, (len(traj), 1, 1))
    target_trajectory[:, :, i] = traj
    joint_target_trajectory = np.concatenate((joint_target_trajectory, target_trajectory), axis=0)

# Upload trajectory table to device as a Warp array (shape: [num_steps, worlds, dofs]).
joint_target_trajectory_wp = wp.array(joint_target_trajectory, dtype=wp.float32, device=device)
# Per-world trajectory clock — advances by SIM_DT × CONTROL_SPEED each substep.
time_step_wp = wp.zeros(WORLD_COUNT, dtype=wp.float32, device=device)

# ctrl is a (world_count, 1, dof_count) Warp array that the kernel writes into;
# set_attribute copies it back to coupler.newton_control each substep.
ctrl = articulation_view.get_attribute("joint_target_pos", coupler.newton_control)

print(
    f"[Trajectory] Pre-computed sinusoidal trajectories for {dof_count} DOFs "
    f"({joint_target_trajectory_wp.shape[0]} trajectory steps).\n"
)

# ─── Movie recording settings ─────────────────────────────────────────────
# Set SAVE_MOVIE = True to record the rendered simulation frames to a video file.
# Requires: pip install imageio imageio-ffmpeg
SAVE_MOVIE = False
MOVIE_OUTPUT_PATH = "demo_claw_newton.mp4"
MOVIE_FPS = SIM_FPS

_movie_writer = None
if SAVE_MOVIE and _vis_available and not USE_OMNIVERSE_VISUALIZATION:
    try:
        import imageio

        _movie_writer = imageio.get_writer(MOVIE_OUTPUT_PATH, fps=MOVIE_FPS)
        print(f"[Movie] Recording simulation to '{MOVIE_OUTPUT_PATH}' at {MOVIE_FPS} fps.\n")
    except ImportError:
        print(
            "WARNING: imageio is not installed — movie recording disabled.\n"
            "         Install with:  pip install imageio imageio-ffmpeg"
        )
        SAVE_MOVIE = False

# ─── Co-simulation loop ────────────────────────────────────────────────────
# Each frame:
#   1. For each substep: update joint targets, then advance Newton via coupler.step().
#   2. Visualize the resulting state.
print(
    f"Running up to {NUM_FRAMES} frame(s) "
    f"(frame_dt={FRAME_DT * 1000:.1f} ms, {SIM_SUBSTEPS} substeps × {SIM_DT * 1000:.1f} ms) ...\n"
)

# Point the camera toward the arm base so the full reach envelope is visible.
if _vis_available and not USE_OMNIVERSE_VISUALIZATION:
    vis.set_camera(
        pos=wp.vec3(3.0, -3.0, 2.5),
        pitch=-15.0,
        yaw=45.0,
    )

sim_time = 0.0

for frame in range(NUM_FRAMES):
    # Stop early if the OpenGL viewer window has been closed by the user.
    # For OmniverseVisualizer, vis.is_running() always returns True.
    if _vis_available and not vis.is_running():
        print(f"\n[Viewer] Window closed by user after frame {frame}.")
        break

    # ── Substep loop: update joint targets + advance Newton ───────────────
    for _ in range(SIM_SUBSTEPS):
        # Compute sinusoidal joint targets for this substep and write them into
        # coupler.newton_control so the solver uses them in the upcoming step.
        wp.launch(
            update_joint_target_trajectory_kernel,
            dim=WORLD_COUNT,
            inputs=[joint_target_trajectory_wp, time_step_wp, SIM_DT * CONTROL_SPEED],
            outputs=[ctrl],
            device=device,
        )
        articulation_view.set_attribute("joint_target_pos", coupler.newton_control, ctrl)

        # Advance Newton (clear forces → collide → step → swap states).
        coupler.step()

    sim_time += FRAME_DT

    # ── Visualization ──────────────────────────────────────────────────────
    if _vis_available:
        vis.begin_frame(sim_time)
        vis.log_state(coupler.newton_state_0)
        vis.end_frame()

        if _movie_writer is not None:
            _movie_writer.append_data(vis.get_frame().numpy())

    # ── Console status (every 50 frames) ──────────────────────────────────
    if (frame + 1) % 50 == 0 or frame == 0:
        print(f"  frame {frame + 1:>4}/{NUM_FRAMES}  sim_time={sim_time:.3f} s")

print()

# ─── Finalize ─────────────────────────────────────────────────────────────
if _movie_writer is not None:
    _movie_writer.close()
    print(f"[Movie] Saved simulation recording to '{MOVIE_OUTPUT_PATH}'.\n")

if vis is not None:
    vis.close()

print("[Coupler] Finalizing NewtonXLBDEMCoupler ...")
coupler.finalize()
print("[Coupler] NewtonXLBDEMCoupler finalized.\n")

print("Demo completed successfully.")
print(f"  Newton version : {newton.__version__}")
print(f"  Warp  version  : {wp.__version__}")
