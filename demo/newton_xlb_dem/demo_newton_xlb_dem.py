"""demo/newton_xlb_dem/demo_newton_xlb_dem.py

Three-way co-simulation demo: Newton (walking robot) + XLB (LBM fluid) + DEM-Engine (particles).

This demo exercises mophi.NewtonXLBDEMCoupler, which manages all three solvers:

  • Newton   — drives an articulated quadruped walking robot using position-based
               dynamics (XPBD).  This is the active physics component: the robot
               walks forward under a sinusoidal trot gait, and its spatial
               representation (body transforms) is extracted at every step.
  • XLB      — a placeholder LBM fluid solver.  The solver is instantiated (if
               the xlb package is available) but no serious fluid physics runs
               during Step() yet.  Future work will use the robot's body transforms
               as a moving boundary condition.
  • DEM-Engine — a placeholder discrete-element solver.  A deme::DEMSolver is
               created on the C++ side but its simulation is not advanced yet.
               Future work will introduce particle–robot coupling.

Both XLB and DEM-Engine are gracefully skipped when their packages / libraries
are not available, so the demo can run Newton-only on a CUDA machine that has
only Newton and Warp installed.

Prerequisites
-------------
  • Build MoPhi with -DMOPHI_BUILD_NEWTON_XLB_DEM=ON (fetches and builds
    DEM-Engine, compiles NewtonXLBDEMCoupler, builds the mophi_core Python
    extension module).
  • pip install newton warp-lang    (Newton rigid-body physics engine)
  • pip install xlb                  (optional — XLB LBM solver)

Running
-------
From the MoPhi build tree (after ``cmake --build``):

    python demo/newton_xlb_dem/demo_newton_xlb_dem.py

Or from the repository root after installing the mophi package:

    python -m demo.newton_xlb_dem.demo_newton_xlb_dem
"""

import math
import sys

# ─── 1. Import MoPhi ─────────────────────────────────────────────────────────
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

# ─── 2. Import Newton + Warp ──────────────────────────────────────────────────
try:
    import newton
    import warp as wp

    _newton_available = True
except ImportError:
    _newton_available = False
    print(
        "WARNING: Newton (or warp) is not installed.  "
        "The demo cannot proceed without Newton.\n"
        "         Install with:  pip install newton warp-lang"
    )
    sys.exit(1)

# ─── 3. Import XLB (optional) ─────────────────────────────────────────────────
try:
    import xlb

    _xlb_available = True
except ImportError:
    _xlb_available = False
    print(
        "INFO: XLB is not installed — the XLB solver will be skipped (placeholder only).\n"
        "      Install with:  pip install xlb"
    )

print("=== MoPhi Newton + XLB + DEM-Engine three-way co-simulation demo ===\n")

# ─── 4. Initialize Warp ───────────────────────────────────────────────────────
wp.init()

# ─── 5. Build the Newton walking-robot scene ──────────────────────────────────
# We construct a quadruped robot using Newton's ModelBuilder.  The robot has:
#   • A torso (box, 0.6 × 0.3 × 0.4 m) connected to the world via a free joint.
#   • Four legs (front-left, front-right, rear-left, rear-right), each with:
#       – An upper-leg segment connected to the torso via a revolute hip joint
#         (rotation about the sagittal z-axis for forward/backward swing).
#       – A lower-leg segment connected to the upper leg via a revolute knee
#         joint (rotation about the same z-axis).
#   • A ground plane for contact.
#
# Joint motors use position control: target_ke (stiffness) and target_kd
# (damping) cause the joint to track the control target angle.  The demo
# updates the control targets with sinusoidal trot-gait signals before each
# coupler.step() call.

print("[Newton] Building quadruped walking-robot model ...")

builder = newton.ModelBuilder()

# ── Torso (floating base) ─────────────────────────────────────────────────────
torso = builder.add_link()
builder.add_shape_box(torso, hx=0.3, hy=0.15, hz=0.2)

# Free joint: lets the torso translate and rotate freely relative to the world.
# The initial position places the robot 0.55 m above the ground so the legs
# clear the floor at the default pose.
torso_joint = builder.add_joint_free(
    parent=-1,
    child=torso,
    parent_xform=wp.transform(p=wp.vec3(0.0, 0.55, 0.0), q=wp.quat_identity()),
    child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
)

# ── Legs ──────────────────────────────────────────────────────────────────────
# Attachment offsets from the torso centre to the hip joint, and trot-gait
# phases (front-left ↔ rear-right in phase; front-right ↔ rear-left in phase).
LEG_LABELS = ["front_left", "front_right", "rear_left", "rear_right"]
LEG_OFFSETS = [
    (0.25, -0.12, 0.18),   # front-left  hip attachment
    (0.25, -0.12, -0.18),  # front-right hip attachment
    (-0.25, -0.12, 0.18),  # rear-left   hip attachment
    (-0.25, -0.12, -0.18), # rear-right  hip attachment
]
# Trot gait: diagonal pairs (FL, RR) share phase 0; (FR, RL) share phase π.
LEG_PHASES = [0.0, math.pi, math.pi, 0.0]

hip_joint_indices = []
knee_joint_indices = []
all_joints = [torso_joint]

for leg_idx, (label, (ox, oy, oz), phase) in enumerate(
    zip(LEG_LABELS, LEG_OFFSETS, LEG_PHASES)
):
    upper_leg = builder.add_link()
    lower_leg = builder.add_link()

    # Upper-leg shape: a narrow box oriented along the leg's long axis (y).
    builder.add_shape_box(upper_leg, hx=0.04, hy=0.12, hz=0.04)
    # Lower-leg shape: slightly narrower.
    builder.add_shape_box(lower_leg, hx=0.035, hy=0.12, hz=0.035)

    # Hip joint — revolute about the z-axis (sagittal plane swing).
    # parent_xform: hip attachment point on the torso.
    # child_xform: top of the upper-leg segment (local +y = up toward hip).
    hip = builder.add_joint_revolute(
        parent=torso,
        child=upper_leg,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(
            p=wp.vec3(ox, oy, oz), q=wp.quat_identity()
        ),
        child_xform=wp.transform(
            p=wp.vec3(0.0, 0.12, 0.0), q=wp.quat_identity()
        ),
        target_ke=800.0,
        target_kd=40.0,
        limit_lower=-0.7,
        limit_upper=0.7,
    )

    # Knee joint — revolute about the z-axis (knee flexion/extension).
    # parent_xform: bottom of upper leg.
    # child_xform: top of lower leg.
    knee = builder.add_joint_revolute(
        parent=upper_leg,
        child=lower_leg,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(
            p=wp.vec3(0.0, -0.12, 0.0), q=wp.quat_identity()
        ),
        child_xform=wp.transform(
            p=wp.vec3(0.0, 0.12, 0.0), q=wp.quat_identity()
        ),
        target_ke=600.0,
        target_kd=30.0,
        limit_lower=-1.2,
        limit_upper=0.1,
    )

    hip_joint_indices.append(hip)
    knee_joint_indices.append(knee)
    all_joints.extend([hip, knee])

# Group all joints into a single articulation (one connected kinematic tree).
builder.add_articulation(all_joints, label="quadruped")
builder.add_ground_plane()

newton_model = builder.finalize()
newton_solver = newton.solvers.SolverXPBD(newton_model)

print(
    f"[Newton] Quadruped model built: {newton_model.body_count} bodies, "
    f"{len(hip_joint_indices)} hip joints, {len(knee_joint_indices)} knee joints.\n"
)

# ─── 6. Build the XLB placeholder scene (if XLB is available) ─────────────────
xlb_simulation = None

if _xlb_available:
    print("[XLB] Setting up placeholder LBM simulation ...")
    try:
        # XLB 0.2+ API: create a minimal 3-D LBM simulation.
        # The grid is intentionally tiny (32³) for a placeholder — real coupling
        # will size the domain to encompass the robot's workspace.
        xlb_simulation = xlb.Simulation(
            omega=1.0,
            grid_shape=(32, 32, 32),
            velocity_set=xlb.velocity_sets.D3Q19(),
            backend=xlb.backends.JAX(),
        )
        print("[XLB] Placeholder LBM simulation created (32×32×32 grid).\n")
    except Exception as exc:
        print(f"[XLB] Could not create XLB simulation ({exc}) — skipping XLB.\n")
        xlb_simulation = None

# ─── 7. Initialize the coupler ────────────────────────────────────────────────
SIM_DT = 1.0 / 500.0  # 2 ms co-simulation time step

coupler = mophi.NewtonXLBDEMCoupler()

print("[Coupler] Initializing NewtonXLBDEMCoupler ...")
coupler.initialize(
    newton_model=newton_model,
    newton_solver=newton_solver,
    xlb_simulation=xlb_simulation,
    sim_dt=SIM_DT,
    num_gpus=1,
)
print("[Coupler] NewtonXLBDEMCoupler initialized.\n")

# ─── 8. Walking control helper ────────────────────────────────────────────────
WALK_FREQ = 1.5   # Gait cycle frequency [Hz]
HIP_AMP   = 0.45  # Hip swing amplitude [rad]
KNEE_AMP  = 0.40  # Knee flexion amplitude [rad]


def update_walking_control(control, step: int, dt: float) -> None:
    """Apply sinusoidal trot-gait targets to the hip and knee joints.

    The trot gait drives diagonal leg pairs in phase:
      front-left  + rear-right:  phase = 0
      front-right + rear-left:   phase = π

    Hip joints swing forward/back (sinusoidal).
    Knee joints flex on the swing phase (cosine-based, always < 0).
    """
    t = step * dt

    # Read the current control targets from GPU, modify, and write back.
    targets = control.joint_target.numpy().copy()

    for i, (hip_j, knee_j, phase) in enumerate(
        zip(hip_joint_indices, knee_joint_indices, LEG_PHASES)
    ):
        hip_angle  = HIP_AMP  * math.sin(2.0 * math.pi * WALK_FREQ * t + phase)
        knee_angle = -KNEE_AMP * (1.0 + math.cos(2.0 * math.pi * WALK_FREQ * t + phase)) / 2.0
        targets[hip_j]  = hip_angle
        targets[knee_j] = knee_angle

    control.joint_target.assign(
        wp.from_numpy(targets, dtype=wp.float32, device="cuda")
    )


# ─── 9. Co-simulation loop ────────────────────────────────────────────────────
NUM_STEPS = 20

print(f"Running {NUM_STEPS} co-simulation step(s) (dt={SIM_DT*1000:.1f} ms each) ...\n")
print(f"{'Step':>5}  {'Torso X [m]':>12}  {'Torso Y [m]':>12}  {'Torso Z [m]':>12}  Representation")
print("-" * 72)

for i in range(NUM_STEPS):
    # Update joint control targets to drive the walking gait.
    update_walking_control(coupler.newton_control, i, SIM_DT)

    # Advance all three solvers by one co-simulation step.
    coupler.step()

    # Extract the robot's spatial representation: one [px, py, pz, qx, qy, qz, qw]
    # per Newton body (torso + 8 leg segments = 9 entries for this quadruped).
    transforms = coupler.get_robot_body_transforms()

    if transforms:
        # The torso body is the first body added (index 0).
        torso_transform = transforms[0]
        px, py, pz = torso_transform[0], torso_transform[1], torso_transform[2]
        qx, qy, qz, qw = (
            torso_transform[3],
            torso_transform[4],
            torso_transform[5],
            torso_transform[6],
        )
        print(
            f"{i + 1:>5}  {px:>12.4f}  {py:>12.4f}  {pz:>12.4f}  "
            f"[{newton_model.body_count} bodies, "
            f"torso quat=({qx:.3f},{qy:.3f},{qz:.3f},{qw:.3f})]"
        )
    else:
        print(f"{i + 1:>5}  (Newton not available — no transforms)")

print()

# ─── 10. Finalize ─────────────────────────────────────────────────────────────
print("[Coupler] Finalizing NewtonXLBDEMCoupler ...")
coupler.finalize()
print("[Coupler] NewtonXLBDEMCoupler finalized.\n")

print("Demo completed successfully.")
print(f"  Newton version : {newton.__version__}")
print(f"  Warp   version : {wp.__version__}")
if _xlb_available:
    print(f"  XLB    version : {xlb.__version__}")
else:
    print("  XLB            : not installed (placeholder skipped)")
print(
    "\nSpatial representation summary:\n"
    "  Each step produced a list of body transforms "
    f"({newton_model.body_count} bodies × 7 values each).\n"
    "  Format: [px, py, pz, qx, qy, qz, qw]  (position [m] + quaternion).\n"
    "  Future work: feed these transforms to DEM-Engine and XLB for full coupling."
)
