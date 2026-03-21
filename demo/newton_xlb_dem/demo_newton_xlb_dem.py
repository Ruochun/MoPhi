"""demo/newton_xlb_dem/demo_newton_xlb_dem.py

Three-way co-simulation demo: Newton (ANYmal C walking robot) + XLB (LBM fluid) + DEM-Engine (particles).

This demo exercises mophi.NewtonXLBDEMCoupler, which manages all three solvers:

  • Newton   — drives the ANYmal C quadruped walking robot using MuJoCo-based
               physics (SolverMuJoCo) controlled by a pre-trained reinforcement
               learning walking policy.  The robot walks forward on a flat ground
               plane, and its spatial representation (body transforms) is extracted
               at every step.
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

The robot setup and walking policy exactly follow Newton's
``newton/examples/robot/example_robot_anymal_c_walk.py``.  The only difference
is that this demo uses a flat ground plane (no procedural terrain) and wraps
all physics inside the MoPhi three-way co-simulation coupler.

Prerequisites
-------------
  • Build MoPhi with -DMOPHI_BUILD_NEWTON_XLB_DEM=ON (fetches and builds
    DEM-Engine, compiles NewtonXLBDEMCoupler, builds the mophi_core Python
    extension module).
  • pip install newton warp-lang torch   (Newton rigid-body physics + PyTorch for RL policy)
  • pip install xlb                       (optional — XLB LBM solver)

Running
-------
From the MoPhi build tree (after ``cmake --build``):

    python demo/newton_xlb_dem/demo_newton_xlb_dem.py

Or from the repository root after installing the mophi package:

    python -m demo.newton_xlb_dem.demo_newton_xlb_dem
"""

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

# ─── 2. Import Newton + Warp + PyTorch ────────────────────────────────────────
try:
    import torch
    import newton
    import warp as wp

    _newton_available = True
    from newton import GeoType
except ImportError:
    _newton_available = False
    print(
        "WARNING: Newton, warp, or torch is not installed.  "
        "The demo cannot proceed without Newton.\n"
        "         Install with:  pip install newton warp-lang torch"
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

print("=== MoPhi Newton (ANYmal C) + XLB + DEM-Engine three-way co-simulation demo ===\n")

# ─── 4. Joint-index remapping ─────────────────────────────────────────────────
# The ANYmal C RL policy was trained with legs ordered [LF, RF, LH, RH] × [HAA, HFE, KFE]
# ("lab" convention), while MuJoCo/Newton uses a different internal ordering.
# These index arrays reorder the 12 joint outputs so they apply to the correct actuators.
# Mirrors newton/examples/robot/example_robot_anymal_c_walk.py.
lab_to_mujoco = [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11]
mujoco_to_lab = [0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11]


# ─── 5. Policy observation helpers ───────────────────────────────────────────
@torch.jit.script
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse of a quaternion (last dimension is [x,y,z,w])."""
    q_w = q[..., 3]
    q_vec = q[..., :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    if q_vec.dim() == 2:
        c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    else:
        c = q_vec * torch.einsum("...i,...i->...", q_vec, v).unsqueeze(-1) * 2.0
    return a - b + c


def compute_obs(actions, state, joint_pos_initial, torch_device, indices, gravity_vec, command):
    """Compute the 48-D observation vector required by the ANYmal C walking policy.

    Mirrors newton/examples/robot/example_robot_anymal_c_walk.py::compute_obs().
    The observation concatenates: base linear velocity (body frame), base angular
    velocity (body frame), projected gravity, velocity command, joint position
    error (lab order), joint velocity (lab order), previous actions.
    """
    root_quat_w = torch.tensor(state.joint_q[3:7], device=torch_device, dtype=torch.float32).unsqueeze(0)
    root_lin_vel_w = torch.tensor(state.joint_qd[:3], device=torch_device, dtype=torch.float32).unsqueeze(0)
    root_ang_vel_w = torch.tensor(state.joint_qd[3:6], device=torch_device, dtype=torch.float32).unsqueeze(0)
    joint_pos_current = torch.tensor(state.joint_q[7:], device=torch_device, dtype=torch.float32).unsqueeze(0)
    joint_vel_current = torch.tensor(state.joint_qd[6:], device=torch_device, dtype=torch.float32).unsqueeze(0)
    vel_b = quat_rotate_inverse(root_quat_w, root_lin_vel_w)
    a_vel_b = quat_rotate_inverse(root_quat_w, root_ang_vel_w)
    grav = quat_rotate_inverse(root_quat_w, gravity_vec)
    joint_pos_rel = joint_pos_current - joint_pos_initial
    joint_vel_rel = joint_vel_current
    rearranged_joint_pos_rel = torch.index_select(joint_pos_rel, 1, indices)
    rearranged_joint_vel_rel = torch.index_select(joint_vel_rel, 1, indices)
    obs = torch.cat([vel_b, a_vel_b, grav, command, rearranged_joint_pos_rel, rearranged_joint_vel_rel, actions], dim=1)
    return obs


# ─── 6. Initialize Warp ───────────────────────────────────────────────────────
wp.init()
torch_device = wp.device_to_torch(wp.get_device())

# ─── 7. Load the ANYmal C robot model ─────────────────────────────────────────
# newton.utils.download_asset("anybotics_anymal_c") downloads the ANYmal C URDF
# and pre-trained RL walking policy from the Newton Assets repository.
# The robot is placed at z = 0.62 m above the flat ground plane (z-up, x = forward).
print("[Newton] Downloading ANYmal C robot assets ...")
asset_path = newton.utils.download_asset("anybotics_anymal_c")
urdf_path = str(asset_path / "urdf" / "anymal.urdf")
policy_path = str(asset_path / "rl_policies" / "anymal_walking_policy_physx.pt")
print(f"[Newton] Assets ready: {asset_path}\n")

print("[Newton] Building ANYmal C model ...")
builder = newton.ModelBuilder()
newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

# Joint and shape defaults matching Newton's anymal example.
builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
    armature=0.06,
    limit_ke=1.0e3,
    limit_kd=1.0e1,
)
builder.default_shape_cfg.ke = 5.0e4
builder.default_shape_cfg.kd = 5.0e2
builder.default_shape_cfg.kf = 1.0e3
builder.default_shape_cfg.mu = 0.75

builder.add_urdf(
    urdf_path,
    xform=wp.transform(
        wp.vec3(0.0, 0.0, 0.62),
        wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5),
    ),
    floating=True,
    enable_self_collisions=False,
    collapse_fixed_joints=True,
    ignore_inertial_definitions=False,
)

# Enlarge foot collision spheres for walking stability on the ground plane.
# Doubling the URDF's small sphere radii prevents the robot from stumbling.
for i in range(len(builder.shape_type)):
    if builder.shape_type[i] == GeoType.SPHERE:
        r = builder.shape_scale[i][0]
        builder.shape_scale[i] = (r * 2.0, 0.0, 0.0)

# Flat ground plane only — no procedural terrain.
builder.add_ground_plane()

# Set initial joint positions to a stable standing pose (from the ANYmal C example).
initial_q = {
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
# builder.joint_q indices: first 6 are the free-joint (position + quaternion),
# then each revolute joint's DOF follows in declaration order.
# Build a name→index mapping once to avoid repeated linear scans.
joint_name_to_idx = {lbl.split("/")[-1]: i for i, lbl in enumerate(builder.joint_label)}
for name, value in initial_q.items():
    idx = joint_name_to_idx.get(name)
    if idx is None:
        raise ValueError(f"Joint '{name}' not found in builder.joint_label")
    builder.joint_q[idx + 6] = value

# Position and velocity control gains.
for i in range(len(builder.joint_target_ke)):
    builder.joint_target_ke[i] = 150
    builder.joint_target_kd[i] = 5

newton_model = builder.finalize()
newton_solver = newton.solvers.SolverMuJoCo(
    newton_model,
    use_mujoco_contacts=False,
    solver="newton",
    ls_parallel=False,
    ls_iterations=50,
    njmax=50,
    nconmax=100,
)

print(
    f"[Newton] ANYmal C model built: {newton_model.body_count} bodies, "
    f"{newton_model.joint_count} joints.\n"
)

# ─── 8. Build the XLB placeholder scene (if XLB is available) ─────────────────
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

# ─── 9. Initialize the coupler ────────────────────────────────────────────────
# sim_dt = 1/200 → 5 ms substep (4 substeps per 50 Hz policy frame,
# matching the inner time step from Newton's anymal example).
SIM_DT = 1.0 / 200.0

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

# ─── 10. Load the ANYmal C walking policy ────────────────────────────────────
print("[Policy] Loading ANYmal C walking policy ...")
policy = torch.jit.load(policy_path, map_location=torch_device)

# Initial joint positions (from the state after coupler initialization and FK eval).
# joint_q[0:7] = free-joint pose (position + quaternion); joint_q[7:] = revolute DOFs.
joint_pos_initial = torch.tensor(
    coupler.newton_state_0.joint_q[7:], device=torch_device, dtype=torch.float32
).unsqueeze(0)

act = torch.zeros(1, 12, device=torch_device, dtype=torch.float32)
lab_to_mujoco_indices = torch.tensor(lab_to_mujoco, device=torch_device)
mujoco_to_lab_indices = torch.tensor(mujoco_to_lab, device=torch_device)
gravity_vec = torch.tensor([[0.0, 0.0, -1.0]], device=torch_device, dtype=torch.float32)
command = torch.zeros((1, 3), device=torch_device, dtype=torch.float32)
command[0, 0] = 1.0  # walk forward (x-direction)

print("[Policy] ANYmal C walking policy loaded.\n")

# ─── 11. Set up Newton's OpenGL visualization window ─────────────────────────
# ViewerGL opens a real-time OpenGL window.  The viewer is non-blocking in the
# render path: begin_frame() / log_state() / end_frame() update the display each
# frame while the simulation continues to advance.  The window can be closed by
# the user at any time; viewer.is_running() returns False once dismissed.
try:
    viewer = newton.viewer.ViewerGL()
    viewer.set_model(newton_model)
    _viewer_available = True
    print("[Viewer] Newton OpenGL visualization window opened.\n")
except Exception as exc:
    viewer = None
    _viewer_available = False
    print(
        f"[Viewer] Could not open Newton OpenGL viewer ({exc}).\n"
        "         The simulation will run without visualization."
    )

# ─── 12. Co-simulation loop ───────────────────────────────────────────────────
# The ANYmal C walking policy runs at 50 Hz (one inference per frame).
# Each frame advances SIM_SUBSTEPS × SIM_DT seconds of physics, matching the
# 4-substep inner loop in Newton's anymal example (frame_dt = 1/50, sim_dt = 1/200).
SIM_SUBSTEPS = 4          # physics substeps per policy frame
FRAME_DT = SIM_DT * SIM_SUBSTEPS  # = 1/50 s — policy control rate
NUM_FRAMES = 250          # ≈ 5 s at 50 Hz (or until the viewer is closed)
sim_time = 0.0

print(
    f"Running up to {NUM_FRAMES} policy frame(s) "
    f"(frame_dt={FRAME_DT * 1000:.1f} ms, {SIM_SUBSTEPS} substeps × {SIM_DT * 1000:.1f} ms) ...\n"
)
print(f"{'Frame':>5}  {'Base X [m]':>12}  {'Base Y [m]':>12}  {'Base Z [m]':>12}  Representation")
print("-" * 72)

# Pre-allocate the 6-element zeros buffer used to prepend free-joint DOFs each frame.
free_joint_zeros = torch.zeros(6, device=torch_device, dtype=torch.float32)

for frame in range(NUM_FRAMES):
    # Stop early if the viewer window has been closed by the user.
    if _viewer_available and not viewer.is_running():
        print(f"\n[Viewer] Window closed by user after frame {frame}.")
        break

    # ── Policy inference: compute observation and get joint targets ──────────
    obs = compute_obs(
        act,
        coupler.newton_state_0,
        joint_pos_initial,
        torch_device,
        lab_to_mujoco_indices,
        gravity_vec,
        command,
    )
    with torch.no_grad():
        act = policy(obs)
        rearranged_act = torch.gather(act, 1, mujoco_to_lab_indices.unsqueeze(0))
        a = joint_pos_initial + 0.5 * rearranged_act
        # Prepend 6 free-joint DOFs (3 linear + 3 angular velocity targets; not actuated).
        a_with_zeros = torch.cat([free_joint_zeros, a.squeeze(0)])
        a_wp = wp.from_torch(a_with_zeros, dtype=wp.float32, requires_grad=False)
        wp.copy(coupler.newton_control.joint_target_pos, a_wp)

    # ── Physics substeps ──────────────────────────────────────────────────────
    for _ in range(SIM_SUBSTEPS):
        coupler.step()
    sim_time += FRAME_DT

    # ── Visualization ─────────────────────────────────────────────────────────
    if _viewer_available:
        viewer.begin_frame(sim_time)
        viewer.log_state(coupler.newton_state_0)
        viewer.end_frame()

    # ── Console status (every 25 frames) ─────────────────────────────────────
    if (frame + 1) % 25 == 0 or frame == 0:
        transforms = coupler.get_robot_body_transforms()
        if transforms:
            base = transforms[0]
            px, py, pz = base[0], base[1], base[2]
            qx, qy, qz, qw = base[3], base[4], base[5], base[6]
            print(
                f"{frame + 1:>5}  {px:>12.4f}  {py:>12.4f}  {pz:>12.4f}  "
                f"[{newton_model.body_count} bodies, "
                f"base quat=({qx:.3f},{qy:.3f},{qz:.3f},{qw:.3f})]"
            )
        else:
            print(f"{frame + 1:>5}  (Newton not available — no transforms)")

print()

# ─── 13. Finalize ─────────────────────────────────────────────────────────────
if _viewer_available:
    viewer.close()

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
    "  Each frame produced a list of body transforms "
    f"({newton_model.body_count} bodies × 7 values each).\n"
    "  Format: [px, py, pz, qx, qy, qz, qw]  (position [m] + quaternion).\n"
    "  Future work: feed these transforms to DEM-Engine and XLB for full coupling."
)

