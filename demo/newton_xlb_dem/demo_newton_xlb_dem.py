"""demo/newton_xlb_dem/demo_newton_xlb_dem.py

Three-way co-simulation demo: Newton (ANYmal C walking robot) + XLB (LBM fluid) + DEME (particles).

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
  • DEME     — a placeholder discrete-element solver (pip install deme).  A
               deme.DEMSolver is created in Python and passed to the coupler, but
               its simulation is not advanced yet.  Future work will introduce
               particle–robot coupling.

Both XLB and DEME are gracefully skipped when their packages are not available,
so the demo can run Newton-only on a CUDA machine that has only Newton and Warp
installed.

The robot setup and walking policy exactly follow Newton's
``newton/examples/robot/example_robot_anymal_c_walk.py``.  The only difference
is that this demo uses a flat ground plane (no procedural terrain) and wraps
all physics inside the MoPhi three-way co-simulation coupler.

Prerequisites
-------------
  • Build MoPhi with -DMOPHI_BUILD_NEWTON_XLB_DEM=ON (compiles
    NewtonXLBDEMCoupler and builds the mophi_core Python extension module).
  • pip install newton warp-lang torch   (Newton rigid-body physics + PyTorch for RL policy)
  • pip install xlb                       (optional — XLB LBM solver)
  • pip install deme                      (optional — DEME discrete-element solver)

Running
-------
From the MoPhi build tree (after ``cmake --build``):

    python demo/newton_xlb_dem/demo_newton_xlb_dem.py

Or from the repository root after installing the mophi package:

    python -m demo.newton_xlb_dem.demo_newton_xlb_dem
"""

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

# ─── Import Newton + Warp + PyTorch ────────────────────────────────────────
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

# ─── Import XLB (optional) ─────────────────────────────────────────────────
try:
    import xlb

    _xlb_available = True
except ImportError:
    _xlb_available = False
    print(
        "INFO: XLB is not installed — the XLB solver will be skipped (placeholder only).\n"
        "      Install with:  pip install xlb"
    )

# ─── Import DEME (optional) ────────────────────────────────────────────────
try:
    import DEME

    _deme_available = True
except ImportError:
    _deme_available = False
    print(
        "INFO: DEME is not installed — the DEME solver will be skipped (placeholder only).\n"
        "      Install with:  pip install deme"
    )

print("=== MoPhi Newton (ANYmal C) + XLB + DEME three-way co-simulation demo ===\n")

# ─── Joint-index remapping ─────────────────────────────────────────────────
# The ANYmal C RL policy was trained with legs ordered [LF, RF, LH, RH] × [HAA, HFE, KFE]
# ("lab" convention), while MuJoCo/Newton uses a different internal ordering.
# These index arrays reorder the 12 joint outputs so they apply to the correct actuators.
# Mirrors newton/examples/robot/example_robot_anymal_c_walk.py.
lab_to_mujoco = [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11]
mujoco_to_lab = [0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11]


# ─── Policy observation helpers ───────────────────────────────────────────
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


# ─── Initialize Warp ───────────────────────────────────────────────────────
wp.init()
torch_device = wp.device_to_torch(wp.get_device())

# ─── Load the ANYmal C robot model ─────────────────────────────────────────
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
# While scanning, also record each sphere's body index and local offset so that
# we can build the foot-tip contact proxy descriptors after finalization.
# Maps builder body-index → {"local_offset": [x,y,z], "sphere_radius": float}
builder_foot_spheres: dict = {}
for i in range(len(builder.shape_type)):
    if builder.shape_type[i] == GeoType.SPHERE:
        r = builder.shape_scale[i][0]
        builder.shape_scale[i] = (r * 2.0, 0.0, 0.0)
        # shape_transform[i].p is the sphere-centre offset in the body's local frame.
        local_p = builder.shape_transform[i].p
        builder_foot_spheres[builder.shape_body[i]] = {
            "local_offset": [float(local_p[0]), float(local_p[1]), float(local_p[2])],
            "sphere_radius": float(r * 2.0),
        }

# Save builder body-name→index mapping before finalize() is called.
builder_body_name_to_idx = {lbl.split("/")[-1]: i for i, lbl in enumerate(builder.body_label)}

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

print(f"[Newton] ANYmal C model built: {newton_model.body_count} bodies, " f"{newton_model.joint_count} joints.\n")

# ─── Identify foot-tip contact proxies ───────────────────────────────────
# ANYmal C has a GeoType.SPHERE collision shape at the distal end of each SHANK
# link.  These spheres are the ground-contact proxies and will serve as coupling
# surfaces for DEM particles and XLB fluid boundaries in future co-sim work.
#
# Geometry representation:
#   Newton represents each foot tip as a sphere shape attached to the SHANK body.
#   The contact proxy is fully described by:
#     • sphere centre in world space: body_pos + rotate(body_quat, local_offset)
#     • sphere radius [m]
#   This is the minimal geometry needed for both particle (DEM) contact detection
#   and fluid (LBM) immersed-boundary coupling.
#
# foot_tip_descriptors — static list (4 entries, [LF, RF, LH, RH]) of dicts:
#   "label"         : str   — short body name, e.g. "LF_SHANK"
#   "body_idx"      : int   — row index into the body_q / body_qd arrays
#   "local_offset"  : list  — sphere centre offset in the shank's body frame [m]
#   "sphere_radius" : float — sphere radius [m]  (constant throughout simulation)
#
# foot_tip_sphere_radii — list[float], per-foot radii in [LF, RF, LH, RH] order.
FOOT_SHANK_NAMES = ["LF_SHANK", "RF_SHANK", "LH_SHANK", "RH_SHANK"]
foot_tip_descriptors = []
for shank_name in FOOT_SHANK_NAMES:
    b_idx = builder_body_name_to_idx.get(shank_name)
    if b_idx is None:
        raise RuntimeError(f"[FootTip] Shank body '{shank_name}' not found in builder body labels")
    sphere_info = builder_foot_spheres.get(b_idx)
    if sphere_info is None:
        raise RuntimeError(f"[FootTip] No sphere shape found attached to shank body '{shank_name}'")
    foot_tip_descriptors.append(
        {
            "label": shank_name,
            "body_idx": b_idx,
            "local_offset": sphere_info["local_offset"],
            "sphere_radius": sphere_info["sphere_radius"],
        }
    )

# Per-foot sphere radii are constant throughout the simulation.
foot_tip_sphere_radii = [d["sphere_radius"] for d in foot_tip_descriptors]

print("[FootTip] Leg-tip contact proxies (sphere geometry):")
for d in foot_tip_descriptors:
    lo = d["local_offset"]
    print(
        f"         {d['label']}: body_idx={d['body_idx']}, "
        f"local_offset=[{lo[0]:.4f}, {lo[1]:.4f}, {lo[2]:.4f}] m, "
        f"sphere_radius={d['sphere_radius']:.4f} m"
    )
print()

# ─── Set up Newton's OpenGL visualization window ─────────────────────────
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

# ─── XLB real LBM simulation setup ───────────────────────────────────────────
# Configures a real 3-D incompressible Navier-Stokes LBM simulation using XLB.
# Physical scenario (world coordinates):
#   • Inlet face at y = +2 (world) = y = NY−1 (LBM grid), blowing in −y direction.
#   • Ground wall at z = 0 (world) = z = 0 (LBM grid), no-slip halfway bounce-back.
#   • Outlet at y = −2 (world) = y = 0 (LBM grid), extrapolation outflow.
#   • All other faces are left open (no explicit BC — acceptable for a low-Re demo).
#
# LBM unit convention:
#   Inlet lattice speed   _XLB_INLET_SPEED  [lu/ts], Mach number Ma ≈ 0.035.
#   Kinematic viscosity   _XLB_NU           [lu],    BGK relaxation ω = 1/(3ν+0.5).
_XLB_SCALE = 4
_XLB_NX, _XLB_NY, _XLB_NZ = 32 * _XLB_SCALE, 32 * _XLB_SCALE, 16 * _XLB_SCALE
_XLB_DOMAIN_MIN = np.array([-2.0, -1.0, 0.0], dtype=np.float64)
_XLB_DOMAIN_MAX = np.array([2.0, 3.0, 2.0], dtype=np.float64)
_XLB_INLET_SPEED = 0.02  # LBM inlet speed [lattice units/timestep]; Ma ≈ 0.035
_XLB_NU = 0.01  # LBM kinematic viscosity [lattice units]
_XLB_OMEGA = 1.0 / (3.0 * _XLB_NU + 0.5)  # BGK relaxation parameter
_XLB_WARMUP_STEPS = 600  # LBM steps to advance before the main loop
_XLB_STEPS_PER_FRAME = 4  # LBM steps advanced per Newton co-simulation frame
_XLB_VIS_INTERVAL = 5  # refresh streamline visualisation every N Newton frames

# State variables for the XLB solver (populated during setup below).
xlb_simulation = None  # passed to coupler.initialize() as a reference handle
_xlb_stepper = None
_xlb_f0 = _xlb_f1 = _xlb_bc_mask = _xlb_missing_mask = None
_xlb_macro = None
_xlb_rho_field = _xlb_u_field = None
_xlb_timestep = 0
_xlb_u_np = None  # velocity in (NX, NY, NZ, 3) layout; updated each vis interval

if _xlb_available:
    print("[XLB] Setting up real LBM simulation ...")
    try:
        from xlb.compute_backend import ComputeBackend as _XLBComputeBackend
        from xlb.precision_policy import PrecisionPolicy as _XLBPrecisionPolicy, Precision as _XLBPrecision
        from xlb.grid import grid_factory as _xlb_grid_factory
        from xlb.operator.boundary_condition import (
            ZouHeBC as _ZouHeBC,
            HalfwayBounceBackBC as _HalfwayBounceBackBC,
            ExtrapolationOutflowBC as _ExtrapolationOutflowBC,
        )
        from xlb.operator.stepper import IncompressibleNavierStokesStepper as _NSEStepper
        from xlb.operator.macroscopic import Macroscopic as _XLBMacroscopic

        _xlb_backend = _XLBComputeBackend.WARP
        _xlb_precision = _XLBPrecisionPolicy.FP32FP32
        _xlb_vel_set = xlb.velocity_set.D3Q19(precision_policy=_xlb_precision, compute_backend=_xlb_backend)
        xlb.init(_xlb_vel_set, _xlb_backend, _xlb_precision)

        _xlb_grid = _xlb_grid_factory((_XLB_NX, _XLB_NY, _XLB_NZ), compute_backend=_xlb_backend)

        # Build non-overlapping boundary index sets.
        _box = _xlb_grid.bounding_box_indices()
        _box_no_e = _xlb_grid.bounding_box_indices(remove_edges=True)

        _xlb_inlet_idx = _box_no_e["back"]  # y_max face  –  incoming flow (-y direction)
        _xlb_outlet_idx = _box_no_e["front"]  # y_min face  –  outflow
        _xlb_wall_idx = [
            _box["bottom"][i] + _box["top"][i] + _box["left"][i] + _box["right"][i] for i in range(_xlb_vel_set.d)
        ]
        _xlb_wall_idx = np.unique(np.array(_xlb_wall_idx), axis=-1).tolist()

        # ZouHeBC velocity BC: prescribed_value must have exactly one non-zero element
        # (the normal component).  For the y = NY−1 face the outward normal is +y, so
        # the functional computes u = −prescribed_value × normal = (0, −speed, 0).
        _xlb_inlet_bc = _ZouHeBC(
            bc_type="velocity",
            prescribed_value=np.array([0.0, _XLB_INLET_SPEED, 0.0]),
            indices=_xlb_inlet_idx,
        )
        _xlb_wall_bc = _HalfwayBounceBackBC(indices=_xlb_wall_idx)
        _xlb_outlet_bc = _ExtrapolationOutflowBC(indices=_xlb_outlet_idx)

        _xlb_stepper = _NSEStepper(
            grid=_xlb_grid,
            boundary_conditions=[_xlb_wall_bc, _xlb_inlet_bc, _xlb_outlet_bc],
            collision_type="BGK",
        )
        _xlb_f0, _xlb_f1, _xlb_bc_mask, _xlb_missing_mask = _xlb_stepper.prepare_fields()

        _xlb_macro = _XLBMacroscopic(_xlb_vel_set, _xlb_precision, _xlb_backend)
        _xlb_rho_field = _xlb_grid.create_field(cardinality=1, dtype=_XLBPrecision.FP32)
        _xlb_u_field = _xlb_grid.create_field(cardinality=3, dtype=_XLBPrecision.FP32)

        # Warm-up: advance the LBM toward an initial near-steady state.
        print(f"[XLB] Running {_XLB_WARMUP_STEPS} warm-up steps (ω = {_XLB_OMEGA:.4f}) ...")
        for _ws in range(_XLB_WARMUP_STEPS):
            _xlb_f0, _xlb_f1 = _xlb_stepper(
                _xlb_f0, _xlb_f1, _xlb_bc_mask, _xlb_missing_mask, _XLB_OMEGA, _xlb_timestep
            )
            _xlb_f0, _xlb_f1 = _xlb_f1, _xlb_f0  # double-buffer swap
            _xlb_timestep += 1

        # Extract macro state; XLB stores fields as (Q/D, NX, NY, NZ) — transpose
        # to (NX, NY, NZ, 3) for the streamline helper.
        _xlb_rho_field, _xlb_u_field = _xlb_macro(_xlb_f0, _xlb_rho_field, _xlb_u_field)
        _xlb_u_np = _xlb_u_field.numpy().transpose(1, 2, 3, 0).astype(np.float32)

        xlb_simulation = _xlb_stepper  # coupler stores this as an opaque reference
        print(f"[XLB] LBM simulation ready: {_XLB_NX}×{_XLB_NY}×{_XLB_NZ} grid.\n")
    except Exception as exc:
        print(f"[XLB] Could not set up XLB simulation ({exc}) — disabling XLB.\n")
        _xlb_stepper = None
        xlb_simulation = None


# ─── XLB flow-field visualisation helpers ─────────────────────────────────────
def _xlb_build_streamlines(
    u,
    domain_min,
    domain_max,
    n_x_seeds=6,
    n_z_seeds=4,
    n_steps=40,
    step_size=0.12,
    seed_y=None,
):
    """Trace arc-length-parameterised streamlines through velocity field *u*.

    Seeds a regular (n_x_seeds × n_z_seeds) grid on the inlet face (y ≈ domain_max)
    and advances each streamline with a normalised Euler step until the path exits
    the domain or the local speed stalls.

    Args:
        u:          (Nx, Ny, Nz, 3) float32 velocity field [m/s or lattice units].
        domain_min: (3,) world-space lower corner of the LBM domain [m].
        domain_max: (3,) world-space upper corner of the LBM domain [m].
        n_x_seeds:  number of seed positions along x.
        n_z_seeds:  number of seed positions along z.
        n_steps:    maximum integration steps per streamline.
        step_size:  arc-length step [m] — controls sample density along lines.
        seed_y:     y-coordinate of the seed plane.  Defaults to domain_max[1] − 0.05
                    (just inside the inlet face at y = domain_max, flow in −y direction).

    Returns:
        pts: (N, 3) float32 — world-space positions along all streamlines.
        spd: (N,)   float32 — velocity magnitude at each sample point.
    """
    nx, ny, nz, _ = u.shape
    if seed_y is None:
        # Default: seed just inside the inlet face (y = domain_max, flow in −y direction).
        seed_y = float(domain_max[1]) - 0.05

    def _interp(pos):
        """Trilinear velocity interpolation at world-space position *pos*."""
        t = (pos - domain_min) / (domain_max - domain_min)
        # Clamp so that the lower-cell index stays at most n-2, keeping (index+1)
        # in-bounds.  The 0.001 epsilon prevents reaching the upper-cell face.
        gx = float(np.clip(t[0] * (nx - 1), 0.0, nx - 1.001))
        gy = float(np.clip(t[1] * (ny - 1), 0.0, ny - 1.001))
        gz = float(np.clip(t[2] * (nz - 1), 0.0, nz - 1.001))
        ix, iy, iz = int(gx), int(gy), int(gz)
        fx, fy, fz = gx - ix, gy - iy, gz - iz
        return (
            u[ix, iy, iz] * (1 - fx) * (1 - fy) * (1 - fz)
            + u[ix + 1, iy, iz] * fx * (1 - fy) * (1 - fz)
            + u[ix, iy + 1, iz] * (1 - fx) * fy * (1 - fz)
            + u[ix, iy, iz + 1] * (1 - fx) * (1 - fy) * fz
            + u[ix + 1, iy + 1, iz] * fx * fy * (1 - fz)
            + u[ix + 1, iy, iz + 1] * fx * (1 - fy) * fz
            + u[ix, iy + 1, iz + 1] * (1 - fx) * fy * fz
            + u[ix + 1, iy + 1, iz + 1] * fx * fy * fz
        )

    # Inset seeds 0.4 m from the x-edges and 0.15 m from the z-edges to avoid
    # the low-speed boundary layers where the no-slip profile approaches zero.
    xs = np.linspace(domain_min[0] + 0.4, domain_max[0] - 0.4, n_x_seeds)
    zs = np.linspace(0.15, domain_max[2] - 0.15, n_z_seeds)

    all_pts: list = []
    all_spd: list = []
    for x0 in xs:
        for z0 in zs:
            pos = np.array([x0, seed_y, z0], dtype=np.float64)
            for _ in range(n_steps):
                if not np.all((pos >= domain_min) & (pos <= domain_max)):
                    break
                vel = _interp(pos).astype(np.float64)
                spd = float(np.linalg.norm(vel))
                if spd < 1e-6:  # stall threshold: treat as zero-velocity region
                    break
                all_pts.append(pos.astype(np.float32).copy())
                all_spd.append(spd)
                pos += step_size * vel / spd  # arc-length step

    if not all_pts:
        return np.zeros((1, 3), dtype=np.float32), np.zeros(1, dtype=np.float32)
    return np.array(all_pts, dtype=np.float32), np.array(all_spd, dtype=np.float32)


def _xlb_make_streamline_warp_arrays(pts, spd):
    """Convert streamline sample arrays to Warp arrays for ViewerGL rendering.

    Colours are mapped from slow → cornflower-blue (0.20, 0.55, 0.85) to
    fast → cyan-white (0.55, 0.90, 1.00); this cool palette contrasts with the
    warm-orange DEM particles without occluding the robot or ground plane.
    """
    spd_max = float(spd.max()) if spd.max() > 0 else 1.0
    t = np.clip(spd / spd_max, 0.0, 1.0).astype(np.float32)
    colors_np = np.stack(
        [
            0.20 + 0.35 * t,  # R: 0.20 → 0.55
            0.55 + 0.35 * t,  # G: 0.55 → 0.90
            0.85 + 0.15 * t,  # B: 0.85 → 1.00
        ],
        axis=1,
    ).astype(np.float32)
    pos_wp = wp.array(pts, dtype=wp.vec3)
    radii_wp = wp.array(
        # 0.018 m radius: small enough to not obstruct the robot view, large
        # enough to be clearly visible at camera distances of 4–8 m.
        np.full(len(pts), 0.018, dtype=np.float32),
        dtype=wp.float32,
    )
    colors_wp = wp.array(colors_np, dtype=wp.vec3)
    return pos_wp, radii_wp, colors_wp


# Build initial streamline visualisation from the LBM macro state.
# If XLB is available use the real macro state; otherwise fall back to an
# analytically-constructed approximation (laminar channel flow in −y direction).
if _xlb_u_np is not None:
    _xlb_sl_u = _xlb_u_np
else:
    # Analytic fallback: parabolic z-profile, flow in −y direction.
    _xlb_sl_u = np.zeros((_XLB_NX, _XLB_NY, _XLB_NZ, 3), dtype=np.float32)
    _xlb_sl_z = np.linspace(0.0, 2.0, _XLB_NZ)
    _xlb_sl_profile = 4.0 * (_xlb_sl_z / 2.0) * (1.0 - _xlb_sl_z / 2.0)
    _xlb_sl_u[:, :, :, 1] = -0.8 * _xlb_sl_profile[np.newaxis, np.newaxis, :]  # −y

_xlb_streamline_pts, _xlb_streamline_spd = _xlb_build_streamlines(_xlb_sl_u, _XLB_DOMAIN_MIN, _XLB_DOMAIN_MAX)
print(f"[XLB] Generated {len(_xlb_streamline_pts)} streamline sample point(s) for flow visualisation.\n")

if _viewer_available:
    _xlb_streamline_pos_wp, _xlb_streamline_radii_wp, _xlb_streamline_colors_wp = _xlb_make_streamline_warp_arrays(
        _xlb_streamline_pts, _xlb_streamline_spd
    )
    print(f"[Viewer] XLB flow streamlines registered ({len(_xlb_streamline_pts)} point(s)).\n")

# ─── Build the DEME placeholder solver (if DEME is available) ─────────────
# Spheres are initially scattered ahead of the robot (+y)
# near the ground — resembling a thin layer of dust or dirt on the surface.
# All spheres share a constant velocity in the -y direction (opposite to the
# robot's forward direction), so they appear to drift towards the robot's face.
# Collision and DEM physics will be handled by DEME in a future update.
_NUM_DEM_SPHERES = 750
# Four representative radius types [m] — coarse dust grain size distribution.
_DEM_RADIUS_TYPES = [0.030, 0.045, 0.060, 0.040]

_rng = np.random.default_rng(seed=42)
_radius_indices = _rng.integers(0, len(_DEM_RADIUS_TYPES), size=_NUM_DEM_SPHERES)
_dem_sphere_radii_np = np.array([_DEM_RADIUS_TYPES[i] for i in _radius_indices], dtype=np.float32)

# Initial positions: scattered in a band ahead of the robot along +xy,
# spread laterally across y, and resting on the ground (z = radius).
_x_init = _rng.uniform(-1.5, 1.5, size=_NUM_DEM_SPHERES).astype(np.float32)
_y_init = _rng.uniform(1.0, 4.5, size=_NUM_DEM_SPHERES).astype(np.float32)
_z_init = _rng.uniform(_dem_sphere_radii_np, _dem_sphere_radii_np + 0.25).astype(np.float32)

# _dem_sphere_positions_np shape (N, 3) — updated every frame.
_dem_sphere_positions_np = np.column_stack([_x_init, _y_init, _z_init])

# Velocity in -y direction [m/s] — opposite to robot's forward (+y) direction,
# so the spheres move towards the robot's face.
_DEM_SPHERE_INIT_VELOCITY_Y = [0.0, -1.0, 0.0]  # m/s
_DEM_SPHERE_COLOR = [0.8, 0.4, 0.1]  # orange — DEM particle colour
# sim_dt = 1/200 → 5 ms substep (4 substeps per 50 Hz policy frame,
# matching the inner time step from Newton's anymal example).
SIM_DT = 1.0 / 200.0

if _viewer_available:
    # Radii and colours are constant throughout the simulation; allocate once.
    # The position array (_dem_sphere_pos_wp) is rebuilt from _dem_sphere_positions_np
    # every frame inside the simulation loop after positions are updated.
    _dem_sphere_radii_wp = wp.array(_dem_sphere_radii_np, dtype=wp.float32)
    _dem_sphere_colors_wp = wp.array(
        np.tile(_DEM_SPHERE_COLOR, (_NUM_DEM_SPHERES, 1)).astype(np.float32),
        dtype=wp.vec3,
    )
    print(f"[Viewer] {_NUM_DEM_SPHERES} DEM sphere(s) registered for visualisation.\n")

deme_solver = None
shank_trackers = []
particles_tracker = None
_FIXED_FAM = 10

if _deme_available:
    print("[DEME] Creating placeholder deme.DEMSolver ...")
    deme_solver = DEME.DEMSolver()
    wall_mat = deme_solver.LoadMaterial({"E": 1e6, "nu": 0.3, "mu": 0.3, "CoR": 0.2})
    deme_solver.AddBCPlane([0, 0, 0], [0, 0, 1], wall_mat)
    deme_solver.SetGravitationalAcceleration([0, 0, -9.81])
    deme_solver.SetErrorOutAvgContacts(500)
    # Load the shank
    ad_hoc_pos = [0.0, 0.0, 0.0]
    for i in range(len(foot_tip_sphere_radii)):
        template_shank = deme_solver.LoadSphereType(1.0, foot_tip_sphere_radii[0], wall_mat)
        shank = deme_solver.AddClumps(template_shank, [ad_hoc_pos])
        shank.SetFamily(_FIXED_FAM)
        shank_trackers.append(deme_solver.Track(shank))
    # Fix shanks physics for DEME
    deme_solver.SetFamilyFixed(_FIXED_FAM)
    # Load particles
    particle_templates = []
    for i in range(len(_DEM_RADIUS_TYPES)):
        particle_templates.append(deme_solver.LoadSphereType(1.0, _DEM_RADIUS_TYPES[i], wall_mat))
    used_types = []
    for i in range(_NUM_DEM_SPHERES):
        used_types.append(particle_templates[_radius_indices[i]])
    particles = deme_solver.AddClumps(used_types, _dem_sphere_positions_np)
    # Init vel
    particles.SetVel(_DEM_SPHERE_INIT_VELOCITY_Y)
    particles_tracker = deme_solver.Track(particles)
    # Init
    deme_solver.SetInitTimeStep(SIM_DT)
    deme_solver.Initialize()

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

# ─── Load the ANYmal C walking policy ────────────────────────────────────
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

# ─── Co-simulation loop ───────────────────────────────────────────────────
# The ANYmal C walking policy runs at 50 Hz (one inference per frame).
# Each frame advances SIM_SUBSTEPS × SIM_DT seconds of physics, matching the
# 4-substep inner loop in Newton's anymal example (frame_dt = 1/50, sim_dt = 1/200).
SIM_SUBSTEPS = 4  # physics substeps per policy frame
FRAME_DT = SIM_DT * SIM_SUBSTEPS  # policy control rate
NUM_FRAMES = 250  # ≈ 5 s at 50 Hz (or until the viewer is closed)
sim_time = 0.0

# ─── Movie recording settings ────────────────────────────────────────────
# Set SAVE_MOVIE = True to record the rendered simulation frames to a video file.
# Requires: pip install imageio imageio-ffmpeg
SAVE_MOVIE = True
MOVIE_OUTPUT_PATH = "demo_newton_xlb_dem.mp4"
MOVIE_FPS = 50  # frames per second for the output video

_movie_writer = None
if SAVE_MOVIE and _viewer_available:
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

print(
    f"Running up to {NUM_FRAMES} policy frame(s) "
    f"(frame_dt={FRAME_DT * 1000:.1f} ms, {SIM_SUBSTEPS} substeps × {SIM_DT * 1000:.1f} ms) ...\n"
)
# print(f"{'Frame':>5}  {'Base X [m]':>12}  {'Base Y [m]':>12}  {'Base Z [m]':>12}  Representation")
# print("-" * 72)

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

    # ── Extract foot-tip contact proxy poses ─────────────────────────────────
    # Fetch all body transforms once; reuse for foot extraction and status line.
    #
    # NOTE on GPU→CPU data movement:
    #   get_robot_body_transforms() calls body_q.numpy() internally, which
    #   performs a synchronous GPU→CPU copy when Newton runs on a CUDA device.
    #   This is acceptable for a demo, but in production co-simulation the copy
    #   will become a bottleneck.
    #
    # TODO (GPU-native foot data): Replace get_robot_body_transforms() with a
    #   GPU-resident path that keeps foot_tip_positions and foot_tip_rotations
    #   as wp.array (or torch.Tensor on the GPU) so they can be fed directly
    #   into DEM-Engine particle contact detection and XLB immersed-boundary
    #   kernels without any CPU round-trip.  Candidate approach: expose the
    #   body_q warp array directly and run a small warp.kernel that reads the
    #   four shank rows + local offsets and writes world-space sphere centres
    #   into a pre-allocated wp.array of shape (4, 3).
    #
    # foot_tip_positions : list[4 × [px, py, pz]]
    #   World-space sphere centre for each foot, in [LF, RF, LH, RH] order [m].
    #   Computed as:  body_pos + rotate(body_quat, local_offset)
    #   where local_offset is the sphere-centre offset in the shank body frame.
    #
    # foot_tip_rotations : list[4 × [qx, qy, qz, qw]]
    #   World-space orientation of each shank body (Warp xyzw quaternion).
    #   Needed for asymmetric contact proxies; provided here for completeness.
    #
    # foot_tip_sphere_radii : list[4 × float]  (constant, defined above)
    #   Sphere radius of each foot's contact proxy [m].
    #
    # TODO: pass (foot_tip_positions, foot_tip_rotations, foot_tip_sphere_radii)
    #       into DEM-Engine and XLB once coupling logic is implemented.
    all_transforms = coupler.get_robot_body_transforms()
    foot_tip_positions = []
    foot_tip_rotations = []
    if all_transforms:
        for d in foot_tip_descriptors:
            t = all_transforms[d["body_idx"]]  # [px, py, pz, qx, qy, qz, qw]
            px, py, pz = t[0], t[1], t[2]
            qx, qy, qz, qw = t[3], t[4], t[5], t[6]
            ox, oy, oz = d["local_offset"]
            # Compute world-space sphere centre:
            #   centre_world = body_pos + rotate(body_quat, local_offset)
            #
            # Using the Rodrigues rotation formula for rotate(q, o):
            #   rotate(q, o) = o + 2*qw*(qv × o) + 2*(qv × (qv × o))
            # where qv = (qx, qy, qz) and qw = scalar part.
            # The +o term (unrotated local offset) is part of the formula itself.
            # Combined with body_pos (p), the full world-space centre is:
            #   centre_world = p + o + 2*qw*(qv×o) + 2*(qv×(qv×o))
            cx = qy * oz - qz * oy  # first cross: qv × o
            cy = qz * ox - qx * oz
            cz = qx * oy - qy * ox
            ccx = qy * cz - qz * cy  # second cross: qv × (qv × o)
            ccy = qz * cx - qx * cz
            ccz = qx * cy - qy * cx
            foot_tip_positions.append(
                [
                    px + ox + 2.0 * (qw * cx + ccx),
                    py + oy + 2.0 * (qw * cy + ccy),
                    pz + oz + 2.0 * (qw * cz + ccz),
                ]
            )
            foot_tip_rotations.append([qx, qy, qz, qw])

        # Feed the info to DEME
        if _deme_available:
            for i in range(len(foot_tip_positions)):
                shank_trackers[i].SetPos(foot_tip_positions[i])
                shank_trackers[i].SetOriQ(foot_tip_rotations[i])

    # ── Physics substeps ──────────────────────────────────────────────────────
    for _ in range(SIM_SUBSTEPS):
        # TODO: No whole-sale stepper. Update this later.
        coupler.step()
        deme_solver.DoStepDynamics()

    sim_time += FRAME_DT

    # ── XLB LBM steps ─────────────────────────────────────────────────────────
    # Advance the LBM solver independently of Newton (no coupling yet).
    # Refreshing the streamline visualisation is deferred to every
    # _XLB_VIS_INTERVAL frames to keep per-frame overhead low.
    if _xlb_stepper is not None:
        for _ in range(_XLB_STEPS_PER_FRAME):
            _xlb_f0, _xlb_f1 = _xlb_stepper(
                _xlb_f0, _xlb_f1, _xlb_bc_mask, _xlb_missing_mask, _XLB_OMEGA, _xlb_timestep
            )
            _xlb_f0, _xlb_f1 = _xlb_f1, _xlb_f0  # double-buffer swap
            _xlb_timestep += 1
        if _viewer_available and frame % _XLB_VIS_INTERVAL == 0:
            _xlb_rho_field, _xlb_u_field = _xlb_macro(_xlb_f0, _xlb_rho_field, _xlb_u_field)
            _xlb_u_np = _xlb_u_field.numpy().transpose(1, 2, 3, 0).astype(np.float32)
            _xlb_streamline_pts, _xlb_streamline_spd = _xlb_build_streamlines(
                _xlb_u_np, _XLB_DOMAIN_MIN, _XLB_DOMAIN_MAX
            )
            _xlb_streamline_pos_wp, _xlb_streamline_radii_wp, _xlb_streamline_colors_wp = (
                _xlb_make_streamline_warp_arrays(_xlb_streamline_pts, _xlb_streamline_spd)
            )

    # ── Print foot-tip positions every frame ──────────────────────────────────
    # if foot_tip_positions:
    #     tip_str = "  ".join(
    #         f"{d['label']}=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})"
    #         for d, p in zip(foot_tip_descriptors, foot_tip_positions)
    #     )
    #     print(f"[f{frame + 1:04d}] foot_tip_positions: {tip_str}")

    # ── Visualization ─────────────────────────────────────────────────────────
    if _viewer_available:
        # Get particles positions
        particles_positions = particles_tracker.Positions()
        # print(particles_positions)
        # Advance sphere positions along -y (towards robot) each frame.
        _dem_sphere_positions_np = np.array(particles_positions)
        _dem_sphere_pos_wp = wp.array(_dem_sphere_positions_np.copy(), dtype=wp.vec3)

        # Follow-camera: position the camera behind the robot (−y direction) so
        # the robot and the DEM particles ahead of it (+y) are both in view.
        # The camera tracks the robot's XY position while staying at a fixed
        # height offset and looking forward (+y, yaw=90) with a slight downward
        # pitch to keep the ground plane visible.
        base_pos = coupler.newton_state_0.joint_q.numpy()[:3]  # [x, y, z] world-space base position
        viewer.set_camera(
            pos=wp.vec3(base_pos[0], base_pos[1] - 6.0, base_pos[2] + 2.0),
            pitch=-10.0,
            yaw=90.0,
        )

        viewer.begin_frame(sim_time)
        viewer.log_state(coupler.newton_state_0)
        # Render DEM placeholder spheres moving towards the robot.
        # Future work will replace these with live DEME particle positions.
        viewer.log_points(
            "dem_particles",
            _dem_sphere_pos_wp,
            radii=_dem_sphere_radii_wp,
            colors=_dem_sphere_colors_wp,
        )
        # Render XLB flow field as dot-chain streamlines.
        # Points are traced from the live LBM macro state (updated every
        # _XLB_VIS_INTERVAL frames); the cool-blue colour palette contrasts with
        # the orange DEM particles and does not block the robot geometry or ground plane.
        viewer.log_points(
            "xlb_streamlines",
            _xlb_streamline_pos_wp,
            radii=_xlb_streamline_radii_wp,
            colors=_xlb_streamline_colors_wp,
        )
        viewer.end_frame()

        if _movie_writer is not None:
            _movie_writer.append_data(viewer.get_frame().numpy())

    # ── Console status (every 25 frames) ─────────────────────────────────────
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
