"""demo/newton_xlb_dem/anymal_robot_multiphysics/demo_anymal_robot_newton_baseline.py

Newton-only baseline demo: ANYmal C walking on flat ground while lightly
contacting a few dynamic boxes managed entirely by Newton.

This demo keeps the same walking policy and a similar camera view as the
ANYmal multiphysics demo, but removes the XLB and DEME solvers. The boxes are
intentionally light so the robot can bump them aside without losing its gait,
which makes this a useful baseline for comparing against the multiphysics case.

Prerequisites
-------------
  • Build MoPhi with Python bindings available.
  • pip install --upgrade newton warp-lang mujoco==3.6.0 torch

Running
-------
From the MoPhi build tree (after ``cmake --build``):

    python demo/newton_xlb_dem/anymal_robot_multiphysics/demo_anymal_robot_newton_baseline.py

Or from the repository root after installing the mophi package:

    python -m demo.newton_xlb_dem.anymal_robot_multiphysics.demo_anymal_robot_newton_baseline
"""

import os
import sys
from importlib.util import find_spec

import numpy as np

# ─── Import MoPhi ─────────────────────────────────────────────────────────
if find_spec("mophi") is None:
    sys.exit(
        "ERROR: Could not import the 'mophi' package.\n"
        "       Build MoPhi with Python bindings enabled and ensure that\n"
        "       the build directory (python/) is on PYTHONPATH."
    )
import mophi

# ─── Import Newton + Warp + MuJoCo + PyTorch ─────────────────────────────
if find_spec("torch") is None:
    mophi.fatal("PyTorch is required for this demo.\nInstall with:  pip install torch")
if find_spec("newton") is None or find_spec("warp") is None or find_spec("mujoco") is None:
    mophi.fatal(
        "Newton, Warp, and MuJoCo are required for this demo.\n"
        "Install with:  pip install --upgrade newton warp-lang mujoco==3.6.0"
    )
import mujoco
import newton
import torch
import warp as wp

mophi.check_newton_warp_mujoco_versions(
    mophi.REQUIRED_NEWTON_VERSION,
    mophi.REQUIRED_WARP_VERSION,
    mophi.REQUIRED_MUJOCO_VERSION,
)

# ─── Import demo-specific utilities ──────────────────────────────────────
# newton_xlb_dem_utils.py lives in demo/newton_xlb_dem (parent directory of this script).
# Prepend that parent directory so the module can be found whether the demo is
# run directly (python demo/...) or via -m.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import newton_xlb_dem_utils as demo_utils  # noqa: E402

# ══════════════════════════════════════════════════════════════════════════════
# Simulation configuration
# All user-tunable constants are defined here. Change values in this section;
# do not scatter magic numbers through the setup code below.
# ══════════════════════════════════════════════════════════════════════════════

# ── Timing ────────────────────────────────────────────────────────────────
RENDER_FPS = 50
NEWTON_DT = 1.0 / 200.0
NUM_FRAMES = 250  # ≈ 5 s at 50 Hz

# ── Robot command ─────────────────────────────────────────────────────────
# Command format follows the ANYmal walking-policy observation: [vx, vy, yaw].
FORWARD_COMMAND = 1.0
LATERAL_COMMAND = 0.0
YAW_COMMAND = 0.0

# ── Newton contact boxes ──────────────────────────────────────────────────
# Toggle dynamic obstacle cubes and their contacts with the robot.
ENABLE_DYNAMIC_CONTACT_BOXES = True
# Boxes are intentionally light so the robot can hit them without losing gait.
BOX_HALF_EXTENTS = np.array([0.08, 0.08, 0.12], dtype=np.float32)
BOX_MASS = 0.25
BOX_CONTACT_KE = 5.0e4
BOX_CONTACT_KD = 5.0e2
BOX_POSES = [
    (0.00, 1.05, float(BOX_HALF_EXTENTS[2])),
    (-0.10, 1.55, float(BOX_HALF_EXTENTS[2])),
    (0.12, 2.05, float(BOX_HALF_EXTENTS[2])),
    (-0.04, 2.55, float(BOX_HALF_EXTENTS[2])),
    (0.08, 3.05, float(BOX_HALF_EXTENTS[2])),
]

# ── Visualization & output ────────────────────────────────────────────────
USE_OMNIVERSE_VISUALIZATION = False
SAVE_MOVIE = True
MOVIE_OUTPUT_PATH = "demo_anymal_robot_newton_baseline.mp4"
MOVIE_FPS = RENDER_FPS
OMNIVERSE_OUTPUT_PATH = "demo_anymal_robot_newton_baseline.usdc"

# ── Derived timing constants ──────────────────────────────────────────────
FRAME_DT = 1.0 / RENDER_FPS
SIM_SUBSTEPS = int(round(FRAME_DT / NEWTON_DT))
if not np.isclose(SIM_SUBSTEPS * NEWTON_DT, FRAME_DT, rtol=0.0, atol=1.0e-12):
    raise ValueError(
        "FRAME_DT must be an integer multiple of NEWTON_DT. "
        f"Got FRAME_DT={FRAME_DT:.12g}, NEWTON_DT={NEWTON_DT:.12g}."
    )

# ─── Joint-index remapping ────────────────────────────────────────────────
# The ANYmal C RL policy was trained with legs ordered [LF, RF, LH, RH] ×
# [HAA, HFE, KFE] ("lab" convention), while MuJoCo/Newton uses a different
# internal ordering. These index arrays reorder the 12 joint outputs so they
# apply to the correct actuators.
NUM_POLICY_JOINTS = 12
lab_to_mujoco = [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11]
mujoco_to_lab = [0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11]


def _set_shape_contact_stiffness(builder: newton.ModelBuilder, shape_idx: int, ke: float, kd: float) -> None:
    """Set contact stiffness and damping for one shape when those fields are available."""
    if hasattr(builder, "shape_ke"):
        builder.shape_ke[shape_idx] = float(ke)
    if hasattr(builder, "shape_kd"):
        builder.shape_kd[shape_idx] = float(kd)


# ─── Initialize Warp ──────────────────────────────────────────────────────
wp.init()
torch_device = wp.device_to_torch(wp.get_device())

# ─── Load the ANYmal C robot model ────────────────────────────────────────
# newton.utils.download_asset("anybotics_anymal_c") downloads the ANYmal C URDF
# and pre-trained RL walking policy from the Newton Assets repository.
print("[Newton] Downloading ANYmal C robot assets ...")
asset_path = newton.utils.download_asset("anybotics_anymal_c")
urdf_path = str(asset_path / "urdf" / "anymal.urdf")
policy_path = str(asset_path / "rl_policies" / "anymal_walking_policy_physx.pt")
print(f"[Newton] Assets ready: {asset_path}\n")

print("[Newton] Building ANYmal C baseline model ...")
builder = newton.ModelBuilder()
newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

# Joint and shape defaults matching Newton's ANYmal example.
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

# Enlarge foot collision spheres for stable walking on the flat ground plane.
demo_utils.scan_and_enlarge_foot_spheres(builder)

# Flat ground plane only — matching the walking policy's nominal environment.
builder.add_ground_plane()

# Set initial joint positions to the stable standing pose used by the ANYmal example.
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
joint_name_to_idx = {lbl.split("/")[-1]: i for i, lbl in enumerate(builder.joint_label)}
for name, value in initial_q.items():
    idx = joint_name_to_idx.get(name)
    if idx is None:
        raise ValueError(f"Joint '{name}' not found in builder.joint_label")
    builder.joint_q[idx + 6] = value

for i in range(len(builder.joint_target_ke)):
    builder.joint_target_ke[i] = 150
    builder.joint_target_kd[i] = 5

# Add optional light dynamic boxes along the forward path so the robot can bump them aside.
if ENABLE_DYNAMIC_CONTACT_BOXES:
    for i, (x, y, z) in enumerate(BOX_POSES):
        box_body = builder.add_link(
            xform=wp.transform(wp.vec3(x, y, z), wp.quat_identity()),
            mass=BOX_MASS,
            label=f"walking_box_{i}",
        )
        builder.add_shape_box(
            box_body,
            hx=float(BOX_HALF_EXTENTS[0]),
            hy=float(BOX_HALF_EXTENTS[1]),
            hz=float(BOX_HALF_EXTENTS[2]),
        )
        _set_shape_contact_stiffness(builder, len(builder.shape_type) - 1, BOX_CONTACT_KE, BOX_CONTACT_KD)
        builder.add_articulation([builder.add_joint_free(box_body)], label=f"walking_box_{i}")

newton_model = builder.finalize()
newton_solver = newton.solvers.SolverMuJoCo(
    newton_model,
    disable_contacts=False,
    use_mujoco_contacts=False,
    solver="newton",
    ls_parallel=False,
    ls_iterations=50,
    njmax=4096,
    nconmax=4096,
)
print(f"[Newton] ANYmal baseline model built: {newton_model.body_count} bodies, {newton_model.joint_count} joints.")
if ENABLE_DYNAMIC_CONTACT_BOXES:
    print(f"[Scene] Added {len(BOX_POSES)} dynamic Newton box obstacle(s).\n")
else:
    print("[Scene] Dynamic Newton box obstacles disabled (vanilla flat-ground walking).\n")

# ─── Initialize Newton state, control, and contacts ──────────────────────
newton_state_0 = newton_model.state()
newton_state_1 = newton_model.state()
newton_control = newton_model.control()
newton_contacts = newton_model.contacts()
newton.eval_fk(newton_model, newton_model.joint_q, newton_model.joint_qd, newton_state_0)

# ─── Visualization backend selection ─────────────────────────────────────
vis = None
_vis_available = False
if USE_OMNIVERSE_VISUALIZATION:
    vis = mophi.OmniverseVisualizer(output_path=OMNIVERSE_OUTPUT_PATH, fps=float(RENDER_FPS))
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
    vis = mophi.create_opengl_visualizer_or_fatal(newton_model)
    _vis_available = True
    print("[Viewer] MoPhi OpenGL visualization window opened.\n")

# Set the initial camera to match the multiphysics demo's viewing angle.
if _vis_available and not USE_OMNIVERSE_VISUALIZATION:
    _init_base_pos = wp.to_torch(newton_state_0.joint_q)[:3].cpu().numpy()
    vis.set_camera(
        pos=wp.vec3(_init_base_pos[0], _init_base_pos[1] - 6.0, _init_base_pos[2] + 2.0),
        pitch=-10.0,
        yaw=90.0,
    )

# ─── Load the ANYmal C walking policy ────────────────────────────────────
print("[Policy] Loading ANYmal C walking policy ...")
policy = torch.jit.load(policy_path, map_location=torch_device)
joint_pos_initial = (
    wp.to_torch(newton_state_0.joint_q)[7 : 7 + NUM_POLICY_JOINTS].to(dtype=torch.float32).unsqueeze(0).clone()
)
act = torch.zeros(1, NUM_POLICY_JOINTS, device=torch_device, dtype=torch.float32)
lab_to_mujoco_indices = torch.tensor(lab_to_mujoco, device=torch_device)
mujoco_to_lab_indices = torch.tensor(mujoco_to_lab, device=torch_device)
gravity_vec = torch.tensor([[0.0, 0.0, -1.0]], device=torch_device, dtype=torch.float32)
command = torch.tensor([[FORWARD_COMMAND, LATERAL_COMMAND, YAW_COMMAND]], device=torch_device, dtype=torch.float32)
free_joint_zeros = torch.zeros(6, device=torch_device, dtype=torch.float32)
print("[Policy] ANYmal C walking policy loaded.\n")

# ─── Movie recording settings ────────────────────────────────────────────
_movie_writer = None
if SAVE_MOVIE and _vis_available and not USE_OMNIVERSE_VISUALIZATION:
    if find_spec("imageio") is None:
        mophi.fatal(
            "imageio is required when SAVE_MOVIE=True.\n"
            "Install with:  pip install imageio imageio-ffmpeg"
        )
    import imageio

    _movie_writer = imageio.get_writer(MOVIE_OUTPUT_PATH, fps=MOVIE_FPS)
    print(f"[Movie] Recording simulation to '{MOVIE_OUTPUT_PATH}' at {MOVIE_FPS} fps.\n")

print(
    f"Running up to {NUM_FRAMES} policy frame(s) "
    f"(frame_dt={FRAME_DT * 1000:.1f} ms, {SIM_SUBSTEPS} Newton substeps × {NEWTON_DT * 1000:.1f} ms) ...\n"
)

sim_time = 0.0
for frame in range(NUM_FRAMES):
    if _vis_available and not vis.is_running():
        print(f"\n[Viewer] Window closed by user after frame {frame}.")
        break

    joint_q_t = wp.to_torch(newton_state_0.joint_q)
    joint_qd_t = wp.to_torch(newton_state_0.joint_qd)
    policy_joint_q_t = torch.cat([joint_q_t[:7], joint_q_t[7 : 7 + NUM_POLICY_JOINTS]])
    policy_joint_qd_t = torch.cat([joint_qd_t[:6], joint_qd_t[6 : 6 + NUM_POLICY_JOINTS]])
    obs = demo_utils.compute_obs(
        act,
        policy_joint_q_t,
        policy_joint_qd_t,
        joint_pos_initial,
        lab_to_mujoco_indices,
        gravity_vec,
        command,
    )
    with torch.no_grad():
        act = policy(obs)
        rearranged_act = torch.gather(act, 1, mujoco_to_lab_indices.unsqueeze(0))
        target_joint_q = joint_pos_initial + 0.5 * rearranged_act
        target_with_zeros = torch.cat([free_joint_zeros, target_joint_q.squeeze(0)])
        control_joint_target_pos_t = wp.to_torch(newton_control.joint_target_pos)
        if control_joint_target_pos_t.numel() < target_with_zeros.numel():
            raise RuntimeError(
                "newton_control.joint_target_pos is smaller than robot target vector "
                f"({control_joint_target_pos_t.numel()} < {target_with_zeros.numel()})."
            )
        control_joint_target_pos_t[: target_with_zeros.numel()] = target_with_zeros

    for _ in range(SIM_SUBSTEPS):
        newton_solver.step(newton_state_0, newton_state_1, newton_control, newton_contacts, NEWTON_DT)
        newton_state_0, newton_state_1 = newton_state_1, newton_state_0

    sim_time += FRAME_DT

    if _vis_available:
        vis.begin_frame(sim_time)
        vis.log_state(newton_state_0)
        vis.end_frame()
        if _movie_writer is not None:
            _movie_writer.append_data(vis.get_frame().numpy())

    if (frame + 1) % 50 == 0 or frame == 0:
        _base_pos_np = newton_state_0.body_q.numpy()[0, :3]
        print(
            f"  frame {frame + 1:>4}/{NUM_FRAMES}  sim_time={sim_time:.3f} s  "
            f"base=({_base_pos_np[0]:+.3f}, {_base_pos_np[1]:+.3f}, {_base_pos_np[2]:+.3f})"
        )

print()

if _movie_writer is not None:
    _movie_writer.close()
    print(f"[Movie] Saved simulation recording to '{MOVIE_OUTPUT_PATH}'.\n")

if _vis_available:
    vis.close()
