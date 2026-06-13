"""First-stage humanoid complex-environment demo.

This demo adapts Newton 1.0.0's public ``example_robot_policy.py`` Unitree G1
walking baseline for MoPhi. Newton downloads the robot model, policy, and YAML
configuration from the public ``newton-assets`` repository on first run.

The demo uses convex contact proxies generated from the robot's visual body
meshes, places lightweight dynamic boxes in its walking path, and allows more
boxes to be spawned interactively. It also adds a finite run, movie generation,
run metadata, and a final-state snapshot.

Keyboard controls
-----------------
  I / K  -- walk forward / backward
  J / L  -- strafe left / right
  U / O  -- turn left / right
  B      -- spawn a dynamic contact box in front of the robot
  P      -- reset

Running
-------
From the MoPhi build tree:

    cmake --build <build-dir> --target demo_humanoid_complex_environment

Or from the repository root after installing the mophi package:

    python demo/newton_xlb_dem/humanoid_complex_environment/demo_humanoid_complex_environment.py
"""

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import warp as wp
import yaml

import mophi
import newton
import newton.examples
import newton.utils
from newton import JointTargetMode, ShapeFlags
from newton.examples.robot import example_robot_policy as newton_robot_policy

# =============================================================================
# Simulation configuration
# All user-tunable constants are defined here.
# =============================================================================

# -- Timing -------------------------------------------------------------------
RENDER_FPS = 50
NEWTON_DT = 1.0 / 200.0
NUM_FRAMES = 500

# -- Robot --------------------------------------------------------------------
ROBOT_NAME = "g1_29dof"
USE_ROBOT_VISUAL_MESH_COLLIDERS = True
ROBOT_MESH_APPROXIMATION_METHOD = "convex_hull"

# -- Contact objects -----------------------------------------------------------
ENABLE_DYNAMIC_CONTACT_BOXES = True
BOX_HALF_EXTENTS = np.array([0.16, 0.16, 0.16], dtype=np.float32)
BOX_MASS = 0.5
BOX_GRID_X_OFFSETS = (-0.35, 0.0, 0.35)
BOX_GRID_Y_POSITIONS = (1.4, 2.0, 2.6)
BOX_POSES = [(float(x), float(y), float(BOX_HALF_EXTENTS[2])) for y in BOX_GRID_Y_POSITIONS for x in BOX_GRID_X_OFFSETS]
ENABLE_INTERACTIVE_OBJECT_SPAWNING = True
INTERACTIVE_OBJECT_SPAWN_KEY = "b"
INTERACTIVE_OBJECT_POOL_SIZE = 24
INTERACTIVE_BOX_HALF_EXTENTS = np.array([0.14, 0.14, 0.14], dtype=np.float32)
INTERACTIVE_BOX_MASS = 0.35
INTERACTIVE_OBJECT_SPAWN_OFFSET = np.array([0.0, 0.9, 0.8], dtype=np.float32)
INTERACTIVE_OBJECT_PARKING_ORIGIN = np.array([50.0, 0.0, 0.2], dtype=np.float32)
INTERACTIVE_OBJECT_PARKING_SPACING = 0.8
MAX_CONTACT_COUNT = 4096
MAX_CONSTRAINT_COUNT = 8192

# -- Visualization ------------------------------------------------------------
USE_OMNIVERSE_VISUALIZATION = False
CAMERA_POSITION = (5.0, -7.0, 2.8)
CAMERA_PITCH = -12.0
CAMERA_YAW = 135.0
REFERENCE_AXIS_ORIGIN = (0.8, -0.8, 0.02)
REFERENCE_SCALE_BAR_CENTER = (0.0, -0.8, 0.06)
SHOW_WAREHOUSE_ENVIRONMENT = True
WAREHOUSE_AISLE_LENGTH = 8.0
WAREHOUSE_AISLE_HALF_WIDTH = 1.7

# -- Output -------------------------------------------------------------------
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "output" / Path(__file__).stem
SAVE_MOVIE = True
MOVIE_OUTPUT_PATH = OUTPUT_DIRECTORY / "demo_humanoid_complex_environment.mp4"
MOVIE_FPS = RENDER_FPS
SAVE_FINAL_STATE = True
FINAL_STATE_OUTPUT_PATH = OUTPUT_DIRECTORY / "final_state.npz"
RUN_METADATA_OUTPUT_PATH = OUTPUT_DIRECTORY / "run_metadata.json"

# -- Derived constants --------------------------------------------------------
FRAME_DT = 1.0 / RENDER_FPS
SIM_SUBSTEPS = int(round(FRAME_DT / NEWTON_DT))
if not np.isclose(SIM_SUBSTEPS * NEWTON_DT, FRAME_DT, rtol=0.0, atol=1.0e-12):
    raise ValueError("FRAME_DT must be an integer multiple of NEWTON_DT.")


def _add_robot_visual_mesh_proxies(builder: newton.ModelBuilder) -> list[int]:
    """Add convex visual-mesh proxies while preserving policy-trained colliders."""
    robot_body_count = len(builder.body_label)
    original_shape_count = len(builder.shape_type)
    visual_mesh_indices = [
        shape_idx
        for shape_idx in range(original_shape_count)
        if 0 <= builder.shape_body[shape_idx] < robot_body_count
        and builder.shape_type[shape_idx] == newton.GeoType.MESH
        and builder.shape_flags[shape_idx] & ShapeFlags.VISIBLE
        and not builder.shape_flags[shape_idx] & ShapeFlags.COLLIDE_SHAPES
    ]

    mesh_proxy_indices = []
    for shape_idx in visual_mesh_indices:
        body_idx = builder.shape_body[shape_idx]
        proxy_idx = builder.add_shape_mesh(
            body=body_idx,
            xform=builder.shape_transform[shape_idx],
            mesh=builder.shape_source[shape_idx],
            scale=builder.shape_scale[shape_idx],
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0,
                ke=builder.default_shape_cfg.ke,
                kd=builder.default_shape_cfg.kd,
                kf=builder.default_shape_cfg.kf,
                mu=builder.default_shape_cfg.mu,
                is_solid=builder.shape_is_solid[shape_idx],
                has_shape_collision=True,
                has_particle_collision=True,
                is_visible=False,
            ),
            label=f"{builder.shape_label[shape_idx]}_contact_proxy",
        )
        mesh_proxy_indices.append(proxy_idx)

    builder.approximate_meshes(
        method=ROBOT_MESH_APPROXIMATION_METHOD,
        shape_indices=mesh_proxy_indices,
    )
    return mesh_proxy_indices


def _filter_mesh_proxies_to_external_objects(
    builder: newton.ModelBuilder,
    mesh_proxy_indices: list[int],
    robot_body_count: int,
    ground_shape_idx: int,
) -> None:
    """Prevent mesh proxies from changing robot-ground and robot-self contact."""
    robot_shape_indices = [
        shape_idx for shape_idx, body_idx in enumerate(builder.shape_body) if 0 <= body_idx < robot_body_count
    ]
    for proxy_idx in mesh_proxy_indices:
        builder.add_shape_collision_filter_pair(proxy_idx, ground_shape_idx)
        for robot_shape_idx in robot_shape_indices:
            if proxy_idx != robot_shape_idx:
                builder.add_shape_collision_filter_pair(proxy_idx, robot_shape_idx)


def _add_dynamic_contact_boxes(builder: newton.ModelBuilder) -> int:
    """Add lightweight free bodies that visibly demonstrate robot contact."""
    if not ENABLE_DYNAMIC_CONTACT_BOXES:
        return 0

    for box_idx, (x, y, z) in enumerate(BOX_POSES):
        body_idx = builder.add_link(
            xform=wp.transform(wp.vec3(x, y, z), wp.quat_identity()),
            mass=BOX_MASS,
            label=f"contact_box_{box_idx}",
        )
        builder.add_shape_box(
            body_idx,
            hx=float(BOX_HALF_EXTENTS[0]),
            hy=float(BOX_HALF_EXTENTS[1]),
            hz=float(BOX_HALF_EXTENTS[2]),
        )
        builder.add_articulation([builder.add_joint_free(body_idx)], label=f"contact_box_{box_idx}")
    return len(BOX_POSES)


def _add_interactive_object_pool(builder: newton.ModelBuilder) -> list[tuple[int, int]]:
    """Preallocate free boxes and return their free-joint position and velocity offsets."""
    if not ENABLE_INTERACTIVE_OBJECT_SPAWNING:
        return []

    joint_starts = []
    for object_idx in range(INTERACTIVE_OBJECT_POOL_SIZE):
        parking_position = INTERACTIVE_OBJECT_PARKING_ORIGIN + np.array(
            [INTERACTIVE_OBJECT_PARKING_SPACING * object_idx, 0.0, 0.0],
            dtype=np.float32,
        )
        label = f"interactive_box_{object_idx}"
        body_idx = builder.add_link(
            xform=wp.transform(wp.vec3(*parking_position), wp.quat_identity()),
            mass=INTERACTIVE_BOX_MASS,
            label=label,
        )
        builder.add_shape_box(
            body_idx,
            hx=float(INTERACTIVE_BOX_HALF_EXTENTS[0]),
            hy=float(INTERACTIVE_BOX_HALF_EXTENTS[1]),
            hz=float(INTERACTIVE_BOX_HALF_EXTENTS[2]),
        )
        joint_idx = builder.add_joint_free(body_idx, label=f"{label}_free_joint")
        builder.add_articulation([joint_idx], label=label)
        joint_starts.append((builder.joint_q_start[joint_idx], builder.joint_qd_start[joint_idx]))
    return joint_starts


class HumanoidContactExample(newton_robot_policy.Example):
    """Newton's G1 policy example with mesh-derived robot contact proxies and boxes."""

    def __init__(
        self,
        viewer,
        robot_config,
        config,
        asset_directory: str,
        mjc_to_physx: list[int],
        physx_to_mjc: list[int],
    ):
        self.frame_dt = NEWTON_DT
        self.decimation = 4
        self.cycle_time = self.frame_dt * self.decimation
        self.sim_time = 0.0
        self.sim_step = 0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt
        self.viewer = viewer
        self.use_mujoco = False
        self.config = config
        self.robot_config = robot_config
        self.device = wp.get_device()
        self.torch_device = "cuda" if self.device.is_cuda else "cpu"

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.1,
            limit_ke=1.0e2,
            limit_kd=1.0e0,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75

        builder.add_usd(
            newton.examples.get_asset(asset_directory + "/" + robot_config.asset_path),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.8)),
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            joint_ordering="dfs",
            hide_collision_shapes=True,
        )
        robot_body_count = len(builder.body_label)
        # Preserve Newton's policy baseline collision model for stable ground
        # contact, then add separate body-mesh proxies for external objects.
        builder.approximate_meshes(ROBOT_MESH_APPROXIMATION_METHOD)
        robot_mesh_proxy_indices = []
        self.robot_mesh_proxy_count = 0
        if USE_ROBOT_VISUAL_MESH_COLLIDERS:
            robot_mesh_proxy_indices = _add_robot_visual_mesh_proxies(builder)
            self.robot_mesh_proxy_count = len(robot_mesh_proxy_indices)

        ground_shape_idx = builder.add_ground_plane()
        _filter_mesh_proxies_to_external_objects(
            builder,
            robot_mesh_proxy_indices,
            robot_body_count,
            ground_shape_idx,
        )
        self.contact_box_count = _add_dynamic_contact_boxes(builder)
        self.interactive_object_joint_starts = _add_interactive_object_pool(builder)

        builder.joint_q[:3] = [0.0, 0.0, 0.76]
        builder.joint_q[3:7] = [0.0, 0.0, 0.7071, 0.7071]
        builder.joint_q[7 : 7 + len(config["mjw_joint_pos"])] = config["mjw_joint_pos"]

        for joint_idx in range(len(config["mjw_joint_stiffness"])):
            builder.joint_target_ke[joint_idx + 6] = config["mjw_joint_stiffness"][joint_idx]
            builder.joint_target_kd[joint_idx + 6] = config["mjw_joint_damping"][joint_idx]
            builder.joint_armature[joint_idx + 6] = config["mjw_joint_armature"][joint_idx]
            builder.joint_target_mode[joint_idx + 6] = int(JointTargetMode.POSITION)

        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=self.use_mujoco,
            solver="newton",
            nconmax=MAX_CONTACT_COUNT,
            njmax=MAX_CONSTRAINT_COUNT,
        )

        self.state_temp = self.model.state()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = newton.Contacts(self.solver.get_max_contact_count(), 0)
        self.robot_joint_q_count = 7 + config["num_dofs"]
        self.robot_joint_qd_count = 6 + config["num_dofs"]
        self.robot_joint_target_count = 6 + config["num_dofs"]
        self.viewer.set_model(self.model)
        self.viewer.vsync = True
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self._initial_joint_q = wp.clone(self.state_0.joint_q)
        self._initial_joint_qd = wp.clone(self.state_0.joint_qd)

        self.physx_to_mjc_indices = torch.tensor(physx_to_mjc, device=self.torch_device, dtype=torch.long)
        self.mjc_to_physx_indices = torch.tensor(mjc_to_physx, device=self.torch_device, dtype=torch.long)
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.torch_device, dtype=torch.float32).unsqueeze(0)
        self.command = torch.zeros((1, 3), device=self.torch_device, dtype=torch.float32)
        self._reset_key_prev = False
        self._spawn_key_prev = False
        self.spawned_object_count = 0
        self.policy = None
        self.joint_pos_initial = None
        self.act = None
        self.rearranged_act = None
        self.capture()
        # CUDA graph capture replaces joint_target_pos with a robot-only buffer.
        # Allocate against the final control shape so CPU and CUDA copies agree.
        self.full_joint_target_pos = torch.zeros(
            self.control.joint_target_pos.shape[0],
            device=self.torch_device,
            dtype=torch.float32,
        )

    def step(self):
        """Advance the inherited policy while excluding free-box coordinates from its inputs."""
        if hasattr(self.viewer, "is_key_down"):
            forward = 1.0 if self.viewer.is_key_down("i") else (-1.0 if self.viewer.is_key_down("k") else 0.0)
            lateral = 0.5 if self.viewer.is_key_down("j") else (-0.5 if self.viewer.is_key_down("l") else 0.0)
            rotation = 1.0 if self.viewer.is_key_down("u") else (-1.0 if self.viewer.is_key_down("o") else 0.0)
            self.command[0, 0] = float(forward)
            self.command[0, 1] = float(lateral)
            self.command[0, 2] = float(rotation)
            reset_down = bool(self.viewer.is_key_down("p"))
            if reset_down and not self._reset_key_prev:
                self.reset()
            self._reset_key_prev = reset_down
            spawn_down = bool(self.viewer.is_key_down(INTERACTIVE_OBJECT_SPAWN_KEY))
            if spawn_down and not self._spawn_key_prev:
                self.spawn_interactive_object()
            self._spawn_key_prev = spawn_down

        policy_state = newton.State()
        policy_state.joint_q = self.state_0.joint_q[: self.robot_joint_q_count]
        policy_state.joint_qd = self.state_0.joint_qd[: self.robot_joint_qd_count]
        observation = newton_robot_policy.compute_obs(
            self.act,
            policy_state,
            self.joint_pos_initial,
            self.torch_device,
            self.physx_to_mjc_indices,
            self.gravity_vec,
            self.command,
        )
        with torch.no_grad():
            self.act = self.policy(observation)
            self.rearranged_act = torch.index_select(self.act, 1, self.mjc_to_physx_indices)
            robot_target = self.joint_pos_initial + self.config["action_scale"] * self.rearranged_act
            robot_target_with_zeros = torch.cat(
                [torch.zeros(6, device=self.torch_device, dtype=torch.float32), robot_target.squeeze(0)]
            )
            self.full_joint_target_pos.zero_()
            self.full_joint_target_pos[: self.robot_joint_target_count] = robot_target_with_zeros
            target_wp = wp.from_torch(self.full_joint_target_pos, dtype=wp.float32, requires_grad=False)
            wp.copy(self.control.joint_target_pos, target_wp)

        for _ in range(self.decimation):
            if self.graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()

        self.sim_time += self.frame_dt
        self.sim_step += 1

    def spawn_interactive_object(self) -> bool:
        """Activate the next pooled object at a configurable robot-relative position."""
        if self.spawned_object_count >= len(self.interactive_object_joint_starts):
            print(f"[Spawn] Interactive object pool is full ({len(self.interactive_object_joint_starts)} objects).")
            return False

        robot_position = self.state_0.joint_q.numpy()[:3]
        spawn_position = robot_position + INTERACTIVE_OBJECT_SPAWN_OFFSET
        joint_q_start, joint_qd_start = self.interactive_object_joint_starts[self.spawned_object_count]
        spawn_q = wp.array(
            [*spawn_position, 0.0, 0.0, 0.0, 1.0],
            dtype=wp.float32,
            device=self.device,
        )
        spawn_qd = wp.zeros(6, dtype=wp.float32, device=self.device)
        for state in (self.state_0, self.state_1):
            wp.copy(state.joint_q, spawn_q, dest_offset=joint_q_start)
            wp.copy(state.joint_qd, spawn_qd, dest_offset=joint_qd_start)
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)

        self.spawned_object_count += 1
        print(
            f"[Spawn] Added interactive box {self.spawned_object_count}/{len(self.interactive_object_joint_starts)} "
            f"at ({spawn_position[0]:.2f}, {spawn_position[1]:.2f}, {spawn_position[2]:.2f})."
        )
        return True

    def reset(self):
        """Reset the robot and return all interactively spawned objects to the pool."""
        super().reset()
        self.spawned_object_count = 0


def _write_run_metadata(frame_count: int, asset_directory: Path, spawned_object_count: int) -> None:
    metadata = {
        "baseline": "newton.examples.robot.example_robot_policy",
        "robot": ROBOT_NAME,
        "newton_version": newton.__version__,
        "asset_directory": str(asset_directory),
        "newton_dt": NEWTON_DT,
        "render_fps": RENDER_FPS,
        "requested_frames": NUM_FRAMES,
        "completed_frames": frame_count,
        "sim_duration": frame_count * FRAME_DT,
        "use_robot_visual_mesh_colliders": USE_ROBOT_VISUAL_MESH_COLLIDERS,
        "dynamic_contact_box_count": len(BOX_POSES) if ENABLE_DYNAMIC_CONTACT_BOXES else 0,
        "interactive_object_pool_size": INTERACTIVE_OBJECT_POOL_SIZE if ENABLE_INTERACTIVE_OBJECT_SPAWNING else 0,
        "interactively_spawned_object_count": spawned_object_count,
        "show_warehouse_environment": SHOW_WAREHOUSE_ENVIRONMENT,
    }
    RUN_METADATA_OUTPUT_PATH.write_text(json.dumps(metadata, indent=4) + "\n", encoding="utf-8")


def main() -> None:
    if USE_OMNIVERSE_VISUALIZATION:
        mophi.fatal("Newton's interactive G1 policy baseline currently requires ViewerGL.")
    if ROBOT_NAME not in newton_robot_policy.ROBOT_CONFIGS:
        mophi.fatal(f"Unknown Newton policy robot '{ROBOT_NAME}'.")

    mophi.check_newton_warp_mujoco_versions(
        mophi.REQUIRED_NEWTON_VERSION,
        mophi.REQUIRED_WARP_VERSION,
        mophi.REQUIRED_MUJOCO_VERSION,
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    wp.init()

    robot_config = newton_robot_policy.ROBOT_CONFIGS[ROBOT_NAME]
    print(f"[Assets] Downloading or locating public Newton assets for '{ROBOT_NAME}' ...")
    try:
        asset_directory = Path(newton.utils.download_asset(robot_config.asset_dir))
    except RuntimeError as exc:
        mophi.fatal(
            "Newton could not download the Unitree G1 assets. Check GitHub connectivity, "
            "then retry with a fresh cache directory:\n"
            "  NEWTON_CACHE_PATH=$HOME/.cache/newton-mophi "
            "python -c \"import newton.utils; print(newton.utils.download_asset('unitree_g1'))\"\n"
            f"Newton error: {exc}"
        )
    print(f"[Assets] Ready at '{asset_directory}'.")

    yaml_path = asset_directory / robot_config.yaml_path
    if not yaml_path.is_file():
        mophi.fatal(f"Downloaded robot configuration was not found: {yaml_path}")
    with yaml_path.open(encoding="utf-8") as yaml_file:
        config = yaml.safe_load(yaml_file)

    policy_path = asset_directory / robot_config.policy_path["mjw"]
    if not policy_path.is_file():
        mophi.fatal(f"Downloaded robot policy was not found: {policy_path}")
    model_path = asset_directory / robot_config.asset_path
    if not model_path.is_file():
        mophi.fatal(
            "The Newton Unitree G1 asset cache is incomplete. The robot model "
            f"was not found:\n  {model_path}\n"
            "Download into a fresh cache directory, then run this demo with the "
            "same NEWTON_CACHE_PATH value:\n"
            "  NEWTON_CACHE_PATH=$HOME/.cache/newton-mophi "
            "python -c \"import newton.utils; print(newton.utils.download_asset('unitree_g1'))\""
        )

    dof_indices = list(range(config["num_dofs"]))
    viewer = newton.viewer.ViewerGL()
    example = HumanoidContactExample(
        viewer,
        robot_config,
        config,
        str(asset_directory),
        dof_indices,
        dof_indices,
    )
    newton_robot_policy.load_policy_and_setup_tensors(
        example,
        str(policy_path),
        config["num_dofs"],
        slice(7, 7 + config["num_dofs"]),
    )
    if not np.isclose(example.frame_dt, NEWTON_DT, rtol=0.0, atol=1.0e-12):
        mophi.fatal(f"Newton G1 policy timestep changed: expected {NEWTON_DT}, got {example.frame_dt}.")
    print(
        f"[Contact] Created {example.robot_mesh_proxy_count} mesh-derived robot contact proxies "
        f"and {example.contact_box_count} initial dynamic boxes. "
        f"Press '{INTERACTIVE_OBJECT_SPAWN_KEY.upper()}' to spawn up to "
        f"{len(example.interactive_object_joint_starts)} additional boxes."
    )

    if hasattr(viewer, "set_camera"):
        viewer.set_camera(wp.vec3(*CAMERA_POSITION), CAMERA_PITCH, CAMERA_YAW)
    mophi.log_orientation_and_scale_reference(
        viewer,
        axis_origin=REFERENCE_AXIS_ORIGIN,
        scale_bar_center=REFERENCE_SCALE_BAR_CENTER,
    )
    if SHOW_WAREHOUSE_ENVIRONMENT:
        mophi.log_warehouse_environment(
            viewer,
            aisle_length=WAREHOUSE_AISLE_LENGTH,
            aisle_half_width=WAREHOUSE_AISLE_HALF_WIDTH,
        )

    movie_writer = None
    if SAVE_MOVIE:
        movie_writer = imageio.get_writer(MOVIE_OUTPUT_PATH, fps=MOVIE_FPS)
        print(f"[Movie] Recording simulation to '{MOVIE_OUTPUT_PATH}' at {MOVIE_FPS} fps.")

    print(
        f"Running up to {NUM_FRAMES} interactive Unitree G1 frame(s) "
        f"({SIM_SUBSTEPS} Newton policy steps x {NEWTON_DT * 1000.0:.1f} ms per rendered frame) ..."
    )

    completed_frames = 0
    try:
        for frame in range(NUM_FRAMES):
            if not viewer.is_running():
                print(f"\n[Viewer] Window closed by user after frame {frame}.")
                break
            for _ in range(SIM_SUBSTEPS):
                example.step()
            example.render()
            completed_frames = frame + 1
            if movie_writer is not None:
                movie_writer.append_data(viewer.get_frame().numpy())
            if completed_frames % RENDER_FPS == 0 or completed_frames == 1:
                print(f"  frame {completed_frames:>4}/{NUM_FRAMES}  sim_time={example.sim_time:.2f} s")
    except KeyboardInterrupt:
        print(f"\n[Input] Stopped by user after frame {completed_frames}.")
    finally:
        if movie_writer is not None:
            movie_writer.close()
        viewer.close()

    if SAVE_FINAL_STATE:
        np.savez_compressed(
            FINAL_STATE_OUTPUT_PATH,
            body_q=example.state_0.body_q.numpy(),
            joint_q=example.state_0.joint_q.numpy(),
        )
        print(f"[Output] Saved final state to '{FINAL_STATE_OUTPUT_PATH}'.")

    _write_run_metadata(completed_frames, asset_directory, example.spawned_object_count)
    print(f"[Output] Saved run metadata to '{RUN_METADATA_OUTPUT_PATH}'.")
    if SAVE_MOVIE:
        print(f"[Movie] Saved simulation recording to '{MOVIE_OUTPUT_PATH}'.")


if __name__ == "__main__":
    main()
