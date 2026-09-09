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
from humanoid_fight_control import scripted_attack_offsets
from humanoid_proxy_geometry import make_deme_mesh_owner_specs, make_scaled_box_mesh_proxies
from humanoid_visualization import configure_scene_lighting, transform_contact_proxy_lines
from mophi.couplers.newton_deme import (
    NewtonDEMEContactCoupler,
    NewtonDEMEMeshOwnerSpec,
    add_deme_mesh_owners,
    extract_newton_shape_triangle_meshes,
    owner_map_from_mesh_bindings,
)
import newton
import newton.examples
from newton import JointTargetMode, ShapeFlags
from newton.examples.robot import example_robot_policy as newton_robot_policy

# =============================================================================
# Simulation configuration
# All user-tunable constants are defined here.
# =============================================================================

# -- Timing -------------------------------------------------------------------
RENDER_FPS = 50
NEWTON_DT = 1.0 / 200.0
POLICY_DECIMATION = 4
NUM_FRAMES = 500

# -- Robot --------------------------------------------------------------------
ROBOT_NAME = "g1_29dof"
ROBOT_INITIAL_POSES = [
    ((0.0, 0.0, 0.76), (0.0, 0.0, 0.7071, 0.7071)),
]
DEFAULT_WALK_COMMANDS = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
USE_ROBOT_VISUAL_MESH_COLLIDERS = True
ROBOT_MESH_APPROXIMATION_METHOD = "convex_hull"
ENABLE_SCRIPTED_FIGHT = False
FIGHT_APPROACH_END_TIME = 1.4
FIGHT_APPROACH_COMMANDS = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
FIGHT_HOLD_COMMANDS = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
FIGHT_ATTACK_SEQUENCE = ()
FIGHT_FIST_CURL = 0.8
FIGHT_PUNCH_MOTION_SCALE = 1.0

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
SCENE_SKY_UPPER_COLOR = (0.48, 0.56, 0.64)
SCENE_SKY_LOWER_COLOR = (0.20, 0.24, 0.28)
SCENE_LIGHT_COLOR = (1.6, 1.7, 1.8)
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
VISUAL_MESH_MANIFEST_OUTPUT_PATH = OUTPUT_DIRECTORY / "visual_mesh_manifest.json"

# -- Contact-proxy visualization ----------------------------------------------
SHOW_CONTACT_PROXY_MESHES = False
CONTACT_PROXY_MESH_PATH = REPOSITORY_ROOT / "data" / "mesh" / "cube.obj"
CONTACT_PROXY_PADDING = 0.005
CONTACT_PROXY_LINE_WIDTH = 0.006
CONTACT_PROXY_COLORS = ((0.1, 0.9, 1.0), (1.0, 0.45, 0.1))

# -- DEME robot-contact resolution -------------------------------------------
ENABLE_DEME_ROBOT_CONTACT = False
DEME_MODULE = None
DEME_DT = 1.0 / 1000.0
DEME_ROBOT_FAMILIES = (10, 11)
DEME_DOMAIN_X = (-2.0, 2.0)
DEME_DOMAIN_Y = (-2.5, 2.5)
DEME_DOMAIN_Z = (-0.5, 2.5)
DEME_CONTACT_MATERIAL = {"E": 5.0e6, "nu": 0.3, "CoR": 0.1, "mu": 0.6, "Crr": 0.0}
DEME_PROXY_DIRECTORY_NAME = "deme_contact_proxies"

# -- Derived constants --------------------------------------------------------
FRAME_DT = 1.0 / RENDER_FPS
POLICY_DT = POLICY_DECIMATION * NEWTON_DT
SIM_SUBSTEPS = int(round(FRAME_DT / POLICY_DT))
DEME_SUBSTEPS = int(round(NEWTON_DT / DEME_DT))
if not np.isclose(SIM_SUBSTEPS * POLICY_DT, FRAME_DT, rtol=0.0, atol=1.0e-12):
    raise ValueError("FRAME_DT must be an integer multiple of POLICY_DT.")
if not np.isclose(DEME_SUBSTEPS * DEME_DT, NEWTON_DT, rtol=0.0, atol=1.0e-12):
    raise ValueError("NEWTON_DT must be an integer multiple of DEME_DT.")


def _configure_viewer_scene_lighting(viewer) -> None:
    configure_scene_lighting(viewer, SCENE_SKY_UPPER_COLOR, SCENE_SKY_LOWER_COLOR, SCENE_LIGHT_COLOR)


def _scripted_attack_offsets(attack_kind: str, side: str, progress: float) -> dict[str, float]:
    return scripted_attack_offsets(attack_kind, side, progress, FIGHT_FIST_CURL, FIGHT_PUNCH_MOTION_SCALE)


def _make_scaled_contact_mesh_proxies(
    visual_meshes: list[dict],
    base_vertices: np.ndarray,
    base_indices: np.ndarray,
) -> list[dict]:
    """Scale the configured OBJ mesh around each visual triangle mesh."""
    try:
        return make_scaled_box_mesh_proxies(visual_meshes, base_vertices, base_indices, CONTACT_PROXY_PADDING)
    except ValueError as exc:
        mophi.fatal(f"{exc} Source: {CONTACT_PROXY_MESH_PATH}")


def _make_deme_robot_mesh_specs(contact_mesh_proxies: list[dict]) -> list[NewtonDEMEMeshOwnerSpec]:
    """Group the fighting proxies into one DEME mesh-owner specification per body."""
    return make_deme_mesh_owner_specs(contact_mesh_proxies, DEME_ROBOT_FAMILIES)


def _create_deme_robot_contact_coupler(example) -> NewtonDEMEContactCoupler:
    """Create fighting-specific DEME proxies and bind MoPhi's GPU coupler."""
    if DEME_MODULE is None:
        mophi.fatal("DEME contact is enabled, but the fighting demo did not provide the deme package module.")
    if len(DEME_ROBOT_FAMILIES) != example.robot_count:
        mophi.fatal(
            f"DEME_ROBOT_FAMILIES must contain one family per robot ({example.robot_count}), "
            f"got {len(DEME_ROBOT_FAMILIES)}."
        )

    solver = DEME_MODULE.DEMSolver([example.device.ordinal])
    material = solver.LoadMaterial(DEME_CONTACT_MATERIAL)
    try:
        mesh_bindings = add_deme_mesh_owners(
            solver,
            material,
            _make_deme_robot_mesh_specs(example.contact_mesh_proxies),
            example.state_0.body_q.numpy(),
            OUTPUT_DIRECTORY / DEME_PROXY_DIRECTORY_NAME,
        )
    except ValueError as exc:
        mophi.fatal(str(exc))
    for family in DEME_ROBOT_FAMILIES:
        # The coupling layer uploads Newton's state. DEME holds each supplied
        # velocity constant while integrating the unprescribed pose.
        solver.SetFamilyPrescribedLinVel(family)
        solver.SetFamilyPrescribedAngVel(family)
        solver.DisableContactBetweenFamilies(family, family)
    solver.SetMeshUniversalContact(True)
    solver.InstructBoxDomainDimension(DEME_DOMAIN_X, DEME_DOMAIN_Y, DEME_DOMAIN_Z)
    solver.SetGravitationalAcceleration([0.0, 0.0, 0.0])
    solver.SetInitTimeStep(DEME_DT)
    solver.SetErrorOutAvgContacts(1000)
    solver.Initialize()

    try:
        owner_map = owner_map_from_mesh_bindings(mesh_bindings)
        coupler = NewtonDEMEContactCoupler()
        coupler.initialize(example.model, solver, owner_map, example.device)
    except (RuntimeError, ValueError) as exc:
        mophi.fatal(str(exc))
    print(
        f"[DEME] Initialized {owner_map.owner_count} articulated body proxy meshes; "
        "only cross-robot mesh contact is enabled."
    )
    return coupler


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


def _collect_robot_visual_meshes(
    builder: newton.ModelBuilder,
    robot_body_start: int,
    robot_body_end: int,
    robot_instance: int,
) -> list[dict]:
    """Retain the visual mesh sources and describe their future DEME inputs."""
    shape_indices = []
    for shape_idx, shape_type in enumerate(builder.shape_type):
        body_idx = builder.shape_body[shape_idx]
        flags = builder.shape_flags[shape_idx]
        if not (
            robot_body_start <= body_idx < robot_body_end
            and shape_type == newton.GeoType.MESH
            and flags & ShapeFlags.VISIBLE
            and not flags & ShapeFlags.COLLIDE_SHAPES
        ):
            continue

        shape_indices.append(shape_idx)

    try:
        extracted_meshes = extract_newton_shape_triangle_meshes(builder, shape_indices)
    except ValueError as exc:
        mophi.fatal(str(exc))
    visual_meshes = []
    for extracted in extracted_meshes:
        shape_idx = extracted.shape_index
        body_idx = extracted.body_index
        visual_meshes.append(
            {
                "shape_index": shape_idx,
                "robot_instance": robot_instance,
                "shape_label": extracted.shape_label,
                "body_index": body_idx,
                "body_label": extracted.body_label,
                "shape_transform": builder.shape_transform[shape_idx],
                "scaled_vertices": extracted.scaled_vertices,
                "vertex_count": len(extracted.scaled_vertices),
                "triangle_count": len(extracted.triangle_indices),
            }
        )
    if not visual_meshes:
        mophi.fatal("The Unitree G1 asset did not expose any non-colliding visual meshes.")
    return visual_meshes


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
        self.decimation = POLICY_DECIMATION
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

        self.robot_visual_meshes = []
        self.robot_body_ranges = []
        for robot_instance, (position, orientation) in enumerate(ROBOT_INITIAL_POSES):
            robot_body_start = len(builder.body_label)
            builder.add_usd(
                newton.examples.get_asset(asset_directory + "/" + robot_config.asset_path),
                xform=wp.transform(wp.vec3(*position), wp.quat(*orientation)),
                collapse_fixed_joints=False,
                enable_self_collisions=False,
                joint_ordering="dfs",
                hide_collision_shapes=True,
            )
            robot_body_end = len(builder.body_label)
            self.robot_body_ranges.append((robot_body_start, robot_body_end))
            self.robot_visual_meshes.extend(
                _collect_robot_visual_meshes(builder, robot_body_start, robot_body_end, robot_instance)
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

        # DEME owns cross-robot contact in the fighting configuration. Newton
        # retains all robot-ground and articulated self-contact behavior.
        if ENABLE_DEME_ROBOT_CONTACT and len(self.robot_body_ranges) > 1:
            robot_shapes = []
            for body_start, body_end in self.robot_body_ranges:
                robot_shapes.append(
                    [
                        shape_idx
                        for shape_idx, body_idx in enumerate(builder.shape_body)
                        if body_start <= body_idx < body_end
                    ]
                )
            for first_robot_idx in range(len(robot_shapes)):
                for second_robot_idx in range(first_robot_idx + 1, len(robot_shapes)):
                    for first_shape_idx in robot_shapes[first_robot_idx]:
                        for second_shape_idx in robot_shapes[second_robot_idx]:
                            builder.add_shape_collision_filter_pair(first_shape_idx, second_shape_idx)

        ground_shape_idx = builder.add_ground_plane()
        _filter_mesh_proxies_to_external_objects(
            builder,
            robot_mesh_proxy_indices,
            robot_body_count,
            ground_shape_idx,
        )
        self.contact_box_count = _add_dynamic_contact_boxes(builder)
        self.interactive_object_joint_starts = _add_interactive_object_pool(builder)

        self.robot_count = len(ROBOT_INITIAL_POSES)
        self.robot_joint_q_count = 7 + config["num_dofs"]
        self.robot_joint_qd_count = 6 + config["num_dofs"]
        self.robot_joint_target_count = 6 + config["num_dofs"]
        for robot_instance, (position, orientation) in enumerate(ROBOT_INITIAL_POSES):
            joint_q_start = robot_instance * self.robot_joint_q_count
            joint_dof_start = robot_instance * self.robot_joint_qd_count
            builder.joint_q[joint_q_start : joint_q_start + 3] = position
            builder.joint_q[joint_q_start + 3 : joint_q_start + 7] = orientation
            builder.joint_q[joint_q_start + 7 : joint_q_start + self.robot_joint_q_count] = config["mjw_joint_pos"]
            for joint_idx in range(len(config["mjw_joint_stiffness"])):
                dof_idx = joint_dof_start + 6 + joint_idx
                builder.joint_target_ke[dof_idx] = config["mjw_joint_stiffness"][joint_idx]
                builder.joint_target_kd[dof_idx] = config["mjw_joint_damping"][joint_idx]
                builder.joint_armature[dof_idx] = config["mjw_joint_armature"][joint_idx]
                builder.joint_target_mode[dof_idx] = int(JointTargetMode.POSITION)

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
        self.viewer.set_model(self.model)
        self.viewer.vsync = True
        self.contact_mesh_proxies = []
        self.contact_proxy_body_indices = None
        self.contact_proxy_local_starts = None
        self.contact_proxy_local_ends = None
        self.contact_proxy_world_starts = None
        self.contact_proxy_world_ends = None
        self.contact_proxy_colors = None
        self.deme_contact_coupler = None
        if SHOW_CONTACT_PROXY_MESHES or ENABLE_DEME_ROBOT_CONTACT:
            self._initialize_contact_proxy_visualization()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self._initial_joint_q = wp.clone(self.state_0.joint_q)
        self._initial_joint_qd = wp.clone(self.state_0.joint_qd)

        self.physx_to_mjc_indices = torch.tensor(physx_to_mjc, device=self.torch_device, dtype=torch.long)
        self.mjc_to_physx_indices = torch.tensor(mjc_to_physx, device=self.torch_device, dtype=torch.long)
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.torch_device, dtype=torch.float32).unsqueeze(0)
        if DEFAULT_WALK_COMMANDS.shape != (self.robot_count, 3):
            mophi.fatal(
                f"DEFAULT_WALK_COMMANDS must have shape ({self.robot_count}, 3), " f"got {DEFAULT_WALK_COMMANDS.shape}."
            )
        self.command = torch.as_tensor(DEFAULT_WALK_COMMANDS, device=self.torch_device, dtype=torch.float32).clone()
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
        expected_target_count = self.robot_count * self.robot_joint_target_count
        if self.full_joint_target_pos.shape[0] < expected_target_count:
            mophi.fatal(
                f"Newton control buffer has {self.full_joint_target_pos.shape[0]} target values, "
                f"but {self.robot_count} robot(s) require {expected_target_count}."
            )

    def capture(self) -> None:
        """Capture simulation with a control buffer sized for every robot instance."""
        self.graph = None
        self.use_cuda_graph = False
        if not ENABLE_DEME_ROBOT_CONTACT and wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device()):
            print("[INFO] Using CUDA graph")
            self.use_cuda_graph = True
            target_count = self.robot_count * self.robot_joint_target_count
            torch_tensor = torch.zeros(target_count, device=self.torch_device, dtype=torch.float32)
            self.control.joint_target_pos = wp.from_torch(torch_tensor, dtype=wp.float32, requires_grad=False)
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def _initialize_contact_proxy_visualization(self) -> None:
        """Build scaled contact meshes and allocate their optional wireframe overlay."""
        if not CONTACT_PROXY_MESH_PATH.is_file():
            mophi.fatal(f"Required contact proxy OBJ was not found: {CONTACT_PROXY_MESH_PATH}")
        self.contact_proxy_base_mesh = mophi.load_obj(str(CONTACT_PROXY_MESH_PATH))
        base_vertices = np.asarray(self.contact_proxy_base_mesh.vertices, dtype=np.float32)
        base_indices = np.asarray(self.contact_proxy_base_mesh.indices, dtype=np.int32)
        self.contact_mesh_proxies = _make_scaled_contact_mesh_proxies(
            self.robot_visual_meshes,
            base_vertices,
            base_indices,
        )
        if not SHOW_CONTACT_PROXY_MESHES:
            return
        body_indices = []
        local_starts = []
        local_ends = []
        colors = []
        for proxy in self.contact_mesh_proxies:
            vertices = proxy["body_local_vertices"]
            triangles = proxy["triangle_indices"].reshape(-1, 3)
            color = CONTACT_PROXY_COLORS[proxy["robot_instance"] % len(CONTACT_PROXY_COLORS)]
            unique_edges = {}
            for triangle in triangles:
                for start_idx, end_idx in (
                    (int(triangle[0]), int(triangle[1])),
                    (int(triangle[1]), int(triangle[2])),
                    (int(triangle[2]), int(triangle[0])),
                ):
                    start_key = tuple(np.round(vertices[start_idx], decimals=8))
                    end_key = tuple(np.round(vertices[end_idx], decimals=8))
                    edge_key = tuple(sorted((start_key, end_key)))
                    unique_edges[edge_key] = (start_idx, end_idx)
            for start_idx, end_idx in unique_edges.values():
                body_indices.append(proxy["body_index"])
                local_starts.append(wp.vec3(*vertices[start_idx]))
                local_ends.append(wp.vec3(*vertices[end_idx]))
                colors.append(wp.vec3(*color))

        self.contact_proxy_body_indices = wp.array(body_indices, dtype=wp.int32, device=self.device)
        self.contact_proxy_local_starts = wp.array(local_starts, dtype=wp.vec3, device=self.device)
        self.contact_proxy_local_ends = wp.array(local_ends, dtype=wp.vec3, device=self.device)
        self.contact_proxy_world_starts = wp.empty(len(local_starts), dtype=wp.vec3, device=self.device)
        self.contact_proxy_world_ends = wp.empty(len(local_ends), dtype=wp.vec3, device=self.device)
        self.contact_proxy_colors = wp.array(colors, dtype=wp.vec3, device=self.device)

    def configure_policy_instances(self) -> None:
        """Expand Newton's single-instance policy tensors across all robots."""
        initial_positions = []
        for robot_instance in range(self.robot_count):
            joint_q_start = robot_instance * self.robot_joint_q_count
            initial_positions.append(
                torch.tensor(
                    self.state_0.joint_q[joint_q_start + 7 : joint_q_start + self.robot_joint_q_count],
                    device=self.torch_device,
                    dtype=torch.float32,
                )
            )
        self.joint_pos_initial = torch.stack(initial_positions)
        self.act = torch.zeros(
            self.robot_count,
            self.config["num_dofs"],
            device=self.torch_device,
            dtype=torch.float32,
        )
        self.rearranged_act = torch.zeros_like(self.act)
        self.policy_joint_name_to_index = {
            joint_name: joint_idx for joint_idx, joint_name in enumerate(self.config["mjw_joint_names"])
        }
        if ENABLE_SCRIPTED_FIGHT:
            for commands_name, commands in (
                ("FIGHT_APPROACH_COMMANDS", FIGHT_APPROACH_COMMANDS),
                ("FIGHT_HOLD_COMMANDS", FIGHT_HOLD_COMMANDS),
            ):
                if commands.shape != (self.robot_count, 3):
                    mophi.fatal(f"{commands_name} must have shape ({self.robot_count}, 3), got {commands.shape}.")
            for start_time, duration, attacker, attack_kind, side in FIGHT_ATTACK_SEQUENCE:
                if duration <= 0.0 or not 0 <= attacker < self.robot_count:
                    mophi.fatal(
                        f"Invalid scripted attack ({start_time}, {duration}, {attacker}, {attack_kind!r}, {side!r})."
                    )
                offsets = _scripted_attack_offsets(attack_kind, side, 0.5)
                missing_names = offsets.keys() - self.policy_joint_name_to_index.keys()
                if missing_names:
                    mophi.fatal(f"Scripted attack references unknown G1 joints: {sorted(missing_names)}")

    def _scripted_fight_residuals(self) -> torch.Tensor:
        """Update approach commands and return the active attack residuals."""
        commands = FIGHT_APPROACH_COMMANDS if self.sim_time < FIGHT_APPROACH_END_TIME else FIGHT_HOLD_COMMANDS
        self.command.copy_(torch.as_tensor(commands, device=self.torch_device, dtype=torch.float32))
        residuals = torch.zeros(
            (self.robot_count, self.config["num_dofs"]), device=self.torch_device, dtype=torch.float32
        )
        for start_time, duration, attacker, attack_kind, side in FIGHT_ATTACK_SEQUENCE:
            progress = (self.sim_time - start_time) / duration
            if 0.0 <= progress <= 1.0:
                for joint_name, offset in _scripted_attack_offsets(attack_kind, side, progress).items():
                    residuals[attacker, self.policy_joint_name_to_index[joint_name]] += offset
        return residuals

    def initialize_deme_contact(self) -> None:
        """Create DEME after Newton's initial body transforms are available."""
        if ENABLE_DEME_ROBOT_CONTACT:
            self.deme_contact_coupler = _create_deme_robot_contact_coupler(self)

    def simulate(self):
        """Advance Newton after applying the current DEME proxy contact resultants."""
        need_state_copy = self.use_cuda_graph and self.sim_substeps % 2 == 1
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            if self.deme_contact_coupler is not None:
                self.deme_contact_coupler.set_deme_owner_state_from_newton(self.state_0)
                self.deme_contact_coupler.step_deme(DEME_SUBSTEPS)
                self.deme_contact_coupler.write_deme_contact_wrenches_to_newton()
                wp.copy(self.state_0.body_f, self.deme_contact_coupler.newton_body_forces)
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            if need_state_copy and substep == self.sim_substeps - 1:
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = self.state_1, self.state_0
        self.solver.update_contacts(self.contacts, self.state_0)

    def step(self):
        """Advance the inherited policy while excluding free-box coordinates from its inputs."""
        if hasattr(self.viewer, "is_key_down"):
            if self.viewer.is_key_down("i"):
                self.command[0, 0] = 1.0
            elif self.viewer.is_key_down("k"):
                self.command[0, 0] = -1.0
            else:
                self.command[0, 0] = float(DEFAULT_WALK_COMMANDS[0, 0])
            if self.viewer.is_key_down("j"):
                self.command[0, 1] = 0.5
            elif self.viewer.is_key_down("l"):
                self.command[0, 1] = -0.5
            else:
                self.command[0, 1] = float(DEFAULT_WALK_COMMANDS[0, 1])
            if self.viewer.is_key_down("u"):
                self.command[0, 2] = 1.0
            elif self.viewer.is_key_down("o"):
                self.command[0, 2] = -1.0
            else:
                self.command[0, 2] = float(DEFAULT_WALK_COMMANDS[0, 2])
            reset_down = bool(self.viewer.is_key_down("p"))
            if reset_down and not self._reset_key_prev:
                self.reset()
            self._reset_key_prev = reset_down
            spawn_down = bool(self.viewer.is_key_down(INTERACTIVE_OBJECT_SPAWN_KEY))
            if spawn_down and not self._spawn_key_prev:
                self.spawn_interactive_object()
            self._spawn_key_prev = spawn_down

        fight_residuals = None
        if ENABLE_SCRIPTED_FIGHT:
            fight_residuals = self._scripted_fight_residuals()

        observations = []
        for robot_instance in range(self.robot_count):
            joint_q_start = robot_instance * self.robot_joint_q_count
            joint_qd_start = robot_instance * self.robot_joint_qd_count
            policy_state = newton.State()
            policy_state.joint_q = self.state_0.joint_q[joint_q_start : joint_q_start + self.robot_joint_q_count]
            policy_state.joint_qd = self.state_0.joint_qd[joint_qd_start : joint_qd_start + self.robot_joint_qd_count]
            observations.append(
                newton_robot_policy.compute_obs(
                    self.act[robot_instance : robot_instance + 1],
                    policy_state,
                    self.joint_pos_initial[robot_instance : robot_instance + 1],
                    self.torch_device,
                    self.physx_to_mjc_indices,
                    self.gravity_vec,
                    self.command[robot_instance : robot_instance + 1],
                )
            )
        observation = torch.cat(observations)
        with torch.no_grad():
            self.act = self.policy(observation)
            self.rearranged_act = torch.index_select(self.act, 1, self.mjc_to_physx_indices)
            robot_target = self.joint_pos_initial + self.config["action_scale"] * self.rearranged_act
            if fight_residuals is not None:
                robot_target += fight_residuals
            robot_target_with_zeros = torch.cat(
                [
                    torch.zeros(self.robot_count, 6, device=self.torch_device, dtype=torch.float32),
                    robot_target,
                ],
                dim=1,
            ).flatten()
            self.full_joint_target_pos.zero_()
            self.full_joint_target_pos[: self.robot_count * self.robot_joint_target_count] = robot_target_with_zeros
            target_wp = wp.from_torch(self.full_joint_target_pos, dtype=wp.float32, requires_grad=False)
            wp.copy(self.control.joint_target_pos, target_wp)

        for _ in range(self.decimation):
            if self.graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()

        self.sim_time += self.cycle_time
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

    def render(self) -> None:
        """Render robot state plus optional visualization-only contact boxes."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        if self.contact_proxy_local_starts is not None:
            wp.launch(
                transform_contact_proxy_lines,
                dim=len(self.contact_proxy_local_starts),
                inputs=[
                    self.state_0.body_q,
                    self.contact_proxy_body_indices,
                    self.contact_proxy_local_starts,
                    self.contact_proxy_local_ends,
                ],
                outputs=[self.contact_proxy_world_starts, self.contact_proxy_world_ends],
                device=self.device,
            )
            self.viewer.log_lines(
                "/contact_proxy_meshes",
                self.contact_proxy_world_starts,
                self.contact_proxy_world_ends,
                self.contact_proxy_colors,
                width=CONTACT_PROXY_LINE_WIDTH,
            )
        self.viewer.end_frame()


def _write_run_metadata(
    frame_count: int,
    asset_directory: Path,
    spawned_object_count: int,
    contact_proxy_mesh_count: int,
) -> None:
    metadata = {
        "baseline": "newton.examples.robot.example_robot_policy",
        "robot": ROBOT_NAME,
        "newton_version": newton.__version__,
        "asset_directory": str(asset_directory),
        "newton_dt": NEWTON_DT,
        "policy_decimation": POLICY_DECIMATION,
        "policy_dt": POLICY_DT,
        "render_fps": RENDER_FPS,
        "scene_sky_upper_color": SCENE_SKY_UPPER_COLOR,
        "scene_sky_lower_color": SCENE_SKY_LOWER_COLOR,
        "scene_light_color": SCENE_LIGHT_COLOR,
        "requested_frames": NUM_FRAMES,
        "completed_frames": frame_count,
        "sim_duration": frame_count * FRAME_DT,
        "robot_count": len(ROBOT_INITIAL_POSES),
        "default_walk_commands": DEFAULT_WALK_COMMANDS.tolist(),
        "scripted_fight_enabled": ENABLE_SCRIPTED_FIGHT,
        "fight_approach_end_time": FIGHT_APPROACH_END_TIME if ENABLE_SCRIPTED_FIGHT else None,
        "fight_attack_sequence": list(FIGHT_ATTACK_SEQUENCE) if ENABLE_SCRIPTED_FIGHT else [],
        "use_robot_visual_mesh_colliders": USE_ROBOT_VISUAL_MESH_COLLIDERS,
        "dynamic_contact_box_count": len(BOX_POSES) if ENABLE_DYNAMIC_CONTACT_BOXES else 0,
        "interactive_object_pool_size": INTERACTIVE_OBJECT_POOL_SIZE if ENABLE_INTERACTIVE_OBJECT_SPAWNING else 0,
        "interactively_spawned_object_count": spawned_object_count,
        "show_warehouse_environment": SHOW_WAREHOUSE_ENVIRONMENT,
        "show_contact_proxy_meshes": SHOW_CONTACT_PROXY_MESHES,
        "contact_proxy_mesh_path": str(CONTACT_PROXY_MESH_PATH),
        "contact_proxy_mesh_count": contact_proxy_mesh_count,
        "contact_proxy_padding": CONTACT_PROXY_PADDING,
        "deme_robot_contact_enabled": ENABLE_DEME_ROBOT_CONTACT,
        "deme_dt": DEME_DT if ENABLE_DEME_ROBOT_CONTACT else None,
        "deme_substeps_per_newton_step": DEME_SUBSTEPS if ENABLE_DEME_ROBOT_CONTACT else 0,
        "deme_robot_families": list(DEME_ROBOT_FAMILIES) if ENABLE_DEME_ROBOT_CONTACT else [],
        "deme_contact_material": DEME_CONTACT_MATERIAL if ENABLE_DEME_ROBOT_CONTACT else None,
    }
    RUN_METADATA_OUTPUT_PATH.write_text(json.dumps(metadata, indent=4) + "\n", encoding="utf-8")


def _write_visual_mesh_manifest(visual_meshes: list[dict]) -> None:
    """Write serializable proof that the high-resolution robot meshes are available."""
    manifest = [
        {
            key: mesh_info[key]
            for key in (
                "robot_instance",
                "shape_index",
                "shape_label",
                "body_index",
                "body_label",
                "vertex_count",
                "triangle_count",
            )
        }
        for mesh_info in visual_meshes
    ]
    VISUAL_MESH_MANIFEST_OUTPUT_PATH.write_text(json.dumps(manifest, indent=4) + "\n", encoding="utf-8")


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
        asset_directory = mophi.download_newton_asset(
            robot_config.asset_dir,
            [robot_config.yaml_path, robot_config.policy_path["mjw"], robot_config.asset_path],
        )
    except RuntimeError as exc:
        mophi.fatal(f"Newton could not prepare the Unitree G1 assets.\n{exc}")
    print(f"[Assets] Ready at '{asset_directory}'.")

    yaml_path = asset_directory / robot_config.yaml_path
    with yaml_path.open(encoding="utf-8") as yaml_file:
        config = yaml.safe_load(yaml_file)

    policy_path = asset_directory / robot_config.policy_path["mjw"]
    model_path = asset_directory / robot_config.asset_path

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
    example.configure_policy_instances()
    example.initialize_deme_contact()
    if not np.isclose(example.frame_dt, NEWTON_DT, rtol=0.0, atol=1.0e-12):
        mophi.fatal(f"Newton G1 policy timestep changed: expected {NEWTON_DT}, got {example.frame_dt}.")
    print(
        f"[Contact] Newton model contains {example.robot_count} robot(s), "
        f"{example.robot_mesh_proxy_count} mesh-derived robot contact proxies, and "
        f"{example.contact_box_count} initial dynamic boxes."
    )
    if example.interactive_object_joint_starts:
        print(
            f"[Contact] Press '{INTERACTIVE_OBJECT_SPAWN_KEY.upper()}' to spawn up to "
            f"{len(example.interactive_object_joint_starts)} additional boxes."
        )
    _write_visual_mesh_manifest(example.robot_visual_meshes)
    print(
        f"[Meshes] Validated {len(example.robot_visual_meshes)} visual body meshes and wrote "
        f"'{VISUAL_MESH_MANIFEST_OUTPUT_PATH}'."
    )

    if hasattr(viewer, "set_camera"):
        viewer.set_camera(wp.vec3(*CAMERA_POSITION), CAMERA_PITCH, CAMERA_YAW)
    _configure_viewer_scene_lighting(viewer)
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
        f"({SIM_SUBSTEPS} policy step(s) x {POLICY_DECIMATION} Newton substeps x "
        f"{NEWTON_DT * 1000.0:.1f} ms per rendered frame) ..."
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

    _write_run_metadata(
        completed_frames,
        asset_directory,
        example.spawned_object_count,
        len(example.contact_mesh_proxies),
    )
    print(f"[Output] Saved run metadata to '{RUN_METADATA_OUTPUT_PATH}'.")
    if SAVE_MOVIE:
        print(f"[Movie] Saved simulation recording to '{MOVIE_OUTPUT_PATH}'.")


if __name__ == "__main__":
    main()
