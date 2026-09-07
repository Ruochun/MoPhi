"""Compare DEME acceleration-derived and direct-wrench contact feedback in Newton.

Two isolated, otherwise identical box-collision lanes are simulated in one
Newton--DEME co-simulation. The lower lane reconstructs contact wrenches from
DEME's contact-acceleration getters; the upper lane uses DEME's reduced-wrench
getter. All per-step physics payloads remain in CUDA memory. The reported A/B
comparison is based on Newton body motion.
"""

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import warp as wp

import mophi
import newton
from mophi.couplers.newton_deme import NewtonDEMEContactCoupler, NewtonDEMEOwnerMap
from mophi.utils import load_package_provider

DEME = load_package_provider("deme")

# =============================================================================
# Simulation configuration
# All user-tunable constants are defined here.
# =============================================================================

# -- Timing -------------------------------------------------------------------
NEWTON_DT = 1.0 / 500.0
DEME_DT = 1.0 / 2000.0
RENDER_FPS = 50
SIM_DURATION_SECONDS = 0.7

# -- Newton and DEME bodies ---------------------------------------------------
BOX_HALF_EXTENTS = (0.20, 0.20, 0.20)
BOX_MASS = 2.0
BOX_PRINCIPAL_MOI = (0.08, 0.11, 0.14)
INITIAL_X_SEPARATION = 1.30
COLLISION_Y_OFFSET = 0.06
INITIAL_SPEED = 1.0
INITIAL_ANGULAR_SPEED = 0.0
LANE_CENTER_Y = (-0.75, 0.75)
LANE_LABELS = ("contact_acceleration", "direct_wrench")
DEME_LANE_FAMILIES = ((10, 11), (20, 21))
DEME_DOMAIN_X = (-2.0, 2.0)
DEME_DOMAIN_Y = (-1.5, 1.5)
DEME_DOMAIN_Z = (-1.0, 1.0)
DEME_CONTACT_MATERIAL = {"E": 2.0e6, "nu": 0.3, "CoR": 0.25, "mu": 0.45, "Crr": 0.0}
CONTACT_PROXY_MESH_PATH = Path(__file__).resolve().parents[3] / "data" / "mesh" / "cube.obj"

# ``mass_moi`` is the physically consistent acceleration-to-wrench conversion.
# ``assumed_unit_mass_moi`` deliberately treats acceleration as force and
# angular acceleration as torque by assuming unit mass and unit MOI.
ACCELERATION_FEEDBACK_MODE = "mass_moi"  # Supported values: "mass_moi", "assumed_unit_mass_moi".

# -- Visualization ------------------------------------------------------------
ENABLE_VISUALIZATION = True
SAVE_MOVIE = False
CAMERA_POSITION = (3.3, -3.5, 2.5)
CAMERA_PITCH = -18.0
CAMERA_YAW = 132.0
REFERENCE_AXIS_ORIGIN = (1.25, -1.25, -0.19)
REFERENCE_SCALE_BAR_CENTER = (0.15, -1.25, -0.16)
ACCELERATION_LANE_COLOR = (0.15, 0.75, 1.0)
WRENCH_LANE_COLOR = (1.0, 0.45, 0.12)

# -- Output -------------------------------------------------------------------
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "output" / Path(__file__).stem
MOTION_OUTPUT_PATH = OUTPUT_DIRECTORY / f"newton_motion_comparison_{ACCELERATION_FEEDBACK_MODE}.npz"
SUMMARY_OUTPUT_PATH = OUTPUT_DIRECTORY / f"comparison_summary_{ACCELERATION_FEEDBACK_MODE}.json"
MOVIE_OUTPUT_PATH = OUTPUT_DIRECTORY / "deme_getter_comparison.mp4"
MOVIE_FPS = RENDER_FPS

# -- Derived constants --------------------------------------------------------
FRAME_DT = 1.0 / RENDER_FPS
NEWTON_SUBSTEPS_PER_FRAME = int(round(FRAME_DT / NEWTON_DT))
DEME_SUBSTEPS_PER_NEWTON_STEP = int(round(NEWTON_DT / DEME_DT))
NUM_FRAMES = int(round(SIM_DURATION_SECONDS / FRAME_DT))
BOX_SIZE = 2.0 * BOX_HALF_EXTENTS[0]
BODY_COUNT = 4
ACCELERATION_BODY_INDICES = (0, 1)
WRENCH_BODY_INDICES = (2, 3)


if ACCELERATION_FEEDBACK_MODE not in ("mass_moi", "assumed_unit_mass_moi"):
    raise ValueError(
        "ACCELERATION_FEEDBACK_MODE must be 'mass_moi' or 'assumed_unit_mass_moi', "
        f"got {ACCELERATION_FEEDBACK_MODE!r}."
    )
if not np.allclose(BOX_HALF_EXTENTS, BOX_HALF_EXTENTS[0]):
    raise ValueError("The OBJ scaling path currently requires a cube; BOX_HALF_EXTENTS must be equal.")
if not np.isclose(NEWTON_SUBSTEPS_PER_FRAME * NEWTON_DT, FRAME_DT, rtol=0.0, atol=1.0e-12):
    raise ValueError("FRAME_DT must be an integer multiple of NEWTON_DT.")
if not np.isclose(DEME_SUBSTEPS_PER_NEWTON_STEP * DEME_DT, NEWTON_DT, rtol=0.0, atol=1.0e-12):
    raise ValueError("NEWTON_DT must be an integer multiple of DEME_DT.")


@wp.kernel
def _compose_comparison_feedback(
    contact_accelerations: wp.array(dtype=wp.vec3),
    contact_angular_accelerations: wp.array(dtype=wp.vec3),
    direct_forces: wp.array(dtype=wp.vec3),
    direct_torques: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.quat),
    mass: float,
    principal_moi: wp.vec3,
    assume_unit_mass_moi: int,
    body_forces: wp.array(dtype=wp.spatial_vector),
):
    body_idx = wp.tid()
    if body_idx < 2:
        force = contact_accelerations[body_idx]
        torque = contact_angular_accelerations[body_idx]
        if assume_unit_mass_moi == 0:
            force = mass * force
            q = orientations[body_idx]
            alpha_local = wp.quat_rotate_inv(q, torque)
            torque_local = wp.cw_mul(principal_moi, alpha_local)
            torque = wp.quat_rotate(q, torque_local)
        body_forces[body_idx] = wp.spatial_vector(force, torque)
    else:
        body_forces[body_idx] = wp.spatial_vector(direct_forces[body_idx], direct_torques[body_idx])


def _initial_body_specs() -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """Return matching lane-local poses and opposing velocities."""
    specs = []
    half_separation = 0.5 * INITIAL_X_SEPARATION
    for lane_center in LANE_CENTER_Y:
        specs.extend(
            [
                ((-half_separation, lane_center - COLLISION_Y_OFFSET, 0.0), (INITIAL_SPEED, 0.0, 0.0)),
                ((half_separation, lane_center + COLLISION_Y_OFFSET, 0.0), (-INITIAL_SPEED, 0.0, 0.0)),
            ]
        )
    return specs


def _build_newton_system():
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    inertia = wp.mat33(
        BOX_PRINCIPAL_MOI[0],
        0.0,
        0.0,
        0.0,
        BOX_PRINCIPAL_MOI[1],
        0.0,
        0.0,
        0.0,
        BOX_PRINCIPAL_MOI[2],
    )
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False)
    body_indices = []
    shape_indices = []
    for body_idx, (position, velocity) in enumerate(_initial_body_specs()):
        body = builder.add_body(
            xform=wp.transform(wp.vec3(*position), wp.quat_identity()),
            mass=BOX_MASS,
            inertia=inertia,
            lock_inertia=True,
            label=f"{LANE_LABELS[body_idx // 2]}_box_{body_idx % 2}",
        )
        builder.joint_qd[builder.joint_qd_start[-1] : builder.joint_qd_start[-1] + 6] = [
            velocity[0],
            velocity[1],
            velocity[2],
            0.0,
            0.0,
            INITIAL_ANGULAR_SPEED if body_idx % 2 == 0 else -INITIAL_ANGULAR_SPEED,
        ]
        shape = builder.add_shape_box(
            body,
            hx=BOX_HALF_EXTENTS[0],
            hy=BOX_HALF_EXTENTS[1],
            hz=BOX_HALF_EXTENTS[2],
            cfg=shape_cfg,
            label=f"comparison_box_shape_{body_idx}",
        )
        body_indices.append(body)
        shape_indices.append(shape)

    model = builder.finalize()
    model.set_gravity((0.0, 0.0, 0.0))
    solver = newton.solvers.SolverMuJoCo(
        model,
        disable_contacts=True,
        use_mujoco_cpu=False,
        solver="newton",
        nconmax=64,
        njmax=128,
    )
    return model, solver, body_indices, shape_indices


def _build_deme_system(device, initial_specs):
    solver = DEME.DEMSolver([device.ordinal])
    if not CONTACT_PROXY_MESH_PATH.is_file():
        mophi.fatal(f"Required contact proxy mesh is missing: {CONTACT_PROXY_MESH_PATH}")

    material = solver.LoadMaterial(DEME_CONTACT_MATERIAL)
    trackers = []
    flat_families = [family for lane_families in DEME_LANE_FAMILIES for family in lane_families]
    for body_idx, (position, _) in enumerate(initial_specs):
        mesh = solver.AddWavefrontMeshObject(str(CONTACT_PROXY_MESH_PATH), material, False)
        mesh.Scale(BOX_SIZE)
        mesh.SetInitPos(position)
        mesh.SetInitQuat([0.0, 0.0, 0.0, 1.0])
        mesh.SetFamily(flat_families[body_idx])
        mesh.SetMass(BOX_MASS)
        mesh.SetMOI(list(BOX_PRINCIPAL_MOI))
        trackers.append(solver.Track(mesh))

    for family in flat_families:
        solver.SetFamilyPrescribedPosition(family)
        solver.SetFamilyPrescribedQuaternion(family)
        solver.SetFamilyPrescribedLinVel(family)
        solver.SetFamilyPrescribedAngVel(family)
        solver.DisableContactBetweenFamilies(family, family)
    for family_a in DEME_LANE_FAMILIES[0]:
        for family_b in DEME_LANE_FAMILIES[1]:
            solver.DisableContactBetweenFamilies(family_a, family_b)

    solver.SetMeshUniversalContact(True)
    solver.InstructBoxDomainDimension(DEME_DOMAIN_X, DEME_DOMAIN_Y, DEME_DOMAIN_Z)
    solver.SetGravitationalAcceleration([0.0, 0.0, 0.0])
    solver.SetInitTimeStep(DEME_DT)
    solver.SetErrorOutAvgContacts(1000)
    solver.Initialize()

    owner_ids = [int(tracker.GetOwnerID()) for tracker in trackers]
    expected_ids = list(range(owner_ids[0], owner_ids[0] + BODY_COUNT))
    if owner_ids != expected_ids:
        mophi.fatal(f"Comparison owners must be consecutive; got {owner_ids}.")
    return solver, owner_ids


def _lane_motion_errors(body_q: np.ndarray, body_qd: np.ndarray) -> dict[str, float]:
    max_position = 0.0
    max_orientation = 0.0
    max_linear_velocity = 0.0
    max_angular_velocity = 0.0
    lane_delta = np.asarray([0.0, LANE_CENTER_Y[1] - LANE_CENTER_Y[0], 0.0], dtype=np.float32)
    for acceleration_idx, wrench_idx in zip(ACCELERATION_BODY_INDICES, WRENCH_BODY_INDICES):
        max_position = max(
            max_position,
            float(np.linalg.norm((body_q[wrench_idx, :3] - lane_delta) - body_q[acceleration_idx, :3])),
        )
        quaternion_dot = float(abs(np.dot(body_q[acceleration_idx, 3:7], body_q[wrench_idx, 3:7])))
        max_orientation = max(max_orientation, 2.0 * float(np.arccos(np.clip(quaternion_dot, 0.0, 1.0))))
        max_linear_velocity = max(
            max_linear_velocity, float(np.linalg.norm(body_qd[wrench_idx, :3] - body_qd[acceleration_idx, :3]))
        )
        max_angular_velocity = max(
            max_angular_velocity, float(np.linalg.norm(body_qd[wrench_idx, 3:] - body_qd[acceleration_idx, 3:]))
        )
    return {
        "position_m": max_position,
        "orientation_rad": max_orientation,
        "linear_velocity_m_per_s": max_linear_velocity,
        "angular_velocity_rad_per_s": max_angular_velocity,
    }


def main() -> None:
    if not hasattr(mophi, "NewtonXLBDEMCoupler"):
        mophi.fatal("NewtonXLBDEMCoupler is unavailable. Rebuild with -DMOPHI_BUILD_NEWTON_XLB_DEM=ON.")
    mophi.check_newton_warp_mujoco_versions(
        mophi.REQUIRED_NEWTON_VERSION,
        mophi.REQUIRED_WARP_VERSION,
        mophi.REQUIRED_MUJOCO_VERSION,
    )
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    wp.init()
    device = wp.get_device()
    if not device.is_cuda:
        mophi.fatal("The DEME getter comparison requires a CUDA Warp device.")

    initial_specs = _initial_body_specs()
    model, newton_solver, body_indices, shape_indices = _build_newton_system()
    if body_indices != list(range(BODY_COUNT)):
        mophi.fatal(f"Expected comparison bodies [0, 1, 2, 3], got {body_indices}.")
    deme_solver, owner_ids = _build_deme_system(device, initial_specs)

    coupler = mophi.NewtonXLBDEMCoupler()
    coupler.initialize(model, newton_solver, None, deme_solver, NEWTON_DT)
    contact_exchange = NewtonDEMEContactCoupler()
    contact_exchange.initialize(
        model,
        deme_solver,
        NewtonDEMEOwnerMap(body_indices, owner_ids),
        device,
    )

    contact_accelerations = wp.empty(BODY_COUNT, dtype=wp.vec3, device=device)
    contact_angular_accelerations = wp.empty(BODY_COUNT, dtype=wp.vec3, device=device)
    direct_forces = wp.empty(BODY_COUNT, dtype=wp.vec3, device=device)
    direct_torques = wp.empty(BODY_COUNT, dtype=wp.vec3, device=device)
    newton_body_forces = wp.zeros(BODY_COUNT, dtype=wp.spatial_vector, device=device)
    coupler.set_newton_body_forces(newton_body_forces)

    visualizer = mophi.OpenGLVisualizer(model) if ENABLE_VISUALIZATION else None
    if visualizer is not None:
        visualizer.set_camera(wp.vec3(*CAMERA_POSITION), CAMERA_PITCH, CAMERA_YAW)
        visualizer.update_shape_colors(
            {
                shape_indices[0]: ACCELERATION_LANE_COLOR,
                shape_indices[1]: ACCELERATION_LANE_COLOR,
                shape_indices[2]: WRENCH_LANE_COLOR,
                shape_indices[3]: WRENCH_LANE_COLOR,
            }
        )
        mophi.log_orientation_and_scale_reference(
            visualizer,
            axis_origin=REFERENCE_AXIS_ORIGIN,
            scale_bar_center=REFERENCE_SCALE_BAR_CENTER,
        )
    movie_writer = imageio.get_writer(MOVIE_OUTPUT_PATH, fps=MOVIE_FPS) if SAVE_MOVIE else None

    times = []
    body_q_history = []
    body_qd_history = []
    error_history = []
    completed_frames = 0
    assume_unit_mass_moi = int(ACCELERATION_FEEDBACK_MODE == "assumed_unit_mass_moi")

    def exchange_and_step() -> None:
        contact_exchange.set_deme_owner_state_from_newton(coupler.newton_state_0)
        contact_exchange.step_deme(DEME_SUBSTEPS_PER_NEWTON_STEP)
        contact_exchange.get_deme_contact_accelerations_to_device(
            contact_accelerations,
            contact_angular_accelerations,
            angular_frame="global",
        )
        contact_exchange.get_deme_contact_wrenches_to_device(direct_forces, direct_torques)
        newton_body_forces.zero_()
        wp.launch(
            _compose_comparison_feedback,
            dim=BODY_COUNT,
            inputs=[
                contact_accelerations,
                contact_angular_accelerations,
                direct_forces,
                direct_torques,
                contact_exchange.deme_owner_orientations,
                BOX_MASS,
                wp.vec3(*BOX_PRINCIPAL_MOI),
                assume_unit_mass_moi,
                newton_body_forces,
            ],
            device=device,
        )
        coupler.step_newton()

    try:
        for frame in range(NUM_FRAMES):
            if visualizer is not None and not visualizer.is_running():
                break
            for _ in range(NEWTON_SUBSTEPS_PER_FRAME):
                exchange_and_step()
            sim_time = (frame + 1) * FRAME_DT
            body_q = coupler.newton_state_0.body_q.numpy()
            body_qd = coupler.newton_state_0.body_qd.numpy()
            errors = _lane_motion_errors(body_q, body_qd)
            times.append(sim_time)
            body_q_history.append(body_q)
            body_qd_history.append(body_qd)
            error_history.append([errors[key] for key in errors])
            completed_frames = frame + 1

            if visualizer is not None:
                visualizer.begin_frame(sim_time)
                visualizer.log_state(coupler.newton_state_0)
                visualizer.end_frame()
                if movie_writer is not None:
                    movie_writer.append_data(visualizer.get_frame().numpy())
    finally:
        if movie_writer is not None:
            movie_writer.close()
        if visualizer is not None:
            visualizer.close()
        contact_exchange.finalize()
        coupler.finalize()

    error_names = (
        "position_m",
        "orientation_rad",
        "linear_velocity_m_per_s",
        "angular_velocity_rad_per_s",
    )
    error_array = np.asarray(error_history, dtype=np.float64).reshape((-1, len(error_names)))
    max_errors = {
        name: float(error_array[:, error_idx].max()) if len(error_array) else None
        for error_idx, name in enumerate(error_names)
    }
    final_errors = (
        {name: float(error_array[-1, error_idx]) for error_idx, name in enumerate(error_names)}
        if len(error_array)
        else {name: None for name in error_names}
    )
    np.savez_compressed(
        MOTION_OUTPUT_PATH,
        time=np.asarray(times, dtype=np.float64),
        body_q=np.asarray(body_q_history, dtype=np.float32),
        body_qd=np.asarray(body_qd_history, dtype=np.float32),
        error_names=np.asarray(error_names),
        motion_errors=error_array,
    )
    summary = {
        "completed_frames": completed_frames,
        "acceleration_feedback_mode": ACCELERATION_FEEDBACK_MODE,
        "newton_dt": NEWTON_DT,
        "deme_dt": DEME_DT,
        "box_mass": BOX_MASS,
        "box_principal_moi": BOX_PRINCIPAL_MOI,
        "max_newton_motion_errors": max_errors,
        "final_newton_motion_errors": final_errors,
        "motion_output": str(MOTION_OUTPUT_PATH),
    }
    SUMMARY_OUTPUT_PATH.write_text(json.dumps(summary, indent=4) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    main()
