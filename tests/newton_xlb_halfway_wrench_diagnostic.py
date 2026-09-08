"""Focused GPU diagnostic for ad-hoc Newton--XLB halfway-bounce-back feedback.

This is an investigation script, not a correctness test for moving-wall LBM.
It isolates force clearing, equilibrium symmetry, background-flow loading,
moving-mask impulses, force-only feedback, and torque-only feedback.

TODO: Remove this diagnostic with the ad-hoc path once true moving-boundary
treatment and force feedback are available.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import warp as wp

import mophi
import newton
import xlb
from mophi.couplers.newton_xlb import (
    NewtonBodyBoxBoundary,
    NewtonXLBHalfwayBounceBackWrench,
    NewtonXLBWrenchExchange,
    XLBPhysicalScaling,
    XLBStepperState,
    velocity_stencil_to_warp,
)
from xlb.compute_backend import ComputeBackend
from xlb.grid import grid_factory
from xlb.operator.boundary_condition import ExtrapolationOutflowBC, HalfwayBounceBackBC, ZouHeBC
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.precision_policy import PrecisionPolicy

# =============================================================================
# Configuration
# =============================================================================

NEWTON_DT = 5.0e-3
XLB_DT = 1.0e-3
XLB_SUBSTEPS = int(round(NEWTON_DT / XLB_DT))
GRID_SHAPE = (24, 24, 24)
DOMAIN_MIN = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
DOMAIN_MAX = np.array([1.0, 1.0, 1.0], dtype=np.float64)
BOX_HALF_EXTENTS = (0.25, 0.25, 0.25)
BOX_MASS = 10.0
BOX_PRINCIPAL_MOI = (0.4166667, 0.4166667, 0.4166667)
OFFSET_COM = (0.0, 0.10, 0.0)
PHYSICAL_DENSITY = 1.0
# This deliberately viscous diagnostic fluid keeps BGK relaxation comfortably
# away from its tau=0.5 stability boundary. It is not intended to model air.
PHYSICAL_KINEMATIC_VISCOSITY = 2.314814814814815e-1
FLOW_SPEED = 0.625
FLOW_WARMUP_STEPS = 80
FEEDBACK_NEWTON_STEPS = 40
MASK_TRANSLATION_PER_NEWTON_STEP = 0.02
PULSE_FORCE = (10.0, 0.0, 0.0)
PULSE_TORQUE = (0.0, 0.0, 1.0)
OUTPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
OUTPUT_PATH = OUTPUT_DIRECTORY / "diagnostics.json"

XLB_SCALING = XLBPhysicalScaling.from_domain(
    DOMAIN_MIN, DOMAIN_MAX, GRID_SHAPE, XLB_DT, PHYSICAL_KINEMATIC_VISCOSITY, PHYSICAL_DENSITY
)
CELL_SIZE = XLB_SCALING.cell_size
if not np.isclose(XLB_SUBSTEPS * XLB_DT, NEWTON_DT):
    raise ValueError("NEWTON_DT must be an integer multiple of XLB_DT.")
LATTICE_NU = XLB_SCALING.lattice_kinematic_viscosity
OMEGA = XLB_SCALING.omega
RELAXATION_TIME = XLB_SCALING.relaxation_time
if RELAXATION_TIME < 0.55:
    raise ValueError("The diagnostic BGK relaxation time must remain safely above 0.5.")


@wp.kernel
def _set_body_position(body_q: wp.array(dtype=wp.transform), position: wp.vec3):
    transform = body_q[0]
    body_q[0] = wp.transform(position, wp.transform_get_rotation(transform))


def _vec(array) -> list[float]:
    return array.numpy()[0].astype(float).tolist()


def _build_newton(com_offset=(0.0, 0.0, 0.0)):
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
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0), wp.quat_identity()),
        com=wp.vec3(*com_offset),
        mass=BOX_MASS,
        inertia=inertia,
        lock_inertia=True,
        label="diagnostic_box",
    )
    builder.add_shape_box(
        body,
        hx=BOX_HALF_EXTENTS[0],
        hy=BOX_HALF_EXTENTS[1],
        hz=BOX_HALF_EXTENTS[2],
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False),
    )
    model = builder.finalize()
    model.set_gravity((0.0, 0.0, 0.0))
    solver = newton.solvers.SolverMuJoCo(
        model,
        disable_contacts=True,
        use_mujoco_cpu=False,
        solver="newton",
        nconmax=16,
        njmax=32,
    )
    coupler = mophi.NewtonXLBDEMCoupler()
    coupler.initialize(model, solver, None, None, NEWTON_DT)
    return model, coupler


def _robot_indices(center=(0.0, 0.0, 0.0)):
    lo = np.floor((np.asarray(center) - BOX_HALF_EXTENTS - DOMAIN_MIN) / CELL_SIZE).astype(int)
    hi = np.floor((np.asarray(center) + BOX_HALF_EXTENTS - DOMAIN_MIN) / CELL_SIZE).astype(int)
    axes = [np.arange(lo[axis], hi[axis] + 1) for axis in range(3)]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    return [x.ravel().tolist(), y.ravel().tolist(), z.ravel().tolist()], lo, hi


def _build_xlb(model, coupler, flow_speed: float):
    device = coupler.newton_state_0.body_q.device
    precision = PrecisionPolicy.FP32FP32
    backend = ComputeBackend.WARP
    velocity_set = xlb.velocity_set.D3Q19(precision_policy=precision, compute_backend=backend)
    xlb.init(velocity_set, backend, precision)
    grid = grid_factory(GRID_SHAPE, compute_backend=backend)
    box = grid.bounding_box_indices()
    box_no_edges = grid.bounding_box_indices(remove_edges=True)
    if flow_speed == 0.0:
        wall_faces = ("bottom", "top", "left", "right", "front", "back")
        boundary_conditions = []
    else:
        wall_faces = ("bottom", "top")
        inlet_speed = float(XLB_SCALING.velocity_to_lattice(flow_speed))
        boundary_conditions = [
            ZouHeBC(
                bc_type="velocity",
                prescribed_value=np.array([inlet_speed, 0.0, 0.0]),
                indices=box_no_edges["left"],
            ),
            ExtrapolationOutflowBC(indices=box_no_edges["right"]),
        ]
    walls = [sum((box[face][axis] for face in wall_faces), []) for axis in range(velocity_set.d)]
    walls = np.unique(np.asarray(walls), axis=-1).tolist()
    wall = HalfwayBounceBackBC(indices=walls)
    robot_indices, grid_min, grid_max = _robot_indices()
    robot = HalfwayBounceBackBC(indices=robot_indices)
    stepper = IncompressibleNavierStokesStepper(
        grid=grid,
        boundary_conditions=[wall, *boundary_conditions, robot],
        collision_type="BGK",
    )
    runtime = XLBStepperState().initialize(stepper, OMEGA)
    bc_mask, missing_mask = runtime.bc_mask, runtime.missing_mask
    stencil = velocity_stencil_to_warp(velocity_set, device)
    boundary = NewtonBodyBoxBoundary()
    boundary.initialize(
        bc_mask,
        missing_mask,
        stencil,
        body_index=0,
        domain_min=DOMAIN_MIN,
        domain_max=DOMAIN_MAX,
        lower_extents=BOX_HALF_EXTENTS,
        boundary_id=robot.id,
        initial_grid_min=grid_min,
        initial_grid_max=grid_max,
    )
    reducer = NewtonXLBHalfwayBounceBackWrench()
    reducer.initialize(
        body_index=0,
        boundary_id=robot.id,
        domain_min=DOMAIN_MIN,
        cell_size=CELL_SIZE,
        physical_density=PHYSICAL_DENSITY,
        xlb_dt=XLB_DT,
        lattice_velocities=stencil,
        opposite_indices=np.asarray(velocity_set.opp_indices, dtype=np.int32),
        device=device,
    )
    exchange = NewtonXLBWrenchExchange()
    exchange.initialize(model, [0], device)
    return runtime, boundary, reducer, exchange


def _step_xlb(system):
    system[0].step()


def _sample_wrench(system, body_q):
    runtime, _, reducer, _ = system
    reducer.reduce(runtime.f0, runtime.bc_mask, runtime.missing_mask, body_q)
    wp.synchronize()
    return {"force": _vec(reducer.forces), "torque": _vec(reducer.torques)}


def _run_newton_pulse():
    model, coupler = _build_newton()
    device = coupler.newton_state_0.body_q.device
    exchange = NewtonXLBWrenchExchange()
    exchange.initialize(model, [0], device)
    force = wp.array([PULSE_FORCE], dtype=wp.vec3, device=device)
    torque = wp.array([PULSE_TORQUE], dtype=wp.vec3, device=device)
    zero = wp.zeros(1, dtype=wp.vec3, device=device)
    body_forces = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=device)
    coupler.set_newton_body_forces(body_forces)
    history = []
    for step in range(4):
        exchange.write_xlb_wrenches_to_newton(
            force if step == 0 else zero,
            torque if step == 0 else zero,
            body_forces,
            clear_destination=True,
        )
        coupler.step_newton()
        history.append(coupler.newton_state_0.body_qd.numpy()[0].astype(float).tolist())
    exchange.finalize()
    coupler.finalize()
    return {"body_velocity_after_each_step": history}


def _run_xlb_case(flow_speed: float, move_mask: bool, feedback_mode: str, com_offset=(0.0, 0.0, 0.0)):
    model, coupler = _build_newton(com_offset)
    system = list(_build_xlb(model, coupler, flow_speed))
    boundary, reducer, exchange = system[1], system[2], system[3]
    body_forces = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=reducer.device)
    zero = wp.zeros(1, dtype=wp.vec3, device=reducer.device)
    coupler.set_newton_body_forces(body_forces)
    for _ in range(FLOW_WARMUP_STEPS):
        _step_xlb(system)
    initial_wrench = _sample_wrench(system, coupler.newton_state_0.body_q)
    initial_force = np.asarray(initial_wrench["force"])
    torque_about_com = np.asarray(initial_wrench["torque"]) - np.cross(np.asarray(com_offset), initial_force)
    history = []
    for step in range(FEEDBACK_NEWTON_STEPS):
        body_forces.zero_()
        if feedback_mode != "off":
            exchange.write_xlb_wrenches_to_newton(
                reducer.forces if feedback_mode != "torque_only" else zero,
                reducer.torques if feedback_mode != "force_only" else zero,
                body_forces,
                clear_destination=False,
            )
        coupler.step_newton()
        if move_mask:
            position = wp.vec3(MASK_TRANSLATION_PER_NEWTON_STEP * float(step + 1), 0.0, 0.0)
            wp.launch(
                _set_body_position,
                dim=1,
                inputs=[coupler.newton_state_0.body_q, position],
                device=reducer.device,
            )
        boundary.update_from_newton(coupler.newton_state_0.body_q)
        for _ in range(XLB_SUBSTEPS):
            _step_xlb(system)
        wrench = _sample_wrench(system, coupler.newton_state_0.body_q)
        history.append(
            {
                "step": step,
                "force": wrench["force"],
                "torque": wrench["torque"],
                "body_velocity": coupler.newton_state_0.body_qd.numpy()[0].astype(float).tolist(),
            }
        )
    boundary.finalize()
    reducer.finalize()
    exchange.finalize()
    system[0].finalize()
    coupler.finalize()
    return {
        "initial_wrench_about_body_origin": initial_wrench,
        "same_initial_torque_shifted_to_com": torque_about_com.astype(float).tolist(),
        "history": history,
    }


def _run_named_case(name):
    cases = {
        "newton_one_step_pulse": _run_newton_pulse,
        "equilibrium_fixed_mask": lambda: _run_xlb_case(0.0, False, "off"),
        "flow_fixed_mask": lambda: _run_xlb_case(FLOW_SPEED, False, "off"),
        "zero_flow_moving_mask": lambda: _run_xlb_case(0.0, True, "off"),
        "flow_force_only_feedback": lambda: _run_xlb_case(FLOW_SPEED, False, "force_only"),
        "flow_torque_only_feedback": lambda: _run_xlb_case(FLOW_SPEED, False, "torque_only"),
        "flow_full_feedback": lambda: _run_xlb_case(FLOW_SPEED, False, "full"),
        "flow_offset_com_full_feedback": lambda: _run_xlb_case(FLOW_SPEED, False, "full", OFFSET_COM),
    }
    if name not in cases:
        raise ValueError(f"Unknown diagnostic case {name!r}.")
    return cases[name]()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="Run one isolated case instead of orchestrating the full suite.")
    parser.add_argument("--case-output", type=Path, help="JSON destination used by an isolated child case.")
    args = parser.parse_args()
    wp.init()
    if not wp.get_device().is_cuda:
        mophi.fatal("This diagnostic requires a CUDA Warp device shared by Newton and XLB.")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    if args.case:
        if args.case_output is None:
            raise ValueError("--case-output is required with --case.")
        args.case_output.write_text(json.dumps(_run_named_case(args.case), indent=2), encoding="utf-8")
        return

    case_names = (
        "newton_one_step_pulse",
        "equilibrium_fixed_mask",
        "flow_fixed_mask",
        "zero_flow_moving_mask",
        "flow_force_only_feedback",
        "flow_torque_only_feedback",
        "flow_full_feedback",
        "flow_offset_com_full_feedback",
    )
    results = {
        "configuration": {
            "newton_dt": NEWTON_DT,
            "xlb_dt": XLB_DT,
            "cell_size": CELL_SIZE,
            "flow_speed": FLOW_SPEED,
            "density": PHYSICAL_DENSITY,
            "physical_kinematic_viscosity": PHYSICAL_KINEMATIC_VISCOSITY,
            "lattice_velocity": FLOW_SPEED * XLB_DT / CELL_SIZE,
            "lattice_kinematic_viscosity": LATTICE_NU,
            "bgk_omega": OMEGA,
            "bgk_relaxation_time": RELAXATION_TIME,
            "process_isolation": "one fresh Python process per case",
        }
    }
    for name in case_names:
        case_path = OUTPUT_DIRECTORY / f"{name}.json"
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--case", name, "--case-output", str(case_path)],
            check=True,
        )
        results[name] = json.loads(case_path.read_text(encoding="utf-8"))
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote Newton--XLB wrench diagnostics to {OUTPUT_PATH}")
    for name, result in results.items():
        if name == "configuration" or "initial_wrench_about_body_origin" not in result:
            continue
        final = result["history"][-1]
        print(
            f"{name}: initial={result['initial_wrench_about_body_origin']}, "
            f"final_wrench={{'force': {final['force']}, 'torque': {final['torque']}}}"
        )


if __name__ == "__main__":
    main()
