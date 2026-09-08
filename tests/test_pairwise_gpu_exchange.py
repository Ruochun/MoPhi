import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from mophi.couplers.newton_deme import NewtonDEMEParticleExchange
from mophi.couplers.newton_xlb import (
    NewtonBodiesBoxBoundary,
    NewtonXLBHalfwayBounceBackWrench,
    XLBPhysicalScaling,
    XLBStepperState,
    bgk_omega_from_lattice_viscosity,
    prescribed_box_grid,
    velocity_stencil_array,
    world_to_grid_index,
)
from mophi.couplers.xlb_deme import XLBDEMEParticleExchange


class NewtonXLBGridTest(unittest.TestCase):
    def test_velocity_stencil_supports_xlb_matrix_objects(self):
        class Matrix:
            values = ((0, 1, -1), (0, 0, 0))

            def __getitem__(self, key):
                return self.values[key[0]][key[1]]

        class VelocitySet:
            c = Matrix()
            d = 2
            q = 3

        np.testing.assert_array_equal(velocity_stencil_array(VelocitySet()), [[0, 0], [1, 0], [-1, 0]])

    def test_stepper_state_owns_buffer_swap_and_timestep(self):
        class Stepper:
            def prepare_fields(self):
                return "f0", "f1", "bc", "missing"

            def __call__(self, f0, f1, bc, missing, omega, timestep):
                self.args = (f0, f1, bc, missing, omega, timestep)
                return "result0", "result1"

        stepper = Stepper()
        runtime = XLBStepperState().initialize(stepper, 1.25, timestep=4)
        self.assertEqual(runtime.step(), "result1")
        self.assertEqual(stepper.args, ("f0", "f1", "bc", "missing", 1.25, 4))
        self.assertEqual((runtime.f0, runtime.f1, runtime.timestep), ("result1", "result0", 5))

    def test_physical_scaling_maps_velocity_and_viscosity(self):
        scaling = XLBPhysicalScaling.from_domain([0, 0, 0], [1, 1, 1], [10, 10, 10], 0.01, 0.1)
        np.testing.assert_allclose(scaling.velocity_to_lattice([1, 0, 0]), [0.1, 0, 0])
        self.assertAlmostEqual(scaling.lattice_kinematic_viscosity, 0.1)
        self.assertAlmostEqual(scaling.omega, bgk_omega_from_lattice_viscosity(0.1))

    def test_world_mapping_clamps_to_interior(self):
        np.testing.assert_array_equal(
            world_to_grid_index([-2.0, 0.5, 3.0], [0, 0, 0], [1, 1, 1], [10, 20, 30]),
            [1, 10, 28],
        )

    def test_box_outside_domain_is_absent(self):
        self.assertEqual(prescribed_box_grid([2, 2, 2], [3, 3, 3], [0, 0, 0], [1, 1, 1], [8, 8, 8]), (None, None))

    def test_halfway_wrench_rejects_host_device(self):
        class HostDevice:
            is_cuda = False

        reducer = NewtonXLBHalfwayBounceBackWrench()
        with self.assertRaisesRegex(ValueError, "requires a CUDA"):
            reducer.initialize(0, 1, [0, 0, 0], 0.1, 1.0, 0.01, object(), [0], HostDevice())

    def test_multi_body_boundary_rejects_inconsistent_mapping(self):
        boundary = NewtonBodiesBoxBoundary()
        with self.assertRaisesRegex(ValueError, "equally sized"):
            boundary.initialize(object(), object(), object(), [0, 1], [(0.1, 0.1, 0.1)], [2, 3], [0] * 3, [1] * 3)


class _Device:
    is_cuda = True
    ordinal = 2


class _Array:
    def __init__(self, pointer):
        self.ptr = pointer
        self.dtype = None
        self.device = _Device()

    def __len__(self):
        return 3


class _Solver:
    def __init__(self):
        self.calls = []

    def GetGPUDeviceIDs(self):
        return [2]

    def GetOwnerPositionToDevice(self, *args):
        self.calls.append(("position", args))

    def GetOwnerVelocityToDevice(self, *args):
        self.calls.append(("velocity", args))

    def GetOwnerOriQToDevice(self, *args):
        self.calls.append(("orientation", args))

    def GetOwnerMassToDevice(self, *args):
        self.calls.append(("mass", args))

    def AddOwnerNextStepAccFromDevice(self, *args):
        self.calls.append(("linear_acceleration", args))

    def AddOwnerNextStepAngAccFromDevice(self, *args):
        self.calls.append(("angular_acceleration", args))


class ParticleExchangeTest(unittest.TestCase):
    def _exercise(self, exchange_type, module):
        solver = _Solver()
        pointers = [101, 202, 303, 404, 505] if exchange_type is NewtonDEMEParticleExchange else [101, 202, 303, 404]
        with patch(f"{module}.wp.empty", side_effect=[_Array(pointer) for pointer in pointers]):
            exchange = exchange_type()
            exchange.initialize(solver, 7, 3, _Device())
        self.assertEqual(exchange.read_deme_particle_state(), (exchange.positions, exchange.velocities))
        self.assertEqual(
            solver.calls,
            [
                ("mass", ((404 if exchange_type is NewtonDEMEParticleExchange else 303), 3, 2, 7, 3)),
                ("position", (101, 3, 2, 7, 3)),
                ("velocity", (202, 3, 2, 7, 3)),
            ],
        )
        if exchange_type is NewtonDEMEParticleExchange:
            self.assertEqual(exchange.read_deme_particle_poses(), (exchange.positions, exchange.orientations))
            self.assertEqual(solver.calls[-2:], [("position", (101, 3, 2, 7, 3)), ("orientation", (303, 3, 2, 7, 3))])
        accelerations = _Array(606)
        accelerations.dtype = wp.vec3
        accelerations.device = exchange.device
        with patch(f"{module}.wp.synchronize_device"):
            exchange.queue_deme_next_step_accelerations(accelerations)
        self.assertEqual(solver.calls[-1], ("linear_acceleration", (7, 606, 2, 3)))

    def test_newton_deme_device_calls(self):
        self._exercise(NewtonDEMEParticleExchange, "mophi.couplers.newton_deme.particles")

    def test_xlb_deme_device_calls(self):
        self._exercise(XLBDEMEParticleExchange, "mophi.couplers.xlb_deme.particles")


if __name__ == "__main__":
    unittest.main()
