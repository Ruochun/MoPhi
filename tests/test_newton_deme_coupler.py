import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import warp as wp

from mophi.couplers.newton_deme import (
    NewtonDEMEContactAccelerationCoupler,
    NewtonDEMEContactCoupler,
    NewtonDEMEMeshOwnerSpec,
    NewtonDEMEOwnerMap,
    NewtonDEMEOwnerPoseExchange,
    add_deme_mesh_owners,
    combine_triangle_meshes,
    extract_newton_shape_triangle_meshes,
    owner_map_from_mesh_bindings,
)


class NewtonDEMEOwnerMapTest(unittest.TestCase):
    def test_accepts_matching_consecutive_owner_span(self):
        owner_map = NewtonDEMEOwnerMap([7, 2, 11], [40, 41, 42])

        self.assertEqual(owner_map.newton_body_indices, (7, 2, 11))
        self.assertEqual(owner_map.deme_owner_ids, (40, 41, 42))
        self.assertEqual(owner_map.first_deme_owner_id, 40)
        self.assertEqual(owner_map.owner_count, 3)

    def test_rejects_empty_mapping(self):
        with self.assertRaisesRegex(ValueError, "at least one body"):
            NewtonDEMEOwnerMap([], [])

    def test_rejects_different_mapping_lengths(self):
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            NewtonDEMEOwnerMap([0, 1], [20])

    def test_accepts_multiple_owners_for_one_newton_body(self):
        owner_map = NewtonDEMEOwnerMap([3, 3], [20, 21])

        self.assertEqual(owner_map.newton_body_indices, (3, 3))

    def test_rejects_nonconsecutive_deme_owners(self):
        with self.assertRaisesRegex(ValueError, "consecutive span"):
            NewtonDEMEOwnerMap([3, 4], [20, 22])


class NewtonDEMEContactCouplerValidationTest(unittest.TestCase):
    def test_rejects_non_cuda_device_before_allocating_buffers(self):
        class Model:
            body_count = 1

        class Device:
            is_cuda = False

        with self.assertRaisesRegex(ValueError, "requires a CUDA"):
            coupler = NewtonDEMEContactCoupler()
            coupler.initialize(Model(), object(), NewtonDEMEOwnerMap([0], [0]), Device())

    def test_methods_require_initialize(self):
        coupler = NewtonDEMEContactCoupler()

        with self.assertRaisesRegex(RuntimeError, "must be initialized"):
            coupler.step_deme()

    def test_acceleration_mode_methods_require_initialize(self):
        coupler = NewtonDEMEContactAccelerationCoupler()

        with self.assertRaisesRegex(RuntimeError, "must be initialized"):
            coupler.write_deme_contact_accelerations_to_newton()

    def test_pose_exchange_methods_require_initialize(self):
        exchange = NewtonDEMEOwnerPoseExchange()

        with self.assertRaisesRegex(RuntimeError, "must be initialized"):
            exchange.set_deme_owner_pose_from_newton(object())

    def test_raw_contact_acceleration_getter_uses_solver_device_api(self):
        class Device:
            ordinal = 2

        device = Device()

        class Array:
            dtype = wp.vec3

            def __init__(self, pointer):
                self.ptr = pointer
                self.device = device

            def __len__(self):
                return 2

        class Solver:
            def __init__(self):
                self.calls = []

            def GetOwnerAccToDevice(self, *args):
                self.calls.append(("linear", args))

            def GetOwnerAngAccGlobalToDevice(self, *args):
                self.calls.append(("angular_global", args))

        coupler = NewtonDEMEContactCoupler()
        coupler.initialized = True
        coupler.device = device
        coupler.owner_map = NewtonDEMEOwnerMap([0, 1], [7, 8])
        coupler.deme_solver = Solver()
        coupler.get_deme_contact_accelerations_to_device(Array(101), Array(202), angular_frame="global")

        self.assertEqual(
            coupler.deme_solver.calls,
            [("linear", (101, 2, 2, 7, 2)), ("angular_global", (202, 2, 2, 7, 2))],
        )
        coupler.initialized = False


class NewtonDEMEMeshOwnerTest(unittest.TestCase):
    def test_extracts_scaled_body_local_newton_mesh(self):
        class Source:
            vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
            indices = [0, 1, 2]

        class Transform:
            p = (1, 2, 3)
            q = (0, 0, 0, 1)

        class Builder:
            shape_type = ["mesh"]
            shape_body = [0]
            shape_source = [Source()]
            shape_scale = [(2, 3, 4)]
            shape_transform = [Transform()]
            shape_label = ["triangle"]
            body_label = ["body"]

        mesh = extract_newton_shape_triangle_meshes(Builder(), [0])[0]
        np.testing.assert_array_equal(mesh.scaled_vertices, [[0, 0, 0], [2, 0, 0], [0, 3, 0]])
        np.testing.assert_array_equal(mesh.body_local_vertices, [[1, 2, 3], [3, 2, 3], [1, 5, 3]])
        np.testing.assert_array_equal(mesh.triangle_indices, [[0, 1, 2]])

    def test_combines_mesh_parts_with_adjusted_indices(self):
        vertices, triangles = combine_triangle_meshes(
            [
                (np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.asarray([[0, 1, 2]])),
                (np.asarray([[0, 0, 1], [1, 0, 1], [0, 1, 1]]), np.asarray([0, 1, 2])),
            ]
        )

        self.assertEqual(vertices.shape, (6, 3))
        np.testing.assert_array_equal(triangles, [[0, 1, 2], [3, 4, 5]])

    def test_adds_owner_with_explicit_physical_properties_and_builds_mapping(self):
        class Owner:
            def SetInitPos(self, value):
                self.position = value

            def SetInitQuat(self, value):
                self.quaternion = value

            def SetFamily(self, value):
                self.family = value

            def SetMass(self, value):
                self.mass = value

            def SetMOI(self, value):
                self.moi = value

        class Tracker:
            def GetOwnerID(self):
                return 12

        class Solver:
            def AddWavefrontMeshObject(self, path, material, load_normals):
                self.path = path
                self.material = material
                self.load_normals = load_normals
                self.owner = Owner()
                return self.owner

            def Track(self, owner):
                self.tracked_owner = owner
                return Tracker()

        spec = NewtonDEMEMeshOwnerSpec(
            name="body_proxy",
            newton_body_index=1,
            family=7,
            mass=2.5,
            principal_moi=(0.2, 0.3, 0.4),
            vertices=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            triangle_indices=np.asarray([[0, 1, 2]]),
        )
        body_q = np.asarray([[0, 0, 0, 0, 0, 0, 1], [1, 2, 3, 0, 0, 0, 1]], dtype=np.float32)
        solver = Solver()
        with TemporaryDirectory() as directory:
            bindings = add_deme_mesh_owners(solver, "material", [spec], body_q, Path(directory))
            mesh_text = bindings[0].mesh_path.read_text(encoding="utf-8")

        self.assertIn("v 1 0 0", mesh_text)
        self.assertIn("f 1 2 3", mesh_text)
        self.assertEqual(solver.owner.position, [1.0, 2.0, 3.0])
        self.assertEqual(solver.owner.family, 7)
        self.assertEqual(solver.owner.mass, 2.5)
        self.assertEqual(solver.owner.moi, [0.2, 0.3, 0.4])
        owner_map = owner_map_from_mesh_bindings(bindings)
        self.assertEqual(owner_map.newton_body_indices, (1,))
        self.assertEqual(owner_map.deme_owner_ids, (12,))


if __name__ == "__main__":
    unittest.main()
