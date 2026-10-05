"""Physical frame/inertia checks for the dynamic GRAB cup preparation path."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from mophi.couplers.newton_deme import prepare_mesh_principal_frame


class PrincipalMeshTests(unittest.TestCase):
    def test_translated_rotated_box_preserves_geometry_and_inertia(self):
        extents = np.array([0.04, 0.06, 0.1])
        mass = 0.25
        box = trimesh.creation.box(extents=extents)
        rotation = Rotation.from_rotvec([0.4, -0.2, 0.7]).as_matrix()
        center = np.array([1.2, -0.3, 0.8])
        vertices = box.vertices @ rotation.T + center
        prepared = prepare_mesh_principal_frame(vertices, box.faces, mass)
        np.testing.assert_allclose(prepared.center_of_mass, center, atol=1e-12)
        np.testing.assert_allclose(
            prepared.vertices @ prepared.principal_to_input.T + prepared.center_of_mass,
            vertices,
            atol=1e-8,
        )
        diagonal = mass / 12 * (np.sum(extents**2) - extents**2)
        expected = rotation @ np.diag(diagonal) @ rotation.T
        actual = prepared.principal_to_input @ np.diag(prepared.principal_moi) @ prepared.principal_to_input.T
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(prepared.principal_to_input), 1.0)
        self.assertAlmostEqual(prepared.volume, np.prod(extents))

    def test_open_surface_cannot_silently_supply_mass_properties(self):
        box = trimesh.creation.box()
        with self.assertRaisesRegex(ValueError, "watertight"):
            prepare_mesh_principal_frame(box.vertices, box.faces[:-1], 1.0)
        for mass in [0.0, -1.0, np.nan]:
            with self.assertRaisesRegex(ValueError, "mass"):
                prepare_mesh_principal_frame(box.vertices, box.faces, mass)


if __name__ == "__main__":
    unittest.main()
