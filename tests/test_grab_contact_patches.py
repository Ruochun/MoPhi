"""Geometric invariants of reusable DEME contact patch preparation."""

import unittest
import numpy as np
from mophi.utils.geometry.mesh_contact_patches import partition_mesh_contact_patches


class PatchTests(unittest.TestCase):
    def test_public_imports_share_one_implementation(self):
        from mophi.utils import partition_mesh_contact_patches as utility
        from mophi.couplers.newton_deme import partition_mesh_contact_patches as legacy

        self.assertIs(utility, partition_mesh_contact_patches)
        self.assertIs(legacy, utility)

    def test_disconnected_and_opposite_faces_never_merge(self):
        v = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [3.0, 1.0, 0.0],
            ]
        )
        f = np.array([[0, 1, 2], [1, 3, 2], [4, 5, 6]])
        np.testing.assert_array_equal(partition_mesh_contact_patches(v, f, 10), [0, 0, 1])
        reversed_f = f.copy()
        reversed_f[1] = reversed_f[1, ::-1]
        self.assertEqual(len(np.unique(partition_mesh_contact_patches(v, reversed_f, 10))), 3)
        np.testing.assert_array_equal(partition_mesh_contact_patches(v, f, 0.1), [0, 1, 2])

    def test_deterministic_connected_regions_with_radius_bound(self):
        v = np.array([[x, y, 0.0] for y in range(5) for x in range(5)])
        f = []
        for y in range(4):
            for x in range(4):
                a = y * 5 + x
                f.extend([[a, a + 1, a + 5], [a + 1, a + 6, a + 5]])
        f = np.array(f)
        before_v = v.copy()
        before_f = f.copy()
        ids = partition_mesh_contact_patches(v, f, 1.5)
        np.testing.assert_array_equal(ids, partition_mesh_contact_patches(v, f, 1.5))
        np.testing.assert_array_equal(v, before_v)
        np.testing.assert_array_equal(f, before_f)
        self.assertGreater(ids.max(), 0)
        for p in np.unique(ids):
            rows = np.flatnonzero(ids == p)
            seed_center = v[f[rows[0]]].mean(axis=0)
            self.assertLessEqual(np.linalg.norm(v[f[rows]] - seed_center, axis=2).max(), 1.5)
            seen = {rows[0]}
            while True:
                more = {i for i in rows if any(len(set(f[i]) & set(f[j])) == 2 for j in seen)}
                if more <= seen:
                    break
                seen |= more
            self.assertEqual(seen, set(rows))

    def test_invalid_input_rejected(self):
        v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        for radius in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                partition_mesh_contact_patches(v, np.array([[0, 1, 2]]), radius)
        with self.assertRaises(ValueError):
            partition_mesh_contact_patches(v, np.array([[0, 0, 1]]), 1)
        with self.assertRaises(ValueError):
            partition_mesh_contact_patches(v, np.array([[0.0, 1.0, 2.0]]), 1)


if __name__ == "__main__":
    unittest.main()
