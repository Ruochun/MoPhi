"""Verify signs, moment arms, and empty spans in cup contact force diagnostics."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "inspect_contact", Path(__file__).resolve().parents[1] / "tests/grab/inspect_contact.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ContactWrenchTests(unittest.TestCase):
    def test_force_couple_and_empty_samples(self):
        center = np.array([5.0, -2.0, 3.0])
        positions = np.tile(center, (3, 1))
        contacts = {
            "sample_offsets": np.array([0, 0, 2, 2]),
            "point_world": center + np.array([[-1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]),
            "force_on_cup_world": np.array([[0.0, -1.0, 2.0], [0.0, 1.0, 2.0]]),
            "normal_world": np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            "model_torque_on_cup_world": np.zeros((2, 3)),
        }
        result = MODULE.reconstruct_contact_wrenches(positions, contacts)
        np.testing.assert_allclose(result["force"], [[0, 0, 0], [0, 0, 4], [0, 0, 0]])
        np.testing.assert_allclose(result["torque"], [[0, 0, 0], [0, 0, 2], [0, 0, 0]])
        np.testing.assert_allclose(result["normal_torque"], 0)
        contacts["sample_offsets"][-1] = 1
        with self.assertRaises(ValueError):
            MODULE.reconstruct_contact_wrenches(positions, contacts)


if __name__ == "__main__":
    unittest.main()
