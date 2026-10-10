"""Check the prescribed hand's coordinate and velocity semantics independently."""

import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo/newton_dem/grab"))

SPEC = importlib.util.spec_from_file_location(
    "grab_contact_demo", Path(__file__).resolve().parents[1] / "demo/newton_dem/grab/grasp_simulation.py"
)
DEMO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEMO)
from grabbing_cup import scene_config as CONFIG


class GrabTrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.hand = {
            "timestamps_s": np.array([1.0, 1.1, 1.2]),
            "hand_origin_world": np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]),
            "hand_vertices_local": np.array(
                [
                    [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    [[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                    [[-3.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                ]
            ),
        }

    def test_interpolated_world_and_slowed_owner_derivative(self):
        offset = np.array([3.0, -2.0, 5.0])
        p, nodes, v = DEMO.sample_grab_hand(self.hand, 1.05, 0.5, offset)
        np.testing.assert_allclose(p, offset + [0.5, 1.0, 1.5])
        np.testing.assert_allclose(nodes, [[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]])
        epsilon = 1e-5
        next_p, _, _ = DEMO.sample_grab_hand(self.hand, 1.05 + epsilon * 0.5, 0.5, offset)
        np.testing.assert_allclose(v, (next_p - p) / epsilon, atol=1e-8)

    def test_lift_requires_support_and_persistent_clearance(self):
        data = np.zeros((3, 21))
        data[:, 0] = [0.8, 0.95, 1.0]
        data[:, 20] = 0.02
        grasp = np.zeros((3, 13))
        checks = DEMO.assess_grab_run(data, grasp, CONFIG)
        self.assertTrue(checks["cup_lifted"])
        self.assertFalse(checks["lift_supported_by_contact"])
        data[:, 16] = CONFIG.CUP_MASS * CONFIG.GRAVITY
        self.assertTrue(DEMO.assess_grab_run(data, grasp, CONFIG)["lift_supported_by_contact"])
        data[1, 20] = 0
        self.assertFalse(DEMO.assess_grab_run(data, grasp, CONFIG)["cup_lifted"])

    def test_scene_configuration_controls_support_threshold(self):
        data = np.zeros((3, 21))
        data[:, 0] = [0.8, 0.95, 1.0]
        data[:, 20] = 0.02
        data[:, 16] = CONFIG.CUP_MASS * CONFIG.GRAVITY
        grasp = np.zeros((3, 13))
        heavier = SimpleNamespace(**{name: getattr(CONFIG, name) for name in dir(CONFIG) if name.isupper()})
        heavier.CUP_MASS = CONFIG.CUP_MASS * 4
        self.assertTrue(DEMO.assess_grab_run(data, grasp, CONFIG)["lift_supported_by_contact"])
        self.assertFalse(DEMO.assess_grab_run(data, grasp, heavier)["lift_supported_by_contact"])
        self.assertTrue(DEMO.assess_grab_run(data, grasp, CONFIG)["lift_supported_by_contact"])

    def test_settling_zeroes_velocity_and_extrapolation_rejected(self):
        _, _, v = DEMO.sample_grab_hand(self.hand, 1.0, 0.0, np.zeros(3))
        np.testing.assert_array_equal(v, np.zeros(3))
        with self.assertRaises(ValueError):
            DEMO.sample_grab_hand(self.hand, 0.9, 0.5, np.zeros(3))


if __name__ == "__main__":
    unittest.main()
