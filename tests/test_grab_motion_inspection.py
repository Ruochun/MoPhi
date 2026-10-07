"""Validation and rendering checks for converted GRAB hand motion."""

from pathlib import Path
import tempfile
import unittest
import numpy as np
from mophi.utils.grab.motion import load_motion


class GrabMotionInspectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "hand.npz"
        self.fields = {
            "format_version": np.int32(1),
            "source_sequence": "s1/example.npz",
            "hand_side": "right",
            "origin_kind": "mean_hand_vertex_position",
            "source_frame_indices": np.array([0, 1, 2], dtype=np.int32),
            "source_fps": np.float64(120),
            "timestamps_s": np.arange(3) / 120,
            "hand_origin_world": np.zeros((3, 3), dtype=np.float32),
            "hand_vertices_local": np.tile(
                np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32),
                (3, 1, 1),
            ),
            "hand_faces": np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        }

    def write(self):
        np.savez_compressed(self.path, **self.fields)

    def test_valid_motion_reconstructs_world_vertices(self):
        self.write()
        motion = load_motion(self.path)
        np.testing.assert_array_equal(
            motion["hand_vertices_local"] + motion["hand_origin_world"][:, None, :], self.fields["hand_vertices_local"]
        )

    def test_rejects_inconsistent_time_and_invalid_faces(self):
        self.fields["timestamps_s"] = np.array([0, 1, 2], dtype=np.float64)
        self.write()
        with self.assertRaisesRegex(ValueError, "timestamps"):
            load_motion(self.path)
        self.fields["timestamps_s"] = np.arange(3) / 120
        self.fields["hand_faces"][0, 0] = 4
        self.write()
        with self.assertRaisesRegex(ValueError, "face indices"):
            load_motion(self.path)

    def test_rejects_wrong_vertex_dtype(self):
        self.fields["hand_vertices_local"] = self.fields["hand_vertices_local"].astype(np.float64)
        self.write()
        with self.assertRaisesRegex(ValueError, "hand_vertices_local must have dtype float32"):
            load_motion(self.path)


if __name__ == "__main__":
    unittest.main()
