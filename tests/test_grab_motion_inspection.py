"""Validation and rendering checks for converted GRAB hand motion."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import imageio.v2 as imageio
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "inspect_grab_hand_motion.py"
SPEC = importlib.util.spec_from_file_location("inspect_grab_hand_motion", SCRIPT)
INSPECT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSPECT)


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

    def test_valid_motion_renders_selected_frames(self):
        self.write()
        motion = INSPECT.load_motion(self.path)
        self.assertEqual(INSPECT.select_frames([motion], None), [0, 1, 2])
        output = Path(self.directory.name) / "inspection.png"
        INSPECT.render([motion], [0, 2], output)
        self.assertGreater(output.stat().st_size, 1000)

    def test_rejects_inconsistent_time_and_invalid_faces(self):
        self.fields["timestamps_s"] = np.array([0, 1, 2], dtype=np.float64)
        self.write()
        with self.assertRaisesRegex(ValueError, "timestamps"):
            INSPECT.load_motion(self.path)
        self.fields["timestamps_s"] = np.arange(3) / 120
        self.fields["hand_faces"][0, 0] = 4
        self.write()
        with self.assertRaisesRegex(ValueError, "face indices"):
            INSPECT.load_motion(self.path)

    def test_rejects_wrong_vertex_dtype(self):
        self.fields["hand_vertices_local"] = self.fields["hand_vertices_local"].astype(np.float64)
        self.write()
        with self.assertRaisesRegex(ValueError, "hand_vertices_local must have dtype float32"):
            INSPECT.load_motion(self.path)

    def test_movie_uses_fixed_source_order_and_encodes_frames(self):
        self.write()
        motion = INSPECT.load_motion(self.path)
        frames = INSPECT.select_movie_frames([motion], None, None, 1)
        self.assertEqual(frames, [0, 1, 2])
        output = Path(self.directory.name) / "inspection.mp4"
        INSPECT.render_movie([motion], frames, output, fps=120)
        with imageio.get_reader(output) as reader:
            self.assertEqual(reader.count_frames(), 3)
        with self.assertRaisesRegex(ValueError, "movie-stride"):
            INSPECT.select_movie_frames([motion], None, None, 0)


if __name__ == "__main__":
    unittest.main()
