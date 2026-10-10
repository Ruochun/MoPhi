"""Coordinate, time, and rotation contracts for prepared GRAB playback."""

from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from mophi.utils.grab.motion import GrabPlayback, load_object_motion, playback_times


class GrabPlaybackTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.hand_path = self.root / "hand.npz"
        self.object_path = self.root / "object.npz"
        self.common = dict(
            format_version=np.int32(1),
            source_sequence="s1/test.npz",
            source_frame_indices=np.array([12, 24], dtype=np.int32),
            source_fps=np.float64(120),
            timestamps_s=np.array([0.1, 0.2]),
        )
        triangle = np.array([[1, 0, 0], [0, 1, 0], [-1, -1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        self.hand = dict(
            **self.common,
            hand_side="right",
            origin_kind="mean_hand_vertex_position",
            hand_vertices_local=np.array([triangle, triangle * 2]),
            hand_faces=faces,
            hand_origin_world=np.array([[0, 0, 1], [2, 0, 1]], dtype=np.float32),
        )
        self.obj = dict(
            **self.common,
            object_name="triangle",
            object_vertices_local=triangle,
            object_faces=faces,
            object_translation_world=np.array([[0, 0, 1], [0, 2, 1]], dtype=np.float32),
            object_quaternion_xyzw=Rotation.from_rotvec(np.deg2rad([[0, 0, 170], [0, 0, -170]]))
            .as_quat()
            .astype(np.float32),
        )
        self.write()

    def write(self):
        np.savez(self.hand_path, **self.hand)
        np.savez(self.object_path, **self.obj)

    def test_unified_viewer_recording_needs_no_simulation(self):
        directory = Path(__file__).resolve().parents[1] / "demo/newton_dem/grab"
        sys.path.insert(0, str(directory))
        spec = importlib.util.spec_from_file_location("grab_viewer", directory / "grasp_viewer.py")
        viewer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(viewer)
        playback = GrabPlayback([self.hand_path], self.object_path)
        from grabbing_cup import scene_config as config

        poses = viewer.recorded_poses(playback, 12, 24, 0.5, config)
        self.assertEqual(poses["source_time_s"][0], 0.1)
        self.assertEqual(poses["source_time_s"][-1], 0.2)
        np.testing.assert_array_equal(poses["source_time_s"][poses["time_s"] < config.SETTLE_TIME], 0.1)
        self.assertEqual(viewer.DEFAULT_MODE, "compare")
        self.assertEqual(config.CASE, "grab")

    def test_midpoint_deformation_and_shortest_rotation(self):
        playback = GrabPlayback([self.hand_path], self.object_path)
        sample = playback.sample(0.15)
        np.testing.assert_allclose(sample["right"][0], [2.5, 0, 1], atol=1e-6)
        # Crossing +/-180 degrees must not turn the object back through zero.
        np.testing.assert_allclose(sample["object"][0], [-1, 1, 1], atol=1e-6)
        for index, time in enumerate(self.common["timestamps_s"]):
            np.testing.assert_allclose(
                playback.sample(time)["right"],
                self.hand["hand_vertices_local"][index] + self.hand["hand_origin_world"][index],
            )
        with self.assertRaisesRegex(ValueError, "outside"):
            playback.sample(0.21)

    def test_mismatched_assets_and_duplicates_fail(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            GrabPlayback([self.hand_path, self.hand_path], self.object_path)
        self.obj["source_sequence"] = "s2/other.npz"
        self.write()
        with self.assertRaisesRegex(ValueError, "same source_sequence"):
            GrabPlayback([self.hand_path], self.object_path)

    def test_invalid_object_mesh_and_quaternion_fail(self):
        self.obj["object_quaternion_xyzw"] *= 2
        self.write()
        with self.assertRaisesRegex(ValueError, "normalized"):
            load_object_motion(self.object_path)
        self.obj["object_quaternion_xyzw"] /= 2
        self.obj["object_faces"][0, 0] = 3
        self.write()
        with self.assertRaisesRegex(ValueError, "face indices"):
            load_object_motion(self.object_path)

    def test_single_frame_clip(self):
        for asset, keys in (
            (self.hand, ["hand_vertices_local", "hand_origin_world"]),
            (self.obj, ["object_translation_world", "object_quaternion_xyzw"]),
        ):
            for key in ["source_frame_indices", "timestamps_s", *keys]:
                asset[key] = asset[key][:1]
        self.write()
        playback = GrabPlayback([self.hand_path], self.object_path)
        self.assertEqual(playback.start_time, playback.end_time)
        self.assertEqual(playback.sample(playback.start_time)["object"].shape, (3, 3))

    def test_playback_cadence_includes_last_pose(self):
        np.testing.assert_allclose(playback_times(0.1, 0.2, 40), [0.1, 0.125, 0.15, 0.175, 0.2])
        np.testing.assert_allclose(playback_times(0.1, 0.2, 40, 2), [0.1, 0.15, 0.2])
        np.testing.assert_allclose(playback_times(0.1, 0.1, 60), [0.1])
        for fps in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                playback_times(0, 1, fps)


if __name__ == "__main__":
    unittest.main()
