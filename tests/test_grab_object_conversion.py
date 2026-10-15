"""Compare converted object playback with GRAB ObjectModel's row-vector rule."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation
from smplx.lbs import batch_rodrigues

from mophi.utils.grab.motion import load_object_motion


class GrabObjectConversionTests(unittest.TestCase):
    def test_export_matches_official_object_transform(self):
        script = Path(__file__).resolve().parents[1] / "scripts/grab/convert_grab_hand_motion.py"
        spec = importlib.util.spec_from_file_location("grab_converter", script)
        converter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(converter)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "grab/s1").mkdir(parents=True)
            mesh = trimesh.creation.box(extents=[0.08, 0.12, 0.17])
            mesh.apply_translation([0.02, -0.01, 0.03])
            mesh.export(root / "object.ply")
            # Include a non-axis-aligned rotation and translations; symmetry must
            # not conceal the transpose error in corresponding vertex positions.
            orient = np.array([[0, 0, 0], [0, 0, np.pi / 2], [0.3, -0.7, 1.1]], dtype=np.float32)
            transl = np.array([[0, 0, 0], [0.1, -0.2, 0.5], [-0.3, 0.4, 0.2]], dtype=np.float32)
            source = root / "grab/s1/test.npz"
            np.savez(
                source,
                n_frames=3,
                framerate=120,
                object={"object_mesh": "object.ply", "params": {"global_orient": orient, "transl": transl}},
            )
            output = converter.convert_object(root, source, root / "prepared", start=1, stop=3)
            data = load_object_motion(output)
            self.assertEqual(int(data["format_version"]), 2)
            np.testing.assert_array_equal(data["source_frame_indices"], [1, 2])
            vertices = data["object_vertices_local"]
            matrices = batch_rodrigues(torch.tensor(orient[1:])).numpy()
            for i, matrix in enumerate(matrices):
                expected = vertices @ matrix + transl[i + 1]
                actual = (
                    Rotation.from_quat(data["object_quaternion_xyzw"][i]).apply(vertices)
                    + data["object_translation_world"][i]
                )
                np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)
                wrong = Rotation.from_rotvec(orient[i + 1]).apply(vertices) + transl[i + 1]
                self.assertGreater(np.max(np.abs(wrong - expected)), 0.01)

    def test_old_simulation_cannot_be_compared_to_corrected_reference(self):
        import sys

        directory = Path(__file__).resolve().parents[1] / "demo/newton_dem/grab"
        sys.path.insert(0, str(directory))
        self.addCleanup(sys.path.remove, str(directory))
        from grasp_viewer import load_render_poses

        with self.assertRaisesRegex(ValueError, "rerun the simulation"):
            load_render_poses(Path("unused"), {}, None, None)


if __name__ == "__main__":
    unittest.main()
