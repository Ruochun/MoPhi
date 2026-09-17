import numpy as np
import pytest

from mophi.utils import sample_grid_between_mesh_surfaces


def _open_box_side_mesh():
    vertices = np.asarray(
        [
            [-1, -1, -1],
            [-1, 1, -1],
            [-1, 1, 1],
            [-1, -1, 1],
            [1, -1, -1],
            [1, 1, -1],
            [1, 1, 1],
            [1, -1, 1],
        ],
        dtype=np.float64,
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]])
    return vertices, triangles


def test_sample_grid_between_open_mesh_surfaces():
    vertices, triangles = _open_box_side_mesh()
    points = sample_grid_between_mesh_surfaces(
        vertices,
        triangles,
        spacing=0.5,
        bounds_min=(-1, -0.5, -0.5),
        bounds_max=(1, 0.5, 0.5),
        ray_axis=0,
        clearance=0.25,
    )

    assert len(points) == 16
    assert np.all(points[:, 0] >= -0.75)
    assert np.all(points[:, 0] <= 0.75)


@pytest.mark.parametrize("spacing", [0.0, -1.0, np.nan])
def test_sample_grid_rejects_invalid_spacing(spacing):
    vertices, triangles = _open_box_side_mesh()
    with pytest.raises(ValueError):
        sample_grid_between_mesh_surfaces(vertices, triangles, spacing, (-1, -1, -1), (1, 1, 1))
