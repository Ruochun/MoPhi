from pathlib import Path

import numpy as np
import pytest

from mophi.utils import triangulate_spheres, write_vtk_file_series, write_vtk_polydata


def test_triangulate_spheres_uses_requested_centers_and_radii():
    points, triangles, sphere_ids = triangulate_spheres(
        [[1.0, 2.0, 3.0], [-1.0, 0.0, 0.0]],
        [0.5, 2.0],
        latitude_segments=4,
        longitude_segments=6,
    )

    assert points.shape == (40, 3)
    assert triangles.shape == (72, 3)
    assert np.array_equal(np.unique(sphere_ids, return_counts=True)[1], [20, 20])
    assert np.allclose(np.linalg.norm(points[sphere_ids == 0] - [1.0, 2.0, 3.0], axis=1), 0.5)
    assert np.allclose(np.linalg.norm(points[sphere_ids == 1] - [-1.0, 0.0, 0.0], axis=1), 2.0)


def test_write_vtk_polydata(tmp_path: Path):
    output = write_vtk_polydata(
        tmp_path / "nested" / "frame.vtk",
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        [[0, 1, 2]],
        vertex_indices=[2],
        cell_scalars={"part_id": [7]},
        point_scalars={"radius": [0.0, 0.0, 0.1]},
    )

    contents = output.read_text(encoding="ascii")
    assert "DATASET POLYDATA" in contents
    assert "POINTS 3 double" in contents
    assert "POLYGONS 1 4" in contents
    assert "VERTICES 1 2" in contents
    assert "CELL_DATA 2" in contents
    assert "POINT_DATA 3" in contents
    assert "SCALARS part_id int 1" in contents
    assert "SCALARS radius double 1" in contents


def test_write_vtk_file_series(tmp_path: Path):
    series_path = tmp_path / "frames.vtk.series"
    write_vtk_file_series(
        series_path,
        [(0.0, tmp_path / "frames" / "frame_000000.vtk"), (0.1, "frames/frame_000001.vtk")],
    )

    contents = series_path.read_text(encoding="utf-8")
    assert '"file-series-version": "1.0"' in contents
    assert '"name": "frames/frame_000000.vtk"' in contents
    assert '"time": 0.0' in contents
    assert '"time": 0.1' in contents


@pytest.mark.parametrize(
    ("points", "triangles"),
    [
        ([[0, 0]], [[0, 0, 0]]),
        ([[0, 0, 0]], [[0, 1, 0]]),
        ([[np.nan, 0, 0]], [[0, 0, 0]]),
    ],
)
def test_write_vtk_polydata_rejects_invalid_geometry(tmp_path: Path, points, triangles):
    with pytest.raises(ValueError):
        write_vtk_polydata(tmp_path / "bad.vtk", points, triangles)
