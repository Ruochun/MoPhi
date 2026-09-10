from pathlib import Path

import numpy as np
import pytest

from mophi.utils import write_vtk_polydata


def test_write_vtk_polydata(tmp_path: Path):
    output = write_vtk_polydata(
        tmp_path / "nested" / "frame.vtk",
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        [[0, 1, 2]],
        cell_scalars={"part_id": [7]},
    )

    contents = output.read_text(encoding="ascii")
    assert "DATASET POLYDATA" in contents
    assert "POINTS 3 double" in contents
    assert "POLYGONS 1 4" in contents
    assert "SCALARS part_id int 1" in contents


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
