"""Dependency-free legacy VTK PolyData output for simulation snapshots."""

from pathlib import Path
from typing import Mapping

import numpy as np


def write_vtk_polydata(
    path: str | Path,
    points,
    triangles,
    *,
    cell_scalars: Mapping[str, object] | None = None,
    title: str = "MoPhi simulation snapshot",
) -> Path:
    """Write triangle geometry and optional integer cell labels as ASCII VTK."""
    output_path = Path(path)
    points_array = np.asarray(points, dtype=np.float64)
    triangles_array = np.asarray(triangles, dtype=np.int64)
    if points_array.ndim != 2 or points_array.shape[1] != 3 or not np.isfinite(points_array).all():
        raise ValueError("points must be a finite array with shape (N, 3)")
    if triangles_array.ndim != 2 or triangles_array.shape[1] != 3:
        raise ValueError("triangles must be an integer array with shape (M, 3)")
    if triangles_array.size and (triangles_array.min() < 0 or triangles_array.max() >= len(points_array)):
        raise ValueError("triangles contain an out-of-range point index")

    scalars = {}
    for name, values in (cell_scalars or {}).items():
        if not name or any(character.isspace() for character in name):
            raise ValueError("cell scalar names must be non-empty and contain no whitespace")
        array = np.asarray(values)
        if array.ndim != 1 or len(array) != len(triangles_array):
            raise ValueError(f"cell scalar {name!r} must have one value per triangle")
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"cell scalar {name!r} must contain integers")
        scalars[name] = array.astype(np.int64, copy=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("# vtk DataFile Version 3.0\n")
        stream.write(f"{title.replace(chr(10), ' ')[:255]}\n")
        stream.write("ASCII\nDATASET POLYDATA\n")
        stream.write(f"POINTS {len(points_array)} double\n")
        np.savetxt(stream, points_array, fmt="%.17g")
        stream.write(f"POLYGONS {len(triangles_array)} {len(triangles_array) * 4}\n")
        if len(triangles_array):
            polygon_rows = np.column_stack((np.full(len(triangles_array), 3), triangles_array))
            np.savetxt(stream, polygon_rows, fmt="%d")
        if scalars:
            stream.write(f"CELL_DATA {len(triangles_array)}\n")
            for name, values in scalars.items():
                stream.write(f"SCALARS {name} int 1\nLOOKUP_TABLE default\n")
                np.savetxt(stream, values, fmt="%d")
    return output_path
