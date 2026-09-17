"""Dependency-free legacy VTK PolyData output for simulation snapshots."""

import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def triangulate_spheres(
    centers,
    radii,
    *,
    latitude_segments: int = 6,
    longitude_segments: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create triangle surfaces for spheres centered at the supplied points.

    Returns ``(points, triangles, sphere_ids)``. ``sphere_ids`` contains one
    source-sphere index per generated point, which callers can use for VTK
    point metadata. Segment counts control output size and visual smoothness.
    """
    centers_array = np.asarray(centers, dtype=np.float64)
    radii_array = np.asarray(radii, dtype=np.float64)
    if centers_array.ndim != 2 or centers_array.shape[1] != 3 or not np.isfinite(centers_array).all():
        raise ValueError("centers must be a finite array with shape (N, 3)")
    if radii_array.ndim == 0:
        radii_array = np.full(len(centers_array), radii_array, dtype=np.float64)
    if radii_array.ndim != 1 or len(radii_array) != len(centers_array):
        raise ValueError("radii must be a scalar or contain one value per center")
    if not np.isfinite(radii_array).all() or np.any(radii_array <= 0.0):
        raise ValueError("radii must be finite and positive")
    if not isinstance(latitude_segments, int) or latitude_segments < 3:
        raise ValueError("latitude_segments must be an integer of at least 3")
    if not isinstance(longitude_segments, int) or longitude_segments < 3:
        raise ValueError("longitude_segments must be an integer of at least 3")

    unit_points = [[0.0, 0.0, 1.0]]
    for latitude_index in range(1, latitude_segments):
        polar_angle = np.pi * latitude_index / latitude_segments
        for longitude_index in range(longitude_segments):
            azimuth = 2.0 * np.pi * longitude_index / longitude_segments
            unit_points.append(
                [
                    np.sin(polar_angle) * np.cos(azimuth),
                    np.sin(polar_angle) * np.sin(azimuth),
                    np.cos(polar_angle),
                ]
            )
    unit_points.append([0.0, 0.0, -1.0])
    unit_points = np.asarray(unit_points, dtype=np.float64)

    unit_triangles = []
    first_ring = 1
    for longitude_index in range(longitude_segments):
        next_longitude = (longitude_index + 1) % longitude_segments
        unit_triangles.append((0, first_ring + longitude_index, first_ring + next_longitude))
    for latitude_index in range(latitude_segments - 2):
        upper_ring = first_ring + latitude_index * longitude_segments
        lower_ring = upper_ring + longitude_segments
        for longitude_index in range(longitude_segments):
            next_longitude = (longitude_index + 1) % longitude_segments
            unit_triangles.append(
                (upper_ring + longitude_index, lower_ring + longitude_index, lower_ring + next_longitude)
            )
            unit_triangles.append(
                (upper_ring + longitude_index, lower_ring + next_longitude, upper_ring + next_longitude)
            )
    south_pole = len(unit_points) - 1
    last_ring = south_pole - longitude_segments
    for longitude_index in range(longitude_segments):
        next_longitude = (longitude_index + 1) % longitude_segments
        unit_triangles.append((last_ring + longitude_index, south_pole, last_ring + next_longitude))
    unit_triangles = np.asarray(unit_triangles, dtype=np.int64)

    points_per_sphere = len(unit_points)
    points = centers_array[:, None, :] + radii_array[:, None, None] * unit_points[None, :, :]
    offsets = np.arange(len(centers_array), dtype=np.int64)[:, None, None] * points_per_sphere
    triangles = unit_triangles[None, :, :] + offsets
    sphere_ids = np.repeat(np.arange(len(centers_array), dtype=np.int64), points_per_sphere)
    return points.reshape((-1, 3)), triangles.reshape((-1, 3)), sphere_ids


def write_vtk_polydata(
    path: str | Path,
    points,
    triangles,
    *,
    vertex_indices=None,
    cell_scalars: Mapping[str, object] | None = None,
    point_scalars: Mapping[str, object] | None = None,
    title: str = "MoPhi simulation snapshot",
) -> Path:
    """Write triangle/vertex geometry and optional scalar labels as ASCII VTK."""
    output_path = Path(path)
    points_array = np.asarray(points, dtype=np.float64)
    triangles_array = np.asarray(triangles, dtype=np.int64)
    if points_array.ndim != 2 or points_array.shape[1] != 3 or not np.isfinite(points_array).all():
        raise ValueError("points must be a finite array with shape (N, 3)")
    if triangles_array.ndim != 2 or triangles_array.shape[1] != 3:
        raise ValueError("triangles must be an integer array with shape (M, 3)")
    if triangles_array.size and (triangles_array.min() < 0 or triangles_array.max() >= len(points_array)):
        raise ValueError("triangles contain an out-of-range point index")
    vertex_indices_array = np.asarray([] if vertex_indices is None else vertex_indices, dtype=np.int64)
    if vertex_indices_array.ndim != 1:
        raise ValueError("vertex_indices must be a one-dimensional integer array")
    if vertex_indices_array.size and (
        vertex_indices_array.min() < 0 or vertex_indices_array.max() >= len(points_array)
    ):
        raise ValueError("vertex_indices contain an out-of-range point index")

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

    point_data = {}
    for name, values in (point_scalars or {}).items():
        if not name or any(character.isspace() for character in name):
            raise ValueError("point scalar names must be non-empty and contain no whitespace")
        array = np.asarray(values)
        if array.ndim != 1 or len(array) != len(points_array):
            raise ValueError(f"point scalar {name!r} must have one value per point")
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError(f"point scalar {name!r} must contain finite numbers")
        point_data[name] = array

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("# vtk DataFile Version 3.0\n")
        stream.write(f"{title.replace(chr(10), ' ')[:255]}\n")
        stream.write("ASCII\nDATASET POLYDATA\n")
        stream.write(f"POINTS {len(points_array)} double\n")
        np.savetxt(stream, points_array, fmt="%.17g")
        if len(vertex_indices_array):
            stream.write(f"VERTICES {len(vertex_indices_array)} {len(vertex_indices_array) * 2}\n")
            vertex_rows = np.column_stack((np.ones(len(vertex_indices_array), dtype=np.int64), vertex_indices_array))
            np.savetxt(stream, vertex_rows, fmt="%d")
        stream.write(f"POLYGONS {len(triangles_array)} {len(triangles_array) * 4}\n")
        if len(triangles_array):
            polygon_rows = np.column_stack((np.full(len(triangles_array), 3), triangles_array))
            np.savetxt(stream, polygon_rows, fmt="%d")
        if scalars:
            stream.write(f"CELL_DATA {len(vertex_indices_array) + len(triangles_array)}\n")
            for name, values in scalars.items():
                stream.write(f"SCALARS {name} int 1\nLOOKUP_TABLE default\n")
                padded_values = np.concatenate((np.zeros(len(vertex_indices_array), dtype=np.int64), values))
                np.savetxt(stream, padded_values, fmt="%d")
        if point_data:
            stream.write(f"POINT_DATA {len(points_array)}\n")
            for name, values in point_data.items():
                if np.issubdtype(values.dtype, np.integer):
                    stream.write(f"SCALARS {name} int 1\nLOOKUP_TABLE default\n")
                    np.savetxt(stream, values, fmt="%d")
                else:
                    stream.write(f"SCALARS {name} double 1\nLOOKUP_TABLE default\n")
                    np.savetxt(stream, values, fmt="%.17g")
    return output_path


def write_vtk_file_series(path: str | Path, frames: Sequence[tuple[float, str | Path]]) -> Path:
    """Write a ParaView JSON file-series manifest for time-stamped VTK files."""
    output_path = Path(path)
    if not output_path.name.endswith(".vtk.series"):
        raise ValueError("a legacy VTK series manifest path must end with '.vtk.series'")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for time, frame_path in frames:
        if not np.isfinite(time):
            raise ValueError("VTK frame times must be finite")
        relative_path = Path(frame_path)
        if relative_path.is_absolute():
            relative_path = Path(os.path.relpath(relative_path, output_path.parent))
        files.append({"name": relative_path.as_posix(), "time": float(time)})
    output_path.write_text(
        json.dumps({"file-series-version": "1.0", "files": files}, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path
