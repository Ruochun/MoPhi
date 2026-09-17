# Open-mesh volume sampling

This document describes `python/mophi/utils/mesh_sampling.py`, specifically
the public `sample_grid_between_mesh_surfaces` utility.

## Purpose and design boundary

The utility creates deterministic Cartesian grid points inside a duct, hopper,
or similar triangle surface that may be open at its ends. It is independent of
DEME, Newton, and particle representations. Demos remain responsible for mesh
selection, physical clearance, coordinate transforms, materials, and boundary
conditions.

For every candidate inside explicit clipping bounds, the sampler casts an
axis-aligned line and requires a triangle intersection on both sides. It then
rejects points closer than `clearance` to any triangle. The explicit bounds
therefore close the unsurfaced directions without modifying the source mesh.

## Usage

```python
from mophi.utils import sample_grid_between_mesh_surfaces

points = sample_grid_between_mesh_surfaces(
    vertices,
    triangles,
    spacing=0.01,
    bounds_min=(-0.1, -0.2, -0.1),
    bounds_max=(0.1, 0.2, 0.1),
    ray_axis=0,
    clearance=0.005,
)
```

Inputs and outputs use caller-selected units, and the result is an `(N, 3)`
NumPy array in the input mesh's coordinate frame. The arguments have these
roles:

- `spacing` is the Cartesian grid spacing on all three axes.
- `bounds_min` and `bounds_max` define the only region considered and also
  clip the two directions not closed by the mesh.
- `ray_axis=0`, `1`, or `2` selects the input mesh's local X, Y, or Z axis.
  It is not a world-space or gravity-axis setting unless the caller's mesh is
  already in that frame. Choose an axis whose lines cross opposing side walls,
  rather than one pointing through an open end.
- `clearance` rejects candidates whose Euclidean distance to any triangle is
  smaller than the requested margin. For spherical particles this normally
  includes their radius and any additional contact envelope.

For example, `ray_axis=0` projects the triangles and candidates onto the local
YZ plane. At each YZ grid location, the sampler reconstructs triangle
intersections along local X and accepts a candidate only if at least one
crossing lies at lower X and another lies at higher X. The explicit Y and Z
bounds decide how much of the open channel is filled.

The robotic DFC printing demo uses this pattern for its open auger proxy. It
samples in the contact mesh's local frame, then transforms accepted positions
through the contact proxy pose into world coordinates before giving them to
DEME. Consequently, its `DEME_AUGER_SAMPLE_BOUNDS_*` and
`DEME_AUGER_SAMPLE_RAY_AXIS` settings must be interpreted in mesh-local axes.

## Supported behavior and limitations

- Triangle winding and watertightness are not required.
- The mesh must bracket the intended volume along the chosen ray axis.
- Explicit bounds are mandatory and should exclude branches or openings that
  do not belong to the desired sampling region.
- The method does not infer semantic cavities and cannot repair missing side
  walls. A poor ray axis or bounds can classify unintended pockets.
- Sampling and clearance checks run on the CPU and are intended for
  initialization, not recurring simulation-loop work.

## Tests

Run the focused tests from the repository root:

```bash
PYTHONPATH=python python -m pytest tests/test_mesh_sampling.py
```
