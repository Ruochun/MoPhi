# VTK snapshot output

This document describes `python/mophi/utils/vtk.py`, the solver-independent
legacy VTK PolyData writer used by MoPhi demos.

## Purpose and design boundary

Newton does not currently provide a native VTK viewer or exporter. MoPhi's
`write_vtk_polydata` utility writes explicitly selected triangle and point
snapshots so they can be inspected in ParaView. `write_vtk_file_series` groups
numbered legacy VTK snapshots into a time series. The physics solver remains
responsible for state and geometry; the caller remains responsible for
transforming geometry into world coordinates and deciding which frame and
objects to export.

The writer has no dependency on the VTK Python package. It produces portable
ASCII legacy `.vtk` files containing triangle polygons, point-vertex cells, and
optional point or integer polygon-cell scalars. The companion
`triangulate_spheres` utility converts sphere centers and radii to explicit
triangle surfaces when a dataset must display spheres without a Glyph filter.

## Usage

```python
from mophi.utils import write_vtk_polydata

write_vtk_polydata(
    "output/frame.vtk",
    points=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
    triangles=[[0, 1, 2]],
    cell_scalars={"part_id": [0]},
)
```

The G-code printing demo writes one numbered file per rendered frame beneath
`output/demo_dfc_3d_printing/vtk/` when `SAVE_VTK_TIME_SERIES` is enabled. Open
`output/demo_dfc_3d_printing/dfc_3d_printing.vtk.series` in ParaView to load the
whole time series. The JSON `.vtk.series` manifest is used because PVD collections
do not reliably identify legacy `.vtk` datasets. Its `scene_part_id` arrays
distinguish CAD parts and the build plate, and
`is_nozzle_contact_mesh` is `1` for the open, inner-surface-only auger mesh
registered for particle contact. `is_deme_build_plate` identifies the finite
plate surface corresponding to the optional infinite DEME analytical plane.
The run's `run_metadata.json` maps the numeric part IDs to names. DEME particles are explicit tessellated sphere surfaces using the DEME
particle radius, so they render as finite-sized spheres immediately upon
loading. The `particle_id` and `particle_radius` point scalars retain their
source identity and physical radius. The demo's latitude/longitude segment
settings trade smoother spheres for larger time-series files.

## Current limitations

- The writer supports triangle and point PolyData, not volumetric cells or
  arbitrary Newton primitives.
- The printing snapshot includes the ten CAD printer meshes and the build
  plate. The analytical infinite ground is not included because it has no
  finite visualization extent.
- Output is ASCII for transparency and portability rather than minimum size.

## Tests

From the repository root with MoPhi available on `PYTHONPATH`:

```bash
PYTHONPATH=python python -m pytest tests/test_vtk.py
```
