# VTK snapshot output

This document describes `python/mophi/utils/vtk.py`, the solver-independent
legacy VTK PolyData writer used by MoPhi demos.

## Purpose and design boundary

Newton does not currently provide a native VTK viewer or exporter. MoPhi's
`write_vtk_polydata` utility writes an explicitly selected triangle snapshot so
it can be inspected in ParaView. The physics solver remains responsible for
state and geometry; the caller remains responsible for transforming geometry
into world coordinates and deciding which frame and objects to export.

The writer has no dependency on the VTK Python package. It produces portable
ASCII legacy `.vtk` files containing points, triangles, and optional integer
cell scalars.

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

The G-code printing demo writes its final ABB printer pose to
`output/demo_dfc_3d_printing/dfc_3d_printing_final_frame.vtk` when
`SAVE_FINAL_VTK_FRAME` is enabled. Open that file directly in ParaView. Its
`scene_part_id` cell array distinguishes CAD parts and the build plate, and
`is_nozzle_contact_mesh` is `1` for the outer-auger mesh registered for Newton
particle contact. The run's `run_metadata.json` maps the numeric part IDs to
names.

## Current limitations

- The writer supports triangle PolyData, not volumetric cells, particles,
  animation, or arbitrary Newton primitives.
- The printing snapshot includes the ten CAD printer meshes and the build
  plate. The analytical infinite ground is not included because it has no
  finite visualization extent.
- Output is ASCII for transparency and portability rather than minimum size.

## Tests

From the repository root with MoPhi available on `PYTHONPATH`:

```bash
PYTHONPATH=python python -m pytest tests/test_vtk.py
```
