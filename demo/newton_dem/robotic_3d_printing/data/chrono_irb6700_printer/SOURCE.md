# ABB IRB6700 printer asset source

These files support `demo/newton_dem/robotic_3d_printing/demo_dfc_3d_printing.py`.
They describe the ABB robot, support beam, outer auger housing, and inner auger
used as the Newton-managed printing device.

The assembly metadata and OBJ meshes were copied from `Demo_for GPU/Demo_Robot`
at commit `406c9efeac07ccd4123993d03bb6aa0ad665e8c7` of:

<https://github.com/Computational-Mechanics-Material-Models/chrono-mechanics>

`Robot_export_j2.json` is retained as the authoritative source for body mass
properties, initial poses, mesh assignments, joint frames, and the tool-center
point. MoPhi reconstructs those records as a Newton articulation; it does not
load the Chrono simulation or use its particle implementation.

`body_10_1_inner.obj` is a derived contact-only version of the outer auger
housing. It contains the inward-facing CAD surface patches along the material
flow path and intentionally omits the housing thickness, exterior wall, end
caps, and unrelated small bores. The original `body_10_1.obj` remains the
visual mesh. The derived mesh is intentionally open and non-watertight.
