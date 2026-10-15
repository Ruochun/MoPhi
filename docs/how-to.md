# MoPhi how-to index

Use this page to find the setup or demo workflow you need. Each topic has its
own prerequisites, commands, behavior notes, and limitations.

## Setup and packaging

- [Getting started](how-to/getting-started.md) — output conventions, cloning,
  configuring, and running a first demo.
- [Python dependencies](how-to/python-dependencies.md) — shared runtime packages and
  pointers to solver-specific dependency sets.
- [Python build workflows](how-to/python-builds.md) — source-tree builds, wheel builds,
  ABI troubleshooting, manylinux repair, and offline wheelhouses.
- [Newton asset cache handling](newton-assets.md) — validated downloads,
  automatic refresh, offline limitations, and cache recovery.

## Coupler guides

- [FERIS + Newton](couplers/feris-newton.md) — prerequisites, build/run commands,
  and FEA contact data exchange.
- [Newton + DEME](couplers/newton-deme.md) — Python coupling APIs, GPU exchange,
  mesh preparation, and limitations.
- [Newton + XLB + DEME](couplers/newton-xlb-deme.md) — compiled coupler setup,
  basic usage, and solver provider configuration.

## Demo workflows

The demo commands use the **source-tree development** workflow: run from the
repository root with `PYTHONPATH=python` and MoPhi built for the active Python
interpreter. If you installed a MoPhi wheel, **omit `PYTHONPATH=python`** so Python
uses the installed package. Also remove any source-checkout path previously
exported in `PYTHONPATH`. Demo scripts, assets, and demo-specific dependencies
are still required. See [Python build workflows](how-to/python-builds.md) for
both setup options.

- [ANYmal demos](how-to/anymal-demos.md) — Newton baseline and Newton/XLB/DEME
  multiphysics behavior, including the XLB wrench diagnostic.
- [Excavation comparison demos](how-to/excavation-demos.md) — DEME terrain feedback
  and the Newton coarse-cube comparison.
- [Robotic 3D printing demos](how-to/robotic-3d-printing.md) — DFC discharge,
  G-code-driven motion, contact proxies, VTK output, and the UR10 workflow.
- [Humanoid demos](how-to/humanoid-demos.md) — interactive walking, Newton/DEME robot
  fighting, and the Newton/XLB swimming concept.
- [Flexible-hand demos](how-to/flexible-hand-demos.md) — Newton and DEME articulated
  hand workloads.
- [GRAB recorded-versus-physics cup, cylinder, cube, and sphere grasps](how-to/grab.md) — the complete workflow:
  prepare NPZ files with `scripts/grab/convert_grab_hand_motion.py`, simulate with
  `demo/newton_dem/grab/grabbing_cup/demo_grab_contact.py`, and produce the comparison with
  `demo/newton_dem/grab/grabbing_cup/render_grab_comparison.py`. The same guide
  covers the cylinder scene: `grabbing_cylinder/demo_grab_cylinder.py` and
  `grabbing_cylinder/render_cylinder_comparison.py` under `demo/newton_dem/grab/`,
  and the [cube workflow](how-to/grab.md#cube-lift-demo) with
  `grabbing_cube/demo_grab_cube.py` and `grabbing_cube/render_cube_comparison.py`,
  plus the [sphere workflow](how-to/grab.md#sphere-lift-demo) with
  `grabbing_sphere/demo_grab_sphere.py` and `grabbing_sphere/render_sphere_comparison.py`.
  Includes prerequisites, NPZ conversion, and run commands. Covers licensed assets,
  MoPhi/DEME/Newton prerequisites, recorded-only viewing, cameras, and diagnostics.

## Demo build requirements

All scripts importing `mophi` require an ABI-compatible `mophi_core` extension
(`MOPHI_BUILD_PYTHON_BINDINGS=ON`, or an installed wheel). “None” below means
no compiled **coupler** is needed. Install the [common dependencies](how-to/python-dependencies.md)
first. CUDA workloads need an NVIDIA GPU; visual viewers also need working
OpenGL drivers. Scenario guides list assets and additional dependencies.

| Demo | Runtime solvers | Required compiled coupler option | CMake demo selection |
|------|-----------------|----------------------------------|---------------------|
| FERIS/Newton | FERIS, Newton | `MOPHI_BUILD_FERIS_NEWTON=ON` | Same option |
| ANYmal baseline | Newton | None | `MOPHI_BUILD_NEWTON_XLB_DEM=ON` (group integration only) |
| ANYmal multiphysics | Newton, XLB, DEME | `MOPHI_BUILD_NEWTON_XLB_DEM=ON` | Same option |
| Claw excavation / coarse cubes | Newton, DEME / Newton | `MOPHI_BUILD_NEWTON_XLB_DEM=ON` | Same option |
| Humanoid environment / fighting | Newton / Newton, DEME | None | `MOPHI_BUILD_NEWTON_XLB_DEM=ON` (group integration only) |
| Humanoid swimming | Newton, XLB | `MOPHI_BUILD_NEWTON_XLB_DEM=ON` | Same option |
| Flexible hands (both), DEME getter comparison | Newton; DEME for DEM variants | `MOPHI_BUILD_NEWTON_XLB_DEM=ON` | Same option |
| GRAB cup/cylinder/cube/sphere | DEME; Newton rendering | None | `MOPHI_BUILD_NEWTON_DEM=ON` |
| DFC discharge | DEME; Newton scene/rendering | None | `MOPHI_BUILD_NEWTON_DEM=ON` |
| Robotic 3D printing | Newton, DEME | `MOPHI_BUILD_NEWTON_XLB_DEM=ON` | Also `MOPHI_BUILD_NEWTON_DEM=ON` |

CMake demo targets and installation additionally require `MOPHI_BUILD_DEMOS=ON`.
Direct `PYTHONPATH=python python demo/...` execution does not need demo selection
flags. Some existing Newton-only demos still instantiate the compiled three-way
class for stepping; their required build option reflects that implementation,
even though XLB/DEME may not participate in their physics. Baselines without that
class remain grouped with the three-way CMake demos for installation.

## Supporting references

- [Visualization](visualization.md)
- [Pairwise GPU exchange](pairwise-gpu-exchange.md)
- [XLB runtime and lattice mapping](xlb-runtime.md)
- [G-code](gcode.md)
- [Mesh contact patches](mesh-contact-patches.md)
- [Mesh sampling](mesh-sampling.md)
- [VTK output](vtk-output.md)
