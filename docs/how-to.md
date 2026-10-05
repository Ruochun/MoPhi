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
- [FERIS + Newton](how-to/feris-newton.md) — build, run, and data exchange for the
  FERIS/Newton coupler.
- [Newton + XLB + DEME setup](how-to/newton-xlb-deme.md) — dependency installation,
  three-way coupler setup, provider configuration, and the serial demo runner.
- [Newton asset cache handling](newton-assets.md) — validated downloads,
  automatic refresh, offline limitations, and cache recovery.

## Demo workflows

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
- [GRAB hand motion](how-to/grab-motion.md) — convert licensed GRAB sequences, inspect
  extracted hand meshes, and render PNG or MP4 previews.
- [GRAB playback in Newton](how-to/grab-playback.md) — prerequisites and preparation
  for `python demo/newton_dem/grab/demo_grab_playback.py`; requires MoPhi,
  Newton/Warp, imageio, SciPy, and prepared GRAB NPZ files.

## Supporting references

- [Visualization](visualization.md)
- [Newton–DEME coupling](newton-deme-coupling.md)
- [Pairwise GPU exchange](pairwise-gpu-exchange.md)
- [XLB runtime and lattice mapping](xlb-runtime.md)
- [G-code](gcode.md)
- [Mesh sampling](mesh-sampling.md)
- [VTK output](vtk-output.md)
