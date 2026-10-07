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
- [GRAB recorded-versus-physics cup grasp](how-to/grab.md) — the complete workflow:
  prepare NPZ files with `scripts/grab/convert_grab_hand_motion.py`, simulate with
  `demo/newton_dem/grab/demo_grab_contact.py`, and produce the comparison with
  `demo/newton_dem/grab/render_grab_comparison.py`. Covers licensed assets,
  MoPhi/DEME/Newton prerequisites, recorded-only viewing, cameras, and diagnostics.

## Supporting references

- [Visualization](visualization.md)
- [Newton–DEME coupling](newton-deme-coupling.md)
- [Pairwise GPU exchange](pairwise-gpu-exchange.md)
- [XLB runtime and lattice mapping](xlb-runtime.md)
- [G-code](gcode.md)
- [Mesh contact patches](mesh-contact-patches.md)
- [Mesh sampling](mesh-sampling.md)
- [VTK output](vtk-output.md)
