# MoPhi How-To Guide

This guide collects the user-facing build and run workflows for MoPhi.

All demos write generated files by default under
`<repository-root>/output/<demo-name>/`. The repository-level `output/`
directory is git-ignored so movies, USD scenes, state snapshots, and other
generated artifacts do not appear in normal staging operations.

Visual demos include a persistent world reference: RGB segments show the
positive X, Y, and Z directions, and a yellow endpoint-marked bar shows one
metre of world-space length.

---

## Python packages used by MoPhi workflows

The packaged MoPhi wheel declares the following Python packages as regular
runtime dependencies, because they are used across MoPhi's current Python-side
workflows, demos, and visualization/movie-recording paths:

- `torch`
- `GitPython`
- `PyYAML`
- `pycollada`
- `mujoco_warp==3.6.0`
- `pyglet`
- `imageio`
- `imageio-ffmpeg`

Notes:

- `demo/newton_xlb_dem/anymal_robot_multiphysics/demo_anymal_robot_multiphysics.py` directly requires `torch`.
- Movie recording paths require `imageio` and `imageio-ffmpeg`.
- When installing from a source checkout instead of a built wheel, install these
  packages alongside the solver packages that your workflow needs.

---

## Quick start

```bash
# 1. Clone MoPhi (--recurse-submodules is required for MoPhiEssentials)
git clone --recurse-submodules https://github.com/Ruochun/MoPhi.git
cd MoPhi

# If you already cloned without --recurse-submodules, initialise the submodule manually:
#   git submodule update --init

# 2. Configure with the desired co-simulation coupler enabled
cmake -B build \
      -DMOPHI_FETCH_FERIS=ON \
      -DMOPHI_FETCH_NEWTON=ON \
      -DMOPHI_BUILD_FERIS_NEWTON=ON
cmake --build build

# 3. Run a demo
PYTHONPATH=python python3 demo/feris_newton/demo_feris_newton.py
```

---

## Linux Python 3.11 wheel (Newton + XLB + DEME, no FERIS)

MoPhi has a standard PEP 517 packaging entry point in `pyproject.toml`
using `scikit-build-core`. The supported packaged distribution path currently
targets **Linux + Python 3.11** and enables:

- Python bindings: **ON**
- Newton + XLB + DEME coupler: **ON**
- FERIS coupler: **OFF**
- demos: **OFF**

The wheel metadata declares the runtime Python dependencies needed for the
supported distributed install experience:

- `newton==1.0.0`
- `warp-lang==1.12.1`
- `mujoco==3.6.0`
- `deme`
- `xlb[cuda]`
- `torch`
- `GitPython`
- `PyYAML`
- `pycollada`
- `mujoco_warp==3.6.0`
- `pyglet`
- `imageio`
- `imageio-ffmpeg`

Build a wheel from a checkout with submodules initialized:

```bash
git clone --recurse-submodules https://github.com/Ruochun/MoPhi.git
cd MoPhi
python3.11 -m pip install --upgrade build
python3.11 -m build --wheel
```

For Linux release distribution, repair the wheel after the build so bundled
native dependencies satisfy manylinux policy:

```bash
python3.11 -m pip install --upgrade auditwheel
auditwheel repair dist/mophi-*.whl -w dist/
```

Then install the repaired wheel with pip; its metadata will pull in Newton,
Warp, MuJoCo, XLB, DEME, and the current MoPhi Python workflow dependencies
automatically:

```bash
python3.11 -m pip install dist/mophi-*.whl
```

Installing from the wheel does **not** build MoPhi from source locally, but the
wheel is **not** a fully self-contained offline installer. `pip` still needs to
resolve the wheel's declared runtime dependencies (`newton`, `warp-lang`,
`mujoco`, `deme`, `xlb[cuda]`, `torch`, `GitPython`, `PyYAML`, `pycollada`,
`mujoco_warp`, `pyglet`, `imageio`, and `imageio-ffmpeg`) from either the
internet or a local wheelhouse.

For an offline installation, pre-download the MoPhi wheel plus all required
dependency wheels, then install from a local wheelhouse:

```bash
python3.11 -m pip install --no-index --find-links /path/to/wheelhouse mophi
```

---

## FERIS + Newton (Python-based GPU physics)

```bash
# Configure: fetch FERIS, install Newton via pip, and build the coupler
cmake -B build \
      -DMOPHI_FETCH_FERIS=ON \
      -DMOPHI_FETCH_NEWTON=ON \
      -DMOPHI_BUILD_FERIS_NEWTON=ON
cmake --build build

# Run the FERIS + Newton demo
PYTHONPATH=python python3 demo/feris_newton/demo_feris_newton.py
```

The `FERISNewtonCoupler` bridges C++ (FERIS) and Python (Newton) inside a single
`step()` call. Pass the Newton model and solver at initialization time; the coupler
manages states, forward-kinematics setup, and the per-step coupling data exchange.

**Coupling data exchange** (FERIS ↔ Newton) is handled by two methods on the C++ side:

| Method | Direction | Purpose |
|--------|-----------|---------|
| `get_feris_node_positions()` | FERIS → Newton | Deformed FEA node positions forwarded to Newton geometry |
| `set_feris_node_forces(forces)` | Newton → FERIS | Contact/body forces from Newton applied as FERIS external loads |

Newton (with required Warp and MuJoCo) is installed by MoPhi via pip at configure time when
`-DMOPHI_FETCH_NEWTON=ON`. You can also install it manually beforehand:

```bash
pip install --upgrade newton==1.0.0 warp-lang==1.12.1 mujoco==3.6.0
```

---

## Newton + XLB + DEME (three-way coupling)

```bash
# Install the required Python packages first
pip install --upgrade newton==1.0.0 warp-lang==1.12.1 mujoco==3.6.0 "xlb[cuda]" deme \
    torch GitPython PyYAML pycollada mujoco_warp==3.6.0 pyglet imageio imageio-ffmpeg

# Configure and build the coupler
cmake -B build \
      -DMOPHI_BUILD_NEWTON_XLB_DEM=ON
cmake --build build

# Run the Python demo (walking robot + XLB + DEME)
PYTHONPATH=python python3 demo/newton_xlb_dem/anymal_robot_multiphysics/demo_anymal_robot_multiphysics.py
# Run the Newton-only walking baseline demo (flat ground + light dynamic boxes)
PYTHONPATH=python python3 demo/newton_xlb_dem/anymal_robot_multiphysics/demo_anymal_robot_newton_baseline.py
# Run the DEME excavator demo (robotic arm with excavator + DEME force feedback)
PYTHONPATH=python python3 demo/newton_xlb_dem/excavation_comparison/demo_claw_newton.py
# Run the Newton coarse-cube comparison demo (same arm policy, coarse rigid terrain)
PYTHONPATH=python python3 demo/newton_xlb_dem/excavation_comparison/demo_claw_newton_cubes.py
# Run the interactive Newton humanoid walking baseline
PYTHONPATH=python python3 demo/newton_xlb_dem/humanoid_complex_environment/demo_humanoid_complex_environment.py
```

The Newton-only ANYmal baseline demo does not instantiate XLB or DEME at
runtime; it keeps the walking-policy setup on flat ground and adds only a few
light Newton-managed contact boxes for baseline comparison against the
multiphysics case.

The humanoid complex-environment demo starts from Newton 1.0.0's public
keyboard-controlled Unitree G1 walking-policy example. On first run, Newton
downloads the G1 model, policy, and YAML configuration from the public
`newton-assets` repository; no manual asset placement is required. This first
stage preserves the policy-trained ground colliders, adds convex Newton contact
proxies from the G1's visual body meshes for external-object contact, and
places lightweight dynamic boxes in its walking path. It also adds MoPhi-style
configuration, movie output, run metadata, and a final-state snapshot. Press
`I` in the viewer to walk forward into the boxes and observe body-object
contact. Interactively added collision objects are planned follow-on stages.

To download the G1 assets before launching the demo and print their cache
location:

```bash
python3 -c "import newton.utils; print(newton.utils.download_asset('unitree_g1'))"
```

Newton stores assets under the platform user cache by default, normally
`~/.cache/newton/` on Linux. Set `NEWTON_CACHE_PATH` to choose a different
persistent cache location:

```bash
NEWTON_CACHE_PATH=/path/to/newton-cache \
python3 -c "import newton.utils; print(newton.utils.download_asset('unitree_g1'))"
```

If the asset download was interrupted and files such as `usd/g1_isaac.usd`
are missing, download into a fresh cache directory:

```bash
export NEWTON_CACHE_PATH="$HOME/.cache/newton-mophi"
python3 -c "import newton.utils; print(newton.utils.download_asset('unitree_g1'))"
```

Keep `NEWTON_CACHE_PATH` set when running the demo.

`NewtonXLBDEMCoupler` is a three-way co-simulation coupler:

- **Newton** (active) — drives an articulated quadruped walking robot using
  position-based dynamics (XPBD). The robot's spatial representation (body
  position + orientation quaternion per body) is extracted at every step via
  `get_robot_body_transforms()`.
- **XLB** — a JAX-based LBM fluid solver that advances each frame. The robot is
  represented as a prescribed moving obstacle whose boundary-condition masks are
  updated directly on the GPU.
- **DEME** — a Python discrete-element solver (pip install deme) that advances a
  live particle simulation alongside the robot.

```python
import math

import mophi, newton, warp as wp

wp.init()
builder = newton.ModelBuilder()
# ... build walking-robot model (see demo script for full quadruped) ...
model = builder.finalize()
solver = newton.solvers.SolverXPBD(model)

coupler = mophi.NewtonXLBDEMCoupler()
coupler.initialize(newton_model=model, newton_solver=solver, sim_dt=2e-3)

for step in range(steps):
    # Update joint control targets for the trot gait before each step.
    targets = coupler.newton_control.joint_target_pos.numpy().copy()
    # ... set sinusoidal hip/knee angles based on step * sim_dt ...
    wp.copy(coupler.newton_control.joint_target_pos,
            wp.from_numpy(targets, dtype=wp.float32, device="cuda"))

    coupler.step_newton()  # advance the Newton rigid-body solver
    coupler.step_deme()    # advance the DEME discrete-element solver

    # Extract spatial representation: [{px,py,pz,qx,qy,qz,qw}, ...] per body.
    transforms = coupler.get_robot_body_transforms()

coupler.finalize()
```

XLB is part of the supported distributed install path for the Newton + XLB +
DEME workflow. The packaged wheel declares `xlb[cuda]` as a runtime
dependency. For source-tree developer builds, you can install it manually or
use the developer-convenience CMake fetch option:

```bash
pip install "xlb[cuda]"
```

Likewise, the packaged wheel declares `deme` as a runtime dependency. For
source-tree developer builds, you can install it manually:

```bash
pip install deme
```
