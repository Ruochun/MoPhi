# MoPhi How-To Guide

This guide collects the user-facing build and run workflows for MoPhi.

---

## Demo-only Python packages

The packaged MoPhi wheel covers the supported Newton + XLB + DEME runtime path,
but some demos and visualization/movie-recording workflows may also require
extra Python packages. Install the demo bundle with:

```bash
python3.11 -m pip install "mophi[demos]"
```

The current demo extra includes:

- `torch`
- `GitPython`
- `pycollada`
- `mujoco_warp==3.6.0`
- `pyglet`
- `imageio`
- `imageio-ffmpeg`

Notes:

- `demo/newton_xlb_dem/demo_newton_xlb_dem.py` directly requires `torch`.
- Movie recording paths require `imageio` and `imageio-ffmpeg`.
- The PR feedback mentioned `mageio`, but no published PyPI package with that
  name was found during validation, so it is not included in the extra.

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
Warp, MuJoCo, XLB, and DEME automatically:

```bash
python3.11 -m pip install dist/mophi-*.whl
```

Installing from the wheel does **not** build MoPhi from source locally, but the
wheel is **not** a fully self-contained offline installer. `pip` still needs to
resolve the wheel's declared runtime dependencies (`newton`, `warp-lang`,
`mujoco`, `deme`, and `xlb[cuda]`) from either the internet or a local
wheelhouse. The same applies to the optional `mophi[demos]` extra.

For an offline installation, pre-download the MoPhi wheel plus all required
dependency wheels, then install from a local wheelhouse:

```bash
python3.11 -m pip install --no-index --find-links /path/to/wheelhouse mophi
python3.11 -m pip install --no-index --find-links /path/to/wheelhouse "mophi[demos]"
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
# Install the required Python solver packages first
pip install --upgrade newton==1.0.0 warp-lang==1.12.1 mujoco==3.6.0 "xlb[cuda]" deme

# Configure and build the coupler
cmake -B build \
      -DMOPHI_BUILD_NEWTON_XLB_DEM=ON
cmake --build build

# Run the Python demo (walking robot + XLB + DEME)
PYTHONPATH=python python3 demo/newton_xlb_dem/demo_newton_xlb_dem.py
# Run another demo (robotic arm with excavator + XLB + DEME)
PYTHONPATH=python python3 demo/newton_xlb_dem/demo_claw_newton.py
```

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
