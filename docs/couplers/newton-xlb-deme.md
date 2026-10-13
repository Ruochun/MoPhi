# Newton + XLB + DEME

## Setup

Run these commands from the repository root in the environment used for the
demo. Use a build directory named for the active Python version; `build-py312`
below assumes Python 3.12.

```bash
# Install the common runtime set from package metadata
python scripts/setup/install_dependencies.py

# Configure and build the coupler
cmake -B build-py312 \
      -DPython3_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
      -DMOPHI_BUILD_NEWTON_XLB_DEM=ON
cmake --build build-py312

# Verify the loaded extension and required class with the same interpreter.
PYTHONPATH=python python -c 'import mophi; print(mophi.mophi_core.__file__); print(mophi.NewtonXLBDEMCoupler)'
```

## Coupler overview

`NewtonXLBDEMCoupler` is a three-way co-simulation coupler:

- **Newton** (active) — drives an articulated quadruped walking robot using
  position-based dynamics (XPBD). The robot's spatial representation (body
  position + orientation quaternion per body) is extracted at every step via
  `get_robot_body_transforms()`.
- **XLB** — a JAX-based LBM fluid solver that advances each frame. The robot is
  represented as a prescribed moving obstacle whose boundary-condition masks are
  updated directly on the GPU.
- **DEME** — a Python discrete-element solver (provided by `pip install "deme"`) that advances a
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

## Provider configuration

The packaged wheel declares XLB and DEME as runtime dependencies. For source
builds, the [common dependency installer](../how-to/python-dependencies.md)
reads the same package requirements, including solver extras. Alternatively,
`MOPHI_FETCH_XLB=ON` and `MOPHI_FETCH_DEME=ON` install their configured providers
during CMake configuration.

The DEME distribution and import module are intentionally configurable. The
default `deme` distribution is installed with the `cuda12` extra and exposes
the `deme` module. To select a distribution
requirement during CMake configuration, set both names as needed:

```bash
cmake -B build-py312 \
      -DPython3_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
      -DMOPHI_PACKAGE_DEME_DISTRIBUTION="deme" \
      -DMOPHI_PACKAGE_DEME_IMPORT_MODULE=deme \
      -DMOPHI_FETCH_DEME=ON
```

At runtime, Python dependencies are described by the general provider registry
in `mophi.utils.package_provider`. It covers every runtime dependency declared
in `pyproject.toml` and keeps each pip distribution name separate from its
Python import module. The default mappings include `warp-lang` to `warp`,
`PyYAML` to `yaml`, `GitPython` to `git`, and `deme` to `deme`.

Every provider accepts the same environment overrides:

```bash
MOPHI_PACKAGE_<NAME>_DISTRIBUTION=<pip-requirement>
MOPHI_PACKAGE_<NAME>_IMPORT_MODULE=<python-module>
```

For example, a DEME-compatible provider with a different module name can be
selected with:

```bash
PYTHONPATH=python MOPHI_PACKAGE_DEME_IMPORT_MODULE=another_module python demo/...
```

Python code resolves a provider uniformly:

```python
from mophi.utils.package_provider import load_package_provider

DEME = load_package_provider("deme")
```

Changing the dependency installed with a packaged wheel remains a one-line
change to the corresponding dependency entry in `pyproject.toml`; update the
matching default in `DEFAULT_PACKAGE_PROVIDERS` at the same time.

For scenario commands, see the [demo workflows](../how-to.md#demo-workflows).
The [demo regression runner](../how-to/getting-started.md#demo-regression-runner)
can run the important demos sequentially.
