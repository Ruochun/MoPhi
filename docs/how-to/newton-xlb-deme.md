# Newton + XLB + DEME setup

```bash
# Install the required Python packages first
pip install --upgrade newton==1.0.0 warp-lang==1.12.1 mujoco==3.6.0 "xlb[cuda]" "deme" \
    torch GitPython PyYAML pycollada mujoco_warp==3.6.0 pyglet imageio imageio-ffmpeg

# Configure and build the coupler
cmake -B build-py312 \
      -DPython3_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
      -DMOPHI_BUILD_NEWTON_XLB_DEM=ON
cmake --build build-py312
```

## Demo regression runner

For a convenient end-to-end check, the repository includes a runner for the
actively developed demos linked from the [how-to index](../how-to.md). It uses the current Python interpreter,
adds the source-tree `python/` directory to `PYTHONPATH`, and waits for each
demo to finish before starting the next. Install and build the prerequisites
from this guide first; interactive viewers must be closed for the runner to
advance.

```bash
python3 scripts/demos/run_important_demos.py
```

The FERIS/Newton demo is excluded by default while that module remains less
developed. Add it explicitly with `--include-feris`. By default the runner
stops at the first failure; use `--keep-going` to exercise the remaining demos
and receive a combined failure summary. Use `--list` to inspect the selected
commands without running them.

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

XLB is part of the supported distributed install path for the Newton + XLB +
DEME workflow. The packaged wheel declares `xlb[cuda]` as a runtime
dependency. For source-tree developer builds, you can install it manually or
use the developer-convenience CMake fetch option:

```bash
pip install "xlb[cuda]"
```

Likewise, the packaged wheel declares `deme[cuda12]` as a runtime dependency. For
source-tree developer builds, you can install it manually:

```bash
pip install "deme"
```

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
MOPHI_PACKAGE_DEME_IMPORT_MODULE=another_module python demo/...
```

Python code resolves a provider uniformly:

```python
from mophi.utils.package_provider import load_package_provider

DEME = load_package_provider("deme")
```

Changing the dependency installed with a packaged wheel remains a one-line
change to the corresponding dependency entry in `pyproject.toml`; update the
matching default in `DEFAULT_PACKAGE_PROVIDERS` at the same time.
