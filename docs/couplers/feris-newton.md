# FERIS + Newton

## Setup

Build with `MOPHI_BUILD_FERIS_NEWTON=ON`; this demo needs the native FERIS
solver, CUDA Toolkit, Eigen, and Newton/Warp/MuJoCo. Use the fetch option below
to build FERIS and its prerequisites as configured by the project.

Install the [common Python dependencies](../how-to/python-dependencies.md), build the
base extension for the active Python with the required options above, and use
an NVIDIA CUDA GPU with OpenGL support for visual demos. Commands below run
from the repository root with `PYTHONPATH=python`; omit that setting for an
installed MoPhi package. See [build workflows](../how-to/python-builds.md) and the
[complete demo matrix](../how-to.md#demo-build-requirements).

```bash
# Configure: fetch FERIS, install Newton via pip, and build the coupler
cmake -B build-py312 \
      -DPython3_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
      -DMOPHI_FETCH_FERIS=ON \
      -DMOPHI_FETCH_NEWTON=ON \
      -DMOPHI_BUILD_FERIS_NEWTON=ON
cmake --build build-py312

# Run the FERIS + Newton demo
PYTHONPATH=python python3 demo/feris_newton/demo_feris_newton.py
```

## Usage and current limitations

`FERISNewtonCoupler` owns the native FERIS objects and holds Python references
to Newton's model and solver. Pass the Newton objects to `initialize()`, advance
with its existing `step()` method, and release resources with `finalize()`.
The [demo](../../demo/feris_newton/demo_feris_newton.py) shows the setup.

This is a groundwork implementation: the combined step advances the solvers,
but geometry transfer and Newton-to-FERIS force feedback remain unfinished.
The C++ hooks `GetFERISNodePositions()` and `SetFERISNodeForces()` are placeholders
and are not exposed as Python methods. Do not treat this as validated two-way
contact coupling. Its combined stepping API predates the per-solver convention
used for new couplers.

`MOPHI_FETCH_NEWTON=ON` installs the configured Newton/Warp/MuJoCo set during
configuration. For installation before configuring, use the shared
[dependency installer](../how-to/python-dependencies.md).
