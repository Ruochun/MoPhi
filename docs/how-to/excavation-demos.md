# Excavation comparison demos

## Prerequisites

Before running either demo, complete [Newton + XLB + DEME — Setup](../couplers/newton-xlb-deme.md#setup), then return to the run commands below.
Both demos use its compiled Newton stepper even when XLB is inactive.

Both scripts instantiate the compiled stepping class and need
`MOPHI_BUILD_NEWTON_XLB_DEM=ON`, including the Newton-only cube comparison.
The excavation scene also needs DEME. Robot assets are loaded through Newton.

Install the [common Python dependencies](python-dependencies.md), build the
base extension for the active Python with the required options above, and use
an NVIDIA CUDA GPU with OpenGL support for visual demos. Commands below run
from the repository root with `PYTHONPATH=python`; omit that setting for an
installed MoPhi package. See [build workflows](python-builds.md) and the
[complete demo matrix](../how-to.md#demo-build-requirements).

## Run

```bash
# Run the DEME excavator demo (robotic arm with excavator + DEME force feedback)
PYTHONPATH=python python3 demo/newton_xlb_dem/excavation_comparison/demo_claw_newton.py
# Run the Newton coarse-cube comparison demo (same arm policy, coarse rigid terrain)
PYTHONPATH=python python3 demo/newton_xlb_dem/excavation_comparison/demo_claw_newton_cubes.py
```

## Coupling behavior

The excavation comparison demo also uses these reusable Newton–DEME exchanges.
The plow's full local transform is applied to Newton's device state and written
to its DEME mesh owner without recurring host staging. DEME contact acceleration
and local angular acceleration are converted with owner mass and inertia into
Newton's device force array, preserving the demo's acceleration-feedback model
and one-step-lag update order. Terrain owner positions and orientations are read
directly into reusable Warp arrays for visualization; movie encoding may still
copy rendered frames to the host, but it is outside the physics exchange path.
