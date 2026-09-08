# XLB runtime and lattice mapping

`python/mophi/couplers/newton_xlb/runtime.py` centralizes XLB mechanics that are
independent of a particular fluid scene.

`XLBStepperState` owns the two population fields, boundary masks, relaxation
rate, timestep counter, and established ping-pong ordering for one XLB step.
Pass it as the XLB object to `NewtonXLBDEMCoupler`; `step_xlb()` then advances
exactly one fluid step, leaving the demo in control of substep ratios.

`XLBPhysicalScaling` validates an isotropic physical domain and converts SI
velocity and kinematic viscosity to lattice units. The standalone helpers
normalize XLB velocity-set storage and calculate BGK omega from lattice
viscosity.

The module does not choose a grid, velocity set, collision model, boundary
conditions, material properties, physical timestep, or multiphysics schedule.
Those choices define the experiment and remain in each demo. Current physical
scaling supports isotropic cells only.

Run the CPU-safe validation suite with:

```bash
PYTHONPATH=python python tests/test_pairwise_gpu_exchange.py
```

The ANYmal and humanoid-swimming demos exercise the complete CUDA stepping
path. Their run commands and prerequisites are listed in `docs/how-to.md`.
