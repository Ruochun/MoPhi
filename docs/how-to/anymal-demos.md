# ANYmal demos

## Run

```bash
# Run the Python demo (walking robot + XLB + DEME)
PYTHONPATH=python python3 demo/newton_xlb_dem/anymal_robot_multiphysics/demo_anymal_robot_multiphysics.py
# Run the Newton-only walking baseline demo (flat ground + light dynamic boxes)
PYTHONPATH=python python3 demo/newton_xlb_dem/anymal_robot_multiphysics/demo_anymal_robot_newton_baseline.py
```

## Multiphysics demo

The ANYmal three-way demo is the first demo migrated to MoPhi's centralized
GPU exchange modules. During its simulation loop, Newton foot-proxy poses flow
to DEME through device setters; DEME contact accelerations, mass, and inertia
flow back into Newton body forces through device getters and Warp kernels;
Newton's base transform updates the XLB obstacle masks on-device; and DEME
particle positions are read directly into reusable Warp arrays. Host copies of
the XLB velocity field and rendered frames remain visualization operations, not
co-simulation data exchange. The demo preserves its historical acceleration-
based DEME feedback rather than changing to wrench reduction during migration.
XLB uses an explicit lattice-to-SI mapping and advances several smaller fluid
steps inside every Newton step; its flow evolution is therefore tied to the
same physical simulation clock rather than to rendering cadence. The demo can
also feed an ad-hoc stationary-wall momentum-exchange wrench back to Newton to
exercise the complete GPU two-way path. Because the moving mask does not impose
the Newton wall velocity, this feedback is illustrative rather than a validated
moving-boundary fluid model. To keep that proof of concept runnable, the demo
applies a small gain and force/torque caps on the GPU. These are explicitly
nonphysical numerical stabilization tricks that contain mask-change impulses;
they do not make the feedback hydrodynamically meaningful.

**TODO:** Remove these tricks when true moving-boundary treatment and true force
feedback are available.

To diagnose that intentionally ad-hoc XLB wrench separately from the robot,
run the small box-based integration script:

```bash
PYTHONPATH=python python tests/newton_xlb_halfway_wrench_diagnostic.py
```

It requires the same CUDA Newton/XLB build as the ANYmal demo and writes its
JSON results under `output/newton_xlb_halfway_wrench_diagnostic/`. Independent
cases test a one-step Newton wrench pulse, equilibrium symmetry, fixed-mask
flow, zero-flow mask motion, and force-only, torque-only, and full feedback.
Each case runs in a fresh process because XLB's compiled Warp stepper retains
boundary-condition state that would otherwise contaminate sequential cases.
The diagnostic intentionally uses a relatively viscous fluid so its BGK
relaxation time stays safely above the marginal `tau = 0.5` limit; it is not an
air-property calibration.
An offset-center-of-mass case reports the torque both about the body origin and
after the corresponding COM shift.
The script is diagnostic evidence for the coupling implementation; it does not
validate stationary bounce-back as a moving-wall physical model.

## Newton baseline

The Newton-only ANYmal baseline demo does not instantiate XLB or DEME at
runtime; it keeps the walking-policy setup on flat ground and adds only a few
light Newton-managed contact boxes for baseline comparison against the
multiphysics case.
