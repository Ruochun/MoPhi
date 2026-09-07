# Pairwise GPU exchange helpers

The pure-Python modules under `python/mophi/couplers/` extract reusable data
transport from the Newton + XLB + DEME demos. They move solver-owned state or
update coupling masks; they do not implement contact, fluid, or particle-force
models.

## Newton--XLB moving boundaries

`mophi.couplers.newton_xlb` maps world-space axis-aligned boxes into an XLB
grid and updates `bc_mask` and `missing_mask` in place with Warp kernels. Use
`prescribed_box_grid(...)` for generic world bounds and
`MovingBoxBoundary.update(...)` to retain the previous bounds between frames.
The existing robot demos retain thin robot-shape compatibility wrappers, but
the GPU kernels now live in the package.

For end-to-end device transport, `NewtonBodyBoxBoundary.update_from_newton(...)`
reads Newton's device-resident `body_q` array and rewrites the XLB masks without
computing grid bounds on the host. It retains copies of the original masks so
cells vacated by the moving body recover their configured boundary values.

In the reverse direction, `NewtonXLBWrenchExchange` scatters global force and
torque arrays produced by an XLB coupling kernel into a Newton
`body_f`-compatible device array. Computing those hydrodynamic wrenches remains
the responsibility of the chosen fluid coupling model.

`NewtonXLBHalfwayBounceBackWrench` provides an explicitly ad-hoc demonstration
path for stationary halfway-bounce-back masks. It sums reflected lattice-link
momentum and its moment about a Newton body center on the GPU, applies the
configured lattice-to-SI scale, and supplies the arrays consumed by
`NewtonXLBWrenchExchange`. It ignores moving-wall velocity and is not a
scientifically valid moving-boundary force model; use it only to demonstrate
the two-way data path until XLB provides a validated moving-wall reaction.

For a runnable two-way proof of concept, `NewtonXLBWrenchExchange` also exposes
`write_numerically_limited_xlb_wrenches_to_newton(...)`. It applies a gain and
independent force/torque magnitude caps entirely on the GPU before scattering
the wrench. This operation is a nonphysical numerical stabilization trick for
containing moving-mask impulses; it does not improve the boundary model or
make the resulting wrench quantitatively meaningful. Demos that use it must
say so at the call site and keep the gain and limits visible as scenario
configuration.

**TODO:** Remove this numerical treatment when true moving-boundary treatment
and true force feedback are available.

The current boundary is an axis-aligned, stationary halfway-bounce-back mask.
It does not encode wall velocity or rotated geometry. Its optional ad-hoc
wrench reduction must not be interpreted as validated hydrodynamic feedback.

**TODO:** Replace this path with true moving-boundary treatment and force feedback.

## Newton--DEME particle state

`NewtonDEMEParticleExchange` reads a consecutive span of DEME clump-owner
positions, velocities, and orientations directly into reusable Warp arrays on
Newton's CUDA device:

```python
exchange = NewtonDEMEParticleExchange()
exchange.initialize(deme, first_owner_id, owner_count, device)
positions, velocities = exchange.read_deme_particle_state()
positions, orientations = exchange.read_deme_particle_poses()
```

The caller chooses the owner span and consumes the arrays in Newton-side Warp
kernels. This helper does not create particles, infer owner IDs, step either
solver, or apply particle forces to Newton. Mesh-owner contact feedback remains
the responsibility of `NewtonDEMEContactCoupler`.
Device accelerations, or device forces converted with DEME's owner masses, can
be queued with `queue_deme_next_step_accelerations(...)` and
`queue_deme_next_step_forces(...)`, respectively.

For legacy simulations whose physics is defined in terms of DEME contact
acceleration rather than a reduced wrench, `NewtonDEMEContactAccelerationCoupler`
updates offset proxy poses from Newton and performs the mass/MOI conversion and
Newton body-force scatter entirely on-device. This preserves that response
model while removing tracker-based host exchange.

## XLB--DEME particle state

`XLBDEMEParticleExchange` provides the same DEME owner-state handoff for XLB
kernels. `queue_deme_next_step_accelerations(...)` passes device-resident linear
and optional local angular accelerations back to DEME. If an XLB coupling law
produces forces, `queue_deme_next_step_forces(...)` converts them to acceleration
on the GPU using masses retrieved from DEME.

The exchange layer intentionally does not choose particle-to-grid interpolation,
drag or buoyancy laws, or force spreading. Those are coupling physics and must
produce the force/acceleration arrays consumed by this transport layer. Global
torques must likewise be converted by the coupling model to DEME's local-frame
angular accelerations before submission.

DEME 3.0.11 device getters and next-step acceleration setters are synchronous.
Both solvers must use the
same CUDA device. Cross-stream event handoff can replace this synchronization
boundary when DEME exposes stream/event-aware calls.

## Tests

From the repository root:

```bash
PYTHONPATH=python python -m unittest discover -s tests -p 'test_pairwise_gpu_exchange.py'
```
