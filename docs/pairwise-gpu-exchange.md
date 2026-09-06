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

The current boundary is an axis-aligned, stationary halfway-bounce-back mask.
It does not encode wall velocity, hydrodynamic force reduction, rotated
geometry, or calculate Newton force feedback by itself.

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
