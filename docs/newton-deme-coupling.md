# Newton–DEME GPU contact coupling

This document describes the reusable Python coupling code under
`python/mophi/couplers/newton_deme/`.

## Purpose and boundary

`NewtonDEMEContactCoupler` transfers mapped Newton rigid-body motion to DEME
mesh owners and returns DEME contact wrenches to a Newton-compatible body-force
array. The per-step physics payload remains in CUDA memory.

The coupler directly retains the Newton model and DEME solver supplied by the
caller. It owns only the body/owner mapping and its GPU exchange buffers. It
does not create either solver, construct contact geometry, select materials or
families, run Newton, or decide the relative solver schedule. Those decisions
remain with the calling workflow.

`NewtonDEMEOwnerMap` maps an arbitrary ordered subset of Newton bodies to one
increasing, consecutive DEME owner span. The consecutive span is required by
DEME's bulk device methods.

## Basic usage

Create and initialize the Newton and DEME solvers and their bodies first. The
DEME owners driven from Newton should use no-expression prescribed linear and
angular velocity when DEME must retain each Newton-fed velocity during its
substeps while integrating position and orientation.

```python
from mophi.couplers.newton_deme import NewtonDEMEContactCoupler, NewtonDEMEOwnerMap

owner_map = NewtonDEMEOwnerMap(
    newton_body_indices=[3, 8, 11],
    deme_owner_ids=[20, 21, 22],
)
coupler = NewtonDEMEContactCoupler()
coupler.initialize(newton_model, deme_solver, owner_map, device)

coupler.set_deme_owner_state_from_newton(newton_state)
coupler.step_deme(deme_substeps)
coupler.write_deme_contact_wrenches_to_newton()
warp.copy(newton_state.body_f, coupler.newton_body_forces)
```

The three calls remain separate so the application controls each solver's
stepping cadence. `set_deme_owner_state_from_newton` uploads world position,
orientation, linear velocity, and global angular velocity. The wrench call
retrieves global force and torque resultants and scatters them into a
Newton-sized `wp.spatial_vector` array.

## Supported behavior

- CUDA-only physics-state and wrench transfer through DEME 3.0.9 device APIs.
- Arbitrary Newton body order mapped to one consecutive DEME owner range.
- Runtime checks for required DEME methods and a shared CUDA device ordinal.
- Explicit synchronization before DEME consumes Warp-produced buffers.
- Independent DEME substepping.
- Zero-filled Newton force entries for bodies outside the coupling map.

## Current limitations

- The DEME device calls are synchronous; stream/event interoperability is not
  yet available, so the CPU still orchestrates the exchange.
- One DEME owner maps to one distinct Newton body. Multiple owners per body and
  wrench accumulation are not yet supported.
- Wrenches replace the coupler's output array. Applications combining multiple
  external-force sources must perform their own accumulation before Newton's
  step.
- Contact mesh preparation and DEME owner creation remain workflow-specific.
- DEME owner mass and moment of inertia must be configured explicitly by the
  caller; the coupler does not infer physical properties from proxy geometry.

## Tests

Run the CPU-safe mapping and validation tests with:

```bash
PYTHONPATH=python python -m unittest discover -s tests -p 'test_newton_deme_coupler.py'
```

The end-to-end GPU path is exercised by the robot-fighting demo and by
`demo/newton_xlb_dem/deme_getter_comparison/`.
