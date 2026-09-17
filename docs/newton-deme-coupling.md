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

`NewtonDEMEOwnerMap` maps an ordered sequence of DEME owners to Newton bodies.
Newton body indices may repeat, allowing several contact meshes to contribute
to one Newton body. The DEME owners must form one increasing, consecutive span
because the transfer uses DEME's bulk device methods.

For one-way kinematic coupling, `NewtonDEMEOwnerPoseExchange` applies optional
body-local offsets and orientations, then writes Newton body poses to DEME
owners through the same bulk device interface. It deliberately does not send
velocities, retrieve contact response, or step either solver, so adopting it
does not silently change a pose-driven demo's contact semantics.

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

A pose-only workflow uses the same mapping without allocating contact-feedback
buffers:

```python
from mophi.couplers.newton_deme import NewtonDEMEOwnerPoseExchange

pose_exchange = NewtonDEMEOwnerPoseExchange()
pose_exchange.initialize(newton_model, deme_solver, owner_map, device)
pose_exchange.set_deme_owner_pose_from_newton(newton_state)
```

The three calls remain separate so the application controls each solver's
stepping cadence. `set_deme_owner_state_from_newton` uploads world position,
orientation, linear velocity, and global angular velocity. The wrench call
retrieves global force and torque resultants and scatters them into a
Newton-sized `wp.spatial_vector` array.

Diagnostics that deliberately compare feedback semantics can call
`get_deme_contact_accelerations_to_device(...)` to fill caller-owned CUDA
arrays with DEME linear and local/global angular contact accelerations. The
diagnostic retains the explicit acceleration-to-wrench operator, while MoPhi
owns the device validation, owner-span arguments, and solver API calls.

Several DEME owners may map to the same Newton body. Their forces and torques
are summed before the Newton body-force entry is written. To combine DEME with
an existing external-force array, pass that array as `destination` and disable
clearing:

```python
coupler.write_deme_contact_wrenches_to_newton(
    destination=combined_body_forces,
    clear_destination=False,
)
```

## Mesh-owner preparation

`extract_newton_shape_triangle_meshes(...)` reads selected Newton mesh shapes,
validates their triangle topology, applies shape scale and local transform, and
returns both scaled source-space and body-local vertices. This keeps generic
Newton mesh extraction out of robot-specific demos.

`NewtonDEMEMeshOwnerSpec` describes one DEME proxy using body-local triangle
geometry, its Newton body index, DEME family, mass, and principal MOI. Physical
properties are mandatory so DEME does not silently derive proxy-body dynamics
from mesh volume.

`combine_triangle_meshes` combines several body-local mesh pieces and adjusts
their indices. `add_deme_mesh_owners` writes the resulting OBJ files, creates
DEME mesh owners, assigns their initial Newton transforms and physical
properties, and retains trackers. After `DEMSolver.Initialize()`, use
`owner_map_from_mesh_bindings` to obtain the runtime mapping:

```python
bindings = add_deme_mesh_owners(
    deme_solver,
    material,
    mesh_specs,
    newton_state.body_q.numpy(),
    output_directory,
)
# Configure DEME and call deme_solver.Initialize() before reading owner IDs.
owner_map = owner_map_from_mesh_bindings(bindings)
```

OBJ generation is an initialization-time operation and may use host memory.
The GPU-only claim applies to recurring state and wrench exchange.

## Supported behavior

- CUDA-only physics-state and wrench transfer through DEME device APIs.
- CUDA-only pose transfer for one-way kinematic DEME owners.
- Arbitrary Newton body order, including multiple owners per Newton body,
  mapped to one consecutive DEME owner range.
- Runtime checks for required DEME methods and a shared CUDA device ordinal.
- Explicit synchronization before DEME consumes Warp-produced buffers.
- Independent DEME substepping.
- Per-body wrench accumulation with either replacement or addition to an
  existing Newton body-force array.
- Reusable body-local triangle-mesh combination, OBJ serialization, and DEME
  owner registration with explicit physical properties.

## Current limitations

- The DEME device calls are synchronous; stream/event interoperability is not
  yet available, so the CPU still orchestrates the exchange. Synchronization is
  isolated behind internal handoff methods so a future DEME stream/event API
  can replace the device-wide barrier without changing callers.
- All owners participating in one coupler must occupy one consecutive DEME ID
  span.
- Mesh loading still uses OBJ files because DEME does not yet expose equivalent
  device-resident mesh construction through this integration.
- The caller still chooses contact materials, family prescriptions, collision
  filtering, and how source geometry becomes `NewtonDEMEMeshOwnerSpec` objects.

## Tests

Run the CPU-safe mapping and validation tests with:

```bash
PYTHONPATH=python python -m unittest discover -s tests -p 'test_newton_deme_coupler.py'
```

The end-to-end GPU path is exercised by the robot-fighting demo and by
`demo/newton_xlb_dem/deme_getter_comparison/`.
