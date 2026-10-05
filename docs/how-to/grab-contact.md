# GRAB cup contact checks (DEME + Newton rendering)

[`demo_grab_contact.py`](../../demo/newton_dem/grab/demo_grab_contact.py) is the
next step after [recorded playback](grab-playback.md). DEME owns and advances
the cup; Newton only renders its current geometry. There is no second rigid-body
integrator and no recorded cup pose is imposed after initialization.

## Prerequisites

Commands below use the source-tree development workflow: run from the repository
root with `PYTHONPATH=python`. With an installed MoPhi wheel, omit that prefix
and remove any source-checkout path previously exported in `PYTHONPATH` so Python
uses the installed package. See [Python build workflows](python-builds.md).

Use the same `ML` conda environment, MoPhi build, graphics setup, and prepared
assets as [GRAB playback](grab-playback.md#prerequisites). Additionally install
DEME 3.0.14 with CUDA 12 runtime support, trimesh for initial mass properties,
fast-simplification for the collision proxy, and rtree for geometry checks:

```bash
conda activate ML
python -m pip install 'deme[cuda12]==3.0.14' trimesh fast-simplification rtree
python -c 'import importlib.metadata as m; print(m.version("deme"))'
```

An NVIDIA CUDA GPU is required. The first run can spend several minutes
compiling DEME kernels; later compatible runs use its kernel cache.
The tested geometry packages in `ML` are trimesh 4.12.2,
fast-simplification 0.2.0, and rtree 1.4.1.
`--no-render` runs the same physics and writes diagnostics without creating an
OpenGL context. No MoPhi C++ solver coupler is required.

Required prepared input is `data/grab/s2/mug_drink_2_object.npz`; `hand_push`
also needs `mug_drink_2_right.npz`. Prepare them using the converter described in
[GRAB asset preparation](grab-motion.md). All outputs remain beneath
`output/demo_grab_contact/<case>/` by default.

## Experiments

| `--case` | What is prescribed | What the check establishes |
|---|---|---|
| `drop` | A stationary analytical floor | The dynamic cup falls, makes contact, and settles. |
| `hand_push` | A frozen GRAB hand shape moves sideways after settling | Hand–cup mesh contact pushes a freely moving cup. This is scripted rigid hand motion, not the full GRAB trajectory. |
| `slide` | A mesh plate moves sideways, with matching owner position and velocity | Tangential contact transmits motion to the cup through friction. |
| `deform_slide` | The same plate moves through vertex updates while its owner stays fixed | Tests whether mesh geometry updates alone provide the surface velocity needed by friction. |

Run the drop and frozen-hand push with Newton rendering and movies:

```bash
PYTHONPATH=python python demo/newton_dem/grab/demo_grab_contact.py --case drop --headless --save-movie --check
PYTHONPATH=python python demo/newton_dem/grab/demo_grab_contact.py --case hand_push --headless --save-movie --check
```

Omit `--headless` for a visible window. Closing it stops the run and marks it
incomplete. Optional `--usd` records mesh geometry to USD instead of OpenGL;
it requires `usd-core` and cannot be combined with `--save-movie`.

Compare physically equivalent moving surfaces:

```bash
PYTHONPATH=python python demo/newton_dem/grab/demo_grab_contact.py --case slide --no-render --check
PYTHONPATH=python python demo/newton_dem/grab/demo_grab_contact.py --case deform_slide --no-render
```

The second case is intentionally a capability diagnostic: a stationary cup
despite the moving plate means deformation velocity is absent from the contact
calculation. Its `summary.json` then reports `surface_velocity_transferred=false`
and `passed=false`. Adding `--check` makes that result exit with an error. Do not
interpret geometric playback of the plate as evidence of correct friction.

Use zero friction as a control and halve the physics timestep for a convergence
comparison (keep outputs separate):

```bash
PYTHONPATH=python python demo/newton_dem/grab/demo_grab_contact.py --case slide --friction 0 --no-render --check \
    --output-dir output/demo_grab_contact/frictionless
PYTHONPATH=python python demo/newton_dem/grab/demo_grab_contact.py --case slide --dt 0.00005 --no-render --check \
    --output-dir output/demo_grab_contact/half_dt
```

With zero friction the check instead requires negligible cup transport, since
that control deliberately removes friction. `--duration`, `--speed`,
`--hand-frame`, `--object`, and `--hand` select other controlled runs. All remaining
physical, rendering, and acceptance parameters are in the top configuration
block. Physics timestep and recording cadence are independent.

## Physical assumptions and recorded outputs

- Cup mass is an explicit experimental parameter, not recovered from GRAB.
  CoM and inertia assume homogeneous density within the source mesh's enclosed
  volume. The cup is initialized upright using its source mesh frame, with a
  small gap above the support, rather than at a recorded world pose.
- The full cup mesh supplies rendering and mass properties. The default collision
  proxy targets 6,000 triangles to reduce contact-detection cost. Its closed volume
  and bidirectional sampled surface error are checked before initialization;
  the limits are explicit configuration parameters. This is a sampled estimate,
  not a guaranteed Hausdorff bound. `--collision-faces 0` uses the full mesh,
  which is much slower in these tests. Diagnostics report proxy error and count.
  The hand has one frozen shape, rotated so its thin direction points along the push direction;
  hand–floor contact is disabled in the pushing case.
  Its wrist boundary is open. DEME warns about its volume/MOI; the prescribed
  hand uses explicit placeholder mass/inertia and no hand dynamics are inferred
  from this open surface. Contact near the wrist cut is not validated here.
- The materials are deliberately compliant calibration parameters, not measured
  ceramic or skin properties. The prescribed surface receives no force feedback.
- The motion starts with a velocity ramp after settling. Rigid motion uses
  matching DEME position/velocity prescriptions evaluated at physics steps.
  `deform_slide` calls `Tracker.UpdateMesh` with indexed vertex coordinates,
  not flattened triangles, at every step and keeps owner velocity zero.
- Mesh–mesh and mesh–analytical contacts are enabled explicitly with
  `SetMeshUniversalContact(True)`.

Each run writes:

- `trajectory.npz`: timestamps, cup CoM pose (xyzw), linear/angular velocity,
  global contact force and torque about CoM, lowest collision-mesh vertex, and
  mass/frame data.
- `summary.json`: numerical metrics and named acceptance checks; `--check` fails
  if any selected check fails. Penetration and peak forces are sampled diagnostics,
  not continuous-time guarantees. Lowest-vertex height measures support-plane
  penetration, not hand–cup penetration.
- Initialization OBJ meshes, plus `contact.mp4`/`last_frame.png` when requested,
  or `contact.usdc` for USD export.

These small diagnostics intentionally synchronize DEME and read host state for
inspection. The deformation diagnostic also stages vertex updates on the host.
They are not a device-resident deforming-hand coupler or a grasping controller.
Full GRAB grasping remains contingent on validating deformation velocities,
contact consistency, and force magnitudes.

DEME uses a float32 physics timestep. The demo passes exact multiples of that
stored value to `DoDynamicsThenSync`, avoiding accidental extra steps caused by
rounding a nominal double-precision duration upward. Physics timestamps use the
same value and the drop check compares pre-contact motion against free fall.

## DEME 3.0.14 deformation-velocity result

With the default cup, proxy, and material parameters in `ML`, the 0.8-second
checks produced these approximate results:

| Check | Result |
|---|---|
| Drop | Passed; pre-contact free-fall error below 0.1 micrometre, sampled support penetration below 0.8 mm. |
| Frozen-hand push | Passed; cup moved 32 mm and followed the hand at about 0.080 m/s. |
| Owner-driven plate | Passed; cup moved 40 mm at about 0.081 m/s late average speed. |
| Zero-friction plate control | Passed; negligible horizontal cup transport. |
| Half-timestep plate | Passed; final cup displacement changed by about 0.094 mm. |
| Vertex-driven plate | Geometry and support contact checks passed; tangential velocity-transfer check failed. |

These are calibration checks for the stated proxy and assumed materials, not
validation against measured human grasp forces or a fully converged force history.

In the tested `ML` environment, rigid owner-driven sliding transports the cup,
but the equivalent `UpdateMesh`-driven translation does **not**. In the default
experiment the former moves the cup about 40 mm; the latter has negligible net
horizontal transport. Both retain support contact and their surface coordinates
match the commanded motion. Thus the deformation case correctly reports
`surface_velocity_transferred=false` rather than passing the calibration.

The default force model uses owner linear and angular velocity. Updating node
positions does not supply the additional local surface velocity needed for
friction and damping. Full GRAB finger deformation must not be declared physically
validated until that velocity contribution is implemented and tested. The
frozen-hand push is a valid rigid prescribed-motion experiment and does not
resolve this remaining deforming-hand requirement.

## Geometry preparation and tests

`prepare_mesh_principal_frame` in
[`mophi.couplers.newton_deme.mesh_owners`](../../python/mophi/couplers/newton_deme/mesh_owners.py)
computes and validates the closed-mesh CoM, transforms vertices to principal axes,
and supplies positive principal moments. It rejects open/invalid volumes and
does not repair or simplify geometry. See [mesh-owner preparation](../newton-deme-coupling.md)
for the reusable API.

```bash
PYTHONPATH=python python -m unittest discover -s tests -p 'test_grab*.py' -v
```
