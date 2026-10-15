# GRAB grasps: recorded motion versus contact physics

This workflow reconstructs recorded hand/object scenes, lets **DEME** move the object
through contact with the prescribed hand, and uses **Newton** to render both
recorded and simulated motion. The comparison exposes the gap between fitted
motion data and a physically reproducible grasp. All scenes default to ten
spread-seed patches for object and hand. See [current results](#current-validation-results)
for measured outcomes and the rotation convention below for asset preparation.

There are three workflow steps. The cup entry points are:

| Step | Script | Default |
|---|---|---|
| Prepare | [`convert_grab_hand_motion.py`](../../scripts/grab/convert_grab_hand_motion.py) | Convert `s2/mug_drink_2`, right hand and cup |
| Simulate | [`demo_grab_contact.py`](../../demo/newton_dem/grab/grabbing_cup/demo_grab_contact.py) | Recorded hand attempts to grab a dynamic cup |
| View | [`render_grab_comparison.py`](../../demo/newton_dem/grab/grabbing_cup/render_grab_comparison.py) | Side-by-side recorded/simulated grasp movie, handle/back-of-hand camera |

The scene folders under `demo/newton_dem/grab/` are `grabbing_cup/`,
`grabbing_cylinder/`, `grabbing_cube/`, and `grabbing_sphere/`. Each has its own
`scene_config.py` and simulation/viewer entry points. They share the experiment
loop in [`grasp_simulation.py`](../../demo/newton_dem/grab/grasp_simulation.py)
and renderer in [`grasp_viewer.py`](../../demo/newton_dem/grab/grasp_viewer.py).
The shared modules receive configuration explicitly; switching scenes does not
modify the other scene's configuration. Geometry preparation and mesh partitioning
continue to use MoPhi utilities. DEME remains responsible for contact and object
integration; Newton renders the resulting geometry. The demo stages hand vertices
through host arrays at each DEME step and reads owner states at diagnostic and
rendering intervals.

The numbered workflow below is for the cup. The [cylinder workflow](#cylinder-lift-demo),
[cube workflow](#cube-lift-demo), and [sphere workflow](#sphere-lift-demo) use
the same prerequisites and converter.

## Noteworthy: GRAB object rotation convention

GRAB's [official ObjectModel](https://github.com/otaheri/GRAB/blob/284cba757bd10364fd38eb883c33d4490c4d98f5/tools/objectmodel.py#L86)
transforms row-vector vertices as `V @ R + translation`. SciPy's
`Rotation.apply(V)` instead evaluates `V @ R.T`. The converter therefore exports
`Rotation.from_rotvec(global_orient).inv().as_quat()` as an active local-to-world
**xyzw** quaternion. Playback and DEME initialization consume that quaternion
normally; do not invert it again. Translation and SMPL-X hand reconstruction
use their respective source conventions.

Object NPZ files use `format_version = 2`; hand files use version 1.
Simulation summaries record `object_motion_format_version`. Loaders and the
comparison viewer enforce this contract. If a file is rejected, regenerate its
object NPZ with `--object-only` using the same source sequence and frame interval,
then rerun simulation and comparison. Do not edit version fields to bypass checks.

Regression checks (require the GRAB preparation and demo dependencies):

```bash
PYTHONPATH=python python -m unittest discover -s tests -p 'test_grab*.py'
```

The conversion test compares asymmetric mesh vertex positions with the official
`V @ batch_rodrigues(global_orient) + translation` rule, and tests reject stale
object exports and simulation results.

## 1. Prerequisites

The shared coupling requirements are in [Newton + DEME — Setup](../couplers/newton-deme.md#setup).
The commands below provide the equivalent base build plus GRAB-specific dependencies;
follow them and the asset preparation steps in this guide.

Run the commands from the repository root with Python dependencies installed
for the same interpreter used to build MoPhi.
The examples use `PYTHONPATH=python` for source-tree development. With an
installed MoPhi wheel, omit that setting and remove any previously exported
source-checkout path. See [Python build workflows](python-builds.md).

- Preparation: NumPy, SciPy, PyTorch, SMPL-X, trimesh, licensed GRAB data and
  authorized SMPL-X models. Standalone MANO models are not needed: we extract
  the hands embedded in the personalized SMPL-X reconstruction.
- Simulation: DEME (`deme`), Newton/Warp, trimesh, fast-simplification, rtree,
  and an NVIDIA CUDA GPU.
- Rendering: working OpenGL drivers, pyglet, imgui-bundle, Pillow, imageio,
  imageio-ffmpeg, and Matplotlib (imported by the viewer and used for patch/force
  plots). Headless rendering still requires graphics support.

```bash
python scripts/setup/install_dependencies.py --extra grab

git submodule update --init --recursive
cmake -S . -B build-grab \
    -DPython3_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
    -DMOPHI_BUILD_PYTHON_BINDINGS=ON -DMOPHI_FETCH_NEWTON=ON \
    -DMOPHI_BUILD_FERIS_NEWTON=OFF -DMOPHI_BUILD_NEWTON_XLB_DEM=OFF \
    -DMOPHI_BUILD_NEWTON_DEM=ON -DMOPHI_BUILD_DEMOS=ON
cmake --build build-grab -j
PYTHONPATH=python python -c 'import mophi, deme, newton; print(mophi.mophi_core.__file__)'
```

No compiled MoPhi solver coupler is needed for this demo. The base `mophi_core`
extension is required. `MOPHI_BUILD_NEWTON_DEM=ON` enables CMake run targets and
demo installation; direct source execution does not depend on that flag.
The [dependency installer](python-dependencies.md) includes a convenience
superset of solvers; XLB is not used by GRAB.
`MOPHI_FETCH_NEWTON=ON` installs the repository-configured Newton/Warp dependencies
into the selected Python interpreter. If installing them manually, use the
compatible package set in [dependency guide](python-dependencies.md); the GRAB workflow
does not require XLB or the three-way compiled coupler.

## 2. Obtain and extract licensed assets

Download from [GRAB](https://grab.is.tue.mpg.de/download.php) under its license.
For this example obtain the **subject s2 GRAB parameters**, object contact
meshes/settings, subject mesh/settings, and SMPL-X hand correspondence assets.
Raw VICON recordings are unnecessary. Obtain SMPL-X models separately through
[SMPL-X](https://smpl-x.is.tue.mpg.de/).

Use the authors' [extraction script](https://github.com/otaheri/GRAB/blob/master/grab/unzip_grab.py)
for the GRAB dataset ZIPs; do not manually flatten their directory hierarchy.
For a fresh setup, put only the dataset ZIPs in a dedicated archive directory:

```bash
# Adjust this one path to your own asset directory.
GRAB_DATA="$HOME/GRAB"
git clone https://github.com/otaheri/GRAB.git /path/to/grab-tools
python /path/to/grab-tools/grab/unzip_grab.py \
    --grab-path "$GRAB_DATA/archives" --extract-path "$GRAB_DATA/extracted"
```

`/path/to/grab-tools` is a placeholder for your chosen tool checkout. Keep model
archives and already extracted files outside the directory passed as
`--grab-path`, since the authors' script recursively treats its files as ZIPs.
Place the authorized gendered SMPL-X model files in `models/smplx/`:

```text
<raw-data-dir>/                   # default: ~/GRAB
  extracted/
    grab/s2/mug_drink_2.npz
    tools/                       # original object/subject paths
      smplx_correspondence/
        rhand_smplx_ids.npy
        rhand_faces.npy           # lhand_* equivalents if exporting the left hand
  models/smplx/
    SMPLX_MALE.npz
    SMPLX_FEMALE.npz
```

Already extracted assets in this layout can be used directly. Converted files
and generated movies remain local and subject to their source licenses;
`data/grab/` and `output/` are Git-ignored. Honor GRAB's consent/withdrawal terms.

## 3. Generate MoPhi NPZ files

```bash
# Defaults: ~/GRAB, s2/mug_drink_2, right hand plus cup.
PYTHONPATH=python python scripts/grab/convert_grab_hand_motion.py

# Equivalent command for a different raw-data location:
PYTHONPATH=python python scripts/grab/convert_grab_hand_motion.py \
    s2/mug_drink_2.npz --raw-data-dir /path/to/GRAB
```

Expected outputs:

- `data/grab/s2/mug_drink_2_right.npz`
- `data/grab/s2/mug_drink_2_object.npz`

`--hand both` also exports the left hand. `--object-only` reuses previously
prepared hands. `--grab-root` overrides the extracted-data directory;
`--model-root` overrides the model directory; `--output-root` changes the output
location. These override the defaults derived from `--raw-data-dir`.
The positional sequence accepts a file path or a subject-relative path beneath
`<grab-root>/grab/`. `--start` and `--stop` select an inclusive/exclusive partial
range and add a frame-range suffix to filenames. `--batch-size` controls memory.

The hand NPZ contains source timestamps/frame IDs, provenance, 778 vertices and
1,538 triangles, geometric-origin translations, and local vertex positions.
World vertices equal `hand_vertices_local + hand_origin_world[:, None, :]`.
The origin is not a physical CoM. The object NPZ contains the original mesh,
translations and **xyzw** quaternions. Both use metres and seconds. No physical
forces, hand mass, or tissue deformation are inferred by conversion.

## 4. Inspect the recorded grasp before simulation

The unified viewer validates prepared NPZ files and can render the recording
without any simulation results:

```bash
PYTHONPATH=python python demo/newton_dem/grab/grabbing_cup/render_grab_comparison.py \
    --mode recorded --output-dir output/render_grab_comparison/recorded
```

This writes `recorded_reference.mp4` and PNG snapshots. For a live Newton window:

```bash
PYTHONPATH=python python demo/newton_dem/grab/grabbing_cup/render_grab_comparison.py \
    --mode recorded --interactive
```

Recorded-only options include `--hand`, `--object`, `--start-frame`,
`--stop-frame`, and `--playback-speed`. Defaults show the **grasp interval**,
source frames 240–380 at half speed, with an initial settling hold matching the
simulation. `--preview-frame 120` renders only one movie-frame PNG.

## 5. Run the complete physics-driven grasp

```bash
PYTHONPATH=python python demo/newton_dem/grab/grabbing_cup/demo_grab_contact.py --headless --save-movie
```

This runs the full default interval and writes to
`output/demo_grab_contact/grab/`.
The command renders without a window and saves `contact.mp4` in that folder.
Use `--no-render` instead of `--headless --save-movie` for physics only; the
comparison viewer can render the saved trajectory afterward. Omit `--headless`
to show the live Newton window.

The hand is frozen for 0.25 s, then advances through the recorded approach,
closure and early lift. Physics uses an explicit 0.1 ms timestep; rendering is
60 FPS. The 120 Hz source motion is interpolated at every physics step. Initial
kernel compilation and per-step mesh updates can make the run take minutes.

DEME alone integrates the cup. Its recorded pose is used only at initialization
and as a diagnostic reference. Hand position/deformation is prescribed; the hand
does not respond to contact forces. Owner translational velocity is supplied,
but finger articulation/wrist rotation velocity is not supplied per node.
The collision proxy has 6,000 triangles; the original cup geometry is retained
for visualization and homogeneous-volume mass/inertia preparation. Both cup and
hand use ten connected regions each from MoPhi's
`partition_mesh_seeded_patches`: seeds spread by face-adjacency distance, then
faces grow into their nearest seed region. Hand patch IDs are computed once on
the initial hand shape and retained during deformation. This does not guarantee
that separate finger contact areas receive separate patches. Changing patch IDs
does not change mesh geometry or mass properties. Patch
grouping changes the contact response and is not a proof of converged physics.

Scene parameters are in [`scene_config.py`](../../demo/newton_dem/grab/grabbing_cup/scene_config.py).
`OBJECT_PATCH_MODE = "spread"` and `OBJECT_PATCH_COUNT = 10` select the default cup
partition. `HAND_PATCH_MODE = "spread"` and `HAND_PATCH_COUNT = 10` configure
the hand in the same way; edit either count there to experiment. Radius and
normal-angle options apply only to the existing `connected` mode, not the count-based utility.
CLI overrides include `--hand`, `--object`, `--start-frame`, `--stop-frame`,
`--playback-speed`, `--dt`, `--mu` (alias `--friction`), `--floor-friction`, `--collision-faces`,
`--cup-patch-radius`, `--hand-patch-radius`, and `--output-dir`. Grasp duration
comes from the selected recording interval; `--duration` applies only to the
optional cup-rest diagnostic. Custom output directories gain a `grab/` child.
A friction value different from the scene default automatically uses
`<scene-output>/mu_<value>/grab/` unless `--output-dir` is supplied, preserving
the default result. This also applies to the legacy `--friction` spelling.

### Simulation outputs and acceptance

| File | Contents |
|---|---|
| `contact.mp4` | Simulation movie saved by `--save-movie` |
| `summary.json` | Parameters, errors, individual checks and aggregate verdict |
| `trajectory.npz` | Sampled cup pose, velocity, contact wrench and inertia |
| `grasp_diagnostics.npz` | Recorded reference, hand command errors, sampled penetration |
| `render_poses.npz` | Exact movie-cadence cup poses, source times, visual geometry and scene offset |
| `patches.npz` | Initial collision geometry and triangle patch IDs |
| `*.obj` | Generated DEME mesh input |

`completed: true` means the simulation finished. **It does not mean the grasp
succeeded.** Inspect the individual lift and reference-position checks in
`summary.json`, along with the comparison movie.
`--check` converts failed acceptance into a nonzero exit after saving results;
leave it off when reproducing the comparison. Lift checks require sustained
clearance and contact support, not merely upward motion.

## 6. Produce the comparison movie

After step 5, the default viewer command uses that same run directory:

```bash
PYTHONPATH=python python demo/newton_dem/grab/grabbing_cup/render_grab_comparison.py
```

The default output is `output/render_grab_comparison/`:

- **`recorded_vs_deme.mp4`**: recorded reference on the left, dynamic cup on the right.
- `recorded_reference.mp4` and `deme_replay.mp4`: individual panels.
- PNG snapshots, `frame_timing.npz`, and `render_manifest.json`.

Both panels share the camera, source timing, hand motion, and world offset.
Replay uses saved simulation poses; it does not rerun physics. If a compatible
run lacks `render_poses.npz`, the viewer interpolates its diagnostic poses and
identifies that approximation in `render_manifest.json`.

```bash
# Palm-side view of the same run.
PYTHONPATH=python python demo/newton_dem/grab/grabbing_cup/render_grab_comparison.py \
    --view palm --output-dir output/render_grab_comparison/palm

# Optional patch image alongside the comparison.
PYTHONPATH=python python demo/newton_dem/grab/grabbing_cup/render_grab_comparison.py --patches

# Interactive replay of the simulated cup.
PYTHONPATH=python python demo/newton_dem/grab/grabbing_cup/render_grab_comparison.py \
    --mode simulation --interactive
```

`--view handle` is the default oblique handle/back-of-hand view;
`--view original` selects the original camera preset. `--camera-position X Y Z` and
`--camera-target X Y Z` override either. Pass a custom run directory as the
positional argument. Interactive mode supports single-panel `recorded` or
`simulation` viewing; comparison movies are rendered headlessly. `--patches`
saves `patches.png` in local metric coordinates and requires the run's partition
archive (`patches.npz`). Recorded-only viewing with `--patches` needs that
archive but does not need a completed trajectory; simulation/comparison viewing
also needs the saved simulation results. It also renders the selected movie; it is
not a standalone preparation or patch-only command. The patch image uses the
fixed local-coordinate `PATCH_VIEW` in the viewer; `--view` selects the movie
camera only.

## Sphere lift demo

The sphere scene uses `s2/spherelarge_lift.npz`. Complete the shared
[prerequisites](#1-prerequisites), including [Newton + DEME setup](../couplers/newton-deme.md#setup),
before running it. Frames 690–830 select a right-hand lift with approach and
initial lowering; the export extends to frame 850 for inspection. Contact labels
show broad right-hand coverage during the lift, but do not establish whether a
physics-driven grasp will succeed.

```bash
# Extract the right hand and the supplied object mesh/trajectory.
PYTHONPATH=python python scripts/grab/convert_grab_hand_motion.py \
    s2/spherelarge_lift.npz --hand right --start 690 --stop 850

# Inspect the recorded reference first.
PYTHONPATH=python python demo/newton_dem/grab/grabbing_sphere/render_sphere_comparison.py \
    --mode recorded --output-dir output/render_sphere_comparison/recorded

# Run DEME contact/dynamics and save a Newton-rendered movie.
PYTHONPATH=python python demo/newton_dem/grab/grabbing_sphere/demo_grab_sphere.py \
    --headless --save-movie

# Generate the comparison and contact-patch image.
PYTHONPATH=python python demo/newton_dem/grab/grabbing_sphere/render_sphere_comparison.py \
    --patches
```

The default inputs are `data/grab/s2/spherelarge_lift_frames_000690_000850_right.npz`
and its matching `_object.npz`. Simulation outputs are under
`output/demo_grab_sphere/grab/`. The comparison movie is
`output/render_sphere_comparison/recorded_vs_deme.mp4`; the same folder contains
`patches.png`, `recorded_reference.mp4`, and `deme_replay.mp4`. The simulation
command also saves `contact.mp4` in its result folder. Use `--no-render` instead
of `--headless --save-movie` to run physics without rendering; the comparison
viewer renders its saved trajectory afterward.
`demo_grab_sphere` is also a CMake run
target when `MOPHI_BUILD_NEWTON_DEM=ON` and `MOPHI_BUILD_DEMOS=ON`.

The [sphere configuration](../../demo/newton_dem/grab/grabbing_sphere/scene_config.py)
uses the supplied closed GRAB mesh, approximately 11.6 × 11.6 × 10.5 cm. It is
slightly flattened, not an ideal sphere. Contact uses a simplified triangle
mesh with a 900-triangle target and ten spread patches; visualization and
uniform-density mass properties use the original mesh. This demo does not
substitute an analytical sphere. The hand has ten patches and prescribed
motion. The assumed object mass, contact material, timestep, playback speed,
and acceptance checks match the cylinder and cube defaults. The sphere uses
0.1 m collision-search bins to reduce detection cost;
this search setting does not change the contact mesh or DEME timestep.

The existing friction experiment option also works here. For example:

```bash
PYTHONPATH=python python demo/newton_dem/grab/grabbing_sphere/demo_grab_sphere.py \
    --mu 0.9 --headless --save-movie
PYTHONPATH=python python demo/newton_dem/grab/grabbing_sphere/render_sphere_comparison.py \
    output/demo_grab_sphere/mu_0.9/grab \
    --output-dir output/render_sphere_comparison/mu_0.9 --patches
```

This preserves the default result and changes only hand–object friction.
Higher friction is an experiment, not a guarantee of a stable grasp. Inspect
`summary.json` and the comparison movie together.

## Cube lift demo

The medium cube scene uses `s2/cubemedium_lift.npz`, with the same
[prerequisites and build setup](#1-prerequisites). It selects a right-hand-only
attempt later in the recording: frames 780–890 cover approach and pickup;
the exported interval includes additional frames for inspection. Raw contact
labels show no left-hand contact over this interval. Labels alone do not
establish whether the physics-driven pickup will succeed.

```bash
# Prepare the hand and original object geometry from licensed GRAB assets.
PYTHONPATH=python python scripts/grab/convert_grab_hand_motion.py \
    s2/cubemedium_lift.npz --hand right --start 780 --stop 910

# Inspect the recorded motion before simulation.
PYTHONPATH=python python demo/newton_dem/grab/grabbing_cube/render_cube_comparison.py \
    --mode recorded --output-dir output/render_cube_comparison/recorded

# Run the physics-driven cube with a Newton-rendered movie.
PYTHONPATH=python python demo/newton_dem/grab/grabbing_cube/demo_grab_cube.py \
    --headless --save-movie

# Render the recorded-versus-simulated comparison and patch inspection image.
PYTHONPATH=python python demo/newton_dem/grab/grabbing_cube/render_cube_comparison.py \
    --patches
```

The converter writes `data/grab/s2/cubemedium_lift_frames_000780_000910_right.npz`
and the matching `_object.npz`. All paths default to these exports, so no
scene-selection arguments are needed. Simulation results are under
`output/demo_grab_cube/grab/`, including `contact.mp4` and `summary.json`.
The comparison is `output/render_cube_comparison/recorded_vs_deme.mp4`;
`patches.png` in the same directory shows the contact partitions. As with the
other scenes, `--no-render` simulates without live graphics and the comparison
viewer can render the saved trajectory afterward.

The [cube configuration](../../demo/newton_dem/grab/grabbing_cube/scene_config.py)
uses the original approximately 8 cm cube for mass properties and visualization,
a 600-triangle target for the collision proxy, and ten spread patches each for
the cube and hand. The assumed mass is 0.25 kg, not a measurement from GRAB.
The timestep, compliant material, friction, half-speed hand motion, settling
hold, and acceptance checks match the cylinder. The original full-resolution
mesh is retained for rendering and mass properties; proxy fidelity is checked
before simulation. See `summary.json` and the movie together when assessing
whether a run achieved supported pickup.

### Higher-friction cube variant

Use `--mu` to try a higher hand–object friction coefficient. The default remains
`0.5`; `0.9` below is a **tentative experiment**, not a validated successful-grasp
preset. No other material parameter, hand trajectory, mass, or acceptance check
changes. Table friction remains separately controlled by `--floor-friction`.
The shared option is also available in the cup, cylinder, and sphere demos.

```bash
# Optional higher-friction trial; leaves the default cube result intact.
PYTHONPATH=python python demo/newton_dem/grab/grabbing_cube/demo_grab_cube.py \
    --mu 0.9 --headless --save-movie

# Compare this trial against the same recorded reference.
PYTHONPATH=python python demo/newton_dem/grab/grabbing_cube/render_cube_comparison.py \
    output/demo_grab_cube/mu_0.9/grab \
    --output-dir output/render_cube_comparison/mu_0.9 --patches
```

The trial writes to `output/demo_grab_cube/mu_0.9/grab/`, and its comparison is
`output/render_cube_comparison/mu_0.9/recorded_vs_deme.mp4`. The effective
coefficient is saved as `friction` in `summary.json`. Inspect that file's checks
and the movie to determine whether pickup is sustained. Add `--check` to the
simulation command if a failed acceptance check should produce a nonzero exit.
Try another `--mu` value to tune the experiment; each nondefault value gets its
own directory. Repeating the same value reuses its directory, so use
`--output-dir` when preserving multiple runs of that value.

## Cylinder lift demo

This scene uses the first right-handed lift in `s2/cylindermedium_lift.npz`.
Obtain the s2 parameter archive and corresponding object/subject/model assets
as described above. Convert a short interval to keep preparation inexpensive:

```bash
PYTHONPATH=python python scripts/grab/convert_grab_hand_motion.py \
    s2/cylindermedium_lift.npz --hand right --start 360 --stop 510
```

This produces `data/grab/s2/cylindermedium_lift_frames_000360_000510_right.npz`
and the matching `_object.npz`. The cylinder configuration refers to those exact
filenames. The simulation uses frames 360–490, including approach, lift, and
the start of lowering, at half playback speed after a settling hold. The raw contact labels identify
the right hand as active during this interval; those proximity labels do not
establish that a physics-driven grasp will succeed.

```bash
# Inspect the recorded reference first.
PYTHONPATH=python python demo/newton_dem/grab/grabbing_cylinder/render_cylinder_comparison.py \
    --mode recorded --output-dir output/render_cylinder_comparison/recorded

# Run the dynamic cylinder and save a Newton-rendered movie.
PYTHONPATH=python python demo/newton_dem/grab/grabbing_cylinder/demo_grab_cylinder.py \
    --headless --save-movie

# Compare recorded and simulated motion and inspect the patches.
PYTHONPATH=python python demo/newton_dem/grab/grabbing_cylinder/render_cylinder_comparison.py \
    --patches
```

Simulation results go to `output/demo_grab_cylinder/grab/`; the comparison goes
to `output/render_cylinder_comparison/recorded_vs_deme.mp4`. `--no-render` can
replace `--headless --save-movie` for a physics-only run; the viewer can render
its saved results afterward. The cylinder uses ten spread patches for both
object and hand. Its assumed mass is 0.25 kg, with homogeneous-volume inertia
computed from the original cylinder mesh; mass is not measured by GRAB.
The collision proxy uses 600 triangles; the original mesh is used for rendering
and mass properties. The geometry gates require sampled surface error below
1.5 mm and relative volume error below 3%. These checks are not a
contact-convergence study.
Materials, hand prescription, timestep, and acceptance criteria match the cup
scene. Edit the cylinder's
[`scene_config.py`](../../demo/newton_dem/grab/grabbing_cylinder/scene_config.py)
to change its experiment without affecting the cup.

For compatibility with existing diagnostics, saved field names such as
`cup_com_world` and `cup_lifted`, plus the optional `cup_rest` case,
refer to the manipulated object in each scene. `--view handle` remains the
shared oblique camera preset name even though the cylinder has no handle.

## Current validation results

These measurements are from the completed default runs with hand–object friction
`mu = 0.5`, ten patches per mesh, and the configured prescribed hand motion.
Values come from each run's `summary.json` under `output/demo_grab_contact/grab/`,
`output/demo_grab_cylinder/grab/`, and `output/demo_grab_sphere/grab/`.

| Scene | Sustained contact-supported pickup | Final bottom height | Final CoM error vs recording | Maximum sampled hand penetration | All checks pass |
|---|---|---:|---:|---:|---|
| Cup | No | −0.20 mm | 143.88 mm | 0.92 mm | No |
| Cylinder | Yes | 43.12 mm | 4.35 mm | 6.20 mm | Yes |
| Sphere | No | −0.06 mm | 115.82 mm | 6.87 mm | No |

All three runs pass completion, time accounting, contact, hand-transfer, and
penetration checks. The cup and sphere fail sustained lift, contact-supported
lift, and final reference-position checks. Negative bottom height denotes slight
penetration below the table. The acceptance tolerance is 20 mm for the
intentionally soft material; passing it does not establish contact convergence.
Cube and higher-friction outcomes are not included in this validation table.

## Interpreting contact and recorded motion

The reference movie prescribes object motion; the physics movie lets DEME move
the object. A reference lift alone does not establish that the reconstructed
hand surface can support the object through contact.

The GRAB authors explicitly report contact undershooting (hovering fingers,
including a thumb example) and overshooting (penetration). They attribute these
to acquisition/fitting error, limited mesh resolution, and soft tissue that
SMPL-X does not model. Their contact annotation uses a **4.5 mm proximity
tolerance**; that is not a measured tissue thickness or a calibrated DEME margin.
See [GRAB §3.2–3.3](https://arxiv.org/html/2008.11200#S3.SS3).

Our converter selects the personalized SMPL-X hand vertices without separate
MANO refitting or hand simplification. Reference rendering uses the original
object mesh. Patches do not move vertices. Apparent gaps should be measured
against source contact labels before attributing a failed lift to a particular
finger or to acquisition error.
Interpretation should also account for omitted per-node articulation velocity,
patch-dependent contact response, and assumed mass/material parameters. A
per-finger gap/contact-label audit and controlled contact-law or trajectory
experiments can help distinguish these effects.

## Optional regression checks

These are diagnostic tools, not steps required for the grasp movie:

```bash
PYTHONPATH=python python demo/newton_dem/grab/grabbing_cup/demo_grab_contact.py \
    --case cup_rest --duration 1 --headless --save-movie --check
PYTHONPATH=python python tests/grab/inspect_contact.py \
    output/demo_grab_contact/cup_rest
PYTHONPATH=python python -m unittest discover -s tests -p 'test_grab*.py' -v
```

Cup-rest records every-step `contacts.npz`; the analysis reconstructs force,
normal/tangential torque and angular impulse. Unit tests cover schemas,
interpolation, mass frames, patch invariants, wrench reconstruction and lift
acceptance. General patch partitioning remains documented under
[mesh contact patches](../mesh-contact-patches.md).

## Geometry and diagnostic interpretation

| Component | DEME contact | Rendering / mass properties |
|-----------|--------------|-----------------------------|
| Cup, cylinder, cube, or sphere | Simplified triangle mesh with fixed patch IDs | Original NPZ object mesh; uniform-density volume properties scaled to `OBJECT_MASS` |
| Recorded hand | Deforming triangle mesh with fixed patch IDs | The same extracted hand surface; prescribed motion |
| Table | Infinite analytical plane | A finite box for display only |

The scene configurations use object-neutral `OBJECT_*` names. Saved `cup_*`
fields and existing `--cup-*` options retain their existing names for
compatibility with prior recordings and inspection tools.

Hand penetration is sampled at hand vertices against the collision mesh at the
configured diagnostic interval. The reported maximum is over those samples,
not a continuous surface or every-substep bound. Thin intersections between
vertices or between diagnostic times can be missed. Inspect the comparison
movie alongside the diagnostics when assessing a grasp.
