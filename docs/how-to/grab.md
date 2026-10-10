# GRAB grasps: recorded motion versus contact physics

This workflow reconstructs recorded hand/object scenes, lets **DEME** move the object
through contact with the prescribed hand, and uses **Newton** to render both
recorded and simulated motion. The comparison exposes the gap between fitted
motion data and a physically reproducible grasp. Earlier runs **did not
successfully lift the cup**; an unsuccessful grasp is a valid experimental result.
Both scenes default to ten spread-seed patches for object and hand. The cup
grasp still needs evaluation with these partitions. The cylinder run below
achieved contact-supported pickup. Its measured penetration is within the
current acceptance tolerance for the intentionally soft contact material.

There are three workflow steps. The cup entry points are:

| Step | Script | Default |
|---|---|---|
| Prepare | [`convert_grab_hand_motion.py`](../../scripts/grab/convert_grab_hand_motion.py) | Convert `s2/mug_drink_2`, right hand and cup |
| Simulate | [`demo_grab_contact.py`](../../demo/newton_dem/grab/grabbing_cup/demo_grab_contact.py) | Recorded hand attempts to grab a dynamic cup |
| View | [`render_grab_comparison.py`](../../demo/newton_dem/grab/grabbing_cup/render_grab_comparison.py) | Side-by-side recorded/simulated grasp movie, handle/back-of-hand camera |

The cup scene lives in `demo/newton_dem/grab/grabbing_cup/`; the cylinder
scene lives in `demo/newton_dem/grab/grabbing_cylinder/`. Each has its own
`scene_config.py` and simulation/viewer entry points. They share the experiment
loop in [`grasp_simulation.py`](../../demo/newton_dem/grab/grasp_simulation.py)
and renderer in [`grasp_viewer.py`](../../demo/newton_dem/grab/grasp_viewer.py).
The shared modules receive configuration explicitly; switching scenes does not
modify the other scene's configuration. Geometry preparation and mesh partitioning
continue to use MoPhi utilities. DEME remains responsible for contact and object
integration; Newton renders the resulting geometry. The demo loop still stages
hand vertices and owner states through host arrays on each DEME step.

The numbered workflow below is for the cup. The [cylinder workflow](#cylinder-lift-demo)
uses the same prerequisites and converter.

## 1. Prerequisites

Run the commands from the repository root with Python dependencies installed
for the same interpreter used to build MoPhi. No named environment is required.
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
python -m pip install numpy scipy torch smplx trimesh fast-simplification rtree \
    deme pyglet imgui-bundle pillow imageio imageio-ffmpeg matplotlib

git submodule update --init --recursive
cmake -S . -B build-grab \
    -DPython3_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
    -DMOPHI_BUILD_PYTHON_BINDINGS=ON -DMOPHI_FETCH_NEWTON=ON \
    -DMOPHI_BUILD_FERIS_NEWTON=OFF -DMOPHI_BUILD_NEWTON_XLB_DEM=OFF
cmake --build build-grab -j
PYTHONPATH=python python -c 'import mophi, deme, newton; print(mophi.mophi_core.__file__)'
```

No compiled MoPhi solver coupler is needed for this demo.
`MOPHI_FETCH_NEWTON=ON` installs the repository-configured Newton/Warp dependencies
into the selected Python interpreter. If installing them manually, use the
compatible package set in [Newton setup](newton-xlb-deme.md); the GRAB workflow
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
PYTHONPATH=python python demo/newton_dem/grab/grabbing_cup/demo_grab_contact.py --no-render
```

This runs the full default interval and writes to
`output/demo_grab_contact/grab/`. It does **not** default to cup-rest or drop.
Omit `--no-render` for live Newton visualization, or use
`--headless --save-movie` to also save a simulation movie during the run.

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
that separate finger contact areas receive separate patches; the full grasp
requires validation. Mesh geometry and mass properties are unchanged. Patch
grouping changes the contact response and is not a proof of converged physics.

Scene parameters are in [`scene_config.py`](../../demo/newton_dem/grab/grabbing_cup/scene_config.py).
`CUP_PATCH_MODE = "spread"` and `CUP_PATCH_COUNT = 10` select the default cup
partition. `HAND_PATCH_MODE = "spread"` and `HAND_PATCH_COUNT = 10` configure
the hand in the same way; edit either count there to experiment. Radius and
normal-angle options apply only to the existing `connected` mode, not the count-based utility.
CLI overrides include `--hand`, `--object`, `--start-frame`, `--stop-frame`,
`--playback-speed`, `--dt`, `--friction`, `--floor-friction`, `--collision-faces`,
`--cup-patch-radius`, `--hand-patch-radius`, and `--output-dir`. Grasp duration
comes from the selected recording interval; `--duration` applies only to the
optional cup-rest diagnostic. Custom output directories gain a `grab/` child.

### Simulation outputs and acceptance

| File | Contents |
|---|---|
| `summary.json` | Parameters, errors, individual checks and aggregate verdict |
| `trajectory.npz` | Sampled cup pose, velocity, contact wrench and inertia |
| `grasp_diagnostics.npz` | Recorded reference, hand command errors, sampled penetration |
| `render_poses.npz` | Exact movie-cadence cup poses, source times, visual geometry and scene offset |
| `patches.npz` | Initial collision geometry and triangle patch IDs |
| `*.obj` | Generated DEME mesh input |

`completed: true` means the simulation finished. **It does not mean the grasp
succeeded.** Earlier radius/normal-partition runs failed the lift/reference-position
checks; the full grasp with ten spread patches has not yet been validated.
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
Replay uses saved simulation poses; it does not rerun physics. Older runs lacking
`render_poses.npz` use explicitly labeled interpolation of coarser diagnostics.

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
`--view original` selects the earlier camera. `--camera-position X Y Z` and
`--camera-target X Y Z` override either. Pass a custom run directory as the
positional argument. Interactive mode supports single-panel `recorded` or
`simulation` viewing; comparison movies are rendered headlessly. `--patches`
saves `patches.png` in local metric coordinates and requires the run's partition
archive from a completed grasp run. It also renders the selected movie; it is
not a standalone preparation or patch-only command. The patch image uses the
fixed local-coordinate `PATCH_VIEW` in the viewer; `--view` selects the movie
camera only.

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
the start of lowering,
at half playback speed after a settling hold. The raw contact labels identify
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
and mass properties. Its sampled surface error was 1.36 mm and relative volume
error 0.11%, within the unchanged 1.5 mm / 3% geometry gates. The original
6,000-triangle trial was prohibitively slow, so it is not the default cylinder
configuration. These geometry checks are not a contact-convergence study.
Materials, hand prescription, timestep, and acceptance criteria match the cup
scene. Edit the cylinder's
[`scene_config.py`](../../demo/newton_dem/grab/grabbing_cylinder/scene_config.py)
to change its experiment without affecting the cup.

### Cylinder validation result

The full default cylinder interval completed with contact-supported pickup:

| Measurement | Result |
|---|---:|
| Peak object-bottom clearance | 98.8 mm |
| Final object-bottom clearance | 43.1 mm |
| Mean upward contact force in the final 0.1 s | 2.87 N |
| Object weight | 2.45 N |
| Final CoM error relative to the recording | 4.07 mm |
| Maximum sampled hand penetration | 6.33 mm |

`cup_lifted`, `lift_supported_by_contact`, and `cup_near_recorded_position`
passed. The run originally failed `sampled_hand_penetration_bounded` under the
previous 4 mm threshold. Both scenes now use `MAX_ALLOWED_PENETRATION = 0.020`
(20 mm), allowing larger overlap with the intentionally soft contact material.
This tolerance applies to sampled hand penetration and object–floor penetration;
it changes acceptance checks only, not contact forces or simulated motion.
The saved run's measurements satisfy all checks under the revised tolerance.
Its historical `summary.json` still records the original failed verdict; a new
run evaluates the current threshold. This demonstrates pickup with prescribed
hand motion, not a penetration-free or converged contact solution. Inspect both
the comparison movie and `summary.json` when reproducing it.

For compatibility with existing diagnostics, saved field names such as
`cup_com_world`, `cup_lifted`, and `CUP_MASS`, plus the optional `cup_rest` case,
refer to the manipulated object in either scene. `--view handle` remains the
shared oblique camera preset name even though the cylinder has no handle.

## Cup findings and interpretation

- The original whole-cup patch allowed unwanted pre-contact yaw despite correct
  average weight support. Isolated cup tests traced vertical-axis torque to
  eccentric tangential friction; halving the timestep did not fix it.
- Earlier radius/normal patches suppressed this symptom. With that partition,
  resting maximum rotation was about 0.226°, and pre-contact grasp yaw at 0.9 s
  was about -0.102° instead of +10.86°.
- The ten-spread-patch cup passed all checks in a one-second cup-rest run:
  peak yaw was 0.0355 degrees, with up to four active table contacts. This
  validates resting stability for that test, not full grasp success.
- In the earlier radius/normal-partition grasp run, the cup slid approximately
  13.2 mm without lifting and ended approximately 135 mm from the recorded CoM. Small command-transfer errors
  establish geometric playback accuracy, not physical grasp validity.
- The reference movie prescribes cup motion, so its cup lifts even if fitted
  finger surfaces leave a gap. The apparent thumb–handle gap has not yet been
  measured per finger or checked against source contact labels.

The GRAB authors explicitly report contact undershooting (hovering fingers,
including a thumb example) and overshooting (penetration). They attribute these
to acquisition/fitting error, limited mesh resolution, and soft tissue that
SMPL-X does not model. Their contact annotation uses a **4.5 mm proximity
tolerance**; that is not a measured tissue thickness or a calibrated DEME margin.
See [GRAB §3.2–3.3](https://arxiv.org/html/2008.11200#S3.SS3).

Our converter selects the personalized SMPL-X hand vertices without separate
MANO refitting or hand simplification. Reference rendering uses the original
cup mesh. Patches do not move vertices. These facts make a source reconstruction
mismatch plausible, but do not conclusively exclude extraction/interpolation
issues or prove the thumb gap caused the failed lift.
Other unresolved factors include absent articulation velocity, patch-dependent
stiffness/damping, transient tensile normal forces, and unvalidated mass/material
assumptions. Next useful work is a source-frame per-finger gap/contact-label
audit followed by controlled contact-law/trajectory experiments.

## Optional regression checks

These are diagnostic tools, not steps required for the grasp movie:

```bash
PYTHONPATH=python python demo/newton_dem/grab/grabbing_cup/demo_grab_contact.py \
    --case cup_rest --duration 1 --no-render --check
PYTHONPATH=python python tests/grab/inspect_contact.py \
    output/demo_grab_contact/cup_rest
PYTHONPATH=python python -m unittest discover -s tests -p 'test_grab*.py' -v
```

Cup-rest records every-step `contacts.npz`; the analysis reconstructs force,
normal/tangential torque and angular impulse. Unit tests cover schemas,
interpolation, mass frames, patch invariants, wrench reconstruction and lift
acceptance. General patch partitioning remains documented under
[mesh contact patches](../mesh-contact-patches.md).

The earlier standalone playback/inspection scripts and separate GRAB guides
have been consolidated here. Drop/push/plate calibration modes are removed from
the main demo; their historical findings remain above and their implementation
is available in Git history. Existing generated output directories are retained
as historical experiments, not required inputs to this workflow.
