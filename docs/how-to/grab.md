# GRAB cup grasp: recorded motion versus contact physics

This workflow reconstructs a recorded hand/cup scene, lets **DEME** move the cup
through contact with the prescribed hand, and uses **Newton** to render both
recorded and simulated motion. The comparison exposes the gap between fitted
motion data and a physically reproducible grasp. The current example **does not
successfully lift the cup**; an unsuccessful grasp is a valid experimental result.

There are three user-facing entry points:

| Step | Script | Default |
|---|---|---|
| Prepare | [`convert_grab_hand_motion.py`](../../scripts/grab/convert_grab_hand_motion.py) | Convert `s2/mug_drink_2`, right hand and cup |
| Simulate | [`demo_grab_contact.py`](../../demo/newton_dem/grab/demo_grab_contact.py) | Recorded hand attempts to grab a dynamic cup |
| View | [`render_grab_comparison.py`](../../demo/newton_dem/grab/render_grab_comparison.py) | Side-by-side recorded/simulated grasp movie, handle/back-of-hand camera |

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
PYTHONPATH=python python demo/newton_dem/grab/render_grab_comparison.py \
    --mode recorded --output-dir output/render_grab_comparison/recorded
```

This writes `recorded_reference.mp4` and PNG snapshots. For a live Newton window:

```bash
PYTHONPATH=python python demo/newton_dem/grab/render_grab_comparison.py \
    --mode recorded --interactive
```

Recorded-only options include `--hand`, `--object`, `--start-frame`,
`--stop-frame`, and `--playback-speed`. Defaults show the **grasp interval**,
source frames 240–380 at half speed, with an initial settling hold matching the
simulation. `--preview-frame 120` renders only one movie-frame PNG.

## 5. Run the complete physics-driven grasp

```bash
PYTHONPATH=python python demo/newton_dem/grab/demo_grab_contact.py --no-render
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
for visualization and homogeneous-volume mass/inertia preparation. Default
connected patches: 721 cup and 828 hand. Patch grouping changes the contact law's
response and is not a proof of converged physics.

Scene parameters are in [`scene_config.py`](../../demo/newton_dem/grab/scene_config.py).
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
succeeded.** The current example fails the lift/reference-position checks.
`--check` converts failed acceptance into a nonzero exit after saving results;
leave it off when reproducing the comparison. Lift checks require sustained
clearance and contact support, not merely upward motion.

## 6. Produce the comparison movie

After step 5, the default viewer command uses that same run directory:

```bash
PYTHONPATH=python python demo/newton_dem/grab/render_grab_comparison.py
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
PYTHONPATH=python python demo/newton_dem/grab/render_grab_comparison.py \
    --view palm --output-dir output/render_grab_comparison/palm

# Optional patch image alongside the comparison.
PYTHONPATH=python python demo/newton_dem/grab/render_grab_comparison.py --patches

# Interactive replay of the simulated cup.
PYTHONPATH=python python demo/newton_dem/grab/render_grab_comparison.py \
    --mode simulation --interactive
```

`--view handle` is the default oblique handle/back-of-hand view;
`--view original` selects the earlier camera. `--camera-position X Y Z` and
`--camera-target X Y Z` override either. Pass a custom run directory as the
positional argument. Interactive mode supports single-panel `recorded` or
`simulation` viewing; comparison movies are rendered headlessly. `--patches`
saves `patches.png` in local metric coordinates and requires the run's partition
archive; it is an inspection image, not a change to contact geometry.

## Findings and interpretation

- The original whole-cup patch allowed unwanted pre-contact yaw despite correct
  average weight support. Isolated cup tests traced vertical-axis torque to
  eccentric tangential friction; halving the timestep did not fix it.
- Connected patches suppressed this symptom. With the current partition,
  resting maximum rotation was about 0.226°, and pre-contact grasp yaw at 0.9 s
  was about -0.102° instead of +10.86°.
- The current cup still slides approximately 13.2 mm without lifting and ends
  approximately 135 mm from the recorded CoM. Small command-transfer errors
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
PYTHONPATH=python python demo/newton_dem/grab/demo_grab_contact.py \
    --case cup_rest --duration 0.3 --no-render --check
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
