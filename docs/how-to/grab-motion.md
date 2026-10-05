# GRAB hand motion conversion

[`scripts/convert_grab_hand_motion.py`](../../scripts/convert_grab_hand_motion.py)
reconstructs the hands embedded in a GRAB SMPL-X body sequence and saves their
fixed-topology surface motion for later DEME contact experiments. It does not
run DEME or reproduce GRAB's separate MANO hand reconstructions.

## Prerequisites and use

Download GRAB under its own license. Use the GRAB authors' `unzip_grab.py` to
place sequence files beneath `~/GRAB/extracted/grab/` and subject templates and
hand correspondence arrays beneath `~/GRAB/extracted/tools/`. Extract the
gendered `SMPLX_MALE.npz` and `SMPLX_FEMALE.npz` files from the authorized
SMPL-X archive into `~/GRAB/models/smplx/`. Install `numpy`, `torch`, `smplx`,
`trimesh`, and `scipy` in the Python environment used for conversion.

```bash
python3 -m pip install numpy torch smplx trimesh scipy
```

From the repository root, convert one sequence:

```bash
python3 scripts/convert_grab_hand_motion.py \
    ~/GRAB/extracted/grab/s2/mug_drink_2.npz
```

`--hand left` or `--hand right` selects one hand. `--batch-size` controls
reconstruction memory use. `--start` and `--stop` select an inclusive-start,
exclusive-stop range of source frames for inspection. `--grab-root`,
`--model-root`, and `--output-root` override the default paths.
Partial frame ranges add `_frames_<start>_<stop>` to the output filename so
they do not overwrite a full sequence.

The default outputs are `data/grab/s2/mug_drink_2_left.npz` and
`data/grab/s2/mug_drink_2_right.npz`, plus `mug_drink_2_object.npz` with the
original object mesh and recorded rigid poses. Use `--object-only` to add the
object archive when hands are already extracted. See
[Newton playback](grab-playback.md) for complete prerequisites and playback.
The whole `data/grab/` tree is ignored
by Git because its contents derive from licensed GRAB data. Do not commit or
redistribute the converted sequences.

## Stored arrays and geometry convention

Each compressed NPZ contains:

| Array | Shape and dtype | Meaning |
|------|-----------------|---------|
| `timestamps_s` | `(T,)`, float64 | Source frame time in seconds. |
| `source_frame_indices` | `(T,)`, int32 | Original frame numbers. |
| `source_fps` | scalar float64 | Original recording rate. |
| `hand_origin_world` | `(T, 3)`, float32 | Arithmetic mean of the hand's vertices in each frame. |
| `hand_vertices_local` | `(T, N, 3)`, float32 | Vertex positions relative to that frame's origin. |
| `hand_faces` | `(M, 3)`, int32 | Fixed triangle indices into the local vertices. |
| `format_version`, `source_sequence`, `hand_side`, `origin_kind` | scalars | Schema and provenance. |

World positions reconstruct as
`hand_vertices_local[t] + hand_origin_world[t, None, :]`. There is no stored
owner rotation: use identity orientation and update local triangle positions
as the hand moves. The origin is a geometric reference, **not** a physical
center of mass. In DEME, indexed vertices must be expanded into its flattened
triangle-node order before using its triangle-node update API.

The source topology is the 778-vertex, 1,538-triangle SMPL-X hand subset.
This converter stores both hands' full time series; it does not simplify the
mesh, infer mass or inertia, set friction parameters, or prescribe the object.
Those are choices for a later contact demo. This format uses float32 geometry,
so world-space reconstruction is accurate to float32 precision. Because the
source is sampled at 120 Hz, a DEME step between recording frames needs an
explicit interpolation policy.

## Check an output

[`scripts/inspect_grab_hand_motion.py`](../../scripts/inspect_grab_hand_motion.py)
validates the NPZ schema, time axis, topology, and local-origin convention,
then renders selected frames in metric world coordinates. It uses MoPhi's
shared hand loader; ensure MoPhi is built for the active interpreter (see
[playback prerequisites](grab-playback.md#prerequisites)). It does not need
SMPL-X, PyTorch, or MANO. From the repository root:

```bash
python3 -m pip install matplotlib scipy
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
python3 scripts/inspect_grab_hand_motion.py \
    data/grab/s2/mug_drink_2_left.npz \
    data/grab/s2/mug_drink_2_right.npz \
    --object --frames 400 602 800
```

`--object` overlays the recorded object, which requires the original GRAB
sequence and object mesh under `~/GRAB/extracted/`. Omit it to inspect the
converted hands alone. `--grab-root` overrides that dataset path, `--output`
chooses the PNG path, and `--frames` accepts original source frame numbers.
Without `--frames`, the script shows the first, middle, and last shared frames.
The default PNG location is `output/inspect_grab_hand_motion/`. A matching
source-object overlay checks coordinate alignment visually; it does not prove
that collision or friction physics is correct.
Supply just one hand file for a closer hand-object view.

For an MP4 of consecutive source frames, also install `imageio` and
`imageio-ffmpeg`, then select a range:

```bash
python3 -m pip install imageio imageio-ffmpeg
python3 scripts/inspect_grab_hand_motion.py \
    data/grab/s2/mug_drink_2_right.npz \
    --object --movie --movie-start 400 --movie-stop 640 --movie-stride 4
```

This example uses source frames 400–639 at every fourth frame. The default
movie FPS is `source_fps / movie_stride`, so recorded time is preserved. Set
`--movie-fps` to change playback speed. Omit the range to render the entire
converted sequence, or use `--output path/to/movie.mp4` for another location.
The movie uses a following camera with a constant metric scale so the hand
stays visible; `--movie-camera fixed` instead preserves one world-space view
of the full trajectory. Rendering a long sequence or a dense object mesh can take
time; `--movie-stride` reduces the number of rendered frames. These MP4s,
like the converted data, remain local and should not be redistributed under
the source-data license.

For a quick numeric check without plotting:

```bash
python3 - <<'PY'
import numpy as np

with np.load('data/grab/s2/mug_drink_2_right.npz') as motion:
    vertices_world = motion['hand_vertices_local'] + motion['hand_origin_world'][:, None, :]
    assert vertices_world.shape[1:] == (778, 3)
    assert motion['hand_faces'].shape == (1538, 3)
    assert np.isfinite(vertices_world).all()
    print(vertices_world.shape, motion['timestamps_s'][0], motion['timestamps_s'][-1])
PY
```

Run the inspection utility's fixture checks with:

```bash
python3 -m unittest discover -s tests -p test_grab_motion_inspection.py
```
