# GRAB playback in Newton

[`demo_grab_playback.py`](../../demo/newton_dem/grab/demo_grab_playback.py)
animates prepared GRAB hand surfaces and the recorded object using Newton's
OpenGL renderer. Both hand deformation and object motion are prescribed.
This milestone computes no contact forces or dynamics; DEME is not required.
Continue with [cup contact checks](grab-contact.md) to let DEME advance the cup.

## Prerequisites

Commands below use the source-tree development workflow: run from the repository
root with `PYTHONPATH=python`. With an installed MoPhi wheel, omit that prefix
and remove any source-checkout path previously exported in `PYTHONPATH` so Python
uses the installed package. See [Python build workflows](python-builds.md).

- Use conda environment `ML`. Tested: Python 3.11, Newton 1.0.0, Warp 1.12.1.
  DEME 3.0.14 is installed in this environment but unused by playback.
- MoPhi with `mophi_core` built for the active Python version. No solver-specific
  C++ coupler is needed.
- NumPy/SciPy for loading and interpolation; Newton/Warp, pyglet, imgui-bundle
  for rendering; imageio/imageio-ffmpeg for PNG and MP4 output.
- Working OpenGL graphics drivers. Interactive mode needs a display (WSLg under
  WSL); `--headless` needs an offscreen OpenGL context and graphics support.
- Prepared hand and object NPZ files. Playback itself needs neither the original
  GRAB directory nor SMPL-X/PyTorch.

Run commands from the repository root:

```bash
conda activate ML
python -m pip install 'newton==1.0.0' 'warp-lang==1.12.1' \
    numpy scipy 'pyglet>=2.1.6,<3' 'imgui-bundle>=1.92.0' imageio imageio-ffmpeg
PYTHONPATH=python python -c 'import mophi; print(mophi.mophi_core.__file__)'
```

If MoPhi's extension is missing or incompatible, build it with CMake and a
C++17 compiler (see [Python build workflows](python-builds.md)):

```bash
git submodule update --init --recursive
cmake -S . -B build-grab-ml \
    -DPython3_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
    -DMOPHI_BUILD_PYTHON_BINDINGS=ON \
    -DMOPHI_BUILD_FERIS_NEWTON=OFF -DMOPHI_BUILD_NEWTON_XLB_DEM=OFF
cmake --build build-grab-ml -j
```

## Prepare assets once

Follow [GRAB asset preparation](grab-motion.md) for licensed downloads, the
authors' unzip procedure, subject templates, correspondence arrays, and SMPL-X
models. Standalone MANO models are unnecessary for these hand surfaces.

```bash
python -m pip install numpy scipy trimesh torch smplx
PYTHONPATH=python python scripts/convert_grab_hand_motion.py \
    ~/GRAB/extracted/grab/s2/mug_drink_2.npz
```

This writes `mug_drink_2_left.npz`, `mug_drink_2_right.npz`, and
`mug_drink_2_object.npz` under `data/grab/s2/`. To reuse existing hand exports:

```bash
PYTHONPATH=python python scripts/convert_grab_hand_motion.py \
    ~/GRAB/extracted/grab/s2/mug_drink_2.npz --object-only
```

The object NPZ stores the original mesh in metres, fixed triangle indices,
world translations, normalized **xyzw** quaternions, timestamps, frame indices,
source FPS, and provenance. No mesh simplification or recentering is performed.
World vertices are `R(q) @ vertex_local + translation`. Translation refers to
the original mesh origin, not a physical CoM. Prepared data remains git-ignored
and retains the source data's license restrictions.

## Run and inspect

```bash
PYTHONPATH=python python demo/newton_dem/grab/demo_grab_playback.py
```

By default, both hands and the mug play through the complete shared recording
at 60 rendered frames per second, then the window closes. Left is blue, right
is orange, and the object is gray. RGB axes show world XYZ and the yellow bar
spans one metre. Coordinates remain in the recorded Z-up world frame.
Use Newton's camera controls and pause control to inspect poses. The camera,
reference placement, colors, and output settings are in the demo's top-level
configuration block.

Record the complete movie without opening a visible window:

```bash
PYTHONPATH=python python demo/newton_dem/grab/demo_grab_playback.py --headless --save-movie
```

Outputs: `output/demo_grab_playback/grab_playback.mp4` and `last_frame.png`.
The PNG shows the final rendered pose. Movie encoding uses `--fps`, runs as
fast as rendering allows, and ignores interactive pauses. On some systems,
CUDA/OpenGL interoperability falls back to copies, affecting performance.

```bash
# Export frame 600, at source time 5 seconds.
PYTHONPATH=python python demo/newton_dem/grab/demo_grab_playback.py --headless \
    --start-time 5 --end-time 5 --output-dir output/demo_grab_playback/frame_600

# Inspect the right hand and object at half speed.
PYTHONPATH=python python demo/newton_dem/grab/demo_grab_playback.py \
    --hands data/grab/s2/mug_drink_2_right.npz --speed 0.5
```

`--hands` accepts one or two files; `--object` selects their matching object.
Time limits use absolute source seconds and include the final pose.
`--max-frames N` limits a smoke test. `--fps` changes rendering cadence, not
recording timestamps. Missing or inconsistent assets terminate playback.

Optional USD export uses the same mesh logging interface:

```bash
python -m pip install usd-core
PYTHONPATH=python python demo/newton_dem/grab/demo_grab_playback.py --usd
```

This writes `output/demo_grab_playback/grab_playback.usdc`, without a baked
camera or PNG/MP4 output. `--usd` and `--save-movie` are mutually exclusive.

## Reusable sampling and limitations

[`mophi.utils.grab_motion`](../../python/mophi/utils/grab_motion.py) owns loading,
validation, asset matching, and sampling. The inspection script shares its hand
loader. Prepared NPZ files are loaded with `allow_pickle=False`.

```python
from mophi.utils.grab_motion import GrabPlayback

clip = GrabPlayback(["data/grab/s2/mug_drink_2_right.npz"],
                    "data/grab/s2/mug_drink_2_object.npz")
vertices = clip.sample(5.0)  # right/object -> world-space (N, 3) float32
faces = clip.faces         # right/object -> (M, 3) int32 triangle indices
```

The shared interval is the intersection of the supplied recordings. Hand
vertices/origins and object translations interpolate linearly. Object rotation
uses shortest-path SLERP. Extrapolation is rejected; single-frame clips work.
The last source interval may be shorter than the nominal cadence. Since every
sample, including the final pose, occupies one movie frame, the movie can extend
past the source duration divided by playback speed by up to two frame periods.

Sampling runs on the CPU and stages vertices to reusable Warp display buffers.
It supplies no contact-point velocities, physical mass/inertia, materials, or
force-responsive deformation. These remain for the DEME milestone. The
renderer requires fixed mesh topology.

Run geometry/time validation and inspection regression tests in `ML`:

```bash
PYTHONPATH=python python -m unittest discover -s tests -p 'test_grab*.py' -v
```
