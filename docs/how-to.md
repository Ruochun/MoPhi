# MoPhi How-To Guide

This guide collects the user-facing build and run workflows for MoPhi.

All demos write generated files by default under
`<repository-root>/output/<demo-name>/`. The repository-level `output/`
directory is git-ignored so movies, USD scenes, state snapshots, and other
generated artifacts do not appear in normal staging operations.

Visual demos include a persistent world reference: RGB segments show the
positive X, Y, and Z directions, and a yellow endpoint-marked bar shows one
metre of world-space length.

---

## Python packages used by MoPhi workflows

The packaged MoPhi wheel declares the following Python packages as regular
runtime dependencies, because they are used across MoPhi's current Python-side
workflows, demos, and visualization/movie-recording paths:

- `torch`
- `GitPython`
- `PyYAML`
- `pycollada`
- `mujoco_warp==3.6.0`
- `pyglet`
- `imageio`
- `imageio-ffmpeg`

Notes:

- `demo/newton_xlb_dem/anymal_robot_multiphysics/demo_anymal_robot_multiphysics.py` directly requires `torch`.
- Movie recording paths require `imageio` and `imageio-ffmpeg`.
- When installing from a source checkout instead of a built wheel, install these
  packages alongside the solver packages that your workflow needs.

---

## Quick start

```bash
# 1. Clone MoPhi (--recurse-submodules is required for MoPhiEssentials)
git clone --recurse-submodules https://github.com/Ruochun/MoPhi.git
cd MoPhi

# If you already cloned without --recurse-submodules, initialise the submodule manually:
#   git submodule update --init

# 2. Configure with the desired co-simulation coupler and active Python
cmake -B build-py311 \
      -DPython3_EXECUTABLE="$(python3 -c 'import sys; print(sys.executable)')" \
      -DMOPHI_FETCH_FERIS=ON \
      -DMOPHI_FETCH_NEWTON=ON \
      -DMOPHI_BUILD_FERIS_NEWTON=ON
cmake --build build-py311

# 3. Run a demo
PYTHONPATH=python python3 demo/feris_newton/demo_feris_newton.py
```

---

## Python build workflows

MoPhi has two distinct Python build workflows. Choose the source-tree workflow
for normal development and the wheel workflow only when producing an
installable or distributable artifact.

| Workflow | Command | Output | Intended use |
|----------|---------|--------|--------------|
| Source-tree developer build | `cmake -B build-py<version> ...` followed by `cmake --build` | `python/mophi/mophi_core.cpython-<version>-<platform>.so` | Editing code and running demos with `PYTHONPATH=python` |
| Single-interpreter wheel | `python -m build --wheel` | One ABI-specific wheel under `dist/` | Testing or distributing one Python version |
| Linux wheel matrix | `python -m cibuildwheel` | `cp311`–`cp314` repaired wheels under `wheelhouse/` | Release artifacts and compatibility testing |

`mophi_core` is a CPython extension. A module built for one Python minor
version cannot be imported by another; for example, Python 3.12 cannot load a
`cpython-311` module.

### Source-tree developer builds

Use one CMake build directory per Python minor version. Activate the intended
environment first and pass its interpreter explicitly:

```bash
# Example: Python 3.12 developer build.
python --version
cmake -B build-py312 \
      -DPython3_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
      -DMOPHI_BUILD_NEWTON_XLB_DEM=ON
cmake --build build-py312
```

A non-packaging developer build places the compiled extension beside the
source package. Verify both the selected interpreter and loaded ABI:

```bash
PYTHONPATH=python python -c \
  "import sys, mophi; print(sys.version); print(mophi.mophi_core.__file__)"
```

The output should contain the matching version, such as:

```text
python/mophi/mophi_core.cpython-312-x86_64-linux-gnu.so
```

Multiple compatible extensions can coexist in `python/mophi/`; CPython chooses
the file matching the running interpreter. Keep the CMake state separate:

```text
build-py311/    # configured with Python 3.11
build-py312/    # configured with Python 3.12
build-py313/    # configured with Python 3.13
build-py314/    # configured with Python 3.14
```

Do not reuse one of these directories for another Python minor version. CMake
caches interpreter discovery, and changing conda environments does not rewrite
an existing cache automatically.

When a source-tree import fails, inspect the interpreter and available modules:

```bash
python -c "import sys; print(sys.executable); print(sys.version)"
find python/mophi -maxdepth 1 -name 'mophi_core*.so' -print
```

If the matching extension is absent, configure and build the corresponding
`build-py<version>` directory. A misleading “partially initialized module” or
“circular import” message usually means no compatible `mophi_core` extension
was found; MoPhi's package initializer now augments this with the detected ABI
details and rebuild instructions.

### Linux Python 3.11–3.14 wheels

MoPhi has a standard PEP 517 packaging entry point in `pyproject.toml`
using `scikit-build-core`. The supported packaged distribution path currently
targets **Linux + CPython 3.11 through 3.14** and enables:

- Python bindings: **ON**
- Newton + XLB + DEME coupler: **ON**
- FERIS coupler: **OFF**
- demos: **OFF**

The wheel metadata declares the runtime Python dependencies needed for the
supported distributed install experience:

- `newton==1.0.0`
- `warp-lang==1.12.1`
- `mujoco==3.6.0`
- `deme3>=3.0.9,<4`
- `xlb[cuda]`
- `torch`
- `GitPython`
- `PyYAML`
- `pycollada`
- `mujoco_warp==3.6.0`
- `pyglet`
- `imageio`
- `imageio-ffmpeg`

#### Build one wheel for the active interpreter

Build from a checkout with submodules initialized. Replace `python3.11` with
the desired interpreter:

```bash
git clone --recurse-submodules https://github.com/Ruochun/MoPhi.git
cd MoPhi
python3.11 -m pip install --upgrade build
python3.11 -m build --wheel
```

`python -m build` invokes scikit-build-core, which configures CMake internally
with `MOPHI_PYTHON_PACKAGING_BUILD=ON`. It does not use or update the developer
extension under `python/mophi/`.

For Linux distribution, repair this single wheel so bundled native libraries
satisfy manylinux policy:

```bash
python3.11 -m pip install --upgrade auditwheel
auditwheel repair dist/mophi-*.whl -w wheelhouse/
```

Test the repaired artifact in a clean environment without `PYTHONPATH=python`,
which would otherwise override the installed wheel with the source checkout:

```bash
python3.11 -m venv /tmp/mophi-wheel-test
/tmp/mophi-wheel-test/bin/python -m pip install wheelhouse/mophi-*.whl
cd /tmp
/tmp/mophi-wheel-test/bin/python -c \
  "import sys, mophi; print(sys.version); print(mophi.mophi_core.__file__)"
```

#### Build the Python-version matrix

The matrix in `pyproject.toml` targets Linux `cp311`, `cp312`, `cp313`, and
`cp314`. Build it with `cibuildwheel` from a Linux host with Docker available:

```bash
python -m pip install --upgrade cibuildwheel
python -m cibuildwheel --output-dir wheelhouse
```

Python 3.11 and 3.12 are the primary validation targets. Python 3.13 and 3.14
are enabled on a best-effort basis because availability may still be limited
by solver dependency wheels rather than MoPhi itself.

`cibuildwheel` builds in manylinux containers, repairs each wheel, and runs the
configured import smoke test. Install a wheel matching the target interpreter;
its metadata pulls in Newton, Warp, MuJoCo, XLB, DEME, and the other runtime
dependencies automatically:

```bash
python3.12 -m pip install wheelhouse/mophi-*-cp312-*.whl
```

Building or installing any wheel does not update the developer extensions in
the source checkout. Likewise, running with `PYTHONPATH=python` tests the source
package, not a wheel installed in the active environment.

Installing from the wheel does **not** build MoPhi from source locally, but the
wheel is **not** a fully self-contained offline installer. `pip` still needs to
resolve the wheel's declared runtime dependencies (`newton`, `warp-lang`,
`mujoco`, `deme3`, `xlb[cuda]`, `torch`, `GitPython`, `PyYAML`, `pycollada`,
`mujoco_warp`, `pyglet`, `imageio`, and `imageio-ffmpeg`) from either the
internet or a local wheelhouse.

For an offline installation, pre-download the MoPhi wheel plus all required
dependency wheels, then install from a local wheelhouse:

```bash
python -m pip install --no-index --find-links /path/to/wheelhouse mophi
```

---

## FERIS + Newton (Python-based GPU physics)

```bash
# Configure: fetch FERIS, install Newton via pip, and build the coupler
cmake -B build-py311 \
      -DPython3_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
      -DMOPHI_FETCH_FERIS=ON \
      -DMOPHI_FETCH_NEWTON=ON \
      -DMOPHI_BUILD_FERIS_NEWTON=ON
cmake --build build-py311

# Run the FERIS + Newton demo
PYTHONPATH=python python3 demo/feris_newton/demo_feris_newton.py
```

The `FERISNewtonCoupler` bridges C++ (FERIS) and Python (Newton) inside a single
`step()` call. Pass the Newton model and solver at initialization time; the coupler
manages states, forward-kinematics setup, and the per-step coupling data exchange.

**Coupling data exchange** (FERIS ↔ Newton) is handled by two methods on the C++ side:

| Method | Direction | Purpose |
|--------|-----------|---------|
| `get_feris_node_positions()` | FERIS → Newton | Deformed FEA node positions forwarded to Newton geometry |
| `set_feris_node_forces(forces)` | Newton → FERIS | Contact/body forces from Newton applied as FERIS external loads |

Newton (with required Warp and MuJoCo) is installed by MoPhi via pip at configure time when
`-DMOPHI_FETCH_NEWTON=ON`. You can also install it manually beforehand:

```bash
pip install --upgrade newton==1.0.0 warp-lang==1.12.1 mujoco==3.6.0
```

---

## Newton + XLB + DEME (three-way coupling)

```bash
# Install the required Python packages first
pip install --upgrade newton==1.0.0 warp-lang==1.12.1 mujoco==3.6.0 "xlb[cuda]" deme3 \
    torch GitPython PyYAML pycollada mujoco_warp==3.6.0 pyglet imageio imageio-ffmpeg

# Configure and build the coupler
cmake -B build-py311 \
      -DPython3_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
      -DMOPHI_BUILD_NEWTON_XLB_DEM=ON
cmake --build build-py311

# Run the Python demo (walking robot + XLB + DEME)
PYTHONPATH=python python3 demo/newton_xlb_dem/anymal_robot_multiphysics/demo_anymal_robot_multiphysics.py
# Run the Newton-only walking baseline demo (flat ground + light dynamic boxes)
PYTHONPATH=python python3 demo/newton_xlb_dem/anymal_robot_multiphysics/demo_anymal_robot_newton_baseline.py
# Run the DEME excavator demo (robotic arm with excavator + DEME force feedback)
PYTHONPATH=python python3 demo/newton_xlb_dem/excavation_comparison/demo_claw_newton.py
# Run the Newton coarse-cube comparison demo (same arm policy, coarse rigid terrain)
PYTHONPATH=python python3 demo/newton_xlb_dem/excavation_comparison/demo_claw_newton_cubes.py
# Run the interactive Newton humanoid walking baseline
PYTHONPATH=python python3 demo/newton_xlb_dem/humanoid_complex_environment/demo_humanoid_complex_environment.py
# Run the two-robot Newton-DEME humanoid fighting demo
PYTHONPATH=python python3 demo/newton_xlb_dem/humanoid_complex_environment/demo_humanoid_robot_fighting.py
# Run the Newton + XLB humanoid swimming idea demo
PYTHONPATH=python python3 demo/newton_xlb_dem/humanoid_complex_environment/demo_humanoid_swimming.py
# Run 64 downward-facing Allegro hands grasping loose Newton cubes
PYTHONPATH=python python3 demo/newton_xlb_dem/flexible_hand_manipulation/demo_flexible_hand_newton.py
# Run one downward-facing Allegro hand interacting with DEME ellipsoidal clumps
PYTHONPATH=python python3 demo/newton_xlb_dem/flexible_hand_manipulation/demo_flexible_hand_deme.py
# Run the robotic 3D-printing co-simulation (Newton arm + DEME spheres)
PYTHONPATH=python python3 demo/newton_dem/robotic_3d_printing/demo_robotic_3d_printing.py
```

### Robotic 3D printing: Newton/DEME co-simulation

`demo_robotic_3d_printing.py` is a Newton + DEME additive-manufacturing demo
whose sphere population currently exercises DEME's custom
`ForceModelWithCohesion.cu` kernel with `DEME_COHESION = 0.0`. This isolates
the custom-kernel path before cohesion or domain changes are enabled. The model
is stored under `data/force_models/` and loaded explicitly by the demo.
DEME's regular-grid sampler derives that population from a configurable
funnel-local box contained within the bowl. The initial Newton funnel transform
maps those sampled points into world space, rather than relying on an unrelated
global center or a fixed particle count.
Newton downloads its public Universal Robots UR10
asset on first use, then the script attaches the hollow
`data/mesh/funnel.obj` print-head mesh to the wrist. A static build plate and
support frame complete the machine. After settling, the arm moves once from its
starting posture toward a modest lateral endpoint, leaving a nearly straight
particle trail without reversing direction. DEME initializes
a small sphere cloud, six analytical box walls from explicit XYZ ranges, and a
separate analytical plane at the build-plate height. The box's lower Z wall is
below the build plate, avoiding coincident contact boundaries.
Newton's machine stage uses zero gravity so its prescribed arm posture does not
sag during warm-up; DEME retains its own downward gravity for particle settling.
The DEME funnel is always an externally driven contact proxy. It starts at the
Newton funnel's configured initial pose, and every Newton substep updates its
DEME tracker pose before DEME advances. There is no independent DEME funnel
trajectory or conditional pose-coupling mode. DEME-to-Newton force feedback is
controlled separately by `ENABLE_DEME_TO_NEWTON_FORCE_FEEDBACK` and defaults to
`False`, so particles do not disturb the prescribed Newton arm motion. When
enabled, DEME writes its contact acceleration, angular acceleration, mass,
inertia, and orientation into caller-owned Warp CUDA arrays. A Warp kernel
forms the world-space wrench in Newton's external body-force array, which the
coupler copies device-to-device into Newton before integration. The code marks
a pending correction to shift torque from the DEME funnel frame to the Newton
end-effector frame before force feedback is used for quantitative studies. Particle
positions are likewise copied directly into a Warp array for rendering. The
robotic-printing demo still uses tracker host pose setters and marks that
exchange point for migration to the device-native setters available in deme3
3.0.9. Its generated movie, USD (when selected), and run metadata are written beneath
`output/demo_robotic_3d_printing/`.

The initial settling interval advances both Newton and DEME. With the default
`RENDER_SETTLING_PHASE = True`, it is rendered and recorded at `RENDER_FPS`
before the arm begins its printing trajectory; disabling that flag keeps the
same physics warm-up but omits its frames from the output.

Build and run it directly or through its CMake target:

```bash
cmake -B build-py311 \
      -DPython3_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
      -DMOPHI_BUILD_NEWTON_XLB_DEM=ON \
      -DMOPHI_BUILD_DEMOS=ON
cmake --build build-py311
PYTHONPATH=python python3 demo/newton_dem/robotic_3d_printing/demo_robotic_3d_printing.py

# Equivalent convenience target (opens the interactive viewer):
cmake --build build-py311 --target demo_robotic_3d_printing
```

Prerequisites are the Newton, Warp, MuJoCo, `deme3`, imageio, and
imageio-ffmpeg packages listed above. Newton and both DEME workers must use the
same logical CUDA device for the direct pointer exchange.

The Newton-only ANYmal baseline demo does not instantiate XLB or DEME at
runtime; it keeps the walking-policy setup on flat ground and adds only a few
light Newton-managed contact boxes for baseline comparison against the
multiphysics case.

The humanoid complex-environment demo starts from Newton 1.0.0's public
keyboard-controlled Unitree G1 walking-policy example. On first run, Newton
downloads the G1 model, policy, and YAML configuration from the public
`newton-assets` repository; no manual asset placement is required. This first
stage preserves the policy-trained ground colliders, adds convex Newton contact
proxies from the G1's visual body meshes for external-object contact, and
places lightweight dynamic boxes in its walking path. It also adds MoPhi-style
configuration, movie output, run metadata, and a final-state snapshot. Press
`I` in the viewer to walk forward into the boxes and observe body-object
contact. Press `B` to interactively spawn an additional dynamic box in front
of the robot. The demo uses a configurable preallocated object pool because
Newton model topology is fixed after finalization; press `P` to reset the
robot and return spawned objects to the pool.
The viewer also shows a non-physical warehouse aisle; set
`SHOW_WAREHOUSE_ENVIRONMENT = False` to hide the decorative scene.

The humanoid robot-fighting demo places two G1 robots face-to-face and commands
both policies forward for five seconds. Newton manages articulated dynamics and
uses its coarse policy colliders for ground contact. DEME resolves cross-robot
mesh contact and feeds each proxy body's force and torque back into Newton;
Newton's duplicate cross-robot collider pairs are disabled. The scene contains
no dynamic boxes, warehouse shelves, or other decorative obstacles. The demo
records a mesh manifest under `output/demo_humanoid_robot_fighting/`, alongside
its movie, metadata, and final-state snapshot.
Set `ROBOT_CONTACT_MODEL` to `"deme"` for this coupled path or `"newton"`
for the original coarse-contact comparison. `SHOW_CONTACT_PROXY_MESHES`
independently turns the wireframe overlay on or off; hiding it does not disable
DEME contact.
It also loads `data/mesh/cube.obj` and draws scaled instances of that triangle
mesh around every visual robot part. The cyan and orange wireframe overlays
follow the two robots' links and show the meshes used for DEME contact. Proxy
components attached to the same Newton body are merged into a body-local OBJ
under `output/demo_humanoid_robot_fighting/deme_contact_proxies/`; configure
the mesh path, padding, line width, colors, or visibility in the demo's
contact-proxy block.
The default script waits until the robots have met, then performs four
alternating punches between 2.05 and 4.85 seconds. Kicks are omitted because
the locomotion controller does not robustly support the required single-leg
balance. `FIGHT_ATTACK_SEQUENCE` stores each attack as
`(start_time, duration, attacker, kind, side)`. The attack trajectory
is added to the learned locomotion target rather than prescribing body poses,
so contact forces and loss of balance remain part of the Newton simulation.
Set `ENABLE_SCRIPTED_FIGHT = False` for the previous approach-only behavior.
The visualization block provides brighter sky/light colors and a closer camera
for viewing strike details; these settings affect rendering only.

The humanoid swimming idea demo places the G1 in an XLB fluid domain without a
ground plane. Newton advances a scripted swimming stroke while analytic
buoyancy, drag, and propulsion keep the robot suspended and moving. Ten
axis-aligned XLB obstacle boxes follow the torso, pelvis, arms, and legs so the
visualized flow responds to the prescribed limb motion. Edit the
`PRESCRIBED_SWIM_MOTION` configuration block to change each swimming joint's
pose offset, amplitude, and phase. This first stage is intentionally one-way:
XLB does not yet calculate forces that feed back into Newton.
The underwater sandbed, rocks, vegetation, and background colors are
renderer-only decorations. Set `SHOW_UNDERWATER_ENVIRONMENT = False` to hide
them.

The flexible-hand Newton demo places 64 downward-facing
Allegro hands in an 8-by-8 laboratory-scale array over individual shallow trays
filled with loose Newton cubes. The flexible-hand DEME demo runs one hand/tray
instance instead, using DEME's HCP sampler to place ellipsoidal clumps above
the tray with small reproducible random initial velocities. DEME reuses
Newton's simplified convex hand contact meshes rather than the fine visual
meshes. These proxies have clump contact disabled during a blocking DEME
settling phase, then switch into an active-contact family for the hand motion.
This stage does not feed DEME forces back into Newton.

Newton stores assets under the platform user cache by default, normally
`~/.cache/newton/` on Linux. Set `NEWTON_CACHE_PATH` to choose a different
persistent cache location:

```bash
NEWTON_CACHE_PATH=/path/to/newton-cache \
PYTHONPATH=python python3 demo/newton_xlb_dem/humanoid_complex_environment/demo_humanoid_robot_fighting.py
```

MoPhi demos validate the particular files they require. If Newton returns an
incomplete cached sparse checkout, MoPhi automatically invokes Newton's
force-refresh path once and validates the replacement. The reusable helper,
supported behavior, and offline limitations are documented in
[`docs/newton-assets.md`](newton-assets.md).

If the automatic refresh cannot repair the checkout because the configured
cache root is unwritable or corrupted, use a new cache directory as a
**last resort**:

```bash
export NEWTON_CACHE_PATH="$HOME/.cache/newton-mophi-fresh"
PYTHONPATH=python python3 demo/newton_xlb_dem/humanoid_complex_environment/demo_humanoid_robot_fighting.py
```

Keep `NEWTON_CACHE_PATH` set for subsequent runs that should reuse this cache.

`NewtonXLBDEMCoupler` is a three-way co-simulation coupler:

- **Newton** (active) — drives an articulated quadruped walking robot using
  position-based dynamics (XPBD). The robot's spatial representation (body
  position + orientation quaternion per body) is extracted at every step via
  `get_robot_body_transforms()`.
- **XLB** — a JAX-based LBM fluid solver that advances each frame. The robot is
  represented as a prescribed moving obstacle whose boundary-condition masks are
  updated directly on the GPU.
- **DEME** — a Python discrete-element solver (provided by `pip install deme3`) that advances a
  live particle simulation alongside the robot.

```python
import math

import mophi, newton, warp as wp

wp.init()
builder = newton.ModelBuilder()
# ... build walking-robot model (see demo script for full quadruped) ...
model = builder.finalize()
solver = newton.solvers.SolverXPBD(model)

coupler = mophi.NewtonXLBDEMCoupler()
coupler.initialize(newton_model=model, newton_solver=solver, sim_dt=2e-3)

for step in range(steps):
    # Update joint control targets for the trot gait before each step.
    targets = coupler.newton_control.joint_target_pos.numpy().copy()
    # ... set sinusoidal hip/knee angles based on step * sim_dt ...
    wp.copy(coupler.newton_control.joint_target_pos,
            wp.from_numpy(targets, dtype=wp.float32, device="cuda"))

    coupler.step_newton()  # advance the Newton rigid-body solver
    coupler.step_deme()    # advance the DEME discrete-element solver

    # Extract spatial representation: [{px,py,pz,qx,qy,qz,qw}, ...] per body.
    transforms = coupler.get_robot_body_transforms()

coupler.finalize()
```

XLB is part of the supported distributed install path for the Newton + XLB +
DEME workflow. The packaged wheel declares `xlb[cuda]` as a runtime
dependency. For source-tree developer builds, you can install it manually or
use the developer-convenience CMake fetch option:

```bash
pip install "xlb[cuda]"
```

Likewise, the packaged wheel declares `deme3` as a runtime dependency. For
source-tree developer builds, you can install it manually:

```bash
pip install deme3
```

The DEME distribution and import module are intentionally configurable. The
default `deme3` distribution exposes the `deme` module. To test another
API-compatible distribution during CMake configuration, set both names as
needed:

```bash
cmake -B build-py311 \
      -DPython3_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
      -DMOPHI_PACKAGE_DEME_DISTRIBUTION=deme \
      -DMOPHI_PACKAGE_DEME_IMPORT_MODULE=deme \
      -DMOPHI_FETCH_DEME=ON
```

At runtime, Python dependencies are described by the general provider registry
in `mophi.utils.package_provider`. It covers every runtime dependency declared
in `pyproject.toml` and keeps each pip distribution name separate from its
Python import module. The default mappings include `warp-lang` to `warp`,
`PyYAML` to `yaml`, `GitPython` to `git`, and `deme3` to `deme`.

Every provider accepts the same environment overrides:

```bash
MOPHI_PACKAGE_<NAME>_DISTRIBUTION=<pip-requirement>
MOPHI_PACKAGE_<NAME>_IMPORT_MODULE=<python-module>
```

For example, a DEME-compatible provider with a different module name can be
selected with:

```bash
MOPHI_PACKAGE_DEME_IMPORT_MODULE=another_module python demo/...
```

Python code resolves a provider uniformly:

```python
from mophi.utils.package_provider import load_package_provider

DEME = load_package_provider("deme")
```

Changing the dependency installed with a packaged wheel remains a one-line
change to the corresponding dependency entry in `pyproject.toml`; update the
matching default in `DEFAULT_PACKAGE_PROVIDERS` at the same time.
