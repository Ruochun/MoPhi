# MoPhi

**MoPhi** is a modular multi-physics co-simulation framework.  It provides the
glue between independent single-physics solvers, letting them exchange data and
march forward together as a tightly-coupled simulation.

---

## Repository structure

```
MoPhi/
├── CMakeLists.txt               # Root build file — all options live here
├── cmake/
│   └── MoPhiExternalUtils.cmake # Macros for registering / fetching external solvers
├── external/
│   ├── MoPhiEssentials/         # Required git submodule — Logger, Real3, utilities
│   ├── ExternalProjects.cmake   # ← add new solver URLs here
│   └── CMakeLists.txt           # Fetches selected external projects
├── src/
│   ├── couplers/                # Multi-physics co-simulation couplers
│   │   └── feris_newton/        # FERIS (C++) + Newton (Python) coupling
│   └── visualization/           # Backend-agnostic visualization layer (pure Python)
│       ├── README.md            # ← design philosophy and API contract
│       ├── opengl_visualizer.py # Real-time OpenGL backend (via Newton's ViewerGL)
│       └── omniverse_visualizer.py # Offline USD export backend
└── python/
    ├── CMakeLists.txt           # pybind11 module — fetched automatically
    ├── bindings/
    │   └── mophi_bindings.cpp   # pybind11 entry point
    └── mophi/
        └── __init__.py          # Python package
```

---

## Prerequisites

| Tool | Minimum version |
|------|----------------|
| CMake | 3.18 |
| C++ compiler | C++17 (GCC 9 / Clang 10 / MSVC 2019) |
| Python (optional) | 3.8 — for Python bindings |
| Git | any recent version — for fetching externals |

---

## Quick start

```bash
# 1. Clone MoPhi (--recurse-submodules is required for MoPhiEssentials)
git clone --recurse-submodules https://github.com/Ruochun/MoPhi.git
cd MoPhi

# If you already cloned without --recurse-submodules, initialise the submodule manually:
#   git submodule update --init

# 2. Configure with the desired co-simulation coupler enabled
cmake -B build \
      -DMOPHI_FETCH_FERIS=ON \
      -DMOPHI_FETCH_NEWTON=ON \
      -DMOPHI_BUILD_FERIS_NEWTON=ON
cmake --build build

# 3. Run a demo
PYTHONPATH=python python3 demo/feris_newton/demo_feris_newton.py
```

### FERIS + Newton (Python-based GPU physics)

```bash
# Configure: fetch FERIS, install Newton via pip, and build the coupler
cmake -B build \
      -DMOPHI_FETCH_FERIS=ON \
      -DMOPHI_FETCH_NEWTON=ON \
      -DMOPHI_BUILD_FERIS_NEWTON=ON
cmake --build build

# Run the FERIS + Newton demo
PYTHONPATH=python python3 demo/feris_newton/demo_feris_newton.py
```

The `FERISNewtonCoupler` bridges C++ (FERIS) and Python (Newton) inside a single
`step()` call.  Pass the Newton model and solver at initialization time; the coupler
manages states, forward-kinematics setup, and the per-step coupling data exchange:

```python
import mophi, newton, warp as wp

wp.init()
builder = newton.ModelBuilder()
# ... configure rigid-body scene ...
model  = builder.finalize()
solver = newton.solvers.SolverXPBD(model)

coupler = mophi.FERISNewtonCoupler()
coupler.initialize(newton_model=model, newton_solver=solver, sim_dt=1e-3)

for _ in range(steps):
    coupler.step()   # advances FERIS, exchanges data, advances Newton

coupler.finalize()
```

**Coupling data exchange** (FERIS ↔ Newton) is handled by two methods on the C++ side:

| Method | Direction | Purpose |
|--------|-----------|---------|
| `get_feris_node_positions()` | FERIS → Newton | Deformed FEA node positions forwarded to Newton geometry |
| `set_feris_node_forces(forces)` | Newton → FERIS | Contact/body forces from Newton applied as FERIS external loads |

Newton (with pinned Warp) is installed by MoPhi via pip at configure time when
`-DMOPHI_FETCH_NEWTON=ON`.  You can also install it manually beforehand:

```bash
pip install --upgrade newton==1.1.0 warp-lang==1.12.1
```

### Newton + XLB + DEME (three-way coupling)

```bash
# Configure: install Newton, XLB, and DEME via pip, and build the coupler
cmake -B build \
      -DMOPHI_FETCH_NEWTON=ON \
      -DMOPHI_FETCH_XLB=ON \
      -DMOPHI_FETCH_DEME=ON \
      -DMOPHI_BUILD_NEWTON_XLB_DEM=ON
cmake --build build

# Run the Python demo (walking robot + XLB + DEME)
PYTHONPATH=python python3 demo/newton_xlb_dem/demo_newton_xlb_dem.py
# Run another demo (robotic arm with excavator + XLB + DEME)
PYTHONPATH=python python3 demo/newton_xlb_dem/demo_claw_newton.py
```

`NewtonXLBDEMCoupler` is a three-way co-simulation coupler:

- **Newton** (active) — drives an articulated quadruped walking robot using
  position-based dynamics (XPBD).  The robot's spatial representation (body
  position + orientation quaternion per body) is extracted at every step via
  `get_robot_body_transforms()`.
- **XLB** (placeholder) — a JAX-based LBM fluid solver is instantiated but not
  yet seriously advanced.  Future work will feed the robot's geometry as a
  moving boundary condition.
- **DEME** (placeholder) — a Python discrete-element solver (pip install deme)
  is instantiated but not yet advanced.  Future work will introduce
  particle–robot coupling.

```python
import mophi, newton, warp as wp, math

wp.init()
builder = newton.ModelBuilder()
# ... build walking-robot model (see demo script for full quadruped) ...
model  = builder.finalize()
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

XLB is optionally installed by MoPhi via pip at configure time when
`-DMOPHI_FETCH_XLB=ON`.  You can also install it manually:

```bash
pip install xlb
```

If you try to enable the co-simulation solver without first fetching the
required externals, CMake will emit a clear error:

```
CMake Error: MoPhi: Co-simulation solver 'FERISNewtonCoupler' requires the
external project 'FERIS', which has not been fetched.
  Fix: re-run CMake with  -DMOPHI_FETCH_FERIS=ON
  URL: https://github.com/Ruochun/FERIS
```

---

## Design: couplers directly own solver instances

Co-simulation solvers in MoPhi (the `couplers/`) hold instances of the
external solver classes directly — there is no wrapper layer between MoPhi and
the solvers' own public APIs.

For example, `PyFERISNewtonCoupler::FERISImpl` owns:
- `std::unique_ptr<feris::GPU_FEAT10_Data>` — FERIS element data
- `std::unique_ptr<feris::SyncedAdamWNocoopSolver>` — FERIS time integrator

The coupler's `initialize()`, `step()`, and `finalize()` methods call the
solver APIs directly (`fea_element->Initialize()`, `fea_solver->Solve()`, etc.).

### Coupling a C++ solver with a Python solver (FERIS + Newton)

When one solver is pure C++ (FERIS) and the other is a pure Python package
(Newton), the coupling cannot be done entirely in C++.  MoPhi's solution is a
**thin Python-facing wrapper struct** (`PyFERISNewtonCoupler`) defined in
`python/bindings/mophi_bindings.cpp`.  This wrapper:

1. Owns the FERIS solver objects in a C++ pimpl member.
2. Stores the Newton model, solver, states, control, and contacts as
   `py::object` members (reference-counted Python objects).
3. Exposes a unified `initialize() / step() / finalize()` interface to Python.
4. Inside `step()`, drives the full coupling cycle:
    - Advances FERIS via `coupler.step()`.
    - Reads `coupler.get_feris_node_positions()` → forwards to Newton (TODO: update
      Newton geometry / anchor points).
    - Advances Newton (clear forces → collide → step → swap states).
    - Reads Newton forces → applies via `coupler.set_feris_node_forces()` (TODO).

The two data-exchange methods (`get_feris_node_positions`, `set_feris_node_forces`) document
the coupling points and will be wired to the real FERIS API calls once the
solver is fully configured.

---

## Enabling external solvers

External C++ solvers (FERIS, DEM-Engine) are downloaded and built in isolation via
CMake's `ExternalProject_Add` (through the `mophi_fetch_external` macro) into
`${CMAKE_BINARY_DIR}/external/<Name>/`.  Pure-Python solvers (Newton, XLB, DEME) are
installed as pip packages at configure time.  Control which ones are fetched with the
following options:

| Option | Default | Effect |
|--------|---------|--------|
| `-DMOPHI_FETCH_FERIS=ON` | OFF | Download FERIS FEA solver |
| `-DMOPHI_FETCH_DEMENGINE=ON` | OFF | Download DEM-Engine DEM solver |
| `-DMOPHI_FETCH_NEWTON=ON` | OFF | Install pinned Newton + Warp (`pip install --upgrade newton==1.1.0 warp-lang==1.12.1`) |
| `-DMOPHI_FETCH_XLB=ON` | OFF | Install XLB Python package (`pip install xlb`) |
| `-DMOPHI_FETCH_DEME=ON` | OFF | Install DEME Python package (`pip install deme`) |

### Adding a new external solver

1. Open `external/ExternalProjects.cmake` and add:
   ```cmake
   mophi_register_external(
       NAME  MySolver
       URL   "https://github.com/org/MySolver"
       TAG   "main"
   )
   ```
2. Add a fetch block in `external/CMakeLists.txt`:
   ```cmake
   if(MOPHI_FETCH_MYSOLVER)
       mophi_fetch_external(MySolver)
   endif()
   ```
3. Add the corresponding `option()` in the root `CMakeLists.txt`.

---

## Co-simulation solvers

| Option | Default | Required externals | Effect |
|--------|---------|-------------------|--------|
| `-DMOPHI_BUILD_FERIS_NEWTON=ON` | OFF | FERIS + Newton (pip) + CUDA Toolkit | Build FERIS + Newton coupler |
| `-DMOPHI_BUILD_NEWTON_XLB_DEM=ON` | OFF | Newton (pip) + XLB (pip, optional) + DEME (pip, optional) | Build Newton + XLB + DEME three-way coupler |

If a required external is not fetched, CMake emits a `FATAL_ERROR` at
configure time with instructions on how to resolve the problem.

### Adding a new co-simulation solver

1. Add `<NewCoupler>.h` / `.cpp` to `src/couplers/`.
2. Add an `if(MOPHI_BUILD_<NAME>)` block in `src/couplers/CMakeLists.txt`
   that calls `mophi_require_externals()` and links the new library into
   `mophi_couplers`.
3. Add the corresponding `option()` in the root `CMakeLists.txt`.
4. Register the new class in `python/bindings/mophi_bindings.cpp`.

---

## Python bindings

Python bindings are built automatically when `MOPHI_BUILD_PYTHON_BINDINGS=ON`
(the default).  pybind11 is fetched automatically if not already available.

After building, add the `python/` directory to your `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/MoPhi/python:$PYTHONPATH
python3 -c "import mophi; help(mophi)"
```

Co-simulation solver classes are only exported when they have been compiled.
The `mophi` package uses a graceful `try/except ImportError` pattern so that
`import mophi` always succeeds regardless of which solvers were built.

---

## Visualization

MoPhi ships a backend-agnostic visualization layer in `src/visualization/`.
Two backends are provided out of the box:

| Class | Backend | Output |
|-------|---------|--------|
| `mophi.OpenGLVisualizer(model)` | Newton `ViewerGL` | Real-time OpenGL window |
| `mophi.OmniverseVisualizer(output_path, fps)` | OpenUSD (`pip install usd-core`) | Offline `.usdc` / `.usda` file |

Demos switch backends by changing a single constructor call; the rest of the
simulation loop is identical for both.

See [`src/visualization/README.md`](src/visualization/README.md) for the full
design philosophy, the public API contract, and instructions for adding new
backends.

---

## CMake option surface

| Option | Default | Effect |
|--------|---------|--------|
| `MOPHI_FETCH_FERIS` | OFF | Download FERIS into `external/FERIS/` |
| `MOPHI_FETCH_DEMENGINE` | OFF | Download DEM-Engine into `external/DEMEngine/` |
| `MOPHI_FETCH_NEWTON` | OFF | Install pinned Newton + Warp (`pip install --upgrade newton==1.1.0 warp-lang==1.12.1`) |
| `MOPHI_FETCH_XLB` | OFF | Install XLB Python package (`pip install xlb`) |
| `MOPHI_FETCH_DEME` | OFF | Install DEME Python package (`pip install deme`) |
| `MOPHI_BUILD_FERIS_NEWTON` | OFF | Build the FERIS+Newton co-simulation coupler |
| `MOPHI_BUILD_NEWTON_XLB_DEM` | OFF | Build the Newton+XLB+DEME three-way coupler |
| `MOPHI_BUILD_PYTHON_BINDINGS` | ON | Build `mophi_core` Python extension |

---

## External solvers

| Name | URL | Role | Type |
|------|-----|------|------|
| FERIS | https://github.com/Ruochun/FERIS | Total-Lagrangian FEA | C++ (CMake) |
| DEM-Engine | https://github.com/projectchrono/DEM-Engine | GPU-based DEM | C++ / CUDA (CMake) |
| Newton | https://github.com/newton-physics/newton | GPU physics (Warp) | Pure Python (pip) |
| XLB | https://github.com/Autodesk/XLB | GPU lattice-Boltzmann fluid | Pure Python / JAX (pip) |
| DEME | https://pypi.org/project/deme/ | Python discrete-element solver | Pure Python (pip) |

---

## License

See [LICENSE](LICENSE).
