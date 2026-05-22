# MoPhi Design and Code Structure

This guide collects the repository structure, architectural notes, and build
surface that were previously embedded directly in the main README.

---

## Repository structure

```text
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
(Newton), the coupling cannot be done entirely in C++. MoPhi's solution is a
**thin Python-facing wrapper struct** (`PyFERISNewtonCoupler`) defined in
`python/bindings/mophi_bindings.cpp`. This wrapper:

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

The two data-exchange methods (`get_feris_node_positions`,
`set_feris_node_forces`) document the coupling points and will be wired to the
real FERIS API calls once the solver is fully configured.

---

## Enabling external solvers

External C++ solvers (FERIS, DEM-Engine) are downloaded and built in isolation via
CMake's `ExternalProject_Add` (through the `mophi_fetch_external` macro) into
`${CMAKE_BINARY_DIR}/external/<Name>/`. Pure-Python solvers (Newton, XLB,
DEME) can still be installed at configure time for local developer
convenience, but wheel builds use `pyproject.toml` dependency metadata
instead. Control which ones are fetched with the following options:

| Option | Default | Effect |
|--------|---------|--------|
| `-DMOPHI_FETCH_FERIS=ON` | OFF | Download FERIS FEA solver |
| `-DMOPHI_FETCH_DEMENGINE=ON` | OFF | Download DEM-Engine DEM solver |
| `-DMOPHI_FETCH_NEWTON=ON` | OFF | Developer convenience: install required Newton + Warp + MuJoCo (`pip install --upgrade newton==1.0.0 warp-lang==1.12.1 mujoco==3.6.0`) during CMake configure |
| `-DMOPHI_FETCH_XLB=ON` | OFF | Developer convenience: install XLB Python package (`pip install "xlb[cuda]"`) during CMake configure |
| `-DMOPHI_FETCH_DEME=ON` | OFF | Developer convenience: install DEME Python package (`pip install deme`) during CMake configure |

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
| `-DMOPHI_BUILD_NEWTON_XLB_DEM=ON` | OFF | Newton (pip) + XLB (pip) + DEME (pip) | Build Newton + XLB + DEME three-way coupler |

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
(the default). pybind11 is fetched automatically if not already available.

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

If you are using WSL and the OpenGL window does not appear, verify that WSLg GUI
support is installed and working in your environment before debugging MoPhi itself.

See [`../src/visualization/README.md`](../src/visualization/README.md) for the full
design philosophy, the public API contract, and instructions for adding new
backends.

---

## CMake option surface

| Option | Default | Effect |
|--------|---------|--------|
| `MOPHI_FETCH_FERIS` | OFF | Download FERIS into `external/FERIS/` |
| `MOPHI_FETCH_DEMENGINE` | OFF | Download DEM-Engine into `external/DEMEngine/` |
| `MOPHI_FETCH_NEWTON` | OFF | Developer convenience: install required Newton + Warp + MuJoCo (`pip install --upgrade newton==1.0.0 warp-lang==1.12.1 mujoco==3.6.0`) during CMake configure |
| `MOPHI_FETCH_XLB` | OFF | Developer convenience: install XLB Python package (`pip install "xlb[cuda]"`) during CMake configure |
| `MOPHI_FETCH_DEME` | OFF | Developer convenience: install DEME Python package (`pip install deme`) during CMake configure |
| `MOPHI_BUILD_FERIS_NEWTON` | OFF | Build the FERIS+Newton co-simulation coupler |
| `MOPHI_BUILD_NEWTON_XLB_DEM` | OFF | Build the Newton+XLB+DEME three-way coupler |
| `MOPHI_BUILD_PYTHON_BINDINGS` | ON | Build `mophi_core` Python extension |
| `MOPHI_PYTHON_PACKAGING_BUILD` | OFF | Configure the Linux Python wheel path (Python bindings ON, Newton+XLB+DEME ON, FERIS OFF, demos OFF, no configure-time pip installs) |

---

## External solvers

| Name | URL | Role | Type |
|------|-----|------|------|
| FERIS | https://github.com/Ruochun/FERIS | Total-Lagrangian FEA | C++ (CMake) |
| DEM-Engine | https://github.com/projectchrono/DEM-Engine | GPU-based DEM | C++ / CUDA (CMake) |
| Newton | https://github.com/newton-physics/newton | GPU physics (Warp) | Pure Python (pip) |
| XLB | https://github.com/Autodesk/XLB | GPU lattice-Boltzmann fluid | Pure Python / JAX (pip) |
| DEME | https://pypi.org/project/deme/ | Python discrete-element solver | Pure Python (pip) |
