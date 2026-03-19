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
│   └── couplers/                # Multi-physics co-simulation solvers
│       ├── TLFEADEMCoupler.{h,cpp}    # Direct coupling of TLFEA + DEM-Engine
│       └── TLFEANewtonCoupler.{h,cpp} # TLFEA (C++) + Newton (Python) coupling
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

# 2. Configure with the required external solvers and enable the co-sim solver
cmake -B build \
      -DMOPHI_FETCH_TLFEA=ON \
      -DMOPHI_FETCH_DEMENGINE=ON \
      -DMOPHI_BUILD_TLFEA_DEM=ON
cmake --build build

# 4. Use the Python package from the build tree
PYTHONPATH=python python3 -c "
import mophi
c = mophi.TLFEADEMCoupler()
c.initialize()
c.step()
c.finalize()
"
```

### TLFEA + Newton (Python-based GPU physics)

```bash
# Configure: fetch TLFEA, install Newton via pip, and build the coupler
cmake -B build \
      -DMOPHI_FETCH_TLFEA=ON \
      -DMOPHI_FETCH_NEWTON=ON \
      -DMOPHI_BUILD_TLFEA_NEWTON=ON
cmake --build build

# Run the Python demo (shows TLFEA solver + Newton double-pendulum together)
PYTHONPATH=python python3 demo/tlfea_newton/demo_tlfea_newton.py
```

Newton is installed by MoPhi via pip at configure time when
`-DMOPHI_FETCH_NEWTON=ON`.  You can also install it manually beforehand:

```bash
pip install newton
```

If you try to enable the co-simulation solver without first fetching the
required externals, CMake will emit a clear error:

```
CMake Error: MoPhi: Co-simulation solver 'TLFEADEMCoupler' requires the
external project 'TLFEA', which has not been fetched.
  Fix: re-run CMake with  -DMOPHI_FETCH_TLFEA=ON
  URL: https://github.com/Ruochun/TLFEA
```

---

## Design: couplers directly own solver instances

Co-simulation solvers in MoPhi (the `couplers/`) hold instances of the
external solver classes directly — there is no wrapper layer between MoPhi and
the solvers' own public APIs.

For example, `TLFEADEMCoupler::Impl` owns:
- `std::unique_ptr<deme::DEMSolver>` — DEM-Engine's main solver class
- `std::unique_ptr<tlfea::SolverBase>` — TLFEA's solver interface

The coupler's `initialize()`, `step()`, and `finalize()` methods call the
solver APIs directly (`dem->Initialize()`, `dem->DoStepDynamics()`,
`fea->Solve()`, etc.).

---

## Enabling external solvers

External solvers are downloaded via CMake's `FetchContent` into `external/<Name>/`.
Control which ones are fetched with the following options:

| Option | Default | Effect |
|--------|---------|--------|
| `-DMOPHI_FETCH_TLFEA=ON` | OFF | Download TLFEA FEA solver |
| `-DMOPHI_FETCH_DEMENGINE=ON` | OFF | Download DEM-Engine DEM solver |
| `-DMOPHI_FETCH_NEWTON=ON` | OFF | Install Newton Python package (`pip install newton`) |

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
| `-DMOPHI_BUILD_TLFEA_DEM=ON` | OFF | TLFEA + DEMEngine | Build TLFEA + DEM-Engine coupler |
| `-DMOPHI_BUILD_TLFEA_NEWTON=ON` | OFF | TLFEA + Newton (pip) + CUDA Toolkit | Build TLFEA + Newton coupler |

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

## CMake option surface

| Option | Default | Effect |
|--------|---------|--------|
| `MOPHI_FETCH_TLFEA` | OFF | Download TLFEA into `external/TLFEA/` |
| `MOPHI_FETCH_DEMENGINE` | OFF | Download DEM-Engine into `external/DEMEngine/` |
| `MOPHI_FETCH_NEWTON` | OFF | Install Newton Python package (`pip install newton`) |
| `MOPHI_BUILD_TLFEA_DEM` | OFF | Build the TLFEA+DEM co-simulation solver |
| `MOPHI_BUILD_TLFEA_NEWTON` | OFF | Build the TLFEA+Newton co-simulation coupler |
| `MOPHI_BUILD_PYTHON_BINDINGS` | ON | Build `mophi_core` Python extension |

---

## External solvers

| Name | URL | Role | Type |
|------|-----|------|------|
| TLFEA | https://github.com/Ruochun/TLFEA | Total-Lagrangian FEA | C++ (CMake) |
| DEM-Engine | https://github.com/projectchrono/DEM-Engine | GPU-based DEM | C++ / CUDA (CMake) |
| Newton | https://github.com/newton-physics/newton | GPU physics (Warp) | Pure Python (pip) |

---

## License

See [LICENSE](LICENSE).
