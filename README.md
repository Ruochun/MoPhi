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
│   ├── ExternalProjects.cmake   # ← add new solver URLs here
│   └── CMakeLists.txt           # Fetches selected external projects
├── src/
│   ├── wrappers/                # Thin C++ wrappers around each external solver
│   │   ├── TLFEAWrapper.{h,cpp}
│   │   └── DEMEngineWrapper.{h,cpp}
│   └── couplers/                # Multi-physics co-simulation solvers
│       └── TLFEADEMCoupler.{h,cpp}
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
# 1. Clone MoPhi
git clone https://github.com/Ruochun/MoPhi.git
cd MoPhi

# 2. Configure (minimal — no external solvers, Python bindings enabled)
cmake -B build

# 3. Build
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

---

## Enabling external solvers

External solvers are downloaded via CMake's `FetchContent` into `external/<Name>/`.
Control which ones are fetched with the following options:

| Option | Default | Effect |
|--------|---------|--------|
| `-DMOPHI_FETCH_TLFEA=ON` | OFF | Download TLFEA FEA solver |
| `-DMOPHI_FETCH_DEMENGINE=ON` | OFF | Download DEM-Engine DEM solver |

Example:
```bash
cmake -B build -DMOPHI_FETCH_TLFEA=ON -DMOPHI_FETCH_DEMENGINE=ON
cmake --build build
```

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
4. Create a wrapper in `src/wrappers/` and integrate it in `src/wrappers/CMakeLists.txt`.

---

## Selecting which co-simulation solvers to build

| Option | Default | Effect |
|--------|---------|--------|
| `-DMOPHI_BUILD_TLFEA_DEM=ON` | ON | Build TLFEA + DEM-Engine coupler |

Each co-simulation solver declares its required externals.  If a required
external has not been fetched, CMake emits a clear error with instructions.

---

## Python bindings

Python bindings are built automatically when `MOPHI_BUILD_PYTHON_BINDINGS=ON`
(the default).  pybind11 is fetched automatically if not already available.

After building, add the `python/` directory to your `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/MoPhi/python:$PYTHONPATH
python3 -c "import mophi; help(mophi)"
```

---

## External solvers

| Name | URL | Role |
|------|-----|------|
| TLFEA | https://github.com/Ruochun/TLFEA | Total-Lagrangian FEA |
| DEM-Engine | https://github.com/projectchrono/DEM-Engine | GPU-based DEM |

---

## License

See [LICENSE](LICENSE).
