# MoPhi — Agent Coding Guide

This file provides instructions and context for coding agents (AI or human) working on the MoPhi repository.
Read it in full before making any changes.

---

## Repository purpose

MoPhi is a **modular multi-physics co-simulation framework**.
Its job is to connect independent single-physics solvers (FEA, DEM, CFD, …) and let them exchange
data and march forward together as a tightly-coupled simulation.

MoPhi does **not** re-implement any physics.
It only provides the glue (the *couplers*) between solvers that already exist elsewhere.

---

## Directory structure

```
MoPhi/
├── CMakeLists.txt               # Root build file — all CMake options live here
├── AGENTS.md                    # This file
├── README.md                    # User-facing documentation
├── .clang-format                # Formatting rules (Chromium base, 120-col, 4-space indent)
├── .format_all                  # Shell script that applies clang-format to the whole repo
├── .gitmodules                  # Git submodule declarations (MoPhiEssentials)
├── cmake/
│   └── MoPhiExternalUtils.cmake # mophi_register_external / mophi_fetch_external /
│                                #   mophi_require_externals macros
├── external/
│   ├── ExternalProjects.cmake   # One mophi_register_external() call per known external
│   ├── CMakeLists.txt           # Activates registered externals when MOPHI_FETCH_*=ON
│   └── MoPhiEssentials/         # Required git submodule — always present
│                                #   Provides Logger, Real3, and other utilities
├── src/
│   ├── couplers/                # One sub-directory per co-simulation coupler
│   │   ├── CMakeLists.txt       # Dispatches to per-coupler add_subdirectory; creates
│   │   │                        #   the aggregate mophi_couplers INTERFACE target
│   │   ├── tlfea_dem/           # TLFEA + DEM-Engine coupler
│   │   │   ├── CMakeLists.txt
│   │   │   └── TLFEADEMCoupler.{h,cpp}
│   │   └── tlfea_newton/        # TLFEA + Newton coupler
│   │       ├── CMakeLists.txt
│   │       ├── TLFEANewtonCoupler.{h,cpp}
│   │       └── PyTLFEANewtonCoupler.h   # Python-facing wrapper (compiled into mophi_core)
│   └── visualization/           # Backend-agnostic visualization layer (pure Python)
│       ├── README.md            # Design philosophy, API contract, extension guide
│       ├── opengl_visualizer.py # Real-time OpenGL backend (via Newton's ViewerGL)
│       └── omniverse_visualizer.py # Offline USD export backend
├── python/
│   ├── CMakeLists.txt           # pybind11 extension (fetched automatically)
│   ├── bindings/
│   │   └── mophi_bindings.cpp   # pybind11 entry point — register every coupler class here
│   └── mophi/
│       ├── __init__.py          # Python package — re-exports classes with try/except
│       ├── visualizers/         # Visualizer backends (mirrored from src/visualization/)
│       │   ├── __init__.py
│       │   ├── opengl_visualizer.py
│       │   └── omniverse_visualizer.py
│       └── utils/               # Shared helper modules (solver-agnostic utilities)
│           ├── __init__.py
│           ├── vis_utils.py     # Internal math helpers for visualizer backends
│           ├── xlb_helpers.py   # XLB (Lattice-Boltzmann) post-processing utilities
│           └── math_utils.py    # General-purpose math helpers (quaternion ops, etc.)
└── demo/
    ├── CMakeLists.txt           # Guards each sub-demo with if(MOPHI_BUILD_<SOLVER>)
    └── <solver_name>/           # One sub-directory per demo
        ├── CMakeLists.txt
        └── demo_<solver_name>.cpp
```

---

## Design philosophy

### 1. Couplers directly own solver instances

Coupler classes own instances of the participating single-physics solvers **directly**,
using each solver's own public API without an intervening wrapper layer.

`TLFEADEMCoupler::Impl` is the canonical example:

```cpp
struct TLFEADEMCoupler::Impl {
    std::unique_ptr<deme::DEMSolver>  dem;   // DEM-Engine's own class
    std::unique_ptr<tlfea::FEASolver> fea;   // TLFEA's own class
};
```

Do **not** introduce wrapper or adapter classes between MoPhi and the external solver APIs.

### 2. Pimpl idiom for all coupler classes

Every coupler hides its implementation detail behind a `struct Impl` and a
`std::unique_ptr<Impl> impl_` member (the [pimpl idiom][pimpl]).
This keeps solver-specific headers (which may be large and/or generated) out of the
coupler's own public header.

[pimpl]: https://en.cppreference.com/w/cpp/language/pimpl

### 3. Lifecycle: Initialize / Finalize

Every coupler must implement exactly two public lifecycle methods:

| Method | Responsibility |
|--------|---------------|
| `Initialize(...)` | Create solver objects, load configs, allocate GPU/memory resources |
| `Finalize()` | Tear down solvers, release all resources |

The destructor calls `Finalize()` automatically when `initialized` is `true`.

### 4. No whole-sale stepper — per-solver stepping methods

**Never** implement a single `Step()` method that advances all solvers in lock-step.
Instead, expose one stepping method per participating solver so that demos can control
the pace of each system independently (e.g., advance Newton every substep, XLB only
once per policy frame, or DEME at a different rate):

| Method | Responsibility |
|--------|---------------|
| `StepNewton()` | Advance the Newton rigid-body solver by one substep |
| `StepDEME()` | Advance the DEME discrete-element solver by one substep |
| `StepXLB()` | Advance the XLB lattice-Boltzmann solver by one substep |

Name the methods after the solver they advance (`StepNewton`, `StepDEME`, `StepXLB`, …).
Use a placeholder/no-op body guarded by the `<solver>_available` flag when a solver
has not been connected yet.

The Python bindings expose these as `step_newton()`, `step_deme()`, `step_xlb()` etc.
(snake_case).  The demo simulation loop then looks like:

```python
for _ in range(SIM_SUBSTEPS):
    coupler.step_newton()
    coupler.step_deme()
# once per policy frame:
coupler.step_xlb()
```

### 5. Non-copyable, movable coupler classes

Because couplers own resources (GPU contexts, open files, etc.) they must **never** be copied.
Every coupler class must:

```cpp
MyNewCoupler(const MyNewCoupler&)            = delete;
MyNewCoupler& operator=(const MyNewCoupler&) = delete;
MyNewCoupler(MyNewCoupler&&)                 = default;
MyNewCoupler& operator=(MyNewCoupler&&)      = default;
```

### 6. Couplers live in the `mophi` namespace

All C++ classes, functions, and types defined in this repository belong to the `mophi` namespace.
There are no nested namespaces at this time.

---

## MoPhiEssentials submodule

MoPhiEssentials (`external/MoPhiEssentials`) is a **required, always-on git submodule**.
It provides foundational utilities shared across all MoPhi-compatible solvers, including:

- `mophi::Logger` — thread-safe singleton logger with verbosity levels
- `MOPHI_INFO(...)`, `MOPHI_WARNING(...)`, `MOPHI_ERROR(...)` macros — printf-style logging
- `MOPHI_STATUS(identifier, ...)` — keyed status-line messages
- `mophi::Real3<T>` — 3D vector template class (CPU + GPU compatible)
- Memory management helpers, GPU utilities, and more

MoPhiEssentials is integrated as a header-only INTERFACE CMake target (`mophi_essentials`) in
CPU-only mode.  CUDA is explicitly disabled inside MoPhiEssentials when it is built as a submodule
of MoPhi (`MOPHI_ENABLE_CUDA=OFF FORCE`), because MoPhi manages CUDA resources per-coupler through
the `MOPHI_FETCH_*` / `MOPHI_BUILD_*` mechanism.  No separate build step is required.

The `mophi_essentials` target is linked transitively to the `mophi_couplers` aggregate INTERFACE
target, so every coupler automatically has access to `#include <core/Logger.hpp>` (and all other
MoPhiEssentials headers) without any additional `target_link_libraries` calls.

### Does MoPhiEssentials interfere with solvers that also use it internally?

No.  Solvers like TLFEA that carry MoPhiEssentials as their own submodule are built via
`ExternalProject_Add` in a **completely isolated build environment** — they have no CMake target
visibility into MoPhi's parent build.  The two copies of MoPhiEssentials are completely independent
and do not conflict.

### Verbosity defaults

The default verbosity is `VERBOSITY_WARNING` (level 2).  `MOPHI_INFO` messages are stored in the
log but are not printed to stdout unless the caller raises the verbosity:

```cpp
mophi::Logger::GetInstance().SetVerbosity(mophi::VERBOSITY_INFO);
```

---

## Visualization layer

MoPhi provides a backend-agnostic Python visualization layer in
`src/visualization/`.  The full design philosophy, public API contract, and
instructions for adding new backends are documented in
[`src/visualization/README.md`](src/visualization/README.md).

Key rules for agents:

- **No `set_model` method** — solver-specific setup is handled at
  construction time (`OpenGLVisualizer(model)`) or inferred lazily from data
  (`OmniverseVisualizer` reads `state.body_q.shape` on the first
  `log_state` call).
- **`log_state` is the only solver-coupled method** — it receives a physics
  solver state object; all other `log_*` / rendering methods are fully
  solver-agnostic.
- **Both backends share the same public interface** — demos switch backends
  with a single constructor change; the simulation loop body is identical.
- The files in `src/visualization/` are **mirrored** to `python/mophi/visualizers/`
  (they must be kept in sync).

---

## Python file naming conventions

- **Never use a leading underscore for file or module names** (e.g. `vis_utils.py`,
  not `_vis_utils.py`).  Leading-underscore names imply implementation-private
  modules that callers should not import; MoPhi modules may be imported by demos
  and agents and must have discoverable names.
- New utility modules shared across multiple backends belong in
  `python/mophi/utils/`.
- New visualizer backends belong in `python/mophi/visualizers/` and must also
  have a mirrored copy in `src/visualization/`.

---

## Demo writing rules

Use these rules for all demo simulation loops.

### Physics step sizes must be explicit and decoupled from render frame rate

- Define each physics system step size as an explicit user-modifiable constant
  (for example `NEWTON_DT`, `DEME_DT`).
- Keep rendering cadence separate (for example `RENDER_FPS`, `FRAME_DT`).
- Derive collaboration loop counts (`SIM_SUBSTEPS`, `DEME_SUBSTEPS`, etc.) from
  those explicit step sizes and `FRAME_DT`; do not derive physics dt values from
  frame rate.

### Demo comments should explain intent, not frozen numeric choices

- In demo comments, explain the purpose of a variable, action, or design choice.
- Avoid encoding specific tuned values in comments when those values are expected
  to change frequently (domain size, particle size, speeds, camera offsets, etc.).
- Keep comments resilient to parameter tuning so they usually remain valid when
  constants are updated.

---

## C++ coding conventions

| Item | Rule |
|------|------|
| Standard | C++17 (`cxx_std_17`) |
| Header guard | `#pragma once` — never `#ifndef / #define` guards |
| Includes | System / third-party headers before project headers |
| Doc comments | `///` Doxygen-style for public API; `//` for implementation notes |
| Line length | 120 characters (enforced by `.clang-format`) |
| Indentation | 4 spaces (enforced by `.clang-format`) |
| Braces | Chromium style (`BasedOnStyle: Chromium` in `.clang-format`) |
| Logging | Use `MOPHI_INFO(...)`, `MOPHI_WARNING(...)`, `MOPHI_ERROR(...)` from `<core/Logger.hpp>` — do **not** use `std::cout` directly |
| Method naming | Public methods use **PascalCase** (e.g. `Initialize`, `Step`, `Finalize`); private methods use **snake_case** (e.g. `build_mesh`) |

### Code formatting

**All** C++ source files (`.cpp`, `.c`, `.h`, `.hpp`, `.cu`, `.cuh`, `.cc`) must be formatted with
`clang-format -style=file` before every commit.
Run the provided script from the repository root:

```bash
bash .format_all
```

The script skips `external/` and `build*/` directories (third-party code).
There is no exception — even single-line changes must be formatted.

---

## CMake conventions

### Options

Every user-visible CMake option is declared with `option()` in the **root** `CMakeLists.txt`.
There are no options in sub-directory `CMakeLists.txt` files.

Options follow two naming groups:

| Group | Prefix | Purpose |
|-------|--------|---------|
| Fetch externals | `MOPHI_FETCH_<NAME>` | Download / update a specific external solver |
| Build solvers | `MOPHI_BUILD_<NAME>` | Compile a specific co-simulation coupler |
| Build extras | `MOPHI_BUILD_PYTHON_BINDINGS`, `MOPHI_BUILD_DEMOS` | Optional build outputs |

### Section banners

Every `CMakeLists.txt` uses `─` banners to delimit logical sections:

```cmake
# ─────────────────────────────────────────────────────────────────────────────
# Section title
# ─────────────────────────────────────────────────────────────────────────────
```

### Optional prerequisites (CUDA, Eigen, …)

Additional tools such as CUDA Toolkit or Eigen3 are searched **once** at the top of the root
`CMakeLists.txt` using `find_package(... QUIET)` so that the result cache variable is available to
all sub-directories:

```cmake
find_package(CUDAToolkit QUIET)
find_package(Eigen3      QUIET)
```

The **QUIET** keyword is mandatory.
A missing CUDA or Eigen installation must **not** cause an error here — the general build must
succeed for users who do not need CUDA or Eigen.

`FATAL_ERROR` messages for missing prerequisites are emitted **only** inside the
`if(MOPHI_BUILD_<SOLVER>)` block in `src/couplers/CMakeLists.txt` that actually requires them.
For example:

```cmake
if(MOPHI_BUILD_TLFEA_DEM)
    if(NOT CUDAToolkit_FOUND)
        message(FATAL_ERROR "MoPhi: Building TLFEADEMCoupler requires the CUDA Toolkit ...")
    endif()
    if(NOT Eigen3_FOUND)
        message(FATAL_ERROR "MoPhi: Building TLFEADEMCoupler requires Eigen3 ...")
    endif()
    ...
endif()
```

**Summary rule**: `find_package(... QUIET)` at the root; `FATAL_ERROR` guarded by the specific
`MOPHI_BUILD_*` option that actually needs the package.

---

## External solver management

### Three-step workflow

1. **Register** — add one `mophi_register_external()` call in `external/ExternalProjects.cmake`.
2. **Fetch** — add an `if(MOPHI_FETCH_<NAME>) mophi_fetch_external(<NAME>) endif()` block in
   `external/CMakeLists.txt`.
3. **Require** — call `mophi_require_externals(<CouplerName> <dep1> [<dep2> ...])` at the top of
   the coupler's `if(MOPHI_BUILD_<NAME>)` block.

### Macro reference

```cmake
mophi_register_external(
    NAME           <name>        # identifier: used for options, dir names, targets
    URL            <url>         # git repository URL
    TAG            <tag>         # branch, tag, or commit SHA
    [LIB_NAME      <lib>]        # static library stem (no "lib" prefix or ".a" suffix)
    [HEADER_SRC    <subdir>]     # copy <subdir>/ → install/include/ when the external
                                 #   has no CMake install rules for headers
    [EXTRA_LIB_NAMES <lib>...]   # additional shared libraries installed by the external
    [EXTRA_CMAKE_ARGS <arg>...]  # extra -D flags forwarded to the external's cmake step
)
```

After `mophi_fetch_external(<NAME>)` succeeds, the IMPORTED target `mophi_ext_<NAME>` is
available for linking.
It carries `INTERFACE_INCLUDE_DIRECTORIES` pointing to the external's installed headers.
When `LIB_NAME` is given it also carries `IMPORTED_LOCATION` for the static library.

### ExternalProject_Add vs FetchContent vs git submodule

| Use case | Tool |
|----------|------|
| Third-party solvers (TLFEA, DEM-Engine, …) | `ExternalProject_Add` via `mophi_fetch_external` |
| Pure build helpers with no generated headers (pybind11) | `FetchContent` |
| Required always-on utilities shared across all solvers (MoPhiEssentials) | git submodule + `add_subdirectory` |

`ExternalProject_Add` builds the external in isolation and guarantees that generated headers
(e.g., `core/ApiVersion.h` from DEM-Engine) exist before any MoPhi source file is compiled.

---

## Adding a new co-simulation solver

Follow these steps exactly; use `TLFEADEMCoupler` as the reference implementation.

### 1. Register and fetch the external solvers

In `external/ExternalProjects.cmake`:

```cmake
mophi_register_external(
    NAME     MySolver
    URL      "https://github.com/org/MySolver"
    TAG      "main"
    LIB_NAME "mysolverlib"
)
```

In `external/CMakeLists.txt`:

```cmake
if(MOPHI_FETCH_MYSOLVER)
    mophi_fetch_external(MySolver)
endif()
```

In the root `CMakeLists.txt` (Options section):

```cmake
option(MOPHI_FETCH_MYSOLVER  "Download/update MySolver external project"   OFF)
option(MOPHI_BUILD_MY_COUPLER "Build the MySolver co-simulation coupler"   OFF)
```

If the new solver requires a system package (CUDA, Eigen, OpenCL, …), add the corresponding
`find_package(... QUIET)` call in the root `CMakeLists.txt` at the prerequisites section.

### 2. Write the coupler header and source

Create a new directory `src/couplers/<name>/` and add these files:

`src/couplers/<name>/MyCoupler.h` — follow the structure of `TLFEADEMCoupler.h`:

- `#pragma once`
- Doxygen `///` class comment that names the external solvers owned and the pimpl members
- Public lifecycle methods: `Initialize(...)`, `Step()`, `Finalize()`
- Private `struct Impl; std::unique_ptr<Impl> impl_;`
- Delete copy, default move

`src/couplers/<name>/MyCoupler.cpp`:

- Include the coupler header first, then `<core/Logger.hpp>`, then external solver headers
- Implement `Impl` with `std::unique_ptr<ExternalSolverClass>` members
- Use `MOPHI_INFO(...)`, `MOPHI_WARNING(...)`, `MOPHI_ERROR(...)` for all output — do **not** use `std::cout` directly
- Destructor calls `Finalize()` when `impl_->initialized` is `true`

If the new solver requires a Python-side component (like Newton), use a **Python-only coupler** pattern instead of the C++ coupler pattern above — see the next section.

After writing source files, **run `.format_all`** before committing.

### 2b. Python-only co-simulation coupler (one solver has no C++ ABI)

When one of the two solvers is a pure Python package (no C++ ABI), there is no
need for a separate C++ coupler class.  Instead, write a single
`PyMyCoupler.h` + `PyMyCoupler.cpp` pair directly in `src/couplers/<name>/`:

`src/couplers/<name>/PyMyCoupler.h` — declares the struct with pimpl:

- `#pragma once`; include only `<pybind11/pybind11.h>` and standard library headers
- `struct TLFEAImpl;` forward-declaration for the pimpl (keeps C++-solver headers out)
- `std::unique_ptr<TLFEAImpl> fea_;` owns the C++ solver via pimpl
- `pybind11::object` members for the Python solver objects
- All lifecycle methods declared (not defined inline): `Initialize(...)`, `Finalize()`
- Per-solver stepping methods declared: `StepNewton()`, `StepDEME()`, `StepXLB()`, … (one per participating solver)
- Coupling data-exchange methods: e.g. `GetNodePositions()`, `SetNodeForces()`
- Delete copy, **no** default move (pybind11 objects inhibit trivial move)

`src/couplers/<name>/PyMyCoupler.cpp` — defines pimpl and implements all methods:

- Include `"PyMyCoupler.h"` first, then `<core/Logger.hpp>`, then C++ solver headers
- `struct PyMyCoupler::TLFEAImpl { std::unique_ptr<CSolverClass> solver; bool initialized{false}; ... };`
- Constructor initializes pimpl and all `pybind11::none()` objects
- Destructor calls `Finalize()` when `fea_->initialized` is `true`
- Use `MOPHI_INFO(...)`, `MOPHI_WARNING(...)`, `MOPHI_ERROR(...)` — never `std::cout`

This `.cpp` is compiled **directly into `mophi_core`** (not as a separate static
library) via `target_sources` in `python/CMakeLists.txt`.

After writing source files, **run `.format_all`** before committing.

### 3. Register the coupler in the build system

Create `src/couplers/<name>/CMakeLists.txt`:

- **C++ coupler** (see `tlfea_dem/CMakeLists.txt`): build a `STATIC` library from
  `MyCoupler.cpp`, link it `PUBLIC` against the required externals, and add it to
  `mophi_couplers INTERFACE`.
- **Python-only coupler** (see `tlfea_newton/CMakeLists.txt`): build an `INTERFACE`
  library (no object files — just carries the link and include requirements), add it
  to `mophi_couplers INTERFACE`.  The `.cpp` implementation is compiled into
  `mophi_core` by `python/CMakeLists.txt`.

In both cases, call `mophi_require_externals()` and emit `FATAL_ERROR` messages for
missing system dependencies (CUDA, Eigen, …) at the top of the file.

In `src/couplers/CMakeLists.txt`, add an `add_subdirectory` call:

```cmake
if(MOPHI_BUILD_MY_COUPLER)
    add_subdirectory(<name>)
endif()
```

### 4. Expose the coupler in Python bindings

In `python/bindings/mophi_bindings.cpp`, include the coupler header:

```cpp
#ifdef MOPHI_HAS_MY_COUPLER
    #include "MyCoupler.h"       // C++ coupler  — OR —
    #include "PyMyCoupler.h"     // Python-only coupler
#endif

// Inside PYBIND11_MODULE:
#ifdef MOPHI_HAS_MY_COUPLER
    // C++ coupler:
    py::class_<mophi::MyCoupler>(m, "MyCoupler", "…docstring…")
        .def(py::init<>())
        .def("initialize", &mophi::MyCoupler::Initialize, /* py::arg ... */)
        .def("step",       &mophi::MyCoupler::Step)
        .def("finalize",   &mophi::MyCoupler::Finalize);

    // Python-only coupler (PyMyCoupler lives in global namespace):
    py::class_<PyMyCoupler>(m, "MyCoupler", "…docstring…")
        .def(py::init<>())
        .def("initialize", &PyMyCoupler::Initialize, /* py::arg ... */)
        .def("step",       &PyMyCoupler::Step)
        .def("finalize",   &PyMyCoupler::Finalize);
#endif
```

In `python/CMakeLists.txt`:

- **C++ coupler**: add `target_include_directories` + `target_compile_definitions`.
- **Python-only coupler**: add the `.cpp` source and compile definition (the include
  directory is already propagated via the INTERFACE CMake target):

```cmake
if(MOPHI_BUILD_MY_COUPLER)
    # C++ coupler:
    target_include_directories(mophi_core PRIVATE "${CMAKE_SOURCE_DIR}/src/couplers/<name>")
    target_compile_definitions(mophi_core PRIVATE MOPHI_HAS_MY_COUPLER)

    # Python-only coupler (no explicit include_directories — INTERFACE target propagates them):
    target_sources(mophi_core PRIVATE "${CMAKE_SOURCE_DIR}/src/couplers/<name>/PyMyCoupler.cpp")
    target_compile_definitions(mophi_core PRIVATE MOPHI_HAS_MY_COUPLER)
endif()
```

In `python/mophi/__init__.py`:

```python
try:
    from .mophi_core import MyCoupler  # noqa: F401
    __all__.append("MyCoupler")
except ImportError:
    pass
```

### 5. Add a demo

Create `demo/<solver_name>/`:

- `CMakeLists.txt` — add an executable, link against `mophi_couplers`
- `demo_<solver_name>.cpp` — minimal end-to-end usage example

In `demo/CMakeLists.txt`:

```cmake
if(MOPHI_BUILD_MY_COUPLER)
    add_subdirectory(<solver_name>)
endif()
```

### 6. Update documentation

Update `README.md` to document:

- The new `MOPHI_FETCH_*` and `MOPHI_BUILD_*` CMake options
- The new external solvers in the "External solvers" table
- Any new prerequisites in the "Prerequisites" table

If the new coupler introduces a new visualization backend, update
`src/visualization/README.md` instead of adding visualization detail to the
main README.

---

## Quick reference: checklist for a new coupler

- [ ] `mophi_register_external()` entry in `external/ExternalProjects.cmake`
- [ ] `if(MOPHI_FETCH_*) mophi_fetch_external() endif()` in `external/CMakeLists.txt`
- [ ] `option(MOPHI_FETCH_*)` and `option(MOPHI_BUILD_*)` in root `CMakeLists.txt`
- [ ] `find_package(... QUIET)` for any new system deps in root `CMakeLists.txt`
- [ ] Create `src/couplers/<name>/` directory
- [ ] `<NewCoupler>.h` — `#pragma once`, pimpl, `Initialize`/`Finalize` lifecycle methods, per-solver step methods, delete copy / default move *(C++ coupler)*
- [ ] `<NewCoupler>.cpp` — `Impl` owns solver instances; lifecycle and per-solver step methods implemented *(C++ coupler)*
- [ ] `PyNewCoupler.h` — declared (not inline) with pimpl `struct TLFEAImpl` + `pybind11::object` members; per-solver step methods declared *(Python-only coupler)*
- [ ] `PyNewCoupler.cpp` — defines pimpl, implements all methods including per-solver step methods *(Python-only coupler)*
- [ ] `src/couplers/<name>/CMakeLists.txt` — `mophi_require_externals()`, FATAL_ERROR guards; **STATIC** lib for C++ coupler or **INTERFACE** lib for Python-only coupler
- [ ] `if(MOPHI_BUILD_*) add_subdirectory(<name>) endif()` in `src/couplers/CMakeLists.txt`
- [ ] `#ifdef MOPHI_HAS_*` binding block in `python/bindings/mophi_bindings.cpp`
- [ ] `target_compile_definitions` + `target_include_directories` (C++ coupler) **or** `target_sources` + `target_compile_definitions` (Python-only coupler) in `python/CMakeLists.txt`
- [ ] `try/except ImportError` in `python/mophi/__init__.py`
- [ ] Demo executable under `demo/<name>/`
- [ ] `README.md` updated
- [ ] `bash .format_all` run and all source files committed formatted
