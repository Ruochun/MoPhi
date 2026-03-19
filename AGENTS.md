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
├── cmake/
│   └── MoPhiExternalUtils.cmake # mophi_register_external / mophi_fetch_external /
│                                #   mophi_require_externals macros
├── external/
│   ├── ExternalProjects.cmake   # One mophi_register_external() call per known external
│   └── CMakeLists.txt           # Activates registered externals when MOPHI_FETCH_*=ON
├── src/
│   └── couplers/                # One .h + .cpp per co-simulation solver
│       ├── CMakeLists.txt       # Builds mophi_coupler_* libraries; adds them to
│       │                        #   the aggregate mophi_couplers INTERFACE target
│       └── TLFEADEMCoupler.{h,cpp}
├── python/
│   ├── CMakeLists.txt           # pybind11 extension (fetched automatically)
│   ├── bindings/
│   │   └── mophi_bindings.cpp   # pybind11 entry point — register every coupler class here
│   └── mophi/
│       └── __init__.py          # Python package — re-exports classes with try/except
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

### 3. Lifecycle: initialize / step / finalize

Every coupler must implement exactly three public lifecycle methods:

| Method | Responsibility |
|--------|---------------|
| `initialize(...)` | Create solver objects, load configs, allocate GPU/memory resources |
| `step()` | Advance both solvers by one co-simulation time step and exchange coupling data |
| `finalize()` | Tear down solvers, release all resources |

The destructor calls `finalize()` automatically when `initialized` is `true`.

### 4. Non-copyable, movable coupler classes

Because couplers own resources (GPU contexts, open files, etc.) they must **never** be copied.
Every coupler class must:

```cpp
MyNewCoupler(const MyNewCoupler&)            = delete;
MyNewCoupler& operator=(const MyNewCoupler&) = delete;
MyNewCoupler(MyNewCoupler&&)                 = default;
MyNewCoupler& operator=(MyNewCoupler&&)      = default;
```

### 5. Couplers live in the `mophi` namespace

All C++ classes, functions, and types defined in this repository belong to the `mophi` namespace.
There are no nested namespaces at this time.

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
| Output | Use `std::cout` prefixed with `[MoPhi] ClassName:` for status messages |

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

### ExternalProject_Add vs FetchContent

| Use case | Tool |
|----------|------|
| Third-party solvers (TLFEA, DEM-Engine, …) | `ExternalProject_Add` via `mophi_fetch_external` |
| Pure build helpers with no generated headers (pybind11) | `FetchContent` |

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

`src/couplers/MyCoupler.h` — follow the structure of `TLFEADEMCoupler.h`:

- `#pragma once`
- Doxygen `///` class comment that names the external solvers owned and the pimpl members
- Public lifecycle methods: `initialize(...)`, `step()`, `finalize()`
- Private `struct Impl; std::unique_ptr<Impl> impl_;`
- Delete copy, default move

`src/couplers/MyCoupler.cpp`:

- Include the coupler header first, then system headers, then external solver headers
- Implement `Impl` with `std::unique_ptr<ExternalSolverClass>` members
- Use `std::cout << "[MoPhi] MyCoupler: ..."` for status messages
- Destructor calls `finalize()` when `impl_->initialized` is `true`

After writing source files, **run `.format_all`** before committing.

### 3. Register the coupler in the build system

In `src/couplers/CMakeLists.txt`, add an `if(MOPHI_BUILD_MY_COUPLER)` block:

```cmake
if(MOPHI_BUILD_MY_COUPLER)
    mophi_require_externals(MyCoupler MySolver)

    # Guard any system-level dependencies required only by this coupler:
    if(NOT SomePkg_FOUND)
        message(FATAL_ERROR "MoPhi: Building MyCoupler requires SomePkg ...")
    endif()

    add_library(mophi_coupler_my STATIC MyCoupler.cpp)
    set_target_properties(mophi_coupler_my PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_features(mophi_coupler_my PUBLIC cxx_std_17)
    target_include_directories(mophi_coupler_my
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:include/mophi/couplers>
    )
    target_link_libraries(mophi_coupler_my
        PUBLIC
            mophi_ext_MySolver
            # … additional system library IMPORTED targets …
    )
    target_link_libraries(mophi_couplers INTERFACE mophi_coupler_my)
endif()
```

Use `PUBLIC` (not `PRIVATE`) linkage on the static coupler library so that the link dependencies
propagate to the Python extension module and demo executables.

### 4. Expose the coupler in Python bindings

In `python/bindings/mophi_bindings.cpp`:

```cpp
#ifdef MOPHI_HAS_MY_COUPLER
    #include "MyCoupler.h"
#endif

// Inside PYBIND11_MODULE:
#ifdef MOPHI_HAS_MY_COUPLER
    py::class_<mophi::MyCoupler>(m, "MyCoupler", "…docstring…")
        .def(py::init<>())
        .def("initialize", &mophi::MyCoupler::initialize, /* py::arg ... */)
        .def("step",       &mophi::MyCoupler::step)
        .def("finalize",   &mophi::MyCoupler::finalize);
#endif
```

In `python/CMakeLists.txt`, add the compile definition:

```cmake
if(MOPHI_BUILD_MY_COUPLER)
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

---

## Quick reference: checklist for a new coupler

- [ ] `mophi_register_external()` entry in `external/ExternalProjects.cmake`
- [ ] `if(MOPHI_FETCH_*) mophi_fetch_external() endif()` in `external/CMakeLists.txt`
- [ ] `option(MOPHI_FETCH_*)` and `option(MOPHI_BUILD_*)` in root `CMakeLists.txt`
- [ ] `find_package(... QUIET)` for any new system deps in root `CMakeLists.txt`
- [ ] `<NewCoupler>.h` — `#pragma once`, pimpl, lifecycle methods, delete copy / default move
- [ ] `<NewCoupler>.cpp` — `Impl` owns solver instances; lifecycle methods implemented
- [ ] `if(MOPHI_BUILD_*)` block in `src/couplers/CMakeLists.txt` with FATAL_ERROR guards for missing system deps
- [ ] `#ifdef MOPHI_HAS_*` binding block in `python/bindings/mophi_bindings.cpp`
- [ ] `target_compile_definitions` for `MOPHI_HAS_*` in `python/CMakeLists.txt`
- [ ] `try/except ImportError` in `python/mophi/__init__.py`
- [ ] Demo executable under `demo/<name>/`
- [ ] `README.md` updated
- [ ] `bash .format_all` run and all source files committed formatted
