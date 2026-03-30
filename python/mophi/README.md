# python/mophi — package directory

This directory is the installable Python package for MoPhi.
It contains three kinds of files:

---

## 1. Hand-authored package files

| File | Purpose |
|------|---------|
| `__init__.py` | Package entry point — imports and re-exports everything under the `mophi` namespace. |

Edit these files directly.

---

## 2. CMake-generated C++ extension module

| File | Origin |
|------|--------|
| `mophi_core*.so` | Compiled by CMake from `python/bindings/*.cpp` (pybind11). |

Do **not** commit this file — it is a build artefact.

---

## 3. Mirrored Python sources (do not edit here)

The following `.py` files are **automatically copied** from their canonical
locations in the `src/` tree by CMake at configure time
(`cmake -B build ...`).  They are committed to the repository only so that
`import mophi` works directly from an un-built source tree (e.g. when running
a demo without a prior CMake configure step).

> **Rule: always edit the source file in `src/`, never the copy here.**
> After editing the source, either re-run CMake configure or copy the file
> manually to keep the two in sync.  The CMake copy step will always win on
> the next configure.

| File in `python/mophi/` | Canonical source | CMake rule |
|-------------------------|------------------|------------|
| `opengl_visualizer.py` | `src/visualization/opengl_visualizer.py` | `src/visualization/CMakeLists.txt` |
| `omniverse_visualizer.py` | `src/visualization/omniverse_visualizer.py` | `src/visualization/CMakeLists.txt` |
| `xlb_helpers.py` | `src/utils/xlb_helpers.py` | `src/utils/CMakeLists.txt` |

### How the copy works

Each source `CMakeLists.txt` uses CMake's `configure_file(... COPYONLY)` to
reproduce the file verbatim into this directory:

```cmake
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/xlb_helpers.py"
    "${CMAKE_SOURCE_DIR}/python/mophi/xlb_helpers.py"
    COPYONLY
)
```

`configure_file` re-runs whenever the source file timestamp changes, so after
editing a file in `src/utils/` or `src/visualization/` a simple
`cmake --build build` is enough to refresh the copy here.

### Adding a new mirrored module

1. Place the new `.py` file under `src/utils/` (solver-agnostic helpers) or
   `src/visualization/` (visualizer backends).
2. Add a `configure_file(... COPYONLY)` entry in the corresponding
   `CMakeLists.txt`.
3. Add a `try/except ImportError` import block for any new public symbols in
   `python/mophi/__init__.py`.
4. Run CMake configure once — the file will appear here automatically.
5. Commit both the source file and the resulting copy.
