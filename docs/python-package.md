# MoPhi Python package architecture

This document explains and justifies the layout and build behavior of the code
under `python/mophi/` and the pybind11 extension configured by
`python/CMakeLists.txt`.

## Canonical source layout

`python/mophi/` is the single canonical location for all installable Python
code. Utility modules live in `python/mophi/utils/`, visualization backends live
in `python/mophi/visualizers/`, and Python-only couplers live in a solver-specific
subpackage under `python/mophi/couplers/`. Edit these files directly; they are
not generated or mirrored from `src/`.

Keeping one source copy prevents a CMake configure from modifying the Git
working tree, eliminates synchronization failures, and makes source-tree and
wheel imports use the same Python files.

The package entry point, `python/mophi/__init__.py`, re-exports the supported
public API. Add new public symbols there or in the appropriate subpackage
`__init__.py`.

## Compiled extension

`mophi_core` is compiled from `python/bindings/*.cpp`. Developer builds place
the interpreter-specific `mophi_core*.so` beside `python/mophi/__init__.py`, so
`PYTHONPATH=python` can load the source package. Packaging builds keep the
extension in the build tree and install it together with the canonical Python
package.

The extension is a build artifact and must not be committed. A separate build
directory is required for each supported CPython minor version because extension
modules are ABI-specific.

## Adding Python code

- Put solver-independent helpers in `python/mophi/utils/`.
- Put visualization backends in `python/mophi/visualizers/`.
- Put reusable Python-only solver coupling in
  `python/mophi/couplers/<coupling_type>/`.
- Keep scenario-specific orchestration in demos rather than generic utilities.
- Add tests that import the module through its installed package path.
- Do not add a duplicate source file under `src/`.

The G-code utility has its design, supported command subset, usage examples,
and test instructions documented in [`gcode.md`](gcode.md).

The dependency-free triangle snapshot writer for ParaView is documented in
[`vtk-output.md`](vtk-output.md).

The GPU-native Newton–DEME contact exchange is documented in
[`newton-deme-coupling.md`](newton-deme-coupling.md).
The additional pairwise device-transfer helpers are documented in
[`pairwise-gpu-exchange.md`](pairwise-gpu-exchange.md).
