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

Keeping one source copy avoids generated copies of tracked Python files and
makes source-tree and wheel imports use the same Python sources. Developer
builds add an ignored compiled extension beside those sources.

The package entry point, `python/mophi/__init__.py`, re-exports the supported
public API. Add new public symbols there or in the appropriate subpackage
`__init__.py`.

## Compiled extension

`mophi_core` is compiled from `python/bindings/*.cpp` plus enabled coupler
sources. Every build keeps its extension under `<build-directory>/python/mophi/`.
Developer default builds publish a copy beside `python/mophi/__init__.py`, so
`PYTHONPATH=python` can load the source package. Publication runs even when no
relink is needed, restoring the selected build's feature set after another
configuration has overwritten the source-tree copy. The explicit
`mophi_python_source` target performs the same publication. Packaging builds
skip publication and install their own extension with the Python package.

The extension is ignored by Git and must not be committed; building does not
modify tracked Python sources. Use a separate build directory for each CPython
minor version and configuration. Configurations for the same Python version
still share the published source-tree filename, so the last publication selects
the active feature set. See [build workflows](how-to/python-builds.md) for switching
configurations and diagnosing missing compiled classes.

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

The solver-independent sampler for grid points bracketed by open triangle
surfaces is documented in [`mesh-sampling.md`](mesh-sampling.md).

The GPU-native Newton–DEME contact exchange is documented in
[`couplers/newton-deme.md`](couplers/newton-deme.md).
The additional pairwise device-transfer helpers are documented in
[`pairwise-gpu-exchange.md`](pairwise-gpu-exchange.md).
