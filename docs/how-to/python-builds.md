# Python build workflows

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

A source build needs CMake, a C++ compiler, Python development headers, and
initialized submodules. FERIS additionally needs its native prerequisites.
Configure checks for the compiled FERIS/Newton and three-way couplers import
Newton, Warp, and MuJoCo, so install those before configuring; the common
[dependency command](python-dependencies.md) is the convenient complete set.
The base extension alone does not require every runtime solver package to build.

`MOPHI_FETCH_*` obtains external sources or installs selected solver packages.
`MOPHI_BUILD_FERIS_NEWTON` and `MOPHI_BUILD_NEWTON_XLB_DEM` compile coupler classes.
`MOPHI_BUILD_NEWTON_DEM` enables Python demo targets and installation under
`MOPHI_BUILD_DEMOS`; it does not gate Python imports or compile another coupler.
See the [matrix](../how-to.md#demo-build-requirements) for each demo's requirements.

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

A non-packaging developer build keeps its compiled extension inside its build
directory and publishes a copy beside the source package on each default build.
Verify both the selected interpreter and loaded ABI:

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

### Base extension only

For demos that need no compiled solver coupler, run from a checkout with
submodules initialized. Install the runtime dependencies, then build the base
extension for your active interpreter (the directory name below assumes Python 3.12):

```bash
python scripts/setup/install_dependencies.py
cmake -S . -B build-base-py312 \
    -DPython3_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
    -DMOPHI_BUILD_PYTHON_BINDINGS=ON \
    -DMOPHI_BUILD_FERIS_NEWTON=OFF \
    -DMOPHI_BUILD_NEWTON_XLB_DEM=OFF
cmake --build build-base-py312
PYTHONPATH=python python -c 'import mophi; print(mophi.mophi_core.__file__)'
```

Then return to your demo guide for assets, extras, and run commands. An installed
wheel already supplies the extension; omit `PYTHONPATH=python` when using it.
Developer builds for the same Python version share the extension under
`python/mophi/`; this base build can replace one containing compiled couplers.
If you also run demos needing a compiled coupler, use that coupler's setup instead;
its extension includes the base functionality. To switch back to an existing
configuration, run `cmake --build <its-build-directory>`; this republishes that
build's extension even when no compilation is necessary. The explicit
`mophi_python_source` target also republishes it. Building only `mophi_core`
publishes after a relink, but does not republish an already up-to-date target.

If a compiled class is missing despite its build flag being enabled, inspect
`mophi.mophi_core.__file__` using the same Python as the demo. Check both
`Python3_EXECUTABLE` and `Python_EXECUTABLE` in `CMakeCache.txt`: older builds
could retain different interpreters. Reconfigure with the intended
`-DPython3_EXECUTABLE=...`; it now also updates the bindings' interpreter selector.
Then run the default build and restart any Python process that already imported
MoPhi. Installed-wheel users should omit the source `PYTHONPATH` setting.

### Linux Python 3.11–3.14 wheels

MoPhi has a standard PEP 517 packaging entry point in `pyproject.toml`
using `scikit-build-core`. The supported packaged distribution path currently
targets **Linux + CPython 3.11 through 3.14** and enables:

- Python bindings: **ON**
- Newton + XLB + DEME coupler: **ON**
- FERIS coupler: **OFF**
- demos: **OFF**

The canonical runtime requirements and optional demo extras are declared in
[`pyproject.toml`](../../pyproject.toml). See [Python dependencies](python-dependencies.md)
for the one-command installer. The packaging profile above is fixed: CMake
options currently do not produce arbitrary subset wheels. Supporting those
would also require matching dependency metadata and exported Python APIs.

#### Build one wheel for the active interpreter

Build from a checkout with submodules initialized. Replace `python3.12` with
the desired interpreter:

```bash
git clone --recurse-submodules https://github.com/Ruochun/MoPhi.git
cd MoPhi
python3.12 -m pip install --upgrade build
python3.12 -m build --wheel
```

`python -m build` invokes scikit-build-core, which configures CMake internally
with `MOPHI_PYTHON_PACKAGING_BUILD=ON`. Build isolation installs the declared
Python build backend; you still need CMake/compiler prerequisites and submodules.
This profile skips solver runtime import checks, so the full pip runtime set
is not required beforehand merely to build a wheel. Install it to run or test
the resulting package. This build does not use or update the developer
extension under `python/mophi/`.

For Linux distribution, repair this single wheel so bundled native libraries
satisfy manylinux policy:

```bash
python3.12 -m pip install --upgrade auditwheel
auditwheel repair dist/mophi-*.whl -w wheelhouse/
```

Test the repaired artifact in a clean environment without `PYTHONPATH=python`,
which would otherwise override the installed wheel with the source checkout:

```bash
python3.12 -m venv /tmp/mophi-wheel-test
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
`mujoco`, `deme[cuda12]`, `xlb[cuda]`, `torch`, `GitPython`, `PyYAML`, `pycollada`,
`mujoco_warp`, `pyglet`, `imageio`, and `imageio-ffmpeg`) from either the
internet or a local wheelhouse.

For an offline installation, pre-download the MoPhi wheel plus all required
dependency wheels, then install from a local wheelhouse:

```bash
python -m pip install --no-index --find-links /path/to/wheelhouse mophi
```

---
