# Getting started

All demos write generated files by default under
`<repository-root>/output/<demo-name>/`. The repository-level `output/`
directory is git-ignored so movies, USD scenes, state snapshots, and other
generated artifacts do not appear in normal staging operations.

Visual demos include a persistent world reference: RGB segments show the
positive X, Y, and Z directions, and a yellow endpoint-marked bar shows one
metre of world-space length.

---

## Clone, build, and run

The examples use Python 3.12. Activate a Python 3.12 environment before
running commands that use `python`, `python3`, or `pip`.

```bash
# 1. Clone MoPhi (--recurse-submodules is required for MoPhiEssentials)
git clone --recurse-submodules https://github.com/Ruochun/MoPhi.git
cd MoPhi

# If you already cloned without --recurse-submodules, initialise the submodule manually:
#   git submodule update --init

# 2. Install the common Python runtime dependencies (does not build MoPhi)
python scripts/setup/install_dependencies.py

# 3. Configure with the desired co-simulation coupler and active Python
cmake -B build-py312 \
      -DPython3_EXECUTABLE="$(python3 -c 'import sys; print(sys.executable)')" \
      -DMOPHI_FETCH_FERIS=ON \
      -DMOPHI_FETCH_NEWTON=ON \
      -DMOPHI_BUILD_FERIS_NEWTON=ON
cmake --build build-py312

# 4. Run a demo
PYTHONPATH=python python3 demo/feris_newton/demo_feris_newton.py
```

## Choose the build for your demo

Use the [demo build matrix](../how-to.md#demo-build-requirements) to select
coupler flags. Even pure-Python demos importing `mophi` currently need its small
native `mophi_core` extension, built for the active interpreter or supplied by
a wheel. They do not all need a compiled solver coupler.

The [dependency installer](python-dependencies.md) supports demo extras and a
`--dry-run` preview. For GRAB, use `--extra grab` and follow [grab.md](grab.md).
For a distribution build, follow [Python build workflows](python-builds.md).

## Demo regression runner

For a convenient end-to-end check, the repository includes a runner for the
actively developed demos linked from the [how-to index](../how-to.md). It uses the current Python interpreter,
adds the source-tree `python/` directory to `PYTHONPATH`, and waits for each
demo to finish before starting the next. Install and build the prerequisites
from each selected demo guide first (see the
[build matrix](../how-to.md#demo-build-requirements)); interactive viewers must be closed for the runner to
advance.

```bash
python3 scripts/demos/run_important_demos.py
```

The FERIS/Newton demo is excluded by default while that module remains less
developed. Add it explicitly with `--include-feris`. By default the runner
stops at the first failure; use `--keep-going` to exercise the remaining demos
and receive a combined failure summary. Use `--list` to inspect the selected
commands without running them.
