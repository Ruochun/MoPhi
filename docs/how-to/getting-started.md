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

# 2. Configure with the desired co-simulation coupler and active Python
cmake -B build-py312 \
      -DPython3_EXECUTABLE="$(python3 -c 'import sys; print(sys.executable)')" \
      -DMOPHI_FETCH_FERIS=ON \
      -DMOPHI_FETCH_NEWTON=ON \
      -DMOPHI_BUILD_FERIS_NEWTON=ON
cmake --build build-py312

# 3. Run a demo
PYTHONPATH=python python3 demo/feris_newton/demo_feris_newton.py
```
