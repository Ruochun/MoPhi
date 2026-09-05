# MoPhi

**MoPhi** is a modular multi-physics co-simulation framework.  It provides the
glue between independent single-physics solvers, letting them exchange data and
march forward together as a tightly-coupled simulation.

MoPhi is aimed at users who want to build cross-solver workflows without
rewriting the participating solvers into one monolithic codebase. Its
coupler-first design extracts tested, reusable on-device data exchange from
scenario-specific demos, making multi-physics experiments and extensible
simulation pipelines easier to assemble as the project grows.

---

## Repository structure

See [docs/design.md#repository-structure](docs/design.md#repository-structure)
for the repository map and code-organization notes.

---

## Prerequisites

| Tool | Minimum version |
|------|----------------|
| CMake | 3.18 |
| C++ compiler | C++17 (GCC 9 / Clang 10 / MSVC 2019) |
| Python (optional) | 3.11–3.14 — for the packaged Linux wheel path |
| Git | any recent version — for fetching externals |

---

## Quick start

Start with the quick-start workflow in
[docs/how-to.md#quick-start](docs/how-to.md#quick-start), which covers cloning,
configuring, building, and running a first demo.

For Python development and distribution, see
[docs/how-to.md#python-build-workflows](docs/how-to.md#python-build-workflows).
It distinguishes per-interpreter source-tree CMake builds from single-wheel
and `cibuildwheel` matrix builds, including ABI troubleshooting and verification.

For the solver-independent G-code parser used by printing workflows, see
[docs/gcode.md](docs/gcode.md).

Newton-backed demos use validated, self-repairing asset downloads described in
[docs/newton-assets.md](docs/newton-assets.md).

The reusable GPU-native Newton–DEME contact-wrench exchange is described in
[docs/newton-deme-coupling.md](docs/newton-deme-coupling.md).

The reusable Newton–XLB, Newton–DEME particle, and XLB–DEME GPU exchange
boundaries are described in
[docs/pairwise-gpu-exchange.md](docs/pairwise-gpu-exchange.md).

### Linux Python 3.11–3.14 wheels (Newton + XLB + DEME, no FERIS)

For the packaged distribution path, see
[docs/how-to.md#linux-python-311314-wheels](docs/how-to.md#linux-python-311314-wheels).
That guide covers wheel building, `auditwheel` repair, and the default runtime
dependency set for Newton + XLB + DEME without FERIS.

### FERIS + Newton (Python-based GPU physics)

The FERIS + Newton workflow, including build and demo commands, lives in
[docs/how-to.md#feris--newton-python-based-gpu-physics](docs/how-to.md#feris--newton-python-based-gpu-physics).

### Newton + XLB + DEME (three-way coupling)

The Newton + XLB + DEME build, demo, and dependency guidance lives in
[docs/how-to.md#newton--xlb--deme-three-way-coupling](docs/how-to.md#newton--xlb--deme-three-way-coupling).

---

## Design: couplers directly own solver instances

See [docs/design.md#design-couplers-directly-own-solver-instances](docs/design.md#design-couplers-directly-own-solver-instances)
for the architecture notes and the FERIS + Newton wrapper explanation.

---

## Enabling external solvers

See [docs/design.md#enabling-external-solvers](docs/design.md#enabling-external-solvers)
for the fetch/install model and extension points.

---

## Co-simulation solvers

See [docs/design.md#co-simulation-solvers](docs/design.md#co-simulation-solvers)
for the solver option matrix and extension checklist.

---

## Python bindings

See [docs/design.md#python-bindings](docs/design.md#python-bindings) for the
binding build/import model and source-tree import guidance. See
[docs/python-package.md](docs/python-package.md) for the canonical package
source layout.

---

## Visualization

See [docs/design.md#visualization](docs/design.md#visualization) for the
summary, and [docs/visualization.md](docs/visualization.md) for
the visualization API contract and backend details.

---

## CMake option surface

See [docs/design.md#cmake-option-surface](docs/design.md#cmake-option-surface)
for the full option table, including the packaging-oriented
`MOPHI_PYTHON_PACKAGING_BUILD` mode.

---

## External solvers

See [docs/design.md#external-solvers](docs/design.md#external-solvers) for the
current solver catalog.

---

## License

See [LICENSE](LICENSE).
