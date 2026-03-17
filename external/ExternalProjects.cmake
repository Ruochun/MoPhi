# ─────────────────────────────────────────────────────────────────────────────
# Registry of known external projects for MoPhi.
#
# To add a new external solver, append a mophi_register_external() call here.
# Each entry needs:
#   NAME  — identifier used for CMake options and directory names
#   URL   — git repository URL
#   TAG   — git branch, tag, or commit SHA to check out
#
# Enabling a project is done at CMake configure time with
#   -DMOPHI_FETCH_<NAME>=ON
# ─────────────────────────────────────────────────────────────────────────────

mophi_register_external(
    NAME TLFEA
    URL  "https://github.com/Ruochun/TLFEA"
    TAG  "main"
)

mophi_register_external(
    NAME DEMEngine
    URL  "https://github.com/projectchrono/DEM-Engine"
    TAG  "main"
)
