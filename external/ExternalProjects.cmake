# ─────────────────────────────────────────────────────────────────────────────
# Registry of known external projects for MoPhi.
#
# To add a new external solver, append a mophi_register_external() call here.
# Each entry requires:
#   NAME           — identifier used for CMake options and directory names
#   URL            — git repository URL
#   TAG            — git branch, tag, or commit SHA to check out
#
# Optional metadata used by mophi_fetch_external():
#   LIB_NAME       — static library stem (without "lib" prefix or ".a" suffix)
#                    used to create an IMPORTED library target and track build
#                    byproducts.  Omit for header-only or unlinked externals.
#   HEADER_SRC     — relative subdirectory inside the source tree whose
#                    contents are copied to install/include/ during the install
#                    step.  Set this for projects that have no CMake install
#                    rules for headers.  Omit for projects that install headers
#                    via their own cmake --install step.
#   EXTRA_CMAKE_ARGS — additional -D flags forwarded to the external cmake
#                      configure step.
#
# Enabling a project is done at CMake configure time with
#   -DMOPHI_FETCH_<NAME>=ON
# ─────────────────────────────────────────────────────────────────────────────

# DEM-Engine has proper CMake install rules that install headers and the static
# library, so no HEADER_SRC override is needed.  The static library is named
# "simulator_multi_gpu".
mophi_register_external(
    NAME     DEMEngine
    URL      "https://github.com/projectchrono/DEM-Engine"
    TAG      "main"
    LIB_NAME "simulator_multi_gpu"
)

# TLFEA does not yet ship CMake install rules.  HEADER_SRC causes
# mophi_fetch_external to copy src/ → install/include/ manually.
# The static library is named "tlfea_lib".
mophi_register_external(
    NAME       TLFEA
    URL        "https://github.com/Ruochun/TLFEA"
    TAG        "main"
    LIB_NAME   "tlfea_lib"
    HEADER_SRC "src"
)
