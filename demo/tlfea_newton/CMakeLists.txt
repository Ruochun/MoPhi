# ─────────────────────────────────────────────────────────────────────────────
# demo/tlfea_newton/CMakeLists.txt
#
# The TLFEA + Newton demo is a pure Python script.  No C++ executable is
# built here; the demo is installed alongside the MoPhi Python package so
# that it can be run with:
#
#   python demo/tlfea_newton/demo_tlfea_newton.py
#
# Prerequisites:
#   • MOPHI_BUILD_TLFEA_NEWTON=ON   — builds the TLFEANewtonCoupler library and
#                                     the mophi_core Python extension module.
#   • pip install newton             — or -DMOPHI_FETCH_NEWTON=ON at configure
#                                     time to have MoPhi install it automatically.
# ─────────────────────────────────────────────────────────────────────────────

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# ── Custom target to run the demo ─────────────────────────────────────────────
# Adds a `demo_tlfea_newton` build target.  Run with:
#   cmake --build <build-dir> --target demo_tlfea_newton
add_custom_target(demo_tlfea_newton
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/demo_tlfea_newton.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Running TLFEA + Newton co-simulation demo"
    VERBATIM
)

# ── Installation ───────────────────────────────────────────────────────────────
install(FILES demo_tlfea_newton.py DESTINATION demo/tlfea_newton)
