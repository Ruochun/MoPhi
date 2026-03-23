# ─────────────────────────────────────────────────────────────────────────────
# demo/newton_xlb_dem/CMakeLists.txt
#
# The Newton + XLB + DEM-Engine three-way co-simulation demo is a pure Python
# script.  No C++ executable is built here; the demo is installed alongside the
# MoPhi Python package so that it can be run with:
#
#   python demo/newton_xlb_dem/demo_newton_xlb_dem.py
#
# Prerequisites:
#   • MOPHI_BUILD_NEWTON_XLB_DEM=ON   — builds NewtonXLBDEMCoupler and the
#                                        mophi_core Python extension module.
#   • pip install newton               — or -DMOPHI_FETCH_NEWTON=ON
#   • pip install warp-lang            — GPU runtime for Newton
#   • pip install xlb                  — or -DMOPHI_FETCH_XLB=ON  (optional)
# ─────────────────────────────────────────────────────────────────────────────

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# ── Custom target to run the demo ─────────────────────────────────────────────
# Adds a `demo_newton_xlb_dem` build target.  Run with:
#   cmake --build <build-dir> --target demo_newton_xlb_dem
add_custom_target(demo_newton_xlb_dem
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/demo_newton_xlb_dem.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Running Newton + XLB + DEM-Engine three-way co-simulation demo"
    VERBATIM
)

# ── Installation ───────────────────────────────────────────────────────────────
install(FILES demo_newton_xlb_dem.py DESTINATION demo/newton_xlb_dem)
