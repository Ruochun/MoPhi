# ─────────────────────────────────────────────────────────────────────────────
# demo/feris_newton/CMakeLists.txt
#
# The FERIS + Newton demo is a pure Python script.  No C++ executable is
# built here; the demo is installed alongside the MoPhi Python package so
# that it can be run with:
#
#   python demo/feris_newton/demo_feris_newton.py
#
# Prerequisites:
#   • MOPHI_BUILD_FERIS_NEWTON=ON   — builds the FERISNewtonCoupler library and
#                                     the mophi_core Python extension module.
#   • pip install --upgrade newton warp-lang mujoco==3.6.0
#                                   — or -DMOPHI_FETCH_NEWTON=ON at configure
#                                     time to have MoPhi install required packages automatically.
# ─────────────────────────────────────────────────────────────────────────────

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# ── Custom target to run the demo ─────────────────────────────────────────────
# Adds a `demo_feris_newton` build target.  Run with:
#   cmake --build <build-dir> --target demo_feris_newton
add_custom_target(demo_feris_newton
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/demo_feris_newton.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Running FERIS + Newton co-simulation demo"
    VERBATIM
)

# ── Installation ───────────────────────────────────────────────────────────────
install(FILES demo_feris_newton.py DESTINATION demo/feris_newton)
