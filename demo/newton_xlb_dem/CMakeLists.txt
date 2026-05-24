# ─────────────────────────────────────────────────────────────────────────────
# demo/newton_xlb_dem/CMakeLists.txt
#
# The Newton + XLB + DEM-Engine three-way co-simulation demos are pure Python
# scripts.  No C++ executables are built here; each demo is installed alongside
# the MoPhi Python package so that it can be run directly:
#
#   python demo/newton_xlb_dem/demo_newton_xlb_dem.py
#   python demo/newton_xlb_dem/excavation_comparison/demo_claw_newton.py
#   python demo/newton_xlb_dem/excavation_comparison/demo_claw_newton_cubes.py
#
# Prerequisites:
#   • MOPHI_BUILD_NEWTON_XLB_DEM=ON   — builds NewtonXLBDEMCoupler and the
#                                        mophi_core Python extension module.
#   • pip install --upgrade newton warp-lang mujoco==3.6.0
#                                      — or -DMOPHI_FETCH_NEWTON=ON
#   • pip install "xlb[cuda]"          — or -DMOPHI_FETCH_XLB=ON
# ─────────────────────────────────────────────────────────────────────────────

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# ── Custom target: ANYmal C walking + XLB + DEME demo ────────────────────────
# Adds a `demo_newton_xlb_dem` build target.  Run with:
#   cmake --build <build-dir> --target demo_newton_xlb_dem
add_custom_target(demo_newton_xlb_dem
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/demo_newton_xlb_dem.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Running Newton + XLB + DEM-Engine three-way co-simulation demo"
    VERBATIM
)

# ── Custom target: UR10 claw arm Newton demo ──────────────────────────────────
# First phase of the claw-plowing-granular-material co-simulation.
# Adds a `demo_claw_newton` build target.  Run with:
#   cmake --build <build-dir> --target demo_claw_newton
add_custom_target(demo_claw_newton
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/excavation_comparison/demo_claw_newton.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Running UR10 claw arm Newton co-simulation demo (claw plowing groundwork phase)"
    VERBATIM
)

# ── Custom target: UR10 claw arm Newton coarse-cube comparison demo ───────────
# Adds a `demo_claw_newton_cubes` build target.  Run with:
#   cmake --build <build-dir> --target demo_claw_newton_cubes
add_custom_target(demo_claw_newton_cubes
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/excavation_comparison/demo_claw_newton_cubes.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Running UR10 claw arm Newton coarse-cube comparison demo"
    VERBATIM
)

# ── Installation ───────────────────────────────────────────────────────────────
# newton_xlb_dem_utils.py is a pre-existing helper module imported by
# demo_newton_xlb_dem.py; it is included here so that the install tree is
# self-contained.
install(
    FILES
        demo_newton_xlb_dem.py
        newton_xlb_dem_utils.py
    DESTINATION demo/newton_xlb_dem
)

install(
    FILES
        excavation_comparison/demo_claw_newton.py
        excavation_comparison/demo_claw_newton_cubes.py
    DESTINATION demo/newton_xlb_dem/excavation_comparison
)
