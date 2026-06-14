# ─────────────────────────────────────────────────────────────────────────────
# demo/newton_xlb_dem/CMakeLists.txt
#
# The Newton + XLB + DEM-Engine three-way co-simulation demos are pure Python
# scripts.  No C++ executables are built here; each demo is installed alongside
# the MoPhi Python package so that it can be run directly:
#
#   python demo/newton_xlb_dem/anymal_robot_multiphysics/demo_anymal_robot_multiphysics.py
#   python demo/newton_xlb_dem/anymal_robot_multiphysics/demo_anymal_robot_newton_baseline.py
#   python demo/newton_xlb_dem/excavation_comparison/demo_claw_newton.py
#   python demo/newton_xlb_dem/excavation_comparison/demo_claw_newton_cubes.py
#   python demo/newton_xlb_dem/humanoid_complex_environment/demo_humanoid_complex_environment.py
#   python demo/newton_xlb_dem/humanoid_complex_environment/demo_humanoid_swimming.py
#   python demo/newton_xlb_dem/humanoid_complex_environment/demo_many_flexible_hands.py
#
# Prerequisites:
#   • MOPHI_BUILD_NEWTON_XLB_DEM=ON   — builds NewtonXLBDEMCoupler and the
#                                        mophi_core Python extension module.
#   • pip install --upgrade newton warp-lang mujoco==3.6.0
#                                      — or -DMOPHI_FETCH_NEWTON=ON
#   • pip install "xlb[cuda]"          — or -DMOPHI_FETCH_XLB=ON
# ─────────────────────────────────────────────────────────────────────────────

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# ── Custom target: ANYmal Newton-only baseline demo ───────────────────────────
# Adds a `demo_anymal_robot_newton_baseline` build target.  Run with:
#   cmake --build <build-dir> --target demo_anymal_robot_newton_baseline
add_custom_target(demo_anymal_robot_newton_baseline
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/anymal_robot_multiphysics/demo_anymal_robot_newton_baseline.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Running ANYmal Newton-only walking baseline demo"
    VERBATIM
)

# ── Custom target: ANYmal multiphysics demo ───────────────────────────────────
# Adds a `demo_anymal_robot_multiphysics` build target.  Run with:
#   cmake --build <build-dir> --target demo_anymal_robot_multiphysics
add_custom_target(demo_anymal_robot_multiphysics
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/anymal_robot_multiphysics/demo_anymal_robot_multiphysics.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Running ANYmal Newton + XLB + DEME multiphysics demo"
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

# ── Custom target: interactive humanoid complex-environment baseline ──────────
# Adds a `demo_humanoid_complex_environment` build target.  Run with:
#   cmake --build <build-dir> --target demo_humanoid_complex_environment
add_custom_target(demo_humanoid_complex_environment
    COMMAND ${Python3_EXECUTABLE}
        "${CMAKE_CURRENT_SOURCE_DIR}/humanoid_complex_environment/demo_humanoid_complex_environment.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Running interactive Newton humanoid complex-environment baseline demo"
    VERBATIM
)

# ── Custom target: humanoid swimming idea demo ────────────────────────────────
add_custom_target(demo_humanoid_swimming
    COMMAND ${Python3_EXECUTABLE}
        "${CMAKE_CURRENT_SOURCE_DIR}/humanoid_complex_environment/demo_humanoid_swimming.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Running Newton + XLB humanoid swimming idea demo"
    VERBATIM
)

# ── Custom target: many articulated hands demo ────────────────────────────────
# Adds a `demo_many_flexible_hands` build target.  Run with:
#   cmake --build <build-dir> --target demo_many_flexible_hands
add_custom_target(demo_many_flexible_hands
    COMMAND ${Python3_EXECUTABLE}
        "${CMAKE_CURRENT_SOURCE_DIR}/humanoid_complex_environment/demo_many_flexible_hands.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Running many independently animated Allegro hands demo"
    VERBATIM
)

# ── Installation ───────────────────────────────────────────────────────────────
# newton_xlb_dem_utils.py is a pre-existing helper module imported by
# demo_anymal_robot_multiphysics.py; it is included here so that the install tree is
# self-contained.
install(
    FILES
        newton_xlb_dem_utils.py
    DESTINATION demo/newton_xlb_dem
)

install(
    FILES
        anymal_robot_multiphysics/demo_anymal_robot_multiphysics.py
        anymal_robot_multiphysics/demo_anymal_robot_newton_baseline.py
    DESTINATION demo/newton_xlb_dem/anymal_robot_multiphysics
)

install(
    FILES
        excavation_comparison/demo_claw_newton.py
        excavation_comparison/demo_claw_newton_cubes.py
    DESTINATION demo/newton_xlb_dem/excavation_comparison
)

install(
    FILES
        humanoid_complex_environment/demo_humanoid_complex_environment.py
        humanoid_complex_environment/demo_humanoid_swimming.py
        humanoid_complex_environment/demo_many_flexible_hands.py
        humanoid_complex_environment/README.md
    DESTINATION demo/newton_xlb_dem/humanoid_complex_environment
)
