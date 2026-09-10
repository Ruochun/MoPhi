# ───────────────────────────────────────────────────────────────────────────
# Newton + DEME demos
#
# The robotic 3D-printing demo couples a Newton machine to DEME particles and
# leaves the final DFC constitutive model as a later material upgrade.
# ────────────────────────────────────────────────────────────────────────────

find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_custom_target(demo_robotic_3d_printing
    COMMAND ${Python3_EXECUTABLE}
        "${CMAKE_CURRENT_SOURCE_DIR}/robotic_3d_printing/demo_robotic_3d_printing.py"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Running Newton/DEME robotic 3D-printing co-simulation"
    VERBATIM
)

add_custom_target(demo_dfc_3d_printing
    COMMAND ${Python3_EXECUTABLE}
        "${CMAKE_CURRENT_SOURCE_DIR}/robotic_3d_printing/demo_dfc_3d_printing.py"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Running G-code-driven Newton DFC printing motion stage"
    VERBATIM
)

install(
    FILES
        robotic_3d_printing/demo_robotic_3d_printing.py
        robotic_3d_printing/demo_dfc_3d_printing.py
    DESTINATION demo/newton_dem/robotic_3d_printing
)

install(
    FILES robotic_3d_printing/data/dfc_single_bead.gcode
    DESTINATION demo/newton_dem/robotic_3d_printing/data
)

install(
    DIRECTORY robotic_3d_printing/data/chrono_irb6700_printer
    DESTINATION demo/newton_dem/robotic_3d_printing/data
)

install(
    FILES
        "${CMAKE_SOURCE_DIR}/demo/shared_utils/__init__.py"
        "${CMAKE_SOURCE_DIR}/demo/shared_utils/abb_irb6700_printer_utils.py"
        "${CMAKE_SOURCE_DIR}/demo/shared_utils/ur10_demo_utils.py"
    DESTINATION demo/shared_utils
)

install(
    FILES "${CMAKE_SOURCE_DIR}/data/mesh/funnel.obj"
    DESTINATION data/mesh
)

install(
    FILES "${CMAKE_SOURCE_DIR}/data/force_models/ForceModelWithCohesion.cu"
    DESTINATION data/force_models
)
