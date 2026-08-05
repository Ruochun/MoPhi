# ───────────────────────────────────────────────────────────────────────────
# Newton + DEME demos
#
# The first robotic 3D-printing phase builds and animates only the Newton
# machine.  Its DEME/DFC material model will be added as a later phase.
# ────────────────────────────────────────────────────────────────────────────

find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_custom_target(demo_robotic_3d_printing
    COMMAND ${Python3_EXECUTABLE}
        "${CMAKE_CURRENT_SOURCE_DIR}/robotic_3d_printing/demo_robotic_3d_printing.py"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Running Newton robotic 3D-printing machine demo"
    VERBATIM
)

install(
    FILES robotic_3d_printing/demo_robotic_3d_printing.py
    DESTINATION demo/newton_dem/robotic_3d_printing
)

install(
    FILES "${CMAKE_SOURCE_DIR}/data/mesh/funnel.obj"
    DESTINATION data/mesh
)
