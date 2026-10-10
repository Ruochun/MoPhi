# ───────────────────────────────────────────────────────────────────────────
# Newton + DEME demos
#
# The robotic 3D-printing demos exercise both the cohesion approximation and
# the DFC material-preparation path used before printing.
# ────────────────────────────────────────────────────────────────────────────

find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_custom_target(demo_grab_contact
    COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_SOURCE_DIR}/python"
        ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/grab/grabbing_cup/demo_grab_contact.py"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Running the recorded-hand DEME cup grasp with Newton rendering"
    VERBATIM
)

add_custom_target(demo_grab_cylinder
    COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_SOURCE_DIR}/python"
        ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/grab/grabbing_cylinder/demo_grab_cylinder.py"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Running the recorded-hand DEME cylinder grasp with Newton rendering"
    VERBATIM
)

install(FILES grab/grasp_simulation.py grab/grasp_viewer.py DESTINATION demo/newton_dem/grab)
install(FILES grab/grabbing_cup/demo_grab_contact.py grab/grabbing_cup/render_grab_comparison.py
    grab/grabbing_cup/scene_config.py DESTINATION demo/newton_dem/grab/grabbing_cup)
install(FILES grab/grabbing_cylinder/demo_grab_cylinder.py grab/grabbing_cylinder/render_cylinder_comparison.py
    grab/grabbing_cylinder/scene_config.py DESTINATION demo/newton_dem/grab/grabbing_cylinder)

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
    COMMENT "Running confined DFC particle-settling stage"
    VERBATIM
)

install(
    FILES
        robotic_3d_printing/demo_robotic_3d_printing.py
        robotic_3d_printing/demo_dfc_3d_printing.py
    DESTINATION demo/newton_dem/robotic_3d_printing
)

install(
    FILES
        robotic_3d_printing/data/dfc_single_bead.gcode
        robotic_3d_printing/data/DFCModel.cu
        robotic_3d_printing/data/DFCModel.SOURCE.md
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
