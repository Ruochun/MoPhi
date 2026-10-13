# ─────────────────────────────────────────────────────────────────────────────
# python/CMakeLists.txt
#
# Builds the mophi_core Python extension module using pybind11.
# pybind11 is fetched automatically via FetchContent if not found on the system.
#
# Each build keeps its own compiled module. Developer builds publish a copy
# beside the source package for `PYTHONPATH=python` execution.
# ─────────────────────────────────────────────────────────────────────────────

# ── Locate or fetch pybind11 ──────────────────────────────────────────────────
# Modern FindPython mode keeps the interpreter, headers, and extension ABI from
# the same Python installation when several CPython versions are present.
set(PYBIND11_FINDPYTHON ON)
if(Python3_EXECUTABLE AND (NOT MOPHI_PYTHON_PACKAGING_BUILD OR NOT Python_EXECUTABLE))
    # Packaging keeps the interpreter supplied by scikit-build-core.
    # Python3_EXECUTABLE is the public interpreter selector used by solver checks
    # and demos. Never let an older FindPython cache select a different ABI.
    set(Python_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH "Python interpreter used for MoPhi bindings" FORCE)
endif()
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 3.0 CONFIG QUIET)

if(NOT pybind11_FOUND)
    message(STATUS "MoPhi: pybind11 not found on the system — fetching via FetchContent ...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY "https://github.com/pybind/pybind11"
        GIT_TAG        "v3.0.4"
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# ── Python extension module ───────────────────────────────────────────────────
pybind11_add_module(mophi_core
    bindings/mophi_bindings.cpp
    bindings/feris_newton_bindings.cpp
    bindings/newton_xlb_dem_bindings.cpp
    bindings/mesh_io_bindings.cpp
)

target_link_libraries(mophi_core
    PRIVATE
        mophi_couplers
)

# PyFERISNewtonCoupler.h and PyFERISNewtonCoupler.cpp are in
# src/couplers/feris_newton/.  The .cpp must be compiled directly into
# mophi_core (not as a standalone static library) because it includes
# pybind11 types that are only available in the pybind11 module build context.
# The include directory and link requirements are propagated via the
# mophi_coupler_feris_newton INTERFACE target (→ mophi_couplers → mophi_core).
if(MOPHI_BUILD_FERIS_NEWTON)
    target_sources(mophi_core
        PRIVATE
            "${CMAKE_SOURCE_DIR}/src/couplers/feris_newton/PyFERISNewtonCoupler.cpp"
    )
    # Enable the FERISNewtonCoupler Python binding via a compile-time definition.
    target_compile_definitions(mophi_core PRIVATE MOPHI_HAS_FERIS_NEWTON_COUPLER)
endif()

# PyNewtonXLBDEMCoupler.h and PyNewtonXLBDEMCoupler.cpp are in
# src/couplers/newton_xlb_dem/.  The .cpp must be compiled directly into
# mophi_core (not as a standalone static library) because it includes
# pybind11 types that are only available in the pybind11 module build context.
# The include directory and link requirements are propagated via the
# mophi_coupler_newton_xlb_dem INTERFACE target (→ mophi_couplers → mophi_core).
if(MOPHI_BUILD_NEWTON_XLB_DEM)
    target_sources(mophi_core
        PRIVATE
            "${CMAKE_SOURCE_DIR}/src/couplers/newton_xlb_dem/PyNewtonXLBDEMCoupler.cpp"
    )
    # Enable the NewtonXLBDEMCoupler Python binding via a compile-time definition.
    target_compile_definitions(mophi_core PRIVATE MOPHI_HAS_NEWTON_XLB_DEM_COUPLER)
endif()

# Keep link outputs private to each build: another configuration must not make
# this target appear up-to-date merely by overwriting a shared source-tree .so.
set_target_properties(mophi_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/mophi"
)

if(NOT MOPHI_PYTHON_PACKAGING_BUILD)
    # Publish after linking, including an explicit --target mophi_core build.
    add_custom_command(TARGET mophi_core POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:mophi_core>" "${CMAKE_CURRENT_SOURCE_DIR}/mophi/$<TARGET_FILE_NAME:mophi_core>"
        VERBATIM
    )
    # Also publish on an incremental default build with no relink. A different
    # build directory may have selected another feature set since the last build.
    add_custom_target(mophi_python_source ALL
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:mophi_core>" "${CMAKE_CURRENT_SOURCE_DIR}/mophi/$<TARGET_FILE_NAME:mophi_core>"
        DEPENDS mophi_core
        COMMENT "Publishing this build's MoPhi extension to the source package"
        VERBATIM
    )
endif()

# ── Installation ──────────────────────────────────────────────────────────────
# Install the Python package tree and the extension module together.
install(TARGETS mophi_core
        LIBRARY DESTINATION mophi
        RUNTIME DESTINATION mophi
        ARCHIVE DESTINATION mophi)
install(DIRECTORY mophi/ DESTINATION mophi
        FILES_MATCHING PATTERN "*.py")
