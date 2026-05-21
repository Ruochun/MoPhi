# ─────────────────────────────────────────────────────────────────────────────
# python/CMakeLists.txt
#
# Builds the mophi_core Python extension module using pybind11.
# pybind11 is fetched automatically via FetchContent if not found on the system.
#
# The compiled module is placed next to the Python package directory so that
# `import mophi` works directly from the build tree.
# ─────────────────────────────────────────────────────────────────────────────

# ── Locate or fetch pybind11 ──────────────────────────────────────────────────
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    message(STATUS "MoPhi: pybind11 not found on the system — fetching via FetchContent ...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY "https://github.com/pybind/pybind11"
        GIT_TAG        "v2.13.6"
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

if(MOPHI_PYTHON_PACKAGING_BUILD)
    set(_mophi_python_module_output_dir "${CMAKE_CURRENT_BINARY_DIR}/mophi")
else()
    set(_mophi_python_module_output_dir "${CMAKE_CURRENT_SOURCE_DIR}/mophi")
endif()

# Developer builds place the compiled .so next to the source-tree package so
# `PYTHONPATH=python` works immediately.  Packaging builds keep artefacts in the
# build tree and rely on the install step for the final wheel layout.
set_target_properties(mophi_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${_mophi_python_module_output_dir}"
)

# ── Installation ──────────────────────────────────────────────────────────────
# Install the Python package tree and the extension module together.
install(TARGETS mophi_core
        LIBRARY DESTINATION mophi
        RUNTIME DESTINATION mophi
        ARCHIVE DESTINATION mophi)
install(DIRECTORY mophi/ DESTINATION mophi
        FILES_MATCHING PATTERN "*.py")
