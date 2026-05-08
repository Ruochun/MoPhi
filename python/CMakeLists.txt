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

# Place the compiled .so next to the Python package so the package __init__.py
# can import it with `from .mophi_core import ...`.
set_target_properties(mophi_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/mophi"
)

# ── Installation ──────────────────────────────────────────────────────────────
# Install the Python package tree and the extension module together.
install(TARGETS mophi_core DESTINATION python/mophi)
install(DIRECTORY mophi/ DESTINATION python/mophi
        FILES_MATCHING PATTERN "*.py")
