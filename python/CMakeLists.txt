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
)

target_link_libraries(mophi_core
    PRIVATE
        mophi_couplers
)

# TLFEADEMCoupler.h is in src/couplers/ — needed by the binding file.
target_include_directories(mophi_core
    PRIVATE
        "${CMAKE_SOURCE_DIR}/src/couplers"
)

# When the TLFEA + DEM co-simulation solver has been compiled, enable its
# Python binding via a compile-time definition.
if(MOPHI_BUILD_TLFEA_DEM)
    target_compile_definitions(mophi_core PRIVATE MOPHI_HAS_TLFEA_DEM_COUPLER)
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
