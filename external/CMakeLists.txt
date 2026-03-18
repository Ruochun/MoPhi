# ─────────────────────────────────────────────────────────────────────────────
# external/CMakeLists.txt
#
# 1. Loads the registry of all known external projects.
# 2. For each project whose MOPHI_FETCH_<NAME> option is ON, registers it as
#    a CMake ExternalProject that will be downloaded, built in isolation, and
#    installed into ${CMAKE_BINARY_DIR}/external/<name>/install/ at build time.
#
# Using ExternalProject_Add (via mophi_fetch_external) ensures that headers
# generated during an external's own cmake configure step (e.g., DEM-Engine's
# core/ApiVersion.h) are present before MoPhi's source files are compiled.
#
# Adding a new external solver:
#   a) Add a mophi_register_external() entry in ExternalProjects.cmake.
#   b) Add a corresponding option() + mophi_fetch_external() block below.
# ─────────────────────────────────────────────────────────────────────────────

# Load the registry (must happen after MoPhiExternalUtils is included).
include(${CMAKE_CURRENT_SOURCE_DIR}/ExternalProjects.cmake)

# ── DEM-Engine ────────────────────────────────────────────────────────────────
if(MOPHI_FETCH_DEMENGINE)
    mophi_fetch_external(DEMEngine)
endif()

# ── TLFEA ─────────────────────────────────────────────────────────────────────
if(MOPHI_FETCH_TLFEA)
    mophi_fetch_external(TLFEA)
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
message(STATUS "MoPhi external projects status:")
foreach(_proj IN LISTS MOPHI_EXTERNAL_REGISTRY)
    if(MOPHI_EXTERNAL_${_proj}_AVAILABLE)
        set(_status "registered (will build into ${MOPHI_EXTERNAL_${_proj}_INSTALL_DIR})")
    else()
        set(_status "not fetched  (enable with -DMOPHI_FETCH_${_proj}=ON)")
    endif()
    message(STATUS "  ${_proj}: ${_status}")
endforeach()
