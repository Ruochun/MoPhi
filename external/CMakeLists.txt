# ─────────────────────────────────────────────────────────────────────────────
# external/CMakeLists.txt
#
# 1. Loads the registry of all known external projects.
# 2. For each project whose MOPHI_FETCH_<NAME> option is ON, fetches it into
#    ${CMAKE_SOURCE_DIR}/external/<name>/ using CMake FetchContent.
#
# Adding a new external solver:
#   a) Add a mophi_register_external() entry in ExternalProjects.cmake.
#   b) Add a corresponding option() + mophi_fetch_external() block below.
# ─────────────────────────────────────────────────────────────────────────────

# Load the registry (must happen after MoPhiExternalUtils is included).
include(${CMAKE_CURRENT_SOURCE_DIR}/ExternalProjects.cmake)

# ── TLFEA ────────────────────────────────────────────────────────────────────
if(MOPHI_FETCH_TLFEA)
    mophi_fetch_external(TLFEA)
endif()

# ── DEM-Engine ───────────────────────────────────────────────────────────────
if(MOPHI_FETCH_DEMENGINE)
    mophi_fetch_external(DEMEngine)
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
message(STATUS "MoPhi external projects status:")
foreach(_proj IN LISTS MOPHI_EXTERNAL_REGISTRY)
    if(MOPHI_EXTERNAL_${_proj}_AVAILABLE)
        set(_status "FETCHED")
    else()
        set(_status "not fetched  (enable with -DMOPHI_FETCH_${_proj}=ON)")
    endif()
    message(STATUS "  ${_proj}: ${_status}")
endforeach()
