include(FetchContent)

# ─────────────────────────────────────────────────────────────────────────────
# MoPhi external-project management utilities
#
# Provides three public entry points:
#
#   mophi_register_external(NAME <name> URL <url> TAG <tag>)
#       Record an external project in the global registry.
#       Call this from external/ExternalProjects.cmake.
#
#   mophi_fetch_external(<name>)
#       Download / populate the named external project into
#       ${CMAKE_SOURCE_DIR}/external/<name> using CMake FetchContent.
#       After a successful call, the cache variable
#       MOPHI_EXTERNAL_<name>_AVAILABLE is set to TRUE.
#
#   mophi_require_externals(<coupler_name> <dep1> [<dep2> ...])
#       Verify that all listed external projects are available and emit a
#       FATAL_ERROR when one is missing, guiding the user on how to resolve
#       the problem.
# ─────────────────────────────────────────────────────────────────────────────

# Internal: ensure the registry list variable exists.
if(NOT DEFINED MOPHI_EXTERNAL_REGISTRY)
    set(MOPHI_EXTERNAL_REGISTRY "" CACHE INTERNAL "Names of all registered external projects")
endif()

# ---------------------------------------------------------------------------
# mophi_register_external(NAME <name> URL <url> TAG <tag>)
# ---------------------------------------------------------------------------
macro(mophi_register_external)
    cmake_parse_arguments(_REG "" "NAME;URL;TAG" "" ${ARGN})
    if(NOT _REG_NAME OR NOT _REG_URL OR NOT _REG_TAG)
        message(FATAL_ERROR "mophi_register_external: NAME, URL, and TAG are all required.")
    endif()

    set(MOPHI_EXTERNAL_${_REG_NAME}_URL       "${_REG_URL}" CACHE INTERNAL "Source URL for ${_REG_NAME}")
    set(MOPHI_EXTERNAL_${_REG_NAME}_TAG       "${_REG_TAG}" CACHE INTERNAL "Git tag/branch for ${_REG_NAME}")
    set(MOPHI_EXTERNAL_${_REG_NAME}_AVAILABLE FALSE         CACHE INTERNAL "Whether ${_REG_NAME} has been fetched")

    list(APPEND MOPHI_EXTERNAL_REGISTRY "${_REG_NAME}")
    list(REMOVE_DUPLICATES MOPHI_EXTERNAL_REGISTRY)
    set(MOPHI_EXTERNAL_REGISTRY "${MOPHI_EXTERNAL_REGISTRY}" CACHE INTERNAL "Names of all registered external projects")
endmacro()

# ---------------------------------------------------------------------------
# mophi_fetch_external(<name>)
# ---------------------------------------------------------------------------
function(mophi_fetch_external name)
    if(NOT DEFINED MOPHI_EXTERNAL_${name}_URL OR "${MOPHI_EXTERNAL_${name}_URL}" STREQUAL "")
        message(FATAL_ERROR
            "MoPhi: Cannot fetch '${name}' — it was not registered.\n"
            "Add a mophi_register_external() entry for '${name}' in "
            "external/ExternalProjects.cmake.")
    endif()

    set(_src_dir "${CMAKE_SOURCE_DIR}/external/${name}")

    message(STATUS "MoPhi: Fetching '${name}' from "
        "${MOPHI_EXTERNAL_${name}_URL} @ ${MOPHI_EXTERNAL_${name}_TAG} ...")

    FetchContent_Declare(
        ${name}
        GIT_REPOSITORY "${MOPHI_EXTERNAL_${name}_URL}"
        GIT_TAG        "${MOPHI_EXTERNAL_${name}_TAG}"
        GIT_SHALLOW    TRUE
        SOURCE_DIR     "${_src_dir}"
    )

    # Populate (download) but do not automatically add_subdirectory; each
    # wrapper's CMakeLists.txt decides how to integrate the external code.
    FetchContent_GetProperties(${name})
    string(TOLOWER "${name}" _name_lc)
    if(NOT ${_name_lc}_POPULATED)
        FetchContent_Populate(${name})
    endif()

    set(MOPHI_EXTERNAL_${name}_AVAILABLE TRUE CACHE INTERNAL "Whether ${name} has been fetched")
    message(STATUS "MoPhi: '${name}' available at ${_src_dir}")
endfunction()

# ---------------------------------------------------------------------------
# mophi_require_externals(<coupler_name> <dep1> [<dep2> ...])
# ---------------------------------------------------------------------------
function(mophi_require_externals coupler_name)
    foreach(_dep IN LISTS ARGN)
        if(NOT MOPHI_EXTERNAL_${_dep}_AVAILABLE)
            message(FATAL_ERROR
                "MoPhi: Co-simulation solver '${coupler_name}' requires "
                "the external project '${_dep}', which has not been fetched.\n\n"
                "  Fix: re-run CMake with  -DMOPHI_FETCH_${_dep}=ON\n"
                "  URL: ${MOPHI_EXTERNAL_${_dep}_URL}\n")
        endif()
    endforeach()
endfunction()
