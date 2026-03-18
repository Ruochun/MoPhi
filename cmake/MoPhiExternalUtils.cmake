include(ExternalProject)

# ─────────────────────────────────────────────────────────────────────────────
# MoPhi external-project management utilities
#
# Provides three public entry points:
#
#   mophi_register_external(
#       NAME     <name>
#       URL      <url>
#       TAG      <tag>
#       [LIB_NAME     <lib>]          # static library stem (no lib prefix / .a)
#       [HEADER_SRC   <subdir>]       # copy <subdir>/ → install/include/ instead
#                                     # of relying on the project's own install rules
#       [EXTRA_LIB_NAMES <lib>...]    # additional shared libraries installed by the
#                                     # external (no lib prefix / suffix); each gets
#                                     # an IMPORTED target and is linked transitively
#       [EXTRA_CMAKE_ARGS <arg>...]   # extra -D flags forwarded to the external build
#   )
#       Record an external project in the global registry.
#       Call this from external/ExternalProjects.cmake.
#
#   mophi_fetch_external(<name>)
#       Register the named external as a CMake ExternalProject that will be
#       downloaded, configured, built, and installed into an isolated directory
#       tree under ${CMAKE_BINARY_DIR}/external/<name>/ at build time.
#
#       Using ExternalProject_Add (rather than FetchContent) ensures that
#       any headers generated during the external's own configure step
#       (e.g., DEM-Engine's core/ApiVersion.h) are present before MoPhi's
#       own source files are compiled.
#
#       Directory layout created for each external:
#         ${CMAKE_BINARY_DIR}/external/<name>/source   — git clone
#         ${CMAKE_BINARY_DIR}/external/<name>/build    — isolated build tree
#         ${CMAKE_BINARY_DIR}/external/<name>/install  — install prefix
#
#       After a successful call the following cache variables are set:
#         MOPHI_EXTERNAL_<name>_AVAILABLE    TRUE
#         MOPHI_EXTERNAL_<name>_INSTALL_DIR  <install prefix>
#
#       An IMPORTED library target named  mophi_ext_<name>  is created that
#       carries the installed include directory (and the library location when
#       LIB_NAME is given).  Link coupler targets against it; the dependency
#       chain guarantees the external is fully built before compilation begins.
#
#   mophi_require_externals(<coupler_name> <dep1> [<dep2> ...])
#       Verify that all listed external projects have been registered and emit a
#       FATAL_ERROR when one is missing, guiding the user on how to resolve
#       the problem.
# ─────────────────────────────────────────────────────────────────────────────

# Internal: ensure the registry list variable exists.
if(NOT DEFINED MOPHI_EXTERNAL_REGISTRY)
    set(MOPHI_EXTERNAL_REGISTRY "" CACHE INTERNAL "Names of all registered external projects")
endif()

# ---------------------------------------------------------------------------
# mophi_register_external(NAME <name> URL <url> TAG <tag>
#                          [LIB_NAME <lib>] [HEADER_SRC <subdir>]
#                          [EXTRA_LIB_NAMES <lib>...]
#                          [EXTRA_CMAKE_ARGS <arg>...])
# ---------------------------------------------------------------------------
macro(mophi_register_external)
    cmake_parse_arguments(_REG "" "NAME;URL;TAG;LIB_NAME;HEADER_SRC" "EXTRA_CMAKE_ARGS;EXTRA_LIB_NAMES" ${ARGN})
    if(NOT _REG_NAME OR NOT _REG_URL OR NOT _REG_TAG)
        message(FATAL_ERROR "mophi_register_external: NAME, URL, and TAG are all required.")
    endif()

    set(MOPHI_EXTERNAL_${_REG_NAME}_URL             "${_REG_URL}"              CACHE INTERNAL "Source URL for ${_REG_NAME}")
    set(MOPHI_EXTERNAL_${_REG_NAME}_TAG             "${_REG_TAG}"              CACHE INTERNAL "Git tag/branch for ${_REG_NAME}")
    set(MOPHI_EXTERNAL_${_REG_NAME}_LIB_NAME        "${_REG_LIB_NAME}"         CACHE INTERNAL "Static library stem for ${_REG_NAME}")
    set(MOPHI_EXTERNAL_${_REG_NAME}_HEADER_SRC      "${_REG_HEADER_SRC}"       CACHE INTERNAL "Header source subdir for ${_REG_NAME}")
    set(MOPHI_EXTERNAL_${_REG_NAME}_EXTRA_CMAKE_ARGS "${_REG_EXTRA_CMAKE_ARGS}" CACHE INTERNAL "Extra CMake args for ${_REG_NAME}")
    set(MOPHI_EXTERNAL_${_REG_NAME}_EXTRA_LIB_NAMES  "${_REG_EXTRA_LIB_NAMES}" CACHE INTERNAL "Additional shared library stems for ${_REG_NAME}")
    set(MOPHI_EXTERNAL_${_REG_NAME}_AVAILABLE       FALSE                      CACHE INTERNAL "Whether ${_REG_NAME} has been registered")

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

    # Isolated directory layout for this external.
    set(_src_dir     "${CMAKE_BINARY_DIR}/external/${name}/source")
    set(_build_dir   "${CMAKE_BINARY_DIR}/external/${name}/build")
    set(_install_dir "${CMAKE_BINARY_DIR}/external/${name}/install")

    # ── CMake arguments forwarded to the external's configure step ──────────
    set(_cmake_args
        "-DCMAKE_INSTALL_PREFIX=${_install_dir}"
        "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    )
    # Forward CUDA architecture selection from the parent project.
    if(DEFINED CMAKE_CUDA_ARCHITECTURES)
        list(APPEND _cmake_args "-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
    endif()
    # Per-project extra args declared in mophi_register_external().
    if(MOPHI_EXTERNAL_${name}_EXTRA_CMAKE_ARGS)
        list(APPEND _cmake_args ${MOPHI_EXTERNAL_${name}_EXTRA_CMAKE_ARGS})
    endif()

    # ── Installed library path (for IMPORTED target + Ninja byproduct) ──────
    set(_lib_suffix "${CMAKE_STATIC_LIBRARY_SUFFIX}")
    if(MOPHI_EXTERNAL_${name}_LIB_NAME)
        set(_lib_file
            "${_install_dir}/lib/lib${MOPHI_EXTERNAL_${name}_LIB_NAME}${_lib_suffix}")
    else()
        set(_lib_file "")
    endif()

    # ── Extra shared libraries produced by this external ────────────────────
    # Some externals install additional shared libraries alongside the main
    # static library (e.g., DEM-Engine installs libDEMERuntimeDataHelper.so in
    # addition to libsimulator_multi_gpu.a).  We pre-compute their installed
    # paths so they can be listed as BUILD_BYPRODUCTS (required by Ninja) and
    # so we can create IMPORTED targets for them after the build.
    set(_extra_lib_files "")
    foreach(_extra IN LISTS MOPHI_EXTERNAL_${name}_EXTRA_LIB_NAMES)
        list(APPEND _extra_lib_files
            "${_install_dir}/lib/lib${_extra}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    endforeach()

    # ── ExternalProject arguments list ──────────────────────────────────────
    set(_ep_args
        GIT_REPOSITORY        "${MOPHI_EXTERNAL_${name}_URL}"
        GIT_TAG               "${MOPHI_EXTERNAL_${name}_TAG}"
        GIT_SHALLOW           TRUE
        GIT_SUBMODULES_RECURSE TRUE
        SOURCE_DIR            "${_src_dir}"
        BINARY_DIR            "${_build_dir}"
        INSTALL_DIR           "${_install_dir}"
        CMAKE_ARGS            ${_cmake_args}
    )

    # When the project does not provide its own CMake install rules, we
    # override INSTALL_COMMAND to copy headers (and library) manually from
    # the source / build tree into the install prefix.
    if(MOPHI_EXTERNAL_${name}_HEADER_SRC)
        list(APPEND _ep_args
            INSTALL_COMMAND
                ${CMAKE_COMMAND} -E make_directory "${_install_dir}/include"
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                    "${_src_dir}/${MOPHI_EXTERNAL_${name}_HEADER_SRC}"
                    "${_install_dir}/include"
        )
        if(_lib_file)
            list(APPEND _ep_args
                COMMAND ${CMAKE_COMMAND} -E make_directory "${_install_dir}/lib"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_build_dir}/lib${MOPHI_EXTERNAL_${name}_LIB_NAME}${_lib_suffix}"
                    "${_lib_file}"
            )
        endif()
    endif()

    # Tell Ninja (and other generators) which files the external build produces
    # so it can correctly determine when to rebuild downstream targets.
    # This must include both the main library and any extra shared libraries.
    set(_all_byproducts "")
    if(_lib_file)
        list(APPEND _all_byproducts "${_lib_file}")
    endif()
    foreach(_extra_file IN LISTS _extra_lib_files)
        list(APPEND _all_byproducts "${_extra_file}")
    endforeach()
    if(_all_byproducts)
        list(APPEND _ep_args BUILD_BYPRODUCTS ${_all_byproducts})
    endif()

    ExternalProject_Add(${name}_external ${_ep_args})

    # ── Pre-create directories so IMPORTED target validation passes ──────────
    file(MAKE_DIRECTORY "${_install_dir}/include")
    if(_lib_file)
        file(MAKE_DIRECTORY "${_install_dir}/lib")
    endif()

    # ── Create an IMPORTED CMake target for downstream use ───────────────────
    # Coupler targets link against mophi_ext_<name>, which carries the correct
    # include directory and (when a library was registered) the library location.
    # The add_dependencies() call ensures the external is fully built and
    # installed before any MoPhi source file that uses it is compiled.
    if(_lib_file)
        add_library(mophi_ext_${name} STATIC IMPORTED GLOBAL)
        set_target_properties(mophi_ext_${name} PROPERTIES
            IMPORTED_LOCATION             "${_lib_file}"
            INTERFACE_INCLUDE_DIRECTORIES "${_install_dir}/include"
        )
    else()
        add_library(mophi_ext_${name} INTERFACE IMPORTED GLOBAL)
        set_target_properties(mophi_ext_${name} PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${_install_dir}/include"
        )
    endif()
    add_dependencies(mophi_ext_${name} ${name}_external)

    # ── Create IMPORTED targets for each extra shared library ────────────────
    # Some externals install additional shared libraries that the main static
    # library links against (e.g., DEM-Engine's DEMERuntimeDataHelper.so).
    # We create an UNKNOWN IMPORTED target for each so that CMake carries the
    # correct library path on every consumer's link line, and we attach them
    # as INTERFACE_LINK_LIBRARIES of the main target so the dependency
    # propagates automatically.
    foreach(_extra IN LISTS MOPHI_EXTERNAL_${name}_EXTRA_LIB_NAMES)
        set(_extra_file "${_install_dir}/lib/lib${_extra}${CMAKE_SHARED_LIBRARY_SUFFIX}")
        file(MAKE_DIRECTORY "${_install_dir}/lib")
        add_library(mophi_ext_${name}_${_extra} UNKNOWN IMPORTED GLOBAL)
        set_target_properties(mophi_ext_${name}_${_extra} PROPERTIES
            IMPORTED_LOCATION "${_extra_file}"
        )
        add_dependencies(mophi_ext_${name}_${_extra} ${name}_external)
        set_property(TARGET mophi_ext_${name} APPEND PROPERTY
            INTERFACE_LINK_LIBRARIES "mophi_ext_${name}_${_extra}"
        )
    endforeach()

    set(MOPHI_EXTERNAL_${name}_AVAILABLE    TRUE             CACHE INTERNAL "Whether ${name} has been registered" FORCE)
    set(MOPHI_EXTERNAL_${name}_INSTALL_DIR  "${_install_dir}" CACHE INTERNAL "Install prefix for ${name}" FORCE)

    message(STATUS "MoPhi: '${name}' registered — "
        "will be built in isolation into ${_install_dir}")
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
