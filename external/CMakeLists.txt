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

# Required dependency constraints for Newton-based MoPhi components.
set(_mophi_newton_required_version "${MOPHI_REQUIRED_NEWTON_VERSION}")
set(_mophi_warp_required_version "${MOPHI_REQUIRED_WARP_VERSION}")
set(_mophi_mujoco_required_version "${MOPHI_REQUIRED_MUJOCO_VERSION}")
set(_mophi_version_check_script "${CMAKE_SOURCE_DIR}/cmake/check_newton_warp_versions.py")

if(MOPHI_PYTHON_PACKAGING_BUILD)
    message(STATUS "MoPhi: Python packaging build enabled — runtime Python dependencies are provided via wheel metadata, not CMake pip installs")
endif()

# ── DEM-Engine ────────────────────────────────────────────────────────────────
if(MOPHI_FETCH_DEMENGINE)
    mophi_fetch_external(DEMEngine)
endif()

# ── FERIS ─────────────────────────────────────────────────────────────────────
if(MOPHI_FETCH_FERIS)
    mophi_fetch_external(FERIS)
endif()

# ── Newton + Warp + MuJoCo (Python packages) ──────────────────────────────────
# Newton is a pure Python physics engine built on NVIDIA Warp.  It has no CMake
# build system and cannot be handled with mophi_fetch_external / ExternalProject_Add.
# When MOPHI_FETCH_NEWTON=ON, MoPhi enforces required package constraints:
#   newton==${_mophi_newton_required_version}, warp-lang==${_mophi_warp_required_version},
#   and mujoco==${_mophi_mujoco_required_version}
# so that couplers and demos stay stable across environments.
#
# Note: this pip install uses whatever Python interpreter CMake finds first
# (the same one that will build the pybind11 extension module).  Users running
# inside a virtual environment should activate it before invoking CMake.
if(MOPHI_FETCH_NEWTON AND NOT MOPHI_PYTHON_PACKAGING_BUILD)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    message(STATUS
        "MoPhi: Installing required Newton/Warp/MuJoCo Python packages via pip "
        "(newton==${_mophi_newton_required_version}, warp-lang==${_mophi_warp_required_version}, "
        "mujoco==${_mophi_mujoco_required_version}) ...")
    execute_process(
        COMMAND           ${Python3_EXECUTABLE} -m pip install --upgrade newton==${_mophi_newton_required_version} warp-lang==${_mophi_warp_required_version} mujoco==${_mophi_mujoco_required_version}
        RESULT_VARIABLE   _newton_install_result
        OUTPUT_VARIABLE   _newton_install_output
        ERROR_VARIABLE    _newton_install_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )
    if(_newton_install_result EQUAL 0)
        message(STATUS "MoPhi: Required Newton/Warp/MuJoCo Python packages installed successfully")
        set(MOPHI_EXTERNAL_Newton_AVAILABLE TRUE
            CACHE INTERNAL "Newton Python package available")
    else()
        message(WARNING
            "MoPhi: pip install newton==${_mophi_newton_required_version} warp-lang==${_mophi_warp_required_version} mujoco==${_mophi_mujoco_required_version} failed "
            "(exit code ${_newton_install_result}):\n"
            "${_newton_install_error}\n"
            "Required Newton/Warp/MuJoCo packages will not be available for couplers that require them.\n"
            "Install them manually with:\n"
            "  pip install --upgrade newton==${_mophi_newton_required_version} "
            "warp-lang==${_mophi_warp_required_version} "
            "mujoco==${_mophi_mujoco_required_version}")
    endif()
else()
    if(MOPHI_FETCH_NEWTON AND MOPHI_PYTHON_PACKAGING_BUILD)
        message(STATUS "MoPhi: Ignoring MOPHI_FETCH_NEWTON during Python packaging build")
    endif()
    # Even without MOPHI_FETCH_NEWTON, detect whether the required
    # Newton/Warp/MuJoCo packages are already installed so the summary line below
    # reports the correct status.
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_FOUND)
        execute_process(
            COMMAND         ${Python3_EXECUTABLE}
                            "${_mophi_version_check_script}"
                            --newton-version "${_mophi_newton_required_version}"
                            --warp-version "${_mophi_warp_required_version}"
                            --mujoco-version "${_mophi_mujoco_required_version}"
                            --quiet
            RESULT_VARIABLE _newton_check_result
            OUTPUT_QUIET
            ERROR_QUIET
        )
        if(_newton_check_result EQUAL 0)
            set(MOPHI_EXTERNAL_Newton_AVAILABLE TRUE
                CACHE INTERNAL "Newton Python package available")
        else()
            set(MOPHI_EXTERNAL_Newton_AVAILABLE FALSE
                CACHE INTERNAL "Newton Python package not available")
        endif()
    endif()
endif()

# ── XLB (Python package) ──────────────────────────────────────────────────────
# XLB (jax-Lattice-Boltzmann) is a pure Python LBM solver — it has no CMake
# build system and cannot be handled with mophi_fetch_external.  When
# MOPHI_FETCH_XLB=ON, MoPhi installs the CUDA-enabled XLB package via pip at configure time.
if(MOPHI_FETCH_XLB AND NOT MOPHI_PYTHON_PACKAGING_BUILD)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    message(STATUS "MoPhi: Installing XLB Python package via pip (xlb[cuda]) ...")
    execute_process(
        COMMAND           ${Python3_EXECUTABLE} -m pip install "xlb[cuda]"
        RESULT_VARIABLE   _xlb_install_result
        OUTPUT_VARIABLE   _xlb_install_output
        ERROR_VARIABLE    _xlb_install_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )
    if(_xlb_install_result EQUAL 0)
        message(STATUS "MoPhi: XLB Python package installed successfully")
        set(MOPHI_EXTERNAL_XLB_AVAILABLE TRUE
            CACHE INTERNAL "XLB Python package available")
    else()
        message(WARNING
            "MoPhi: pip install xlb[cuda] failed (exit code ${_xlb_install_result}):\n"
            "${_xlb_install_error}\n"
            "XLB will not be available for the demo.  "
            "Install it manually with:  pip install \"xlb[cuda]\"")
    endif()
else()
    if(MOPHI_FETCH_XLB AND MOPHI_PYTHON_PACKAGING_BUILD)
        message(STATUS "MoPhi: Ignoring MOPHI_FETCH_XLB during Python packaging build")
    endif()
    # Even without MOPHI_FETCH_XLB, detect a pre-installed XLB so the
    # summary line below reports the correct status.
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_FOUND)
        execute_process(
            COMMAND         ${Python3_EXECUTABLE} -c "import xlb"
            RESULT_VARIABLE _xlb_check_result
            OUTPUT_QUIET
            ERROR_QUIET
        )
        if(_xlb_check_result EQUAL 0)
            set(MOPHI_EXTERNAL_XLB_AVAILABLE TRUE
                CACHE INTERNAL "XLB Python package available")
        else()
            set(MOPHI_EXTERNAL_XLB_AVAILABLE FALSE
                CACHE INTERNAL "XLB Python package not available")
        endif()
    endif()
endif()

# ── DEME (Python package) ─────────────────────────────────────────────────────
# The DEME provider is a Python distribution — it has no CMake build system
# and cannot be handled with mophi_fetch_external.  When MOPHI_FETCH_DEME=ON,
# MoPhi installs the configured distribution via pip.  Its distribution and
# import names are deliberately separate so API-compatible providers are easy
# to substitute.
if(MOPHI_FETCH_DEME AND NOT MOPHI_PYTHON_PACKAGING_BUILD)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    message(STATUS "MoPhi: Installing DEME provider '${MOPHI_PACKAGE_DEME_DISTRIBUTION}' via pip ...")
    execute_process(
        COMMAND           ${Python3_EXECUTABLE} -m pip install ${MOPHI_PACKAGE_DEME_DISTRIBUTION}
        RESULT_VARIABLE   _deme_install_result
        OUTPUT_VARIABLE   _deme_install_output
        ERROR_VARIABLE    _deme_install_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )
    if(_deme_install_result EQUAL 0)
        message(STATUS "MoPhi: DEME provider installed successfully")
        set(MOPHI_EXTERNAL_DEME_AVAILABLE TRUE
            CACHE INTERNAL "DEME Python package available")
    else()
        message(WARNING
            "MoPhi: pip install ${MOPHI_PACKAGE_DEME_DISTRIBUTION} failed (exit code ${_deme_install_result}):\n"
            "${_deme_install_error}\n"
            "DEME will not be available for the coupler.  "
            "Install it manually with:  pip install ${MOPHI_PACKAGE_DEME_DISTRIBUTION}")
    endif()
else()
    if(MOPHI_FETCH_DEME AND MOPHI_PYTHON_PACKAGING_BUILD)
        message(STATUS "MoPhi: Ignoring MOPHI_FETCH_DEME during Python packaging build")
    endif()
    # Even without MOPHI_FETCH_DEME, detect a pre-installed DEME so the
    # summary line below reports the correct status.
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_FOUND)
        execute_process(
            COMMAND         ${Python3_EXECUTABLE} -c "import importlib; importlib.import_module('${MOPHI_PACKAGE_DEME_IMPORT_MODULE}')"
            RESULT_VARIABLE _deme_check_result
            OUTPUT_QUIET
            ERROR_QUIET
        )
        if(_deme_check_result EQUAL 0)
            set(MOPHI_EXTERNAL_DEME_AVAILABLE TRUE
                CACHE INTERNAL "DEME Python package available")
        else()
            set(MOPHI_EXTERNAL_DEME_AVAILABLE FALSE
                CACHE INTERNAL "DEME Python package not available")
        endif()
    else()
        set(MOPHI_EXTERNAL_DEME_AVAILABLE FALSE
            CACHE INTERNAL "DEME Python package not available")
    endif()
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
