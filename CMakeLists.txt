cmake_minimum_required(VERSION 3.18)
project(MoPhi VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─────────────────────────────────────────────────────────────────────────────
# Options: external project fetching
#
# Each option controls whether the named external solver is downloaded into
# external/<name>/ during the CMake configure step.  New solvers should be
# registered in external/ExternalProjects.cmake and given a corresponding
# option here.
# ─────────────────────────────────────────────────────────────────────────────
option(MOPHI_FETCH_TLFEA     "Download/update the TLFEA external project"     OFF)
option(MOPHI_FETCH_DEMENGINE "Download/update the DEM-Engine external project" OFF)

# ─────────────────────────────────────────────────────────────────────────────
# Options: co-simulation solvers to build
#
# Each option enables compilation of a specific multi-physics co-simulation
# solver.  The corresponding external projects must have been fetched first
# (via the MOPHI_FETCH_* options above); CMake emits a FATAL_ERROR at configure
# time if a required external is missing.
# ─────────────────────────────────────────────────────────────────────────────
option(MOPHI_BUILD_TLFEA_DEM "Build the TLFEA + DEM-Engine co-simulation solver" OFF)

# ─────────────────────────────────────────────────────────────────────────────
# Options: Python bindings
# ─────────────────────────────────────────────────────────────────────────────
option(MOPHI_BUILD_PYTHON_BINDINGS "Build Python bindings via pybind11" ON)

# ─────────────────────────────────────────────────────────────────────────────
# Options: demos
# ─────────────────────────────────────────────────────────────────────────────
option(MOPHI_BUILD_DEMOS "Build demonstration programs" ON)

# ─────────────────────────────────────────────────────────────────────────────
# CUDA Toolkit
#
# Searched once here so the result is available to all sub-directories.
# Co-simulation solvers that depend on CUDA (e.g. TLFEADEMCoupler) emit a
# CMake FATAL_ERROR at configure time when CUDAToolkit is not found rather
# than deferring the failure to a cryptic link-time "undefined reference".
# ─────────────────────────────────────────────────────────────────────────────
find_package(CUDAToolkit QUIET)

# ─────────────────────────────────────────────────────────────────────────────
# Infrastructure
# ─────────────────────────────────────────────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(MoPhiExternalUtils)

# ─────────────────────────────────────────────────────────────────────────────
# External projects
# ─────────────────────────────────────────────────────────────────────────────
add_subdirectory(external)

# ─────────────────────────────────────────────────────────────────────────────
# Core library (co-simulation couplers)
# ─────────────────────────────────────────────────────────────────────────────
add_subdirectory(src)

# ─────────────────────────────────────────────────────────────────────────────
# Python bindings
# ─────────────────────────────────────────────────────────────────────────────
if(MOPHI_BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Demos
# ─────────────────────────────────────────────────────────────────────────────
if(MOPHI_BUILD_DEMOS)
    add_subdirectory(demo)
endif()
