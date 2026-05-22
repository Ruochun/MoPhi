cmake_minimum_required(VERSION 3.18)
project(MoPhi VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Required Python dependency constraints for Newton-based couplers/demos.
set(MOPHI_REQUIRED_NEWTON_VERSION "1.0.0" CACHE INTERNAL "Required Newton Python package version")
set(MOPHI_REQUIRED_WARP_VERSION "1.12.1" CACHE INTERNAL "Required Warp Python package version")
set(MOPHI_REQUIRED_MUJOCO_VERSION "3.6.0" CACHE INTERNAL "Required MuJoCo Python package version")

# ─────────────────────────────────────────────────────────────────────────────
# Options: external project fetching
#
# Each option controls whether the named external solver is downloaded into
# external/<name>/ during the CMake configure step.  New solvers should be
# registered in external/ExternalProjects.cmake and given a corresponding
# option here.
# ─────────────────────────────────────────────────────────────────────────────
option(MOPHI_FETCH_FERIS     "Download/update the FERIS external project"     OFF)
option(MOPHI_FETCH_DEMENGINE "Download/update the DEM-Engine external project" OFF)
option(MOPHI_FETCH_NEWTON    "Install required Newton + Warp + MuJoCo Python packages (newton==${MOPHI_REQUIRED_NEWTON_VERSION}, warp-lang==${MOPHI_REQUIRED_WARP_VERSION}, mujoco==${MOPHI_REQUIRED_MUJOCO_VERSION})" OFF)
option(MOPHI_FETCH_XLB       "Install the XLB Python package (pip install \"xlb[cuda]\")"       OFF)
option(MOPHI_FETCH_DEME      "Install the DEME Python package (pip install deme)"     OFF)

# ─────────────────────────────────────────────────────────────────────────────
# Options: co-simulation solvers to build
#
# Each option enables compilation of a specific multi-physics co-simulation
# solver.  The corresponding external projects must have been fetched first
# (via the MOPHI_FETCH_* options above); CMake emits a FATAL_ERROR at configure
# time if a required external is missing.
# ─────────────────────────────────────────────────────────────────────────────
option(MOPHI_BUILD_FERIS_NEWTON "Build the FERIS + Newton co-simulation coupler"     OFF)
option(MOPHI_BUILD_NEWTON_XLB_DEM "Build the Newton + XLB + DEM-Engine three-way co-simulation coupler" OFF)

# ─────────────────────────────────────────────────────────────────────────────
# Options: Python bindings
# ─────────────────────────────────────────────────────────────────────────────
option(MOPHI_BUILD_PYTHON_BINDINGS "Build Python bindings via pybind11" ON)

# ─────────────────────────────────────────────────────────────────────────────
# Options: demos
# ─────────────────────────────────────────────────────────────────────────────
option(MOPHI_BUILD_DEMOS "Build demonstration programs" ON)

# ─────────────────────────────────────────────────────────────────────────────
# Options: Python packaging
# ─────────────────────────────────────────────────────────────────────────────
option(MOPHI_PYTHON_PACKAGING_BUILD "Configure the supported Python packaging build (Linux wheel, Python bindings ON, Newton+XLB+DEME ON, FERIS OFF, demos OFF)" OFF)

if(MOPHI_PYTHON_PACKAGING_BUILD)
    set(MOPHI_FETCH_FERIS OFF CACHE BOOL "Download/update the FERIS external project" FORCE)
    set(MOPHI_FETCH_DEMENGINE OFF CACHE BOOL "Download/update the DEM-Engine external project" FORCE)
    set(MOPHI_FETCH_NEWTON OFF CACHE BOOL "Install required Newton + Warp + MuJoCo Python packages (newton==${MOPHI_REQUIRED_NEWTON_VERSION}, warp-lang==${MOPHI_REQUIRED_WARP_VERSION}, mujoco==${MOPHI_REQUIRED_MUJOCO_VERSION})" FORCE)
    set(MOPHI_FETCH_XLB OFF CACHE BOOL "Install the XLB Python package (pip install \"xlb[cuda]\")" FORCE)
    set(MOPHI_FETCH_DEME OFF CACHE BOOL "Install the DEME Python package (pip install deme)" FORCE)
    set(MOPHI_BUILD_FERIS_NEWTON OFF CACHE BOOL "Build the FERIS + Newton co-simulation coupler" FORCE)
    set(MOPHI_BUILD_NEWTON_XLB_DEM ON CACHE BOOL "Build the Newton + XLB + DEM-Engine three-way co-simulation coupler" FORCE)
    set(MOPHI_BUILD_PYTHON_BINDINGS ON CACHE BOOL "Build Python bindings via pybind11" FORCE)
    set(MOPHI_BUILD_DEMOS OFF CACHE BOOL "Build demonstration programs" FORCE)
endif()

# ─────────────────────────────────────────────────────────────────────────────
# CUDA Toolkit
#
# Searched once here so the result is available to all sub-directories.
# Co-simulation solvers that depend on CUDA (e.g. FERISNewtonCoupler) emit a
# CMake FATAL_ERROR at configure time when CUDAToolkit is not found rather
# than deferring the failure to a cryptic link-time "undefined reference".
# ─────────────────────────────────────────────────────────────────────────────
find_package(CUDAToolkit QUIET)

# ─────────────────────────────────────────────────────────────────────────────
# Eigen3
#
# Searched once here so the result is available to all sub-directories.
# Co-simulation solvers that depend on Eigen (e.g. FERISNewtonCoupler via FERIS)
# emit a CMake FATAL_ERROR at configure time when Eigen3 is not found.
# ─────────────────────────────────────────────────────────────────────────────
find_package(Eigen3 QUIET)

# ─────────────────────────────────────────────────────────────────────────────
# Infrastructure
# ─────────────────────────────────────────────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(MoPhiExternalUtils)

# ─────────────────────────────────────────────────────────────────────────────
# MoPhiEssentials submodule (required, always on)
#
# MoPhiEssentials provides foundational utilities — Logger, Real3, memory
# helpers — shared across all MoPhi-compatible solvers.  It is a required
# git submodule (external/MoPhiEssentials) and is always included.
#
# CUDA is explicitly disabled here: MoPhi manages CUDA resources per-coupler
# via the MOPHI_FETCH_* / MOPHI_BUILD_* mechanism.  The CPU-only build of
# MoPhiEssentials produces a header-only INTERFACE target (mophi_essentials)
# that carries the correct include directories and requires no separate build
# step.
#
# Does not interfere with solvers that also use MoPhiEssentials internally
# (e.g. FERIS) — those are built in complete isolation via ExternalProject_Add
# and have no CMake target visibility into this parent build.
# ─────────────────────────────────────────────────────────────────────────────
set(MOPHI_ENABLE_CUDA OFF CACHE BOOL "Disable CUDA inside the MoPhiEssentials submodule (MoPhi manages CUDA per-coupler)" FORCE)
add_subdirectory(external/MoPhiEssentials)

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
