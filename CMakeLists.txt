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
# solver.  The solver is always compiled (even as a stub) so the Python
# bindings remain available; this flag is reserved for future use when solver
# selection should control compilation of heavyweight components.
# ─────────────────────────────────────────────────────────────────────────────
option(MOPHI_BUILD_TLFEA_DEM "Build the TLFEA + DEM-Engine co-simulation solver" ON)

# ─────────────────────────────────────────────────────────────────────────────
# Options: Python bindings
# ─────────────────────────────────────────────────────────────────────────────
option(MOPHI_BUILD_PYTHON_BINDINGS "Build Python bindings via pybind11" ON)

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
# Core library (wrappers + co-simulation couplers)
# ─────────────────────────────────────────────────────────────────────────────
add_subdirectory(src)

# ─────────────────────────────────────────────────────────────────────────────
# Python bindings
# ─────────────────────────────────────────────────────────────────────────────
if(MOPHI_BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()
