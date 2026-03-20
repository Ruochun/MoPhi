# ─────────────────────────────────────────────────────────────────────────────
# src/couplers/newton_xlb_dem/CMakeLists.txt
#
# Registers the Newton + XLB + DEM-Engine three-way co-simulation coupler.
# This file is only included when MOPHI_BUILD_NEWTON_XLB_DEM=ON (guarded in the
# parent CMakeLists.txt).
#
# There is no static library to build here: the coupler logic lives in
# PyNewtonXLBDEMCoupler.{h,cpp} and is compiled directly into the mophi_core
# Python extension module by python/CMakeLists.txt.  This CMakeLists.txt
# only performs prerequisite checks and creates an INTERFACE target that
# propagates the required link and include requirements to mophi_core.
# ─────────────────────────────────────────────────────────────────────────────

# DEM-Engine must be registered (fetched) at configure time.
# mophi_require_externals emits a FATAL_ERROR with actionable instructions
# if DEMEngine was not enabled via -DMOPHI_FETCH_DEMENGINE=ON.
mophi_require_externals(NewtonXLBDEMCoupler DEMEngine)

# DEM-Engine is a CUDA-based library — its headers and runtime require the
# CUDA Toolkit.
if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR
        "MoPhi: Building NewtonXLBDEMCoupler requires the CUDA Toolkit "
        "(DEM-Engine is a CUDA-based DEM library), "
        "but CMake could not find it.\n\n"
        "  • Install CUDA Toolkit and ensure 'nvcc' is on PATH, or\n"
        "  • Set CUDAToolkit_ROOT to the CUDA installation directory.\n")
endif()

# Newton and XLB are pure Python packages — no C++ linking is required for them.
# PyNewtonXLBDEMCoupler.cpp is compiled directly into the mophi_core Python
# extension module (see python/CMakeLists.txt).

# ── INTERFACE target ──────────────────────────────────────────────────────────
# Carries the DEM-Engine + CUDA link and include requirements to mophi_core via
# the existing mophi_couplers INTERFACE chain.  No object files are produced.
add_library(mophi_coupler_newton_xlb_dem INTERFACE)

target_include_directories(mophi_coupler_newton_xlb_dem
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/mophi/couplers>
)

# CUDA::cudart provides the CUDA runtime (cudaMalloc, cudaMemcpy, …) and makes
# cuda_runtime.h available so DEM-Engine's __host__/__device__ headers compile
# with the plain C++ compiler.
# CUDA::cuda_driver covers driver-API symbols used by DEM-Engine.
# CUDA::nvrtc is needed for DEM-Engine's JIT kernel compilation.
target_link_libraries(mophi_coupler_newton_xlb_dem
    INTERFACE
        mophi_essentials
        mophi_ext_DEMEngine
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::nvrtc
)

target_link_libraries(mophi_couplers INTERFACE mophi_coupler_newton_xlb_dem)
