# ─────────────────────────────────────────────────────────────────────────────
# src/couplers/tlfea_newton/CMakeLists.txt
#
# Builds the TLFEA + Newton co-simulation coupler library.
# This file is only included when MOPHI_BUILD_TLFEA_NEWTON=ON (guarded in the
# parent CMakeLists.txt).
# ─────────────────────────────────────────────────────────────────────────────

# TLFEA must be registered (fetched) at configure time.
# mophi_require_externals emits a FATAL_ERROR with actionable instructions
# if TLFEA was not enabled via -DMOPHI_FETCH_TLFEA=ON.
mophi_require_externals(TLFEANewtonCoupler TLFEA)

# TLFEA is a CUDA-based FEA library — its headers annotate functions with
# __host__ and __device__, which require the CUDA Toolkit to compile.
# Catch a missing CUDA installation here with an actionable message rather
# than letting the build fail later with cryptic "__host__ does not name a
# type" errors from TLFEA headers.
if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR
        "MoPhi: Building TLFEANewtonCoupler requires the CUDA Toolkit "
        "(TLFEA is a CUDA-based FEA library whose headers use __host__/__device__), "
        "but CMake could not find it.\n\n"
        "  • Install CUDA Toolkit and ensure 'nvcc' is on PATH, or\n"
        "  • Set CUDAToolkit_ROOT to the CUDA installation directory.\n")
endif()

# TLFEA is an Eigen-based FEA library — catch a missing Eigen installation
# here with an actionable message rather than a cryptic "file not found" for
# Eigen headers.
if(NOT Eigen3_FOUND)
    message(FATAL_ERROR
        "MoPhi: Building TLFEANewtonCoupler requires Eigen3 "
        "(TLFEA is an Eigen-based FEA library), but CMake could not find it.\n\n"
        "  • Install Eigen3 (e.g. 'apt install libeigen3-dev' on Debian/Ubuntu), or\n"
        "  • Set Eigen3_DIR to the directory containing Eigen3Config.cmake.\n")
endif()

# Newton is a pure Python package — no C++ linking is required.
# PyTLFEANewtonCoupler (the Python-facing wrapper that bridges TLFEA and Newton)
# lives in PyTLFEANewtonCoupler.h alongside this directory and is compiled as
# part of the mophi_core Python extension module.
# Users must have Newton installed: pip install newton, or enable
# -DMOPHI_FETCH_NEWTON=ON to have MoPhi install it automatically.

add_library(mophi_coupler_tlfea_newton STATIC
    TLFEANewtonCoupler.cpp
)

# Required when the static library is linked into a shared library
# (e.g., the Python extension module).
set_target_properties(mophi_coupler_tlfea_newton PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_features(mophi_coupler_tlfea_newton PUBLIC cxx_std_17)

target_include_directories(mophi_coupler_tlfea_newton
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/mophi/couplers>
)

# PUBLIC (not PRIVATE) is required here because mophi_coupler_tlfea_newton is
# a static library: any executable or shared library that links against it must
# also receive the external libraries on its own link line.
#
# CUDA::cudart provides the CUDA runtime (cudaMalloc, cudaMemcpy, …) that
# TLFEA's static library depends on, and also makes cuda_runtime.h available
# on the include path so that TLFEA's __host__/__device__ annotated headers
# compile correctly with the plain C++ compiler.
# CUDA::cuda_driver covers driver-API symbols used by TLFEA.
# CUDA::nvrtc is NOT required here — that is specific to DEM-Engine's JIT
# kernel compilation and is not used by TLFEA.
target_link_libraries(mophi_coupler_tlfea_newton
    PUBLIC
        mophi_essentials
        mophi_ext_TLFEA
        Eigen3::Eigen
        CUDA::cudart
        CUDA::cuda_driver
)

target_link_libraries(mophi_couplers INTERFACE mophi_coupler_tlfea_newton)
