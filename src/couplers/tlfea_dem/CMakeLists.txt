# ─────────────────────────────────────────────────────────────────────────────
# src/couplers/tlfea_dem/CMakeLists.txt
#
# Builds the TLFEA + DEM-Engine co-simulation solver library.
# This file is only included when MOPHI_BUILD_TLFEA_DEM=ON (guarded in the
# parent CMakeLists.txt).
# ─────────────────────────────────────────────────────────────────────────────

# Both externals must be registered (fetched) at configure time.
# mophi_require_externals emits a FATAL_ERROR with actionable instructions
# if either external was not enabled via -DMOPHI_FETCH_*=ON.
mophi_require_externals(TLFEADEMCoupler TLFEA DEMEngine)

# DEM-Engine is a CUDA library — its installed static library contains
# CUDA runtime calls (cudaMalloc, cudaMemcpy, …).  MoPhi must therefore
# find the CUDA Toolkit and link CUDA::cudart into every consumer.
# Catch a missing CUDA installation here with an actionable message rather
# than letting the build fail later with cryptic "undefined reference"
# errors for cudaMalloc and friends.
if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR
        "MoPhi: Building TLFEADEMCoupler requires the CUDA Toolkit "
        "(DEM-Engine is a CUDA library), but CMake could not find it.\n\n"
        "  • Install CUDA Toolkit and ensure 'nvcc' is on PATH, or\n"
        "  • Set CUDAToolkit_ROOT to the CUDA installation directory.\n")
endif()

# TLFEA is an Eigen-based FEA library — it uses Eigen heavily for linear
# algebra (matrices, vectors, sparse solvers).  Catch a missing Eigen
# installation here with an actionable message rather than letting the build
# fail later with cryptic "file not found" errors for Eigen headers.
if(NOT Eigen3_FOUND)
    message(FATAL_ERROR
        "MoPhi: Building TLFEADEMCoupler requires Eigen3 "
        "(TLFEA is an Eigen-based FEA library), but CMake could not find it.\n\n"
        "  • Install Eigen3 (e.g. 'apt install libeigen3-dev' on Debian/Ubuntu), or\n"
        "  • Set Eigen3_DIR to the directory containing Eigen3Config.cmake.\n")
endif()

add_library(mophi_coupler_tlfea_dem STATIC
    TLFEADEMCoupler.cpp
)

# Required when the static library is linked into a shared library
# (e.g., the Python extension module).
set_target_properties(mophi_coupler_tlfea_dem PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_features(mophi_coupler_tlfea_dem PUBLIC cxx_std_17)

target_include_directories(mophi_coupler_tlfea_dem
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/mophi/couplers>
)

# TLFEA's installed .cuh headers include <MoPhiEssentials.h> and rely on
# MOPHI_GPU_CALL and DualArray, which are only compiled when MOPHI_USE_CUDA is
# defined.  MoPhi's own mophi_essentials submodule is built CPU-only
# (MOPHI_ENABLE_CUDA=OFF FORCE) so it never adds that definition.  We must add
# it here so that the TLFEA headers compile correctly in this translation unit.
# PRIVATE is sufficient because the pimpl design hides all TLFEA headers from
# the public TLFEADEMCoupler.h — consumers of this library never see them.
target_compile_definitions(mophi_coupler_tlfea_dem PRIVATE MOPHI_USE_CUDA)

# Link against the IMPORTED targets created by mophi_fetch_external().
# Each target carries:
#   - INTERFACE_INCLUDE_DIRECTORIES pointing to <external>/install/include
#     (which contains all generated and source headers after the external
#     has been built and installed)
#   - IMPORTED_LOCATION pointing to the installed static library
# The add_dependencies() inside mophi_fetch_external() ensures the
# external is fully built before this coupler is compiled.
#
# PUBLIC (not PRIVATE) is required here because mophi_coupler_tlfea_dem is
# a static library: its object files do not absorb the symbols from the
# external libraries.  Any executable or shared library that links against
# mophi_coupler_tlfea_dem must also receive the external libraries on its
# own link line — PUBLIC propagates them automatically.
#
# CUDA::cudart provides the CUDA runtime (cudaMalloc, cudaMemcpy, …)
# that DEM-Engine's static library depends on.  CUDA::cuda_driver covers
# driver-API symbols.  CUDA::nvrtc is required for DEM-Engine's JIT kernel
# compilation (nvrtcGetLoweredName, nvrtcCompileProgram, etc.).
# CUDA::cusparse and CUDA::cublas are required because the TLFEA solver and
# element .cuh headers include <cusparse.h> and <cublas_v2.h> respectively.
# All are propagated PUBLIC for the same reason.
target_link_libraries(mophi_coupler_tlfea_dem
    PUBLIC
        mophi_essentials
        mophi_ext_DEMEngine
        mophi_ext_TLFEA
        Eigen3::Eigen
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::nvrtc
        CUDA::cusparse
        CUDA::cublas
)

target_link_libraries(mophi_couplers INTERFACE mophi_coupler_tlfea_dem)
