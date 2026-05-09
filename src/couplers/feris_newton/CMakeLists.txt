# ─────────────────────────────────────────────────────────────────────────────
# src/couplers/feris_newton/CMakeLists.txt
#
# Registers the FERIS + Newton co-simulation coupler.
# This file is only included when MOPHI_BUILD_FERIS_NEWTON=ON (guarded in the
# parent CMakeLists.txt).
#
# There is no static library to build here: the coupler logic lives in
# PyFERISNewtonCoupler.{h,cpp} and is compiled directly into the mophi_core
# Python extension module by python/CMakeLists.txt.  This CMakeLists.txt
# only performs prerequisite checks and creates an INTERFACE target that
# propagates the required link and include requirements to mophi_core.
# ─────────────────────────────────────────────────────────────────────────────

# FERIS must be registered (fetched) at configure time.
# mophi_require_externals emits a FATAL_ERROR with actionable instructions
# if FERIS was not enabled via -DMOPHI_FETCH_FERIS=ON.
mophi_require_externals(FERISNewtonCoupler FERIS)

# FERIS is a CUDA-based FEA library — its headers annotate functions with
# __host__ and __device__, which require the CUDA Toolkit to compile.
# Catch a missing CUDA installation here with an actionable message rather
# than letting the build fail later with cryptic "__host__ does not name a
# type" errors from FERIS headers.
if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR
        "MoPhi: Building FERISNewtonCoupler requires the CUDA Toolkit "
        "(FERIS is a CUDA-based FEA library whose headers use __host__/__device__), "
        "but CMake could not find it.\n\n"
        "  • Install CUDA Toolkit and ensure 'nvcc' is on PATH, or\n"
        "  • Set CUDAToolkit_ROOT to the CUDA installation directory.\n")
endif()

# FERIS is an Eigen-based FEA library — catch a missing Eigen installation
# here with an actionable message rather than a cryptic "file not found" for
# Eigen headers.
if(NOT Eigen3_FOUND)
    message(FATAL_ERROR
        "MoPhi: Building FERISNewtonCoupler requires Eigen3 "
        "(FERIS is an Eigen-based FEA library), but CMake could not find it.\n\n"
        "  • Install Eigen3 (e.g. 'apt install libeigen3-dev' on Debian/Ubuntu), or\n"
        "  • Set Eigen3_DIR to the directory containing Eigen3Config.cmake.\n")
endif()

# Newton is a pure Python package — no C++ linking is required.
# PyFERISNewtonCoupler.cpp is compiled directly into the mophi_core Python
# extension module (see python/CMakeLists.txt).

# ── INTERFACE target ──────────────────────────────────────────────────────────
# Carries the FERIS + CUDA link and include requirements to mophi_core via the
# existing mophi_couplers INTERFACE chain.  No object files are produced here.
add_library(mophi_coupler_feris_newton INTERFACE)

target_include_directories(mophi_coupler_feris_newton
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/mophi/couplers>
)

# FERIS's installed .cuh headers include <MoPhiEssentials.h> and rely on
# MOPHI_GPU_CALL and DualArray, which are only compiled when MOPHI_USE_CUDA is
# defined.  MoPhi's own mophi_essentials submodule is built CPU-only
# (MOPHI_ENABLE_CUDA=OFF FORCE) so it never adds that definition.  We must
# propagate it as INTERFACE so that mophi_core (the Python extension module
# that compiles PyFERISNewtonCoupler.cpp) also sees it.
target_compile_definitions(mophi_coupler_feris_newton INTERFACE MOPHI_USE_CUDA)

# CUDA::cusparse and CUDA::cublas are required because the FERIS solver and
# element .cuh headers include <cusparse.h> and <cublas_v2.h> respectively.
# CUDA::cuda_driver covers driver-API symbols used by FERIS.
# CUDA::nvrtc is NOT required here — that is specific to DEM-Engine's JIT
# kernel compilation and is not used by FERIS.
target_link_libraries(mophi_coupler_feris_newton
    INTERFACE
        mophi_essentials
        mophi_ext_FERIS
        Eigen3::Eigen
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::cusparse
        CUDA::cublas
)

target_link_libraries(mophi_couplers INTERFACE mophi_coupler_feris_newton)
