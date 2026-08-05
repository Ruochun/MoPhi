# ─────────────────────────────────────────────────────────────────────────────
# src/couplers/newton_xlb_dem/CMakeLists.txt
#
# Registers the Newton + XLB + DEME three-way co-simulation coupler.
# This file is only included when MOPHI_BUILD_NEWTON_XLB_DEM=ON (guarded in the
# parent CMakeLists.txt).
#
# There is no static library to build here: the coupler logic lives in
# PyNewtonXLBDEMCoupler.{h,cpp} and is compiled directly into the mophi_core
# Python extension module by python/CMakeLists.txt.  This CMakeLists.txt
# only creates an INTERFACE target that propagates the required include and
# link requirements to mophi_core.
#
# Newton, XLB, and DEME are all pure Python packages — no C++ externals or
# CUDA Toolkit are required for this coupler.  Newton-based couplers require
# specific package constraints:
#   pip install --upgrade newton warp-lang mujoco==3.6.0
# plus XLB/DEME as needed:
#   pip install "xlb[cuda]" deme3
# or enable -DMOPHI_FETCH_NEWTON=ON / -DMOPHI_FETCH_XLB=ON /
# -DMOPHI_FETCH_DEME=ON at CMake configure time to have MoPhi install them
# automatically.
# ─────────────────────────────────────────────────────────────────────────────

# ── INTERFACE target ──────────────────────────────────────────────────────────
# Carries the include path for PyNewtonXLBDEMCoupler.h to mophi_core via the
# existing mophi_couplers INTERFACE chain.  No object files are produced.
add_library(mophi_coupler_newton_xlb_dem INTERFACE)

target_include_directories(mophi_coupler_newton_xlb_dem
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/mophi/couplers>
)

target_link_libraries(mophi_coupler_newton_xlb_dem
    INTERFACE
        mophi_essentials
)

target_link_libraries(mophi_couplers INTERFACE mophi_coupler_newton_xlb_dem)
