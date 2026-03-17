# ─────────────────────────────────────────────────────────────────────────────
# src/couplers/CMakeLists.txt
#
# Builds one static library per co-simulation solver.
#
# Adding a new co-simulation solver:
#   1. Add <NewCoupler>.h / .cpp to this directory.
#   2. Add a new add_library() + target_link_libraries() block below.
#   3. Add the new library to the mophi_couplers interface target so the rest
#      of the build tree (and Python bindings) can find it.
#   4. If it requires specific externals, call mophi_require_externals() to
#      emit a helpful error when those externals are not yet fetched.
# ─────────────────────────────────────────────────────────────────────────────

# ── TLFEA + DEM-Engine placeholder co-simulation solver ───────────────────────
if(MOPHI_BUILD_TLFEA_DEM)
    # Optional: enforce that both externals are present when the real solver
    # is requested.  Comment out for stub/placeholder mode.
    # mophi_require_externals(TLFEADEMCoupler TLFEA DEMEngine)
endif()

add_library(mophi_coupler_tlfea_dem STATIC
    TLFEADEMCoupler.cpp
)

# Required when the static library is linked into a shared library
# (e.g., the Python extension module).
set_target_properties(mophi_coupler_tlfea_dem PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(mophi_coupler_tlfea_dem
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/mophi/couplers>
)

target_link_libraries(mophi_coupler_tlfea_dem
    PUBLIC mophi_wrappers
)

# ── Aggregate interface target ────────────────────────────────────────────────
# All targets that need co-simulation solvers should link against mophi_couplers.
add_library(mophi_couplers INTERFACE)
target_link_libraries(mophi_couplers INTERFACE mophi_coupler_tlfea_dem)
