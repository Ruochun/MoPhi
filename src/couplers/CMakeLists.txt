# ─────────────────────────────────────────────────────────────────────────────
# src/couplers/CMakeLists.txt
#
# Builds one static library per co-simulation solver.
#
# Adding a new co-simulation solver:
#   1. Add <NewCoupler>.h / .cpp to this directory.
#   2. Add a new if(MOPHI_BUILD_<NAME>) block below that:
#      a) Calls mophi_require_externals() to abort at CMake time if any
#         required external has not been fetched.
#      b) Defines the library and links it into mophi_couplers.
#   3. Add the corresponding option() in the root CMakeLists.txt.
# ─────────────────────────────────────────────────────────────────────────────

# Aggregate interface target — always created so that downstream targets
# (Python bindings, etc.) can link against it unconditionally.
add_library(mophi_couplers INTERFACE)

# ── TLFEA + DEM-Engine co-simulation solver ───────────────────────────────────
if(MOPHI_BUILD_TLFEA_DEM)
    # Both externals must be present at configure time.  mophi_require_externals
    # emits a FATAL_ERROR with actionable instructions if either is missing.
    mophi_require_externals(TLFEADEMCoupler TLFEA DEMEngine)

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
        PRIVATE
            # DEM-Engine public headers live under src/DEM/ inside the repo.
            "${CMAKE_SOURCE_DIR}/external/DEMEngine/src"
            # TLFEA public headers live under src/ inside the repo.
            "${CMAKE_SOURCE_DIR}/external/TLFEA/src"
    )

    # Link the solver libraries once they expose CMake targets.
    # Uncomment and adjust when DEM-Engine / TLFEA provide installable targets:
    # target_link_libraries(mophi_coupler_tlfea_dem PUBLIC DEM-Engine tlfea)

    target_link_libraries(mophi_couplers INTERFACE mophi_coupler_tlfea_dem)
endif()

