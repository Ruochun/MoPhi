# ─────────────────────────────────────────────────────────────────────────────
# src/couplers/CMakeLists.txt
#
# Builds one static library per co-simulation solver.
#
# Adding a new co-simulation solver:
#   1. Add <NewCoupler>.h / .cpp to this directory.
#   2. Add a new if(MOPHI_BUILD_<NAME>) block below that:
#      a) Calls mophi_require_externals() to abort at CMake time if any
#         required external has not been registered.
#      b) Defines the library, links it against the mophi_ext_* IMPORTED
#         targets (which also carry the correct include directories), and
#         adds it to mophi_couplers.
#   3. Add the corresponding option() in the root CMakeLists.txt.
# ─────────────────────────────────────────────────────────────────────────────

# Aggregate interface target — always created so that downstream targets
# (Python bindings, etc.) can link against it unconditionally.
add_library(mophi_couplers INTERFACE)

# ── TLFEA + DEM-Engine co-simulation solver ───────────────────────────────────
if(MOPHI_BUILD_TLFEA_DEM)
    # Both externals must be registered (fetched) at configure time.
    # mophi_require_externals emits a FATAL_ERROR with actionable instructions
    # if either external was not enabled via -DMOPHI_FETCH_*=ON.
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
    )

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
    target_link_libraries(mophi_coupler_tlfea_dem
        PUBLIC
            mophi_ext_DEMEngine
            mophi_ext_TLFEA
    )

    target_link_libraries(mophi_couplers INTERFACE mophi_coupler_tlfea_dem)
endif()
