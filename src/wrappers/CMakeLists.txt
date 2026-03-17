# ─────────────────────────────────────────────────────────────────────────────
# src/wrappers/CMakeLists.txt
#
# Builds the mophi_wrappers static library, which wraps each supported external
# solver with a uniform MoPhi C++ interface.
#
# Adding a new solver wrapper:
#   1. Add its .h / .cpp files to this directory.
#   2. Add the source files to the target_sources() call below.
#   3. If the solver needs to be fetched, add a conditional block that:
#      a) Calls mophi_fetch_external() or relies on the external/ step.
#      b) Adds the solver's include path and link libraries.
#      c) Passes -DMOPHI_HAS_<SOLVER> via target_compile_definitions().
# ─────────────────────────────────────────────────────────────────────────────

add_library(mophi_wrappers STATIC
    TLFEAWrapper.cpp
    DEMEngineWrapper.cpp
)

# Required when the static library is linked into a shared library
# (e.g., the Python extension module).
set_target_properties(mophi_wrappers PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(mophi_wrappers
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/mophi/wrappers>
)

target_compile_features(mophi_wrappers PUBLIC cxx_std_17)

# ── TLFEA integration ─────────────────────────────────────────────────────────
if(MOPHI_EXTERNAL_TLFEA_AVAILABLE)
    target_compile_definitions(mophi_wrappers PUBLIC MOPHI_HAS_TLFEA)
    target_include_directories(mophi_wrappers
        PUBLIC "${CMAKE_SOURCE_DIR}/external/TLFEA/src")
    # Uncomment and adjust once TLFEA exposes a CMake target:
    # target_link_libraries(mophi_wrappers PUBLIC tlfea)
endif()

# ── DEM-Engine integration ────────────────────────────────────────────────────
if(MOPHI_EXTERNAL_DEMEngine_AVAILABLE)
    target_compile_definitions(mophi_wrappers PUBLIC MOPHI_HAS_DEMENGINE)
    target_include_directories(mophi_wrappers
        PUBLIC "${CMAKE_SOURCE_DIR}/external/DEMEngine/src")
    # Uncomment and adjust once DEM-Engine exposes a CMake target:
    # target_link_libraries(mophi_wrappers PUBLIC DEM-Engine)
endif()
