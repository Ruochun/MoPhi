# ─────────────────────────────────────────────────────────────────────────────
# src/couplers/CMakeLists.txt
#
# Builds one static library per co-simulation solver.  Each solver lives in
# its own subdirectory and has its own CMakeLists.txt.
#
# Adding a new co-simulation solver:
#   1. Create src/couplers/<name>/ with <NewCoupler>.h, .cpp, and CMakeLists.txt.
#   2. Add an if(MOPHI_BUILD_<NAME>) add_subdirectory(<name>) endif() block below.
#   3. Add the corresponding option() in the root CMakeLists.txt.
# ─────────────────────────────────────────────────────────────────────────────

# Aggregate interface target — always created so that downstream targets
# (Python bindings, demos, etc.) can link against it unconditionally.
# mophi_essentials is also linked here so that any consumer of mophi_couplers
# (e.g. the Python extension module) automatically inherits access to
# Logger and other MoPhiEssentials utilities.
# Note: individual coupler libraries must also link mophi_essentials directly
# in their own target_link_libraries() so that their own .cpp compilation has
# access to <core/Logger.hpp>.
add_library(mophi_couplers INTERFACE)
target_link_libraries(mophi_couplers INTERFACE mophi_essentials)

# ── TLFEA + DEM-Engine co-simulation solver ───────────────────────────────────
if(MOPHI_BUILD_TLFEA_DEM)
    add_subdirectory(tlfea_dem)
endif()

# ── TLFEA + Newton co-simulation coupler ──────────────────────────────────────
if(MOPHI_BUILD_TLFEA_NEWTON)
    add_subdirectory(tlfea_newton)
endif()

# ── Newton + XLB + DEM-Engine three-way co-simulation coupler ─────────────────
if(MOPHI_BUILD_NEWTON_XLB_DEM)
    add_subdirectory(newton_xlb_dem)
endif()

