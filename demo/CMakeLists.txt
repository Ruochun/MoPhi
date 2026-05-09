# ─────────────────────────────────────────────────────────────────────────────
# demo/CMakeLists.txt
#
# Collects all MoPhi demonstration programs.  Each demo lives in its own
# subdirectory and is only added when the co-simulation solver it exercises
# has been enabled.
#
# Adding a new demo:
#   1. Create demo/<name>/<name>.cpp (and any supporting files).
#   2. Add an if(MOPHI_BUILD_<SOLVER>) block below that calls
#      add_subdirectory(demo/<name>).
# ─────────────────────────────────────────────────────────────────────────────

# ── FERIS + Newton demo ───────────────────────────────────────────────────────
if(MOPHI_BUILD_FERIS_NEWTON)
    add_subdirectory(feris_newton)
endif()

# ── Newton + XLB + DEM-Engine three-way co-simulation demo ────────────────────
if(MOPHI_BUILD_NEWTON_XLB_DEM)
    add_subdirectory(newton_xlb_dem)
endif()
