# ─────────────────────────────────────────────────────────────────────────────
# src/utils/CMakeLists.txt
#
# MoPhi utility helpers — pure Python modules that are solver-agnostic and
# usable both inside couplers and directly by end-user scripts / demos.
#
# CMake copies them into the Python package directory (python/mophi/) at
# configure time so that `import mophi` works directly from the build tree
# without any additional PYTHONPATH manipulation.  The existing install rule in
# python/CMakeLists.txt (install(DIRECTORY mophi/ ...)) picks up the copied
# files automatically at install time.
# ─────────────────────────────────────────────────────────────────────────────

# ── Copy utility Python sources into the mophi package ───────────────────────
# configure_file with COPYONLY reproduces the file verbatim in the destination.
# Destination is the source-tree Python package directory so that build-tree
# imports (e.g. `python demo/newton_xlb_dem/demo_newton_xlb_dem.py`) resolve
# `mophi.xlb_build_streamlines` without requiring a `cmake --install` step first.
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/xlb_helpers.py"
    "${CMAKE_SOURCE_DIR}/python/mophi/xlb_helpers.py"
    COPYONLY
)
