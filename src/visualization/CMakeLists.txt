# ─────────────────────────────────────────────────────────────────────────────
# src/visualization/CMakeLists.txt
#
# The MoPhi visualization module is a pure Python package — no C++ compilation
# is required.  Source Python files live here (src/visualization/) to keep them
# co-located with other MoPhi source modules.
#
# CMake copies them into the Python package directory (python/mophi/) at
# configure time so that `import mophi` works directly from the build tree
# without any additional PYTHONPATH manipulation.  The existing install rule in
# python/CMakeLists.txt (install(DIRECTORY mophi/ ...)) picks up the copied
# files automatically at install time.
# ─────────────────────────────────────────────────────────────────────────────

# ── Copy visualization Python sources into the mophi package ──────────────────
# configure_file with COPYONLY reproduces the file verbatim in the destination.
# Destination is the source-tree Python package directory so that build-tree
# imports (e.g. `python demo/newton_xlb_dem/demo_newton_xlb_dem.py`) resolve
# `mophi.OpenGLVisualizer` without requiring a `cmake --install` step first.
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/opengl_visualizer.py"
    "${CMAKE_SOURCE_DIR}/python/mophi/opengl_visualizer.py"
    COPYONLY
)
