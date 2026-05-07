# ─────────────────────────────────────────────────────────────────────────────
# src/visualization/CMakeLists.txt
#
# The MoPhi visualization module is a pure Python package — no C++ compilation
# is required.  Source Python files live here (src/visualization/) to keep them
# co-located with other MoPhi source modules.
#
# The canonical Python package copies live in python/mophi/visualizers/ and
# must be kept in sync with the src/visualization/ sources manually (or via
# editor tooling).  No CMake copy step is performed so that the build never
# silently overwrites the package tree.
# ─────────────────────────────────────────────────────────────────────────────
