# ─────────────────────────────────────────────────────────────────────────────
# src/utils/CMakeLists.txt
#
# MoPhi utility helpers — pure Python modules that are solver-agnostic and
# usable both inside couplers and directly by end-user scripts / demos.
#
# The canonical Python package copies live in python/mophi/utils/ and must
# be kept in sync with the src/utils/ sources manually (or via editor tooling).
# No CMake copy step is performed so that the build never silently overwrites
# the package tree.
# ─────────────────────────────────────────────────────────────────────────────
