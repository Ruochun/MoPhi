# ─────────────────────────────────────────────────────────────────────────────
# demo/newton_xlb_dem/CMakeLists.txt
#
# The Newton + XLB + DEME three-way co-simulation demo is a pure Python
# script.  No C++ executable is built here; the demo is installed alongside the
# MoPhi Python package so that it can be run with:
#
#   python demo/newton_xlb_dem/demo_newton_xlb_dem.py
#
# Prerequisites:
#   • MOPHI_BUILD_NEWTON_XLB_DEM=ON   — builds NewtonXLBDEMCoupler and the
#                                        mophi_core Python extension module.
#   • python3 -m pip install newton    — or -DMOPHI_FETCH_NEWTON=ON
#   • python3 -m pip install warp-lang — GPU runtime for Newton
#   • python3 -m pip install xlb       — or -DMOPHI_FETCH_XLB=ON  (optional)
#   • python3 -m pip install deme      — or -DMOPHI_FETCH_DEME=ON  (optional)
# ─────────────────────────────────────────────────────────────────────────────

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# ── Discover site-packages from the selected Python interpreter ───────────────
# When the demo is run via `cmake --build ... --target demo_newton_xlb_dem`,
# PYTHONPATH is not inherited from the shell.  We construct it explicitly so
# that both the MoPhi Python extension module (in the build tree) and any
# pip-installed packages (newton, xlb, deme, warp) are importable.
#
# site.getsitepackages() returns the system/venv site-packages directories;
# site.getusersitepackages() returns the per-user install location.  Including
# both ensures packages installed with or without --user are found.
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "import site; sp = getattr(site,'getsitepackages',lambda:[])(); print(':'.join(sp + [site.getusersitepackages()]))"
    OUTPUT_VARIABLE _demo_site_packages
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

# Assemble PYTHONPATH: MoPhi build output first, then discovered site-packages,
# then any existing PYTHONPATH from the shell environment.
set(_demo_pythonpath "${CMAKE_BINARY_DIR}/python")
if(_demo_site_packages)
    string(APPEND _demo_pythonpath ":${_demo_site_packages}")
endif()
if(DEFINED ENV{PYTHONPATH} AND NOT "$ENV{PYTHONPATH}" STREQUAL "")
    string(APPEND _demo_pythonpath ":$ENV{PYTHONPATH}")
endif()

# ── Custom target to run the demo ─────────────────────────────────────────────
# Adds a `demo_newton_xlb_dem` build target.  Run with:
#   cmake --build <build-dir> --target demo_newton_xlb_dem
add_custom_target(demo_newton_xlb_dem
    COMMAND ${CMAKE_COMMAND} -E env
        "PYTHONPATH=${_demo_pythonpath}"
        ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/demo_newton_xlb_dem.py"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Running Newton + XLB + DEME three-way co-simulation demo"
    VERBATIM
)

# ── Installation ───────────────────────────────────────────────────────────────
install(FILES demo_newton_xlb_dem.py DESTINATION demo/newton_xlb_dem)
