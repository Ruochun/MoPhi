# ─────────────────────────────────────────────────────────────────────────────
# demo/tlfea_newton/CMakeLists.txt
#
# The TLFEA + Newton demo is a pure Python script.  No C++ executable is
# built here; the demo is installed alongside the MoPhi Python package so
# that it can be run with:
#
#   python demo/tlfea_newton/demo_tlfea_newton.py
#
# Prerequisites:
#   • MOPHI_BUILD_TLFEA_NEWTON=ON   — builds the TLFEANewtonCoupler library and
#                                     the mophi_core Python extension module.
#   • python3 -m pip install newton  — or -DMOPHI_FETCH_NEWTON=ON at configure
#                                     time to have MoPhi install it automatically.
# ─────────────────────────────────────────────────────────────────────────────

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# ── Discover site-packages from the selected Python interpreter ───────────────
# When the demo is run via `cmake --build ... --target demo_tlfea_newton`,
# PYTHONPATH is not inherited from the shell.  We construct it explicitly so
# that both the MoPhi Python extension module (in the build tree) and any
# pip-installed packages (newton, warp) are importable.
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "import site; sp = getattr(site,'getsitepackages',lambda:[])(); print(':'.join(sp + [site.getusersitepackages()]))"
    OUTPUT_VARIABLE _demo_site_packages
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

set(_demo_pythonpath "${CMAKE_BINARY_DIR}/python")
if(_demo_site_packages)
    string(APPEND _demo_pythonpath ":${_demo_site_packages}")
endif()
if(DEFINED ENV{PYTHONPATH} AND NOT "$ENV{PYTHONPATH}" STREQUAL "")
    string(APPEND _demo_pythonpath ":$ENV{PYTHONPATH}")
endif()

# ── Custom target to run the demo ─────────────────────────────────────────────
# Adds a `demo_tlfea_newton` build target.  Run with:
#   cmake --build <build-dir> --target demo_tlfea_newton
add_custom_target(demo_tlfea_newton
    COMMAND ${CMAKE_COMMAND} -E env
        "PYTHONPATH=${_demo_pythonpath}"
        ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/demo_tlfea_newton.py"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Running TLFEA + Newton co-simulation demo"
    VERBATIM
)

# ── Installation ───────────────────────────────────────────────────────────────
install(FILES demo_tlfea_newton.py DESTINATION demo/tlfea_newton)
