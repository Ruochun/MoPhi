# ─────────────────────────────────────────────────────────────────────────────
# src/couplers/CMakeLists.txt
#
# Registers each co-simulation coupler.  Each coupler lives in its own
# subdirectory and has its own CMakeLists.txt.  Python-only couplers produce
# an INTERFACE target (no object files); pure C++ couplers produce a STATIC
# library.
#
# Adding a new co-simulation coupler:
#   1. Create src/couplers/<name>/ with <NewCoupler>.{h,cpp} and CMakeLists.txt.
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

# ── Shared Python prerequisites for Newton-based couplers ─────────────────────
# FERISNewtonCoupler and NewtonXLBDEMCoupler both require pinned Python package
# versions to avoid API/runtime breakage from upstream releases:
#   newton==1.1.0 and warp-lang==1.12.1
if(MOPHI_BUILD_FERIS_NEWTON OR MOPHI_BUILD_NEWTON_XLB_DEM)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c
                "import importlib, sys
required = {'newton': '1.1.0', 'warp': '1.12.1'}
installed = {}
errors = []
for module_name, version in required.items():
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        errors.append(f\"{module_name}: missing (required {version})\")
        continue
    found = getattr(mod, '__version__', None)
    installed[module_name] = found if found is not None else 'unknown'
    if found != version:
        errors.append(f\"{module_name}: found {installed[module_name]} (required {version})\")
if errors:
    details = '\\n'.join(errors)
    sys.stderr.write(
        'MoPhi: Newton-based couplers require pinned Python dependency versions.\\n'
        'Required:\\n'
        '  - newton==1.1.0\\n'
        '  - warp-lang==1.12.1\\n'
        f'Problems:\\n{details}\\n\\n'
        'Install/fix with:\\n'
        '  python -m pip install --upgrade newton==1.1.0 warp-lang==1.12.1\\n'
    )
    sys.exit(1)"
        RESULT_VARIABLE _mophi_newton_warp_version_check_result
        OUTPUT_QUIET
        ERROR_VARIABLE _mophi_newton_warp_version_check_error
        ERROR_STRIP_TRAILING_WHITESPACE
    )
    if(NOT _mophi_newton_warp_version_check_result EQUAL 0)
        message(FATAL_ERROR
            "${_mophi_newton_warp_version_check_error}")
    endif()
endif()

# ── FERIS + Newton co-simulation coupler ──────────────────────────────────────
if(MOPHI_BUILD_FERIS_NEWTON)
    add_subdirectory(feris_newton)
endif()

# ── Newton + XLB + DEM-Engine three-way co-simulation coupler ─────────────────
if(MOPHI_BUILD_NEWTON_XLB_DEM)
    add_subdirectory(newton_xlb_dem)
endif()
