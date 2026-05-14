"""
mophi — multi-physics co-simulation framework

This package exposes the C++ core (mophi_core) under a clean Python namespace.
Co-simulation solvers are available only when the corresponding external
projects have been fetched and the solver was built (MOPHI_BUILD_* options).

Example usage (requires -DMOPHI_BUILD_FERIS_NEWTON=ON at CMake configure time)::

    import mophi
    coupler = mophi.FERISNewtonCoupler()
    coupler.initialize()
    coupler.step()
    coupler.finalize()
"""

import sys as _sys

from . import mophi_core  # noqa: F401 — make the C++ module accessible

__all__ = []

REQUIRED_NEWTON_VERSION = "1.0.0"
REQUIRED_WARP_VERSION = "1.12.1"
REQUIRED_MUJOCO_VERSION = "3.6.0"

__all__ += [
    "REQUIRED_NEWTON_VERSION",
    "REQUIRED_WARP_VERSION",
    "REQUIRED_MUJOCO_VERSION",
]

# Verbosity constants (always available — come from mophi_core unconditionally).
try:
    from .mophi_core import VERBOSITY_ERROR, VERBOSITY_WARNING, VERBOSITY_INFO  # noqa: F401

    __all__ += ["VERBOSITY_ERROR", "VERBOSITY_WARNING", "VERBOSITY_INFO"]
except ImportError:
    pass

# FERISNewtonCoupler is only present when built with MOPHI_BUILD_FERIS_NEWTON=ON.
try:
    from .mophi_core import FERISNewtonCoupler  # noqa: F401

    __all__.append("FERISNewtonCoupler")
except ImportError:
    pass

# NewtonXLBDEMCoupler is only present when built with MOPHI_BUILD_NEWTON_XLB_DEM=ON.
try:
    from .mophi_core import NewtonXLBDEMCoupler  # noqa: F401

    __all__.append("NewtonXLBDEMCoupler")
except ImportError:
    pass

# xlb helpers are pure-NumPy utilities; numpy is always available.
# xlb_make_streamline_warp_arrays additionally requires warp (Newton dependency)
# and performs a lazy import internally, so it is safe to re-export here.
try:
    from .utils.xlb_helpers import (  # noqa: F401
        xlb_build_streamlines,
        xlb_make_streamline_warp_arrays,
        xlb_make_y_plane_seeds,
    )

    __all__ += ["xlb_build_streamlines", "xlb_make_streamline_warp_arrays", "xlb_make_y_plane_seeds"]
except ImportError:
    pass

# General-purpose math helpers (quaternion operations, etc.).
try:
    from .utils.math_utils import quat_mul  # noqa: F401

    __all__.append("quat_mul")
except ImportError:
    pass

# Package-version helper utilities.
try:
    from .utils.version_checker import (  # noqa: F401
        check_newton_warp_mujoco_versions,
        check_newton_warp_versions,
        check_python_package_versions,
    )

    __all__ += [
        "check_newton_warp_mujoco_versions",
        "check_newton_warp_versions",
        "check_python_package_versions",
    ]
except ImportError:
    pass

# SurfaceMesh and load_obj are always available — they depend only on the
# mophi_essentials submodule which is always compiled into mophi_core.
try:
    from .mophi_core import SurfaceMesh, load_obj  # noqa: F401

    __all__ += ["SurfaceMesh", "load_obj"]
except ImportError:
    pass


def fatal(msg: str) -> None:
    """Print an error message and terminate the process with exit code 1.

    This is the preferred way for MoPhi demos to report unrecoverable errors
    (missing data files, unsatisfied build requirements, etc.).  It avoids
    bare ``sys.exit()`` calls scattered through demo code and provides a
    consistent error format.

    .. note::
        This function calls :func:`sys.exit`, which raises :class:`SystemExit`.
        It is designed for top-level demo scripts and should not be called from
        library code or test fixtures where clean shutdown is required.

    Args:
        msg: Human-readable description of the error, including any remediation
             hint (e.g. which CMake flag to enable or package to install).

    Example::

        if not hasattr(mophi, "NewtonXLBDEMCoupler"):
            mophi.fatal(
                "mophi.NewtonXLBDEMCoupler is not available.\\n"
                "Re-build MoPhi with -DMOPHI_BUILD_NEWTON_XLB_DEM=ON."
            )
    """
    _sys.exit(f"\n[MoPhi FATAL] {msg}\n")


__all__ += ["fatal"]


def create_opengl_visualizer_or_fatal(model):
    """Create OpenGLVisualizer or terminate with a clear fatal message."""
    if "OpenGLVisualizer" not in globals():
        fatal(
            "mophi.OpenGLVisualizer is not available.\n"
            "Install with:  pip install --upgrade "
            f"newton=={REQUIRED_NEWTON_VERSION} "
            f"warp-lang=={REQUIRED_WARP_VERSION} "
            f"mujoco=={REQUIRED_MUJOCO_VERSION}\n"
            "Or switch to Omniverse visualization."
        )
    try:
        return OpenGLVisualizer(model)
    except Exception as exc:
        fatal(f"OpenGL visualization initialization failed: {exc}")


__all__ += ["create_opengl_visualizer_or_fatal"]

# OpenGLVisualizer wraps Newton's ViewerGL behind a stable MoPhi interface.
# Available whenever newton is installed; no special CMake flag required.
try:
    from .visualizers.opengl_visualizer import OpenGLVisualizer  # noqa: F401

    __all__.append("OpenGLVisualizer")
except ImportError:
    pass

# OmniverseVisualizer writes co-simulation state to a USD/USDC scene file.
# Same public interface as OpenGLVisualizer; switch backends with a single flag.
# Available whenever pxr (OpenUSD) is installed: pip install usd-core
try:
    from .visualizers.omniverse_visualizer import OmniverseVisualizer  # noqa: F401

    __all__.append("OmniverseVisualizer")
except ImportError:
    pass
