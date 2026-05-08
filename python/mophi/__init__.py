"""
mophi — multi-physics co-simulation framework

This package exposes the C++ core (mophi_core) under a clean Python namespace.
Co-simulation solvers are available only when the corresponding external
projects have been fetched and the solver was built (MOPHI_BUILD_* options).

Example usage (requires -DMOPHI_BUILD_TLFEA_DEM=ON at CMake configure time)::

    import mophi
    coupler = mophi.TLFEADEMCoupler()
    coupler.initialize()
    coupler.step()
    coupler.finalize()
"""

from . import mophi_core  # noqa: F401 — make the C++ module accessible

__all__ = []

# Verbosity constants (always available — come from mophi_core unconditionally).
try:
    from .mophi_core import VERBOSITY_ERROR, VERBOSITY_WARNING, VERBOSITY_INFO  # noqa: F401

    __all__ += ["VERBOSITY_ERROR", "VERBOSITY_WARNING", "VERBOSITY_INFO"]
except ImportError:
    pass

# TLFEADEMCoupler is only present when built with MOPHI_BUILD_TLFEA_DEM=ON.
try:
    from .mophi_core import TLFEADEMCoupler  # noqa: F401

    __all__.append("TLFEADEMCoupler")
except ImportError:
    pass

# TLFEANewtonCoupler is only present when built with MOPHI_BUILD_TLFEA_NEWTON=ON.
try:
    from .mophi_core import TLFEANewtonCoupler  # noqa: F401

    __all__.append("TLFEANewtonCoupler")
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

# SurfaceMesh and load_obj are always available — they depend only on the
# mophi_essentials submodule which is always compiled into mophi_core.
try:
    from .mophi_core import SurfaceMesh, load_obj  # noqa: F401

    __all__ += ["SurfaceMesh", "load_obj"]
except ImportError:
    pass

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
