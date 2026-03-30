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

# xlb_build_streamlines is a pure-NumPy utility; numpy is always available.
try:
    from .xlb_helpers import xlb_build_streamlines  # noqa: F401

    __all__.append("xlb_build_streamlines")
except ImportError:
    pass

# OpenGLVisualizer wraps Newton's ViewerGL behind a stable MoPhi interface.
# Available whenever newton is installed; no special CMake flag required.
try:
    from .opengl_visualizer import OpenGLVisualizer  # noqa: F401

    __all__.append("OpenGLVisualizer")
except ImportError:
    pass

# OmniverseVisualizer writes co-simulation state to a USD/USDC scene file.
# Same public interface as OpenGLVisualizer; switch backends with a single flag.
# Available whenever pxr (OpenUSD) is installed: pip install usd-core
try:
    from .omniverse_visualizer import OmniverseVisualizer  # noqa: F401

    __all__.append("OmniverseVisualizer")
except ImportError:
    pass
