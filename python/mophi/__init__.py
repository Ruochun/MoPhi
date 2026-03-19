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
