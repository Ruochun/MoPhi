"""
mophi — multi-physics co-simulation framework

This package exposes the C++ core (mophi_core) under a clean Python namespace.
All co-simulation solvers and solver wrappers are re-exported here.

Example usage::

    import mophi
    coupler = mophi.TLFEADEMCoupler()
    coupler.initialize()
    coupler.step()
    coupler.finalize()
"""

from .mophi_core import (          # noqa: F401 — public API
    TLFEAWrapper,
    DEMEngineWrapper,
    TLFEADEMCoupler,
    tlfea_available,
    dem_engine_available,
)

__all__ = [
    "TLFEAWrapper",
    "DEMEngineWrapper",
    "TLFEADEMCoupler",
    "tlfea_available",
    "dem_engine_available",
]
