#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// TLFEADEMCoupler is only registered when the library was compiled with
// MOPHI_BUILD_TLFEA_DEM=ON (both external projects fetched and built).
// The CMake target sets MOPHI_HAS_TLFEA_DEM_COUPLER when that is the case.
#ifdef MOPHI_HAS_TLFEA_DEM_COUPLER
#include "TLFEADEMCoupler.h"
#endif

namespace py = pybind11;

// ─────────────────────────────────────────────────────────────────────────────
// mophi_core — Python extension module
//
// Every multi-physics co-simulation solver built in MoPhi must be exposed here
// (or in a sibling binding file).  The Python package in python/mophi/ imports
// this module and re-exports the classes under the user-facing `mophi` namespace.
// ─────────────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(mophi_core, m) {
    m.doc() = "MoPhi — multi-physics co-simulation framework (C++ core)";

#ifdef MOPHI_HAS_TLFEA_DEM_COUPLER
    // ── TLFEADEMCoupler ───────────────────────────────────────────────────────
    py::class_<mophi::TLFEADEMCoupler>(m, "TLFEADEMCoupler",
        "Co-simulation solver coupling TLFEA (FEA) and DEM-Engine (DEM).\n\n"
        "Use initialize() / step() / finalize() to drive the coupled simulation.\n\n"
        "Impl holds deme::DEMSolver and tlfea::SolverBase directly; no wrapper\n"
        "layer sits between this coupler and the solver APIs.")
        .def(py::init<>())
        .def("initialize", &mophi::TLFEADEMCoupler::initialize,
             py::arg("tlfea_config") = "",
             py::arg("dem_config")   = "",
             py::arg("num_gpus")     = 1u,
             "Initialize both solvers.  num_gpus controls how many GPUs "
             "are handed to the DEM-Engine solver.")
        .def("step",     &mophi::TLFEADEMCoupler::step,
             "Advance both solvers by one co-simulation time step.")
        .def("finalize", &mophi::TLFEADEMCoupler::finalize,
             "Finalize both solvers and release all resources.");
#endif
}
