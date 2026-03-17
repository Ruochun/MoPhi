#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "TLFEADEMCoupler.h"
#include "TLFEAWrapper.h"
#include "DEMEngineWrapper.h"

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

    // ── Version / availability helpers ──────────────────────────────────────
    m.def("tlfea_available",      &mophi::TLFEAWrapper::isAvailable,
          "Return True if MoPhi was built with the TLFEA back-end.");
    m.def("dem_engine_available", &mophi::DEMEngineWrapper::isAvailable,
          "Return True if MoPhi was built with the DEM-Engine back-end.");

    // ── TLFEAWrapper ─────────────────────────────────────────────────────────
    py::class_<mophi::TLFEAWrapper>(m, "TLFEAWrapper",
        "Wrapper around the TLFEA finite-element solver.")
        .def(py::init<>())
        .def("initialize", &mophi::TLFEAWrapper::initialize,
             py::arg("config_file") = "",
             "Initialize the TLFEA solver.")
        .def("step",     &mophi::TLFEAWrapper::step,
             "Advance the simulation by one time step.")
        .def("finalize", &mophi::TLFEAWrapper::finalize,
             "Finalize the solver and release resources.")
        .def_static("is_available", &mophi::TLFEAWrapper::isAvailable,
             "Return True if TLFEA is compiled in.");

    // ── DEMEngineWrapper ──────────────────────────────────────────────────────
    py::class_<mophi::DEMEngineWrapper>(m, "DEMEngineWrapper",
        "Wrapper around the DEM-Engine discrete-element solver.")
        .def(py::init<>())
        .def("initialize", &mophi::DEMEngineWrapper::initialize,
             py::arg("config_file") = "",
             "Initialize the DEM-Engine solver.")
        .def("step",     &mophi::DEMEngineWrapper::step,
             "Advance the simulation by one time step.")
        .def("finalize", &mophi::DEMEngineWrapper::finalize,
             "Finalize the solver and release resources.")
        .def_static("is_available", &mophi::DEMEngineWrapper::isAvailable,
             "Return True if DEM-Engine is compiled in.");

    // ── TLFEADEMCoupler ───────────────────────────────────────────────────────
    py::class_<mophi::TLFEADEMCoupler>(m, "TLFEADEMCoupler",
        "Placeholder co-simulation solver coupling TLFEA and DEM-Engine.\n\n"
        "Use initialize() / step() / finalize() to drive the coupled simulation.")
        .def(py::init<>())
        .def("initialize", &mophi::TLFEADEMCoupler::initialize,
             py::arg("tlfea_config") = "",
             py::arg("dem_config")   = "",
             "Initialize both solvers.")
        .def("step",     &mophi::TLFEADEMCoupler::step,
             "Advance both solvers by one co-simulation time step.")
        .def("finalize", &mophi::TLFEADEMCoupler::finalize,
             "Finalize both solvers and release all resources.");
}
