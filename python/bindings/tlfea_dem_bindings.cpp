#include <pybind11/pybind11.h>

// TLFEADEMCoupler is only registered when the library was compiled with
// MOPHI_BUILD_TLFEA_DEM=ON (both external projects fetched and built).
// The CMake target sets MOPHI_HAS_TLFEA_DEM_COUPLER when that is the case.
#ifdef MOPHI_HAS_TLFEA_DEM_COUPLER
    #include "TLFEADEMCoupler.h"
#endif

namespace py = pybind11;

// ─────────────────────────────────────────────────────────────────────────────
// register_tlfea_dem — pybind11 registration for TLFEADEMCoupler.
//
// Registers the TLFEADEMCoupler class into the mophi_core module.
// Called unconditionally from PYBIND11_MODULE in mophi_bindings.cpp;
// the function body is a no-op when MOPHI_HAS_TLFEA_DEM_COUPLER is not defined.
// ─────────────────────────────────────────────────────────────────────────────

void register_tlfea_dem(py::module_& m) {
#ifdef MOPHI_HAS_TLFEA_DEM_COUPLER
    py::class_<mophi::TLFEADEMCoupler>(m, "TLFEADEMCoupler",
                                       "Co-simulation solver coupling TLFEA (FEA) and DEM-Engine (DEM).\n\n"
                                       "Use initialize() / step() / finalize() to drive the coupled simulation.\n\n"
                                       "Impl holds deme::DEMSolver and tlfea::SolverBase directly; no wrapper\n"
                                       "layer sits between this coupler and the solver APIs.")
        .def(py::init<>())
        .def("initialize", &mophi::TLFEADEMCoupler::Initialize, py::arg("tlfea_config") = "",
             py::arg("dem_config") = "", py::arg("num_gpus") = 1u,
             "Initialize both solvers.  num_gpus controls how many GPUs "
             "are handed to the DEM-Engine solver.")
        .def("step", &mophi::TLFEADEMCoupler::Step, "Advance both solvers by one co-simulation time step.")
        .def("finalize", &mophi::TLFEADEMCoupler::Finalize, "Finalize both solvers and release all resources.");
#endif
}
