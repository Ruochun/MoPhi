#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// TLFEADEMCoupler is only registered when the library was compiled with
// MOPHI_BUILD_TLFEA_DEM=ON (both external projects fetched and built).
// The CMake target sets MOPHI_HAS_TLFEA_DEM_COUPLER when that is the case.
#ifdef MOPHI_HAS_TLFEA_DEM_COUPLER
    #include "TLFEADEMCoupler.h"
#endif

// TLFEANewtonCoupler is only registered when the library was compiled with
// MOPHI_BUILD_TLFEA_NEWTON=ON (TLFEA external fetched and built).
// Newton itself is a pure Python package and is not linked into this module.
// PyTLFEANewtonCoupler (defined in src/couplers/tlfea_newton/PyTLFEANewtonCoupler.h)
// owns both the C++ TLFEANewtonCoupler and the Newton model/solver objects,
// bridging the two solvers inside a single step() call.
#ifdef MOPHI_HAS_TLFEA_NEWTON_COUPLER
    #include "PyTLFEANewtonCoupler.h"  // src/couplers/tlfea_newton/
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
        .def("initialize", &mophi::TLFEADEMCoupler::Initialize, py::arg("tlfea_config") = "",
             py::arg("dem_config") = "", py::arg("num_gpus") = 1u,
             "Initialize both solvers.  num_gpus controls how many GPUs "
             "are handed to the DEM-Engine solver.")
        .def("step", &mophi::TLFEADEMCoupler::Step, "Advance both solvers by one co-simulation time step.")
        .def("finalize", &mophi::TLFEADEMCoupler::Finalize, "Finalize both solvers and release all resources.");
#endif

#ifdef MOPHI_HAS_TLFEA_NEWTON_COUPLER
    // ── TLFEANewtonCoupler ────────────────────────────────────────────────────
    // Exposed as PyTLFEANewtonCoupler so that the Python class owns both the C++
    // TLFEANewtonCoupler (TLFEA side) and the Newton model/solver Python objects.
    // step() drives the full co-simulation cycle including data exchange.
    // The struct is defined in src/couplers/tlfea_newton/PyTLFEANewtonCoupler.h.
    py::class_<PyTLFEANewtonCoupler>(m, "TLFEANewtonCoupler",
                                     "Co-simulation coupler coupling TLFEA (FEA) and Newton (Python GPU physics).\n\n"
                                     "Owns the TLFEA solver on the C++ side and Newton model/solver objects on the\n"
                                     "Python side.  Pass a newton.Model and a Newton solver to initialize() so that\n"
                                     "step() can advance both solvers and exchange coupling data between them:\n\n"
                                     "  TLFEA → Newton : deformed node positions forwarded to Newton geometry\n"
                                     "  Newton → TLFEA : contact/body forces applied to TLFEA as external loads\n\n"
                                     "Usage::\n\n"
                                     "    import mophi, newton, warp as wp\n"
                                     "    wp.init()\n"
                                     "    builder = newton.ModelBuilder()\n"
                                     "    # ... configure model ...\n"
                                     "    model  = builder.finalize()\n"
                                     "    solver = newton.solvers.SolverXPBD(model)\n"
                                     "    coupler = mophi.TLFEANewtonCoupler()\n"
                                     "    coupler.initialize(newton_model=model, newton_solver=solver)\n"
                                     "    for _ in range(steps):\n"
                                     "        coupler.step()  # drives TLFEA + Newton + data exchange\n"
                                     "    coupler.finalize()")
        .def(py::init<>())
        .def("initialize", &PyTLFEANewtonCoupler::Initialize, py::arg("tlfea_config") = "",
             py::arg("newton_model") = py::none(), py::arg("newton_solver") = py::none(),
             py::arg("sim_dt") = 1.0 / 1000.0,
             "Initialize TLFEA and (optionally) bind a Newton model and solver.\n\n"
             "newton_model must be a newton.Model; newton_solver must be a Newton solver\n"
             "instance (e.g. newton.solvers.SolverXPBD(model)).  When both are provided\n"
             "the coupler creates Newton states, control, and contacts internally and\n"
             "evaluates initial forward kinematics.  sim_dt sets the Newton integration\n"
             "time step in seconds (default 1 ms).\n\n"
             "Omit newton_model / newton_solver (or pass None) to run TLFEA only.")
        .def("step", &PyTLFEANewtonCoupler::Step,
             "Advance one co-simulation step.\n\n"
             "Sequence: advance TLFEA → extract TLFEA node positions → advance Newton\n"
             "(clear forces, collide, step, swap states) → (TODO) apply Newton forces\n"
             "to TLFEA.  If no Newton solver was provided, only TLFEA is advanced.")
        .def("finalize", &PyTLFEANewtonCoupler::Finalize,
             "Finalize both solvers and release all resources including Newton references.");
#endif
}
