#include <pybind11/pybind11.h>

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
// register_tlfea_newton — pybind11 registration for TLFEANewtonCoupler.
//
// Registers the TLFEANewtonCoupler class (backed by PyTLFEANewtonCoupler) into
// the mophi_core module.  Called unconditionally from PYBIND11_MODULE in
// mophi_bindings.cpp; the function body is a no-op when
// MOPHI_HAS_TLFEA_NEWTON_COUPLER is not defined.
// ─────────────────────────────────────────────────────────────────────────────

void register_tlfea_newton(py::module_& m) {
#ifdef MOPHI_HAS_TLFEA_NEWTON_COUPLER
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
