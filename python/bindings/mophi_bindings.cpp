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

// NewtonXLBDEMCoupler is only registered when the library was compiled with
// MOPHI_BUILD_NEWTON_XLB_DEM=ON (DEMEngine external fetched and built).
// Newton and XLB are pure Python packages and are not linked into this module.
// PyNewtonXLBDEMCoupler (defined in src/couplers/newton_xlb_dem/PyNewtonXLBDEMCoupler.h)
// bridges all three solvers: Newton drives a walking robot in Python, XLB provides
// a placeholder LBM fluid solver in Python, and DEM-Engine is a placeholder C++
// discrete-element solver held via a pimpl DEMImpl.
#ifdef MOPHI_HAS_NEWTON_XLB_DEM_COUPLER
    #include "PyNewtonXLBDEMCoupler.h"  // src/couplers/newton_xlb_dem/
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

#ifdef MOPHI_HAS_NEWTON_XLB_DEM_COUPLER
    // ── NewtonXLBDEMCoupler ───────────────────────────────────────────────────
    // Three-way co-simulation coupler: Newton (walking robot, Python GPU), XLB
    // (LBM fluid solver, Python GPU), and DEM-Engine (discrete elements, C++/CUDA).
    // Newton is the primary active solver; XLB and DEM-Engine are placeholders that
    // are instantiated but do not advance serious physics in Step() yet.
    // PyNewtonXLBDEMCoupler is defined in src/couplers/newton_xlb_dem/.
    py::class_<PyNewtonXLBDEMCoupler>(m, "NewtonXLBDEMCoupler",
                                      "Three-way co-simulation coupler coupling Newton (articulated rigid-body "
                                      "physics), XLB (lattice-Boltzmann fluid solver), and DEM-Engine (discrete "
                                      "elements).\n\n"
                                      "Newton drives a walking robot and produces a spatial representation "
                                      "(body transforms) at every step.  XLB and DEM-Engine are placeholder "
                                      "solvers that are instantiated but not seriously advanced yet; future "
                                      "work will feed the robot geometry into both.\n\n"
                                      "Usage::\n\n"
                                      "    import mophi, newton, warp as wp\n"
                                      "    wp.init()\n"
                                      "    builder = newton.ModelBuilder()\n"
                                      "    # ... build walking robot ...\n"
                                      "    model  = builder.finalize()\n"
                                      "    solver = newton.solvers.SolverXPBD(model)\n"
                                      "    coupler = mophi.NewtonXLBDEMCoupler()\n"
                                      "    coupler.initialize(newton_model=model, newton_solver=solver)\n"
                                      "    for step in range(steps):\n"
                                      "        # optionally update coupler.newton_control before each step\n"
                                      "        coupler.step()\n"
                                      "        transforms = coupler.get_robot_body_transforms()\n"
                                      "    coupler.finalize()")
        .def(py::init<>())
        .def("initialize", &PyNewtonXLBDEMCoupler::Initialize, py::arg("newton_model") = py::none(),
             py::arg("newton_solver") = py::none(), py::arg("xlb_simulation") = py::none(),
             py::arg("sim_dt") = 1.0 / 1000.0, py::arg("num_gpus") = 1u,
             "Initialize all three solvers.\n\n"
             "newton_model must be a newton.Model; newton_solver must be a Newton solver\n"
             "instance (e.g. newton.solvers.SolverXPBD(model)).  When both are provided\n"
             "the coupler evaluates initial forward kinematics and is ready to step.\n"
             "xlb_simulation may be an XLB simulation object or None (skip XLB).\n"
             "sim_dt sets the co-simulation time step in seconds (default 1 ms).\n"
             "num_gpus controls how many GPUs are handed to DEM-Engine (default 1).")
        .def("step", &PyNewtonXLBDEMCoupler::Step,
             "Advance one co-simulation step.\n\n"
             "Sequence: clear Newton forces → Newton collide → Newton step → swap states\n"
             "→ DEM-Engine placeholder (no-op) → XLB placeholder (no-op).\n"
             "Call get_robot_body_transforms() after step() to read the spatial\n"
             "representation of the robot.")
        .def("finalize", &PyNewtonXLBDEMCoupler::Finalize,
             "Finalize all solvers and release all resources including Python references.")
        .def("get_robot_body_transforms", &PyNewtonXLBDEMCoupler::GetRobotBodyTransforms,
             "Return the spatial representation of the robot as a list of body transforms.\n\n"
             "Each entry is [px, py, pz, qx, qy, qz, qw] — the world-space position\n"
             "and orientation quaternion (Warp convention: x,y,z,w) of one robot body.\n"
             "The list has one entry per Newton body (model.body_count).\n"
             "Returns an empty list when Newton has not been initialized.")
        .def_readwrite("newton_control", &PyNewtonXLBDEMCoupler::newton_control,
                       "The Newton control object.  Write joint_target values here before\n"
                       "calling step() to drive the robot's joints.");
#endif
}
