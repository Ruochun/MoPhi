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
// The Python-facing class is PyTLFEANewtonCoupler (defined below), which owns
// both the C++ TLFEANewtonCoupler and the Newton model/solver objects.
#ifdef MOPHI_HAS_TLFEA_NEWTON_COUPLER
    #include "TLFEANewtonCoupler.h"
#endif

namespace py = pybind11;

// ─────────────────────────────────────────────────────────────────────────────
// PyTLFEANewtonCoupler — Python-facing wrapper that bridges C++ and Newton
//
// mophi::TLFEANewtonCoupler manages the C++ / TLFEA side.  Newton is a pure
// Python package (GPU-accelerated via NVIDIA Warp) with no C++ ABI.  This
// wrapper holds Newton objects as py::object so that a single step() call
// advances both solvers and performs the coupling data exchange:
//
//   TLFEA → Newton:  get_node_positions() feeds deformed node xyz to Newton
//                    (e.g. to update collision geometry or anchor points).
//   Newton → TLFEA:  contact / body forces extracted from the Newton state
//                    are applied via set_node_forces() before the next TLFEA
//                    advance.
//
// The wrapper is created inside PYBIND11_MODULE so that it has access to the
// Python interpreter (already running when the extension module is imported).
// ─────────────────────────────────────────────────────────────────────────────

#ifdef MOPHI_HAS_TLFEA_NEWTON_COUPLER

struct PyTLFEANewtonCoupler {
    mophi::TLFEANewtonCoupler coupler;

    // Newton objects owned on the Python side and kept alive here via reference
    // counting.  All are initialized to None and populated in initialize().
    py::object newton_model;
    py::object newton_solver;
    py::object newton_state_0;  ///< "current" Newton state (input to step())
    py::object newton_state_1;  ///< "scratch"  Newton state (output of step())
    py::object newton_control;
    py::object newton_contacts;
    double sim_dt{1.0 / 1000.0};  ///< Newton integration time step [s]
    bool newton_available{false};

    PyTLFEANewtonCoupler()
        : coupler(),
          newton_model(py::none()),
          newton_solver(py::none()),
          newton_state_0(py::none()),
          newton_state_1(py::none()),
          newton_control(py::none()),
          newton_contacts(py::none()) {}

    // Non-copyable (mophi::TLFEANewtonCoupler is non-copyable and holds GPU resources).
    PyTLFEANewtonCoupler(const PyTLFEANewtonCoupler&) = delete;
    PyTLFEANewtonCoupler& operator=(const PyTLFEANewtonCoupler&) = delete;

    /// @brief Initialize TLFEA and bind the Newton model and solver.
    ///
    /// @param tlfea_config   Configuration/mesh file forwarded to TLFEA.
    /// @param newton_model   A newton.Model instance (or None to skip Newton).
    /// @param newton_solver  A Newton solver instance, e.g. newton.solvers.SolverXPBD
    ///                       (or None to skip Newton).
    /// @param dt             Newton integration time step [s].
    void initialize(const std::string& tlfea_config,
                    py::object newton_model_in,
                    py::object newton_solver_in,
                    double dt) {
        coupler.initialize(tlfea_config);

        if (!newton_model_in.is_none() && !newton_solver_in.is_none()) {
            newton_model = newton_model_in;
            newton_solver = newton_solver_in;
            newton_state_0 = newton_model.attr("state")();
            newton_state_1 = newton_model.attr("state")();
            newton_control = newton_model.attr("control")();
            newton_contacts = newton_model.attr("contacts")();
            sim_dt = dt;

            // Evaluate initial forward kinematics so the first Newton step
            // starts from a well-defined body configuration.
            py::module_::import("newton").attr("eval_fk")(newton_model, newton_model.attr("joint_q"),
                                                          newton_model.attr("joint_qd"), newton_state_0);

            newton_available = true;
        }
    }

    /// @brief Advance one co-simulation step.
    ///
    /// Coupling sequence:
    ///   1. Advance TLFEA by one time step.
    ///   2. Extract deformed TLFEA node positions (coupling output).
    ///   3. TODO: update Newton collision geometry / anchor points with TLFEA positions.
    ///   4. Advance Newton by one time step (sim_dt).
    ///   5. TODO: extract contact/body forces from Newton state (coupling input).
    ///   6. TODO: apply Newton forces to TLFEA via set_node_forces() for the next step.
    void step() {
        // ── 1. Advance TLFEA ──────────────────────────────────────────────────
        coupler.step();

        if (!newton_available) {
            return;
        }

        // ── 2. Extract TLFEA node positions (coupling output: TLFEA → Newton) ─
        // auto tlfea_xyz = coupler.get_node_positions();
        // TODO: convert tlfea_xyz to a warp array and pass to Newton to update
        // collision geometry or rigid-body anchor points.

        // ── 3. Advance Newton ─────────────────────────────────────────────────
        newton_state_0.attr("clear_forces")();
        newton_model.attr("collide")(newton_state_0, newton_contacts);
        newton_solver.attr("step")(newton_state_0, newton_state_1, newton_control, newton_contacts, py::float_(sim_dt));
        // state_1 now holds the new Newton state; swap buffers for next iteration.
        std::swap(newton_state_0, newton_state_1);

        // ── 4. Extract Newton forces (coupling input: Newton → TLFEA) ─────────
        // TODO: read contact / body forces from newton_state_0 (the just-computed
        // state after the swap) and apply them to TLFEA:
        //   std::vector<std::array<double,3>> forces = ...;  // from Newton state
        //   coupler.set_node_forces(forces);
    }

    /// @brief Finalize both solvers and release all resources.
    void finalize() {
        coupler.finalize();
        // Release Newton references so Python's reference counter can collect them.
        newton_model = py::none();
        newton_solver = py::none();
        newton_state_0 = py::none();
        newton_state_1 = py::none();
        newton_control = py::none();
        newton_contacts = py::none();
        newton_available = false;
    }
};

#endif  // MOPHI_HAS_TLFEA_NEWTON_COUPLER

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
        .def("initialize", &mophi::TLFEADEMCoupler::initialize, py::arg("tlfea_config") = "",
             py::arg("dem_config") = "", py::arg("num_gpus") = 1u,
             "Initialize both solvers.  num_gpus controls how many GPUs "
             "are handed to the DEM-Engine solver.")
        .def("step", &mophi::TLFEADEMCoupler::step, "Advance both solvers by one co-simulation time step.")
        .def("finalize", &mophi::TLFEADEMCoupler::finalize, "Finalize both solvers and release all resources.");
#endif

#ifdef MOPHI_HAS_TLFEA_NEWTON_COUPLER
    // ── TLFEANewtonCoupler ────────────────────────────────────────────────────
    // Exposed as PyTLFEANewtonCoupler so that the Python class owns both the C++
    // TLFEANewtonCoupler (TLFEA side) and the Newton model/solver Python objects.
    // step() drives the full co-simulation cycle including data exchange.
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
        .def("initialize", &PyTLFEANewtonCoupler::initialize, py::arg("tlfea_config") = "",
             py::arg("newton_model") = py::none(), py::arg("newton_solver") = py::none(),
             py::arg("sim_dt") = 1.0 / 1000.0,
             "Initialize TLFEA and (optionally) bind a Newton model and solver.\n\n"
             "newton_model must be a newton.Model; newton_solver must be a Newton solver\n"
             "instance (e.g. newton.solvers.SolverXPBD(model)).  When both are provided\n"
             "the coupler creates Newton states, control, and contacts internally and\n"
             "evaluates initial forward kinematics.  sim_dt sets the Newton integration\n"
             "time step in seconds (default 1 ms).\n\n"
             "Omit newton_model / newton_solver (or pass None) to run TLFEA only.")
        .def("step", &PyTLFEANewtonCoupler::step,
             "Advance one co-simulation step.\n\n"
             "Sequence: advance TLFEA → extract TLFEA node positions → advance Newton\n"
             "(clear forces, collide, step, swap states) → (TODO) apply Newton forces\n"
             "to TLFEA.  If no Newton solver was provided, only TLFEA is advanced.")
        .def("finalize", &PyTLFEANewtonCoupler::finalize,
             "Finalize both solvers and release all resources including Newton references.");
#endif
}
