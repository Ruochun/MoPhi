#pragma once

#include <pybind11/pybind11.h>

#include "TLFEANewtonCoupler.h"

// ─────────────────────────────────────────────────────────────────────────────
// PyTLFEANewtonCoupler — Python-facing wrapper that bridges C++ and Newton
//
// mophi::TLFEANewtonCoupler manages the C++ / TLFEA side.  Newton is a pure
// Python package (GPU-accelerated via NVIDIA Warp) with no C++ ABI.  This
// wrapper holds Newton objects as pybind11::object so that a single step()
// call advances both solvers and performs the coupling data exchange:
//
//   TLFEA → Newton:  get_node_positions() feeds deformed node xyz to Newton
//                    (e.g. to update collision geometry or anchor points).
//   Newton → TLFEA:  contact / body forces extracted from the Newton state
//                    are applied via set_node_forces() before the next TLFEA
//                    advance.
//
// This class lives in src/couplers/tlfea_newton/ so that the coupler logic
// is co-located with the rest of the TLFEA+Newton coupler code.  It is
// compiled as part of the mophi_core Python extension module (not as a
// standalone library) because it depends on pybind11 types.
// ─────────────────────────────────────────────────────────────────────────────

struct PyTLFEANewtonCoupler {
    mophi::TLFEANewtonCoupler coupler;

    // Newton objects owned on the Python side and kept alive here via reference
    // counting.  All are initialized to None and populated in initialize().
    pybind11::object newton_model;
    pybind11::object newton_solver;
    pybind11::object newton_state_0;  ///< "current" Newton state (input to step())
    pybind11::object newton_state_1;  ///< "scratch"  Newton state (output of step())
    pybind11::object newton_control;
    pybind11::object newton_contacts;
    double sim_dt{1.0 / 1000.0};  ///< Newton integration time step [s]
    bool newton_available{false};

    PyTLFEANewtonCoupler()
        : coupler(),
          newton_model(pybind11::none()),
          newton_solver(pybind11::none()),
          newton_state_0(pybind11::none()),
          newton_state_1(pybind11::none()),
          newton_control(pybind11::none()),
          newton_contacts(pybind11::none()) {}

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
                    pybind11::object newton_model_in,
                    pybind11::object newton_solver_in,
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
            pybind11::module_::import("newton").attr("eval_fk")(newton_model, newton_model.attr("joint_q"),
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
        newton_solver.attr("step")(newton_state_0, newton_state_1, newton_control, newton_contacts,
                                   pybind11::float_(sim_dt));
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
        newton_model = pybind11::none();
        newton_solver = pybind11::none();
        newton_state_0 = pybind11::none();
        newton_state_1 = pybind11::none();
        newton_control = pybind11::none();
        newton_contacts = pybind11::none();
        newton_available = false;
    }
};
