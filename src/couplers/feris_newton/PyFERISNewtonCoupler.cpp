#include "PyFERISNewtonCoupler.h"

#include <core/Logger.hpp>

// ── CUDA runtime ──────────────────────────────────────────────────────────────
// Must be included before any FERIS header.  FERIS is a CUDA-based library
// whose headers annotate functions with __host__ and __device__.  These
// keywords are only defined once cuda_runtime.h has been included; without this
// include the plain C++ compiler would report "__host__ does not name a type".
#include <cuda_runtime.h>

// ── FERIS ─────────────────────────────────────────────────────────────────────
// FEASolver.h is a convenience header that pulls in all FERIS element types and
// solver types.  FERISImpl below owns a GPU_FEAT10_Data (TET10 element) and a
// SyncedAdamWNocoopSolver (AdamW-Nocoop time integrator) created in Initialize()
// and driven in Step() via the solver's Solve() method.
#include <feris/FEASolver.h>

// ─────────────────────────────────────────────────────────────────────────────
// FERISImpl — pimpl that hides feris::GPU_FEAT10_Data and
//             feris::SyncedAdamWNocoopSolver from PyFERISNewtonCoupler.h
// ─────────────────────────────────────────────────────────────────────────────

struct PyFERISNewtonCoupler::FERISImpl {
    /// FERIS TET10 element data (mesh geometry, DOFs, forces).  Created and
    /// initialized in Initialize(); torn down in Finalize() via Destroy().
    std::unique_ptr<feris::GPU_FEAT10_Data> fea_element;

    /// FERIS AdamW-Nocoop time integrator.  Created after fea_element in
    /// Initialize() and driven in Step() via its Solve() method.
    std::unique_ptr<feris::SyncedAdamWNocoopSolver> fea_solver;

    bool initialized{false};
    double time_step{1e-4};  ///< Co-simulation time step [s].
};

// ── Constructor / Destructor ───────────────────────────────────────────────────

PyFERISNewtonCoupler::PyFERISNewtonCoupler()
    : fea_(std::make_unique<FERISImpl>()),
      newton_model(pybind11::none()),
      newton_solver(pybind11::none()),
      newton_state_0(pybind11::none()),
      newton_state_1(pybind11::none()),
      newton_control(pybind11::none()),
      newton_contacts(pybind11::none()) {
    MOPHI_INFO("PyFERISNewtonCoupler: created");
}

PyFERISNewtonCoupler::~PyFERISNewtonCoupler() {
    if (fea_ && fea_->initialized) {
        Finalize();
    }
}

// ── Public interface ───────────────────────────────────────────────────────────

void PyFERISNewtonCoupler::Initialize(const std::string& feris_config,
                                      pybind11::object newton_model_in,
                                      pybind11::object newton_solver_in,
                                      double dt) {
    MOPHI_INFO("PyFERISNewtonCoupler: initializing ...");

    // ── FERIS ─────────────────────────────────────────────────────────────────
    // Create TET10 element data with placeholder dimensions (0 elements / 0 nodes).
    // TODO: load the actual mesh from feris_config and use the real element count /
    //       node count once mesh-loading support is implemented.
    fea_->fea_element = std::make_unique<feris::GPU_FEAT10_Data>(0, 0);
    fea_->fea_element->Initialize();
    // Create AdamW-Nocoop solver bound to the element.
    fea_->fea_solver = std::make_unique<feris::SyncedAdamWNocoopSolver>(fea_->fea_element.get(),
                                                                        fea_->fea_element->get_n_constraint());
    const std::string fea_suffix = feris_config.empty() ? "" : (" (config: " + feris_config + ")");
    MOPHI_INFO("PyFERISNewtonCoupler: feris::GPU_FEAT10_Data + SyncedAdamWNocoopSolver created%s", fea_suffix.c_str());

    fea_->initialized = true;

    // ── Newton ────────────────────────────────────────────────────────────────
    // Newton runs in Python; no C++ initialization is needed.
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
        MOPHI_INFO("PyFERISNewtonCoupler: Newton model and solver bound (sim_dt=%.6f s)", sim_dt);
    } else {
        MOPHI_INFO("PyFERISNewtonCoupler: no Newton solver provided — FERIS only");
    }

    MOPHI_INFO("PyFERISNewtonCoupler: initialized");
}

void PyFERISNewtonCoupler::Step() {
    if (!fea_->initialized) {
        MOPHI_ERROR("PyFERISNewtonCoupler::Step() called before Initialize().");
    }

    // ── 1. Advance FERIS ──────────────────────────────────────────────────────
    // Uncomment once the solver has been fully configured and initialized:
    // fea_->fea_solver->Solve();

    if (!newton_available) {
        return;
    }

    // ── 2. Extract FERIS node positions (coupling output: FERIS → Newton) ─────
    // auto feris_xyz = GetFERISNodePositions();
    // TODO: convert feris_xyz to a warp array and pass to Newton to update
    // collision geometry or rigid-body anchor points.

    // ── 3. Advance Newton ─────────────────────────────────────────────────────
    newton_state_0.attr("clear_forces")();
    newton_model.attr("collide")(newton_state_0, newton_contacts);
    newton_solver.attr("step")(newton_state_0, newton_state_1, newton_control, newton_contacts,
                               pybind11::float_(sim_dt));
    // state_1 now holds the new Newton state; swap buffers for next iteration.
    std::swap(newton_state_0, newton_state_1);

    // ── 4. Extract Newton forces (coupling input: Newton → FERIS) ─────────────
    // TODO: read contact / body forces from newton_state_0 (the just-computed
    // state after the swap) and apply them to FERIS:
    //   std::vector<std::array<double,3>> forces = ...;  // from Newton state
    //   SetFERISNodeForces(forces);
}

void PyFERISNewtonCoupler::Finalize() {
    MOPHI_INFO("PyFERISNewtonCoupler: finalizing ...");
    fea_->fea_solver.reset();
    if (fea_->fea_element) {
        fea_->fea_element->Destroy();
        fea_->fea_element.reset();
    }
    fea_->initialized = false;
    // Release Newton references so Python's reference counter can collect them.
    newton_model = pybind11::none();
    newton_solver = pybind11::none();
    newton_state_0 = pybind11::none();
    newton_state_1 = pybind11::none();
    newton_control = pybind11::none();
    newton_contacts = pybind11::none();
    newton_available = false;
    MOPHI_INFO("PyFERISNewtonCoupler: finalized");
}

std::vector<std::array<double, 3>> PyFERISNewtonCoupler::GetFERISNodePositions() const {
    if (!fea_->initialized) {
        return {};
    }
    // TODO: Return actual deformed node positions from FERIS once the solver is
    // fully configured, e.g.:
    //   VectorXR x12, y12, z12;
    //   fea_->fea_element->RetrievePositionToCPU(x12, y12, z12);
    //   // convert to std::vector<std::array<double,3>> and return
    // For now this is a placeholder that returns an empty vector.
    return {};
}

void PyFERISNewtonCoupler::SetFERISNodeForces(const std::vector<std::array<double, 3>>& forces) {
    if (!fea_->initialized) {
        MOPHI_ERROR("PyFERISNewtonCoupler::SetFERISNodeForces() called before Initialize().");
    }
    if (forces.empty()) {
        return;
    }
    // TODO: Apply external forces to FERIS nodes once the solver is fully configured,
    // e.g.:
    //   VectorXR h_f_ext(fea_->fea_element->get_n_coef() * 3);
    //   // populate h_f_ext from forces vector ...
    //   fea_->fea_element->SetExternalForce(h_f_ext);
    // For now this is a placeholder.
}
