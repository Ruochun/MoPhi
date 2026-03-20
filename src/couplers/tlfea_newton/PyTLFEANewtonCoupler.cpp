#include "PyTLFEANewtonCoupler.h"

#include <core/Logger.hpp>

// ── CUDA runtime ──────────────────────────────────────────────────────────────
// Must be included before any TLFEA header.  TLFEA is a CUDA-based library
// whose headers annotate functions with __host__ and __device__.  These
// keywords are only defined once cuda_runtime.h has been included; without this
// include the plain C++ compiler would report "__host__ does not name a type".
#include <cuda_runtime.h>

// ── TLFEA ─────────────────────────────────────────────────────────────────────
// FEASolver is TLFEA's top-level simulation driver that manages everything in
// the package.  TLFEAImpl below owns a FEASolver instance created in
// Initialize() and driven in Step() via its Solve() method.
#include <tlfea/FEASolver.h>

// ─────────────────────────────────────────────────────────────────────────────
// TLFEAImpl — pimpl that hides tlfea::FEASolver from PyTLFEANewtonCoupler.h
// ─────────────────────────────────────────────────────────────────────────────

struct PyTLFEANewtonCoupler::TLFEAImpl {
    /// TLFEA's top-level simulation driver.  Created in Initialize() and driven
    /// in Step() via its Solve() method.
    std::unique_ptr<tlfea::FEASolver> fea;

    bool initialized{false};
    double time_step{1e-4};  ///< Co-simulation time step [s].
};

// ── Constructor / Destructor ───────────────────────────────────────────────────

PyTLFEANewtonCoupler::PyTLFEANewtonCoupler()
    : fea_(std::make_unique<TLFEAImpl>()),
      newton_model(pybind11::none()),
      newton_solver(pybind11::none()),
      newton_state_0(pybind11::none()),
      newton_state_1(pybind11::none()),
      newton_control(pybind11::none()),
      newton_contacts(pybind11::none()) {
    MOPHI_INFO("PyTLFEANewtonCoupler: created");
}

PyTLFEANewtonCoupler::~PyTLFEANewtonCoupler() {
    if (fea_ && fea_->initialized) {
        Finalize();
    }
}

// ── Public interface ───────────────────────────────────────────────────────────

void PyTLFEANewtonCoupler::Initialize(const std::string& tlfea_config,
                                      pybind11::object newton_model_in,
                                      pybind11::object newton_solver_in,
                                      double dt) {
    MOPHI_INFO("PyTLFEANewtonCoupler: initializing ...");

    // ── TLFEA ─────────────────────────────────────────────────────────────────
    fea_->fea = std::make_unique<tlfea::FEASolver>();
    const std::string fea_suffix = tlfea_config.empty() ? "" : (" (config: " + tlfea_config + ")");
    MOPHI_INFO("PyTLFEANewtonCoupler: tlfea::FEASolver created%s", fea_suffix.c_str());

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
        MOPHI_INFO("PyTLFEANewtonCoupler: Newton model and solver bound (sim_dt=%.6f s)", sim_dt);
    } else {
        MOPHI_INFO("PyTLFEANewtonCoupler: no Newton solver provided — TLFEA only");
    }

    MOPHI_INFO("PyTLFEANewtonCoupler: initialized");
}

void PyTLFEANewtonCoupler::Step() {
    if (!fea_->initialized) {
        MOPHI_ERROR("PyTLFEANewtonCoupler::Step() called before Initialize().");
    }

    // ── 1. Advance TLFEA ──────────────────────────────────────────────────────
    // Uncomment once the solver has been fully configured and initialized:
    // fea_->fea->Solve();

    if (!newton_available) {
        return;
    }

    // ── 2. Extract TLFEA node positions (coupling output: TLFEA → Newton) ─────
    // auto tlfea_xyz = GetNodePositions();
    // TODO: convert tlfea_xyz to a warp array and pass to Newton to update
    // collision geometry or rigid-body anchor points.

    // ── 3. Advance Newton ─────────────────────────────────────────────────────
    newton_state_0.attr("clear_forces")();
    newton_model.attr("collide")(newton_state_0, newton_contacts);
    newton_solver.attr("step")(newton_state_0, newton_state_1, newton_control, newton_contacts,
                               pybind11::float_(sim_dt));
    // state_1 now holds the new Newton state; swap buffers for next iteration.
    std::swap(newton_state_0, newton_state_1);

    // ── 4. Extract Newton forces (coupling input: Newton → TLFEA) ─────────────
    // TODO: read contact / body forces from newton_state_0 (the just-computed
    // state after the swap) and apply them to TLFEA:
    //   std::vector<std::array<double,3>> forces = ...;  // from Newton state
    //   SetNodeForces(forces);
}

void PyTLFEANewtonCoupler::Finalize() {
    MOPHI_INFO("PyTLFEANewtonCoupler: finalizing ...");
    fea_->fea.reset();
    fea_->initialized = false;
    // Release Newton references so Python's reference counter can collect them.
    newton_model = pybind11::none();
    newton_solver = pybind11::none();
    newton_state_0 = pybind11::none();
    newton_state_1 = pybind11::none();
    newton_control = pybind11::none();
    newton_contacts = pybind11::none();
    newton_available = false;
    MOPHI_INFO("PyTLFEANewtonCoupler: finalized");
}

std::vector<std::array<double, 3>> PyTLFEANewtonCoupler::GetNodePositions() const {
    if (!fea_->initialized) {
        return {};
    }
    // TODO: Return actual deformed node positions from TLFEA once the solver is
    // fully configured, e.g.:
    //   return fea_->fea->GetNodePositions();
    // For now this is a placeholder that returns an empty vector.
    return {};
}

void PyTLFEANewtonCoupler::SetNodeForces(const std::vector<std::array<double, 3>>& forces) {
    if (!fea_->initialized) {
        MOPHI_ERROR("PyTLFEANewtonCoupler::SetNodeForces() called before Initialize().");
    }
    if (forces.empty()) {
        return;
    }
    // TODO: Apply external forces to TLFEA nodes once the solver is fully configured,
    // e.g.:
    //   fea_->fea->SetExternalForces(forces);
    // For now this is a placeholder.
}
