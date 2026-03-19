#include "TLFEANewtonCoupler.h"

#include <core/Logger.hpp>

// ── CUDA runtime ──────────────────────────────────────────────────────────────
// Must be included before any TLFEA header.  TLFEA is a CUDA-based library
// whose headers annotate functions with __host__ and __device__.  These
// keywords are only defined once cuda_runtime.h has been included; without this
// include the plain C++ compiler would report "__host__ does not name a type".
#include <cuda_runtime.h>

// ── TLFEA ─────────────────────────────────────────────────────────────────────
// FEASolver is TLFEA's top-level simulation driver that manages everything in
// the package.  The Impl below owns a FEASolver instance created in initialize()
// and driven in step() via its Solve() method.
#include <tlfea/FEASolver.h>

namespace mophi {

// ── Pimpl ─────────────────────────────────────────────────────────────────────

struct TLFEANewtonCoupler::Impl {
    /// TLFEA's top-level simulation driver.  Created in initialize() and driven
    /// in step() via its Solve() method.  Newton is a pure Python package and
    /// therefore has no C++ representation in this struct.
    std::unique_ptr<tlfea::FEASolver> fea;

    bool initialized{false};
    double time_step{1e-4};  ///< Co-simulation time step [s].
};

// ── Constructor / Destructor ───────────────────────────────────────────────────

TLFEANewtonCoupler::TLFEANewtonCoupler() : impl_(std::make_unique<Impl>()) {
    MOPHI_INFO("TLFEANewtonCoupler: created");
}

TLFEANewtonCoupler::~TLFEANewtonCoupler() {
    if (impl_ && impl_->initialized) {
        finalize();
    }
}

// ── Public interface ───────────────────────────────────────────────────────────

void TLFEANewtonCoupler::initialize(const std::string& tlfea_config) {
    MOPHI_INFO("TLFEANewtonCoupler: initializing ...");

    // ── TLFEA ─────────────────────────────────────────────────────────────────
    // Create FEASolver — TLFEA's top-level simulation driver.
    impl_->fea = std::make_unique<tlfea::FEASolver>();
    const std::string fea_suffix = tlfea_config.empty() ? "" : (" (config: " + tlfea_config + ")");
    MOPHI_INFO("TLFEANewtonCoupler: tlfea::FEASolver created%s", fea_suffix.c_str());

    // Newton runs in Python; no C++ initialization is needed here.
    // The Python binding (PyTLFEANewtonCoupler) owns the Newton objects and
    // drives Newton steps alongside this C++ coupler.
    MOPHI_INFO("TLFEANewtonCoupler: Newton session is managed by PyTLFEANewtonCoupler (binding layer)");

    impl_->initialized = true;
    MOPHI_INFO("TLFEANewtonCoupler: initialized");
}

void TLFEANewtonCoupler::step() {
    if (!impl_->initialized) {
        MOPHI_ERROR("TLFEANewtonCoupler::step() called before initialize().");
    }

    // FEA step: advance the TLFEA simulation by one time step.
    // Uncomment once the solver has been fully configured and initialized:
    // impl_->fea->Solve();

    // The Python binding (PyTLFEANewtonCoupler) calls this method as part of its
    // own step(), which also drives Newton and performs data exchange via
    // get_node_positions() and set_node_forces().
}

void TLFEANewtonCoupler::finalize() {
    MOPHI_INFO("TLFEANewtonCoupler: finalizing ...");
    impl_->fea.reset();
    impl_->initialized = false;
    MOPHI_INFO("TLFEANewtonCoupler: finalized");
}

std::vector<std::array<double, 3>> TLFEANewtonCoupler::get_node_positions() const {
    if (!impl_->initialized) {
        return {};
    }
    // TODO: Return actual deformed node positions from TLFEA once the solver is
    // fully configured, e.g.:
    //   return impl_->fea->GetNodePositions();
    // For now this is a placeholder that returns an empty vector.
    return {};
}

void TLFEANewtonCoupler::set_node_forces(const std::vector<std::array<double, 3>>& forces) {
    if (!impl_->initialized) {
        MOPHI_ERROR("TLFEANewtonCoupler::set_node_forces() called before initialize().");
    }
    if (forces.empty()) {
        return;
    }
    // TODO: Apply external forces to TLFEA nodes once the solver is fully configured,
    // e.g.:
    //   impl_->fea->SetExternalForces(forces);
    // For now this is a placeholder.
}

}  // namespace mophi
