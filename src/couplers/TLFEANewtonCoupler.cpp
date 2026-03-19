#include "TLFEANewtonCoupler.h"

#include <core/Logger.hpp>

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
    // The Python driver script creates and manages the Newton session directly.
    MOPHI_INFO("TLFEANewtonCoupler: Newton session is managed by the Python driver");

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

    // Newton step is driven from Python; this C++ method advances only TLFEA.
    // In a full co-simulation the Python layer would call this method and then
    // advance the Newton solver, exchanging forces/displacements between the two.
}

void TLFEANewtonCoupler::finalize() {
    MOPHI_INFO("TLFEANewtonCoupler: finalizing ...");
    impl_->fea.reset();
    impl_->initialized = false;
    MOPHI_INFO("TLFEANewtonCoupler: finalized");
}

}  // namespace mophi
