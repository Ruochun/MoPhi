#include "TLFEADEMCoupler.h"

#include <core/Logger.hpp>

// ── DEM-Engine ────────────────────────────────────────────────────────────────
// Header lives at external/DEMEngine/src/DEM/API.h, added to the include path
// by src/couplers/CMakeLists.txt.  This translation unit is only compiled when
// MOPHI_BUILD_TLFEA_DEM=ON AND both externals have been fetched; CMake emits a
// FATAL_ERROR before compilation if either external is missing.
#include <DEM/API.h>

// ── TLFEA ─────────────────────────────────────────────────────────────────────
// FEASolver is TLFEA's top-level simulation driver that manages everything in
// the package.  The Impl below owns a FEASolver instance created in Initialize()
// and driven in Step() via its Solve() method.
#include <tlfea/FEASolver.h>

namespace mophi {

// ── Pimpl ────────────────────────────────────────────────────────────────────

struct TLFEADEMCoupler::Impl {
    /// DEM-Engine's main solver.  Created (not yet initialized) in Initialize(),
    /// torn down in Finalize().  In a real co-simulation the user would configure
    /// domain size, materials, and particle templates on this object before
    /// calling dem->Initialize().
    std::unique_ptr<deme::DEMSolver> dem;

    /// TLFEA's top-level simulation driver.  Created in Initialize() and
    /// driven in Step() via its Solve() method.
    std::unique_ptr<tlfea::FEASolver> fea;

    bool initialized{false};
    double time_step{1e-4};  ///< Co-simulation time step [s].
};

// ── Constructor / Destructor ──────────────────────────────────────────────────

TLFEADEMCoupler::TLFEADEMCoupler() : impl_(std::make_unique<Impl>()) {
    MOPHI_INFO("TLFEADEMCoupler: created");
}

TLFEADEMCoupler::~TLFEADEMCoupler() {
    if (impl_ && impl_->initialized) {
        Finalize();
    }
}

// ── Public interface ──────────────────────────────────────────────────────────

void TLFEADEMCoupler::Initialize(const std::string& tlfea_config,
                                 const std::string& dem_config,
                                 unsigned int num_gpus) {
    MOPHI_INFO("TLFEADEMCoupler: initializing ...");

    // ── DEM-Engine ────────────────────────────────────────────────────────────
    // Create DEMSolver with the requested number of GPUs.  In a real simulation
    // the user would configure domain, materials, and clump templates on this
    // object and then call impl_->dem->Initialize() before the time-stepping loop.
    impl_->dem = std::make_unique<deme::DEMSolver>(num_gpus);
    const std::string dem_suffix = dem_config.empty() ? "" : (" (config: " + dem_config + ")");
    MOPHI_INFO("TLFEADEMCoupler: deme::DEMSolver created (nGPUs=%u)%s", num_gpus, dem_suffix.c_str());

    // ── TLFEA ─────────────────────────────────────────────────────────────────
    // Create FEASolver — TLFEA's top-level simulation driver.
    impl_->fea = std::make_unique<tlfea::FEASolver>();
    const std::string fea_suffix = tlfea_config.empty() ? "" : (" (config: " + tlfea_config + ")");
    MOPHI_INFO("TLFEADEMCoupler: tlfea::FEASolver created%s", fea_suffix.c_str());

    impl_->initialized = true;
    MOPHI_INFO("TLFEADEMCoupler: initialized");
}

void TLFEADEMCoupler::Step() {
    if (!impl_->initialized) {
        MOPHI_ERROR("TLFEADEMCoupler::Step() called before Initialize().");
    }

    // DEM step: advance by one co-simulation time step.
    // Uncomment once the solver has been fully configured and initialized:
    // impl_->dem->DoDynamicsThenSync(impl_->time_step);

    // FEA step: advance the TLFEA simulation by one time step.
    // impl_->fea->Solve();
}

void TLFEADEMCoupler::Finalize() {
    MOPHI_INFO("TLFEADEMCoupler: finalizing ...");
    impl_->fea.reset();
    impl_->dem.reset();
    impl_->initialized = false;
    MOPHI_INFO("TLFEADEMCoupler: finalized");
}

}  // namespace mophi
