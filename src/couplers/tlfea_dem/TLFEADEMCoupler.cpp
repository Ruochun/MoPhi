#include "TLFEADEMCoupler.h"

#include <core/Logger.hpp>

// ── DEM-Engine ────────────────────────────────────────────────────────────────
// Header lives at external/DEMEngine/src/DEM/API.h, added to the include path
// by src/couplers/CMakeLists.txt.  This translation unit is only compiled when
// MOPHI_BUILD_TLFEA_DEM=ON AND both externals have been fetched; CMake emits a
// FATAL_ERROR before compilation if either external is missing.
#include <DEM/API.h>

// ── TLFEA ─────────────────────────────────────────────────────────────────────
// FEASolver.h is a convenience header that pulls in all TLFEA element types and
// solver types.  The Impl below uses GPU_FEAT10_Data (TET10 element) and
// SyncedAdamWNocoopSolver (AdamW-Nocoop time integrator) created in Initialize()
// and driven in Step() via the solver's Solve() method.
#include <tlfea/FEASolver.h>

namespace mophi {

// ── Pimpl ────────────────────────────────────────────────────────────────────

struct TLFEADEMCoupler::Impl {
    /// DEM-Engine's main solver.  Created (not yet initialized) in Initialize(),
    /// torn down in Finalize().  In a real co-simulation the user would configure
    /// domain size, materials, and particle templates on this object before
    /// calling dem->Initialize().
    std::unique_ptr<deme::DEMSolver> dem;

    /// TLFEA TET10 element data (mesh geometry, DOFs, forces).  Created and
    /// initialized in Initialize(); torn down in Finalize() via Destroy().
    std::unique_ptr<tlfea::GPU_FEAT10_Data> fea_element;

    /// TLFEA AdamW-Nocoop time integrator.  Created after fea_element in
    /// Initialize() and driven in Step() via its Solve() method.
    std::unique_ptr<tlfea::SyncedAdamWNocoopSolver> fea_solver;

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
    // Create TET10 element data with placeholder dimensions (0 elements / 0 nodes).
    // TODO: load the actual mesh from tlfea_config and use the real element count /
    //       node count once mesh-loading support is implemented.
    impl_->fea_element = std::make_unique<tlfea::GPU_FEAT10_Data>(0, 0);
    impl_->fea_element->Initialize();
    // Create AdamW-Nocoop solver bound to the element.
    impl_->fea_solver = std::make_unique<tlfea::SyncedAdamWNocoopSolver>(impl_->fea_element.get(),
                                                                         impl_->fea_element->get_n_constraint());
    const std::string fea_suffix = tlfea_config.empty() ? "" : (" (config: " + tlfea_config + ")");
    MOPHI_INFO("TLFEADEMCoupler: tlfea::GPU_FEAT10_Data + SyncedAdamWNocoopSolver created%s", fea_suffix.c_str());

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
    // impl_->fea_solver->Solve();
}

void TLFEADEMCoupler::Finalize() {
    MOPHI_INFO("TLFEADEMCoupler: finalizing ...");
    impl_->fea_solver.reset();
    if (impl_->fea_element) {
        impl_->fea_element->Destroy();
        impl_->fea_element.reset();
    }
    impl_->dem.reset();
    impl_->initialized = false;
    MOPHI_INFO("TLFEADEMCoupler: finalized");
}

}  // namespace mophi
