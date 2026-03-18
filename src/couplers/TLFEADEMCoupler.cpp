#include "TLFEADEMCoupler.h"

#include <iostream>
#include <stdexcept>

// ── DEM-Engine ────────────────────────────────────────────────────────────────
// Header lives at external/DEMEngine/src/DEM/API.h, added to the include path
// by src/couplers/CMakeLists.txt.  This translation unit is only compiled when
// MOPHI_BUILD_TLFEA_DEM=ON AND both externals have been fetched; CMake emits a
// FATAL_ERROR before compilation if either external is missing.
#include <DEM/API.h>

// ── TLFEA ─────────────────────────────────────────────────────────────────────
// TLFEA does not yet expose a single top-level "simulation driver" class.
// SolverBase is the common abstract interface for all TLFEA iterative solvers
// (SyncedNesterov, SyncedAdamW, …).  The Impl below holds a SolverBase pointer
// that is left as nullptr until a full FEA simulation setup (mesh loading,
// material parameters, boundary conditions) is added in a future release.
// Header lives at external/TLFEA/src/solvers/SolverBase.h.
#include <solvers/SolverBase.h>

namespace mophi {

// ── Pimpl ────────────────────────────────────────────────────────────────────

struct TLFEADEMCoupler::Impl {
    /// DEM-Engine's main solver.  Created (not yet initialized) in initialize(),
    /// torn down in finalize().  In a real co-simulation the user would configure
    /// domain size, materials, and particle templates on this object before
    /// calling dem->Initialize().
    std::unique_ptr<deme::DEMSolver> dem;

    /// TLFEA solver interface.  Left as nullptr until a concrete FEA simulation
    /// setup (ElementBase + material + boundary conditions) can be wired in.
    /// When a real FEA driver exists, step() will call fea->Solve().
    std::unique_ptr<tlfea::SolverBase> fea;

    bool   initialized{false};
    double time_step{1e-4};  ///< Co-simulation time step [s].
};

// ── Constructor / Destructor ──────────────────────────────────────────────────

TLFEADEMCoupler::TLFEADEMCoupler()
    : impl_(std::make_unique<Impl>()) {
    std::cout << "[MoPhi] TLFEADEMCoupler: created\n";
}

TLFEADEMCoupler::~TLFEADEMCoupler() {
    if (impl_ && impl_->initialized) {
        finalize();
    }
}

// ── Public interface ──────────────────────────────────────────────────────────

void TLFEADEMCoupler::initialize(const std::string& tlfea_config,
                                  const std::string& dem_config,
                                  unsigned int       num_gpus) {
    std::cout << "[MoPhi] TLFEADEMCoupler: initializing ...\n";

    // ── DEM-Engine ────────────────────────────────────────────────────────────
    // Create DEMSolver with the requested number of GPUs.  In a real simulation
    // the user would configure domain, materials, and clump templates on this
    // object and then call impl_->dem->Initialize() before the time-stepping loop.
    impl_->dem = std::make_unique<deme::DEMSolver>(num_gpus);
    std::cout << "[MoPhi] TLFEADEMCoupler: deme::DEMSolver created (nGPUs=" << num_gpus << ")"
              << (dem_config.empty() ? "" : " (config: " + dem_config + ")")
              << "\n";

    // ── TLFEA ─────────────────────────────────────────────────────────────────
    // TLFEA solvers require an ElementBase* (mesh + DOF data) in their
    // constructors; a standalone driver class does not exist yet.  The fea
    // pointer is left null as a placeholder.
    impl_->fea = nullptr;
    std::cout << "[MoPhi] TLFEADEMCoupler: tlfea::SolverBase placeholder ready"
              << (tlfea_config.empty() ? "" : " (config: " + tlfea_config + ")")
              << "\n";

    impl_->initialized = true;
    std::cout << "[MoPhi] TLFEADEMCoupler: initialized\n";
}

void TLFEADEMCoupler::step() {
    if (!impl_->initialized) {
        throw std::runtime_error("TLFEADEMCoupler::step() called before initialize().");
    }

    // DEM step: advance by one co-simulation time step.
    // Uncomment once the solver has been fully configured and initialized:
    // impl_->dem->DoDynamicsThenSync(impl_->time_step);

    // FEA step: call when a concrete SolverBase subclass has been assigned.
    if (impl_->fea) {
        impl_->fea->Solve();
    }
}

void TLFEADEMCoupler::finalize() {
    std::cout << "[MoPhi] TLFEADEMCoupler: finalizing ...\n";
    impl_->fea.reset();
    impl_->dem.reset();
    impl_->initialized = false;
    std::cout << "[MoPhi] TLFEADEMCoupler: finalized\n";
}

} // namespace mophi
