#include "TLFEADEMCoupler.h"

#include <iostream>

#include "TLFEAWrapper.h"
#include "DEMEngineWrapper.h"

namespace mophi {

// ── Pimpl ────────────────────────────────────────────────────────────────────

struct TLFEADEMCoupler::Impl {
    TLFEAWrapper    tlfea;
    DEMEngineWrapper dem;
    bool initialized{false};
};

// ── Constructor / Destructor ──────────────────────────────────────────────────

TLFEADEMCoupler::TLFEADEMCoupler()
    : impl_(std::make_unique<Impl>()) {
    std::cout << "[MoPhi] TLFEADEMCoupler: created  "
              << "(TLFEA available=" << TLFEAWrapper::isAvailable()
              << ", DEM-Engine available=" << DEMEngineWrapper::isAvailable()
              << ")\n";
}

TLFEADEMCoupler::~TLFEADEMCoupler() {
    if (impl_ && impl_->initialized) {
        finalize();
    }
}

// ── Public interface ──────────────────────────────────────────────────────────

void TLFEADEMCoupler::initialize(const std::string& tlfea_config,
                                  const std::string& dem_config) {
    std::cout << "[MoPhi] TLFEADEMCoupler: initializing ...\n";
    impl_->tlfea.initialize(tlfea_config);
    impl_->dem.initialize(dem_config);
    impl_->initialized = true;
    std::cout << "[MoPhi] TLFEADEMCoupler: initialized\n";
}

void TLFEADEMCoupler::step() {
    if (!impl_->initialized) {
        throw std::runtime_error("TLFEADEMCoupler::step() called before initialize().");
    }
    // Future: exchange data between solvers before / after each sub-step.
    impl_->tlfea.step();
    impl_->dem.step();
}

void TLFEADEMCoupler::finalize() {
    std::cout << "[MoPhi] TLFEADEMCoupler: finalizing ...\n";
    impl_->tlfea.finalize();
    impl_->dem.finalize();
    impl_->initialized = false;
    std::cout << "[MoPhi] TLFEADEMCoupler: finalized\n";
}

} // namespace mophi
