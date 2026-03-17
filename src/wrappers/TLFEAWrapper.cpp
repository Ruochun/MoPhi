#include "TLFEAWrapper.h"

#include <iostream>
#include <stdexcept>

// When TLFEA is fetched and its CMakeLists.txt integrated, the build system
// defines MOPHI_HAS_TLFEA and adds the necessary include paths / link targets.
#ifdef MOPHI_HAS_TLFEA
// #include <tlfea/tlfea.h>   // ← uncomment once TLFEA exposes a public header
#endif

namespace mophi {

// ── Pimpl ────────────────────────────────────────────────────────────────────

struct TLFEAWrapper::Impl {
#ifdef MOPHI_HAS_TLFEA
    // std::unique_ptr<tlfea::Simulation> sim;  // ← real solver handle
#endif
    bool initialized{false};
};

// ── Constructor / Destructor ──────────────────────────────────────────────────

TLFEAWrapper::TLFEAWrapper()
    : impl_(std::make_unique<Impl>()) {}

TLFEAWrapper::~TLFEAWrapper() {
    if (impl_ && impl_->initialized) {
        finalize();
    }
}

// ── Public interface ──────────────────────────────────────────────────────────

void TLFEAWrapper::initialize(const std::string& config_file) {
#ifdef MOPHI_HAS_TLFEA
    // impl_->sim = std::make_unique<tlfea::Simulation>(config_file);
    // impl_->sim->initialize();
    impl_->initialized = true;
    std::cout << "[MoPhi] TLFEAWrapper: initialized"
              << (config_file.empty() ? "" : " with " + config_file) << "\n";
#else
    (void)config_file;
    std::cout << "[MoPhi] TLFEAWrapper: TLFEA not fetched — running in stub mode.\n"
              << "         Re-run CMake with -DMOPHI_FETCH_TLFEA=ON to enable.\n";
    impl_->initialized = true;
#endif
}

void TLFEAWrapper::step() {
    if (!impl_->initialized) {
        throw std::runtime_error("TLFEAWrapper::step() called before initialize().");
    }
#ifdef MOPHI_HAS_TLFEA
    // impl_->sim->step();
#else
    std::cout << "[MoPhi] TLFEAWrapper::step(): stub\n";
#endif
}

void TLFEAWrapper::finalize() {
#ifdef MOPHI_HAS_TLFEA
    // impl_->sim->finalize();
    impl_->initialized = false;
    std::cout << "[MoPhi] TLFEAWrapper: finalized\n";
#else
    impl_->initialized = false;
    std::cout << "[MoPhi] TLFEAWrapper::finalize(): stub\n";
#endif
}

bool TLFEAWrapper::isAvailable() noexcept {
#ifdef MOPHI_HAS_TLFEA
    return true;
#else
    return false;
#endif
}

} // namespace mophi
