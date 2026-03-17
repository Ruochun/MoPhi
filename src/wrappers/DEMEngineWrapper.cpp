#include "DEMEngineWrapper.h"

#include <iostream>
#include <stdexcept>

// When DEMEngine is fetched and its CMakeLists.txt integrated, the build system
// defines MOPHI_HAS_DEMENGINE and adds the necessary include paths / link targets.
#ifdef MOPHI_HAS_DEMENGINE
// #include <DEM/API.h>   // ← uncomment once DEM-Engine exposes a public header
#endif

namespace mophi {

// ── Pimpl ────────────────────────────────────────────────────────────────────

struct DEMEngineWrapper::Impl {
#ifdef MOPHI_HAS_DEMENGINE
    // std::unique_ptr<deme::DEMSolver> solver;  // ← real solver handle
#endif
    bool initialized{false};
};

// ── Constructor / Destructor ──────────────────────────────────────────────────

DEMEngineWrapper::DEMEngineWrapper()
    : impl_(std::make_unique<Impl>()) {}

DEMEngineWrapper::~DEMEngineWrapper() {
    if (impl_ && impl_->initialized) {
        finalize();
    }
}

// ── Public interface ──────────────────────────────────────────────────────────

void DEMEngineWrapper::initialize(const std::string& config_file) {
#ifdef MOPHI_HAS_DEMENGINE
    // impl_->solver = std::make_unique<deme::DEMSolver>();
    // impl_->solver->Initialize();
    impl_->initialized = true;
    std::cout << "[MoPhi] DEMEngineWrapper: initialized"
              << (config_file.empty() ? "" : " with " + config_file) << "\n";
#else
    (void)config_file;
    std::cout << "[MoPhi] DEMEngineWrapper: DEMEngine not fetched — running in stub mode.\n"
              << "         Re-run CMake with -DMOPHI_FETCH_DEMENGINE=ON to enable.\n";
    impl_->initialized = true;
#endif
}

void DEMEngineWrapper::step() {
    if (!impl_->initialized) {
        throw std::runtime_error("DEMEngineWrapper::step() called before initialize().");
    }
#ifdef MOPHI_HAS_DEMENGINE
    // impl_->solver->DoDynamicsThenSync(dt);
#else
    std::cout << "[MoPhi] DEMEngineWrapper::step(): stub\n";
#endif
}

void DEMEngineWrapper::finalize() {
#ifdef MOPHI_HAS_DEMENGINE
    // impl_->solver.reset();
    impl_->initialized = false;
    std::cout << "[MoPhi] DEMEngineWrapper: finalized\n";
#else
    impl_->initialized = false;
    std::cout << "[MoPhi] DEMEngineWrapper::finalize(): stub\n";
#endif
}

bool DEMEngineWrapper::isAvailable() noexcept {
#ifdef MOPHI_HAS_DEMENGINE
    return true;
#else
    return false;
#endif
}

} // namespace mophi
