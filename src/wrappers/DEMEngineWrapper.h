#pragma once

#include <memory>
#include <string>

namespace mophi {

/// @brief Wrapper around the DEM-Engine (Discrete Element Method) solver.
///
/// Provides a uniform MoPhi interface regardless of whether the DEMEngine
/// external project has been fetched.  When built without DEMEngine the methods
/// log a diagnostic message so the rest of the framework can still compile and
/// the Python bindings remain functional.
///
/// Integration status is reflected by the compile-time macro MOPHI_HAS_DEMENGINE,
/// which is defined by src/wrappers/CMakeLists.txt when DEMEngine is available.
class DEMEngineWrapper {
public:
    DEMEngineWrapper();
    ~DEMEngineWrapper();

    // Non-copyable, movable.
    DEMEngineWrapper(const DEMEngineWrapper&)            = delete;
    DEMEngineWrapper& operator=(const DEMEngineWrapper&) = delete;
    DEMEngineWrapper(DEMEngineWrapper&&)                 = default;
    DEMEngineWrapper& operator=(DEMEngineWrapper&&)      = default;

    /// @brief Initialize the DEM-Engine solver.
    /// @param config_file  Path to a solver configuration / scene description file.
    void initialize(const std::string& config_file = "");

    /// @brief Advance the simulation by one time step.
    void step();

    /// @brief Finalize the solver and release all resources.
    void finalize();

    /// @returns true if the solver was compiled with the real DEM-Engine back-end.
    static bool isAvailable() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mophi
