#pragma once

#include <memory>
#include <string>

namespace mophi {

/// @brief Placeholder co-simulation solver coupling TLFEA and DEM-Engine.
///
/// This is the first concrete example of a MoPhi multi-physics solver.  It
/// demonstrates the pattern that all future co-simulation solvers should follow:
///
///   1. Own instances of the participating single-physics solvers directly
///      (via pimpl), using each solver's own public API without an intervening
///      wrapper layer.
///   2. Provide initialize() / step() / finalize() lifecycle methods.
///   3. Expose a Python binding via the python/ sub-directory.
///
/// Impl holds:
///   - std::unique_ptr<deme::DEMSolver>      — DEM-Engine's main solver class
///   - std::unique_ptr<tlfea::SolverBase>    — TLFEA's solver interface
///     (placeholder nullptr until TLFEA gains a top-level simulation driver)
///
/// Building this class requires both the TLFEA and DEMEngine external projects
/// to have been fetched.  CMake emits a FATAL_ERROR at configure time if either
/// is missing and MOPHI_BUILD_TLFEA_DEM=ON is requested.
///
/// Later, initialize(), step(), and finalize() will be filled in with real
/// data-exchange logic (force/displacement mapping, time-step synchronization,
/// etc.).
class TLFEADEMCoupler {
public:
    TLFEADEMCoupler();
    ~TLFEADEMCoupler();

    // Non-copyable, movable.
    TLFEADEMCoupler(const TLFEADEMCoupler&)            = delete;
    TLFEADEMCoupler& operator=(const TLFEADEMCoupler&) = delete;
    TLFEADEMCoupler(TLFEADEMCoupler&&)                 = default;
    TLFEADEMCoupler& operator=(TLFEADEMCoupler&&)      = default;

    /// @brief Initialize both solvers.
    /// @param tlfea_config   Configuration/mesh file forwarded to TLFEA.
    /// @param dem_config     Configuration/scene file forwarded to DEM-Engine.
    /// @param num_gpus       Number of GPUs to hand to DEM-Engine (default 1).
    void initialize(const std::string& tlfea_config = "",
                    const std::string& dem_config   = "",
                    unsigned int       num_gpus     = 1);

    /// @brief Advance both solvers by one co-simulation time step.
    void step();

    /// @brief Finalize both solvers and release all resources.
    void finalize();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mophi
