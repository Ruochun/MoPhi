#pragma once

#include <memory>
#include <string>

namespace mophi {

/// @brief Placeholder co-simulation solver coupling TLFEA and DEM-Engine.
///
/// This is the first concrete example of a MoPhi multi-physics solver.  It
/// demonstrates the pattern that all future co-simulation solvers should follow:
///
///   1. Own wrapper instances for every participating single-physics solver.
///   2. Provide initialize() / step() / finalize() lifecycle methods.
///   3. Expose a Python binding via the python/ sub-directory.
///
/// The placeholder implementation simply creates the two solver wrappers and
/// immediately destroys them, which is sufficient to verify that the whole
/// build and Python-lization infrastructure is in place.
///
/// Later, the bodies of initialize(), step(), and finalize() will be filled in
/// with real data-exchange logic (force/displacement mapping, time-step
/// synchronization, etc.).
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
    void initialize(const std::string& tlfea_config = "",
                    const std::string& dem_config   = "");

    /// @brief Advance both solvers by one co-simulation time step.
    void step();

    /// @brief Finalize both solvers and release all resources.
    void finalize();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mophi
