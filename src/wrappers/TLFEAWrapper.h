#pragma once

#include <memory>
#include <string>

namespace mophi {

/// @brief Wrapper around the TLFEA (Total-Lagrangian Finite Element Analysis) solver.
///
/// Provides a uniform MoPhi interface regardless of whether the TLFEA external
/// project has been fetched.  When built without TLFEA the methods log a
/// diagnostic message so the rest of the framework can still compile and the
/// Python bindings remain functional.
///
/// Integration status is reflected by the compile-time macro MOPHI_HAS_TLFEA,
/// which is defined by src/wrappers/CMakeLists.txt when TLFEA is available.
class TLFEAWrapper {
public:
    TLFEAWrapper();
    ~TLFEAWrapper();

    // Non-copyable, movable.
    TLFEAWrapper(const TLFEAWrapper&)            = delete;
    TLFEAWrapper& operator=(const TLFEAWrapper&) = delete;
    TLFEAWrapper(TLFEAWrapper&&)                 = default;
    TLFEAWrapper& operator=(TLFEAWrapper&&)      = default;

    /// @brief Initialize the TLFEA solver.
    /// @param config_file  Path to a solver configuration / mesh file.
    void initialize(const std::string& config_file = "");

    /// @brief Advance the simulation by one time step.
    void step();

    /// @brief Finalize the solver and release all resources.
    void finalize();

    /// @returns true if the solver was compiled with the real TLFEA back-end.
    static bool isAvailable() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mophi
