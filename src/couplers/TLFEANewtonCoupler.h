#pragma once

#include <memory>
#include <string>

namespace mophi {

/// @brief Co-simulation coupler coupling TLFEA (FEA) and Newton (Python-based GPU physics).
///
/// TLFEANewtonCoupler holds TLFEA's C++ FEA solver on the C++ side.  Newton is a
/// pure Python physics engine (GPU-accelerated, built on NVIDIA Warp) and therefore
/// runs entirely in the Python layer.  The coupling data exchange happens in the
/// Python driver script, not inside this C++ class.
///
/// Impl holds:
///   - std::unique_ptr<tlfea::FEASolver>  — TLFEA's top-level simulation driver
///
/// TLFEA is a CUDA-based FEA library — its headers contain CUDA annotations
/// (__host__/__device__) that require the CUDA Toolkit to compile.  Building
/// this class therefore requires the CUDA Toolkit even though DEM-Engine is not
/// involved.
///
/// Building this class requires the TLFEA external project to have been fetched.
/// CMake emits a FATAL_ERROR at configure time if TLFEA is missing and
/// MOPHI_BUILD_TLFEA_NEWTON=ON is requested.
class TLFEANewtonCoupler {
  public:
    TLFEANewtonCoupler();
    ~TLFEANewtonCoupler();

    // Non-copyable, movable.
    TLFEANewtonCoupler(const TLFEANewtonCoupler&) = delete;
    TLFEANewtonCoupler& operator=(const TLFEANewtonCoupler&) = delete;
    TLFEANewtonCoupler(TLFEANewtonCoupler&&) = default;
    TLFEANewtonCoupler& operator=(TLFEANewtonCoupler&&) = default;

    /// @brief Initialize the TLFEA solver.
    /// @param tlfea_config  Configuration/mesh file forwarded to TLFEA (empty = default).
    void initialize(const std::string& tlfea_config = "");

    /// @brief Advance the TLFEA solver by one co-simulation time step.
    ///
    /// In a real co-simulation the Newton side would be stepped in Python, and the
    /// exchanged data (forces, displacements) would be passed in and out of this
    /// method.  For now the body is intentionally a placeholder.
    void step();

    /// @brief Finalize the TLFEA solver and release all resources.
    void finalize();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace mophi
