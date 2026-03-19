#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace mophi {

/// @brief Co-simulation coupler coupling TLFEA (FEA) and Newton (Python-based GPU physics).
///
/// TLFEANewtonCoupler holds TLFEA's C++ FEA solver on the C++ side.  Newton is a
/// pure Python physics engine (GPU-accelerated, built on NVIDIA Warp).  The coupling
/// is orchestrated by PyTLFEANewtonCoupler (src/couplers/tlfea_newton/PyTLFEANewtonCoupler.h),
/// which owns Newton objects (model, solver, states) as py::object alongside this class.
///
/// Data exchange coupling points:
///   - get_node_positions() → TLFEA deformed node positions forwarded to Newton geometry
///   - set_node_forces()    ← contact/body forces from Newton applied to TLFEA nodes
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
    /// In a full co-simulation the Python layer calls this, then drives Newton,
    /// exchanging data via get_node_positions() / set_node_forces().  For now
    /// the body is a placeholder.
    void step();

    /// @brief Finalize the TLFEA solver and release all resources.
    void finalize();

    /// @brief Return deformed positions of all TLFEA simulation nodes.
    ///
    /// Each entry is {x, y, z} in the deformed configuration.  This is the
    /// primary coupling output: the Python layer reads these positions after
    /// every TLFEA step and forwards them to Newton (e.g. as collision geometry
    /// or as rigid-body anchor points).
    ///
    /// @return Empty vector before initialize() is called.  Filled with actual
    ///         node positions once the solver has been fully configured and
    ///         impl_->fea->GetNodePositions() is available.
    std::vector<std::array<double, 3>> get_node_positions() const;

    /// @brief Apply external forces to TLFEA simulation nodes (e.g. from Newton).
    ///
    /// This is the primary coupling input: the Python layer calls this after
    /// every Newton step to feed contact or body forces computed by Newton back
    /// into the TLFEA solver before the next TLFEA advance.
    ///
    /// @param forces  One {fx, fy, fz} entry per simulation node.  An empty
    ///                vector is silently ignored.
    void set_node_forces(const std::vector<std::array<double, 3>>& forces);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace mophi
