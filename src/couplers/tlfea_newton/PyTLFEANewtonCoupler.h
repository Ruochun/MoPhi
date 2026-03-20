#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>

// ─────────────────────────────────────────────────────────────────────────────
// PyTLFEANewtonCoupler — co-simulation coupler bridging TLFEA (C++) and Newton
//
// TLFEA is a CUDA-based FEA solver whose headers carry __host__/__device__
// annotations.  The pimpl (TLFEAImpl) keeps those headers out of this
// declaration and limits TLFEA-specific compilation to PyTLFEANewtonCoupler.cpp.
//
// Newton is a pure Python package (GPU-accelerated via NVIDIA Warp) with no
// C++ ABI.  Newton objects are held as pybind11::object and driven through the
// Python C-API.
//
// step() coupling sequence:
//   1. Advance TLFEA by one time step.
//   2. Extract deformed TLFEA node positions (coupling output).
//   3. TODO: update Newton collision geometry / anchor points with the TLFEA positions.
//   4. Advance Newton by one time step (clear forces -> collide -> step -> swap).
//   5. TODO: extract contact / body forces from Newton state.
//   6. TODO: apply those forces to TLFEA via SetNodeForces() for the next step.
//
// Lives in src/couplers/tlfea_newton/.  The declaration is compiled as part of
// the mophi_core Python extension module (not as a standalone library) because
// it depends on pybind11 types.  The implementation (PyTLFEANewtonCoupler.cpp)
// is added as a source to mophi_core by python/CMakeLists.txt.
// ─────────────────────────────────────────────────────────────────────────────

struct PyTLFEANewtonCoupler {
    /// pimpl hiding tlfea::FEASolver and CUDA headers — defined in PyTLFEANewtonCoupler.cpp.
    struct TLFEAImpl;
    std::unique_ptr<TLFEAImpl> fea_;

    // Newton objects owned on the Python side and kept alive here via reference
    // counting.  All are initialized to None and populated in Initialize().
    pybind11::object newton_model;
    pybind11::object newton_solver;
    pybind11::object newton_state_0;  ///< "current" Newton state (input to Step())
    pybind11::object newton_state_1;  ///< "scratch"  Newton state (output of Step())
    pybind11::object newton_control;
    pybind11::object newton_contacts;
    double sim_dt{1.0 / 1000.0};  ///< Newton integration time step [s]
    bool newton_available{false};

    PyTLFEANewtonCoupler();
    ~PyTLFEANewtonCoupler();

    // Non-copyable (holds GPU resources via TLFEA and/or pybind11 objects).
    PyTLFEANewtonCoupler(const PyTLFEANewtonCoupler&) = delete;
    PyTLFEANewtonCoupler& operator=(const PyTLFEANewtonCoupler&) = delete;

    /// @brief Initialize TLFEA and bind the Newton model and solver.
    ///
    /// @param tlfea_config   Configuration/mesh file path forwarded to TLFEA (empty string = no config file).
    /// @param newton_model   A newton.Model instance (or None to skip Newton).
    /// @param newton_solver  A Newton solver instance, e.g. newton.solvers.SolverXPBD
    ///                       (or None to skip Newton).
    /// @param dt             Newton integration time step [s].
    void Initialize(const std::string& tlfea_config,
                    pybind11::object newton_model_in,
                    pybind11::object newton_solver_in,
                    double dt);

    /// @brief Advance one co-simulation step (TLFEA + data exchange + Newton).
    void Step();

    /// @brief Finalize both solvers and release all resources.
    void Finalize();

    /// @brief Return deformed positions of all TLFEA simulation nodes.
    ///
    /// Primary coupling output: the Python layer reads these positions after
    /// every TLFEA step and forwards them to Newton (e.g. as collision geometry
    /// or as rigid-body anchor points).
    std::vector<std::array<double, 3>> GetNodePositions() const;

    /// @brief Apply external forces to TLFEA simulation nodes (e.g. from Newton).
    ///
    /// Primary coupling input: the Python layer calls this after every Newton
    /// step to feed contact or body forces back into TLFEA before the next advance.
    ///
    /// @param forces  One {fx, fy, fz} entry per simulation node.
    void SetNodeForces(const std::vector<std::array<double, 3>>& forces);
};
