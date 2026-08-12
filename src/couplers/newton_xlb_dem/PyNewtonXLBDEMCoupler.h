#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <core/Logger.hpp>
#include <pybind11/pybind11.h>

// ─────────────────────────────────────────────────────────────────────────────
// PyNewtonXLBDEMCoupler — three-way co-simulation coupler bridging
//   • Newton (Python GPU rigid-body / articulation physics via NVIDIA Warp)
//   • XLB    (Python GPU lattice-Boltzmann fluid solver via JAX)
//   • DEME   (Python discrete-element solver, provided by the configured distribution)
//
// Newton is the primary physics solver in this coupler.  It drives an
// articulated walking robot and produces a spatial representation of the robot
// (body position + orientation quaternions) at every time step.
//
// XLB (jax-lattice-boltzmann) is a real LBM fluid solver.  The robot is
// represented as a prescribed AABB obstacle that moves with the Newton base
// body each frame.  bc_mask and missing_mask are updated on-device via Warp
// kernels without any CPU round-trip.
//
// DEME is an active discrete-element solver in this setup: a Python
// deme.DEMSolver object is provided by the caller, retained by the coupler,
// and advanced during StepDEME().  DEME is used via its Python API so that
// no C++ build of DEM-Engine is required for this coupler.
//
// Per-solver stepping methods (no whole-sale stepper):
//   StepNewton() — advance Newton by one substep (clear forces → collide → step → swap states).
//   StepDEME()   — advance DEME by one substep (calls DoStepDynamics() on the Python object).
//   StepXLB()    — reserved hook for XLB stepping; currently a no-op because
//                  the Python demo advances the XLB stepper directly.
//
// The demo controls the pace of each system independently.  For example, Newton and DEME
// may be advanced every substep while XLB is advanced only once per policy frame.
//
// On-device data communication infrastructure
// ──────────────────────────────────────────
// All three solvers are built on NVIDIA Warp and keep their physics state
// in device-resident wp.array objects.  This coupler exposes the following
// accessor methods so that callers can reach device arrays directly, without
// a GPU→CPU copy:
//
//   GetNewtonBodyQArray()        — returns newton_state_0.body_q  (Warp array,
//                            shape (body_count, 7), dtype wp.transform)
//   GetXLBBCMaskArray()       — returns the stored XLB bc_mask Warp array
//                            (shape (1, NX, NY, NZ), dtype uint8)
//   GetXLBMissingMaskArray()  — returns the stored XLB missing_mask Warp array
//                            (shape (Q, NX, NY, NZ), dtype bool)
//   SetXLBMasks(bc, mm)    — binds the XLB bc_mask / missing_mask Warp arrays
//                            to this coupler instance
//
// The demo uses GetNewtonBodyQArray() to derive the robot AABB on GPU and then
// launches Warp kernels that write directly into GetXLBBCMaskArray() and
// GetXLBMissingMaskArray(), replacing the former CPU numpy path entirely.
//
// Lives in src/couplers/newton_xlb_dem/.
// The declaration is compiled as part of the mophi_core Python extension module
// (not as a standalone library) because it depends on pybind11 types.
// The implementation (PyNewtonXLBDEMCoupler.cpp) is added as a source to
// mophi_core by python/CMakeLists.txt.
// ─────────────────────────────────────────────────────────────────────────────

struct PyNewtonXLBDEMCoupler {
    // DEME object: kept alive via pybind11 reference counting (may be None).
    pybind11::object deme_solver;  ///< deme.DEMSolver Python instance, or None

    // Newton objects: kept alive here via pybind11 reference counting.
    // All are initialized to None and populated in Initialize().
    pybind11::object newton_model;
    pybind11::object newton_solver;
    pybind11::object newton_state_0;  ///< "current"  Newton state (written by StepNewton())
    pybind11::object newton_state_1;  ///< "scratch"   Newton state (work buffer in StepNewton())
    pybind11::object newton_control;
    pybind11::object newton_contacts;

    // XLB object: kept alive via pybind11 reference counting (may be None).
    pybind11::object xlb_simulation;  ///< XLB simulation instance, or None

    // XLB device-resident mask arrays (Warp arrays, bound via SetXLBMasks()).
    // Kept here so that GPU-kernel-based coupling code can obtain the device
    // handles in one place without holding additional module-level references.
    pybind11::object xlb_bc_mask;       ///< wp.array (1, NX, NY, NZ) uint8,  or None
    pybind11::object xlb_missing_mask;  ///< wp.array (Q, NX, NY, NZ) bool,   or None

    // External body forces injected by a coupled solver (e.g. DEME contact forces).
    // Stored as a wp.array of shape (body_count,) dtype=wp.spatial_vector, or None.
    // Applied inside StepNewton() after clear_forces()+collide() and before step(),
    // so Newton integrates them together with internal joint-actuator torques.
    // Set via SetNewtonBodyForces(); cleared to None in Finalize().
    pybind11::object newton_ext_body_forces;  ///< wp.array (body_count,) spatial_vector, or None

    double sim_dt{1.0 / 1000.0};  ///< Co-simulation time step [s]
    int step_count{0};            ///< Number of StepNewton() calls completed
    bool deme_available{false};
    bool newton_available{false};
    bool xlb_available{false};
    bool initialized{false};  ///< True after Initialize() completes; checked in step methods and destructor

    PyNewtonXLBDEMCoupler();
    ~PyNewtonXLBDEMCoupler();

    // Non-copyable (pybind11 objects inhibit copy).
    PyNewtonXLBDEMCoupler(const PyNewtonXLBDEMCoupler&) = delete;
    PyNewtonXLBDEMCoupler& operator=(const PyNewtonXLBDEMCoupler&) = delete;

    /// @brief Initialize all three solvers.
    ///
    /// @param newton_model_in    A newton.Model instance for the walking robot (or None).
    /// @param newton_solver_in   A Newton solver instance, e.g. newton.solvers.SolverXPBD
    ///                           (or None to skip Newton).
    /// @param xlb_simulation_in  An XLB simulation instance (or None to skip XLB).
    /// @param deme_solver_in     A deme.DEMSolver Python instance (or None to skip DEME).
    /// @param dt                 Co-simulation time step [s].
    void Initialize(pybind11::object newton_model_in,
                    pybind11::object newton_solver_in,
                    pybind11::object xlb_simulation_in,
                    pybind11::object deme_solver_in,
                    double dt);

    /// @brief Advance Newton by one substep.
    ///
    /// Sequence: clear forces → collide → step → swap states.
    /// Call this every substep.  Joint targets must be written to
    /// \c newton_control before this call.
    void StepNewton();

    /// @brief Advance DEME by one substep.
    ///
    /// Calls \c DoStepDynamics() on the bound deme.DEMSolver Python object.
    /// When no DEME solver was provided this method is a no-op.
    void StepDEME();

    /// @brief Advance XLB by one substep.
    ///
    /// Placeholder — calls \c xlb_simulation.step() once future fluid–robot
    /// coupling is implemented.  When no XLB simulation was provided this
    /// method is a no-op.
    void StepXLB();

    /// @brief Finalize all solvers and release all resources.
    void Finalize();

    /// @brief Set the MoPhi logger verbosity level.
    void SetVerbosity(mophi::verbosity_t verbose);

    /// @brief Return the current spatial representation of the robot.
    ///
    /// Each entry is {px, py, pz, qx, qy, qz, qw} — the world-space position
    /// and orientation quaternion of one robot body.  The vector has one entry
    /// per Newton body (model.body_count).  Returns an empty vector when Newton
    /// has not been initialized.
    ///
    /// @note This method performs a synchronous GPU→CPU copy via body_q.numpy().
    ///       For GPU-native (zero-copy) access, use GetNewtonBodyQArray() instead and
    ///       work directly with the returned Warp array (e.g. wp.to_torch()).
    std::vector<std::array<double, 7>> GetRobotBodyTransforms() const;

    // ── On-device data communication infrastructure ────────────────────────────
    // These methods expose the device-resident Warp arrays owned by each solver
    // so that coupling code can read from Newton and write to XLB entirely on-
    // device, with no CPU round-trip.

    /// @brief Return the device-resident body_q Warp array from newton_state_0.
    ///
    /// body_q has shape (body_count, 7) and dtype wp.transform — each row stores
    /// [px, py, pz, qx, qy, qz, qw] for one robot body.
    /// Returns None when Newton has not been initialized.
    ///
    /// Callers can use this to avoid the full GPU→CPU copy of GetRobotBodyTransforms():
    ///   • body_q.numpy()           — CPU copy, same as before but through the
    ///                                 device-handle interface (good for small reads)
    ///   • wp.to_torch(body_q)[...]  — zero-copy PyTorch view on the same device
    pybind11::object GetNewtonBodyQArray() const;

    /// @brief Bind the XLB bc_mask and missing_mask Warp arrays to this coupler.
    ///
    /// Both arrays are device-resident wp.array objects returned by the XLB
    /// stepper's prepare_fields() call.  After binding, GetXLBBCMaskArray() and
    /// GetXLBMissingMaskArray() return these handles so that Warp GPU kernels in
    /// the demo can update the obstacle masks in-place without any CPU copy or
    /// array reallocation.
    ///
    /// @param bc_mask       wp.array of shape (1, NX, NY, NZ), dtype uint8.
    /// @param missing_mask  wp.array of shape (Q, NX, NY, NZ), dtype bool.
    void SetXLBMasks(pybind11::object bc_mask, pybind11::object missing_mask);

    /// @brief Return the stored XLB bc_mask Warp array, or None if not yet set.
    pybind11::object GetXLBBCMaskArray() const;

    /// @brief Return the stored XLB missing_mask Warp array, or None if not yet set.
    pybind11::object GetXLBMissingMaskArray() const;

    // ── DEME → Newton force feedback ───────────────────────────────────────────

    /// @brief Register an external body-force array to be applied in the next StepNewton() call.
    ///
    /// @p ext_forces_wp must be a wp.array of shape (body_count,) with dtype
    /// wp.spatial_vector.  Each entry is [fx, fy, fz, tx, ty, tz] in world space.
    ///
    /// StepNewton() copies the forces directly between device arrays into \c newton_state_0.body_f after
    /// clear_forces() and collide() but before step().  The stored array is replaced every time this method is called;
    /// pass \c None to disable force injection for subsequent steps.
    ///
    /// The caller may populate the array directly from another CUDA solver.
    void SetNewtonBodyForces(pybind11::object ext_forces_wp);
};
