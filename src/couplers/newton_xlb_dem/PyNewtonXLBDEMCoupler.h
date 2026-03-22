#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>

// ─────────────────────────────────────────────────────────────────────────────
// PyNewtonXLBDEMCoupler — three-way co-simulation coupler bridging
//   • Newton (Python GPU rigid-body / articulation physics via NVIDIA Warp)
//   • XLB    (Python GPU lattice-Boltzmann fluid solver via JAX)
//   • DEME   (Python discrete-element solver, pip install deme)
//
// Newton is the primary physics solver in this coupler.  It drives an
// articulated walking robot and produces a spatial representation of the robot
// (body position + orientation quaternions) at every time step.
//
// XLB (jax-lattice-boltzmann) is a placeholder for now: the solver is
// instantiated inside Initialize() but its step() method is not called with
// any real physics during Step().  Future work will feed the robot's spatial
// representation into XLB as a moving boundary condition.
//
// DEME is likewise a placeholder: a deme.DEMSolver Python object is created
// inside Initialize() but the simulation is not advanced during Step().
// Future work will introduce particle–robot coupling (particles interacting
// with the robot's surface mesh).  DEME is used via pip install deme so that
// no C++ build of DEM-Engine is required for this coupler.
//
// step() coupling sequence:
//   1. Advance Newton by one time step (clears forces, collides, steps, swaps states).
//   2. Extract robot body transforms (the spatial representation).
//   3. TODO: feed robot geometry into DEME particle field.
//   4. TODO: feed robot geometry into XLB fluid boundary.
//   5. DEME placeholder step (no-op for now).
//   6. XLB placeholder step (no-op for now).
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
    pybind11::object newton_state_0;  ///< "current"  Newton state (input to Step())
    pybind11::object newton_state_1;  ///< "scratch"   Newton state (output of Step())
    pybind11::object newton_control;
    pybind11::object newton_contacts;

    // XLB object: kept alive via pybind11 reference counting (may be None).
    pybind11::object xlb_simulation;  ///< XLB simulation instance, or None

    double sim_dt{1.0 / 1000.0};  ///< Co-simulation time step [s]
    int step_count{0};            ///< Number of Step() calls completed
    bool deme_available{false};
    bool newton_available{false};
    bool xlb_available{false};
    bool initialized{false};  ///< True after all Initialize() steps complete

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
    /// @param dt                 Co-simulation time step [s].
    /// @param num_gpus           Number of GPUs to pass to the DEME solver.
    void Initialize(pybind11::object newton_model_in,
                    pybind11::object newton_solver_in,
                    pybind11::object xlb_simulation_in,
                    double dt,
                    unsigned int num_gpus);

    /// @brief Advance one co-simulation step.
    ///
    /// Advances Newton (clear forces → collide → step → swap states), then
    /// performs no-op placeholder steps for DEME and XLB.
    void Step();

    /// @brief Finalize all solvers and release all resources.
    void Finalize();

    /// @brief Return the current spatial representation of the robot.
    ///
    /// Each entry is {px, py, pz, qx, qy, qz, qw} — the world-space position
    /// and orientation quaternion of one robot body.  The vector has one entry
    /// per Newton body (model.body_count).  Returns an empty vector when Newton
    /// has not been initialized.
    ///
    /// This is the primary coupling output: the Python layer (or future C++
    /// coupling code) reads these transforms after every Newton step and uses
    /// them to update DEME's particle field geometry and XLB's moving
    /// boundary condition.
    std::vector<std::array<double, 7>> GetRobotBodyTransforms() const;
};
