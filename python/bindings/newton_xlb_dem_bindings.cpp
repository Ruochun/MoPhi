#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// NewtonXLBDEMCoupler is only registered when the library was compiled with
// MOPHI_BUILD_NEWTON_XLB_DEM=ON.  Newton, XLB, and DEME are all pure Python
// packages and are not linked into this module.
// PyNewtonXLBDEMCoupler (defined in src/couplers/newton_xlb_dem/PyNewtonXLBDEMCoupler.h)
// bridges all three solvers: Newton drives a walking robot in Python, XLB provides
// a real LBM fluid solver in Python, and DEME is a Python discrete-element solver
// (pip install deme) imported at runtime via pybind11.
#ifdef MOPHI_HAS_NEWTON_XLB_DEM_COUPLER
    #include "PyNewtonXLBDEMCoupler.h"  // src/couplers/newton_xlb_dem/
#endif

namespace py = pybind11;

// ─────────────────────────────────────────────────────────────────────────────
// register_newton_xlb_dem — pybind11 registration for NewtonXLBDEMCoupler.
//
// Registers the NewtonXLBDEMCoupler class (backed by PyNewtonXLBDEMCoupler) into
// the mophi_core module.  Called unconditionally from PYBIND11_MODULE in
// mophi_bindings.cpp; the function body is a no-op when
// MOPHI_HAS_NEWTON_XLB_DEM_COUPLER is not defined.
// ─────────────────────────────────────────────────────────────────────────────

void register_newton_xlb_dem(py::module_& m) {
#ifdef MOPHI_HAS_NEWTON_XLB_DEM_COUPLER
    // Three-way co-simulation coupler: Newton (walking robot, Python GPU), XLB
    // (LBM fluid solver, Python GPU), and DEM-Engine (discrete elements, C++/CUDA).
    // Newton is the primary active solver; XLB and DEM-Engine are placeholders that
    // are instantiated but do not advance serious physics in Step() yet.
    // PyNewtonXLBDEMCoupler is defined in src/couplers/newton_xlb_dem/.
    py::class_<PyNewtonXLBDEMCoupler>(m, "NewtonXLBDEMCoupler",
                                      "Three-way co-simulation coupler coupling Newton (articulated rigid-body "
                                      "physics), XLB (lattice-Boltzmann fluid solver), and DEME (Python "
                                      "discrete-element solver, pip install deme).\n\n"
                                      "Newton drives a walking robot and produces a spatial representation "
                                      "(body transforms) at every step.  XLB and DEME are placeholder "
                                      "solvers that are instantiated but not seriously advanced yet; future "
                                      "work will feed the robot geometry into both.\n\n"
                                      "Usage::\n\n"
                                      "    import mophi, newton, warp as wp, deme\n"
                                      "    wp.init()\n"
                                      "    builder = newton.ModelBuilder()\n"
                                      "    # ... build walking robot ...\n"
                                      "    model  = builder.finalize()\n"
                                      "    solver = newton.solvers.SolverXPBD(model)\n"
                                      "    dem = deme.DEMSolver(1)\n"
                                      "    coupler = mophi.NewtonXLBDEMCoupler()\n"
                                      "    coupler.set_verbosity(mophi.VERBOSITY_INFO)\n"
                                      "    coupler.initialize(newton_model=model, newton_solver=solver,\n"
                                      "                       deme_solver=dem)\n"
                                      "    for step in range(steps):\n"
                                      "        # optionally update coupler.newton_control before each step\n"
                                      "        coupler.step_newton()  # advance Newton one substep\n"
                                      "        coupler.step_deme()    # advance DEME one substep\n"
                                      "        # coupler.step_xlb()  # advance XLB (placeholder)\n"
                                      "        transforms = coupler.get_robot_body_transforms()\n"
                                      "    coupler.finalize()")
        .def(py::init<>())
        .def("initialize", &PyNewtonXLBDEMCoupler::Initialize, py::arg("newton_model") = py::none(),
             py::arg("newton_solver") = py::none(), py::arg("xlb_simulation") = py::none(),
             py::arg("deme_solver") = py::none(), py::arg("sim_dt") = 1.0 / 1000.0,
             "Initialize all three solvers.\n\n"
             "newton_model must be a newton.Model; newton_solver must be a Newton solver\n"
             "instance (e.g. newton.solvers.SolverXPBD(model)).  When both are provided\n"
             "the coupler evaluates initial forward kinematics and is ready to step.\n"
             "xlb_simulation may be an XLB simulation object or None (skip XLB).\n"
             "deme_solver may be a deme.DEMSolver instance or None (skip DEME).\n"
             "sim_dt sets the co-simulation time step in seconds (default 1 ms).")
        .def("step_newton", &PyNewtonXLBDEMCoupler::StepNewton,
             "Advance Newton by one substep.\n\n"
             "Sequence: clear forces → collide → step → swap states.\n"
             "Write joint targets into newton_control before calling this.")
        .def("step_deme", &PyNewtonXLBDEMCoupler::StepDEME,
             "Advance DEME by one substep.\n\n"
             "Calls DoStepDynamics() on the bound deme.DEMSolver.\n"
             "No-op when no DEME solver was provided.")
        .def("step_xlb", &PyNewtonXLBDEMCoupler::StepXLB,
             "Advance XLB by one substep (placeholder).\n\n"
             "No-op until fluid–robot coupling is implemented.\n"
             "Call once per policy frame, independently of Newton / DEME.")
        .def("finalize", &PyNewtonXLBDEMCoupler::Finalize,
             "Finalize all solvers and release all resources including Python references.")
        .def("set_verbosity", &PyNewtonXLBDEMCoupler::SetVerbosity, py::arg("verbose"),
             "Set the MoPhi logger verbosity level (e.g. mophi.VERBOSITY_INFO).")
        .def("get_robot_body_transforms", &PyNewtonXLBDEMCoupler::GetRobotBodyTransforms,
             "Return the spatial representation of the robot as a list of body transforms.\n\n"
             "Each entry is [px, py, pz, qx, qy, qz, qw] — the world-space position\n"
             "and orientation quaternion (Warp convention: x,y,z,w) of one robot body.\n"
             "The list has one entry per Newton body (model.body_count).\n"
             "Returns an empty list when Newton has not been initialized.\n\n"
             "NOTE: this method performs a synchronous GPU→CPU copy.  For the\n"
             "GPU-native path, use get_newton_body_q_array() instead.")
        .def("get_newton_body_q_array", &PyNewtonXLBDEMCoupler::GetNewtonBodyQArray,
             "Return the device-resident body_q Warp array from newton_state_0.\n\n"
             "body_q has shape (body_count, 7) and stores each body's world-space\n"
             "transform as [px, py, pz, qx, qy, qz, qw].\n"
             "Returns None when Newton has not been initialized.\n\n"
             "Usage patterns:\n"
             "  body_q.numpy()          — synchronous GPU→CPU copy (small reads OK)\n"
             "  wp.to_torch(body_q)[...] — zero-copy PyTorch view, stays on device")
        .def("set_xlb_masks", &PyNewtonXLBDEMCoupler::SetXLBMasks, py::arg("bc_mask"), py::arg("missing_mask"),
             "Bind the XLB bc_mask and missing_mask Warp arrays to this coupler.\n\n"
             "Both arrays are device-resident wp.array objects returned by the XLB\n"
             "stepper's prepare_fields() call.  After binding, get_xlb_bc_mask_array()\n"
             "and get_xlb_missing_mask_array() provide the device handles so that Warp\n"
             "GPU kernels can update the obstacle masks in-place without any CPU copy.\n\n"
             "  bc_mask      : wp.array shape (1, NX, NY, NZ), dtype uint8\n"
             "  missing_mask : wp.array shape (Q, NX, NY, NZ), dtype bool")
        .def("get_xlb_bc_mask_array", &PyNewtonXLBDEMCoupler::GetXLBBCMaskArray,
             "Return the stored XLB bc_mask Warp array, or None if not yet set.\n\n"
             "The array lives on the same device as the XLB stepper.  Warp kernels\n"
             "can write to it in-place to update the robot obstacle boundary condition.")
        .def("get_xlb_missing_mask_array", &PyNewtonXLBDEMCoupler::GetXLBMissingMaskArray,
             "Return the stored XLB missing_mask Warp array, or None if not yet set.\n\n"
             "The array lives on the same device as the XLB stepper.  Warp kernels\n"
             "can write to it in-place to update the lattice pull-direction flags\n"
             "for the robot obstacle boundary condition.")
        .def_readwrite("newton_control", &PyNewtonXLBDEMCoupler::newton_control,
                       "The Newton control object.  Write joint_target_pos values here before\n"
                       "calling step() to drive the robot's joints.")
        .def_readwrite("newton_state_0", &PyNewtonXLBDEMCoupler::newton_state_0,
                       "The current Newton state (after the most recent step()).\n"
                       "Pass this to newton.viewer.ViewerGL.log_state() to visualize the scene.");
#endif
}
