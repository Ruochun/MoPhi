#include "PyNewtonXLBDEMCoupler.h"

#include <core/Logger.hpp>

// ── pybind11 numpy support ────────────────────────────────────────────────────
// Required for the numpy array conversions in GetRobotBodyTransforms().
#include <pybind11/numpy.h>

// ── Constructor / Destructor ───────────────────────────────────────────────────

PyNewtonXLBDEMCoupler::PyNewtonXLBDEMCoupler()
    : deme_solver(pybind11::none()),
      newton_model(pybind11::none()),
      newton_solver(pybind11::none()),
      newton_state_0(pybind11::none()),
      newton_state_1(pybind11::none()),
      newton_control(pybind11::none()),
      newton_contacts(pybind11::none()),
      xlb_simulation(pybind11::none()) {
    MOPHI_INFO("PyNewtonXLBDEMCoupler: created");
}

PyNewtonXLBDEMCoupler::~PyNewtonXLBDEMCoupler() {
    if (initialized) {
        Finalize();
    }
}

// ── Public interface ───────────────────────────────────────────────────────────

void PyNewtonXLBDEMCoupler::Initialize(pybind11::object newton_model_in,
                                       pybind11::object newton_solver_in,
                                       pybind11::object xlb_simulation_in,
                                       pybind11::object deme_solver_in,
                                       double dt) {
    MOPHI_INFO("PyNewtonXLBDEMCoupler: initializing ...");

    // ── DEME (Python, pip install deme) ───────────────────────────────────────
    // Accept a pre-built deme.DEMSolver instance from the Python caller.
    // Physics configuration and deme_solver.initialize() will be added later.
    if (!deme_solver_in.is_none()) {
        deme_solver = deme_solver_in;
        deme_available = true;
        MOPHI_INFO("PyNewtonXLBDEMCoupler: deme.DEMSolver bound [placeholder]");
    } else {
        MOPHI_INFO("PyNewtonXLBDEMCoupler: no DEME solver provided — DEME step will be skipped");
    }

    // ── Newton ────────────────────────────────────────────────────────────────
    // Newton runs in Python; no C++ initialization is needed beyond storing the
    // Python objects and evaluating initial forward kinematics.
    if (!newton_model_in.is_none() && !newton_solver_in.is_none()) {
        newton_model = newton_model_in;
        newton_solver = newton_solver_in;
        newton_state_0 = newton_model.attr("state")();
        newton_state_1 = newton_model.attr("state")();
        newton_control = newton_model.attr("control")();
        newton_contacts = newton_model.attr("contacts")();
        sim_dt = dt;

        // Evaluate initial forward kinematics so the first Newton step
        // starts from a well-defined body configuration.
        pybind11::module_::import("newton").attr("eval_fk")(newton_model, newton_model.attr("joint_q"),
                                                            newton_model.attr("joint_qd"), newton_state_0);

        newton_available = true;
        MOPHI_INFO("PyNewtonXLBDEMCoupler: Newton model and solver bound (sim_dt=%.6f s)", sim_dt);
    } else {
        MOPHI_INFO("PyNewtonXLBDEMCoupler: no Newton solver provided — Newton step will be skipped");
    }

    // ── XLB (placeholder) ────────────────────────────────────────────────────
    // Accept a pre-built XLB simulation from the Python caller.  Future coupling
    // logic will feed robot geometry into XLB as a moving boundary condition.
    if (!xlb_simulation_in.is_none()) {
        xlb_simulation = xlb_simulation_in;
        xlb_available = true;
        MOPHI_INFO("PyNewtonXLBDEMCoupler: XLB simulation bound [placeholder]");
    } else {
        MOPHI_INFO("PyNewtonXLBDEMCoupler: no XLB simulation provided — XLB step will be skipped");
    }

    MOPHI_INFO("PyNewtonXLBDEMCoupler: initialized");

    initialized = true;
}

//// TODO: No whole-sale stepper. Each physics step their own way.
void PyNewtonXLBDEMCoupler::Step() {
    if (!initialized) {
        MOPHI_ERROR("PyNewtonXLBDEMCoupler::Step() called before Initialize().");
    }

    // ── 1. Advance Newton ─────────────────────────────────────────────────────
    if (newton_available) {
        newton_state_0.attr("clear_forces")();
        newton_model.attr("collide")(newton_state_0, newton_contacts);
        newton_solver.attr("step")(newton_state_0, newton_state_1, newton_control, newton_contacts,
                                   pybind11::float_(sim_dt));
        // state_1 now holds the new Newton state; swap buffers for next iteration.
        std::swap(newton_state_0, newton_state_1);
    }

    // ── 2. Extract robot spatial representation (coupling output) ─────────────
    // GetRobotBodyTransforms() returns the body transforms from the just-updated
    // Newton state.  Future coupling logic should consume these transforms to:
    //   • Update DEME's particle-field geometry (robot surface mesh).
    //   • Update XLB's fluid boundary (robot surface as moving obstacle).
    // auto transforms = GetRobotBodyTransforms();
    // TODO: feed transforms into DEME and XLB.

    // ── 3. DEME placeholder step ──────────────────────────────────────────────
    // No physics is advanced yet.  Once particle–robot coupling is implemented,
    // this will call the appropriate deme_solver step method.
    if (deme_available) {
        // MOPHI_INFO("PyNewtonXLBDEMCoupler: DEME step (placeholder, no-op)");
    }

    // ── 4. XLB placeholder step ───────────────────────────────────────────────
    // No fluid physics is advanced yet.  Once fluid–robot coupling is implemented,
    // this will call xlb_simulation.step() with the robot geometry boundary.
    if (xlb_available) {
        // MOPHI_INFO("PyNewtonXLBDEMCoupler: XLB step (placeholder, no-op)");
    }

    ++step_count;
}

void PyNewtonXLBDEMCoupler::Finalize() {
    MOPHI_INFO("PyNewtonXLBDEMCoupler: finalizing ...");

    // Release DEME Python reference so Python's reference counter can collect it.
    deme_solver = pybind11::none();
    deme_available = false;

    initialized = false;

    // Release Newton references so Python's reference counter can collect them.
    newton_model = pybind11::none();
    newton_solver = pybind11::none();
    newton_state_0 = pybind11::none();
    newton_state_1 = pybind11::none();
    newton_control = pybind11::none();
    newton_contacts = pybind11::none();
    newton_available = false;

    // Release XLB reference.
    xlb_simulation = pybind11::none();
    xlb_available = false;

    step_count = 0;

    MOPHI_INFO("PyNewtonXLBDEMCoupler: finalized");
}

void PyNewtonXLBDEMCoupler::SetVerbosity(mophi::verbosity_t verbose) {
    mophi::Logger::GetInstance().SetVerbosity(verbose);
}

std::vector<std::array<double, 7>> PyNewtonXLBDEMCoupler::GetRobotBodyTransforms() const {
    if (!newton_available) {
        return {};
    }

    namespace py = pybind11;

    // Retrieve body_q from the current Newton state.
    // body_q is a Warp array of shape [body_count, 7].
    // Each row encodes one body's world-space transform as
    //   [px, py, pz, qx, qy, qz, qw]
    // (position in metres + unit quaternion, Warp convention: x,y,z,w).
    py::object body_q = newton_state_0.attr("body_q");

    // .numpy() copies the Warp GPU array to a CPU numpy ndarray.
    py::object np_obj = body_q.attr("numpy")();

    // Ensure float64 for the C++ output vector.
    py::module_ np = py::module_::import("numpy");
    np_obj = np.attr("asarray")(np_obj, py::arg("dtype") = "float64");

    auto buf = np_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    py::buffer_info info = buf.request();

    if (info.ndim != 2 || info.shape[1] != 7) {
        MOPHI_WARNING(
            "PyNewtonXLBDEMCoupler::GetRobotBodyTransforms: "
            "unexpected body_q shape — expected (N, 7), got ndim=%zd shape[1]=%zd",
            static_cast<size_t>(info.ndim), info.ndim >= 2 ? static_cast<size_t>(info.shape[1]) : 0UL);
        return {};
    }

    const auto num_bodies = static_cast<int>(info.shape[0]);
    std::vector<std::array<double, 7>> transforms(num_bodies);
    const auto* ptr = static_cast<const double*>(info.ptr);

    for (int i = 0; i < num_bodies; ++i) {
        for (int j = 0; j < 7; ++j) {
            transforms[i][j] = ptr[i * 7 + j];
        }
    }

    return transforms;
}
