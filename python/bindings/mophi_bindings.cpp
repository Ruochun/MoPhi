#include <pybind11/pybind11.h>

#include <core/Logger.hpp>

namespace py = pybind11;

// ─────────────────────────────────────────────────────────────────────────────
// Per-coupler registration functions — each defined in its own binding file.
// The functions are always compiled (no #ifdef guards at the call site) so that
// adding a new coupler requires only creating a new *_bindings.cpp file and
// adding one register_*() call here.  Each function is a no-op when its coupler
// was not enabled at build time.
// ─────────────────────────────────────────────────────────────────────────────

void register_feris_newton(py::module_& m);
void register_newton_xlb_dem(py::module_& m);
void register_mesh_io(py::module_& m);

// ─────────────────────────────────────────────────────────────────────────────
// mophi_core — Python extension module
//
// Every multi-physics co-simulation solver built in MoPhi must be exposed here
// (or in a sibling binding file).  The Python package in python/mophi/ imports
// this module and re-exports the classes under the user-facing `mophi` namespace.
// ─────────────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(mophi_core, m) {
    m.doc() = "MoPhi — multi-physics co-simulation framework (C++ core)";

    // ── Verbosity constants ────────────────────────────────────────────────────
    // Expose mophi::verbosity_t values so Python callers can write e.g.:
    //   coupler.set_verbosity(mophi.VERBOSITY_INFO)
    // The integer values match the constants defined in <core/Logger.hpp>.
    m.attr("VERBOSITY_ERROR") = static_cast<int>(mophi::VERBOSITY_ERROR);
    m.attr("VERBOSITY_WARNING") = static_cast<int>(mophi::VERBOSITY_WARNING);
    m.attr("VERBOSITY_INFO") = static_cast<int>(mophi::VERBOSITY_INFO);

    // ── Coupler registrations ─────────────────────────────────────────────────
    register_feris_newton(m);
    register_newton_xlb_dem(m);

    // ── MoPhiEssentials mesh I/O ──────────────────────────────────────────────
    // SurfaceMesh and load_obj are always registered regardless of which
    // couplers are enabled; they depend only on the mophi_essentials submodule.
    register_mesh_io(m);
}
