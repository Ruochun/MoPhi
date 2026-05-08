#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <common/Mesh.hpp>
#include <utils/MeshIO.hpp>

namespace py = pybind11;

// ─────────────────────────────────────────────────────────────────────────────
// register_mesh_io — pybind11 registration for SurfaceMesh and load_obj.
//
// Exposes MoPhiEssentials' SurfaceMesh data structure and the LoadOBJ utility
// to Python so that demos can load geometry files using the same in-tree
// mesh-loading code that the C++ couplers use, without re-implementing a
// separate Python OBJ parser.
// ─────────────────────────────────────────────────────────────────────────────

void register_mesh_io(py::module_& m) {
    // ── SurfaceMesh ─────────────────────────────────────────────────────────
    // Wraps mophi::SurfaceMesh from <common/Mesh.hpp>.  The two most commonly
    // needed properties — vertex positions and triangle indices — are exposed as
    // NumPy arrays so they can be passed directly to solvers such as newton.Mesh.
    py::class_<mophi::SurfaceMesh>(m, "SurfaceMesh",
                                   "Triangle surface mesh loaded from an OBJ (or STL/PLY) file via "
                                   "MoPhiEssentials.\n\n"
                                   "Obtain a SurfaceMesh by calling :func:`mophi.load_obj`.  The two "
                                   "properties used most often are:\n\n"
                                   "  mesh.vertices — float32 NumPy array of shape (N, 3)\n"
                                   "  mesh.indices  — int32 NumPy array of shape (F*3,)  (flat face list)\n\n"
                                   "Both arrays can be passed directly to ``newton.Mesh()`` or to the DEME "
                                   "kinematic-mesh API without any further conversion.")
        .def(py::init<>())
        .def_property_readonly("num_vertices", &mophi::SurfaceMesh::NumVertices, "Number of vertices in the mesh.")
        .def_property_readonly("num_faces", &mophi::SurfaceMesh::NumFaces, "Number of triangular faces in the mesh.")
        .def_property_readonly(
            "vertices",
            [](const mophi::SurfaceMesh& self) {
                auto n = static_cast<py::ssize_t>(self.vertices.size());
                py::array_t<float> arr({n, static_cast<py::ssize_t>(3)});
                auto buf = arr.mutable_unchecked<2>();
                for (py::ssize_t i = 0; i < n; ++i) {
                    buf(i, 0) = static_cast<float>(self.vertices[i].x());
                    buf(i, 1) = static_cast<float>(self.vertices[i].y());
                    buf(i, 2) = static_cast<float>(self.vertices[i].z());
                }
                return arr;
            },
            "Vertex positions as a float32 NumPy array of shape (N, 3).")
        .def_property_readonly(
            "indices",
            [](const mophi::SurfaceMesh& self) {
                auto f = static_cast<py::ssize_t>(self.faces.size());
                py::array_t<int32_t> arr({f * static_cast<py::ssize_t>(3)});
                auto buf = arr.mutable_unchecked<1>();
                for (py::ssize_t i = 0; i < f; ++i) {
                    buf(i * 3 + 0) = self.faces[i][0];
                    buf(i * 3 + 1) = self.faces[i][1];
                    buf(i * 3 + 2) = self.faces[i][2];
                }
                return arr;
            },
            "Triangle face indices as a flat int32 NumPy array of shape (F*3,).\n\n"
            "Each consecutive triple [i*3, i*3+1, i*3+2] gives the three vertex "
            "indices of triangle i (0-based).")
        .def(
            "scale", [](mophi::SurfaceMesh& self, double factor) { self.Scale(factor); }, py::arg("factor"),
            "Scale all vertex coordinates uniformly by *factor* in-place.\n\n"
            "Useful for converting between unit systems (e.g. pass 0.01 to "
            "convert a centimetre-authored OBJ to metres).");

    // ── load_obj ─────────────────────────────────────────────────────────────
    // Thin wrapper around mophi::LoadOBJ from <utils/MeshIO.hpp>.
    // The optional *scale* factor is applied after loading so that callers can
    // perform unit conversion (e.g. cm → m) in a single call.
    m.def(
        "load_obj",
        [](const std::string& path, double scale) {
            mophi::SurfaceMesh mesh;
            if (!mophi::LoadOBJ(path, mesh)) {
                throw std::runtime_error("mophi.load_obj: failed to load OBJ file: " + path);
            }
            if (scale != 1.0) {
                mesh.Scale(scale);
            }
            return mesh;
        },
        py::arg("path"), py::arg("scale") = 1.0,
        "Load a Wavefront OBJ file into a SurfaceMesh using MoPhiEssentials' LoadOBJ.\n\n"
        "Args:\n"
        "    path:  Path to the .obj file.\n"
        "    scale: Uniform scale factor applied to all vertices after loading.\n"
        "           Pass 0.01 to convert centimetre OBJ coordinates to metres.\n\n"
        "Returns:\n"
        "    A :class:`SurfaceMesh` with .vertices (float32, (N,3)) and\n"
        "    .indices (int32, (F*3,)) ready for use with newton.Mesh or DEME.\n\n"
        "Raises:\n"
        "    RuntimeError: If the file cannot be parsed by LoadOBJ.");
}
