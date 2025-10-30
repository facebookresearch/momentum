/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/geometry/mesh_pybind.h"

#include "pymomentum/geometry/momentum_geometry.h"

#include <momentum/math/intersection.h>
#include <momentum/math/mesh.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <optional>
#include <vector>

#include <fmt/format.h>

namespace py = pybind11;
namespace mm = momentum;

namespace pymomentum {

namespace {

mm::Mesh createMesh(
    const py::array_t<float>& vertices,
    const py::array_t<int>& faces,
    const std::optional<py::array_t<float>>& normals,
    const std::vector<std::vector<int32_t>>& lines,
    std::optional<py::array_t<uint8_t>> colors,
    const std::vector<float>& confidence,
    std::optional<py::array_t<float>> texcoords,
    std::optional<py::array_t<int>> texcoord_faces,
    const std::vector<std::vector<int32_t>>& texcoord_lines) {
  mm::Mesh mesh;
  MT_THROW_IF(vertices.ndim() != 2, "vertices must be a 2D array");
  MT_THROW_IF(vertices.shape(1) != 3, "vertices must have size n x 3");

  MT_THROW_IF(faces.ndim() != 2, "faces must be a 2D array");
  MT_THROW_IF(faces.shape(1) != 3, "faces must have size n x 3");
  const auto nVerts = vertices.shape(0);

  mesh.vertices = asVectorList<float, 3>(vertices);
  mesh.faces = asVectorList<int, 3>(faces);
  for (const auto& f : mesh.faces) {
    MT_THROW_IF(
        f.x() >= nVerts || f.y() >= nVerts || f.z() >= nVerts, "face index exceeded vertex count");
  }

  if (normals.has_value()) {
    MT_THROW_IF(normals->ndim() != 2 || normals->shape(1) != 3, "normals must have size n x 3");
    MT_THROW_IF(
        normals->shape(0) != nVerts, "vertices and normals must have the same number of rows");

    mesh.normals = asVectorList<float, 3>(normals.value());
  } else {
    mesh.updateNormals();
  }

  for (const auto& l : lines) {
    MT_THROW_IF(
        !l.empty() && *std::max_element(l.begin(), l.end()) >= nVerts,
        "line index exceeded vertex count");
  }
  mesh.lines = lines;

  if (colors && colors->size() != 0) {
    MT_THROW_IF((colors->ndim() != 2 || colors->shape(1) != 3), "colors should have size n x 3");
    MT_THROW_IF(
        (colors->shape(0) != nVerts), "colors should be empty or equal to the number of vertices");
    mesh.colors = asVectorList<uint8_t, 3>(*colors);
  }

  MT_THROW_IF(
      confidence.size() != 0 && confidence.size() != nVerts,
      "confidence should be empty or equal to the number of vertices");
  mesh.confidence = confidence;

  int nTextureCoords = 0;
  if (texcoords && texcoords->size() != 0) {
    MT_THROW_IF(
        texcoords->ndim() != 2 && texcoords->shape(1) != 2,
        "texcoords should be empty or must have size n x 2");
    nTextureCoords = texcoords->shape(0);
    mesh.texcoords = asVectorList<float, 2>(*texcoords);
  }

  if (texcoord_faces && texcoord_faces->size() != 0) {
    MT_THROW_IF(
        texcoord_faces->ndim() != 2 || texcoord_faces->shape(1) != 3,
        "texcoord_faces should be empty or must have size n x 3");
    MT_THROW_IF(
        texcoord_faces->shape(0) != faces.shape(0),
        "texcoords_faces should be empty or equal to the size of faces");
    mesh.texcoord_faces = asVectorList<int32_t, 3>(*texcoord_faces);

    for (const auto& f : mesh.texcoord_faces) {
      MT_THROW_IF(
          f.x() >= nTextureCoords || f.y() >= nTextureCoords || f.z() >= nTextureCoords,
          "texcoord face index exceeded texcoord count");
    }
  }

  mesh.texcoord_lines = texcoord_lines;

  return mesh;
}

} // namespace

void registerMeshBindings(py::class_<mm::Mesh>& meshClass) {
  // =====================================================
  // momentum::Mesh
  // - vertices
  // - normals
  // - faces
  // - colors
  // =====================================================
  meshClass
      .def(
          py::init(&createMesh),
          R"(
:param vertices: n x 3 array of vertex locations.
:param faces: n x 3 array of triangles.
:param normals: Optional n x 3 array of vertex normals.  If not passed in, vertex normals will be computed automatically.
:param lines: Optional list of lines, where each line is a list of vertex indices.
:param colors: Optional n x 3 array of vertex colors.
:param confidence: Optional n x 1 array of vertex confidence values.
:param texcoords: Optional n x 2 array of texture coordinates.
:param texcoord_faces: Optional n x 3 array of triangles in the texture map.  Each triangle corresponds to a triangle on the mesh, but indices should refer to the texcoord array.
:param texcoord_lines: Optional list of lines, where each line is a list of texture coordinate indices.
          )",
          py::arg("vertices"),
          py::arg("faces"),
          py::kw_only(),
          py::arg("normals") = std::optional<py::array_t<float>>{},
          py::arg("lines") = std::vector<std::vector<int32_t>>{},
          py::arg("colors") = std::optional<py::array_t<uint8_t>>{},
          py::arg("confidence") = std::vector<float>{},
          py::arg("texcoords") = std::optional<py::array_t<float>>{},
          py::arg("texcoord_faces") = std::optional<py::array_t<int>>{},
          py::arg("texcoord_lines") = std::vector<std::vector<int32_t>>{})
      .def_property_readonly(
          "n_vertices",
          [](const mm::Mesh& mesh) { return mesh.vertices.size(); },
          ":return: The number of vertices in the mesh.")
      .def_property_readonly(
          "n_faces",
          [](const mm::Mesh& mesh) { return mesh.faces.size(); },
          ":return: The number of faces in the mesh.")
      .def_property_readonly(
          "vertices",
          [](const mm::Mesh& mesh) { return pymomentum::asArray(mesh.vertices); },
          ":return: The vertices of the mesh in a [n x 3] numpy array.")
      .def_property_readonly(
          "normals",
          [](const mm::Mesh& mesh) { return pymomentum::asArray(mesh.normals); },
          ":return: The per-vertex normals of the mesh in a [n x 3] numpy array.")
      .def_property_readonly(
          "faces",
          [](const mm::Mesh& mesh) { return pymomentum::asArray(mesh.faces); },
          ":return: The triangles of the mesh in an [n x 3] numpy array.")
      .def_readonly("lines", &mm::Mesh::lines, "list of list of vertex indices per line")
      .def_property_readonly(
          "colors",
          [](const mm::Mesh& mesh) { return pymomentum::asArray(mesh.colors); },
          ":return: Per-vertex colors if available; returned as a (possibly empty) [n x 3] numpy array.")
      .def_readonly("confidence", &mm::Mesh::confidence, "list of per-vertex confidences")
      .def_property_readonly(
          "texcoords",
          [](const mm::Mesh& mesh) { return pymomentum::asArray(mesh.texcoords); },
          "texture coordinates as m x 3 array.  Note that the number of texture coordinates may "
          "be different from the number of vertices as there can be cuts in the texture map.  "
          "Use texcoord_faces to index the texture coordinates.")
      .def_property_readonly(
          "texcoord_faces",
          [](const mm::Mesh& mesh) { return pymomentum::asArray(mesh.texcoord_faces); },
          "n x 3 faces in the texture map.  Each face maps 1-to-1 to a face in the original "
          "mesh but indexes into the texcoords array.")
      .def_readonly(
          "texcoord_lines",
          &mm::Mesh::texcoord_lines,
          "Texture coordinate indices for each line.  ")
      .def(
          "self_intersections",
          [](const mm::Mesh& mesh) {
            const auto intersections = mm::intersectMesh(mesh);
            return py::array(py::cast(intersections));
          },
          "Test if the mesh self intersects anywhere and return all intersecting face pairs")
      .def(
          "with_updated_normals",
          [](const mm::Mesh& mesh) {
            mm::Mesh result = mesh;
            result.updateNormals();
            return result;
          })
      .def("__repr__", [](const mm::Mesh& m) {
        return fmt::format(
            "Mesh(vertices={}, faces={}, has_normals={}, has_colors={}, has_texcoords={})",
            m.vertices.size(),
            m.faces.size(),
            !m.normals.empty() ? "True" : "False",
            !m.colors.empty() ? "True" : "False",
            !m.texcoords.empty() ? "True" : "False");
      });
}

} // namespace pymomentum
