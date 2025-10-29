/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/geometry/character_pybind.h"
#include "pymomentum/geometry/gltf_builder_pybind.h"
#include "pymomentum/geometry/limit_pybind.h"
#include "pymomentum/geometry/momentum_geometry.h"
#include "pymomentum/geometry/momentum_io.h"
#include "pymomentum/geometry/parameter_transform_pybind.h"
#include "pymomentum/geometry/skeleton_pybind.h"
#include "pymomentum/geometry/skin_weights_pybind.h"
#include "pymomentum/tensor_momentum/tensor_blend_shape.h"
#include "pymomentum/tensor_momentum/tensor_joint_parameters_to_positions.h"
#include "pymomentum/tensor_momentum/tensor_kd_tree.h"
#include "pymomentum/tensor_momentum/tensor_mppca.h"
#include "pymomentum/tensor_momentum/tensor_parameter_transform.h"
#include "pymomentum/tensor_momentum/tensor_skeleton_state.h"
#include "pymomentum/tensor_momentum/tensor_skinning.h"

#include <momentum/character/blend_shape.h>
#include <momentum/character/character.h>
#include <momentum/character/character_utility.h>
#include <momentum/character/fwd.h>
#include <momentum/character/inverse_parameter_transform.h>
#include <momentum/character/joint.h>
#include <momentum/character/locator.h>
#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character/skin_weights.h>
#include <momentum/io/fbx/fbx_io.h>
#include <momentum/io/gltf/gltf_io.h>
#include <momentum/io/legacy_json/legacy_json_io.h>
#include <momentum/io/marker/coordinate_system.h>
#include <momentum/io/shape/blend_shape_io.h>
#include <momentum/math/intersection.h>
#include <momentum/math/mesh.h>
#include <momentum/math/mppca.h>
#include <momentum/test/character/character_helpers.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/python.h>
#include <Eigen/Core>

#include <algorithm>
#include <limits>

#include <fmt/format.h>

namespace py = pybind11;
namespace mm = momentum;

using namespace pymomentum;

namespace {
mm::Character createTestCharacter(int num_joints, bool with_blendshapes) {
  auto character = momentum::createTestCharacter<float>(num_joints);
  if (with_blendshapes) {
    character = momentum::withTestBlendShapes(character);
  }
  return character;
}
} // namespace

PYBIND11_MODULE(geometry, m) {
  // TODO more explanation
  m.doc() = "Geometry and forward kinematics for momentum models.  ";
  m.attr("__name__") = "pymomentum.geometry";

  pybind11::module_::import("torch"); // @dep=//caffe2:torch

  m.attr("PARAMETERS_PER_JOINT") = mm::kParametersPerJoint;

  py::enum_<mm::FBXUpVector>(m, "FBXUpVector")
      .value("XAxis", mm::FBXUpVector::XAxis)
      .value("YAxis", mm::FBXUpVector::YAxis)
      .value("ZAxis", mm::FBXUpVector::ZAxis);

  py::enum_<mm::UpVector>(m, "UpVector")
      .value("X", mm::UpVector::X)
      .value("Y", mm::UpVector::Y)
      .value("Z", mm::UpVector::Z);

  py::enum_<mm::FBXFrontVector>(m, "FBXFrontVector")
      .value("ParityEven", mm::FBXFrontVector::ParityEven)
      .value("ParityOdd", mm::FBXFrontVector::ParityOdd);

  py::enum_<mm::FBXCoordSystem>(m, "FBXCoordSystem")
      .value("RightHanded", mm::FBXCoordSystem::RightHanded)
      .value("LeftHanded", mm::FBXCoordSystem::LeftHanded);

  // We need to forward-declare classes so that if we refer to them they get
  // typed correctly; otherwise we end up with "momentum::Locator" in the
  // docstrings/type descriptors.
  auto characterClass = py::class_<mm::Character>(
      m, "Character", "A complete momentum character including its skeleton and mesh.");
  auto parameterTransformClass = py::class_<mm::ParameterTransform>(
      m,
      "ParameterTransform",
      "Maps reduced model parameters to full joint parameters for character animation. "
      "This class handles the transformation from a compact set of model parameters "
      "(typically ~50) used in optimization to the full 7*nJoints parameters needed "
      "for skeleton posing.");
  auto inverseParameterTransformClass = py::class_<mm::InverseParameterTransform>(
      m,
      "InverseParameterTransform",
      "Inverse parameter transform that maps from full joint parameters back to "
      "reduced model parameters. Used for fitting model parameters to existing "
      "joint parameter data.");
  auto meshClass = py::class_<mm::Mesh>(
      m,
      "Mesh",
      "A 3D mesh containing vertices, faces, normals, and optional texture coordinates. "
      "Supports triangular meshes with additional data like vertex colors, confidence values, "
      "and line segments for wireframe rendering.");
  auto jointClass = py::class_<mm::Joint>(
      m,
      "Joint",
      "A single joint in a character skeleton. Contains the joint's name, parent relationship, "
      "pre-rotation, and translation offset from its parent joint.");
  auto skeletonClass = py::class_<mm::Skeleton>(
      m,
      "Skeleton",
      "A hierarchical skeleton structure containing joints with parent-child relationships. "
      "Each joint has a name, parent index, pre-rotation, and translation offset. "
      "Used for character animation and forward kinematics.");
  auto skinWeightsClass = py::class_<mm::SkinWeights>(
      m,
      "SkinWeights",
      "Linear blend skinning weights that define how mesh vertices are influenced by skeleton joints. "
      "Contains weight values and joint indices for each vertex, enabling smooth deformation "
      "of the mesh during character animation.");
  auto locatorClass = py::class_<mm::Locator>(
      m,
      "Locator",
      "A 3D point attached to a skeleton joint used for inverse kinematics constraints. "
      "Locators define target positions that the character should reach during pose optimization, "
      "with configurable weights and axis locks for fine-grained control.");
  auto skinnedLocatorClass = py::class_<mm::SkinnedLocator>(m, "SkinnedLocator");
  auto blendShapeClass = py::class_<mm::BlendShape, std::shared_ptr<mm::BlendShape>>(
      m,
      "BlendShape",
      "A blend shape basis for facial expressions and corrective shapes. "
      "Contains a base mesh and a set of shape vectors that can be linearly "
      "combined to create different facial expressions or body shape variations.");
  auto capsuleClass = py::class_<mm::TaperedCapsule>(
      m,
      "TaperedCapsule",
      "A tapered capsule primitive used for collision detection and physics simulation. "
      "Represents a capsule with potentially different radii at each end, attached to a skeleton joint.");
  auto markerClass = py::class_<mm::Marker>(
      m,
      "Marker",
      "A 3D marker used in motion capture systems. Contains position data and occlusion status "
      "for tracking points on subjects during motion capture recording.");
  auto markerSequenceClass = py::class_<mm::MarkerSequence>(
      m,
      "MarkerSequence",
      "A sequence of motion capture marker data over time. Contains marker positions "
      "and occlusion status for each frame, along with frame rate information.");
  auto fbxCoordSystemInfoClass = py::class_<mm::FBXCoordSystemInfo>(
      m,
      "FBXCoordSystemInfo",
      "FBX coordinate system information containing up vector, front vector, and handedness. "
      "Used when importing/exporting FBX files to ensure proper coordinate system conversion.");
  auto parameterLimitClass = py::class_<mm::ParameterLimit>(
      m,
      "ParameterLimit",
      "A constraint on model or joint parameters used to enforce realistic poses. "
      "Supports various limit types including min/max bounds, linear relationships, "
      "ellipsoid constraints, and half-plane constraints.");
  auto gltfOptionsClass =
      py::class_<mm::GltfOptions>(m, "GltfOptions", "Storage options for Gltf export.");

  gltfOptionsClass.def(py::init<>())
      .def(
          py::init<bool, bool, bool, bool, bool>(),
          py::arg("extensions") = true,
          py::arg("collisions") = true,
          py::arg("locators") = true,
          py::arg("mesh") = true,
          py::arg("blendShapes") = true)
      .def(
          "__repr__",
          [](const mm::GltfOptions& self) {
            return fmt::format(
                "GltfData(extensions={}, collisions={}, locators={}, mesh={}, blendShapes={})",
                self.extensions,
                self.collisions,
                self.locators,
                self.mesh,
                self.blendShapes);
          })
      .def_readwrite("extensions", &mm::GltfOptions::extensions, "Save momentum extensions")
      .def_readwrite("collisions", &mm::GltfOptions::collisions, "Save collision geometry")
      .def_readwrite("locators", &mm::GltfOptions::locators, "Save locator data")
      .def_readwrite("mesh", &mm::GltfOptions::mesh, "Save mesh data")
      .def_readwrite("blend_shapes", &mm::GltfOptions::blendShapes, "Save blend shape data");

  // =====================================================
  // momentum::Mesh
  // - vertices
  // - normals
  // - faces
  // - colors
  // =====================================================
  meshClass
      .def(
          py::init([](const py::array_t<float>& vertices,
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
                  f.x() >= nVerts || f.y() >= nVerts || f.z() >= nVerts,
                  "face index exceeded vertex count");
            }

            if (normals.has_value()) {
              MT_THROW_IF(
                  normals->ndim() != 2 || normals->shape(1) != 3, "normals must have size n x 3");
              MT_THROW_IF(
                  normals->shape(0) != nVerts,
                  "vertices and normals must have the same number of rows");

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
              MT_THROW_IF(
                  (colors->ndim() != 2 || colors->shape(1) != 3), "colors should have size n x 3");
              MT_THROW_IF(
                  (colors->shape(0) != nVerts),
                  "colors should be empty or equal to the number of vertices");
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
          }),
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

  blendShapeClass
      .def_property_readonly(
          "base_shape",
          [](const mm::BlendShape& blendShape) {
            return pymomentum::asArray(blendShape.getBaseShape());
          },
          ":return: The base shape of the blend shape solver.")
      .def_property_readonly(
          "shape_vectors",
          [](const mm::BlendShape& blendShape) -> py::array_t<float> {
            const Eigen::MatrixXf& shapeVectors = blendShape.getShapeVectors();
            const Eigen::Index nVerts = shapeVectors.rows() / 3;
            MT_THROW_IF(shapeVectors.rows() % 3 != 0, "Invalid blend shape basis.");
            py::array_t<float> result(std::vector<ptrdiff_t>{shapeVectors.cols(), nVerts, 3});
            py::buffer_info buf = result.request();
            memcpy(buf.ptr, shapeVectors.data(), result.nbytes());
            return result;
          },
          ":return: The base shape of the blend shape solver.")
      .def_static(
          "load",
          &loadBlendShapeFromFile,
          R"(Load a blend shape basis from a file.

:param path: The path to a blend shape file.
:param num_expected_shapes: Trim the shape basis if it contains more shapes than this.  Pass -1 (the default) to leave the shapes untouched.
:param num_expected_vertices: Trim the shape basis if it contains more vertices than this.  Pass -1 (the default) to leave the shapes untouched.
:return: A :class:`BlendShape`.)",
          py::arg("path"),
          py::arg("num_expected_shapes") = -1,
          py::arg("num_expected_vertices") = -1)
      .def_static(
          "from_bytes",
          &loadBlendShapeFromBytes,
          R"(Load a blend shape basis from bytes in memory.

:param blend_shape_bytes: A chunk of bytes containing the blend shape basis.
:param num_expected_shapes: Trim the shape basis if it contains more shapes than this.  Pass -1 (the default) to leave the shapes untouched.
:param num_expected_vertices: Trim the shape basis if it contains more vertices than this.  Pass -1 (the default) to leave the shapes untouched.
:return: a :class:`BlendShape`.)",
          py::arg("blend_shape_bytes"),
          py::arg("num_expected_shapes") = -1,
          py::arg("num_expected_vertices") = -1)
      .def("to_bytes", &saveBlendShapeToBytes, R"(Save a blend shape basis to bytes in memory.)")
      .def("save", &saveBlendShapeToFile, R"(Save a blend shape basis to a file.)", py::arg("path"))
      .def_static(
          "from_tensors",
          &loadBlendShapeFromTensors,
          R"(Create a blend shape basis from numpy.ndarrays.

:param base_shape: A [nPts x 3] ndarray containing the base shape.
:param shape_vectors: A [nShapes x nPts x 3] ndarray containing the blend shape basis.
:return: a :class:`BlendShape`.)",
          py::arg("base_shape"),
          py::arg("shape_vectors"))
      .def_property_readonly(
          "n_shapes",
          [](const mm::BlendShape& blendShape) { return blendShape.shapeSize(); },
          "Number of shapes in the blend shape basis.")
      .def_property_readonly(
          "n_vertices",
          [](const mm::BlendShape& blendShape) { return blendShape.modelSize(); },
          "Number of vertices in the mesh.")
      .def(
          "compute_shape",
          [](py::object blendShape, at::Tensor coeffs) {
            return applyBlendShapeCoefficients(blendShape, coeffs);
          },
          R"(Apply the blend shape coefficients to compute the rest shape.

The resulting shape is equal to the base shape plus a linear combination of the shape vectors.

:param coeffs: A torch.Tensor of size [n_batch x n_shapes] containing blend shape coefficients.
:result: A [n_batch x n_vertices x 3] tensor containing the vertex positions.)",
          py::arg("coeffs"))
      .def("__repr__", [](const mm::BlendShape& bs) {
        return fmt::format("BlendShape(shapes={}, vertices={})", bs.shapeSize(), bs.modelSize());
      });

  // =====================================================
  // momentum::Locator
  // - name
  // - parent
  // - offset
  // =====================================================
  locatorClass
      .def(
          py::init<
              const std::string&,
              const size_t,
              const Eigen::Vector3f&,
              const Eigen::Vector3i&,
              float,
              const Eigen::Vector3f&,
              const Eigen::Vector3f&>(),
          py::arg("name") = "uninitialized",
          py::arg("parent") = mm::kInvalidIndex,
          py::arg("offset") = Eigen::Vector3f::Zero(),
          py::arg("locked") = Eigen::Vector3i::Zero(),
          py::arg("weight") = 1.0f,
          py::arg("limit_origin") = Eigen::Vector3f::Zero(),
          py::arg("limit_weight") = Eigen::Vector3f::Zero())
      .def_readonly("name", &mm::Locator::name, "The locator's name.")
      .def_readonly("parent", &mm::Locator::parent, "The locator's parent joint index.")
      .def_readonly(
          "offset", &mm::Locator::offset, "The locator's offset to parent joint location.")
      .def_readonly(
          "locked",
          &mm::Locator::locked,
          "Flag per axes to indicate whether that axis can be moved during optimization or not.")
      .def_readonly(
          "weight", &mm::Locator::weight, "Weight for this locator during IK optimization.")
      .def_readonly(
          "limit_origin",
          &mm::Locator::limitOrigin,
          "Defines the limit reference position. equal to offset on loading.")
      .def_readonly(
          "limit_weight",
          &mm::Locator::limitOrigin,
          "Defines how close an unlocked locator should stay to it's original position")
      .def("__repr__", [](const mm::Locator& l) {
        return fmt::format(
            "Locator(name={}, parent={}, offset=[{}, {}, {}])",
            l.name,
            l.parent,
            l.offset.x(),
            l.offset.y(),
            l.offset.z());
      });

  // ==============================================>>>>>>> REPLACE
  // momentum::SkinnedLocator
  // - name
  // - parents
  // - skinWeights
  // - position
  // - weight
  // =====================================================
  skinnedLocatorClass
      .def(
          py::init([](const std::string& name,
                      const Eigen::VectorXi& parents,
                      const Eigen::VectorXf& skinWeights,
                      const std::optional<Eigen::Vector3f>& position,
                      float weight) {
            if (parents.size() != skinWeights.size()) {
              throw std::runtime_error("parents and skin_weights must have the same size");
            }

            if (parents.size() > mm::kMaxSkinJoints) {
              throw std::runtime_error(
                  fmt::format(
                      "parents and skin_weights must have at most {} elements",
                      mm::kMaxSkinJoints));
            }

            Eigen::Matrix<uint32_t, mm::kMaxSkinJoints, 1> parentsTmp =
                Eigen::Matrix<uint32_t, mm::kMaxSkinJoints, 1>::Zero();
            Eigen::Matrix<float, mm::kMaxSkinJoints, 1> skinWeightsTmp =
                Eigen::Matrix<float, mm::kMaxSkinJoints, 1>::Zero();

            for (size_t i = 0; i < parents.size(); ++i) {
              if (parents(i) < 0) {
                throw std::runtime_error(
                    "parents must be non-negative, but got " + std::to_string(parents(i)));
              }

              if (skinWeights(i) < 0) {
                throw std::runtime_error(
                    "skin_weights must be non-negative, but got " + std::to_string(skinWeights(i)));
              }
              parentsTmp(i) = parents(i);
              skinWeightsTmp(i) = skinWeights(i);
            }

            return mm::SkinnedLocator(
                name,
                parentsTmp,
                skinWeightsTmp,
                position.value_or(Eigen::Vector3f::Zero()),
                weight);
          }),
          py::arg("name"),
          py::arg("parents"),
          py::arg("skin_weights"),
          py::arg("position") = std::nullopt,
          py::arg("weight") = 1.0f)
      .def_readonly("name", &mm::SkinnedLocator::name, "The skinned locator's name.")
      .def_property_readonly(
          "parents",
          [](const mm::SkinnedLocator& locator) { return locator.parents; },
          "Indices of the parent joints in the skeleton.")
      .def_property_readonly(
          "skin_weights",
          [](const mm::SkinnedLocator& locator) { return locator.skinWeights; },
          "Skinning weights for the parent joints.")
      .def_readonly(
          "position",
          &mm::SkinnedLocator::position,
          "Position relative to rest pose of the character.")
      .def_readonly(
          "weight",
          &mm::SkinnedLocator::weight,
          "Influence weight of this locator when used in constraints.")
      .def("__repr__", [](const mm::SkinnedLocator& l) {
        return fmt::format(
            "SkinnedLocator(name={}, position=[{}, {}, {}], weight={})",
            l.name,
            l.position.x(),
            l.position.y(),
            l.position.z(),
            l.weight);
      });

  // =====================================================
  // momentum::Mppca
  // - Mppca()
  // - Mppca(pi, mu, W, sigma2, names)
  // - numModels
  // - dimension
  // - names
  // - getModel(iModel)
  // =====================================================
  py::class_<mm::Mppca, std::shared_ptr<mm::Mppca>>(
      m,
      "Mppca",
      R"(Probability distribution over poses, used by the PosePriorErrorFunction.
Currently contains a mixture of probabilistic PCA models.

Each PPCA model is a Gaussian with mean mu and covariance (sigma^2*I + W*W^T).
)")
      .def(py::init())
      .def(
          py::init(&createMppcaModel),
          R"(Construct an Mppca model from numpy arrays.
:param pi: The (nModels) mixture weights
:param mu: The (nModels x dimension) mean vectors.
:param W: The (nModels x dimension x nParams) weight matrices.
:param sigma2: The (nModels) squared sigma uniform variance parameters.
:param names: The (nDimension) names of the affected parameters.
)",
          py::arg("pi"),
          py::arg("mu"),
          py::arg("W"),
          py::arg("sigma"),
          py::arg("names"))
      .def_readonly(
          "n_mixtures",
          &mm::Mppca::p,
          R"(The number of individual Gaussians in the mixture model.)")
      .def_readonly("n_dimension", &mm::Mppca::d, R"(The dimension of the parameter space.)")
      .def_readonly("names", &mm::Mppca::names, R"(The names of the parameters.)")
      .def(
          "to_tensors",
          &mppcaToTensors,
          R"(Return the parameters defining the mixture of probabilistic PCA models.

Each PPCA model a Gaussian N(mu, cov) where the covariance matrix is
(sigma*sigma*I + W * W^T).  pi is the mixture weight for this particular Gaussian.

Note that mu is a vector of length :meth:`dimension` and W is a matrix of dimension :meth:`dimension` x q
where q is the dimensionality of the PCA subspace.

The resulting tensors are as follows:

* pi: a [n]-dimensional tensor containing the mixture weights.  It sums to 1.
* mu: a [n x d]-dimensional tensor containing the mean pose for each mixture.
* weights: a [n x d x q]-dimensional tensor containing the q vectors spanning the PCA space.
* sigma: a [n]-dimensional tensor containing the uniform part of the covariance matrix.
* param_idx: a [d]-dimensional tensor containing the indices of the parameters.

:param parameter_transform: An optional parameter transform used to map the parameters; if not present, then the param_idx tensor will be empty.
:return: an tuple (pi, mean, weights, sigma, param_idx) for the Probabilistic PCA model.)",
          py::arg("parameter_transform") = std::optional<const mm::ParameterTransform*>())
      .def("get_mixture", &getMppcaModel, py::arg("i_model"))
      .def_static(
          "load",
          &loadPosePriorFromFile,
          "Load a mixture PCA model (e.g. poseprior.mppca).",
          py::arg("mppca_filename"))
      .def_static(
          "save",
          &savePosePriorToFile,
          "Save a mixture PCA model to file (e.g. poseprior.mppca).",
          py::arg("mppca"),
          py::arg("mppca_filename"))
      .def_static(
          "from_bytes",
          &loadPosePriorFromBytes,
          "Load a mixture PCA model (e.g. poseprior.mppca).",
          py::arg("mppca_bytes"))
      .def("__repr__", [](const mm::Mppca& mppca) {
        return fmt::format("Mppca(mixtures={}, dimension={})", mppca.p, mppca.d);
      });

  // Class TaperedCapsule, defining the properties:
  //    transformation
  //    radius
  //    parent
  //    length
  capsuleClass
      .def(
          py::init<>([](int parent,
                        const std::optional<Eigen::Matrix4f>& transformation,
                        const std::optional<Eigen::Vector2f>& radius,
                        float length) {
            mm::TaperedCapsule capsule;
            capsule.transformation = transformation.value_or(Eigen::Matrix4f::Identity());
            capsule.radius = radius.value_or(Eigen::Vector2f::Ones());
            capsule.parent = parent;
            capsule.length = length;
            return capsule;
          }),
          R"(Create a capsule using its transformation, radius, parent and length.

:param transformation: Transformation defining the orientation and starting point relative to the parent coordinate system.
:param radius: Start and end radius for the capsule.
:param parent: Parent joint to which the capsule is attached.
:param length: Length of the capsule in local space.)",
          py::arg("parent"),
          py::arg("transformation") = std::optional<Eigen::Matrix4f>{},
          py::arg("radius") = std::optional<Eigen::Vector2f>{},
          py::arg("length") = 1.0f)
      .def_property_readonly(
          "transformation",
          [](const mm::TaperedCapsule& capsule) -> Eigen::Matrix4f {
            return capsule.transformation.toMatrix();
          },
          "Transformation defining the orientation and starting point relative to the parent coordinate system")
      .def_property_readonly(
          "radius",
          [](const mm::TaperedCapsule& capsule) -> Eigen::Vector2f { return capsule.radius; },
          "Start and end radius for the capsule.")
      .def_property_readonly(
          "parent",
          [](const mm::TaperedCapsule& capsule) -> int { return capsule.parent; },
          "Parent joint to which the capsule is attached.")
      .def_property_readonly(
          "length", [](const mm::TaperedCapsule& capsule) -> float { return capsule.length; })
      .def("__repr__", [](const mm::TaperedCapsule& tc) {
        return fmt::format(
            "TaperedCapsule(parent={}, length={}, radius=[{}, {}])",
            tc.parent,
            tc.length,
            tc.radius[0],
            tc.radius[1]);
      });

  // Class Marker, defining the properties:
  //    name
  //    pos
  //    occluded
  markerClass.def(py::init())
      .def(
          py::init<const std::string&, const Eigen::Vector3d&, const bool>(),
          R"(Create a marker with the specified properties.

          :param name: The name of the marker
          :param pos: The 3D position of the marker
          :param occluded: Whether the marker is occluded with no position info
          )",
          py::arg("name"),
          py::arg("pos"),
          py::arg("occluded"))
      .def_readwrite("name", &mm::Marker::name, "Name of the marker")
      .def_readwrite("pos", &mm::Marker::pos, "Marker 3d position")
      .def_readwrite(
          "occluded", &mm::Marker::occluded, "True if the marker is occluded with no position info")
      .def("__repr__", [](const mm::Marker& m) {
        return fmt::format(
            "Marker(name='{}', pos=[{} {} {}], occluded={})",
            m.name,
            m.pos.x(),
            m.pos.y(),
            m.pos.z(),
            m.occluded ? "True" : "False");
      });

  // Class MarkerSequence, defining the properties:
  //    name
  //    frames
  //    fps
  markerSequenceClass.def(py::init())
      .def_readwrite("name", &mm::MarkerSequence::name, "Name of the subject")
      .def_readwrite("frames", &mm::MarkerSequence::frames, "Marker data in [nframes][nMarkers]")
      .def_readwrite("fps", &mm::MarkerSequence::fps, "Frame rate")
      .def("__repr__", [](const mm::MarkerSequence& ms) {
        return fmt::format(
            "MarkerSequence(name='{}', frames={}, fps={})", ms.name, ms.frames.size(), ms.fps);
      });

  // =====================================================
  // momentum::FBXCoordSystemInfo
  // - upVector
  // - frontVector
  // - coordSystem

  // =====================================================

  fbxCoordSystemInfoClass
      .def(
          py::init([](const momentum::FBXUpVector upVector,
                      const momentum::FBXFrontVector frontVector,
                      const momentum::FBXCoordSystem coordSystem) {
            return momentum::FBXCoordSystemInfo{upVector, frontVector, coordSystem};
          }),
          py::arg("upVector"),
          py::arg("frontVector"),
          py::arg("coordSystem"))
      .def_property_readonly(
          "upVector",
          [](const mm::FBXCoordSystemInfo& coordSystemInfo) { return coordSystemInfo.upVector; },
          "Returns the up vector.")
      .def_property_readonly(
          "frontVector",
          [](const mm::FBXCoordSystemInfo& coordSystemInfo) { return coordSystemInfo.frontVector; },
          "Returns the front vector.")
      .def_property_readonly(
          "coordSystem",
          [](const mm::FBXCoordSystemInfo& coordSystemInfo) { return coordSystemInfo.coordSystem; },
          "Returns the coordinate system.")
      .def("__repr__", [](const mm::FBXCoordSystemInfo& info) {
        std::string upVectorStr;
        switch (info.upVector) {
          case mm::FBXUpVector::XAxis:
            upVectorStr = "XAxis";
            break;
          case mm::FBXUpVector::YAxis:
            upVectorStr = "YAxis";
            break;
          case mm::FBXUpVector::ZAxis:
            upVectorStr = "ZAxis";
            break;
          default:
            upVectorStr = "Unknown";
            break;
        }

        std::string frontVectorStr;
        switch (info.frontVector) {
          case mm::FBXFrontVector::ParityEven:
            frontVectorStr = "ParityEven";
            break;
          case mm::FBXFrontVector::ParityOdd:
            frontVectorStr = "ParityOdd";
            break;
          default:
            frontVectorStr = "Unknown";
            break;
        }

        std::string coordSystemStr;
        switch (info.coordSystem) {
          case mm::FBXCoordSystem::RightHanded:
            coordSystemStr = "RightHanded";
            break;
          case mm::FBXCoordSystem::LeftHanded:
            coordSystemStr = "LeftHanded";
            break;
          default:
            coordSystemStr = "Unknown";
            break;
        }

        return fmt::format(
            "FBXCoordSystemInfo(upVector={}, frontVector={}, coordSystem={})",
            upVectorStr,
            frontVectorStr,
            coordSystemStr);
      });

  // loadMotion(gltfFilename)
  m.def(
      "load_motion",
      &loadMotion,
      R"(Load a motion sequence from a gltf file.

Unless you can guarantee that the parameters in the motion files match your existing character,
you will likely want to retarget the parameters using the :meth:`mapParameters` function.

:parameter gltf_filename: A .gltf file; e.g. character_s0.glb.
:return: a tuple [motionData, motionParameterNames, identityData, identityParameterNames].
      )",
      py::arg("gltf_filename"));

  // loadMarkersFromFile(path, mainSubjectOnly)
  // TODO(T138941756): Expose the loadMarker and loadMarkersForMainSubject
  // APIs separately from markerIO.h loadMarkersFromFile(path,
  // mainSubjectOnly)
  m.def(
      "load_markers",
      &loadMarkersFromFile,
      R"(Load 3d mocap marker data from file.

:param path: A marker data file: .c3d, .gltf, or .trc.
:param main_subject_only: True to load only one subject's data.
:param up: The up vector to use for the coordinate system, default to Y.
:return: an array of MarkerSequence, one per subject in the file.
      )",
      py::arg("path"),
      py::arg("main_subject_only") = true,
      py::arg("up") = mm::UpVector::Y);

  // mapModelParameters_names(motionData, sourceParameterNames,
  // targetCharacter)
  m.def(
      "map_model_parameters",
      &mapModelParameters_names,
      R"(Remap model parameters from one character to another.

:param motionData: The source motion data as a nFrames x nParams torch.Tensor.
:param sourceParameterNames: The source parameter names as a list of strings (e.g. c.parameterTransform.name).
:param targetCharacter: The target character to remap onto.

:return: The motion with the parameters remapped to match the passed-in Character. The fields with no match are filled with zero.
      )",
      py::arg("motion_data"),
      py::arg("source_parameter_names"),
      py::arg("target_character"),
      py::arg("verbose") = false);

  // mapModelParameters(motionData, sourceCharacter, targetCharacter)
  m.def(
      "map_model_parameters",
      &mapModelParameters,
      R"(Remap model parameters from one character to another.

:param motionData: The source motion data as a nFrames x nParams torch.Tensor.
:param sourceCharacter: The source character.
:param targetCharacter: The target character to remap onto.
:param verbose: If true, print out warnings about missing parameters.

:return: The motion with the parameters remapped to match the passed-in Character. The fields with no match are filled with zero.
      )",
      py::arg("motion_data"),
      py::arg("source_character"),
      py::arg("target_character"),
      py::arg("verbose") = true);

  // mapJointParameters(motionData, sourceCharacter, targetCharacter)
  m.def(
      "map_joint_parameters",
      &mapJointParameters,
      R"(Remap joint parameters from one character to another.

:param motionData: The source motion data as a [nFrames x (nBones * 7)] torch.Tensor.
:param sourceCharacter: The source character.
:param targetCharacter: The target character to remap onto.

:return: The motion with the parameters remapped to match the passed-in Character. The fields with no match are filled with zero.
      )",
      py::arg("motion_data"),
      py::arg("source_character"),
      py::arg("target_character"));

  // uniformRandomToModelParameters(character, unifNoise)
  m.def(
      "uniform_random_to_model_parameters",
      &uniformRandomToModelParameters,
      R"(Convert a uniform noise vector into a valid body pose.

:parameter character: The character to use.
:parameter unifNoise: A uniform noise tensor, with dimensions (nBatch x nModelParams).
:return: A torch.Tensor with dimensions (nBatch x nModelParams).)",
      py::arg("character"),
      py::arg("unif_noise"));

  m.def(
      "apply_parameter_transform",
      [](py::object character, at::Tensor modelParameters) {
        return applyParamTransform(character, modelParameters);
      },
      R"(Apply the parameter transform to a [nBatch x nParams] tensor of model parameters.
This is functionally identical to :meth:`ParameterTransform.apply` except that it allows
batching on the character.

:param character: A character or list of characters.
:type character: Union[Character, List[Character]]
:param model_parameters: A [nBatch x nParams] tensor of model parameters.

:return: a tensor of joint parameters.)",
      py::arg("character"),
      py::arg("model_parameters"));

  m.def(
      "model_parameters_to_blend_shape_coefficients",
      &modelParametersToBlendShapeCoefficients,
      R"(Extract the model parameters that correspond to the blend shape coefficients, in the order
required to call `meth:BlendShape.compute_shape`.

:param character: A character.
:parameter model_parameters: A [nBatch x nParams] tensor of model parameters.

:return: A [nBatch x nBlendShape] torch.Tensor of blend shape coefficients.)",
      py::arg("character"),
      py::arg("model_parameters"));

  // modelParametersToPositions(character, modelParameters, parents, offsets)
  m.def(
      "model_parameters_to_positions",
      &modelParametersToPositions,
      R"(Convert model parameters to 3D positions relative to skeleton joints using forward kinematics.  You can use this (for example) to
supervise a model to produce the correct 3D ground truth.

Working directly from modelParameters is preferable to mapping to jointParameters first because it does a better job exploiting the
sparsity in the model and therefore can be made somewhat faster.

:param character: Character to use.
:type character: Union[Character, List[Character]]
:param model_parameters: Model parameter tensor, with dimension (nBatch x nModelParams).
:param parents: Joint parents, on for each target position.
:param offsets: 3-d offset in each joint's local space.
:return: Tensor of size (nBatch x nParents x 3), representing the world-space position of each point.
)",
      py::arg("character"),
      py::arg("model_parameters"),
      py::arg("parents"),
      py::arg("offsets"));

  // jointParametersToPositions(character, jointParameters, parents, offsets)
  m.def(
      "joint_parameters_to_positions",
      &jointParametersToPositions,
      R"(Convert joint parameters to 3D positions relative to skeleton joints using forward kinematics.  You can use this (for example) to
supervise a model to produce the correct 3D ground truth.

You should prefer :meth:`model_parameters_to_positions` when working from modelParameters because it is better able to exploit sparsity; this
function is provided as a convenience because motion read from external files generally uses jointParameters.

:param character: Character to use.
:type character: Union[Character, List[Character]]
:param joint_parameters: Joint parameter tensor, with dimension (nBatch x (7*nJoints)).
:param parents: Joint parents, on for each target position.
:param offsets: 3-d offset in each joint's local space.
:return: Tensor of size (nBatch x nParents x 3), representing the world-space position of each point.
)",
      py::arg("character"),
      py::arg("joint_parameters"),
      py::arg("parents"),
      py::arg("offsets"));

  // modelParametersToSkeletonState(characters, modelParameters)
  m.def(
      "model_parameters_to_skeleton_state",
      &modelParametersToSkeletonState,
      R"(Map from the k modelParameters to the 8*nJoints global skeleton state.

The skeletonState is stored (tx, ty, tz; rx, ry, rz, rw; s) and each maps the transform from the joint's local space to worldspace.
Rotations are Quaternions in the ((x, y, z), w) format.  This is deliberately identical to the representation used in a legacy format.)

:param character: Character to use.
:type character: Union[Character, List[Character]]
:param model_parameters: torch.Tensor containing the (nBatch x nModelParameters) model parameters.

:return: torch.Tensor of size (nBatch x nJoints x 8) containing the skeleton state; should be also compatible with a legacy format's skeleton state representation.)",
      py::arg("character"),
      py::arg("model_parameters"));

  // modelParametersToLocalSkeletonState(characters, modelParameters)
  m.def(
      "model_parameters_to_local_skeleton_state",
      &modelParametersToLocalSkeletonState,
      R"(Map from the k modelParameters to the 8*nJoints local skeleton state.

The skeletonState is stored (tx, ty, tz; rx, ry, rz, rw; s) and each maps the transform from the joint's local space to its parent joint space.
Rotations are Quaternions in the ((x, y, z), w) format.  This is deliberately identical to the representation used in a legacy format.)

:param character: Character to use.
:type character: Union[Character, List[Character]]
:param model_parameters: torch.Tensor containing the (nBatch x nModelParameters) model parameters.

:return: torch.Tensor of size (nBatch x nJoints x 8) containing the skeleton state; should be also compatible with a legacy format's skeleton state representation.)",
      py::arg("character"),
      py::arg("model_parameters"));

  // jointParametersToSkeletonState(character, jointParameters)
  m.def(
      "joint_parameters_to_skeleton_state",
      &jointParametersToSkeletonState,
      R"(Map from the 7*nJoints jointParameters to the 8*nJoints global skeleton state.

The skeletonState is stored (tx, ty, tz; rx, ry, rz, rw; s) and each maps the transform from the joint's local space to worldspace.
Rotations are Quaternions in the ((x, y, z), w) format.  This is deliberately identical to the representation used in a legacy format.)

:param character: Character to use.
:type character: Union[Character, List[Character]]
:param joint_parameters: torch.Tensor containing the (nBatch x nJointParameters) joint parameters.

:return: torch.Tensor of size (nBatch x nJoints x 8) containing the skeleton state; should be also compatible with a legacy format's skeleton state representation.)",
      py::arg("character"),
      py::arg("joint_parameters"));

  // jointParametersToLocalSkeletonState(character, jointParameters)
  m.def(
      "joint_parameters_to_local_skeleton_state",
      &jointParametersToLocalSkeletonState,
      R"(Map from the 7*nJoints jointParameters (representing transforms to the parent joint) to the 8*nJoints local skeleton state.

The skeletonState is stored (tx, ty, tz; rx, ry, rz, rw; s) and each maps the transform from the joint's local space to its parent joint space.
Rotations are Quaternions in the ((x, y, z), w) format.  This is deliberately identical to the representation used in a legacy format.)

:param character: Character to use.
:type character: Union[Character, List[Character]]
:param joint_parameters: torch.Tensor containing the (nBatch x nJointParameters) joint parameters.

:return: torch.Tensor of size (nBatch x nJoints x 8) containing the skeleton state; should be also compatible with a legacy format's skeleton state representation.)",
      py::arg("character"),
      py::arg("joint_parameters"));

  // localSkeletonStateToJointParameters(character, skelState)
  m.def(
      "skeleton_state_to_joint_parameters",
      &skeletonStateToJointParameters,
      R"(Map from the 8*nJoints skeleton state (representing transforms to world-space) to the 7*nJoints jointParameters.  This performs the following operations:

* Removing the translation offset.
* Inverting out the pre-rotation.
* Converting to Euler angles.

The skeleton state is stored (tx, ty, tz; rx, ry, rz, rw; s) and transforms from the joint's local space to world-space.
The joint parameters are stored (tx, ty, tz; ry, rz, ry; s) where rotations are in Euler angles and are relative to the parent joint.

:param character: Character to use.
:param skel_state: torch.Tensor containing the ([nBatch] x nJoints x 8) skeleton state.

:return: torch.Tensor of size ([nBatch] x nJoints x 7) containing the joint parameters.)",
      py::arg("character"),
      py::arg("skel_state"));

  // localSkeletonStateToJointParameters(character, skelState)
  m.def(
      "local_skeleton_state_to_joint_parameters",
      &localSkeletonStateToJointParameters,
      R"(Map from the 8*nJoints local skeleton state to the 7*nJoints jointParameters.  This performs the following operations:

* Removing the translation offset.
* Inverting out the pre-rotation.
* Converting to Euler angles.

The local skeleton state is stored (tx, ty, tz; rx, ry, rz, rw; s) and each maps the transform from the joint's local space to its parent joint space.
The joint parameters are stored (tx, ty, tz; ry, rz, ry; s) where rotations are in Euler angles and are relative to the parent joint.

:param character: Character to use.
:param local_skel_state: torch.Tensor containing the ([nBatch] x nJoints x 8) skeleton state.

:return: torch.Tensor of size ([nBatch] x nJoints x 7) containing the joint parameters.)",
      py::arg("character"),
      py::arg("local_skel_state"));

  // stripLowerBodyVertices(character)
  m.def(
      "strip_lower_body_vertices",
      &stripLowerBodyVertices,
      R"(Returns a character where all vertices below the waist have been stripped out (without modifying the skeleton).
This can be useful for visualization if you don't want the legs to distract.

:param character: Full-body character.
:return: A new character with only the upper body visible.)",
      py::arg("character"));

  m.def(
      "strip_joints",
      [](const momentum::Character& c, const std::vector<std::string>& joints_in) {
        std::vector<size_t> joints;
        for (const auto& j : joints_in) {
          const auto idx = c.skeleton.getJointIdByName(j);
          MT_THROW_IF(
              idx == momentum::kInvalidIndex,
              "Trying to remove nonexistent joint '{}' from skeleton.",
              j);
          joints.push_back(idx);
        }

        return momentum::removeJoints(c, joints);
      },
      R"(Returns a character where the passed-in joints and all joints parented underneath them have been removed.

:param character: Full-body character.
:param joint_names: Names of the joints to remove.
:return: A new character with only the upper body visible.)",
      py::arg("character"),
      py::arg("joint_names"));

  // replace_skeleton_recursive(character, activeParameters)
  m.def(
      "replace_skeleton_hierarchy",
      momentum::replaceSkeletonHierarchy,
      R"(Replaces the part of target_character's skeleton rooted at target_root with the part of
source_character's skeleton rooted at source_root.
This is used e.g. to swap one character's hand skeleton with another.

:param source_character: Source character.
:param target_character: Target character.
:param source_root: Root of the source skeleton hierarchy to be copied.
:param target_root: Root of the target skeleton hierarchy to be replaced.
:return: A new skeleton that is identical to tgt_skeleton except that everything under target_root
   has been replaced by the part of source_character rooted at source_root.
    )",
      py::arg("source_character"),
      py::arg("target_character"),
      py::arg("source_root"),
      py::arg("target_root"));

  // reduceMeshByVertices(character, activeVertices)
  m.def(
      "reduce_mesh_by_vertices",
      [](const momentum::Character& character, const py::array_t<bool>& activeVertices) {
        return momentum::reduceMeshByVertices(character, boolArrayToVector(activeVertices));
      },
      R"(Reduces the mesh to only include the specified vertices and associated faces.

Creates a new character with mesh reduced to the specified vertices. This function
handles all character components including mesh vertices/faces, skin weights,
blend shapes, pose shapes, and other mesh-related data.

:param character: Full-body character.
:param active_vertices: A boolean array marking which vertices should be retained.
:return: A new character whose mesh only includes the marked vertices and their associated data.)",
      py::arg("character"),
      py::arg("active_vertices"));

  // reduceMeshByFaces(character, activeFaces)
  m.def(
      "reduce_mesh_by_faces",
      [](const momentum::Character& character, const py::array_t<bool>& activeFaces) {
        return momentum::reduceMeshByFaces(character, boolArrayToVector(activeFaces));
      },
      R"(Reduces the mesh to only include the specified faces and associated vertices.

Creates a new character with mesh reduced to the specified faces. This function
handles all character components including mesh vertices/faces, skin weights,
blend shapes, pose shapes, and other mesh-related data.

:param character: Full-body character.
:param active_faces: A boolean array marking which faces should be retained.
:return: A new character whose mesh only includes the marked faces and their associated data.)",
      py::arg("character"),
      py::arg("active_faces"));

  // reduceToSelectedModelParameters(character, activeParameters)
  m.def(
      "reduce_to_selected_model_parameters",
      [](const momentum::Character& character, at::Tensor activeParameters) {
        return character.simplifyParameterTransform(
            tensorToParameterSet(character.parameterTransform, activeParameters));
      },
      R"(Strips out unused parameters from the parameter transform.

:param character: Full-body character.
:param activeParameters: A boolean tensor marking which parameters should be retained.
:return: A new character whose parameter transform only includes the marked parameters.)",
      py::arg("character"),
      py::arg("active_parameters"));

  m.def(
      "find_closest_points",
      &findClosestPoints,
      R"(For each point in the points_source tensor, find the closest point in the points_target tensor.  This version of find_closest points supports both 2- and 3-dimensional point sets.

:param points_source: [nBatch x nPoints x dim] tensor of source points (dim must be 2 or 3).
:param points_target: [nBatch x nPoints x dim] tensor of target points (dim must be 2 or 3).
:param max_dist: Maximum distance to search, can be used to speed up the method by allowing the search to return early.  Defaults to FLT_MAX.
:return: A tuple of three tensors.  The first is [nBatch x nPoints x dim] and contains the closest point for each point in the target set.
         The second is [nBatch x nPoints] and contains the index of each closest point in the target set (or -1 if none).
         The third is [nBatch x nPoints] and is a boolean tensor indicating whether a valid closest point was found for each source point.
      )",
      py::arg("points_source"),
      py::arg("points_target"),
      py::arg("max_dist") = std::numeric_limits<float>::max());

  m.def(
      "find_closest_points",
      &findClosestPointsWithNormals,
      R"(For each point in the points_source tensor, find the closest point in the points_target tensor whose normal is compatible (n_source . n_target > max_normal_dot).
Using the normal is a good way to avoid certain kinds of bad matches, such as matching the front of the body against depth values from the back of the body.

:param points_source: [nBatch x nPoints x 3] tensor of source points.
:param normals_source: [nBatch x nPoints x 3] tensor of source normals (must be normalized).
:param points_target: [nBatch x nPoints x 3] tensor of target points.
:param normals_target: [nBatch x nPoints x 3] tensor of target normals (must be normalized).
:param max_dist: Maximum distance to search, can be used to speed up the method by allowing the search to return early.  Defaults to FLT_MAX.
:param max_normal_dot: Maximum dot product allowed between the source and target normal.  Defaults to 0.
:return: A tuple of three tensors.  The first is [nBatch x nPoints x dim] and contains the closest point for each point in the target set.
         The second is [nBatch x nPoints] and contains the index of each closest point in the target set (or -1 if none).
         The third is [nBatch x nPoints] and is a boolean tensor indicating whether a valid closest point was found for each source point.
      )",
      py::arg("points_source"),
      py::arg("normals_source"),
      py::arg("points_target"),
      py::arg("normals_target"),
      py::arg("max_dist") = std::numeric_limits<float>::max(),
      py::arg("max_normal_dot") = 0.0f);

  m.def(
      "find_closest_points_on_mesh",
      &findClosestPointsOnMesh,
      R"(For each point in the points_source tensor, find the closest point in the target mesh.

  :param points_source: [nBatch x nPoints x 3] tensor of source points.
  :param vertices_target: [nBatch x nPoints x 3] tensor of target vertices.
  :param faces_target: [nBatch x nPoints x 3] tensor of target faces.
  :return: A tuple of four tensors, (valid, points, face_index, bary).  The first is [nBatch x nPoints] and specifies if the closest point result is valid.
           The second is [nBatch x nPoints x 3] and contains the actual closest point (or 0, 0, 0 if invalid).
           The third is [nBatch x nPoints] and contains the index of the closest face (or -1 if invalid).
           The fourth is [nBatch x nPoints x 3] and contains the barycentric coordinates of the closest point on the face (or 0, 0, 0 if invalid).
        )",
      py::arg("points_source"),
      py::arg("vertices_target"),
      py::arg("faces_target"));

  m.def(
      "replace_rest_mesh",
      &replaceRestMesh,
      R"(Return a new :class:`Character` with the rest mesh positions replaced by the passed-in positions.
        Can be used to e.g. bake the blend shapes into the character's mesh.  Does not allow changing the topology.

:param rest_vertex_positions: nVert x 3 numpy array of vertex positions.
        )",
      py::arg("character"),
      py::arg("rest_vertex_positions"));

  m.def(
      "compute_vertex_normals",
      &computeVertexNormals,
      R"(
Computes vertex normals for a triangle mesh given its positions.

:param vertex_positions: [nBatch] x nVert x 3 Tensor of vertex positions.
:param triangles: nTriangles x 3 Tensor of triangle indices.
:return: Smooth per-vertex normals.
    )",
      py::arg("vertex_positions"),
      py::arg("triangles"));

  m.def(
      "create_test_character",
      &createTestCharacter,
      R"(Create a simple 3-joint test character with blendshapes.  This is useful for writing
confidence tests that execute quickly and don't rely on outside files.

The mesh is made by a few vertices on the line segment from (1,0,0) to (1,1,0) and a few dummy
faces. The skeleton has three joints: root at (0,0,0), joint1 parented by root, at world-space
(0,1,0), and joint2 parented by joint1, at world-space (0,2,0).
The character has only one parameter limit: min-max type [-0.1, 0.1] for root.

:parameter numJoints: The number of joints in the resulting character.
:return: A simple character with 3 joints, 10 model parameters, and 5 blendshapes.
      )",
      py::arg("num_joints") = 3,
      py::arg("with_blendshapes") = false);

  // createTestPosePrior()
  m.def(
      "create_test_mppca",
      &momentum::createDefaultPosePrior<float>,
      R"(Create a pose prior that acts on the simple 3-joint test character.

:return: A simple pose prior.)");

  registerJointBindings(jointClass);
  registerSkeletonBindings(skeletonClass);

  registerSkinWeightsBindings(skinWeightsClass);

  registerParameterTransformBindings(parameterTransformClass);
  registerInverseParameterTransformBindings(inverseParameterTransformClass);

  registerLimits(m, parameterLimitClass);

  registerCharacterBindings(characterClass);

  // Register GltfBuilder bindings
  registerGltfBuilderBindings(m);
}
