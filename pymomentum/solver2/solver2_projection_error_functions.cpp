/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pymomentum/solver2/solver2_error_functions.h>

#include <momentum/character/character.h>
#include <momentum/character_solver/camera_projection_error_function.h>
#include <momentum/character_solver/projection_error_function.h>
#include <momentum/character_solver/vertex_projection_error_function.h>
#include <pymomentum/solver2/solver2_utility.h>

#include <fmt/format.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <variant>

namespace py = pybind11;
namespace mm = momentum;

namespace pymomentum {

void addProjectionErrorFunctions(py::module_& m) {
  // Projection error function
  py::class_<mm::ProjectionConstraintDataT<float>>(m, "ProjectionConstraint")
      .def(
          "__repr__",
          [](const mm::ProjectionConstraintDataT<float>& self) {
            return fmt::format(
                "ProjectionConstraint(parent={}, weight={}, offset=[{:.3f}, {:.3f}, {:.3f}], target=[{:.3f}, {:.3f}])",
                self.parent,
                self.weight,
                self.offset.x(),
                self.offset.y(),
                self.offset.z(),
                self.target.x(),
                self.target.y());
          })
      .def_readonly(
          "parent", &mm::ProjectionConstraintDataT<float>::parent, "The parent joint index")
      .def_readonly(
          "weight", &mm::ProjectionConstraintDataT<float>::weight, "The weight of the constraint")
      .def_readonly(
          "offset", &mm::ProjectionConstraintDataT<float>::offset, "The offset in parent space")
      .def_readonly(
          "target", &mm::ProjectionConstraintDataT<float>::target, "The target 2D position")
      .def_readonly(
          "projection",
          &mm::ProjectionConstraintDataT<float>::projection,
          "The 3x4 projection matrix");
  py::class_<
      mm::ProjectionErrorFunctionT<float>,
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::ProjectionErrorFunctionT<float>>>(
      m,
      "ProjectionErrorFunction",
      R"(The ProjectionErrorFunction projects a 3D point to a 2D point using a projection matrix.
This is useful for camera-based constraints where you want to match a 3D point to a 2D observation.)")
      .def(
          "__repr__",
          [](const mm::ProjectionErrorFunctionT<float>& self) {
            return fmt::format(
                "ProjectionErrorFunction(weight={}, num_constraints={})",
                self.getWeight(),
                self.getNumConstraints());
          })
      .def(
          py::init<>(
              [](const mm::Character& character,
                 float nearClip,
                 float weight) -> std::shared_ptr<mm::ProjectionErrorFunctionT<float>> {
                validateWeight(weight, "weight");
                auto result = std::make_shared<mm::ProjectionErrorFunctionT<float>>(
                    character.skeleton, character.parameterTransform, nearClip);
                result->setWeight(weight);
                return result;
              }),
          R"(Initialize a ProjectionErrorFunction.

            :param character: The character to use.
            :param near_clip: The near clip distance. Points closer than this distance to the camera will be ignored.
            :param weight: The weight applied to the error function.)",
          py::keep_alive<1, 2>(),
          py::arg("character"),
          py::kw_only(),
          py::arg("near_clip") = 1.0f,
          py::arg("weight") = 1.0f)
      .def(
          "add_constraint",
          [](mm::ProjectionErrorFunctionT<float>& self,
             const Eigen::Matrix<float, 3, 4>& projection,
             const Eigen::Vector2f& target,
             int parent,
             const std::optional<Eigen::Vector3f>& offset,
             float weight) {
            validateJointIndex(parent, "parent", self.getSkeleton());
            validateWeight(weight, "weight");
            mm::ProjectionConstraintDataT<float> constraint;
            constraint.projection = projection;
            constraint.parent = parent;
            constraint.offset = offset.value_or(Eigen::Vector3f::Zero());
            constraint.weight = weight;
            constraint.target = target;
            self.addConstraint(constraint);
          },
          R"(Adds a projection constraint to the error function.

        :param projection: The 3x4 projection matrix.
        :param target: The 2D target point.
        :param parent: The index of the parent joint.
        :param offset: The offset from the parent joint in local space.
        :param weight: The weight of the constraint.)",
          py::arg("projection"),
          py::arg("target"),
          py::arg("parent"),
          py::arg("offset") = std::nullopt,
          py::arg("weight") = 1.0f)
      .def(
          "add_constraints",
          [](mm::ProjectionErrorFunctionT<float>& self,
             const py::array_t<float>& projection,
             const py::array_t<float>& target,
             const py::array_t<int>& parent,
             const std::optional<py::array_t<float>>& offset,
             const std::optional<py::array_t<float>>& weight) {
            ArrayShapeValidator validator;
            const int nConsIdx = -1;
            validator.validate(
                projection, "projection", {nConsIdx, 3, 4}, {"n_cons", "rows", "cols"});
            validator.validate(target, "target", {nConsIdx, 2}, {"n_cons", "xy"});
            validator.validate(parent, "parent", {nConsIdx}, {"n_cons"});
            validateJointIndex(parent, "parent", self.getSkeleton());
            validator.validate(offset, "offset", {nConsIdx, 3}, {"n_cons", "xyz"});
            validator.validate(weight, "weight", {nConsIdx}, {"n_cons"});

            auto projectionAcc = projection.unchecked<3>();
            auto targetAcc = target.unchecked<2>();
            auto parentAcc = parent.unchecked<1>();
            auto offsetAcc =
                offset.has_value() ? std::make_optional(offset->unchecked<2>()) : std::nullopt;
            auto weightAcc =
                weight.has_value() ? std::make_optional(weight->unchecked<1>()) : std::nullopt;

            py::gil_scoped_release release;

            for (py::ssize_t i = 0; i < parent.shape(0); ++i) {
              mm::ProjectionConstraintDataT<float> constraint;

              // Set projection matrix
              Eigen::Matrix<float, 3, 4> proj;
              for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 4; ++col) {
                  proj(row, col) = projectionAcc(i, row, col);
                }
              }
              constraint.projection = proj;

              // Set target
              constraint.target = Eigen::Vector2f(targetAcc(i, 0), targetAcc(i, 1));

              // Set parent
              constraint.parent = parentAcc(i);

              // Set offset
              constraint.offset = offsetAcc.has_value()
                  ? Eigen::Vector3f((*offsetAcc)(i, 0), (*offsetAcc)(i, 1), (*offsetAcc)(i, 2))
                  : Eigen::Vector3f::Zero();

              // Set weight
              constraint.weight = weightAcc.has_value() ? (*weightAcc)(i) : 1.0f;

              self.addConstraint(constraint);
            }
          },
          R"(Adds multiple projection constraints to the error function.

        :param projection: A numpy array of shape (n, 3, 4) for the projection matrices.
        :param target: A numpy array of shape (n, 2) for the 2D target points.
        :param parent: A numpy array of size n for the indices of the parent joints.
        :param offset: A numpy array of shape (n, 3) for the offsets from the parent joints in local space.
        :param weight: A numpy array of size n for the weights of the constraints.)",
          py::arg("projection"),
          py::arg("target"),
          py::arg("parent"),
          py::arg("offset") = std::nullopt,
          py::arg("weight") = std::nullopt)
      .def(
          "clear_constraints",
          &mm::ProjectionErrorFunctionT<float>::clearConstraints,
          "Clears all projection constraints from the error function.")
      .def(
          "empty",
          &mm::ProjectionErrorFunctionT<float>::empty,
          "Returns true if there are no constraints.")
      .def(
          "num_constraints",
          &mm::ProjectionErrorFunctionT<float>::getNumConstraints,
          "Returns the number of constraints.")
      .def_property_readonly(
          "constraints",
          &mm::ProjectionErrorFunctionT<float>::getConstraints,
          "Returns the list of projection constraints.");

  // Vertex projection error function
  py::class_<mm::VertexProjectionDataT<float>>(
      m, "VertexProjectionConstraint", "Read-only access to a vertex projection constraint.")
      .def(
          "__repr__",
          [](const mm::VertexProjectionDataT<float>& self) {
            return fmt::format(
                "VertexProjectionConstraint(vertex_index={}, weight={}, target_position=[{:.3f}, {:.3f}])",
                self.vertexIndex,
                self.weight,
                self.target.x(),
                self.target.y());
          })
      .def_readonly(
          "vertex_index",
          &mm::VertexProjectionDataT<float>::vertexIndex,
          "Returns the index of the vertex to project.")
      .def_readonly(
          "weight",
          &mm::VertexProjectionDataT<float>::weight,
          "Returns the weight of the constraint.")
      .def_readonly(
          "target_position",
          &mm::VertexProjectionDataT<float>::target,
          "Returns the target 2D position.")
      .def_readonly(
          "projection",
          &mm::VertexProjectionDataT<float>::projectionMatrix,
          "Returns the 3x4 projection matrix.");

  py::class_<
      mm::VertexProjectionErrorFunctionT<float>,
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::VertexProjectionErrorFunctionT<float>>>(
      m,
      "VertexProjectionErrorFunction",
      R"(The VertexProjectionErrorFunction projects 3D vertices onto 2D points using a projection matrix.
This is useful for camera-based constraints where you want to match a 3D vertex to a 2D observation.)")
      .def(
          "__repr__",
          [](const mm::VertexProjectionErrorFunctionT<float>& self) {
            return fmt::format(
                "VertexProjectionErrorFunction(weight={}, num_constraints={})",
                self.getWeight(),
                self.getNumConstraints());
          })
      .def(
          py::init<>(
              [](const mm::Character& character,
                 size_t maxThreads,
                 float weight) -> std::shared_ptr<mm::VertexProjectionErrorFunctionT<float>> {
                validateWeight(weight, "weight");
                auto result = std::make_shared<mm::VertexProjectionErrorFunctionT<float>>(
                    character, character.parameterTransform);
                result->setMaxThreads(static_cast<uint32_t>(maxThreads));
                result->setWeight(weight);
                return result;
              }),
          R"(Initialize a VertexProjectionErrorFunction.

            :param character: The character to use.
            :param max_threads: The maximum number of threads to use for computation.)",
          py::keep_alive<1, 2>(),
          py::arg("character"),
          py::kw_only(),
          py::arg("max_threads") = 0,
          py::arg("weight") = 1.0f)
      .def(
          "add_constraint",
          [](mm::VertexProjectionErrorFunctionT<float>& self,
             int vertexIndex,
             float weight,
             const Eigen::Vector2f& targetPosition,
             const Eigen::Matrix<float, 3, 4>& projection) {
            validateVertexIndex(vertexIndex, "vertex_index", self.getCharacter());
            validateWeight(weight, "weight");
            self.addConstraint(
                mm::VertexProjectionDataT<float>(vertexIndex, targetPosition, projection, weight));
          },
          R"(Adds a vertex projection constraint to the error function.

        :param vertex_index: The index of the vertex to project.
        :param weight: The weight of the constraint.
        :param target_position: The target 2D position.
        :param projection: The 3x4 projection matrix.)",
          py::arg("vertex_index"),
          py::arg("weight"),
          py::arg("target_position"),
          py::arg("projection"))
      .def(
          "clear_constraints",
          &mm::VertexProjectionErrorFunctionT<float>::clearConstraints,
          "Clears all vertex projection constraints from the error function.")
      .def_property_readonly(
          "constraints",
          [](const mm::VertexProjectionErrorFunctionT<float>& self) {
            return self.getConstraints();
          },
          "Returns the list of vertex projection constraints.")
      .def(
          "add_constraints",
          [](mm::VertexProjectionErrorFunctionT<float>& self,
             const py::array_t<int>& vertexIndex,
             const py::array_t<float>& weight,
             const py::array_t<float>& targetPosition,
             const py::array_t<float>& projection) {
            ArrayShapeValidator validator;
            const int nConsIdx = -1;
            validator.validate(vertexIndex, "vertex_index", {nConsIdx}, {"n_cons"});
            validateVertexIndex(vertexIndex, "vertex_index", self.getCharacter());
            validator.validate(weight, "weight", {nConsIdx}, {"n_cons"});
            validator.validate(targetPosition, "target_position", {nConsIdx, 2}, {"n_cons", "xy"});
            validator.validate(
                projection, "projection", {nConsIdx, 3, 4}, {"n_cons", "rows", "cols"});

            auto vertexIndexAcc = vertexIndex.unchecked<1>();
            auto weightAcc = weight.unchecked<1>();
            auto targetPositionAcc = targetPosition.unchecked<2>();
            auto projectionAcc = projection.unchecked<3>();

            py::gil_scoped_release release;

            for (py::ssize_t i = 0; i < vertexIndex.shape(0); ++i) {
              // Set projection matrix
              Eigen::Matrix<float, 3, 4> proj;
              for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 4; ++col) {
                  proj(row, col) = projectionAcc(i, row, col);
                }
              }

              self.addConstraint(
                  mm::VertexProjectionDataT<float>(
                      static_cast<int>(vertexIndexAcc(i)),
                      Eigen::Vector2f(targetPositionAcc(i, 0), targetPositionAcc(i, 1)),
                      proj,
                      weightAcc(i)));
            }
          },
          R"(Adds multiple vertex projection constraints to the error function.

        :param vertex_index: A numpy array of size n for the indices of the vertices to project.
        :param weight: A numpy array of size n for the weights of the constraints.
        :param target_position: A numpy array of shape (n, 2) for the target 2D positions.
        :param projection: A numpy array of shape (n, 3, 4) for the projection matrices.)",
          py::arg("vertex_index"),
          py::arg("weight"),
          py::arg("target_position"),
          py::arg("projection"));

  // Camera projection error function
  py::class_<mm::ProjectionConstraintT<float>>(
      m, "CameraProjectionConstraint", "Read-only access to a camera projection constraint.")
      .def(
          "__repr__",
          [](const mm::ProjectionConstraintT<float>& self) {
            return fmt::format(
                "CameraProjectionConstraint(parent={}, weight={}, offset=[{:.3f}, {:.3f}, {:.3f}], target=[{:.3f}, {:.3f}])",
                self.parent,
                self.weight,
                self.offset.x(),
                self.offset.y(),
                self.offset.z(),
                self.target.x(),
                self.target.y());
          })
      .def_readonly("parent", &mm::ProjectionConstraintT<float>::parent, "The parent joint index.")
      .def_readonly(
          "offset",
          &mm::ProjectionConstraintT<float>::offset,
          "The offset from the parent joint in local space.")
      .def_readonly(
          "weight", &mm::ProjectionConstraintT<float>::weight, "The weight of the constraint.")
      .def_readonly(
          "target", &mm::ProjectionConstraintT<float>::target, "The target 2D pixel position.");

  py::class_<
      mm::CameraProjectionErrorFunctionT<float>,
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::CameraProjectionErrorFunctionT<float>>>(
      m,
      "CameraProjectionErrorFunction",
      R"(Projects 3D skeleton points through a bone-parented camera using an IntrinsicsModel
and penalizes reprojection error in pixel space.

The camera is rigidly attached to a skeleton joint (camera_parent) with a fixed offset
(camera_offset). If camera_parent is -1, the camera is static in world space and
camera_offset is used directly as the eye-from-world transform.)")
      .def(
          "__repr__",
          [](const mm::CameraProjectionErrorFunctionT<float>& self) {
            return fmt::format(
                "CameraProjectionErrorFunction(weight={}, num_constraints={})",
                self.getWeight(),
                self.getNumConstraints());
          })
      .def(
          py::init<>(
              [](const mm::Character& character,
                 std::variant<mm::Camera, std::shared_ptr<const mm::IntrinsicsModel>>
                     cameraOrIntrinsics,
                 std::optional<int> cameraParent,
                 std::optional<Eigen::Matrix4f> cameraOffset,
                 float weight) -> std::shared_ptr<mm::CameraProjectionErrorFunctionT<float>> {
                validateWeight(weight, "weight");
                size_t parent = mm::kInvalidIndex;
                if (cameraParent.has_value()) {
                  validateJointIndex(*cameraParent, "camera_parent", character.skeleton);
                  parent = static_cast<size_t>(*cameraParent);
                }
                std::shared_ptr<const mm::IntrinsicsModel> intrinsicsModel;
                Eigen::Affine3f offset;
                if (auto* camera = std::get_if<mm::Camera>(&cameraOrIntrinsics)) {
                  intrinsicsModel = camera->intrinsicsModel();
                  if (cameraOffset.has_value()) {
                    Eigen::Affine3f additionalOffset;
                    additionalOffset.matrix() = *cameraOffset;
                    offset = additionalOffset * camera->eyeFromWorld();
                  } else {
                    offset = camera->eyeFromWorld();
                  }
                } else {
                  intrinsicsModel =
                      std::get<std::shared_ptr<const mm::IntrinsicsModel>>(cameraOrIntrinsics);
                  if (!intrinsicsModel) {
                    throw std::invalid_argument("intrinsics_model must not be None.");
                  }
                  offset.matrix() = cameraOffset.value_or(Eigen::Matrix4f::Identity());
                }
                auto result = std::make_shared<mm::CameraProjectionErrorFunctionT<float>>(
                    character.skeleton,
                    character.parameterTransform,
                    std::move(intrinsicsModel),
                    parent,
                    offset);
                result->setWeight(weight);
                return result;
              }),
          R"(Initialize a CameraProjectionErrorFunction.

            The second argument can be either a :class:`pymomentum.camera.Camera` or a
            :class:`pymomentum.camera.IntrinsicsModel` (e.g.
            :class:`pymomentum.camera.PinholeIntrinsicsModel`).

            When a :class:`pymomentum.camera.Camera` is provided, its intrinsics and
            eye-from-world transform are used. If ``camera_offset`` is also provided, it
            is composed as an additional transform: ``camera_offset @ camera.eye_from_world``.

            When a :class:`pymomentum.camera.IntrinsicsModel` is provided, the
            ``camera_offset`` parameter specifies the eye-from-world transform (defaults
            to identity if not provided).

            :param character: The character to use.
            :param camera: A :class:`pymomentum.camera.Camera` or
                :class:`pymomentum.camera.IntrinsicsModel` (e.g.
                :class:`pymomentum.camera.PinholeIntrinsicsModel`).
            :param camera_parent: The joint index that the camera is attached to, or
                None for a static camera in world space.
            :param camera_offset: A 4x4 transformation matrix. When used with a
                :class:`pymomentum.camera.Camera`, this is composed as an additional
                transform on top of the camera's eye-from-world. When used with a
                :class:`pymomentum.camera.IntrinsicsModel`, this is the eye-from-world
                transform directly.
            :param weight: The weight applied to the error function.)",
          py::keep_alive<1, 2>(),
          py::keep_alive<1, 3>(),
          py::arg("character"),
          py::arg("camera"),
          py::kw_only(),
          py::arg("camera_parent") = py::none(),
          py::arg("camera_offset") = py::none(),
          py::arg("weight") = 1.0f)
      .def(
          "add_constraint",
          [](mm::CameraProjectionErrorFunctionT<float>& self,
             int parent,
             const Eigen::Vector2f& target,
             const std::optional<Eigen::Vector3f>& offset,
             float weight) {
            validateJointIndex(parent, "parent", self.getSkeleton());
            validateWeight(weight, "weight");
            mm::ProjectionConstraintT<float> constraint;
            constraint.parent = parent;
            constraint.target = target;
            constraint.offset = offset.value_or(Eigen::Vector3f::Zero());
            constraint.weight = weight;
            self.addConstraint(constraint);
          },
          R"(Adds a camera projection constraint.

        :param parent: The index of the parent joint.
        :param target: The 2D target pixel position.
        :param offset: The offset from the parent joint in local space.
        :param weight: The weight of the constraint.)",
          py::arg("parent"),
          py::arg("target"),
          py::arg("offset") = std::nullopt,
          py::arg("weight") = 1.0f)
      .def(
          "add_constraints",
          [](mm::CameraProjectionErrorFunctionT<float>& self,
             const py::array_t<int>& parent,
             const py::array_t<float>& target,
             const std::optional<py::array_t<float>>& offset,
             const std::optional<py::array_t<float>>& weight) {
            ArrayShapeValidator validator;
            const int nConsIdx = -1;
            validator.validate(parent, "parent", {nConsIdx}, {"n_cons"});
            validator.validate(target, "target", {nConsIdx, 2}, {"n_cons", "xy"});
            validateJointIndex(parent, "parent", self.getSkeleton());
            validator.validate(offset, "offset", {nConsIdx, 3}, {"n_cons", "xyz"});
            validator.validate(weight, "weight", {nConsIdx}, {"n_cons"});

            auto parentAcc = parent.unchecked<1>();
            auto targetAcc = target.unchecked<2>();
            auto offsetAcc =
                offset.has_value() ? std::make_optional(offset->unchecked<2>()) : std::nullopt;
            auto weightAcc =
                weight.has_value() ? std::make_optional(weight->unchecked<1>()) : std::nullopt;

            py::gil_scoped_release release;

            std::vector<mm::ProjectionConstraintT<float>> constraints;
            constraints.reserve(parent.shape(0));
            for (py::ssize_t i = 0; i < parent.shape(0); ++i) {
              mm::ProjectionConstraintT<float> constraint;
              constraint.parent = parentAcc(i);
              constraint.target = Eigen::Vector2f(targetAcc(i, 0), targetAcc(i, 1));
              constraint.offset = offsetAcc.has_value()
                  ? Eigen::Vector3f((*offsetAcc)(i, 0), (*offsetAcc)(i, 1), (*offsetAcc)(i, 2))
                  : Eigen::Vector3f::Zero();
              constraint.weight = weightAcc.has_value() ? (*weightAcc)(i) : 1.0f;
              constraints.push_back(constraint);
            }
            self.setConstraints(std::move(constraints));
          },
          R"(Adds multiple camera projection constraints.

        :param parent: A numpy array of size n for the indices of the parent joints.
        :param target: A numpy array of shape (n, 2) for the 2D target pixel positions.
        :param offset: A numpy array of shape (n, 3) for the offsets from the parent joints in local space.
        :param weight: A numpy array of size n for the weights of the constraints.)",
          py::arg("parent"),
          py::arg("target"),
          py::arg("offset") = std::nullopt,
          py::arg("weight") = std::nullopt)
      .def(
          "clear_constraints",
          &mm::CameraProjectionErrorFunctionT<float>::clearConstraints,
          "Clears all camera projection constraints.")
      .def(
          "num_constraints",
          &mm::CameraProjectionErrorFunctionT<float>::getNumConstraints,
          "Returns the number of constraints.")
      .def_property_readonly(
          "constraints",
          &mm::CameraProjectionErrorFunctionT<float>::getConstraints,
          "Returns the list of camera projection constraints.");
}

} // namespace pymomentum
