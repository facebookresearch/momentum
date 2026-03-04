/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pymomentum/solver2/solver2_error_functions.h>

#include <momentum/character/character.h>
#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character_solver/collision_error_function.h>
#include <momentum/character_solver/height_error_function.h>
#include <momentum/character_solver/limit_error_function.h>
#include <momentum/character_solver/model_parameters_error_function.h>
#include <momentum/character_solver/orientation_error_function.h>
#include <momentum/character_solver/point_triangle_vertex_error_function.h>
#include <momentum/character_solver/pose_prior_error_function.h>
#include <momentum/character_solver/position_error_function.h>
#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/character_solver/state_error_function.h>
#include <momentum/character_solver/vertex_sdf_error_function.h>
#include <pymomentum/python_utility/eigen_quaternion.h>
#include <pymomentum/solver2/solver2_utility.h>

#include <fmt/format.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Core>

namespace py = pybind11;
namespace mm = momentum;

namespace pymomentum {

void addErrorFunctions(py::module_& m) {
  py::class_<
      mm::LimitErrorFunction,
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::LimitErrorFunction>>(
      m,
      "LimitErrorFunction",
      "A skeleton error function that penalizes joints that exceed their limits.")
      .def(
          "__repr__",
          [](const mm::LimitErrorFunction& self) {
            return fmt::format("LimitErrorFunction(weight={})", self.getWeight());
          })
      .def(
          py::init<>([](const mm::Character& character, float weight) {
            validateWeight(weight, "weight");
            auto result = std::make_shared<mm::LimitErrorFunction>(character);
            result->setWeight(weight);
            return result;
          }),
          py::keep_alive<1, 2>(),
          py::arg("character"),
          py::kw_only(),
          py::arg("weight") = 1.0f);

  py::class_<
      mm::HeightErrorFunction,
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::HeightErrorFunction>>(
      m,
      "HeightErrorFunction",
      R"(Error function for character height constraints.

This error function measures the height of a character mesh by projecting
all vertices onto a specified "up" direction and computing the difference
between the maximum and minimum projections.

Unlike most error functions, this one only depends on specific active parameters
that are automatically determined: blend shape, face expression, and scale parameters
only (not pose parameters). This allows constraining height independent of pose.)")
      .def(
          "__repr__",
          [](const mm::HeightErrorFunction& self) {
            return fmt::format(
                "HeightErrorFunction(weight={}, target_height={:.3f})",
                self.getWeight(),
                self.getTargetHeight());
          })
      .def(
          py::init<>([](const mm::Character& character,
                        float targetHeight,
                        const std::optional<Eigen::Vector3f>& upDirection,
                        size_t k,
                        float weight) {
            validateWeight(weight, "weight");
            auto result = std::make_shared<mm::HeightErrorFunction>(
                character, targetHeight, upDirection.value_or(Eigen::Vector3f::UnitY()), k);
            result->setWeight(weight);
            return result;
          }),
          R"(Initialize a HeightErrorFunction.

:param character: The character to measure height for.
:param target_height: The target height for the character (required).
:param up_direction: The direction to measure height along (defaults to Y-axis [0, 1, 0]).
:param k: Number of vertices to average for min/max height calculation (defaults to 1).
:param weight: The weight applied to the error function.)",
          py::keep_alive<1, 2>(),
          py::arg("character"),
          py::arg("target_height"),
          py::kw_only(),
          py::arg("up_direction") = std::nullopt,
          py::arg("k") = 1,
          py::arg("weight") = 1.0f)
      .def_property(
          "target_height",
          &mm::HeightErrorFunction::getTargetHeight,
          &mm::HeightErrorFunction::setTargetHeight,
          "The target height for the character.")
      .def_property(
          "up_direction",
          &mm::HeightErrorFunction::getUpDirection,
          &mm::HeightErrorFunction::setUpDirection,
          "The up direction for height measurement.");

  py::class_<
      mm::ModelParametersErrorFunction,
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::ModelParametersErrorFunction>>(
      m,
      "ModelParametersErrorFunction",
      "A skeleton error function that penalizes the difference between the target model parameters and the current model parameters.")
      .def(
          "__repr__",
          [](const mm::ModelParametersErrorFunction& self) {
            return fmt::format("ModelParametersErrorFunction(weight={})", self.getWeight());
          })
      .def(
          py::init<>([](const mm::Character& character,
                        const std::optional<py::array_t<float>>& targetParams,
                        const std::optional<py::array_t<float>>& weights,
                        float weight) {
            validateWeight(weight, "weight");
            validateWeights(weights, "weights");
            auto result = std::make_shared<mm::ModelParametersErrorFunction>(character);
            result->setWeight(weight);

            if (targetParams.has_value() || weights.has_value()) {
              const auto nParams = character.parameterTransform.numAllModelParameters();
              result->setTargetParameters(
                  arrayToVec(targetParams, nParams, 0.0f, "target_parameters"),
                  arrayToVec(weights, nParams, 1.0f, "weights"));
            }
            return result;
          }),
          R"(Initialize a ModelParametersErrorFunction.

:param character: The character to use.
:param target_parameters: The target model parameters to achieve.
:param weights: Optional weights for each parameter, defaulting to 1 if not provided.
:param weight: The weight applied to the error function.)",

          py::keep_alive<1, 2>(),
          py::arg("character"),
          py::arg("target_parameters") = std::optional<py::array_t<float>>{},
          py::kw_only(),
          py::arg("weights") = std::optional<py::array_t<float>>{},
          py::arg("weight") = 1.0f)
      .def(
          "set_target_parameters",
          [](mm::ModelParametersErrorFunction& self,
             const py::array_t<float>& targetParams,
             const std::optional<py::array_t<float>>& weights) {
            const auto nParams = self.getParameterTransform().numAllModelParameters();
            self.setTargetParameters(
                arrayToVec(targetParams, nParams, "target_parameters"),
                arrayToVec(weights, nParams, 1.0f, "weights"));
          },
          R"(Sets the target model parameters and optional weights.

:param target_parameters: The target model parameters to achieve.
:param weights: Optional weights for each parameter, defaulting to 1 if not provided.)",
          py::arg("target_parameters"),
          py::arg("weights") = std::optional<py::array_t<float>>{});

  py::class_<mm::PositionDataT<float>>(m, "PositionData")
      .def(
          "__repr__",
          [](const mm::PositionDataT<float>& self) {
            return fmt::format(
                "PositionData(parent={}, weight={}, offset=[{:.3f}, {:.3f}, {:.3f}], target=[{:.3f}, {:.3f}, {:.3f}])",
                self.parent,
                self.weight,
                self.offset.x(),
                self.offset.y(),
                self.offset.z(),
                self.target.x(),
                self.target.y(),
                self.target.z());
          })
      .def_readonly("parent", &mm::PositionDataT<float>::parent, "The parent joint index")
      .def_readonly("weight", &mm::PositionDataT<float>::weight, "The weight of the constraint")
      .def_readonly("offset", &mm::PositionDataT<float>::offset, "The offset in parent space")
      .def_readonly("target", &mm::PositionDataT<float>::target, "The target position");

  py::class_<
      mm::PositionErrorFunction,
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::PositionErrorFunction>>(
      m,
      "PositionErrorFunction",
      R"(A skeleton error function that tries to move locators to a target position.

Uses a generalized loss function that support various forms of losses such as L1, L2, and Cauchy.  Details can be found in the paper 'A General and Adaptive Robust Loss Function' by Jonathan T. Barron.
      )")
      .def(
          "__repr__",
          [](const mm::PositionErrorFunction& self) {
            return fmt::format(
                "PositionErrorFunction(weight={}, num_constraints={})",
                self.getWeight(),
                self.numConstraints());
          })
      .def(
          py::init<>(
              [](const mm::Character& character, float lossAlpha, float lossC, float weight)
                  -> std::shared_ptr<mm::PositionErrorFunction> {
                validateWeight(weight, "weight");
                auto result =
                    std::make_shared<mm::PositionErrorFunction>(character, lossAlpha, lossC);
                result->setWeight(weight);
                return result;
              }),
          R"(Initialize a PositionErrorFunction.

              :param character: The character to use.
              :param alpha: P-norm to use; 2 is Euclidean distance, 1 is L1 norm, 0 is Cauchy norm.
              :param c: c parameter in the generalized loss function, this is roughly equivalent to normalizing by the standard deviation in a Gaussian distribution.
              :param weight: The weight applied to the error function.)",
          py::keep_alive<1, 2>(),
          py::arg("character"),
          py::kw_only(),
          py::arg("alpha") = 2.0f,
          py::arg("c") = 1.0f,
          py::arg("weight") = 1.0f)
      .def(
          "add_constraint",
          [](mm::PositionErrorFunction& errf,
             int parent,
             const Eigen::Vector3f& target,
             const std::optional<Eigen::Vector3f>& offset,
             float weight,
             const std::string& name) {
            validateJointIndex(parent, "parent", errf.getSkeleton());
            validateWeight(weight, "weight");
            errf.addConstraint(
                mm::PositionData(
                    offset.value_or(Eigen::Vector3f::Zero()), target, parent, weight, name));
          },
          R"(Adds a new constraint to the error function.

:param parent: The index of the joint that is the parent of the locator.
:param offset: The offset of the locator in the parent joint local space.
:param target: The target position of the locator.
:param weight: The weight of the constraint.
:param name: The name of the constraint (for debugging).
        )",
          py::arg("parent"),
          py::kw_only(),
          py::arg("target"),
          py::arg("offset") = std::nullopt,
          py::arg("weight") = 1.0f,
          py::arg("name") = std::string{})
      .def_property_readonly(
          "constraints",
          [](const mm::PositionErrorFunction& self) { return self.getConstraints(); },
          "Returns the list of position constraints.")
      .def(
          "clear_constraints",
          [](mm::PositionErrorFunction& self) { self.clearConstraints(); },
          "Clears all constraints from the error function.")
      .def(
          "add_constraints",
          [](mm::PositionErrorFunction& errf,
             const py::array_t<int>& parent,
             const py::array_t<float, 2>& target,
             const std::optional<py::array_t<float, 2>>& offset,
             const std::optional<py::array_t<float>>& weight,
             const std::optional<std::vector<std::string>>& name) {
            ArrayShapeValidator validator;
            const int nConsIdx = -1;
            validator.validate(parent, "parent", {nConsIdx}, {"n_cons"});
            validateJointIndex(parent, "parent", errf.getSkeleton());
            validator.validate(offset, "offset", {nConsIdx, 3}, {"n_cons", "xyz"});
            validator.validate(target, "target", {nConsIdx, 3}, {"n_cons", "xyz"});
            validator.validate(weight, "weight", {nConsIdx}, {"n_cons"});
            validateWeights(weight, "weight");

            if (name.has_value() && name->size() != parent.shape(0)) {
              throw std::runtime_error(
                  fmt::format(
                      "Invalid names; expected {} names but got {}",
                      parent.shape(0),
                      name->size()));
            }

            auto parent_acc = parent.unchecked<1>();
            auto offset_acc =
                offset.has_value() ? std::make_optional(offset->unchecked<2>()) : std::nullopt;
            auto target_acc = target.unchecked<2>();
            auto weight_acc =
                weight.has_value() ? std::make_optional(weight->unchecked<1>()) : std::nullopt;

            py::gil_scoped_release release;

            for (size_t iCons = 0; iCons < parent_acc.shape(0); ++iCons) {
              errf.addConstraint(
                  mm::PositionData(
                      offset_acc.has_value() ? Eigen::Vector3f(
                                                   (*offset_acc)(iCons, 0),
                                                   (*offset_acc)(iCons, 1),
                                                   (*offset_acc)(iCons, 2))
                                             : Eigen::Vector3f::Zero(),
                      Eigen::Vector3f(
                          target_acc(iCons, 0), target_acc(iCons, 1), target_acc(iCons, 2)),
                      parent_acc(iCons),
                      weight_acc.has_value() ? (*weight_acc)(iCons) : 1.0f,
                      name.has_value() ? name->at(iCons) : std::string{}));
            }
          },
          R"(Adds a new constraint to the error function.

:param parent: The index of the joints that are the parents of the locator.
:param offset: The offset of the locators from the parent joint.
:param target: The target positions of the locator.
:param weight: The weight of the constraint.
:param names: The names of the constraints (for debugging).)",
          py::arg("parent"),
          py::kw_only(),
          py::arg("target"),
          py::arg("offset") = std::nullopt,
          py::arg("weight") = std::nullopt,
          py::arg("name") = std::optional<std::vector<std::string>>{});

  py::enum_<mm::RotationErrorType>(m, "RotationErrorType")
      .value(
          "RotationMatrixDifference",
          mm::RotationErrorType::RotationMatrixDifference,
          R"(Frobenius norm of rotation matrix difference: ||R1 - R2||_F^2.

This is the default method. It computes the squared Frobenius norm of the
difference between two rotation matrices. While not a geodesic distance,
it has smooth derivatives everywhere and is computationally efficient.)")
      .value(
          "QuaternionLogMap",
          mm::RotationErrorType::QuaternionLogMap,
          R"(Logarithmic map of relative rotation: ||log(R1^{-1} * R2)||^2.

This method computes the squared norm of the logarithmic map of the relative
rotation quaternion. This gives the squared geodesic distance on SO(3), which
has a clear geometric interpretation. It uses numerically robust logmap
computation with Taylor series for small angles.)");

  py::class_<
      mm::StateErrorFunction,
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::StateErrorFunction>>(m, "StateErrorFunction")
      .def(
          "__repr__",
          [](const mm::StateErrorFunction& self) {
            return fmt::format("StateErrorFunction(weight={})", self.getWeight());
          })
      .def(
          py::init<>([](const mm::Character& character,
                        float weight,
                        float positionWeight,
                        float rotationWeight,
                        const std::optional<py::array_t<float>>& jointPositionWeights,
                        const std::optional<py::array_t<float>>& jointRotationWeights,
                        mm::RotationErrorType rotationErrorType) {
            validateWeight(weight, "weight");
            validateWeight(positionWeight, "position_weight");
            validateWeight(rotationWeight, "rotation_weight");
            validateWeights(jointPositionWeights, "joint_position_weights");
            validateWeights(jointRotationWeights, "joint_rotation_weights");

            auto result = std::make_shared<mm::StateErrorFunction>(character, rotationErrorType);
            result->setWeight(weight);
            result->setWeights(positionWeight, rotationWeight);

            if (jointPositionWeights.has_value() || jointRotationWeights.has_value()) {
              result->setTargetWeights(
                  getJointWeights(
                      jointPositionWeights, character.skeleton, "joint_position_weights"),
                  getJointWeights(
                      jointRotationWeights, character.skeleton, "joint_rotation_weights"));
            }

            return result;
          }),
          R"(Constructs a StateErrorFunction for a given character.

:param character: The character for which the error function is defined.
:param weight: The weight for the error function. Defaults to 1.0.
:param position_weight: The weight for position errors. Defaults to 1.0.
:param rotation_weight: The weight for rotation errors. Defaults to 1.0.
:param joint_position_weights: Optional numpy array of per-joint position weights.
                               Must match the number of joints in the skeleton.
:param joint_rotation_weights: Optional numpy array of per-joint rotation weights.
                               Must match the number of joints in the skeleton.
:param rotation_error_type: The type of rotation error to use. Defaults to RotationMatrixDifference.
                            See :class:`RotationErrorType` for options.)",
          py::arg("character"),
          py::kw_only(),
          py::arg("weight") = 1.0f,
          py::arg("position_weight") = 1.0f,
          py::arg("rotation_weight") = 1.0f,
          py::arg("joint_position_weights") = std::optional<py::array_t<float>>{},
          py::arg("joint_rotation_weights") = std::optional<py::array_t<float>>{},
          py::arg("rotation_error_type") = mm::RotationErrorType::RotationMatrixDifference)
      .def(
          "set_target_state",
          [](mm::StateErrorFunction& self, const py::array_t<float>& targetStateArray) {
            if (targetStateArray.ndim() != 2 || targetStateArray.shape(1) != 8) {
              throw std::runtime_error("Expected target state array of shape (njoints, 8)");
            }

            const auto targetTransforms = toTransformList(targetStateArray);
            if (targetTransforms.size() != self.getSkeleton().joints.size()) {
              throw std::runtime_error(
                  fmt::format(
                      "Expected target state array of shape (njoints, 8) where nJoints={} but got {}.",
                      self.getSkeleton().joints.size(),
                      getDimStr(targetStateArray)));
            }

            self.setTargetState(targetTransforms);
          },
          R"(Sets the target skeleton state for the error function.

:param target_state: A numpy array of shape (njoints, 8) where each row contains
                     3D position, 4D quaternion (rotation), and 1D scale for each joint.)",
          py::arg("target_state"))
      .def_property(
          "joint_position_weights",
          [](const mm::StateErrorFunction& self) { return self.getPositionWeights(); },
          [](mm::StateErrorFunction& self, const py::array_t<float>& weights) {
            validateWeights(weights, "weights");
            self.setPositionTargetWeights(
                getJointWeights(weights, self.getSkeleton(), "joint_position_weights"));
          },
          R"(The per-joint position weights.)")
      .def_property(
          "joint_rotation_weights",
          [](const mm::StateErrorFunction& self) { return self.getRotationWeights(); },
          [](mm::StateErrorFunction& self, const py::array_t<float>& weights) {
            validateWeights(weights, "weights");
            self.setRotationTargetWeights(
                getJointWeights(weights, self.getSkeleton(), "joint_position_weights"));
          },
          R"(The per-joint rotation weights.)");

  py::enum_<mm::VertexConstraintType>(m, "VertexConstraintType")
      .value("Position", mm::VertexConstraintType::Position, "Target the vertex position")
      .value(
          "Plane",
          mm::VertexConstraintType::Plane,
          "Point-to-plane distance using the target normal")
      .value(
          "Normal",
          mm::VertexConstraintType::Normal,
          "Point-to-plane distance using the source (body) normal")
      .value(
          "SymmetricNormal",
          mm::VertexConstraintType::SymmetricNormal,
          "Point-to-plane using a 50/50 mix of source and target normal");

  py::class_<mm::VertexConstraint>(m, "VertexConstraint")
      .def(
          "__repr__",
          [](const mm::VertexConstraint& self) {
            return fmt::format(
                "VertexConstraint(vertex_index={}, weight={}, target_position=[{:.3f}, {:.3f}, {:.3f}])",
                self.vertexIndex,
                self.weight,
                self.targetPosition.x(),
                self.targetPosition.y(),
                self.targetPosition.z());
          })
      .def_readonly(
          "vertex_index",
          &mm::VertexConstraint::vertexIndex,
          "The index of the vertex to constrain.")
      .def_readonly("weight", &mm::VertexConstraint::weight, "The weight of the constraint.")
      .def_readonly(
          "target_position",
          &mm::VertexConstraint::targetPosition,
          "The target position for the vertex.")
      .def_readonly(
          "target_normal",
          &mm::VertexConstraint::targetNormal,
          "The target normal for the vertex.");

  py::class_<
      mm::VertexErrorFunction,
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::VertexErrorFunction>>(m, "VertexErrorFunction")
      .def(
          "__repr__",
          [](const mm::VertexErrorFunction& self) {
            return fmt::format(
                "VertexErrorFunction(weight={}, num_constraints={})",
                self.getWeight(),
                self.numConstraints());
          })
      .def(
          py::init<>([](const mm::Character& character,
                        mm::VertexConstraintType constraintType,
                        float weight,
                        size_t maxThreads) {
            validateWeight(weight, "weight");
            auto result =
                std::make_shared<mm::VertexErrorFunction>(character, constraintType, maxThreads);
            result->setWeight(weight);
            return result;
          }),
          R"(A skeleton error function that penalizes vertex distance to a target point.

  :param character: The character to use.
  :param constraint_type: The type of vertex constraint to apply.
  :param max_threads: The maximum number of threads to use for computation.)",
          py::arg("character"),
          py::arg("constraint_type") = mm::VertexConstraintType::Position,
          py::arg("weight") = 1.0f,
          py::arg("max_threads") = 0)
      .def(
          "add_constraint",
          [](mm::VertexErrorFunction& self,
             int vertexIndex,
             float weight,
             const Eigen::Vector3f& targetPosition,
             const Eigen::Vector3f& targetNormal) {
            validateVertexIndex(vertexIndex, "vertex_index", self.getCharacter());
            validateWeight(weight, "weight");
            self.addConstraint(vertexIndex, weight, targetPosition, targetNormal);
          },
          R"(Adds a vertex constraint to the error function.

  :param vertex_index: The index of the vertex to constrain.
  :param weight: The weight of the constraint.
  :param target_position: The target position for the vertex.
  :param target_normal: The target normal for the vertex.)",
          py::arg("vertex_index"),
          py::arg("weight"),
          py::arg("target_position"),
          py::arg("target_normal"))
      .def(
          "add_constraints",
          [](mm::VertexErrorFunction& errf,
             const py::array_t<int>& vertexIndex,
             const py::array_t<float, 2>& targetPosition,
             const py::array_t<float, 2>& targetNormal,
             const py::array_t<float>& weight) {
            ArrayShapeValidator validator;
            const int nConsIdx = -1;
            validator.validate(vertexIndex, "vertex_index", {nConsIdx}, {"n_cons"});
            validateVertexIndex(vertexIndex, "vertex_index", errf.getCharacter());
            validator.validate(targetPosition, "target_position", {nConsIdx, 3}, {"n_cons", "xyz"});
            validator.validate(targetNormal, "target_normal", {nConsIdx, 3}, {"n_cons", "xyz"});
            validator.validate(weight, "weight", {nConsIdx}, {"n_cons"});

            auto vertex_index_acc = vertexIndex.unchecked<1>();
            auto target_position_acc = targetPosition.unchecked<2>();
            auto target_normal_acc = targetNormal.unchecked<2>();
            auto weight_acc = weight.unchecked<1>();

            validateWeights(weight, "weight");

            py::gil_scoped_release release;

            for (size_t iCons = 0; iCons < vertex_index_acc.shape(0); ++iCons) {
              errf.addConstraint(
                  vertex_index_acc(iCons),
                  weight_acc(iCons),
                  Eigen::Vector3f(
                      target_position_acc(iCons, 0),
                      target_position_acc(iCons, 1),
                      target_position_acc(iCons, 2)),
                  Eigen::Vector3f(
                      target_normal_acc(iCons, 0),
                      target_normal_acc(iCons, 1),
                      target_normal_acc(iCons, 2)));
            }
          },
          R"(Adds multiple vertex constraints to the error function.

  :param vertex_index: The indices of the vertices to constrain.
  :param target_position: The target positions for the vertices.
  :param target_normal: The target normals for the vertices.
  :param weight: The weights of the constraints.)",
          py::arg("vertex_index"),
          py::arg("target_position"),
          py::arg("target_normal"),
          py::arg("weight"))
      .def(
          "clear_constraints",
          &mm::VertexErrorFunction::clearConstraints,
          "Clears all vertex constraints from the error function.")
      .def_property_readonly(
          "constraints",
          [](const mm::VertexErrorFunction& self) { return self.getConstraints(); },
          "Returns the list of vertex constraints.");

  py::class_<mm::PointTriangleVertexConstraint>(m, "PointTriangleVertexConstraint")
      .def(
          "__repr__",
          [](const mm::PointTriangleVertexConstraint& self) {
            return fmt::format(
                "PointTriangleVertexConstraint(src_vertex_index={}, depth={:.3f}, tgt_triangle_indices=[{} {} {}], tgt_bary_coords=[{:.3f}, {:.3f}, {:.3f}], weight={:.3f})",
                self.srcVertexIndex,
                self.depth,
                self.tgtTriangleIndices.x(),
                self.tgtTriangleIndices.y(),
                self.tgtTriangleIndices.z(),
                self.tgtTriangleBaryCoords.x(),
                self.tgtTriangleBaryCoords.y(),
                self.tgtTriangleBaryCoords.z(),
                self.weight);
          })
      .def_readonly(
          "src_vertex_index",
          &mm::PointTriangleVertexConstraint::srcVertexIndex,
          "The index of the vertex to constrain.")
      .def_readonly(
          "depth", &mm::PointTriangleVertexConstraint::depth, "The depth of the constraint.")
      .def_readonly(
          "weight", &mm::PointTriangleVertexConstraint::weight, "The weight of the constraint.")
      .def_readonly(
          "tgt_triangle_indices",
          &mm::PointTriangleVertexConstraint::tgtTriangleIndices,
          "The target triangle indices.")
      .def_readonly(
          "tgt_bary_coords",
          &mm::PointTriangleVertexConstraint::tgtTriangleBaryCoords,
          "The target barycentric coordinates.");

  py::class_<
      mm::PointTriangleVertexErrorFunction,
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::PointTriangleVertexErrorFunction>>(m, "PointTriangleVertexErrorFunction")
      .def(
          "__repr__",
          [](const mm::PointTriangleVertexErrorFunction& self) {
            return fmt::format("PointTriangleVertexErrorFunction(weight={})", self.getWeight());
          })
      .def(
          py::init<>(
              [](const mm::Character& character,
                 mm::VertexConstraintType constraintType,
                 float weight) -> std::shared_ptr<mm::PointTriangleVertexErrorFunction> {
                if (!character.mesh || !character.skinWeights) {
                  throw std::runtime_error("No mesh or skin weights found");
                }

                auto result = std::make_shared<mm::PointTriangleVertexErrorFunction>(
                    character, constraintType);
                result->setWeight(weight);
                return result;
              }),
          R"(A skeleton error function that penalizes the distance between a vertex and the location within a triangle.)",
          py::keep_alive<1, 2>(),
          py::arg("character"),
          py::arg("type") = mm::VertexConstraintType::Position,
          py::arg("weight") = 1.0f)
      .def(
          "add_constraints",
          [](mm::PointTriangleVertexErrorFunction& errf,
             const py::array_t<int>& srcVertexIndex,
             const py::array_t<int, 3>& tgtTriangleIndex,
             const py::array_t<float, 3>& tgtTriangleBaryCoord,
             const py::array_t<float>& depth,
             const py::array_t<float>& weight) {
            ArrayShapeValidator validator;
            const int nConsIdx = -1;
            validator.validate(srcVertexIndex, "source_vertex_index", {nConsIdx}, {"n_cons"});
            validateVertexIndex(srcVertexIndex, "source_vertex_index", errf.getCharacter());
            validator.validate(
                tgtTriangleIndex, "target_triangle_index", {nConsIdx, 3}, {"n_cons", "idx"});
            validateVertexIndex(tgtTriangleIndex, "target_triangle_index", errf.getCharacter());
            validator.validate(
                tgtTriangleBaryCoord,
                "target_triangle_bary_coord",
                {nConsIdx, 3},
                {"n_cons", "xyz"});
            validator.validate(depth, "depth", {nConsIdx}, {"n_cons"});
            validator.validate(weight, "weight", {nConsIdx}, {"n_cons"});

            auto src_vertex_index_acc = srcVertexIndex.unchecked<1>();
            auto tgt_triangle_index_acc = tgtTriangleIndex.unchecked<2>();
            auto tgt_triangle_bary_coord_acc = tgtTriangleBaryCoord.unchecked<2>();
            auto depth_acc = depth.unchecked<1>();
            auto weight_acc = weight.unchecked<1>();

            py::gil_scoped_release release;

            for (size_t iCons = 0; iCons < src_vertex_index_acc.shape(0); ++iCons) {
              errf.addConstraint(
                  src_vertex_index_acc(iCons),
                  Eigen::Vector3i(
                      tgt_triangle_index_acc(iCons, 0),
                      tgt_triangle_index_acc(iCons, 1),
                      tgt_triangle_index_acc(iCons, 2)),
                  Eigen::Vector3f(
                      tgt_triangle_bary_coord_acc(iCons, 0),
                      tgt_triangle_bary_coord_acc(iCons, 1),
                      tgt_triangle_bary_coord_acc(iCons, 2)),
                  depth_acc(iCons),
                  weight_acc(iCons));
            }
          },
          R"(Adds a new constraint to the error function.

  :param parents: The indices of the joints that are the parents of the locators.
  :param offsets: The offsets of the locators from the parent joints.
  :param targets: The target positions of the locators.
  :param weights: The weights of the constraints.
  :param names: The names of the constraints (for debugging).)",
          py::arg("source_vertex_index"),
          py::arg("target_triangle_index"),
          py::arg("target_triangle_bary_coord"),
          py::arg("depth"),
          py::arg("weight"))
      .def_property_readonly(
          "constraints",
          [](const mm::PointTriangleVertexErrorFunction& self) { return self.getConstraints(); },
          "Returns the list of point triangle constraints.")
      .def(
          "clear_constraints",
          &mm::PointTriangleVertexErrorFunction::clearConstraints,
          "Clears all vertex constraints from the error function.");

  py::class_<
      mm::PosePriorErrorFunction,
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::PosePriorErrorFunction>>(m, "PosePriorErrorFunction")
      .def(
          "__repr__",
          [](const mm::PosePriorErrorFunction& self) {
            return fmt::format("PosePriorErrorFunction(weight={})", self.getWeight());
          })
      .def(
          py::init<>([](const mm::Character& character,
                        std::shared_ptr<const mm::Mppca> posePrior,
                        float weight) {
            validateWeight(weight, "weight");
            auto result = std::make_shared<mm::PosePriorErrorFunction>(character, posePrior);
            result->setWeight(weight);
            return result;
          }),
          R"(A skeleton error function that uses a mixture of PCA models pose prior to penalize deviations from expected poses.

  :param character: The character to use.
  :param pose_prior: The pose prior model to use.
  :param weight: The weight applied to the error function.)",
          py::arg("character"),
          py::arg("pose_prior"),
          py::kw_only(),
          py::arg("weight") = 1.0f)
      .def(
          "set_pose_prior",
          &mm::PosePriorErrorFunction::setPosePrior,
          R"(Sets the pose prior model for the error function.

  :param pose_prior: The pose prior model to use.)",
          py::arg("pose_prior"))
      .def(
          "log_probability",
          [](const mm::PosePriorErrorFunction& self, const py::array_t<float>& modelParameters) {
            return self.logProbability(
                toModelParameters(modelParameters, self.getParameterTransform()));
          },
          R"(Computes the log probability of the given model parameters under the pose prior.

  :param model_parameters: The model parameters to evaluate.
  :return: The log probability value.)",
          py::arg("model_parameters"));

  py::class_<mm::OrientationData>(m, "OrientationData")
      .def(
          "__repr__",
          [](const mm::OrientationData& self) {
            return fmt::format(
                "OrientationData(parent={}, weight={}, offset=[{:.3f}, {:.3f}, {:.3f}, {:.3f}], target=[{:.3f}, {:.3f}, {:.3f}, {:.3f}])",
                self.parent,
                self.weight,
                self.offset.x(),
                self.offset.y(),
                self.offset.z(),
                self.offset.w(),
                self.target.x(),
                self.target.y(),
                self.target.z(),
                self.target.w());
          })
      .def_readonly("parent", &mm::OrientationData::parent, "The parent joint index")
      .def_readonly("weight", &mm::OrientationData::weight, "The weight of the constraint")
      .def_readonly("offset", &mm::OrientationData::offset, "The offset in parent space")
      .def_readonly("target", &mm::OrientationData::target, "The target orientation");

  py::class_<
      mm::OrientationErrorFunctionT<float>,
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::OrientationErrorFunctionT<float>>>(
      m,
      "OrientationErrorFunction",
      R"(A skeleton error function that minimizes the F-norm of the element-wise difference of a 3x3
rotation matrix to a target rotation.)")
      .def(
          "__repr__",
          [](const mm::OrientationErrorFunctionT<float>& self) {
            return fmt::format(
                "OrientationErrorFunction(weight={}, num_constraints={})",
                self.getWeight(),
                self.numConstraints());
          })
      .def(
          py::init<>(
              [](const mm::Character& character, float lossAlpha, float lossC, float weight)
                  -> std::shared_ptr<mm::OrientationErrorFunctionT<float>> {
                validateWeight(weight, "weight");
                auto result = std::make_shared<mm::OrientationErrorFunctionT<float>>(
                    character, lossAlpha, lossC);
                result->setWeight(weight);
                return result;
              }),
          R"(Initialize an OrientationErrorFunction.

              :param character: The character to use.
              :param alpha: P-norm to use; 2 is Euclidean distance, 1 is L1 norm, 0 is Cauchy norm.
              :param c: c parameter in the generalized loss function, this is roughly equivalent to normalizing by the standard deviation in a Gaussian distribution.
              :param weight: The weight applied to the error function.)",
          py::keep_alive<1, 2>(),
          py::arg("character"),
          py::kw_only(),
          py::arg("alpha") = 2.0f,
          py::arg("c") = 1.0f,
          py::arg("weight") = 1.0f)
      .def(
          "add_constraint",
          [](mm::OrientationErrorFunctionT<float>& self,
             const Eigen::Vector4f& target,
             int parent,
             const std::optional<Eigen::Vector4f>& offset,
             float weight,
             const std::string& name) {
            validateJointIndex(parent, "parent", self.getSkeleton());
            validateWeight(weight, "weight");
            self.addConstraint(
                mm::OrientationDataT<float>(
                    offset.has_value() ? toQuaternion(*offset) : Eigen::Quaternionf::Identity(),
                    toQuaternion(target),
                    parent,
                    weight,
                    name));
          },
          R"(Adds an orientation constraint to the error function.

      :param offset: The rotation offset in the parent space.
      :param target: The target rotation in global space.
      :param parent: The index of the parent joint.
      :param weight: The weight of the constraint.
      :param name: The name of the constraint (for debugging).)",
          py::arg("target"),
          py::arg("parent"),
          py::arg("offset") = std::nullopt,
          py::arg("weight") = 1.0f,
          py::arg("name") = std::string{})
      .def(
          "clear_constraints",
          [](mm::OrientationErrorFunctionT<float>& self) { self.clearConstraints(); },
          R"(Clears all constraints.)")
      .def_property_readonly(
          "constraints",
          [](const mm::OrientationErrorFunctionT<float>& self) { return self.getConstraints(); },
          "Returns the list of orientation constraints.")
      .def(
          "add_constraints",
          [](mm::OrientationErrorFunctionT<float>& self,
             const py::array_t<float>& target,
             const py::array_t<int>& parent,
             const std::optional<py::array_t<float>>& offset,
             const std::optional<py::array_t<float>>& weight,
             const std::optional<std::vector<std::string>>& name) {
            ArrayShapeValidator validator;
            const int nConsIdx = -1;
            validator.validate(offset, "offset", {nConsIdx, 4}, {"n_cons", "xyzw"});
            validator.validate(target, "target", {nConsIdx, 4}, {"n_cons", "xyzw"});
            validator.validate(parent, "parent", {nConsIdx}, {"n_cons"});
            validateJointIndex(parent, "parent", self.getSkeleton());
            validator.validate(weight, "weight", {nConsIdx}, {"n_cons"});

            if (name.has_value() && name->size() != parent.shape(0)) {
              throw std::runtime_error(
                  fmt::format(
                      "Invalid names; expected {} names but got {}",
                      parent.shape(0),
                      name->size()));
            }

            auto offsetAcc =
                offset.has_value() ? std::make_optional(offset->unchecked<2>()) : std::nullopt;
            auto targetAcc = target.unchecked<2>();
            auto parentAcc = parent.unchecked<1>();
            auto weightAcc =
                weight.has_value() ? std::make_optional(weight->unchecked<1>()) : std::nullopt;

            py::gil_scoped_release release;

            for (py::ssize_t i = 0; i < parent.shape(0); ++i) {
              validateJointIndex(parentAcc(i), "parent", self.getSkeleton());
              self.addConstraint(
                  mm::OrientationDataT<float>(
                      offsetAcc.has_value() ? Eigen::Quaternionf(
                                                  (*offsetAcc)(i, 3),
                                                  (*offsetAcc)(i, 0),
                                                  (*offsetAcc)(i, 1),
                                                  (*offsetAcc)(i, 2))
                                            : Eigen::Quaternionf::Identity(),
                      Eigen::Quaternionf(
                          targetAcc(i, 3), targetAcc(i, 0), targetAcc(i, 1), targetAcc(i, 2)),
                      parentAcc(i),
                      weightAcc.has_value() ? (*weightAcc)(i) : 1.0f,
                      name.has_value() ? name->at(i) : std::string{}));
            }
          },
          R"(Adds multiple orientation constraints to the error function.

      :param target: A numpy array of shape (n, 4) for the (x, y, z, w) quaternion target rotations in global space.
      :param parent: A numpy array of size n for the indices of the parent joints.
      :param offset: A numpy array of shape (n, 4) for the (x, y, z, w) quaternion rotation offsets in the parent space.
      :param weight: A numpy array of size n for the weights of the constraints.
      :param name: An optional list of names for the constraints (for debugging).)",
          py::arg("target"),
          py::arg("parent"),
          py::arg("offset") = std::nullopt,
          py::arg("weight") = std::nullopt,
          py::arg("name") = std::nullopt);

  py::class_<
      mm::CollisionErrorFunction,
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::CollisionErrorFunction>>(m, "CollisionErrorFunction")
      .def(
          "__repr__",
          [](const mm::CollisionErrorFunction& self) {
            return fmt::format(
                "CollisionErrorFunction(weight={}, collision_pairs={})",
                self.getWeight(),
                self.getCollisionPairs().size());
          })
      .def(
          py::init<>([](const mm::Character& character, float weight) {
            validateWeight(weight, "weight");
            auto result = std::make_shared<mm::CollisionErrorFunction>(character);
            result->setWeight(weight);
            return result;
          }),
          R"(A skeleton error function that penalizes collisions between character parts.

        :param character: The character to use.
        :param weight: The weight applied to the error function.)",
          py::arg("character"),
          py::kw_only(),
          py::arg("weight") = 1.0f)
      .def(
          "get_collision_pairs",
          [](const mm::CollisionErrorFunction& self) {
            const auto& collisionPairs = self.getCollisionPairs();
            py::array_t<float> result(
                std::vector<py::ssize_t>{(py::ssize_t)collisionPairs.size(), 2});
            auto resultAcc = result.mutable_unchecked<2>();
            for (size_t i = 0; i < collisionPairs.size(); ++i) {
              resultAcc(i, 0) = collisionPairs[i].x();
              resultAcc(i, 1) = collisionPairs[i].y();
            }
            return result;
          },
          R"(Returns the pairs of colliding elements.)");

  // Vertex SDF error function
  py::class_<
      mm::VertexSDFErrorFunction,
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::VertexSDFErrorFunction>>(m, "VertexSDFErrorFunction")
      .def(
          "__repr__",
          [](const mm::VertexSDFErrorFunction& self) {
            return fmt::format(
                "VertexSDFErrorFunction(weight={}, num_constraints={})",
                self.getWeight(),
                self.numConstraints());
          })
      .def(
          py::init<>([](const mm::Character& character,
                        const mm::SDFColliderT<float>& sdfCollider,
                        float weight) {
            validateWeight(weight, "weight");
            auto result = std::make_shared<mm::VertexSDFErrorFunction>(character, sdfCollider);
            result->setWeight(weight);
            return result;
          }),
          R"(A skeleton error function that penalizes deviation of mesh vertex SDF distance from a target value.

Unlike CollisionErrorFunction (which penalizes only penetration), this targets an arbitrary signed
distance value against a single SDF, enabling use cases like fitting a mesh to a target isosurface.

:param character: The character to use.
:param sdf_collider: The SDF collider to test against (with optional parent joint and offset).
:param weight: The weight applied to the error function.)",
          py::arg("character"),
          py::arg("sdf_collider"),
          py::kw_only(),
          py::arg("weight") = 1.0f)
      .def(
          "add_constraint",
          [](mm::VertexSDFErrorFunction& self,
             int vertexIndex,
             float targetDistance,
             float weight) {
            validateWeight(weight, "weight");
            mm::VertexSDFConstraintT<float> constraint;
            constraint.vertexIndex = vertexIndex;
            constraint.targetDistance = targetDistance;
            constraint.weight = weight;
            self.addConstraint(constraint);
          },
          R"(Add a constraint specifying a target signed distance for a vertex.

:param vertex_index: The mesh vertex index.
:param target_distance: The target signed distance value (0 = surface, positive = outside, negative = inside).
:param weight: Per-constraint weight (default: 1.0).)",
          py::arg("vertex_index"),
          py::arg("target_distance"),
          py::kw_only(),
          py::arg("weight") = 1.0f)
      .def(
          "clear_constraints",
          &mm::VertexSDFErrorFunction::clearConstraints,
          R"(Clear all constraints.)")
      .def_property_readonly(
          "num_constraints",
          &mm::VertexSDFErrorFunction::numConstraints,
          R"(The number of active constraints.)");

  // Call sub-registration functions
  addAimAxisErrorFunctions(m);
  addProjectionErrorFunctions(m);
  addDistanceErrorFunctions(m);
}

} // namespace pymomentum
