/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/character.h>
#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character_sequence_solver/model_parameters_sequence_error_function.h>
#include <momentum/character_sequence_solver/sequence_solver.h>
#include <momentum/character_sequence_solver/state_sequence_error_function.h>
#include <momentum/character_sequence_solver/vertex_sequence_error_function.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Core>

#include <pymomentum/solver2/solver2_sequence_error_functions.h>
#include <pymomentum/solver2/solver2_utility.h>

namespace py = pybind11;
namespace mm = momentum;

namespace pymomentum {

void addSequenceErrorFunctions(pybind11::module_& m) {
  py::class_<
      mm::StateSequenceErrorFunction,
      mm::SequenceErrorFunction,
      std::shared_ptr<mm::StateSequenceErrorFunction>>(
      m, "StateSequenceErrorFunction")
      .def(
          py::init<>(
              [](const mm::Character& character,
                 float weight,
                 float positionWeight,
                 float rotationWeight,
                 const std::optional<py::array_t<float>>& jointPositionWeights,
                 const std::optional<py::array_t<float>>&
                     jointRotationWeights) {
                auto result =
                    std::make_shared<mm::StateSequenceErrorFunction>(character);
                result->setWeight(weight);
                result->setWeights(positionWeight, rotationWeight);

                const auto nJoints = character.skeleton.joints.size();
                if (jointPositionWeights.has_value()) {
                  result->setPositionTargetWeights(arrayToVec(
                      jointPositionWeights,
                      nJoints,
                      1.0f,
                      "joint_position_weights"));
                }

                if (jointRotationWeights.has_value()) {
                  result->setRotationTargetWeights(arrayToVec(
                      jointRotationWeights,
                      nJoints,
                      1.0f,
                      "joint_rotation_weights"));
                }

                return result;
              }),
          R"(A sequence error function that penalizes changes in global position and rotation.

:param character: The character to use.
:param weight: The weight of the error function.
:param position_weight: The weight of the position error.  Defaults to 1.0.
:param rotation_weight: The weight of the rotation error.  Defaults to 1.0.
:param joint_position_weights: The weights of the position error for each joint.  Defaults to all 1s.
:param joint_rotation_weights: The weights of the rotation error for each joint.  Defaults to all 1s.)",
          py::arg("character"),
          py::kw_only(),
          py::arg("weight") = 1.0f,
          py::arg("position_weight") = 1.0f,
          py::arg("rotation_weight") = 1.0f,
          py::arg("joint_position_weights") = std::optional<Eigen::VectorXf>{},
          py::arg("joint_rotation_weights") = std::optional<Eigen::VectorXf>{})
      .def(
          "set_target_state",
          [](mm::StateSequenceErrorFunction& self,
             const py::array_t<float>& targetStateArray) {
            if (targetStateArray.ndim() != 2 ||
                targetStateArray.shape(1) != 8) {
              throw std::runtime_error(
                  "Expected target state array of shape (njoints, 8)");
            }

            const auto targetTransforms = toTransformList(targetStateArray);
            if (targetTransforms.size() != self.getSkeleton().joints.size()) {
              throw std::runtime_error(
                  "Expected target state array of shape (njoints, 8) where nJoints=" +
                  std::to_string(self.getSkeleton().joints.size()) +
                  " but got " + getDimStr(targetStateArray) + ".");
            }

            self.setTargetState(targetTransforms);
          },
          R"(Sets the target skeleton state for the error function.

  :param target_state: A numpy array of shape (njoints, 8) where each row contains
                       3D position, 4D quaternion (rotation), and 1D scale for each joint.)",
          py::arg("target_state"));

  py::class_<
      mm::ModelParametersSequenceErrorFunction,
      mm::SequenceErrorFunction,
      std::shared_ptr<mm::ModelParametersSequenceErrorFunction>>(
      m, "ModelParametersSequenceErrorFunction")
      .def(
          py::init<>([](const mm::Character& character,
                        float weight,
                        const std::optional<Eigen::VectorXf>& targetWeights) {
            auto result =
                std::make_shared<mm::ModelParametersSequenceErrorFunction>(
                    character);
            result->setWeight(weight);

            if (targetWeights.has_value()) {
              if (targetWeights->size() !=
                  character.parameterTransform.numAllModelParameters()) {
                throw std::runtime_error(
                    "Invalid target weights; expected " +
                    std::to_string(
                        character.parameterTransform.numAllModelParameters()) +
                    " values but got " + std::to_string(targetWeights->size()));
              }
              result->setTargetWeights(targetWeights.value());
            }

            return result;
          }),
          R"(A sequence error function that penalizes changes in model parameters.

:param character: The character to use.
:param weight: The weight of the error function.
:param target_weights: The weights for each model parameter.  Defaults to all 1s.)",
          py::arg("character"),
          py::kw_only(),
          py::arg("weight") = 1.0f,
          py::arg("target_weights") = std::optional<Eigen::VectorXf>{});

  py::class_<mm::VertexVelocityConstraintT<float>>(
      m, "VertexVelocityConstraint")
      .def(
          "__repr__",
          [](const mm::VertexVelocityConstraintT<float>& self) {
            return fmt::format(
                "VertexVelocityConstraint(vertex_index={}, weight={}, target_velocity=[{:.3f}, {:.3f}, {:.3f}])",
                self.vertexIndex,
                self.weight,
                self.targetVelocity.x(),
                self.targetVelocity.y(),
                self.targetVelocity.z());
          })
      .def_readonly(
          "vertex_index",
          &mm::VertexVelocityConstraintT<float>::vertexIndex,
          "The index of the vertex to constrain.")
      .def_readonly(
          "weight",
          &mm::VertexVelocityConstraintT<float>::weight,
          "The weight of the constraint.")
      .def_readonly(
          "target_velocity",
          &mm::VertexVelocityConstraintT<float>::targetVelocity,
          "The target velocity for the vertex.");

  py::class_<
      mm::VertexSequenceErrorFunctionT<float>,
      mm::SequenceErrorFunction,
      std::shared_ptr<mm::VertexSequenceErrorFunctionT<float>>>(
      m, "VertexSequenceErrorFunction")
      .def(
          "__repr__",
          [](const mm::VertexSequenceErrorFunctionT<float>& self) {
            return fmt::format(
                "VertexSequenceErrorFunction(weight={}, num_constraints={})",
                self.getWeight(),
                self.numConstraints());
          })
      .def(
          py::init<>([](const mm::Character& character, float weight) {
            auto result =
                std::make_shared<mm::VertexSequenceErrorFunctionT<float>>(
                    character);
            result->setWeight(weight);
            return result;
          }),
          R"(A sequence error function that penalizes differences in vertex velocity between consecutive frames.

This function computes vertex velocities by taking the difference between consecutive frames
and penalizes deviations from target vertex velocities. It's useful for controlling vertex
motion in animations, cloth simulation, or any scenario where smooth vertex movement is desired.

:param character: The character to use.
:param weight: The weight applied to the error function.)",
          py::keep_alive<1, 2>(),
          py::arg("character"),
          py::kw_only(),
          py::arg("weight") = 1.0f)
      .def(
          "add_constraint",
          [](mm::VertexSequenceErrorFunctionT<float>& self,
             int vertexIndex,
             float weight,
             const Eigen::Vector3f& targetVelocity) {
            validateVertexIndex(
                vertexIndex, "vertex_index", self.getCharacter());
            self.addConstraint(vertexIndex, weight, targetVelocity);
          },
          R"(Adds a vertex velocity constraint to the error function.

:param vertex_index: The index of the vertex to constrain.
:param weight: The weight of the constraint.
:param target_velocity: The target velocity for the vertex (difference between consecutive frames).)",
          py::arg("vertex_index"),
          py::arg("weight"),
          py::arg("target_velocity"))
      .def(
          "add_constraints",
          [](mm::VertexSequenceErrorFunctionT<float>& self,
             const py::array_t<int>& vertexIndex,
             const py::array_t<float>& weight,
             const py::array_t<float>& targetVelocity) {
            ArrayShapeValidator validator;
            const int nConsIdx = -1;
            validator.validate(
                vertexIndex, "vertex_index", {nConsIdx}, {"n_cons"});
            validateVertexIndex(
                vertexIndex, "vertex_index", self.getCharacter());
            validator.validate(weight, "weight", {nConsIdx}, {"n_cons"});
            validator.validate(
                targetVelocity,
                "target_velocity",
                {nConsIdx, 3},
                {"n_cons", "xyz"});

            auto vertexIndexAcc = vertexIndex.unchecked<1>();
            auto weightAcc = weight.unchecked<1>();
            auto targetVelocityAcc = targetVelocity.unchecked<2>();

            py::gil_scoped_release release;

            for (py::ssize_t i = 0; i < vertexIndex.shape(0); ++i) {
              self.addConstraint(
                  vertexIndexAcc(i),
                  weightAcc(i),
                  Eigen::Vector3f(
                      targetVelocityAcc(i, 0),
                      targetVelocityAcc(i, 1),
                      targetVelocityAcc(i, 2)));
            }
          },
          R"(Adds multiple vertex velocity constraints to the error function.

:param vertex_index: A numpy array of size n for the indices of the vertices to constrain.
:param weight: A numpy array of size n for the weights of the constraints.
:param target_velocity: A numpy array of shape (n, 3) for the target velocities of the vertices.)",
          py::arg("vertex_index"),
          py::arg("weight"),
          py::arg("target_velocity"))
      .def(
          "clear_constraints",
          &mm::VertexSequenceErrorFunctionT<float>::clearConstraints,
          "Clears all vertex velocity constraints from the error function.")
      .def_property_readonly(
          "constraints",
          [](const mm::VertexSequenceErrorFunctionT<float>& self) {
            return self.getConstraints();
          },
          "Returns the list of vertex velocity constraints.")
      .def_property_readonly(
          "num_constraints",
          &mm::VertexSequenceErrorFunctionT<float>::numConstraints,
          "Returns the number of vertex velocity constraints.")
      .def_property_readonly(
          "character",
          &mm::VertexSequenceErrorFunctionT<float>::getCharacter,
          "Returns the character used by this error function.",
          py::return_value_policy::reference_internal)
      .def_property(
          "weight",
          &mm::VertexSequenceErrorFunctionT<float>::getWeight,
          &mm::VertexSequenceErrorFunctionT<float>::setWeight,
          "The weight applied to the error function.");
}

} // namespace pymomentum
