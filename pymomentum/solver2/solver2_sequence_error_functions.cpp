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
}

} // namespace pymomentum
