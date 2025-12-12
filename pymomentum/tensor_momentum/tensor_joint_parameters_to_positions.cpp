/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/tensor_momentum/tensor_joint_parameters_to_positions.h"

#include "pymomentum/python_utility/python_utility.h"
#include "pymomentum/tensor_momentum/tensor_momentum_utility.h"
#include "pymomentum/tensor_momentum/tensor_parameter_transform.h"
#include "pymomentum/tensor_utility/autograd_utility.h"
#include "pymomentum/tensor_utility/tensor_utility.h"

#include <momentum/character/character.h>
#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/common/exception.h>

#ifndef PYMOMENTUM_LIMITED_TORCH_API
#include <torch/csrc/jit/python/python_ivalue.h>
#endif
#include <Eigen/Core>

namespace pymomentum {

using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

namespace {

#ifndef PYMOMENTUM_LIMITED_TORCH_API

template <typename T>
void jointParametersToPositions(
    const momentum::Skeleton& skeleton,
    Eigen::Ref<const Eigen::VectorX<T>> jointParameters,
    Eigen::Ref<const Eigen::VectorXi> parents,
    Eigen::Ref<const Eigen::VectorX<T>> offsets,
    Eigen::Ref<Eigen::VectorX<T>> positions) {
  const momentum::SkeletonStateT<T> skelState(jointParameters, skeleton);

  const int n = parents.size();
  MT_THROW_IF(offsets.size() != 3 * n, "Mismatched offsets size in jointParametersToPositions()");

  MT_THROW_IF(
      positions.size() != 3 * n, "Mismatched positions size in jointParametersToPositions()");

  for (int i = 0; i < n; ++i) {
    const int parent = parents[i];
    const Eigen::Vector3<T> offset = offsets.template segment<3>(3 * i);
    const Eigen::Vector3<T> p_world = skelState.jointState[parent].transform * offset;
    positions.template segment<3>(3 * i) = p_world;
  }
}

template <typename T>
void d_jointParametersToPositions(
    const momentum::Skeleton& skeleton,
    Eigen::Ref<const Eigen::VectorX<T>> jointParameters,
    Eigen::Ref<const Eigen::VectorXi> parents,
    Eigen::Ref<const Eigen::VectorX<T>> offsets,
    Eigen::Ref<const Eigen::VectorX<T>> dLoss_dPositions,
    Eigen::Ref<Eigen::VectorX<T>> dLoss_jointParameters,
    Eigen::Ref<Eigen::VectorX<T>> dLoss_offsets) {
  const momentum::SkeletonStateT<T> skelState(jointParameters, skeleton);

  const int n = parents.size();
  MT_THROW_IF(offsets.size() != 3 * n, "Mismatched offsets size in d_modelParametersToPositions()");

  dLoss_jointParameters.setZero();

  for (int i = 0; i < n; ++i) {
    const Eigen::Vector3<T> dLoss_dPosition = dLoss_dPositions.template segment<3>(3 * i);

    const int parent = parents[i];
    const Eigen::Vector3<T> p_world =
        skelState.jointState[parent].transform * offsets.template segment<3>(3 * i);

    for (int k = 0; k < 3; ++k) {
      dLoss_offsets(3 * i + k) =
          (skelState.jointState[parent].transform.toLinear() * Eigen::Vector3<T>::Unit(k))
              .dot(dLoss_dPosition);
    }

    // loop over all joints the constraint is attached to and calculate gradient
    size_t jointIndex = parents[i];
    while (jointIndex != momentum::kInvalidIndex) {
      // check for valid index
      if (jointIndex >= skeleton.joints.size()) {
        break;
      }

      const auto& jointState = skelState.jointState[jointIndex];
      const Eigen::Vector3<T> posd = p_world - jointState.translation();
      const size_t paramIdx = jointIndex * momentum::kParametersPerJoint;

      // calculate derivatives based on active joints
      for (size_t d = 0; d < 3; d++) {
        dLoss_jointParameters[paramIdx + d] +=
            dLoss_dPosition.dot(jointState.getTranslationDerivative(d));
      }

      for (size_t d = 0; d < 3; d++) {
        dLoss_jointParameters[paramIdx + 3 + d] +=
            dLoss_dPosition.dot(jointState.getRotationDerivative(d, posd));
      }

      dLoss_jointParameters[paramIdx + 6] +=
          dLoss_dPosition.dot(jointState.getScaleDerivative(posd));

      // go to the next joint
      if (jointIndex < skeleton.joints.size()) {
        jointIndex = skeleton.joints[jointIndex].parent;
      } else {
        break;
      }
    }
  }
}

template <typename T>
struct JointParametersToPositionsFunction
    : public torch::autograd::Function<JointParametersToPositionsFunction<T>> {
 public:
  static variable_list forward(
      AutogradContext* ctx,
      PyObject* characters_in,
      at::Tensor jointParameters,
      at::Tensor parents,
      at::Tensor offsets);

  static variable_list backward(AutogradContext* ctx, variable_list grad_jointParameters);
};

template <typename T>
variable_list JointParametersToPositionsFunction<T>::forward(
    AutogradContext* ctx,
    PyObject* characters_in,
    at::Tensor jointParameters,
    at::Tensor parents,
    at::Tensor offsets) {
  const int nJointParameters = static_cast<int>(
      momentum::kParametersPerJoint *
      anyCharacter(characters_in, "joint_parameters_to_positions()").skeleton.joints.size());

  const auto nPoints_idx = -1;

  TensorChecker checker("jointParametersToPositions");
  const auto input_device = jointParameters.device();

  jointParameters = flattenJointParameters(
      anyCharacter(characters_in, "joint_parameters_to_positions()"), jointParameters);

  bool squeeze = false;
  jointParameters = checker.validateAndFixTensor(
      jointParameters,
      "jointParameters",
      {nJointParameters},
      {"nJointParams"},
      toScalarType<T>(),
      true,
      false,
      &squeeze);

  parents = checker.validateAndFixTensor(
      parents, "parents", {nPoints_idx}, {"nPoints"}, at::kInt, true, true);
  checkValidBoneIndex(
      parents, anyCharacter(characters_in, "joint_parameters_to_positions()"), "parents");

  offsets = checker.validateAndFixTensor(
      offsets, "offsets", {nPoints_idx, 3}, {"nPoints", "xyz"}, toScalarType<T>(), true, true);

  const auto nPoints = checker.getBoundValue(nPoints_idx);
  const auto nBatch = checker.getBatchSize();
  const auto characters = toCharacterList(characters_in, nBatch, "joint_parameters_to_positions()");

  ctx->saved_data["character"] = c10::ivalue::ConcretePyObjectHolder::create(characters_in);
  ctx->save_for_backward({jointParameters, parents, offsets});

  at::Tensor result = at::zeros({nBatch, nPoints, 3}, at::CPU(toScalarType<T>()));

  for (int64_t k = 0; k < nBatch; ++k) {
    const auto character = characters[k];

    at::Tensor jointParameters_cur = jointParameters.select(0, k);
    at::Tensor result_cur = result.select(0, k);
    at::Tensor parents_cur = parents.select(0, k);
    at::Tensor offsets_cur = offsets.select(0, k);

    jointParametersToPositions<T>(
        character->skeleton,
        toEigenMap<T>(jointParameters_cur),
        toEigenMap<int>(parents_cur),
        toEigenMap<T>(offsets_cur),
        toEigenMap<T>(result_cur));
  }

  if (squeeze) {
    result = result.squeeze(0);
  }

  return {result.to(input_device)};
}

template <typename T>
variable_list JointParametersToPositionsFunction<T>::backward(
    AutogradContext* ctx,
    variable_list grad_outputs) {
  MT_THROW_IF(
      grad_outputs.size() != 1,
      "Invalid grad_outputs in ApplyParameterTransformFunction::backward");

  // Restore variables:
  const int nJointParameters = static_cast<int>(
      momentum::kParametersPerJoint *
      anyCharacter(ctx->saved_data["character"].toPyObject(), "jointParametersToPositions()")
          .skeleton.joints.size());

  const auto saved = ctx->get_saved_variables();
  auto savedItr = std::begin(saved);
  auto jointParameters = *savedItr++;
  auto parents = *savedItr++;
  auto offsets = *savedItr++;

  const auto input_device =
      grad_outputs[0].device(); // grad_output size is asserted in the beginning

  auto dLoss_dPositions = grad_outputs[0].contiguous().to(at::DeviceType::CPU, toScalarType<T>());

  bool squeeze_jointParams = false;
  bool squeeze_offsets = false;

  const auto nPoints_idx = -1;

  TensorChecker checker("jointParametersToPositions");

  jointParameters = checker.validateAndFixTensor(
      jointParameters,
      "jointParameters",
      {nJointParameters},
      {"nJointParameters"},
      toScalarType<T>(),
      true,
      false,
      &squeeze_jointParams);

  parents = checker.validateAndFixTensor(parents, "parents", {nPoints_idx}, {"nPoints"}, at::kInt);

  offsets = checker.validateAndFixTensor(
      offsets,
      "offsets",
      {nPoints_idx, 3},
      {"nPoints", "xyz"},
      toScalarType<T>(),
      true,
      false,
      &squeeze_offsets);

  const auto nPoints = checker.getBoundValue(nPoints_idx);
  const auto nBatch = checker.getBatchSize();
  const auto characters = toCharacterList(
      ctx->saved_data["character"].toPyObject(), nBatch, "jointParametersToPositions()");

  at::Tensor d_jointParameters = at::zeros({nBatch, nJointParameters}, at::CPU(toScalarType<T>()));
  at::Tensor d_offsets = at::zeros({nBatch, nPoints, 3}, at::CPU(toScalarType<T>()));

  for (int64_t k = 0; k < nBatch; ++k) {
    const auto character = characters[k];

    at::Tensor jointParameters_cur = jointParameters.select(0, k);
    at::Tensor offsets_cur = offsets.select(0, k);
    at::Tensor parents_cur = parents.select(0, k);
    at::Tensor d_jointParameters_cur = d_jointParameters.select(0, k);
    at::Tensor dLoss_dPositions_cur = dLoss_dPositions.select(0, k);
    at::Tensor d_offsets_cur = d_offsets.select(0, k);

    d_jointParametersToPositions<T>(
        character->skeleton,
        toEigenMap<T>(jointParameters_cur),
        toEigenMap<int>(parents_cur),
        toEigenMap<T>(offsets_cur),
        toEigenMap<T>(dLoss_dPositions_cur),
        toEigenMap<T>(d_jointParameters_cur),
        toEigenMap<T>(d_offsets_cur));
  }

  if (squeeze_jointParams) {
    d_jointParameters = d_jointParameters.sum(0);
  }

  if (squeeze_offsets) {
    d_offsets = d_offsets.sum(0);
  }

  return {
      at::Tensor(), d_jointParameters.to(input_device), at::Tensor(), d_offsets.to(input_device)};
}

#endif // PYMOMENTUM_LIMITED_TORCH_API

} // anonymous namespace

at::Tensor jointParametersToPositions(
    [[maybe_unused]] pybind11::object characters_in,
    [[maybe_unused]] at::Tensor jointParameters,
    [[maybe_unused]] at::Tensor parents,
    [[maybe_unused]] at::Tensor offsets) {
#ifndef PYMOMENTUM_LIMITED_TORCH_API
  return applyTemplatedAutogradFunction<JointParametersToPositionsFunction>(
      characters_in.ptr(), jointParameters, parents, offsets)[0];
#else
  MT_THROW("jointParametersToPositions is not supported in limited PyTorch API mode");
#endif
}

at::Tensor modelParametersToPositions(
    pybind11::object characters,
    at::Tensor modelParameters,
    at::Tensor parents,
    at::Tensor offsets) {
  return jointParametersToPositions(
      characters, applyParamTransform(characters, modelParameters), parents, offsets);
}

} // namespace pymomentum
