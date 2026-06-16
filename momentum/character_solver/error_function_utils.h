/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character/parameter_transform.h>
#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>

namespace momentum {

template <typename T>
void gradient_jointParams_to_modelParams(
    const T& grad_jointParams,
    const Eigen::Index iJointParam,
    const ParameterTransform& parameterTransform,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  // explicitly multiply with the parameter transform to generate parameter space gradients
  for (auto index = parameterTransform.transform.outerIndexPtr()[iJointParam];
       index < parameterTransform.transform.outerIndexPtr()[iJointParam + 1];
       ++index) {
    gradient[parameterTransform.transform.innerIndexPtr()[index]] +=
        grad_jointParams * parameterTransform.transform.valuePtr()[index];
  }
}

template <typename T>
void jacobian_jointParams_to_modelParams(
    const Eigen::Ref<const Eigen::VectorX<T>>& jacobian_jointParams,
    const Eigen::Index iJointParam,
    const ParameterTransform& parameterTransform,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian) {
  // explicitly multiply with the parameter transform to generate parameter space gradients
  for (auto index = parameterTransform.transform.outerIndexPtr()[iJointParam];
       index < parameterTransform.transform.outerIndexPtr()[iJointParam + 1];
       ++index) {
    jacobian.col(parameterTransform.transform.innerIndexPtr()[index]) +=
        jacobian_jointParams * parameterTransform.transform.valuePtr()[index];
  }
}

template <typename T>
void jacobian_jointParams_to_modelParams(
    const T& jacobian_jointParams,
    const Eigen::Index iJointParam,
    const ParameterTransform& parameterTransform,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian) {
  // explicitly multiply with the parameter transform to generate parameter space gradients
  for (auto index = parameterTransform.transform.outerIndexPtr()[iJointParam];
       index < parameterTransform.transform.outerIndexPtr()[iJointParam + 1];
       ++index) {
    jacobian(0, parameterTransform.transform.innerIndexPtr()[index]) +=
        jacobian_jointParams * parameterTransform.transform.valuePtr()[index];
  }
}

/// Overload accepting MatrixBase for Eigen expression types (e.g. row blocks).
template <typename T, typename Derived>
void jacobian_jointParams_to_modelParams(
    const T& jacobian_jointParams,
    const Eigen::Index iJointParam,
    const ParameterTransform& parameterTransform,
    Eigen::MatrixBase<Derived>& jacobian) {
  for (auto index = parameterTransform.transform.outerIndexPtr()[iJointParam];
       index < parameterTransform.transform.outerIndexPtr()[iJointParam + 1];
       ++index) {
    jacobian(0, parameterTransform.transform.innerIndexPtr()[index]) +=
        jacobian_jointParams * parameterTransform.transform.valuePtr()[index];
  }
}

template <typename T>
void jacobian_jointParams_to_modelParams(
    const T& jacobian_jointParams,
    const Eigen::Index iJointParam,
    const Eigen::Index iRow,
    const ParameterTransform& parameterTransform,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian) {
  // explicitly multiply with the parameter transform to generate parameter space gradients
  for (auto index = parameterTransform.transform.outerIndexPtr()[iJointParam];
       index < parameterTransform.transform.outerIndexPtr()[iJointParam + 1];
       ++index) {
    jacobian(iRow, parameterTransform.transform.innerIndexPtr()[index]) +=
        jacobian_jointParams * parameterTransform.transform.valuePtr()[index];
  }
}

/// Accumulate the derivatives of `direction . position` with respect to the model parameters along
/// the joint parent chain, walking from `startJoint` up to (but not including) `stopJoint`.
///
/// For each active joint parameter, `sink(value, jointParamIndex)` routes the joint-parameter
/// derivative into the destination buffer (a gradient vector or a single Jacobian row). Pass
/// `stopJoint = kInvalidIndex` to walk all the way to the skeleton root. `scaleCorrection` is added
/// to the scale-parameter derivative to account for a collision primitive's support radius scaling
/// with the joint's uniform scale.
template <typename T, typename SinkFn>
void accumulateChainDerivatives(
    const SkeletonStateT<T>& state,
    const Skeleton& skeleton,
    const VectorX<bool>& activeJointParams,
    const Vector3<T>& position,
    const Vector3<T>& direction,
    T weight,
    size_t startJoint,
    size_t stopJoint,
    T scaleCorrection,
    const SinkFn& sink) {
  size_t jointIndex = startJoint;
  while (jointIndex != kInvalidIndex && jointIndex != stopJoint) {
    const auto& jointState = state.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Vector3<T> posd = position - jointState.translation();

    for (size_t d = 0; d < 3; ++d) {
      if (activeJointParams[paramIndex + d]) {
        sink(direction.dot(jointState.getTranslationDerivative(d)) * weight, paramIndex + d);
      }
      if (activeJointParams[paramIndex + 3 + d]) {
        sink(direction.dot(jointState.getRotationDerivative(d, posd)) * weight, paramIndex + 3 + d);
      }
    }
    if (activeJointParams[paramIndex + 6]) {
      sink(
          (direction.dot(jointState.getScaleDerivative(posd)) + scaleCorrection) * weight,
          paramIndex + 6);
    }

    jointIndex = skeleton.joints[jointIndex].parent;
  }
}

} // namespace momentum
