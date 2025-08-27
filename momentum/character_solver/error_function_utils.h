/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character/parameter_transform.h>

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

} // namespace momentum
