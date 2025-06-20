/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/diff_ik/fully_differentiable_motion_error_function.h>

#include "momentum/math/utility.h"

#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/diff_ik/ceres_utility.h>

#include <algorithm>

namespace momentum {

template <typename T>
FullyDifferentiableMotionErrorFunctionT<T>::FullyDifferentiableMotionErrorFunctionT(
    const Skeleton& skel,
    const ParameterTransform& pt)
    : ModelParametersErrorFunctionT<T>(skel, pt) {}

template <typename T>
FullyDifferentiableMotionErrorFunctionT<T>::~FullyDifferentiableMotionErrorFunctionT() = default;

template <typename T>
std::vector<std::string> FullyDifferentiableMotionErrorFunctionT<T>::inputs() const {
  return {kTargetParameters, kTargetWeights};
}

template <typename T>
Eigen::Index FullyDifferentiableMotionErrorFunctionT<T>::getInputSize(
    const std::string& name) const {
  if (name == kTargetParameters || name == kTargetWeights) {
    return this->parameterTransform_.numAllModelParameters();
  } else {
    MT_THROW("Unknown input to FullyDifferentiableMotionErrorFunctionT: {}", name);
  }
}

template <typename T>
void FullyDifferentiableMotionErrorFunctionT<T>::getInputImp(
    const std::string& name,
    Eigen::Ref<Eigen::VectorX<T>> value) const {
  if (name == kTargetParameters) {
    const auto& targetParameters = this->getTargetParameters();
    if (targetParameters.size() != this->parameterTransform_.numAllModelParameters()) {
      value = Eigen::VectorX<T>::Zero(this->parameterTransform_.numAllModelParameters());
    } else {
      value = targetParameters.v;
    }
  } else if (name == kTargetWeights) {
    const auto& targetWeights = this->getTargetWeights();
    if (targetWeights.size() != this->parameterTransform_.numAllModelParameters()) {
      value = Eigen::VectorX<T>::Zero(this->parameterTransform_.numAllModelParameters());
    } else {
      value = targetWeights;
    }
  } else {
    MT_THROW("Unknown input to FullyDifferentiableMotionErrorFunctionT: {}", name);
  }
}

template <typename T>
void FullyDifferentiableMotionErrorFunctionT<T>::setInputImp(
    const std::string& name,
    Eigen::Ref<const Eigen::VectorX<T>> value) {
  if (name == kTargetParameters) {
    this->setTargetParameters(value, this->getTargetWeights());
  } else if (name == kTargetWeights) {
    this->setTargetParameters(this->getTargetParameters(), value);
  } else {
    MT_THROW("Unknown input to FullyDifferentiableMotionErrorFunctionT: {}", name);
  }
}

template <typename T>
Eigen::VectorX<T> FullyDifferentiableMotionErrorFunctionT<T>::d_gradient_d_input_dot(
    const std::string& inputName,
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& /* state */,
    Eigen::Ref<const Eigen::VectorX<T>> inputVec) {
  // The error function is just
  //    \sum_i (w_i * (param_i - targetParam_i))^2
  // The gradient is then
  //    \sum_i 2 * w_i^2 * (param_i - targetParam_i)
  //
  // The two input gradients are thus:
  //   dGrad_dw_i = 4 * w_i * (param_i - targetParam_i)
  //   dGrad_dtargetParam_i = -2 * w_i^2;
  const auto& targetParameters = this->getTargetParameters();
  const auto& targetWeights = this->getTargetWeights();

  if (targetWeights.size() != this->parameterTransform_.numAllModelParameters() ||
      targetParameters.size() != this->parameterTransform_.numAllModelParameters()) {
    return Eigen::VectorX<T>::Zero(this->parameterTransform_.numAllModelParameters());
  }

  if (inputName == kTargetParameters) {
    return T(-2) * this->weight_ * ModelParametersErrorFunctionT<T>::kMotionWeight *
        targetWeights.array().square() * inputVec.array();
  } else if (inputName == kTargetWeights) {
    return T(4) * this->weight_ * ModelParametersErrorFunctionT<T>::kMotionWeight *
        targetWeights.array() * (params.v - targetParameters.v).array() * inputVec.array();
  } else {
    MT_THROW("Unknown input to FullyDifferentiableMotionErrorFunctionT: {}", inputName);
  }
}

template class FullyDifferentiableMotionErrorFunctionT<float>;
template class FullyDifferentiableMotionErrorFunctionT<double>;

} // namespace momentum
