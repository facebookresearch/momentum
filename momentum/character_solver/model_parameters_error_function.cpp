/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/model_parameters_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/types.h"
#include "momentum/common/profile.h"

namespace momentum {

template <typename T>
ModelParametersErrorFunctionT<T>::ModelParametersErrorFunctionT(
    const Skeleton& skel,
    const ParameterTransform& pt)
    : SkeletonErrorFunctionT<T>(skel, pt) {}

template <typename T>
ModelParametersErrorFunctionT<T>::ModelParametersErrorFunctionT(const Character& character)
    : ModelParametersErrorFunctionT<T>(character.skeleton, character.parameterTransform) {}

template <typename T>
ModelParametersErrorFunctionT<T>::ModelParametersErrorFunctionT(
    const Character& character,
    const ParameterSet& active)
    : ModelParametersErrorFunctionT<T>(character.skeleton, character.parameterTransform) {
  targetParameters_.v.setZero(character.parameterTransform.numAllModelParameters());
  targetWeights_.setZero(character.parameterTransform.numAllModelParameters());

  for (Eigen::Index i = 0; i < character.parameterTransform.numAllModelParameters(); ++i) {
    if (active.test(i)) {
      targetWeights_(i) = 1;
    }
  }
}

template <typename T>
double ModelParametersErrorFunctionT<T>::getError(
    const ModelParametersT<T>& parameters,
    const SkeletonStateT<T>& /* state */,
    const MeshStateT<T>& /* meshState */) {
  // ignore if we don't have any reasonable data
  if (targetParameters_.size() != parameters.size() || targetWeights_.size() != parameters.size()) {
    return 0.0;
  }

  // calculate difference between parameters and desired parameters
  double error = 0;
  for (Eigen::Index i = 0; i < parameters.size(); ++i) {
    if (this->enabledParameters_.test(i)) {
      const auto pdiff = targetWeights_(i) * (parameters(i) - targetParameters_(i));
      error += pdiff * pdiff;
    }
  }

  // return error
  return error * this->weight_ * kMotionWeight;
}

template <typename T>
double ModelParametersErrorFunctionT<T>::getGradient(
    const ModelParametersT<T>& parameters,
    const SkeletonStateT<T>& /* state */,
    const MeshStateT<T>& /* meshState */,
    Ref<Eigen::VectorX<T>> gradient) {
  // ignore if we don't have any reasonable data
  if (targetParameters_.size() != parameters.size() || targetWeights_.size() != parameters.size()) {
    return 0.0;
  }

  double error = 0;
  for (Eigen::Index i = 0; i < parameters.size(); ++i) {
    if (this->enabledParameters_.test(i)) {
      const auto pdiff = targetWeights_(i) * (parameters(i) - targetParameters_(i));
      gradient(i) += 2.0f * targetWeights_(i) * pdiff * this->weight_ * kMotionWeight;
      error += pdiff * pdiff;
    }
  }

  // return error
  return error * this->weight_ * kMotionWeight;
}

template <typename T>
size_t ModelParametersErrorFunctionT<T>::getJacobianSize() const {
  return (targetWeights_.array() > T(0)).count();
}

template <typename T>
double ModelParametersErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& parameters,
    const SkeletonStateT<T>& /* state */,
    const MeshStateT<T>& /* meshState */,
    Ref<Eigen::MatrixX<T>> jacobian,
    Ref<Eigen::VectorX<T>> residual,
    int& usedRows) {
  MT_PROFILE_FUNCTION();

  usedRows = 0;

  // ignore if we don't have any reasonable data
  if (targetParameters_.size() != parameters.size() || targetWeights_.size() != parameters.size()) {
    return 0.0;
  }

  const float sWeight = std::sqrt(this->weight_ * kMotionWeight);

  Eigen::Index out = 0;

  // calculate difference between parameters and desired parameters
  double error = 0;
  for (Eigen::Index i = 0; i < parameters.size(); ++i) {
    if (this->enabledParameters_.test(i) && targetWeights_(i) > 0) {
      const auto pdiff = targetWeights_(i) * (parameters(i) - targetParameters_(i));
      error += pdiff * pdiff;
      residual(out) = pdiff * sWeight;
      jacobian(out, i) = sWeight * targetWeights_(i);
      ++out;
    }
  }

  usedRows = gsl::narrow_cast<int>(out);

  // return error
  return error * this->weight_ * kMotionWeight;
}

template class ModelParametersErrorFunctionT<float>;
template class ModelParametersErrorFunctionT<double>;

} // namespace momentum
