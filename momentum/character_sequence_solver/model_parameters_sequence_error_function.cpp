/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_sequence_solver/model_parameters_sequence_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"

namespace momentum {

template <typename T>
ModelParametersSequenceErrorFunctionT<T>::ModelParametersSequenceErrorFunctionT(
    const Skeleton& skel,
    const ParameterTransform& pt)
    : SequenceErrorFunctionT<T>(skel, pt) {
  targetWeights_.setOnes(pt.numAllModelParameters());
}

template <typename T>
ModelParametersSequenceErrorFunctionT<T>::ModelParametersSequenceErrorFunctionT(
    const Character& character)
    : ModelParametersSequenceErrorFunctionT<T>(character.skeleton, character.parameterTransform) {}

template <typename T>
double ModelParametersSequenceErrorFunctionT<T>::getError(
    gsl::span<const ModelParametersT<T>> modelParameters,
    gsl::span<const SkeletonStateT<T>> /* skelStates */,
    gsl::span<const MeshStateT<T>> /* meshStates */) const {
  // ignore if we don't have any reasonable data
  if (targetWeights_.size() !=
      gsl::narrow_cast<Eigen::Index>(this->parameterTransform_.numAllModelParameters())) {
    return 0.0;
  }

  const auto np = gsl::narrow_cast<Eigen::Index>(this->parameterTransform_.numAllModelParameters());

  MT_CHECK(modelParameters.size() == 2);
  const auto& prevParams = modelParameters[0];
  const auto& nextParams = modelParameters[1];

  // calculate difference between parameters and desired parameters
  double error = 0;
  for (Eigen::Index i = 0; i < np; ++i) {
    if (this->enabledParameters_.test(i) && targetWeights_(i) > 0) {
      const auto pdiff = targetWeights_(i) * (nextParams(i) - prevParams(i));
      error += pdiff * pdiff;
    }
  }

  // return error
  return error * this->weight_ * kMotionWeight;
}

template <typename T>
double ModelParametersSequenceErrorFunctionT<T>::getGradient(
    gsl::span<const ModelParametersT<T>> modelParameters,
    gsl::span<const SkeletonStateT<T>> /* skelStates */,
    gsl::span<const MeshStateT<T>> /* meshStates */,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  // ignore if we don't have any reasonable data
  if (targetWeights_.size() !=
      gsl::narrow_cast<Eigen::Index>(this->parameterTransform_.numAllModelParameters())) {
    return 0.0;
  }

  const auto np = gsl::narrow_cast<Eigen::Index>(this->parameterTransform_.numAllModelParameters());

  MT_CHECK(modelParameters.size() == 2);
  const auto& prevParams = modelParameters[0];
  const auto& nextParams = modelParameters[1];

  Eigen::Ref<Eigen::VectorX<T>> prevGrad = gradient.segment(0, np);
  Eigen::Ref<Eigen::VectorX<T>> nextGrad = gradient.segment(np, np);

  double error = 0;
  for (Eigen::Index i = 0; i < np; ++i) {
    if (this->enabledParameters_.test(i) && targetWeights_(i) > 0) {
      const auto pdiff = targetWeights_(i) * (nextParams(i) - prevParams(i));
      prevGrad(i) -= 2.0f * targetWeights_(i) * pdiff * this->weight_ * kMotionWeight;
      nextGrad(i) += 2.0f * targetWeights_(i) * pdiff * this->weight_ * kMotionWeight;
      error += pdiff * pdiff;
    }
  }

  // return error
  return error * this->weight_ * kMotionWeight;
}

template <typename T>
size_t ModelParametersSequenceErrorFunctionT<T>::getJacobianSize() const {
  return (targetWeights_.array() > 0).count();
}

template <typename T>
double ModelParametersSequenceErrorFunctionT<T>::getJacobian(
    gsl::span<const ModelParametersT<T>> modelParameters,
    gsl::span<const SkeletonStateT<T>> /* skelStates */,
    gsl::span<const MeshStateT<T>> /* meshStates */,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    int& usedRows) const {
  MT_PROFILE_FUNCTION();
  MT_CHECK(
      jacobian.cols() ==
      gsl::narrow_cast<Eigen::Index>(
          numFrames() * this->parameterTransform_.numAllModelParameters()));

  usedRows = 0;

  const auto np = gsl::narrow_cast<Eigen::Index>(this->parameterTransform_.numAllModelParameters());

  // ignore if we don't have any reasonable data
  if (targetWeights_.size() !=
      gsl::narrow_cast<Eigen::Index>(this->parameterTransform_.numAllModelParameters())) {
    return 0.0;
  }

  const float sWeight = std::sqrt(this->weight_ * kMotionWeight);

  MT_CHECK(modelParameters.size() == 2);
  const auto& prevParams = modelParameters[0];
  const auto& nextParams = modelParameters[1];
  Eigen::Ref<Eigen::MatrixX<T>> prevJac = jacobian.topLeftCorner(jacobian.rows(), np);
  Eigen::Ref<Eigen::MatrixX<T>> nextJac = jacobian.block(0, np, jacobian.rows(), np);

  Eigen::Index out = 0;

  // calculate difference between parameters and desired parameters
  double error = 0;
  for (Eigen::Index i = 0; i < np; ++i) {
    if (this->enabledParameters_.test(i) && targetWeights_(i) > 0) {
      const auto pdiff = targetWeights_(i) * (nextParams(i) - prevParams(i));
      error += pdiff * pdiff;
      residual(out) = pdiff * sWeight;
      prevJac(out, i) = -sWeight * targetWeights_(i);
      nextJac(out, i) = sWeight * targetWeights_(i);
      ++out;
    }
  }

  MT_CHECK(out <= residual.rows() && out <= jacobian.rows());

  usedRows = gsl::narrow_cast<int>(out);

  // return error
  return error * this->weight_ * kMotionWeight;
}

template class ModelParametersSequenceErrorFunctionT<float>;
template class ModelParametersSequenceErrorFunctionT<double>;

} // namespace momentum
