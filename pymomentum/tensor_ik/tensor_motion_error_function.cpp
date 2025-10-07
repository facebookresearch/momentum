/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/tensor_ik/tensor_motion_error_function.h"

#include <momentum/character/character.h>
#include <momentum/diff_ik/fully_differentiable_motion_error_function.h>

namespace pymomentum {

using momentum::FullyDifferentiableMotionErrorFunction;
using momentum::FullyDifferentiableMotionErrorFunctionT;

namespace {

template <typename T>
class TensorMotionErrorFunction : public TensorErrorFunction<T> {
 public:
  TensorMotionErrorFunction(
      size_t batchSize,
      size_t nFrames,
      const momentum::ParameterTransform& paramTransform,
      at::Tensor targetParameters,
      at::Tensor targetWeights);

 protected:
  std::shared_ptr<momentum::SkeletonErrorFunctionT<T>> createErrorFunctionImp(
      const momentum::Character& character,
      size_t iBatch,
      size_t jFrame) const override;
};

template <typename T>
TensorMotionErrorFunction<T>::TensorMotionErrorFunction(
    size_t batchSize,
    size_t nFrames,
    const momentum::ParameterTransform& paramTransform,
    at::Tensor targetParameters,
    at::Tensor targetWeights)
    : TensorErrorFunction<T>(
          "Motion",
          "motion",
          batchSize,
          nFrames,
          {{FullyDifferentiableMotionErrorFunction::kTargetParameters,
            targetParameters,
            {(int)paramTransform.numAllModelParameters()},
            TensorType::TYPE_FLOAT,
            TensorInput::DIFFERENTIABLE,
            TensorInput::REQUIRED},
           {FullyDifferentiableMotionErrorFunction::kTargetWeights,
            targetWeights,
            {(int)paramTransform.numAllModelParameters()},
            TensorType::TYPE_FLOAT,
            TensorInput::DIFFERENTIABLE,
            TensorInput::OPTIONAL}},
          {}) {}

template <typename T>
std::shared_ptr<momentum::SkeletonErrorFunctionT<T>>
TensorMotionErrorFunction<T>::createErrorFunctionImp(
    const momentum::Character& character,
    size_t iBatch,
    size_t jFrame) const {
  auto result = std::make_unique<momentum::FullyDifferentiableMotionErrorFunctionT<T>>(
      character.skeleton, character.parameterTransform);
  const auto& weights =
      this->getTensorInput(FullyDifferentiableMotionErrorFunction::kTargetWeights);
  result->setTargetParameters(
      this->getTensorInput(FullyDifferentiableMotionErrorFunction::kTargetParameters)
          .template toEigenMap<T>(iBatch, jFrame),
      weights.isTensorEmpty()
          ? Eigen::VectorX<T>::Ones(character.parameterTransform.numAllModelParameters()).eval()
          : weights.template toEigenMap<T>(iBatch, jFrame));
  return result;
}

} // anonymous namespace

template <typename T>
std::unique_ptr<TensorErrorFunction<T>> createMotionErrorFunction(
    size_t batchSize,
    size_t nFrames,
    const momentum::ParameterTransform& paramTransform,
    at::Tensor targetParameters,
    at::Tensor targetWeights) {
  return std::make_unique<TensorMotionErrorFunction<T>>(
      batchSize, nFrames, paramTransform, targetParameters, targetWeights);
}

template std::unique_ptr<TensorErrorFunction<float>> createMotionErrorFunction<float>(
    size_t batchSize,
    size_t nFrames,
    const momentum::ParameterTransform& paramTransform,
    at::Tensor targetParameters,
    at::Tensor targetWeights);
template std::unique_ptr<TensorErrorFunction<double>> createMotionErrorFunction<double>(
    size_t batchSize,
    size_t nFrames,
    const momentum::ParameterTransform& paramTransform,
    at::Tensor targetParameters,
    at::Tensor targetWeights);

} // namespace pymomentum
