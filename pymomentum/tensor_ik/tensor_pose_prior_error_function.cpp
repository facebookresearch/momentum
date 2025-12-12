/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/tensor_ik/tensor_pose_prior_error_function.h"

#include <momentum/character/character.h>
#include <momentum/character_solver/pose_prior_error_function.h>
#include <momentum/diff_ik/fully_differentiable_pose_prior_error_function.h>
#include <momentum/math/mppca.h>

namespace pymomentum {

namespace {

template <typename T>
class TensorPosePriorErrorFunction : public TensorErrorFunction<T> {
 public:
  explicit TensorPosePriorErrorFunction(
      size_t batchSize,
      size_t nFrames,
      const momentum::Mppca* posePrior_model);

 protected:
  std::shared_ptr<momentum::SkeletonErrorFunctionT<T>> createErrorFunctionImp(
      const momentum::Character& character,
      size_t /* unused iBatch */,
      size_t /* unused nFrames */) const override;

 private:
  std::shared_ptr<const momentum::MppcaT<T>> _posePrior_model;
};

template <typename T>
TensorPosePriorErrorFunction<T>::TensorPosePriorErrorFunction(
    size_t batchSize,
    size_t nFrames,
    const momentum::Mppca* posePrior_model)
    : TensorErrorFunction<T>(
          "PosePrior",
          "pose_prior",
          batchSize,
          nFrames,
          {{momentum::FullyDifferentiablePosePriorErrorFunction::kPi,
            at::Tensor(),
            {},
            TensorType::TYPE_SENTINEL,
            TensorInput::NON_DIFFERENTIABLE,
            TensorInput::OPTIONAL},
           {momentum::FullyDifferentiablePosePriorErrorFunction::kMu,
            at::Tensor(),
            {},
            TensorType::TYPE_SENTINEL,
            TensorInput::NON_DIFFERENTIABLE,
            TensorInput::OPTIONAL},
           {momentum::FullyDifferentiablePosePriorErrorFunction::kW,
            at::Tensor(),
            {},
            TensorType::TYPE_SENTINEL,
            TensorInput::NON_DIFFERENTIABLE,
            TensorInput::OPTIONAL},
           {momentum::FullyDifferentiablePosePriorErrorFunction::kSigma,
            at::Tensor(),
            {},
            TensorType::TYPE_SENTINEL,
            TensorInput::NON_DIFFERENTIABLE,
            TensorInput::OPTIONAL},
           {momentum::FullyDifferentiablePosePriorErrorFunction::kParameterIndices,
            at::Tensor(),
            {},
            TensorType::TYPE_SENTINEL,
            TensorInput::NON_DIFFERENTIABLE,
            TensorInput::OPTIONAL},
           {"pose_prior",
            at::Tensor(),
            {},
            TensorType::TYPE_SENTINEL,
            TensorInput::NON_DIFFERENTIABLE,
            TensorInput::REQUIRED}},
          {}) {
  if (posePrior_model != nullptr) {
    _posePrior_model = std::make_shared<momentum::MppcaT<T>>(posePrior_model->cast<T>());
  }
}

template <typename T>
std::shared_ptr<momentum::SkeletonErrorFunctionT<T>>
TensorPosePriorErrorFunction<T>::createErrorFunctionImp(
    const momentum::Character& character,
    size_t /* unused iBatch */,
    size_t /* unused jFrame */) const {
  auto result = std::make_unique<momentum::PosePriorErrorFunctionT<T>>(
      character.skeleton, character.parameterTransform, _posePrior_model);
  return result;
}

} // anonymous namespace

template <typename T>
std::unique_ptr<TensorErrorFunction<T>> createPosePriorErrorFunction(
    size_t batchSize,
    size_t nFrames,
    const momentum::Mppca* posePrior_model) {
  return std::make_unique<TensorPosePriorErrorFunction<T>>(batchSize, nFrames, posePrior_model);
}

template std::unique_ptr<TensorErrorFunction<float>> createPosePriorErrorFunction<float>(
    size_t batchSize,
    size_t nFrames,
    const momentum::Mppca* posePrior_model);

template std::unique_ptr<TensorErrorFunction<double>> createPosePriorErrorFunction<double>(
    size_t batchSize,
    size_t nFrames,
    const momentum::Mppca* posePrior_model);

} // namespace pymomentum
