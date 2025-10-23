/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/tensor_ik/tensor_marker_error_function.h"

#include "pymomentum/tensor_ik/tensor_error_function_utility.h"

#include <momentum/character/character.h>
#include <momentum/diff_ik/fully_differentiable_orientation_error_function.h>
#include <momentum/diff_ik/fully_differentiable_position_error_function.h>

namespace pymomentum {

namespace {

constexpr const int NCONS_IDX = -1;

using momentum::FullyDifferentiableOrientationErrorFunction;
using momentum::FullyDifferentiableOrientationErrorFunctionT;
using momentum::FullyDifferentiablePositionErrorFunction;
using momentum::FullyDifferentiablePositionErrorFunctionT;

template <typename T>
class TensorPositionErrorFunction : public TensorErrorFunction<T> {
 public:
  TensorPositionErrorFunction(
      size_t batchSize,
      size_t nFrames,
      at::Tensor parents,
      at::Tensor offsets,
      at::Tensor weights,
      at::Tensor targets);

 protected:
  std::shared_ptr<momentum::SkeletonErrorFunctionT<T>> createErrorFunctionImp(
      const momentum::Character& character,
      size_t iBatch,
      size_t jFrame) const override;
};

template <typename T>
TensorPositionErrorFunction<T>::TensorPositionErrorFunction(
    size_t batchSize,
    size_t nFrames,
    at::Tensor parents,
    at::Tensor offsets,
    at::Tensor weights,
    at::Tensor targets)
    : TensorErrorFunction<T>(
          "Position",
          "position_cons",
          batchSize,
          nFrames,
          {{FullyDifferentiablePositionErrorFunction::kParents,
            parents,
            {NCONS_IDX},
            TensorType::TYPE_INT,
            TensorInput::NON_DIFFERENTIABLE,
            TensorInput::REQUIRED},
           {FullyDifferentiablePositionErrorFunction::kOffsets,
            offsets,
            {NCONS_IDX, 3},
            TensorType::TYPE_FLOAT,
            TensorInput::DIFFERENTIABLE,
            TensorInput::OPTIONAL},
           {FullyDifferentiablePositionErrorFunction::kWeights,
            weights,
            {NCONS_IDX},
            TensorType::TYPE_FLOAT,
            TensorInput::DIFFERENTIABLE,
            TensorInput::OPTIONAL},
           {FullyDifferentiablePositionErrorFunction::kTargets,
            targets,
            {NCONS_IDX, 3},
            TensorType::TYPE_FLOAT,
            TensorInput::DIFFERENTIABLE,
            TensorInput::REQUIRED}},
          {{NCONS_IDX, "nConstraints"}}) {}

template <typename T>
std::shared_ptr<momentum::SkeletonErrorFunctionT<T>>
TensorPositionErrorFunction<T>::createErrorFunctionImp(
    const momentum::Character& character,
    size_t iBatch,
    size_t jFrame) const {
  auto result = std::make_unique<momentum::FullyDifferentiablePositionErrorFunctionT<T>>(
      character.skeleton, character.parameterTransform);

  const auto weights = this->getTensorInput(FullyDifferentiablePositionErrorFunction::kWeights)
                           .template toEigenMap<T>(iBatch, jFrame);
  const auto offsets = this->getTensorInput(FullyDifferentiablePositionErrorFunction::kOffsets)
                           .template toEigenMap<T>(iBatch, jFrame);
  const auto parents = this->getTensorInput(FullyDifferentiablePositionErrorFunction::kParents)
                           .template toEigenMap<int>(iBatch, jFrame);
  const auto targets = this->getTensorInput(FullyDifferentiablePositionErrorFunction::kTargets)
                           .template toEigenMap<T>(iBatch, jFrame);

  const auto nCons = this->sharedSize(NCONS_IDX);
  for (Eigen::Index i = 0; i < nCons; ++i) {
    result->addConstraint(
        momentum::PositionConstraintT<T>(
            extractVector<T, 3>(offsets, i, Eigen::Vector3<T>::Zero()),
            extractVector<T, 3>(targets, i),
            extractScalar<int>(parents, i),
            extractScalar<T>(weights, i, T(1))));
  }

  return result;
}

template <typename T>
class TensorOrientationErrorFunction : public TensorErrorFunction<T> {
 public:
  TensorOrientationErrorFunction(
      size_t batchSize,
      size_t nFrames,
      at::Tensor parents,
      at::Tensor offsets,
      at::Tensor weights,
      at::Tensor targets);

 protected:
  std::shared_ptr<momentum::SkeletonErrorFunctionT<T>> createErrorFunctionImp(
      const momentum::Character& character,
      size_t iBatch,
      size_t jFrame) const override;
};

template <typename T>
TensorOrientationErrorFunction<T>::TensorOrientationErrorFunction(
    size_t batchSize,
    size_t nFrames,
    at::Tensor parents,
    at::Tensor orientation_offsets,
    at::Tensor weights,
    at::Tensor orientation_targets)
    : TensorErrorFunction<T>(
          "Orientation",
          "orientation_cons",
          batchSize,
          nFrames,
          {{FullyDifferentiableOrientationErrorFunction::kParents,
            parents,
            {NCONS_IDX},
            TensorType::TYPE_INT,
            TensorInput::NON_DIFFERENTIABLE,
            TensorInput::REQUIRED},
           {FullyDifferentiableOrientationErrorFunction::kOffsets,
            orientation_offsets,
            {NCONS_IDX, 4},
            TensorType::TYPE_FLOAT,
            TensorInput::DIFFERENTIABLE,
            TensorInput::OPTIONAL},
           {FullyDifferentiableOrientationErrorFunction::kWeights,
            weights,
            {NCONS_IDX},
            TensorType::TYPE_FLOAT,
            TensorInput::DIFFERENTIABLE,
            TensorInput::OPTIONAL},
           {FullyDifferentiableOrientationErrorFunction::kTargets,
            orientation_targets,
            {NCONS_IDX, 4},
            TensorType::TYPE_FLOAT,
            TensorInput::DIFFERENTIABLE,
            TensorInput::REQUIRED}},
          {{NCONS_IDX, "nConstraints"}}) {}

template <typename T>
std::shared_ptr<momentum::SkeletonErrorFunctionT<T>>
TensorOrientationErrorFunction<T>::createErrorFunctionImp(
    const momentum::Character& character,
    size_t iBatch,
    size_t jFrame) const {
  auto result = std::make_unique<momentum::FullyDifferentiableOrientationErrorFunctionT<T>>(
      character.skeleton, character.parameterTransform);

  const auto weights = this->getTensorInput(FullyDifferentiableOrientationErrorFunction::kWeights)
                           .template toEigenMap<T>(iBatch, jFrame);
  const auto orientation_offsets =
      this->getTensorInput(FullyDifferentiableOrientationErrorFunction::kOffsets)
          .template toEigenMap<T>(iBatch, jFrame);
  const auto parents = this->getTensorInput(FullyDifferentiableOrientationErrorFunction::kParents)
                           .template toEigenMap<int>(iBatch, jFrame);
  const auto orientation_targets =
      this->getTensorInput(FullyDifferentiableOrientationErrorFunction::kTargets)
          .template toEigenMap<T>(iBatch, jFrame);

  const auto nCons = this->sharedSize(NCONS_IDX);
  for (Eigen::Index i = 0; i < nCons; ++i) {
    result->addConstraint(
        momentum::OrientationConstraintT<T>(
            extractQuaternion<T>(orientation_offsets, i),
            extractQuaternion<T>(orientation_targets, i),
            extractScalar<int>(parents, i),
            extractScalar<T>(weights, i, T(1))));
  }

  return result;
}

} // anonymous namespace

template <typename T>
std::unique_ptr<TensorErrorFunction<T>> createPositionErrorFunction(
    size_t batchSize,
    size_t nFrames,
    at::Tensor parents,
    at::Tensor offsets,
    at::Tensor weights,
    at::Tensor targets) {
  return std::make_unique<TensorPositionErrorFunction<T>>(
      batchSize, nFrames, parents, offsets, weights, targets);
}

template <typename T>
std::unique_ptr<TensorErrorFunction<T>> createOrientationErrorFunction(
    size_t batchSize,
    size_t nFrames,
    at::Tensor parents,
    at::Tensor orientation_offsets,
    at::Tensor weights,
    at::Tensor orientation_targets) {
  return std::make_unique<TensorOrientationErrorFunction<T>>(
      batchSize, nFrames, parents, orientation_offsets, weights, orientation_targets);
}

template std::unique_ptr<TensorErrorFunction<float>> createPositionErrorFunction<float>(
    size_t batchSize,
    size_t nFrames,
    at::Tensor parents,
    at::Tensor offsets,
    at::Tensor weights,
    at::Tensor targets);

template std::unique_ptr<TensorErrorFunction<double>> createPositionErrorFunction<double>(
    size_t batchSize,
    size_t nFrames,
    at::Tensor parents,
    at::Tensor offsets,
    at::Tensor weights,
    at::Tensor targets);

template std::unique_ptr<TensorErrorFunction<float>> createOrientationErrorFunction<float>(
    size_t batchSize,
    size_t nFrames,
    at::Tensor parents,
    at::Tensor orientation_offsets,
    at::Tensor weights,
    at::Tensor orientation_targets);

template std::unique_ptr<TensorErrorFunction<double>> createOrientationErrorFunction<double>(
    size_t batchSize,
    size_t nFrames,
    at::Tensor parents,
    at::Tensor orientation_offsets,
    at::Tensor weights,
    at::Tensor orientation_targets);

} // namespace pymomentum
