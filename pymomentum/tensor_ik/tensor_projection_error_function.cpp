/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/tensor_ik/tensor_projection_error_function.h"

#include "pymomentum/tensor_ik/tensor_error_function_utility.h"

#include <momentum/character/character.h>
#include <momentum/character_solver/projection_error_function.h>
#include <momentum/diff_ik/fully_differentiable_projection_error_function.h>

namespace pymomentum {

using momentum::FullyDifferentiableProjectionErrorFunction;
using momentum::FullyDifferentiableProjectionErrorFunctionT;

namespace {

const int NCONS_IDX = -1;

template <typename T>
class TensorProjectionErrorFunction : public TensorErrorFunction<T> {
 public:
  TensorProjectionErrorFunction(
      size_t batchSize,
      size_t nFrames,
      at::Tensor projections,
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
TensorProjectionErrorFunction<T>::TensorProjectionErrorFunction(
    size_t batchSize,
    size_t nFrames,
    at::Tensor projections,
    at::Tensor parents,
    at::Tensor offsets,
    at::Tensor weights,
    at::Tensor targets)
    : TensorErrorFunction<T>(
          "Projection",
          "projection_cons",
          batchSize,
          nFrames,
          {{FullyDifferentiableProjectionErrorFunction::kProjections,
            projections,
            {NCONS_IDX, 3, 4},
            TensorType::TYPE_FLOAT,
            TensorInput::DIFFERENTIABLE,
            TensorInput::REQUIRED},
           {FullyDifferentiableProjectionErrorFunction::kParents,
            parents,
            {NCONS_IDX},
            TensorType::TYPE_INT,
            TensorInput::NON_DIFFERENTIABLE,
            TensorInput::REQUIRED},
           {FullyDifferentiableProjectionErrorFunction::kOffsets,
            offsets,
            {NCONS_IDX, 3},
            TensorType::TYPE_FLOAT,
            TensorInput::DIFFERENTIABLE,
            TensorInput::OPTIONAL},
           {FullyDifferentiableProjectionErrorFunction::kWeights,
            weights,
            {NCONS_IDX},
            TensorType::TYPE_FLOAT,
            TensorInput::DIFFERENTIABLE,
            TensorInput::OPTIONAL},
           {FullyDifferentiableProjectionErrorFunction::kTargets,
            targets,
            {NCONS_IDX, 2},
            TensorType::TYPE_FLOAT,
            TensorInput::DIFFERENTIABLE,
            TensorInput::OPTIONAL}},
          {{NCONS_IDX, "nConstraints"}}) {}

template <typename T>
std::shared_ptr<momentum::SkeletonErrorFunctionT<T>>
TensorProjectionErrorFunction<T>::createErrorFunctionImp(
    const momentum::Character& character,
    size_t iBatch,
    size_t jFrame) const {
  auto errorFun = std::make_unique<FullyDifferentiableProjectionErrorFunctionT<T>>(
      character.skeleton, character.parameterTransform);

  using momentum ::FullyDifferentiableProjectionErrorFunction;

  // The Eigen map below are vectors for data of this sample in a batch.
  // One sample is made by NCAM cameras.
  // Each camera has NCONS constraints.
  // So each Eigen map actually stores 2D data.
  const auto weights = this->getTensorInput(FullyDifferentiableProjectionErrorFunction::kWeights)
                           .template toEigenMap<T>(iBatch, jFrame);
  const auto offsets =
      this->getTensorInput(momentum::FullyDifferentiableProjectionErrorFunction::kOffsets)
          .template toEigenMap<T>(iBatch, jFrame);
  const auto parents =
      this->getTensorInput(momentum::FullyDifferentiableProjectionErrorFunction::kParents)
          .template toEigenMap<int>(iBatch, jFrame);
  const auto targets = this->getTensorInput(FullyDifferentiableProjectionErrorFunction::kTargets)
                           .template toEigenMap<T>(iBatch, jFrame);
  const auto projections =
      this->getTensorInput(FullyDifferentiableProjectionErrorFunction::kProjections)
          .template toEigenMap<T>(iBatch, jFrame);

  const auto nCons = this->sharedSize(NCONS_IDX);
  for (Eigen::Index kCons = 0; kCons < nCons; ++kCons) {
    momentum::ProjectionConstraintDataT<T> constraintData;
    constraintData.target = extractVector<T, 2>(targets, kCons, Eigen::Vector2<T>::Zero());
    constraintData.parent = extractScalar<int>(parents, kCons);
    constraintData.offset = extractVector<T, 3>(offsets, kCons, Eigen::Vector3<T>::Zero());
    constraintData.weight = extractScalar<T>(weights, kCons, T(1));
    constraintData.projection = extractMatrix<T, 3, 4>(projections, kCons);
    errorFun->addConstraint(constraintData);
  }

  return errorFun;
}

} // End of anonymous namespace

template <typename T>
std::unique_ptr<TensorErrorFunction<T>> createProjectionErrorFunction(
    size_t batchSize,
    size_t nFrames,
    at::Tensor projections,
    at::Tensor parents,
    at::Tensor offsets,
    at::Tensor weights,
    at::Tensor targets) {
  return std::make_unique<TensorProjectionErrorFunction<T>>(
      batchSize, nFrames, projections, parents, offsets, weights, targets);
}

template std::unique_ptr<TensorErrorFunction<float>> createProjectionErrorFunction<float>(
    size_t batchSize,
    size_t nFrames,
    at::Tensor projections,
    at::Tensor parents,
    at::Tensor offsets,
    at::Tensor weights,
    at::Tensor targets);

template std::unique_ptr<TensorErrorFunction<double>> createProjectionErrorFunction<double>(
    size_t batchSize,
    size_t nFrames,
    at::Tensor projections,
    at::Tensor parents,
    at::Tensor offsets,
    at::Tensor weights,
    at::Tensor targets);

} // namespace pymomentum
