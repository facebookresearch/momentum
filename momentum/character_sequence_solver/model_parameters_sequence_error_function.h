/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/skeleton_state.h>
#include <momentum/character_sequence_solver/fwd.h>
#include <momentum/character_sequence_solver/sequence_error_function.h>

namespace momentum {

template <typename T>
class ModelParametersSequenceErrorFunctionT : public SequenceErrorFunctionT<T> {
 public:
  ModelParametersSequenceErrorFunctionT(const Skeleton& skel, const ParameterTransform& pt);
  ModelParametersSequenceErrorFunctionT(const Character& character);

  [[nodiscard]] size_t numFrames() const final {
    return 2;
  }

  double getError(
      gsl::span<const ModelParametersT<T>> modelParameters,
      gsl::span<const SkeletonStateT<T>> skelStates) const final;
  double getGradient(
      gsl::span<const ModelParametersT<T>> modelParameters,
      gsl::span<const SkeletonStateT<T>> skelStates,
      Eigen::Ref<Eigen::VectorX<T>> gradient) const final;

  // modelParameters: [numFrames() * parameterTransform] parameter vector
  // skelStates: [numFrames()] array of skeleton states
  // jacobian: [getJacobianSize()] x [numFrames() * parameterTransform] Jacobian matrix
  // residual: [getJacobianSize()] residual vector.
  double getJacobian(
      gsl::span<const ModelParametersT<T>> modelParameters,
      gsl::span<const SkeletonStateT<T>> skelStates,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      Eigen::Ref<Eigen::VectorX<T>> residual,
      int& usedRows) const final;

  [[nodiscard]] size_t getJacobianSize() const final;

  void setTargetWeights(const Eigen::VectorX<T>& weights) {
    this->targetWeights_ = weights;
  }

  [[nodiscard]] const Eigen::VectorX<T>& getTargetWeights() const {
    return this->targetWeights_;
  }

 private:
  Eigen::VectorX<T> targetWeights_;

 public:
  static constexpr T kMotionWeight = 1e-1;
};

} // namespace momentum
