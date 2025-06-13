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
class StateSequenceErrorFunctionT : public SequenceErrorFunctionT<T> {
 public:
  StateSequenceErrorFunctionT(const Skeleton& skel, const ParameterTransform& pt);
  explicit StateSequenceErrorFunctionT(const Character& character);

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

  void setTargetWeights(const Eigen::VectorX<T>& posWeight, const Eigen::VectorX<T>& rotWeight);
  void setWeights(const float posWeight, const float rotationWeight) {
    posWgt_ = posWeight;
    rotWgt_ = rotationWeight;
  }

  void reset();

  [[nodiscard]] const Eigen::VectorX<T>& getPositionWeights() const {
    return targetPositionWeights_;
  }
  [[nodiscard]] const Eigen::VectorX<T>& getRotationWeights() const {
    return targetRotationWeights_;
  }
  [[nodiscard]] const T& getPositionWeight() const {
    return posWgt_;
  }
  [[nodiscard]] const T& getRotationWeight() const {
    return rotWgt_;
  }

 private:
  Eigen::VectorX<T> targetPositionWeights_;
  Eigen::VectorX<T> targetRotationWeights_;

  T posWgt_;
  T rotWgt_;

 public:
  // weights for the error functions
  static constexpr T kPositionWeight = 1e-3f;
  static constexpr T kOrientationWeight = 1e+0f;
};

} // namespace momentum
