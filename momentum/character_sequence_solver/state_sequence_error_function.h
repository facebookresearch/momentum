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
#include <momentum/character_solver/state_error_function.h>

namespace momentum {

/// Error function that penalizes the difference between the current skeleton state and the next
/// skeleton state. Note that by default this penalizes any difference between adjacent skeleton
/// states. If we specify a target state, this will be applied as an offset to the current state,
/// that is, the error will penalize the difference between offset * currentState and nextState.
template <typename T>
class StateSequenceErrorFunctionT : public SequenceErrorFunctionT<T> {
 public:
  StateSequenceErrorFunctionT(
      const Skeleton& skel,
      const ParameterTransform& pt,
      RotationErrorType rotationErrorType = RotationErrorType::RotationMatrixDifference);
  explicit StateSequenceErrorFunctionT(
      const Character& character,
      RotationErrorType rotationErrorType = RotationErrorType::RotationMatrixDifference);

  [[nodiscard]] size_t numFrames() const final {
    return 2;
  }

  double getError(
      std::span<const ModelParametersT<T>> modelParameters,
      std::span<const SkeletonStateT<T>> skelStates,
      std::span<const MeshStateT<T>> /* meshStates */) const final;
  double getGradient(
      std::span<const ModelParametersT<T>> modelParameters,
      std::span<const SkeletonStateT<T>> skelStates,
      std::span<const MeshStateT<T>> /* meshStates */,
      Eigen::Ref<Eigen::VectorX<T>> gradient) const final;

  // modelParameters: [numFrames() * parameterTransform] parameter vector
  // skelStates: [numFrames()] array of skeleton states
  // jacobian: [getJacobianSize()] x [numFrames() * parameterTransform] Jacobian matrix
  // residual: [getJacobianSize()] residual vector.
  double getJacobian(
      std::span<const ModelParametersT<T>> modelParameters,
      std::span<const SkeletonStateT<T>> skelStates,
      std::span<const MeshStateT<T>> /* meshStates */,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      Eigen::Ref<Eigen::VectorX<T>> residual,
      int& usedRows) const final;

  [[nodiscard]] size_t getJacobianSize() const final;

  /// Set the target weights for each joint's position and rotation.
  /// @param posWeight Per-joint position weights.
  /// @param rotWeight Per-joint rotation weights.
  void setTargetWeights(const Eigen::VectorX<T>& posWeight, const Eigen::VectorX<T>& rotWeight);

  /// Set the target position weights for each joint.
  /// @param posWeight Per-joint position weights.
  void setPositionTargetWeights(const Eigen::VectorX<T>& posWeight);

  /// Set the target rotation weights for each joint.
  /// @param rotWeight Per-joint rotation weights.
  void setRotationTargetWeights(const Eigen::VectorX<T>& rotWeight);
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

  /// Set the target state for the skeleton.
  ///
  /// This function allows specifying a target state that the skeleton should aim to achieve.
  /// The target state is represented as a list of transforms, one for each joint.
  /// The error will penalize the difference between offset * currentState and nextState.
  /// @param target A list of transforms representing the target state for each joint.
  void setTargetState(TransformListT<T> target);

 private:
  Eigen::VectorX<T> targetPositionWeights_;
  Eigen::VectorX<T> targetRotationWeights_;

  TransformListT<T> targetState_;

  T posWgt_;
  T rotWgt_;

  const RotationErrorType rotationErrorType_;

 public:
  // weights for the error functions
  static constexpr T kPositionWeight = 1e-3f;
  static constexpr T kOrientationWeight = 1e+0f;
};

} // namespace momentum
