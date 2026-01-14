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

/// Error function that penalizes the deviation of per-joint velocity magnitude from a target speed.
///
/// Given two consecutive frames, this error function computes the velocity for each joint as:
///   velocity = pos[t+1] - pos[t]
///
/// The residual for each joint is the difference between the velocity magnitude and a target speed:
///   residual = ||velocity|| - targetSpeed
///
/// This is useful for constraining character motion to maintain a specific speed profile, such as
/// ensuring that joints move at a consistent pace during walking or running animations.
///
/// Note: This error function only constrains position velocity magnitude, not rotation velocity.
/// When the velocity is very small (near zero), a small epsilon is added to avoid numerical issues
/// in the gradient computation.
template <typename T>
class VelocityMagnitudeSequenceErrorFunctionT : public SequenceErrorFunctionT<T> {
 public:
  VelocityMagnitudeSequenceErrorFunctionT(const Skeleton& skel, const ParameterTransform& pt);
  explicit VelocityMagnitudeSequenceErrorFunctionT(const Character& character);

  [[nodiscard]] size_t numFrames() const final {
    return 2;
  }

  [[nodiscard]] double getError(
      std::span<const ModelParametersT<T>> modelParameters,
      std::span<const SkeletonStateT<T>> skelStates,
      std::span<const MeshStateT<T>> meshStates) const final;

  [[nodiscard]] double getGradient(
      std::span<const ModelParametersT<T>> modelParameters,
      std::span<const SkeletonStateT<T>> skelStates,
      std::span<const MeshStateT<T>> meshStates,
      Eigen::Ref<Eigen::VectorX<T>> gradient) const final;

  double getJacobian(
      std::span<const ModelParametersT<T>> modelParameters,
      std::span<const SkeletonStateT<T>> skelStates,
      std::span<const MeshStateT<T>> meshStates,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      Eigen::Ref<Eigen::VectorX<T>> residual,
      int& usedRows) const final;

  [[nodiscard]] size_t getJacobianSize() const final;

  /// Set the per-joint weights for the velocity magnitude error.
  /// @param weights Per-joint weights vector. Size must match the number of joints.
  void setTargetWeights(const Eigen::VectorX<T>& weights);

  /// Set a single target speed applied to all joints.
  /// @param speed The target speed (velocity magnitude) for all joints.
  void setTargetSpeed(T speed);

  /// Set per-joint target speeds.
  /// @param speeds Vector of target speeds, one per joint.
  void setTargetSpeeds(const Eigen::VectorX<T>& speeds);

  /// Reset weights to ones and target speeds to zero.
  void reset();

  [[nodiscard]] const Eigen::VectorX<T>& getTargetWeights() const {
    return targetWeights_;
  }

  [[nodiscard]] const Eigen::VectorX<T>& getTargetSpeeds() const {
    return targetSpeeds_;
  }

 private:
  Eigen::VectorX<T> targetWeights_;
  Eigen::VectorX<T> targetSpeeds_;
};

} // namespace momentum
