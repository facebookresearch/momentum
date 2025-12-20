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

/// Error function that penalizes the acceleration of joint positions across three consecutive
/// frames using a standard finite difference stencil [-1, 2, -1].
///
/// The acceleration residual for each joint is computed as:
///   accel = pos[t+1] - 2*pos[t] + pos[t-1] - targetAcceleration
///
/// This is useful for ballistic motion constraints where joints should follow a specific
/// acceleration (e.g., gravity). By default, the target acceleration is zero, which penalizes
/// any non-zero acceleration (smoothness constraint on velocity).
///
/// Note: This error function only constrains position acceleration, not rotation acceleration,
/// as rotation acceleration involves significantly more complex mathematics.
template <typename T>
class AccelerationSequenceErrorFunctionT : public SequenceErrorFunctionT<T> {
 public:
  AccelerationSequenceErrorFunctionT(const Skeleton& skel, const ParameterTransform& pt);
  explicit AccelerationSequenceErrorFunctionT(const Character& character);

  [[nodiscard]] size_t numFrames() const final {
    return 3;
  }

  double getError(
      std::span<const ModelParametersT<T>> modelParameters,
      std::span<const SkeletonStateT<T>> skelStates,
      std::span<const MeshStateT<T>> meshStates) const final;

  double getGradient(
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

  /// Set the per-joint weights for the acceleration error.
  /// @param weights Per-joint weights vector. Size must match the number of joints.
  void setTargetWeights(const Eigen::VectorX<T>& weights);

  /// Set a single target acceleration applied to all joints.
  /// This is a convenience method for uniform acceleration like gravity.
  /// @param acceleration The target acceleration vector (e.g., (0, -9.8, 0) * dt^2 for gravity).
  void setTargetAcceleration(const Eigen::Vector3<T>& acceleration);

  /// Set per-joint target accelerations.
  /// @param accelerations Vector of target accelerations, one per joint.
  void setTargetAccelerations(const std::vector<Eigen::Vector3<T>>& accelerations);

  /// Reset weights to ones and target accelerations to zero.
  void reset();

  [[nodiscard]] const Eigen::VectorX<T>& getTargetWeights() const {
    return targetWeights_;
  }

  [[nodiscard]] const std::vector<Eigen::Vector3<T>>& getTargetAccelerations() const {
    return targetAccelerations_;
  }

 private:
  Eigen::VectorX<T> targetWeights_;
  std::vector<Eigen::Vector3<T>> targetAccelerations_;
};

} // namespace momentum
