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

/// Base class for error functions that penalize finite difference derivatives of joint positions
/// across consecutive frames using a configurable stencil.
///
/// This class provides a generic framework for computing derivatives of arbitrary order using
/// finite difference stencils. Derived classes specify the stencil coefficients to compute
/// specific derivatives:
/// - 2nd derivative (acceleration): [-1, 2, -1] stencil, 3 frames
/// - 3rd derivative (jerk): [1, -3, 3, -1] stencil, 4 frames
///
/// The residual for each joint is computed as:
///   derivative = sum(stencilCoefficients[i] * pos[t+i]) - targetValue
///
/// Note: This error function only constrains position derivatives, not rotation derivatives,
/// as rotation derivatives involve significantly more complex mathematics.
template <typename T>
class FiniteDifferenceSequenceErrorFunctionT : public SequenceErrorFunctionT<T> {
 public:
  FiniteDifferenceSequenceErrorFunctionT(
      const Skeleton& skel,
      const ParameterTransform& pt,
      const std::vector<T>& stencilCoefficients);
  explicit FiniteDifferenceSequenceErrorFunctionT(
      const Character& character,
      const std::vector<T>& stencilCoefficients);

  [[nodiscard]] size_t numFrames() const final {
    return stencilCoefficients_.size();
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

  /// Set the per-joint weights for the derivative error.
  /// @param weights Per-joint weights vector. Size must match the number of joints.
  void setTargetWeights(const Eigen::VectorX<T>& weights);

  /// Set a single target value applied to all joints.
  /// This is a convenience method for uniform targets (e.g., gravity for acceleration).
  /// @param targetValue The target derivative value (e.g., (0, -9.8, 0) * dt^2 for gravity
  /// acceleration).
  void setTargetValue(const Eigen::Vector3<T>& targetValue);

  /// Set per-joint target values.
  /// @param targetValues Vector of target derivative values, one per joint.
  void setTargetValues(const std::vector<Eigen::Vector3<T>>& targetValues);

  /// Reset weights to ones and target values to zero.
  void reset();

  [[nodiscard]] const Eigen::VectorX<T>& getTargetWeights() const {
    return targetWeights_;
  }

  [[nodiscard]] const std::vector<Eigen::Vector3<T>>& getTargetValues() const {
    return targetValues_;
  }

  [[nodiscard]] const std::vector<T>& getStencilCoefficients() const {
    return stencilCoefficients_;
  }

 private:
  std::vector<T> stencilCoefficients_;
  Eigen::VectorX<T> targetWeights_;
  std::vector<Eigen::Vector3<T>> targetValues_;
};

} // namespace momentum
