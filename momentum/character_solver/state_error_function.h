/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/skeleton_state.h>
#include <momentum/character_solver/fwd.h>
#include <momentum/character_solver/skeleton_error_function.h>

namespace momentum {

/// Defines different methods for computing rotation error between two quaternions.
enum class RotationErrorType {
  /// Frobenius norm of rotation matrix difference: ||R1 - R2||_F^2
  ///
  /// This is the default method. It computes the squared Frobenius norm of the
  /// difference between two rotation matrices. While not a geodesic distance,
  /// it has smooth derivatives everywhere and is computationally efficient.
  RotationMatrixDifference,

  /// Logarithmic map of relative rotation: ||log(R1^{-1} * R2)||^2
  ///
  /// This method computes the squared norm of the logarithmic map of the relative
  /// rotation quaternion. This gives the squared geodesic distance on SO(3), which
  /// has a clear geometric interpretation. It uses numerically robust logmap
  /// computation with Taylor series for small angles.
  QuaternionLogMap,
};

template <typename T>
class StateErrorFunctionT : public SkeletonErrorFunctionT<T> {
 public:
  StateErrorFunctionT(
      const Skeleton& skel,
      const ParameterTransform& pt,
      RotationErrorType rotationErrorType = RotationErrorType::RotationMatrixDifference);
  explicit StateErrorFunctionT(
      const Character& character,
      RotationErrorType rotationErrorType = RotationErrorType::RotationMatrixDifference);

  [[nodiscard]] double getError(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState) final;

  double getGradient(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Eigen::Ref<Eigen::VectorX<T>> gradient) final;

  double getJacobian(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      Eigen::Ref<Eigen::VectorX<T>> residual,
      int& usedRows) final;

  [[nodiscard]] size_t getJacobianSize() const final;

  void reset();
  void setTargetState(const SkeletonStateT<T>* target);
  void setTargetState(const SkeletonStateT<T>& target);
  void setTargetState(TransformListT<T> target);
  void setTargetWeight(const Eigen::VectorX<T>& weights);
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

  [[nodiscard]] const TransformListT<T>& getTargetState() const {
    return this->targetState_;
  }

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
  TransformListT<T> targetState_;
  Eigen::VectorX<T> targetPositionWeights_;
  Eigen::VectorX<T> targetRotationWeights_;

  T posWgt_;
  T rotWgt_;

  const RotationErrorType rotationErrorType_;

 public:
  // weights for the error functions
  static constexpr T kPositionWeight = 1e-3f;
  static constexpr T kOrientationWeight = 1e+0f;
};

} // namespace momentum
