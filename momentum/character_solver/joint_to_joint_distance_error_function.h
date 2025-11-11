/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character_solver/fwd.h>
#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/math/types.h>

namespace momentum {

/// Constraint that enforces a distance relationship between two points attached to different
/// joints.
///
/// Each point is specified by a joint index and an offset in the local coordinate system of that
/// joint.
template <typename T>
struct JointToJointDistanceConstraintT {
  /// Index of the first joint.
  size_t joint1 = kInvalidIndex;

  /// Offset from joint1 in the local coordinate system of joint1.
  Vector3<T> offset1 = Vector3<T>::Zero();

  /// Index of the second joint.
  size_t joint2 = kInvalidIndex;

  /// Offset from joint2 in the local coordinate system of joint2.
  Vector3<T> offset2 = Vector3<T>::Zero();

  /// Target distance between the two points (in world space).
  T targetDistance = T(0);

  /// Weight for this constraint.
  T weight = T(1);

  template <typename T2>
  JointToJointDistanceConstraintT<T2> cast() const {
    return {
        this->joint1,
        this->offset1.template cast<T2>(),
        this->joint2,
        this->offset2.template cast<T2>(),
        static_cast<T2>(this->targetDistance),
        static_cast<T2>(this->weight)};
  }
};

/// Error function that penalizes deviation from a target distance between two points attached to
/// different joints.
///
/// This is useful for enforcing distance constraints between different parts of a character, such
/// as maintaining a fixed distance between hands or ensuring two joints stay a certain distance
/// apart.
template <typename T>
class JointToJointDistanceErrorFunctionT : public SkeletonErrorFunctionT<T> {
 public:
  explicit JointToJointDistanceErrorFunctionT(const Skeleton& skel, const ParameterTransform& pt);

  explicit JointToJointDistanceErrorFunctionT(const Character& character);

  [[nodiscard]] double getError(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState) final;

  double getGradient(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Ref<VectorX<T>> gradient) override;

  double getJacobian(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Ref<MatrixX<T>> jacobian,
      Ref<VectorX<T>> residual,
      int& usedRows) override;

  [[nodiscard]] size_t getJacobianSize() const final;

  /// Add a constraint between two joints with local offsets.
  void addConstraint(
      size_t joint1,
      const Vector3<T>& offset1,
      size_t joint2,
      const Vector3<T>& offset2,
      T targetDistance,
      T weight = T(1));

  /// Clear all constraints.
  void clearConstraints();

  /// Get all constraints.
  [[nodiscard]] const std::vector<JointToJointDistanceConstraintT<T>>& getConstraints() const {
    return constraints_;
  }

  /// Default weight for distance constraints.
  static constexpr T kDistanceWeight = 1e-2f;

 private:
  std::vector<JointToJointDistanceConstraintT<T>> constraints_;
};

} // namespace momentum
