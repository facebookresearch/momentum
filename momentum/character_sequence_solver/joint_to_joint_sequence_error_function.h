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
#include <momentum/math/types.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace momentum {

/// Constraint data for the joint-to-joint sequence error function.
///
/// Describes a pair of joints whose relative transform should remain stable
/// across consecutive frames.
template <typename T>
struct JointToJointSequenceConstraintT {
  /// Index of the source joint (the joint whose relative pose we track).
  size_t sourceJoint = kInvalidIndex;

  /// Index of the reference joint (the frame of reference).
  size_t referenceJoint = kInvalidIndex;

  /// Weight for this constraint.
  T weight = T(1);

  /// Name of the constraint (for debugging).
  std::string name = {};
};

/// Error function that penalizes frame-to-frame changes in the relative
/// transform between two joints.
///
/// For each constraint with source joint A and reference joint B, computes the
/// relative transform T_rel = T_B^{-1} * T_A at two consecutive frames. This
/// expresses the source joint frame in the reference joint frame. It then
/// penalizes changes in relative translation and relative rotation:
///
///   error = weight * (
///       positionWeight * ||T_rel_{i+1}.translation - T_rel_i.translation||^2
///     + orientationWeight * ||R(T_rel_{i+1}) - R(T_rel_i)||_F^2)
///
/// The source/reference direction is intentionally fixed; reversing the pair
/// expresses the residual in the other joint's frame and can produce a different
/// translation error when the relative orientation or reference scale changes.
///
/// @tparam T Scalar type (float or double)
template <typename T>
class JointToJointSequenceErrorFunctionT : public SequenceErrorFunctionT<T> {
 public:
  JointToJointSequenceErrorFunctionT(const Skeleton& skel, const ParameterTransform& pt);
  explicit JointToJointSequenceErrorFunctionT(const Character& character);

  [[nodiscard]] size_t numFrames() const final {
    return 2;
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

  /// Add a constraint using the struct.
  void addConstraint(const JointToJointSequenceConstraintT<T>& constraint);

  /// Add a constraint with individual parameters.
  void addConstraint(
      size_t sourceJoint,
      size_t referenceJoint,
      T weight = T(1),
      const std::string& name = {});

  /// Clear all constraints.
  void clearConstraints();

  /// Get all constraints.
  [[nodiscard]] const std::vector<JointToJointSequenceConstraintT<T>>& getConstraints() const {
    return constraints_;
  }

  /// Returns the number of constraints.
  [[nodiscard]] size_t getNumConstraints() const {
    return constraints_.size();
  }

  void setPositionWeight(T w) {
    posWgt_ = w;
  }
  void setOrientationWeight(T w) {
    rotWgt_ = w;
  }

  [[nodiscard]] T getPositionWeight() const {
    return posWgt_;
  }
  [[nodiscard]] T getOrientationWeight() const {
    return rotWgt_;
  }

 private:
  std::vector<JointToJointSequenceConstraintT<T>> constraints_;
  T posWgt_ = T(1);
  T rotWgt_ = T(1);
};

} // namespace momentum
