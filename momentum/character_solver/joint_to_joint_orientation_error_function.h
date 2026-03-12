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

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace momentum {

/// Constraint data for joint-to-joint orientation error between two joints.
///
/// This constraint measures the orientation of a source joint relative to a reference
/// joint, and compares it to a target orientation (expressed as a quaternion in the
/// reference joint's coordinate frame).
template <typename T>
struct JointToJointOrientationDataT {
  /// Index of the source joint (the joint whose orientation we want to constrain).
  size_t sourceJoint = kInvalidIndex;

  /// Index of the reference joint (the joint whose coordinate frame we use).
  size_t referenceJoint = kInvalidIndex;

  /// Target relative orientation as a quaternion (R_ref^T * R_src should match this).
  Eigen::Quaternion<T> target = Eigen::Quaternion<T>::Identity();

  /// Weight for this constraint.
  T weight = T(1);

  /// Name of the constraint (for debugging).
  std::string name = {};

  template <typename T2>
  JointToJointOrientationDataT<T2> cast() const {
    return {
        this->sourceJoint,
        this->referenceJoint,
        this->target.template cast<T2>(),
        static_cast<T2>(this->weight),
        this->name};
  }
};

/// Error function that penalizes deviation from a target orientation expressed
/// relative to a reference joint's coordinate frame.
///
/// This is useful for constraints where you want to control the relative orientation
/// between two body parts, such as keeping a hand at a specific orientation relative
/// to the body, or maintaining a specific rotational relationship between two joints.
///
/// The error is computed as the Frobenius norm of the difference between the actual
/// and target relative rotation matrices:
///   error = ||R_ref^T * R_src - R_target||_F^2
///
/// where R_ref and R_src are the global rotations of the reference and source joints
/// respectively, and R_target is the target relative rotation.
template <typename T>
class JointToJointOrientationErrorFunctionT : public SkeletonErrorFunctionT<T> {
 public:
  explicit JointToJointOrientationErrorFunctionT(
      const Skeleton& skel,
      const ParameterTransform& pt);

  explicit JointToJointOrientationErrorFunctionT(const Character& character);

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

  /// Add a joint-to-joint orientation constraint.
  ///
  /// @param[in] sourceJoint Index of the source joint.
  /// @param[in] referenceJoint Index of the reference joint.
  /// @param[in] target Target relative orientation (R_ref^T * R_src should match this).
  /// @param[in] weight Weight for this constraint.
  /// @param[in] name Name of the constraint (for debugging).
  void addConstraint(
      size_t sourceJoint,
      size_t referenceJoint,
      const Eigen::Quaternion<T>& target,
      T weight = T(1),
      const std::string& name = {});

  /// Add a constraint using the JointToJointOrientationDataT struct.
  void addConstraint(const JointToJointOrientationDataT<T>& constraint);

  /// Clear all constraints.
  void clearConstraints();

  /// Get all constraints.
  [[nodiscard]] const std::vector<JointToJointOrientationDataT<T>>& getConstraints() const {
    return constraints_;
  }

  /// Get number of constraints.
  [[nodiscard]] size_t numConstraints() const {
    return constraints_.size();
  }

 private:
  std::vector<JointToJointOrientationDataT<T>> constraints_;
};

} // namespace momentum
