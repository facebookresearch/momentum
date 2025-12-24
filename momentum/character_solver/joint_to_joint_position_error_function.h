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

/// Constraint data for joint-to-joint position error between two joints.
///
/// This constraint measures the position of a point attached to the source joint
/// in the coordinate frame of the reference joint, and compares it to a target
/// position (also expressed in the reference joint's coordinate frame).
template <typename T>
struct JointToJointPositionDataT {
  /// Index of the source joint (the joint whose position we want to constrain).
  size_t sourceJoint = kInvalidIndex;

  /// Offset from sourceJoint in the local coordinate system of sourceJoint.
  Vector3<T> sourceOffset = Vector3<T>::Zero();

  /// Index of the reference joint (the joint whose coordinate frame we use).
  size_t referenceJoint = kInvalidIndex;

  /// Offset from referenceJoint in the local coordinate system of referenceJoint.
  /// This defines the origin of the reference frame.
  Vector3<T> referenceOffset = Vector3<T>::Zero();

  /// Target position in the reference joint's coordinate frame.
  Vector3<T> target = Vector3<T>::Zero();

  /// Weight for this constraint.
  T weight = T(1);

  /// Name of the constraint (for debugging).
  std::string name = {};

  template <typename T2>
  JointToJointPositionDataT<T2> cast() const {
    return {
        this->sourceJoint,
        this->sourceOffset.template cast<T2>(),
        this->referenceJoint,
        this->referenceOffset.template cast<T2>(),
        this->target.template cast<T2>(),
        static_cast<T2>(this->weight),
        this->name};
  }
};

/// Error function that penalizes deviation from a target position expressed in
/// a reference joint's coordinate frame.
///
/// This is useful for constraints where you want to control the relative position
/// between two body parts, such as keeping a hand at a specific position relative
/// to the body, or maintaining a specific relationship between two joints that
/// should move together.
///
/// The error is computed as:
///   error = T_ref^{-1} * (T_src * src_offset) - target
///
/// where T_ref and T_src are the global transformations of the reference and
/// source joints respectively.
template <typename T>
class JointToJointPositionErrorFunctionT : public SkeletonErrorFunctionT<T> {
 public:
  explicit JointToJointPositionErrorFunctionT(const Skeleton& skel, const ParameterTransform& pt);

  explicit JointToJointPositionErrorFunctionT(const Character& character);

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

  /// Add a joint-to-joint position constraint.
  ///
  /// @param[in] sourceJoint Index of the source joint.
  /// @param[in] sourceOffset Offset from sourceJoint in local coordinates.
  /// @param[in] referenceJoint Index of the reference joint.
  /// @param[in] referenceOffset Offset from referenceJoint in local coordinates.
  /// @param[in] target Target position in the reference joint's coordinate frame.
  /// @param[in] weight Weight for this constraint.
  /// @param[in] name Name of the constraint (for debugging).
  void addConstraint(
      size_t sourceJoint,
      const Vector3<T>& sourceOffset,
      size_t referenceJoint,
      const Vector3<T>& referenceOffset,
      const Vector3<T>& target,
      T weight = T(1),
      const std::string& name = {});

  /// Add a constraint using the JointToJointPositionDataT struct.
  void addConstraint(const JointToJointPositionDataT<T>& constraint);

  /// Clear all constraints.
  void clearConstraints();

  /// Get all constraints.
  [[nodiscard]] const std::vector<JointToJointPositionDataT<T>>& getConstraints() const {
    return constraints_;
  }

  /// Get number of constraints.
  [[nodiscard]] size_t numConstraints() const {
    return constraints_.size();
  }

 private:
  std::vector<JointToJointPositionDataT<T>> constraints_;

  /// Intermediate storage for joint gradients.
  VectorX<T> jointGrad_;
};

} // namespace momentum
