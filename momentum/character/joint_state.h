/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/joint.h>
#include <momentum/character/types.h>
#include <momentum/math/transform.h>
#include <momentum/math/types.h>

namespace momentum {

/// Represents the state of a joint with local and global transformations.
///
/// Joint transform hierarchy is defined as:
///     WorldTransform = ParentWorldTransform * Tz * Ty * Tx * Rpre * R * S
///
/// Where:
/// - Tx, Ty, Tz are translations along each axis
/// - Rpre is the pre-rotation from the joint definition
/// - R is the joint rotation (applied in order: rz * ry * rx)
/// - S is the uniform scale factor
///
/// Each joint has 7 parameters: 3 translation, 3 rotation, and 1 scale.
///
/// The ordering has direct implications for derivative computation:
///
/// - **Translation Jacobian**: Because translation is applied in the parent's frame
///   (before local rotation), the translation derivative for axis `i` is simply
///   `translationAxis.col(i)` — the parent's rotated axis direction. It does NOT
///   depend on the joint's own rotation.
///
/// - **Rotation Jacobian**: Because rotation is applied after translation and
///   pre-rotation, the rotation derivative for a point `p` is
///   `rotationAxis.col(i).cross(p - jointPosition)`. The pre-rotation (`Rpre`) is
///   baked into `rotationAxis` and is not a free parameter.
///
/// - **Scale Jacobian**: Because scale is applied last (innermost), it scales the
///   offset from the joint position: `(p - jointPosition) * ln(2)` (since scale
///   is stored as `exp2(parameter[6])`).
///
/// Swapping the order (e.g., applying rotation before translation) would change
/// which frame the translation operates in, producing incorrect Jacobians in all
/// error functions.
template <typename T>
struct JointStateT {
  /// Local transformation relative to parent joint
  ///
  /// Defined by the joint parameters (translation, rotation, scale)
  TransformT<T> localTransform;

  /// Global transformation in world space
  ///
  /// Combines parent transformations with local transformation
  TransformT<T> transform;

  /// Translation axes in global space
  ///
  /// Each column represents one translation axis (X, Y, Z) in global coordinates
  Matrix3<T> translationAxis;

  /// Rotation axes in global space
  ///
  /// Each column represents one rotation axis (X, Y, Z) in global coordinates
  Matrix3<T> rotationAxis;

  /// Indicates if derivative data needs recomputation
  ///
  /// When true, derivative methods should not be called as they may return invalid results
  bool derivDirty = true;

  /// Updates joint state based on joint definition and parameters
  ///
  /// Recursively computes all transformations from parent to child, including
  /// local and global transforms, and optionally derivative information.
  ///
  /// @param joint The joint definition containing offset and pre-rotation
  /// @param parameters The 7 joint parameters [tx, ty, tz, rx, ry, rz, scale]
  /// @param parentState Optional parent joint state for hierarchical transformations
  /// @param computeDeriv If true, computes `translationAxis` and `rotationAxis`
  ///   matrices needed for Jacobian computation in error functions. Set to false
  ///   ONLY when the state is used for visualization, FK output inspection, or IO —
  ///   never when the state will be passed to a solver or error function.
  ///   Skipping derivatives saves ~30% of per-joint FK cost.
  ///   When false, `derivDirty` is set to true and derivative accessors
  ///   (`getRotationDerivative`, `getTranslationDerivative`, `getScaleDerivative`)
  ///   return invalid results.
  void set(
      const JointT<T>& joint,
      const JointVectorT<T>& parameters,
      const JointStateT<T>* parentState = nullptr,
      bool computeDeriv = true) noexcept;

  /// Calculates the derivative of a point with respect to rotation around a specific axis
  ///
  /// @param index The rotation axis index (0=X, 1=Y, 2=Z)
  /// @param ref The reference point in global space
  /// @return The derivative vector
  [[nodiscard]] Vector3<T> getRotationDerivative(size_t index, const Vector3<T>& ref) const;

  /// Calculates the derivative with respect to translation along a specific axis
  ///
  /// @param index The translation axis index (0=X, 1=Y, 2=Z)
  /// @return The derivative vector
  [[nodiscard]] Vector3<T> getTranslationDerivative(size_t index) const;

  /// Calculates the derivative of a point with respect to uniform scaling
  ///
  /// @param ref The reference point in global space
  /// @return The derivative vector
  [[nodiscard]] Vector3<T> getScaleDerivative(const Vector3<T>& ref) const noexcept;

  /// Copies state from another JointState with potentially different scalar type
  ///
  /// @tparam T2 Source scalar type
  /// @param rhs Source joint state to copy from
  template <typename T2>
  void set(const JointStateT<T2>& rhs);

  /// Access the joint's local rotation quaternion
  [[nodiscard]] const Quaternion<T>& localRotation() const {
    return localTransform.rotation;
  }

  /// Access the joint's local rotation quaternion (mutable)
  Quaternion<T>& localRotation() {
    return localTransform.rotation;
  }

  /// Access the joint's local translation vector
  [[nodiscard]] const Vector3<T>& localTranslation() const {
    return localTransform.translation;
  }

  /// Access the joint's local translation vector (mutable)
  Vector3<T>& localTranslation() {
    return localTransform.translation;
  }

  /// Access the joint's local scale factor
  ///
  /// This scale propagates to all descendant joints
  ///
  /// The scale parameter is stored as a log2 value and applied as `exp2(parameter[6])`.
  /// This means:
  /// - Parameter value 0.0 → scale = 1.0 (no scaling)
  /// - Parameter value 1.0 → scale = 2.0 (double size)
  /// - Parameter value -1.0 → scale = 0.5 (half size)
  /// - The Jacobian of scale involves `exp2(param) * ln(2)`, not 1.0
  ///
  /// This encoding keeps scale always positive and makes the parameter space
  /// more linear for optimization (uniform steps in log-space = uniform ratios).
  [[nodiscard]] const T& localScale() const {
    return localTransform.scale;
  }

  /// Access the joint's local scale factor (mutable)
  T& localScale() {
    return localTransform.scale;
  }

  /// Access the joint's global rotation quaternion
  [[nodiscard]] const Quaternion<T>& rotation() const {
    return transform.rotation;
  }

  /// Access the joint's global rotation quaternion (mutable)
  Quaternion<T>& rotation() {
    return transform.rotation;
  }

  /// Access the joint's global position vector
  [[nodiscard]] const Vector3<T>& translation() const {
    return transform.translation;
  }

  /// Access the joint's global position vector (mutable)
  Vector3<T>& translation() {
    return transform.translation;
  }

  /// Access the X component of the global position
  [[nodiscard]] const T& x() const {
    return transform.translation.x();
  }

  /// Access the X component of the global position (mutable)
  T& x() {
    return transform.translation.x();
  }

  /// Access the Y component of the global position
  [[nodiscard]] const T& y() const {
    return transform.translation.y();
  }

  /// Access the Y component of the global position (mutable)
  T& y() {
    return transform.translation.y();
  }

  /// Access the Z component of the global position
  [[nodiscard]] const T& z() const {
    return transform.translation.z();
  }

  /// Access the Z component of the global position (mutable)
  T& z() {
    return transform.translation.z();
  }

  /// Access the W component of the global rotation quaternion
  [[nodiscard]] const T& quatW() const {
    return transform.rotation.w();
  }

  /// Access the W component of the global rotation quaternion (mutable)
  T& quatW() {
    return transform.rotation.w();
  }

  /// Access the X component of the global rotation quaternion
  [[nodiscard]] const T& quatX() const {
    return transform.rotation.x();
  }

  /// Access the X component of the global rotation quaternion (mutable)
  T& quatX() {
    return transform.rotation.x();
  }

  /// Access the Y component of the global rotation quaternion
  [[nodiscard]] const T& quatY() const {
    return transform.rotation.y();
  }

  /// Access the Y component of the global rotation quaternion (mutable)
  T& quatY() {
    return transform.rotation.y();
  }

  /// Access the Z component of the global rotation quaternion
  [[nodiscard]] const T& quatZ() const {
    return transform.rotation.z();
  }

  /// Access the Z component of the global rotation quaternion (mutable)
  T& quatZ() {
    return transform.rotation.z();
  }

  /// Access the joint's global scale factor
  ///
  /// This is the cumulative scale from all parent joints
  [[nodiscard]] const T& scale() const {
    return transform.scale;
  }

  /// Access the joint's global scale factor (mutable)
  T& scale() {
    return transform.scale;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// Structure describing the state of all joints in a skeleton
template <typename T>
using JointStateListT = std::vector<JointStateT<T>>;

using JointState = JointStateT<float>;
using JointStateList = JointStateListT<float>;

} // namespace momentum
