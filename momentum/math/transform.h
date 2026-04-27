/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/math/types.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace momentum {

/// Lightweight transform with separate rotation, translation, and uniform scaling.
///
/// Applied to a point as: `p' = translation + rotation * (scale * p)`.
///
/// @tparam T The scalar type (float or double).
///
/// @par Design rationale
/// `TransformT` stores rotation (quaternion), translation, and uniform scale separately
/// instead of as a 4x4 matrix. Preferred over `Eigen::Affine3` throughout Momentum because:
/// - Composition is faster: quaternion multiply + scaled translation is cheaper than 4x4
///   matrix multiply for uniform-scale transforms.
/// - Decomposition is free: extracting rotation, translation, or scale is a field access,
///   not a matrix decomposition.
/// - FK derivatives are simpler: the Jacobian of rotation, translation, and scale are
///   computed independently (see `JointStateT`).
///
/// Use `Eigen::Affine3` only at IO boundaries (FBX, glTF) where the format stores 4x4
/// matrices. Convert via `TransformT::fromAffine3()` / `toAffine3()`.
template <typename T>
struct TransformT {
  /// Unit quaternion encoding the rotation. Not enforced — callers are responsible for
  /// keeping it normalized; non-unit quaternions will scale the rotated vector.
  Quaternion<T> rotation;
  Vector3<T> translation;
  /// Uniform scale factor applied before rotation and translation.
  T scale;

  /// Creates a transform with the given rotation, identity translation, and unit scale.
  [[nodiscard]] static TransformT<T> makeRotation(const Quaternion<T>& rotation_in);

  /// Creates a transform with the given translation, identity rotation, and unit scale.
  [[nodiscard]] static TransformT<T> makeTranslation(const Vector3<T>& translation_in);

  /// Creates a transform with the given scale, identity rotation, and zero translation.
  [[nodiscard]] static TransformT<T> makeScale(const T& scale_in);

  /// Creates a random transform.
  ///
  /// @param[in] translation If true, sample translation uniformly in [-1, 1] per axis;
  ///   otherwise zero.
  /// @param[in] rotation If true, sample a uniformly random unit quaternion; otherwise
  ///   identity.
  /// @param[in] scale If true, sample scale uniformly in [0.5, 2.0]; otherwise 1.
  [[nodiscard]] static TransformT<T>
  makeRandom(bool translation = true, bool rotation = true, bool scale = true);

  /// Decomposes an `Eigen::Affine3` into rotation, translation, and uniform scale.
  /// @pre `other.linear()` is a uniformly scaled rotation (no shear, isotropic scale);
  /// the result is undefined for general affine transforms.
  [[nodiscard]] static TransformT<T> fromAffine3(const Affine3<T>& other);

  /// Decomposes a 4x4 homogeneous matrix into rotation, translation, and uniform scale.
  /// @pre The upper-left 3x3 block is a uniformly scaled rotation and the bottom row is
  /// `[0 0 0 1]`; the result is undefined otherwise.
  [[nodiscard]] static TransformT<T> fromMatrix(const Matrix4<T>& other);

  /// Creates the identity transform (zero translation, identity rotation, unit scale).
  TransformT() : rotation(Quaternion<T>::Identity()), translation(Vector3<T>::Zero()), scale(1) {}

  /// @tparam T2 Source scalar type.
  template <typename T2>
  explicit TransformT(const TransformT<T2>& other)
      : rotation(other.rotation.template cast<T>()),
        translation(other.translation.template cast<T>()),
        scale(other.scale) {}

  explicit TransformT(
      const Vector3<T>& translation_in,
      const Quaternion<T>& rotation_in = Quaternion<T>::Identity(),
      const T& scale_in = T(1))
      : rotation(rotation_in), translation(translation_in), scale(scale_in) {}

  explicit TransformT(
      Vector3<T>&& translation_in,
      Quaternion<T>&& rotation_in = Quaternion<T>::Identity(),
      T&& scale_in = T(1))
      : rotation(std::move(rotation_in)),
        translation(std::move(translation_in)),
        scale(std::move(scale_in)) {}

  /// @see fromAffine3 for preconditions.
  explicit TransformT(const Affine3<T>& other) {
    *this = fromAffine3(other);
  }

  /// @see fromMatrix for preconditions.
  explicit TransformT(const Matrix4<T>& other) {
    *this = fromMatrix(other);
  }

  /// @see fromAffine3 for preconditions.
  TransformT<T>& operator=(const Affine3<T>& other) {
    *this = fromAffine3(other);
    return *this;
  }

  /// @see fromMatrix for preconditions.
  TransformT<T>& operator=(const Matrix4<T>& other) {
    *this = fromMatrix(other);
    return *this;
  }

  /// Composes transforms: `(*this) * other` applies `other` first, then `*this`.
  ///
  /// Equivalent to multiplying the homogeneous matrices:
  ///     [ s_1*R_1 t_1 ] * [ s_2*R_2 t_2 ] = [ s_1*s_2*R_1*R_2  s_1*R_1*t_2 + t_1 ]
  ///     [     0    1  ]   [     0    1  ]   [        0                     1     ]
  [[nodiscard]] TransformT<T> operator*(const TransformT<T>& other) const {
    const Vector3<T> trans = translation + rotation * (scale * other.translation);
    const Quaternion<T> rot = rotation * other.rotation;
    const T newScale = scale * other.scale;
    return TransformT<T>(std::move(trans), std::move(rot), std::move(newScale));
  }

  /// Composes with an `Affine3`. The result is a general `Affine3` because `other` may
  /// contain shear or non-uniform scale that cannot be represented by a `TransformT`.
  [[nodiscard]] Affine3<T> operator*(const Affine3<T>& other) const {
    Affine3<T> out = Affine3<T>::Identity();
    out.linear().noalias() = toLinear() * other.linear();
    out.translation().noalias() = rotation * (scale * other.translation()) + translation;
    return out;
  }

  [[nodiscard]] Vector3<T> operator*(const Vector3<T>& other) const {
    return transformPoint(other);
  }

  explicit operator Affine3<T>() const {
    return toAffine3();
  }

  [[nodiscard]] Affine3<T> toAffine3() const;

  /// Returns the equivalent 4x4 homogeneous matrix `[s*R | t; 0 0 0 1]`.
  [[nodiscard]] Matrix4<T> toMatrix() const {
    Matrix4<T> out = Matrix4<T>::Zero();
    out.template topLeftCorner<3, 3>().noalias() = rotation.toRotationMatrix() * scale;
    out.template topRightCorner<3, 1>() = translation;
    out(3, 3) = T(1);
    return out;
  }

  /// Returns the rotation as a 3x3 orthogonal matrix (scale not included).
  [[nodiscard]] Matrix3<T> toRotationMatrix() const {
    return rotation.toRotationMatrix();
  }

  /// Returns `scale * R`, the linear part of the transform.
  [[nodiscard]] Matrix3<T> toLinear() const {
    return toRotationMatrix() * scale;
  }

  /// Returns the rotated unit-X axis (i.e. the first column of the rotation matrix).
  /// Inlined hand-expansion of `Eigen::Quaternion::toRotationMatrix()` for the X column;
  /// faster than constructing the full 3x3 matrix when only one axis is needed.
  /// @pre `rotation` is a unit quaternion; otherwise the returned vector is scaled.
  [[nodiscard]] Vector3<T> getAxisX() const {
    Vector3<T> axis;

    const T ty = T(2) * rotation.y();
    const T tz = T(2) * rotation.z();
    const T twy = ty * rotation.w();
    const T twz = tz * rotation.w();
    const T txy = ty * rotation.x();
    const T txz = tz * rotation.x();
    const T tyy = ty * rotation.y();
    const T tzz = tz * rotation.z();

    axis[0] = T(1) - (tyy + tzz);
    axis[1] = txy + twz;
    axis[2] = txz - twy;

    return axis;
  }

  /// Applies the full transform to a point: `translation + rotation * (scale * pt)`.
  [[nodiscard]] Vector3<T> transformPoint(const Vector3<T>& pt) const {
    return translation + rotation * (scale * pt).eval();
  }

  /// Applies only the rotation to a direction vector (translation and scale are ignored).
  [[nodiscard]] Vector3<T> rotate(const Vector3<T>& vec) const;

  /// Returns the inverse transform such that `(*this) * inverse() == identity`.
  /// @pre `scale != 0`.
  [[nodiscard]] TransformT<T> inverse() const;

  /// @tparam T2 Target scalar type.
  template <typename T2>
  [[nodiscard]] TransformT<T2> cast() const {
    return TransformT<T2>(
        this->translation.template cast<T2>(),
        this->rotation.template cast<T2>(),
        static_cast<T2>(this->scale));
  }

  /// Approximate equality test using `tol` as a relative precision.
  /// Currently routed through the `Affine3` representation; this conflates rotation,
  /// translation, and scale tolerances.
  // TODO: Compare components directly so the tolerance has well-defined meaning per
  // component (and so two quaternions that differ by sign compare equal).
  [[nodiscard]] bool isApprox(
      const TransformT<T>& other,
      T tol = Eigen::NumTraits<T>::dummy_precision()) const {
    return toAffine3().isApprox(other.toAffine3(), tol);
  }
};

/// Per-joint transform list describing the state of all joints in a skeleton (one entry
/// per joint, indexed by joint id).
template <typename T>
using TransformListT = std::vector<TransformT<T>>;

/// Weighted blend of transforms: rotations are blended via averaged quaternions,
/// translations and scales linearly. `transforms` and `weights` must have the same size.
template <typename T>
TransformT<T> blendTransforms(
    momentum::span<const TransformT<T>> transforms,
    momentum::span<const T> weights);

/// Spherical linear interpolation between two transforms.
/// @param weight Interpolation parameter in [0, 1]; 0 returns `t1`, 1 returns `t2`.
template <typename T>
TransformT<T> slerp(const TransformT<T>& t1, const TransformT<T>& t2, T weight);

using Transform = TransformT<float>;
using TransformList = TransformListT<float>;

} // namespace momentum
