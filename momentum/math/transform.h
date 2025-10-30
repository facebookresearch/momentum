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

/// Lightweight transform with separate rotation, translation, and uniform scaling
///
/// This class provides a more efficient alternative to Eigen's Affine3 for transformations
/// that only involve rotation, translation, and uniform scaling. It stores these components
/// separately, which allows for more efficient operations and memory usage.
///
/// @tparam T The scalar type (float or double)
template <typename T>
struct TransformT {
  Quaternion<T> rotation;
  Vector3<T> translation;
  T scale;

  /// Creates a transform with only rotation
  [[nodiscard]] static TransformT<T> makeRotation(const Quaternion<T>& rotation_in);

  /// Creates a transform with only translation
  [[nodiscard]] static TransformT<T> makeTranslation(const Vector3<T>& translation_in);

  /// Creates a transform with only scaling
  [[nodiscard]] static TransformT<T> makeScale(const T& scale_in);

  /// Creates a random transform
  ///
  /// @param[in] translation If true, use random translation (-1 to 1)
  /// @param[in] rotation If true, use random rotation
  /// @param[in] scale If true, use random scale (0.5 to 2.0)
  [[nodiscard]] static TransformT<T>
  makeRandom(bool translation = true, bool rotation = true, bool scale = true);

  /// Converts from Eigen's Affine3
  [[nodiscard]] static TransformT<T> fromAffine3(const Affine3<T>& other);

  /// Converts from 4x4 matrix
  [[nodiscard]] static TransformT<T> fromMatrix(const Matrix4<T>& other);

  /// Creates identity transform
  TransformT() : rotation(Quaternion<T>::Identity()), translation(Vector3<T>::Zero()), scale(1) {
    // Empty
  }

  /// Copy constructor with type conversion
  ///
  /// @tparam T2 Source scalar type
  template <typename T2>
  explicit TransformT(const TransformT<T2>& other)
      : rotation(other.rotation.template cast<T>()),
        translation(other.translation.template cast<T>()),
        scale(other.scale) {
    // Empty
  }

  /// Constructor with components
  explicit TransformT(
      const Vector3<T>& translation_in,
      const Quaternion<T>& rotation_in = Quaternion<T>::Identity(),
      const T& scale_in = T(1))
      : rotation(rotation_in), translation(translation_in), scale(scale_in) {
    // Empty
  }

  /// Move constructor
  explicit TransformT(
      Vector3<T>&& translation_in,
      Quaternion<T>&& rotation_in = Quaternion<T>::Identity(),
      T&& scale_in = T(1))
      : rotation(std::move(rotation_in)),
        translation(std::move(translation_in)),
        scale(std::move(scale_in)) {
    // Empty
  }

  /// Constructor from Affine3
  explicit TransformT(const Affine3<T>& other) {
    *this = fromAffine3(other);
  }

  /// Constructor from 4x4 matrix
  explicit TransformT(const Matrix4<T>& other) {
    *this = fromMatrix(other);
  }

  /// Assignment from Affine3
  TransformT<T>& operator=(const Affine3<T>& other) {
    *this = fromAffine3(other);
    return *this;
  }

  /// Assignment from 4x4 matrix
  TransformT<T>& operator=(const Matrix4<T>& other) {
    *this = fromMatrix(other);
    return *this;
  }

  /// Combines transforms (applies other first, then this)
  [[nodiscard]] TransformT<T> operator*(const TransformT<T>& other) const {
    // [ s_1*R_1 t_1 ] * [ s_2*R_2 t_2 ] = [ s_1*s_2*R_1*R_2  s_1*R_1*t_2 + t_1 ]
    // [     0    1  ]   [     0    1  ]   [        0                     1     ]
    const Vector3<T> trans = translation + rotation * (scale * other.translation);
    const Quaternion<T> rot = rotation * other.rotation;
    const T newScale = scale * other.scale;
    return TransformT<T>(std::move(trans), std::move(rot), std::move(newScale));
  }

  /// Multiplies with Affine3
  [[nodiscard]] Affine3<T> operator*(const Affine3<T>& other) const {
    Affine3<T> out = Affine3<T>::Identity();
    out.linear().noalias() = toLinear() * other.linear();
    out.translation().noalias() = rotation * (scale * other.translation()) + translation;
    return out;
  }

  /// Transforms a point
  [[nodiscard]] Vector3<T> operator*(const Vector3<T>& other) const {
    return transformPoint(other);
  }

  /// Converts to Affine3
  explicit operator Affine3<T>() const {
    return toAffine3();
  }

  /// Converts to Affine3
  [[nodiscard]] Affine3<T> toAffine3() const;

  /// Converts to 4x4 matrix
  [[nodiscard]] Matrix4<T> toMatrix() const {
    Matrix4<T> out = Matrix4<T>::Zero();
    out.template topLeftCorner<3, 3>().noalias() = rotation.toRotationMatrix() * scale;
    out.template topRightCorner<3, 1>() = translation;
    out(3, 3) = T(1);
    return out;
  }

  /// Returns rotation as 3x3 matrix
  [[nodiscard]] Matrix3<T> toRotationMatrix() const {
    return rotation.toRotationMatrix();
  }

  /// Returns rotation*scale as 3x3 matrix
  [[nodiscard]] Matrix3<T> toLinear() const {
    return toRotationMatrix() * scale;
  }

  /// Gets the X axis of the rotation
  [[nodiscard]] Vector3<T> getAxisX() const {
    // Optimized version of Eigen::Quaternion::toRotationMatrix()
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

  /// Applies full transform to a point
  [[nodiscard]] Vector3<T> transformPoint(const Vector3<T>& pt) const {
    return translation + rotation * (scale * pt).eval();
  }

  /// Applies only rotation to a vector
  [[nodiscard]] Vector3<T> rotate(const Vector3<T>& vec) const;

  /// Computes inverse transform
  [[nodiscard]] TransformT<T> inverse() const;

  /// Converts to different scalar type
  ///
  /// @tparam T2 Target scalar type
  template <typename T2>
  [[nodiscard]] TransformT<T2> cast() const {
    return TransformT<T2>(
        this->translation.template cast<T2>(),
        this->rotation.template cast<T2>(),
        static_cast<T2>(this->scale));
  }

  /// Checks if this transform is approximately equal to another
  [[nodiscard]] bool isApprox(
      const TransformT<T>& other,
      T tol = Eigen::NumTraits<T>::dummy_precision()) const {
    // TODO: Improve
    return toAffine3().isApprox(other.toAffine3(), tol);
  }
};

template <typename T>
using TransformListT =
    std::vector<TransformT<T>>; // structure describing a the state of all joints in a skeleton

template <typename T>
TransformT<T> blendTransforms(
    std::span<const TransformT<T>> transforms,
    std::span<const T> weights);

/// Spherical linear interpolation between two transforms
template <typename T>
TransformT<T> slerp(const TransformT<T>& t1, const TransformT<T>& t2, T weight);

using Transform = TransformT<float>;
using TransformList = TransformListT<float>;

} // namespace momentum
