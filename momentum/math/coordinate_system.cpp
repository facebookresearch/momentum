/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/math/coordinate_system.h"

namespace momentum {

namespace {

// Returns the scale factor in meters for a given LengthUnit.
template <typename T>
constexpr T unitInMeters(LengthUnit u) {
  switch (u) {
    case LengthUnit::Meter:
      return T(1);
    case LengthUnit::Decimeter:
      return T(0.1);
    case LengthUnit::Centimeter:
      return T(0.01);
    case LengthUnit::Millimeter:
      return T(0.001);
  }
  return T(1); // unreachable
}

// Computes the signed permutation matrix P that converts coordinates from
// system 'from' to system 'to': v_to = P * v_from.
//
// Each coordinate system defines a canonical assignment of semantic axes
// (right, forward, up) to world axes (x, y, z):
//   Y-up right-hand: right=+X, forward=-Z, up=+Y
//   Z-up right-hand: right=+X, forward=+Y, up=+Z
//   X-up right-hand: right=+Y, forward=+Z, up=+X
// For left-handed systems, the forward axis is negated.
//
// The change-of-basis matrix is: P = toAxes * fromAxes^T
// where each column of 'axes' maps a semantic axis to its world direction.
template <typename T>
Matrix3<T> permutationMatrix(const CoordinateSystem& from, const CoordinateSystem& to) {
  auto getAxes = [](UpAxis up, Handedness hand) -> Matrix3<T> {
    Matrix3<T> m;
    switch (up) {
      case UpAxis::Y:
        m.col(0) = Vector3<T>(1, 0, 0);
        m.col(1) = (hand == Handedness::Right) ? Vector3<T>(0, 0, -1) : Vector3<T>(0, 0, 1);
        m.col(2) = Vector3<T>(0, 1, 0);
        break;
      case UpAxis::Z:
        m.col(0) = Vector3<T>(1, 0, 0);
        m.col(1) = (hand == Handedness::Right) ? Vector3<T>(0, 1, 0) : Vector3<T>(0, -1, 0);
        m.col(2) = Vector3<T>(0, 0, 1);
        break;
      case UpAxis::X:
        m.col(0) = Vector3<T>(0, 1, 0);
        m.col(1) = (hand == Handedness::Right) ? Vector3<T>(0, 0, 1) : Vector3<T>(0, 0, -1);
        m.col(2) = Vector3<T>(1, 0, 0);
        break;
    }
    return m;
  };

  const Matrix3<T> fromAxes = getAxes(from.up, from.hand);
  const Matrix3<T> toAxes = getAxes(to.up, to.hand);
  return toAxes * fromAxes.transpose();
}

} // namespace

template <typename T>
T scaleFactor(const CoordinateSystem& from, const CoordinateSystem& to) {
  return unitInMeters<T>(from.unit) / unitInMeters<T>(to.unit);
}

template <typename T>
Vector3<T>
changeVector(const Vector3<T>& v, const CoordinateSystem& from, const CoordinateSystem& to) {
  const T s = scaleFactor<T>(from, to);
  const Matrix3<T> P = permutationMatrix<T>(from, to);
  return s * (P * v);
}

template <typename T>
Quaternion<T>
changeQuaternion(const Quaternion<T>& q, const CoordinateSystem& from, const CoordinateSystem& to) {
  const Matrix3<T> R = q.toRotationMatrix();
  const Matrix3<T> newR = changeRotationMatrix(R, from, to);
  return Quaternion<T>(newR);
}

template <typename T>
Matrix3<T> changeRotationMatrix(
    const Matrix3<T>& m,
    const CoordinateSystem& from,
    const CoordinateSystem& to) {
  const Matrix3<T> P = permutationMatrix<T>(from, to);
  return P * m * P.transpose();
}

// --- Convenience wrappers ---

template <typename T>
Vector3<T> toMomentumVector(const Vector3<T>& v, const CoordinateSystem& from) {
  return changeVector(v, from, kMomentumCoordinateSystem);
}

template <typename T>
Vector3<T> fromMomentumVector(const Vector3<T>& v, const CoordinateSystem& to) {
  return changeVector(v, kMomentumCoordinateSystem, to);
}

template <typename T>
Quaternion<T> toMomentumQuaternion(const Quaternion<T>& q, const CoordinateSystem& from) {
  return changeQuaternion(q, from, kMomentumCoordinateSystem);
}

template <typename T>
Quaternion<T> fromMomentumQuaternion(const Quaternion<T>& q, const CoordinateSystem& to) {
  return changeQuaternion(q, kMomentumCoordinateSystem, to);
}

template <typename T>
Matrix3<T> toMomentumRotationMatrix(const Matrix3<T>& m, const CoordinateSystem& from) {
  return changeRotationMatrix(m, from, kMomentumCoordinateSystem);
}

template <typename T>
Matrix3<T> fromMomentumRotationMatrix(const Matrix3<T>& m, const CoordinateSystem& to) {
  return changeRotationMatrix(m, kMomentumCoordinateSystem, to);
}

// --- Explicit template instantiations ---

template float scaleFactor<float>(const CoordinateSystem&, const CoordinateSystem&);
template double scaleFactor<double>(const CoordinateSystem&, const CoordinateSystem&);

template Vector3<float>
changeVector(const Vector3<float>&, const CoordinateSystem&, const CoordinateSystem&);
template Vector3<double>
changeVector(const Vector3<double>&, const CoordinateSystem&, const CoordinateSystem&);

template Quaternion<float>
changeQuaternion(const Quaternion<float>&, const CoordinateSystem&, const CoordinateSystem&);
template Quaternion<double>
changeQuaternion(const Quaternion<double>&, const CoordinateSystem&, const CoordinateSystem&);

template Matrix3<float>
changeRotationMatrix(const Matrix3<float>&, const CoordinateSystem&, const CoordinateSystem&);
template Matrix3<double>
changeRotationMatrix(const Matrix3<double>&, const CoordinateSystem&, const CoordinateSystem&);

template Vector3<float> toMomentumVector(const Vector3<float>&, const CoordinateSystem&);
template Vector3<double> toMomentumVector(const Vector3<double>&, const CoordinateSystem&);
template Vector3<float> fromMomentumVector(const Vector3<float>&, const CoordinateSystem&);
template Vector3<double> fromMomentumVector(const Vector3<double>&, const CoordinateSystem&);

template Quaternion<float> toMomentumQuaternion(const Quaternion<float>&, const CoordinateSystem&);
template Quaternion<double> toMomentumQuaternion(
    const Quaternion<double>&,
    const CoordinateSystem&);
template Quaternion<float> fromMomentumQuaternion(
    const Quaternion<float>&,
    const CoordinateSystem&);
template Quaternion<double> fromMomentumQuaternion(
    const Quaternion<double>&,
    const CoordinateSystem&);

template Matrix3<float> toMomentumRotationMatrix(const Matrix3<float>&, const CoordinateSystem&);
template Matrix3<double> toMomentumRotationMatrix(const Matrix3<double>&, const CoordinateSystem&);
template Matrix3<float> fromMomentumRotationMatrix(const Matrix3<float>&, const CoordinateSystem&);
template Matrix3<double> fromMomentumRotationMatrix(
    const Matrix3<double>&,
    const CoordinateSystem&);

} // namespace momentum
