/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/math/types.h>

namespace momentum {

/// Specifies which axis points "up" in a 3D coordinate system.
enum class UpAxis {
  X = 0,
  Y,
  Z,
};

/// Specifies the handedness (chirality) of a coordinate system.
enum class Handedness {
  Left = 0,
  Right,
};

/// Specifies the unit of length measurement.
enum class LengthUnit {
  Meter = 0,
  Decimeter,
  Centimeter,
  Millimeter,
};

/// A 3D coordinate system defined by its up axis, handedness, and length unit.
/// Plain aggregate; all member combinations are valid.
struct CoordinateSystem {
  UpAxis up;
  Handedness hand;
  LengthUnit unit;
};

/// Momentum's canonical coordinate system: Y-up, right-handed, centimeters.
inline constexpr CoordinateSystem kMomentumCoordinateSystem{
    UpAxis::Y,
    Handedness::Right,
    LengthUnit::Centimeter};

/// Returns the length scale factor when converting a length from `from`'s unit to `to`'s unit.
/// For example, with `from.unit == Meter` and `to.unit == Centimeter` the result is 100.
/// Depends only on the `unit` fields of `from` and `to`; ignores `up` and `hand`.
template <typename T>
[[nodiscard]] T scaleFactor(const CoordinateSystem& from, const CoordinateSystem& to);

/// Converts a 3D position vector between coordinate systems.
/// Applies axis permutation, optional handedness flip, and unit scaling.
template <typename T>
[[nodiscard]] Vector3<T>
changeVector(const Vector3<T>& v, const CoordinateSystem& from, const CoordinateSystem& to);

/// Converts a quaternion rotation between coordinate systems.
/// Applies axis permutation (and optional handedness flip). No unit scaling — rotations are
/// dimensionless.
template <typename T>
[[nodiscard]] Quaternion<T>
changeQuaternion(const Quaternion<T>& q, const CoordinateSystem& from, const CoordinateSystem& to);

/// Converts a 3x3 rotation matrix between coordinate systems.
/// Computes `R_to = P * R_from * P^T` where `P` is the change-of-basis permutation (and reflection
/// for handedness flips) matrix. No unit scaling.
/// @pre `m` is a proper rotation matrix (orthonormal, det == +1).
template <typename T>
[[nodiscard]] Matrix3<T>
changeRotationMatrix(const Matrix3<T>& m, const CoordinateSystem& from, const CoordinateSystem& to);

// Convenience wrappers for converting to/from Momentum's canonical coordinate system
// (kMomentumCoordinateSystem: Y-up, right-handed, centimeters).

/// Converts a position vector from `from` into Momentum's canonical coordinate system.
template <typename T>
[[nodiscard]] Vector3<T> toMomentumVector(const Vector3<T>& v, const CoordinateSystem& from);

/// Converts a position vector from Momentum's canonical coordinate system into `to`.
template <typename T>
[[nodiscard]] Vector3<T> fromMomentumVector(const Vector3<T>& v, const CoordinateSystem& to);

/// Converts a quaternion rotation from `from` into Momentum's canonical coordinate system.
template <typename T>
[[nodiscard]] Quaternion<T> toMomentumQuaternion(
    const Quaternion<T>& q,
    const CoordinateSystem& from);

/// Converts a quaternion rotation from Momentum's canonical coordinate system into `to`.
template <typename T>
[[nodiscard]] Quaternion<T> fromMomentumQuaternion(
    const Quaternion<T>& q,
    const CoordinateSystem& to);

/// Converts a 3x3 rotation matrix from `from` into Momentum's canonical coordinate system.
template <typename T>
[[nodiscard]] Matrix3<T> toMomentumRotationMatrix(
    const Matrix3<T>& m,
    const CoordinateSystem& from);

/// Converts a 3x3 rotation matrix from Momentum's canonical coordinate system into `to`.
template <typename T>
[[nodiscard]] Matrix3<T> fromMomentumRotationMatrix(
    const Matrix3<T>& m,
    const CoordinateSystem& to);

} // namespace momentum
