/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/math/coordinate_system.h>

namespace momentum {

/// The glTF coordinate system: Y-up, right-handed, meters.
inline constexpr CoordinateSystem kGltfCoordinateSystem{
    UpAxis::Y,
    Handedness::Right,
    LengthUnit::Meter};

// --- Vector conversions (coordinate system change: meters <-> centimeters) ---

[[nodiscard]] inline Vector3f toMomentumVec3f(const std::array<float, 3>& vec) noexcept {
  return toMomentumVector(Vector3f(vec[0], vec[1], vec[2]), kGltfCoordinateSystem);
}

[[nodiscard]] inline Vector3f fromMomentumVec3f(const Vector3f& vec) noexcept {
  return fromMomentumVector(vec, kGltfCoordinateSystem);
}

inline void toMomentumVec3f(std::vector<Vector3f>& vec) {
  if (vec.empty()) {
    return;
  }
  const auto s = scaleFactor<float>(kGltfCoordinateSystem, kMomentumCoordinateSystem);
  Map<VectorXf>(&vec[0][0], vec.size() * 3) *= s;
}

inline void fromMomentumVec3f(std::vector<Vector3f>& vec) {
  if (vec.empty()) {
    return;
  }
  const auto s = scaleFactor<float>(kMomentumCoordinateSystem, kGltfCoordinateSystem);
  Map<VectorXf>(&vec[0][0], vec.size() * 3) *= s;
}

inline void fromMomentumVec3f(VectorXf& vec) {
  if (vec.size() == 0) {
    return;
  }
  const auto s = scaleFactor<float>(kMomentumCoordinateSystem, kGltfCoordinateSystem);
  vec *= s;
}

// --- Quaternion component reordering (NOT a coordinate system operation) ---
// glTF stores quaternions as [x, y, z, w], Eigen stores as [w, x, y, z].

[[nodiscard]] inline Quaternionf toMomentumQuaternionf(
    const std::array<float, 4>& gltfQuat) noexcept {
  return {gltfQuat[3], gltfQuat[0], gltfQuat[1], gltfQuat[2]};
}

[[nodiscard]] inline std::array<float, 4> fromMomentumQuaternionf(
    const Quaternionf& quat) noexcept {
  return {quat.x(), quat.y(), quat.z(), quat.w()};
}

} // namespace momentum
