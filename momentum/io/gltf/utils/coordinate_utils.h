/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/math/constants.h>
#include <momentum/math/types.h>

namespace momentum {

// Utility functions to convert gltf coordinate system to momentum coordinate system.
// #TODO: load coordinates by setting up CoordinateSystems so we don't need to worry about
// conversion.
[[nodiscard]] inline Vector3f toMomentumVec3f(const std::array<float, 3>& vec) noexcept {
  return Vector3f(vec[0], vec[1], vec[2]) * toCm();
}

[[nodiscard]] inline Vector3f fromMomentumVec3f(const Vector3f& vec) noexcept {
  return vec * toM();
}

[[nodiscard]] inline Quaternionf toMomentumQuaternionf(
    const std::array<float, 4>& gltfQuat) noexcept {
  return {gltfQuat[3], gltfQuat[0], gltfQuat[1], gltfQuat[2]};
}

[[nodiscard]] inline std::array<float, 4> fromMomentumQuaternionf(
    const Quaternionf& quat) noexcept {
  return {quat.x(), quat.y(), quat.z(), quat.w()};
}

inline void toMomentumVec3f(std::vector<Vector3f>& vec) {
  if (vec.empty()) {
    return;
  }

  Map<VectorXf>(&vec[0][0], vec.size() * 3) *= toCm();
}

inline void fromMomentumVec3f(std::vector<Vector3f>& vec) {
  if (vec.empty()) {
    return;
  }

  Map<VectorXf>(&vec[0][0], vec.size() * 3) *= toM();
}

inline void fromMomentumVec3f(VectorXf& vec) {
  if (vec.size() == 0) {
    return;
  }

  vec *= toM();
}

} // namespace momentum
