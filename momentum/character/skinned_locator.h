/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/skin_weights.h>
#include <momentum/character/types.h>
#include <momentum/math/utility.h>

namespace momentum {

/// A locator attached to multiple joints, with its runtime position determined by
/// blending the skinning transforms (LBS) of those joints.
///
/// Unlike a regular `Locator` (which is local to a single parent joint), a
/// `SkinnedLocator`'s position is defined relative to the character's rest pose.
/// This is intended to model mocap markers attached to the actor's skin: points on
/// e.g. the shoulder are more accurately modeled by blending multiple joint
/// transforms. The locator can also be constrained to slide along the mesh surface
/// using e.g. `SkinnedLocatorTriangleErrorFunction`.
struct SkinnedLocator {
  std::string name;

  /// Indices of the parent joints used for skinning. The final position is the
  /// `skinWeights`-weighted blend of transforms from each (valid) parent.
  Eigen::Matrix<uint32_t, kMaxSkinJoints, 1> parents;

  /// Skinning weight for each entry in `parents`. Typically sums to 1.0.
  Eigen::Matrix<float, kMaxSkinJoints, 1> skinWeights;

  /// Position in the character's rest-pose coordinate frame.
  Vector3f position = Vector3f::Zero();

  /// Influence weight when used as a constraint in error functions.
  float weight = 1.0f;

  /// Offset from the skin surface in centimeters, used when solving for body shape.
  /// Accounts for physical marker thickness (e.g. half the marker diameter).
  float skinOffset = 0.0f;

  SkinnedLocator(
      const std::string& name = "uninitialized",
      const Eigen::Matrix<uint32_t, kMaxSkinJoints, 1>& parents =
          Eigen::Matrix<uint32_t, kMaxSkinJoints, 1>::Zero(),
      const Eigen::Matrix<float, kMaxSkinJoints, 1>& skinWeights =
          Eigen::Matrix<float, kMaxSkinJoints, 1>::Zero(),
      const Vector3f& position = Vector3f::Zero(),
      const float weight = 1.0f,
      const float skinOffset = 0.0f)
      : name(name),
        parents(parents),
        skinWeights(skinWeights),
        position(position),
        weight(weight),
        skinOffset(skinOffset) {}

  /// Equality comparison; uses approximate (`isApprox`) comparison for floating-point fields.
  inline bool operator==(const SkinnedLocator& locator) const {
    return (
        (name == locator.name) && (parents == locator.parents) &&
        skinWeights.isApprox(locator.skinWeights) && position.isApprox(locator.position) &&
        isApprox(weight, locator.weight) && isApprox(skinOffset, locator.skinOffset));
  }
};

/// A collection of locators attached to a skeleton
using SkinnedLocatorList = std::vector<SkinnedLocator>;

} // namespace momentum
