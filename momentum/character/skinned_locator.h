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

/// A skinned locator is a locator which can be attached to multiple bones.
/// The locator's position is defined relative to the rest pose of the character (and not
/// local to a single parent bone as with a regular Locator) and its position at runtime is
/// determined by blending the skinning transforms (using LBS).
///
/// The purpose of the SkinnedLocator is to model mocap markers attached to the actor's skin:
/// the location of points on e.g. the shoulder is more accurately modeled by blending
/// multiple transforms.  In addition, the locator can be constrained to slide along the
/// surface of the mesh using e.g. the SkinnedLocatorTriangleErrorFunction.
///
/// Locators can be used for various purposes such as tracking specific points
/// on a character, defining constraints, or serving as targets for inverse kinematics.
struct SkinnedLocator {
  /// Name identifier for the locator
  std::string name;

  /// Index of the parent joints in the skeleton.  The final position of the locator
  /// is determined by blending the transforms from each (valid) parent.
  Eigen::Matrix<uint32_t, kMaxSkinJoints, 1> parents;

  /// Skinning weight for each parent joint.
  Eigen::Matrix<float, kMaxSkinJoints, 1> skinWeights;

  /// Position relative to rest pose of the character
  Vector3f position;

  /// Influence weight of this locator when used in constraints
  float weight;

  /// Creates a locator with the specified properties
  ///
  /// @param name Identifier for the locator
  /// @param parents Indices of the parent joints
  /// @param weights Skinning weights for the parent joints
  /// @param weight Influence weight in constraints
  SkinnedLocator(
      const std::string& name = "uninitialized",
      const Eigen::Matrix<uint32_t, kMaxSkinJoints, 1>& parents =
          Eigen::Matrix<uint32_t, kMaxSkinJoints, 1>::Zero(),
      const Eigen::Matrix<float, kMaxSkinJoints, 1>& skinWeights =
          Eigen::Matrix<float, kMaxSkinJoints, 1>::Zero(),
      const Vector3f& position = Vector3f::Zero(),
      const float weight = 1.0f)
      : name(name),
        parents(parents),
        skinWeights(skinWeights),
        position(position),
        weight(weight) {}

  /// Compares two locators for equality, using approximate comparison for floating-point values
  ///
  /// @param locator The locator to compare with
  /// @return True if all properties are equal (or approximately equal for floating-point values)
  inline bool operator==(const SkinnedLocator& locator) const {
    return (
        (name == locator.name) && (parents == locator.parents) &&
        skinWeights.isApprox(locator.skinWeights) && position.isApprox(locator.position) &&
        isApprox(weight, locator.weight));
  }
};

/// A collection of locators attached to a skeleton
using SkinnedLocatorList = std::vector<SkinnedLocator>;

} // namespace momentum
