/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/types.h>
#include <momentum/math/utility.h>

namespace momentum {

/// Represents a point attached to a joint in a skeleton.
///
/// Locators can be used for various purposes such as tracking specific points
/// on a character, defining constraints, or serving as targets for inverse kinematics.
struct Locator {
  std::string name;

  /// Index of the parent joint in the skeleton.
  size_t parent;

  /// Position in the parent joint's local coordinate frame.
  Vector3f offset;

  /// Per-axis lock flags (x, y, z): 1 = locked, 0 = free.
  Vector3i locked;

  /// Influence weight when used as a constraint in error functions.
  float weight;

  /// Reference position for limit constraints; typically equals `offset` when loaded.
  Vector3f limitOrigin;

  /// Per-axis strength pulling the locator back toward `limitOrigin`.
  /// Zero on an axis disables the limit on that axis.
  Vector3f limitWeight;

  /// True if the locator is attached to the skin of a person (e.g. as in mocap tracking);
  /// used to determine whether the locator can safely be converted to a skinned locator.
  bool attachedToSkin = false;

  /// Offset from the skin surface, used when solving for body shape using locators.
  float skinOffset = 0.0f;

  Locator(
      const std::string& name = "uninitialized",
      const size_t parent = kInvalidIndex,
      const Vector3f& offset = Vector3f::Zero(),
      const Vector3i& locked = Vector3i::Zero(),
      const float weight = 1.0f,
      const Vector3f& limitOrigin = Vector3f::Zero(),
      const Vector3f& limitWeight = Vector3f::Zero(),
      bool attachedToSkin = false,
      float skinOffset = 0.0f)
      : name(name),
        parent(parent),
        offset(offset),
        locked(locked),
        weight(weight),
        limitOrigin(limitOrigin),
        limitWeight(limitWeight),
        attachedToSkin(attachedToSkin),
        skinOffset(skinOffset) {}

  /// Equality comparison; uses approximate (`isApprox`) comparison for floating-point fields.
  inline bool operator==(const Locator& locator) const {
    return (
        (name == locator.name) && (parent == locator.parent) && offset.isApprox(locator.offset) &&
        locked.isApprox(locator.locked) && isApprox(weight, locator.weight) &&
        limitOrigin.isApprox(locator.limitOrigin) && limitWeight.isApprox(locator.limitWeight) &&
        attachedToSkin == locator.attachedToSkin && isApprox(skinOffset, locator.skinOffset));
  }
};

/// A collection of locators attached to a skeleton
using LocatorList = std::vector<Locator>;

} // namespace momentum
