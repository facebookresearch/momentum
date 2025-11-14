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
  /// Name identifier for the locator
  std::string name;

  /// Index of the parent joint in the skeleton
  size_t parent;

  /// Position relative to the parent joint's coordinate system
  Vector3f offset;

  /// Specifies which axes (x,y,z) are locked (1) or free (0)
  Vector3i locked;

  /// Influence weight of this locator when used in constraints
  float weight;

  /// Reference position for limit constraints, typically equal to offset when loaded
  Vector3f limitOrigin;

  /// Controls how strongly the locator should maintain its original position
  /// Higher values create stronger constraints, zero means completely free
  Vector3f limitWeight;

  /// Indicates whether the locator is attached to the skin of a person (e.g. as in mocap tracking),
  /// used to determine whether the locator can safely be converted to a skinned locator.
  bool attachedToSkin = false;

  /// Offset from the skin surface, used when trying to solve for body shape using locators.
  float skinOffset = 0.0f;

  /// Creates a locator with the specified properties
  ///
  /// @param name Identifier for the locator
  /// @param parent Index of the parent joint
  /// @param offset Position relative to the parent joint
  /// @param locked Axes that are locked (1) or free (0)
  /// @param weight Influence weight in constraints
  /// @param limitOrigin Reference position for limit constraints
  /// @param limitWeight Strength of position maintenance constraints
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

  /// Compares two locators for equality, using approximate comparison for floating-point values
  ///
  /// @param locator The locator to compare with
  /// @return True if all properties are equal (or approximately equal for floating-point values)
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
