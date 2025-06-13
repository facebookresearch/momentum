/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character/locator.h>
#include <momentum/character/types.h>

namespace momentum {

/// Tracks the current world positions of locators in a skeleton.
///
/// This class maintains the transformed positions of locators based on the current
/// state of the skeleton they're attached to.
struct LocatorState {
  /// World-space positions of all locators, updated when the skeleton moves
  std::vector<Vector3f> position;

 public:
  /// Creates an empty locator state with no positions
  LocatorState() noexcept = default;

  /// Creates a locator state and immediately updates positions based on the given skeleton state
  ///
  /// @param skeletonState Current pose of the skeleton
  /// @param referenceLocators List of locators to track
  LocatorState(const SkeletonState& skeletonState, const LocatorList& referenceLocators) noexcept {
    update(skeletonState, referenceLocators);
  }

  /// Updates the world positions of all locators based on the current skeleton pose
  ///
  /// @param skeletonState Current pose of the skeleton
  /// @param referenceLocators List of locators to update positions for
  void update(const SkeletonState& skeletonState, const LocatorList& referenceLocators) noexcept;
};

} // namespace momentum
