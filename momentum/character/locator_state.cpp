/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/locator_state.h>

#include <momentum/character/skeleton_state.h>

namespace momentum {

// functions
void LocatorState::update(
    const SkeletonState& skeletonState,
    const LocatorList& referenceLocators) noexcept {
  const size_t numLocators = referenceLocators.size();

  // resize output arrays
  position.resize(numLocators);

  // get joint state
  const auto& jointState = skeletonState.jointState;

  // go over all locators
  for (size_t locatorID = 0; locatorID < numLocators; locatorID++) {
    // reference for quick access
    const Locator& locator = referenceLocators[locatorID];

    // get parent id
    const size_t& parentId = locator.parent;

    // transform each locator by its parents transformation and store it in the locator state
    position[locatorID] = jointState[parentId].transformation * locator.offset;
  }
}

} // namespace momentum
