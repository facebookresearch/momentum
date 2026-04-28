/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character/locator_state.h"

#include "momentum/character/skeleton_state.h"

namespace momentum {

void LocatorState::update(
    const SkeletonState& skeletonState,
    const LocatorList& referenceLocators) noexcept {
  const size_t numLocators = referenceLocators.size();

  position.resize(numLocators);

  const auto& jointState = skeletonState.jointState;

  for (size_t locatorID = 0; locatorID < numLocators; locatorID++) {
    const Locator& locator = referenceLocators[locatorID];

    const size_t& parentId = locator.parent;

    // Transform each locator by its parent joint's world transform.
    position[locatorID] = jointState[parentId].transform * locator.offset;
  }
}

} // namespace momentum
