/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character/locator.h>

#include <momentum/common/filesystem.h>
#include <span>

namespace momentum {

enum class LocatorSpace {
  Local = 0, // local offset in the joint space; useful for transferring locators from one model to
             // another.
  Global = 1, // global offset in the character's space; useful for authoring in Maya/Blender
              // without parenting the locators.
};

LocatorList loadLocators(
    const filesystem::path& filename,
    const Skeleton& skeleton,
    const ParameterTransform& parameterTransform);

LocatorList loadLocatorsFromBuffer(
    std::span<const std::byte> rawData,
    const Skeleton& skeleton,
    const ParameterTransform& parameterTransform);

void saveLocators(
    const filesystem::path& filename,
    const LocatorList& locators,
    const Skeleton& skeleton,
    LocatorSpace space = LocatorSpace::Global);

} // namespace momentum
