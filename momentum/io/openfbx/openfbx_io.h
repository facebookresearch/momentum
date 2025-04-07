/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/common/filesystem.h>
#include <momentum/math/types.h>

#include <gsl/span>

namespace momentum {

// Using keepLocators means the Nulls in the transform hierarchy will be turned into Locators.
// This is different from historical momentum behavior so it's off by default.
// Permissive mode allows loading mesh-only characters (without skin weights).
Character loadOpenFbxCharacter(
    const filesystem::path& inputPath,
    bool keepLocators = false,
    bool permissive = false);

// Permissive mode allows loading mesh-only characters (without skin weights).
Character loadOpenFbxCharacter(
    gsl::span<const std::byte> inputSpan,
    bool keepLocators = false,
    bool permissive = false);

// Permissive mode allows loading mesh-only characters (without skin weights).
std::tuple<Character, std::vector<MatrixXf>, float> loadOpenFbxCharacterWithMotion(
    const filesystem::path& inputPath,
    bool keepLocators = false,
    bool permissive = false);

// Permissive mode allows loading mesh-only characters (without skin weights).
std::tuple<Character, std::vector<MatrixXf>, float> loadOpenFbxCharacterWithMotion(
    gsl::span<const std::byte> inputSpan,
    bool keepLocators = false,
    bool permissive = false);

} // namespace momentum
