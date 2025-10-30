/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/common/filesystem.h>
#include <momentum/math/types.h>

#include <span>

namespace momentum {

/// Loads a character from a URDF file.
template <typename T = float>
CharacterT<T> loadUrdfCharacter(const filesystem::path& filepath);

/// Loads a character from a URDF file using the provided byte data.
template <typename T = float>
[[nodiscard]] CharacterT<T> loadUrdfCharacter(std::span<const std::byte> bytes);

} // namespace momentum
