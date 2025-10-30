/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/collision_geometry.h>
#include <momentum/character/fwd.h>
#include <momentum/character/locator.h>
#include <momentum/math/fwd.h>

#include <nlohmann/json.hpp>
#include <span>
#include <string>

namespace momentum {

/// Loads a Character from a legacy JSON format that was previously used by older Python libraries
/// (and some other tools).
///
/// The legacy JSON format contains:
/// - "skeleton": FBX skeleton structure with joints, hierarchy, and transforms
/// - "skinnedmodel": Mesh data with skinning weights
/// - "collision": Optional collision geometry (tapered capsules)
///
/// @param[in] jsonPath Path to the legacy JSON file
/// @return The loaded Character object
/// @throws std::runtime_error if the file cannot be loaded or parsed
[[nodiscard]] Character loadCharacterFromLegacyJson(const std::string& jsonPath);

/// Loads a Character from a legacy JSON buffer.
///
/// @param[in] jsonBuffer Buffer containing the legacy JSON data
/// @return The loaded Character object
/// @throws std::runtime_error if the buffer cannot be parsed
[[nodiscard]] Character loadCharacterFromLegacyJsonBuffer(std::span<const std::byte> jsonBuffer);

/// Loads a Character from a legacy JSON string.
///
/// @param[in] jsonString String containing the legacy JSON data
/// @return The loaded Character object
/// @throws std::runtime_error if the string cannot be parsed
[[nodiscard]] Character loadCharacterFromLegacyJsonString(const std::string& jsonString);

/// Saves a Character to legacy JSON format.
///
/// This function converts a momentum::Character back to the legacy JSON format
/// for compatibility with existing tools and workflows.
///
/// @param[in] character The Character to save
/// @param[in] jsonPath Path where to save the legacy JSON file
/// @throws std::runtime_error if the file cannot be written
void saveCharacterToLegacyJson(const Character& character, const std::string& jsonPath);

/// Converts a Character to legacy JSON string.
///
/// @param[in] character The Character to convert
/// @return String containing the legacy JSON representation
[[nodiscard]] std::string characterToLegacyJsonString(const Character& character);

/// Converts a Character to legacy JSON object.
///
/// @param[in] character The Character to convert
/// @return nlohmann::json object containing the legacy JSON representation
[[nodiscard]] nlohmann::json characterToLegacyJson(const Character& character);

} // namespace momentum
