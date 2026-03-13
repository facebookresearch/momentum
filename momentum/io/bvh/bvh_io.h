/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/character/types.h>
#include <momentum/common/filesystem.h>
#include <momentum/math/types.h>

#include <span>
#include <tuple>
#include <vector>

namespace momentum {

/// Loads a character from a BVH (BioVision Hierarchy) file.
///
/// The character is created with an identity ParameterTransform where model parameters
/// map directly to joint parameters (7 per joint: tx, ty, tz, rx, ry, rz, sc).
///
/// @param[in] filepath The path to the BVH file.
/// @return The loaded Character object.
[[nodiscard]] Character loadBvhCharacter(const filesystem::path& filepath);

/// Loads a character from a BVH file buffer.
///
/// @param[in] bytes The buffer containing the BVH file data.
/// @return The loaded Character object.
[[nodiscard]] Character loadBvhCharacter(std::span<const std::byte> bytes);

/// Loads a character with motion data from a BVH (BioVision Hierarchy) file.
///
/// Returns the character with an identity ParameterTransform, along with the motion data
/// represented as model parameters and the frame rate. BVH rotation values are converted
/// from degrees to radians and mapped to Momentum's ZYX Euler convention.
///
/// @param[in] filepath The path to the BVH file.
/// @return A tuple of (Character, MotionParameters, fps).
[[nodiscard]] std::tuple<Character, MotionParameters, float> loadBvhCharacterWithMotion(
    const filesystem::path& filepath);

/// Loads a character with motion data from a BVH file buffer.
///
/// @param[in] bytes The buffer containing the BVH file data.
/// @return A tuple of (Character, MotionParameters, fps).
[[nodiscard]] std::tuple<Character, MotionParameters, float> loadBvhCharacterWithMotion(
    std::span<const std::byte> bytes);

/// Saves a character to a BVH file with a single rest-pose frame.
///
/// Writes the skeleton hierarchy with default BVH channel layout: root joints get 6 channels
/// (Xposition, Yposition, Zposition, Zrotation, Xrotation, Yrotation), non-root joints get
/// 3 rotation channels (Zrotation, Xrotation, Yrotation).
///
/// @param[in] filepath The output BVH file path.
/// @param[in] character The character to save.
void saveBvhCharacter(const filesystem::path& filepath, const Character& character);

/// Saves a character with motion data to a BVH file.
///
/// Converts Momentum model parameters back to BVH channel values. Rotation values are
/// converted from radians (Momentum ZYX convention) to degrees in the BVH channel order.
///
/// @param[in] filepath The output BVH file path.
/// @param[in] character The character to save.
/// @param[in] motionParams The motion parameters (parameter names + motion matrix).
/// @param[in] fps The frame rate for the BVH file.
void saveBvhCharacterWithMotion(
    const filesystem::path& filepath,
    const Character& character,
    const MotionParameters& motionParams,
    float fps);

/// Saves a character with motion data to a byte buffer in BVH format.
///
/// @param[in] character The character to save.
/// @param[in] motionParams The motion parameters (parameter names + motion matrix).
/// @param[in] fps The frame rate for the BVH file.
/// @return The BVH file content as a byte buffer.
[[nodiscard]] std::vector<std::byte> saveBvhCharacterWithMotion(
    const Character& character,
    const MotionParameters& motionParams,
    float fps);

} // namespace momentum
