/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character/marker.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character/types.h>
#include <momentum/common/filesystem.h>
#include <momentum/io/file_save_options.h>
#include <momentum/math/types.h>

#include <cstddef>
#include <span>
#include <string>
#include <tuple>
#include <vector>

namespace momentum {

/// Load a USD character from a local file path.
///
/// @param[in] inputPath The path to the USD character file.
/// @return The loaded Character object.
Character loadUsdCharacter(const filesystem::path& inputPath);

/// Load a USD character from a buffer.
///
/// @param[in] inputSpan The buffer containing the USD character data.
/// @return The loaded Character object.
Character loadUsdCharacter(std::span<const std::byte> inputSpan);

/// Load a USD character with skeleton states from a local file path.
///
/// @param[in] inputPath The path to the USD character file.
/// @return A tuple of (Character, skeleton states, frame times in seconds).
std::tuple<Character, std::vector<SkeletonState>, std::vector<float>>
loadUsdCharacterWithSkeletonStates(const filesystem::path& inputPath);

/// Load a USD character with skeleton states from a buffer.
///
/// @param[in] inputSpan The buffer containing the USD character data.
/// @return A tuple of (Character, skeleton states, frame times in seconds).
std::tuple<Character, std::vector<SkeletonState>, std::vector<float>>
loadUsdCharacterWithSkeletonStates(std::span<const std::byte> inputSpan);

/// Save a character to a USD file.
///
/// @param[in] filename The path to save the USD file.
/// @param[in] character The Character object to save.
/// @param[in] options Optional file save options for controlling output.
void saveUsd(
    const filesystem::path& filename,
    const Character& character,
    const FileSaveOptions& options = FileSaveOptions());

/// Save a character with skeleton state animation to a USD file.
///
/// @param[in] filename The path to save the USD file.
/// @param[in] character The Character object to save.
/// @param[in] fps Frames per second for the animation.
/// @param[in] skeletonStates The skeleton states to save (one per frame).
/// @param[in] options Optional file save options for controlling output.
void saveUsdCharacter(
    const filesystem::path& filename,
    const Character& character,
    float fps,
    std::span<const SkeletonState> skeletonStates,
    const FileSaveOptions& options = FileSaveOptions());

/// Load a USD character with model-parameter motion from a local file path.
///
/// @param[in] inputPath The path to the USD character file.
/// @return A tuple of (Character, motion parameters, identity parameters, fps).
std::tuple<Character, MotionParameters, IdentityParameters, float> loadUsdCharacterWithMotion(
    const filesystem::path& inputPath);

/// Load a USD character with model-parameter motion from a buffer.
///
/// @param[in] inputSpan The buffer containing the USD character data.
/// @return A tuple of (Character, motion parameters, identity parameters, fps).
std::tuple<Character, MotionParameters, IdentityParameters, float> loadUsdCharacterWithMotion(
    std::span<const std::byte> inputSpan);

/// Save a character with model-parameter motion to a USD file.
///
/// @param[in] filename The path to save the USD file.
/// @param[in] character The Character object to save.
/// @param[in] fps Frames per second for the animation.
/// @param[in] motion The motion data (parameter names, poses matrix).
/// @param[in] offsets The identity parameters (joint names, offset values).
/// @param[in] markerSequence Optional marker sequence to save.
/// @param[in] options Optional file save options for controlling output.
void saveUsdCharacterWithMotion(
    const filesystem::path& filename,
    const Character& character,
    float fps = 120.0f,
    const MotionParameters& motion = {},
    const IdentityParameters& offsets = {},
    std::span<const std::vector<Marker>> markerSequence = {},
    const FileSaveOptions& options = FileSaveOptions());

/// Load a marker sequence from a USD file.
///
/// @param[in] inputPath The path to the USD file.
/// @return The loaded MarkerSequence.
MarkerSequence loadUsdMarkerSequence(const filesystem::path& inputPath);

} // namespace momentum
