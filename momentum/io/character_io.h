/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character/marker.h>
#include <momentum/common/filesystem.h>
#include <momentum/io/file_save_options.h>
#include <momentum/math/fwd.h>
#include <momentum/math/types.h>

#include <string>
#include <vector>

namespace momentum {

/// Represents the supported character formats.
enum class CharacterFormat : uint8_t {
  Fbx, ///< FBX file format.
  Gltf, ///< glTF file format.
  Usd, ///< USD file format.
  Unknown ///< Unknown or unsupported file format.
};

/// High level function to load a character of any type, with a local path.
///
/// @param[in] characterPath The path to the character file.
/// @param[in] parametersPath The optional path to the file containing additional parameters for the
/// character.
/// @param[in] locatorsPath The optional path to the file containing additional locators for the
/// character.
/// @return The loaded Character object.
///
/// Currently, only supports .glb and .fbx. If you want to parse from a non-local path, you may need
/// to parse it using your favorite resource retriever into a buffer and use the buffer version of
/// this function.
[[nodiscard]] Character loadFullCharacter(
    const std::string& characterPath,
    const std::string& parametersPath = std::string(),
    const std::string& locatorsPath = std::string());

/// Buffer version of loadFullCharacter function, supports .glb and .fbx file formats.
///
/// @param[in] format The character file format.
/// @param[in] characterBuffer The buffer containing the character data.
/// @param[in] paramBuffer The optional buffer containing additional parameters for the character.
/// @param[in] locBuffer The optional buffer containing additional locators for the character.
/// @return The loaded Character object.
[[nodiscard]] Character loadFullCharacterFromBuffer(
    CharacterFormat format,
    std::span<const std::byte> characterBuffer,
    std::span<const std::byte> paramBuffer = std::span<const std::byte>(),
    std::span<const std::byte> locBuffer = std::span<const std::byte>());

/// High level function to save a character with motion and markers to any supported format.
///
/// The format is determined by the file extension (.fbx, .glb, .gltf).
/// This is a unified interface that automatically selects between FBX and GLTF based on extension.
/// @param[in] filename The path where the character file will be saved.
/// @param[in] character The Character object to save.
/// @param[in] fps Frame rate for the animation (default: 120.0f).
/// @param[in] motion The motion represented in model parameters (numModelParams, numFrames).
/// @param[in] markerSequence Optional marker sequence data to save with the character.
/// @param[in] options Optional file save options for controlling output (default:
/// FileSaveOptions{}).
void saveCharacter(
    const filesystem::path& filename,
    const Character& character,
    float fps = 120.f,
    const MatrixXf& motion = MatrixXf(),
    std::span<const std::vector<Marker>> markerSequence = {},
    const FileSaveOptions& options = FileSaveOptions());

/// High level function to save a character with motion in skeleton states and markers to any
/// supported format.
///
/// The format is determined by the file extension (.fbx, .glb, .gltf).
/// This is a unified interface that automatically selects between FBX and GLTF based on extension.
/// @param[in] filename The path where the character file will be saved.
/// @param[in] character The Character object to save.
/// @param[in] fps Frame rate for the animation.
/// @param[in] skeletonStates The motion represented in skeleton states (ie. JointStates).
/// @param[in] markerSequence Optional marker sequence data to save with the character.
/// @param[in] options Optional file save options for controlling output (default:
/// FileSaveOptions{}).
void saveCharacter(
    const filesystem::path& filename,
    const Character& character,
    float fps,
    std::span<const SkeletonState> skeletonStates,
    std::span<const std::vector<Marker>> markerSequence = {},
    const FileSaveOptions& options = FileSaveOptions());
} // namespace momentum
