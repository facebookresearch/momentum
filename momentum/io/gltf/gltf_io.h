/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/marker.h>
#include <momentum/character/types.h>
#include <momentum/common/filesystem.h>
#include <momentum/io/file_save_options.h>
#include <momentum/math/types.h>

#include <fx/gltf.h>
#include <span>

#include <tuple>
#include <vector>

namespace momentum {

Character loadGltfCharacter(const fx::gltf::Document& model);

Character loadGltfCharacter(const filesystem::path& gltfFilename);

Character loadGltfCharacter(std::span<const std::byte> byteSpan);

std::tuple<MotionParameters, IdentityParameters, float> loadMotion(
    const filesystem::path& gltfFilename);

/// Load a glTF character from a local file path.
///
/// This function assumes the file format of the given path is glTF without checking the extension,
/// so please ensure the file is in glTF format.
///
/// @param[in] filepath The path to the glTF character file.
/// @return A tuple containing the loaded Character object, the motion represented in model
/// parameters, the identity vector represented as joint parameters, and the fps.
std::tuple<Character, MatrixXf, VectorXf, float> loadCharacterWithMotion(
    const filesystem::path& gltfFilename);

/// Load a glTF character from a buffer.
///
/// @param[in] byteSpan The buffer containing the glTF character data.
/// @return A tuple containing the loaded Character object, the motion represented in model
/// parameters, the identity vector represented as joint parameters, and the fps.
std::tuple<Character, MatrixXf, VectorXf, float> loadCharacterWithMotion(
    std::span<const std::byte> byteSpan);

/// Load a GLTF Character with motion in the form of skeleton states (transform matrices)
///
/// Unlike the other loadCharacterWithMotion functions, this function does not require the
/// file to be saved using momentum's functionality that saves model parameters in a
/// custom GTLF extension, however the resulting skeleton states may be harder to work with.
std::tuple<Character, std::vector<SkeletonState>, std::vector<float>>
loadCharacterWithSkeletonStates(std::span<const std::byte> byteSpan);

std::tuple<Character, std::vector<SkeletonState>, std::vector<float>>
loadCharacterWithSkeletonStates(const filesystem::path& gltfFilename);

/// Maps the loaded motion onto the input character by matching joint names and parameter names.
///
/// This function assumes the file format of the given path is glTF without checking the extension,
/// so please ensure the file is in glTF format.
///
/// @param[in] gltfFilename The path to the glTF motion file.
/// @param[in] character The Character object to map the motion onto.
/// @return A tuple containing the motion represented in model parameters, the identity vector
/// represented as joint parameters, and the fps. The model parameters and joint parameters are
/// mapped to the input character by name matching.
std::tuple<MatrixXf, VectorXf, float> loadMotionOnCharacter(
    const filesystem::path& gltfFilename,
    const Character& character);

/// Buffer version of loadMotionOnCharacter()
///
/// @param[in] byteSpan The buffer containing the glTF motion data.
/// @param[in] character The Character object to map the motion onto.
/// @return A tuple containing the motion represented in model parameters, the identity vector
/// represented as joint parameters, and the fps. The model parameters and joint parameters are
/// mapped to the input character by name matching.
std::tuple<MatrixXf, VectorXf, float> loadMotionOnCharacter(
    std::span<const std::byte> byteSpan,
    const Character& character);

/// Loads a MarkerSequence from a file.
///
/// @param[in] filename Path to the file containing the MarkerSequence data.
/// @return A MarkerSequence object containing motion capture marker data.
MarkerSequence loadMarkerSequence(const filesystem::path& filename);

fx::gltf::Document makeCharacterDocument(
    const Character& character,
    float fps = 120.0f,
    const MotionParameters& motion = {},
    const IdentityParameters& offsets = {},
    std::span<const std::vector<Marker>> markerSequence = {},
    bool embedResource = true,
    const FileSaveOptions& options = FileSaveOptions());

/// Saves character motion to a glb file.
///
/// @param[in] motion The model parameters representing the motion of the character (numModelParams,
/// numFrames)
/// @param[in] offsets Offset values per joint capturing the skeleton bone lengths using translation
/// and scale offset (7*numJoints, 1)
/// @param[in] options Optional file save options for controlling output (default:
/// FileSaveOptions{}).
void saveGltfCharacter(
    const filesystem::path& filename,
    const Character& character,
    float fps = 120.0f,
    const MotionParameters& motion = {},
    const IdentityParameters& offsets = {},
    std::span<const std::vector<Marker>> markerSequence = {},
    const FileSaveOptions& options = FileSaveOptions());

/// Saves character skeleton states to a glb file.
///
/// @param[in] skeletonStates The skeleton states for each frame of the motion sequence (numFrames,
/// numJoints, 8)
/// @param[in] options Optional file save options for controlling output (default:
/// FileSaveOptions{}).
void saveGltfCharacter(
    const filesystem::path& filename,
    const Character& character,
    float fps,
    std::span<const SkeletonState> skeletonStates,
    std::span<const std::vector<Marker>> markerSequence = {},
    const FileSaveOptions& options = FileSaveOptions());

std::vector<std::byte> saveCharacterToBytes(
    const Character& character,
    float fps = 120.0f,
    const MotionParameters& motion = {},
    const IdentityParameters& offsets = {},
    std::span<const std::vector<Marker>> markerSequence = {},
    const FileSaveOptions& options = FileSaveOptions());

} // namespace momentum
