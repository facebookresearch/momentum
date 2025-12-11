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

/// Load a glTF character from a document.
///
/// @param[in] model The glTF document to load from.
/// @return The loaded Character object.
Character loadGltfCharacter(const fx::gltf::Document& model);

/// Load a glTF character from a file path.
///
/// This function assumes the file format is glTF without checking the extension.
///
/// @param[in] gltfFilename The path to the glTF file.
/// @return The loaded Character object.
Character loadGltfCharacter(const filesystem::path& gltfFilename);

/// Load a glTF character from a byte buffer.
///
/// @param[in] byteSpan The buffer containing the glTF character data.
/// @return The loaded Character object.
Character loadGltfCharacter(std::span<const std::byte> byteSpan);

/// Load motion data from a glTF file.
///
/// @param[in] gltfFilename The path to the glTF file.
/// @return A tuple containing the motion parameters, identity parameters, and fps.
std::tuple<MotionParameters, IdentityParameters, float> loadMotion(
    const filesystem::path& gltfFilename);

/// Load per-frame timestamps from a glTF file.
///
/// This function loads timestamps that were saved using GltfBuilder::addMotion.
///
/// @param[in] gltfFilename The path to the glTF file.
/// @return A vector of per-frame timestamps (usually in microseconds).
std::vector<int64_t> loadMotionTimestamps(const filesystem::path& gltfFilename);

/// Load a glTF character with motion from a file path.
///
/// This function loads both the character and motion data from a glTF file. The motion is
/// represented in model parameters and the identity vector is represented as joint parameters.
/// This function assumes the file format is glTF without checking the extension.
///
/// @param[in] gltfFilename The path to the glTF file.
/// @return A tuple containing the loaded Character object, the motion represented in model
/// parameters, the identity vector represented as joint parameters, and the fps.
std::tuple<Character, MatrixXf, momentum::JointParameters, float> loadCharacterWithMotion(
    const filesystem::path& gltfFilename);

/// Load a glTF character from a buffer.
///
/// @param[in] byteSpan The buffer containing the glTF character data.
/// @return A tuple containing the loaded Character object, the motion represented in model
/// parameters, the identity vector represented as joint parameters, and the fps.
std::tuple<Character, MatrixXf, momentum::JointParameters, float> loadCharacterWithMotion(
    std::span<const std::byte> byteSpan);

/// Load a glTF character with motion and apply scale parameters to the motion.
///
/// This function loads both the character and motion data from a glTF file, applying the
/// character's scale parameters directly to the motion as model parameters. This approach is
/// preferred over the deprecated method of storing scales in an offset vector in the parameter
/// transform. The scale parameters from the identity vector are integrated into the motion data
/// for each frame.
///
/// @param[in] byteSpan The buffer containing the glTF character data.
/// @return A tuple containing the loaded Character object, the motion represented in model
/// parameters with scales applied, the identity parameters as model parameters, and the fps.
std::tuple<Character, MatrixXf, momentum::ModelParameters, float>
loadCharacterWithMotionModelParameterScales(std::span<const std::byte> byteSpan);

/// Load a glTF character with motion and apply scale parameters to the motion.
///
/// This function loads both the character and motion data from a glTF file, applying the
/// character's scale parameters directly to the motion as model parameters. This approach is
/// preferred over the deprecated method of storing scales in an offset vector in the parameter
/// transform. The scale parameters from the identity vector are integrated into the motion data
/// for each frame.
///
/// @param[in] gltfFilename The path to the glTF file.
/// @return A tuple containing the loaded Character object, the motion represented in model
/// parameters with scales applied, the identity parameters as model parameters, and the fps.
std::tuple<Character, MatrixXf, momentum::ModelParameters, float>
loadCharacterWithMotionModelParameterScales(const filesystem::path& gltfFilename);

/// Load a glTF character with motion in the form of skeleton states.
///
/// This function loads motion data as skeleton states (transform matrices) instead of model
/// parameters. Unlike the other loadCharacterWithMotion functions, this function does not require
/// the file to be saved using momentum's custom glTF extension for model parameters. However, the
/// resulting skeleton states may be harder to work with than model parameters.
///
/// @param[in] byteSpan The buffer containing the glTF character data.
/// @return A tuple containing the loaded Character object, a vector of skeleton states for each
/// frame, and a vector of timestamps.
std::tuple<Character, std::vector<SkeletonState>, std::vector<float>>
loadCharacterWithSkeletonStates(std::span<const std::byte> byteSpan);

/// Load a glTF character with motion in the form of skeleton states.
///
/// This function loads motion data as skeleton states (transform matrices) instead of model
/// parameters. Unlike the other loadCharacterWithMotion functions, this function does not require
/// the file to be saved using momentum's custom glTF extension for model parameters. However, the
/// resulting skeleton states may be harder to work with than model parameters.
///
/// @param[in] gltfFilename The path to the glTF file.
/// @return A tuple containing the loaded Character object, a vector of skeleton states for each
/// frame, and a vector of timestamps.
std::tuple<Character, std::vector<SkeletonState>, std::vector<float>>
loadCharacterWithSkeletonStates(const filesystem::path& gltfFilename);

/// Load motion from a glTF file and map it to a character.
///
/// This function loads motion data from a glTF file and maps it onto the input character by
/// matching joint names and parameter names. This is useful when the motion was saved with a
/// different character but you want to apply it to your character. This function assumes the file
/// format is glTF without checking the extension.
///
/// @param[in] gltfFilename The path to the glTF motion file.
/// @param[in] character The Character object to map the motion onto.
/// @return A tuple containing the motion represented in model parameters, the identity vector
/// represented as joint parameters, and the fps. The model parameters and joint parameters are
/// mapped to the input character by name matching.
std::tuple<MatrixXf, JointParameters, float> loadMotionOnCharacter(
    const filesystem::path& gltfFilename,
    const Character& character);

/// Load motion from a buffer and map it to a character.
///
/// This function loads motion data from a buffer and maps it onto the input character by matching
/// joint names and parameter names. This is useful when the motion was saved with a different
/// character but you want to apply it to your character.
///
/// @param[in] byteSpan The buffer containing the glTF motion data.
/// @param[in] character The Character object to map the motion onto.
/// @return A tuple containing the motion represented in model parameters, the identity vector
/// represented as joint parameters, and the fps. The model parameters and joint parameters are
/// mapped to the input character by name matching.
std::tuple<MatrixXf, JointParameters, float> loadMotionOnCharacter(
    std::span<const std::byte> byteSpan,
    const Character& character);

/// Load a marker sequence from a glTF file.
///
/// This function loads motion capture marker data from a glTF file.
///
/// @param[in] filename Path to the glTF file containing the marker sequence data.
/// @return A MarkerSequence object containing the loaded motion capture marker data.
MarkerSequence loadMarkerSequence(const filesystem::path& filename);

/// Create a glTF document from a character.
///
/// This function creates a glTF document containing the character data and optionally motion,
/// identity offsets, and marker sequences.
///
/// @param[in] character The Character object to save.
/// @param[in] fps Frame rate of the motion in frames per second (default: 120.0f).
/// @param[in] motion Optional motion parameters to include in the document.
/// @param[in] offsets Optional identity parameters (joint offsets) to include in the document.
/// @param[in] markerSequence Optional marker sequence data to include in the document.
/// @param[in] embedResource Whether to embed resources in the document (default: true).
/// @param[in] options Optional file save options for controlling output.
/// @return A glTF document containing the character data.
fx::gltf::Document makeCharacterDocument(
    const Character& character,
    float fps = 120.0f,
    const MotionParameters& motion = {},
    const IdentityParameters& offsets = {},
    std::span<const std::vector<Marker>> markerSequence = {},
    bool embedResource = true,
    const FileSaveOptions& options = FileSaveOptions());

/// Save a character with motion to a glTF file.
///
/// This function saves a character along with optional motion data, identity parameters, and marker
/// sequences to a glTF file.
///
/// @param[in] filename The path where the glTF file will be saved.
/// @param[in] character The Character object to save.
/// @param[in] fps Frame rate of the motion in frames per second (default: 120.0f).
/// @param[in] motion Optional motion parameters to save (default: empty).
/// @param[in] offsets Optional identity parameters capturing skeleton bone lengths (default:
/// empty).
/// @param[in] markerSequence Optional marker sequence data to save (default: empty).
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

/// Save a character with skeleton states to a glTF file.
///
/// This function saves a character along with skeleton states (transform matrices for each joint
/// at each frame) to a glTF file. This is an alternative to saving model parameters.
///
/// @param[in] filename The path where the glTF file will be saved.
/// @param[in] character The Character object to save.
/// @param[in] fps Frame rate of the motion in frames per second.
/// @param[in] skeletonStates The skeleton states for each frame of the motion sequence.
/// @param[in] markerSequence Optional marker sequence data to save (default: empty).
/// @param[in] options Optional file save options for controlling output (default:
/// FileSaveOptions{}).
void saveGltfCharacter(
    const filesystem::path& filename,
    const Character& character,
    float fps,
    std::span<const SkeletonState> skeletonStates,
    std::span<const std::vector<Marker>> markerSequence = {},
    const FileSaveOptions& options = FileSaveOptions());

/// Save a character with motion to a byte buffer.
///
/// This function saves a character along with optional motion data, identity parameters, and marker
/// sequences to a byte buffer instead of a file.
///
/// @param[in] character The Character object to save.
/// @param[in] fps Frame rate of the motion in frames per second (default: 120.0f).
/// @param[in] motion Optional motion parameters to save (default: empty).
/// @param[in] offsets Optional identity parameters capturing skeleton bone lengths (default:
/// empty).
/// @param[in] markerSequence Optional marker sequence data to save (default: empty).
/// @param[in] options Optional file save options for controlling output (default:
/// FileSaveOptions{}).
/// @return A byte buffer containing the glTF data.
std::vector<std::byte> saveCharacterToBytes(
    const Character& character,
    float fps = 120.0f,
    const MotionParameters& motion = {},
    const IdentityParameters& offsets = {},
    std::span<const std::vector<Marker>> markerSequence = {},
    const FileSaveOptions& options = FileSaveOptions());

} // namespace momentum
