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
#include <momentum/math/types.h>

#include <span>

#include <string_view>

namespace momentum {

// KeepLocators Specifies whether Nulls in the transform hierarchy should be turned into Locators.
enum class KeepLocators { No, Yes };

// LoadBlendShapes Specifies whether blendshapes should be loaded or not
enum class LoadBlendShapes { No, Yes };

// Using keepLocators means the Nulls in the transform hierarchy will be turned into Locators.
// This is different from historical momentum behavior so it's off by default.
// Permissive mode allows loading  mesh-only characters (without skin weights).
Character loadFbxCharacter(
    const filesystem::path& inputPath,
    KeepLocators keepLocators = KeepLocators::No,
    bool permissive = false,
    LoadBlendShapes loadBlendShapes = LoadBlendShapes::No,
    bool stripNamespaces = true);

// Using keepLocators means the Nulls in the transform hierarchy will be turned into Locators.
// This is different from historical momentum behavior so it's off by default.
// Permissive mode allows loading  mesh-only characters (without skin weights).
Character loadFbxCharacter(
    std::span<const std::byte> inputSpan,
    KeepLocators keepLocators = KeepLocators::No,
    bool permissive = false,
    LoadBlendShapes loadBlendShapes = LoadBlendShapes::No,
    bool stripNamespaces = true);

// Permissive mode allows loading mesh-only characters (without skin weights).
std::tuple<Character, std::vector<MatrixXf>, float> loadFbxCharacterWithMotion(
    const filesystem::path& inputPath,
    KeepLocators keepLocators = KeepLocators::No,
    bool permissive = false,
    LoadBlendShapes loadBlendShapes = LoadBlendShapes::No,
    bool stripNamespaces = true);

// Permissive mode allows loading mesh-only characters (without skin weights).
std::tuple<Character, std::vector<MatrixXf>, float> loadFbxCharacterWithMotion(
    std::span<const std::byte> inputSpan,
    KeepLocators keepLocatorss = KeepLocators::No,
    bool permissive = false,
    LoadBlendShapes loadBlendShape = LoadBlendShapes::No,
    bool stripNamespaces = true);

/// Save a character with animation to an FBX file.
/// @param filename Path to the output FBX file
/// @param character The character to save
/// @param poses Model parameters for each frame (empty for bind pose only)
/// @param identity Identity pose parameters (empty to use bind pose)
/// @param framerate Animation framerate in frames per second
/// @param markerSequence Optional marker sequence data to save with the character
/// @param options Optional file save options for controlling output (default: FileSaveOptions{})
void saveFbx(
    const filesystem::path& filename,
    const Character& character,
    const MatrixXf& poses = MatrixXf(),
    const VectorXf& identity = VectorXf(),
    double framerate = 120.0,
    std::span<const std::vector<Marker>> markerSequence = {},
    const FileSaveOptions& options = FileSaveOptions());

/// Save a character with animation using joint parameters directly.
/// @param filename Path to the output FBX file
/// @param character The character to save
/// @param jointParams Joint parameters for each frame (empty for bind pose only)
/// @param framerate Animation framerate in frames per second
/// @param markerSequence Optional marker sequence data to save with the character
/// @param options Optional file save options for controlling output (default: FileSaveOptions{})
void saveFbxWithJointParams(
    const filesystem::path& filename,
    const Character& character,
    const MatrixXf& jointParams = MatrixXf(),
    double framerate = 120.0,
    std::span<const std::vector<Marker>> markerSequence = {},
    const FileSaveOptions& options = FileSaveOptions());

/// Save a character with animation using skeleton states directly.
/// @param filename Path to the output FBX file
/// @param character The character to save
/// @param skeletonStates SkeletonState for each frame (empty for bind pose only)
/// @param framerate Animation framerate in frames per second
/// @param markerSequence Optional marker sequence data to save with the character
/// @param options Optional file save options for controlling output (default: FileSaveOptions{})
void saveFbxWithSkeletonStates(
    const filesystem::path& filename,
    const Character& character,
    std::span<const SkeletonState> skeletonStates,
    double framerate = 120.0,
    std::span<const std::vector<Marker>> markerSequence = {},
    const FileSaveOptions& options = FileSaveOptions());

/// Save a character model (skeleton and mesh) without animation.
/// @param filename Path to the output FBX file
/// @param character The character to save
/// @param options Optional file save options for controlling output (default: FileSaveOptions{})
void saveFbxModel(
    const filesystem::path& filename,
    const Character& character,
    const FileSaveOptions& options = FileSaveOptions());

/// Loads a MarkerSequence from an FBX file.
///
/// This function reads motion capture marker data from an FBX file and returns
/// it as a MarkerSequence. The markers must be stored in the FBX scene hierarchy
/// under a "Markers" root node, and each marker node must have the custom property
/// "Momentum_Marker" to be recognized. The Markers root node must have the custom
/// property "Momentum_Markers_Root" to be identified.
///
/// @param[in] filename Path to the FBX file containing marker data.
/// @param stripNamespaces Removes namespace from joints when true. True by default
/// @return A MarkerSequence object containing the marker animation data, including
///         marker positions per frame and fps. Returns an empty sequence if no
///         markers or animations are found.
MarkerSequence loadFbxMarkerSequence(const filesystem::path& filename, bool stripNamespaces = true);

} // namespace momentum
