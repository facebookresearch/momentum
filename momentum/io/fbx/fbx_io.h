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
    Permissive permissive = Permissive::No,
    LoadBlendShapes loadBlendShapes = LoadBlendShapes::No,
    bool stripNamespaces = true);

// Using keepLocators means the Nulls in the transform hierarchy will be turned into Locators.
// This is different from historical momentum behavior so it's off by default.
// Permissive mode allows loading  mesh-only characters (without skin weights).
Character loadFbxCharacter(
    std::span<const std::byte> inputSpan,
    KeepLocators keepLocators = KeepLocators::No,
    Permissive permissive = Permissive::No,
    LoadBlendShapes loadBlendShapes = LoadBlendShapes::No,
    bool stripNamespaces = true);

/// Load every character of an FBX scene.
///
/// Unlike loadFbxCharacter, which merges the whole scene into one Character, this returns one
/// Character per skeleton root (a LIMB_NODE, also under grouping NULL_NODEs) and one per unskinned
/// mesh outside a skeleton, such as an object written by FbxBuilder::addAnimatedMesh, which becomes
/// a one-joint character. A skinned mesh goes to the character its skin links to. An unskinned mesh
/// in a skeleton, such as one written by FbxBuilder::addRigidBody, is bound rigidly to the nearest
/// joint above it. Each character has an identity parameter transform. No animation is loaded.
///
/// @note The name comes from the root joint's `name` property when present, otherwise from the root
/// node name. The singular loader leaves the name empty when that property is absent.
///
/// @param inputPath Path to the FBX file.
/// @param keepLocators Turns Nulls in the transform hierarchy into Locators when Yes.
/// @param permissive When Yes, loads joints with unsupported rotation settings and skips a mesh
/// that belongs to no character, with a warning. When No, such a mesh makes the load fail.
/// @param loadBlendShapes Loads blend shapes when Yes.
/// @param stripNamespaces Removes namespaces from node names when true. True by default.
/// @return One Character per character in the scene, in scene order.
std::vector<Character> loadFbxCharacters(
    const filesystem::path& inputPath,
    KeepLocators keepLocators = KeepLocators::No,
    Permissive permissive = Permissive::No,
    LoadBlendShapes loadBlendShapes = LoadBlendShapes::No,
    bool stripNamespaces = true);

/// Load every character of an FBX scene from an in-memory buffer.
///
/// See the filename overload.
///
/// @param inputSpan Buffer holding the FBX file contents.
/// @param keepLocators Turns Nulls in the transform hierarchy into Locators when Yes.
/// @param permissive When Yes, loads joints with unsupported rotation settings and skips a mesh
/// that belongs to no character, with a warning. When No, such a mesh makes the load fail.
/// @param loadBlendShapes Loads blend shapes when Yes.
/// @param stripNamespaces Removes namespaces from node names when true. True by default.
/// @return One Character per character in the scene, in scene order.
std::vector<Character> loadFbxCharacters(
    std::span<const std::byte> inputSpan,
    KeepLocators keepLocators = KeepLocators::No,
    Permissive permissive = Permissive::No,
    LoadBlendShapes loadBlendShapes = LoadBlendShapes::No,
    bool stripNamespaces = true);

/// Load every character of an FBX scene together with its motion.
///
/// Characters are those of loadFbxCharacters. Each character's motion holds one joint-parameter
/// matrix per readable animation stack, in stack order, read only from the curves on its own
/// joints. The lists are index-aligned across characters; a matrix is empty when its stack does not
/// animate the character. Column i of every matrix is the pose at i / fps seconds, so the matrices
/// of one stack share their frame times; each matrix ends at the last key on its character's own
/// joints, so the matrices of one stack can have different frame counts.
///
/// @param inputPath Path to the FBX file.
/// @param keepLocators Turns Nulls in the transform hierarchy into Locators when Yes.
/// @param permissive When Yes, loads joints with unsupported rotation settings and skips a mesh
/// that belongs to no character, with a warning. When No, such a mesh makes the load fail.
/// @param loadBlendShapes Loads blend shapes when Yes.
/// @param stripNamespaces Removes namespaces from node names when true. True by default.
/// @return The characters, each with its per-stack motion, and the scene frame rate.
std::tuple<std::vector<std::pair<Character, std::vector<MatrixXf>>>, float>
loadFbxCharactersWithMotion(
    const filesystem::path& inputPath,
    KeepLocators keepLocators = KeepLocators::No,
    Permissive permissive = Permissive::No,
    LoadBlendShapes loadBlendShapes = LoadBlendShapes::No,
    bool stripNamespaces = true);

/// Load every character of an FBX scene together with its motion, from an in-memory buffer.
///
/// See the filename overload.
///
/// @param inputSpan Buffer holding the FBX file contents.
/// @param keepLocators Turns Nulls in the transform hierarchy into Locators when Yes.
/// @param permissive When Yes, loads joints with unsupported rotation settings and skips a mesh
/// that belongs to no character, with a warning. When No, such a mesh makes the load fail.
/// @param loadBlendShapes Loads blend shapes when Yes.
/// @param stripNamespaces Removes namespaces from node names when true. True by default.
/// @return The characters, each with its per-stack motion, and the scene frame rate.
std::tuple<std::vector<std::pair<Character, std::vector<MatrixXf>>>, float>
loadFbxCharactersWithMotion(
    std::span<const std::byte> inputSpan,
    KeepLocators keepLocators = KeepLocators::No,
    Permissive permissive = Permissive::No,
    LoadBlendShapes loadBlendShapes = LoadBlendShapes::No,
    bool stripNamespaces = true);

// Permissive mode allows loading mesh-only characters (without skin weights).
std::tuple<Character, std::vector<MatrixXf>, float> loadFbxCharacterWithMotion(
    const filesystem::path& inputPath,
    KeepLocators keepLocators = KeepLocators::No,
    Permissive permissive = Permissive::No,
    LoadBlendShapes loadBlendShapes = LoadBlendShapes::No,
    bool stripNamespaces = true);

// Permissive mode allows loading mesh-only characters (without skin weights).
std::tuple<Character, std::vector<MatrixXf>, float> loadFbxCharacterWithMotion(
    std::span<const std::byte> inputSpan,
    KeepLocators keepLocatorss = KeepLocators::No,
    Permissive permissive = Permissive::No,
    LoadBlendShapes loadBlendShape = LoadBlendShapes::No,
    bool stripNamespaces = true);

/// Save a character with animation to an FBX file.
///
/// Blend shape and face expression weights are automatically extracted from the model parameters
/// (poses) when the character has blend shapes or face expression blend shapes configured in its
/// parameter transform. No separate weight arguments are needed.
///
/// @param filename Path to the output FBX file
/// @param character The character to save
/// @param poses Model parameters with shape (numAllModelParameters x numFrames), one column per
/// frame; empty for the bind pose only. With columns, a different row count throws.
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
///
/// Blend shape and face expression weights are automatically extracted from the model parameters
/// when the character has blend shapes or face expression blend shapes configured in its
/// parameter transform. Since joint parameters do not contain blend shape weights, this function
/// does not save blend shape animation. Use saveFbx() with model parameters to include blend shape
/// animation.
///
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
///
/// Since skeleton states do not contain blend shape weights, this function does not save blend
/// shape animation. Use saveFbx() with model parameters to include blend shape animation.
///
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

/// Load per-frame blend shape animation weights from an FBX file.
///
/// Reads the DeformPercent animation curves on FbxBlendShapeChannel objects in the FBX scene.
/// Returns a matrix of shape (numShapes x numFrames) with values in [0, 1] range.
/// Returns an empty matrix if no blend shape animation is found.
///
/// @param filename Path to the FBX file containing blend shape animation.
/// @return A matrix of blend shape weights (numShapes x numFrames), values in [0, 1].
MatrixXf loadFbxBlendShapeWeights(const filesystem::path& filename);

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

/// Loads a MarkerSequence from an in-memory FBX byte buffer.
///
/// Same semantics as the filename overload. Uses OpenFBX, which natively accepts a byte
/// buffer, so this is true zero-copy.
///
/// @param[in] inputSpan Buffer holding the FBX file contents.
/// @param stripNamespaces Removes namespace from joints when true. True by default.
/// @return A MarkerSequence object containing the marker animation data.
MarkerSequence loadFbxMarkerSequence(
    std::span<const std::byte> inputSpan,
    bool stripNamespaces = true);

} // namespace momentum
