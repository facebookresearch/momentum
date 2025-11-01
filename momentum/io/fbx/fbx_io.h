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
#include <momentum/math/types.h>

#include <span>

#include <string_view>

namespace momentum {

// UpVector Specifies which canonical axis represents up in the system
// (typically Y or Z). Maps to fbxsdk::FbxAxisSystem::EUpVector
enum class FBXUpVector { XAxis = 1, YAxis = 2, ZAxis = 3 };

// FrontVector  Vector with origin at the screen pointing toward the camera.
// This is a subset of enum EUpVector because axis cannot be repeated.
// We use the system of "parity" to define this vector because its value (X,Y or
// Z axis) really depends on the up-vector. The EPreDefinedAxisSystem list the
// up-vector, parity and coordinate system values for the predefined systems.
// Maps to fbxsdk::FbxAxisSystem::EFrontVector
enum class FBXFrontVector { ParityEven = 1, ParityOdd = 2 };

// CoordSystem Specifies the third vector of the system.
// Maps to fbxsdk::FbxAxisSystem::ECoordSystem
enum class FBXCoordSystem { RightHanded, LeftHanded };

// KeepLocators Specifies whether Nulls in the transform hierarchy should be turned into Locators.
enum class KeepLocators { No, Yes };

// LoadBlendShapes Specifies whether blendshapes should be loaded or not
enum class LoadBlendShapes { No, Yes };

// A struct containing the up, front vectors and coordinate system
struct FBXCoordSystemInfo {
  // Default to the same orientations as FbxAxisSystem::eMayaYUp
  FBXUpVector upVector = FBXUpVector::YAxis;
  FBXFrontVector frontVector = FBXFrontVector::ParityOdd;
  FBXCoordSystem coordSystem = FBXCoordSystem::RightHanded;
};

// Using keepLocators means the Nulls in the transform hierarchy will be turned into Locators.
// This is different from historical momentum behavior so it's off by default.
// Permissive mode allows loading  mesh-only characters (without skin weights).
Character loadFbxCharacter(
    const filesystem::path& inputPath,
    KeepLocators keepLocators = KeepLocators::No,
    bool permissive = false,
    LoadBlendShapes loadBlendShapes = LoadBlendShapes::No);

// Using keepLocators means the Nulls in the transform hierarchy will be turned into Locators.
// This is different from historical momentum behavior so it's off by default.
// Permissive mode allows loading  mesh-only characters (without skin weights).
Character loadFbxCharacter(
    std::span<const std::byte> inputSpan,
    KeepLocators keepLocators = KeepLocators::No,
    bool permissive = false,
    LoadBlendShapes loadBlendShapes = LoadBlendShapes::No);

// Permissive mode allows loading mesh-only characters (without skin weights).
std::tuple<Character, std::vector<MatrixXf>, float> loadFbxCharacterWithMotion(
    const filesystem::path& inputPath,
    KeepLocators keepLocators = KeepLocators::No,
    bool permissive = false,
    LoadBlendShapes loadBlendShapes = LoadBlendShapes::No);

// Permissive mode allows loading mesh-only characters (without skin weights).
std::tuple<Character, std::vector<MatrixXf>, float> loadFbxCharacterWithMotion(
    std::span<const std::byte> inputSpan,
    KeepLocators keepLocatorss = KeepLocators::No,
    bool permissive = false,
    LoadBlendShapes loadBlendShape = LoadBlendShapes::No);

/// Save a character with animation to an FBX file.
/// @param filename Path to the output FBX file
/// @param character The character to save
/// @param poses Model parameters for each frame (empty for bind pose only)
/// @param identity Identity pose parameters (empty to use bind pose)
/// @param framerate Animation framerate in frames per second
/// @param saveMesh Whether to include mesh geometry in the output
/// @param coordSystemInfo Coordinate system configuration for the FBX file
/// @param permissive Permissive mode allows saving mesh-only characters (without skin weights)
/// @param fbxNamespace Optional namespace to prepend to all node names (e.g., "ns" will become
/// "ns:")
void saveFbx(
    const filesystem::path& filename,
    const Character& character,
    const MatrixXf& poses = MatrixXf(),
    const VectorXf& identity = VectorXf(),
    double framerate = 120.0,
    bool saveMesh = false,
    const FBXCoordSystemInfo& coordSystemInfo = FBXCoordSystemInfo(),
    bool permissive = false,
    const std::vector<std::vector<Marker>>& markerSequence = {},
    std::string_view fbxNamespace = "");

/// Save a character with animation using joint parameters directly.
/// @param filename Path to the output FBX file
/// @param character The character to save
/// @param jointParams Joint parameters for each frame (empty for bind pose only)
/// @param framerate Animation framerate in frames per second
/// @param saveMesh Whether to include mesh geometry in the output
/// @param coordSystemInfo Coordinate system configuration for the FBX file
/// @param permissive Permissive mode allows saving mesh-only characters (without skin weights)
/// @param markerSequence Optional marker sequence data to save with the character
/// @param fbxNamespace Optional namespace to prepend to all node names (e.g., "ns" will become
/// "ns:")
void saveFbxWithJointParams(
    const filesystem::path& filename,
    const Character& character,
    const MatrixXf& jointParams = MatrixXf(),
    double framerate = 120.0,
    bool saveMesh = false,
    const FBXCoordSystemInfo& coordSystemInfo = FBXCoordSystemInfo(),
    bool permissive = false,
    const std::vector<std::vector<Marker>>& markerSequence = {},
    std::string_view fbxNamespace = "");

void saveFbxWithSkelState(
    const filesystem::path& filename,
    const Character& character,
    std::span<const SkeletonState> skeletonStates,
    double framerate = 120.0,
    bool saveMesh = false,
    const FBXCoordSystemInfo& coordSystemInfo = FBXCoordSystemInfo(),
    bool permissive = false,
    const std::vector<std::vector<Marker>>& markerSequence = {},
    std::string_view fbxNamespace = "");

/// Save a character model (skeleton and mesh) without animation.
/// @param filename Path to the output FBX file
/// @param character The character to save
/// @param coordSystemInfo Coordinate system configuration for the FBX file
/// @param permissive Permissive mode allows saving mesh-only characters (without skin weights)
/// @param fbxNamespace Optional namespace to prepend to all node names (e.g., "ns" will become
/// "ns:")
void saveFbxModel(
    const filesystem::path& filename,
    const Character& character,
    const FBXCoordSystemInfo& coordSystemInfo = FBXCoordSystemInfo(),
    bool permissive = false,
    std::string_view fbxNamespace = "");

/// Loads a MarkerSequence from an FBX file.
///
/// This function reads motion capture marker data from an FBX file and returns
/// it as a MarkerSequence. The markers must be stored in the FBX scene hierarchy
/// under a "Markers" root node, and each marker node must have the custom property
/// "Momentum_Marker" to be recognized. The Markers root node must have the custom
/// property "Momentum_Markers_Root" to be identified.
///
/// @param[in] filename Path to the FBX file containing marker data.
/// @return A MarkerSequence object containing the marker animation data, including
///         marker positions per frame and fps. Returns an empty sequence if no
///         markers or animations are found.
MarkerSequence loadFbxMarkerSequence(const filesystem::path& filename);

} // namespace momentum
