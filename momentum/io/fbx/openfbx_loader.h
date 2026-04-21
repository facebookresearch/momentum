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
#include <momentum/io/fbx/fbx_io.h>
#include <momentum/math/types.h>

#include <span>

namespace momentum {

// Custom property names for identifying Momentum-specific FBX nodes
constexpr const char* kMomentumMarkersRootProperty = "Momentum_Markers_Root";
constexpr const char* kMomentumMarkerProperty = "Momentum_Marker";

// Using keepLocators means the Nulls in the transform hierarchy will be turned into Locators.
// This is different from historical momentum behavior so it's off by default.
// Permissive mode allows loading mesh-only characters (without skin weights).
Character loadOpenFbxCharacter(
    const filesystem::path& inputPath,
    KeepLocators keepLocators = KeepLocators::No,
    Permissive permissive = Permissive::No,
    LoadBlendShapes loadBlendShapes = LoadBlendShapes::No,
    bool stripNamespaces = true);

// Permissive mode allows loading mesh-only characters (without skin weights).
Character loadOpenFbxCharacter(
    std::span<const std::byte> inputData,
    KeepLocators keepLocators = KeepLocators::No,
    Permissive permissive = Permissive::No,
    LoadBlendShapes loadBlendShapes = LoadBlendShapes::No,
    bool stripNamespaces = true);

// Permissive mode allows loading mesh-only characters (without skin weights).
std::tuple<Character, std::vector<MatrixXf>, float> loadOpenFbxCharacterWithMotion(
    const filesystem::path& inputPath,
    KeepLocators keepLocators = KeepLocators::No,
    Permissive permissive = Permissive::No,
    LoadBlendShapes loadBlendShapes = LoadBlendShapes::No,
    bool stripNamespaces = true);

// Permissive mode allows loading mesh-only characters (without skin weights).
std::tuple<Character, std::vector<MatrixXf>, float> loadOpenFbxCharacterWithMotion(
    std::span<const std::byte> inputData,
    KeepLocators keepLocators = KeepLocators::No,
    Permissive permissive = Permissive::No,
    LoadBlendShapes loadBlendShapes = LoadBlendShapes::No,
    bool stripNamespaces = true);

MarkerSequence loadOpenFbxMarkerSequence(
    const filesystem::path& filename,
    bool stripNamespaces = true);

/// Load a MarkerSequence from an in-memory FBX byte buffer.
///
/// Same semantics as the filename overload but parses the FBX scene from a buffer instead of
/// reading from disk. OpenFBX accepts a raw byte pointer + length, so this is true zero-copy.
///
/// @param[in] inputData Buffer holding the FBX file contents.
/// @param[in] stripNamespaces Removes namespace from joints when true. True by default.
/// @return A MarkerSequence object containing the marker animation data, or an empty sequence
///         if no markers or animations are found.
MarkerSequence loadOpenFbxMarkerSequence(
    std::span<const std::byte> inputData,
    bool stripNamespaces = true);

} // namespace momentum
