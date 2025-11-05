/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/fbx/fbx_io.h"

#include "momentum/character/character.h"
#include "momentum/common/exception.h"
#include "momentum/io/fbx/openfbx_loader.h"

namespace momentum {

Character loadFbxCharacter(
    const filesystem::path& inputPath,
    KeepLocators keepLocators,
    bool permissive,
    LoadBlendShapes loadBlendShapes,
    bool stripNamespaces) {
  return loadOpenFbxCharacter(
      inputPath, keepLocators, permissive, loadBlendShapes, stripNamespaces);
}

Character loadFbxCharacter(
    std::span<const std::byte> inputSpan,
    KeepLocators keepLocators,
    bool permissive,
    LoadBlendShapes loadBlendShapes,
    bool stripNamespaces) {
  return loadOpenFbxCharacter(
      inputSpan, keepLocators, permissive, loadBlendShapes, stripNamespaces);
}

std::tuple<Character, std::vector<MatrixXf>, float> loadFbxCharacterWithMotion(
    const filesystem::path& inputPath,
    KeepLocators keepLocators,
    bool permissive,
    LoadBlendShapes loadBlendShapes,
    bool stripNamespaces) {
  return loadOpenFbxCharacterWithMotion(
      inputPath, keepLocators, permissive, loadBlendShapes, stripNamespaces);
}

std::tuple<Character, std::vector<MatrixXf>, float> loadFbxCharacterWithMotion(
    std::span<const std::byte> inputSpan,
    KeepLocators keepLocators,
    bool permissive,
    LoadBlendShapes loadBlendShapes,
    bool stripNamespaces) {
  return loadOpenFbxCharacterWithMotion(
      inputSpan, keepLocators, permissive, loadBlendShapes, stripNamespaces);
}

void saveFbx(
    const filesystem::path& /* filename */,
    const Character& /* character */,
    const MatrixXf& /* poses */,
    const VectorXf& /* identity */,
    double /* framerate */,
    bool /* saveMesh */,
    const FBXCoordSystemInfo& /* coordSystemInfo */,
    bool /* permissive */,
    const std::vector<std::vector<Marker>>& /* markerSequence */,
    std::string_view /* fbxNamespace */) {
  MT_THROW(
      "FBX saving is not supported in OpenFBX-only mode. FBX loading is available via OpenFBX, but saving requires the full Autodesk FBX SDK.");
}

void saveFbxWithJointParams(
    const filesystem::path& /* filename */,
    const Character& /* character */,
    const MatrixXf& /* jointParams */,
    double /* framerate */,
    bool /* saveMesh */,
    const FBXCoordSystemInfo& /* coordSystemInfo */,
    bool /* permissive */,
    const std::vector<std::vector<Marker>>& /* markerSequence */,
    std::string_view /* fbxNamespace */) {
  MT_THROW(
      "FBX saving is not supported in OpenFBX-only mode. FBX loading is available via OpenFBX, but saving requires the full Autodesk FBX SDK.");
}

void saveFbxModel(
    const filesystem::path& /* filename */,
    const Character& /* character */,
    const FBXCoordSystemInfo& /* coordSystemInfo */,
    bool /* permissive */,
    std::string_view /* fbxNamespace */) {
  MT_THROW(
      "FBX saving is not supported in OpenFBX-only mode. FBX loading is available via OpenFBX, but saving requires the full Autodesk FBX SDK.");
}

MarkerSequence loadFbxMarkerSequence(const filesystem::path& filename, bool stripNamespaces) {
  return loadOpenFbxMarkerSequence(filename, stripNamespaces);
}

} // namespace momentum
