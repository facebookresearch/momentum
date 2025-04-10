/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/fbx/fbx_io.h"

#include "momentum/character/character.h"
#include "momentum/common/exception.h"
#include "momentum/io/openfbx/openfbx_io.h"

namespace momentum {

Character loadFbxCharacter(const filesystem::path& inputPath, bool keepLocators, bool permissive) {
  return loadOpenFbxCharacter(inputPath, keepLocators, permissive);
}

Character
loadFbxCharacter(gsl::span<const std::byte> inputSpan, bool keepLocators, bool permissive) {
  return loadOpenFbxCharacter(inputSpan, keepLocators, permissive);
}

std::tuple<Character, std::vector<MatrixXf>, float>
loadFbxCharacterWithMotion(const filesystem::path& inputPath, bool keepLocators, bool permissive) {
  return loadOpenFbxCharacterWithMotion(inputPath, keepLocators, permissive);
}

std::tuple<Character, std::vector<MatrixXf>, float> loadFbxCharacterWithMotion(
    gsl::span<const std::byte> inputSpan,
    bool keepLocators,
    bool permissive) {
  return loadOpenFbxCharacterWithMotion(inputSpan, keepLocators, permissive);
}

void saveFbx(
    const filesystem::path& filename,
    const Character& character,
    const MatrixXf& poses,
    const VectorXf& identity,
    double framerate,
    bool saveMesh,
    const FBXCoordSystemInfo& coordSystemInfo,
    bool permissive) {
  MT_THROW("FbxSDK is not supported on your platform.");
}

void saveFbxWithJointParams(
    const filesystem::path& filename,
    const Character& character,
    const MatrixXf& jointParams,
    double framerate,
    bool saveMesh,
    const FBXCoordSystemInfo& coordSystemInfo,
    bool permissive) {
  MT_THROW("FbxSDK is not supported on your platform.");
}

void saveFbxModel(
    const filesystem::path& filename,
    const Character& character,
    const FBXCoordSystemInfo& coordSystemInfo,
    bool permissive) {
  MT_THROW("FbxSDK is not supported on your platform.");
}

} // namespace momentum
