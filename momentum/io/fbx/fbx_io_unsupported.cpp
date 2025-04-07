/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/fbx/fbx_io.h"

#include "momentum/character/character.h"
#include "momentum/common/exception.h"

namespace momentum {

Character loadFbxCharacter(const filesystem::path& inputPath, bool permissive) {
  MT_THROW("FbxSDK is not supported on your platform. Please use loadOpenFbxCharacter instead.");
  return Character();
}

Character loadFbxCharacter(gsl::span<const std::byte> inputSpan, bool permissive) {
  MT_THROW("FbxSDK is not supported on your platform. Please use loadOpenFbxCharacter instead.");
  return Character();
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
