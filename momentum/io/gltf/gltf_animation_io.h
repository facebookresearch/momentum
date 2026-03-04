/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/character/types.h>
#include <momentum/common/filesystem.h>

#include <fx/gltf.h>

#include <span>
#include <tuple>
#include <variant>
#include <vector>

namespace momentum {

/// Get motion parameters from a glTF model.
///
/// @param[in] model The glTF document containing motion data.
/// @return A tuple containing the motion parameters and identity parameters.
std::tuple<MotionParameters, IdentityParameters> getMotionFromModel(fx::gltf::Document& model);

/// Calculate FPS from glTF animation data.
///
/// @param[in] model The glTF document containing animation data.
/// @return Estimated FPS, or 0.0f if unable to calculate.
float calculateFpsFromAnimationData(const fx::gltf::Document& model);

/// Load motion data from a glTF model.
///
/// @param[in,out] model The glTF document containing motion data.
/// @return A tuple containing the motion parameters, identity parameters, and fps.
std::tuple<MotionParameters, IdentityParameters, float> loadMotionFromModel(
    fx::gltf::Document& model);

/// Load character with motion from input (common implementation).
///
/// @param[in] input Either a filesystem path or a byte span containing glTF data.
/// @param[in] mapParameters Whether to map parameters to the character (default: false).
/// @return A tuple containing the character, motion matrix, joint parameters, and fps.
std::tuple<Character, MatrixXf, JointParameters, float> loadCharacterWithMotionCommon(
    const std::variant<filesystem::path, std::span<const std::byte>>& input,
    bool mapParameters = false);

/// Load motion onto a character (common implementation).
///
/// @param[in] input Either a filesystem path or a byte span containing glTF data.
/// @param[in] character The character to map motion onto.
/// @return A tuple containing the motion matrix, joint parameters, and fps.
std::tuple<MatrixXf, JointParameters, float> loadMotionOnCharacterCommon(
    const std::variant<filesystem::path, std::span<const std::byte>>& input,
    const Character& character);

/// Load character with skeleton states from input (common implementation).
///
/// @param[in] input Either a filesystem path or a byte span containing glTF data.
/// @return A tuple containing the character, skeleton states, and timestamps.
std::tuple<Character, std::vector<SkeletonState>, std::vector<float>>
loadCharacterWithSkeletonStatesCommon(
    const std::variant<filesystem::path, std::span<const std::byte>>& input);

} // namespace momentum
