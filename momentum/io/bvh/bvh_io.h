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
#include <momentum/math/types.h>

#include <span>
#include <tuple>
#include <vector>

namespace momentum {

/// Loads a character from a BVH (BioVision Hierarchy) file.
///
/// The character is created with an identity ParameterTransform where model parameters
/// map directly to joint parameters (7 per joint: tx, ty, tz, rx, ry, rz, sc).
///
/// @param[in] filepath The path to the BVH file.
/// @return The loaded Character object.
[[nodiscard]] Character loadBvhCharacter(const filesystem::path& filepath);

/// Loads a character from a BVH file buffer.
///
/// @param[in] bytes The buffer containing the BVH file data.
/// @return The loaded Character object.
[[nodiscard]] Character loadBvhCharacter(std::span<const std::byte> bytes);

/// Loads a character with motion data from a BVH (BioVision Hierarchy) file.
///
/// Returns the character with an identity ParameterTransform, along with the motion data
/// represented as model parameters and the frame rate. BVH rotation values are converted
/// from degrees to radians and mapped to Momentum's ZYX Euler convention.
///
/// @param[in] filepath The path to the BVH file.
/// @return A tuple of (Character, MotionParameters, fps).
[[nodiscard]] std::tuple<Character, MotionParameters, float> loadBvhCharacterWithMotion(
    const filesystem::path& filepath);

/// Loads a character with motion data from a BVH file buffer.
///
/// @param[in] bytes The buffer containing the BVH file data.
/// @return A tuple of (Character, MotionParameters, fps).
[[nodiscard]] std::tuple<Character, MotionParameters, float> loadBvhCharacterWithMotion(
    std::span<const std::byte> bytes);

/// Saves a character to a BVH file with a single frame of its rest pose (zero model parameters) at
/// 30 frames per second. See saveBvhCharacterWithMotion() for the file content.
///
/// @param[in] filepath The output BVH file path.
/// @param[in] character The character to save.
void saveBvhCharacter(const filesystem::path& filepath, const Character& character);

/// Saves a character with motion data to a BVH file.
///
/// The joint parameters of frame i are `transform * motion.col(i) + offsets`, as
/// FbxBuilder::addMotion() computes them: the identity offsets take the place of the offsets of
/// the parameter transform, and empty offsets add nothing.
///
/// The HIERARCHY section has one ROOT per root joint. Root joints, and joints whose translation is
/// not zero (to the 4 written decimals) in some frame, get 6 channels (Xposition, Yposition,
/// Zposition, Zrotation, Xrotation, Yrotation); other joints get the 3 rotation channels. The
/// position channels hold the translation joint parameters, which are relative to the joint's
/// OFFSET, as loadBvhCharacterWithMotion() reads them. BVH joints have no pre-rotation, so each
/// joint's pre-rotation is folded into its rotation channels, in degrees.
///
/// BVH cannot store joint scales, meshes, locators, collision geometry, physical properties or
/// metadata: they are not saved, with a warning. The function throws for a character without
/// joints, a motion with another number of rows than the model parameters, offsets with another
/// size than the joint parameters, and a frame rate that is not positive; the file does not change
/// then.
///
/// @param[in] filepath The output BVH file path.
/// @param[in] character The character to save.
/// @param[in] motionParams The motion: model parameters of the character (numAllModelParameters x
/// frames). The parameter names are not used.
/// @param[in] fps The frame rate for the BVH file.
/// @param[in] offsets The identity joint parameters (numJointParameters values), or empty.
void saveBvhCharacterWithMotion(
    const filesystem::path& filepath,
    const Character& character,
    const MotionParameters& motionParams,
    float fps,
    const VectorXf& offsets = VectorXf());

/// Saves a character with motion data given as joint parameters to a BVH file, as
/// saveBvhCharacterWithMotion() saves model parameters. The parameter transform of the character
/// is not used.
///
/// @param[in] filepath The output BVH file path.
/// @param[in] character The character to save.
/// @param[in] jointParams The joint parameters of the character (numJointParameters x frames).
/// @param[in] fps The frame rate for the BVH file.
void saveBvhCharacterWithJointParams(
    const filesystem::path& filepath,
    const Character& character,
    const MatrixXf& jointParams,
    float fps);

/// Saves a character with motion data to a byte buffer in BVH format, with the content of
/// saveBvhCharacterWithMotion().
///
/// @param[in] character The character to save.
/// @param[in] motionParams The motion: model parameters of the character (numAllModelParameters x
/// frames). The parameter names are not used.
/// @param[in] fps The frame rate for the BVH file.
/// @param[in] offsets The identity joint parameters (numJointParameters values), or empty.
/// @return The BVH file content as a byte buffer.
[[nodiscard]] std::vector<std::byte> saveBvhCharacterWithMotion(
    const Character& character,
    const MotionParameters& motionParams,
    float fps,
    const VectorXf& offsets = VectorXf());

} // namespace momentum
