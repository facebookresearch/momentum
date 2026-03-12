/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/common/filesystem.h>

#include <string>
#include <vector>

namespace momentum {

/// Command-line options for the convert_model tool.
struct Options {
  std::string input_model_file;
  std::string input_params_file;
  std::string input_locator_file;
  std::string input_motion_file;
  std::string output_model_file;
  std::string output_locator_local;
  std::string output_locator_global;
  std::string fbx_namespace;
  bool save_markers = false;
  bool character_mesh_save = false;
};

/// Bundles the result of loading motion data from a file.
struct MotionData {
  MatrixXf poses;
  JointParameters offsets;
  float fps = 120.0f;
};

/// Finds the index of the longest motion in a list of motions.
/// Returns -1 if no valid motion is found.
int findLongestMotion(const std::vector<MatrixXf>& motions);

/// Loads motion data from a GLB file. If no separate model was provided,
/// the character is also loaded from the GLB file.
MotionData loadGlbMotion(
    const filesystem::path& motionPath,
    Character& character,
    bool hasModel,
    const Options& options);

/// Loads motion data from an FBX file. If no separate model was provided,
/// the character is also initialized from the FBX file.
MotionData loadFbxMotion(
    const filesystem::path& motionPath,
    Character& character,
    bool hasModel,
    const Options& options);

/// Initializes the character from FBX data when no separate model was provided.
/// Applies parameter transform and locators if specified in the options.
void initializeCharacterFromFbx(
    Character& character,
    const Character& fbxCharacter,
    const Options& options);

/// Converts joint-parameter motion data to model parameters using
/// inverse parameter transform. May lose information.
MatrixXf convertToModelParams(const MatrixXf& motion, const ParameterTransform& parameterTransform);

/// Converts joint-parameter motion from a source skeleton to model parameters
/// on a target character by mapping global joint transforms based on joint names,
/// then converting back to joint parameters based on the target skeleton's bind
/// pose. This handles cases where the source and target skeletons have different
/// joint orders or rest/bind poses.
MatrixXf convertMotionWithJointMapping(
    const MatrixXf& motion,
    const Skeleton& sourceSkeleton,
    const Character& targetCharacter);

} // namespace momentum
