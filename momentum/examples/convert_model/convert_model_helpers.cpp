/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define DEFAULT_LOG_CHANNEL "convert_model"

#include "convert_model_helpers.h"

#include <momentum/character/character_utility.h>
#include <momentum/character/inverse_parameter_transform.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/common/log.h>
#include <momentum/io/character_io.h>
#include <momentum/io/fbx/fbx_io.h>
#include <momentum/io/gltf/gltf_io.h>
#include <momentum/io/skeleton/locator_io.h>
#include <momentum/io/skeleton/parameter_transform_io.h>
#include <momentum/io/skeleton/parameters_io.h>

namespace momentum {

int findLongestMotion(const std::vector<MatrixXf>& motions) {
  if (motions.empty() || (motions.size() == 1 && motions.at(0).cols() == 0)) {
    MT_LOGW("No motion loaded from file");
    return -1;
  }

  int motionIndex = -1;
  size_t nFrames = 0;
  for (size_t iMotion = 0; iMotion < motions.size(); ++iMotion) {
    const size_t length = motions.at(iMotion).cols();
    if (length > nFrames) {
      nFrames = length;
      motionIndex = static_cast<int>(iMotion);
    }
  }
  if (nFrames > 0 && motions.size() > 1) {
    MT_LOGW("More than one motion found; only taking the longest one");
  }
  return motionIndex;
}

MotionData loadGlbMotion(
    const filesystem::path& motionPath,
    Character& character,
    bool hasModel,
    const Options& options) {
  MT_LOGI("Loading motion from glb...");
  MotionData result;
  if (hasModel) {
    std::tie(result.poses, result.offsets, result.fps) =
        loadMotionOnCharacter(motionPath, character);
    return result;
  }

  std::tie(character, result.poses, result.offsets, result.fps) =
      loadCharacterWithMotion(motionPath);
  if (!options.input_params_file.empty()) {
    MT_LOGW("Ignoring input parameter transform {}.", options.input_params_file);
  }
  if (!options.input_locator_file.empty()) {
    MT_LOGW("Ignoring input locators {}.", options.input_locator_file);
  }
  return result;
}

void initializeCharacterFromFbx(
    Character& character,
    const Character& fbxCharacter,
    const Options& options) {
  character = fbxCharacter;
  if (!options.input_params_file.empty()) {
    auto def = loadMomentumModel(options.input_params_file);
    loadParameters(def, character);
  } else {
    character.parameterTransform = ParameterTransform::identity(character.skeleton.getJointNames());
  }
  if (!options.input_locator_file.empty()) {
    character.locators =
        loadLocators(options.input_locator_file, character.skeleton, character.parameterTransform);
  }
}

MatrixXf convertToModelParams(
    const MatrixXf& motion,
    const ParameterTransform& parameterTransform) {
  const size_t nFrames = motion.cols();
  MatrixXf poses;
  poses.setZero(parameterTransform.numAllModelParameters(), nFrames);
  InverseParameterTransform inversePt(parameterTransform);
  for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
    poses.col(iFrame) = inversePt.apply(motion.col(iFrame)).pose.v;
  }
  return poses;
}

MatrixXf convertMotionWithJointMapping(
    const MatrixXf& motion,
    const Skeleton& sourceSkeleton,
    const Character& targetCharacter) {
  const size_t nFrames = motion.cols();
  const size_t nJoints = targetCharacter.skeleton.joints.size();
  MatrixXf poses;
  poses.setZero(targetCharacter.parameterTransform.numAllModelParameters(), nFrames);
  InverseParameterTransform inversePt(targetCharacter.parameterTransform);
  SkeletonState sourceState;
  TransformList transforms(nJoints);

  // Build name mapping: indexMap[target_index] = source_index
  const auto jointNames = targetCharacter.skeleton.getJointNames();
  std::vector<size_t> indexMap(nJoints, kInvalidIndex);
  for (size_t iJoint = 0; iJoint < nJoints; ++iJoint) {
    indexMap[iJoint] = sourceSkeleton.getJointIdByName(jointNames[iJoint]);
  }

  for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
    sourceState.set(motion.col(iFrame), sourceSkeleton);

    for (size_t jJoint = 0; jJoint < nJoints; ++jJoint) {
      if (indexMap[jJoint] == kInvalidIndex) {
        // No correspondence in source; use rest pose
        const Joint& joint = targetCharacter.skeleton.joints[jJoint];
        const Transform local(joint.translationOffset, joint.preRotation);
        const size_t parentIndex = targetCharacter.skeleton.joints[jJoint].parent;
        if (parentIndex == kInvalidIndex) {
          transforms[jJoint] = local;
        } else {
          transforms[jJoint] = transforms[parentIndex] * local;
        }
      } else {
        transforms[jJoint] = sourceState.jointState[indexMap[jJoint]].transform;
      }
    }
    const auto jointParams = skeletonStateToJointParameters(transforms, targetCharacter.skeleton);
    poses.col(iFrame) = inversePt.apply(jointParams).pose.v;
  }
  return poses;
}

MotionData loadFbxMotion(
    const filesystem::path& motionPath,
    Character& character,
    bool hasModel,
    const Options& options) {
  MT_LOGI("Loading motion from fbx...");
  MotionData result;

  auto [c, motions, framerate] =
      loadFbxCharacterWithMotion(motionPath, KeepLocators::Yes, Permissive::No);

  const int motionIndex = findLongestMotion(motions);

  if (!hasModel) {
    initializeCharacterFromFbx(character, c, options);
  }

  if (motionIndex < 0) {
    return result;
  }

  // To account for differences in joint order and rest pose between the source
  // and target skeletons, map global joint transforms by name then convert back
  // to joint parameters based on the target skeleton's bind pose.
  result.poses = convertMotionWithJointMapping(motions.at(motionIndex), c.skeleton, character);
  result.fps = framerate;
  result.offsets = character.parameterTransform.zero();
  return result;
}

} // namespace momentum
