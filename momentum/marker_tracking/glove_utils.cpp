/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/marker_tracking/glove_utils.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character/types.h"
#include "momentum/common/checks.h"
#include "momentum/common/log.h"
#include "momentum/math/utility.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <string>
#include <vector>

namespace momentum {

Character addGloveBones(
    Character character,
    const GloveConfig& cfg,
    const std::array<GloveOffset, 2>& offsets,
    const std::string& prefix) {
  auto& newSkel = character.skeleton;
  auto& newTransform = character.parameterTransform;
  auto& newInvBindPose = character.inverseBindPose;

  for (size_t hand = 0; hand < 2; ++hand) {
    const auto& wristName = cfg.wristJointNames[hand];

    const size_t wristIdx = newSkel.getJointIdByName(wristName);
    if (wristIdx == kInvalidIndex) {
      MT_LOGW("addGloveBones: wrist joint '{}' not found, skipping", wristName);
      continue;
    }

    // Skip if glove bone already exists
    const std::string gloveBoneName = prefix + wristName;
    if (newSkel.getJointIdByName(gloveBoneName) != kInvalidIndex) {
      continue;
    }

    // Create a new joint as child of the wrist
    Joint joint;
    joint.name = gloveBoneName;
    joint.parent = wristIdx;
    joint.translationOffset = offsets[hand].translation;
    joint.preRotation = eulerToQuaternion<float>(
        offsets[hand].rotationEulerXYZ.cast<float>(), 0, 1, 2, EulerConvention::Intrinsic);

    newSkel.joints.push_back(joint);
    newInvBindPose.push_back(Affine3f::Identity());

    // Resize offsets and activeJointParams for the new joint
    newTransform.offsets.conservativeResize(newTransform.offsets.size() + kParametersPerJoint);
    newTransform.activeJointParams.conservativeResize(
        newTransform.activeJointParams.size() + kParametersPerJoint);
    for (size_t j = newTransform.offsets.size() - kParametersPerJoint;
         j < static_cast<size_t>(newTransform.offsets.size());
         ++j) {
      newTransform.offsets(j) = 0.0f;
    }
    for (size_t j = newTransform.activeJointParams.size() - kParametersPerJoint;
         j < static_cast<size_t>(newTransform.activeJointParams.size());
         ++j) {
      newTransform.activeJointParams(j) = true;
    }

    // Resize the transform matrix rows for the new joint (no new columns)
    const int newRows = static_cast<int>(newSkel.joints.size()) * kParametersPerJoint;
    const int curCols = static_cast<int>(newTransform.name.size());
    newTransform.transform.conservativeResize(newRows, curCols);
  }

  return character;
}

Character addGloveCalibrationParameters(
    Character character,
    const GloveConfig& cfg,
    const std::string& prefix) {
  auto& newSkel = character.skeleton;
  auto& newTransform = character.parameterTransform;

  ParameterSet gloveSet;
  std::vector<Eigen::Triplet<float>> triplets;

  // Parameter name suffixes for translation and rotation DOFs
  static const std::array<std::string, 6> dofNames{"_tx", "_ty", "_tz", "_rx", "_ry", "_rz"};
  // Corresponding joint parameter indices: TX=0, TY=1, TZ=2, RX=3, RY=4, RZ=5
  static const std::array<size_t, 6> dofIndices{TX, TY, TZ, RX, RY, RZ};

  for (size_t hand = 0; hand < 2; ++hand) {
    const std::string gloveBoneName = prefix + cfg.wristJointNames[hand];
    const size_t jointIdx = newSkel.getJointIdByName(gloveBoneName);
    if (jointIdx == kInvalidIndex) {
      MT_LOGW("addGloveCalibrationParameters: glove bone '{}' not found, skipping", gloveBoneName);
      continue;
    }

    // Add 6 DOF parameters (TX, TY, TZ, RX, RY, RZ) for this glove bone
    for (size_t d = 0; d < 6; ++d) {
      const std::string paramName = gloveBoneName + dofNames[d];
      const auto row =
          static_cast<int>(jointIdx) * kParametersPerJoint + static_cast<int>(dofIndices[d]);
      const auto col = static_cast<int>(newTransform.name.size());

      triplets.emplace_back(row, col, 1.0f);

      MT_CHECK(
          newTransform.name.size() < kMaxModelParams,
          "Number of model parameters reached the {} limit.",
          kMaxModelParams);
      gloveSet.set(newTransform.name.size());
      newTransform.name.push_back(paramName);
    }
  }

  newTransform.parameterSets["gloves"] = gloveSet;

  // Update the parameter transform matrix
  const int newRows = static_cast<int>(newSkel.joints.size()) * kParametersPerJoint;
  const int newCols = static_cast<int>(newTransform.name.size());
  newTransform.transform.conservativeResize(newRows, newCols);
  SparseRowMatrixf additionalTransforms(newRows, newCols);
  additionalTransforms.setFromTriplets(triplets.begin(), triplets.end());
  newTransform.transform += additionalTransforms;

  return character;
}

Character
createGloveCharacter(const Character& src, const GloveConfig& cfg, const std::string& prefix) {
  return addGloveCalibrationParameters(addGloveBones(src, cfg, {}, prefix), cfg, prefix);
}

std::array<GloveOffset, 2> extractGloveOffsetsFromCharacter(
    const Character& gloveChar,
    const CharacterParameters& params,
    const GloveConfig& cfg) {
  std::array<GloveOffset, 2> result;

  // Apply parameter transform to get joint parameters
  const JointParameters jointParams = gloveChar.parameterTransform.apply(params);

  for (size_t hand = 0; hand < 2; ++hand) {
    const std::string gloveBoneName = "glove_" + cfg.wristJointNames[hand];
    const size_t jointIdx = gloveChar.skeleton.getJointIdByName(gloveBoneName);
    if (jointIdx == kInvalidIndex) {
      continue;
    }

    const size_t base = jointIdx * kParametersPerJoint;
    result[hand].translation =
        Vector3f(jointParams.v(base + TX), jointParams.v(base + TY), jointParams.v(base + TZ));
    result[hand].rotationEulerXYZ =
        Vector3f(jointParams.v(base + RX), jointParams.v(base + RY), jointParams.v(base + RZ));
  }

  return result;
}

} // namespace momentum
