/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/marker_tracking/tracker_utils.h"

#include "momentum/character/character.h"
#include "momentum/character/parameter_limits.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_solver/position_error_function.h"
#include "momentum/common/log.h"

using namespace momentum;

namespace momentum {

std::vector<std::vector<PositionData>> createConstraintData(
    const gsl::span<const std::vector<Marker>> markerData,
    const LocatorList& locators) {
  std::vector<std::vector<PositionData>> results(markerData.size());

  // map locator name to its index
  std::map<std::string, size_t> locatorLookup;
  for (size_t i = 0; i < locators.size(); i++) {
    locatorLookup[locators[i].name] = i;
  }

  // create a list of position constraints per frame
  for (size_t iFrame = 0; iFrame < markerData.size(); ++iFrame) {
    const auto& markerList = markerData[iFrame];
    for (const auto& jMarker : markerList) {
      if (jMarker.occluded) {
        continue;
      }
      auto query = locatorLookup.find(jMarker.name);
      if (query == locatorLookup.end()) {
        continue;
      }
      size_t locatorIdx = query->second;

      results.at(iFrame).emplace_back(PositionData(
          locators.at(locatorIdx).offset,
          jMarker.pos.cast<float>(),
          locators.at(locatorIdx).parent,
          locators.at(locatorIdx).weight));
    }
  }

  return results;
}

Character createLocatorCharacter(const Character& sourceCharacter, const std::string& prefix) {
  // create a copy
  Character character = sourceCharacter;

  auto& newSkel = character.skeleton;
  auto& newTransform = character.parameterTransform;
  auto& newLocators = character.locators;
  auto& newLimits = character.parameterLimits;
  auto& newInvBindPose = character.inverseBindPose;

  ParameterSet locatorSet;

  // create a new additional joint for each locator
  std::vector<Eigen::Triplet<float>> triplets;
  for (auto& newLocator : newLocators) {
    // ignore locator if it is fixed
    if (newLocator.locked.all()) {
      continue;
    }

    // create a new joint for the locator
    Joint joint;
    joint.name = std::string(prefix + newLocator.name);
    joint.parent = newLocator.parent;
    joint.translationOffset = newLocator.offset;

    // insert joint
    const size_t id = newSkel.joints.size();
    newSkel.joints.push_back(joint);
    newInvBindPose.push_back(Affine3f::Identity());

    // create parameter for the added joint
    static const std::array<std::string, 3> tNames{"_tx", "_ty", "_tz"};
    for (size_t j = 0; j < 3; j++) {
      if (newLocator.locked[j] != 0) {
        continue;
      }
      const std::string jname = joint.name + tNames[j];
      triplets.emplace_back(
          static_cast<int>(id) * kParametersPerJoint + static_cast<int>(j),
          static_cast<int>(newTransform.name.size()),
          1.0f);
      MT_CHECK(
          newTransform.name.size() < kMaxModelParams,
          "Number of model parameters reached the {} limit.",
          kMaxModelParams);
      locatorSet.set(newTransform.name.size());
      newTransform.name.push_back(jname);

      // check if we need joint limit
      if (newLocator.limitWeight[j] > 0.0f) {
        ParameterLimit p;
        p.type = MinMaxJoint;
        p.data.minMaxJoint.jointIndex = id;
        p.data.minMaxJoint.jointParameter = j;
        const float referencePosition = newLocator.limitOrigin[j] - newLocator.offset[j];
        p.data.minMaxJoint.limits = Vector2f(referencePosition, referencePosition);
        p.weight = newLocator.limitWeight[j];
        newLimits.push_back(p);
      }
    }
    newTransform.offsets.conservativeResize(newTransform.offsets.size() + kParametersPerJoint);
    newTransform.activeJointParams.conservativeResize(
        newTransform.activeJointParams.size() + kParametersPerJoint);
    for (size_t j = newTransform.offsets.size() - gsl::narrow<int>(kParametersPerJoint);
         j < newTransform.offsets.size();
         ++j) {
      newTransform.offsets(j) = 0.0;
    }

    for (size_t j = newTransform.activeJointParams.size() - gsl::narrow<int>(kParametersPerJoint);
         j < newTransform.activeJointParams.size();
         ++j) {
      newTransform.activeJointParams(j) = true;
    }

    // reattach locator to new joint
    newLocator.parent = id;
    newLocator.offset.setZero();
  }
  newTransform.parameterSets["locators"] = locatorSet;

  // update parameter transform
  const int newRows = static_cast<int>(newSkel.joints.size()) * kParametersPerJoint;
  const int newCols = static_cast<int>(newTransform.name.size());
  newTransform.transform.conservativeResize(newRows, newCols);
  SparseRowMatrixf additionalTransforms(newRows, newCols);
  additionalTransforms.setFromTriplets(triplets.begin(), triplets.end());
  newTransform.transform += additionalTransforms;

  // return the new character
  return character;
}

LocatorList extractLocatorsFromCharacter(
    const Character& locatorCharacter,
    const CharacterParameters& calibParams) {
  const SkeletonState state(
      locatorCharacter.parameterTransform.apply(calibParams), locatorCharacter.skeleton);
  const LocatorList& locators = locatorCharacter.locators;
  const auto& skeleton = locatorCharacter.skeleton;

  LocatorList result = locators;
  // revert each locator to the original attachment and add an offset
  for (size_t i = 0; i < locators.size(); i++) {
    // only map locators back that are not fixed
    if (locators[i].locked.all()) {
      continue;
    }

    // joint id
    const size_t jointId = locators[i].parent;

    // get global locator position
    const Vector3f pos = state.jointState[jointId].transformation * locators[i].offset;

    // change attachment to original joint
    result[i].parent = skeleton.joints[jointId].parent;

    // calculate new offset
    const Vector3f offset = state.jointState[result[i].parent].transformation.inverse() * pos;

    // change offset to current state
    result[i].offset = offset;
  }

  // return
  return result;
}

ModelParameters extractParameters(const ModelParameters& params, const ParameterSet& parameterSet) {
  ModelParameters newParams = params;
  for (size_t iParam = 0; iParam < newParams.size(); ++iParam) {
    // TODO: check index out of bound
    if (!parameterSet[iParam]) {
      newParams[iParam] = 0.0;
    }
  }
  return newParams;
}

std::tuple<Eigen::VectorXf, LocatorList> extractIdAndLocatorsFromParams(
    const ModelParameters& param,
    const Character& sourceCharacter,
    const Character& targetCharacter) {
  ModelParameters idParam =
      extractParameters(param, targetCharacter.parameterTransform.getScalingParameters());
  CharacterParameters fullParams;
  fullParams.pose = param;
  LocatorList locators = extractLocatorsFromCharacter(sourceCharacter, fullParams);

  return {idParam.v.head(targetCharacter.parameterTransform.numAllModelParameters()), locators};
}

void fillIdentity(
    const ParameterSet& idSet,
    const ModelParameters& identity,
    Eigen::MatrixXf& motionNoId) {
  MT_CHECK(identity.v.size() == motionNoId.rows());

  const size_t numParams = motionNoId.rows();
  const size_t numFrames = motionNoId.cols();

  for (size_t iParam = 0; iParam < numParams; ++iParam) {
    if (!idSet.test(iParam)) {
      continue;
    }
    for (size_t jFrame = 0; jFrame < numFrames; ++jFrame) {
      motionNoId(iParam, jFrame) += identity.v(iParam);
    }
  }
}

void removeIdentity(
    const ParameterSet& idSet,
    const ModelParameters& identity,
    Eigen::MatrixXf& motionWithId) {
  const size_t numParams = motionWithId.rows();
  const size_t numFrames = motionWithId.cols();

  for (size_t iParam = 0; iParam < numParams; ++iParam) {
    if (!idSet.test(iParam)) {
      continue;
    }
    for (size_t jFrame = 0; jFrame < numFrames; ++jFrame) {
      motionWithId(iParam, jFrame) -= identity.v(iParam);
    }
  }
}

std::vector<std::vector<Marker>> extractMarkersFromMotion(
    const Character& character,
    const Eigen::MatrixXf& motion) {
  const size_t nFrames = motion.cols();
  std::vector<std::vector<Marker>> result(nFrames);
  for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
    const JointParameters params(motion.col(iFrame));
    const auto& states = SkeletonState(params, character.skeleton).jointState;

    std::vector<Marker>& markers = result.at(iFrame);
    for (const auto& loc : character.locators) {
      const Vector3d pos = (states.at(loc.parent).transformation * loc.offset).cast<double>();
      markers.emplace_back(Marker{loc.name, pos, false});
    }
  }

  return result;
}

} // namespace momentum
