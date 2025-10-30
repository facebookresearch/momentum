/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/marker_tracking/tracker_utils.h"

#include "momentum/character/blend_shape.h"
#include "momentum/character/blend_shape_skinning.h"
#include "momentum/character/character.h"
#include "momentum/character/inverse_parameter_transform.h"
#include "momentum/character/parameter_limits.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_solver/position_error_function.h"
#include "momentum/character_solver/skinned_locator_error_function.h"
#include "momentum/math/mesh.h"
#include "momentum/math/utility.h"

#include "momentum/common/log.h"

#include "axel/math/PointTriangleProjectionDefinitions.h"

using namespace momentum;

namespace momentum {

std::vector<std::vector<PositionData>> createConstraintData(
    const std::span<const std::vector<Marker>> markerData,
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

std::vector<std::vector<SkinnedLocatorConstraint>> createSkinnedConstraintData(
    const std::span<const std::vector<Marker>> markerData,
    const SkinnedLocatorList& locators) {
  std::vector<std::vector<SkinnedLocatorConstraint>> results(markerData.size());

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

      SkinnedLocatorConstraint skinnedConstraint;
      skinnedConstraint.locatorIndex = gsl::narrow_cast<int>(locatorIdx);
      skinnedConstraint.targetPosition = jMarker.pos.cast<float>();
      skinnedConstraint.weight = locators[locatorIdx].weight;
      results.at(iFrame).push_back(skinnedConstraint);
    }
  }

  return results;
}

std::pair<Eigen::Vector<uint32_t, kMaxSkinJoints>, Eigen::Vector<float, kMaxSkinJoints>>
averageTriangleSkinWeights(
    const Character& character,
    int triangleIndex,
    const Eigen::Vector3f& barycentric) {
  Eigen::VectorXf skinWeights = Eigen::VectorXf::Zero(character.skeleton.joints.size());

  const auto& mesh = *character.mesh;
  const auto& skin = *character.skinWeights;

  // get the triangle
  const auto& triangle = mesh.faces[triangleIndex];
  for (int kTriVert = 0; kTriVert < 3; ++kTriVert) {
    for (int kSkinWeight = 0; kSkinWeight < skin.index.cols(); ++kSkinWeight) {
      const auto& skinIndex = skin.index(triangle(kTriVert), kSkinWeight);
      const auto& skinWeight = skin.weight(triangle(kTriVert), kSkinWeight);
      skinWeights[skinIndex] += skinWeight * barycentric[kTriVert];
    }
  }

  std::vector<std::pair<float, uint32_t>> sortedWeights;
  for (Eigen::Index i = 0; i < skinWeights.size(); ++i) {
    if (skinWeights[i] > 0.0f) {
      sortedWeights.emplace_back(skinWeights[i], static_cast<uint32_t>(i));
    }
  }
  std::sort(sortedWeights.begin(), sortedWeights.end(), std::greater<>());

  Eigen::Vector<float, kMaxSkinJoints> resultWeights = Eigen::Vector<float, kMaxSkinJoints>::Zero();
  Eigen::Vector<uint32_t, kMaxSkinJoints> resultIndices =
      Eigen::Vector<uint32_t, kMaxSkinJoints>::Zero();
  for (size_t i = 0; i < kMaxSkinJoints; ++i) {
    if (i < sortedWeights.size()) {
      resultWeights[i] = sortedWeights[i].first;
      resultIndices[i] = sortedWeights[i].second;
    }
  }

  resultWeights /= resultWeights.sum();

  return {resultIndices, resultWeights};
}

namespace {

struct ClosestPointOnMeshResult {
  bool valid = false;
  size_t triangleIdx = kInvalidIndex;
  Eigen::Vector3f baryCoords = Eigen::Vector3f::Zero();
  float distance = std::numeric_limits<float>::max();
};

} // namespace

ClosestPointOnMeshResult closestPointOnMeshMatchingParent(
    const momentum::Mesh& mesh,
    const momentum::SkinWeights& skin,
    const Eigen::Vector3f& p_world,
    uint32_t parentIdx,
    float cutoffWeight = 0.1) {
  ClosestPointOnMeshResult result;
  for (size_t iTri = 0; iTri < mesh.faces.size(); ++iTri) {
    const auto& f = mesh.faces[iTri];

    float skinWeight = 0;
    for (int kTriVert = 0; kTriVert < 3; ++kTriVert) {
      for (int kSkinWeight = 0; kSkinWeight < skin.index.cols(); ++kSkinWeight) {
        if (skin.index(f(kTriVert), kSkinWeight) == parentIdx) {
          skinWeight += skin.weight(f(kTriVert), kSkinWeight);
        }
      }
    }
    skinWeight /= 3.0f;

    if (skinWeight < cutoffWeight) {
      continue;
    }

    // Note: the return value from the does _not_ have anything to do with
    // whether the projection is valid, it just indicates whether the projection is
    // inside the triangle or not, so we can ignore it and just use the projected point directly.
    Eigen::Vector3f p_tri;
    Eigen::Vector3f bary;
    axel::projectOnTriangle(
        p_world, mesh.vertices[f(0)], mesh.vertices[f(1)], mesh.vertices[f(2)], p_tri, &bary);
    const float dist = (p_tri - p_world).norm();
    if (!result.valid || dist < result.distance) {
      result = {true, iTri, bary, dist};
    }
  }

  return result;
}

momentum::Character locatorsToSkinnedLocators(
    const momentum::Character& sourceCharacter,
    float maxDistance) {
  if (!sourceCharacter.mesh || !sourceCharacter.skinWeights) {
    return sourceCharacter;
  }

  const JointParameters restJointParams =
      JointParameters::Zero(sourceCharacter.parameterTransform.numJointParameters());
  const SkeletonState restState(restJointParams, sourceCharacter.skeleton);

  SkinnedLocatorList skinnedLocators;
  LocatorList locators;
  for (size_t i = 0; i < sourceCharacter.locators.size(); ++i) {
    const auto& locator = sourceCharacter.locators[i];
    const auto& offset = locator.offset;
    const auto& parent = locator.parent;
    const Eigen::Vector3f p_world = restState.jointState[parent].transform * offset;

    // Find the closest point on the mesh to the locator.
    const auto closestPointResult = closestPointOnMeshMatchingParent(
        *sourceCharacter.mesh,
        *sourceCharacter.skinWeights,
        p_world,
        gsl::narrow_cast<uint32_t>(parent));
    if (!closestPointResult.valid || closestPointResult.distance > maxDistance) {
      locators.push_back(locator);
      continue;
    }

    SkinnedLocator skinnedLocator;
    skinnedLocator.name = locator.name;
    skinnedLocator.position = p_world;
    skinnedLocator.weight = locator.weight;

    // Set the locator's parent to the closest face.
    std::tie(skinnedLocator.parents, skinnedLocator.skinWeights) = averageTriangleSkinWeights(
        sourceCharacter, closestPointResult.triangleIdx, closestPointResult.baryCoords);
    skinnedLocators.push_back(skinnedLocator);
  }

  // Strip out regular locators and replace with skinned locators:
  return {
      sourceCharacter.skeleton,
      sourceCharacter.parameterTransform,
      sourceCharacter.parameterLimits,
      locators,
      sourceCharacter.mesh.get(),
      sourceCharacter.skinWeights.get(),
      sourceCharacter.collision.get(),
      sourceCharacter.poseShapes.get(),
      sourceCharacter.blendShape,
      sourceCharacter.faceExpressionBlendShape,
      sourceCharacter.name,
      sourceCharacter.inverseBindPose,
      skinnedLocators};
}

std::vector<momentum::SkinnedLocatorTriangleConstraintT<float>> createSkinnedLocatorMeshConstraints(
    const momentum::Character& character,
    float targetDepth) {
  if (!character.mesh || !character.skinWeights) {
    return {};
  }

  // skip the triangle tree if we don't need it:
  if (character.skinnedLocators.empty()) {
    return {};
  }

  const auto& mesh = *character.mesh;

  std::vector<momentum::SkinnedLocatorTriangleConstraintT<float>> result;
  result.reserve(character.skinnedLocators.size());
  for (size_t i = 0; i < character.skinnedLocators.size(); ++i) {
    const auto& locator = character.skinnedLocators[i];
    const Eigen::Vector3f p_world = locator.position;

    // Find the closest point on the mesh to the locator.
    const auto closestPointResult = closestPointOnMeshMatchingParent(
        *character.mesh, *character.skinWeights, p_world, locator.parents[0], targetDepth);
    if (!closestPointResult.valid) {
      continue;
    }

    momentum::SkinnedLocatorTriangleConstraintT<float> constr;
    constr.locatorIndex = i;
    constr.tgtTriangleIndices = mesh.faces[closestPointResult.triangleIdx];
    constr.tgtTriangleBaryCoords = closestPointResult.baryCoords;
    constr.depth = targetDepth;
    result.push_back(constr);
  }

  return result;
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

  // Check if locators is empty
  if (locators.empty()) {
    return result;
  }

  // revert each locator to the original attachment and add an offset
  for (size_t i = 0; i < locators.size(); i++) {
    // only map locators back that are not fixed
    if (locators[i].locked.all()) {
      continue;
    }

    // joint id
    const size_t jointId = locators[i].parent;

    // Check bounds for jointState access
    if (jointId >= state.jointState.size()) {
      MT_LOGW(
          "Joint ID {} is out of bounds for jointState (size: {})",
          jointId,
          state.jointState.size());
      continue;
    }

    // get global locator position
    const Vector3f pos = state.jointState[jointId].transform * locators[i].offset;

    // change attachment to original joint
    result[i].parent = skeleton.joints[jointId].parent;

    // Check bounds for parent joint access
    if (result[i].parent >= state.jointState.size()) {
      MT_LOGW(
          "Parent joint ID {} is out of bounds for jointState (size: {})",
          result[i].parent,
          state.jointState.size());
      continue;
    }

    // calculate new offset
    const Vector3f offset = state.jointState[result[i].parent].transform.inverse() * pos;

    // change offset to current state
    result[i].offset = offset;
  }

  // return
  return result;
}

SkinnedLocatorList extractSkinnedLocatorsFromCharacter(
    const Character& locatorCharacter,
    const CharacterParameters& calibParams) {
  const SkeletonState state(
      locatorCharacter.parameterTransform.apply(calibParams), locatorCharacter.skeleton);
  const SkinnedLocatorList& skinnedLocators = locatorCharacter.skinnedLocators;
  const auto& pt = locatorCharacter.parameterTransform;

  SkinnedLocatorList result = skinnedLocators;

  for (size_t i = 0; i < skinnedLocators.size(); i++) {
    if (i < pt.skinnedLocatorParameters.size()) {
      auto paramIdx = pt.skinnedLocatorParameters[i];
      for (int k = 0; k < 3; ++k) {
        result.at(i).position(k) += calibParams.pose[paramIdx + k];
      }
    }
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

std::tuple<Eigen::VectorXf, LocatorList, SkinnedLocatorList> extractIdAndLocatorsFromParams(
    const ModelParameters& param,
    const Character& sourceCharacter,
    const Character& targetCharacter) {
  ModelParameters idParam =
      extractParameters(param, targetCharacter.parameterTransform.getScalingParameters());
  CharacterParameters fullParams;
  fullParams.pose = param;
  LocatorList locators = extractLocatorsFromCharacter(sourceCharacter, fullParams);
  SkinnedLocatorList skinnedLocators =
      extractSkinnedLocatorsFromCharacter(sourceCharacter, fullParams);

  return {
      idParam.v.head(targetCharacter.parameterTransform.numAllModelParameters()),
      locators,
      skinnedLocators};
}

Mesh extractBlendShapeFromParams(
    const momentum::ModelParameters& param,
    const momentum::Character& sourceCharacter) {
  Mesh result = *sourceCharacter.mesh;
  auto blendWeights = extractBlendWeights(sourceCharacter.parameterTransform, param);
  result.vertices = sourceCharacter.blendShape->computeShape<float>(blendWeights);
  return result;
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

ModelParameters jointIdentityToModelIdentity(
    const Character& c,
    const ParameterSet& idSet,
    const JointParameters& jointIdentity) {
  // convert from joint to model parameters using only the parameters active in idSet
  auto scalingTransform = c.parameterTransform.simplify(idSet);
  momentum::ModelParameters scaleParameters =
      InverseParameterTransform(scalingTransform).apply(jointIdentity).pose;
  momentum::ModelParameters identity =
      ModelParameters::Zero(c.parameterTransform.numAllModelParameters());
  size_t iScale = 0;
  for (size_t iParam = 0; iParam < c.parameterTransform.numAllModelParameters(); iParam++) {
    if (idSet.test(iParam)) {
      identity[iParam] = scaleParameters[iScale];
      ++iScale;
    }
  }

  return identity;
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
      const Vector3d pos = (states.at(loc.parent).transform * loc.offset).cast<double>();
      markers.emplace_back(Marker{loc.name, pos, false});
    }
  }

  return result;
}

} // namespace momentum
