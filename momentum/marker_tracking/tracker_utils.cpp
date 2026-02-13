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
#include "momentum/math/constants.h"
#include "momentum/math/mesh.h"
#include "momentum/math/utility.h"

#include "momentum/common/exception.h"
#include "momentum/common/log.h"

#include "axel/math/PointTriangleProjectionDefinitions.h"

#include <algorithm>
#include <queue>
#include <span>
#include <unordered_map>
#include <unordered_set>

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
  Eigen::Vector3f trianglePoint = Eigen::Vector3f::Zero();
  float distance = std::numeric_limits<float>::max();
};

} // namespace

/// Check if two joints are related (same joint, parent-child, or child-parent relationship).
/// @param skeleton The skeleton containing the joint hierarchy
/// @param jointA First joint index
/// @param jointB Second joint index
/// @return True if the joints are related (same, or one is the parent of the other)
bool isRelatedJoint(const momentum::Skeleton& skeleton, uint32_t jointA, uint32_t jointB) {
  if (jointA == jointB) {
    return true;
  }
  // Check if jointB is the parent of jointA
  if (jointA < skeleton.joints.size() && skeleton.joints[jointA].parent == jointB) {
    return true;
  }
  // Check if jointA is the parent of jointB
  if (jointB < skeleton.joints.size() && skeleton.joints[jointB].parent == jointA) {
    return true;
  }
  return false;
}

ClosestPointOnMeshResult closestPointOnMeshMatchingParent(
    const momentum::Mesh& mesh,
    const momentum::SkinWeights& skin,
    const momentum::Skeleton& skeleton,
    const Eigen::Vector3f& p_world,
    uint32_t parentIdx,
    float cutoffWeight = 0.02) {
  ClosestPointOnMeshResult result;

  for (size_t iTri = 0; iTri < mesh.faces.size(); ++iTri) {
    const auto& f = mesh.faces[iTri];

    // Check bounds for face indices (including negative values)
    MT_THROW_IF(
        f(0) < 0 || f(1) < 0 || f(2) < 0 || f(0) >= static_cast<int>(mesh.vertices.size()) ||
            f(1) >= static_cast<int>(mesh.vertices.size()) ||
            f(2) >= static_cast<int>(mesh.vertices.size()),
        "Face index out of bounds at triangle {}: face indices ({}, {}, {}), mesh vertices size {}",
        iTri,
        f(0),
        f(1),
        f(2),
        mesh.vertices.size());

    // Sum skin weights across the joint, its parent, and its children
    float skinWeight = 0;
    for (int kTriVert = 0; kTriVert < 3; ++kTriVert) {
      for (int kSkinWeight = 0; kSkinWeight < skin.index.cols(); ++kSkinWeight) {
        const uint32_t skinJointIdx = skin.index(f(kTriVert), kSkinWeight);
        if (isRelatedJoint(skeleton, skinJointIdx, parentIdx)) {
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
      result = {true, iTri, bary, p_tri, dist};
    }
  }

  return result;
}

momentum::Character locatorsToSkinnedLocators(
    const momentum::Character& sourceCharacter,
    float maxDistance,
    float minSkinWeight,
    bool verbose) {
  if (!sourceCharacter.mesh || !sourceCharacter.skinWeights) {
    MT_LOGI_IF(
        verbose,
        "Can't convert locators to skinned locators because mesh or skin weights are missing");
    return sourceCharacter;
  }

  const JointParameters restJointParams =
      JointParameters::Zero(sourceCharacter.parameterTransform.numJointParameters());
  const SkeletonState restState(restJointParams, sourceCharacter.skeleton);

  SkinnedLocatorList skinnedLocators;
  LocatorList locators;
  for (size_t i = 0; i < sourceCharacter.locators.size(); ++i) {
    const auto& locator = sourceCharacter.locators[i];

    if (!locator.attachedToSkin) {
      MT_LOGI_IF(
          verbose,
          "Not converting locator {} to skinned locator because it is not attached to skin",
          locator.name);
      locators.push_back(locator);
      continue;
    }

    const auto& offset = locator.offset;
    const auto& parent = locator.parent;
    const Eigen::Vector3f p_world = restState.jointState[parent].transform * offset;

    // Find the closest point on the mesh to the locator.
    const auto closestPointResult = closestPointOnMeshMatchingParent(
        *sourceCharacter.mesh,
        *sourceCharacter.skinWeights,
        sourceCharacter.skeleton,
        p_world,
        gsl::narrow_cast<uint32_t>(parent),
        minSkinWeight);
    if (!closestPointResult.valid) {
      MT_LOGI_IF(
          verbose,
          "Not converting locator {} to skinned locator because no closest point found matching parent {} ({})",
          locator.name,
          sourceCharacter.skeleton.joints.at(parent).name,
          parent);
      locators.push_back(locator);
      continue;
    }

    if (closestPointResult.distance > maxDistance) {
      MT_LOGI_IF(
          verbose,
          "Not converting locator {} to skinned locator because distance {} is greater than threshold {}",
          locator.name,
          closestPointResult.distance,
          maxDistance);
      locators.push_back(locator);
      continue;
    }

    SkinnedLocator skinnedLocator;
    skinnedLocator.name = locator.name;
    skinnedLocator.position = p_world;
    skinnedLocator.weight = locator.weight;
    MT_LOGT("Converting locator {} to skinned locator", locator.name);

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

momentum::Character skinnedLocatorsToLocators(const momentum::Character& sourceCharacter) {
  if (sourceCharacter.skinnedLocators.empty()) {
    MT_LOGI("No skinned locators to convert");
    return sourceCharacter;
  }

  const JointParameters restJointParams =
      JointParameters::Zero(sourceCharacter.parameterTransform.numJointParameters());
  const SkeletonState restState(restJointParams, sourceCharacter.skeleton);

  LocatorList locators = sourceCharacter.locators;

  for (const auto& skinnedLocator : sourceCharacter.skinnedLocators) {
    // Find the bone with the highest skin weight
    int maxWeightIdx = 0;
    float maxWeight = skinnedLocator.skinWeights[0];

    for (int i = 1; i < kMaxSkinJoints; ++i) {
      if (skinnedLocator.skinWeights[i] > maxWeight) {
        maxWeight = skinnedLocator.skinWeights[i];
        maxWeightIdx = i;
      }
    }

    // Get the parent bone index
    const uint32_t parentBoneIdx = skinnedLocator.parents[maxWeightIdx];

    // Transform the position from rest pose space to local bone space
    const Eigen::Vector3f offset =
        restState.jointState[parentBoneIdx].transform.inverse() * skinnedLocator.position;

    // Create a regular locator
    Locator locator;
    locator.name = skinnedLocator.name;
    locator.parent = parentBoneIdx;
    locator.offset = offset;
    locator.weight = skinnedLocator.weight;
    locator.locked = Vector3i::Zero();
    locator.limitOrigin = offset;
    locator.limitWeight = Vector3f::Zero();

    MT_LOGT(
        "Converting skinned locator {} to regular locator attached to bone {} ({})",
        skinnedLocator.name,
        sourceCharacter.skeleton.joints.at(parentBoneIdx).name,
        parentBoneIdx);

    locators.push_back(locator);
  }

  // Return character with skinned locators converted to regular locators
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
      {}};
}

std::vector<std::vector<size_t>> buildTriangleAdjacency(const Mesh& mesh) {
  std::vector<std::vector<size_t>> adjacency(mesh.faces.size());

  if (mesh.faces.empty()) {
    return adjacency;
  }

  // Build edge-to-triangles map
  // Edge is represented as pair of sorted vertex indices
  struct PairHash {
    size_t operator()(const std::pair<int, int>& p) const {
      return std::hash<int64_t>{}(
          (static_cast<int64_t>(p.first) << 32) | static_cast<uint32_t>(p.second));
    }
  };
  std::unordered_map<std::pair<int, int>, std::vector<size_t>, PairHash> edgeToTriangles;

  for (size_t iTri = 0; iTri < mesh.faces.size(); ++iTri) {
    const Eigen::Vector3i& face = mesh.faces[iTri];
    for (int i = 0; i < 3; ++i) {
      int v0 = face(i);
      int v1 = face((i + 1) % 3);
      // Sort to create canonical edge representation
      if (v0 > v1) {
        std::swap(v0, v1);
      }
      edgeToTriangles[{v0, v1}].push_back(iTri);
    }
  }

  // Build adjacency from edge-to-triangles map
  for (const auto& [edge, triangles] : edgeToTriangles) {
    // For each pair of triangles sharing this edge, mark them as adjacent
    for (size_t i = 0; i < triangles.size(); ++i) {
      for (size_t j = i + 1; j < triangles.size(); ++j) {
        adjacency[triangles[i]].push_back(triangles[j]);
        adjacency[triangles[j]].push_back(triangles[i]);
      }
    }
  }

  // Remove duplicates from adjacency lists
  for (auto& neighbors : adjacency) {
    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
  }

  return adjacency;
}

std::vector<CandidateTriangle> findCandidateTrianglesDfs(
    const Mesh& mesh,
    const SkinWeights& skin,
    const Skeleton& skeleton,
    std::span<const std::vector<size_t>> adjacency,
    size_t startTriangleIdx,
    const Eigen::Vector3f& initialPoint,
    const Eigen::Vector3f& referenceNormal,
    uint32_t parentJointIdx,
    float maxSearchDistanceCm,
    float maxNormalAngleDeg) {
  std::vector<CandidateTriangle> candidates;

  if (startTriangleIdx >= mesh.faces.size()) {
    return candidates;
  }

  const float cosMaxAngle = std::cos(maxNormalAngleDeg * pi<float>() / 180.0f);
  const float cutoffWeight = 0.02f;

  std::unordered_set<size_t> visited;
  std::queue<size_t> toVisit;

  toVisit.push(startTriangleIdx);
  visited.insert(startTriangleIdx);

  while (!toVisit.empty()) {
    size_t triIdx = toVisit.front();
    toVisit.pop();

    MT_CHECK(triIdx < mesh.faces.size());

    const Eigen::Vector3i& face = mesh.faces[triIdx];
    const Eigen::Vector3f& v0 = mesh.vertices[face(0)];
    const Eigen::Vector3f& v1 = mesh.vertices[face(1)];
    const Eigen::Vector3f& v2 = mesh.vertices[face(2)];

    // Compute centroid
    const Eigen::Vector3f centroid = (v0 + v1 + v2) / 3.0f;

    // Distance check - if too far, don't add and don't continue DFS from here
    const float dist = (centroid - initialPoint).norm();
    if (dist > maxSearchDistanceCm) {
      continue;
    }

    // Compute triangle normal
    Eigen::Vector3f normal = (v1 - v0).cross(v2 - v0);
    const float normalNorm = normal.norm();
    if (normalNorm < 1e-8f) {
      continue; // Degenerate triangle
    }
    normal /= normalNorm;

    // Normal angle check
    if (normal.dot(referenceNormal) < cosMaxAngle) {
      continue;
    }

    // Skin weight check - sum skin weights across the joint and related joints
    float skinWeight = 0;
    for (int kTriVert = 0; kTriVert < 3; ++kTriVert) {
      for (int kSkinWeight = 0; kSkinWeight < skin.index.cols(); ++kSkinWeight) {
        const uint32_t skinJointIdx = skin.index(face(kTriVert), kSkinWeight);
        if (isRelatedJoint(skeleton, skinJointIdx, parentJointIdx)) {
          skinWeight += skin.weight(face(kTriVert), kSkinWeight);
        }
      }
    }
    skinWeight /= 3.0f;

    if (skinWeight < cutoffWeight) {
      continue;
    }

    // All checks passed - add to candidates
    CandidateTriangle candidate;
    candidate.triangleIdx = triIdx;
    candidate.vertexIndices = face;
    candidates.push_back(candidate);

    // Continue DFS to adjacent triangles
    if (triIdx < adjacency.size()) {
      for (size_t neighborIdx : adjacency[triIdx]) {
        if (visited.insert(neighborIdx).second) {
          toVisit.push(neighborIdx);
        }
      }
    }
  }

  return candidates;
}

std::vector<momentum::SkinnedLocatorTriangleConstraintT<float>> createSkinnedLocatorMeshConstraints(
    const momentum::Character& character,
    float targetDepth,
    float maxSearchDistanceCm,
    float maxNormalAngleDeg) {
  if (!character.mesh || !character.skinWeights) {
    return {};
  }

  // skip the triangle tree if we don't need it:
  if (character.skinnedLocators.empty()) {
    return {};
  }

  const Mesh& mesh = *character.mesh;

  // Build triangle adjacency if we need to find candidate triangles
  std::vector<std::vector<size_t>> adjacency;
  if (maxSearchDistanceCm > 0.0f) {
    adjacency = buildTriangleAdjacency(mesh);
  }

  std::vector<momentum::SkinnedLocatorTriangleConstraintT<float>> result;
  result.reserve(character.skinnedLocators.size());
  for (size_t i = 0; i < character.skinnedLocators.size(); ++i) {
    const auto& locator = character.skinnedLocators[i];
    const Eigen::Vector3f p_world = locator.position;

    // Find the closest point on the mesh to the locator.
    const auto closestPointResult = closestPointOnMeshMatchingParent(
        *character.mesh,
        *character.skinWeights,
        character.skeleton,
        p_world,
        locator.parents[0],
        targetDepth);
    if (!closestPointResult.valid) {
      continue;
    }

    momentum::SkinnedLocatorTriangleConstraintT<float> constr;
    constr.locatorIndex = i;
    constr.tgtTriangleIndices = mesh.faces[closestPointResult.triangleIdx];
    constr.tgtTriangleBaryCoords = closestPointResult.baryCoords;
    constr.depth = targetDepth;

    // Populate candidate triangles if sliding is enabled
    if (maxSearchDistanceCm > 0.0f) {
      // Compute reference normal from the initial triangle
      const Eigen::Vector3f& v0 = mesh.vertices[constr.tgtTriangleIndices[0]];
      const Eigen::Vector3f& v1 = mesh.vertices[constr.tgtTriangleIndices[1]];
      const Eigen::Vector3f& v2 = mesh.vertices[constr.tgtTriangleIndices[2]];
      Eigen::Vector3f referenceNormal = (v1 - v0).cross(v2 - v0);
      const float normalNorm = referenceNormal.norm();
      if (normalNorm > 1e-8f) {
        referenceNormal /= normalNorm;
      } else {
        referenceNormal = Eigen::Vector3f::UnitZ();
      }

      constr.candidateTriangles = findCandidateTrianglesDfs(
          mesh,
          *character.skinWeights,
          character.skeleton,
          adjacency,
          closestPointResult.triangleIdx,
          p_world,
          referenceNormal,
          locator.parents[0],
          maxSearchDistanceCm,
          maxNormalAngleDeg);
    }

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
    MT_THROW_IF(
        jointId >= state.jointState.size(),
        "Joint ID {} is out of bounds for jointState (size: {})",
        jointId,
        state.jointState.size());

    // get global locator position
    const Vector3f pos = state.jointState[jointId].transform * locators[i].offset;

    // change attachment to original joint
    result[i].parent = skeleton.joints[jointId].parent;

    // Check bounds for parent joint access
    MT_THROW_IF(
        result[i].parent >= state.jointState.size(),
        "Parent joint ID {} is out of bounds for jointState (size: {})",
        result[i].parent,
        state.jointState.size());

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
  MT_CHECK(param.size() == sourceCharacter.parameterTransform.numAllModelParameters());
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
