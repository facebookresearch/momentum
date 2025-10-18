/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character/character_utility.h"
#include "momentum/character/blend_shape.h"

#include "momentum/character/joint.h"
#include "momentum/character/pose_shape.h"
#include "momentum/character/skin_weights.h"
#include "momentum/common/exception.h"
#include "momentum/common/log.h"
#include "momentum/math/mesh.h"

#include <unordered_set>

namespace momentum {

namespace {

Skeleton scale(const Skeleton& skel, float scale) {
  Skeleton result = skel;
  for (auto& j : result.joints) {
    j.translationOffset *= scale;
  }
  return result;
}

std::unique_ptr<Mesh> scale(const std::unique_ptr<Mesh>& mesh, float scale) {
  if (!mesh) {
    return {};
  }

  auto result = std::make_unique<Mesh>(*mesh);
  for (auto& v : result->vertices) {
    v *= scale;
  }
  return result;
}

LocatorList scale(const LocatorList& locators, float scale) {
  LocatorList result = locators;
  for (auto& l : result) {
    l.offset *= scale;
    l.limitOrigin *= scale;
  }
  return result;
}

ParameterLimits scale(const ParameterLimits& limits, float scale) {
  ParameterLimits result = limits;
  for (auto& limit : result) {
    if (limit.type == LimitType::Ellipsoid) {
      limit.data.ellipsoid.ellipsoid.translation() *= scale;
      limit.data.ellipsoid.ellipsoidInv.translation() *= scale;
      limit.data.ellipsoid.offset *= scale;
    }
  }

  return result;
}

std::unique_ptr<CollisionGeometry> scale(
    const std::unique_ptr<CollisionGeometry>& geom,
    float scale) {
  if (!geom) {
    return {};
  }

  auto result = std::make_unique<CollisionGeometry>(*geom);
  for (auto& capsule : (*result)) {
    capsule.transformation.translation *= scale;
    capsule.radius *= scale;
    capsule.length *= scale;
  }

  return result;
}

TransformationList scaleInverseBindPose(
    const TransformationList& transformations,
    const float scale) {
  TransformationList result = transformations;
  MT_CHECK(scale > 0.0f);
  for (auto& t : result) {
    t.translation() *= scale;
  }
  return result;
}

template <typename T>
std::vector<T> mapParents(const std::vector<T>& objects, const std::vector<size_t>& jointMapping) {
  std::vector<T> result;
  result.reserve(objects.size());
  for (auto o : objects) {
    MT_CHECK(
        o.parent < jointMapping.size(),
        "Parent {} exceeds joint mapping size {}",
        o.parent,
        jointMapping.size());
    o.parent = jointMapping[o.parent];
    if (o.parent != kInvalidIndex) {
      result.push_back(o);
    }
  }
  return result;
}

ParameterLimits mapParameterLimits(
    const ParameterLimits& limits,
    const std::vector<size_t>& jointMapping,
    const std::vector<size_t>& parameterMapping) {
  ParameterLimits result;
  result.reserve(limits.size());
  for (auto l : limits) {
    switch (l.type) {
      case LimitType::MinMax:
        l.data.minMax.parameterIndex = parameterMapping[l.data.minMax.parameterIndex];
        if (l.data.minMax.parameterIndex != kInvalidIndex) {
          result.push_back(l);
        }
        break;
      case LimitType::MinMaxJoint:
      case LimitType::MinMaxJointPassive:
        l.data.minMaxJoint.jointIndex = jointMapping[l.data.minMaxJoint.jointIndex];
        if (l.data.minMaxJoint.jointIndex != kInvalidIndex) {
          result.push_back(l);
        }
        break;
      case LimitType::Linear:
        l.data.linear.referenceIndex = parameterMapping[l.data.linear.referenceIndex];
        l.data.linear.targetIndex = parameterMapping[l.data.linear.targetIndex];
        if (l.data.linear.referenceIndex != kInvalidIndex &&
            l.data.linear.targetIndex != kInvalidIndex) {
          result.push_back(l);
        }
        break;
      case LimitType::LinearJoint:
        l.data.linearJoint.referenceJointIndex =
            parameterMapping[l.data.linearJoint.referenceJointIndex];
        l.data.linearJoint.targetJointIndex = parameterMapping[l.data.linearJoint.targetJointIndex];
        if (l.data.linearJoint.referenceJointIndex != kInvalidIndex &&
            l.data.linearJoint.targetJointIndex != kInvalidIndex) {
          result.push_back(l);
        }
        break;
      case LimitType::HalfPlane:
        l.data.halfPlane.param1 = parameterMapping[l.data.halfPlane.param1];
        l.data.halfPlane.param2 = parameterMapping[l.data.halfPlane.param2];
        if (l.data.halfPlane.param1 != kInvalidIndex && l.data.halfPlane.param2 != kInvalidIndex) {
          result.push_back(l);
        }
        break;
      case LimitType::Ellipsoid:
        l.data.ellipsoid.parent = jointMapping[l.data.ellipsoid.parent];
        l.data.ellipsoid.ellipsoidParent = jointMapping[l.data.ellipsoid.ellipsoidParent];
        if (l.data.ellipsoid.parent != kInvalidIndex &&
            l.data.ellipsoid.ellipsoidParent != kInvalidIndex) {
          result.push_back(l);
        }
        break;
      case LimitType::LimitTypeCount: {
        // nothing to do here.
        break;
      }
    }
  }
  return result;
}

SkinWeights mapSkinWeights(SkinWeights weights, const std::vector<size_t>& jointMapping) {
  SkinWeights result(weights);

  for (Eigen::Index iRow = 0; iRow < weights.index.rows(); ++iRow) {
    for (Eigen::Index jCol = 0; jCol < weights.index.cols(); ++jCol) {
      if (weights.weight(iRow, jCol) == 0) {
        continue;
      }

      const auto mapped = jointMapping[result.index(iRow, jCol)];
      MT_CHECK(mapped != kInvalidIndex);
      result.index(iRow, jCol) = (int)mapped;
    }
  }

  return result;
}

template <typename T>
std::vector<T> mergeVectors(const std::vector<T>& left, const std::vector<T>& right) {
  std::vector<T> result;
  result.reserve(left.size() + right.size());
  std::copy(std::begin(left), std::end(left), std::back_inserter(result));
  std::copy(std::begin(right), std::end(right), std::back_inserter(result));
  return result;
}

std::vector<Eigen::Triplet<float>> toTriplets(const SparseRowMatrixf& mat) {
  std::vector<Eigen::Triplet<float>> result;
  for (int k = 0; k < mat.outerSize(); ++k) {
    for (SparseRowMatrixf::InnerIterator it(mat, k); it; ++it) {
      result.emplace_back(it.row(), it.col(), it.value());
    }
  }
  return result;
}

std::vector<size_t> addMappedParameters(
    const ParameterTransform& paramTransformOrig,
    const std::vector<size_t>& jointMapping,
    ParameterTransform& paramTransformResult) {
  std::vector<bool> validParamsOrig(paramTransformOrig.numAllModelParameters());

  // Pass 1: figure out which parameters we're going to keep:
  for (Eigen::Index kJointParam = 0; kJointParam < paramTransformOrig.transform.outerSize();
       ++kJointParam) {
    const Eigen::Index jointIndex = kJointParam / kParametersPerJoint;
    const size_t mappedJointIndex = jointMapping[jointIndex];
    if (mappedJointIndex == kInvalidIndex) {
      continue;
    }

    // Invalid joint index, so any model parameters are invalid too:
    for (SparseRowMatrixf::InnerIterator it(paramTransformOrig.transform, kJointParam); it; ++it) {
      validParamsOrig[it.col()] = true;
    }
  }

  std::vector<size_t> origParamToNewParam(
      static_cast<size_t>(paramTransformOrig.numAllModelParameters()), kInvalidIndex);
  std::unordered_set<std::string> existingParams(
      paramTransformResult.name.begin(), paramTransformResult.name.end());
  for (Eigen::Index iParamOld = 0; iParamOld < paramTransformOrig.numAllModelParameters();
       ++iParamOld) {
    if (!validParamsOrig[iParamOld]) {
      continue;
    }

    MT_THROW_IF(
        existingParams.find(paramTransformOrig.name[iParamOld]) != existingParams.end(),
        "Duplicate parameter {} found while merging parameter transforms.",
        paramTransformOrig.name[iParamOld]);

    origParamToNewParam[iParamOld] = paramTransformResult.name.size();
    paramTransformResult.name.push_back(paramTransformOrig.name[iParamOld]);
  }

  std::vector<Eigen::Triplet<float>> tripletsNew = toTriplets(paramTransformResult.transform);
  for (Eigen::Index kJointParam = 0; kJointParam < paramTransformOrig.transform.outerSize();
       ++kJointParam) {
    const Eigen::Index jointIndex = kJointParam / kParametersPerJoint;
    const Eigen::Index offset = kJointParam % kParametersPerJoint;
    const size_t mappedJointIndex = jointMapping[jointIndex];
    if (mappedJointIndex == kInvalidIndex) {
      continue;
    }

    const auto mappedJointParam = mappedJointIndex * kParametersPerJoint + offset;

    for (SparseRowMatrixf::InnerIterator it(paramTransformOrig.transform, kJointParam); it; ++it) {
      const auto iOldParam = it.col();
      const auto iNewParam = origParamToNewParam[iOldParam];

      if (iNewParam != SIZE_MAX) {
        tripletsNew.emplace_back((int)mappedJointParam, (int)iNewParam, it.value());
      }
    }
  }
  paramTransformResult.transform.resize(
      paramTransformResult.transform.rows(), paramTransformResult.name.size());
  paramTransformResult.transform.setFromTriplets(tripletsNew.begin(), tripletsNew.end());

  // Update the parameter sets:
  for (const auto& paramSetOrig : paramTransformOrig.parameterSets) {
    for (Eigen::Index iOrigParam = 0; iOrigParam < paramTransformOrig.numAllModelParameters();
         ++iOrigParam) {
      if (!paramSetOrig.second.test(iOrigParam)) {
        continue;
      }

      const auto iNewParam = origParamToNewParam[iOrigParam];
      if (iNewParam != kInvalidIndex) {
        paramTransformResult.parameterSets[paramSetOrig.first].set(iNewParam);
      }
    }
  }

  return origParamToNewParam;
}

LocatorList removeDuplicateLocators(LocatorList locators, const LocatorList& toRemove) {
  std::unordered_set<std::string> toRemoveNames;
  for (const auto& l : toRemove) {
    toRemoveNames.insert(l.name);
  }

  locators.erase(
      std::remove_if(
          locators.begin(),
          locators.end(),
          [&toRemoveNames](const Locator& l) { return toRemoveNames.count(l.name) > 0; }),
      locators.end());

  return locators;
}

} // namespace

Character scaleCharacter(const Character& character, float s) {
  return {
      scale(character.skeleton, s),
      character.parameterTransform,
      scale(character.parameterLimits, s),
      scale(character.locators, s),
      scale(character.mesh, s).get(),
      character.skinWeights.get(),
      scale(character.collision, s).get(),
      character.poseShapes.get(),
      character.blendShape,
      character.faceExpressionBlendShape,
      character.name,
      scaleInverseBindPose(character.inverseBindPose, s)};
}

namespace {

Skeleton transformSkeleton(const Skeleton& skeleton, const Eigen::Affine3f& xform) {
  const Eigen::Vector3f singularValues = xform.linear().jacobiSvd().singularValues();
  for (Eigen::Index i = 0; i < 3; ++i) {
    MT_CHECK(
        singularValues(i) > 0.99 && singularValues(i) < 1.01,
        "Transform should not include scale or shear.");
  }

  const Eigen::Quaternionf rotation(xform.linear());
  const Eigen::Vector3f translation(xform.translation());

  Skeleton result(skeleton);
  MT_CHECK(!result.joints.empty());
  MT_CHECK(!skeleton.joints.empty());
  if (!result.joints.empty() && !skeleton.joints.empty()) {
    result.joints.front().preRotation = rotation * skeleton.joints.front().preRotation;
    result.joints.front().translationOffset =
        rotation * skeleton.joints.front().translationOffset + translation;
  }
  return result;
}

BlendShape_const_p transformBlendShape(
    const BlendShape_const_p& blendShape,
    const Eigen::Affine3f& xform) {
  if (!blendShape) {
    return nullptr;
  }

  const Eigen::Vector3f singularValues = xform.linear().jacobiSvd().singularValues();
  for (Eigen::Index i = 0; i < 3; ++i) {
    MT_CHECK(
        singularValues(i) > 0.99 && singularValues(i) < 1.01,
        "Transform should not include scale or shear.");
  }

  const Eigen::Quaternionf rotation(xform.linear());
  const Eigen::Vector3f translation(xform.translation());

  momentum::BlendShape_p blendShapeTransformed =
      std::make_unique<momentum::BlendShape>(*blendShape);

  // apply the full transformation to the blendshape basis
  std::vector<Vector3f> baseShape = blendShapeTransformed->getBaseShape();
  for (auto& v : baseShape) {
    v = xform * v;
  }
  blendShapeTransformed->setBaseShape(baseShape);

  // apply only the rotation to the shape vectors
  Eigen::MatrixXf shapeVectors = blendShapeTransformed->getShapeVectors();
  // Apply rotation to each 3D displacement within each shape vector column
  for (Eigen::Index iShape = 0; iShape < shapeVectors.cols(); ++iShape) {
    for (Eigen::Index iVertex = 0; iVertex < shapeVectors.rows() / 3; ++iVertex) {
      Eigen::Vector3f displacement = shapeVectors.block<3, 1>(iVertex * 3, iShape);
      shapeVectors.block<3, 1>(iVertex * 3, iShape) = rotation * displacement;
    }
  }
  blendShapeTransformed->setShapeVectors(shapeVectors);

  return blendShapeTransformed;
}

std::unique_ptr<Mesh> transformMesh(const std::unique_ptr<Mesh>& mesh, const Eigen::Affine3f& xf) {
  if (!mesh) {
    return {};
  }

  auto result = std::make_unique<Mesh>(*mesh);
  for (auto& v : result->vertices) {
    v = xf * v;
  }

  for (auto& v : result->normals) {
    v = xf.linear() * v;
  }

  return result;
}

TransformationList transformInverseBindPose(
    const TransformationList& inverseBindPose,
    const Eigen::Affine3f& xf) {
  TransformationList result;
  result.reserve(inverseBindPose.size());
  const Eigen::Affine3f xfInv = xf.inverse();
  for (const auto& m : inverseBindPose) {
    result.push_back(m * xfInv);
  }
  return result;
}

} // namespace

Character transformCharacter(const Character& character, const Affine3f& xform) {
  return {
      transformSkeleton(character.skeleton, xform),
      character.parameterTransform,
      character.parameterLimits,
      character.locators,
      transformMesh(character.mesh, xform).get(),
      character.skinWeights.get(),
      character.collision.get(),
      character.poseShapes.get(),
      transformBlendShape(character.blendShape, xform),
      character.faceExpressionBlendShape,
      character.name,
      transformInverseBindPose(character.inverseBindPose, xform)};
}

Character replaceSkeletonHierarchy(
    const Character& srcCharacter,
    const Character& tgtCharacter,
    const std::string& srcRootJoint,
    const std::string& tgtRootJoint) {
  const Skeleton& srcSkeleton = srcCharacter.skeleton;
  const Skeleton& tgtSkeleton = tgtCharacter.skeleton;

  MT_THROW_IF(srcSkeleton.joints.empty(), "Trying to reparent empty skeleton.");

  const auto tgtRoot = tgtSkeleton.getJointIdByName(tgtRootJoint);
  MT_THROW_IF(
      tgtRoot == kInvalidIndex,
      "Unable to re-root skeleton, target root joint {} not found.",
      tgtRootJoint);

  const auto srcRoot = srcSkeleton.getJointIdByName(srcRootJoint);
  MT_THROW_IF(
      srcRoot == kInvalidIndex,
      "Unable to re-root skeleton, source root joint {} not found.",
      srcRootJoint);

  Skeleton combinedSkeleton;

  // Make sure we don't insert duplicates:
  std::unordered_map<std::string, size_t> combinedSkeletonJointMapping;

  auto addJoint =
      [&](const Skeleton& skeleton, size_t jointIndex, std::vector<size_t>& jointMap) -> size_t {
    const Joint& origJoint = skeleton.joints[jointIndex];
    Joint combinedJoint = origJoint;
    MT_THROW_IF(
        combinedSkeletonJointMapping.find(combinedJoint.name) != combinedSkeletonJointMapping.end(),
        "Duplicate joint '{}' found while reparenting.",
        origJoint.name);
    const size_t combinedIndex = combinedSkeleton.joints.size();
    jointMap[jointIndex] = combinedIndex;
    combinedSkeletonJointMapping.insert({combinedJoint.name, combinedIndex});

    // Remap the parent:
    if (origJoint.parent != kInvalidIndex) {
      auto itr = combinedSkeletonJointMapping.find(skeleton.joints[origJoint.parent].name);
      MT_CHECK(itr != combinedSkeletonJointMapping.end());
      combinedJoint.parent = itr->second;
    }

    combinedSkeleton.joints.push_back(combinedJoint);
    return combinedIndex;
  };

  std::vector<size_t> srcToCombinedJoints(srcCharacter.skeleton.joints.size(), kInvalidIndex);
  std::vector<size_t> tgtToCombinedJoints(tgtCharacter.skeleton.joints.size(), kInvalidIndex);

  for (size_t iTgtJoint = 0; iTgtJoint < tgtSkeleton.joints.size(); ++iTgtJoint) {
    // Add the replacement joints just after the new parent rather than at the very end; this will
    // improve coherency a bit.
    if (iTgtJoint == tgtRoot) {
      addJoint(tgtSkeleton, iTgtJoint, tgtToCombinedJoints);

      // Copy the joints from the source skeleton over in place of the target root:
      for (size_t kSrcJoint = srcRoot + 1; kSrcJoint < srcSkeleton.joints.size(); ++kSrcJoint) {
        if (srcSkeleton.isAncestor(kSrcJoint, srcRoot)) {
          addJoint(srcSkeleton, kSrcJoint, srcToCombinedJoints);
        }
      }
    } else if (!tgtSkeleton.isAncestor(iTgtJoint, tgtRoot)) {
      addJoint(tgtSkeleton, iTgtJoint, tgtToCombinedJoints);
    }
  }

  // For locators, use the parent joint to decide whether to use the src locator or
  // the target locator.
  // For duplicate locators that are on both the src and tgt skeletons (but parented to different
  // bones), we will always take the src locator.  This is because we trust the locators on the
  // primary skeleton's hands more than the secondary skeleton.
  const LocatorList combinedLocators = [&]() {
    const LocatorList remappedSrcLocators =
        mapParents<Locator>(srcCharacter.locators, srcToCombinedJoints);
    const LocatorList remappedTgtLocators =
        mapParents<Locator>(tgtCharacter.locators, tgtToCombinedJoints);
    return mergeVectors(
        removeDuplicateLocators(remappedTgtLocators, remappedSrcLocators), remappedSrcLocators);
  }();
  {
    std::unordered_set<std::string> locatorNames;
    for (const auto& l : combinedLocators) {
      MT_LOGW_IF(locatorNames.count(l.name) != 0, "Duplicate locator: {}", l.name);
      locatorNames.insert(l.name);
    }
  }

  CollisionGeometry combinedCollisionGeometry;
  if (tgtCharacter.collision) {
    combinedCollisionGeometry = mergeVectors(
        combinedCollisionGeometry,
        mapParents<TaperedCapsule>(*tgtCharacter.collision, tgtToCombinedJoints));
  }
  if (srcCharacter.collision) {
    combinedCollisionGeometry = mergeVectors(
        combinedCollisionGeometry,
        mapParents<TaperedCapsule>(*srcCharacter.collision, srcToCombinedJoints));
  }

  // Build the merged parameter transform:
  ParameterTransform combinedParamTransform;
  combinedParamTransform.offsets =
      VectorXf::Zero(combinedSkeleton.joints.size() * kParametersPerJoint);
  combinedParamTransform.transform.resize(combinedSkeleton.joints.size() * kParametersPerJoint, 0);
  const std::vector<size_t> tgtToCombinedParameters = addMappedParameters(
      tgtCharacter.parameterTransform, tgtToCombinedJoints, combinedParamTransform);
  const std::vector<size_t> srcToCombinedParameters = addMappedParameters(
      srcCharacter.parameterTransform, srcToCombinedJoints, combinedParamTransform);
  combinedParamTransform.activeJointParams = combinedParamTransform.computeActiveJointParams();

  const ParameterLimits combinedParameterLimits = mergeVectors(
      mapParameterLimits(
          tgtCharacter.parameterLimits, tgtToCombinedJoints, tgtToCombinedParameters),
      mapParameterLimits(
          srcCharacter.parameterLimits, srcToCombinedJoints, srcToCombinedParameters));

  std::unique_ptr<SkinWeights> skinWeightsCombined;
  {
    // Mapping for every joint in the target skeleton to a joint in the combined skeleton.
    // The goal here is something that we can use to map e.g. the skinning weights.
    // Rules are:
    //  1. if the target joint is in the combined skeleton, then map it directly.
    //  2. otherwise, find a name with a matching joint in the source skeleton.
    //  3. if not, then go up one level of the hierarchy and try again.
    std::vector<size_t> tgtToCombinedWithParents(
        tgtCharacter.skeleton.joints.size(), kInvalidIndex);
    for (size_t iTgtJoint = 0; iTgtJoint < tgtSkeleton.joints.size(); ++iTgtJoint) {
      size_t tgtParent = iTgtJoint;
      while (tgtParent != kInvalidIndex) {
        const auto itr = combinedSkeletonJointMapping.find(tgtSkeleton.joints[tgtParent].name);
        if (itr != combinedSkeletonJointMapping.end()) {
          tgtToCombinedWithParents[iTgtJoint] = itr->second;
          break;
        }
        tgtParent = tgtSkeleton.joints[iTgtJoint].parent;
      }
      MT_CHECK(tgtParent != kInvalidIndex);
    }

    if (tgtCharacter.skinWeights) {
      skinWeightsCombined = std::make_unique<SkinWeights>(
          mapSkinWeights(*tgtCharacter.skinWeights, tgtToCombinedWithParents));
    }
  }

  return {
      combinedSkeleton,
      combinedParamTransform,
      combinedParameterLimits,
      combinedLocators,
      tgtCharacter.mesh.get(),
      skinWeightsCombined.get(),
      combinedCollisionGeometry.empty() ? nullptr : &combinedCollisionGeometry,
      tgtCharacter.poseShapes.get(),
      tgtCharacter.blendShape};
}

Character removeJoints(const Character& character, gsl::span<const size_t> jointsToRemove) {
  std::vector<bool> toRemove(character.skeleton.joints.size(), false);
  for (const auto& j : jointsToRemove) {
    MT_THROW_IF(j >= character.skeleton.joints.size(), "Invalid joint found in removeJoints.");
    toRemove[j] = true;
  }

  // Remove all joints parented under the target joints as well:
  const auto& srcSkeleton = character.skeleton;
  std::vector<size_t> srcToResultJoints(srcSkeleton.joints.size(), kInvalidIndex);
  Skeleton resultSkeleton;
  for (size_t iJoint = 0; iJoint < srcSkeleton.joints.size(); ++iJoint) {
    bool shouldRemove = false;

    size_t parent = iJoint;
    while (parent != kInvalidIndex) {
      if (toRemove[parent]) {
        shouldRemove = true;
        break;
      }
      if (parent < character.skeleton.joints.size()) {
        parent = character.skeleton.joints[parent].parent;
      } else {
        break;
      }
    }

    if (shouldRemove) {
      continue;
    }

    srcToResultJoints[iJoint] = resultSkeleton.joints.size();
    auto joint = srcSkeleton.joints[iJoint];
    if (joint.parent != kInvalidIndex) {
      joint.parent = srcToResultJoints[joint.parent];
    }
    resultSkeleton.joints.push_back(joint);
  }

  ParameterTransform resultParamTransform =
      ParameterTransform::empty(resultSkeleton.joints.size() * kParametersPerJoint);
  const std::vector<size_t> srcToResultParameters =
      addMappedParameters(character.parameterTransform, srcToResultJoints, resultParamTransform);

  std::unique_ptr<SkinWeights> resultSkinWeights;
  {
    // Mapping for every joint in the target skeleton to a joint in the combined skeleton.
    std::vector<size_t> srcToResultJointsWithParents(
        character.skeleton.joints.size(), kInvalidIndex);
    for (size_t iSrcJoint = 0; iSrcJoint < srcSkeleton.joints.size(); ++iSrcJoint) {
      size_t srcParent = iSrcJoint;
      // Anything skinned to a deleted joint should get skinned to its parent instead:
      while (srcParent != kInvalidIndex) {
        srcToResultJointsWithParents[iSrcJoint] = srcToResultJoints[srcParent];
        if (srcToResultJointsWithParents[iSrcJoint] != kInvalidIndex) {
          break;
        }
        srcParent = srcSkeleton.joints[srcParent].parent;
      }
    }

    if (character.skinWeights) {
      resultSkinWeights = std::make_unique<SkinWeights>(
          mapSkinWeights(*character.skinWeights, srcToResultJointsWithParents));
    }
  }

  // For locators, use the parent joint to decide whether to use the src locator or
  // the target locator.
  const LocatorList resultLocators = mapParents<Locator>(character.locators, srcToResultJoints);

  CollisionGeometry resultCollisionGeometry;
  if (character.collision) {
    resultCollisionGeometry = mapParents<TaperedCapsule>(*character.collision, srcToResultJoints);
  }

  const ParameterLimits resultParameterLimits =
      mapParameterLimits(character.parameterLimits, srcToResultJoints, srcToResultParameters);

  return {
      resultSkeleton,
      resultParamTransform,
      resultParameterLimits,
      resultLocators,
      character.mesh.get(),
      resultSkinWeights.get(),
      resultCollisionGeometry.empty() ? nullptr : &resultCollisionGeometry,
      character.poseShapes.get(),
      character.blendShape};
}

MatrixXf mapMotionToCharacter(
    const MotionParameters& inputMotion,
    const Character& targetCharacter) {
  // re-arrange poses according to the character names
  const auto& parameterNames = std::get<0>(inputMotion);
  const auto& motion = std::get<1>(inputMotion);

  MT_CHECK(
      static_cast<int>(parameterNames.size()) == motion.rows(),
      "The number of parameter names {} does not match the number of rows {} in the motion.",
      parameterNames.size(),
      motion.rows());

  MatrixXf result =
      MatrixXf::Zero(targetCharacter.parameterTransform.numAllModelParameters(), motion.cols());
  for (size_t i = 0; i < parameterNames.size(); i++) {
    const auto index = targetCharacter.parameterTransform.getParameterIdByName(parameterNames[i]);
    if (index != kInvalidIndex) {
      result.row(index) = motion.row(i);
    } else {
      MT_LOGW("Model parameter {} not found in source motion", parameterNames[i]);
    }
  }

  return result;
}

VectorXf mapIdentityToCharacter(
    const IdentityParameters& inputIdentity,
    const Character& targetCharacter) {
  // re-arrange offsets according to the character names
  const auto& jointNames = std::get<0>(inputIdentity);
  const auto& identity = std::get<1>(inputIdentity);

  MT_CHECK(
      static_cast<int>(jointNames.size() * kParametersPerJoint) == identity.size(),
      "The number of joint parameters {} does not match the identity vector size {}.",
      jointNames.size() * kParametersPerJoint,
      identity.size());

  VectorXf result = VectorXf::Zero(targetCharacter.skeleton.joints.size() * kParametersPerJoint);
  for (size_t i = 0; i < jointNames.size(); i++) {
    const auto index = targetCharacter.skeleton.getJointIdByName(jointNames[i]);
    if (index != kInvalidIndex) {
      result.template middleRows<kParametersPerJoint>(index * kParametersPerJoint).noalias() =
          identity.template middleRows<kParametersPerJoint>(i * kParametersPerJoint);
    } else {
      MT_LOGW("Joint {} not found in source identity", jointNames[i]);
    }
  }

  return result;
}

namespace {

/// Reduces base shape vertices based on active vertex selection
///
/// @param baseVertices Original base shape vertices
/// @param activeVertices Boolean vector indicating which vertices to keep
/// @param newVertexCount Number of vertices in the reduced mesh
/// @return Reduced base shape vertices
std::vector<Vector3f> reduceBaseShapeVertices(
    const std::vector<Vector3f>& baseVertices,
    const std::vector<bool>& activeVertices,
    size_t newVertexCount) {
  std::vector<Vector3f> newBaseVertices;
  newBaseVertices.reserve(newVertexCount);

  for (size_t i = 0; i < activeVertices.size(); ++i) {
    if (activeVertices[i]) {
      newBaseVertices.push_back(baseVertices[i]);
    }
  }

  return newBaseVertices;
}

/// Reduces blend shape vectors based on active vertex selection
///
/// @param shapeVectors Original shape vectors matrix [vertexCount*3 x numShapes]
/// @param activeVertices Boolean vector indicating which vertices to keep
/// @param newVertexCount Number of vertices in the reduced mesh
/// @return Reduced shape vectors matrix [newVertexCount*3 x numShapes]
MatrixXf reduceBlendShapeVectors(
    const MatrixXf& shapeVectors,
    const std::vector<bool>& activeVertices,
    size_t newVertexCount) {
  MatrixXf newShapeVectors(newVertexCount * 3, shapeVectors.cols());

  size_t newIdx = 0;
  for (size_t i = 0; i < activeVertices.size(); ++i) {
    if (activeVertices[i]) {
      newShapeVectors.block(newIdx * 3, 0, 3, shapeVectors.cols()) =
          shapeVectors.block(i * 3, 0, 3, shapeVectors.cols());
      newIdx++;
    }
  }

  return newShapeVectors;
}

std::pair<std::vector<size_t>, std::vector<size_t>> createIndexMapping(
    const std::vector<bool>& activeElements) {
  std::vector<size_t> forwardMapping;
  std::vector<size_t> reverseMapping(activeElements.size(), kInvalidIndex);
  for (size_t i = 0; i < activeElements.size(); ++i) {
    if (activeElements[i]) {
      reverseMapping[i] = forwardMapping.size();
      forwardMapping.push_back(i);
    }
  }

  return {forwardMapping, reverseMapping};
}

template <typename T>
std::vector<T> selectVertices(
    const std::vector<T>& sourceVector,
    const std::vector<size_t>& selected) {
  std::vector<T> result;
  result.reserve(selected.size());

  for (unsigned long i : selected) {
    MT_CHECK(i < sourceVector.size());
    result.push_back(sourceVector[i]);
  }

  return result;
}

std::vector<Eigen::Vector3i> remapFaces(
    const std::vector<Eigen::Vector3i>& faces,
    const std::vector<size_t>& mapping,
    const std::vector<bool>& activeFaces) {
  const size_t nNewFaceCount = std::count(activeFaces.begin(), activeFaces.end(), true);

  std::vector<Eigen::Vector3i> remappedFaces;
  remappedFaces.reserve(nNewFaceCount);

  for (size_t iFace = 0; iFace < faces.size(); ++iFace) {
    if (!activeFaces[iFace]) {
      continue;
    }

    const auto& face = faces[iFace];
    MT_CHECK(
        mapping[face[0]] != kInvalidIndex && mapping[face[1]] != kInvalidIndex &&
        mapping[face[2]] != kInvalidIndex);

    remappedFaces.emplace_back(mapping[face[0]], mapping[face[1]], mapping[face[2]]);
  }

  return remappedFaces;
}

std::vector<std::vector<int32_t>> remapLines(
    const std::vector<std::vector<int32_t>>& lines,
    const std::vector<size_t>& mapping) {
  std::vector<std::vector<int32_t>> remappedLines;
  remappedLines.reserve(lines.size());

  for (const auto& line : lines) {
    std::vector<int32_t> newLine;
    newLine.reserve(line.size());

    bool lineValid = true;
    for (int32_t vertexIdx : line) {
      if (vertexIdx >= 0 && vertexIdx < static_cast<int32_t>(mapping.size()) &&
          mapping[vertexIdx] != kInvalidIndex) {
        newLine.push_back(static_cast<int32_t>(mapping[vertexIdx]));
      } else {
        lineValid = false;
        break;
      }
    }

    if (lineValid && !newLine.empty()) {
      remappedLines.push_back(std::move(newLine));
    }
  }

  return remappedLines;
}

std::vector<bool> verticesToFaces(
    const std::vector<Eigen::Vector3i>& faces,
    const std::vector<bool>& activeVertices) {
  std::vector<bool> activeFaces(faces.size(), false);

  for (size_t faceIdx = 0; faceIdx < faces.size(); ++faceIdx) {
    const auto& face = faces[faceIdx];
    // Face is active only if all its vertices are active and within bounds
    if (face[0] >= 0 && face[0] < static_cast<int>(activeVertices.size()) && face[1] >= 0 &&
        face[1] < static_cast<int>(activeVertices.size()) && face[2] >= 0 &&
        face[2] < static_cast<int>(activeVertices.size()) && activeVertices[face[0]] &&
        activeVertices[face[1]] && activeVertices[face[2]]) {
      activeFaces[faceIdx] = true;
    }
  }

  return activeFaces;
}

std::vector<bool> facesToVertices(
    const std::vector<Eigen::Vector3i>& faces,
    const std::vector<bool>& activeFaces,
    size_t vertexCount) {
  MT_CHECK(
      activeFaces.size() == faces.size(),
      "Active faces size ({}) does not match face count ({})",
      activeFaces.size(),
      faces.size());

  std::vector<bool> activeVertices(vertexCount, false);

  for (size_t faceIdx = 0; faceIdx < activeFaces.size(); ++faceIdx) {
    if (activeFaces[faceIdx]) {
      const auto& face = faces[faceIdx];
      if (face[0] >= 0 && face[0] < static_cast<int>(vertexCount)) {
        activeVertices[face[0]] = true;
      }
      if (face[1] >= 0 && face[1] < static_cast<int>(vertexCount)) {
        activeVertices[face[1]] = true;
      }
      if (face[2] >= 0 && face[2] < static_cast<int>(vertexCount)) {
        activeVertices[face[2]] = true;
      }
    }
  }

  return activeVertices;
}

/// Reduces mesh components based on active vertices and faces
///
/// @param character Character to be reduced
/// @param activeVertices Boolean vector indicating which vertices to keep
/// @param activeFaces Boolean vector indicating which faces to keep
/// @return A new character with all components reduced accordingly
template <typename T>
CharacterT<T> reduceMeshComponents(
    const CharacterT<T>& character,
    const std::vector<bool>& activeVertices,
    const std::vector<bool>& activeFaces) {
  MT_CHECK(character.mesh, "Cannot reduce mesh: character has no mesh");
  MT_CHECK(
      activeVertices.size() == character.mesh->vertices.size(),
      "Active vertices size ({}) does not match mesh vertex count ({})",
      activeVertices.size(),
      character.mesh->vertices.size());
  MT_CHECK(
      activeFaces.size() == character.mesh->faces.size(),
      "Active faces size ({}) does not match mesh face count ({})",
      activeFaces.size(),
      character.mesh->faces.size());

  // Create a mapping from old vertex indices to new vertex indices using generic function
  const auto [forwardVertexMapping, reverseVertexMapping] = createIndexMapping(activeVertices);

  // Create new mesh with reduced vertices and faces
  auto newMesh = std::make_unique<Mesh>();

  // Copy active vertices using generic mapping function
  newMesh->vertices = selectVertices(character.mesh->vertices, forwardVertexMapping);

  if (!character.mesh->normals.empty()) {
    newMesh->normals = selectVertices(character.mesh->normals, forwardVertexMapping);
  }

  if (!character.mesh->colors.empty()) {
    newMesh->colors = selectVertices(character.mesh->colors, forwardVertexMapping);
  }

  if (!character.mesh->confidence.empty()) {
    newMesh->confidence = selectVertices(character.mesh->confidence, forwardVertexMapping);
  }

  newMesh->faces = remapFaces(character.mesh->faces, reverseVertexMapping, activeFaces);

  if (!character.mesh->lines.empty()) {
    newMesh->lines = remapLines(character.mesh->lines, reverseVertexMapping);
  }

  // Handle texture coordinates properly using generic functions
  if (!character.mesh->texcoord_faces.empty() && !character.mesh->texcoords.empty()) {
    // Create active texture triangles based on active faces
    std::vector<bool> activeTextureTriangles = activeFaces;
    activeTextureTriangles.resize(character.mesh->texcoord_faces.size(), false);

    const std::vector<bool> activeTextureVertices = facesToVertices(
        character.mesh->texcoord_faces, activeTextureTriangles, character.mesh->texcoords.size());

    auto [forwardTextureVertexMapping, reverseTextureVertexMapping] =
        createIndexMapping(activeTextureVertices);

    // Copy active texture vertices using generic mapping function
    newMesh->texcoords = selectVertices(character.mesh->texcoords, forwardTextureVertexMapping);

    newMesh->texcoord_faces = remapFaces(
        character.mesh->texcoord_faces, reverseTextureVertexMapping, activeTextureTriangles);

    newMesh->texcoord_lines =
        remapLines(character.mesh->texcoord_lines, reverseTextureVertexMapping);
  }

  // Create new skin weights if they exist
  std::unique_ptr<SkinWeights> newSkinWeights;
  if (character.skinWeights) {
    newSkinWeights = std::make_unique<SkinWeights>();
    newSkinWeights->index.resize(forwardVertexMapping.size(), kMaxSkinJoints);
    newSkinWeights->weight.resize(forwardVertexMapping.size(), kMaxSkinJoints);

    for (size_t newRowIdx = 0; newRowIdx < forwardVertexMapping.size(); ++newRowIdx) {
      const auto oldRowIdx = forwardVertexMapping[newRowIdx];
      newSkinWeights->index.row(newRowIdx) = character.skinWeights->index.row(oldRowIdx);
      newSkinWeights->weight.row(newRowIdx) = character.skinWeights->weight.row(oldRowIdx);
    }
  }

  // Create new pose shapes if they exist
  std::unique_ptr<PoseShape> newPoseShapes;
  if (character.poseShapes) {
    newPoseShapes = std::make_unique<PoseShape>(*character.poseShapes);

    // Reduce the base shape (stored as flat vector [x1,y1,z1,x2,y2,z2,...])
    VectorXf newBaseShape(forwardVertexMapping.size() * 3);
    size_t newIdx = 0;
    for (size_t i = 0; i < activeVertices.size(); ++i) {
      if (activeVertices[i]) {
        newBaseShape.template segment<3>(newIdx * 3) =
            character.poseShapes->baseShape.template segment<3>(i * 3);
        newIdx++;
      }
    }
    newPoseShapes->baseShape = std::move(newBaseShape);

    // Reduce the shape vectors matrix
    MatrixXf newShapeVectors(
        forwardVertexMapping.size() * 3, character.poseShapes->shapeVectors.cols());
    newIdx = 0;
    for (size_t i = 0; i < activeVertices.size(); ++i) {
      if (activeVertices[i]) {
        newShapeVectors.block(newIdx * 3, 0, 3, character.poseShapes->shapeVectors.cols()) =
            character.poseShapes->shapeVectors.block(
                i * 3, 0, 3, character.poseShapes->shapeVectors.cols());
        newIdx++;
      }
    }
    newPoseShapes->shapeVectors = std::move(newShapeVectors);
  }

  // Create new blend shapes if they exist
  BlendShape_const_p newBlendShape;
  if (character.blendShape) {
    auto blendShape = std::make_shared<BlendShape>();

    // Reduce base shape vertices
    std::vector<Vector3f> newBaseVertices = reduceBaseShapeVertices(
        character.blendShape->getBaseShape(), activeVertices, forwardVertexMapping.size());
    blendShape->setBaseShape(newBaseVertices);

    // Reduce shape vectors using shared function
    MatrixXf newShapeVectors = reduceBlendShapeVectors(
        character.blendShape->getShapeVectors(), activeVertices, forwardVertexMapping.size());
    blendShape->setShapeVectors(newShapeVectors);

    newBlendShape = blendShape;
  }

  // Create new face expression blend shapes if they exist
  BlendShapeBase_const_p newFaceExpressionBlendShape;
  if (character.faceExpressionBlendShape) {
    auto faceBlendShape = std::make_shared<BlendShapeBase>(
        forwardVertexMapping.size(), character.faceExpressionBlendShape->shapeSize());

    // Reduce shape vectors using shared function
    MatrixXf newShapeVectors = reduceBlendShapeVectors(
        character.faceExpressionBlendShape->getShapeVectors(),
        activeVertices,
        forwardVertexMapping.size());
    faceBlendShape->setShapeVectors(newShapeVectors);

    newFaceExpressionBlendShape = faceBlendShape;
  }

  // Construct and return the new character with all updated components
  return CharacterT<T>(
      character.skeleton,
      character.parameterTransform,
      character.parameterLimits,
      character.locators,
      newMesh.get(),
      newSkinWeights.get(),
      character.collision.get(),
      newPoseShapes.get(),
      std::move(newBlendShape),
      std::move(newFaceExpressionBlendShape),
      character.name,
      character.inverseBindPose);
}

} // namespace

template <typename T>
CharacterT<T> reduceMeshByVertices(
    const CharacterT<T>& character,
    const std::vector<bool>& activeVertices) {
  MT_CHECK(character.mesh, "Cannot reduce mesh: character has no mesh");
  MT_CHECK(
      activeVertices.size() == character.mesh->vertices.size(),
      "Active vertices size ({}) does not match mesh vertex count ({})",
      activeVertices.size(),
      character.mesh->vertices.size());

  // Convert vertex selection to face selection
  const auto activeFaces = verticesToFaces(*character.mesh, activeVertices);

  return reduceMeshComponents(character, activeVertices, activeFaces);
}

template <typename T>
CharacterT<T> reduceMeshByFaces(
    const CharacterT<T>& character,
    const std::vector<bool>& activeFaces) {
  MT_CHECK(character.mesh, "Cannot reduce mesh: character has no mesh");
  MT_CHECK(
      activeFaces.size() == character.mesh->faces.size(),
      "Active faces size ({}) does not match mesh face count ({})",
      activeFaces.size(),
      character.mesh->faces.size());

  // Convert face selection to vertex selection
  const auto activeVertices = facesToVertices(*character.mesh, activeFaces);

  return reduceMeshComponents(character, activeVertices, activeFaces);
}

template <typename T>
std::vector<bool> verticesToFaces(const MeshT<T>& mesh, const std::vector<bool>& activeVertices) {
  MT_CHECK(
      activeVertices.size() == mesh.vertices.size(),
      "Active vertices size ({}) does not match mesh vertex count ({})",
      activeVertices.size(),
      mesh.vertices.size());

  return verticesToFaces(mesh.faces, activeVertices);
}

template <typename T>
std::vector<bool> facesToVertices(const MeshT<T>& mesh, const std::vector<bool>& activeFaces) {
  MT_CHECK(
      activeFaces.size() == mesh.faces.size(),
      "Active faces size ({}) does not match mesh face count ({})",
      activeFaces.size(),
      mesh.faces.size());

  return facesToVertices(mesh.faces, activeFaces, mesh.vertices.size());
}

// Explicit instantiations for commonly used types
template CharacterT<float> reduceMeshByVertices<float>(
    const CharacterT<float>& character,
    const std::vector<bool>& activeVertices);

template CharacterT<double> reduceMeshByVertices<double>(
    const CharacterT<double>& character,
    const std::vector<bool>& activeVertices);

template CharacterT<float> reduceMeshByFaces<float>(
    const CharacterT<float>& character,
    const std::vector<bool>& activeFaces);

template CharacterT<double> reduceMeshByFaces<double>(
    const CharacterT<double>& character,
    const std::vector<bool>& activeFaces);

template std::vector<bool> verticesToFaces<float>(
    const MeshT<float>& mesh,
    const std::vector<bool>& activeVertices);

template std::vector<bool> verticesToFaces<double>(
    const MeshT<double>& mesh,
    const std::vector<bool>& activeVertices);

template std::vector<bool> facesToVertices<float>(
    const MeshT<float>& mesh,
    const std::vector<bool>& activeFaces);

template std::vector<bool> facesToVertices<double>(
    const MeshT<double>& mesh,
    const std::vector<bool>& activeFaces);

} // namespace momentum
