/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/character_utility.h>

#include <momentum/character/joint.h>
#include <momentum/character/skin_weights.h>
#include <momentum/common/exception.h>
#include <momentum/common/log.h>
#include <momentum/math/mesh.h>

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
    capsule.transformation.translation() *= scale;
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
  return Character(
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
      scaleInverseBindPose(character.inverseBindPose, s));
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
  result.joints.front().preRotation = rotation * skeleton.joints.front().preRotation;
  result.joints.front().translationOffset =
      rotation * skeleton.joints.front().translationOffset + translation;
  return result;
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
  return Character(
      transformSkeleton(character.skeleton, xform),
      character.parameterTransform,
      character.parameterLimits,
      character.locators,
      transformMesh(character.mesh, xform).get(),
      character.skinWeights.get(),
      character.collision.get(),
      character.poseShapes.get(),
      character.blendShape,
      character.faceExpressionBlendShape,
      character.name,
      transformInverseBindPose(character.inverseBindPose, xform));
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

  return Character(
      combinedSkeleton,
      combinedParamTransform,
      combinedParameterLimits,
      combinedLocators,
      tgtCharacter.mesh.get(),
      skinWeightsCombined.get(),
      combinedCollisionGeometry.empty() ? nullptr : &combinedCollisionGeometry,
      tgtCharacter.poseShapes.get(),
      tgtCharacter.blendShape);
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
      parent = character.skeleton.joints[parent].parent;
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

  return Character(
      resultSkeleton,
      resultParamTransform,
      resultParameterLimits,
      resultLocators,
      character.mesh.get(),
      resultSkinWeights.get(),
      resultCollisionGeometry.empty() ? nullptr : &resultCollisionGeometry,
      character.poseShapes.get(),
      character.blendShape);
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

} // namespace momentum
