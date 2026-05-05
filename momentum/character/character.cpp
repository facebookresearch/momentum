/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character/character.h"

#include "momentum/character/blend_shape.h"
#include "momentum/character/blend_shape_skinning.h"
#include "momentum/character/character_state.h"
#include "momentum/character/collision_geometry.h"
#include "momentum/character/joint.h"
#include "momentum/character/linear_skinning.h"
#include "momentum/character/parameter_transform.h"
#include "momentum/character/pose_shape.h"
#include "momentum/character/skin_weights.h"
#include "momentum/common/checks.h"
#include "momentum/math/mesh.h"

#include <numeric>
#include <utility>

namespace momentum {

template <typename T>
CharacterT<T>::CharacterT(
    const Skeleton& s,
    const ParameterTransform& pt,
    const ParameterLimits& pl,
    const LocatorList& l,
    const Mesh* m,
    const SkinWeights* sw,
    const CollisionGeometry* cg,
    const PoseShape* bs,
    BlendShape_const_p blendShapes,
    BlendShapeBase_const_p faceExpressionBlendShapes,
    const std::string& nameIn,
    const momentum::TransformationList& inverseBindPose_in,
    const SkinnedLocatorList& skinnedLocators,
    std::string_view metadataIn)
    : skeleton(s),
      parameterTransform(pt),
      parameterLimits(pl),
      locators(l),
      skinnedLocators(skinnedLocators),
      blendShape(std::move(blendShapes)),
      faceExpressionBlendShape(std::move(faceExpressionBlendShapes)),
      inverseBindPose(inverseBindPose_in),
      name(nameIn),
      metadata(metadataIn) {
  if (m) {
    mesh = std::make_unique<Mesh>(*m);
    // Skin weights are only meaningful in the presence of a mesh; drop them otherwise.
    if (sw) {
      skinWeights = std::make_unique<SkinWeights>(*sw);
    }
  }

  if (bs) {
    poseShapes = std::make_unique<PoseShape>(*bs);
  }

  if (inverseBindPose.empty()) {
    initInverseBindPose();
  }
  MT_CHECK(this->inverseBindPose.size() == skeleton.joints.size());

  if (cg) {
    collision = std::make_unique<CollisionGeometry>(*cg);
  }

  resetJointMap();
}

template <typename T>
CharacterT<T>::CharacterT(const CharacterT& c)
    : skeleton(c.skeleton),
      parameterTransform(c.parameterTransform),
      parameterLimits(c.parameterLimits),
      locators(c.locators),
      skinnedLocators(c.skinnedLocators),
      blendShape(c.blendShape),
      faceExpressionBlendShape(c.faceExpressionBlendShape),
      inverseBindPose(c.inverseBindPose),
      jointMap(c.jointMap),
      name(c.name),
      metadata(c.metadata) {
  if (c.mesh) {
    mesh = std::make_unique<Mesh>(*c.mesh);
    // Skin weights are only meaningful in the presence of a mesh; drop them otherwise.
    if (c.skinWeights) {
      skinWeights = std::make_unique<SkinWeights>(*c.skinWeights);
    }
  }

  if (c.poseShapes) {
    poseShapes = std::make_unique<PoseShape>(*c.poseShapes);
  }

  if (c.collision) {
    collision = std::make_unique<CollisionGeometry>(*c.collision);
  }
}

template <typename T>
CharacterT<T>::CharacterT(CharacterT&& c) noexcept = default;

template <typename T>
CharacterT<T>::CharacterT() = default;

template <typename T>
CharacterT<T>::~CharacterT() = default;

template <typename T>
CharacterT<T>& CharacterT<T>::operator=(const CharacterT& rhs) {
  // Copy-and-swap: build a temporary copy, then swap members in for strong exception safety.
  CharacterT<T> tmp(rhs);

  std::swap(skeleton, tmp.skeleton);
  std::swap(parameterTransform, tmp.parameterTransform);
  std::swap(parameterLimits, tmp.parameterLimits);
  std::swap(locators, tmp.locators);
  std::swap(skinnedLocators, tmp.skinnedLocators);
  std::swap(inverseBindPose, tmp.inverseBindPose);
  std::swap(jointMap, tmp.jointMap);
  std::swap(mesh, tmp.mesh);
  std::swap(poseShapes, tmp.poseShapes);
  std::swap(skinWeights, tmp.skinWeights);
  std::swap(collision, tmp.collision);
  std::swap(blendShape, tmp.blendShape);
  std::swap(faceExpressionBlendShape, tmp.faceExpressionBlendShape);
  std::swap(name, tmp.name);
  std::swap(metadata, tmp.metadata);

  return *this;
}

template <typename T>
CharacterT<T>& CharacterT<T>::operator=(CharacterT&& rhs) noexcept = default;

template <typename T>
std::vector<bool> CharacterT<T>::parametersToActiveJoints(const ParameterSet& parameterSet) const {
  const auto nJoints = skeleton.joints.size();
  std::vector<bool> result(nJoints, false);
  for (int k = 0; k < parameterTransform.transform.outerSize(); ++k) {
    for (SparseRowMatrixf::InnerIterator it(parameterTransform.transform, k); it; ++it) {
      // Row index encodes (jointIndex * kParametersPerJoint + jointParamSlot); col is the model
      // parameter that drives it.
      const auto& row = it.row();
      const auto& col = it.col();

      if (!parameterSet.test(col)) {
        continue;
      }

      const auto jointIndex = row / kParametersPerJoint;
      MT_CHECK(jointIndex < nJoints, "{} vs {}", jointIndex, nJoints);
      result[jointIndex] = true;
    }
  }

  return result;
}

template <typename T>
ParameterSet CharacterT<T>::activeJointsToParameters(const std::vector<bool>& activeJoints) const {
  MT_CHECK(
      activeJoints.size() == skeleton.joints.size(),
      "{} is not {}",
      activeJoints.size(),
      skeleton.joints.size());

  ParameterSet result;

  for (int k = 0; k < parameterTransform.transform.outerSize(); ++k) {
    for (SparseRowMatrixf::InnerIterator it(parameterTransform.transform, k); it; ++it) {
      const auto globalParam = it.row();
      const auto kParam = it.col();
      MT_CHECK(
          kParam < (Eigen::Index)parameterTransform.numAllModelParameters(),
          "{} vs {}",
          kParam,
          parameterTransform.numAllModelParameters());
      const auto jointIndex = globalParam / kParametersPerJoint;

      if (activeJoints[jointIndex] != 0) {
        result.set(kParam, true);
      }
    }
  }

  return result;
}

template <typename T>
CharacterT<T> CharacterT<T>::simplifySkeleton(const std::vector<bool>& activeJoints) const {
  MT_CHECK(
      activeJoints.size() == skeleton.joints.size(),
      "{} is not {}",
      activeJoints.size(),
      skeleton.joints.size());
  MT_THROW_IF(
      std::find(activeJoints.begin(), activeJoints.end(), true) == activeJoints.end(),
      "In simplifySkeleton, no joints are enabled; resulting skeleton must have at least one valid joint.");

  Skeleton simplifiedSkeleton;

  // -------------------------------------------------------------------
  //  start by remapping the skeleton and parameter transform
  // -------------------------------------------------------------------

  // Running count of joints kept in the simplified skeleton.
  size_t jointCount = 0;

  // Build a parameter transform whose offsets are zero for active joints but preserved for
  // inactive ones, so the reference state below "bakes" the inactive offsets in.
  auto zeroTransform = parameterTransform;

  for (size_t jointIndex = 0; jointIndex < activeJoints.size(); ++jointIndex) {
    // Zero the offsets for active joints; inactive joints keep their offsets so their effect is
    // captured in the reference state's world transforms below.
    if (activeJoints[jointIndex]) {
      for (size_t o = 0; o < kParametersPerJoint; o++) {
        zeroTransform.offsets(jointIndex * kParametersPerJoint + o) = 0.0f;
      }
    }
  }

  // Maps each original joint index to its position in `simplifiedSkeleton.joints` (or
  // kInvalidIndex if the joint is dropped). Populated below as we iterate.
  std::vector<size_t> intermediateJointMap(skeleton.joints.size(), kInvalidIndex);

  // Reference state where disabled joints contribute their offsets but enabled joints sit at
  // their bind values. Used to bake inactive joint offsets into active children.
  const SkeletonState referenceState(
      zeroTransform.apply(ModelParameters::Zero(zeroTransform.numAllModelParameters())),
      skeleton,
      false);

  // For each original joint index, the last simplified-joint index seen at-or-above it in the
  // hierarchy. Used to find the new parent for an active joint after intermediate parents are
  // pruned.
  std::vector<size_t> lastJoint(skeleton.joints.size(), kInvalidIndex);

  std::vector<size_t> simplifiedJointMap(skeleton.joints.size(), kInvalidIndex);
  for (size_t aIndex = 0; aIndex < skeleton.joints.size(); aIndex++) {
    const Joint& j = skeleton.joints[aIndex];

    // Walk up the original hierarchy to find the nearest active ancestor; that becomes this
    // joint's parent in the simplified skeleton, with the offset re-expressed in the new
    // parent's local frame.
    Vector3f offset = j.translationOffset;
    size_t currentParent = kInvalidIndex;
    size_t sIndex = aIndex;
    if (j.parent != kInvalidIndex) {
      while (currentParent == kInvalidIndex && sIndex != kInvalidIndex) {
        sIndex = skeleton.joints[sIndex].parent;
        if (sIndex == kInvalidIndex) {
          break;
        }
        currentParent = lastJoint[sIndex];
      }
      // Re-express the joint's translation in the (new) parent's local frame; if no parent
      // remains, the translation is in world space (root joint case).
      if (sIndex != kInvalidIndex) {
        offset = referenceState.jointState[sIndex].transform.inverse() *
            referenceState.jointState[aIndex].translation();
      } else {
        offset = referenceState.jointState[aIndex].translation();
      }
    }

    if (activeJoints[aIndex]) {
      Joint jt = j;

      jt.parent = currentParent;

      jt.translationOffset = offset;

      // Compose pre-rotations from every dropped intermediate ancestor into this joint, so the
      // pruned hierarchy still produces the same orientation chain.
      size_t parent = j.parent;
      while (parent != sIndex && parent != kInvalidIndex) {
        jt.preRotation = skeleton.joints[parent].preRotation * jt.preRotation;
        parent = skeleton.joints[parent].parent;
      }

      simplifiedSkeleton.joints.push_back(jt);

      intermediateJointMap[aIndex] = jointCount;
      jointCount++;
      lastJoint[aIndex] = simplifiedSkeleton.joints.size() - 1;

      simplifiedJointMap[aIndex] = jointCount - 1;
    } else {
      // Inactive joints are not added to the simplified skeleton; instead, point their entry in
      // simplifiedJointMap at the nearest active ancestor so consumers (locators, skin weights,
      // etc.) can still resolve them.
      size_t index = aIndex;
      while (index != kInvalidIndex && intermediateJointMap[index] == kInvalidIndex) {
        index = skeleton.joints[index].parent;
      }
      MT_THROW_IF(
          index == kInvalidIndex,
          "During skeleton simplification, inactive joint '{}' has no valid parent joint.  "
          "Every joint in the simplified skeleton must have at least one parent joint that is not disabled.",
          skeleton.joints.at(aIndex).name);
      simplifiedJointMap[aIndex] = intermediateJointMap[index];
    }
  }
  MT_THROW_IF(
      simplifiedSkeleton.joints.empty(),
      "During skeleton simplification, resulting skeleton was empty.  Simplified skeleton must have at least one valid joint.");

  const ParameterTransform simplifiedTransform = mapParameterTransformJoints(
      parameterTransform, simplifiedSkeleton.joints.size(), intermediateJointMap);

  CharacterT<T> result(simplifiedSkeleton, simplifiedTransform);
  result.name = name;
  result.metadata = metadata;

  result.jointMap = simplifiedJointMap;

  // -------------------------------------------------------------------
  //  remap the parameterlimits
  // -------------------------------------------------------------------
  result.parameterLimits = result.remapParameterLimits(parameterLimits, *this);

  // -------------------------------------------------------------------
  //  remap the locators if we have any
  // -------------------------------------------------------------------
  result.locators = result.remapLocators(locators, *this);
  result.skinnedLocators = result.remapSkinnedLocators(skinnedLocators, *this);

  // -------------------------------------------------------------------
  //  remap the mesh and skinning if present
  // -------------------------------------------------------------------

  // Bind states for source and target skeleton — used by the collision-remap block below to
  // transport collision shapes between the two joint frames.
  const SkeletonState sourceBindState(parameterTransform.bindPose(), skeleton, false);
  const SkeletonState targetBindState(result.parameterTransform.bindPose(), result.skeleton, false);

  if (mesh && skinWeights) {
    result.mesh = std::make_unique<Mesh>(*mesh);
    result.skinWeights =
        std::make_unique<SkinWeights>(result.remapSkinWeights(*skinWeights, *this));
  }

  // -------------------------------------------------------------------
  //  remap the collision if we have any
  // -------------------------------------------------------------------
  if (collision != nullptr) {
    result.collision = std::make_unique<CollisionGeometry>(*collision);
    for (auto&& c : *result.collision) {
      const auto oldParent = c.parent;
      c.parent = result.jointMap[c.parent];
      // Re-express the collision shape's local transform in the new parent's frame:
      //   newLocal = targetParentBind^-1 * sourceParentBind * oldLocal
      c.transformation =
          (targetBindState.jointState[c.parent].transform.inverse() *
           sourceBindState.jointState[oldParent].transform * c.transformation);
    }
  }

  return result;
}

template <typename T>
CharacterT<T> CharacterT<T>::simplifyParameterTransform(const ParameterSet& parameterSet) const {
  MT_CHECK(
      parameterSet.count() > 0, "No active parameters in the input to simplifyParameterTransform.");

  auto [subsetParamTransform, subsetParamLimits] =
      subsetParameterTransform(parameterTransform, parameterLimits, parameterSet);

  return CharacterT(
      skeleton,
      subsetParamTransform,
      subsetParamLimits,
      locators,
      mesh.get(),
      skinWeights.get(),
      collision.get(),
      poseShapes.get(),
      blendShape,
      faceExpressionBlendShape,
      name,
      inverseBindPose,
      skinnedLocators,
      metadata);
}

template <typename T>
CharacterT<T> CharacterT<T>::simplify(const ParameterSet& activeParams) const {
  auto activeJoints = parametersToActiveJoints(activeParams);
  // always keep the root joint to ensure valid simplification.
  // This is needed because while previously we generally parametrized the root joint with
  // the root transform, modern skeletons sometimes have a body_world above b_root that is
  // not parametrized, but if we throw it out we end up with an invalid character.
  if (!activeJoints.empty()) {
    activeJoints[0] = true;
  }

  return simplifySkeleton(activeJoints);
}

template <typename T>
SkinWeights CharacterT<T>::remapSkinWeights(
    const SkinWeights& inSkinWeights,
    const CharacterT& originalCharacter) const {
  MT_CHECK(
      originalCharacter.parameterTransform.numAllModelParameters() ==
          parameterTransform.numAllModelParameters(),
      "{} is not {}",
      originalCharacter.parameterTransform.numAllModelParameters(),
      parameterTransform.numAllModelParameters());
  MT_CHECK(
      jointMap.size() == originalCharacter.skeleton.joints.size(),
      "{} is not {}",
      jointMap.size(),
      originalCharacter.skeleton.joints.size());

  SkinWeights result = inSkinWeights;

  for (int v = 0; v < result.index.rows(); v++) {
    // Translate each joint index from the original character to the simplified character.
    for (int i = 0; i < gsl::narrow_cast<int>(kMaxSkinJoints); i++) {
      result.index(v, i) = static_cast<uint32_t>(jointMap[result.index(v, i)]);
    }

    // Multiple original joints may collapse to the same simplified joint; merge their weights
    // into the first slot and zero the duplicates.
    for (int i = 0; i < gsl::narrow_cast<int>(kMaxSkinJoints); i++) {
      const auto& joint = result.index(v, i);
      for (int j = i + 1; j < gsl::narrow_cast<int>(kMaxSkinJoints); j++) {
        if (result.index(v, j) == joint) {
          result.weight(v, i) += result.weight(v, j);
          result.weight(v, j) = 0.0f;
        }
      }
    }

    // Sort influences by weight (descending) so callers that truncate to fewer slots keep the
    // most significant ones.
    for (int i = 0; i < gsl::narrow_cast<int>(kMaxSkinJoints); i++) {
      for (int j = i + 1; j < gsl::narrow_cast<int>(kMaxSkinJoints); j++) {
        if (result.weight(v, i) < result.weight(v, j)) {
          std::swap(result.weight(v, i), result.weight(v, j));
          std::swap(result.index(v, i), result.index(v, j));
        }
      }
    }
  }

  return result;
}

template <typename T>
ParameterLimits CharacterT<T>::remapParameterLimits(
    const ParameterLimits& limits,
    const CharacterT& originalCharacter) const {
  MT_CHECK(
      originalCharacter.parameterTransform.numAllModelParameters() ==
          parameterTransform.numAllModelParameters(),
      "{} is not {}",
      originalCharacter.parameterTransform.numAllModelParameters(),
      parameterTransform.numAllModelParameters());
  MT_CHECK(
      jointMap.size() == originalCharacter.skeleton.joints.size(),
      "{} is not {}",
      jointMap.size(),
      originalCharacter.skeleton.joints.size());

  ParameterLimits result;

  // create bind states for both source and target skeleton
  const SkeletonState sourceBindState(
      originalCharacter.parameterTransform.bindPose(), originalCharacter.skeleton, false);
  const SkeletonState targetBindState(parameterTransform.bindPose(), skeleton, false);

  result = limits;
  for (auto&& limit : result) {
    // only need to remap limits that depend on joints
    if (limit.type == MinMaxJoint || limit.type == MinMaxJointPassive) {
      auto& data = limit.data.minMaxJoint;
      data.jointIndex = jointMap[data.jointIndex];
    } else if (limit.type == Ellipsoid) {
      // ellipsoid limit requires re-targeting the ellipsoid to the new joint's local system
      auto& data = limit.data.ellipsoid;
      const auto sourceParent = data.parent;
      const auto targetParent = jointMap[sourceParent];
      MT_CHECK(
          targetParent < targetBindState.jointState.size(),
          "{} vs {}",
          targetParent,
          targetBindState.jointState.size());
      data.parent = targetParent;

      const auto targetTransformationInverse =
          targetBindState.jointState[targetParent].transform.inverse();
      const auto sourceTransformation = sourceBindState.jointState[sourceParent].transform;
      data.offset = targetTransformationInverse * sourceTransformation * data.offset;

      const auto sourceEllipsoidParent = data.ellipsoidParent;
      data.ellipsoidParent = jointMap[data.ellipsoidParent];
      const auto targetEllipsoidInverse =
          targetBindState.jointState[data.ellipsoidParent].transform.inverse();
      const auto sourceEllipsoid = sourceBindState.jointState[sourceEllipsoidParent].transform;
      data.ellipsoid = targetEllipsoidInverse * sourceEllipsoid * data.ellipsoid;
      data.ellipsoidInv = data.ellipsoid.inverse();
    }
  }

  return result;
}

template <typename T>
SkinnedLocatorList CharacterT<T>::remapSkinnedLocators(
    const SkinnedLocatorList& locs,
    const CharacterT& originalCharacter) const {
  MT_CHECK(
      originalCharacter.parameterTransform.numAllModelParameters() ==
          parameterTransform.numAllModelParameters(),
      "{} is not {}",
      originalCharacter.parameterTransform.numAllModelParameters(),
      parameterTransform.numAllModelParameters());
  MT_CHECK(
      jointMap.size() == originalCharacter.skeleton.joints.size(),
      "{} is not {}",
      jointMap.size(),
      originalCharacter.skeleton.joints.size());

  SkinnedLocatorList result = locs;

  // Note: unlike remapLocators(), this function does not need the source/target bind states.
  // A skinned locator stores a single bind-frame offset that is blended across its weighted
  // parents at evaluation time; it has no per-influence offset to re-express. When collapsed
  // joints share the same bind transform (the typical case for redundant intermediate joints),
  // the merged-weight result is unchanged. If callers ever simplify across joints whose bind
  // transforms differ, the locator offset will need explicit remapping.

  for (auto& sl : result) {
    // Translate each parent joint index from the original character to the simplified character.
    for (int i = 0; i < gsl::narrow_cast<int>(kMaxSkinJoints); i++) {
      sl.parents(i) = static_cast<uint32_t>(jointMap[sl.parents(i)]);
    }

    // Multiple original joints may collapse to the same simplified joint; merge their weights
    // into the first slot and zero the duplicates.
    for (int i = 0; i < gsl::narrow_cast<int>(kMaxSkinJoints); i++) {
      const auto& joint = sl.parents(i);
      for (int j = i + 1; j < gsl::narrow_cast<int>(kMaxSkinJoints); j++) {
        if (sl.parents(j) == joint) {
          sl.skinWeights(i) += sl.skinWeights(j);
          sl.skinWeights(j) = 0.0f;
        }
      }
    }

    // Sort influences by weight (descending) so callers that truncate to fewer slots keep the
    // most significant ones.
    for (int i = 0; i < gsl::narrow_cast<int>(kMaxSkinJoints); i++) {
      for (int j = i + 1; j < gsl::narrow_cast<int>(kMaxSkinJoints); j++) {
        if (sl.skinWeights(i) < sl.skinWeights(j)) {
          std::swap(sl.skinWeights(i), sl.skinWeights(j));
          std::swap(sl.parents(i), sl.parents(j));
        }
      }
    }
  }

  return result;
}

template <typename T>
LocatorList CharacterT<T>::remapLocators(
    const LocatorList& locs,
    const CharacterT& originalCharacter) const {
  MT_CHECK(
      originalCharacter.parameterTransform.numAllModelParameters() ==
          parameterTransform.numAllModelParameters(),
      "{} is not {}",
      originalCharacter.parameterTransform.numAllModelParameters(),
      parameterTransform.numAllModelParameters());
  MT_CHECK(
      jointMap.size() == originalCharacter.skeleton.joints.size(),
      "{} is not {}",
      jointMap.size(),
      originalCharacter.skeleton.joints.size());

  LocatorList result;

  // Bind states for source and target skeletons; used below to re-express each locator's offset
  // in the new parent joint's local frame.
  const SkeletonState sourceBindState(
      originalCharacter.parameterTransform.bindPose(), originalCharacter.skeleton, false);
  const SkeletonState targetBindState(parameterTransform.bindPose(), skeleton, false);

  for (const auto& sourceLocator : locs) {
    result.emplace_back(sourceLocator);
    auto& loc = result.back();
    loc.parent = jointMap[loc.parent];
    // newOffset = targetParentBind^-1 * sourceParentBind * oldOffset
    loc.offset = targetBindState.jointState[loc.parent].transform.inverse() *
        sourceBindState.jointState[sourceLocator.parent].transform * sourceLocator.offset;
  }

  return result;
}

template <typename T>
CharacterParameters CharacterT<T>::bindPose() const {
  CharacterParameters result;
  result.pose = VectorXf::Zero(parameterTransform.numAllModelParameters());
  result.offsets = VectorXf::Zero(skeleton.joints.size() * kParametersPerJoint);
  return result;
}

template <typename T>
void CharacterT<T>::initParameterTransform() {
  size_t nParams = skeleton.joints.size() * kParametersPerJoint;
  parameterTransform = ParameterTransform::empty(nParams);
}

template <typename T>
void CharacterT<T>::resetJointMap() {
  // Identity map: jointMap[i] = i. Used when no skeleton simplification has been performed.
  jointMap.resize(skeleton.joints.size());
  std::iota(jointMap.begin(), jointMap.end(), 0);
}

template <typename T>
void CharacterT<T>::initInverseBindPose() {
  const SkeletonState bindState(parameterTransform.bindPose(), skeleton, false);
  if (!inverseBindPose.empty()) {
    inverseBindPose.clear();
  }
  inverseBindPose.reserve(bindState.jointState.size());
  for (const auto& t : bindState.jointState) {
    inverseBindPose.push_back(t.transform.inverse().toAffine3());
  }
}

template <typename T>
CharacterParameters CharacterT<T>::splitParameters(
    const CharacterT& character,
    const CharacterParameters& parameters,
    const ParameterSet& parameterSet) {
  auto parameterSplit = parameters;
  auto offsetSplit = parameters;
  for (int j = 0; j < offsetSplit.pose.size(); j++) {
    if (parameterSet.test(j)) {
      parameterSplit.pose[j] = 0.0f;
    } else {
      offsetSplit.pose[j] = 0.0f;
    }
  }
  parameterSplit.offsets = character.parameterTransform.apply(offsetSplit);
  return parameterSplit;
}

template <typename T>
CharacterT<T> CharacterT<T>::withBlendShape(
    BlendShape_const_p blendShape_in,
    Eigen::Index maxBlendShapes) const {
  CharacterT<T> result = *this;
  result.addBlendShape(blendShape_in, maxBlendShapes);
  return result;
}

template <typename T>
void CharacterT<T>::addBlendShape(
    const BlendShape_const_p& blendShape_in,
    Eigen::Index maxBlendShapes) {
  MT_CHECK(this->mesh);
  if (!blendShape_in) {
    std::tie(parameterTransform, parameterLimits) =
        addBlendShapeParameters(parameterTransform, parameterLimits, 0);
    this->blendShape.reset();
    return;
  }

  // TODO should we accommodate passing in a BlendShape with "extra" vertices (for the
  // higher subdivision levels)?
  MT_CHECK(
      blendShape_in->modelSize() == this->mesh->vertices.size(),
      "{} is not {}",
      blendShape_in->modelSize(),
      this->mesh->vertices.size());

  blendShape = blendShape_in;

  // Augment the parameter transform with blend shape parameters:
  std::tie(parameterTransform, parameterLimits) = addBlendShapeParameters(
      parameterTransform, parameterLimits, std::min(maxBlendShapes, blendShape_in->shapeSize()));
}

template <typename T>
CharacterT<T> CharacterT<T>::withFaceExpressionBlendShape(
    BlendShapeBase_const_p blendShape_in,
    Eigen::Index maxBlendShapes) const {
  CharacterT<T> res = *this;
  res.addFaceExpressionBlendShape(blendShape_in, maxBlendShapes);
  return res;
}

template <typename T>
void CharacterT<T>::addFaceExpressionBlendShape(
    const BlendShapeBase_const_p& blendShape_in,
    Eigen::Index maxBlendShapes) {
  MT_CHECK(mesh);

  if (!blendShape_in) {
    std::tie(parameterTransform, parameterLimits) =
        addFaceExpressionParameters(parameterTransform, parameterLimits, 0);
    this->faceExpressionBlendShape.reset();
    return;
  }

  MT_CHECK(
      blendShape_in->modelSize() == mesh->vertices.size(),
      "{} is not {}",
      blendShape_in->modelSize(),
      mesh->vertices.size());

  faceExpressionBlendShape = blendShape_in;

  auto nBlendShapes = blendShape_in->shapeSize();
  if (maxBlendShapes > 0) {
    nBlendShapes = std::min(maxBlendShapes, nBlendShapes);
  }

  // Augment the parameter transform with blend shape parameters:
  std::tie(parameterTransform, parameterLimits) =
      addFaceExpressionParameters(parameterTransform, parameterLimits, nBlendShapes);
}

template <typename T>
CharacterT<T> CharacterT<T>::bakeBlendShape(const BlendWeights& blendWeights) const {
  CharacterT<T> result = *this;
  MT_CHECK(result.mesh);
  if (this->blendShape) {
    result.mesh->vertices = this->blendShape->template computeShape<float>(blendWeights);
    result.mesh->updateNormals();
  }

  std::tie(result.parameterTransform, result.parameterLimits) = subsetParameterTransform(
      this->parameterTransform,
      this->parameterLimits,
      ~this->parameterTransform.getBlendShapeParameters());

  return result;
}

template <typename T>
CharacterT<T> CharacterT<T>::bakeBlendShape(const ModelParameters& modelParams) const {
  return bakeBlendShape(extractBlendWeights(this->parameterTransform, modelParams));
}

template <typename T>
CharacterT<T> CharacterT<T>::bake(
    const ModelParameters& modelParams,
    bool bakeBlendShapes,
    bool bakeScales) const {
  // 1. Clone the character so we don't mutate the original.
  CharacterT<T> result = *this;
  MT_CHECK(result.mesh);

  // 2. Blend-shape baking: replace the rest-pose vertices with the blend-shape evaluation.
  if (bakeBlendShapes && this->blendShape) {
    const BlendWeights blendWeights = extractBlendWeights(result.parameterTransform, modelParams);
    result.mesh->vertices = this->blendShape->template computeShape<float>(blendWeights);
  }

  // 3. Scale (or any joint transform) baking: skin the rest mesh through the posed skeleton so
  // the resulting vertices already include the joint transforms; then strip those parameters
  // below.
  if (bakeScales && this->skinWeights) {
    const SkeletonState posed(
        result.parameterTransform.apply(modelParams),
        result.skeleton,
        /*computeDeriv =*/false);
    Mesh restMesh = *result.mesh;
    applySSD(result.inverseBindPose, *result.skinWeights, restMesh, posed, *result.mesh);
  }

  result.mesh->updateNormals();

  // 4. Strip baked parameters from the rig
  ParameterSet baked = ParameterSet();
  if (bakeBlendShapes) {
    baked |= result.parameterTransform.getBlendShapeParameters();
  }
  if (bakeScales) {
    baked |= result.parameterTransform.getScalingParameters();
  }

  std::tie(result.parameterTransform, result.parameterLimits) = subsetParameterTransform(
      result.parameterTransform,
      result.parameterLimits,
      ~baked); // keep unbaked params only

  return result;
}

template struct CharacterT<float>;
template struct CharacterT<double>;

} // namespace momentum
