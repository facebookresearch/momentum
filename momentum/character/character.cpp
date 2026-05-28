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
#include "momentum/character/character_utility.h"
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
#include <vector>

namespace momentum {

namespace {

ParameterTransform zeroActiveJointOffsets(
    ParameterTransform parameterTransform,
    const std::vector<bool>& activeJoints) {
  for (size_t jointIndex = 0; jointIndex < activeJoints.size(); ++jointIndex) {
    if (!activeJoints[jointIndex]) {
      continue;
    }

    for (size_t offset = 0; offset < kParametersPerJoint; ++offset) {
      parameterTransform.offsets(jointIndex * kParametersPerJoint + offset) = 0.0f;
    }
  }

  return parameterTransform;
}

struct SimplifiedJointPlacement {
  size_t parent = kInvalidIndex;
  size_t retainedAncestor = kInvalidIndex;
  Vector3f translationOffset = Vector3f::Zero();
};

SimplifiedJointPlacement getSimplifiedJointPlacement(
    const Skeleton& skeleton,
    const SkeletonState& referenceState,
    const std::vector<size_t>& lastJoint,
    const size_t jointIndex) {
  const Joint& joint = skeleton.joints.at(jointIndex);
  if (joint.parent == kInvalidIndex) {
    return {kInvalidIndex, jointIndex, joint.translationOffset};
  }

  size_t retainedAncestor = jointIndex;
  size_t simplifiedParent = kInvalidIndex;
  while (simplifiedParent == kInvalidIndex && retainedAncestor != kInvalidIndex) {
    retainedAncestor = skeleton.joints.at(retainedAncestor).parent;
    if (retainedAncestor != kInvalidIndex) {
      simplifiedParent = lastJoint.at(retainedAncestor);
    }
  }

  const Vector3f offset = retainedAncestor != kInvalidIndex
      ? referenceState.jointState.at(retainedAncestor).transform.inverse() *
          referenceState.jointState.at(jointIndex).translation()
      : referenceState.jointState.at(jointIndex).translation();
  return {simplifiedParent, retainedAncestor, offset};
}

Joint makeSimplifiedJoint(
    const Skeleton& skeleton,
    const size_t jointIndex,
    const SimplifiedJointPlacement& placement) {
  Joint result = skeleton.joints.at(jointIndex);
  result.parent = placement.parent;
  result.translationOffset = placement.translationOffset;

  size_t parent = skeleton.joints.at(jointIndex).parent;
  while (parent != placement.retainedAncestor && parent != kInvalidIndex) {
    result.preRotation = skeleton.joints.at(parent).preRotation * result.preRotation;
    parent = skeleton.joints.at(parent).parent;
  }

  return result;
}

size_t findSimplifiedAncestor(
    const Skeleton& skeleton,
    const std::vector<size_t>& intermediateJointMap,
    const size_t jointIndex) {
  size_t ancestor = jointIndex;
  while (ancestor != kInvalidIndex && intermediateJointMap.at(ancestor) == kInvalidIndex) {
    ancestor = skeleton.joints.at(ancestor).parent;
  }

  MT_THROW_IF(
      ancestor == kInvalidIndex,
      "During skeleton simplification, inactive joint '{}' has no valid parent joint.  "
      "Every joint in the simplified skeleton must have at least one parent joint that is not disabled.",
      skeleton.joints.at(jointIndex).name);
  return intermediateJointMap.at(ancestor);
}

Matrix3f inertiaInJointFrame(const JointPhysicalProperties& properties) {
  const Matrix3f inertiaRotation = properties.inertiaRotation.toRotationMatrix();
  return inertiaRotation * properties.inertia * inertiaRotation.transpose();
}

void addParallelAxisShift(Matrix3f& inertia, const float mass, const Vector3f& offset) {
  const float dx = offset.x();
  const float dy = offset.y();
  const float dz = offset.z();
  inertia(0, 0) += mass * (dy * dy + dz * dz);
  inertia(1, 1) += mass * (dx * dx + dz * dz);
  inertia(2, 2) += mass * (dx * dx + dy * dy);

  const float xy = -mass * dx * dy;
  const float xz = -mass * dx * dz;
  const float yz = -mass * dy * dz;
  inertia(0, 1) += xy;
  inertia(1, 0) += xy;
  inertia(0, 2) += xz;
  inertia(2, 0) += xz;
  inertia(1, 2) += yz;
  inertia(2, 1) += yz;
}

Matrix3f inertiaShiftedToCenter(
    const JointPhysicalProperties& properties,
    const Vector3f& centerOfMassOffset) {
  Matrix3f result = inertiaInJointFrame(properties);
  const Vector3f offset = properties.centerOfMassOffset - centerOfMassOffset;
  addParallelAxisShift(result, properties.mass, offset);
  return result;
}

void symmetrizeInPlace(Matrix3f& inertia) {
  const float xy = 0.5f * (inertia(0, 1) + inertia(1, 0));
  const float xz = 0.5f * (inertia(0, 2) + inertia(2, 0));
  const float yz = 0.5f * (inertia(1, 2) + inertia(2, 1));
  inertia(0, 1) = xy;
  inertia(1, 0) = xy;
  inertia(0, 2) = xz;
  inertia(2, 0) = xz;
  inertia(1, 2) = yz;
  inertia(2, 1) = yz;
}

void mergePhysicalProperties(
    JointPhysicalProperties& target,
    const JointPhysicalProperties& source) {
  const float mergedMass = target.mass + source.mass;
  const Vector3f mergedCenterOfMassOffset = mergedMass > 0.0f
      ? (target.mass * target.centerOfMassOffset + source.mass * source.centerOfMassOffset) /
          mergedMass
      : target.centerOfMassOffset;
  Matrix3f mergedInertia = inertiaShiftedToCenter(target, mergedCenterOfMassOffset) +
      inertiaShiftedToCenter(source, mergedCenterOfMassOffset);
  symmetrizeInPlace(mergedInertia);

  target.mass = mergedMass;
  target.centerOfMassOffset = mergedCenterOfMassOffset;
  target.inertia = mergedInertia;
  target.inertiaRotation = Quaternionf::Identity();
}

PhysicalProperties remapPhysicalProperties(
    const PhysicalProperties& physicalProperties,
    const Skeleton& sourceSkeleton,
    const Skeleton& targetSkeleton,
    const std::vector<size_t>& jointMap,
    const SkeletonState& sourceBindState,
    const SkeletonState& targetBindState) {
  PhysicalProperties result;
  result.reserve(physicalProperties.size());
  std::vector<size_t> resultIndexByJoint(targetSkeleton.joints.size(), kInvalidIndex);

  for (const auto& jointProperties : physicalProperties) {
    const size_t oldJointIndex =
        resolvePhysicalPropertiesJointIndex(jointProperties, sourceSkeleton);
    if (oldJointIndex == kInvalidIndex || oldJointIndex >= jointMap.size()) {
      continue;
    }

    const size_t newJointIndex = jointMap[oldJointIndex];
    if (newJointIndex == kInvalidIndex || newJointIndex >= targetSkeleton.joints.size()) {
      continue;
    }

    JointPhysicalProperties mappedProperties = jointProperties;
    mappedProperties.jointIndex = newJointIndex;
    mappedProperties.jointName = targetSkeleton.joints.at(newJointIndex).name;

    const auto oldToNewLocal = targetBindState.jointState.at(newJointIndex).transform.inverse() *
        sourceBindState.jointState.at(oldJointIndex).transform;
    mappedProperties.centerOfMassOffset = oldToNewLocal * jointProperties.centerOfMassOffset;
    mappedProperties.inertiaRotation =
        (oldToNewLocal.rotation * jointProperties.inertiaRotation).normalized();

    // Skeleton simplification can fold multiple source joints into one target joint. Keep the
    // one-body-per-joint invariant by merging those bodies in the target joint frame.
    size_t& resultIndex = resultIndexByJoint.at(newJointIndex);
    if (resultIndex == kInvalidIndex) {
      resultIndex = result.size();
      result.push_back(std::move(mappedProperties));
    } else {
      mergePhysicalProperties(result.at(resultIndex), mappedProperties);
    }
  }

  return result;
}

std::unique_ptr<CollisionGeometry> remapCollisionGeometry(
    const CollisionGeometry* collision,
    const std::vector<size_t>& jointMap,
    const SkeletonState& sourceBindState,
    const SkeletonState& targetBindState) {
  if (collision == nullptr) {
    return {};
  }

  auto result = std::make_unique<CollisionGeometry>(*collision);
  for (auto& collisionGeometry : *result) {
    const auto oldParent = collisionGeometry.parent;
    collisionGeometry.parent = jointMap.at(collisionGeometry.parent);
    // Re-express the collision shape's local transform in the new parent's frame:
    //   newLocal = targetParentBind^-1 * sourceParentBind * oldLocal
    collisionGeometry.transformation =
        targetBindState.jointState.at(collisionGeometry.parent).transform.inverse() *
        sourceBindState.jointState.at(oldParent).transform * collisionGeometry.transformation;
  }

  return result;
}

} // namespace

Character::Character(
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
    std::string_view metadataIn,
    const PhysicalProperties& physicalPropertiesIn)
    : skeleton(s),
      parameterTransform(pt),
      parameterLimits(pl),
      locators(l),
      skinnedLocators(skinnedLocators),
      physicalProperties(physicalPropertiesIn),
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

Character::Character(const Character& c)
    : skeleton(c.skeleton),
      parameterTransform(c.parameterTransform),
      parameterLimits(c.parameterLimits),
      locators(c.locators),
      skinnedLocators(c.skinnedLocators),
      physicalProperties(c.physicalProperties),
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

Character::Character(Character&& c) noexcept = default;

Character::Character() = default;

Character::~Character() = default;

Character& Character::operator=(const Character& rhs) {
  // Copy-and-swap: build a temporary copy, then swap members in for strong exception safety.
  Character tmp(rhs);

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
  std::swap(physicalProperties, tmp.physicalProperties);
  std::swap(blendShape, tmp.blendShape);
  std::swap(faceExpressionBlendShape, tmp.faceExpressionBlendShape);
  std::swap(name, tmp.name);
  std::swap(metadata, tmp.metadata);

  return *this;
}

Character& Character::operator=(Character&& rhs) noexcept = default;

std::vector<bool> Character::parametersToActiveJoints(const ParameterSet& parameterSet) const {
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

ParameterSet Character::activeJointsToParameters(const std::vector<bool>& activeJoints) const {
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

Character Character::simplifySkeleton(const std::vector<bool>& activeJoints) const {
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

  // Build a parameter transform whose offsets are zero for active joints but preserved for
  // inactive ones, so the reference state below "bakes" the inactive offsets in.
  const auto zeroTransform = zeroActiveJointOffsets(parameterTransform, activeJoints);

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
    // `retainedAncestor` identifies the original ancestor whose simplified mapping supplies the
    // new parent frame. Dropped joints between it and this joint are folded into the kept joint.
    const auto placement = getSimplifiedJointPlacement(skeleton, referenceState, lastJoint, aIndex);

    if (activeJoints[aIndex]) {
      simplifiedSkeleton.joints.push_back(makeSimplifiedJoint(skeleton, aIndex, placement));
      const size_t newJointIndex = simplifiedSkeleton.joints.size() - 1;
      intermediateJointMap.at(aIndex) = newJointIndex;
      lastJoint.at(aIndex) = newJointIndex;
      simplifiedJointMap.at(aIndex) = newJointIndex;
    } else {
      // Inactive joints are not added to the simplified skeleton; instead, point their entry in
      // simplifiedJointMap at the nearest active ancestor so consumers (locators, skin weights,
      // etc.) can still resolve them.
      simplifiedJointMap.at(aIndex) =
          findSimplifiedAncestor(skeleton, intermediateJointMap, aIndex);
    }
  }
  MT_THROW_IF(
      simplifiedSkeleton.joints.empty(),
      "During skeleton simplification, resulting skeleton was empty.  Simplified skeleton must have at least one valid joint.");

  const ParameterTransform simplifiedTransform = mapParameterTransformJoints(
      parameterTransform, simplifiedSkeleton.joints.size(), intermediateJointMap);

  Character result(simplifiedSkeleton, simplifiedTransform);
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

  // -------------------------------------------------------------------
  //  remap physical joint properties if present
  // -------------------------------------------------------------------
  result.physicalProperties = remapPhysicalProperties(
      physicalProperties,
      skeleton,
      result.skeleton,
      result.jointMap,
      sourceBindState,
      targetBindState);

  if (mesh && skinWeights) {
    result.mesh = std::make_unique<Mesh>(*mesh);
    result.skinWeights =
        std::make_unique<SkinWeights>(result.remapSkinWeights(*skinWeights, *this));
  }

  // -------------------------------------------------------------------
  //  remap the collision if we have any
  // -------------------------------------------------------------------
  result.collision =
      remapCollisionGeometry(collision.get(), result.jointMap, sourceBindState, targetBindState);

  return result;
}

Character Character::simplifyParameterTransform(const ParameterSet& parameterSet) const {
  MT_CHECK(
      parameterSet.count() > 0, "No active parameters in the input to simplifyParameterTransform.");

  auto [subsetParamTransform, subsetParamLimits] =
      subsetParameterTransform(parameterTransform, parameterLimits, parameterSet);

  return {
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
      metadata,
      physicalProperties};
}

Character Character::simplify(const ParameterSet& activeParams) const {
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

SkinWeights Character::remapSkinWeights(
    const SkinWeights& inSkinWeights,
    const Character& originalCharacter) const {
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

ParameterLimits Character::remapParameterLimits(
    const ParameterLimits& limits,
    const Character& originalCharacter) const {
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

SkinnedLocatorList Character::remapSkinnedLocators(
    const SkinnedLocatorList& locs,
    const Character& originalCharacter) const {
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

LocatorList Character::remapLocators(const LocatorList& locs, const Character& originalCharacter)
    const {
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

CharacterParameters Character::bindPose() const {
  CharacterParameters result;
  result.pose = VectorXf::Zero(parameterTransform.numAllModelParameters());
  result.offsets = VectorXf::Zero(skeleton.joints.size() * kParametersPerJoint);
  return result;
}

void Character::initParameterTransform() {
  size_t nParams = skeleton.joints.size() * kParametersPerJoint;
  parameterTransform = ParameterTransform::empty(nParams);
}

void Character::resetJointMap() {
  // Identity map: jointMap[i] = i. Used when no skeleton simplification has been performed.
  jointMap.resize(skeleton.joints.size());
  std::iota(jointMap.begin(), jointMap.end(), 0);
}

void Character::initInverseBindPose() {
  const SkeletonState bindState(parameterTransform.bindPose(), skeleton, false);
  if (!inverseBindPose.empty()) {
    inverseBindPose.clear();
  }
  inverseBindPose.reserve(bindState.jointState.size());
  for (const auto& t : bindState.jointState) {
    inverseBindPose.push_back(t.transform.inverse().toAffine3());
  }
}

CharacterParameters Character::splitParameters(
    const Character& character,
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

Character Character::withBlendShape(
    const BlendShape_const_p& blendShape_in,
    Eigen::Index maxBlendShapes) const {
  Character result = *this;
  result.addBlendShape(blendShape_in, maxBlendShapes);
  return result;
}

void Character::addBlendShape(
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

Character Character::withFaceExpressionBlendShape(
    const BlendShapeBase_const_p& blendShape_in,
    Eigen::Index maxBlendShapes) const {
  Character res = *this;
  res.addFaceExpressionBlendShape(blendShape_in, maxBlendShapes);
  return res;
}

void Character::addFaceExpressionBlendShape(
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

Character Character::bakeBlendShape(const BlendWeights& blendWeights) const {
  Character result = *this;
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

Character Character::bakeBlendShape(const ModelParameters& modelParams) const {
  return bakeBlendShape(extractBlendWeights(this->parameterTransform, modelParams));
}

Character Character::bake(const ModelParameters& modelParams, bool bakeBlendShapes, bool bakeScales)
    const {
  // 1. Clone the character so we don't mutate the original.
  Character result = *this;
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

} // namespace momentum
