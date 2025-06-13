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
    const momentum::TransformationList& inverseBindPose_in)
    : skeleton(s),
      parameterTransform(pt),
      parameterLimits(pl),
      locators(l),
      blendShape(std::move(blendShapes)),
      faceExpressionBlendShape(std::move(faceExpressionBlendShapes)),
      inverseBindPose(inverseBindPose_in),
      name(nameIn) {
  if (m) {
    mesh = std::make_unique<Mesh>(*m);
    // create skinweights copy only if both mesh and skinweights exist
    if (sw) {
      skinWeights = std::make_unique<SkinWeights>(*sw);
    }
  }

  // copy pose blendshapes if present
  if (bs) {
    poseShapes = std::make_unique<PoseShape>(*bs);
  }

  // initialize bindpose inverse list
  if (inverseBindPose.empty()) {
    initInverseBindPose();
  }
  MT_CHECK(this->inverseBindPose.size() == skeleton.joints.size());

  // copy collision geometry if it exists
  if (cg) {
    collision = std::make_unique<CollisionGeometry>(*cg);
  }

  // fill jointMap with identity
  resetJointMap();
}

template <typename T>
CharacterT<T>::CharacterT(const CharacterT& c)
    : skeleton(c.skeleton),
      parameterTransform(c.parameterTransform),
      parameterLimits(c.parameterLimits),
      locators(c.locators),
      blendShape(c.blendShape),
      faceExpressionBlendShape(c.faceExpressionBlendShape),
      inverseBindPose(c.inverseBindPose),
      jointMap(c.jointMap),
      name(c.name) {
  if (c.mesh) {
    mesh = std::make_unique<Mesh>(*c.mesh);
    // create skinweights copy only if both mesh and skinweights exist
    if (c.skinWeights) {
      skinWeights = std::make_unique<SkinWeights>(*c.skinWeights);
    }
  }

  // copy pose blendshapes if present
  if (c.poseShapes) {
    poseShapes = std::make_unique<PoseShape>(*c.poseShapes);
  }

  // copy collision geometry if it exists
  if (c.collision) {
    collision = std::make_unique<CollisionGeometry>(*c.collision);
  }
}

template <typename T>
CharacterT<T>::CharacterT() = default;

template <typename T>
CharacterT<T>::~CharacterT() = default;

template <typename T>
CharacterT<T>& CharacterT<T>::operator=(const CharacterT& rhs) {
  // create copy
  CharacterT<T> tmp(rhs);

  // now swap out
  std::swap(skeleton, tmp.skeleton);
  std::swap(parameterTransform, tmp.parameterTransform);
  std::swap(parameterLimits, tmp.parameterLimits);
  std::swap(locators, tmp.locators);
  std::swap(inverseBindPose, tmp.inverseBindPose);
  std::swap(jointMap, tmp.jointMap);
  std::swap(mesh, tmp.mesh);
  std::swap(poseShapes, tmp.poseShapes);
  std::swap(skinWeights, tmp.skinWeights);
  std::swap(collision, tmp.collision);
  std::swap(blendShape, tmp.blendShape);
  std::swap(faceExpressionBlendShape, tmp.faceExpressionBlendShape);
  std::swap(name, tmp.name);

  return *this;
}

template <typename T>
std::vector<bool> CharacterT<T>::parametersToActiveJoints(const ParameterSet& parameterSet) const {
  // create a list of joints that are currently enabled
  const auto nJoints = skeleton.joints.size();
  std::vector<bool> result(nJoints, false);
  for (int k = 0; k < parameterTransform.transform.outerSize(); ++k) {
    for (SparseRowMatrixf::InnerIterator it(parameterTransform.transform, k); it; ++it) {
      // row is influenced joint, col is parameter
      const auto& row = it.row();
      const auto& col = it.col();

      // disable any joints that are not set
      if (!parameterSet.test(col)) {
        continue;
      }

      // set joint as enabled
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

  // iterate over all non-zero entries of the matrix
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

// create simplified skeleton for enabled parameters
template <typename T>
CharacterT<T> CharacterT<T>::simplifySkeleton(const std::vector<bool>& enabledJoints) const {
  MT_CHECK(
      enabledJoints.size() == skeleton.joints.size(),
      "{} is not {}",
      enabledJoints.size(),
      skeleton.joints.size());
  MT_THROW_IF(
      std::find(enabledJoints.begin(), enabledJoints.end(), true) == enabledJoints.end(),
      "In simplifySkeleton, no joints are enabled; resulting skeleton must have at least one valid joint.");

  Skeleton simplifiedSkeleton;

  // -------------------------------------------------------------------
  //  start by remapping the skeleton and parameter transform
  // -------------------------------------------------------------------

  // remember parameter mapping from skeleton to anim skeleton
  size_t jointCount = 0;

  // create a parameter transform with the correct offsets
  auto zeroTransform = parameterTransform;

  for (size_t jointIndex = 0; jointIndex < enabledJoints.size(); ++jointIndex) {
    // zero out all offsets related to joints that are enabled, keep the disabled ones set
    if (enabledJoints[jointIndex]) {
      for (size_t o = 0; o < kParametersPerJoint; o++) {
        zeroTransform.offsets(jointIndex * kParametersPerJoint + o) = 0.0f;
      }
    }
  }

  // some state for mapping joints to new joints
  std::vector<size_t> intermediateJointMap(skeleton.joints.size(), kInvalidIndex);

  // create a reference state for the model where all disabled joints keep their offset influence
  const SkeletonState referenceState(
      zeroTransform.apply(ModelParameters::Zero(zeroTransform.numAllModelParameters())),
      skeleton,
      false);

  // create map from anim skeleton to new joints
  std::vector<size_t> lastJoint(skeleton.joints.size(), kInvalidIndex);

  // go over the anim skeleton, check which joints are active and convert them
  std::vector<size_t> simplifiedJointMap(skeleton.joints.size(), kInvalidIndex);
  for (size_t aIndex = 0; aIndex < skeleton.joints.size(); aIndex++) {
    const Joint& j = skeleton.joints[aIndex];

    // find the parent joint that belongs here
    Vector3f offset = j.translationOffset;
    size_t currentParent = kInvalidIndex;
    size_t sIndex = aIndex;
    if (j.parent != kInvalidIndex) {
      // find the last not disabled parent joint
      while (currentParent == kInvalidIndex && sIndex != kInvalidIndex) {
        sIndex = skeleton.joints[sIndex].parent;
        currentParent = lastJoint[sIndex];
      }
      // calculate the new offset of the joint in the new parent space
      if (sIndex != kInvalidIndex) {
        offset = referenceState.jointState[sIndex].transformation.inverse() *
            referenceState.jointState[aIndex].translation();
      } else {
        offset = referenceState.jointState[aIndex].translation();
      }
    }

    // is joint enabled?
    if (enabledJoints[aIndex]) {
      // create copy
      Joint jt = j;

      // set new parent
      jt.parent = currentParent;

      // set updated offset
      jt.translationOffset = offset;

      // update prerotation for the new parent joint by summing up
      size_t parent = j.parent;
      while (parent != sIndex && parent != kInvalidIndex) {
        jt.preRotation = skeleton.joints[parent].preRotation * jt.preRotation;
        parent = skeleton.joints[parent].parent;
      }

      // add it
      simplifiedSkeleton.joints.push_back(jt);

      // add jointmap entry
      intermediateJointMap[aIndex] = jointCount;
      jointCount++;
      lastJoint[aIndex] = simplifiedSkeleton.joints.size() - 1;

      // add joint to joint mapping
      simplifiedJointMap[aIndex] = jointCount - 1;
    } else {
      size_t index = aIndex;
      while (index != kInvalidIndex && intermediateJointMap[index] == kInvalidIndex) {
        index = skeleton.joints[index].parent;
      }
      // Find the valid joint above this one in the hierarchy:
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

  // create character result
  CharacterT<T> result(simplifiedSkeleton, simplifiedTransform);

  result.jointMap = simplifiedJointMap;

  // -------------------------------------------------------------------
  //  remap the parameterlimits
  // -------------------------------------------------------------------
  result.parameterLimits = result.remapParameterLimits(parameterLimits, *this);

  // -------------------------------------------------------------------
  //  remap the locators if we have any
  // -------------------------------------------------------------------
  result.locators = result.remapLocators(locators, *this);

  // -------------------------------------------------------------------
  //  remap the mesh and skinning if present
  // -------------------------------------------------------------------

  // create bind states for both source and target skeleton
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
      c.transformation = targetBindState.jointState[c.parent].transformation.inverse() *
          sourceBindState.jointState[oldParent].transformation * c.transformation;
    }
  }

  // return the map
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
      name);
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

  // create bind states for both source and target skeleton
  const SkeletonState sourceBindState(
      originalCharacter.parameterTransform.bindPose(), originalCharacter.skeleton, false);
  const SkeletonState targetBindState(parameterTransform.bindPose(), skeleton, false);

  // go over all vertices
  for (int v = 0; v < result.index.rows(); v++) {
    // remap the parent bones according to the map
    for (int i = 0; i < gsl::narrow_cast<int>(kMaxSkinJoints); i++)
      result.index(v, i) = gsl::narrow<uint32_t>(jointMap[result.index(v, i)]);

    // join together all weights with the same parent
    for (int i = 0; i < gsl::narrow_cast<int>(kMaxSkinJoints); i++) {
      const auto& joint = result.index(v, i);
      for (int j = i + 1; j < gsl::narrow_cast<int>(kMaxSkinJoints); j++) {
        // if we have the same joint multiple times, add things up
        if (result.index(v, j) == joint) {
          result.weight(v, i) += result.weight(v, j);
          result.weight(v, j) = 0.0f;
        }
      }
    }

    // do a clean up pass, sorting things by weight
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
          targetBindState.jointState[targetParent].transformation.inverse();
      const auto sourceTransformation = sourceBindState.jointState[sourceParent].transformation;
      data.offset = targetTransformationInverse * sourceTransformation * data.offset;

      const auto sourceEllipsoidParent = data.ellipsoidParent;
      data.ellipsoidParent = jointMap[data.ellipsoidParent];
      const auto targetEllipsoidInverse =
          targetBindState.jointState[data.ellipsoidParent].transformation.inverse();
      const auto sourceEllipsoid = sourceBindState.jointState[sourceEllipsoidParent].transformation;
      data.ellipsoid = targetEllipsoidInverse * sourceEllipsoid * data.ellipsoid;
      data.ellipsoidInv = data.ellipsoid.inverse();
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

  // create bind states for both source and target skeleton
  const SkeletonState sourceBindState(
      originalCharacter.parameterTransform.bindPose(), originalCharacter.skeleton, false);
  const SkeletonState targetBindState(parameterTransform.bindPose(), skeleton, false);

  // go over each locator and remap it
  for (const auto& sourceLocator : locs) {
    result.emplace_back(sourceLocator);
    auto& loc = result.back();
    loc.parent = jointMap[loc.parent];
    loc.offset = targetBindState.jointState[loc.parent].transformation.inverse() *
        sourceBindState.jointState[sourceLocator.parent].transformation * sourceLocator.offset;
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
  // fill jointMap with identity
  jointMap.resize(skeleton.joints.size());
  std::iota(jointMap.begin(), jointMap.end(), 0);
}

template <typename T>
void CharacterT<T>::initInverseBindPose() {
  // initialize bindpose inverse list
  const SkeletonState bindState(parameterTransform.bindPose(), skeleton, false);
  if (!inverseBindPose.empty()) {
    inverseBindPose.clear();
  }
  inverseBindPose.reserve(bindState.jointState.size());
  for (const auto& t : bindState.jointState) {
    inverseBindPose.push_back(t.transformation.inverse());
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
    BlendShape_p blendShape_in,
    Eigen::Index maxBlendShapes,
    const bool overwriteBaseShape) const {
  CharacterT<T> result = *this;
  result.addBlendShape(blendShape_in, maxBlendShapes, overwriteBaseShape);
  return result;
}

template <typename T>
void CharacterT<T>::addBlendShape(
    const BlendShape_p& blendShape_in,
    Eigen::Index maxBlendShapes,
    const bool overwriteBaseShape) {
  MT_CHECK(this->mesh);
  MT_CHECK(blendShape_in);

  // TODO should we accommodate passing in a BlendShape with "extra" vertices (for the
  // higher subdivision levels)?
  MT_CHECK(
      blendShape_in->modelSize() == this->mesh->vertices.size(),
      "{} is not {}",
      blendShape_in->modelSize(),
      this->mesh->vertices.size());

  if (overwriteBaseShape) {
    // We need to use the base shape from the mesh rather than the one that's stored in the file.
    // TODO (fbogo): Check if this operation is actually meaningful.
    blendShape_in->setBaseShape(this->mesh->vertices);
  }

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
  MT_CHECK(blendShape_in);
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

  // 2. Blend-shape baking (you already had this, just guarded).
  if (bakeBlendShapes && this->blendShape) {
    const BlendWeights blendWeights = extractBlendWeights(result.parameterTransform, modelParams);
    result.mesh->vertices = this->blendShape->template computeShape<float>(blendWeights);
  }

  // 3. Scale (or any joint transform) baking
  if (bakeScales && this->skinWeights) {
    // Evaluate the posed skeleton with modelParams.
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
