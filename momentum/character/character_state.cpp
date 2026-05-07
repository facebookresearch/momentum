/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character/character_state.h"

#include "momentum/character/blend_shape_skinning.h"
#include "momentum/character/collision_geometry_state.h"
#include "momentum/character/linear_skinning.h"
#include "momentum/character/locator_state.h"
#include "momentum/character/pose_shape.h"
#include "momentum/common/checks.h"
#include "momentum/math/mesh.h"

namespace momentum {

template <typename T>
CharacterStateT<T>::CharacterStateT() = default;

template <typename T>
CharacterStateT<T>::CharacterStateT(const CharacterStateT& other)
    : parameters{other.parameters},
      skeletonState{other.skeletonState},
      locatorState{other.locatorState},
      meshState{other.meshState ? std::make_unique<Mesh>(*other.meshState) : nullptr},
      collisionState{
          other.collisionState ? std::make_unique<CollisionGeometryState>(*other.collisionState)
                               : nullptr} {}

template <typename T>
CharacterStateT<T>::CharacterStateT(CharacterStateT&& c) noexcept = default;

template <typename T>
CharacterStateT<T>& CharacterStateT<T>::operator=(CharacterStateT&& rhs) noexcept = default;

template <typename T>
CharacterStateT<T>::~CharacterStateT() = default;

template <typename T>
CharacterStateT<T>::CharacterStateT(
    const CharacterT<T>& referenceCharacter,
    bool updateMesh,
    bool updateCollision) {
  setBindPose(referenceCharacter, updateMesh, updateCollision);
}

template <typename T>
CharacterStateT<T>::CharacterStateT(
    const CharacterParameters& parameters,
    const CharacterT<T>& referenceCharacter,
    bool updateMesh,
    bool updateCollision,
    bool applyLimits) {
  set(parameters, referenceCharacter, updateMesh, updateCollision, applyLimits);
}

template <typename T>
void CharacterStateT<T>::setBindPose(
    const CharacterT<T>& referenceCharacter,
    bool updateMesh,
    bool updateCollision) {
  set(referenceCharacter.bindPose(), referenceCharacter, updateMesh, updateCollision);
}

template <typename T>
void CharacterStateT<T>::set(
    const CharacterParameters& inParameters,
    const CharacterT<T>& referenceCharacter,
    bool updateMesh,
    bool updateCollision,
    bool applyLimits) {
  parameters = inParameters;
  if (parameters.offsets.size() == 0) {
    parameters.offsets = VectorXf::Zero(referenceCharacter.parameterTransform.numJointParameters());
  }
  MT_CHECK(
      parameters.pose.size() == referenceCharacter.parameterTransform.numAllModelParameters(),
      "{} is not {}",
      parameters.pose.size(),
      referenceCharacter.parameterTransform.numAllModelParameters());
  MT_CHECK(
      parameters.offsets.size() == referenceCharacter.parameterTransform.numJointParameters(),
      "{} is not {}",
      parameters.offsets.size(),
      referenceCharacter.parameterTransform.numJointParameters());

  // Skeleton state must be computed first — locator, mesh, and collision updates below
  // all read from it.
  JointParameters jointParams = referenceCharacter.parameterTransform.apply(parameters);
  if (applyLimits) {
    jointParams = applyPassiveJointParameterLimits(referenceCharacter.parameterLimits, jointParams);
  }
  skeletonState.set(jointParams, referenceCharacter.skeleton);

  locatorState.update(skeletonState, referenceCharacter.locators);

  if (updateMesh && referenceCharacter.mesh && referenceCharacter.skinWeights) {
    // (Re)allocate the cached mesh whenever vertex count diverges from the reference
    // — covers both first call and switching to a different reference character.
    if (!meshState || meshState->vertices.size() != referenceCharacter.mesh->vertices.size()) {
      meshState = std::make_unique<Mesh>(*referenceCharacter.mesh);
    }

    // Pose blendshapes take precedence: corrective vertex offsets driven by joint pose are
    // applied before linear skinning so the SSD pass deforms the corrected bind-pose vertices.
    if (referenceCharacter.poseShapes) {
      const auto vs = referenceCharacter.poseShapes->compute(skeletonState);

      meshState->vertices = applySSD(
          referenceCharacter.inverseBindPose, *referenceCharacter.skinWeights, vs, skeletonState);

      meshState->updateNormals();
    } else if (referenceCharacter.blendShape) {
      // skinWithBlendShapes() is only defined for `Character` (i.e., CharacterT<float>), so the
      // double-precision instantiation must materialize a float Character to call it. Every
      // CharacterT<T> member is itself non-templated (skeleton, parameterTransform, mesh,
      // skinWeights, inverseBindPose are all stored as float types regardless of T), so the
      // temporary built below is bit-equivalent to the source — no float<->double conversion is
      // needed. The cost is the per-call deep copies the constructor performs for `mesh`,
      // `skinWeights`, `collision`, and `poseShapes`; that overhead could be avoided by
      // templating `skinWithBlendShapes` on the character type if it ever shows up in profiles.
      if constexpr (std::is_same_v<T, float>) {
        skinWithBlendShapes(
            static_cast<const Character&>(referenceCharacter),
            skeletonState,
            parameters.pose,
            *meshState);
      } else {
        Character tempCharacter(
            referenceCharacter.skeleton,
            referenceCharacter.parameterTransform,
            referenceCharacter.parameterLimits,
            referenceCharacter.locators,
            referenceCharacter.mesh.get(),
            referenceCharacter.skinWeights.get(),
            referenceCharacter.collision.get(),
            referenceCharacter.poseShapes.get(),
            referenceCharacter.blendShape,
            nullptr, // no face expression blend shape
            referenceCharacter.name,
            referenceCharacter.inverseBindPose);

        skinWithBlendShapes(tempCharacter, skeletonState, parameters.pose, *meshState);
      }
    } else {
      applySSD(
          referenceCharacter.inverseBindPose,
          *referenceCharacter.skinWeights,
          *referenceCharacter.mesh,
          skeletonState,
          *meshState);
    }
  }

  if (updateCollision && referenceCharacter.collision) {
    if (!collisionState || collisionState->origin.size() != referenceCharacter.collision->size()) {
      collisionState = std::make_unique<CollisionGeometryState>();
    }
    collisionState->update(skeletonState, *referenceCharacter.collision);
  }
}

template struct CharacterStateT<float>;
template struct CharacterStateT<double>;

} // namespace momentum
