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
                               : nullptr} {
  // Empty
}

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

  // create skeleton state first
  JointParameters jointParams = referenceCharacter.parameterTransform.apply(parameters);
  if (applyLimits) {
    jointParams = applyPassiveJointParameterLimits(referenceCharacter.parameterLimits, jointParams);
  }
  skeletonState.set(jointParams, referenceCharacter.skeleton);

  // create locator state
  locatorState.update(skeletonState, referenceCharacter.locators);

  // create skinning if we have a mesh
  if (updateMesh && referenceCharacter.mesh && referenceCharacter.skinWeights) {
    // check if we need to create a mesh
    if (!meshState || meshState->vertices.size() != referenceCharacter.mesh->vertices.size()) {
      meshState = std::make_unique<Mesh>(*referenceCharacter.mesh);
    }

    // check if we have pose blendshapes and use them before skinning if there
    if (referenceCharacter.poseShapes) {
      // calculate pose shapes
      const auto vs = referenceCharacter.poseShapes->compute(skeletonState);

      // apply skinning
      meshState->vertices = applySSD(
          referenceCharacter.inverseBindPose, *referenceCharacter.skinWeights, vs, skeletonState);

      // update normals
      meshState->updateNormals();
    } else if (referenceCharacter.blendShape) {
      // For CharacterT<float>, we can cast directly to Character
      // For CharacterT<double>, we need to cast to Character (CharacterT<float>) first
      if constexpr (std::is_same_v<T, float>) {
        skinWithBlendShapes(
            static_cast<const Character&>(referenceCharacter),
            skeletonState,
            parameters.pose,
            *meshState);
      } else {
        // For double, create a temporary Character (CharacterT<float>) object
        // This is a simplified conversion that may not preserve all properties
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
            nullptr, // No face expression blend shape
            referenceCharacter.name,
            referenceCharacter.inverseBindPose);

        skinWithBlendShapes(tempCharacter, skeletonState, parameters.pose, *meshState);
      }
    } else {
      // apply simple skinning
      applySSD(
          referenceCharacter.inverseBindPose,
          *referenceCharacter.skinWeights,
          *referenceCharacter.mesh,
          skeletonState,
          *meshState);
    }
  }

  // update collision geometry if present
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
