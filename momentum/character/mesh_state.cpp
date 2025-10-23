/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character/mesh_state.h"

#include "momentum/character/blend_shape.h"
#include "momentum/character/blend_shape_base.h"
#include "momentum/character/blend_shape_skinning.h"
#include "momentum/character/character.h"
#include "momentum/character/linear_skinning.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/common/profile.h"

namespace momentum {

template <typename T>
MeshStateT<T>::MeshStateT(
    const ModelParametersT<T>& parameters,
    const SkeletonStateT<T>& state,
    const Character& character) noexcept {
  update(parameters, state, character);
}

template <typename T>
MeshStateT<T>::~MeshStateT() noexcept = default;

template <typename T>
MeshStateT<T>::MeshStateT(const MeshStateT& other) {
  if (other.neutralMesh_) {
    neutralMesh_ = std::make_unique<MeshT<T>>(*other.neutralMesh_);
  }
  if (other.restMesh_) {
    restMesh_ = std::make_unique<MeshT<T>>(*other.restMesh_);
  }
  if (other.posedMesh_) {
    posedMesh_ = std::make_unique<MeshT<T>>(*other.posedMesh_);
  }
}

template <typename T>
MeshStateT<T>& MeshStateT<T>::operator=(const MeshStateT& other) {
  if (this != &other) {
    *this = MeshStateT(other);
  }
  return *this;
}

template <typename T>
void MeshStateT<T>::update(
    const ModelParametersT<T>& parameters,
    const SkeletonStateT<T>& state,
    const Character& character) {
  MT_PROFILE_FUNCTION();

  // Initialize mesh objects on first call if character has mesh data
  if (!restMesh_ && character.mesh) {
    neutralMesh_ = std::make_unique<MeshT<T>>(character.mesh->template cast<T>());
    restMesh_ = std::make_unique<MeshT<T>>(character.mesh->template cast<T>());
    posedMesh_ = std::make_unique<MeshT<T>>(character.mesh->template cast<T>());
  }

  if (!restMesh_ || !posedMesh_) {
    return; // No mesh to update
  }

  // Update rest mesh with blend shapes and face expressions
  bool doUpdateNormals = false;

  // Apply blend shapes if available
  if (character.blendShape) {
    const BlendWeightsT<T> blendWeights =
        extractBlendWeights(character.parameterTransform, parameters);
    character.blendShape->computeShape(blendWeights, restMesh_->vertices);
    doUpdateNormals = true;
  }

  // Apply face expression blend shapes if available
  if (character.faceExpressionBlendShape) {
    if (!character.blendShape) {
      // Set restMesh back to neutral, removing potential previous expressions.
      // Note that if the character comes with (shape) blendShape, the previous if block already
      // takes care of this step.
      Eigen::Map<Eigen::VectorX<T>> outputVec(
          &restMesh_->vertices[0][0], restMesh_->vertices.size() * 3);
      const Eigen::Map<Eigen::VectorX<T>> baseVec(
          &neutralMesh_->vertices[0][0], neutralMesh_->vertices.size() * 3);
      outputVec = baseVec.template cast<T>();
    }
    const BlendWeightsT<T> faceExpressionBlendWeights =
        extractFaceExpressionBlendWeights(character.parameterTransform, parameters);
    character.faceExpressionBlendShape->applyDeltas(
        faceExpressionBlendWeights, restMesh_->vertices);
    doUpdateNormals = true;
  }

  // Update normals if blend shapes were applied
  if (doUpdateNormals) {
    restMesh_->updateNormals();
  }

  // Apply skinning to get the posed mesh
  applySSD(
      cast<T>(character.inverseBindPose), *character.skinWeights, *restMesh_, state, *posedMesh_);
}

template struct MeshStateT<float>;
template struct MeshStateT<double>;

} // namespace momentum
