/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/skinning_weight_iterator.h"

#include <momentum/character/skinned_locator.h>
#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character/skin_weights.h"
#include "momentum/common/checks.h"
#include "momentum/math/mesh.h"

namespace momentum {

template <typename T>
SkinningWeightIteratorT<T>::SkinningWeightIteratorT(
    const Character& character,
    const MeshT<T>& restMesh,
    const SkeletonStateT<T>& skelState,
    Eigen::Index vertexIndex)
    : character(character) {
  const auto& skinWeights = *character.skinWeights;
  nBoneWeights = 0;
  {
    for (uint32_t i = 0; i < kMaxSkinJoints; ++i) {
      const auto w = skinWeights.weight(vertexIndex, i);
      const auto parentBone = skinWeights.index(vertexIndex, i);
      if (w > 0) {
        if (parentBone < character.inverseBindPose.size() &&
            parentBone < skelState.jointState.size()) {
          boneWeights[nBoneWeights++] = {
              parentBone,
              w,
              T(w) *
                  (skelState.jointState[parentBone].transform *
                   (character.inverseBindPose[parentBone].template cast<T>() *
                    restMesh.vertices[vertexIndex]))};
        }
      }
    }
    std::sort(
        boneWeights.begin(), boneWeights.begin() + nBoneWeights, std::greater<BoneWeightT<T>>());
  }
  checkInvariants();
}

template <typename T>
SkinningWeightIteratorT<T>::SkinningWeightIteratorT(
    const Character& character,
    const SkinnedLocator& locator,
    const Eigen::Vector3<T>& locatorPosition,
    const SkeletonStateT<T>& skelState)
    : character(character) {
  nBoneWeights = 0;
  for (uint32_t i = 0; i < kMaxSkinJoints; ++i) {
    const auto w = locator.skinWeights[i];
    const auto parentBone = locator.parents[i];
    if (w > 0 && parentBone < character.inverseBindPose.size() &&
        parentBone < skelState.jointState.size()) {
      boneWeights[nBoneWeights++] = {
          parentBone,
          w,
          T(w) *
              (skelState.jointState[parentBone].transform *
               (character.inverseBindPose[parentBone].template cast<T>() *
                locatorPosition.template cast<T>()))};
    }
  }
  std::sort(
      boneWeights.begin(), boneWeights.begin() + nBoneWeights, std::greater<BoneWeightT<T>>());
}

template <typename T>
bool SkinningWeightIteratorT<T>::finished() const {
  return nBoneWeights == 0;
}

// Returns the tuple <parent bone index, bone weight, vertex position in world space wrt the
// current bone>
template <typename T>
std::tuple<size_t, T, Eigen::Vector3<T>> SkinningWeightIteratorT<T>::next() {
  MT_CHECK(nBoneWeights != 0);
  const BoneWeightT<T> result = boneWeights[0];

  boneWeights[0].parentBone = this->character.skeleton.joints[result.parentBone].parent;
  if (boneWeights[0].parentBone == kInvalidIndex) {
    // Reached the root, so we're done with this one:
    for (int i = 1; i < nBoneWeights; ++i) {
      boneWeights[i - 1] = boneWeights[i];
    }
    --nBoneWeights;
  } else {
    // We decreased the bone index by moving up to its parent; now figure out
    // where the bone weight should be relocated in the list.
    int i = 1;
    while (i < nBoneWeights && boneWeights[i - 1].parentBone < boneWeights[i].parentBone) {
      std::swap(boneWeights[i - 1], boneWeights[i]);
      ++i;
    }

    if (i < nBoneWeights && boneWeights[i - 1].parentBone == boneWeights[i].parentBone) {
      // Merge them:
      boneWeights[i - 1] += boneWeights[i];

      // strip out the duplicate:
      ++i;
      while (i < nBoneWeights) {
        boneWeights[i - 1] = boneWeights[i];
        ++i;
      }
      --nBoneWeights;
    }
  }

  checkInvariants();
  return {result.parentBone, result.weight, result.weightedWorldSpacePoint / result.weight};
}

template <typename T>
void SkinningWeightIteratorT<T>::checkInvariants() {
#ifndef NDEBUG
  if (nBoneWeights > 0) {
    // Check the invariants:
    float sum = 0;
    for (int i = 0; i < nBoneWeights; ++i) {
      MT_CHECK(boneWeights[i].parentBone < this->character.skeleton.joints.size());
      MT_CHECK(boneWeights[i].weight > 0 && boneWeights[i].weight <= 1.001);
      MT_CHECK(i == 0 || boneWeights[i - 1].parentBone > boneWeights[i].parentBone);
      sum += boneWeights[i].weight;
    }
    MT_CHECK(sum > 0.99f && sum < 1.01f);
  }
#endif
}

template class SkinningWeightIteratorT<float>;
template class SkinningWeightIteratorT<double>;

} // namespace momentum
