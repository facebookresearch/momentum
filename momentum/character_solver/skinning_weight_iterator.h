/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character/skin_weights.h>
#include <momentum/math/fwd.h>

namespace momentum {

template <typename T>
struct BoneWeightT {
  size_t parentBone{};
  float weight{};
  Eigen::Vector3<T> weightedWorldSpacePoint;

  bool operator<(const BoneWeightT<T>& rhs) const {
    return parentBone < rhs.parentBone;
  }

  bool operator>(const BoneWeightT<T>& rhs) const {
    return parentBone > rhs.parentBone;
  }

  BoneWeightT<T>& operator+=(const BoneWeightT<T>& rhs) {
    this->weight += rhs.weight;
    this->weightedWorldSpacePoint += rhs.weightedWorldSpacePoint;
    return *this;
  }
};

// When we compute the derivatives of skinned points, we usually
// end up with a loop like this:
//   for each skinned point i
//     for each skinning bone and weight B_ij, w_ij
//       for each ancestor k
//          compute the weighted change in the skinned point wrt the ancestor
// The redundancy here is that most points are skinned to a set of bones in the
// same hierarchy (for example, two parts of the arm) and so it's redundant to
// walk up the whole hierarchy from scratch for each skinned bone.
//
// This class instead does a single pass up the tree from leaf-most to root-most
// skinned bones, accumulating skinning weights along the way.  By packaging it
// in this class we hide the complexity and simplify the jacobian/gradient
// calculation.
template <typename T>
class SkinningWeightIteratorT {
 public:
  SkinningWeightIteratorT(
      const Character& character,
      const MeshT<T>& restMesh,
      const SkeletonStateT<T>& skelState,
      Eigen::Index vertexIndex);

  SkinningWeightIteratorT(
      const Character& character,
      const SkinnedLocator& locator,
      const Eigen::Vector3<T>& locatorPosition,
      const SkeletonStateT<T>& skelState);

  [[nodiscard]] bool finished() const;

  // Returns the tuple <parent bone index, bone weight, vertex position in world space wrt the
  // current bone>
  std::tuple<size_t, T, Eigen::Vector3<T>> next();

 private:
  void checkInvariants();

  std::array<BoneWeightT<T>, kMaxSkinJoints> boneWeights;
  int nBoneWeights;
  const Character& character;
};

} // namespace momentum
