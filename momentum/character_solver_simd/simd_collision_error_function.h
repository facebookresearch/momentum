/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/collision_geometry.h>
#include <momentum/character/collision_geometry_state.h>
#include <momentum/character_solver/fwd.h>
#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/common/aligned.h>
#include <momentum/simd/simd.h>

#include <axel/BoundingBox.h>

#include <vector>

namespace momentum {

/// Represents an error function that penalizes self-intersections between collision geometries of
/// the character.
///
/// The function pre-filters collision geometries to ignore those attached to the same joint, those
/// intersecting in the rest pose, or those very close in the rest pose.
template <typename T>
class SimdCollisionErrorFunctionT : public SkeletonErrorFunctionT<T> {
 public:
  explicit SimdCollisionErrorFunctionT(
      const Skeleton& skel,
      const ParameterTransform& pt,
      const CollisionGeometry& collisionGeometry);

  explicit SimdCollisionErrorFunctionT(const Character& character);

  [[nodiscard]] double getError(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& /* meshState */) final;

  double getGradient(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& /* meshState */,
      Ref<VectorX<T>> gradient) override;

  double getJacobian(
      const ModelParametersT<T>& /*unused*/,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& /* meshState */,
      Ref<MatrixX<T>> jacobian,
      Ref<VectorX<T>> residual,
      int& usedRows) override;

  [[nodiscard]] size_t getJacobianSize() const final;

 protected:
  void updateCollisionPairs();

  /// Update collisionState_, aabbs_, and collisionPairs_ given the new skeleton state.
  void computeBroadPhase(const SkeletonStateT<T>& state);

  /// Pre-computed valid collision pair with common ancestor.
  struct CollisionPairInfo {
    size_t indexA;
    size_t indexB;
    size_t commonAncestor;
  };

  const CollisionGeometry collisionGeometry_;

  /// Pre-filtered list of valid collision pairs.
  std::vector<CollisionPairInfo> validPairs_;

  /// Per-capsule list of candidate collision partner indices (populated each frame by broadphase).
  std::vector<std::vector<int, AlignedAllocator<int, momentum::kSimdAlignment>>> collisionPairs_;

  /// Pre-computed common ancestor lookup table indexed by [indexA * numCapsules + indexB].
  std::vector<size_t> commonAncestors_;

  /// Per-capsule AABBs, updated each frame for broadphase culling.
  std::vector<axel::BoundingBox<T>> aabbs_;

  CollisionGeometryStateT<T> collisionState_;

  // weights for the error functions
  static constexpr T kCollisionWeight = 5e-3f;
};

} // namespace momentum
