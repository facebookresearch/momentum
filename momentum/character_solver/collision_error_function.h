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

#include <axel/BoundingBox.h>

#include <vector>

namespace momentum {

template <typename T>
class CollisionErrorFunctionT : public SkeletonErrorFunctionT<T> {
 public:
  explicit CollisionErrorFunctionT(
      const Skeleton& skel,
      const ParameterTransform& pt,
      const CollisionGeometry& cg);

  explicit CollisionErrorFunctionT(const Character& character);

  [[nodiscard]] double getError(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState) final;

  double getGradient(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Ref<VectorX<T>> gradient) override;

  double getJacobian(
      const ModelParametersT<T>& /*unused*/,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Ref<MatrixX<T>> jacobian,
      Ref<VectorX<T>> residual,
      int& usedRows) override;

  [[nodiscard]] size_t getJacobianSize() const final;

  [[nodiscard]] std::vector<Vector2i> getCollisionPairs() const;

 protected:
  void updateCollisionPairs();

  /// Update collisionState_ and aabbs_ given the new skeleton state.
  void computeBroadPhase(const SkeletonStateT<T>& state);

  /// AABB broadphase pre-check followed by narrow-phase capsule overlap test.
  /// Returns true if the pair collides, filling distance, cp, and overlap.
  [[nodiscard]] bool checkCollision(
      const CollisionGeometryStateT<T>& colState,
      size_t indexA,
      size_t indexB,
      T& distance,
      Vector2<T>& cp,
      T& overlap) const;

  /// Accumulate gradient contributions along a joint chain from startJoint up to (but not
  /// including) stopJoint. The sign of weight controls the push direction.
  void accumulateGradientAlongChain(
      const SkeletonStateT<T>& state,
      const Vector3<T>& position,
      const Vector3<T>& direction,
      T weight,
      size_t startJoint,
      size_t stopJoint,
      Ref<VectorX<T>> gradient) const;

  /// Accumulate Jacobian contributions along a joint chain from startJoint up to (but not
  /// including) stopJoint. The sign of factor controls the push direction.
  void accumulateJacobianAlongChain(
      const SkeletonStateT<T>& state,
      const Vector3<T>& position,
      const Vector3<T>& direction,
      T factor,
      size_t startJoint,
      size_t stopJoint,
      Ref<MatrixX<T>> jacobian,
      int row) const;

  /// Pre-computed valid collision pair with common ancestor.
  struct CollisionPairInfo {
    size_t indexA;
    size_t indexB;
    size_t commonAncestor;
  };

  const CollisionGeometry collisionGeometry_;

  /// Pre-filtered list of valid collision pairs (excludes rest-pose overlaps and adjacent joints).
  std::vector<CollisionPairInfo> validPairs_;

  /// Per-capsule AABBs, updated each frame for broadphase culling.
  std::vector<axel::BoundingBox<T>> aabbs_;

  CollisionGeometryStateT<T> collisionState_;

  // weights for the error functions
  static constexpr T kCollisionWeight = 5e-3f;
};

} // namespace momentum
