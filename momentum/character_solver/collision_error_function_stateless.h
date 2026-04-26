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

/// NOTE: This is a resurrected version of the `CollisionErrorFunction` (previously updated by
/// D50818775). Its stateless nature makes it ideal for testing in multi-threaded settings.
///
/// Represents an error function that penalizes self-intersections between collision geometries of
/// the character.
///
/// The function pre-filters collision geometries to ignore those attached to the same joint, those
/// intersecting in the rest pose, or those very close in the rest pose.
template <typename T>
class CollisionErrorFunctionStatelessT : public SkeletonErrorFunctionT<T> {
 public:
  explicit CollisionErrorFunctionStatelessT(
      const Skeleton& skel,
      const ParameterTransform& pt,
      const CollisionGeometry& cg);

  explicit CollisionErrorFunctionStatelessT(const Character& character);

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

  /// Update collisionState and aabbs_ given the new skeleton state.
  void computeBroadPhase(const SkeletonStateT<T>& state);

  /// AABB broadphase pre-check followed by narrow-phase capsule overlap test.
  [[nodiscard]] bool checkCollision(
      const CollisionGeometryStateT<T>& colState,
      size_t indexA,
      size_t indexB,
      T& distance,
      Vector2<T>& cp,
      T& overlap) const;

  /// Accumulate gradient contributions along a joint chain.
  /// scaleCorrection is an additive term for the scale derivative (for capsule radius correction).
  void accumulateGradientAlongChain(
      const SkeletonStateT<T>& state,
      const Vector3<T>& position,
      const Vector3<T>& direction,
      T weight,
      size_t startJoint,
      size_t stopJoint,
      Ref<VectorX<T>> gradient,
      T scaleCorrection = T(0)) const;

  /// Accumulate Jacobian contributions along a joint chain.
  /// scaleCorrection is an additive term for the scale derivative (for capsule radius correction).
  void accumulateJacobianAlongChain(
      const SkeletonStateT<T>& state,
      const Vector3<T>& position,
      const Vector3<T>& direction,
      T factor,
      size_t startJoint,
      size_t stopJoint,
      Ref<MatrixX<T>> jacobian,
      int row,
      T scaleCorrection = T(0)) const;

  /// Pre-computed valid collision pair with common ancestor.
  struct CollisionPairInfo {
    size_t indexA;
    size_t indexB;
    size_t commonAncestor;
  };

  const CollisionGeometry collisionGeometry;

  /// Pre-filtered list of valid collision pairs.
  std::vector<CollisionPairInfo> validPairs_;

  /// Per-capsule AABBs, updated each frame for broadphase culling.
  std::vector<axel::BoundingBox<T>> aabbs_;

  CollisionGeometryStateT<T> collisionState;

  // weights for the error functions
  static constexpr T kCollisionWeight = 5e-3f;
};

} // namespace momentum
