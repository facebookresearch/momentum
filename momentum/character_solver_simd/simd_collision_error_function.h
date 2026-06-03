/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/collision_geometry.h>
#include <momentum/character/collision_geometry_state.h>
#include <momentum/character_solver/collision_error_function.h>
#include <momentum/character_solver/fwd.h>
#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/common/aligned.h>
#include <momentum/simd/simd.h>

#include <axel/BoundingBox.h>

#include <optional>
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
      const MeshStateT<T>& meshState) final;

  double getGradient(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Ref<VectorX<T>> gradient) override;

  double getJacobian(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Ref<MatrixX<T>> jacobian,
      Ref<VectorX<T>> residual,
      int& usedRows) override;

  [[nodiscard]] size_t getJacobianSize() const final;

 protected:
  void updateCollisionPairs();

  /// Update collisionState_, aabbs_, and collisionPairs_ given the new skeleton state.
  void computeBroadPhase(const SkeletonStateT<T>& state);

  /// Synchronize scalar fallback settings before delegating non-capsule collision work.
  CollisionErrorFunctionT<T>& syncScalarFallback();

  /// Accumulate gradient contributions along a joint chain from startJoint up to (but not
  /// including) stopJoint. The sign of weight controls the push direction.
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

  /// Accumulate the shared scale-gradient contribution above a collision pair's common ancestor.
  void accumulateCommonAncestorScaleGradient(
      size_t commonAncestor,
      T scaleGradient,
      Ref<VectorX<T>> gradient) const;

  /// Accumulate Jacobian contributions along a joint chain from startJoint up to (but not
  /// including) stopJoint. The sign of factor controls the push direction.
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

  /// Accumulate the shared scale-Jacobian entry above a collision pair's common ancestor.
  void accumulateCommonAncestorScaleJacobian(
      size_t commonAncestor,
      T scaleJacobian,
      int row,
      Ref<MatrixX<T>> jacobian) const;

  /// Pre-computed valid collision pair with common ancestor.
  struct CollisionPairInfo {
    size_t indexA;
    size_t indexB;
    size_t commonAncestor;
  };

  const CollisionGeometry collisionGeometry_;

  /// Scalar collision error function used when any primitive is not a tapered capsule.
  /// Constructed once because @c collisionGeometry_ is immutable; runtime solver settings are
  /// synchronized before each delegated call.
  std::optional<CollisionErrorFunctionT<T>> scalarFallback_;

  /// Pre-filtered list of valid collision pairs.
  std::vector<CollisionPairInfo> validPairs_;

  /// Per-capsule list of candidate collision partner indices (populated each frame by broadphase).
  std::vector<std::vector<int, AlignedAllocator<int, momentum::kSimdAlignment>>> collisionPairs_;

  /// Pre-computed common ancestor lookup table indexed by [indexA * numCapsules + indexB].
  std::vector<size_t> commonAncestors_;

  /// Per-capsule AABBs, updated each frame for broadphase culling.
  std::vector<axel::BoundingBox<T>> aabbs_;

  CollisionGeometryStateT<T> collisionState_;

  /// True when the collision geometry contains a non-tapered-capsule primitive, in which case
  /// getError/getGradient/getJacobian delegate to @c scalarFallback_ instead of the SIMD path.
  bool useScalarFallback_ = false;

  // weights for the error functions
  static constexpr T kCollisionWeight = 5e-3f;
};

} // namespace momentum
