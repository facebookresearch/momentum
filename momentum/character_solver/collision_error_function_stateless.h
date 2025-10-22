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

  const CollisionGeometry collisionGeometry;

  // [0] and [1] index the collision geometry, [2] indexes the joint after
  // which we can stop comparing as it's parent to both
  std::vector<Vector3i> collisionPairs;

  VectorX<bool> collisionActive;

  CollisionGeometryStateT<T> collisionState;

  // weights for the error functions
  static constexpr T kCollisionWeight = 5e-3f;
};

} // namespace momentum
