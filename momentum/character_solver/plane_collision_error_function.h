/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character_solver/fwd.h>
#include <momentum/character_solver/plane_collision_query.h>
#include <momentum/character_solver/skeleton_error_function.h>

#include <span>
#include <vector>

namespace momentum {

/// Penalizes collision between character collision primitives and a fixed plane.
///
/// The plane is represented by `planeNormal.dot(x) - planeOffset = 0`; the non-penetrating side is
/// `planeNormal.dot(x) >= planeOffset`. Capsules, ellipsoids, and boxes use the same collision
/// geometry state and support-radius convention as CollisionErrorFunctionT.
template <typename T>
class PlaneCollisionErrorFunctionT : public SkeletonErrorFunctionT<T> {
 public:
  explicit PlaneCollisionErrorFunctionT(
      const Skeleton& skel,
      const ParameterTransform& pt,
      const CollisionGeometry& cg,
      const Vector3<T>& planeNormal,
      T planeOffset);

  explicit PlaneCollisionErrorFunctionT(
      const Character& character,
      const Vector3<T>& planeNormal,
      T planeOffset);

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

  /// Set the collision plane after normalizing @p planeNormal and the matching offset.
  void setPlane(const Vector3<T>& planeNormal, T planeOffset) {
    collisionQuery_.setPlane(planeNormal, planeOffset);
  }

  /// Return the normalized plane normal used by collision and contact queries.
  [[nodiscard]] const Vector3<T>& getPlaneNormal() const {
    return collisionQuery_.getPlaneNormal();
  }

  /// Return the plane offset after normalization by setPlane().
  [[nodiscard]] T getPlaneOffset() const {
    return collisionQuery_.getPlaneOffset();
  }

  /// Return the reusable query helper used by this error function.
  [[nodiscard]] PlaneCollisionQueryT<T>& getCollisionQuery() {
    return collisionQuery_;
  }

  /// Return the reusable query helper used by this error function.
  [[nodiscard]] const PlaneCollisionQueryT<T>& getCollisionQuery() const {
    return collisionQuery_;
  }

  [[nodiscard]] std::vector<size_t> getCollidingPrimitives() const;

  /// Query detailed collision state for all currently-colliding primitives.
  ///
  /// Uses the collision state from the most recent solver iteration (computed during
  /// getJacobian/getError/getGradient), or the rest pose state immediately after construction.
  [[nodiscard]] std::vector<PlaneCollisionDebugEntryT<T>> getCollisionDebugInfo() const;

  /// Return world-space primitive surface points at or near the plane for an explicit state.
  ///
  /// A primitive contributes when its signed surface distance to the plane is <=
  /// @p contactMargin. The returned point is on the primitive surface, deepest toward the plane.
  [[nodiscard]] std::vector<Vector3<T>> getContactPoints(
      std::span<const JointStateT<T>> states,
      T contactMargin);

  /// Return one world-space primitive surface point per parent joint at or near the plane.
  ///
  /// A primitive contributes when its signed surface distance to the plane is <=
  /// @p contactMargin. When multiple primitives under the same parent joint contribute, the
  /// returned point is the deepest one toward the plane.
  [[nodiscard]] std::vector<Vector3<T>> getContactPointsByParent(
      std::span<const JointStateT<T>> states,
      T contactMargin);

  /// Return one detailed world-space primitive surface point per parent joint at or near the plane.
  ///
  /// A primitive contributes when its signed surface distance to the plane is <=
  /// @p contactMargin. When multiple primitives under the same parent joint contribute, the
  /// returned point is the deepest one toward the plane.
  [[nodiscard]] std::vector<PlaneCollisionContactPointT<T>> getContactPointsByParentWithDetails(
      std::span<const JointStateT<T>> states,
      T contactMargin);

 protected:
  /// Accumulate gradient contributions along a joint chain up to the root.
  ///
  /// `scaleCorrection` accounts for the primitive support radius changing with uniform scale.
  void accumulateGradientAlongChain(
      const SkeletonStateT<T>& state,
      const Vector3<T>& position,
      const Vector3<T>& direction,
      T weight,
      size_t startJoint,
      Ref<VectorX<T>> gradient,
      T scaleCorrection = T(0)) const;

  /// Accumulate Jacobian contributions along a joint chain up to the root.
  ///
  /// `scaleCorrection` accounts for the primitive support radius changing with uniform scale.
  void accumulateJacobianAlongChain(
      const SkeletonStateT<T>& state,
      const Vector3<T>& position,
      const Vector3<T>& direction,
      T factor,
      size_t startJoint,
      Ref<MatrixX<T>> jacobian,
      int row,
      T scaleCorrection = T(0)) const;

  PlaneCollisionQueryT<T> collisionQuery_;

  static constexpr T kCollisionWeight = 5e-3f;
};

} // namespace momentum
