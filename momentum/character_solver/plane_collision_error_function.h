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

#include <span>
#include <vector>

namespace momentum {

/// Per-primitive collision debug information returned by getCollisionDebugInfo().
template <typename T>
struct PlaneCollisionDebugEntryT {
  size_t primitiveIndex;
  size_t parentJoint;
  T overlap;
  T signedDistance;
  T radius;
  Vector3<T> position;
};

using PlaneCollisionDebugEntry = PlaneCollisionDebugEntryT<float>;
using PlaneCollisionDebugEntryd = PlaneCollisionDebugEntryT<double>;

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
      const Vector3<T>& planeNormal = Vector3<T>::UnitY(),
      T planeOffset = T(0));

  explicit PlaneCollisionErrorFunctionT(
      const Character& character,
      const Vector3<T>& planeNormal = Vector3<T>::UnitY(),
      T planeOffset = T(0));

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

  void setPlane(const Vector3<T>& planeNormal, T planeOffset);

  [[nodiscard]] const Vector3<T>& getPlaneNormal() const {
    return planeNormal_;
  }

  [[nodiscard]] T getPlaneOffset() const {
    return planeOffset_;
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

 protected:
  struct PlaneCollisionResult {
    T signedDistance = T(0);
    T radius = T(0);
    T overlap = T(0);
    Vector3<T> position = Vector3<T>::Zero();
  };

  struct ActivePlaneCollision {
    size_t primitiveIndex = kInvalidIndex;
    PlaneCollisionResult result;
  };

  void updatePrimitiveList();

  void updateCandidateCollisionState(const SkeletonStateT<T>& state, T contactMargin);

  [[nodiscard]] bool mayPrimitiveContactPlane(
      const SkeletonStateT<T>& state,
      size_t primitiveIndex,
      T contactMargin) const;

  [[nodiscard]] T primitiveParentSpaceBound(size_t primitiveIndex) const;

  void prepareContactPointQuery(std::span<const JointStateT<T>> states, T contactMargin);

  [[nodiscard]] bool checkCollision(size_t primitiveIndex, PlaneCollisionResult& result) const;

  void updateActiveParentCollisions();

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

  const CollisionGeometry collisionGeometry_;
  std::vector<size_t> primitiveIndices_;
  std::vector<size_t> candidatePrimitiveIndices_;
  std::vector<T> primitiveParentSpaceBounds_;
  std::vector<ActivePlaneCollision> activeParentCollisions_;
  CollisionGeometryStateT<T> collisionState_;
  Vector3<T> planeNormal_;
  T planeOffset_;

  static constexpr T kCollisionWeight = 5e-3f;
};

} // namespace momentum
