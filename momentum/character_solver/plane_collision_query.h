/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/collision_geometry.h>
#include <momentum/character/collision_geometry_state.h>
#include <momentum/character/fwd.h>
#include <momentum/character/types.h>
#include <momentum/character_solver/fwd.h>
#include <momentum/math/types.h>

#include <cstddef>
#include <span>
#include <vector>

namespace momentum {

/// Per-primitive collision debug information returned by plane-collision queries.
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

/// World-space plane contact point selected for support/contact queries.
template <typename T>
struct PlaneCollisionContactPointT {
  size_t parentJoint = kInvalidIndex;
  T surfaceDistance = T(0);
  Vector3<T> position = Vector3<T>::Zero();
};

using PlaneCollisionContactPoint = PlaneCollisionContactPointT<float>;
using PlaneCollisionContactPointd = PlaneCollisionContactPointT<double>;

/// Stateful query helper for collision primitives against a fixed world plane.
///
/// The plane is represented by `planeNormal.dot(x) - planeOffset = 0`; the non-penetrating side is
/// `planeNormal.dot(x) >= planeOffset`. Capsules, ellipsoids, and boxes use the same collision
/// geometry state and support-radius convention as CollisionErrorFunctionT.
template <typename T>
class PlaneCollisionQueryT {
 public:
  /// Collision information for one primitive selected by the current query state.
  struct Result {
    size_t primitiveIndex = kInvalidIndex;
    size_t parentJoint = kInvalidIndex;
    T signedDistance = T(0);
    T radius = T(0);
    T overlap = T(0);
    Vector3<T> position = Vector3<T>::Zero();

    /// Signed distance from the primitive surface point to the plane.
    [[nodiscard]] T surfaceDistance() const {
      return signedDistance - radius;
    }
  };

  /// Construct a query over explicit collision geometry.
  ///
  /// The primitive list is initialized immediately, but candidate and active collision state remain
  /// empty until update() is called for a skeleton state. Contact-point helpers that take explicit
  /// joint states update that state internally for the duration of the query.
  explicit PlaneCollisionQueryT(
      const CollisionGeometry& cg,
      const Vector3<T>& planeNormal,
      T planeOffset);

  /// Construct a query over a character's collision geometry and initialize it at the zero pose.
  explicit PlaneCollisionQueryT(
      const Character& character,
      const Vector3<T>& planeNormal,
      T planeOffset);

  /// Set the collision plane after normalizing @p planeNormal and the matching offset.
  ///
  /// @p planeOffset must use the same scale as @p planeNormal. Equivalent plane equations such as
  /// `2 * normal.dot(x) - 2 * offset = 0` are stored in one canonical form, so
  /// getPlaneOffset() returns the offset divided by the input normal length.
  void setPlane(const Vector3<T>& planeNormal, T planeOffset);

  /// Return the normalized plane normal used by collision and contact queries.
  [[nodiscard]] const Vector3<T>& getPlaneNormal() const {
    return planeNormal_;
  }

  /// Return the plane offset after normalization by setPlane().
  [[nodiscard]] T getPlaneOffset() const {
    return planeOffset_;
  }

  /// Number of character-attached collision primitives owned by this query.
  [[nodiscard]] size_t primitiveCount() const {
    return primitiveIndices_.size();
  }

  /// Number of primitives retained by the most recent broad-phase update.
  [[nodiscard]] size_t candidatePrimitiveCount() const {
    return candidatePrimitiveIndices_.size();
  }

  /// Update candidate primitive state for a skeleton state and signed plane-distance margin.
  void update(const SkeletonStateT<T>& state, T contactMargin);

  /// Update candidate primitive state for explicit joint states and signed plane-distance margin.
  void update(std::span<const JointStateT<T>> states, T contactMargin);

  /// Select the deepest active collision under each parent joint from current candidates.
  void updateActiveParentCollisions();

  /// Return the current deepest active collisions, one per parent joint.
  [[nodiscard]] const std::vector<Result>& activeParentCollisions() const {
    return activeParentCollisions_;
  }

  /// Return collision primitive indices that penetrate the plane in the current query state.
  [[nodiscard]] std::vector<size_t> getCollidingPrimitives() const;

  /// Query detailed collision state for all currently-colliding primitives.
  [[nodiscard]] std::vector<PlaneCollisionDebugEntryT<T>> getCollisionDebugInfo() const;

  /// Return world-space primitive surface points at or near the plane for explicit joint states.
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

 private:
  void updatePrimitiveList();

  [[nodiscard]] bool mayPrimitiveContactPlane(
      const SkeletonStateT<T>& state,
      size_t primitiveIndex,
      T contactMargin) const;

  [[nodiscard]] T primitiveParentSpaceBound(size_t primitiveIndex) const;

  void validateContactMargin(T contactMargin) const;

  void validateJointStateParents(std::span<const JointStateT<T>> states) const;

  [[nodiscard]] bool checkCollision(size_t primitiveIndex, Result& result) const;

  const CollisionGeometry collisionGeometry_;
  std::vector<size_t> primitiveIndices_;
  std::vector<size_t> candidatePrimitiveIndices_;
  std::vector<T> primitiveParentSpaceBounds_;
  std::vector<Result> activeParentCollisions_;
  std::vector<size_t> parentContactIndices_;
  std::vector<T> parentSurfaceDistances_;
  CollisionGeometryStateT<T> collisionState_;
  Vector3<T> planeNormal_;
  T planeOffset_;
};

} // namespace momentum
