/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/collision_geometry.h>
#include <momentum/character/fwd.h>
#include <momentum/character/skeleton.h>

#include <axel/BoundingBox.h>

namespace momentum {

/// Represents collision geometry states using a Structure of Arrays (SoA) format.
template <typename T>
struct CollisionGeometryStateT {
  // ASCII Art to visualize the capsule:
  //
  //           ^
  //           :
  //           :________________________________________
  //          /:                                        :\
  //        /  :                                        : \
  //       |   :(origin)                                :  |
  //       |   O-------------(direction)--------------->:--|------------------> X-axis
  //       |   :                                        :  |
  //        \  : (radius[0])                            : /  (radius[1])
  //          \:________________________________________:/
  //
  //
  // Note: radius[0] and radius[1] can be different.

  /// Capsule's origin in global coordinates.
  std::vector<Vector3<T>> origin;

  /// Capsule's direction vector (representing the X-axis) in global coordinates.
  std::vector<Vector3<T>> direction;

  /// Scaled radii of the capsule's endpoints.
  std::vector<Vector2<T>> radius;

  /// Signed difference between the capsule's radii (radius[1] - radius[0]).
  std::vector<T> delta;

  /// Updates the state based on a given skeleton state and collision geometry.
  void update(const SkeletonStateT<T>& skeletonState, const CollisionGeometry& collisionGeometry);
};

/// Determines if two tapered capsules overlap.
///
/// @param originA Origin of the first capsule
/// @param directionA Direction vector of the first capsule (length encodes the segment length)
/// @param radiiA Radii at the endpoints of the first capsule
/// @param deltaA Signed difference between radii of the first capsule (radiiA[1] - radiiA[0])
/// @param originB Origin of the second capsule
/// @param directionB Direction vector of the second capsule (length encodes the segment length)
/// @param radiiB Radii at the endpoints of the second capsule
/// @param deltaB Signed difference between radii of the second capsule (radiiB[1] - radiiB[0])
/// @param outDistance Output: distance between the closest points on the two centerline
///   segments. Only written when the closest-point computation succeeds; left unchanged
///   when the function returns false due to an early exit from `closestPointsOnSegments`.
/// @param outClosestPoints Output: parametric positions in [0, 1] along each segment of
///   the closest points (outClosestPoints[0] for capsule A, outClosestPoints[1] for B).
/// @param outOverlap Output: amount of overlap (positive if overlapping). Only written
///   when the closest-point computation succeeds.
/// @return True if the capsules overlap and the closest points are not coincident
///   (separation >= ~1e-8 for float, ~1e-17 for double).
template <typename T>
bool overlaps(
    const Vector3<T>& originA,
    const Vector3<T>& directionA,
    const Vector2<T>& radiiA,
    T deltaA,
    const Vector3<T>& originB,
    const Vector3<T>& directionB,
    const Vector2<T>& radiiB,
    T deltaB,
    T& outDistance,
    Vector2<T>& outClosestPoints,
    T& outOverlap) {
  // Use the sum of max radii as a proximity cutoff so closestPointsOnSegments can early-out
  // when segments are guaranteed too far apart for the capsules to overlap.
  const T maxRadiiSum = radiiA.maxCoeff() + radiiB.maxCoeff();

  auto [success, closestDist, closestPoints] =
      closestPointsOnSegments<T>(originA, directionA, originB, directionB, maxRadiiSum);

  if (!success) {
    return false;
  }

  outClosestPoints = closestPoints;

  // Linearly interpolate each tapered radius at the closest-point parameter along its segment,
  // then sum to get the combined radius the gap must clear for the capsules to be disjoint.
  const T radiusAtClosestPoints =
      radiiA[0] + closestPoints[0] * deltaA + radiiB[0] + closestPoints[1] * deltaB;

  outOverlap = radiusAtClosestPoints - closestDist;
  outDistance = closestDist;

  // Reject coincident centerlines (closestDist below Eps) to avoid degenerate contact normals
  // downstream when the two segments are effectively the same line.
  return (outOverlap > T(0)) && (closestDist >= Eps<T>(1e-8, 1e-17));
}

/// Updates an axis-aligned bounding box to encompass a tapered capsule.
///
/// @param aabb The bounding box to update
/// @param originA Origin of the capsule
/// @param direction Direction vector of the capsule
/// @param radii Radii at the endpoints of the capsule
template <typename T>
void updateAabb(
    axel::BoundingBox<T>& aabb,
    const Vector3<T>& originA,
    const Vector3<T>& direction,
    const Vector2<T>& radii) {
  const Vector3<T> radius0 = Vector3<T>::Constant(radii[0]);
  const Vector3<T> radius1 = Vector3<T>::Constant(radii[1]);
  const Vector3<T> originB = originA + direction;

  const Vector3<T> minA = originA - radius0;
  const Vector3<T> maxA = originA + radius0;
  const Vector3<T> minB = originB - radius1;
  const Vector3<T> maxB = originB + radius1;

  aabb.aabb.min() = minA.cwiseMin(minB);
  aabb.aabb.max() = maxA.cwiseMax(maxB);
}

/// Determines whether a pair of collision geometry capsules should be included as a valid
/// collision pair.
///
/// A pair is excluded if:
/// - The capsules overlap in rest pose (permanently overlapping, e.g. adjacent body parts),
///   unless @p filterRestPoseOverlaps is false
/// - The capsules are attached to the same joint or directly adjacent joints
///
/// @param collisionState The collision geometry state in rest pose
/// @param collisionGeometry The collision geometry definition
/// @param skeleton The skeleton used for adjacency checks
/// @param i Index of the first capsule
/// @param j Index of the second capsule
/// @param filterRestPoseOverlaps When true, exclude pairs that overlap in rest pose
/// @return True if the pair is a valid collision pair (should be checked for collisions)
template <typename T, typename S>
[[nodiscard]] bool isValidCollisionPair(
    const CollisionGeometryStateT<T>& collisionState,
    const CollisionGeometry& collisionGeometry,
    const SkeletonT<S>& skeleton,
    size_t i,
    size_t j,
    bool filterRestPoseOverlaps = true) {
  const size_t p0 = collisionGeometry[i].parent;
  const size_t p1 = collisionGeometry[j].parent;

  // World-fixed capsules (kInvalidIndex parent):
  // - If both are world-fixed, they can't collide with each other in a meaningful way.
  // - If exactly one is world-fixed, it's always a valid collision pair (no rest-pose
  //   overlap or adjacency checks apply since the world geometry is independent of
  //   the character's skeleton).
  if (p0 == kInvalidIndex || p1 == kInvalidIndex) {
    return p0 != p1;
  }

  // Check for same or adjacent joints
  if (skeleton.isSameOrAdjacentJoints(p0, p1)) {
    return false;
  }

  if (!filterRestPoseOverlaps) {
    return true;
  }

  // Check for rest-pose overlap (permanently overlapping pairs like adjacent body parts)
  T distance;
  Vector2<T> cp;
  T overlap;
  return !overlaps(
      collisionState.origin[i],
      collisionState.direction[i],
      collisionState.radius[i],
      collisionState.delta[i],
      collisionState.origin[j],
      collisionState.direction[j],
      collisionState.radius[j],
      collisionState.delta[j],
      distance,
      cp,
      overlap);
}

} // namespace momentum
