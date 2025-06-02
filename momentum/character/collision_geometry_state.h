/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/collision_geometry.h>
#include <momentum/character/fwd.h>

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
/// @param directionA Direction vector of the first capsule
/// @param radiiA Radii at the endpoints of the first capsule
/// @param deltaA Difference between radii of the first capsule
/// @param originB Origin of the second capsule
/// @param directionB Direction vector of the second capsule
/// @param radiiB Radii at the endpoints of the second capsule
/// @param deltaB Difference between radii of the second capsule
/// @param outDistance Output parameter for the distance between the closest points
/// (normalized). No modification on return false.
/// @param outOverlap Output parameter for the overlap amount (positive if overlapping)
/// @return True if the capsules overlap, false otherwise
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
  // Sum of the maximum radii of the tapered capsules
  const T maxRadiiSum = radiiA.maxCoeff() + radiiB.maxCoeff();

  // Determine the closest points on the segments of the tapered capsules
  auto [success, closestDist, closestPoints] =
      closestPointsOnSegments<T>(originA, directionA, originB, directionB, maxRadiiSum);

  if (!success) {
    return false;
  }

  // Store the closest points to the output argument
  outClosestPoints = closestPoints;

  // Calculate the radii at the closest points
  const T radiusAtClosestPoints =
      radiiA[0] + closestPoints[0] * deltaA + radiiB[0] + closestPoints[1] * deltaB;

  // Determine the overlap and distance between the closest points
  outOverlap = radiusAtClosestPoints - closestDist;
  outDistance = closestDist;

  // Check for overlap and sufficient proximity
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

} // namespace momentum
