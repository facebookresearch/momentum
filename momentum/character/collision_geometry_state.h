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

#include <algorithm>

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

  /// Primitive origin/center in global coordinates.
  std::vector<Vector3<T>> origin;

  /// Capsule's direction vector (representing the X-axis) in global coordinates.
  std::vector<Vector3<T>> direction;

  /// Scaled radii of the capsule's endpoints.
  std::vector<Vector2<T>> radius;

  /// Signed difference between the capsule's radii (radius[1] - radius[0]).
  std::vector<T> delta;

  /// Primitive types matching the source collision geometry.
  std::vector<CollisionPrimitiveType> type;

  /// Primitive orientation in global coordinates.
  std::vector<Quaternion<T>> orientation;

  /// Scaled ellipsoid radii in global coordinates.
  std::vector<Vector3<T>> ellipsoidRadii;

  /// Scaled box half extents in global coordinates.
  std::vector<Vector3<T>> boxHalfExtents;

  /// Updates the state based on a given skeleton state and collision geometry.
  void update(const SkeletonStateT<T>& skeletonState, const CollisionGeometry& collisionGeometry);
};

/// Narrow-phase collision information for a pair of primitives.
template <typename T>
struct CollisionOverlapResultT {
  /// Distance between the primitive center features used by the narrow phase.
  T distance = T(0);

  /// Parametric closest-point positions. Capsule entries use [0, 1]; centered primitives use 0.
  Vector2<T> closestPoints = Vector2<T>::Zero();

  /// Amount of overlap, positive when colliding.
  T overlap = T(0);

  /// Center feature position for the first primitive.
  Vector3<T> positionA = Vector3<T>::Zero();

  /// Center feature position for the second primitive.
  Vector3<T> positionB = Vector3<T>::Zero();

  /// Effective radius of the first primitive along the contact direction.
  T radiusA = T(0);

  /// Effective radius of the second primitive along the contact direction.
  T radiusB = T(0);
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

/// Return the clamped segment parameter for the closest point to a world-space point.
template <typename T>
[[nodiscard]] T closestPointOnSegmentParameter(
    const Vector3<T>& origin,
    const Vector3<T>& direction,
    const Vector3<T>& point) {
  const T directionNormSq = direction.squaredNorm();
  if (directionNormSq <= Eps<T>(1e-8, 1e-17)) {
    return T(0);
  }

  return std::clamp((point - origin).dot(direction) / directionNormSq, T(0), T(1));
}

/// Return the support radius of an oriented ellipsoid along a unit world-space direction.
template <typename T>
[[nodiscard]] T ellipsoidRadiusAlongDirection(
    const Quaternion<T>& orientation,
    const Vector3<T>& radii,
    const Vector3<T>& unitDirection) {
  const Vector3<T> localDirection = orientation.inverse() * unitDirection;
  return radii.cwiseAbs().cwiseProduct(localDirection).norm();
}

/// Return the support radius of an oriented box along a unit world-space direction.
template <typename T>
[[nodiscard]] T boxRadiusAlongDirection(
    const Quaternion<T>& orientation,
    const Vector3<T>& halfExtents,
    const Vector3<T>& unitDirection) {
  const Vector3<T> localDirection = orientation.inverse() * unitDirection;
  return halfExtents.cwiseAbs().dot(localDirection.cwiseAbs());
}

/// Return whether the primitive is represented by a center, orientation, and support radius.
[[nodiscard]] inline bool isCenteredPrimitive(CollisionPrimitiveType type) {
  return type == CollisionPrimitiveType::Ellipsoid || type == CollisionPrimitiveType::Box;
}

/// Return the support radius of an ellipsoid or box along a unit world-space direction.
template <typename T>
[[nodiscard]] T centeredPrimitiveRadiusAlongDirection(
    CollisionPrimitiveType type,
    const Quaternion<T>& orientation,
    const Vector3<T>& ellipsoidRadii,
    const Vector3<T>& boxHalfExtents,
    const Vector3<T>& unitDirection) {
  if (type == CollisionPrimitiveType::Ellipsoid) {
    return ellipsoidRadiusAlongDirection(orientation, ellipsoidRadii, unitDirection);
  }
  if (type == CollisionPrimitiveType::Box) {
    return boxRadiusAlongDirection(orientation, boxHalfExtents, unitDirection);
  }
  return T(0);
}

/// Test overlap between a tapered capsule and an oriented centered primitive.
///
/// The centered primitive can be an ellipsoid or box. The function fills the
/// closest capsule parameter, support radii, distance, and signed overlap used
/// by the solver when a non-degenerate overlap is found.
///
/// @param capsuleOrigin Origin of the capsule segment in world space.
/// @param capsuleDirection Direction vector from capsule start to end in world space.
/// @param capsuleRadii Radii at the capsule start and end points.
/// @param capsuleDelta Signed difference between capsule radii (`capsuleRadii[1] -
///   capsuleRadii[0]`).
/// @param primitiveType Centered primitive type; must be ellipsoid or box.
/// @param primitiveCenter Primitive center in world space.
/// @param primitiveOrientation Primitive orientation in world space.
/// @param ellipsoidRadii Ellipsoid radii along its local axes; ignored for boxes.
/// @param boxHalfExtents Box half extents along its local axes; ignored for ellipsoids.
/// @param outDistance Output: distance between the capsule centerline point and primitive center.
/// @param outCapsuleParam Output: parametric position in [0, 1] along the capsule segment.
/// @param outOverlap Output: signed overlap amount, positive when overlapping.
/// @param outCapsuleRadius Output: capsule support radius at `outCapsuleParam`.
/// @param outPrimitiveRadius Output: primitive support radius along the contact direction.
/// @return True when the capsule and centered primitive overlap with a non-degenerate contact
///   direction.
template <typename T>
[[nodiscard]] bool overlapsCapsuleCenteredPrimitive(
    const Vector3<T>& capsuleOrigin,
    const Vector3<T>& capsuleDirection,
    const Vector2<T>& capsuleRadii,
    T capsuleDelta,
    CollisionPrimitiveType primitiveType,
    const Vector3<T>& primitiveCenter,
    const Quaternion<T>& primitiveOrientation,
    const Vector3<T>& ellipsoidRadii,
    const Vector3<T>& boxHalfExtents,
    T& outDistance,
    T& outCapsuleParam,
    T& outOverlap,
    T& outCapsuleRadius,
    T& outPrimitiveRadius) {
  outCapsuleParam =
      closestPointOnSegmentParameter(capsuleOrigin, capsuleDirection, primitiveCenter);
  const Vector3<T> capsulePoint = capsuleOrigin + capsuleDirection * outCapsuleParam;
  const Vector3<T> separation = capsulePoint - primitiveCenter;
  outDistance = separation.norm();
  if (outDistance < Eps<T>(1e-8, 1e-17)) {
    return false;
  }

  const Vector3<T> normal = separation / outDistance;
  outCapsuleRadius = capsuleRadii[0] + outCapsuleParam * capsuleDelta;
  outPrimitiveRadius = centeredPrimitiveRadiusAlongDirection(
      primitiveType, primitiveOrientation, ellipsoidRadii, boxHalfExtents, normal);
  outOverlap = outCapsuleRadius + outPrimitiveRadius - outDistance;
  return outOverlap > T(0);
}

/// Determines whether two collision primitives overlap.
///
/// Capsule-capsule pairs use the existing closest-segment query. Centered primitive pairs use a
/// support radius along the center-feature separation direction, which gives a stable contact
/// normal for the solver and preserves the same overlap convention used by tapered capsules.
///
/// @param collisionState World-space collision state for `collisionGeometry`.
/// @param collisionGeometry Collision primitives corresponding to `collisionState`.
/// @param indexA First primitive index.
/// @param indexB Second primitive index.
/// @param result Output: closest points, contact positions, support radii, distance, and overlap.
/// @return True when the primitive pair overlaps with a non-degenerate contact direction.
template <typename T>
[[nodiscard]] bool overlaps(
    const CollisionGeometryStateT<T>& collisionState,
    const CollisionGeometry& collisionGeometry,
    size_t indexA,
    size_t indexB,
    CollisionOverlapResultT<T>& result) {
  const auto typeA = collisionGeometry[indexA].type;
  const auto typeB = collisionGeometry[indexB].type;

  if (typeA == CollisionPrimitiveType::TaperedCapsule &&
      typeB == CollisionPrimitiveType::TaperedCapsule) {
    if (!overlaps(
            collisionState.origin[indexA],
            collisionState.direction[indexA],
            collisionState.radius[indexA],
            collisionState.delta[indexA],
            collisionState.origin[indexB],
            collisionState.direction[indexB],
            collisionState.radius[indexB],
            collisionState.delta[indexB],
            result.distance,
            result.closestPoints,
            result.overlap)) {
      return false;
    }

    result.positionA =
        collisionState.origin[indexA] + collisionState.direction[indexA] * result.closestPoints[0];
    result.positionB =
        collisionState.origin[indexB] + collisionState.direction[indexB] * result.closestPoints[1];
    result.radiusA =
        collisionState.radius[indexA][0] + result.closestPoints[0] * collisionState.delta[indexA];
    result.radiusB =
        collisionState.radius[indexB][0] + result.closestPoints[1] * collisionState.delta[indexB];
    return true;
  }

  if (typeA == CollisionPrimitiveType::TaperedCapsule && isCenteredPrimitive(typeB)) {
    if (!overlapsCapsuleCenteredPrimitive(
            collisionState.origin[indexA],
            collisionState.direction[indexA],
            collisionState.radius[indexA],
            collisionState.delta[indexA],
            typeB,
            collisionState.origin[indexB],
            collisionState.orientation[indexB],
            collisionState.ellipsoidRadii[indexB],
            collisionState.boxHalfExtents[indexB],
            result.distance,
            result.closestPoints[0],
            result.overlap,
            result.radiusA,
            result.radiusB)) {
      return false;
    }

    result.closestPoints[1] = T(0);
    result.positionA =
        collisionState.origin[indexA] + collisionState.direction[indexA] * result.closestPoints[0];
    result.positionB = collisionState.origin[indexB];
    return true;
  }

  if (isCenteredPrimitive(typeA) && typeB == CollisionPrimitiveType::TaperedCapsule) {
    if (!overlapsCapsuleCenteredPrimitive(
            collisionState.origin[indexB],
            collisionState.direction[indexB],
            collisionState.radius[indexB],
            collisionState.delta[indexB],
            typeA,
            collisionState.origin[indexA],
            collisionState.orientation[indexA],
            collisionState.ellipsoidRadii[indexA],
            collisionState.boxHalfExtents[indexA],
            result.distance,
            result.closestPoints[1],
            result.overlap,
            result.radiusB,
            result.radiusA)) {
      return false;
    }

    result.closestPoints[0] = T(0);
    result.positionA = collisionState.origin[indexA];
    result.positionB =
        collisionState.origin[indexB] + collisionState.direction[indexB] * result.closestPoints[1];
    return true;
  }

  if (isCenteredPrimitive(typeA) && isCenteredPrimitive(typeB)) {
    const Vector3<T> separation = collisionState.origin[indexA] - collisionState.origin[indexB];
    result.distance = separation.norm();
    if (result.distance < Eps<T>(1e-8, 1e-17)) {
      return false;
    }

    const Vector3<T> normal = separation / result.distance;
    result.radiusA = centeredPrimitiveRadiusAlongDirection(
        typeA,
        collisionState.orientation[indexA],
        collisionState.ellipsoidRadii[indexA],
        collisionState.boxHalfExtents[indexA],
        normal);
    result.radiusB = centeredPrimitiveRadiusAlongDirection(
        typeB,
        collisionState.orientation[indexB],
        collisionState.ellipsoidRadii[indexB],
        collisionState.boxHalfExtents[indexB],
        normal);
    result.overlap = result.radiusA + result.radiusB - result.distance;
    result.positionA = collisionState.origin[indexA];
    result.positionB = collisionState.origin[indexB];
    result.closestPoints = Vector2<T>::Zero();
    return result.overlap > T(0);
  }

  return false;
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

/// Updates an axis-aligned bounding box to encompass an oriented ellipsoid.
///
/// Uses the tight ellipsoid AABB half-extent h_i = sqrt(sum_j (R_ij * radii_j)^2),
/// i.e. the L2 norm of each rotation-matrix row scaled component-wise by the radii.
/// This is the exact AABB of the ellipsoid, unlike the looser L1/OBB bound
/// (sum_j |R_ij| * radii_j) which over-estimates by up to a factor of sqrt(3).
///
/// @param aabb The bounding box to update
/// @param center Center of the ellipsoid
/// @param orientation Orientation of the ellipsoid
/// @param radii Radii along the ellipsoid's local axes
template <typename T>
void updateEllipsoidAabb(
    axel::BoundingBox<T>& aabb,
    const Vector3<T>& center,
    const Quaternion<T>& orientation,
    const Vector3<T>& radii) {
  const Eigen::Matrix<T, 3, 3> rotation = orientation.toRotationMatrix();
  const Vector3<T> absRadii = radii.cwiseAbs();
  Vector3<T> halfExtents;
  for (int i = 0; i < 3; ++i) {
    halfExtents[i] = rotation.row(i).cwiseProduct(absRadii.transpose()).norm();
  }
  aabb.aabb.min() = center - halfExtents;
  aabb.aabb.max() = center + halfExtents;
}

/// Updates an axis-aligned bounding box to encompass an oriented box.
///
/// Uses the exact OBB half-extent h_i = sum_j |R_ij| * halfExtents_j (L1 over the
/// rotated extents). For a box this is the tight, correct AABB; do NOT switch it to
/// the ellipsoid L2 row-norm, which would under-bound the box and miss collisions.
///
/// @param aabb The bounding box to update
/// @param center Center of the box
/// @param orientation Orientation of the box
/// @param halfExtents Half extents along the box's local axes
template <typename T>
void updateBoxAabb(
    axel::BoundingBox<T>& aabb,
    const Vector3<T>& center,
    const Quaternion<T>& orientation,
    const Vector3<T>& halfExtents) {
  const Eigen::Matrix<T, 3, 3> absRotation = orientation.toRotationMatrix().cwiseAbs();
  const Vector3<T> aabbHalfExtents = absRotation * halfExtents.cwiseAbs();
  aabb.aabb.min() = center - aabbHalfExtents;
  aabb.aabb.max() = center + aabbHalfExtents;
}

/// Updates an axis-aligned bounding box to encompass a collision primitive.
template <typename T>
void updateAabb(
    axel::BoundingBox<T>& aabb,
    const CollisionGeometryStateT<T>& collisionState,
    size_t index) {
  if (collisionState.type[index] == CollisionPrimitiveType::TaperedCapsule) {
    updateAabb(
        aabb,
        collisionState.origin[index],
        collisionState.direction[index],
        collisionState.radius[index]);
    return;
  }

  if (collisionState.type[index] == CollisionPrimitiveType::Ellipsoid) {
    updateEllipsoidAabb(
        aabb,
        collisionState.origin[index],
        collisionState.orientation[index],
        collisionState.ellipsoidRadii[index]);
    return;
  }

  if (collisionState.type[index] == CollisionPrimitiveType::Box) {
    updateBoxAabb(
        aabb,
        collisionState.origin[index],
        collisionState.orientation[index],
        collisionState.boxHalfExtents[index]);
  }
}

/// Determines whether a pair of collision geometry primitives should be included as a valid
/// collision pair.
///
/// A pair is excluded if:
/// - The primitives overlap in rest pose (permanently overlapping, e.g. adjacent body parts),
///   unless @p filterRestPoseOverlaps is false
/// - The primitives are attached to the same joint or directly adjacent joints
///
/// @param collisionState The collision geometry state in rest pose
/// @param collisionGeometry The collision geometry definition
/// @param skeleton The skeleton used for adjacency checks
/// @param i Index of the first primitive
/// @param j Index of the second primitive
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

  // World-fixed primitives (kInvalidIndex parent):
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
  CollisionOverlapResultT<T> overlap;
  return !overlaps(collisionState, collisionGeometry, i, j, overlap);
}

} // namespace momentum
