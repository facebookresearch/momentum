/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/math/types.h>

#include <span>
#include <vector>

namespace momentum {

/// Oriented plane used to classify support contacts and project support polygons.
///
/// The default is Momentum's Y-up ground plane. Its in-plane basis intentionally preserves the
/// existing world-XZ support-polygon coordinates: @ref uAxis is +X and @ref vAxis is +Z. That
/// makes the default basis left-handed, so callers should treat (u, v) as projection coordinates,
/// not as a right-handed frame. Points with signed distance <= the caller's margin are on the
/// contact side of the plane.
template <typename T>
struct SupportPlaneT {
  /// Unit plane normal. Points on the non-penetrating side have non-negative signed distance.
  Vector3<T> normal = Vector3<T>::UnitY();

  /// Plane offset in `normal.dot(point) - offset = 0`.
  T offset = T(0);

  /// First unit in-plane basis axis used for 2D support-polygon coordinates.
  Vector3<T> uAxis = Vector3<T>::UnitX();

  /// Second unit in-plane basis axis. Chosen to preserve the default Y-up world-XZ coordinates.
  Vector3<T> vAxis = Vector3<T>::UnitZ();

  /// Construct an oriented support plane and in-plane coordinate basis.
  ///
  /// @param planeNormal Plane normal before normalization.
  /// @param planeOffset Plane offset matching @p planeNormal; normalized with the normal.
  /// @param uAxisHint Preferred first in-plane axis. If it is collinear with @p planeNormal, a
  /// stable perpendicular axis is chosen instead.
  explicit SupportPlaneT(
      const Vector3<T>& planeNormal = Vector3<T>::UnitY(),
      T planeOffset = T(0),
      const Vector3<T>& uAxisHint = Vector3<T>::UnitX());

  /// Return one world-space point on the plane, equal to `normal * offset`.
  [[nodiscard]] Vector3<T> origin() const;

  /// Return `normal.dot(point) - offset`; negative values are below/behind the plane.
  [[nodiscard]] T signedDistance(const Vector3<T>& point) const;

  /// Return @p point projected orthogonally onto the plane.
  [[nodiscard]] Vector3<T> projectPoint(const Vector3<T>& point) const;

  /// Return 2D support-plane coordinates of @p point after projection.
  [[nodiscard]] Vector2<T> coordinates(const Vector3<T>& point) const;

  /// Return the world-space point with the given 2D support-plane coordinates.
  [[nodiscard]] Vector3<T> pointFromCoordinates(const Vector2<T>& point) const;
};

using SupportPlane = SupportPlaneT<float>;
using SupportPlaned = SupportPlaneT<double>;

/// Compute the signed 2D cross product of vectors from an origin.
///
/// @param origin Shared origin for both vectors.
/// @param a End point of the first vector.
/// @param b End point of the second vector.
/// @return Positive when `origin -> a -> b` turns counter-clockwise.
[[nodiscard]] float cross2d(const Vector2f& origin, const Vector2f& a, const Vector2f& b);

/// Compute the convex hull of 2D support points.
///
/// Duplicate points and collinear interior boundary points are omitted.
/// Degenerate inputs return zero, one, or two points.
///
/// @param points Support points in the floor plane.
/// @return Convex hull vertices in counter-clockwise order.
[[nodiscard]] std::vector<Vector2f> computeConvexHull2d(std::span<const Vector2f> points);

/// Compute the 2D support polygon from world-space support points.
///
/// Support points are projected onto @p supportPlane before hull construction.
///
/// @param points World-space contact/support points.
/// @param supportPlane Plane and in-plane basis used for projection.
/// @return Convex hull vertices in support-plane coordinates.
[[nodiscard]] std::vector<Vector2f> computeSupportPolygonFromWorldPoints(
    std::span<const Vector3f> points,
    const SupportPlane& supportPlane);

/// Compute the 2D support polygon from world-space support points on the Y-up ground plane.
[[nodiscard]] std::vector<Vector2f> computeSupportPolygonFromWorldPoints(
    std::span<const Vector3f> points);

} // namespace momentum
