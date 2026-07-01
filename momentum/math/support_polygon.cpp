/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/math/support_polygon.h"

#include "momentum/common/exception.h"
#include "momentum/math/utility.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace momentum {

namespace {

template <typename T>
Vector3<T> fallbackSupportPlaneAxis(const Vector3<T>& normal) {
  Vector3<T> axis = Vector3<T>::UnitX();
  T axisAlignment = std::abs(normal.x());
  if (const T yAlignment = std::abs(normal.y()); yAlignment < axisAlignment) {
    axis = Vector3<T>::UnitY();
    axisAlignment = yAlignment;
  }
  if (std::abs(normal.z()) < axisAlignment) {
    axis = Vector3<T>::UnitZ();
  }
  return (axis - normal * axis.dot(normal)).normalized();
}

bool supportPointFinite(const Vector2f& point) {
  return std::isfinite(point.x()) && std::isfinite(point.y());
}

bool supportPointLess(const Vector2f& lhs, const Vector2f& rhs) {
  if (lhs.x() == rhs.x()) {
    return lhs.y() < rhs.y();
  }
  return lhs.x() < rhs.x();
}

bool supportPointEqual(const Vector2f& lhs, const Vector2f& rhs) {
  return lhs.x() == rhs.x() && lhs.y() == rhs.y();
}

void appendSupportHullPoint(std::vector<Vector2f>& hull, const Vector2f& point) {
  while (hull.size() >= 2) {
    if (cross2d(hull.at(hull.size() - 2), hull.back(), point) > 0.0f) {
      break;
    }
    hull.pop_back();
  }
  hull.push_back(point);
}

void removeLastSupportHullPoint(std::vector<Vector2f>& hull) {
  if (!hull.empty()) {
    hull.pop_back();
  }
}

} // namespace

template <typename T>
SupportPlaneT<T>::SupportPlaneT(
    const Vector3<T>& planeNormal,
    const T planeOffset,
    const Vector3<T>& uAxisHint) {
  MT_THROW_IF(!planeNormal.allFinite(), "Support plane normal must contain only finite values");
  MT_THROW_IF(!std::isfinite(planeOffset), "Support plane offset must be finite");
  const T normalNorm = planeNormal.norm();
  MT_THROW_IF(normalNorm <= Eps<T>(1e-8f, 1e-17), "Support plane normal must be non-zero");

  normal = planeNormal / normalNorm;
  offset = planeOffset / normalNorm;

  MT_THROW_IF(!uAxisHint.allFinite(), "Support plane axis hint must contain only finite values");
  Vector3<T> tangent = uAxisHint - normal * uAxisHint.dot(normal);
  const T tangentNorm = tangent.norm();
  if (tangentNorm > Eps<T>(1e-8f, 1e-17)) {
    uAxis = tangent / tangentNorm;
  } else {
    uAxis = fallbackSupportPlaneAxis(normal);
  }
  vAxis = uAxis.cross(normal).normalized();
}

template <typename T>
Vector3<T> SupportPlaneT<T>::origin() const {
  return offset * normal;
}

template <typename T>
T SupportPlaneT<T>::signedDistance(const Vector3<T>& point) const {
  return normal.dot(point) - offset;
}

template <typename T>
Vector3<T> SupportPlaneT<T>::projectPoint(const Vector3<T>& point) const {
  return point - signedDistance(point) * normal;
}

template <typename T>
Vector2<T> SupportPlaneT<T>::coordinates(const Vector3<T>& point) const {
  const Vector3<T> fromOrigin = projectPoint(point) - origin();
  return Vector2<T>(fromOrigin.dot(uAxis), fromOrigin.dot(vAxis));
}

template <typename T>
Vector3<T> SupportPlaneT<T>::pointFromCoordinates(const Vector2<T>& point) const {
  return origin() + point.x() * uAxis + point.y() * vAxis;
}

float cross2d(const Vector2f& origin, const Vector2f& a, const Vector2f& b) {
  const Vector2f originToA = a - origin;
  const Vector2f originToB = b - origin;
  return originToA.x() * originToB.y() - originToA.y() * originToB.x();
}

std::vector<Vector2f> computeConvexHull2d(const std::span<const Vector2f> points) {
  std::vector<Vector2f> sortedPoints;
  sortedPoints.reserve(points.size());
  for (const Vector2f& point : points) {
    MT_THROW_IF(
        !supportPointFinite(point), "Support polygon point must contain only finite values");
    sortedPoints.push_back(point);
  }

  std::ranges::sort(sortedPoints, supportPointLess);
  sortedPoints.erase(
      std::unique(sortedPoints.begin(), sortedPoints.end(), supportPointEqual), sortedPoints.end());
  if (sortedPoints.size() <= 2) {
    return sortedPoints;
  }

  std::vector<Vector2f> lower;
  lower.reserve(sortedPoints.size());
  for (const Vector2f& point : sortedPoints) {
    appendSupportHullPoint(lower, point);
  }

  std::vector<Vector2f> upper;
  upper.reserve(sortedPoints.size());
  for (size_t i = sortedPoints.size(); i-- > 0;) {
    appendSupportHullPoint(upper, sortedPoints.at(i));
  }

  removeLastSupportHullPoint(lower);
  removeLastSupportHullPoint(upper);
  lower.insert(lower.end(), upper.begin(), upper.end());
  return lower;
}

std::vector<Vector2f> computeSupportPolygonFromWorldPoints(const std::span<const Vector3f> points) {
  return computeSupportPolygonFromWorldPoints(points, SupportPlane());
}

std::vector<Vector2f> computeSupportPolygonFromWorldPoints(
    const std::span<const Vector3f> points,
    const SupportPlane& supportPlane) {
  std::vector<Vector2f> projectedPoints;
  projectedPoints.reserve(points.size());
  for (const Vector3f& point : points) {
    MT_THROW_IF(!point.allFinite(), "Support point must contain only finite values");
    projectedPoints.push_back(supportPlane.coordinates(point));
  }
  return computeConvexHull2d(projectedPoints);
}

template struct SupportPlaneT<float>;
template struct SupportPlaneT<double>;

} // namespace momentum
