/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <Eigen/Geometry>

#include "axel/common/Constants.h"
#include "axel/math/RayTriangleIntersection.h"

#include <cmath>

namespace axel {

template <typename T>
bool rayTriangleIntersect(
    const Eigen::Vector3<T>& beginRay,
    const Eigen::Vector3<T>& direction,
    const Eigen::Vector3<T>& p0,
    const Eigen::Vector3<T>& p1,
    const Eigen::Vector3<T>& p2,
    Eigen::Vector3<T>& intersectionPoint,
    T& tOut,
    T& u,
    T& v) {
  constexpr T kEpsilon = detail::eps<T>();
  const Eigen::Vector3<T> v0v1 = p1 - p0;
  const Eigen::Vector3<T> v0v2 = p2 - p0;
  const Eigen::Vector3<T> pvec = direction.cross(v0v2);
  const T det = v0v1.dot(pvec);
  const T detScale = v0v1.norm() * direction.norm() * v0v2.norm();
  if (!(detScale > T(0)) || !std::isfinite(detScale) || !std::isfinite(det) ||
      std::abs(det) <= kEpsilon * detScale) {
    return false;
  }

  const T invDet = T(1) / det;

  const Eigen::Vector3<T> tvec = beginRay - p0;
  u = tvec.dot(pvec) * invDet;
  if (!std::isfinite(u) || u < -kEpsilon || u > T(1) + kEpsilon) {
    return false;
  }

  const Eigen::Vector3<T> qvec = tvec.cross(v0v1);
  v = direction.dot(qvec) * invDet;
  if (!std::isfinite(v) || v < -kEpsilon || u + v > T(1) + kEpsilon) {
    return false;
  }

  tOut = v0v2.dot(qvec) * invDet;
  if (!std::isfinite(tOut)) {
    return false;
  }
  intersectionPoint = beginRay + tOut * direction;
  return intersectionPoint.allFinite();
}

template bool rayTriangleIntersect(
    const Eigen::Vector3<float>& beginRay,
    const Eigen::Vector3<float>& direction,
    const Eigen::Vector3<float>& p0,
    const Eigen::Vector3<float>& p1,
    const Eigen::Vector3<float>& p2,
    Eigen::Vector3<float>& intersectionPoint,
    float& tOut,
    float& u,
    float& v);
template bool rayTriangleIntersect(
    const Eigen::Vector3<double>& beginRay,
    const Eigen::Vector3<double>& direction,
    const Eigen::Vector3<double>& p0,
    const Eigen::Vector3<double>& p1,
    const Eigen::Vector3<double>& p2,
    Eigen::Vector3<double>& intersectionPoint,
    double& tOut,
    double& u,
    double& v);

template <typename T>
bool rayTriangleIntersect(
    const Eigen::Vector3<T>& beginRay,
    const Eigen::Vector3<T>& direction,
    const Eigen::Vector3<T>& p0,
    const Eigen::Vector3<T>& p1,
    const Eigen::Vector3<T>& p2,
    Eigen::Vector3<T>& intersectionPoint,
    T& tOut) {
  T u = 0.0;
  T v = 0.0;
  return rayTriangleIntersect<T>(beginRay, direction, p0, p1, p2, intersectionPoint, tOut, u, v);
}

template bool rayTriangleIntersect(
    const Eigen::Vector3<float>& beginRay,
    const Eigen::Vector3<float>& direction,
    const Eigen::Vector3<float>& p0,
    const Eigen::Vector3<float>& p1,
    const Eigen::Vector3<float>& p2,
    Eigen::Vector3<float>& intersectionPoint,
    float& tOut);
template bool rayTriangleIntersect(
    const Eigen::Vector3<double>& beginRay,
    const Eigen::Vector3<double>& direction,
    const Eigen::Vector3<double>& p0,
    const Eigen::Vector3<double>& p1,
    const Eigen::Vector3<double>& p2,
    Eigen::Vector3<double>& intersectionPoint,
    double& tOut);

} // namespace axel
