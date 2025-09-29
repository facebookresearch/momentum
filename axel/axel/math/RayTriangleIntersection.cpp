/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <Eigen/Geometry>

#include "axel/common/Constants.h"
#include "axel/math/RayTriangleIntersection.h"

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

  // ray and triangle are parallel if direction is parallel to v0v2, which is indicated by pvec
  // being the zero vector; adding this check to avoid having a non-zero determinant due to
  // normalization of an almost-zero pvec
  if (pvec.norm() < kEpsilon) {
    return false;
  }

  const T det = v0v1.dot(pvec);
  // #ifdef CULLING
  //  if the determinant is negative the triangle is backfacing
  //  if the determinant is close to 0, the ray misses the triangle
  //  if (det < kEpsilon) { return false; }

  // ray and triangle are parallel if det is close to 0
  // calculate determinant with normalized vectors to check for parallelism
  // invDet could become quite large - consider using det and lower threshold (maybe 1e-12)
  const T det_norm = (v0v1.normalized()).dot(pvec.normalized());
  if (std::abs(det_norm) < kEpsilon) {
    return false;
  }

  const T invDet = 1 / det;

  const Eigen::Vector3<T> tvec = beginRay - p0;
  u = tvec.dot(pvec) * invDet;
  if (u < -kEpsilon || u > 1 + kEpsilon) {
    return false;
  }

  const Eigen::Vector3<T> qvec = tvec.cross(v0v1);
  v = direction.dot(qvec) * invDet;
  if (v < -kEpsilon || u + v > 1 + kEpsilon) {
    return false;
  }

  tOut = v0v2.dot(qvec) * invDet;
  intersectionPoint = beginRay + tOut * direction;
  return true;
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
