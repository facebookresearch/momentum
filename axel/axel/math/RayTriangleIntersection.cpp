/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <Eigen/Geometry>

#include "axel/common/Constants.h"
#include "axel/math/RayTriangleIntersection.h"

#include <drjit/array.h>
#include <drjit/packet.h>

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

template <typename T, size_t LaneWidth>
WideMask<WideScalar<T, LaneWidth>> rayTriangleIntersectWide(
    const Eigen::Vector3<T>& beginRay,
    const Eigen::Vector3<T>& direction,
    const WideVec3<T, LaneWidth>& p0,
    const WideVec3<T, LaneWidth>& p1,
    const WideVec3<T, LaneWidth>& p2,
    WideVec3<T, LaneWidth>& intersectionPoint,
    WideScalar<T, LaneWidth>& tOut,
    WideScalar<T, LaneWidth>& u,
    WideScalar<T, LaneWidth>& v) {
  using WideScalarT = WideScalar<T, LaneWidth>;
  using WideVec3T = WideVec3<T, LaneWidth>;
  using WideMaskT = WideMask<WideScalarT>;

  constexpr T kEpsilon = detail::eps<T>();

  // Convert ray to SIMD format
  const WideVec3T rayOrigin{
      WideScalarT(beginRay.x()), WideScalarT(beginRay.y()), WideScalarT(beginRay.z())};
  const WideVec3T rayDirection{
      WideScalarT(direction.x()), WideScalarT(direction.y()), WideScalarT(direction.z())};

  // Compute triangle edges (same as scalar version)
  const WideVec3T v0v1 = p1 - p0;
  const WideVec3T v0v2 = p2 - p0;

  // Cross product: direction × v0v2
  const WideVec3T pvec = drjit::cross(rayDirection, v0v2);

  // Check for parallel ray and triangle (same logic as scalar version)
  const WideScalarT pvecNorm = drjit::norm(pvec);
  const WideMaskT notParallel = pvecNorm >= WideScalarT(kEpsilon);

  // Early exit if all triangles are parallel
  if (!drjit::any(notParallel)) {
    u = WideScalarT(0);
    v = WideScalarT(0);
    tOut = WideScalarT(std::numeric_limits<T>::max());
    intersectionPoint = rayOrigin;
    return WideMaskT(false);
  }

  // Compute determinant
  const WideScalarT det = drjit::dot(v0v1, pvec);

  // Check for parallelism using normalized vectors (same as scalar version)
  const WideVec3T v0v1_norm = drjit::normalize(v0v1);
  const WideVec3T pvec_norm = drjit::normalize(pvec);
  const WideScalarT det_norm = drjit::dot(v0v1_norm, pvec_norm);
  const WideMaskT detValid = drjit::abs(det_norm) >= WideScalarT(kEpsilon);

  const WideScalarT invDet = WideScalarT(1) / det;

  // Compute tvec = rayOrigin - p0
  const WideVec3T tvec = rayOrigin - p0;

  // Compute u parameter
  u = drjit::dot(tvec, pvec) * invDet;
  const WideMaskT uValid = (u >= WideScalarT(-kEpsilon)) & (u <= WideScalarT(1 + kEpsilon));

  // Compute qvec = tvec × v0v1
  const WideVec3T qvec = drjit::cross(tvec, v0v1);

  // Compute v parameter
  v = drjit::dot(rayDirection, qvec) * invDet;
  const WideMaskT vValid = (v >= WideScalarT(-kEpsilon)) & ((u + v) <= WideScalarT(1 + kEpsilon));

  // Compute t parameter
  tOut = drjit::dot(v0v2, qvec) * invDet;

  // Compute intersection point
  intersectionPoint = rayOrigin + tOut * rayDirection;

  // Combine all validity checks
  const WideMaskT valid = notParallel & detValid & uValid & vValid;

  // Set invalid results to default values
  u = drjit::select(valid, u, WideScalarT(0));
  v = drjit::select(valid, v, WideScalarT(0));
  tOut = drjit::select(valid, tOut, WideScalarT(std::numeric_limits<T>::max()));
  intersectionPoint = drjit::select(valid, intersectionPoint, rayOrigin);

  return valid;
}

template <typename T, size_t LaneWidth>
WideMask<WideScalar<T, LaneWidth>> rayTriangleIntersectWide(
    const Eigen::Vector3<T>& beginRay,
    const Eigen::Vector3<T>& direction,
    const WideVec3<T, LaneWidth>& p0,
    const WideVec3<T, LaneWidth>& p1,
    const WideVec3<T, LaneWidth>& p2,
    WideVec3<T, LaneWidth>& intersectionPoint,
    WideScalar<T, LaneWidth>& tOut) {
  WideScalar<T, LaneWidth> u, v;
  return rayTriangleIntersectWide<T, LaneWidth>(
      beginRay, direction, p0, p1, p2, intersectionPoint, tOut, u, v);
}

// Explicit instantiations for common SIMD widths and default lane widths
template WideMask<WideScalar<float, 4>> rayTriangleIntersectWide<float, 4>(
    const Eigen::Vector3<float>&,
    const Eigen::Vector3<float>&,
    const WideVec3<float, 4>&,
    const WideVec3<float, 4>&,
    const WideVec3<float, 4>&,
    WideVec3<float, 4>&,
    WideScalar<float, 4>&,
    WideScalar<float, 4>&,
    WideScalar<float, 4>&);

template WideMask<WideScalar<float, 8>> rayTriangleIntersectWide<float, 8>(
    const Eigen::Vector3<float>&,
    const Eigen::Vector3<float>&,
    const WideVec3<float, 8>&,
    const WideVec3<float, 8>&,
    const WideVec3<float, 8>&,
    WideVec3<float, 8>&,
    WideScalar<float, 8>&,
    WideScalar<float, 8>&,
    WideScalar<float, 8>&);

template WideMask<WideScalar<double, 2>> rayTriangleIntersectWide<double, 2>(
    const Eigen::Vector3<double>&,
    const Eigen::Vector3<double>&,
    const WideVec3<double, 2>&,
    const WideVec3<double, 2>&,
    const WideVec3<double, 2>&,
    WideVec3<double, 2>&,
    WideScalar<double, 2>&,
    WideScalar<double, 2>&,
    WideScalar<double, 2>&);

template WideMask<WideScalar<double, 4>> rayTriangleIntersectWide<double, 4>(
    const Eigen::Vector3<double>&,
    const Eigen::Vector3<double>&,
    const WideVec3<double, 4>&,
    const WideVec3<double, 4>&,
    const WideVec3<double, 4>&,
    WideVec3<double, 4>&,
    WideScalar<double, 4>&,
    WideScalar<double, 4>&,
    WideScalar<double, 4>&);

// Versions without barycentric coordinates
template WideMask<WideScalar<float, 4>> rayTriangleIntersectWide<float, 4>(
    const Eigen::Vector3<float>&,
    const Eigen::Vector3<float>&,
    const WideVec3<float, 4>&,
    const WideVec3<float, 4>&,
    const WideVec3<float, 4>&,
    WideVec3<float, 4>&,
    WideScalar<float, 4>&);

template WideMask<WideScalar<float, 8>> rayTriangleIntersectWide<float, 8>(
    const Eigen::Vector3<float>&,
    const Eigen::Vector3<float>&,
    const WideVec3<float, 8>&,
    const WideVec3<float, 8>&,
    const WideVec3<float, 8>&,
    WideVec3<float, 8>&,
    WideScalar<float, 8>&);

template WideMask<WideScalar<double, 2>> rayTriangleIntersectWide<double, 2>(
    const Eigen::Vector3<double>&,
    const Eigen::Vector3<double>&,
    const WideVec3<double, 2>&,
    const WideVec3<double, 2>&,
    const WideVec3<double, 2>&,
    WideVec3<double, 2>&,
    WideScalar<double, 2>&);

template WideMask<WideScalar<double, 4>> rayTriangleIntersectWide<double, 4>(
    const Eigen::Vector3<double>&,
    const Eigen::Vector3<double>&,
    const WideVec3<double, 4>&,
    const WideVec3<double, 4>&,
    const WideVec3<double, 4>&,
    WideVec3<double, 4>&,
    WideScalar<double, 4>&);

} // namespace axel
