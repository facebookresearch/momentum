/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "axel/math/RayBoxIntersection.h"

#include <algorithm>
#include <limits>

namespace axel {

template <typename ScalarType>
RayBoxIntersectionResult<ScalarType> intersectRayBox(
    const Ray3<ScalarType>& ray,
    const Eigen::Vector3<ScalarType>& boxMin,
    const Eigen::Vector3<ScalarType>& boxMax) {
  // Based on the proven implementation from BoundingBox.cpp intersects() method
  // Reference:
  // https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection

  const Eigen::Vector3<ScalarType>& origin = ray.origin;
  const Eigen::Vector3<ScalarType>& direction = ray.direction;

  // X-axis intersection
  ScalarType tmin = (boxMin.x() - origin.x()) / direction.x();
  ScalarType tmax = (boxMax.x() - origin.x()) / direction.x();

  if (tmin > tmax) {
    std::swap(tmin, tmax);
  }

  // Y-axis intersection
  ScalarType tymin = (boxMin.y() - origin.y()) / direction.y();
  ScalarType tymax = (boxMax.y() - origin.y()) / direction.y();

  if (tymin > tymax) {
    std::swap(tymin, tymax);
  }

  // Check if ray misses the box in Y
  if ((tmin > tymax) || (tymin > tmax)) {
    return {false, ScalarType(0), ScalarType(0)};
  }

  if (tymin > tmin) {
    tmin = tymin;
  }

  if (tymax < tmax) {
    tmax = tymax;
  }

  // Z-axis intersection
  ScalarType tzmin = (boxMin.z() - origin.z()) / direction.z();
  ScalarType tzmax = (boxMax.z() - origin.z()) / direction.z();

  if (tzmin > tzmax) {
    std::swap(tzmin, tzmax);
  }

  // Check if ray misses the box in Z
  if ((tmin > tzmax) || (tzmin > tmax)) {
    return {false, ScalarType(0), ScalarType(0)};
  }

  if (tzmin > tmin) {
    tmin = tzmin;
  }

  if (tzmax < tmax) {
    tmax = tzmax;
  }

  // Now we have the intersection range [tmin, tmax] with the box
  // Check if this range overlaps with the ray's parameter range [ray.minT, ray.maxT]

  // Handle the case where ray.minT > ray.maxT (backwards ray)
  const ScalarType actualMinT = std::min(ray.minT, ray.maxT);
  const ScalarType actualMaxT = std::max(ray.minT, ray.maxT);

  // Check if the intersection range overlaps with the ray parameter range
  // Ranges [tmin, tmax] and [actualMinT, actualMaxT] overlap if:
  // tmin <= actualMaxT && tmax >= actualMinT
  if (tmin <= actualMaxT && tmax >= actualMinT) {
    // There is an intersection within the ray's parameter range
    return {true, tmin, tmax};
  }

  return {false, ScalarType(0), ScalarType(0)};
}

template <typename ScalarType, size_t SimdWidth>
RayBoxIntersectionResult<
    drjit::Packet<ScalarType, SimdWidth>,
    typename drjit::Packet<ScalarType, SimdWidth>::MaskType>
intersectRayBoxes(
    const Ray3<ScalarType>& ray,
    const drjit::Array<drjit::Packet<ScalarType, SimdWidth>, 3>& boxMin,
    const drjit::Array<drjit::Packet<ScalarType, SimdWidth>, 3>& boxMax) {
  using FloatP = drjit::Packet<ScalarType, SimdWidth>;
  using MaskP = typename FloatP::MaskType;

  const Eigen::Vector3<ScalarType>& origin = ray.origin;
  const Eigen::Vector3<ScalarType>& direction = ray.direction;

  // X-axis intersection
  FloatP tmin = (boxMin[0] - FloatP(origin.x())) / FloatP(direction.x());
  FloatP tmax = (boxMax[0] - FloatP(origin.x())) / FloatP(direction.x());

  // Swap tmin and tmax if needed for X-axis
  MaskP xSwapMask = tmin > tmax;
  FloatP temp = drjit::select(xSwapMask, tmin, tmax);
  tmin = drjit::select(xSwapMask, tmax, tmin);
  tmax = temp;

  // Y-axis intersection
  FloatP tymin = (boxMin[1] - FloatP(origin.y())) / FloatP(direction.y());
  FloatP tymax = (boxMax[1] - FloatP(origin.y())) / FloatP(direction.y());

  // Swap tymin and tymax if needed for Y-axis
  MaskP ySwapMask = tymin > tymax;
  temp = drjit::select(ySwapMask, tymin, tymax);
  tymin = drjit::select(ySwapMask, tymax, tymin);
  tymax = temp;

  // Check if ray misses the box in Y
  MaskP yHits = (tmin <= tymax) & (tymin <= tmax);

  // Update tmin and tmax based on Y intersection
  tmin = drjit::select(tymin > tmin, tymin, tmin);
  tmax = drjit::select(tymax < tmax, tymax, tmax);

  // Z-axis intersection
  FloatP tzmin = (boxMin[2] - FloatP(origin.z())) / FloatP(direction.z());
  FloatP tzmax = (boxMax[2] - FloatP(origin.z())) / FloatP(direction.z());

  // Swap tzmin and tzmax if needed for Z-axis
  MaskP zSwapMask = tzmin > tzmax;
  temp = drjit::select(zSwapMask, tzmin, tzmax);
  tzmin = drjit::select(zSwapMask, tzmax, tzmin);
  tzmax = temp;

  // Check if ray misses the box in Z
  MaskP zHits = (tmin <= tzmax) & (tzmin <= tmax);

  // Update tmin and tmax based on Z intersection
  tmin = drjit::select(tzmin > tmin, tzmin, tmin);
  tmax = drjit::select(tzmax < tmax, tzmax, tmax);

  // Now we have the intersection range [tmin, tmax] with each box
  // Check if this range overlaps with the ray's parameter range [ray.minT, ray.maxT]

  // Handle the case where ray.minT > ray.maxT (backwards ray)
  const ScalarType actualMinT = std::min(ray.minT, ray.maxT);
  const ScalarType actualMaxT = std::max(ray.minT, ray.maxT);

  // Check if the intersection range overlaps with the ray parameter range
  MaskP tRangeValid = (tmin <= FloatP(actualMaxT)) & (tmax >= FloatP(actualMinT));

  // Combine all intersection tests
  MaskP intersects = yHits & zHits & tRangeValid;

  return {intersects, tmin, tmax};
}

// Explicit template instantiations for common types
template RayBoxIntersectionResult<float> intersectRayBox<float>(
    const Ray3<float>& ray,
    const Eigen::Vector3<float>& boxMin,
    const Eigen::Vector3<float>& boxMax);

template RayBoxIntersectionResult<double> intersectRayBox<double>(
    const Ray3<double>& ray,
    const Eigen::Vector3<double>& boxMin,
    const Eigen::Vector3<double>& boxMax);

// SIMD instantiations for common widths using drjit::Array
template RayBoxIntersectionResult<
    drjit::Packet<float, 4>,
    typename drjit::Packet<float, 4>::MaskType>
intersectRayBoxes<float, 4>(
    const Ray3<float>& ray,
    const drjit::Array<drjit::Packet<float, 4>, 3>& boxMin,
    const drjit::Array<drjit::Packet<float, 4>, 3>& boxMax);

template RayBoxIntersectionResult<
    drjit::Packet<float, 8>,
    typename drjit::Packet<float, 8>::MaskType>
intersectRayBoxes<float, 8>(
    const Ray3<float>& ray,
    const drjit::Array<drjit::Packet<float, 8>, 3>& boxMin,
    const drjit::Array<drjit::Packet<float, 8>, 3>& boxMax);

template RayBoxIntersectionResult<
    drjit::Packet<double, 4>,
    typename drjit::Packet<double, 4>::MaskType>
intersectRayBoxes<double, 4>(
    const Ray3<double>& ray,
    const drjit::Array<drjit::Packet<double, 4>, 3>& boxMin,
    const drjit::Array<drjit::Packet<double, 4>, 3>& boxMax);

// TriBvh uses TreeSplit = 8 for both float and double, so we need width 8 for both
template RayBoxIntersectionResult<
    drjit::Packet<double, 8>,
    typename drjit::Packet<double, 8>::MaskType>
intersectRayBoxes<double, 8>(
    const Ray3<double>& ray,
    const drjit::Array<drjit::Packet<double, 8>, 3>& boxMin,
    const drjit::Array<drjit::Packet<double, 8>, 3>& boxMax);

template struct RayBoxIntersectionResult<
    drjit::Packet<float, 4>,
    typename drjit::Packet<float, 4>::MaskType>;

template struct RayBoxIntersectionResult<
    drjit::Packet<float, 8>,
    typename drjit::Packet<float, 8>::MaskType>;

template struct RayBoxIntersectionResult<
    drjit::Packet<double, 4>,
    typename drjit::Packet<double, 4>::MaskType>;

template struct RayBoxIntersectionResult<
    drjit::Packet<double, 8>,
    typename drjit::Packet<double, 8>::MaskType>;

} // namespace axel
