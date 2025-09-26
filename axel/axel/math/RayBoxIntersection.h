/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "axel/Ray.h"

#include <drjit/array.h>
#include <drjit/packet.h>
#include <Eigen/Core>
#include <tuple>

namespace axel {

/**
 * Templated result struct for ray-box intersection results.
 * Provides named fields for better readability compared to std::tuple.
 * Works for both scalar (MaskType = bool) and SIMD (MaskType = SIMD mask) cases.
 */
template <typename ScalarType, typename MaskType = bool>
struct RayBoxIntersectionResult {
  MaskType intersects; // True/mask if ray intersects the box within [ray.minT, ray.maxT]
  ScalarType tMin; // Intersection entry distance(s)
  ScalarType tMax; // Intersection exit distance(s)
};

/**
 * SIMD ray-box intersection test using drjit::Array (Vector3SP) format.
 * This function accepts drjit::Array<drjit::Packet<ScalarType, SimdWidth>, 3>
 * which is commonly used in SIMD-optimized code as Vector3SP.
 *
 * Tests a single ray against multiple bounding boxes packed in SIMD vectors.
 * Based on the proven scalar logic from BoundingBox.cpp but extended to handle
 * ray parameter ranges (minT, maxT) and SIMD operations.
 *
 * @param ray The ray to test (origin, direction, minT, maxT)
 * @param boxMin SIMD vector containing minimum corners of multiple boxes
 * @param boxMax SIMD vector containing maximum corners of multiple boxes
 * @return RayBoxIntersectionResult with named fields:
 *   - intersects: SIMD mask indicating which boxes intersect the ray
 *   - tMin: SIMD vector of intersection entry distances
 *   - tMax: SIMD vector of intersection exit distances
 */
template <typename ScalarType, size_t SimdWidth>
RayBoxIntersectionResult<
    drjit::Packet<ScalarType, SimdWidth>,
    typename drjit::Packet<ScalarType, SimdWidth>::MaskType>
intersectRayBoxes(
    const Ray3<ScalarType>& ray,
    const drjit::Array<drjit::Packet<ScalarType, SimdWidth>, 3>& boxMin,
    const drjit::Array<drjit::Packet<ScalarType, SimdWidth>, 3>& boxMax);

/**
 * Scalar ray-box intersection test with ray parameter range support.
 * Based on the logic from BoundingBox.cpp with added minT/maxT handling.
 *
 * @param ray The ray to test
 * @param boxMin Box minimum corner
 * @param boxMax Box maximum corner
 * @return RayBoxIntersectionResult with named fields:
 *   - intersects: true if ray intersects the box within [ray.minT, ray.maxT]
 *   - tMin: intersection entry distance
 *   - tMax: intersection exit distance
 */
template <typename ScalarType>
RayBoxIntersectionResult<ScalarType> intersectRayBox(
    const Ray3<ScalarType>& ray,
    const Eigen::Vector3<ScalarType>& boxMin,
    const Eigen::Vector3<ScalarType>& boxMax);

} // namespace axel
