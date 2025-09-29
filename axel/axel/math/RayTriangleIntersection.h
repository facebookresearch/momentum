/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <Eigen/Core>

#include "axel/common/VectorizationTypes.h"

namespace axel {

template <typename T>
bool rayTriangleIntersect(
    const Eigen::Vector3<T>& beginRay,
    const Eigen::Vector3<T>& direction,
    const Eigen::Vector3<T>& p0,
    const Eigen::Vector3<T>& p1,
    const Eigen::Vector3<T>& p2,
    Eigen::Vector3<T>& intersectionPoint,
    T& tOut);

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
    T& v);

/**
 * @brief Wide vector version of ray-triangle intersection using Möller-Trumbore algorithm
 *
 * This function performs SIMD ray-triangle intersection tests using the same algorithm
 * as the scalar version, ensuring consistent results across all lanes.
 *
 * @tparam T Scalar type (float or double)
 * @tparam LaneWidth SIMD width (number of triangles to test simultaneously)
 * @param beginRay Ray origin (scalar)
 * @param direction Ray direction (scalar)
 * @param p0 First vertices of triangles (wide vector)
 * @param p1 Second vertices of triangles (wide vector)
 * @param p2 Third vertices of triangles (wide vector)
 * @param intersectionPoint Output intersection points (wide vector)
 * @param tOut Output t parameters (wide vector)
 * @param u Output u barycentric coordinates (wide vector)
 * @param v Output v barycentric coordinates (wide vector)
 * @return Wide mask indicating which triangles were hit
 */
template <typename T, size_t LaneWidth = kNativeLaneWidth<T>>
WideMask<WideScalar<T, LaneWidth>> rayTriangleIntersectWide(
    const Eigen::Vector3<T>& beginRay,
    const Eigen::Vector3<T>& direction,
    const WideVec3<T, LaneWidth>& p0,
    const WideVec3<T, LaneWidth>& p1,
    const WideVec3<T, LaneWidth>& p2,
    WideVec3<T, LaneWidth>& intersectionPoint,
    WideScalar<T, LaneWidth>& tOut,
    WideScalar<T, LaneWidth>& u,
    WideScalar<T, LaneWidth>& v);

/**
 * @brief Wide vector version without barycentric coordinates
 */
template <typename T, size_t LaneWidth = kNativeLaneWidth<T>>
WideMask<WideScalar<T, LaneWidth>> rayTriangleIntersectWide(
    const Eigen::Vector3<T>& beginRay,
    const Eigen::Vector3<T>& direction,
    const WideVec3<T, LaneWidth>& p0,
    const WideVec3<T, LaneWidth>& p1,
    const WideVec3<T, LaneWidth>& p2,
    WideVec3<T, LaneWidth>& intersectionPoint,
    WideScalar<T, LaneWidth>& tOut);

} // namespace axel
