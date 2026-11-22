/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <Eigen/Core>

namespace axel {

/// Intersects the parametric line `beginRay + t * direction` with a triangle.
///
/// Despite the function name, intersections with negative `tOut` are reported; callers that need a
/// ray or segment must bound `tOut` themselves. Zero-length directions, degenerate triangles, and
/// near-parallel lines return false using a scale-invariant angular tolerance.
template <typename T>
bool rayTriangleIntersect(
    const Eigen::Vector3<T>& beginRay,
    const Eigen::Vector3<T>& direction,
    const Eigen::Vector3<T>& p0,
    const Eigen::Vector3<T>& p1,
    const Eigen::Vector3<T>& p2,
    Eigen::Vector3<T>& intersectionPoint,
    T& tOut);

/// Intersects the same parametric line and also returns triangle barycentric coordinates `u` and
/// `v`; the third coordinate is `1 - u - v`.
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

} // namespace axel
