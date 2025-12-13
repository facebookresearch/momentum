/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <Eigen/Geometry>

#include "axel/math/PointTriangleProjection.h"
#include "axel/math/PointTriangleProjectionDefinitions.h"

namespace axel {

template bool projectOnTriangle(
    const Eigen::Vector3<float>& p,
    const Eigen::Vector3<float>& a,
    const Eigen::Vector3<float>& b,
    const Eigen::Vector3<float>& c,
    Eigen::Vector3<float>& q,
    Eigen::Vector3<float>* barycentric);
template bool projectOnTriangle(
    const Eigen::Vector3<double>& p,
    const Eigen::Vector3<double>& a,
    const Eigen::Vector3<double>& b,
    const Eigen::Vector3<double>& c,
    Eigen::Vector3<double>& q,
    Eigen::Vector3<double>* barycentric);

// NOLINTNEXTLINE(bugprone-macro-parentheses)
#define INSTANTIATE_PROJECT_ON_TRIANGLE(Scalar)            \
  template WideMask<WideScalar<Scalar>> projectOnTriangle( \
      const WideVec3<Scalar>& p,                           \
      const WideVec3<Scalar>& a,                           \
      const WideVec3<Scalar>& b,                           \
      const WideVec3<Scalar>& c,                           \
      WideVec3<Scalar>& q,                                 \
      WideVec3<Scalar>* barycentric);

INSTANTIATE_PROJECT_ON_TRIANGLE(float)
INSTANTIATE_PROJECT_ON_TRIANGLE(double)

} // namespace axel
