/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <Eigen/Geometry>

#include "axel/math/CoplanarityCheck.h"
#include "axel/math/EdgeEdgeDistance.h"
#include "axel/math/PointTriangleProjection.h"

namespace axel {

bool ccdEdgeEdge(
    const Eigen::Vector3d& x1,
    const Eigen::Vector3d& x2,
    const Eigen::Vector3d& x3,
    const Eigen::Vector3d& x4,
    const Eigen::Vector3d& v1,
    const Eigen::Vector3d& v2,
    const Eigen::Vector3d& v3,
    const Eigen::Vector3d& v4,
    const double distanceThreshold,
    const double dt) {
  // Find times that triangle and point are coplanar.
  std::array<double, 4> t{0.0, 0.0, 0.0, 0.0};
  const int solutions = timesCoplanar(std::span{t}.first<3>(), x1, x2, x3, x4, v1, v2, v3, v4);
  // Include dt as a potential solution in case numerical imprecisions do not account for it.
  t[solutions] = dt;
  for (int32_t sol = 0; sol <= solutions; sol++) {
    const double time = t[sol];
    // If time of collision is not during the current time step, we can skip.
    if (time <= 0.0 || time > dt) {
      continue;
    }

    // Compute the collider edge at the time when all points are coplanar
    // and check if the distance between the two edges is close enough.
    double s;
    double v;
    double distance;
    const bool success = distanceEdgeEdge(
        x1 + time * v1, x2 + time * v2, x3 + time * v3, x4 + time * v4, s, v, distance);
    if (success && distance < distanceThreshold) {
      return true;
    }
  }
  return false;
}

bool ccdVertexTriangle(
    const Eigen::Vector3d& x1,
    const Eigen::Vector3d& x2,
    const Eigen::Vector3d& x3,
    const Eigen::Vector3d& x4,
    const Eigen::Vector3d& v1,
    const Eigen::Vector3d& v2,
    const Eigen::Vector3d& v3,
    const Eigen::Vector3d& v4,
    const double distanceThreshold,
    const double dt) {
  // Find times that triangle and point are coplanar.
  std::array<double, 4> t{0.0, 0.0, 0.0, 0.0};
  const int solutions = timesCoplanar(std::span{t}.first<3>(), x1, x2, x3, x4, v1, v2, v3, v4);
  // Include dt as a potential solution in case numerical imprecisions do not account for it.
  t[solutions] = dt;

  for (int32_t sol = 0; sol <= solutions; sol++) {
    const double time = t[sol];
    // If time of collision is not during the current time step, we can skip.
    if (time <= 0.0 || time > dt) {
      continue;
    }
    const Eigen::Vector3d x1_at_t = x1 + time * v1;
    const Eigen::Vector3d x2_at_t = x2 + time * v2;
    const Eigen::Vector3d x3_at_t = x3 + time * v3;
    const Eigen::Vector3d x4_at_t = x4 + time * v4;
    Eigen::Vector3d q = Eigen::Vector3d::Zero();
    const bool insideTriangle = projectOnTriangle(x4_at_t, x1_at_t, x2_at_t, x3_at_t, q);

    const double distSq = (x4_at_t - q).squaredNorm();
    const double distanceThresholdSq = distanceThreshold * distanceThreshold;
    if (insideTriangle && (distSq < distanceThresholdSq)) {
      return true;
    }
  }
  return false;
}

} // namespace axel
