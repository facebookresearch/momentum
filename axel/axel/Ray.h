/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <Eigen/Core>

namespace axel {
inline constexpr double kRayMinDistanceEpsilon{1e-3};

template <typename ScalarType>
struct Ray3 {
  Eigen::Vector3<ScalarType> origin;
  ScalarType minT = static_cast<ScalarType>(kRayMinDistanceEpsilon);

  Eigen::Vector3<ScalarType> direction;
  ScalarType maxT = std::numeric_limits<ScalarType>::max();

  inline Ray3() = default;

  inline Ray3(const Eigen::Vector3<ScalarType>& origin, const Eigen::Vector3<ScalarType>& direction)
      : origin(origin), direction(direction) {}

  inline Ray3(
      const Eigen::Vector3<ScalarType>& origin,
      const Eigen::Vector3<ScalarType>& direction,
      const ScalarType& maxT)
      : origin(origin), direction(direction), maxT(maxT) {}

  inline Ray3(
      const Eigen::Vector3<ScalarType>& origin,
      const Eigen::Vector3<ScalarType>& direction,
      const ScalarType& maxT,
      const ScalarType& minT)
      : origin(origin), minT(minT), direction(direction), maxT(maxT) {}
};

using Ray3d = Ray3<double>;
using Ray3f = Ray3<float>;

} // namespace axel
