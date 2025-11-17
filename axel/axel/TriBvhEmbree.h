/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <embree4/rtcore.h>
#include <optional>

#include "axel/BvhCommon.h"
#include "axel/Ray.h"

namespace axel {

template <typename S>
class TriBvhEmbree {
 public:
  TriBvhEmbree(const Eigen::MatrixX3<S>& positions, const Eigen::MatrixX3i& triangles);
  ~TriBvhEmbree();

  TriBvhEmbree(const TriBvhEmbree&) = delete;
  TriBvhEmbree& operator=(const TriBvhEmbree&) = delete;

  TriBvhEmbree(TriBvhEmbree&& other) noexcept;
  TriBvhEmbree& operator=(TriBvhEmbree&& other) noexcept;

  /**
   * @brief Returns the closest hit with the given query ray.
   * If the ray hits nothing, returns a std::nullopt.
   */
  std::optional<IntersectionResult<S>> closestHit(const Ray3<S>& ray) const;

  /**
   * @brief Checks whether the given ray intersects any of the primitives in the Bvh.
   */
  bool anyHit(const Ray3<S>& ray) const;

  ClosestSurfacePointResult<S> closestSurfacePoint(const Eigen::Vector3<S>& query) const;

  /**
   * @brief Returns the total number of primitives in the tree. It should be equal to the number of
   * triangles that Bvh was constructed with.
   */
  [[nodiscard]] size_t getPrimitiveCount() const;

 private:
  // TODO(nemanjab): Wrap this in a singleton to allow usage across multiple BVHs.
  RTCDevice device_;
  RTCScene scene_;
  Eigen::MatrixX3<S> positions_;
  Eigen::MatrixX3i triangles_;
};

} // namespace axel
