/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <optional>

#ifndef AXEL_NO_DISPENSO
#include <dispenso/parallel_for.h>
#endif

#include "axel/Bvh.h"
#include "axel/Ray.h"

namespace axel {

inline constexpr double kDefaultBoundingBoxThickness{0.0};

template <typename S, size_t LeafCapacity = 1>
class TriBvh {
 public:
  using Scalar = S;
  using QueryBuffer = typename BvhBase<S>::QueryBuffer;
  using ClosestSurfacePointResult = ::axel::ClosestSurfacePointResult<S>;

  TriBvh() = default;

  /**
   * @brief Construct a new Bvh from the given positions and triangles. The constructor assumes
   * that the integrity of provided positions and triangles is checked beforehand.
   */
  explicit TriBvh(
      Eigen::MatrixX3<S>&& positions,
      Eigen::MatrixX3i&& triangles,
      const std::optional<S>& boundingBoxThickness = kDefaultBoundingBoxThickness);

  /**
   * @brief Performs a query with a given bounding box.
   * All primitives whose bounding box intersects with the query get returned as a list of their
   * IDs, e.g. the triangle indices.
   */
  [[nodiscard]] std::vector<uint32_t> boxQuery(const BoundingBox<S>& box) const;

  /**
   * @brief Performs a query with a given bounding box.
   * Stores up to 512 primitive IDs in the query buffer, such that their bounding boxes
   * intersect with the query.
   * The number of valid hits in the buffer is returned.
   * NOTE: This method performs no dynamic memory allocation.
   */
  uint32_t boxQuery(const BoundingBox<S>& box, QueryBuffer& hits) const;

  /**
   * @brief Returns all primitives hit by the line formed from the ray's direction.
   */
  [[nodiscard]] std::vector<uint32_t> lineHits(const Ray3<S>& ray) const;

  /**
   * @brief Returns the closest hit with the given query ray.
   * If the ray hits nothing, returns a std::nullopt.
   */
  [[nodiscard]] std::optional<IntersectionResult<S>> closestHit(const Ray3<S>& ray) const;

  /**
   * @brief Returns all hits with the given query ray.
   */
  [[nodiscard]] std::vector<IntersectionResult<S>> allHits(const Ray3<S>& ray) const;

  /**
   * @brief Checks whether the given ray intersects any of the primitives in the Bvh.
   */
  [[nodiscard]] bool anyHit(const Ray3<S>& ray) const;

  [[nodiscard]] ClosestSurfacePointResult closestSurfacePoint(const Eigen::Vector3<S>& query) const;

  /**
   * @brief Returns the total number of internal nodes in the tree.
   */
  [[nodiscard]] size_t getNodeCount() const;

  /**
   * @brief Returns the total number of primitives in the tree. It should be equal to the number of
   * triangles that Bvh was constructed with.
   */
  [[nodiscard]] size_t getPrimitiveCount() const;

 private:
  // This memory layout is required for gather operations in vectorized code.
  Eigen::Matrix<S, Eigen::Dynamic, 3, Eigen::RowMajor> positions_;
  Eigen::MatrixX3i triangles_;
  Bvh<S, LeafCapacity> bvh_;
};

using TriBvhf = TriBvh<float>;
using TriBvhd = TriBvh<double>;

extern template class TriBvh<float>;
extern template class TriBvh<double>;
extern template class TriBvh<float, kNativeLaneWidth<float>>;
extern template class TriBvh<double, kNativeLaneWidth<double>>;

template <typename Derived, typename F>
void closestSurfacePoints(
    const TriBvh<typename Derived::Scalar>& bvh,
    const Eigen::PlainObjectBase<Derived>& queryPoints,
    F&& resultFunc) {
#ifdef AXEL_NO_DISPENSO
  for (uint32_t i = 0; i < queryPoints.rows(); ++i) {
    resultFunc(i, bvh.closestSurfacePoint(queryPoints.row(i)));
  }
#else
  dispenso::parallel_for(
      0, queryPoints.rows(), [&bvh, &queryPoints, &resultFunc](const uint32_t i) {
        resultFunc(i, bvh.closestSurfacePoint(queryPoints.row(i)));
      });
#endif
}

template <typename Derived1, typename Derived2, typename Derived3, typename Derived4>
void closestSurfacePoints(
    const TriBvh<typename Derived1::Scalar>& bvh,
    const Eigen::PlainObjectBase<Derived1>& queryPoints,
    Eigen::PlainObjectBase<Derived2>& closestSquareDistances,
    Eigen::PlainObjectBase<Derived3>& closestTriangles,
    Eigen::PlainObjectBase<Derived4>& closestPoints) {
  using S = typename Derived1::Scalar;
  static_assert(
      std::is_same_v<typename Derived2::Scalar, S> && std::is_same_v<typename Derived4::Scalar, S>,
      "All output matrices must have the same scalar type.");
  closestSquareDistances.resize(queryPoints.rows(), 1);
  closestTriangles.resize(queryPoints.rows(), 1);
  closestPoints.resize(queryPoints.rows(), 3);

  const auto closestSquareDistancesLambda =
      [&bvh, &queryPoints, &closestSquareDistances, &closestPoints, &closestTriangles](
          const uint32_t i) {
        const auto result = bvh.closestSurfacePoint(queryPoints.row(i));
        closestPoints.row(i) = result.point;
        closestTriangles(i) = static_cast<int32_t>(result.triangleIdx);
        closestSquareDistances(i) =
            (Eigen::Vector3<S>(queryPoints.row(i)) - result.point).squaredNorm();
      };

#ifdef AXEL_NO_DISPENSO
  for (uint32_t i = 0; i < queryPoints.rows(); ++i) {
    closestSquareDistancesLambda(i);
  }
#else
  dispenso::parallel_for(0, queryPoints.rows(), closestSquareDistancesLambda);
#endif
}

} // namespace axel
