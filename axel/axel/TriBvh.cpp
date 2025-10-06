/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "axel/TriBvh.h"

#include "axel/Checks.h"
#include "axel/Profile.h"
#include "axel/math/BoundingBoxUtils.h"
#include "axel/math/PointTriangleProjection.h"

namespace axel {
namespace {
template <typename S>
struct HitResult {
  S bary0;
  S bary1;
  S t{-1.0};
  uint32_t triangleIdx = kInvalidTriangleIdx;
};

template <typename T, typename MatrixType>
Eigen::Vector3<T> interpolatePosition(
    const Eigen::Vector3i& triangleIndices,
    const MatrixType& positions,
    const Eigen::Vector3<T>& bary) {
  return positions.row(triangleIndices[0]) * bary[0] + positions.row(triangleIndices[1]) * bary[1] +
      positions.row(triangleIndices[2]) * bary[2];
}

template <typename S, size_t LeafCapacity>
Bvh<S, LeafCapacity> buildBvh(
    const Eigen::MatrixX3<S>& positions,
    const Eigen::MatrixX3i& triangles,
    const std::optional<S>& boundingBoxThickness) {
  XR_PROFILE_EVENT("build_bvh");
  const Eigen::Index triangleCount{triangles.rows()};
  std::vector<BoundingBox<S>> boundingBoxes(triangleCount);

  const auto boundingBoxComputeLambda = [&](const Eigen::Index triangleIndex) {
    const Eigen::Vector3i triangle = triangles.row(triangleIndex);

    const Eigen::Index positionCount{positions.rows()};
    XR_CHECK(
        triangle[0] < positionCount && triangle[1] < positionCount && triangle[2] < positionCount,
        "Triangle {} index out of bounds! It has vertices {}, {}, {}, but the last vertex is at {}.",
        triangleIndex,
        triangle[0],
        triangle[1],
        triangle[2],
        positionCount - 1);
    Eigen::Vector3<S> minBound;
    Eigen::Vector3<S> maxBound;
    if (boundingBoxThickness.has_value()) {
      getMinMaxBoundTriangle<S>(
          positions.row(triangle[0]),
          positions.row(triangle[1]),
          positions.row(triangle[2]),
          minBound,
          maxBound,
          boundingBoxThickness.value());
    } else {
      getAdaptiveMinMaxBoundTriangle<S>(
          positions.row(triangle[0]),
          positions.row(triangle[1]),
          positions.row(triangle[2]),
          minBound,
          maxBound,
          /*multiplicativeFactor*/ 0.01);
    }
    boundingBoxes[triangleIndex] =
        BoundingBox<S>(minBound, maxBound, static_cast<int>(triangleIndex));
  };
#ifdef AXEL_NO_DISPENSO
  for (size_t triangleIndex = 0; triangleIndex < triangleCount; ++triangleIndex) {
    boundingBoxComputeLambda(triangleIndex);
  }
#else
  dispenso::parallel_for(0, triangleCount, boundingBoxComputeLambda);
#endif
  return Bvh<S, LeafCapacity>{boundingBoxes};
}
} // namespace

template <typename S, size_t LeafCapacity>
TriBvh<S, LeafCapacity>::TriBvh(
    Eigen::MatrixX3<S>&& positions,
    Eigen::MatrixX3i&& triangles,
    const std::optional<S>& boundingBoxThickness)
    : positions_{std::move(positions)},
      triangles_{std::move(triangles)},
      bvh_{buildBvh<S, LeafCapacity>(positions_, triangles_, boundingBoxThickness)} {}

template <typename S, size_t LeafCapacity>
std::vector<uint32_t> TriBvh<S, LeafCapacity>::boxQuery(const BoundingBox<S>& box) const {
  return bvh_.query(box);
}

template <typename S, size_t LeafCapacity>
uint32_t TriBvh<S, LeafCapacity>::boxQuery(const BoundingBox<S>& box, QueryBuffer& hits) const {
  return bvh_.query(box, hits);
}

template <typename S, size_t LeafCapacity>
std::vector<uint32_t> TriBvh<S, LeafCapacity>::lineHits(const Ray3<S>& ray) const {
  return bvh_.query(ray.origin, ray.direction);
}

template <typename S, size_t LeafCapacity>
std::optional<IntersectionResult<S>> TriBvh<S, LeafCapacity>::closestHit(const Ray3<S>& ray) const {
  auto result = bvh_.template rayQueryClosestHit<HitResult<S>>(
      ray.origin,
      ray.direction,
      ray.minT,
      ray.maxT,
      [this, &ray](const uint32_t primIdx, HitResult<S>& result, S& maxT) {
        const Eigen::Vector3i tri = triangles_.row(primIdx);
        const Eigen::Vector3<S> p0 = positions_.row(tri[0]);
        const Eigen::Vector3<S> p1 = positions_.row(tri[1]);
        const Eigen::Vector3<S> p2 = positions_.row(tri[2]);

        S u;
        S v;
        S tOut;
        Eigen::Vector3<S> itsPoint;
        if (rayTriangleIntersect(ray.origin, ray.direction, p0, p1, p2, itsPoint, tOut, u, v) &&
            tOut >= ray.minT && tOut <= maxT) {
          result.t = tOut;
          result.triangleIdx = primIdx;
          result.bary0 = u;
          result.bary1 = v;
          maxT = tOut;
        }
      });
  if (result.triangleIdx == kInvalidTriangleIdx) {
    return std::nullopt;
  }

  const Eigen::Vector3<S> bary(
      static_cast<S>(1.0) - result.bary0 - result.bary1, result.bary0, result.bary1);
  return IntersectionResult<S>{
      static_cast<int32_t>(result.triangleIdx),
      result.t,
      interpolatePosition(triangles_.row(result.triangleIdx), positions_, bary),
      bary};
}

template <typename S, size_t LeafCapacity>
std::vector<IntersectionResult<S>> TriBvh<S, LeafCapacity>::allHits(const Ray3<S>& ray) const {
  const auto results = bvh_.template rayQueryClosestHit<std::vector<HitResult<S>>>(
      ray.origin,
      ray.direction,
      ray.minT,
      ray.maxT,
      [this, &ray](const uint32_t primIdx, auto& results, S& maxT) {
        const Eigen::Vector3i tri = triangles_.row(primIdx);
        const Eigen::Vector3<S> p0 = positions_.row(tri[0]);
        const Eigen::Vector3<S> p1 = positions_.row(tri[1]);
        const Eigen::Vector3<S> p2 = positions_.row(tri[2]);

        S u;
        S v;
        S tOut;
        Eigen::Vector3<S> itsPoint;
        if (rayTriangleIntersect(ray.origin, ray.direction, p0, p1, p2, itsPoint, tOut, u, v) &&
            tOut >= ray.minT && tOut <= maxT) {
          // We don't update the maxT here because we want to intersect with all the triangles.
          results.push_back({u, v, tOut, primIdx});
        }
      });

  std::vector<IntersectionResult<S>> intersections;
  intersections.reserve(results.size());
  for (const auto& result : results) {
    const Eigen::Vector3<S> bary(1.0 - result.bary0 - result.bary1, result.bary0, result.bary1);
    intersections.push_back(
        {static_cast<int32_t>(result.triangleIdx),
         result.t,
         interpolatePosition(triangles_.row(result.triangleIdx), positions_, bary),
         bary});
  }
  return intersections;
}

template <typename S, size_t LeafCapacity>
bool TriBvh<S, LeafCapacity>::anyHit(const Ray3<S>& ray) const {
  return bvh_.rayQueryAnyHit(
      ray.origin, ray.direction, ray.minT, ray.maxT, [this, &ray](const uint32_t primIdx) {
        const Eigen::Vector3i tri = triangles_.row(primIdx);
        const Eigen::Vector3<S> p0 = positions_.row(tri[0]);
        const Eigen::Vector3<S> p1 = positions_.row(tri[1]);
        const Eigen::Vector3<S> p2 = positions_.row(tri[2]);

        S u;
        S v;
        S tOut;
        Eigen::Vector3<S> itsPoint;
        return rayTriangleIntersect(ray.origin, ray.direction, p0, p1, p2, itsPoint, tOut, u, v) &&
            tOut >= ray.minT && tOut <= ray.maxT;
      });
}

template <typename S, size_t LeafCapacity>
ClosestSurfacePointResult<S> TriBvh<S, LeafCapacity>::closestSurfacePoint(
    const Eigen::Vector3<S>& query) const {
  // If LeafCapacity is equal to the native SIMD width, we can vectorize.
  if constexpr (LeafCapacity > 1 && LeafCapacity == kNativeLaneWidth<S>) {
    WideVec3<S> wideQuery;
    wideQuery[0] = query[0];
    wideQuery[1] = query[1];
    wideQuery[2] = query[2];

    // Note: We have to define wide int scalar depending on the native lane width of the
    // floating point type (e.g. AVX = either 4 for doubles or 8 for floats)
    using WideScalari = WideScalar<int32_t, kNativeLaneWidth<S>>;
    return bvh_.queryClosestSurfacePointPacket(
        query,
        [this, &wideQuery](
            const Eigen::Vector3<S>& /*query*/,
            const uint32_t primCount,
            const WideScalari& indices,
            WideVec3<S>& projections,
            WideVec3<S>& barycentrics,
            WideScalar<S>& squaredDistances) {
          // Is there a more efficient way to create this mask?
          const auto mask = drjit::arange<WideScalari>(LeafCapacity) < primCount;
          const auto v0Ind = drjit::gather<WideScalari>(triangles_.col(0).data(), indices, mask);
          const auto v1Ind = drjit::gather<WideScalari>(triangles_.col(1).data(), indices, mask);
          const auto v2Ind = drjit::gather<WideScalari>(triangles_.col(2).data(), indices, mask);
          projectOnTriangle(
              wideQuery,
              drjit::gather<WideVec3<S>>(positions_.data(), v0Ind, mask),
              drjit::gather<WideVec3<S>>(positions_.data(), v1Ind, mask),
              drjit::gather<WideVec3<S>>(positions_.data(), v2Ind, mask),
              projections,
              &barycentrics);
          squaredDistances = drjit::squared_norm(projections - wideQuery);
        });
  } else {
    return bvh_.queryClosestSurfacePoint(
        query,
        [this](
            const Eigen::Vector3<S>& query,
            const uint32_t primIdx,
            Eigen::Vector3<S>& projection,
            Eigen::Vector3<S>& barycentric) {
          const Eigen::Vector3i& face = triangles_.row(primIdx);
          projectOnTriangle<S>(
              query,
              positions_.row(face(0)),
              positions_.row(face(1)),
              positions_.row(face(2)),
              projection,
              &barycentric);
        });
  }
}

template <typename S, size_t LeafCapacity>
size_t TriBvh<S, LeafCapacity>::getNodeCount() const {
  return bvh_.getNodeCount();
}

template <typename S, size_t LeafCapacity>
size_t TriBvh<S, LeafCapacity>::getPrimitiveCount() const {
  return bvh_.getPrimitiveCount();
}

template class TriBvh<float>;
template class TriBvh<double>;
template class TriBvh<float, kNativeLaneWidth<float>>;
template class TriBvh<double, kNativeLaneWidth<double>>;

} // namespace axel
