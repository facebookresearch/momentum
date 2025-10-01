/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>
#include <optional>

#ifndef AXEL_NO_DISPENSO
#include <dispenso/parallel_for.h>
#endif

#include <drjit/array.h>
#include <drjit/fwd.h>
#include <drjit/packet.h>
#include <gsl/span>

#include "axel/Bvh.h"
#include "axel/Ray.h"
#include "axel/common/VectorizationTypes.h"

namespace axel {

inline constexpr double kDefaultBoundingBoxThickness{0.0};

/**
 * @brief High-performance triangle BVH with SIMD optimizations
 *
 * This implementation provides significant performance improvements through
 * SIMD-optimized triangle processing and 8-way tree splits for improved
 * cache efficiency and parallel processing.
 *
 * ## SIMD Optimization Strategy
 *
 * The core optimization comes from two key data structure changes:
 *
 * 1. **Triangle Blocks**: Triangles are packed into SIMD-aligned blocks where
 *    each block contains 4-8 triangles (depending on scalar type). This enables
 *    vectorized ray-triangle intersection tests across multiple triangles simultaneously.
 *
 * 2. **8-way Tree Nodes**: Each internal node has exactly 8 children, allowing
 *    SIMD bounding box intersection tests to process all 8 children in parallel
 *    using 256-bit SIMD instructions.
 *
 * ## Performance Benefits
 *
 * - **Ray Intersection**: Process 4-8 triangles per SIMD instruction
 * - **Bounding Box Tests**: Process 8 child boxes per SIMD instruction
 * - **Cache Efficiency**: 8-way splits reduce tree depth and improve locality
 * - **Memory Layout**: SIMD-aligned data structures optimize memory access patterns
 *
 * @tparam S Scalar type (float or double). Note: double precision may have
 *           reduced SIMD efficiency compared to float.
 *
 * Performance Notes:
 * - Float precision: Optimal SIMD performance with native lane width
 * - Double precision: Reduced SIMD width but maintains precision
 * - 8-way tree splits provide better cache efficiency than binary trees
 * - Triangle blocks are SIMD-aligned for optimal memory access
 */
template <typename S>
class TriBvh {
 public:
  using Scalar = S;
  using SizeType = uint32_t;
  using QueryBuffer = typename BvhBase<S>::QueryBuffer;
  using ClosestSurfacePointResult = ::axel::ClosestSurfacePointResult<S>;

  // Compile-time scalar type validation
  static_assert(
      std::is_same_v<S, float> || std::is_same_v<S, double>,
      "TriBvh only supports float and double scalar types");

 private:
  // SIMD type definitions for optimized triangle processing
  // Using axel vectorization types for consistency with ray-triangle intersection
  using FloatP = WideScalar<S, kNativeLaneWidth<S>>;
  using IntP = drjit::Packet<int32_t, kNativeLaneWidth<S>>;
  using UIntP = drjit::Packet<uint32_t, kNativeLaneWidth<S>>;
  using Vector3P = WideVec3<S, kNativeLaneWidth<S>>;

  // Tree-specific SIMD types for 8-way splits
  static constexpr int TreeSplit = 8;
  static constexpr int TriangleBlockSize = kNativeLaneWidth<S>;
  using IntSP = drjit::Packet<int32_t, TreeSplit>;
  using FloatSP = drjit::Packet<S, TreeSplit>;
  using Vector3SP = drjit::Array<FloatSP, 3>;

  // SIMD width information for debugging/profiling
  static constexpr size_t SimdWidth = kNativeLaneWidth<S>;
  static constexpr size_t NativeLaneWidth = kNativeLaneWidth<S>;

  // Invalid index constants for better code readability
  static constexpr uint32_t kInvalidNodeIdx = std::numeric_limits<uint32_t>::max();
  static constexpr uint32_t kInvalidTriangleBlockIdx = std::numeric_limits<uint32_t>::max();

  /**
   * @brief Triangle storage optimized for SIMD operations
   *
   * To enable efficient SIMD processing, triangles are stored in blocks where each block
   * contains exactly TriangleBlockSize triangles (typically 4-8 depending on scalar type).
   * This allows vectorized operations across multiple triangles simultaneously.
   *
   * Memory Layout Example (for TriangleBlockSize = 4):
   * triPos[0] = [v0x_tri0, v0x_tri1, v0x_tri2, v0x_tri3] (vertex 0, x-coordinates)
   * triPos[1] = [v1x_tri0, v1x_tri1, v1x_tri2, v1x_tri3] (vertex 1, x-coordinates)
   * triPos[2] = [v2x_tri0, v2x_tri1, v2x_tri2, v2x_tri3] (vertex 2, x-coordinates)
   *
   * Each triPos[i] is a Vector3P containing [x, y, z] SIMD vectors.
   *
   * Empty slots in incomplete blocks are filled with FAR_VALUE positions and
   * INT32_MAX indices to ensure safe SIMD operations without branches.
   */
  struct TriangleBlock {
    std::array<Vector3P, 3> triPos; // The three vertices of each triangle, SIMD-packed
                                    // [vertex0_simd, vertex1_simd, vertex2_simd]
    UIntP triIndex; // Triangle indices in SIMD format (unsigned for consistency)
                    // [index0, index1, index2, index3, ...] up to TriangleBlockSize
  };

  /**
   * @brief 8-way tree node optimized for SIMD bounding box tests
   *
   * Each internal node has exactly 8 children, corresponding to an octree-style spatial
   * subdivision. This design enables SIMD processing of 8 bounding box intersection
   * tests simultaneously using 256-bit SIMD instructions.
   *
   * The 8-way split provides several advantages:
   * - SIMD efficiency: Process 8 box tests in parallel
   * - Cache efficiency: Better spatial locality than binary trees
   * - Reduced tree depth: Fewer levels to traverse
   *
   * Bounding boxes are stored as separate min/max SIMD vectors rather than
   * Eigen::AlignedBox objects to avoid SIMD type instantiation issues.
   */
  struct Node {
    Node(
        const Vector3SP& bboxMin_in,
        const Vector3SP& bboxMax_in,
        const std::array<SizeType, TreeSplit>& children_in);

    Node(SizeType trianglesStart_in, SizeType trianglesEnd_in);

    // SIMD bounding boxes for the 8 children
    // Stored as separate min/max vectors for SIMD compatibility
    // childBoxMin[0] = [child0_minX, child1_minX, ..., child7_minX]
    // childBoxMin[1] = [child0_minY, child1_minY, ..., child7_minY]
    // childBoxMin[2] = [child0_minZ, child1_minZ, ..., child7_minZ]
    Vector3SP childBoxMin;
    Vector3SP childBoxMax;

    // Indices of the 8 child nodes in the nodes_ array
    // Missing/invalid children are marked with kInvalidNodeIdx
    std::array<SizeType, TreeSplit> children{};

    [[nodiscard]] bool isLeaf() const {
      return trianglesStart != kInvalidTriangleBlockIdx;
    }

    // For leaf nodes: range of triangle blocks in triangleBlocks_ array
    // These indices define which TriangleBlocks belong to this leaf
    SizeType trianglesStart = kInvalidTriangleBlockIdx; // Start index (inclusive)
    SizeType trianglesEnd = kInvalidTriangleBlockIdx; // End index (exclusive)
  };

 public:
  TriBvh() = default;

  /**
   * @brief Construct a new Bvh from the given positions and triangles. The constructor assumes
   * that the integrity of provided positions and triangles is checked beforehand.
   * This is the backwards-compatible constructor that takes Eigen matrices.
   */
  explicit TriBvh(
      const Eigen::MatrixX3<S>& positions,
      const Eigen::MatrixX3i& triangles,
      const std::optional<S>& boundingBoxThickness = kDefaultBoundingBoxThickness);

  /**
   * @brief Construct a new Bvh from the given positions and triangles using gsl::span inputs.
   * This constructor provides maximum flexibility by accepting spans, allowing BVH construction
   * from both complete containers (std::vector, arrays) and subranges of existing data
   * without requiring data copying or extraction.
   *
   * This is particularly useful when you have a large mesh but only want to build a BVH
   * for a specific subset of triangles and vertices, or when working with std::vector inputs
   * (which are automatically converted to spans).
   *
   * @param vertices Span of vertex positions (works with std::vector, arrays, subranges)
   * @param triangles Span of triangle indices (indices must be valid within the vertices span)
   * @param boundingBoxThickness Optional thickness to add to bounding boxes
   */
  explicit TriBvh(
      gsl::span<const Eigen::Vector3<S>> vertices,
      gsl::span<const Eigen::Vector3i> triangles,
      const std::optional<S>& boundingBoxThickness = kDefaultBoundingBoxThickness);

  /**
   * @brief Performs a query with a given bounding box.
   * All primitives whose bounding box intersects with the query get returned as a list of their
   * IDs, e.g. the triangle indices.
   */
  std::vector<uint32_t> boxQuery(const BoundingBox<S>& box) const;

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
  std::vector<uint32_t> lineHits(const Ray3<S>& ray) const;

  /**
   * @brief Returns the closest hit with the given query ray.
   * If the ray hits nothing, returns a std::nullopt.
   */
  std::optional<IntersectionResult<S>> closestHit(const Ray3<S>& ray) const;

  /**
   * @brief Returns all hits with the given query ray.
   */
  std::vector<IntersectionResult<S>> allHits(const Ray3<S>& ray) const;

  /**
   * @brief Checks whether the given ray intersects any of the primitives in the Bvh.
   */
  bool anyHit(const Ray3<S>& ray) const;

  /**
   * @brief Find the closest surface point with optional maximum distance threshold for early exit.
   * This optimization allows the search to exit early when we only care about nearby points.
   *
   * @param query The query point
   * @param maxDist Maximum distance to search. Points further than this will be ignored.
   *                Use std::nullopt (default) for unlimited search.
   * @return ClosestSurfacePointResult with valid result only if found within maxDist
   */
  [[nodiscard]] ClosestSurfacePointResult closestSurfacePoint(
      const Eigen::Vector3<S>& query,
      const std::optional<S>& maxDist = std::nullopt) const;

  /**
   * @brief Returns the total number of internal nodes in the tree.
   */
  size_t getNodeCount() const;

  /**
   * @brief Returns the total number of primitives in the tree. It should be equal to the number of
   * triangles that Bvh was constructed with.
   */
  size_t getPrimitiveCount() const;

  // Validation function for debugging tree construction
  void validate(
      const std::vector<Eigen::Vector3i>& triangles,
      const std::vector<Eigen::Vector3<S>>& vertices) const;

  // Debug function to print tree statistics
  void printTreeStats() const;

 private:
  // SIMD-optimized data structures for high-performance triangle processing
  std::vector<TriangleBlock> triangleBlocks_;
  std::vector<Node> nodes_;
  SizeType root_ = kInvalidNodeIdx;
  SizeType numTriangles_ = 0;
  SizeType depth_ = 0;
  Eigen::AlignedBox<S, 3> boundingBox_;

  // Templated helper methods for tree building
  template <typename TriangleContainer, typename VertexContainer>
  std::pair<SizeType, Eigen::AlignedBox<S, 3>> split(
      const TriangleContainer& triangles,
      const VertexContainer& vertices,
      std::vector<std::tuple<uint32_t, Eigen::Vector3<S>>>& triIndices,
      SizeType start,
      SizeType end,
      SizeType depth,
      S boundingBoxThickness);

  // SIMD intersection helper methods for ray traversal
  [[nodiscard]] std::tuple<typename IntSP::MaskType, FloatSP, FloatSP> intersectBBox(
      const Ray3<S>& ray,
      const Vector3SP& boxMin,
      const Vector3SP& boxMax,
      const S& t0,
      const S& t1) const;

  [[nodiscard]] std::pair<FloatP, Vector3P> intersectTriangle(
      const Ray3<S>& ray,
      const Vector3P& v0,
      const Vector3P& v1,
      const Vector3P& v2) const;

  // SIMD helper methods for closest surface point queries
  [[nodiscard]] FloatSP pointToBoxDistanceSquared(
      const Eigen::Vector3<S>& query,
      const Vector3SP& boxMin,
      const Vector3SP& boxMax) const;

  [[nodiscard]] std::tuple<S, Eigen::Vector3<S>, Eigen::Vector3<S>>
  pointToTriangleDistanceSquaredScalar(
      const Eigen::Vector3<S>& query,
      const Eigen::Vector3<S>& v0,
      const Eigen::Vector3<S>& v1,
      const Eigen::Vector3<S>& v2) const;

  [[nodiscard]] std::tuple<FloatP, Vector3P, Vector3P> pointToTriangleDistanceSquared(
      const Eigen::Vector3<S>& query,
      const Vector3P& v0,
      const Vector3P& v1,
      const Vector3P& v2) const;

  // SIMD helper method for box query operations
  [[nodiscard]] typename IntSP::MaskType intersectBoxWithBox(
      const BoundingBox<S>& queryBox,
      const Vector3SP& boxMin,
      const Vector3SP& boxMax) const;
};

using TriBvhf = TriBvh<float>;
using TriBvhd = TriBvh<double>;

extern template class TriBvh<float>;
extern template class TriBvh<double>;

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
