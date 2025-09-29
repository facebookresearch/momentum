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
#include "axel/math/RayBoxIntersection.h"
#include "axel/math/RayTriangleIntersection.h"

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <set>

namespace axel {

namespace {

// Anonymous helper functions for accessing triangles and vertices from different container types
Eigen::Vector3i getTriangle(gsl::span<const Eigen::Vector3i> triangles, size_t index) {
  return triangles[index];
}

Eigen::Vector3i getTriangle(const Eigen::MatrixX3i& triangles, size_t index) {
  return triangles.row(index);
}

template <typename S>
Eigen::Vector3<S> getVertex(gsl::span<const Eigen::Vector3<S>> vertices, size_t index) {
  return vertices[index];
}

template <typename S>
Eigen::Vector3<S> getVertex(const Eigen::MatrixX3<S>& vertices, size_t index) {
  return vertices.row(index);
}

} // anonymous namespace

constexpr float kFarValueFloat = 1e10f;

// Constants for optimized tree building
constexpr size_t kMaxDepth = 16;
constexpr size_t kLeafBlocks = 4;

// Node constructors for 8-way tree structure
template <typename S>
TriBvh<S>::Node::Node(
    const Vector3SP& bboxMin_in,
    const Vector3SP& bboxMax_in,
    const std::array<SizeType, TreeSplit>& children_in)
    : childBoxMin(bboxMin_in),
      childBoxMax(bboxMax_in),
      children(children_in),
      trianglesStart(kInvalidTriangleBlockIdx),
      trianglesEnd(kInvalidTriangleBlockIdx) {}

template <typename S>
TriBvh<S>::Node::Node(const SizeType trianglesStart_in, const SizeType trianglesEnd_in)
    : trianglesStart(trianglesStart_in), trianglesEnd(trianglesEnd_in) {
  std::fill(children.begin(), children.end(), kInvalidNodeIdx);
  for (int i = 0; i < 3; ++i) {
    childBoxMin[i] = FloatSP(static_cast<S>(kFarValueFloat));
  }

  for (int i = 0; i < 3; ++i) {
    childBoxMax[i] = FloatSP(static_cast<S>(kFarValueFloat));
  }
}

// Helper function to compute triangle center - templated for different container types
template <typename S, typename TriangleContainer, typename VertexContainer>
Eigen::Vector3<S> computeTriangleCenter(
    const TriangleContainer& triangles,
    size_t triangleIndex,
    const VertexContainer& vertices) {
  const auto triangle = getTriangle(triangles, triangleIndex);
  Eigen::Vector3<S> result = getVertex<S>(vertices, triangle[0]);

  for (int i = 1; i < 3; ++i) {
    result += (getVertex<S>(vertices, triangle[i]) - result) / (i + 1);
  }

  return result;
}

// Backwards-compatible constructor with Eigen matrices
template <typename S>
TriBvh<S>::TriBvh(
    const Eigen::MatrixX3<S>& positions,
    const Eigen::MatrixX3i& triangles,
    const std::optional<S>& boundingBoxThickness)
    : depth_(0) {
  numTriangles_ = static_cast<uint32_t>(triangles.rows());
  if (numTriangles_ == 0) {
    return;
  }

  // Validate that we're getting reasonable SIMD width
  static_assert(TriangleBlockSize >= 1, "SIMD width must be at least 1");

  // Store bounding box thickness for use during tree construction
  S bboxThickness = boundingBoxThickness.value_or(static_cast<S>(0));
  if (bboxThickness < static_cast<S>(0)) {
    bboxThickness = static_cast<S>(0);
  }

  // Create triangle indices with centers for spatial partitioning
  std::vector<std::tuple<uint32_t, Eigen::Vector3<S>>> triangleIndices;
  triangleIndices.reserve(numTriangles_);
  for (uint32_t iTri = 0; iTri < numTriangles_; ++iTri) {
    triangleIndices.emplace_back(iTri, computeTriangleCenter<S>(triangles, iTri, positions));
  }

  // Build the tree using recursive splitting - no conversion needed
  std::tie(root_, boundingBox_) =
      split(triangles, positions, triangleIndices, 0, numTriangles_, 0, bboxThickness);
}

// Flexible constructor with gsl::span inputs for maximum compatibility
template <typename S>
TriBvh<S>::TriBvh(
    gsl::span<const Eigen::Vector3<S>> vertices,
    gsl::span<const Eigen::Vector3i> triangles,
    const std::optional<S>& boundingBoxThickness)
    : depth_(0) {
  numTriangles_ = static_cast<uint32_t>(triangles.size());
  if (numTriangles_ == 0) {
    return;
  }

  // Validate that we're getting reasonable SIMD width
  static_assert(TriangleBlockSize >= 1, "SIMD width must be at least 1");

  // Store bounding box thickness for use during tree construction
  S bboxThickness = boundingBoxThickness.value_or(static_cast<S>(0));
  if (bboxThickness < static_cast<S>(0)) {
    bboxThickness = static_cast<S>(0);
  }

  // Create triangle indices with centers for spatial partitioning
  // gsl::span works seamlessly with our templated helper functions
  std::vector<std::tuple<uint32_t, Eigen::Vector3<S>>> triangleIndices;
  triangleIndices.reserve(numTriangles_);
  for (uint32_t iTri = 0; iTri < numTriangles_; ++iTri) {
    triangleIndices.emplace_back(iTri, computeTriangleCenter<S>(triangles, iTri, vertices));
  }

  // Build the tree using recursive splitting - gsl::span works with our templated split function
  std::tie(root_, boundingBox_) =
      split(triangles, vertices, triangleIndices, 0, numTriangles_, 0, bboxThickness);
}

// Tree building split function with 8-way spatial subdivision - now templated for different
// container types
template <typename S>
template <typename TriangleContainer, typename VertexContainer>
std::pair<typename TriBvh<S>::SizeType, Eigen::AlignedBox<S, 3>> TriBvh<S>::split(
    const TriangleContainer& triangles,
    const VertexContainer& vertices,
    std::vector<std::tuple<uint32_t, Eigen::Vector3<S>>>& triIndices,
    const SizeType start,
    const SizeType end,
    SizeType depth,
    S boundingBoxThickness) {
  XR_CHECK(end > start, "Invalid range for split");

  depth_ = std::max(depth, depth_);

  // Compute bounding box for this range of triangles
  Eigen::AlignedBox<S, 3> box;
  for (SizeType i = start; i != end; ++i) {
    const auto [triIndex, triCenter] = triIndices[i];
    const auto tri = getTriangle(triangles, triIndex);
    for (int j = 0; j < 3; ++j) {
      box.extend(getVertex(vertices, tri[j]));
    }
  }

  // Apply bounding box thickness to avoid zero-thickness boxes
  if (boundingBoxThickness > static_cast<S>(0)) {
    const Eigen::Vector3<S> thickness = Eigen::Vector3<S>::Constant(boundingBoxThickness);
    box.min() -= thickness;
    box.max() += thickness;
  }

  const SizeType nBlocks = (end - start) / TriangleBlockSize;

  // Create leaf node if we've reached depth limit or have few enough triangles
  if (depth_ == kMaxDepth || nBlocks <= kLeafBlocks) {
    const auto node_id = static_cast<SizeType>(nodes_.size());

    auto blocksStart = static_cast<SizeType>(triangleBlocks_.size());

    // Pack triangles into SIMD blocks
    for (SizeType curBlockStart = start; curBlockStart < end; curBlockStart += TriangleBlockSize) {
      // Initialize triangle block with vectorized types directly
      TriangleBlock triangleBlock;

      // Initialize with far values and invalid indices
      triangleBlock.triIndex = UIntP(kInvalidTriangleIdx);

      for (int iTriVert = 0; iTriVert < 3; ++iTriVert) {
        triangleBlock.triPos[iTriVert] = Vector3P(
            FloatP(static_cast<S>(kFarValueFloat)),
            FloatP(static_cast<S>(kFarValueFloat)),
            FloatP(static_cast<S>(kFarValueFloat)));
      }

      const SizeType curBlockEnd = std::min(curBlockStart + TriangleBlockSize, end);
      for (SizeType curTri = curBlockStart; curTri != curBlockEnd; ++curTri) {
        const SizeType offset = curTri - curBlockStart;
        const auto [triIndex, triCenter] = triIndices[curTri];

        // Set triangle index directly in vectorized format
        triangleBlock.triIndex[offset] = triIndex;

        const auto tri = getTriangle(triangles, triIndex);
        for (int jTriVert = 0; jTriVert < 3; ++jTriVert) {
          const auto vertex = getVertex(vertices, tri[jTriVert]);
          // Pack vertex coordinates directly into vectorized types
          triangleBlock.triPos[jTriVert][0][offset] = vertex[0];
          triangleBlock.triPos[jTriVert][1][offset] = vertex[1];
          triangleBlock.triPos[jTriVert][2][offset] = vertex[2];
        }
      }

      triangleBlocks_.push_back(triangleBlock);
    }

    auto blocksEnd = static_cast<SizeType>(triangleBlocks_.size());

    nodes_.push_back(Node(blocksStart, blocksEnd));
    return std::make_pair(node_id, box);
  }

  // Create 8-way split for internal node
  const Eigen::Vector3<S> low = Eigen::Vector3<S>::Constant(-std::numeric_limits<S>::max());
  const Eigen::Vector3<S> mid = box.center();
  const Eigen::Vector3<S> high = Eigen::Vector3<S>::Constant(std::numeric_limits<S>::max());

  std::array<Eigen::AlignedBox<S, 3>, TreeSplit> boxes = {
      Eigen::AlignedBox<S, 3>(
          Eigen::Vector3<S>(low.x(), low.y(), low.z()),
          Eigen::Vector3<S>(mid.x(), mid.y(), mid.z())),
      Eigen::AlignedBox<S, 3>(
          Eigen::Vector3<S>(low.x(), low.y(), mid.z()),
          Eigen::Vector3<S>(mid.x(), mid.y(), high.z())),
      Eigen::AlignedBox<S, 3>(
          Eigen::Vector3<S>(low.x(), mid.y(), low.z()),
          Eigen::Vector3<S>(mid.x(), high.y(), mid.z())),
      Eigen::AlignedBox<S, 3>(
          Eigen::Vector3<S>(low.x(), mid.y(), mid.z()),
          Eigen::Vector3<S>(mid.x(), high.y(), high.z())),
      Eigen::AlignedBox<S, 3>(
          Eigen::Vector3<S>(mid.x(), low.y(), low.z()),
          Eigen::Vector3<S>(high.x(), mid.y(), mid.z())),
      Eigen::AlignedBox<S, 3>(
          Eigen::Vector3<S>(mid.x(), low.y(), mid.z()),
          Eigen::Vector3<S>(high.x(), mid.y(), high.z())),
      Eigen::AlignedBox<S, 3>(
          Eigen::Vector3<S>(mid.x(), mid.y(), low.z()),
          Eigen::Vector3<S>(high.x(), high.y(), mid.z())),
      Eigen::AlignedBox<S, 3>(
          Eigen::Vector3<S>(mid.x(), mid.y(), mid.z()),
          Eigen::Vector3<S>(high.x(), high.y(), high.z()))};

  std::array<SizeType, TreeSplit> children{};

  SizeType lastEnd = start;
  for (uint32_t iBox = 0; iBox < TreeSplit; ++iBox) {
    const auto it = std::partition(
        triIndices.begin() + lastEnd,
        triIndices.begin() + end,
        [&](const std::tuple<uint32_t, Eigen::Vector3<S>>& triIndexAndCenter) -> bool {
          const auto& [triIndex, triCenter] = triIndexAndCenter;
          return boxes[iBox].contains(triCenter);
        });

    const auto curEnd = static_cast<uint32_t>(std::distance(triIndices.begin(), it));

    // No triangles in current child:
    if (lastEnd == curEnd) {
      children[iBox] = kInvalidNodeIdx;
      // Make an empty box:
      boxes[iBox] = Eigen::AlignedBox<S, 3>(mid, mid);
    } else {
      // Recursively split this child
      std::tie(children[iBox], boxes[iBox]) =
          split(triangles, vertices, triIndices, lastEnd, curEnd, depth + 1, boundingBoxThickness);
    }
    lastEnd = curEnd;
  }
  XR_CHECK(lastEnd == end, "Triangle partitioning failed");

  // Create SIMD bounding box data
  Vector3SP bbox_min;
  Vector3SP bbox_max;
  for (int i = 0; i < 3; ++i) {
    for (uint32_t j = 0; j < TreeSplit; ++j) {
      bbox_min[i][j] = boxes[j].min()[i];
      bbox_max[i][j] = boxes[j].max()[i];
    }
  }

  const auto node_id = static_cast<SizeType>(nodes_.size());
  nodes_.push_back(Node(bbox_min, bbox_max, children));

  return std::make_pair(node_id, box);
}

// SIMD bounding box intersection test
template <typename S>
typename TriBvh<S>::IntSP::MaskType TriBvh<S>::intersectBoxWithBox(
    const BoundingBox<S>& queryBox,
    const Vector3SP& boxMin,
    const Vector3SP& boxMax) const {
  // Convert query box to SIMD format for comparison
  const Vector3SP queryMin = Vector3SP(
      FloatSP(queryBox.aabb.min().x()),
      FloatSP(queryBox.aabb.min().y()),
      FloatSP(queryBox.aabb.min().z()));

  const Vector3SP queryMax = Vector3SP(
      FloatSP(queryBox.aabb.max().x()),
      FloatSP(queryBox.aabb.max().y()),
      FloatSP(queryBox.aabb.max().z()));

  // Test intersection on all three axes
  auto intersects = typename IntSP::MaskType(true);

  for (int i = 0; i < 3; ++i) {
    // Boxes intersect on axis i if:
    // queryMin[i] <= boxMax[i] && queryMax[i] >= boxMin[i]
    const auto axisIntersects = (queryMin[i] <= boxMax[i]) & (queryMax[i] >= boxMin[i]);
    intersects = intersects & axisIntersects;
  }

  return intersects;
}

template <typename S>
std::vector<uint32_t> TriBvh<S>::boxQuery(const BoundingBox<S>& box) const {
  std::vector<uint32_t> results;

  if (numTriangles_ == 0 || root_ == kInvalidNodeIdx) {
    return results;
  }

  // Use explicit stack for traversal
  const size_t kMaxStackSize = (TreeSplit - 1) * kMaxDepth + 1;
  std::array<SizeType, kMaxStackSize> nodeStack{};

  // Start with root on the stack
  nodeStack[0] = root_;
  SizeType stackSize = 1;

  while (stackSize != 0) {
    // Pop the top of the stack
    --stackSize;
    const SizeType cur = nodeStack[stackSize];
    const auto& cur_node = nodes_[cur];

    if (cur_node.isLeaf()) {
      // Process leaf node - test individual triangles against query box
      const SizeType triBlocksStart = cur_node.trianglesStart;
      const SizeType triBlocksEnd = cur_node.trianglesEnd;

      for (SizeType iBlock = triBlocksStart; iBlock < triBlocksEnd; ++iBlock) {
        const auto& triBlock = triangleBlocks_[iBlock];

        // Vectorized triangle-box intersection test for the entire block

        // Compute triangle bounding boxes for the entire block using SIMD
        Vector3P triangleBoxMin = triBlock.triPos[0];
        Vector3P triangleBoxMax = triBlock.triPos[0];

        // Extend with other vertices
        for (int vert = 1; vert < 3; ++vert) {
          triangleBoxMin = drjit::minimum(triangleBoxMin, triBlock.triPos[vert]);
          triangleBoxMax = drjit::maximum(triangleBoxMax, triBlock.triPos[vert]);
        }

        // Test all triangle bounding boxes against query box using SIMD
        const Vector3P queryMin = Vector3P(
            FloatP(box.aabb.min().x()), FloatP(box.aabb.min().y()), FloatP(box.aabb.min().z()));

        const Vector3P queryMax = Vector3P(
            FloatP(box.aabb.max().x()), FloatP(box.aabb.max().y()), FloatP(box.aabb.max().z()));

        // Test intersection on all three axes
        auto intersects = typename FloatP::MaskType(true);
        for (int i = 0; i < 3; ++i) {
          // Boxes intersect on axis i if:
          // queryMin[i] <= triangleBoxMax[i] && queryMax[i] >= triangleBoxMin[i]
          const auto axisIntersects =
              (queryMin[i] <= triangleBoxMax[i]) & (queryMax[i] >= triangleBoxMin[i]);
          intersects = intersects & axisIntersects;
        }

        // Also need to mask out invalid triangles
        const auto validTriangles = (triBlock.triIndex != UIntP(kInvalidTriangleIdx));
        intersects = intersects & validTriangles;

        // Only unpack if any triangles intersect
        if (drjit::any(intersects)) {
          for (uint32_t i = 0; i < TriangleBlockSize; ++i) {
            if (intersects[i]) {
              results.push_back(static_cast<uint32_t>(triBlock.triIndex[i]));
            }
          }
        }
      }

    } else {
      // Process internal node - test bounding box intersections
      const auto mask = intersectBoxWithBox(box, cur_node.childBoxMin, cur_node.childBoxMax);

      // Add intersecting children to stack
      for (int i = 0; i < TreeSplit; ++i) {
        if (mask[i] && cur_node.children[i] != kInvalidNodeIdx) {
          nodeStack[stackSize] = cur_node.children[i];
          ++stackSize;
        }
      }
    }
  }

  // Remove duplicates and sort
  std::sort(results.begin(), results.end());
  results.erase(std::unique(results.begin(), results.end()), results.end());

  return results;
}

template <typename S>
uint32_t TriBvh<S>::boxQuery(const BoundingBox<S>& box, QueryBuffer& hits) const {
  if (numTriangles_ == 0 || root_ == kInvalidNodeIdx) {
    return 0;
  }

  // Use explicit stack for traversal
  const size_t kMaxStackSize = (TreeSplit - 1) * kMaxDepth + 1;
  std::array<SizeType, kMaxStackSize> nodeStack{};

  // Start with root on the stack
  nodeStack[0] = root_;
  SizeType stackSize = 1;

  uint32_t hitCount = 0;
  const auto maxHits = static_cast<uint32_t>(hits.size());

  while (stackSize != 0 && hitCount < maxHits) {
    // Pop the top of the stack
    --stackSize;
    const SizeType cur = nodeStack[stackSize];
    const auto& cur_node = nodes_[cur];

    if (cur_node.isLeaf()) {
      // Process leaf node - test individual triangles against query box up to buffer limit
      const SizeType triBlocksStart = cur_node.trianglesStart;
      const SizeType triBlocksEnd = cur_node.trianglesEnd;

      for (SizeType iBlock = triBlocksStart; iBlock < triBlocksEnd && hitCount < maxHits;
           ++iBlock) {
        const auto& triBlock = triangleBlocks_[iBlock];

        // Vectorized triangle-box intersection test for the entire block

        // Compute triangle bounding boxes for the entire block using SIMD
        Vector3P triangleBoxMin = triBlock.triPos[0];
        Vector3P triangleBoxMax = triBlock.triPos[0];

        // Extend with other vertices
        for (int vert = 1; vert < 3; ++vert) {
          triangleBoxMin = drjit::minimum(triangleBoxMin, triBlock.triPos[vert]);
          triangleBoxMax = drjit::maximum(triangleBoxMax, triBlock.triPos[vert]);
        }

        // Test all triangle bounding boxes against query box using SIMD
        const Vector3P queryMin = Vector3P(
            FloatP(box.aabb.min().x()), FloatP(box.aabb.min().y()), FloatP(box.aabb.min().z()));

        const Vector3P queryMax = Vector3P(
            FloatP(box.aabb.max().x()), FloatP(box.aabb.max().y()), FloatP(box.aabb.max().z()));

        // Test intersection on all three axes
        auto intersects = typename FloatP::MaskType(true);
        for (int i = 0; i < 3; ++i) {
          // Boxes intersect on axis i if:
          // queryMin[i] <= triangleBoxMax[i] && queryMax[i] >= triangleBoxMin[i]
          const auto axisIntersects =
              (queryMin[i] <= triangleBoxMax[i]) & (queryMax[i] >= triangleBoxMin[i]);
          intersects = intersects & axisIntersects;
        }

        // Also need to mask out invalid triangles
        const auto validTriangles = (triBlock.triIndex != UIntP(kInvalidTriangleIdx));
        intersects = intersects & validTriangles;

        // Only unpack if any triangles intersect, and up to buffer limit
        if (drjit::any(intersects)) {
          for (uint32_t i = 0; i < TriangleBlockSize && hitCount < maxHits; ++i) {
            if (intersects[i]) {
              hits[hitCount] = static_cast<uint32_t>(triBlock.triIndex[i]);
              ++hitCount;
            }
          }
        }
      }

    } else {
      // Process internal node - test bounding box intersections
      const auto mask = intersectBoxWithBox(box, cur_node.childBoxMin, cur_node.childBoxMax);

      // Add intersecting children to stack
      for (int i = 0; i < TreeSplit; ++i) {
        if (mask[i] && cur_node.children[i] != kInvalidNodeIdx) {
          nodeStack[stackSize] = cur_node.children[i];
          ++stackSize;
        }
      }
    }
  }

  return hitCount;
}

template <typename S>
std::vector<uint32_t> TriBvh<S>::lineHits(const Ray3<S>& ray) const {
  std::vector<uint32_t> results;

  if (numTriangles_ == 0 || root_ == kInvalidNodeIdx) {
    return results;
  }

  // Use explicit stack for traversal (similar to ray intersection but collect triangle indices)
  const size_t kMaxStackSize = (TreeSplit - 1) * kMaxDepth + 1;

  struct NodeElt {
    SizeType nodeIdx;
    S near_tVal;
  };

  std::array<NodeElt, kMaxStackSize> nodeStack{};

  // Start with root on the stack
  nodeStack[0].nodeIdx = root_;
  nodeStack[0].near_tVal = ray.minT;
  SizeType stackSize = 1;

  // Use a large maxT for line hits (we want all triangles the line passes through)
  const S line_maxT = std::numeric_limits<S>::max();

  while (stackSize != 0) {
    // Pop the top of the stack
    --stackSize;
    const SizeType cur = nodeStack[stackSize].nodeIdx;
    const S cur_min_t = nodeStack[stackSize].near_tVal;

    if (cur_min_t >= line_maxT) {
      continue;
    }

    const auto& cur_node = nodes_[cur];

    if (cur_node.isLeaf()) {
      // Process leaf node - collect all triangle indices in blocks that intersect the ray line
      const SizeType triBlocksStart = cur_node.trianglesStart;
      const SizeType triBlocksEnd = cur_node.trianglesEnd;

      for (SizeType iBlock = triBlocksStart; iBlock < triBlocksEnd; ++iBlock) {
        const auto& triBlock = triangleBlocks_[iBlock];

        // For lineHits, we don't need to do actual triangle intersection
        // We just collect all valid triangle indices in blocks that the ray reaches
        for (uint32_t i = 0; i < TriangleBlockSize; ++i) {
          if (triBlock.triIndex[i] != kInvalidTriangleIdx) {
            results.push_back(static_cast<uint32_t>(triBlock.triIndex[i]));
          }
        }
      }

    } else {
      // Process internal node - test bounding boxes
      const auto [mask, tMin, tMax] =
          intersectBBox(ray, cur_node.childBoxMin, cur_node.childBoxMax, ray.minT, line_maxT);

      // Missed all boxes
      if (!drjit::any(mask)) {
        continue;
      }

      // Add children to stack
      for (int i = 0; i < TreeSplit; ++i) {
        if (!mask[i]) {
          continue;
        }

        if (cur_node.children[i] == kInvalidNodeIdx) {
          continue;
        }

        nodeStack[stackSize].nodeIdx = cur_node.children[i];
        nodeStack[stackSize].near_tVal = tMin[i];
        ++stackSize;
      }
    }
  }

  // Remove duplicates and sort (triangle indices might be duplicated if they appear in multiple
  // blocks)
  std::sort(results.begin(), results.end());
  results.erase(std::unique(results.begin(), results.end()), results.end());

  return results;
}

// SIMD bounding box intersection function using our tested ray-box intersection implementation
template <typename S>
std::tuple<
    typename TriBvh<S>::IntSP::MaskType,
    typename TriBvh<S>::FloatSP,
    typename TriBvh<S>::FloatSP>
TriBvh<S>::intersectBBox(
    const Ray3<S>& ray,
    const Vector3SP& boxMin,
    const Vector3SP& boxMax,
    const S& t0,
    const S& t1) const {
  // Create a ray with the proper parameter range for intersection testing
  Ray3<S> testRay = ray;
  testRay.minT = t0;
  testRay.maxT = t1;

  // Use our tested ray-box intersection function directly with Vector3SP
  auto [mask, tMin, tMax] = intersectRayBoxes<S, TreeSplit>(testRay, boxMin, boxMax);

  return std::make_tuple(mask, tMin, tMax);
}

// SIMD triangle intersection function using the corrected wide vector implementation
template <typename S>
std::pair<typename TriBvh<S>::FloatP, typename TriBvh<S>::Vector3P> TriBvh<S>::intersectTriangle(
    const Ray3<S>& ray,
    const Vector3P& v0,
    const Vector3P& v1,
    const Vector3P& v2) const {
  // Use the corrected wide vector ray-triangle intersection with axel types
  Vector3P intersectionPoint;
  FloatP t, u, v;

  // Call with default template parameters that match the explicit instantiations
  auto hitMask = rayTriangleIntersectWide<S>(
      ray.origin, ray.direction, v0, v1, v2, intersectionPoint, t, u, v);

  // Apply ray t-range constraints
  auto tInRange = (t >= FloatP(ray.minT)) & (t <= FloatP(ray.maxT));
  auto validMask = hitMask & tInRange;

  // Set invalid intersections to max value
  t = drjit::select(validMask, t, FloatP(std::numeric_limits<S>::max()));

  // Compute barycentric coordinates (w, u, v) where w = 1 - u - v
  const auto w = FloatP(static_cast<S>(1)) - u - v;

  return {t, Vector3P(w, u, v)};
}

template <typename S>
std::optional<IntersectionResult<S>> TriBvh<S>::closestHit(const Ray3<S>& ray) const {
  if (numTriangles_ == 0 || root_ == kInvalidNodeIdx) {
    return std::nullopt;
  }

  // Use explicit stack for traversal with optimized 8-way tree structure
  const size_t kMaxStackSize = (TreeSplit - 1) * kMaxDepth + 1;

  struct NodeElt {
    SizeType nodeIdx;
    S near_tVal;
  };

  std::array<NodeElt, kMaxStackSize> nodeStack{};

  // Start with root on the stack
  nodeStack[0].nodeIdx = root_;
  nodeStack[0].near_tVal = ray.minT;
  SizeType stackSize = 1;

  S closest_t = ray.maxT;
  uint32_t closest_t_index = std::numeric_limits<uint32_t>::max();
  Eigen::Vector3<S> closest_t_bary = Eigen::Vector3<S>::Zero();

  while (stackSize != 0) {
    // Pop the top of the stack
    --stackSize;
    const SizeType cur = nodeStack[stackSize].nodeIdx;
    const S cur_min_t = nodeStack[stackSize].near_tVal;

    if (cur_min_t >= closest_t) {
      continue;
    }

    const auto& cur_node = nodes_[cur];

    if (cur_node.isLeaf()) {
      // Process leaf node - test triangles in triangle blocks
      const SizeType triBlocksStart = cur_node.trianglesStart;
      const SizeType triBlocksEnd = cur_node.trianglesEnd;

      auto intersect_t = FloatP(std::numeric_limits<S>::max());
      Vector3P intersect_bary;
      auto intersect_index = UIntP(kInvalidTriangleIdx);

      for (SizeType iBlock = triBlocksStart; iBlock < triBlocksEnd; ++iBlock) {
        const auto& triBlock = triangleBlocks_[iBlock];

        auto triIntersectRes =
            intersectTriangle(ray, triBlock.triPos[0], triBlock.triPos[1], triBlock.triPos[2]);

        auto mask = (triIntersectRes.first < intersect_t);
        intersect_t = drjit::select(mask, triIntersectRes.first, intersect_t);
        intersect_bary = drjit::select(mask, triIntersectRes.second, intersect_bary);
        intersect_index = drjit::select(mask, triBlock.triIndex, intersect_index);
      }

      auto anyHit = (intersect_t < FloatP(closest_t));
      if (!drjit::any(anyHit)) {
        continue;
      }

      // Extract the closest hit from SIMD results
      S min_t_cur = std::numeric_limits<S>::max();
      int min_t_index = -1;
      for (uint32_t i = 0; i < TriangleBlockSize; ++i) {
        if (intersect_t[i] < min_t_cur && intersect_index[i] != kInvalidTriangleIdx) {
          min_t_cur = intersect_t[i];
          min_t_index = static_cast<int>(i);
        }
      }

      if (min_t_cur >= closest_t) {
        continue;
      }

      // Update closest hit
      closest_t = min_t_cur;
      closest_t_index = static_cast<uint32_t>(intersect_index[min_t_index]);
      for (int j = 0; j < 3; ++j) {
        closest_t_bary[j] = intersect_bary[j][min_t_index];
      }

    } else {
      // Process internal node - test bounding boxes
      const auto [mask, tMin, tMax] =
          intersectBBox(ray, cur_node.childBoxMin, cur_node.childBoxMax, ray.minT, closest_t);

      // Missed all boxes
      if (!drjit::any(mask)) {
        continue;
      }

      // Add children to stack
      for (int i = 0; i < TreeSplit; ++i) {
        if (!mask[i]) {
          continue;
        }

        if (cur_node.children[i] == kInvalidNodeIdx) {
          continue;
        }

        nodeStack[stackSize].nodeIdx = cur_node.children[i];
        nodeStack[stackSize].near_tVal = tMin[i];
        ++stackSize;
      }
    }
  }

  // Return result if we found a valid intersection
  if (closest_t < ray.maxT && closest_t_index != std::numeric_limits<uint32_t>::max()) {
    IntersectionResult<S> result;
    result.triangleId = static_cast<int32_t>(closest_t_index);
    result.hitDistance = closest_t;
    result.hitPoint = ray.origin + closest_t * ray.direction;
    result.baryCoords = closest_t_bary;
    return result;
  }

  return std::nullopt;
}

template <typename S>
std::vector<IntersectionResult<S>> TriBvh<S>::allHits(const Ray3<S>& ray) const {
  std::vector<IntersectionResult<S>> results;

  if (numTriangles_ == 0 || root_ == kInvalidNodeIdx) {
    return results;
  }

  // Use explicit stack for traversal with optimized 8-way tree structure
  const size_t kMaxStackSize = (TreeSplit - 1) * kMaxDepth + 1;

  struct NodeElt {
    SizeType nodeIdx;
    S near_tVal;
  };

  std::array<NodeElt, kMaxStackSize> nodeStack{};

  // Start with root on the stack
  nodeStack[0].nodeIdx = root_;
  nodeStack[0].near_tVal = ray.minT;
  SizeType stackSize = 1;

  // Note: Unlike closestHit, we don't update maxT during traversal
  // We want to find ALL intersections within the ray bounds
  const S segment_t = ray.maxT;

  while (stackSize != 0) {
    // Pop the top of the stack
    --stackSize;
    const SizeType cur = nodeStack[stackSize].nodeIdx;
    const S cur_min_t = nodeStack[stackSize].near_tVal;

    if (cur_min_t >= segment_t) {
      continue;
    }

    const auto& cur_node = nodes_[cur];

    if (cur_node.isLeaf()) {
      // Process leaf node - test triangles in triangle blocks
      const SizeType triBlocksStart = cur_node.trianglesStart;
      const SizeType triBlocksEnd = cur_node.trianglesEnd;

      for (SizeType iBlock = triBlocksStart; iBlock < triBlocksEnd; ++iBlock) {
        const auto& triBlock = triangleBlocks_[iBlock];

        auto triIntersectRes =
            intersectTriangle(ray, triBlock.triPos[0], triBlock.triPos[1], triBlock.triPos[2]);

        auto anyHit = (triIntersectRes.first < FloatP(segment_t));
        if (!drjit::any(anyHit)) {
          continue;
        }

        // Extract all valid hits from SIMD results
        for (uint32_t i = 0; i < TriangleBlockSize; ++i) {
          if (anyHit[i] && triBlock.triIndex[i] != kInvalidTriangleIdx) {
            // Create intersection result for this hit
            IntersectionResult<S> result;
            result.triangleId = static_cast<int32_t>(triBlock.triIndex[i]);
            result.hitDistance = triIntersectRes.first[i];
            result.hitPoint = ray.origin + result.hitDistance * ray.direction;

            // Extract barycentric coordinates
            for (int j = 0; j < 3; ++j) {
              result.baryCoords[j] = triIntersectRes.second[j][i];
            }

            results.push_back(result);
          }
        }
      }

    } else {
      // Process internal node - test bounding boxes
      const auto [mask, tMin, tMax] =
          intersectBBox(ray, cur_node.childBoxMin, cur_node.childBoxMax, ray.minT, segment_t);

      // Missed all boxes
      if (!drjit::any(mask)) {
        continue;
      }

      // Add children to stack
      for (int i = 0; i < TreeSplit; ++i) {
        if (!mask[i]) {
          continue;
        }

        if (cur_node.children[i] == kInvalidNodeIdx) {
          continue;
        }

        nodeStack[stackSize].nodeIdx = cur_node.children[i];
        nodeStack[stackSize].near_tVal = tMin[i];
        ++stackSize;
      }
    }
  }

  // Sort results by hit distance (optional, but often expected)
  std::sort(
      results.begin(),
      results.end(),
      [](const IntersectionResult<S>& a, const IntersectionResult<S>& b) {
        return a.hitDistance < b.hitDistance;
      });

  return results;
}

template <typename S>
bool TriBvh<S>::anyHit(const Ray3<S>& ray) const {
  if (numTriangles_ == 0 || root_ == kInvalidNodeIdx) {
    return false;
  }

  // Use explicit stack for traversal (similar to closestHit but with early exit)
  const size_t kMaxStackSize = (TreeSplit - 1) * kMaxDepth + 1;

  struct NodeElt {
    SizeType nodeIdx;
    S near_tVal;
  };

  std::array<NodeElt, kMaxStackSize> nodeStack{};

  // Start with root on the stack
  nodeStack[0].nodeIdx = root_;
  nodeStack[0].near_tVal = ray.minT;
  SizeType stackSize = 1;

  while (stackSize != 0) {
    // Pop the top of the stack
    --stackSize;
    const SizeType cur = nodeStack[stackSize].nodeIdx;
    const S cur_min_t = nodeStack[stackSize].near_tVal;

    if (cur_min_t >= ray.maxT) {
      continue;
    }

    const auto& cur_node = nodes_[cur];

    if (cur_node.isLeaf()) {
      // Process leaf node - test triangles in triangle blocks
      const SizeType triBlocksStart = cur_node.trianglesStart;
      const SizeType triBlocksEnd = cur_node.trianglesEnd;

      for (SizeType iBlock = triBlocksStart; iBlock < triBlocksEnd; ++iBlock) {
        const auto& triBlock = triangleBlocks_[iBlock];

        auto triIntersectRes =
            intersectTriangle(ray, triBlock.triPos[0], triBlock.triPos[1], triBlock.triPos[2]);

        auto anyHit = (triIntersectRes.first < FloatP(ray.maxT));
        if (drjit::any(anyHit)) {
          // Check if any of the hits are valid
          for (uint32_t i = 0; i < TriangleBlockSize; ++i) {
            if (anyHit[i] && triBlock.triIndex[i] != kInvalidTriangleIdx) {
              return true; // Early exit on first valid hit
            }
          }
        }
      }

    } else {
      // Process internal node - test bounding boxes
      const auto [mask, tMin, tMax] =
          intersectBBox(ray, cur_node.childBoxMin, cur_node.childBoxMax, ray.minT, ray.maxT);

      // Missed all boxes
      if (!drjit::any(mask)) {
        continue;
      }

      // Add children to stack
      for (int i = 0; i < TreeSplit; ++i) {
        if (!mask[i]) {
          continue;
        }

        if (cur_node.children[i] == kInvalidNodeIdx) {
          continue;
        }

        nodeStack[stackSize].nodeIdx = cur_node.children[i];
        nodeStack[stackSize].near_tVal = tMin[i];
        ++stackSize;
      }
    }
  }

  return false; // No intersection found
}

// SIMD point-to-bounding-box distance function - simplified using clamping
template <typename S>
typename TriBvh<S>::FloatSP TriBvh<S>::pointToBoxDistanceSquared(
    const Eigen::Vector3<S>& query,
    const Vector3SP& boxMin,
    const Vector3SP& boxMax) const {
  // Clamp query point to box bounds on each axis
  Vector3SP queryP = Vector3SP(FloatSP(query[0]), FloatSP(query[1]), FloatSP(query[2]));
  Vector3SP clampedPoint = Vector3SP(
      drjit::clip(queryP[0], boxMin[0], boxMax[0]),
      drjit::clip(queryP[1], boxMin[1], boxMax[1]),
      drjit::clip(queryP[2], boxMin[2], boxMax[2]));

  // Compute squared distance between query point and clamped point
  Vector3SP diff = queryP - clampedPoint;
  return diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
}

// Scalar point-to-triangle distance function using proper axel projection
template <typename S>
std::tuple<S, Eigen::Vector3<S>, Eigen::Vector3<S>> TriBvh<S>::pointToTriangleDistanceSquaredScalar(
    const Eigen::Vector3<S>& query,
    const Eigen::Vector3<S>& v0,
    const Eigen::Vector3<S>& v1,
    const Eigen::Vector3<S>& v2) const {
  Eigen::Vector3<S> closestPoint;
  Eigen::Vector3<S> baryCoords;

  // Use the proper axel projectOnTriangle function
  projectOnTriangle(query, v0, v1, v2, closestPoint, &baryCoords);

  // Compute squared distance
  const S distSq = (query - closestPoint).squaredNorm();

  return {distSq, closestPoint, baryCoords};
}

// SIMD point-to-triangle distance function using vectorized projectOnTriangle
template <typename S>
std::tuple<typename TriBvh<S>::FloatP, typename TriBvh<S>::Vector3P, typename TriBvh<S>::Vector3P>
TriBvh<S>::pointToTriangleDistanceSquared(
    const Eigen::Vector3<S>& query,
    const Vector3P& v0,
    const Vector3P& v1,
    const Vector3P& v2) const {
  // Create SIMD version of query point by broadcasting to all lanes
  Vector3P queryP = Vector3P(FloatP(query.x()), FloatP(query.y()), FloatP(query.z()));

  // Use vectorized projectOnTriangle function
  Vector3P closestPoint;
  Vector3P baryCoords;

  // The vectorized projectOnTriangle function expects WideVec3<S> which is the same as our Vector3P
  projectOnTriangle<S>(queryP, v0, v1, v2, closestPoint, &baryCoords);

  // Compute squared distance using SIMD operations
  Vector3P diff = queryP - closestPoint;
  FloatP distSq = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];

  return {distSq, closestPoint, baryCoords};
}

template <typename S>
typename TriBvh<S>::ClosestSurfacePointResult TriBvh<S>::closestSurfacePoint(
    const Eigen::Vector3<S>& query,
    const std::optional<S>& maxDist) const {
  ClosestSurfacePointResult result;
  result.triangleIdx = kInvalidTriangleIdx;

  if (numTriangles_ == 0 || root_ == kInvalidNodeIdx) {
    return result;
  }

  // Use explicit stack for traversal (similar to ray intersection)
  const size_t kMaxStackSize = (TreeSplit - 1) * kMaxDepth + 1;

  struct NodeElt {
    SizeType nodeIdx;
    S minDistSq;
  };

  std::array<NodeElt, kMaxStackSize> nodeStack{};

  // Start with root on the stack
  nodeStack[0].nodeIdx = root_;
  nodeStack[0].minDistSq = static_cast<S>(0);
  SizeType stackSize = 1;

  // Set initial closest distance to maxDist if specified, otherwise unlimited
  S closestDistSq = maxDist.has_value()
      ? maxDist.value() * maxDist.value() // Convert to squared distance
      : std::numeric_limits<S>::max();
  uint32_t closestTriangleIdx = kInvalidTriangleIdx;
  Eigen::Vector3<S> closestPoint = query;
  Eigen::Vector3<S> closestBaryCoords = Eigen::Vector3<S>::Zero();

  while (stackSize != 0) {
    // Pop the top of the stack
    --stackSize;
    const SizeType cur = nodeStack[stackSize].nodeIdx;
    const S curMinDistSq = nodeStack[stackSize].minDistSq;

    // Prune if this node can't contain a closer point
    if (curMinDistSq >= closestDistSq) {
      continue;
    }

    const auto& curNode = nodes_[cur];

    if (curNode.isLeaf()) {
      // Process leaf node - test triangles in triangle blocks
      const SizeType triBlocksStart = curNode.trianglesStart;
      const SizeType triBlocksEnd = curNode.trianglesEnd;

      for (SizeType iBlock = triBlocksStart; iBlock < triBlocksEnd; ++iBlock) {
        const auto& triBlock = triangleBlocks_[iBlock];

        auto [distSqP, closestPointP, baryCoordsP] = pointToTriangleDistanceSquared(
            query, triBlock.triPos[0], triBlock.triPos[1], triBlock.triPos[2]);

        // Extract the closest result from SIMD
        for (uint32_t i = 0; i < TriangleBlockSize; ++i) {
          if (triBlock.triIndex[i] != kInvalidTriangleIdx && distSqP[i] <= closestDistSq) {
            closestDistSq = distSqP[i];
            closestTriangleIdx = static_cast<uint32_t>(triBlock.triIndex[i]);
            closestPoint =
                Eigen::Vector3<S>(closestPointP[0][i], closestPointP[1][i], closestPointP[2][i]);
            closestBaryCoords =
                Eigen::Vector3<S>(baryCoordsP[0][i], baryCoordsP[1][i], baryCoordsP[2][i]);
          }
        }
      }

    } else {
      // Process internal node - test bounding boxes and add children to stack
      const auto distSqToBoxes =
          pointToBoxDistanceSquared(query, curNode.childBoxMin, curNode.childBoxMax);

      // Add children to stack, sorted by distance (closest first)
      std::array<std::pair<S, SizeType>, TreeSplit> childDistances;
      int validChildren = 0;

      for (int i = 0; i < TreeSplit; ++i) {
        if (curNode.children[i] != kInvalidNodeIdx && distSqToBoxes[i] < closestDistSq) {
          childDistances[validChildren] = {distSqToBoxes[i], curNode.children[i]};
          ++validChildren;
        }
      }

      // Sort children by distance (closest first)
      std::sort(
          childDistances.begin(),
          childDistances.begin() + validChildren,
          [](const auto& a, const auto& b) { return a.first < b.first; });

      // Add children to stack in reverse order (so closest is processed first)
      for (int i = validChildren - 1; i >= 0; --i) {
        nodeStack[stackSize].nodeIdx = childDistances[i].second;
        nodeStack[stackSize].minDistSq = childDistances[i].first;
        ++stackSize;
      }
    }
  }

  // Set up result - only return valid result if within maxDist (when specified)
  if (closestTriangleIdx != kInvalidTriangleIdx) {
    // If maxDist was specified, verify the result is within bounds
    if (maxDist.has_value()) {
      const S actualDist = std::sqrt(closestDistSq);
      if (actualDist > maxDist.value()) {
        // Result is beyond maxDist, return invalid result
        result.triangleIdx = kInvalidTriangleIdx;
        return result;
      }
    }

    result.point = closestPoint;
    result.triangleIdx = closestTriangleIdx;
    result.baryCoords = closestBaryCoords;
  }

  return result;
}

template <typename S>
size_t TriBvh<S>::getNodeCount() const {
  return nodes_.size();
}

template <typename S>
size_t TriBvh<S>::getPrimitiveCount() const {
  return numTriangles_;
}

// Validation function for debugging tree construction
template <typename S>
void TriBvh<S>::validate(
    const std::vector<Eigen::Vector3i>& triangles,
    const std::vector<Eigen::Vector3<S>>& vertices) const {
  if (numTriangles_ == 0) {
    XR_CHECK(root_ == kInvalidNodeIdx, "Empty tree should have invalid root");
    XR_CHECK(triangleBlocks_.empty(), "Empty tree should have no triangle blocks");
    XR_CHECK(nodes_.empty(), "Empty tree should have no nodes");
    return;
  }

  XR_CHECK(root_ != kInvalidNodeIdx, "Non-empty tree should have valid root");
  XR_CHECK(root_ < nodes_.size(), "Root index should be valid");

  // Validate that all triangles are accounted for
  std::set<uint32_t> foundTriangles;

  // Traverse all leaf nodes and collect triangle indices
  std::function<void(SizeType)> validateNode = [&](SizeType nodeIdx) {
    XR_CHECK(nodeIdx < nodes_.size(), "Node index {} out of bounds", nodeIdx);

    const auto& node = nodes_[nodeIdx];

    if (node.isLeaf()) {
      // Validate leaf node
      XR_CHECK(
          node.trianglesStart != kInvalidTriangleBlockIdx,
          "Leaf node should have valid triangle start");
      XR_CHECK(
          node.trianglesEnd != kInvalidTriangleBlockIdx,
          "Leaf node should have valid triangle end");
      XR_CHECK(node.trianglesStart <= node.trianglesEnd, "Invalid triangle range");
      XR_CHECK(node.trianglesEnd <= triangleBlocks_.size(), "Triangle end out of bounds");

      // Check all triangles in this leaf's blocks
      for (SizeType iBlock = node.trianglesStart; iBlock < node.trianglesEnd; ++iBlock) {
        const auto& triBlock = triangleBlocks_[iBlock];

        for (uint32_t i = 0; i < TriangleBlockSize; ++i) {
          if (triBlock.triIndex[i] != kInvalidTriangleIdx) {
            auto triIdx = static_cast<uint32_t>(triBlock.triIndex[i]);
            XR_CHECK(triIdx < triangles.size(), "Triangle index {} out of bounds", triIdx);

            // Check for duplicates
            XR_CHECK(
                foundTriangles.find(triIdx) == foundTriangles.end(),
                "Duplicate triangle index {}",
                triIdx);
            foundTriangles.insert(triIdx);

            // Validate triangle positions match input data
            const auto& tri = triangles[triIdx];
            for (int j = 0; j < 3; ++j) {
              for (int k = 0; k < 3; ++k) {
                S expected = vertices[tri[j]][k];
                S actual = triBlock.triPos[j][k][i];
                S diff = std::abs(expected - actual);
                XR_CHECK(
                    diff < static_cast<S>(1e-6),
                    "Triangle {} vertex {} coord {} mismatch: expected {}, got {}",
                    triIdx,
                    j,
                    k,
                    expected,
                    actual);
              }
            }
          }
        }
      }
    } else {
      // Validate internal node
      bool hasValidChild = false;
      for (int i = 0; i < TreeSplit; ++i) {
        if (node.children[i] != kInvalidNodeIdx) {
          hasValidChild = true;
          validateNode(node.children[i]);
        }
      }
      XR_CHECK(hasValidChild, "Internal node should have at least one valid child");
    }
  };

  validateNode(root_);

  // Check that we found all triangles
  XR_CHECK(
      foundTriangles.size() == numTriangles_,
      "Found {} triangles, expected {}",
      foundTriangles.size(),
      numTriangles_);

  // Check that triangle indices are in expected range
  for (uint32_t i = 0; i < numTriangles_; ++i) {
    XR_CHECK(foundTriangles.find(i) != foundTriangles.end(), "Missing triangle index {}", i);
  }
}

// Debug function to print tree statistics
template <typename S>
void TriBvh<S>::printTreeStats() const {
  std::cout << "=== TriBvh Tree Statistics ===" << std::endl;
  std::cout << "Triangles: " << numTriangles_ << std::endl;
  std::cout << "Nodes: " << nodes_.size() << std::endl;
  std::cout << "Triangle Blocks: " << triangleBlocks_.size() << std::endl;
  std::cout << "Depth: " << depth_ << std::endl;
  std::cout << "Root: " << root_ << std::endl;
  std::cout << "SIMD Width: " << TriangleBlockSize << std::endl;
  std::cout << "Tree Split: " << TreeSplit << std::endl;

  if (numTriangles_ > 0) {
    std::cout << "Bounding Box: [" << boundingBox_.min().transpose() << "] to ["
              << boundingBox_.max().transpose() << "]" << std::endl;
  }

  // Count leaf vs internal nodes
  size_t leafNodes = 0;
  size_t internalNodes = 0;
  size_t totalTrianglesInBlocks = 0;

  for (const auto& node : nodes_) {
    if (node.isLeaf()) {
      leafNodes++;
      // Count triangles in this leaf
      for (SizeType iBlock = node.trianglesStart; iBlock < node.trianglesEnd; ++iBlock) {
        const auto& triBlock = triangleBlocks_[iBlock];
        for (uint32_t i = 0; i < TriangleBlockSize; ++i) {
          if (triBlock.triIndex[i] != kInvalidTriangleIdx) {
            totalTrianglesInBlocks++;
          }
        }
      }
    } else {
      internalNodes++;
    }
  }

  std::cout << "Leaf Nodes: " << leafNodes << std::endl;
  std::cout << "Internal Nodes: " << internalNodes << std::endl;
  std::cout << "Triangles in Blocks: " << totalTrianglesInBlocks << std::endl;

  if (totalTrianglesInBlocks != numTriangles_) {
    std::cout << "WARNING: Triangle count mismatch!" << std::endl;
  }

  std::cout << "==============================" << std::endl;
}

template class TriBvh<float>;
template class TriBvh<double>;

} // namespace axel
