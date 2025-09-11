/*
 * Portions Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* which bore the following notice:
Copyright (c) 2012 Brandon Pelfrey
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions: The above copyright notice and this
permission notice shall be included in all copies or substantial portions of the Software. THE
SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#pragma once

#include <vector>

#include "axel/BvhBase.h"
#include "axel/BvhCommon.h"
#include "axel/common/Types.h"
#include "axel/common/VectorizationTypes.h"

// BVH based on https://github.com/brandonpelfrey/Fast-BVH

namespace axel {

template <typename S>
struct BvhFlatNode {
  BoundingBox<S> bbox;
  uint32_t start{}, nPrims{}, rightOffset{};

  [[nodiscard]] bool isLeaf() const {
    return rightOffset == 0;
  }

  [[nodiscard]] uint32_t getLeftIndex(uint32_t index) const {
    return index + 1;
  }

  [[nodiscard]] uint32_t getRightIndex(uint32_t index) const {
    return index + rightOffset;
  }

  // Invokes a callback function for each primitive within the node. If the callback function
  // returns a boolean, a return value of true indicates that the iteration should stop, while
  // false allows it to continue.
  template <typename Callback>
  auto eachPrimitive(std::vector<BoundingBox<S>>& primitives, Callback&& callback);

  // Invokes a callback function for each primitive within the node. If the callback function
  // returns a boolean, a return value of true indicates that the iteration should stop, while
  // false allows it to continue.
  template <typename Callback>
  auto eachPrimitive(const std::vector<BoundingBox<S>>& primitives, Callback&& callback) const;

  // Invokes a callback function for each primitive within the node starting from a specified
  // index. If the callback function returns a boolean, a return value of true indicates that the
  // iteration should stop, while false allows it to continue.
  template <typename Callback>
  auto
  eachPrimitive(std::vector<BoundingBox<S>>& primitives, uint32_t startIndex, Callback&& callback);

  // Invokes a callback function for each primitive within the node starting from a specified
  // index. If the callback function returns a boolean, a return value of true indicates that the
  // iteration should stop, while false allows it to continue.
  template <typename Callback>
  auto eachPrimitive(
      const std::vector<BoundingBox<S>>& primitives,
      uint32_t startIndex,
      Callback&& callback) const;
};

template <typename S, size_t LeafCapacity = 1>
class Bvh final : public BvhBase<S> {
 public:
  using BvhType = Bvh<S, LeafCapacity>;
  using Scalar = S;
  using QueryBuffer = typename BvhBase<S>::QueryBuffer;
  static constexpr size_t kLeafCapacity = LeafCapacity;
  static constexpr size_t kMaxStackDepth = 64;

  Bvh() = default;
  explicit Bvh(const std::vector<BoundingBox<S>>& buildPrimitives);
  ~Bvh() override = default;

  Bvh(const Bvh&) = default;
  Bvh& operator=(const Bvh&) = default;

  Bvh(Bvh&&) = default;
  Bvh& operator=(Bvh&&) = default;

  // Sets the bounding boxes of the leaf nodes of the BVH. This function calls build() internally,
  // so there's no need to call build() again after setting the bounding boxes.
  void setBoundingBoxes(const std::vector<BoundingBox<S>>& bboxes) override;

  // Rebuilds the BVH based on the bounding boxes previously set via setBoundingBoxes(). This
  // function is useful when the bounding boxes, accessed via getPrimitives(), are updated and a
  // rebuild of the tree is needed.
  void build();

  [[nodiscard]] std::vector<BoundingBox<S>>& getPrimitives();

  [[nodiscard]] const std::vector<BoundingBox<S>>& getPrimitives() const;

  // Refits the tree to account for changes in the bounding boxes of leaf nodes. This is achieved
  // by updating the entire tree in a bottom-up manner to ensure accurate collision detection.
  void refit();

  // Queries the BVH for bounding boxes intersecting with the specified box.
  [[nodiscard]] std::vector<uint32_t> query(const BoundingBox<S>& box) const override;

  // Overloaded query function that stores intersecting bounding boxes in a provided buffer,
  // returning the count.
  uint32_t query(const BoundingBox<S>& box, QueryBuffer& hits) const override;

  // Queries the BVH for bounding boxes intersected by a ray specified by its origin and direction.
  [[nodiscard]] std::vector<uint32_t> query(
      const Eigen::Vector3<S>& origin,
      const Eigen::Vector3<S>& direction) const override;

  // Queries the BVH for the closest point on the surface to a specified query point.
  template <typename ClosestSurfacePointFunc>
  [[nodiscard]] ClosestSurfacePointResult<S> queryClosestSurfacePoint(
      const Eigen::Vector3<S>& query,
      ClosestSurfacePointFunc&& func) const;
  template <typename ClosestSurfacePointFunc>
  [[nodiscard]] ClosestSurfacePointResult<S> queryClosestSurfacePointPacket(
      const Eigen::Vector3<S>& query,
      ClosestSurfacePointFunc&& func) const;

  // Queries the BVH for the closest hit along a ray within a specified range of t-values.
  template <typename ReturnType, typename RayPrimitiveIntersector>
  [[nodiscard]] ReturnType rayQueryClosestHit(
      const Eigen::Vector3<S>& origin,
      const Eigen::Vector3<S>& direction,
      S minT,
      S maxT,
      RayPrimitiveIntersector&& func) const;

  // Queries the BVH for any hit along a ray within a specified range of t-values, returning true if
  // such an intersection exists, or false otherwise.
  template <typename RayPrimitiveIntersector>
  [[nodiscard]] bool rayQueryAnyHit(
      const Eigen::Vector3<S>& origin,
      const Eigen::Vector3<S>& direction,
      S minT,
      S maxT,
      RayPrimitiveIntersector&& func) const;

  // Invokes the provided callback function for each pair of overlapping bounding boxes within this
  // BVH. The callback function should return false if the traversal should stop, or false
  // otherwise.
  template <typename Callback>
  void traverseOverlappingPairs(Callback&& func) const;

  // Invokes the provided callback function for each pair of overlapping bounding boxes between this
  // BVH and another BVH. Note that the indices passed to the callback are relative to each BVH's
  // own primitive list. The callback function should return false if the traversal should stop,
  // or false otherwise.
  template <typename Callback>
  void traverseOverlappingPairs(const BvhType& other, Callback&& func) const;

  // Returns the total number of nodes in the BVH.
  [[nodiscard]] Size getNodeCount() const;

  // Returns the total number of primitives (bounding boxes) in the BVH.
  [[nodiscard]] Size getPrimitiveCount() const override;

  // Checks if all bounding boxes in the BVH tree are correctly structured.
  // This method is mainly used for debugging purposes.
  [[nodiscard]] bool checkBoundingBoxes() const;

 private:
  // This bool is used to select either a scalar or a packet-based callback type.
  template <bool IsPacket, bool NeedBarycentric, typename ClosestSurfacePointFunc>
  S queryClosestHelper(
      uint32_t nodeIdx,
      const Eigen::Vector3<S>& query,
      const ClosestSurfacePointFunc& func,
      S sqrDist,
      ClosestSurfacePointResult<S>& result) const;

  std::vector<BvhFlatNode<S>> flatTree_;
  std::vector<BoundingBox<S>> buildPrimitives_;

  Size nodeCount_{0};
};

using Bvhf = Bvh<float, 1>;
using Bvhd = Bvh<double, 1>;

extern template class Bvh<float, 1>;
extern template class Bvh<double, 1>;
extern template class Bvh<float, kNativeLaneWidth<float>>;
extern template class Bvh<double, kNativeLaneWidth<double>>;

} // namespace axel

#include <axel/Checks.h>

//------------------------------------------------------------------------------
// Implementation
//------------------------------------------------------------------------------

namespace axel {

template <typename S>
template <typename Callback>
auto BvhFlatNode<S>::eachPrimitive(std::vector<BoundingBox<S>>& primitives, Callback&& callback) {
  static_assert(
      std::is_invocable_v<Callback, BoundingBox<S>&>,
      "Callback function must be invocable with a BoundingBox<S>& argument");

  if constexpr (std::is_same_v<bool, std::invoke_result_t<Callback, BoundingBox<S>&>>) {
    for (uint32_t i = start; i < start + nPrims; ++i) {
      if (!callback(primitives[i])) {
        return false;
      }
    }
    return true;
  } else {
    for (uint32_t i = start; i < start + nPrims; ++i) {
      callback(primitives[i]);
    }
  }
}

template <typename S>
template <typename Callback>
auto BvhFlatNode<S>::eachPrimitive(
    const std::vector<BoundingBox<S>>& primitives,
    Callback&& callback) const {
  static_assert(
      std::is_invocable_v<Callback, const BoundingBox<S>&>,
      "Callback function must be invocable with a const BoundingBox<S>& argument");

  if constexpr (std::is_same_v<bool, std::invoke_result_t<Callback, const BoundingBox<S>&>>) {
    for (uint32_t i = start; i < start + nPrims; ++i) {
      if (!callback(primitives[i])) {
        return false;
      }
    }
    return true;
  } else {
    for (uint32_t i = start; i < start + nPrims; ++i) {
      callback(primitives[i]);
    }
  }
}

template <typename S>
template <typename Callback>
auto BvhFlatNode<S>::eachPrimitive(
    std::vector<BoundingBox<S>>& primitives,
    uint32_t startIndex,
    Callback&& callback) {
  static_assert(
      std::is_invocable_v<Callback, BoundingBox<S>&>,
      "Callback function must be invocable with a BoundingBox<S>& argument");

  if constexpr (std::is_same_v<bool, std::invoke_result_t<Callback, BoundingBox<S>&>>) {
    for (uint32_t i = start + startIndex; i < start + nPrims; ++i) {
      if (!callback(primitives[i])) {
        return false;
      }
    }
    return true;
  } else {
    for (uint32_t i = start + startIndex; i < start + nPrims; ++i) {
      callback(primitives[i]);
    }
  }
}

template <typename S>
template <typename Callback>
auto BvhFlatNode<S>::eachPrimitive(
    const std::vector<BoundingBox<S>>& primitives,
    uint32_t startIndex,
    Callback&& callback) const {
  static_assert(
      std::is_invocable_v<Callback, const BoundingBox<S>&>,
      "Callback function must be invocable with a const BoundingBox<S>& argument");

  if constexpr (std::is_same_v<bool, std::invoke_result_t<Callback, const BoundingBox<S>&>>) {
    for (uint32_t i = start + startIndex; i < start + nPrims; ++i) {
      if (!callback(primitives[i])) {
        return false;
      }
    }
    return true;
  } else {
    for (uint32_t i = start + startIndex; i < start + nPrims; ++i) {
      callback(primitives[i]);
    }
  }
}

namespace {

template <typename S, typename Callback>
bool traverseCollision(
    const std::vector<BvhFlatNode<S>>& nodesA,
    const Size indexA,
    const std::vector<BoundingBox<S>>& leafAabbsA,
    const std::vector<BvhFlatNode<S>>& nodesB,
    const Size indexB,
    const std::vector<BoundingBox<S>>& leafAabbsB,
    Callback&& func) {
  // Retrieve the current nodes from both trees
  const auto& nodeA = nodesA[indexA];
  const auto& nodeB = nodesB[indexB];

  // Early exit if bounding boxes of the current nodes do not intersect
  if (!nodeA.bbox.intersects(nodeB.bbox)) {
    return false;
  }

  // Handle leaf-leaf intersection by checking all bounding boxes at the leaf nodes
  if (nodeA.isLeaf() && nodeB.isLeaf()) {
    nodeA.eachPrimitive(leafAabbsA, [&](const BoundingBox<S>& aabbA) -> bool {
      return nodeB.eachPrimitive(leafAabbsB, [&](const BoundingBox<S>& aabbB) -> bool {
        return func(aabbA.id, aabbB.id); // true means continue traversal
      }); // true means continue traversal
    });
    return false;
  }

  // Determine which BVH to descend based on leaf status or bounding box size
  const bool descendA = nodeB.isLeaf() ||
      (!nodeA.isLeaf() && (nodeA.bbox.squaredVolume() > nodeB.bbox.squaredVolume()));

  // Recursively traverse the chosen BVH and its sibling
  const Size nextIndexA1 = descendA ? nodeA.getLeftIndex(indexA) : indexA;
  const Size nextIndexA2 = descendA ? nodeA.getRightIndex(indexA) : indexA;
  const Size nextIndexB1 = descendA ? indexB : nodeB.getLeftIndex(indexB);
  const Size nextIndexB2 = descendA ? indexB : nodeB.getRightIndex(indexB);
  return traverseCollision(
             nodesA,
             nextIndexA1,
             leafAabbsA,
             nodesB,
             nextIndexB1,
             leafAabbsB,
             std::forward<Callback>(func)) ||
      traverseCollision(
             nodesA,
             nextIndexA2,
             leafAabbsA,
             nodesB,
             nextIndexB2,
             leafAabbsB,
             std::forward<Callback>(func));
}

template <typename S, typename Callback>
bool traverseSelfCollision(
    const std::vector<BvhFlatNode<S>>& nodes,
    const Size index,
    const std::vector<BoundingBox<S>>& leafAabbs,
    Callback&& func) {
  const auto& node = nodes[index];

  if (node.isLeaf()) {
    return false;
  }

  const Size leftChildIndex = node.getLeftIndex(index);
  if (traverseSelfCollision(nodes, leftChildIndex, leafAabbs, std::forward<Callback>(func))) {
    return true;
  }

  const Size rightChildIndex = node.getRightIndex(index);
  if (traverseSelfCollision(nodes, rightChildIndex, leafAabbs, std::forward<Callback>(func))) {
    return true;
  }

  if (traverseCollision(
          nodes,
          leftChildIndex,
          leafAabbs,
          nodes,
          rightChildIndex,
          leafAabbs,
          std::forward<Callback>(func))) {
    return true;
  }

  return false;
}

} // namespace

template <typename S, size_t LeafCapacity>
template <typename ClosestSurfacePointFunc>
[[nodiscard]] ClosestSurfacePointResult<S> Bvh<S, LeafCapacity>::queryClosestSurfacePoint(
    const Eigen::Vector3<S>& query,
    ClosestSurfacePointFunc&& func) const {
  // TODO(nemanjab): Implement this without recursion. It should be faster.
  ClosestSurfacePointResult<S> res{};
  if constexpr (std::is_invocable_r_v<
                    void,
                    ClosestSurfacePointFunc,
                    const Eigen::Vector3<S>&,
                    Index,
                    Eigen::Vector3<S>&>) {
    queryClosestHelper</*IsPacket=*/false, /*NeedBarycentric=*/false>(
        0, query, std::forward<ClosestSurfacePointFunc>(func), std::numeric_limits<S>::max(), res);
  } else if (std::is_invocable_r_v<
                 void,
                 ClosestSurfacePointFunc,
                 const Eigen::Vector3<S>&,
                 Index,
                 Eigen::Vector3<S>&,
                 Eigen::Vector3<S>&>) {
    queryClosestHelper</*IsPacket=*/false, /*NeedBarycentric=*/true>(
        0, query, std::forward<ClosestSurfacePointFunc>(func), std::numeric_limits<S>::max(), res);
  } else {
    XR_CHECK(
        false,
        "Callback function must be invocable with query type, primitive ID and return the closest projection point.");
  }
  return res;
}

template <typename S, size_t LeafCapacity>
template <typename ClosestSurfacePointFunc>
[[nodiscard]] ClosestSurfacePointResult<S> Bvh<S, LeafCapacity>::queryClosestSurfacePointPacket(
    const Eigen::Vector3<S>& query,
    ClosestSurfacePointFunc&& func) const {
  // TODO(nemanjab, T176575677): Implement this without recursion. It should be faster.
  ClosestSurfacePointResult<S> res{};
  if constexpr (std::is_invocable_r_v<
                    void,
                    ClosestSurfacePointFunc,
                    const Eigen::Vector3<S>&,
                    const uint32_t,
                    const WideScalar<int32_t, LeafCapacity>&,
                    WideVec3<S, LeafCapacity>&,
                    WideScalar<S, LeafCapacity>&>) {
    queryClosestHelper</*IsPacket=*/true, /*NeedBarycentric=*/false>(
        0, query, std::forward<ClosestSurfacePointFunc>(func), std::numeric_limits<S>::max(), res);
  } else if (std::is_invocable_r_v<
                 void,
                 ClosestSurfacePointFunc,
                 const Eigen::Vector3<S>&,
                 const uint32_t,
                 const WideScalar<int32_t, LeafCapacity>&,
                 WideVec3<S, LeafCapacity>&,
                 WideVec3<S, LeafCapacity>&,
                 WideScalar<S, LeafCapacity>&>) {
    queryClosestHelper</*IsPacket=*/true, /*NeedBarycentric=*/true>(
        0, query, std::forward<ClosestSurfacePointFunc>(func), std::numeric_limits<S>::max(), res);
  } else {
    XR_CHECK(
        false,
        "Callback function must be invocable with query, prim count, prim IDs and return projections and their distances through out param.");
  }
  return res;
}

template <typename S, size_t LeafCapacity>
template <bool IsPacket, bool NeedBarycentric, typename ClosestSurfacePointFunc>
S Bvh<S, LeafCapacity>::queryClosestHelper(
    const uint32_t nodeIdx,
    const Eigen::Vector3<S>& query,
    const ClosestSurfacePointFunc& func,
    S sqrDist, // NOLINT
    ClosestSurfacePointResult<S>& result) const {
  const auto& currNode = flatTree_[nodeIdx];

  // If we hit the leaf, update the closest distance.
  if (currNode.isLeaf()) {
    // We can still send non-packet query even if LeafCapacity is bigger than 1, which is good for
    // debugging.
    if constexpr (IsPacket) {
      // We send up to N (depending on leaf capacity) primitives at once.
      // The callback can use packet-based vectorization for data parallelism.
      std::array<uint32_t, LeafCapacity> primIndices; // NOLINT
      for (uint32_t o = 0; o < currNode.nPrims; ++o) {
        primIndices[o] = buildPrimitives_[currNode.start + o].id;
      }
      WideVec3<S, LeafCapacity> projections;
      WideScalar<S, LeafCapacity> squaredDistances;
      [[maybe_unused]] WideVec3<S, LeafCapacity> barycentrics;
      if constexpr (NeedBarycentric) {
        func(
            query,
            currNode.nPrims,
            drjit::load<WideScalar<int32_t, LeafCapacity>>(primIndices.data()),
            projections,
            barycentrics,
            squaredDistances);
      } else {
        func(
            query,
            currNode.nPrims,
            drjit::load<WideScalar<int32_t, LeafCapacity>>(primIndices.data()),
            projections,
            squaredDistances);
      }
      for (int i = 0; i < currNode.nPrims; ++i) {
        if (squaredDistances[i] < sqrDist) {
          sqrDist = squaredDistances[i];
          result.point = Eigen::Vector3<S>{projections[0][i], projections[1][i], projections[2][i]};
          result.triangleIdx = primIndices[i];
          if constexpr (NeedBarycentric) {
            result.baryCoords =
                Eigen::Vector3<S>{barycentrics[0][i], barycentrics[1][i], barycentrics[2][i]};
          }
        }
      }
    } else {
      // The base scalar case.
      for (uint32_t o = 0; o < currNode.nPrims; ++o) {
        const BoundingBox<S>& obj = buildPrimitives_[currNode.start + o];
        Eigen::Vector3<S> projection;
        [[maybe_unused]] Eigen::Vector3<S> barycentric;
        if constexpr (NeedBarycentric) {
          func(query, obj.id, projection, barycentric);
        } else {
          func(query, obj.id, projection);
        }
        const S distSq = (projection - query).squaredNorm();
        if (distSq < sqrDist) {
          result.point = std::move(projection);
          result.triangleIdx = obj.id;
          if constexpr (NeedBarycentric) {
            result.baryCoords = std::move(barycentric);
          }
          sqrDist = distSq;
        }
      }
    }
  } else {
    const auto& checkSubtree = [this, &query, &sqrDist, &func, &result](const uint32_t nodeIdx) {
      ClosestSurfacePointResult<S> tempRes;
      const S distSq =
          queryClosestHelper<IsPacket, NeedBarycentric>(nodeIdx, query, func, sqrDist, tempRes);
      if (distSq < sqrDist) {
        sqrDist = distSq;
        result = tempRes;
      }
    };

    const auto& leftBox = flatTree_[currNode.getLeftIndex(nodeIdx)].bbox.aabb;
    const auto& rightBox = flatTree_[currNode.getRightIndex(nodeIdx)].bbox.aabb;

    // If we are inside an intermediate box, we need to explore the whole subtree.
    // This is because the primitive could be at any distance from the query.
    const bool checkedLeftSubtree = leftBox.contains(query);
    if (checkedLeftSubtree) {
      checkSubtree(currNode.getLeftIndex(nodeIdx));
    }
    const bool checkedRightSubtree = rightBox.contains(query);
    if (checkedRightSubtree) {
      checkSubtree(currNode.getRightIndex(nodeIdx));
    }

    // For the case where the query isn't inside, we'll need distance to the AABBs.
    const S leftExtDistSq = leftBox.squaredExteriorDistance(query);
    const S rightExtDistSq = rightBox.squaredExteriorDistance(query);
    if (leftExtDistSq < rightExtDistSq) {
      if (!checkedLeftSubtree && leftExtDistSq < sqrDist) {
        checkSubtree(currNode.getLeftIndex(nodeIdx));
      }
      if (!checkedRightSubtree && rightExtDistSq < sqrDist) {
        checkSubtree(currNode.getRightIndex(nodeIdx));
      }
    } else {
      if (!checkedRightSubtree && rightExtDistSq < sqrDist) {
        checkSubtree(currNode.getRightIndex(nodeIdx));
      }
      if (!checkedLeftSubtree && leftExtDistSq < sqrDist) {
        checkSubtree(currNode.getLeftIndex(nodeIdx));
      }
    }
  }

  return sqrDist;
}

template <typename S, size_t LeafCapacity>
template <typename ReturnType, typename RayPrimitiveIntersector>
ReturnType Bvh<S, LeafCapacity>::rayQueryClosestHit(
    const Eigen::Vector3<S>& origin,
    const Eigen::Vector3<S>& direction,
    S /* minT */,
    S maxT,
    RayPrimitiveIntersector&& func) const {
  static_assert(
      std::is_invocable_r_v<void, RayPrimitiveIntersector, Index, ReturnType&, S&>,
      "Callback function must be invocable with a primitive index, result type to update and maximum distance to update.");

  std::array<uint32_t, kMaxStackDepth> todoStack{};
  Index stackLevel = 0;
  todoStack[stackLevel] = 0; // Start from the root, node 0.

  ReturnType result{};

  const Eigen::Vector3<S> dirInv = direction.cwiseInverse();
  while (stackLevel >= 0) {
    // Pop off the next node to work on.
    const auto currNode = todoStack[stackLevel--];
    const auto& node(flatTree_[currNode]);

    // This is the leaf condition.
    if (node.isLeaf()) {
      node.eachPrimitive(
          buildPrimitives_, [&](const BoundingBox<S>& prim) { func(prim.id, result, maxT); });
    } else { // Not a leaf => internal node with 2 children boxes.
      if (flatTree_[node.getLeftIndex(currNode)].bbox.intersectsBranchless(origin, dirInv)) {
        todoStack[++stackLevel] = node.getLeftIndex(currNode);
      }
      if (flatTree_[node.getRightIndex(currNode)].bbox.intersectsBranchless(origin, dirInv)) {
        todoStack[++stackLevel] = node.getRightIndex(currNode);
      }
    }
  }
  return result;
}

template <typename S, size_t LeafCapacity>
template <typename RayPrimitiveIntersector>
bool Bvh<S, LeafCapacity>::rayQueryAnyHit(
    const Eigen::Vector3<S>& origin,
    const Eigen::Vector3<S>& direction,
    S /* minT */,
    S /* maxT */,
    RayPrimitiveIntersector&& func) const {
  static_assert(
      std::is_invocable_r_v<bool, RayPrimitiveIntersector, Index>,
      "Callback function must be invocable with a primitive index and return if the hit was valid.");

  std::array<uint32_t, kMaxStackDepth> todoStack{};
  Index stackLevel = 0;
  todoStack[stackLevel] = 0; // Start from the root, node 0.

  const Eigen::Vector3<S> dirInv = direction.cwiseInverse();
  while (stackLevel >= 0) {
    // Pop off the next node to work on.
    const auto currNode = todoStack[stackLevel--];
    const auto& node(flatTree_[currNode]);

    // Is leaf -> Intersect
    if (node.isLeaf()) {
      for (Index o = 0; o < node.nPrims; ++o) {
        if (func(buildPrimitives_[node.start + o].id)) {
          return true;
        }
      }
    } else { // Not a leaf => internal node with 2 children boxes.
      if (flatTree_[node.getLeftIndex(currNode)].bbox.intersectsBranchless(origin, dirInv)) {
        todoStack[++stackLevel] = node.getLeftIndex(currNode);
      }
      if (flatTree_[node.getRightIndex(currNode)].bbox.intersectsBranchless(origin, dirInv)) {
        todoStack[++stackLevel] = node.getRightIndex(currNode);
      }
    }
  }
  return false;
}

template <typename S, size_t LeafCapacity>
template <typename Callback>
void Bvh<S, LeafCapacity>::traverseOverlappingPairs(Callback&& func) const {
  static_assert(
      std::is_invocable_r_v<bool, Callback, Index, Index>,
      "Callback function must be invocable with two Index arguments and return bool");

  if (nodeCount_ == 0) {
    return;
  }
  traverseSelfCollision<S>(flatTree_, 0, buildPrimitives_, std::forward<Callback>(func));
}

template <typename S, size_t LeafCapacity>
template <typename Callback>
void Bvh<S, LeafCapacity>::traverseOverlappingPairs(
    const Bvh<S, LeafCapacity>& other,
    Callback&& func) const {
  static_assert(
      std::is_invocable_r_v<bool, Callback, Index, Index>,
      "Callback function must be invocable with two Index arguments and return bool");

  if (nodeCount_ == 0 || other.nodeCount_ == 0) {
    return;
  }
  traverseCollision(
      flatTree_,
      0,
      buildPrimitives_,
      other.flatTree_,
      0,
      other.buildPrimitives_,
      std::forward<Callback>(func));
}

} // namespace axel
