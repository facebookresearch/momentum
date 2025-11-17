/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>
#include <optional>
#include <vector>

#include <embree4/rtcore.h>

#include "axel/BoundingBox.h"
#include "axel/BvhBase.h"

namespace axel {

/**
 * @brief Encapsulates an Embree BVH for a single geometry.
 *
 * The data structure can be initialized/updated with setBoundingBoxes(...)
 * or preferably with a vector of RTCBoundPrimitive structs through setPrimitives(...).
 *
 * The BVH supports bounding boxes and ray queries to conform to the interface
 * of the other BVH implementations in this library.
 */
class BVHEmbree final : public BvhBased {
 public:
  struct alignas(16) Node {
    // A bounding box with SIMD-friendly storage.
    std::array<RTCBounds, 2> bounds{};
    std::array<BVHEmbree::Node*, 2> children{nullptr, nullptr};
    uint32_t primID{0};
    bool isLeaf{false};

    Node() = default;
    explicit Node(const RTCBuildPrimitive& bounds);
  };

  /**
   * @brief Construct a new, empty BVH based on Embree.
   */
  explicit BVHEmbree(std::optional<uint32_t> embreeThreadCount = std::nullopt);
  ~BVHEmbree() override;

  BVHEmbree(const BVHEmbree& other) = delete;
  BVHEmbree(BVHEmbree&& other) noexcept;
  BVHEmbree& operator=(const BVHEmbree& other) = delete;
  BVHEmbree& operator=(BVHEmbree&& other) noexcept;

  void setBoundingBoxes(const std::vector<BoundingBoxd>& bboxes) override;

  /**
   * @brief Updates the BVH with a new vector of bounding boxes.
   * The input parameter can't be made const because Embree takes a non-const ptr
   * to its data.
   * @param prims The new bounding boxes, in Embree's expected, aligned structs.
   */
  void setPrimitives(std::vector<RTCBuildPrimitive>& prims);

  uint32_t query(const BoundingBoxd& box, QueryBuffer& hits) const override;
  [[nodiscard]] std::vector<unsigned int> query(const BoundingBoxd& box) const override;
  [[nodiscard]] std::vector<unsigned int> query(
      const Eigen::Vector3d& origin,
      const Eigen::Vector3d& direction) const override;

  [[nodiscard]] Size getPrimitiveCount() const override;

 private:
  // Manually ref-counted handle to the device used as Embree's context.
  RTCDevice device_{nullptr};

  // Struct for BVH build parameters.
  RTCBuildArguments buildArguments_;

  // Owning handle to the BVH handle.
  RTCBVH bvh_{nullptr};

  // This is a non-owning pointer to Embree's internal thread-local memory allocation.
  // All the memory will be freed once we call rtcReleaseBVH(bvh_) in the destructor of this class.
  Node* root_{nullptr};
};

} // namespace axel
