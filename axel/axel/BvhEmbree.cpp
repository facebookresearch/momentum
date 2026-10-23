/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "axel/BvhEmbree.h"

#include <utility>

#include <dispenso/parallel_for.h>

#include <Eigen/Dense>

#include "axel/Profile.h"

#define DEFAULT_LOG_CHANNEL "AXEL: Embree"
#include "axel/Log.h"

namespace axel {
namespace {

/**
 * @brief A simple log function for any errors that Embree might want to log.
 */
void logEmbreeError(void* /*userPtr*/, RTCError err, const char* msg) {
  switch (err) {
    case RTC_ERROR_INVALID_ARGUMENT: {
      XR_LOGE("INVALID_ARGUMENT: {}", msg);
      break;
    }
    case RTC_ERROR_INVALID_OPERATION: {
      XR_LOGE("INVALID_OPERATION: {}", msg);
      break;
    }
    case RTC_ERROR_OUT_OF_MEMORY: {
      XR_LOGE("OUT_OF_MEMORY: {}", msg);
      break;
    }
    case RTC_ERROR_UNSUPPORTED_CPU: {
      XR_LOGE("UNSUPPORTED_CPU: {}", msg);
      break;
    }
    case RTC_ERROR_CANCELLED: {
      XR_LOGE("CANCELLED: {}", msg);
      break;
    }
#if RTC_VERSION >= 40300
    case RTC_ERROR_LEVEL_ZERO_RAYTRACING_SUPPORT_MISSING: {
      XR_LOGE("LEVEL_ZERO_RAYTRACING_SUPPORT_MISSING: {}", msg);
      break;
    }
#endif
    case RTC_ERROR_NONE: {
      break; // There's no error.
    }
    case RTC_ERROR_UNKNOWN: {
      XR_LOGE("UNKNOWN_ERROR: {}", msg);
      break;
    }
  }
}

/**
 * @brief The callback invoked when Embree wants to create a leaf node.
 * @param alloc Thread-local memory allocator of the thread that calls the function.
 * @param prims The array of primitives to store in this leaf.
 * @param numPrims The number of primitives in the prims array. By default length is 1.
 * @param userPtr Optional pointer to the user data.
 * @return A void pointer to the created leaf node.
 */
void* createLeafNode(
    RTCThreadLocalAllocator alloc,
    const RTCBuildPrimitive* prims,
    size_t /*numPrims*/,
    void* /*userPtr*/) {
  void* ptr = rtcThreadLocalAlloc(alloc, sizeof(BVHEmbree::Node), alignof(BVHEmbree::Node));
  // Create a node here with placement new; Embree will free the memory once the BVH
  // handle is released.
  return static_cast<void*>(new (ptr) BVHEmbree::Node(prims[0])); // NOLINT
}

/**
 * @brief The callback invoked when Embree wants to create an inner/non-leaf node.
 * @param alloc Thread-local memory allocator of the thread that calls the function.
 * @param numChildren The number of children to store in this node, by default it is 2.
 * @param userPtr Optional pointer to the user data.
 * @return A void pointer to the created inner node.
 */
void* createInnerNode(
    RTCThreadLocalAllocator alloc,
    unsigned int /*numChildren*/,
    void* /*userPtr*/) {
  void* ptr = rtcThreadLocalAlloc(alloc, sizeof(BVHEmbree::Node), alignof(BVHEmbree::Node));
  // Create a node here with placement new; Embree will free the memory once the BVH
  // handle is released.
  return static_cast<void*>(new (ptr) BVHEmbree::Node());
}

/**
 * @brief The callback invoked when Embree wants to assign the children nodes to an inner/non-leaf
 * node.
 * @param nodePtr Pointer to the inner node.
 * @param childPtr Pointer to children nodes.
 * @param numChildren Number of values in childPtr. By default, the number is 2.
 * @param userPtr Optional pointer to the user data.
 */
void setChildren(void* nodePtr, void** childPtr, unsigned int /*numChildren*/, void* /*userPtr*/) {
  auto* node = static_cast<BVHEmbree::Node*>(nodePtr);
  node->children[0] = static_cast<BVHEmbree::Node*>(childPtr[0]); // NOLINT
  node->children[1] = static_cast<BVHEmbree::Node*>(childPtr[1]); // NOLINT
}

/**
 * @brief The callback invoked when Embree wants to set bounds for children of an inner node.
 * @param nodePtr Pointer to the inner node.
 * @param bounds Pointer to the boundary structures of the children.
 * @param numChildren Number of values in bounds. By default, the number is 2.
 * @param userPtr Optional pointer to the user data.
 */
void setBounds(
    void* nodePtr,
    const RTCBounds** bounds,
    unsigned int /*numChildren*/,
    void* /*userPtr*/) {
  auto* node = static_cast<BVHEmbree::Node*>(nodePtr);
  node->bounds[0] = *bounds[0]; // NOLINT
  node->bounds[1] = *bounds[1]; // NOLINT
}

/**
 * @brief Performs AABB-AABB intersection with Eigen. The method is asymmetric
 * in its inputs because bounding box 'b' always stays the same throughout queries.
 * @param a Bounding box that has aligned memory for min and max corners for aligned loading.
 * @param bLo Min corner of the query bounding box.
 * @param bHi Max corner of the query bounding box.
 * @return true if the bounding boxes overlap, false otherwise.
 */
inline bool aabbIntersectEigen(const RTCBounds& a, const Eigen::AlignedBox3d& b) {
  const Eigen::AlignedBox3d aBox(
      Eigen::Vector3d{a.lower_x, a.lower_y, a.lower_z},
      Eigen::Vector3d{a.upper_x, a.upper_y, a.upper_z});
  return aBox.intersects(b);
}

/**
 * @brief Checks for an intersection between AABB and a ray using Eigen,
 * based on the slab method.
 * @return True on a valid intersection, false otherwise.
 */

inline bool aabbRayIntersectEigen(
    const RTCBounds& bbox,
    const Eigen::Vector3d& rayOrigin,
    const Eigen::Vector3d& rayInvDirection) {
  const Eigen::Vector3d bmin(bbox.lower_x, bbox.lower_y, bbox.lower_z);
  const Eigen::Vector3d bmax(bbox.upper_x, bbox.upper_y, bbox.upper_z);

  double tmin = 0.0;
  double tmax = std::numeric_limits<double>::max();
  for (int32_t i = 0; i < 3; ++i) {
    const double t1 = (bmin[i] - rayOrigin[i]) * rayInvDirection[i];
    const double t2 = (bmax[i] - rayOrigin[i]) * rayInvDirection[i];

    // These mins and maxs rely on standard NaN comparison to work for all edge-cases.
    tmin = std::min(std::max(tmin, t1), std::max(tmin, t2));
    tmax = std::max(std::min(tmax, t1), std::min(tmax, t2));
  }

  return tmin <= tmax;
}

int32_t embreeDeviceRefCount{0};
RTCDevice embreeDevice{nullptr};

RTCDevice createEmbreeDevice(std::optional<uint32_t> embreeThreadCount) {
  // Recommended by Embree docs to slightly increase performance in some cases.
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

  // Please see https://spec.oneapi.io/oneart/latest/embree-spec.html#rtcnewdevice.
  // 0 implies all available hardware threads.
  const uint32_t threadCount{embreeThreadCount.value_or(0)};
  const std::string params{fmt::format("threads={}", threadCount)};
  RTCDevice device = rtcNewDevice(params.c_str());
  rtcSetDeviceErrorFunction(device, logEmbreeError, nullptr);
  return device;
}

} // namespace

BVHEmbree::Node::Node(const RTCBuildPrimitive& bounds) : primID(bounds.primID), isLeaf{true} {}

BVHEmbree::BVHEmbree(std::optional<uint32_t> embreeThreadCount)
    : buildArguments_{rtcDefaultBuildArguments()} {
  XR_LOGT("Constructing Embree BVH...");
  if (embreeDeviceRefCount == 0) {
    embreeDevice = createEmbreeDevice(embreeThreadCount);
  }
  device_ = embreeDevice;
  ++embreeDeviceRefCount;

  bvh_ = rtcNewBVH(device_);
  buildArguments_.byteSize = sizeof(buildArguments_);
  buildArguments_.buildFlags = RTC_BUILD_FLAG_DYNAMIC;
  buildArguments_.buildQuality = RTC_BUILD_QUALITY_LOW;
  buildArguments_.maxBranchingFactor = 2;
  buildArguments_.maxDepth = 1024;
  buildArguments_.sahBlockSize = 1;
  buildArguments_.minLeafSize = 1;
  buildArguments_.maxLeafSize = 1;
  buildArguments_.traversalCost = 1.0f;
  buildArguments_.intersectionCost = 1.0f;
  buildArguments_.bvh = bvh_;
  buildArguments_.createNode = createInnerNode;
  buildArguments_.setNodeChildren = setChildren;
  buildArguments_.setNodeBounds = setBounds;
  buildArguments_.createLeaf = createLeafNode;
  buildArguments_.splitPrimitive = nullptr;
  buildArguments_.buildProgress = nullptr;
  buildArguments_.userPtr = nullptr;
}

BVHEmbree::~BVHEmbree() {
  rtcReleaseBVH(bvh_);

  --embreeDeviceRefCount;
  if (embreeDeviceRefCount == 0) {
    rtcReleaseDevice(embreeDevice);
  }
}

BVHEmbree::BVHEmbree(BVHEmbree&& other) noexcept
    : device_{other.device_},
      buildArguments_{other.buildArguments_},
      bvh_{std::exchange(other.bvh_, nullptr)},
      root_{std::exchange(other.root_, nullptr)} {
  ++embreeDeviceRefCount;
}

BVHEmbree& BVHEmbree::operator=(BVHEmbree&& other) noexcept {
  if (this == &other) {
    return *this;
  }

  device_ = other.device_;
  buildArguments_ = other.buildArguments_;

  rtcReleaseBVH(bvh_); // Release the old BVH before grabbing the one from "other".
  bvh_ = std::exchange(other.bvh_, nullptr);
  root_ = std::exchange(other.root_, nullptr);
  return *this;
}

void BVHEmbree::setBoundingBoxes(const std::vector<BoundingBoxd>& bboxes) {
  std::vector<RTCBuildPrimitive> prims;
  {
    XR_PROFILE_EVENT("BVHEmbree::convert_bbox_to_prims");
    prims.reserve(bboxes.size() * 2);
    prims.resize(bboxes.size());
    dispenso::parallel_for(0, bboxes.size(), [&bboxes, &prims](const size_t i) {
      prims[i].lower_x = static_cast<float>(bboxes[i].min().x());
      prims[i].lower_y = static_cast<float>(bboxes[i].min().y());
      prims[i].lower_z = static_cast<float>(bboxes[i].min().z());
      prims[i].upper_x = static_cast<float>(bboxes[i].max().x());
      prims[i].upper_y = static_cast<float>(bboxes[i].max().y());
      prims[i].upper_z = static_cast<float>(bboxes[i].max().z());
      prims[i].geomID = 0;
      prims[i].primID = bboxes[i].id;
    });
  }

  setPrimitives(prims);
}

void BVHEmbree::setPrimitives(std::vector<RTCBuildPrimitive>& prims) {
  buildArguments_.primitives = prims.data();
  buildArguments_.primitiveCount = prims.size();
  buildArguments_.primitiveArrayCapacity = prims.capacity();

  if (prims.empty()) {
    // If we have built a BVH and our intention is to reclaim all the memory
    // with the empty prims vector, we need to recreate the handle.
    // rtcReleaseBvh(...) also releases all the memory
    // allocated by rtcThreadLocalAlloc(...) used in the build callbacks.
    if (root_) {
      rtcReleaseBVH(bvh_);
      bvh_ = rtcNewBVH(device_);
      root_ = nullptr;
    }

    return;
  }

  root_ = static_cast<Node*>(rtcBuildBVH(&buildArguments_));
}

uint32_t BVHEmbree::query(const BoundingBoxd& box, QueryBuffer& hits) const {
  if (!root_ || buildArguments_.primitiveCount == 0) {
    return 0;
  }

  std::array<const Node*, 1024> todo{};
  int32_t stackPtr = 0;
  todo[stackPtr] = root_;

  uint32_t hitCount{0};
  while (stackPtr >= 0 && hitCount < hits.size()) {
    const Node* node = todo[stackPtr--];

    // If we found a leaf, add it to the hit buffer.
    if (node->isLeaf) {
      hits[hitCount++] = node->primID;
    } else {
      // If we are processing an inner node, potentially add its children to the stack.
      if (aabbIntersectEigen(node->bounds[0], box.aabb)) {
        todo[++stackPtr] = node->children[0];
      }
      if (aabbIntersectEigen(node->bounds[1], box.aabb)) {
        todo[++stackPtr] = node->children[1];
      }
    }
  }

  return hitCount;
}

std::vector<unsigned int> BVHEmbree::query(const BoundingBoxd& box) const {
  std::vector<unsigned int> hits{};
  if (!root_ || buildArguments_.primitiveCount == 0) {
    return hits;
  }

  std::array<const Node*, 1024> todo{};
  int32_t stackPtr = 0;
  todo[stackPtr] = root_;

  while (stackPtr >= 0) {
    const Node* node = todo[stackPtr--];

    // If we found a leaf, add it to the hit buffer.
    if (node->isLeaf) {
      hits.push_back(node->primID);
    } else {
      // If we are processing an inner node, potentially add its children to the stack.
      if (aabbIntersectEigen(node->bounds[0], box.aabb)) {
        todo[++stackPtr] = node->children[0];
      }
      if (aabbIntersectEigen(node->bounds[1], box.aabb)) {
        todo[++stackPtr] = node->children[1];
      }
    }
  }

  return hits;
}

std::vector<unsigned int> BVHEmbree::query(
    const Eigen::Vector3d& origin,
    const Eigen::Vector3d& direction) const {
  std::vector<unsigned int> hits;
  if (!root_ || buildArguments_.primitiveCount == 0) {
    return hits;
  }

  const Eigen::Vector3d rayInvDirection = 1.0 / direction.array();

  // Working set
  std::array<const Node*, 1024> todo{};
  int32_t stackPtr = 0;
  todo[stackPtr] = root_;

  while (stackPtr >= 0) {
    const Node* node = todo[stackPtr--];

    // We hit a leaf, we can add it to the hit buffer.
    if (node->isLeaf) {
      hits.push_back(node->primID);
    } else {
      // If we are processing an inner node, potentially add its children to the stack.
      if (aabbRayIntersectEigen(node->bounds[0], origin, rayInvDirection)) {
        todo[++stackPtr] = node->children[0];
      }

      if (aabbRayIntersectEigen(node->bounds[1], origin, rayInvDirection)) {
        todo[++stackPtr] = node->children[1];
      }
    }
  }
  return hits;
}

Size BVHEmbree::getPrimitiveCount() const {
  return buildArguments_.primitiveCount;
}

} // namespace axel
