/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "axel/SimdKdTree.h"

#include <drjit/array.h>
#include <drjit/array_router.h>
#include <drjit/packet.h>
#include <drjit/util.h>
#include <gsl/gsl>

#include "axel/Profile.h"

namespace axel {

namespace {

inline constexpr size_t kSimdPacketSize = drjit::DefaultSize;
inline constexpr size_t kSimdAlignment = kSimdPacketSize * 4;

using FloatP = drjit::Packet<float, kSimdPacketSize>;
using DoubleP = drjit::Packet<double, kSimdPacketSize>;
using IntP = drjit::Packet<int, kSimdPacketSize>;

using Vector3fP = drjit::Array<FloatP, 3>;
using Vector2fP = drjit::Array<FloatP, 2>;

template <int Dim>
struct VecfP;

template <>
struct VecfP<2> {
  static constexpr int Dim = 2;

  FloatP data[Dim];

  const FloatP& operator[](size_t i) const {
    return data[i];
  }

  FloatP& operator[](size_t i) {
    return data[i];
  }
};

template <>
struct VecfP<3> {
  static constexpr int Dim = 3;

  FloatP data[Dim];

  const FloatP& operator[](size_t i) const {
    return data[i];
  }

  FloatP& operator[](size_t i) {
    return data[i];
  }
};

template <>
struct VecfP<4> {
  static constexpr int Dim = 4;

  FloatP data[Dim];

  const FloatP& operator[](size_t i) const {
    return data[i];
  }

  FloatP& operator[](size_t i) {
    return data[i];
  }
};

} // namespace

template <int32_t nDim>
struct SimdKdTreef<nDim>::Implementation {
  /// In order to enable the use of SIMD intrinsics, we will store the points in blocks of
  /// kSimdPacketSize. Empty point values will get kFarValueFloat for their location (so we can
  /// safely use it in difference and squared norm operations) and std::numeric_limits<int>::max()
  /// for their index.
  struct PointBlock {
    VecfP<nDim> values; // [ {x1 x2 x3 x4 x5 x6 x7 x8}, {y1 y2 y3 y4 y5 y6 y7 y8}, ... ]
    IntP indices; // [i1 i2 i3 i4 i5 i6 i7 i8]
  };

  struct NormalBlock {
    VecfP<nDim> values;
  };

  struct ColorBlock {
    VecfP<4> values;
  };

  std::vector<PointBlock> pointBlocks;
  std::vector<NormalBlock> normalBlocks;
  std::vector<ColorBlock> colorBlocks;
};

template <int32_t nDim>
SimdKdTreef<nDim>::SimdKdTreef(
    std::span<const Vec> points_in,
    std::span<const Vec> normals_in,
    std::span<const Col> colors_in)
    : impl_(std::make_unique<Implementation>()) {
  init(points_in, normals_in, colors_in);
}

template <int32_t nDim>
SimdKdTreef<nDim>::~SimdKdTreef() {
  // Do nothing
}

template <int32_t nDim>
bool SimdKdTreef<nDim>::empty() const {
  return (numPoints_ == 0);
}

template <int32_t nDim>
const typename SimdKdTreef<nDim>::Box& SimdKdTreef<nDim>::boundingBox() const {
  return bbox_;
}

template <int32_t nDim>
typename SimdKdTreef<nDim>::SizeType SimdKdTreef<nDim>::size() const {
  return numPoints_;
}

template <int32_t nDim>
typename SimdKdTreef<nDim>::SizeType SimdKdTreef<nDim>::depth() const {
  return depth_;
}

template <int32_t nDim>
std::tuple<bool, typename SimdKdTreef<nDim>::SizeType, typename SimdKdTreef<nDim>::Scalar>
SimdKdTreef<nDim>::closestPoint(const Vec& queryPoint, Scalar maxSqrDist) const {
  bool foundPoint = false;
  SizeType bestPoint = std::numeric_limits<SizeType>::max();
  Scalar bestSqrDist = maxSqrDist;

  if (empty()) {
    return std::make_tuple(foundPoint, bestPoint, bestSqrDist);
  }

  // Use an explicit stack for speed:
  std::array<SizeType, kMaxDepth + 1> nodeStack;

  // Start with just the root on the stack:
  SizeType stackSize = 1;
  nodeStack[0] = root_;

  VecfP<nDim> query_p;
  for (SizeType iDim = 0; iDim < nDim; ++iDim) {
    query_p[iDim] = queryPoint[iDim];
  }

  while (stackSize != 0) {
    // Pop the top of the stack:
    const SizeType cur = nodeStack[--stackSize];
    const auto& curNode = nodes_[cur];

    // Check again if the current best distance is smaller than the distance to the current node
    // split. if so we can skip the node
    if (curNode.box.squaredExteriorDistance(queryPoint) > bestSqrDist) {
      continue;
    }

    if (curNode.isLeaf()) {
      const SizeType pointBlocksStart = curNode.pointBlocksStart;
      const SizeType pointBlocksEnd = curNode.pointBlocksEnd;
      XR_CHECK(pointBlocksEnd > pointBlocksStart, "Invalid point block indices.");

      FloatP bestSqrDistBlock = bestSqrDist;
      IntP bestSqrDistIndicesBlock = INT_MAX;

      for (SizeType iBlock = pointBlocksStart; iBlock != pointBlocksEnd; ++iBlock) {
        const typename Implementation::PointBlock& block = impl_->pointBlocks[iBlock];
        FloatP sqrDist = 0;
        for (SizeType iDim = 0; iDim < nDim; ++iDim) {
          const FloatP dimDiff = block.values[iDim] - query_p[iDim];
          sqrDist += dimDiff * dimDiff;
        }

        const auto lessThanMask = sqrDist < bestSqrDistBlock;
        bestSqrDistBlock = drjit::select(lessThanMask, sqrDist, bestSqrDistBlock);
        bestSqrDistIndicesBlock =
            drjit::select(lessThanMask, block.indices, bestSqrDistIndicesBlock);
      }

      alignas(kSimdAlignment) float bestSqrDist_extract[kSimdPacketSize];
      drjit::store<FloatP>(bestSqrDist_extract, bestSqrDistBlock);
      alignas(kSimdAlignment) SizeType bestSqrDistIndices_extract[kSimdPacketSize];
      drjit::store<IntP>(bestSqrDistIndices_extract, bestSqrDistIndicesBlock);

      // Find best point horizontally within the current best packet
      for (SizeType k = 0; k < gsl::narrow<SizeType>(kSimdPacketSize); ++k) {
        if (bestSqrDist_extract[k] < bestSqrDist) {
          XR_CHECK(
              bestSqrDistIndices_extract[k] != INT_MAX,
              "The best squared distance index shouldn't be INT_MAX here.");
          foundPoint = true;
          bestPoint = bestSqrDistIndices_extract[k];
          bestSqrDist = bestSqrDist_extract[k];
        }
      }
    } else {
      // We need to descend on
      const unsigned char splitDim = curNode.splitDim;
      const Scalar splitVal = curNode.splitVal;
      const Scalar queryPointVal = queryPoint[splitDim];
      const Scalar distToSplit = (queryPointVal - splitVal);
      const Scalar sqrDistToSplit = distToSplit * distToSplit;

      if (queryPointVal >= splitVal) {
        // We only need to descend if the distance to the split is less than the distance to our
        // best point so far.
        if (sqrDistToSplit < bestSqrDist) {
          nodeStack[stackSize++] = curNode.leftChild;
        }

        // Since the query point is in the right half of the space we think it more likely that
        // we'll find the closest point in that half of the space, so we should descend on it first.
        // Since we grab the top entry of the stack first, this implies pushing it last.
        nodeStack[stackSize++] = curNode.rightChild;
      } else {
        // Same as above:
        if (sqrDistToSplit < bestSqrDist) {
          nodeStack[stackSize++] = curNode.rightChild;
        }

        nodeStack[stackSize++] = curNode.leftChild;
      }
    }
  }

  XR_CHECK(!foundPoint || bestPoint < numPoints_);
  return std::make_tuple(foundPoint, bestPoint, bestSqrDist);
}

template <int32_t nDim>
std::tuple<bool, typename SimdKdTreef<nDim>::SizeType, typename SimdKdTreef<nDim>::Scalar>
SimdKdTreef<nDim>::closestPoint(
    const Vec& queryPoint,
    const Vec& queryNormal,
    Scalar maxSqrDist,
    Scalar maxNormalDot) const {
  XR_PROFILE_EVENT("SimdKdTreef: ClosedPoint");

  bool foundPoint = false;
  SizeType bestPoint = std::numeric_limits<SizeType>::max();
  Scalar bestSqrDist = maxSqrDist;

  XR_CHECK(hasNormals_);

  if (empty()) {
    return std::make_tuple(foundPoint, bestPoint, bestSqrDist);
  }

  // Use an explicit stack for speed:
  std::array<SizeType, kMaxDepth + 1> nodeStack;

  // Start with just the root on the stack:
  SizeType stackSize = 1;
  nodeStack[0] = root_;

  VecfP<nDim> query_p;
  VecfP<nDim> query_n;
  for (SizeType iDim = 0; iDim < nDim; ++iDim) {
    query_p[iDim] = queryPoint[iDim];
    query_n[iDim] = queryNormal[iDim];
  }

  while (stackSize != 0) {
    // Pop the top of the stack:
    const SizeType cur = nodeStack[--stackSize];
    const auto& curNode = nodes_[cur];

    // Check again if the current best distance is smaller than the distance to the current node
    // split. if so we can skip the node
    if (curNode.box.squaredExteriorDistance(queryPoint) > bestSqrDist) {
      continue;
    }

    if (curNode.isLeaf()) {
      const SizeType pointBlocksStart = curNode.pointBlocksStart;
      const SizeType pointBlocksEnd = curNode.pointBlocksEnd;
      XR_CHECK(pointBlocksEnd > pointBlocksStart, "Invalid point block indices.");

      FloatP bestSqrDistBlock = bestSqrDist;
      IntP bestSqrDistIndicesBlock = INT_MAX;

      for (SizeType iBlock = pointBlocksStart; iBlock != pointBlocksEnd; ++iBlock) {
        const typename Implementation::PointBlock& pointBlock = impl_->pointBlocks[iBlock];
        FloatP sqrDist = 0;
        for (SizeType iDim = 0; iDim < nDim; ++iDim) {
          const FloatP dimDiff = pointBlock.values[iDim] - query_p[iDim];
          sqrDist += dimDiff * dimDiff;
        }

        // Skip if no closer point is found
        const auto lessThanMask = sqrDist < bestSqrDistBlock;
        if (drjit::none(lessThanMask)) {
          continue;
        }

        // Take the dot product between the query normal and the point normals:
        const typename Implementation::NormalBlock& normalBlock = impl_->normalBlocks[iBlock];
        FloatP normalDotProd = 0;
        for (SizeType iDim = 0; iDim < nDim; ++iDim) {
          normalDotProd += query_n[iDim] * normalBlock.values[iDim];
        }

        // Only take points where the distance is smaller and the normal agrees.
        const auto finalMask = lessThanMask & (normalDotProd >= maxNormalDot);
        bestSqrDistBlock = drjit::select(finalMask, sqrDist, bestSqrDistBlock);
        bestSqrDistIndicesBlock =
            drjit::select(finalMask, pointBlock.indices, bestSqrDistIndicesBlock);
      }

      alignas(kSimdAlignment) float bestSqrDist_extract[kSimdPacketSize];
      drjit::store<FloatP>(bestSqrDist_extract, bestSqrDistBlock);
      alignas(kSimdAlignment) SizeType bestSqrDistIndices_extract[kSimdPacketSize];
      drjit::store<IntP>(bestSqrDistIndices_extract, bestSqrDistIndicesBlock);

      // Find best point horizontally within the current best packet
      for (SizeType k = 0; k < gsl::narrow<SizeType>(kSimdPacketSize); ++k) {
        if (bestSqrDistIndices_extract[k] == INT_MAX) {
          continue;
        }

        if (bestSqrDist_extract[k] < bestSqrDist) {
          foundPoint = true;
          bestPoint = bestSqrDistIndices_extract[k];
          bestSqrDist = bestSqrDist_extract[k];
        }
      }
    } else {
      // We need to descend on
      const unsigned char splitDim = curNode.splitDim;
      const Scalar splitVal = curNode.splitVal;
      const Scalar queryPointVal = queryPoint[splitDim];
      const Scalar distToSplit = (queryPointVal - splitVal);
      const Scalar sqrDistToSplit = distToSplit * distToSplit;

      if (queryPointVal >= splitVal) {
        // We only need to descend if the distance to the split is less than the distance to our
        // best point so far.
        if (sqrDistToSplit < bestSqrDist) {
          nodeStack[stackSize++] = curNode.leftChild;
        }

        // Since the query point is in the right half of the space we think it more likely that
        // we'll find the closest point in that half of the space, so we should descend on it first.
        // Since we grab the top entry of the stack first, this implies pushing it last.
        nodeStack[stackSize++] = curNode.rightChild;
      } else {
        // Same as above:
        if (sqrDistToSplit < bestSqrDist) {
          nodeStack[stackSize++] = curNode.rightChild;
        }

        nodeStack[stackSize++] = curNode.leftChild;
      }
    }
  }

  XR_CHECK(!foundPoint || bestPoint < numPoints_);
  return std::make_tuple(foundPoint, bestPoint, bestSqrDist);
}

template <int32_t nDim>
std::tuple<bool, typename SimdKdTreef<nDim>::SizeType, typename SimdKdTreef<nDim>::Scalar>
SimdKdTreef<nDim>::closestPoint(
    const Vec& queryPoint,
    const Vec& queryNormal,
    const Col& queryColor,
    Scalar maxSqrDist,
    Scalar maxNormalDot,
    Scalar maxColorSqrDist) const {
  Scalar bestSqrDist = maxSqrDist;
  SizeType bestPoint = std::numeric_limits<SizeType>::max();
  bool foundPoint = false;

  XR_CHECK(hasNormals_);

  if (empty()) {
    return std::make_tuple(foundPoint, bestPoint, bestSqrDist);
  }

  // Use an explicit stack for speed:
  std::array<SizeType, kMaxDepth + 1> nodeStack;

  // Start with just the root on the stack:
  SizeType stackSize = 1;
  nodeStack[0] = root_;

  VecfP<nDim> query_p;
  VecfP<nDim> query_n;
  VecfP<kColorDimensions> query_c;
  for (SizeType iDim = 0; iDim < nDim; ++iDim) {
    query_p[iDim] = queryPoint[iDim];
    query_n[iDim] = queryNormal[iDim];
  }
  for (SizeType iDim = 0; iDim < kColorDimensions; ++iDim) {
    query_c[iDim] = queryColor[iDim];
  }

  while (stackSize != 0) {
    // Pop the top of the stack:
    const SizeType cur = nodeStack[--stackSize];
    const auto& curNode = nodes_[cur];

    // Check again if the current best distance is smaller than the distance to the current node
    // split. if so we can skip the node
    if (curNode.box.squaredExteriorDistance(queryPoint) > bestSqrDist) {
      continue;
    }

    if (curNode.isLeaf()) {
      const SizeType pointBlocksStart = curNode.pointBlocksStart;
      const SizeType pointBlocksEnd = curNode.pointBlocksEnd;
      XR_CHECK(pointBlocksEnd > pointBlocksStart, "Invalid point block indices.");

      FloatP bestSqrDistBlock = bestSqrDist;
      IntP bestSqrDistIndicesBlock = INT_MAX;

      for (SizeType iBlock = pointBlocksStart; iBlock != pointBlocksEnd; ++iBlock) {
        const typename Implementation::PointBlock& pointBlock = impl_->pointBlocks[iBlock];
        FloatP sqrDist = 0;
        for (SizeType iDim = 0; iDim < nDim; ++iDim) {
          const FloatP dimDiff = pointBlock.values[iDim] - query_p[iDim];
          sqrDist += dimDiff * dimDiff;
        }

        // Skip if no closer point is found
        const auto lessThanMask = sqrDist < bestSqrDistBlock;
        if (drjit::none(lessThanMask)) {
          continue;
        }

        // Take the dot product between the query normal and the point normals:
        const typename Implementation::NormalBlock& normalBlock = impl_->normalBlocks[iBlock];
        FloatP normalDotProd = 0;
        for (SizeType iDim = 0; iDim < nDim; ++iDim) {
          normalDotProd += query_n[iDim] * normalBlock.values[iDim];
        }

        // Take the squared difference between the query color and the point colors:
        const typename Implementation::ColorBlock& colorBlock = impl_->colorBlocks[iBlock];
        FloatP colorSqrDist = 0;
        for (SizeType iDim = 0; iDim < kColorDimensions - 1; ++iDim) // skipping for alpha for now
        {
          const FloatP dimMul = queryColor[iDim] - colorBlock.values[iDim];
          colorSqrDist += dimMul * dimMul;
        }
        // Multiply with certainty value (last color value)
        colorSqrDist *= colorBlock.values[kColorDimensions - 1];
        colorSqrDist *= queryColor[kColorDimensions - 1];

        // Only take points where the distance is smaller and the normal agrees.
        const auto normalMask = lessThanMask & (normalDotProd >= maxNormalDot);
        const auto finalMask = normalMask & (colorSqrDist <= maxColorSqrDist);
        bestSqrDistBlock = drjit::select(finalMask, sqrDist, bestSqrDistBlock);
        bestSqrDistIndicesBlock =
            drjit::select(finalMask, pointBlock.indices, bestSqrDistIndicesBlock);
      }

      alignas(kSimdAlignment) float bestSqrDist_extract[kSimdPacketSize];
      drjit::store<FloatP>(bestSqrDist_extract, bestSqrDistBlock);
      alignas(kSimdAlignment) SizeType bestSqrDistIndices_extract[kSimdPacketSize];
      drjit::store<IntP>(bestSqrDistIndices_extract, bestSqrDistIndicesBlock);

      for (SizeType k = 0; k < gsl::narrow<SizeType>(kSimdPacketSize); ++k) {
        if (bestSqrDistIndices_extract[k] == INT_MAX) {
          continue;
        }

        if (bestSqrDist_extract[k] < bestSqrDist) {
          foundPoint = true;
          bestPoint = bestSqrDistIndices_extract[k];
          bestSqrDist = bestSqrDist_extract[k];
        }
      }
    } else {
      // We need to descend on
      const unsigned char splitDim = curNode.splitDim;
      const Scalar splitVal = curNode.splitVal;
      const Scalar queryPointVal = queryPoint[splitDim];
      const Scalar distToSplit = (queryPointVal - splitVal);
      const Scalar sqrDistToSplit = distToSplit * distToSplit;

      if (queryPointVal >= splitVal) {
        // We only need to descend if the distance to the split is less than the distance to our
        // best point so far.
        if (sqrDistToSplit < bestSqrDist) {
          nodeStack[stackSize++] = curNode.leftChild;
        }

        // Since the query point is in the right half of the space we think it more likely that
        // we'll find the closest point in that half of the space, so we should descend on it first.
        // Since we grab the top entry of the stack first, this implies pushing it last.
        nodeStack[stackSize++] = curNode.rightChild;
      } else {
        // Same as above:
        if (sqrDistToSplit < bestSqrDist) {
          nodeStack[stackSize++] = curNode.rightChild;
        }

        nodeStack[stackSize++] = curNode.leftChild;
      }
    }
  }

  XR_CHECK(!foundPoint || bestPoint < numPoints_);
  return std::make_tuple(foundPoint, bestPoint, bestSqrDist);
}

template <int32_t nDim>
std::tuple<bool, typename SimdKdTreef<nDim>::SizeType, typename SimdKdTreef<nDim>::Scalar>
SimdKdTreef<nDim>::closestPointWithAcceptance(
    const Vec& queryPoint,
    Scalar maxSqrDist,
    const std::function<bool(SizeType pointIndex)>& acceptanceFunction) const {
  Scalar bestSqrDist = maxSqrDist;
  SizeType bestPoint = std::numeric_limits<SizeType>::max();
  bool foundPoint = false;

  if (empty()) {
    return std::make_tuple(foundPoint, bestPoint, bestSqrDist);
  }

  // Use an explicit stack for speed:
  std::array<SizeType, kMaxDepth + 1> nodeStack;

  // Start with just the root on the stack:
  SizeType stackSize = 1;
  nodeStack[0] = root_;

  VecfP<nDim> query_p;
  for (SizeType iDim = 0; iDim < nDim; ++iDim) {
    query_p[iDim] = queryPoint[iDim];
  }

  while (stackSize != 0) {
    // Pop the top of the stack:
    const SizeType cur = nodeStack[--stackSize];
    const auto& curNode = nodes_[cur];

    // Check again if the current best distance is smaller than the distance to the current node
    // split. if so we can skip the node
    if (curNode.box.squaredExteriorDistance(queryPoint) > bestSqrDist) {
      continue;
    }

    if (curNode.isLeaf()) {
      const SizeType pointBlocksStart = curNode.pointBlocksStart;
      const SizeType pointBlocksEnd = curNode.pointBlocksEnd;
      XR_CHECK(pointBlocksEnd > pointBlocksStart, "Invalid point block indices.");

      FloatP bestSqrDistBlock = bestSqrDist;
      IntP bestSqrDistIndicesBlock = INT_MAX;

      for (SizeType iBlock = pointBlocksStart; iBlock != pointBlocksEnd; ++iBlock) {
        const typename Implementation::PointBlock& pointBlock = impl_->pointBlocks[iBlock];
        FloatP sqrDist = 0;
        for (SizeType iDim = 0; iDim < nDim; ++iDim) {
          const FloatP dimDiff = pointBlock.values[iDim] - query_p[iDim];
          sqrDist += dimDiff * dimDiff;
        }

        auto lessThanMask = sqrDist < bestSqrDistBlock;
        for (auto i = 0u; i < getSimdPacketSize(); ++i) {
          lessThanMask[i] = lessThanMask[i] && acceptanceFunction(pointBlock.indices[i]);
        }
        bestSqrDistBlock = drjit::select(lessThanMask, sqrDist, bestSqrDistBlock);
        bestSqrDistIndicesBlock =
            drjit::select(lessThanMask, pointBlock.indices, bestSqrDistIndicesBlock);
      }

      alignas(kSimdAlignment) float bestSqrDist_extract[kSimdPacketSize];
      drjit::store<FloatP>(bestSqrDist_extract, bestSqrDistBlock);
      alignas(kSimdAlignment) SizeType bestSqrDistIndices_extract[kSimdPacketSize];
      drjit::store<IntP>(bestSqrDistIndices_extract, bestSqrDistIndicesBlock);

      // Find best point horizontally within the current best packet
      for (SizeType k = 0; k < gsl::narrow<SizeType>(kSimdPacketSize); ++k) {
        if (bestSqrDist_extract[k] < bestSqrDist) {
          XR_CHECK(bestSqrDistIndices_extract[k] != INT_MAX, "Invalid index.");
          XR_CHECK(bestSqrDistIndices_extract[k] < numPoints_, "Invalid index.");
          foundPoint = true;
          bestPoint = bestSqrDistIndices_extract[k];
          bestSqrDist = bestSqrDist_extract[k];
        }
      }
    } else {
      // We need to descend on
      const unsigned char splitDim = curNode.splitDim;
      const Scalar splitVal = curNode.splitVal;
      const Scalar queryPointVal = queryPoint[splitDim];
      const Scalar distToSplit = (queryPointVal - splitVal);
      const Scalar sqrDistToSplit = distToSplit * distToSplit;

      if (queryPointVal >= splitVal) {
        // We only need to descend if the distance to the split is less than the distance to our
        // best point so far.
        if (sqrDistToSplit < bestSqrDist) {
          nodeStack[stackSize++] = curNode.leftChild;
        }

        // Since the query point is in the right half of the space we think it more likely that
        // we'll find the closest point in that half of the space, so we should descend on it first.
        // Since we grab the top entry of the stack first, this implies pushing it last.
        nodeStack[stackSize++] = curNode.rightChild;
      } else {
        // Same as above:
        if (sqrDistToSplit < bestSqrDist) {
          nodeStack[stackSize++] = curNode.rightChild;
        }

        nodeStack[stackSize++] = curNode.leftChild;
      }
    }
  }

  XR_CHECK(!foundPoint || bestPoint < numPoints_);
  return std::make_tuple(foundPoint, bestPoint, bestSqrDist);
}

template <int32_t nDim>
void SimdKdTreef<nDim>::pointsInNSphere(
    const Vec& center_in,
    Scalar radius,
    std::vector<SizeType>& points) const {
  XR_CHECK(radius >= Scalar(0));

  points.clear();
  if (empty()) {
    return;
  }

  const Scalar radiusSqr = radius * radius;
  const FloatP radiusSqrBlock = radiusSqr;

  // Use an explicit stack for speed:
  std::array<SizeType, kMaxDepth + 1> nodeStack;

  // Start with just the root on the stack:
  SizeType stackSize = 1;
  nodeStack[0] = root_;

  VecfP<nDim> query_p;
  for (SizeType iDim = 0; iDim < nDim; ++iDim) {
    query_p[iDim] = center_in[iDim];
  }

  while (stackSize != 0) {
    // Pop the top of the stack:
    const SizeType cur = nodeStack[--stackSize];
    const auto& curNode = nodes_[cur];

    // Check if the distance is smaller than the distance to the current node split. if so we can
    // skip the node. This may help as the bbox is tighter than the split
    if (curNode.box.squaredExteriorDistance(center_in) > radiusSqr) {
      continue;
    }

    if (curNode.isLeaf()) {
      const SizeType pointBlocksStart = curNode.pointBlocksStart;
      const SizeType pointBlocksEnd = curNode.pointBlocksEnd;
      XR_CHECK(pointBlocksEnd > pointBlocksStart, "Invalid point block indices.");

      for (SizeType iBlock = pointBlocksStart; iBlock != pointBlocksEnd; ++iBlock) {
        const typename Implementation::PointBlock& block = impl_->pointBlocks[iBlock];
        FloatP sqrDist = 0;
        for (SizeType iDim = 0; iDim < nDim; ++iDim) {
          const FloatP dimDiff = block.values[iDim] - query_p[iDim];
          sqrDist += dimDiff * dimDiff;
        }

        if (drjit::none(sqrDist <= radiusSqrBlock)) {
          continue;
        }

        alignas(kSimdAlignment) float sqrDist_extract[kSimdPacketSize];
        drjit::store<FloatP>(sqrDist_extract, sqrDist);
        alignas(kSimdAlignment) SizeType indices_extract[kSimdPacketSize];
        drjit::store<IntP>(indices_extract, block.indices);

        for (SizeType k = 0; k < gsl::narrow<SizeType>(kSimdPacketSize); ++k) {
          if (sqrDist_extract[k] <= radiusSqr) {
            XR_CHECK(indices_extract[k] != INT_MAX, "Invalid index.");
            points.push_back(indices_extract[k]);
          }
        }
      }
    } else {
      // We potentially need to descend on both children. Unlike the closest point case, there's no
      // reason to favor one side over the other:
      const unsigned char splitDim = curNode.splitDim;
      const Scalar splitVal = curNode.splitVal;
      const Scalar centerVal = center_in[splitDim];

      if (centerVal - radius <= splitVal) {
        nodeStack[stackSize++] = curNode.leftChild;
      }

      if (centerVal + radius >= splitVal) {
        nodeStack[stackSize++] = curNode.rightChild;
      }
    }
  }
}

template <int32_t nDim>
void SimdKdTreef<nDim>::validate() const {
  std::vector<char> touchedPoints(numPoints_, 0);
  validateInternal(root_, bbox_, touchedPoints);

#ifndef NDEBUG
  // Make sure every point is in the tree exactly once:
  for (const auto& p : touchedPoints) {
    XR_CHECK(p == 1);
  }
#endif
}

template <int32_t nDim>
typename SimdKdTreef<nDim>::SizeType SimdKdTreef<nDim>::split(
    std::vector<std::pair<SizeType, Vec>>& points,
    std::span<const Vec> normals,
    std::span<const Col> colors,
    SizeType start,
    SizeType end,
    SizeType depth) {
  XR_CHECK(end > 0);

  depth_ = std::max(depth, depth_);

  // Calculate the actual bounding box of the points in this node
  Box box;
  for (SizeType i = start; i < end; ++i) {
    box.extend(points[i].second);
  }

  // Fast ceiling of an integer division:
  //   * 0 for 0
  //   * 1 for [kSimdPacketSize * 0 + 1, ..., kSimdPacketSize * 1]
  //   * 2 for [kSimdPacketSize * 1 + 1, ..., kSimdPacketSize * 2]
  //   * ...
  const SizeType nBlocks =
      (end - start) / getSimdPacketSize() + !!((end - start) % getSimdPacketSize());

  if (nBlocks <= 8 || depth >= kMaxDepth) {
    return createLeafNode(points, normals, colors, start, end, box);
  }

  // Split on the longest axis:
  typename Box::Index splitDim = 0;
  box.diagonal().maxCoeff(&splitDim);
  Scalar splitVal = box.center()[splitDim];

  // Reorder by the splitVal
  const auto it = std::partition(
      points.begin() + start, points.begin() + end, [=](const std::pair<SizeType, Vec>& e) -> bool {
        return e.second[splitDim] < splitVal;
      });

  // Get the index of the mid point
  SizeType mid = static_cast<SizeType>(std::distance(points.begin(), it));

  // If the split doesn't carve any blocks off, then we need to do something to avoid an infinite
  // recursion; in this case we'll fall back to computing the median partition:
  if ((mid - start) < gsl::narrow<SizeType>(getSimdPacketSize()) ||
      (end - mid) < gsl::narrow<SizeType>(getSimdPacketSize())) {
    mid = start + (end - start) / 2;
    std::nth_element(
        points.begin() + start,
        points.begin() + mid,
        points.begin() + end,
        [=](const std::pair<SizeType, Vec>& left, const std::pair<SizeType, Vec>& right) -> bool {
          return left.second[splitDim] < right.second[splitDim];
        });
    splitVal = points[mid].second[splitDim];
  }

  XR_CHECK(start < mid && mid < end);

  // Create a node. Note that we intentionally create the parent node before its children's nodes.
  // This way we ensure that at least the left child's node is always located right next to the
  // parent's node, which would minimize cache misses by maximazing the closeness to its children.
  // The tree structure would be something like:
  //                      [ 0]
  //                   /        \
  //          [ 1]                    [ 8]
  //         /    \                  /    \
  //    [ 2]        [ 5]        [ 9]        [12]
  //    /  \        /  \        /  \        /  \
  // [ 3]  [ 4]  [ 6]  [ 7]  [10]  [11]  [13]  [14]
  //
  // An alternative way would be putting the parent in the middle, but this could lead to two memory
  // fetches at worst for each child when the distance between the parent and its children while the
  // other approach would need only one for the right child at worst.
  const auto nodeId = gsl::narrow<SizeType>(nodes_.size());
  nodes_.emplace_back(
      splitVal,
      gsl::narrow<unsigned char>(splitDim),
      /* tempLeftChild */ 0,
      /* tempRightChild */ 0,
      box);

  // Now generate the 'left' and 'right' subtrees.
  const SizeType leftChild = split(points, normals, colors, start, mid, depth + 1);
  const SizeType rightChild = split(points, normals, colors, mid, end, depth + 1);

  // Set the parent node's children now that we know them.
  auto& parentNode = this->nodes_[nodeId];
  parentNode.leftChild = leftChild;
  parentNode.rightChild = rightChild;

  return nodeId;
}

template <int32_t nDim>
typename SimdKdTreef<nDim>::SizeType SimdKdTreef<nDim>::createLeafNode(
    std::vector<std::pair<SizeType, Vec>>& points,
    std::span<const Vec> normals,
    std::span<const Col> colors,
    SizeType start,
    SizeType end,
    const Box& box) {
  const SizeType blocksStart = gsl::narrow<SizeType>(impl_->pointBlocks.size());

  alignas(kSimdAlignment) float pValues[nDim][kSimdPacketSize];
  alignas(kSimdAlignment) float nValues[nDim][kSimdPacketSize];
  alignas(kSimdAlignment) float cValues[kColorDimensions][kSimdPacketSize];
  alignas(kSimdAlignment) std::array<SizeType, kSimdPacketSize> indices;

  for (SizeType curBlockStart = start; curBlockStart < end; curBlockStart += kSimdPacketSize) {
    // Initialize pValues, nValues, cValues, and indices with the default values
    for (SizeType iDim = 0; iDim < nDim; ++iDim) {
      drjit::store(pValues[iDim], FloatP(kFarValueFloat));
      drjit::store(nValues[iDim], drjit::zeros<FloatP>());
    }
    for (SizeType iDim = 0; iDim < kColorDimensions; ++iDim) {
      drjit::store(cValues[iDim], drjit::zeros<FloatP>());
    }
    std::fill(indices.begin(), indices.end(), std::numeric_limits<int>::max());

    const SizeType curBlockEnd =
        std::min(curBlockStart + gsl::narrow<SizeType>(kSimdPacketSize), end);
    for (SizeType curPoint = curBlockStart; curPoint != curBlockEnd; ++curPoint) {
      const SizeType offset = curPoint - curBlockStart;
      const SizeType pointIndex = points[curPoint].first;

      for (SizeType iDim = 0; iDim < nDim; ++iDim) {
        pValues[iDim][offset] = points[curPoint].second(iDim);
      }
      if (!normals.empty()) {
        for (SizeType iDim = 0; iDim < nDim; ++iDim) {
          nValues[iDim][offset] = normals[pointIndex](iDim);
        }
      }
      if (!colors.empty()) {
        for (SizeType iDim = 0; iDim < kColorDimensions; ++iDim) {
          cValues[iDim][offset] = colors[pointIndex](iDim);
        }
      }
      indices[offset] = pointIndex;
    }

    // PointBlock
    typename Implementation::PointBlock pointBlock;
    for (SizeType iDim = 0; iDim < nDim; ++iDim) {
      pointBlock.values[iDim] = drjit::load<FloatP>(pValues[iDim]);
    }
    pointBlock.indices = drjit::load<IntP>(indices.data());
    impl_->pointBlocks.push_back(pointBlock);

    // NormalBlock
    if (!normals.empty()) {
      typename Implementation::NormalBlock normalBlock;
      for (SizeType iDim = 0; iDim < nDim; ++iDim) {
        normalBlock.values[iDim] = drjit::load<FloatP>(nValues[iDim]);
      }
      impl_->normalBlocks.push_back(normalBlock);
    }

    // ColorBlock
    if (!colors.empty()) {
      typename Implementation::ColorBlock colorBlock;
      for (SizeType iDim = 0; iDim < kColorDimensions; ++iDim) {
        colorBlock.values[iDim] = drjit::load<FloatP>(cValues[iDim]);
      }
      impl_->colorBlocks.push_back(colorBlock);
    }
  }

  const SizeType blocksEnd = gsl::narrow<SizeType>(impl_->pointBlocks.size());

  const auto nodeId = gsl::narrow<SizeType>(nodes_.size());
  nodes_.emplace_back(blocksStart, blocksEnd, box);

  return nodeId;
}

// Ideally we would replace this with a c++11 forwarding constructor.
template <int32_t nDim>
void SimdKdTreef<nDim>::init(
    std::span<const Vec> points_in,
    std::span<const Vec> normals_in,
    std::span<const Col> colors_in) {
  XR_CHECK(points_in.size() < std::numeric_limits<SizeType>::max());
  XR_CHECK(normals_in.empty() || normals_in.size() == points_in.size());
  XR_CHECK(colors_in.empty() || colors_in.size() == points_in.size());

  numPoints_ = gsl::narrow<SizeType>(points_in.size());
  root_ = std::numeric_limits<SizeType>::max();
  depth_ = 0;

  hasNormals_ = (points_in.size() == normals_in.size());

  if (numPoints_ == 0) {
    return;
  }

  // Create a copy of points to manipulate its order for performance
  std::vector<std::pair<SizeType, Vec>> points;
  points.reserve(numPoints_);
  for (SizeType i = 0; i < numPoints_; ++i) {
    points.push_back(std::make_pair(i, points_in[i]));
    bbox_.extend(points_in[i]);
  }

  impl_->pointBlocks.reserve(numPoints_ / kSimdPacketSize + !!(numPoints_ % kSimdPacketSize));

  if (!normals_in.empty()) {
    impl_->normalBlocks.reserve(impl_->pointBlocks.capacity());
  }
  if (!colors_in.empty()) {
    impl_->colorBlocks.reserve(impl_->pointBlocks.capacity());
  }

  root_ = split(points, normals_in, colors_in, 0, numPoints_, 0);
}

template <int32_t nDim>
size_t SimdKdTreef<nDim>::getSimdPacketSize() const {
  return kSimdPacketSize;
}

template <int32_t nDim>
void SimdKdTreef<nDim>::validateInternal(
    SizeType nodeId,
    const Box& box,
    std::vector<char>& touched) const {
  if (touched.empty()) {
    return;
  }

  const Node& node = nodes_[nodeId];
  if (node.isLeaf()) {
    XR_CHECK(node.pointBlocksEnd > node.pointBlocksStart, "Invalid point block indices.");

    for (SizeType i = node.pointBlocksStart; i != node.pointBlocksEnd; ++i) {
      alignas(kSimdAlignment) float pValues[nDim][kSimdPacketSize];
      for (SizeType iDim = 0; iDim < nDim; ++iDim) {
        drjit::store(pValues[iDim], impl_->pointBlocks[i].values[iDim]);
      }

      alignas(kSimdAlignment) SizeType indices[kSimdPacketSize];
      drjit::store(indices, impl_->pointBlocks[i].indices);

      for (SizeType jPoint = 0; jPoint < gsl::narrow<SizeType>(kSimdPacketSize); ++jPoint) {
        if (indices[jPoint] == INT_MAX) {
          for (SizeType iDim = 0; iDim < nDim; ++iDim) {
            XR_CHECK(pValues[iDim][jPoint] == kFarValueFloat);
          }
        } else {
          XR_CHECK(indices[jPoint] >= 0 && indices[jPoint] < numPoints_);
          touched[indices[jPoint]]++;
          Vec point;
          for (SizeType iDim = 0; iDim < nDim; ++iDim) {
            point(iDim) = pValues[iDim][jPoint];
          }
          XR_CHECK(box.contains(point));
        }
      }
    }
  } else {
    XR_CHECK(node.rightChild != nodeId);

    Box childBox = box;
    childBox.max()[node.splitDim] = node.splitVal;
    validateInternal(node.leftChild, childBox, touched);

    childBox = box;
    childBox.min()[node.splitDim] = node.splitVal;
    validateInternal(node.rightChild, childBox, touched);
  }
}

template <int32_t nDim>
SimdKdTreef<nDim>::Node::Node(
    Scalar splitVal_in,
    unsigned char splitDim_in,
    SizeType leftChild_in,
    SizeType rightChild_in,
    const Box& box_in)
    : splitVal(splitVal_in),
      splitDim(splitDim_in),
      leftChild(leftChild_in),
      rightChild(rightChild_in),
      box(box_in) {
  XR_CHECK(splitDim_in < nDim);
}

template <int32_t nDim>
SimdKdTreef<nDim>::Node::Node(SizeType leftChild_in, SizeType rightChild_in, const Box& box_in)
    : splitVal(std::numeric_limits<Scalar>::max()),
      splitDim(std::numeric_limits<unsigned char>::max()),
      pointBlocksStart(leftChild_in),
      pointBlocksEnd(rightChild_in),
      box(box_in) {
  XR_CHECK(0 <= pointBlocksStart);
  XR_CHECK(pointBlocksStart < pointBlocksEnd);
}

template <int32_t nDim>
bool SimdKdTreef<nDim>::Node::isLeaf() const {
  return (splitDim == std::numeric_limits<unsigned char>::max());
}

template class SimdKdTreef<3>;
template class SimdKdTreef<2>;

#ifdef AXEL_ENABLE_AVX

template <int32_t nDim>
SimdKdTreeAvxf<nDim>::SimdKdTreeAvxf(
    std::span<const Vec> points_in,
    std::span<const Vec> normals_in,
    std::span<const Col> colors_in) {
  init(points_in, normals_in, colors_in);
}

template <int32_t nDim>
SimdKdTreeAvxf<nDim>::~SimdKdTreeAvxf() {
  // Do nothing
}

template <int32_t nDim>
typename SimdKdTreeAvxf<nDim>::SizeType SimdKdTreeAvxf<nDim>::createLeafNode(
    std::vector<std::pair<SizeType, Vec>>& points,
    std::span<const Vec> normals,
    std::span<const Col> colors,
    SizeType start,
    SizeType end,
    const Box& box) {
  const SizeType node_id = (SizeType)this->nodes_.size();

  const SizeType blocksStart = (SizeType)pointBlocks_.size();
  for (SizeType curBlockStart = start; curBlockStart < end; curBlockStart += AVX_FLOAT_BLOCK_SIZE) {
    alignas(AVX_ALIGNMENT) float pValues[nDim][AVX_FLOAT_BLOCK_SIZE];
    alignas(AVX_ALIGNMENT) float nValues[nDim][AVX_FLOAT_BLOCK_SIZE];
    alignas(AVX_ALIGNMENT) float cValues[kColorDimensions][AVX_FLOAT_BLOCK_SIZE];

    for (SizeType iDim = 0; iDim < nDim; ++iDim) {
      _mm256_store_ps(pValues[iDim], _mm256_set1_ps(kFarValueFloat));
      _mm256_store_ps(nValues[iDim], _mm256_setzero_ps());
    }
    for (SizeType iDim = 0; iDim < kColorDimensions; ++iDim) {
      _mm256_store_ps(cValues[iDim], _mm256_setzero_ps());
    }

    alignas(AVX_ALIGNMENT) SizeType indices[AVX_FLOAT_BLOCK_SIZE] = {
        INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX};

    const SizeType curBlockEnd = std::min(curBlockStart + AVX_FLOAT_BLOCK_SIZE, end);
    for (SizeType curPoint = curBlockStart; curPoint != curBlockEnd; ++curPoint) {
      const SizeType offset = curPoint - curBlockStart;
      const SizeType pointIndex = points[curPoint].first;
      indices[offset] = pointIndex;

      for (SizeType iDim = 0; iDim < nDim; ++iDim) {
        pValues[iDim][offset] = points[curPoint].second[iDim];
      }

      if (!normals.empty()) {
        for (SizeType iDim = 0; iDim < nDim; ++iDim) {
          nValues[iDim][offset] = normals[pointIndex][iDim];
        }
      }

      if (!colors.empty()) {
        for (SizeType iDim = 0; iDim < kColorDimensions; ++iDim) {
          cValues[iDim][offset] = colors[pointIndex][iDim];
        }
      }
    }

    PointBlock pointBlock;
    for (SizeType iDim = 0; iDim < nDim; ++iDim) {
      pointBlock.values[iDim] = _mm256_load_ps(pValues[iDim]);
    }

    pointBlock.indices = _mm256_load_si256((const __m256i*)indices);
    pointBlocks_.push_back(pointBlock);

    if (!normals.empty()) {
      NormalBlock normalBlock;
      for (SizeType iDim = 0; iDim < nDim; ++iDim) {
        normalBlock.values[iDim] = _mm256_load_ps(nValues[iDim]);
      }
      normalBlocks_.push_back(normalBlock);
    }

    if (!colors.empty()) {
      ColorBlock colorBlock;
      for (SizeType iDim = 0; iDim < kColorDimensions; ++iDim) {
        colorBlock.values[iDim] = _mm256_load_ps(cValues[iDim]);
      }
      colorBlocks_.push_back(colorBlock);
    }
  }

  const SizeType blocksEnd = (SizeType)pointBlocks_.size();

  this->nodes_.push_back(Node(blocksStart, blocksEnd, box));
  return node_id;
}

template <int32_t nDim>
std::tuple<bool, typename SimdKdTreeAvxf<nDim>::SizeType, typename SimdKdTreeAvxf<nDim>::Scalar>
SimdKdTreeAvxf<nDim>::closestPoint(const Vec& queryPoint, Scalar maxSqrDist) const {
  bool foundPoint = false;
  SizeType bestPoint = std::numeric_limits<SizeType>::max();
  Scalar bestSqrDist = maxSqrDist;

  if (this->empty()) {
    return std::make_tuple(foundPoint, bestPoint, bestSqrDist);
  }

  // Use an explicit stack for speed:
  std::array<SizeType, kMaxDepth + 1> nodeStack;

  // Start with just the root on the stack:
  SizeType stackSize = 1;
  nodeStack[0] = this->root_;

  __m256 query_p[nDim];
  for (SizeType iDim = 0; iDim < nDim; ++iDim) {
    query_p[iDim] = _mm256_broadcast_ss(&queryPoint[iDim]);
  }

  while (stackSize != 0) {
    // Pop the top of the stack:
    const SizeType cur = nodeStack[--stackSize];
    const auto& curNode = this->nodes_[cur];

    // check again if the current best distance is smaller than the distance to the current node
    // split. if so we can skip the node
    if (curNode.box.squaredExteriorDistance(queryPoint) > bestSqrDist) {
      continue;
    }

    if (curNode.isLeaf()) {
      const SizeType pointBlocksStart = curNode.pointBlocksStart;
      const SizeType pointBlocksEnd = curNode.pointBlocksEnd;
      XR_CHECK(pointBlocksEnd > pointBlocksStart, "Invalid point block indices.");

      __m256 bestSqrDistBlock = _mm256_broadcast_ss(&bestSqrDist);
      __m256i bestSqrDistIndicesBlock = _mm256_set1_epi32(INT_MAX);

      for (SizeType iBlock = pointBlocksStart; iBlock != pointBlocksEnd; ++iBlock) {
        const PointBlock& block = pointBlocks_[iBlock];
        __m256 sqrDist = _mm256_setzero_ps();
        for (SizeType iDim = 0; iDim < nDim; ++iDim) {
          const __m256 dimDiff = _mm256_sub_ps(block.values[iDim], query_p[iDim]);
          sqrDist = _mm256_add_ps(sqrDist, _mm256_mul_ps(dimDiff, dimDiff));
        }

        const __m256i lessThanMask =
            _mm256_castps_si256(_mm256_cmp_ps(sqrDist, bestSqrDistBlock, _CMP_LT_OS));
        bestSqrDistBlock = _mm256_min_ps(sqrDist, bestSqrDistBlock);
        bestSqrDistIndicesBlock = _mm256_or_si256(
            _mm256_and_si256(lessThanMask, block.indices),
            _mm256_andnot_si256(lessThanMask, bestSqrDistIndicesBlock));
      }

      alignas(AVX_ALIGNMENT) float bestSqrDist_extract[AVX_FLOAT_BLOCK_SIZE];
      _mm256_store_ps(bestSqrDist_extract, bestSqrDistBlock);

      alignas(AVX_ALIGNMENT) SizeType bestSqrDistIndices_extract[AVX_FLOAT_BLOCK_SIZE];
      _mm256_store_si256((__m256i*)bestSqrDistIndices_extract, bestSqrDistIndicesBlock);

      for (SizeType k = 0; k < AVX_FLOAT_BLOCK_SIZE; ++k) {
        if (bestSqrDist_extract[k] < bestSqrDist) {
          XR_CHECK(bestSqrDistIndices_extract[k] != INT_MAX);
          foundPoint = true;
          bestPoint = bestSqrDistIndices_extract[k];
          bestSqrDist = bestSqrDist_extract[k];
        }
      }
    } else {
      // We need to descend on
      const unsigned char splitDim = curNode.splitDim;
      const Scalar splitVal = curNode.splitVal;
      const Scalar queryPointVal = queryPoint[splitDim];
      const Scalar distToSplit = (queryPointVal - splitVal);
      const Scalar sqrDistToSplit = distToSplit * distToSplit;

      if (queryPointVal >= splitVal) {
        // We only need to descend if the distance to the split is less
        // than the distance to our best point so far.
        if (sqrDistToSplit < bestSqrDist) {
          nodeStack[stackSize++] = curNode.leftChild;
        }

        // Since the query point is in the right half of the space
        // we think it more likely that we'll find the closest point
        // in that half of the space, so we should descend on it first.
        // Since we grab the top entry of the stack first, this implies
        // pushing it last.
        nodeStack[stackSize++] = curNode.rightChild;
      } else {
        // Same as above:
        if (sqrDistToSplit < bestSqrDist) {
          nodeStack[stackSize++] = curNode.rightChild;
        }

        nodeStack[stackSize++] = curNode.leftChild;
      }
    }
  }

  XR_CHECK(!foundPoint || bestPoint < this->numPoints_);
  return std::make_tuple(foundPoint, bestPoint, bestSqrDist);
}

template <int32_t nDim>
std::tuple<bool, typename SimdKdTreeAvxf<nDim>::SizeType, typename SimdKdTreeAvxf<nDim>::Scalar>
SimdKdTreeAvxf<nDim>::closestPoint(
    const Vec& queryPoint,
    const Vec& queryNormal,
    Scalar maxSqrDist,
    Scalar maxNormalDot) const {
  bool foundPoint = false;
  SizeType bestPoint = std::numeric_limits<SizeType>::max();
  Scalar bestSqrDist = maxSqrDist;

  XR_CHECK(this->hasNormals_);

  if (this->empty()) {
    return std::make_tuple(foundPoint, bestPoint, bestSqrDist);
  }

  // Use an explicit stack for speed:
  std::array<SizeType, kMaxDepth + 1> nodeStack;

  // Start with just the root on the stack:
  SizeType stackSize = 1;
  nodeStack[0] = this->root_;

  __m256 query_p[nDim];
  __m256 query_normal[nDim];
  for (SizeType iDim = 0; iDim < nDim; ++iDim) {
    query_p[iDim] = _mm256_broadcast_ss(&queryPoint[iDim]);
    query_normal[iDim] = _mm256_broadcast_ss(&queryNormal[iDim]);
  }

  while (stackSize != 0) {
    // Pop the top of the stack:
    const SizeType cur = nodeStack[--stackSize];
    const auto& curNode = this->nodes_[cur];

    // check again if the current best distance is smaller than the distance to the current node
    // split. if so we can skip the node
    if (curNode.box.squaredExteriorDistance(queryPoint) > bestSqrDist) {
      continue;
    }

    if (curNode.isLeaf()) {
      const SizeType pointBlocksStart = curNode.pointBlocksStart;
      const SizeType pointBlocksEnd = curNode.pointBlocksEnd;
      XR_CHECK(pointBlocksEnd > pointBlocksStart, "Invalid point block indices.");

      __m256 bestSqrDistBlock = _mm256_broadcast_ss(&bestSqrDist);
      __m256i bestSqrDistIndicesBlock = _mm256_set1_epi32(INT_MAX);

      for (SizeType iBlock = pointBlocksStart; iBlock != pointBlocksEnd; ++iBlock) {
        const PointBlock& pointBlock = pointBlocks_[iBlock];
        __m256 sqrDist = _mm256_setzero_ps();
        for (SizeType iDim = 0; iDim < nDim; ++iDim) {
          const __m256 dimDiff = _mm256_sub_ps(pointBlock.values[iDim], query_p[iDim]);
          sqrDist = _mm256_add_ps(sqrDist, _mm256_mul_ps(dimDiff, dimDiff));
        }

        const __m256 lessThanMask = _mm256_cmp_ps(sqrDist, bestSqrDistBlock, _CMP_LT_OS);

        if (_mm256_movemask_ps(lessThanMask) == 0) {
          continue;
        }

        // Take the dot product between the query normal and the point normals:
        const NormalBlock& normalBlock = normalBlocks_[iBlock];
        __m256 normalDotProd = _mm256_setzero_ps();
        for (SizeType iDim = 0; iDim < nDim; ++iDim) {
          const __m256 dimMul = _mm256_mul_ps(query_normal[iDim], normalBlock.values[iDim]);
          normalDotProd = _mm256_add_ps(normalDotProd, dimMul);
        }

        // Only take points where the distance is smaller and the normal agrees.
        const __m256 finalMask = _mm256_and_ps(
            lessThanMask,
            _mm256_cmp_ps(normalDotProd, _mm256_broadcast_ss(&maxNormalDot), _CMP_GE_OS));
        bestSqrDistBlock = _mm256_or_ps(
            _mm256_and_ps(finalMask, sqrDist), _mm256_andnot_ps(finalMask, bestSqrDistBlock));

        const __m256i finalMaskInt = _mm256_castps_si256(finalMask);
        bestSqrDistIndicesBlock = _mm256_or_si256(
            _mm256_and_si256(finalMaskInt, pointBlock.indices),
            _mm256_andnot_si256(finalMaskInt, bestSqrDistIndicesBlock));
      }

      alignas(AVX_ALIGNMENT) float bestSqrDist_extract[AVX_FLOAT_BLOCK_SIZE];
      _mm256_store_ps(bestSqrDist_extract, bestSqrDistBlock);
      alignas(AVX_ALIGNMENT) SizeType bestSqrDistIndices_extract[AVX_FLOAT_BLOCK_SIZE];
      _mm256_store_si256((__m256i*)bestSqrDistIndices_extract, bestSqrDistIndicesBlock);

      for (SizeType k = 0; k < AVX_FLOAT_BLOCK_SIZE; ++k) {
        if (bestSqrDistIndices_extract[k] == INT_MAX) {
          continue;
        }

        if (bestSqrDist_extract[k] < bestSqrDist) {
          foundPoint = true;
          bestPoint = bestSqrDistIndices_extract[k];
          bestSqrDist = bestSqrDist_extract[k];
        }
      }
    } else {
      // We need to descend on
      const unsigned char splitDim = curNode.splitDim;
      const Scalar splitVal = curNode.splitVal;
      const Scalar queryPointVal = queryPoint[splitDim];
      const Scalar distToSplit = (queryPointVal - splitVal);
      const Scalar sqrDistToSplit = distToSplit * distToSplit;

      if (queryPointVal >= splitVal) {
        // We only need to descend if the distance to the split is less than the distance to our
        // best point so far.
        if (sqrDistToSplit < bestSqrDist) {
          nodeStack[stackSize++] = curNode.leftChild;
        }

        // Since the query point is in the right half of the space we think it more likely that
        // we'll find the closest point in that half of the space, so we should descend on it first.
        // Since we grab the top entry of the stack first, this implies pushing it last.
        nodeStack[stackSize++] = curNode.rightChild;
      } else {
        // Same as above:
        if (sqrDistToSplit < bestSqrDist) {
          nodeStack[stackSize++] = curNode.rightChild;
        }

        nodeStack[stackSize++] = curNode.leftChild;
      }
    }
  }

  XR_CHECK(!foundPoint || bestPoint < this->numPoints_);
  return std::make_tuple(foundPoint, bestPoint, bestSqrDist);
}

template <int32_t nDim>
std::tuple<bool, typename SimdKdTreeAvxf<nDim>::SizeType, typename SimdKdTreeAvxf<nDim>::Scalar>
SimdKdTreeAvxf<nDim>::closestPoint(
    const Vec& queryPoint,
    const Vec& queryNormal,
    const Col& queryColor,
    Scalar maxSqrDist,
    Scalar maxNormalDot,
    Scalar maxColorSqrDist) const {
  Scalar bestSqrDist = maxSqrDist;
  SizeType bestPoint = std::numeric_limits<SizeType>::max();
  bool foundPoint = false;

  XR_CHECK(this->hasNormals_);

  if (this->empty()) {
    return std::make_tuple(foundPoint, bestPoint, bestSqrDist);
  }

  // Use an explicit stack for speed:
  std::array<SizeType, kMaxDepth + 1> nodeStack;

  // Start with just the root on the stack:
  SizeType stackSize = 1;
  nodeStack[0] = this->root_;

  __m256 query_p[nDim];
  __m256 query_normal[nDim];
  __m256 query_color[kColorDimensions];
  for (SizeType iDim = 0; iDim < nDim; ++iDim) {
    query_p[iDim] = _mm256_broadcast_ss(&queryPoint[iDim]);
    query_normal[iDim] = _mm256_broadcast_ss(&queryNormal[iDim]);
  }
  for (SizeType iDim = 0; iDim < kColorDimensions; ++iDim) {
    query_color[iDim] = _mm256_broadcast_ss(&queryColor[iDim]);
  }

  while (stackSize != 0) {
    // Pop the top of the stack:
    const SizeType cur = nodeStack[--stackSize];
    const auto& curNode = this->nodes_[cur];

    // check again if the current best distance is smaller than the distance to the current node
    // split. if so we can skip the node
    if (curNode.box.squaredExteriorDistance(queryPoint) > bestSqrDist) {
      continue;
    }

    if (curNode.isLeaf()) {
      const SizeType pointBlocksStart = curNode.pointBlocksStart;
      const SizeType pointBlocksEnd = curNode.pointBlocksEnd;
      XR_CHECK(pointBlocksEnd > pointBlocksStart, "Invalid point block indices.");

      __m256 bestSqrDistBlock = _mm256_broadcast_ss(&bestSqrDist);
      __m256i bestSqrDistIndicesBlock = _mm256_set1_epi32(INT_MAX);

      for (SizeType iBlock = pointBlocksStart; iBlock != pointBlocksEnd; ++iBlock) {
        const PointBlock& pointBlock = pointBlocks_[iBlock];
        __m256 sqrDist = _mm256_setzero_ps();
        for (SizeType iDim = 0; iDim < nDim; ++iDim) {
          const __m256 dimDiff = _mm256_sub_ps(pointBlock.values[iDim], query_p[iDim]);
          sqrDist = _mm256_add_ps(sqrDist, _mm256_mul_ps(dimDiff, dimDiff));
        }

        const __m256 lessThanMask = _mm256_cmp_ps(sqrDist, bestSqrDistBlock, _CMP_LT_OS);

        if (_mm256_movemask_ps(lessThanMask) == 0) {
          continue;
        }

        // Take the dot product between the query normal and the point normals:
        const NormalBlock& normalBlock = normalBlocks_[iBlock];
        __m256 normalDotProd = _mm256_setzero_ps();
        for (SizeType iDim = 0; iDim < nDim; ++iDim) {
          const __m256 dimMul = _mm256_mul_ps(query_normal[iDim], normalBlock.values[iDim]);
          normalDotProd = _mm256_add_ps(normalDotProd, dimMul);
        }

        // Take the squared difference between the query color and the point colors:
        const ColorBlock& colorBlock = colorBlocks_[iBlock];
        __m256 colorSqrDist = _mm256_setzero_ps();
        for (SizeType iDim = 0; iDim < kColorDimensions - 1; ++iDim) {
          const __m256 dimMul = _mm256_sub_ps(query_color[iDim], colorBlock.values[iDim]);
          colorSqrDist = _mm256_add_ps(colorSqrDist, _mm256_mul_ps(dimMul, dimMul));
        }
        // multiply with certainty value (last color value)
        colorSqrDist = _mm256_mul_ps(colorSqrDist, colorBlock.values[kColorDimensions - 1]);
        colorSqrDist = _mm256_mul_ps(colorSqrDist, query_color[kColorDimensions - 1]);

        // Only take points where the distance is smaller and the normal agrees.
        const __m256 normalMask = _mm256_and_ps(
            lessThanMask,
            _mm256_cmp_ps(normalDotProd, _mm256_broadcast_ss(&maxNormalDot), _CMP_GE_OS));
        const __m256 finalMask = _mm256_and_ps(
            normalMask,
            _mm256_cmp_ps(colorSqrDist, _mm256_broadcast_ss(&maxColorSqrDist), _CMP_LE_OS));
        bestSqrDistBlock = _mm256_or_ps(
            _mm256_and_ps(finalMask, sqrDist), _mm256_andnot_ps(finalMask, bestSqrDistBlock));

        const __m256i finalMaskInt = _mm256_castps_si256(finalMask);
        bestSqrDistIndicesBlock = _mm256_or_si256(
            _mm256_and_si256(finalMaskInt, pointBlock.indices),
            _mm256_andnot_si256(finalMaskInt, bestSqrDistIndicesBlock));
      }

      alignas(AVX_ALIGNMENT) float bestSqrDist_extract[AVX_FLOAT_BLOCK_SIZE];
      _mm256_store_ps(bestSqrDist_extract, bestSqrDistBlock);
      alignas(AVX_ALIGNMENT) SizeType bestSqrDistIndices_extract[AVX_FLOAT_BLOCK_SIZE];
      _mm256_store_si256((__m256i*)bestSqrDistIndices_extract, bestSqrDistIndicesBlock);

      for (SizeType k = 0; k < AVX_FLOAT_BLOCK_SIZE; ++k) {
        if (bestSqrDistIndices_extract[k] == INT_MAX) {
          continue;
        }

        if (bestSqrDist_extract[k] < bestSqrDist) {
          foundPoint = true;
          bestPoint = bestSqrDistIndices_extract[k];
          bestSqrDist = bestSqrDist_extract[k];
        }
      }
    } else {
      // We need to descend on
      const unsigned char splitDim = curNode.splitDim;
      const Scalar splitVal = curNode.splitVal;
      const Scalar queryPointVal = queryPoint[splitDim];
      const Scalar distToSplit = (queryPointVal - splitVal);
      const Scalar sqrDistToSplit = distToSplit * distToSplit;

      if (queryPointVal >= splitVal) {
        // We only need to descend if the distance to the split is less than the distance to our
        // best point so far.
        if (sqrDistToSplit < bestSqrDist) {
          nodeStack[stackSize++] = curNode.leftChild;
        }

        // Since the query point is in the right half of the space we think it more likely that
        // we'll find the closest point in that half of the space, so we should descend on it first.
        // Since we grab the top entry of the stack first, this implies pushing it last.
        nodeStack[stackSize++] = curNode.rightChild;
      } else {
        // Same as above:
        if (sqrDistToSplit < bestSqrDist) {
          nodeStack[stackSize++] = curNode.rightChild;
        }

        nodeStack[stackSize++] = curNode.leftChild;
      }
    }
  }

  XR_CHECK(!foundPoint || bestPoint < this->numPoints_);
  return std::make_tuple(foundPoint, bestPoint, bestSqrDist);
}

template <int32_t nDim>
void SimdKdTreeAvxf<nDim>::pointsInNSphere(
    const Vec& center_in,
    Scalar radius,
    std::vector<SizeType>& points) const {
  XR_CHECK(radius >= Scalar(0));

  points.clear();
  if (this->empty()) {
    return;
  }

  const Scalar radiusSqr = radius * radius;
  alignas(AVX_ALIGNMENT) const __m256 radiusSqrBlock = _mm256_broadcast_ss(&radiusSqr);

  // Use an explicit stack for speed:
  std::array<SizeType, kMaxDepth + 1> nodeStack;

  // Start with just the root on the stack:
  SizeType stackSize = 1;
  nodeStack[0] = this->root_;

  __m256 query_p[nDim];
  for (SizeType iDim = 0; iDim < nDim; ++iDim) {
    query_p[iDim] = _mm256_broadcast_ss(&center_in[iDim]);
  }

  while (stackSize != 0) {
    // Pop the top of the stack:
    const SizeType cur = nodeStack[--stackSize];
    const auto& curNode = this->nodes_[cur];

    // check if the distance is smaller than the distance to the current node split. if so we can
    // skip the node. this may help as the bbox is tighter than the split
    if (curNode.box.squaredExteriorDistance(center_in) > radiusSqr) {
      continue;
    }

    if (curNode.isLeaf()) {
      const SizeType pointBlocksStart = curNode.pointBlocksStart;
      const SizeType pointBlocksEnd = curNode.pointBlocksEnd;
      XR_CHECK(pointBlocksEnd > pointBlocksStart, "Invalid point block indices.");

      for (SizeType iBlock = pointBlocksStart; iBlock != pointBlocksEnd; ++iBlock) {
        const PointBlock& block = pointBlocks_[iBlock];
        __m256 sqrDist = _mm256_setzero_ps();
        for (SizeType iDim = 0; iDim < nDim; ++iDim) {
          alignas(AVX_ALIGNMENT) const __m256 dimDiff =
              _mm256_sub_ps(block.values[iDim], query_p[iDim]);
          sqrDist = _mm256_add_ps(sqrDist, _mm256_mul_ps(dimDiff, dimDiff));
        }

        if (_mm256_movemask_ps(_mm256_cmp_ps(sqrDist, radiusSqrBlock, _CMP_LE_OS)) == 0) {
          continue;
        }

        alignas(AVX_ALIGNMENT) SizeType indices_extract[AVX_FLOAT_BLOCK_SIZE];
        _mm256_store_si256((__m256i*)indices_extract, block.indices);

        alignas(AVX_ALIGNMENT) float sqrDist_extract[AVX_FLOAT_BLOCK_SIZE];
        _mm256_store_ps(sqrDist_extract, sqrDist);

        for (SizeType k = 0; k < AVX_FLOAT_BLOCK_SIZE; ++k) {
          if (sqrDist_extract[k] <= radiusSqr) {
            XR_CHECK(indices_extract[k] != INT_MAX);
            points.push_back(indices_extract[k]);
          }
        }
      }
    } else {
      // We potentially need to descend on both children.  Unlike
      // the closest point case, there's no reason to favor one
      // side over the other:
      const unsigned char splitDim = curNode.splitDim;
      const Scalar splitVal = curNode.splitVal;
      const Scalar centerVal = center_in[splitDim];

      if (centerVal - radius <= splitVal) {
        nodeStack[stackSize++] = curNode.leftChild;
      }

      if (centerVal + radius >= splitVal) {
        nodeStack[stackSize++] = curNode.rightChild;
      }
    }
  }
}

template <int32_t nDim>
size_t SimdKdTreeAvxf<nDim>::getSimdPacketSize() const {
  return AVX_FLOAT_BLOCK_SIZE;
}

template <int32_t nDim>
void SimdKdTreeAvxf<nDim>::validateInternal(
    SizeType nodeId,
    const Box& box,
    std::vector<char>& touched) const {
  if (touched.empty()) {
    return;
  }

  const Node& node = this->nodes_[nodeId];
  if (node.isLeaf()) {
    XR_CHECK(node.pointBlocksEnd > node.pointBlocksStart);

    for (SizeType i = node.pointBlocksStart; i != node.pointBlocksEnd; ++i) {
      alignas(AVX_ALIGNMENT) float pValues[nDim][AVX_FLOAT_BLOCK_SIZE];
      for (SizeType iDim = 0; iDim < nDim; ++iDim) {
        _mm256_store_ps(pValues[iDim], pointBlocks_[i].values[iDim]);
      }

      alignas(AVX_ALIGNMENT) SizeType indices[AVX_FLOAT_BLOCK_SIZE];
      _mm256_store_si256((__m256i*)indices, pointBlocks_[i].indices);

      for (SizeType jPoint = 0; jPoint < AVX_FLOAT_BLOCK_SIZE; ++jPoint) {
        if (indices[jPoint] == INT_MAX) {
          for (SizeType iDim = 0; iDim < nDim; ++iDim) {
            XR_CHECK(pValues[iDim][jPoint] == kFarValueFloat);
          }
        } else {
          XR_CHECK(indices[jPoint] >= 0 && indices[jPoint] < this->numPoints_);
          touched[indices[jPoint]]++;
          Vec point;
          for (SizeType iDim = 0; iDim < nDim; ++iDim) {
            point(iDim) = pValues[iDim][jPoint];
          }
          XR_CHECK(box.contains(point));
        }
      }
    }
  } else {
    XR_CHECK(node.rightChild != nodeId);

    Box childBox = box;
    childBox.max()[node.splitDim] = node.splitVal;
    validateInternal(node.leftChild, childBox, touched);

    childBox = box;
    childBox.min()[node.splitDim] = node.splitVal;
    validateInternal(node.rightChild, childBox, touched);
  }
}

// Ideally we would replace this with a c++11 forwarding constructor.
template <int32_t nDim>
void SimdKdTreeAvxf<nDim>::init(
    std::span<const Vec> points_in,
    std::span<const Vec> normals_in,
    std::span<const Col> colors_in) {
  XR_CHECK(points_in.size() < std::numeric_limits<SizeType>::max());
  XR_CHECK(normals_in.empty() || normals_in.size() == points_in.size());
  XR_CHECK(colors_in.empty() || colors_in.size() == points_in.size());

  this->numPoints_ = gsl::narrow<SizeType>(points_in.size());
  this->root_ = std::numeric_limits<SizeType>::max();
  this->depth_ = 0;

  this->hasNormals_ = (points_in.size() == normals_in.size());

  if (this->numPoints_ == 0) {
    return;
  }

  // Create a copy of points to manipulate its order for performance
  std::vector<std::pair<SizeType, Vec>> points;
  points.reserve(this->numPoints_);
  for (SizeType i = 0; i < this->numPoints_; ++i) {
    points.push_back(std::make_pair(i, points_in[i]));
    this->bbox_.extend(points_in[i]);
  }

  this->pointBlocks_.reserve(
      this->numPoints_ / AVX_FLOAT_BLOCK_SIZE + !!(this->numPoints_ % AVX_FLOAT_BLOCK_SIZE));

  if (!normals_in.empty()) {
    normalBlocks_.reserve(this->pointBlocks_.capacity());
  }
  if (!colors_in.empty()) {
    colorBlocks_.reserve(this->pointBlocks_.capacity());
  }

  this->root_ = this->split(points, normals_in, colors_in, 0, this->numPoints_, 0);
}

template class SimdKdTreeAvxf<3>;
template class SimdKdTreeAvxf<2>;

#endif

} // namespace axel
