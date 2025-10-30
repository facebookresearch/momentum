/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <span>

#ifdef AXEL_ENABLE_AVX
#include <immintrin.h>
#include <xmmintrin.h>
#endif

#include <axel/Checks.h>

// TODO: Keeping this temporarily
#define AXEL_DEFINE_POINTERS(x)                   \
  using x##_p = ::std::shared_ptr<x>;             \
  using x##_u = ::std::unique_ptr<x>;             \
  using x##_w = ::std::weak_ptr<x>;               \
  using x##_const_p = ::std::shared_ptr<const x>; \
  using x##_const_u = ::std::unique_ptr<const x>; \
  using x##_const_w = ::std::weak_ptr<const x>;

namespace axel {

/// A k-d tree is a spatial data structure that supports fast nearest-points lookup.
///
/// For details, see: http://en.wikipedia.org/wiki/k-d_tree
///
/// This particular k-d tree is a bit unusual, in that it only stores points at the leaf nodes. This
/// allows some amount of speedup because we can then blast through the points 10 at a time or so,
/// exploiting cache locality.
template <int32_t nDim>
class SimdKdTreef {
 public:
  static constexpr int32_t kDim = nDim;
  static constexpr int32_t kColorDimensions = 4;

  /// We will use this value for storing the "too far away" distance instead of FLT_MAX because we
  /// can square it and still remain well within the range of acceptable values.
  static constexpr float kFarValueFloat = 1e+10f;

  using SizeType = int32_t; // We use signed integer indices for easier compatibility with SSE
  using Scalar = float;
  using Vec = Eigen::Matrix<Scalar, kDim, 1>;
  using Col = Eigen::Matrix<Scalar, kColorDimensions, 1>;
  using Box = Eigen::AlignedBox<Scalar, kDim>;

  /// Limiting the total number of nodes to 2^20 or 1,048,576 to prevent infinite recursion.
  static constexpr SizeType kMaxDepth = 20;

  /// Constructs k-d tree from points, normals, and colors.
  explicit SimdKdTreef(
      std::span<const Vec> points = std::span<Vec>{},
      std::span<const Vec> normals = std::span<Vec>{},
      std::span<const Col> colors = std::span<Col>{});

  /// Destructor.
  virtual ~SimdKdTreef();

  /// Disables copy constructor.
  SimdKdTreef(const SimdKdTreef&) = delete;

  /// Disables assignment operator.
  SimdKdTreef& operator=(const SimdKdTreef&) = delete;

  /// Returns (1) whether a point was found, (2) the index of the point, and (3) the squared
  /// distance.
  [[nodiscard]] virtual std::tuple<bool, SizeType, Scalar> closestPoint(
      const Vec& queryPoint,
      Scalar maxSqrDist = std::numeric_limits<Scalar>::max()) const;
  // TODO: Consider making the result as a structure for better readability

  /// This version of closest points also validates that the normals agree; that is, it won't return
  /// a closest point whose normal dotted with the passed-in normal is negative.
  [[nodiscard]] virtual std::tuple<bool, SizeType, Scalar> closestPoint(
      const Vec& queryPoint,
      const Vec& queryNormal,
      Scalar maxSqrDist = std::numeric_limits<Scalar>::max(),
      Scalar maxNormalDot = 0.0f) const;

  /// This version of closest points also validates that the normals and colors agree; that is, it
  /// won't return a closest point whose normal dotted with the passed-in normal is negative.
  [[nodiscard]] virtual std::tuple<bool, SizeType, Scalar> closestPoint(
      const Vec& queryPoint,
      const Vec& queryNormal,
      const Col& color,
      Scalar maxSqrDist = std::numeric_limits<Scalar>::max(),
      Scalar maxNormalDot = 0.0f,
      Scalar maxColorSqrDist = 0.5f) const;

  /// Allows the user to mark out which points are considered 'acceptable.'
  ///
  /// The passed-in function should map from a __m256i (the indices) to an __m256i which uses the
  /// standard AVX convention (all ones in the appropriate spots). It also needs to be tolerant to
  /// possible INT_MAX values since we may not have actual indices for all 4 points.
  ///
  /// Here's an example acceptance function that looks for odd-valued points:
  /// @code
  /// closestPointWithAcceptance(queryPoint, maxSqrDist, [](SizeType index) -> bool { return !(index
  /// % 2); });
  /// @endcode
  [[nodiscard]] std::tuple<bool, SizeType, Scalar> closestPointWithAcceptance(
      const Vec& queryPoint,
      Scalar maxSqrDist,
      const std::function<bool(SizeType pointIndex)>& acceptanceFunction) const;
  // TODO: Consider take a SIMD version of acceptance function. Currently, non-vectorized function
  // is used.

  /// Queries a sphere against the tree.
  virtual void pointsInNSphere(const Vec& center, Scalar radius, std::vector<SizeType>& points)
      const;

  /// Returns whether this tree has no points.
  [[nodiscard]] bool empty() const;

  /// Returns the most outter bounding box.
  [[nodiscard]] const Box& boundingBox() const;

  /// Returns the number of points.
  [[nodiscard]] SizeType size() const;

  /// Returns the depth of the tree.
  [[nodiscard]] SizeType depth() const;

  /// Checks that the KD tree is valid. This could be arbitrarily slow, so it should only be used
  /// for debugging.
  void validate() const;

 protected:
  /// Returns the SIMD packet size (e.g., 8 for AVX)
  [[nodiscard]] virtual size_t getSimdPacketSize() const;

  /// Internal implementation for validate()
  virtual void validateInternal(SizeType node, const Box& box, std::vector<char>& touched) const;

  /// Builds a tree; split the range [start, end) in two, generating a new Node at the split.
  /// Returns the index of that new node.
  SizeType split(
      std::vector<std::pair<SizeType, Vec>>& points,
      std::span<const Vec> normals,
      std::span<const Col> colors,
      SizeType start,
      SizeType end,
      SizeType depth);

  /// Creates a leaf node when the number of points is equal to or greater than 8 SIMD blocks.
  virtual SizeType createLeafNode(
      std::vector<std::pair<SizeType, Vec>>& points,
      std::span<const Vec> normals,
      std::span<const Col> colors,
      SizeType start,
      SizeType end,
      const Box& box);

  /// Each node in the tree splits the space in half
  struct Node {
    /// Constructs as an internal node
    Node(
        Scalar splitVal_in,
        unsigned char splitDim_in,
        SizeType leftChild_in,
        SizeType rightChild_in,
        const Box& box_in);

    /// Constructs as a leaf node
    Node(SizeType leftChild_in, SizeType rightChild_in, const Box& box_in);

    /// Returns whether this is a leaf node.
    [[nodiscard]] bool isLeaf() const;

    /// For internal nodes, the split value. For leaf nodes, std::numeric_limits<Scalar>::max().
    const Scalar splitVal;

    /// For internal nodes, the index to the left child node. For leaf nodes,
    /// std::numeric_limits<unsigned char>::max().
    const unsigned char splitDim;

    union {
      /// For internal nodes, the index to the left child node.
      SizeType leftChild;

      /// For leaf nodes, the block start of the list of points.
      const SizeType pointBlocksStart;
    };

    union {
      /// For internal nodes, the index to the right child node.
      SizeType rightChild;

      /// For leaf nodes, the block end of the list of points.
      const SizeType pointBlocksEnd;
    };

    /// Inner bounding box
    const Box box;
  };

  /// The axis-aligned bounding box of the whole tree.
  Box bbox_;

  /// The number of points in the tree.
  SizeType numPoints_;

  /// The list of nodes in the tree.
  std::vector<Node> nodes_;

  /// The node index to the root node.
  SizeType root_;

  /// The largest depth of the tree.
  SizeType depth_;

  /// Used purely for debugging, to make sure we never use a normal-less k-d tree to check against
  /// normals.
  bool hasNormals_;

 private:
  /// Initializes the k-d tree. This function is intended to be only called by the constructor.
  void init(
      std::span<const Vec> points_in,
      std::span<const Vec> normals_in,
      std::span<const Col> colors_in);

  /// PIMPL idiom to hide the SIMD specific implementation.
  struct Implementation;

  /// PIMPL instance for the SIMD implementation.
  std::unique_ptr<Implementation> impl_;
};

using SimdKdTree2f = SimdKdTreef<2>;
using SimdKdTree3f = SimdKdTreef<3>;

AXEL_DEFINE_POINTERS(SimdKdTree2f);
AXEL_DEFINE_POINTERS(SimdKdTree3f);

extern template class SimdKdTreef<2>;
extern template class SimdKdTreef<3>;

#ifdef AXEL_ENABLE_AVX

template <int32_t nDim>
class SimdKdTreeAvxf : public SimdKdTreef<nDim> {
 public:
  using Base = SimdKdTreef<nDim>;

  using Base::kColorDimensions;
  using Base::kFarValueFloat;

  using Box = typename Base::Box;
  using Col = typename Base::Col;
  using Scalar = typename Base::Scalar;
  using SizeType = typename Base::SizeType;
  using Vec = typename Base::Vec;

  static constexpr SizeType kMaxDepth = Base::kMaxDepth;

  /// Constructs k-d tree from points, normals, and colors
  explicit SimdKdTreeAvxf(
      std::span<const Vec> points = std::span<Vec>{},
      std::span<const Vec> normals = std::span<Vec>{},
      std::span<const Col> colors = std::span<Col>{});

  /// Destructor
  ~SimdKdTreeAvxf() override;

  /// Disables copy constructor.
  SimdKdTreeAvxf(const SimdKdTreeAvxf&) = delete;

  /// Disables assignment operator.
  SimdKdTreeAvxf& operator=(const SimdKdTreeAvxf&) = delete;

  // Documentation inherited
  [[nodiscard]] std::tuple<bool, SizeType, Scalar> closestPoint(
      const Vec& queryPoint,
      Scalar maxSqrDist = std::numeric_limits<Scalar>::max()) const override;

  // Documentation inherited
  [[nodiscard]] std::tuple<bool, SizeType, Scalar> closestPoint(
      const Vec& queryPoint,
      const Vec& normal,
      Scalar maxSqrDist = std::numeric_limits<Scalar>::max(),
      Scalar maxNormalDot = 0.0f) const override;

  // Documentation inherited
  [[nodiscard]] std::tuple<bool, SizeType, Scalar> closestPoint(
      const Vec& queryPoint,
      const Vec& normal,
      const Col& color,
      Scalar maxSqrDist = std::numeric_limits<Scalar>::max(),
      Scalar maxNormalDot = 0.0f,
      Scalar maxColorSqrDist = 0.5f) const override;

  /// Allows the user to mark out which points are considered 'acceptable.'
  ///
  /// The passed-in function should map from a __m256i (the indices) to an __m256i which uses the
  /// standard AVX convention (all ones in the appropriate spots). It also needs to be tolerant to
  /// possible INT_MAX values since we may not have actual indices for all 4 points.
  ///
  /// Here's an example acceptance function that looks for odd-valued points:
  ///
  /// @code
  ///  [] (const __m256i indices_in) -> __m256i
  ///  {
  ///      const int all_zeros = 0;
  ///      const int all_ones = ~all_zeros;
  ///
  ///      alignas(AVX_ALIGNMENT) int32_t indices[8];
  ///      _mm256_store_si256 ((__m256i*) indices, indices_in);
  ///
  ///      alignas(AVX_ALIGNMENT) int32_t result[8];
  ///      for (int32_t j = 0; j < 8; ++j)
  ///      {
  ///          if (indices[j] % 2 == 0)
  ///              result[j] = all_ones;
  ///          else
  ///              result[j] = all_zeros;
  ///      }
  ///
  ///      return _mm256_load_si256 ((const __m256i*) result);
  ///  } );
  /// @endcode
  template <typename AcceptanceFunction>
  [[nodiscard]] std::tuple<bool, SizeType, Scalar> closestPointWithAcceptance(
      const Vec& queryPoint,
      Scalar maxSqrDist,
      const AcceptanceFunction& acceptanceFunction) const;

  // Documentation inherited
  void pointsInNSphere(const Vec& center, Scalar radius, std::vector<SizeType>& points)
      const override;

 protected:
  // Documentation inherited
  [[nodiscard]] size_t getSimdPacketSize() const override;

  // Documentation inherited
  void validateInternal(SizeType node, const Box& box, std::vector<char>& touched) const override;

  // Documentation inherited
  SizeType createLeafNode(
      std::vector<std::pair<SizeType, Vec>>& points,
      std::span<const Vec> normals,
      std::span<const Col> colors,
      SizeType start,
      SizeType end,
      const Box& box) override;

 private:
  using Node = typename Base::Node;

  static constexpr int32_t AVX_FLOAT_BLOCK_SIZE = 8;
  static constexpr int32_t AVX_ALIGNMENT = AVX_FLOAT_BLOCK_SIZE * 4;

  /// In order to enable the use of AVX intrinsics, we will store the points in blocks of 8. Empty
  /// point values will get FAR_VALUE for their location (so we can safely use it in difference and
  /// squared norm operations) and INT_MAX for their index.
  struct PointBlock {
    __m256 values[nDim]; // [ {x1 x2 x3 x4 x5 x6 x7 x8}, {y1 y2 y3 y4 y5 y6 y7 y8}, ... ]
    __m256i indices; // [i1 i2 i3 i4 i5 i6 i7 i8]
  };

  struct NormalBlock {
    __m256 values[nDim];
  };

  struct ColorBlock {
    __m256 values[4];
  };

  /// Initializes the k-d tree. This function is intended to be only called by the constructor.
  void init(
      std::span<const Vec> points_in,
      std::span<const Vec> normals_in,
      std::span<const Col> colors_in);

  std::vector<PointBlock> pointBlocks_;
  std::vector<NormalBlock> normalBlocks_;
  std::vector<ColorBlock> colorBlocks_;
};

template <int32_t nDim>
template <typename AcceptanceFunction>
std::tuple<bool, typename SimdKdTreeAvxf<nDim>::SizeType, typename SimdKdTreeAvxf<nDim>::Scalar>
SimdKdTreeAvxf<nDim>::closestPointWithAcceptance(
    const Vec& queryPoint,
    Scalar maxSqrDist,
    const AcceptanceFunction& acceptanceFunction) const {
  Scalar bestSqrDist = maxSqrDist;
  SizeType bestPoint = std::numeric_limits<SizeType>::max();
  bool foundPoint = false;

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
    query_p[iDim] = _mm256_broadcast_ss(&queryPoint(iDim));
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
      XR_CHECK(pointBlocksEnd > pointBlocksStart);

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
        const __m256 userMask = _mm256_castsi256_ps(acceptanceFunction(pointBlock.indices));
        const __m256 finalMask = _mm256_and_ps(lessThanMask, userMask);
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
        if (bestSqrDist_extract[k] < bestSqrDist) {
          XR_CHECK(bestSqrDistIndices_extract[k] != INT_MAX);
          XR_CHECK(bestSqrDistIndices_extract[k] < this->numPoints_);
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

using SimdKdTreeAvx3f = SimdKdTreeAvxf<3>;
using SimdKdTreeAvx2f = SimdKdTreeAvxf<2>;

extern template class SimdKdTreeAvxf<3>;
extern template class SimdKdTreeAvxf<2>;

#endif

} // namespace axel
