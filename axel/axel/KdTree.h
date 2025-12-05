/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <optional>

#include <nanoflann/nanoflann.hpp>
#include <Eigen/Core>
#include <Eigen/Eigen>

#ifdef True
#undef True
#endif // True

#ifdef False
#undef False
#endif // False

namespace axel {

namespace detail {

template <typename EigenType>
inline constexpr bool IsEigenTypeWithStorage =
    std::is_base_of_v<Eigen::PlainObjectBase<std::decay_t<EigenType>>, std::decay_t<EigenType>>;

template <typename EigenType, Eigen::Index Length>
inline constexpr bool IsContiguousEigenVectorBaseWithLength =
    (std::decay_t<EigenType>::RowsAtCompileTime == Length ||
     std::decay_t<EigenType>::ColsAtCompileTime == Length) &&
    (std::decay_t<EigenType>::RowsAtCompileTime == 1 ||
     std::decay_t<EigenType>::ColsAtCompileTime == 1) &&
    std::decay_t<EigenType>::InnerStrideAtCompileTime == 1; // Make sure the memory is contiguous.
} // namespace detail

enum class SortByDistance { True, False };

template <typename EigenMatrixType>
class KdTree {
 public:
  // The scalar type of the point coordinates.
  using Scalar = typename EigenMatrixType::Scalar;

  // The storage of data needs to be done with an Eigen::Matrix.
  static_assert(
      detail::IsEigenTypeWithStorage<EigenMatrixType>,
      "KdTree's current implementation only works with Eigen matrices.");

  // Dimensionality of the tree is determined by compile-time columns in the point matrix.
  // It can also be dynamic, i.e. Eigen::Dynamic.
  static constexpr Eigen::Index kDimensionality = EigenMatrixType::ColsAtCompileTime;

  using TreeAdaptorType = nanoflann::KDTreeEigenMatrixAdaptor<EigenMatrixType, kDimensionality>;

  // The underlying index for elements, i.e. Eigen::Index.
  using IndexType = typename TreeAdaptorType::IndexType;

  using RadiusSearchResults = std::vector<std::pair<IndexType, Scalar>>;

  struct KnnSearchResults {
    std::vector<IndexType> indices;
    std::vector<Scalar> squaredDistances;
  };

  struct ClosestSearchResult {
    IndexType index{};
    Scalar squaredDistance{};
  };

  // Only allow lvalue and rvalue references of points that are exactly the Eigen::Matrix type
  // we used to construct the tree.
  template <
      typename T,
      typename = std::enable_if_t<
          detail::IsEigenTypeWithStorage<T> && // This check might be obsolete due to transitivity.
          std::is_same_v<std::decay_t<T>, std::decay_t<EigenMatrixType>>>>
  explicit KdTree(T&& points)
      : points_(std::forward<T>(points)), tree_(static_cast<int32_t>(points_.cols()), points_) {
    tree_.index->buildIndex();
  }

  // Returns the number of points in the tree.
  [[nodiscard]] size_t getSize() const {
    return static_cast<size_t>(tree_.index->size());
  }

  [[nodiscard]] bool isEmpty() const {
    return tree_.index->size() == 0;
  }

  /**
   * @brief Returns all nearby points in the tree to a given query point with the given radius.
   *
   * @param query The point to query the tree with.
   * @param radius The radius to search with.
   * @param sort If SortByDistance::True, the results are sorted by distance.
   * @param results The results contain the indices and squared distances of the points within the
   * given radius.
   */
  template <typename EigenVectorType>
  void radiusSearch(
      EigenVectorType&& query,
      const Scalar radius,
      const SortByDistance sort,
      RadiusSearchResults& results) const {
    static_assert(detail::IsContiguousEigenVectorBaseWithLength<EigenVectorType, kDimensionality>);
    nanoflann::SearchParams params{};
    params.sorted = sort == SortByDistance::True;
    tree_.index->radiusSearch(query.data(), radius * radius, results, params);
  }

  /**
   * @brief Returns all nearby points in the tree to a given query point with the given radius.
   *
   * @param query The point to query the tree with.
   * @param radius The radius to search with.
   * @param sort If SortByDistance::True, the results are sorted by distance.
   * @return The results contain the indices and squared distances of the points within the given
   * radius.
   */
  template <typename EigenVectorType>
  RadiusSearchResults
  radiusSearch(EigenVectorType&& query, const Scalar radius, const SortByDistance sort) const {
    static_assert(detail::IsContiguousEigenVectorBaseWithLength<EigenVectorType, kDimensionality>);
    RadiusSearchResults results{};
    radiusSearch(query, radius, sort, results);
    return results;
  }

  /**
   * @brief Returns the K closest points in the tree to a given query point.
   *
   * @param query The point to query the tree with.
   * @param numClosest The number of closest points to return.
   * @param results The K closest points, denoted by their index and squared distance. If the tree
   * contains fewer than @param numClosest points, returns all the points in the tree instead.
   */
  template <typename EigenVectorType>
  void knnSearch(EigenVectorType&& query, size_t numClosest, KnnSearchResults& results) const {
    static_assert(detail::IsContiguousEigenVectorBaseWithLength<EigenVectorType, kDimensionality>);

    numClosest = std::min(numClosest, getSize());
    results.indices.resize(numClosest);
    results.squaredDistances.resize(numClosest);
    if (numClosest == 0) {
      return; // Have to early-exit here because nanoflann doesn't support empty queries.
    }

    tree_.index->knnSearch(
        query.data(), numClosest, results.indices.data(), results.squaredDistances.data());
  }

  /**
   * @brief Returns the K closest points in the tree to a given query point.
   *
   * @param query The point to query the tree with.
   * @param numClosest The number of closest points to return.
   * @return The K closest points, denoted by their index and squared distance. If the tree contains
   * fewer than @param numClosest points, returns all the points in the tree instead.
   */
  template <typename EigenVectorType>
  KnnSearchResults knnSearch(EigenVectorType&& query, const size_t numClosest) const {
    static_assert(detail::IsContiguousEigenVectorBaseWithLength<EigenVectorType, kDimensionality>);
    KnnSearchResults results{};
    knnSearch(query, numClosest, results);
    return results;
  }

  /**
   * @brief Returns the closest point in the tree to a given query point.

   * @param query The point to query the tree with.
   * @return The closest point. If the tree is empty, returns std::nullopt.
   */
  template <typename EigenVectorType>
  std::optional<ClosestSearchResult> closestSearch(EigenVectorType&& query) const {
    static_assert(detail::IsContiguousEigenVectorBaseWithLength<EigenVectorType, kDimensionality>);

    if (isEmpty()) {
      return std::nullopt;
    }

    ClosestSearchResult result{};
    tree_.index->knnSearch(query.data(), /*num_closest=*/1, &result.index, &result.squaredDistance);
    return result;
  }

 private:
  EigenMatrixType points_;
  TreeAdaptorType tree_;
};

template <typename ScalarType>
using KdTree2 = KdTree<Eigen::Matrix<ScalarType, Eigen::Dynamic, 2>>;
using KdTree2f = KdTree2<float>;
using KdTree2d = KdTree2<double>;

template <typename ScalarType>
using KdTree3 = KdTree<Eigen::Matrix<ScalarType, Eigen::Dynamic, 3>>;
using KdTree3f = KdTree3<float>;
using KdTree3d = KdTree3<double>;

} // namespace axel
