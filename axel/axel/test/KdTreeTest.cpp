/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gmock/gmock.h>

#include "axel/KdTree.h"
#include "axel/common/Constants.h"

namespace axel::test {
namespace {

using ::testing::AllOf;
using ::testing::DoubleNear;
using ::testing::Each;
using ::testing::Eq;
using ::testing::FloatNear;
using ::testing::Ge;
using ::testing::IsEmpty;
using ::testing::Lt;
using ::testing::Not;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAreArray;

// TODO(nemanjab): Unify this with the BvhTest as "test_utils" once migration of both data
// structures is done.
template <typename T>
struct MeshData {
  Eigen::MatrixX3<T> positions;
  Eigen::MatrixX3i triangles;
};

template <typename T>
struct CubeMeshParameters {
  T edgeLength;
};

template <typename T>
MeshData<T> generateCube(const T& edgeLength) {
  MeshData<T> mesh{};
  mesh.positions.resize(8, 3);
  mesh.triangles.resize(12, 3);

  // 4 vertices of the bottom face.
  mesh.positions.row(0) = Eigen::Vector3<T>(0., 0., 0.);
  mesh.positions.row(1) = Eigen::Vector3<T>(edgeLength, 0., 0.);
  mesh.positions.row(2) = Eigen::Vector3<T>(0., edgeLength, 0.);
  mesh.positions.row(3) = Eigen::Vector3<T>(edgeLength, edgeLength, 0.);

  // 4 vertices of the top face.
  mesh.positions.row(4) = Eigen::Vector3<T>(0., 0., edgeLength);
  mesh.positions.row(5) = Eigen::Vector3<T>(edgeLength, 0., edgeLength);
  mesh.positions.row(6) = Eigen::Vector3<T>(0., edgeLength, edgeLength);
  mesh.positions.row(7) = Eigen::Vector3<T>(edgeLength, edgeLength, edgeLength);

  mesh.triangles.row(0) = Eigen::Vector3i(0, 2, 1);
  mesh.triangles.row(1) = Eigen::Vector3i(1, 2, 3);
  mesh.triangles.row(2) = Eigen::Vector3i(1, 5, 4);
  mesh.triangles.row(3) = Eigen::Vector3i(1, 4, 0);
  mesh.triangles.row(4) = Eigen::Vector3i(0, 4, 2);
  mesh.triangles.row(5) = Eigen::Vector3i(2, 4, 6);
  mesh.triangles.row(6) = Eigen::Vector3i(1, 3, 5);
  mesh.triangles.row(7) = Eigen::Vector3i(3, 7, 5);
  mesh.triangles.row(8) = Eigen::Vector3i(3, 2, 6);
  mesh.triangles.row(9) = Eigen::Vector3i(3, 6, 7);
  mesh.triangles.row(10) = Eigen::Vector3i(4, 5, 6);
  mesh.triangles.row(11) = Eigen::Vector3i(5, 7, 6);

  return mesh;
}

using ScalarTypes = testing::Types<float, double>;

template <typename T>
struct KdTreeTest : testing::Test {
  using Type = T;
  static constexpr T kEdgeLength = 3.0;
};

TYPED_TEST_SUITE(KdTreeTest, ScalarTypes);

TYPED_TEST(KdTreeTest, EigenMatrixDetection) {
  using T = typename TestFixture::Type;

  const Eigen::MatrixX3<T> positions = generateCube<T>(TestFixture::kEdgeLength).positions;

  // Identifies any Eigen expressions that own their storage.
  EXPECT_TRUE(detail::IsEigenTypeWithStorage<decltype(positions)>);
  EXPECT_TRUE(detail::IsEigenTypeWithStorage<decltype(positions.transpose().eval())>);
  EXPECT_TRUE(detail::IsEigenTypeWithStorage<decltype(positions.transpose().matrix().eval())>);
  EXPECT_FALSE(detail::IsEigenTypeWithStorage<decltype(positions.transpose())>);
  EXPECT_FALSE(detail::IsEigenTypeWithStorage<decltype(positions.transpose().matrix())>);

  // Identifies both row and column vectors.
  EXPECT_TRUE((detail::IsContiguousEigenVectorBaseWithLength<Eigen::Vector2<T>, 2>));
  EXPECT_TRUE((detail::IsContiguousEigenVectorBaseWithLength<Eigen::Vector3<T>, 3>));
  EXPECT_TRUE((detail::IsContiguousEigenVectorBaseWithLength<Eigen::Vector4<T>, 4>));
  EXPECT_TRUE((detail::IsContiguousEigenVectorBaseWithLength<Eigen::VectorX<T>, Eigen::Dynamic>));
  EXPECT_TRUE((detail::IsContiguousEigenVectorBaseWithLength<Eigen::RowVector2<T>, 2>));
  EXPECT_TRUE((detail::IsContiguousEigenVectorBaseWithLength<Eigen::RowVector3<T>, 3>));
  EXPECT_TRUE((detail::IsContiguousEigenVectorBaseWithLength<Eigen::RowVector4<T>, 4>));
  EXPECT_TRUE(
      (detail::IsContiguousEigenVectorBaseWithLength<Eigen::RowVectorX<T>, Eigen::Dynamic>));

  // Check that block expressions that results in vectors have valid memory layout.
  // E.g. a row vector block from a column-major matrix cannot be used before materialization.
  const Eigen::Matrix<T, 3, 3, Eigen::RowMajor> rm;
  EXPECT_TRUE((detail::IsContiguousEigenVectorBaseWithLength<decltype(rm.row(0)), 3>));
  EXPECT_FALSE((detail::IsContiguousEigenVectorBaseWithLength<decltype(rm.col(0)), 3>));

  const Eigen::Matrix<T, 3, 3, Eigen::ColMajor> cm;
  EXPECT_FALSE((detail::IsContiguousEigenVectorBaseWithLength<decltype(cm.row(0)), 3>));
  EXPECT_TRUE((detail::IsContiguousEigenVectorBaseWithLength<decltype(cm.col(0)), 3>));

  // A Matrix shouldn't match the vector type.
  EXPECT_FALSE((detail::IsContiguousEigenVectorBaseWithLength<Eigen::Matrix3<T>, 3>));
  EXPECT_FALSE((detail::IsContiguousEigenVectorBaseWithLength<Eigen::MatrixX<T>, 3>));
}

TYPED_TEST(KdTreeTest, Construction) {
  using T = typename TestFixture::Type;

  const Eigen::MatrixX3<T> positions = generateCube<T>(TestFixture::kEdgeLength).positions;

  const KdTree<Eigen::MatrixX3<T>> kdTree1(positions); // From lvalue ref.
  EXPECT_EQ(kdTree1.getSize(), positions.rows());

  Eigen::MatrixX3<T> positionsCopy = positions;
  const KdTree<Eigen::MatrixX3<T>> kdTree2(std::move(positionsCopy)); // From rvalue ref.
  EXPECT_EQ(kdTree2.getSize(), positions.rows());

  // Empty tree.
  const KdTree<Eigen::MatrixX3<T>> emptyTree(Eigen::MatrixX3<T>{});
  EXPECT_EQ(emptyTree.getSize(), 0);

  // Does not compile as it fails the sfinae check for sametypeness.
  // const Eigen::Matrix3Xd transposedPositions = positions.transpose();
  // const KdTree<Eigen::MatrixX3<T>> kdTree3(transposedPositions);

  // Does not compile as it fails the sfinae check for storage requirements.
  // const KdTree<Eigen::MatrixX3<T>> kdTree4(transposedPositions.transpose());
}

TYPED_TEST(KdTreeTest, Construction_DynamicDimension) {
  using T = typename TestFixture::Type;
  const Eigen::MatrixX<T> positions = generateCube<T>(TestFixture::kEdgeLength).positions;
  const KdTree<Eigen::MatrixX<T>> kdTreeDynamicDim(positions);
  EXPECT_EQ(kdTreeDynamicDim.getSize(), positions.rows());
}

TYPED_TEST(KdTreeTest, RadiusSearch_BoundingSphereAllPoints) {
  using T = typename TestFixture::Type;
  const Eigen::MatrixX3<T> positions = generateCube<T>(TestFixture::kEdgeLength).positions;
  const KdTree<Eigen::MatrixX3<T>> kdTree(positions);
  EXPECT_EQ(kdTree.getSize(), positions.rows());

  // Place in the center and set the radius to slightly higher than diameter.
  const Eigen::Vector3<T> queryPoint = Eigen::Vector3<T>::Ones() * TestFixture::kEdgeLength / 2.0;
  const T radius = TestFixture::kEdgeLength * std::sqrt(3.0) * 0.5;
  const auto results = kdTree.radiusSearch(queryPoint, radius * 1.01, SortByDistance::False);
  EXPECT_THAT(results, SizeIs(positions.rows()));
  for (auto& [index, distanceSquared] : results) {
    EXPECT_THAT(index, AllOf(Ge(0), Lt(positions.rows())));
    EXPECT_NEAR(std::sqrt(distanceSquared), radius, detail::eps<T>());
  }
}

TYPED_TEST(KdTreeTest, RadiusSearch_OnePoint) {
  using T = typename TestFixture::Type;
  const Eigen::MatrixX3<T> positions = generateCube<T>(TestFixture::kEdgeLength).positions;
  const KdTree<Eigen::MatrixX3<T>> kdTree(positions);
  EXPECT_EQ(kdTree.getSize(), positions.rows());

  // Place in the 0-vertex and search such that we get only that one.
  const Eigen::Vector3<T> queryPoint{Eigen::Vector3<T>::Zero()};
  const T radius{TestFixture::kEdgeLength - std::numeric_limits<T>::epsilon()};
  EXPECT_THAT(kdTree.radiusSearch(queryPoint, radius, SortByDistance::False), SizeIs(1));
}

TYPED_TEST(KdTreeTest, RadiusSearch_SortByDistance_ReturnsAscendingOrder) {
  using T = typename TestFixture::Type;
  const Eigen::MatrixX3<T> positions = generateCube<T>(TestFixture::kEdgeLength).positions;
  const KdTree<Eigen::MatrixX3<T>> kdTree(positions);
  EXPECT_EQ(kdTree.getSize(), positions.rows());

  // Place in the center and set the radius to slightly higher than diameter.
  const Eigen::Vector3<T> queryPoint{Eigen::Vector3<T>::Zero()};
  const T radius{TestFixture::kEdgeLength * std::sqrt(T(3))};
  const auto results{kdTree.radiusSearch(queryPoint, radius * 1.01, SortByDistance::True)};
  EXPECT_THAT(results, SizeIs(positions.rows()));
  ASSERT_THAT(results, Not(IsEmpty()));
  for (uint32_t i = 0; i < results.size() - 1; ++i) {
    EXPECT_LE(results[i].second, results[i + 1].second);
  }
}

TYPED_TEST(KdTreeTest, RadiusSearch_DifferentQueryVectorTypes) {
  using T = typename TestFixture::Type;

  const Eigen::MatrixX3<T> positions = generateCube<T>(TestFixture::kEdgeLength).positions;
  const KdTree<Eigen::MatrixX3<T>> kdTree(positions);
  EXPECT_EQ(kdTree.getSize(), positions.rows());

  const T radius{TestFixture::kEdgeLength * std::sqrt(T(3)) * T(1.01)};

  // Column vector.
  const Eigen::Vector3<T> queryPoint{Eigen::Vector3<T>::Zero()};
  EXPECT_THAT(
      kdTree.radiusSearch(queryPoint, radius, SortByDistance::True), SizeIs(positions.rows()));

  // Row vector.
  const Eigen::RowVector3<T> rowQueryPoint{Eigen::Vector3<T>::Zero()};
  EXPECT_THAT(
      kdTree.radiusSearch(rowQueryPoint, radius, SortByDistance::True), SizeIs(positions.rows()));

  // Vector-like/block expression, has to be contiguous, so we cast to RowMajor in this unit test.
  const Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor> rmPositions = positions;
  EXPECT_THAT(
      kdTree.radiusSearch(rmPositions.row(0), radius, SortByDistance::False),
      SizeIs(positions.rows()));
}

TYPED_TEST(KdTreeTest, KnnSearch) {
  using T = typename TestFixture::Type;

  const Eigen::MatrixX3<T> positions = generateCube<T>(TestFixture::kEdgeLength).positions;
  const KdTree<Eigen::MatrixX3<T>> kdTree(positions);
  EXPECT_EQ(kdTree.getSize(), positions.rows());

  const Eigen::Vector3<T> queryPoint{1.5, 1.5, 0.0};
  constexpr size_t kNbrCount{4};
  const auto results{kdTree.knnSearch(queryPoint, kNbrCount)};
  EXPECT_THAT(results.indices, SizeIs(kNbrCount));
  EXPECT_THAT(results.indices, SizeIs(results.squaredDistances.size()));
  if constexpr (std::is_same_v<T, float>) {
    EXPECT_THAT(
        results.squaredDistances,
        Each(FloatNear(TestFixture::kEdgeLength * TestFixture::kEdgeLength * 0.5, 1e-6)));
  } else {
    EXPECT_THAT(
        results.squaredDistances,
        Each(DoubleNear(TestFixture::kEdgeLength * TestFixture::kEdgeLength * 0.5, 1e-15)));
  }
  EXPECT_THAT(results.indices, UnorderedElementsAreArray({0, 1, 2, 3}));
}

TYPED_TEST(KdTreeTest, KnnSearch_MoreNbrsRequestedThanPoints) {
  using T = typename TestFixture::Type;

  const Eigen::MatrixX3<T> positions = generateCube<T>(TestFixture::kEdgeLength).positions;
  const KdTree<Eigen::MatrixX3<T>> kdTree(positions);
  EXPECT_EQ(kdTree.getSize(), positions.rows());

  const Eigen::Vector3<T> queryPoint{1.5, 1.5, 0.0};
  constexpr size_t kNbrCount{40};
  const auto results{kdTree.knnSearch(queryPoint, kNbrCount)};
  EXPECT_THAT(results.indices, SizeIs(positions.rows()));
  EXPECT_THAT(results.indices, SizeIs(results.squaredDistances.size()));
}

TYPED_TEST(KdTreeTest, KnnSearch_ZeroNbrsRequested) {
  using T = typename TestFixture::Type;

  const Eigen::MatrixX3<T> positions = generateCube<T>(TestFixture::kEdgeLength).positions;
  const KdTree<Eigen::MatrixX3<T>> kdTree(positions);
  EXPECT_EQ(kdTree.getSize(), positions.rows());

  const Eigen::Vector3<T> queryPoint{1.5, 1.5, 0.0};
  constexpr size_t kNbrCount{0};
  const auto results{kdTree.knnSearch(queryPoint, kNbrCount)};
  EXPECT_THAT(results.indices, IsEmpty());
  EXPECT_THAT(results.squaredDistances, IsEmpty());
}

TYPED_TEST(KdTreeTest, ClosestSearch) {
  using T = typename TestFixture::Type;

  const Eigen::MatrixX3<T> positions = generateCube<T>(TestFixture::kEdgeLength).positions;
  const KdTree<Eigen::MatrixX3<T>> kdTree(positions);
  EXPECT_EQ(kdTree.getSize(), positions.rows());

  const Eigen::Vector3<T> queryPoint{1.0, 0.0, 0.0};
  const auto result{kdTree.closestSearch(queryPoint)};
  ASSERT_THAT(result, Not(Eq(std::nullopt)));
  EXPECT_THAT(result->index, 0);
  EXPECT_EQ(result->squaredDistance, 1.0);
}

TYPED_TEST(KdTreeTest, ClosestSearch_EmptyTree) {
  using T = typename TestFixture::Type;

  const KdTree<Eigen::MatrixX3<T>> kdTree(Eigen::MatrixX3<T>{});
  EXPECT_TRUE(kdTree.isEmpty());

  const Eigen::Vector3<T> queryPoint{1.0, 0.0, 0.0};
  const auto result{kdTree.closestSearch(queryPoint)};
  EXPECT_THAT(result, Eq(std::nullopt));
}

} // namespace
} // namespace axel::test
