/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "axel/TriBvh.h"
#include "axel/common/Constants.h"
#include "axel/test/Helper.h"

#include <igl/AABB.h>
#include <perception/test_helpers/EigenChecks.h>
#include <test_helpers/FollyMatchers.h>

namespace axel::test {
namespace {

using ::testing::DoubleNear;
using ::testing::FloatNear;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

template <typename S>
constexpr SphereMeshParameters<S> kSphereParams{3.0, 64, 64};

using ScalarTypes = testing::Types<float, double>;

template <typename S, size_t LeafCapacity>
TriBvh<S, LeafCapacity> makeBvh(const MeshData<S>& meshData) {
  auto pos = meshData.positions;
  auto tris = meshData.triangles;
  return TriBvh<S, LeafCapacity>{std::move(pos), std::move(tris), 0.0};
}

template <typename T>
struct TriBvhTest : testing::Test {
  using Type = T;
  static constexpr T kEdgeLength = 3.0;
};

TYPED_TEST_SUITE(TriBvhTest, ScalarTypes);

TYPED_TEST(TriBvhTest, rayHitsAny) {
  using S = typename TestFixture::Type;

  auto sphere = generateSphere(kSphereParams<S>);
  const Eigen::Index triangleCount{sphere.triangles.rows()};
  constexpr S kThickness{1e-2};
  const TriBvh<S> bvh{std::move(sphere.positions), std::move(sphere.triangles), kThickness};

  EXPECT_EQ(bvh.getPrimitiveCount(), triangleCount);

  // Hits.
  EXPECT_TRUE(bvh.anyHit(Ray3<S>(Eigen::Vector3<S>(-10, 0, 0), Eigen::Vector3<S>(1, 0, 0))));

  // Ignores hits too far.
  EXPECT_FALSE(bvh.anyHit(Ray3<S>(Eigen::Vector3<S>(-10, 0, 0), Eigen::Vector3<S>(1, 0, 0), 6.0)));

  // No hits despite bvh query not empty.
  EXPECT_FALSE(bvh.anyHit(
      Ray3<S>(Eigen::Vector3<S>(0, 0, 3.0 + kThickness * 1.01), Eigen::Vector3<S>(0, 0, 1))));

  // Query has at least one triangle here because it is within the triangle's thick bounding box.
  EXPECT_THAT(
      bvh.lineHits(
          Ray3<S>(Eigen::Vector3<S>(0, 0, 3.0 + kThickness * 0.5), Eigen::Vector3<S>(0, 0, 1))),
      Not(IsEmpty()));

  // When starting outside the bounding box, no potential hits will be registered on the
  // line.
  EXPECT_THAT(
      bvh.lineHits(
          Ray3<S>(Eigen::Vector3<S>(0, 0, 3.0 + kThickness * 1.5), Eigen::Vector3<S>(0, 0, 1))),
      IsEmpty());
}

TYPED_TEST(TriBvhTest, rayHitsClosest) {
  using S = typename TestFixture::Type;

  auto sphere = generateSphere(kSphereParams<S>);
  const TriBvh<S> bvh{std::move(sphere.positions), std::move(sphere.triangles)};

  // Hits.
  std::optional<IntersectionResult<S>> closestHit;
  closestHit = bvh.closestHit(Ray3<S>(Eigen::Vector3<S>(-10, 0, 0), Eigen::Vector3<S>(1, 0, 0)));
  ASSERT_NE(closestHit, std::nullopt);
  EXPECT_EIGEN_MATRIX_NEAR(closestHit->hitPoint, Eigen::Vector3<S>(-3, 0, 0), 1e-2);

  // Ignores hits too far.
  closestHit =
      bvh.closestHit(Ray3<S>(Eigen::Vector3<S>(-10, 0, 0), Eigen::Vector3<S>(1, 0, 0), 6.0));
  EXPECT_EQ(closestHit, std::nullopt);

  // Ignores hits before origin.
  closestHit = bvh.closestHit(Ray3<S>(Eigen::Vector3<S>(0, -2, 0), Eigen::Vector3<S>(0, 1, 0)));
  ASSERT_NE(closestHit, std::nullopt);
  EXPECT_NEAR(closestHit->hitDistance, 5.0, 1e-2);
  EXPECT_EIGEN_MATRIX_NEAR(closestHit->hitPoint, Eigen::Vector3<S>(0, 3, 0), 1e-2);

  // No hits
  EXPECT_EQ(
      bvh.closestHit(Ray3<S>(Eigen::Vector3<S>(0, 0, 10), Eigen::Vector3<S>(0, 0, 1))),
      std::nullopt);
}

TYPED_TEST(TriBvhTest, rayAllHits) {
  using S = typename TestFixture::Type;

  auto sphere = generateSphere(kSphereParams<S>);
  rotate(
      sphere,
      (Eigen::AngleAxis<S>(M_PI / 13, Eigen::Vector3<S>::UnitX()) *
       Eigen::AngleAxis<S>(M_PI / 13, Eigen::Vector3<S>::UnitY()) *
       Eigen::AngleAxis<S>(M_PI / 13, Eigen::Vector3<S>::UnitZ()))
          .toRotationMatrix()); // this is to avoid hitting the special case where the ray hits an
                                // edge or a vertex
  const TriBvh<S> bvh{std::move(sphere.positions), std::move(sphere.triangles)};

  // Hits.
  std::vector<IntersectionResult<S>> allHits;
  allHits = bvh.allHits(Ray3<S>(Eigen::Vector3<S>(-10, 0, 0), Eigen::Vector3<S>(1, 0, 0)));
  EXPECT_THAT(allHits, SizeIs(2));
  if constexpr (std::is_same_v<S, float>) {
    EXPECT_THAT(
        (std::vector<S>{allHits.at(0).hitDistance, allHits.at(1).hitDistance}),
        UnorderedElementsAre(FloatNear(7.0, 1e-2), FloatNear(13.0, 1e-2)));
  } else if constexpr (std::is_same_v<S, double>) {
    EXPECT_THAT(
        (std::vector<S>{allHits.at(0).hitDistance, allHits.at(1).hitDistance}),
        UnorderedElementsAre(DoubleNear(7.0, 1e-2), DoubleNear(13.0, 1e-2)));
  }

  // Ignores hits too far.
  allHits = bvh.allHits(Ray3<S>(Eigen::Vector3<S>(-10, 0, 0), Eigen::Vector3<S>(1, 0, 0), 6.0));
  EXPECT_THAT(allHits, IsEmpty());

  // Ignores hits before origin.
  allHits = bvh.allHits(Ray3<S>(Eigen::Vector3<S>(0, -2, 0), Eigen::Vector3<S>(0, 1, 0)));
  EXPECT_THAT(allHits, SizeIs(1));
  EXPECT_EIGEN_MATRIX_NEAR(allHits.at(0).hitPoint, Eigen::Vector3<S>(0, 3, 0), 1e-2);

  // No hits
  EXPECT_THAT(
      bvh.allHits(Ray3<S>(Eigen::Vector3<S>(0, 0, 10), Eigen::Vector3<S>(0, 0, 1))), IsEmpty());
}

TYPED_TEST(TriBvhTest, differentThickness) {
  using S = typename TestFixture::Type;

  const Eigen::Vector3<S> org{3.5, 0.0, 3.5};
  const Eigen::Vector3<S> dir{0.0, 0.0, -1.0};

  // Thick BVH has its bboxes expanded by 1 unit. The given ray parameters will hit through that.
  auto s1 = generateSphere(kSphereParams<S>);
  auto s2 = s1;
  const TriBvh<S> bvhWithThickness{std::move(s1.positions), std::move(s1.triangles), 1.0};
  EXPECT_THAT(bvhWithThickness.lineHits(Ray3<S>(org, dir)), Not(IsEmpty()));

  // BVH without thickness will not register any potential hit.
  const TriBvh<S> bvhNoThickness{std::move(s2.positions), std::move(s2.triangles), 0.0};
  EXPECT_THAT(bvhNoThickness.lineHits(Ray3<S>(org, dir)), IsEmpty());

  // None of the hits should actually be primitive hits.
  EXPECT_EQ(bvhWithThickness.closestHit(Ray3<S>(org, dir)), std::nullopt);
  EXPECT_EQ(bvhNoThickness.closestHit(Ray3<S>(org, dir)), std::nullopt);
}

TYPED_TEST(TriBvhTest, ClosestSurfacePoint_SameResultsAsIgl) {
  using S = typename TestFixture::Type;

  auto sphere = generateSphere(kSphereParams<S>);
  const Eigen::MatrixX<S> positions = sphere.positions;
  const Eigen::MatrixXi faces = sphere.triangles;

  igl::AABB<Eigen::MatrixX<S>, 3> aabb;
  aabb.init(positions, faces);

  const auto checkQueries = [&](const auto& bvh) {
    // Our query points will come from the vertices of a simple grid mesh.
    constexpr GridMeshParameters<S> kGridParams{10.0, 10.0, 1.0};
    const Eigen::MatrixX<S> queryPoints =
        generateGrid(kGridParams, Eigen::Vector3<S>(0, 0, 5)).positions;

    for (uint32_t i = 0; i < queryPoints.rows(); ++i) {
      const Eigen::Vector3<S> query = queryPoints.row(i);

      int32_t tri0{};
      Eigen::RowVector3<S> p0{};
      aabb.squared_distance(positions, faces, query, tri0, p0);

      const auto result = bvh.closestSurfacePoint(query);
      const auto& p1 = result.point;
      const auto& tri1 = result.triangleIdx;
      const auto& baryCoords = result.baryCoords.value();

      EXPECT_NEAR(
          (query - Eigen::Vector3<S>(p0)).norm(), (query - p1).norm(), detail::eps<S>(1e-6, 1e-14))
          << "Failed for: " << query.transpose();

      // Check barycentric coordinates
      const Eigen::Vector3<S> v0 = positions.row(faces(tri1, 0));
      const Eigen::Vector3<S> v1 = positions.row(faces(tri1, 1));
      const Eigen::Vector3<S> v2 = positions.row(faces(tri1, 2));
      const Eigen::Vector3<S> expectedP =
          baryCoords(0) * v0 + baryCoords(1) * v1 + baryCoords(2) * v2;
      EXPECT_NEAR((expectedP - p1).norm(), 0.0, detail::eps<S>(1e-6, 1e-14))
          << "Failed for: " << query.transpose() << " with baryCoords: " << baryCoords.transpose();
    }
  };

  checkQueries(makeBvh<S, 1>(sphere));
  checkQueries(makeBvh<S, kNativeLaneWidth<S>>(sphere));
}

TYPED_TEST(TriBvhTest, ClosestSurfacePoint_ExpectZeroDistFromMeshPoints) {
  using S = typename TestFixture::Type;

  auto sphere = generateSphere(kSphereParams<S>);

  const Eigen::MatrixX3<S> queryPoints = sphere.positions;
  const TriBvh<S> bvh{std::move(sphere.positions), std::move(sphere.triangles), detail::eps<S>()};

  for (uint32_t i = 0; i < queryPoints.rows(); ++i) {
    const Eigen::Vector3<S> query = queryPoints.row(i);
    const auto result = bvh.closestSurfacePoint(query);
    const auto& p1 = result.point;
    const auto& baryCoords = result.baryCoords.value();
    EXPECT_NEAR((query - p1).norm(), 0.0, detail::eps<S>()) << "Failed for: " << query.transpose();

    // Check barycentric coordinates
    int numOnes = 0;
    for (int j = 0; j < 3; ++j) {
      if (std::abs(baryCoords(j) - 1.0) < detail::eps<S>()) {
        numOnes++;
      } else {
        EXPECT_NEAR(baryCoords(j), 0.0, detail::eps<S>());
      }
    }
    EXPECT_EQ(numOnes, 1);
  }
}

TYPED_TEST(TriBvhTest, ClosestSurfacePoints_SameResultsAsIgl) {
  using S = typename TestFixture::Type;

  auto sphere = generateSphere(kSphereParams<S>);
  const Eigen::MatrixX<S> positions = sphere.positions;
  const Eigen::MatrixXi faces = sphere.triangles;

  const TriBvh<S> bvh{std::move(sphere.positions), std::move(sphere.triangles), 0.0};

  igl::AABB<Eigen::MatrixX<S>, 3> aabb;
  aabb.init(positions, faces);

  // Our query points will come from the vertices of a simple grid mesh.
  constexpr GridMeshParameters<S> kGridParams{10.0, 10.0, 1.0};
  const Eigen::MatrixX<S> queryPoints = generateGrid(kGridParams).positions;

  Eigen::MatrixX<S> closestPoints;
  Eigen::MatrixX<S> closestSquareDistances;
  Eigen::MatrixXi closestIndices;
  aabb.squared_distance(
      positions, faces, queryPoints, closestSquareDistances, closestIndices, closestPoints);

  Eigen::MatrixX<S> bvhClosestPoints;
  Eigen::MatrixX<S> bvhClosestSquareDistances;
  Eigen::MatrixXi bvhClosestIndices;
  closestSurfacePoints(
      bvh, queryPoints, bvhClosestSquareDistances, bvhClosestIndices, bvhClosestPoints);

  ASSERT_EQ(bvhClosestPoints.rows(), queryPoints.rows());
  ASSERT_EQ(closestPoints.rows(), queryPoints.rows());

  ASSERT_EQ(bvhClosestSquareDistances.rows(), queryPoints.rows());
  ASSERT_EQ(closestSquareDistances.rows(), queryPoints.rows());

  ASSERT_EQ(bvhClosestIndices.rows(), queryPoints.rows());
  ASSERT_EQ(closestIndices.rows(), queryPoints.rows());

  // Check that the multi-query API returns the same results as igl::AABB.
  for (uint32_t i = 0; i < queryPoints.rows(); ++i) {
    EXPECT_NEAR(
        bvhClosestSquareDistances(i), closestSquareDistances(i), detail::eps<S>(1e-4, 1e-8));
    EXPECT_NEAR(
        bvhClosestPoints.row(i).squaredNorm(),
        closestPoints.row(i).squaredNorm(),
        detail::eps<S>(1e-3, 1e-8));
  }

  for (uint32_t i = 0; i < queryPoints.rows(); ++i) {
    const auto result = bvh.closestSurfacePoint(queryPoints.row(i));
    EXPECT_NEAR(
        result.point.squaredNorm(),
        bvhClosestPoints.row(i).squaredNorm(),
        detail::eps<S>(1e-4, 1e-8));
    EXPECT_EQ(result.triangleIdx, bvhClosestIndices(i));
  }

  // Check that the callback-based API also returns the same results.
  std::vector<ClosestSurfacePointResult<S>> results(queryPoints.rows());
  closestSurfacePoints(bvh, queryPoints, [&results](const uint32_t idx, const auto& result) {
    results[idx] = result;
  });

  for (uint32_t i = 0; i < results.size(); ++i) {
    EXPECT_EIGEN_MATRIX_NEAR(results[i].point, Eigen::Vector3<S>(bvhClosestPoints.row(i)), 1e-8);
    EXPECT_EQ(results[i].triangleIdx, bvhClosestIndices(i));
  }
}

TYPED_TEST(TriBvhTest, RayQuery_AnyHits_Simple) {
  using S = typename TestFixture::Type;

  Eigen::MatrixX3<S> positions(3, 3);
  positions.row(0) = Eigen::Vector3<S>(-1., 0., -2.0);
  positions.row(1) = Eigen::Vector3<S>(+1., 0., -2.0);
  positions.row(2) = Eigen::Vector3<S>(+0., 1., -2.0);

  Eigen::MatrixX3i triangles(1, 3);
  triangles.row(0) = Eigen::Vector3i(0, 1, 2);

  TriBvh<S> bvh(std::move(positions), std::move(triangles), 0.1);

  // Ray shoots straight through the triangle, hits bottom edge.
  EXPECT_TRUE(bvh.anyHit(Ray3<S>({0, 0, 0}, {0, 0, -1}, 3.0, 0.0)));

  // Ray shoots straight slightly below the triangle, but doesn't hit.
  EXPECT_FALSE(bvh.anyHit(Ray3<S>({0, -0.1, 0}, {0, 0, -1}, 3.0, 0.0)));

  // Ray shoots straight through the triangle, but is shorter than the distance.
  EXPECT_FALSE(bvh.anyHit(Ray3<S>({0, 0, 0}, {0, 0, -1}, 1.0, 0.0)));

  // Ray shoots "backwards" with direction inverted and hits with minDist.
  EXPECT_TRUE(bvh.anyHit(Ray3<S>({0, 0, 0}, {0, 0, 1}, 0.0, -2.0)));
}

} // namespace
} // namespace axel::test
