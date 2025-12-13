/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "axel/TriBvhEmbree.h"
#include "axel/common/Constants.h"
#include "axel/test/Helper.h"

#include <igl/AABB.h>
#include <perception/test_helpers/EigenChecks.h>
#include <test_helpers/FollyMatchers.h>

namespace axel::test {
namespace {

template <typename S>
constexpr SphereMeshParameters<S> kSphereParams{3.0, 64, 64};

using ScalarTypes = testing::Types<float, double>;

template <typename T>
struct TriBvhEmbreeTest : testing::Test {
  using Type = T;
  static constexpr T kEdgeLength = 3.0;
};

TYPED_TEST_SUITE(TriBvhEmbreeTest, ScalarTypes);

TYPED_TEST(TriBvhEmbreeTest, rayHitsAny) {
  using S = typename TestFixture::Type;

  auto sphere = generateSphere(kSphereParams<S>);
  const Eigen::Index triangleCount{sphere.triangles.rows()};
  const TriBvhEmbree<S> bvh{std::move(sphere.positions), std::move(sphere.triangles)};

  EXPECT_EQ(bvh.getPrimitiveCount(), triangleCount);

  // Hits.
  EXPECT_TRUE(bvh.anyHit(Ray3<S>(Eigen::Vector3<S>(-10, 0, 0), Eigen::Vector3<S>(1, 0, 0))));

  // Ignores hits too far.
  EXPECT_FALSE(bvh.anyHit(Ray3<S>(Eigen::Vector3<S>(-10, 0, 0), Eigen::Vector3<S>(1, 0, 0), 6.0)));

  // No hits despite bvh query not empty.
  EXPECT_FALSE(bvh.anyHit(Ray3<S>(Eigen::Vector3<S>(0, 0, 3.0), Eigen::Vector3<S>(0, 0, 1))));
}

TYPED_TEST(TriBvhEmbreeTest, rayHitsClosest) {
  using S = typename TestFixture::Type;

  auto sphere = generateSphere(kSphereParams<S>);
  const TriBvhEmbree<S> bvh{std::move(sphere.positions), std::move(sphere.triangles)};

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

TYPED_TEST(TriBvhEmbreeTest, ClosestSurfacePoint_SameResultsAsIgl) {
  using S = typename TestFixture::Type;

  auto sphere = generateSphere(kSphereParams<S>);
  const Eigen::MatrixX<S> positions = sphere.positions;
  const Eigen::MatrixXi faces = sphere.triangles;

  const TriBvhEmbree<S> bvh{std::move(sphere.positions), std::move(sphere.triangles)};

  igl::AABB<Eigen::MatrixX<S>, 3> aabb;
  aabb.init(positions, faces);

  // Our query points will come from the vertices of a simple grid mesh.
  constexpr GridMeshParameters<S> kGridParams{10.0, 10.0, 1.0};
  const Eigen::MatrixX<S> queryPoints =
      generateGrid(kGridParams, Eigen::Vector3<S>(0, 0, 5)).positions;

  for (uint32_t i = 0; i < queryPoints.rows(); ++i) {
    const Eigen::Vector3<S> query = queryPoints.row(i);

    int32_t tri0 = 0;
    Eigen::RowVector3<S> p0{};
    aabb.squared_distance(positions, faces, query, tri0, p0);

    const auto result = bvh.closestSurfacePoint(query);
    const auto& p1 = result.point;

    EXPECT_NEAR(
        (query - Eigen::Vector3<S>(p0)).norm(), (query - p1).norm(), detail::eps<S>(1e-6, 1e-10))
        << "Failed for: " << query.transpose();
  }
}

} // namespace
} // namespace axel::test
