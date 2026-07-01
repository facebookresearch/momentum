/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/math/support_polygon.h>

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>

using namespace momentum;

namespace {

bool supportHullContains(const std::vector<Vector2f>& hull, const Vector2f& point) {
  for (const Vector2f& vertex : hull) {
    if ((vertex - point).norm() < 1e-4f) {
      return true;
    }
  }
  return false;
}

} // namespace

TEST(SupportPolygonTest, ComputeSupportPolygonFromWorldPointsReturnsContactConvexHull) {
  const std::vector<Vector3f> supportPoints = {
      Vector3f(0.0f, -1.0f, 0.0f),
      Vector3f(4.0f, 2.0f, 0.0f),
      Vector3f(0.0f, 0.5f, 4.0f),
      Vector3f(1.0f, 12.0f, 1.0f),
  };

  const std::vector<Vector2f> hull = computeSupportPolygonFromWorldPoints(supportPoints);

  ASSERT_EQ(hull.size(), 3u);
  EXPECT_TRUE(supportHullContains(hull, Vector2f(0.0f, 0.0f)));
  EXPECT_TRUE(supportHullContains(hull, Vector2f(4.0f, 0.0f)));
  EXPECT_TRUE(supportHullContains(hull, Vector2f(0.0f, 4.0f)));
}

TEST(SupportPolygonTest, SupportPlaneProjectsCustomPlaneCoordinates) {
  const SupportPlane supportPlane(Vector3f::UnitX(), 2.0f, Vector3f::UnitY());

  EXPECT_TRUE(supportPlane.origin().isApprox(Vector3f(2.0f, 0.0f, 0.0f)));
  EXPECT_NEAR(supportPlane.signedDistance(Vector3f(3.0f, 4.0f, -5.0f)), 1.0f, 1e-5f);

  const Vector3f point(2.0f, 4.0f, -5.0f);
  const Vector2f coordinates = supportPlane.coordinates(point);
  EXPECT_TRUE(coordinates.isApprox(Vector2f(4.0f, 5.0f)));
  EXPECT_TRUE(supportPlane.pointFromCoordinates(coordinates).isApprox(point));

  const std::vector<Vector3f> supportPoints = {
      Vector3f(2.0f, 0.0f, 0.0f),
      Vector3f(2.0f, 3.0f, 0.0f),
      Vector3f(2.0f, 0.0f, -3.0f),
      Vector3f(5.0f, 1.0f, -1.0f),
  };
  const std::vector<Vector2f> hull =
      computeSupportPolygonFromWorldPoints(supportPoints, supportPlane);

  ASSERT_EQ(hull.size(), 3u);
  EXPECT_TRUE(supportHullContains(hull, Vector2f(0.0f, 0.0f)));
  EXPECT_TRUE(supportHullContains(hull, Vector2f(3.0f, 0.0f)));
  EXPECT_TRUE(supportHullContains(hull, Vector2f(0.0f, 3.0f)));
}

TEST(SupportPolygonTest, ComputeConvexHull2dReturnsDegenerateHulls) {
  EXPECT_TRUE(computeConvexHull2d({}).empty());

  const std::vector<Vector2f> onePoint = {Vector2f(1.0f, 2.0f)};
  const std::vector<Vector2f> onePointHull = computeConvexHull2d(onePoint);
  ASSERT_EQ(onePointHull.size(), 1u);
  EXPECT_TRUE(supportHullContains(onePointHull, Vector2f(1.0f, 2.0f)));

  const std::vector<Vector2f> collinearPoints = {
      Vector2f(0.0f, 0.0f),
      Vector2f(2.0f, 0.0f),
      Vector2f(4.0f, 0.0f),
  };
  const std::vector<Vector2f> collinearHull = computeConvexHull2d(collinearPoints);
  ASSERT_EQ(collinearHull.size(), 2u);
  EXPECT_TRUE(supportHullContains(collinearHull, Vector2f(0.0f, 0.0f)));
  EXPECT_TRUE(supportHullContains(collinearHull, Vector2f(4.0f, 0.0f)));
  EXPECT_FALSE(supportHullContains(collinearHull, Vector2f(2.0f, 0.0f)));
}

TEST(SupportPolygonTest, ComputeConvexHull2dRemovesDuplicatePoints) {
  const std::vector<Vector2f> points = {
      Vector2f(0.0f, 0.0f),
      Vector2f(0.0f, 0.0f),
      Vector2f(1.0f, 0.0f),
      Vector2f(0.0f, 1.0f),
  };

  const std::vector<Vector2f> hull = computeConvexHull2d(points);

  ASSERT_EQ(hull.size(), 3u);
  EXPECT_TRUE(supportHullContains(hull, Vector2f(0.0f, 0.0f)));
  EXPECT_TRUE(supportHullContains(hull, Vector2f(1.0f, 0.0f)));
  EXPECT_TRUE(supportHullContains(hull, Vector2f(0.0f, 1.0f)));
}

TEST(SupportPolygonTest, ComputeConvexHull2dRejectsNonFinitePoints) {
  const auto nanValue = std::bit_cast<float>(std::uint32_t{0x7fc00000});
  const auto infValue = std::bit_cast<float>(std::uint32_t{0x7f800000});
  const std::vector<Vector2f> points = {
      Vector2f(0.0f, 0.0f),
      Vector2f(nanValue, 1.0f),
  };

  EXPECT_THROW(static_cast<void>(computeConvexHull2d(points)), std::exception);
  const std::vector<Vector3f> worldPoints = {Vector3f(infValue, 0.0f, 0.0f)};
  EXPECT_THROW(
      static_cast<void>(computeSupportPolygonFromWorldPoints(worldPoints)), std::exception);

  const std::vector<Vector3f> nonFiniteYPoints = {Vector3f(1.0f, infValue, 2.0f)};
  EXPECT_THROW(
      static_cast<void>(computeSupportPolygonFromWorldPoints(nonFiniteYPoints)), std::exception);
}
