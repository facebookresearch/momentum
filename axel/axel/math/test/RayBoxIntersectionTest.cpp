/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "axel/math/RayBoxIntersection.h"
#include "axel/test/Helper.h"

#include <perception/test_helpers/EigenChecks.h>

namespace axel::test {
namespace {

template <typename T>
struct RayBoxIntersectionTest : testing::Test {
  using Type = T;
};

using ScalarTypes = testing::Types<float, double>;
TYPED_TEST_SUITE(RayBoxIntersectionTest, ScalarTypes);

TYPED_TEST(RayBoxIntersectionTest, BasicHit) {
  using S = typename TestFixture::Type;

  // Create a simple ray shooting towards +Z
  Ray3<S> ray;
  ray.origin = Eigen::Vector3<S>(0, 0, 0);
  ray.direction = Eigen::Vector3<S>(0, 0, 1);
  ray.minT = static_cast<S>(0.1);
  ray.maxT = static_cast<S>(10.0);

  // Create a box that the ray should hit
  Eigen::Vector3<S> boxMin(-1, -1, 2);
  Eigen::Vector3<S> boxMax(1, 1, 3);

  auto result = intersectRayBox(ray, boxMin, boxMax);

  EXPECT_TRUE(result.intersects);
  EXPECT_NEAR(result.tMin, static_cast<S>(2.0), static_cast<S>(1e-6));
  EXPECT_NEAR(result.tMax, static_cast<S>(3.0), static_cast<S>(1e-6));
}

TYPED_TEST(RayBoxIntersectionTest, BasicMiss) {
  using S = typename TestFixture::Type;

  // Create a ray shooting towards +Z
  Ray3<S> ray;
  ray.origin = Eigen::Vector3<S>(0, 0, 0);
  ray.direction = Eigen::Vector3<S>(0, 0, 1);
  ray.minT = static_cast<S>(0.1);
  ray.maxT = static_cast<S>(10.0);

  // Create a box that the ray should miss (offset in X)
  Eigen::Vector3<S> boxMin(2, -1, 2);
  Eigen::Vector3<S> boxMax(3, 1, 3);

  auto result = intersectRayBox(ray, boxMin, boxMax);

  EXPECT_FALSE(result.intersects);
}

TYPED_TEST(RayBoxIntersectionTest, NegativeMaxT) {
  using S = typename TestFixture::Type;

  // Test the failing case from TriBvhTest.cpp:
  // Ray3<S>({0, 0, 0}, {0, 0, 1}, 0.0, -2.0)
  // This creates a ray with origin=(0,0,0), direction=(0,0,1), maxT=0.0, minT=-2.0
  Ray3<S> ray;
  ray.origin = Eigen::Vector3<S>(0, 0, 0);
  ray.direction = Eigen::Vector3<S>(0, 0, 1);
  ray.maxT = static_cast<S>(0.0); // 3rd parameter
  ray.minT = static_cast<S>(-2.0); // 4th parameter

  // Create a box at z=-2 (same as the test triangle)
  // Bounding Box: [-1  0 -2] to [ 1  1 -2]
  Eigen::Vector3<S> boxMin(-1, 0, -2);
  Eigen::Vector3<S> boxMax(1, 1, -2);

  auto result = intersectRayBox(ray, boxMin, boxMax);

  EXPECT_TRUE(result.intersects) << "Ray with negative maxT should intersect box at z=-2";
  EXPECT_NEAR(result.tMin, static_cast<S>(-2.0), static_cast<S>(1e-6));
  EXPECT_NEAR(result.tMax, static_cast<S>(-2.0), static_cast<S>(1e-6));
}

TYPED_TEST(RayBoxIntersectionTest, RayTooShort) {
  using S = typename TestFixture::Type;

  // Create a ray that stops before reaching the box
  Ray3<S> ray;
  ray.origin = Eigen::Vector3<S>(0, 0, 0);
  ray.direction = Eigen::Vector3<S>(0, 0, 1);
  ray.minT = static_cast<S>(0.1);
  ray.maxT = static_cast<S>(1.0); // Ray stops at t=1.0

  // Create a box at z=2-3, which requires t=2.0 to reach
  Eigen::Vector3<S> boxMin(-1, -1, 2);
  Eigen::Vector3<S> boxMax(1, 1, 3);

  auto result = intersectRayBox(ray, boxMin, boxMax);

  EXPECT_FALSE(result.intersects) << "Ray should not intersect box beyond its maxT";
}

TYPED_TEST(RayBoxIntersectionTest, RayStartsTooLate) {
  using S = typename TestFixture::Type;

  // Create a ray that starts after passing through the box
  Ray3<S> ray;
  ray.origin = Eigen::Vector3<S>(0, 0, 0);
  ray.direction = Eigen::Vector3<S>(0, 0, 1);
  ray.minT = static_cast<S>(5.0); // Ray starts at t=5.0
  ray.maxT = static_cast<S>(10.0);

  // Create a box at z=2-3, which is intersected at t=2.0-3.0
  Eigen::Vector3<S> boxMin(-1, -1, 2);
  Eigen::Vector3<S> boxMax(1, 1, 3);

  auto result = intersectRayBox(ray, boxMin, boxMax);

  EXPECT_FALSE(result.intersects) << "Ray should not intersect box before its minT";
}

TYPED_TEST(RayBoxIntersectionTest, RayOriginInsideBox) {
  using S = typename TestFixture::Type;

  // Create a ray starting inside the box
  Ray3<S> ray;
  ray.origin = Eigen::Vector3<S>(0, 0, 0);
  ray.direction = Eigen::Vector3<S>(0, 0, 1);
  ray.minT = static_cast<S>(-1.0);
  ray.maxT = static_cast<S>(1.0);

  // Create a box containing the origin
  Eigen::Vector3<S> boxMin(-2, -2, -2);
  Eigen::Vector3<S> boxMax(2, 2, 2);

  auto result = intersectRayBox(ray, boxMin, boxMax);

  EXPECT_TRUE(result.intersects) << "Ray starting inside box should intersect";
  EXPECT_LE(result.tMin, static_cast<S>(0.0)) << "Entry should be at or before ray origin";
  EXPECT_GE(result.tMax, static_cast<S>(0.0)) << "Exit should be at or after ray origin";
}

TYPED_TEST(RayBoxIntersectionTest, NegativeDirection) {
  using S = typename TestFixture::Type;

  // Create a ray shooting towards -Z
  Ray3<S> ray;
  ray.origin = Eigen::Vector3<S>(0, 0, 0);
  ray.direction = Eigen::Vector3<S>(0, 0, -1);
  ray.minT = static_cast<S>(0.1);
  ray.maxT = static_cast<S>(10.0);

  // Create a box in the negative Z direction
  Eigen::Vector3<S> boxMin(-1, -1, -3);
  Eigen::Vector3<S> boxMax(1, 1, -2);

  auto result = intersectRayBox(ray, boxMin, boxMax);

  EXPECT_TRUE(result.intersects);
  EXPECT_NEAR(result.tMin, static_cast<S>(2.0), static_cast<S>(1e-6));
  EXPECT_NEAR(result.tMax, static_cast<S>(3.0), static_cast<S>(1e-6));
}

TYPED_TEST(RayBoxIntersectionTest, CompareWithBoundingBoxMethod) {
  using S = typename TestFixture::Type;

  // Test consistency with the BoundingBox.cpp intersects() method
  Ray3<S> ray;
  ray.origin = Eigen::Vector3<S>(1, 2, 3);
  ray.direction = Eigen::Vector3<S>(-1, -1, -1).normalized();
  ray.minT = static_cast<S>(0.0);
  ray.maxT = static_cast<S>(100.0);

  Eigen::Vector3<S> boxMin(0, 0, 0);
  Eigen::Vector3<S> boxMax(5, 5, 5);

  auto result = intersectRayBox(ray, boxMin, boxMax);

  // Create equivalent BoundingBox and test
  BoundingBox<S> bbox(boxMin, boxMax);
  bool bboxIntersects = bbox.intersects(ray.origin, ray.direction);

  EXPECT_EQ(result.intersects, bboxIntersects)
      << "Our function should agree with BoundingBox.intersects() for basic cases";
}

} // namespace
} // namespace axel::test
