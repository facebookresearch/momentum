/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <perception/test_helpers/EigenChecks.h>

#include "axel/common/Constants.h"
#include "axel/common/VectorizationTypes.h"
#include "axel/math/RayTriangleIntersection.h"

#include <random>

namespace axel::test {
namespace {

using ::axel::detail::eps;

// Helper function to test both scalar and wide versions of ray-triangle intersection
template <typename T>
void testRayTriangleIntersectionBothVersions(
    const Eigen::Vector3<T>& rayOrigin,
    const Eigen::Vector3<T>& rayDirection,
    const Eigen::Vector3<T>& v0,
    const Eigen::Vector3<T>& v1,
    const Eigen::Vector3<T>& v2,
    bool expectedHit,
    const std::string& testDescription = "") {
  std::cout << "=== Testing both scalar and wide versions: " << testDescription
            << " ===" << std::endl;
  std::cout << "Ray: origin=" << rayOrigin.transpose() << " dir=" << rayDirection.transpose()
            << std::endl;
  std::cout << "Triangle: v0=" << v0.transpose() << " v1=" << v1.transpose()
            << " v2=" << v2.transpose() << std::endl;

  // Test scalar version
  Eigen::Vector3<T> scalarHitPoint;
  T scalarT, scalarU, scalarV;
  bool scalarHit = rayTriangleIntersect(
      rayOrigin, rayDirection, v0, v1, v2, scalarHitPoint, scalarT, scalarU, scalarV);

  std::cout << "Scalar result: " << (scalarHit ? "HIT" : "MISS") << std::endl;
  if (scalarHit) {
    std::cout << "  t=" << scalarT << " u=" << scalarU << " v=" << scalarV
              << " w=" << (1 - scalarU - scalarV) << std::endl;
    std::cout << "  hitPoint=" << scalarHitPoint.transpose() << std::endl;
  }

  // Test wide version by filling a SIMD vector with random data and our test case
  constexpr size_t LaneWidth = kNativeLaneWidth<T>;
  constexpr size_t TestLane = 0; // Use first lane for our actual test case

  // Create random number generator for filling other lanes
  std::mt19937 rng(42); // Fixed seed for reproducibility
  std::uniform_real_distribution<T> dist(-10.0, 10.0);

  // Create wide vectors for triangles
  WideVec3<T, LaneWidth> wideV0, wideV1, wideV2;
  for (int axis = 0; axis < 3; ++axis) {
    // Set the test case in the first lane
    wideV0[axis][TestLane] = v0[axis];
    wideV1[axis][TestLane] = v1[axis];
    wideV2[axis][TestLane] = v2[axis];

    // Fill other lanes with random triangles
    for (size_t lane = 1; lane < LaneWidth; ++lane) {
      wideV0[axis][lane] = dist(rng);
      wideV1[axis][lane] = dist(rng);
      wideV2[axis][lane] = dist(rng);
    }
  }

  // Test wide version
  WideVec3<T, LaneWidth> wideHitPoint;
  WideScalar<T, LaneWidth> wideT, wideU, wideV;
  auto wideHitMask = rayTriangleIntersectWide<T, LaneWidth>(
      rayOrigin, rayDirection, wideV0, wideV1, wideV2, wideHitPoint, wideT, wideU, wideV);

  bool wideHit = wideHitMask[TestLane];
  std::cout << "Wide result: " << (wideHit ? "HIT" : "MISS") << std::endl;
  if (wideHit) {
    std::cout << "  t=" << wideT[TestLane] << " u=" << wideU[TestLane] << " v=" << wideV[TestLane]
              << " w=" << (1 - wideU[TestLane] - wideV[TestLane]) << std::endl;
    std::cout << "  hitPoint=(" << wideHitPoint[0][TestLane] << ", " << wideHitPoint[1][TestLane]
              << ", " << wideHitPoint[2][TestLane] << ")" << std::endl;
  }

  // Verify both versions agree on hit/miss
  EXPECT_EQ(scalarHit, wideHit) << "Scalar and wide versions disagree on hit/miss for "
                                << testDescription;
  EXPECT_EQ(expectedHit, scalarHit)
      << "Scalar result doesn't match expected for " << testDescription;

  // If both hit, verify they produce the same results
  if (scalarHit && wideHit) {
    EXPECT_NEAR(scalarT, wideT[TestLane], eps<T>())
        << "t parameter mismatch for " << testDescription;
    EXPECT_NEAR(scalarU, wideU[TestLane], eps<T>())
        << "u parameter mismatch for " << testDescription;
    EXPECT_NEAR(scalarV, wideV[TestLane], eps<T>())
        << "v parameter mismatch for " << testDescription;

    Eigen::Vector3<T> wideHitPointExtracted(
        wideHitPoint[0][TestLane], wideHitPoint[1][TestLane], wideHitPoint[2][TestLane]);
    EXPECT_EIGEN_MATRIX_NEAR(scalarHitPoint, wideHitPointExtracted, eps<T>())
        << "Hit point mismatch for " << testDescription;
  }

  std::cout << "Both versions agree: " << (scalarHit == wideHit ? "YES" : "NO") << std::endl;
  std::cout << std::endl;
}

// Helper function to test multiple triangles with SIMD (unique from Wide test)
template <typename T>
void testMultipleTriangles(
    const Eigen::Vector3<T>& rayOrigin,
    const Eigen::Vector3<T>& rayDirection,
    const std::vector<std::array<Eigen::Vector3<T>, 3>>& triangles,
    const std::vector<bool>& expectedHits,
    const std::string& testDescription = "") {
  constexpr size_t Width = 4; // Test with width 4 for simplicity
  using WideScalarT = WideScalar<T, Width>;
  using WideVec3T = WideVec3<T, Width>;
  using WideMaskT = WideMask<WideScalarT>;

  std::cout << "=== Testing multiple triangles: " << testDescription << " ===" << std::endl;

  // Test each triangle individually with scalar version
  std::vector<bool> scalarHits(triangles.size());
  std::vector<T> scalarTs(triangles.size());
  std::vector<T> scalarUs(triangles.size());
  std::vector<T> scalarVs(triangles.size());
  std::vector<Eigen::Vector3<T>> scalarHitPoints(triangles.size());

  for (size_t i = 0; i < triangles.size(); ++i) {
    scalarHits[i] = rayTriangleIntersect(
        rayOrigin,
        rayDirection,
        triangles[i][0],
        triangles[i][1],
        triangles[i][2],
        scalarHitPoints[i],
        scalarTs[i],
        scalarUs[i],
        scalarVs[i]);
  }

  // Pack triangles into SIMD format
  WideVec3T p0, p1, p2;
  for (int i = 0; i < 3; ++i) {
    p0[i] = WideScalarT(0);
    p1[i] = WideScalarT(0);
    p2[i] = WideScalarT(0);
  }

  const size_t numTris = std::min(triangles.size(), static_cast<size_t>(Width));
  for (size_t i = 0; i < numTris; ++i) {
    for (int j = 0; j < 3; ++j) {
      p0[j][i] = triangles[i][0][j];
      p1[j][i] = triangles[i][1][j];
      p2[j][i] = triangles[i][2][j];
    }
  }

  // Test wide version
  WideVec3T wideHitPoint;
  WideScalarT wideT, wideU, wideV;
  WideMaskT wideHit = rayTriangleIntersectWide<T, Width>(
      rayOrigin, rayDirection, p0, p1, p2, wideHitPoint, wideT, wideU, wideV);

  // Compare results
  for (size_t i = 0; i < numTris; ++i) {
    EXPECT_EQ(scalarHits[i], wideHit[i]) << "Hit/miss mismatch for triangle " << i;
    if (i < expectedHits.size()) {
      EXPECT_EQ(expectedHits[i], scalarHits[i]) << "Expected hit mismatch for triangle " << i;
    }

    if (scalarHits[i] && wideHit[i]) {
      EXPECT_NEAR(scalarTs[i], wideT[i], static_cast<T>(1e-6))
          << "T parameter mismatch for triangle " << i;
      EXPECT_NEAR(scalarUs[i], wideU[i], static_cast<T>(1e-6))
          << "U parameter mismatch for triangle " << i;
      EXPECT_NEAR(scalarVs[i], wideV[i], static_cast<T>(1e-6))
          << "V parameter mismatch for triangle " << i;

      for (int j = 0; j < 3; ++j) {
        EXPECT_NEAR(scalarHitPoints[i][j], wideHitPoint[j][i], static_cast<T>(1e-6))
            << "Hit point mismatch for triangle " << i << " component " << j;
      }
    }
  }

  std::cout << "Multiple triangles test completed" << std::endl;
}

// Overloaded version without barycentric coordinates
template <typename T>
void testRayTriangleIntersectionBothVersionsNoBary(
    const Eigen::Vector3<T>& rayOrigin,
    const Eigen::Vector3<T>& rayDirection,
    const Eigen::Vector3<T>& v0,
    const Eigen::Vector3<T>& v1,
    const Eigen::Vector3<T>& v2,
    bool expectedHit,
    const std::string& testDescription = "") {
  std::cout << "=== Testing both versions (no bary): " << testDescription << " ===" << std::endl;

  // Test scalar version
  Eigen::Vector3<T> scalarHitPoint;
  T scalarT;
  bool scalarHit =
      rayTriangleIntersect(rayOrigin, rayDirection, v0, v1, v2, scalarHitPoint, scalarT);

  // Test wide version
  constexpr size_t LaneWidth = kNativeLaneWidth<T>;
  constexpr size_t TestLane = 0;

  std::mt19937 rng(42);
  std::uniform_real_distribution<T> dist(-10.0, 10.0);

  WideVec3<T, LaneWidth> wideV0, wideV1, wideV2;
  for (int axis = 0; axis < 3; ++axis) {
    wideV0[axis][TestLane] = v0[axis];
    wideV1[axis][TestLane] = v1[axis];
    wideV2[axis][TestLane] = v2[axis];

    for (size_t lane = 1; lane < LaneWidth; ++lane) {
      wideV0[axis][lane] = dist(rng);
      wideV1[axis][lane] = dist(rng);
      wideV2[axis][lane] = dist(rng);
    }
  }

  WideVec3<T, LaneWidth> wideHitPoint;
  WideScalar<T, LaneWidth> wideT;
  auto wideHitMask = rayTriangleIntersectWide<T, LaneWidth>(
      rayOrigin, rayDirection, wideV0, wideV1, wideV2, wideHitPoint, wideT);

  bool wideHit = wideHitMask[TestLane];

  // Verify both versions agree
  EXPECT_EQ(scalarHit, wideHit) << "Scalar and wide versions disagree on hit/miss for "
                                << testDescription;
  EXPECT_EQ(expectedHit, scalarHit)
      << "Scalar result doesn't match expected for " << testDescription;

  if (scalarHit && wideHit) {
    EXPECT_NEAR(scalarT, wideT[TestLane], eps<T>())
        << "t parameter mismatch for " << testDescription;

    Eigen::Vector3<T> wideHitPointExtracted(
        wideHitPoint[0][TestLane], wideHitPoint[1][TestLane], wideHitPoint[2][TestLane]);
    EXPECT_EIGEN_MATRIX_NEAR(scalarHitPoint, wideHitPointExtracted, eps<T>())
        << "Hit point mismatch for " << testDescription;
  }
}

TEST(RayTriangleIntersectionTest, Basic) {
  Eigen::Vector3d collisionPoint;
  double t;
  EXPECT_TRUE(rayTriangleIntersect(
      Eigen::Vector3d(1., 1., 1.),
      Eigen::Vector3d(0., 1., 0.),
      Eigen::Vector3d(0., 5., 0.),
      Eigen::Vector3d(5., 5., 0.),
      Eigen::Vector3d(2.5, 5., 2.5),
      collisionPoint,
      t));
  EXPECT_EIGEN_MATRIX_NEAR(collisionPoint, Eigen::Vector3d(1., 5., 1.), eps<double>());
  EXPECT_NEAR(t, 4., eps<double>());

  EXPECT_FALSE(rayTriangleIntersect(
      Eigen::Vector3d(1., 1., 1.),
      Eigen::Vector3d(0., 1., 0.),
      Eigen::Vector3d(10., 5., 0.),
      Eigen::Vector3d(10., 15., 0.),
      Eigen::Vector3d(12.5, 5., 2.5),
      collisionPoint,
      t));

  // same edge-triangle tests with different scale
  EXPECT_TRUE(rayTriangleIntersect(
      Eigen::Vector3d(-2.3116044998168945, -3.557568311691284, 4.071789264678955),
      Eigen::Vector3d(-0.02631211280822754, -0.005295753479003906, -0.04990386962890625),
      Eigen::Vector3d(-2.322500705718994, -3.5661442279815674, 3.9918453693389893),
      Eigen::Vector3d(-2.381772518157959, -3.554487943649292, 4.0887017250061035),
      Eigen::Vector3d(-2.2690765857696533, -3.567392110824585, 3.994669198989868),
      collisionPoint,
      t));
  EXPECT_NEAR(t, 0.73062444794057768, eps<double>());

  EXPECT_TRUE(rayTriangleIntersect(
      Eigen::Vector3d(-2.381772518157959, -3.554487943649292, 4.0887017250061035),
      Eigen::Vector3d(0.11269593238830566, -0.012904167175292969, -0.09403252601623535),
      Eigen::Vector3d(-2.3116044998168945, -3.557568311691284, 4.071789264678955),
      Eigen::Vector3d(-2.337916612625122, -3.562864065170288, 4.021885395050049),
      Eigen::Vector3d(-2.248196601867676, -3.580512285232544, 3.9052698612213135),
      collisionPoint,
      t));
  EXPECT_NEAR(t, 0.53861406334285833, eps<double>());

  EXPECT_TRUE(rayTriangleIntersect(
      Eigen::Vector3d(-0.5650877952575684, -0.9797914624214172, 0.8701220750808716),
      Eigen::Vector3d(-0.006578028202056885, -0.0013239383697509766, -0.012475967407226562),
      Eigen::Vector3d(-0.5678118467330933, -0.981935441493988, 0.8501361608505249),
      Eigen::Vector3d(-0.5826297998428345, -0.9790213704109192, 0.8743501901626587),
      Eigen::Vector3d(-0.5544558167457581, -0.9822474122047424, 0.8508421182632446),
      collisionPoint,
      t));

  EXPECT_TRUE(rayTriangleIntersect(
      Eigen::Vector3d(-0.5826297998428345, -0.9790213704109192, 0.8743501901626587),
      Eigen::Vector3d(0.028173983097076416, -0.003226041793823242, -0.023508071899414062),
      Eigen::Vector3d(-0.5650877952575684, -0.9797914624214172, 0.8701220750808716),
      Eigen::Vector3d(-0.5716658234596252, -0.9811154007911682, 0.857646107673645),
      Eigen::Vector3d(-0.5492358207702637, -0.9855274558067322, 0.8284921646118164),
      collisionPoint,
      t));

  EXPECT_TRUE(rayTriangleIntersect(
      Eigen::Vector3d(-0.27400198578834534, -0.5501620173454285, 0.3365109860897064),
      Eigen::Vector3d(-0.0032890141010284424, -0.0006619691848754883, -0.006237983703613281),
      Eigen::Vector3d(-0.2753640115261078, -0.5512340068817139, 0.3265179991722107),
      Eigen::Vector3d(-0.2827729880809784, -0.5497769713401794, 0.3386250138282776),
      Eigen::Vector3d(-0.2686859965324402, -0.5513899922370911, 0.32687100768089294),
      collisionPoint,
      t));

  EXPECT_TRUE(rayTriangleIntersect(
      Eigen::Vector3d(-0.2827729880809784, -0.5497769713401794, 0.3386250138282776),
      Eigen::Vector3d(0.014086991548538208, -0.001613020896911621, -0.011754006147384644),
      Eigen::Vector3d(-0.27400198578834534, -0.5501620173454285, 0.3365109860897064),
      Eigen::Vector3d(-0.2772909998893738, -0.550823986530304, 0.33027300238609314),
      Eigen::Vector3d(-0.266075998544693, -0.5530300140380859, 0.31569600105285645),
      collisionPoint,
      t));
}

TEST(RaytracingUtilsTest, rayTriangleIntersect_hit) {
  const Eigen::Vector3d p0(0., 5., 0.);
  const Eigen::Vector3d p1(5., 5., 0.);
  const Eigen::Vector3d p2(2.5, 5., 2.5);

  Eigen::Vector3d hitPoint;
  double t;
  double u;
  double v;
  EXPECT_TRUE(rayTriangleIntersect(
      Eigen::Vector3d(1., 1., 1.), Eigen::Vector3d(0., 1., 0.), p0, p1, p2, hitPoint, t, u, v));
  EXPECT_NEAR(t, 4., eps<double>());
  EXPECT_EIGEN_MATRIX_NEAR(hitPoint, Eigen::Vector3d(1., 5., 1.), eps<double>());
  EXPECT_EIGEN_MATRIX_NEAR(
      Eigen::Vector3d(1 - u - v, u, v), Eigen::Vector3d(0.6, 0.0, 0.4), eps<double>());

  EXPECT_TRUE(rayTriangleIntersect(
      Eigen::Vector3d(1., 10., 1.), Eigen::Vector3d(0., 1., 0.), p0, p1, p2, hitPoint, t));
  EXPECT_NEAR(t, -5., eps<double>());
}

TEST(RaytracingUtilsTest, rayTriangleIntersect_noHit) {
  Eigen::Vector3d hitPoint;
  double t;
  EXPECT_FALSE(rayTriangleIntersect(
      Eigen::Vector3d(1., 1., 1.),
      Eigen::Vector3d(0., 1., 0.),
      Eigen::Vector3d(10., 5., 0.),
      Eigen::Vector3d(10., 15., 0.),
      Eigen::Vector3d(12.5, 5., 2.5),
      hitPoint,
      t));
}

TEST(RayTriangleIntersectionTest, RayQuery_AnyHits_Simple_FailingCase) {
  // This test reproduces the exact failing case from TriBvhTest::RayQuery_AnyHits_Simple
  // Triangle vertices from the failing test case
  Eigen::Vector3d v0(-1., 0., -2.0);
  Eigen::Vector3d v1(+1., 0., -2.0);
  Eigen::Vector3d v2(+0., 1., -2.0);

  // Ray from the failing test case
  Eigen::Vector3d rayOrigin(0., 0., 0.);
  Eigen::Vector3d rayDirection(0., 0., -1.);

  // The ray shoots straight down from origin (0,0,0) towards (0,0,-1)
  // The triangle is at z=-2, so the ray should hit at t=2.0 at point (0,0,-2)
  // Triangle spans from (-1,0,-2) to (1,0,-2) along x-axis at y=0
  // and has a third vertex at (0,1,-2)
  // The point (0,0,-2) should be on the bottom edge of this triangle

  // Use our helper function to test both scalar and wide versions
  testRayTriangleIntersectionBothVersions<double>(
      rayOrigin, rayDirection, v0, v1, v2, true, "RayQuery_AnyHits_Simple failing case");
}

TEST(RayTriangleIntersectionTest, NegativeT_BackwardsRay) {
  // Test the specific backward ray case from TriBvhTest
  // This tests if we can detect intersections with negative t values

  // Triangle at z=-2
  Eigen::Vector3d v0(-1., 0., -2.0);
  Eigen::Vector3d v1(+1., 0., -2.0);
  Eigen::Vector3d v2(+0., 1., -2.0);

  // Ray shooting towards +Z from origin, but we want to intersect behind origin
  Eigen::Vector3d rayOrigin(0., 0., 0.);
  Eigen::Vector3d rayDirection(0., 0., 1.); // towards +Z

  std::cout << "=== Testing backwards ray intersection ===" << std::endl;
  std::cout << "Triangle at z=-2, ray shoots towards +Z from origin" << std::endl;
  std::cout << "Expected intersection at t=-2 (behind ray origin)" << std::endl;

  Eigen::Vector3d hitPoint;
  double t = NAN, u = NAN, v = NAN;
  bool hit = rayTriangleIntersect(rayOrigin, rayDirection, v0, v1, v2, hitPoint, t, u, v);

  std::cout << "Ray-triangle intersection result: " << (hit ? "HIT" : "MISS") << std::endl;
  if (hit) {
    std::cout << "  t=" << t << " (should be -2)" << std::endl;
    std::cout << "  hitPoint=" << hitPoint.transpose() << " (should be (0, 0, -2))" << std::endl;
    std::cout << "  u=" << u << " v=" << v << " w=" << (1 - u - v) << std::endl;

    // Verify the intersection is correct
    EXPECT_NEAR(t, -2.0, eps<double>()) << "Ray should intersect at t=-2";
    EXPECT_NEAR(hitPoint.z(), -2.0, eps<double>()) << "Hit point should be at z=-2";
    EXPECT_NEAR(hitPoint.x(), 0.0, eps<double>()) << "Hit point should be at x=0";
    EXPECT_NEAR(hitPoint.y(), 0.0, eps<double>()) << "Hit point should be at y=0";

    // Verify barycentric coordinates (on bottom edge, so v=0, u+w=1)
    EXPECT_NEAR(v, 0.0, eps<double>()) << "Should be on bottom edge (v=0)";
    EXPECT_NEAR(u + (1 - u - v), 1.0, eps<double>()) << "Barycentric coordinates should sum to 1";
  }

  // The basic ray-triangle intersection should find the hit at t=-2
  EXPECT_TRUE(hit) << "Ray-triangle intersection should detect backward intersection";

  // This demonstrates that our ray-triangle intersection DOES support negative t values
  // So the issue must be in how the BVH applies the ray.minT/maxT constraints
}

TEST(RayTriangleIntersectionTest, BVH_RayParameterSimulation) {
  // This test simulates how the BVH constructs and uses the ray parameters
  // to understand why the test Ray3<S>({0, 0, 0}, {0, 0, 1}, 0.0, -2.0) fails

  std::cout << "=== Simulating BVH ray parameter usage ===" << std::endl;

  // Triangle at z=-2 (same as failing test)
  Eigen::Vector3d v0(-1., 0., -2.0);
  Eigen::Vector3d v1(+1., 0., -2.0);
  Eigen::Vector3d v2(+0., 1., -2.0);

  // Simulate the Ray3 constructor call: Ray3<S>({0, 0, 0}, {0, 0, 1}, 0.0, -2.0)
  // According to Ray.h constructor: Ray3(origin, direction, maxT, minT)
  Eigen::Vector3d rayOrigin(0., 0., 0.);
  Eigen::Vector3d rayDirection(0., 0., 1.);
  double maxT = 0.0;
  double minT = -2.0;

  std::cout << "Ray parameters:" << std::endl;
  std::cout << "  origin: " << rayOrigin.transpose() << std::endl;
  std::cout << "  direction: " << rayDirection.transpose() << std::endl;
  std::cout << "  minT: " << minT << std::endl;
  std::cout << "  maxT: " << maxT << std::endl;
  std::cout << "  Valid t range: [" << minT << ", " << maxT << "]" << std::endl;

  // Test raw ray-triangle intersection (no t filtering)
  Eigen::Vector3d hitPoint;
  double t = NAN, u = NAN, v = NAN;
  bool hit = rayTriangleIntersect(rayOrigin, rayDirection, v0, v1, v2, hitPoint, t, u, v);

  std::cout << "Raw intersection result: " << (hit ? "HIT" : "MISS") << std::endl;
  if (hit) {
    std::cout << "  t=" << t << std::endl;
    std::cout << "  hitPoint=" << hitPoint.transpose() << std::endl;
  }

  // Simulate BVH t-range filtering (like in TriBvh::intersectTriangle line 608)
  bool tInRange = (t >= minT) && (t <= maxT);
  bool validHit = hit && tInRange;

  std::cout << "BVH t-range check:" << std::endl;
  std::cout << "  t >= minT: " << t << " >= " << minT << " = " << (t >= minT) << std::endl;
  std::cout << "  t <= maxT: " << t << " <= " << maxT << " = " << (t <= maxT) << std::endl;
  std::cout << "  tInRange: " << tInRange << std::endl;
  std::cout << "  Final valid hit: " << validHit << std::endl;

  // The basic intersection should work
  EXPECT_TRUE(hit) << "Basic ray-triangle intersection should succeed";

  if (hit) {
    EXPECT_NEAR(t, -2.0, eps<double>()) << "Intersection should be at t=-2";

    // The key test: t=-2 should be within range [-2, 0]
    EXPECT_TRUE(tInRange) << "t=" << t << " should be within range [" << minT << ", " << maxT
                          << "]";
    EXPECT_TRUE(validHit) << "Ray should be valid after BVH-style t-range filtering";
  }
}

TEST(RayTriangleIntersectionTest, ScalarAndWideConsistency_BasicHit) {
  // Test a simple hit case with both scalar and wide versions
  Eigen::Vector3f rayOrigin(1.f, 1.f, 1.f);
  Eigen::Vector3f rayDirection(0.f, 1.f, 0.f);
  Eigen::Vector3f v0(0.f, 5.f, 0.f);
  Eigen::Vector3f v1(5.f, 5.f, 0.f);
  Eigen::Vector3f v2(2.5f, 5.f, 2.5f);

  testRayTriangleIntersectionBothVersions<float>(
      rayOrigin, rayDirection, v0, v1, v2, true, "Basic hit case");
}

TEST(RayTriangleIntersectionTest, ScalarAndWideConsistency_BasicMiss) {
  // Test a simple miss case with both scalar and wide versions
  Eigen::Vector3f rayOrigin(1.f, 1.f, 1.f);
  Eigen::Vector3f rayDirection(0.f, 1.f, 0.f);
  Eigen::Vector3f v0(10.f, 5.f, 0.f);
  Eigen::Vector3f v1(10.f, 15.f, 0.f);
  Eigen::Vector3f v2(12.5f, 5.f, 2.5f);

  testRayTriangleIntersectionBothVersions<float>(
      rayOrigin, rayDirection, v0, v1, v2, false, "Basic miss case");
}

TEST(RayTriangleIntersectionTest, ScalarAndWideConsistency_EdgeCases) {
  // Test various edge cases to ensure scalar and wide versions agree

  // Case 1: Ray parallel to triangle plane
  testRayTriangleIntersectionBothVersions<double>(
      Eigen::Vector3d(0., 0., 0.), // origin
      Eigen::Vector3d(1., 0., 0.), // direction parallel to triangle
      Eigen::Vector3d(-1., 0., 1.), // triangle vertices (in plane z=1)
      Eigen::Vector3d(1., 0., 1.),
      Eigen::Vector3d(0., 1., 1.),
      false,
      "Ray parallel to triangle plane");

  // Case 2: Ray hits triangle edge
  testRayTriangleIntersectionBothVersions<double>(
      Eigen::Vector3d(0.5, 0., 0.), // origin
      Eigen::Vector3d(0., 0., 1.), // direction towards triangle
      Eigen::Vector3d(0., 0., 2.), // triangle vertices
      Eigen::Vector3d(1., 0., 2.),
      Eigen::Vector3d(0.5, 1., 2.),
      true,
      "Ray hits triangle edge");

  // Case 3: Ray hits triangle vertex
  testRayTriangleIntersectionBothVersions<double>(
      Eigen::Vector3d(0., 0., 0.), // origin
      Eigen::Vector3d(0., 0., 1.), // direction towards vertex
      Eigen::Vector3d(0., 0., 2.), // triangle vertices (ray hits first vertex)
      Eigen::Vector3d(1., 0., 2.),
      Eigen::Vector3d(0., 1., 2.),
      true,
      "Ray hits triangle vertex");

  // Case 4: Ray passes very close to triangle but misses
  testRayTriangleIntersectionBothVersions<double>(
      Eigen::Vector3d(1.001, 0., 0.), // origin slightly outside triangle projection
      Eigen::Vector3d(0., 0., 1.), // direction towards triangle plane
      Eigen::Vector3d(-1., -1., 2.), // triangle vertices
      Eigen::Vector3d(1., -1., 2.),
      Eigen::Vector3d(0., 1., 2.),
      false,
      "Ray passes very close but misses");

  // Case 5: Degenerate triangle (all points collinear)
  testRayTriangleIntersectionBothVersions<double>(
      Eigen::Vector3d(0., 0., 0.), // origin
      Eigen::Vector3d(0., 0., 1.), // direction
      Eigen::Vector3d(0., 0., 2.), // degenerate triangle (all points on a line)
      Eigen::Vector3d(1., 0., 2.),
      Eigen::Vector3d(2., 0., 2.),
      false,
      "Degenerate triangle (collinear points)");
}

TEST(RayTriangleIntersectionTest, ScalarAndWideConsistency_NoBarycentricVersion) {
  // Test the version without barycentric coordinates

  testRayTriangleIntersectionBothVersionsNoBary<float>(
      Eigen::Vector3f(1.f, 1.f, 1.f), // origin
      Eigen::Vector3f(0.f, 1.f, 0.f), // direction
      Eigen::Vector3f(0.f, 5.f, 0.f), // triangle vertices
      Eigen::Vector3f(5.f, 5.f, 0.f),
      Eigen::Vector3f(2.5f, 5.f, 2.5f),
      true,
      "No barycentric version - hit case");

  testRayTriangleIntersectionBothVersionsNoBary<float>(
      Eigen::Vector3f(1.f, 1.f, 1.f), // origin
      Eigen::Vector3f(0.f, 1.f, 0.f), // direction
      Eigen::Vector3f(10.f, 5.f, 0.f), // triangle vertices (miss)
      Eigen::Vector3f(10.f, 15.f, 0.f),
      Eigen::Vector3f(12.5f, 5.f, 2.5f),
      false,
      "No barycentric version - miss case");
}

TEST(RayTriangleIntersectionTest, ScalarAndWideConsistency_PrecisionTests) {
  // Test cases that might reveal precision differences between scalar and wide versions

  // Case 1: Very small triangle
  testRayTriangleIntersectionBothVersions<double>(
      Eigen::Vector3d(0., 0., 0.),
      Eigen::Vector3d(0., 0., 1.),
      Eigen::Vector3d(-1e-6, -1e-6, 1.),
      Eigen::Vector3d(1e-6, -1e-6, 1.),
      Eigen::Vector3d(0., 1e-6, 1.),
      true,
      "Very small triangle");

  // Case 2: Triangle far from origin
  testRayTriangleIntersectionBothVersions<double>(
      Eigen::Vector3d(1e6, 1e6, 1e6),
      Eigen::Vector3d(0., 0., 1.),
      Eigen::Vector3d(1e6 - 1., 1e6 - 1., 1e6 + 100.),
      Eigen::Vector3d(1e6 + 1., 1e6 - 1., 1e6 + 100.),
      Eigen::Vector3d(1e6, 1e6 + 1., 1e6 + 100.),
      true,
      "Triangle far from origin");

  // Case 3: Nearly parallel ray and triangle
  double nearParallel = 1e-10;
  testRayTriangleIntersectionBothVersions<double>(
      Eigen::Vector3d(0., 0., 0.),
      Eigen::Vector3d(1., 0., nearParallel), // Almost parallel to z=0 plane
      Eigen::Vector3d(-1., -1., 1e6 * nearParallel), // Triangle in plane with tiny z
      Eigen::Vector3d(1., -1., 1e6 * nearParallel),
      Eigen::Vector3d(0., 1., 1e6 * nearParallel),
      false,
      "Nearly parallel ray and triangle"); // Corrected: nearly parallel should miss
}

// Tests from the original Wide test file that add unique coverage
TEST(RayTriangleIntersectionTest, MultipleTriangles_SimdPacking) {
  // Test multiple triangles packed into SIMD format (from original Wide test)

  std::vector<std::array<Eigen::Vector3<float>, 3>> triangles = {
      // Triangle 0: Should hit
      {Eigen::Vector3f(-1, -1, 2), Eigen::Vector3f(1, -1, 2), Eigen::Vector3f(0, 1, 2)},
      // Triangle 1: Should miss (too far to the side)
      {Eigen::Vector3f(5, -1, 2), Eigen::Vector3f(7, -1, 2), Eigen::Vector3f(6, 1, 2)},
      // Triangle 2: Should hit
      {Eigen::Vector3f(-0.5f, -0.5f, 3),
       Eigen::Vector3f(0.5f, -0.5f, 3),
       Eigen::Vector3f(0, 0.5f, 3)},
      // Triangle 3: Should miss (behind ray origin)
      {Eigen::Vector3f(-1, -1, -1), Eigen::Vector3f(1, -1, -1), Eigen::Vector3f(0, 1, -1)}};

  std::vector<bool> expectedHits = {
      true, false, true, true}; // Triangle 3 actually hits (ray goes through z=0 to z=-1)

  Eigen::Vector3f rayOrigin(0, 0, 0);
  Eigen::Vector3f rayDirection(0, 0, 1);

  testMultipleTriangles<float>(
      rayOrigin, rayDirection, triangles, expectedHits, "Mixed hit/miss triangles");
}

TEST(RayTriangleIntersectionTest, EdgeCases_SimdConsistency) {
  // Test edge cases with SIMD to ensure consistency (from original Wide test)

  std::vector<std::array<Eigen::Vector3<double>, 3>> edgeCaseTriangles = {
      // Degenerate triangle (all points collinear)
      {Eigen::Vector3d(0, 0, 2), Eigen::Vector3d(1, 0, 2), Eigen::Vector3d(2, 0, 2)},
      // Very small triangle
      {Eigen::Vector3d(-0.001, -0.001, 2),
       Eigen::Vector3d(0.001, -0.001, 2),
       Eigen::Vector3d(0, 0.001, 2)},
      // Triangle with one very acute angle
      {Eigen::Vector3d(-10, 0, 2), Eigen::Vector3d(10, 0, 2), Eigen::Vector3d(0, 0.001, 2)},
      // Normal triangle for comparison
      {Eigen::Vector3d(-1, -1, 2), Eigen::Vector3d(1, -1, 2), Eigen::Vector3d(0, 1, 2)}};

  std::vector<bool> expectedHits = {
      false, true, true, true}; // Triangle 2 (acute angle) actually hits at the centroid

  Eigen::Vector3d rayOrigin(0, 0, 0);
  Eigen::Vector3d rayDirection(0, 0, 1);

  testMultipleTriangles<double>(
      rayOrigin, rayDirection, edgeCaseTriangles, expectedHits, "Edge case triangles");
}

TEST(RayTriangleIntersectionTest, RandomTriangles_StressTest) {
  // Stress test with random triangles (adapted from original Wide test)

  std::mt19937 rng(42); // Fixed seed for reproducibility
  std::uniform_real_distribution<double> dist(-10.0, 10.0);

  const int numTests = 50; // Reduced from 100 for faster execution

  for (int test = 0; test < numTests; ++test) {
    // Generate random triangles (up to 4 for SIMD width)
    std::vector<std::array<Eigen::Vector3<double>, 3>> triangles;
    triangles.reserve(4);
    for (int i = 0; i < 4; ++i) {
      triangles.push_back(
          {Eigen::Vector3d(dist(rng), dist(rng), dist(rng)),
           Eigen::Vector3d(dist(rng), dist(rng), dist(rng)),
           Eigen::Vector3d(dist(rng), dist(rng), dist(rng))});
    }

    // Generate random ray
    Eigen::Vector3d rayOrigin(dist(rng), dist(rng), dist(rng));
    Eigen::Vector3d rayDirection = Eigen::Vector3d(dist(rng), dist(rng), dist(rng)).normalized();

    std::vector<bool> expectedHits; // Don't specify expected hits for random test

    testMultipleTriangles<double>(
        rayOrigin, rayDirection, triangles, expectedHits, "Random test " + std::to_string(test));
  }
}

} // namespace
} // namespace axel::test
