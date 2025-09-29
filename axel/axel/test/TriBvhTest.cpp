/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "axel/TriBvh.h"
#include "axel/common/Constants.h"
#include "axel/math/RayTriangleIntersection.h"
#include "axel/test/Helper.h"

#include <igl/AABB.h>
#include <perception/test_helpers/EigenChecks.h>
#include <test_helpers/FollyMatchers.h>

namespace axel::test {
namespace {

using ::test_helpers::HoldsError;
using ::testing::DoubleNear;
using ::testing::FloatNear;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

// Sphere generation parameters used in tests
template <typename S>
constexpr SphereMeshParameters<S> getSphereParams() {
  return SphereMeshParameters<S>{3.0, 64, 64};
}

using ScalarTypes = testing::Types<float, double>;

// Brute force box query implementation for validation
template <typename S>
std::vector<uint32_t> bruteForceBoxQuery(
    const BoundingBox<S>& queryBox,
    const Eigen::MatrixX3<S>& positions,
    const Eigen::MatrixX3i& triangles) {
  std::vector<uint32_t> results;

  for (int triIdx = 0; triIdx < triangles.rows(); ++triIdx) {
    // Compute triangle bounding box
    Eigen::AlignedBox<S, 3> triangleBBox;
    for (int vertIdx = 0; vertIdx < 3; ++vertIdx) {
      const int posIdx = triangles(triIdx, vertIdx);
      triangleBBox.extend(positions.row(posIdx).transpose());
    }

    // Test intersection with query box
    if (queryBox.aabb.intersects(triangleBBox)) {
      results.push_back(static_cast<uint32_t>(triIdx));
    }
  }

  // Sort results for comparison
  std::sort(results.begin(), results.end());
  return results;
}

template <typename S>
TriBvh<S> makeBvh(const MeshData<S>& meshData) {
  auto pos = meshData.positions;
  auto tris = meshData.triangles;

  // Store original data for validation
  std::vector<Eigen::Vector3<S>> vertices;
  vertices.reserve(pos.rows());
  for (int i = 0; i < pos.rows(); ++i) {
    vertices.emplace_back(pos.row(i));
  }

  std::vector<Eigen::Vector3i> triangleList;
  triangleList.reserve(tris.rows());
  for (int i = 0; i < tris.rows(); ++i) {
    triangleList.emplace_back(tris.row(i));
  }

  auto bvh = TriBvh<S>{std::move(pos), std::move(tris), 0.0};

  // Validate BVH construction for test correctness
  bvh.validate(triangleList, vertices);

  return bvh;
}

template <typename T>
struct TriBvhTest : testing::Test {
  using Type = T;
  static constexpr T kEdgeLength = 3.0;
};

TYPED_TEST_SUITE(TriBvhTest, ScalarTypes);

TYPED_TEST(TriBvhTest, EmptyTree) {
  using S = typename TestFixture::Type;

  // Test both constructors with empty input
  {
    // Test Eigen matrix constructor with empty matrices
    Eigen::MatrixX3<S> emptyPositions(0, 3);
    Eigen::MatrixX3i emptyTriangles(0, 3);

    TriBvh<S> bvh(emptyPositions, emptyTriangles);

    // Verify empty tree properties
    EXPECT_EQ(bvh.getPrimitiveCount(), 0);
    EXPECT_EQ(bvh.getNodeCount(), 0);

    // Verify validation works for empty tree
    std::vector<Eigen::Vector3<S>> emptyVertices;
    std::vector<Eigen::Vector3i> emptyTriangleList;
    EXPECT_NO_THROW(bvh.validate(emptyTriangleList, emptyVertices));

    // Test ray queries on empty tree
    Ray3<S> testRay(Eigen::Vector3<S>(0, 0, 0), Eigen::Vector3<S>(1, 0, 0));

    EXPECT_FALSE(bvh.anyHit(testRay));
    EXPECT_EQ(bvh.closestHit(testRay), std::nullopt);
    EXPECT_THAT(bvh.allHits(testRay), IsEmpty());
    EXPECT_THAT(bvh.lineHits(testRay), IsEmpty());

    // Test box queries on empty tree
    BoundingBox<S> testBox;
    testBox.aabb =
        Eigen::AlignedBox<S, 3>(Eigen::Vector3<S>(-1, -1, -1), Eigen::Vector3<S>(1, 1, 1));

    EXPECT_THAT(bvh.boxQuery(testBox), IsEmpty());

    typename TriBvh<S>::QueryBuffer buffer;
    EXPECT_EQ(bvh.boxQuery(testBox, buffer), 0);

    // Test closest surface point on empty tree
    auto result = bvh.closestSurfacePoint(Eigen::Vector3<S>(0, 0, 0));
    EXPECT_EQ(result.triangleIdx, kInvalidTriangleIdx);

    // Test with maxDist parameter
    auto resultWithMaxDist = bvh.closestSurfacePoint(Eigen::Vector3<S>(0, 0, 0), S(10.0));
    EXPECT_EQ(resultWithMaxDist.triangleIdx, kInvalidTriangleIdx);
  }

  {
    // Test std::vector constructor with empty vectors
    std::vector<Eigen::Vector3<S>> emptyVertices;
    std::vector<Eigen::Vector3i> emptyTriangles;

    TriBvh<S> bvh(emptyVertices, emptyTriangles);

    // Verify empty tree properties
    EXPECT_EQ(bvh.getPrimitiveCount(), 0);
    EXPECT_EQ(bvh.getNodeCount(), 0);

    // Verify validation works for empty tree
    EXPECT_NO_THROW(bvh.validate(emptyTriangles, emptyVertices));

    // Test ray queries on empty tree
    Ray3<S> testRay(Eigen::Vector3<S>(0, 0, 0), Eigen::Vector3<S>(1, 0, 0));

    EXPECT_FALSE(bvh.anyHit(testRay));
    EXPECT_EQ(bvh.closestHit(testRay), std::nullopt);
    EXPECT_THAT(bvh.allHits(testRay), IsEmpty());
    EXPECT_THAT(bvh.lineHits(testRay), IsEmpty());

    // Test closest surface point on empty tree
    auto result = bvh.closestSurfacePoint(Eigen::Vector3<S>(0, 0, 0));
    EXPECT_EQ(result.triangleIdx, kInvalidTriangleIdx);
  }

  // Test that printTreeStats doesn't crash on empty tree
  {
    std::vector<Eigen::Vector3<S>> emptyVertices;
    std::vector<Eigen::Vector3i> emptyTriangles;
    TriBvh<S> bvh(emptyVertices, emptyTriangles);

    std::cout << "=== Testing printTreeStats on empty tree ===" << std::endl;
    EXPECT_NO_THROW(bvh.printTreeStats());
  }
}

TYPED_TEST(TriBvhTest, BoxQuery) {
  using S = typename TestFixture::Type;

  // Create simple test geometry - a cube made of triangles
  Eigen::MatrixX3<S> positions(8, 3);
  positions.row(0) = Eigen::Vector3<S>(-1, -1, -1); // Bottom face
  positions.row(1) = Eigen::Vector3<S>(1, -1, -1);
  positions.row(2) = Eigen::Vector3<S>(1, 1, -1);
  positions.row(3) = Eigen::Vector3<S>(-1, 1, -1);
  positions.row(4) = Eigen::Vector3<S>(-1, -1, 1); // Top face
  positions.row(5) = Eigen::Vector3<S>(1, -1, 1);
  positions.row(6) = Eigen::Vector3<S>(1, 1, 1);
  positions.row(7) = Eigen::Vector3<S>(-1, 1, 1);

  // Create 12 triangles (2 per face of cube)
  Eigen::MatrixX3i triangles(12, 3);
  // Bottom face (z = -1)
  triangles.row(0) = Eigen::Vector3i(0, 1, 2);
  triangles.row(1) = Eigen::Vector3i(0, 2, 3);
  // Top face (z = 1)
  triangles.row(2) = Eigen::Vector3i(4, 6, 5);
  triangles.row(3) = Eigen::Vector3i(4, 7, 6);
  // Front face (y = -1)
  triangles.row(4) = Eigen::Vector3i(0, 5, 1);
  triangles.row(5) = Eigen::Vector3i(0, 4, 5);
  // Back face (y = 1)
  triangles.row(6) = Eigen::Vector3i(3, 2, 6);
  triangles.row(7) = Eigen::Vector3i(3, 6, 7);
  // Left face (x = -1)
  triangles.row(8) = Eigen::Vector3i(0, 3, 7);
  triangles.row(9) = Eigen::Vector3i(0, 7, 4);
  // Right face (x = 1)
  triangles.row(10) = Eigen::Vector3i(1, 5, 6);
  triangles.row(11) = Eigen::Vector3i(1, 6, 2);

  const TriBvh<S> bvh{std::move(positions), std::move(triangles), S(0.0)};

  // Test 1: Query box that completely contains the cube (should return all triangles)
  {
    BoundingBox<S> largeBox;
    largeBox.aabb =
        Eigen::AlignedBox<S, 3>(Eigen::Vector3<S>(-2, -2, -2), Eigen::Vector3<S>(2, 2, 2));

    auto results = bvh.boxQuery(largeBox);

    EXPECT_EQ(results.size(), 12); // Should find all 12 triangles

    // Check that all triangle indices are valid and unique
    std::set<uint32_t> uniqueResults(results.begin(), results.end());
    EXPECT_EQ(uniqueResults.size(), 12);
    for (uint32_t i = 0; i < 12; ++i) {
      EXPECT_TRUE(uniqueResults.find(i) != uniqueResults.end());
    }
  }

  // Test 2: Query box that intersects only part of the cube
  {
    BoundingBox<S> partialBox;
    partialBox.aabb = Eigen::AlignedBox<S, 3>(
        Eigen::Vector3<S>(-0.5, -0.5, -0.5),
        Eigen::Vector3<S>(0.5, 0.5, 2)); // Extends beyond cube in +Z only

    auto results = bvh.boxQuery(partialBox);

    EXPECT_GT(results.size(), 0); // Should find some triangles
    EXPECT_LE(results.size(), 12); // But not necessarily all

    // All returned indices should be valid
    for (uint32_t idx : results) {
      EXPECT_LT(idx, 12);
    }
  }

  // Test 3: Query box that doesn't intersect the cube at all
  {
    BoundingBox<S> missBox;
    missBox.aabb = Eigen::AlignedBox<S, 3>(
        Eigen::Vector3<S>(10, 10, 10), Eigen::Vector3<S>(11, 11, 11)); // Far away from cube

    auto results = bvh.boxQuery(missBox);

    EXPECT_EQ(results.size(), 0); // Should find no triangles
  }

  // Test 4: Very small query box inside the cube
  {
    BoundingBox<S> smallBox;
    smallBox.aabb = Eigen::AlignedBox<S, 3>(
        Eigen::Vector3<S>(-0.1, -0.1, -0.1),
        Eigen::Vector3<S>(0.1, 0.1, 0.1)); // Tiny box at center

    auto results = bvh.boxQuery(smallBox);

    // All returned indices should be valid
    for (uint32_t idx : results) {
      EXPECT_LT(idx, 12);
    }
  }

  // Test 5: Test the buffer-based boxQuery method
  {
    BoundingBox<S> testBox;
    testBox.aabb = Eigen::AlignedBox<S, 3>(
        Eigen::Vector3<S>(-1.5, -1.5, -1.5), Eigen::Vector3<S>(1.5, 1.5, 1.5));

    typename TriBvh<S>::QueryBuffer buffer;
    uint32_t hitCount = bvh.boxQuery(testBox, buffer);

    EXPECT_GT(hitCount, 0);
    EXPECT_LE(hitCount, std::min<uint32_t>(12, buffer.size()));

    // Check that buffer contains valid triangle indices
    for (uint32_t i = 0; i < hitCount; ++i) {
      EXPECT_LT(buffer[i], 12);
    }

    // Compare with vector-based method
    auto vectorResults = bvh.boxQuery(testBox);
    if (vectorResults.size() <= buffer.size()) {
      // If all results fit in buffer, they should match
      EXPECT_EQ(hitCount, vectorResults.size());
    } else {
      // Buffer should be full
      EXPECT_EQ(hitCount, buffer.size());
    }
  }
}

TYPED_TEST(TriBvhTest, BoxQuerySphere) {
  using S = typename TestFixture::Type;

  // Test with a more complex geometry (sphere)
  auto sphere = generateSphere(getSphereParams<S>());
  const TriBvh<S> bvh{std::move(sphere.positions), std::move(sphere.triangles), S(0.0)};

  std::cout << "=== BoxQuery Sphere Tests ===" << std::endl;
  std::cout << "Sphere triangles: " << bvh.getPrimitiveCount() << std::endl;

  // Test 1: Query box that should contain the entire sphere
  {
    BoundingBox<S> largeBox;
    largeBox.aabb =
        Eigen::AlignedBox<S, 3>(Eigen::Vector3<S>(-5, -5, -5), Eigen::Vector3<S>(5, 5, 5));

    auto results = bvh.boxQuery(largeBox);
    std::cout << "Large sphere box query returned " << results.size() << " triangles" << std::endl;

    EXPECT_EQ(results.size(), bvh.getPrimitiveCount()); // Should find all triangles
  }

  // Test 2: Query box that intersects only part of the sphere
  {
    BoundingBox<S> partialBox;
    partialBox.aabb = Eigen::AlignedBox<S, 3>(
        Eigen::Vector3<S>(-2.5, -2.5, -2.5),
        Eigen::Vector3<S>(2.5, 2.5, 2.5)); // Box that reaches sphere surface

    auto results = bvh.boxQuery(partialBox);
    std::cout << "Partial sphere box query returned " << results.size() << " triangles"
              << std::endl;

    EXPECT_GT(results.size(), 0); // Should find some triangles
    EXPECT_LT(results.size(), bvh.getPrimitiveCount()); // But not all
  }

  // Test 3: Query box at sphere surface
  {
    BoundingBox<S> surfaceBox;
    surfaceBox.aabb = Eigen::AlignedBox<S, 3>(
        Eigen::Vector3<S>(2.5, -0.5, -0.5),
        Eigen::Vector3<S>(3.5, 0.5, 0.5)); // Box at sphere surface

    auto results = bvh.boxQuery(surfaceBox);
    std::cout << "Surface sphere box query returned " << results.size() << " triangles"
              << std::endl;

    EXPECT_GT(results.size(), 0); // Should find triangles at the surface
  }

  // Test 4: Query box outside the sphere
  {
    BoundingBox<S> outsideBox;
    outsideBox.aabb = Eigen::AlignedBox<S, 3>(
        Eigen::Vector3<S>(10, 10, 10), Eigen::Vector3<S>(11, 11, 11)); // Far from sphere

    auto results = bvh.boxQuery(outsideBox);
    std::cout << "Outside sphere box query returned " << results.size() << " triangles"
              << std::endl;

    EXPECT_EQ(results.size(), 0); // Should find no triangles
  }
}

TYPED_TEST(TriBvhTest, BoxQueryBruteForceComparison) {
  using S = typename TestFixture::Type;

  // Create a more complex test geometry - a cube made of triangles
  Eigen::MatrixX3<S> positions(8, 3);
  positions.row(0) = Eigen::Vector3<S>(-1, -1, -1); // Bottom face
  positions.row(1) = Eigen::Vector3<S>(1, -1, -1);
  positions.row(2) = Eigen::Vector3<S>(1, 1, -1);
  positions.row(3) = Eigen::Vector3<S>(-1, 1, -1);
  positions.row(4) = Eigen::Vector3<S>(-1, -1, 1); // Top face
  positions.row(5) = Eigen::Vector3<S>(1, -1, 1);
  positions.row(6) = Eigen::Vector3<S>(1, 1, 1);
  positions.row(7) = Eigen::Vector3<S>(-1, 1, 1);

  // Create 12 triangles (2 per face of cube)
  Eigen::MatrixX3i triangles(12, 3);
  // Bottom face (z = -1)
  triangles.row(0) = Eigen::Vector3i(0, 1, 2);
  triangles.row(1) = Eigen::Vector3i(0, 2, 3);
  // Top face (z = 1)
  triangles.row(2) = Eigen::Vector3i(4, 6, 5);
  triangles.row(3) = Eigen::Vector3i(4, 7, 6);
  // Front face (y = -1)
  triangles.row(4) = Eigen::Vector3i(0, 5, 1);
  triangles.row(5) = Eigen::Vector3i(0, 4, 5);
  // Back face (y = 1)
  triangles.row(6) = Eigen::Vector3i(3, 2, 6);
  triangles.row(7) = Eigen::Vector3i(3, 6, 7);
  // Left face (x = -1)
  triangles.row(8) = Eigen::Vector3i(0, 3, 7);
  triangles.row(9) = Eigen::Vector3i(0, 7, 4);
  // Right face (x = 1)
  triangles.row(10) = Eigen::Vector3i(1, 5, 6);
  triangles.row(11) = Eigen::Vector3i(1, 6, 2);

  // Keep copies for brute force comparison
  const Eigen::MatrixX3<S> originalPositions = positions;
  const Eigen::MatrixX3i originalTriangles = triangles;

  const TriBvh<S> bvh{std::move(positions), std::move(triangles), S(0.0)};

  std::cout << "=== BoxQuery Brute Force Comparison Tests ===" << std::endl;
  std::cout << "Total triangles: " << bvh.getPrimitiveCount() << std::endl;

  // Test multiple query boxes of different sizes and positions
  std::vector<BoundingBox<S>> testBoxes;

  // Test case 1: Box that contains entire geometry
  {
    BoundingBox<S> box;
    box.aabb = Eigen::AlignedBox<S, 3>(Eigen::Vector3<S>(-2, -2, -2), Eigen::Vector3<S>(2, 2, 2));
    testBoxes.push_back(box);
  }

  // Test case 2: Box that intersects part of the geometry
  {
    BoundingBox<S> box;
    box.aabb = Eigen::AlignedBox<S, 3>(
        Eigen::Vector3<S>(-0.5, -0.5, -0.5), Eigen::Vector3<S>(1.5, 1.5, 1.5));
    testBoxes.push_back(box);
  }

  // Test case 3: Box that doesn't intersect geometry
  {
    BoundingBox<S> box;
    box.aabb = Eigen::AlignedBox<S, 3>(Eigen::Vector3<S>(5, 5, 5), Eigen::Vector3<S>(6, 6, 6));
    testBoxes.push_back(box);
  }

  // Test case 4: Small box at corner
  {
    BoundingBox<S> box;
    box.aabb =
        Eigen::AlignedBox<S, 3>(Eigen::Vector3<S>(0.8, 0.8, 0.8), Eigen::Vector3<S>(1.2, 1.2, 1.2));
    testBoxes.push_back(box);
  }

  // Test case 5: Box that clips one face
  {
    BoundingBox<S> box;
    box.aabb = Eigen::AlignedBox<S, 3>(
        Eigen::Vector3<S>(-2, -0.2, -0.2), Eigen::Vector3<S>(-0.8, 0.2, 0.2));
    testBoxes.push_back(box);
  }

  for (size_t testIdx = 0; testIdx < testBoxes.size(); ++testIdx) {
    const auto& queryBox = testBoxes[testIdx];

    std::cout << "\n--- Test Case " << (testIdx + 1) << " ---" << std::endl;
    std::cout << "Query box: [" << queryBox.aabb.min().transpose() << "] to ["
              << queryBox.aabb.max().transpose() << "]" << std::endl;

    // Get BVH results (sorted)
    auto bvhResults = bvh.boxQuery(queryBox);
    std::sort(bvhResults.begin(), bvhResults.end());

    // Get brute force results (already sorted)
    auto bruteForceResults = bruteForceBoxQuery(queryBox, originalPositions, originalTriangles);

    std::cout << "BVH results: " << bvhResults.size() << " triangles";
    if (!bvhResults.empty()) {
      std::cout << " [";
      for (size_t i = 0; i < std::min<size_t>(5, bvhResults.size()); ++i) {
        if (i > 0) {
          std::cout << ", ";
        }
        std::cout << bvhResults[i];
      }
      if (bvhResults.size() > 5) {
        std::cout << ", ...";
      }
      std::cout << "]";
    }
    std::cout << std::endl;

    std::cout << "Brute force results: " << bruteForceResults.size() << " triangles";
    if (!bruteForceResults.empty()) {
      std::cout << " [";
      for (size_t i = 0; i < std::min<size_t>(5, bruteForceResults.size()); ++i) {
        if (i > 0) {
          std::cout << ", ";
        }
        std::cout << bruteForceResults[i];
      }
      if (bruteForceResults.size() > 5) {
        std::cout << ", ...";
      }
      std::cout << "]";
    }
    std::cout << std::endl;

    // Compare results
    EXPECT_EQ(bvhResults.size(), bruteForceResults.size())
        << "Test case " << (testIdx + 1) << ": Result count mismatch";
    EXPECT_EQ(bvhResults, bruteForceResults)
        << "Test case " << (testIdx + 1) << ": Results don't match";

    // Also test the buffer-based method
    typename TriBvh<S>::QueryBuffer buffer;
    uint32_t bufferHitCount = bvh.boxQuery(queryBox, buffer);

    std::cout << "Buffer method returned: " << bufferHitCount << " triangles" << std::endl;

    if (bvhResults.size() <= buffer.size()) {
      // All results should fit in buffer
      EXPECT_EQ(bufferHitCount, bvhResults.size())
          << "Test case " << (testIdx + 1) << ": Buffer count mismatch";

      // Check that buffer contents match (need to sort buffer results)
      std::vector<uint32_t> bufferResults(buffer.begin(), buffer.begin() + bufferHitCount);
      std::sort(bufferResults.begin(), bufferResults.end());
      EXPECT_EQ(bufferResults, bvhResults)
          << "Test case " << (testIdx + 1) << ": Buffer results don't match";
    } else {
      // Buffer should be full
      EXPECT_EQ(bufferHitCount, buffer.size())
          << "Test case " << (testIdx + 1) << ": Buffer should be full";
    }
  }

  std::cout << "\nAll brute force comparison tests completed!" << std::endl;
}

TYPED_TEST(TriBvhTest, rayHitsAny) {
  using S = typename TestFixture::Type;

  auto sphere = generateSphere(getSphereParams<S>());
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

  auto sphere = generateSphere(getSphereParams<S>());
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

  auto sphere = generateSphere(getSphereParams<S>());
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
  auto s1 = generateSphere(getSphereParams<S>());
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

  auto sphere = generateSphere(getSphereParams<S>());
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

  checkQueries(makeBvh<S>(sphere));
}

TYPED_TEST(TriBvhTest, ClosestSurfacePoint_ExpectZeroDistFromMeshPoints) {
  using S = typename TestFixture::Type;

  auto sphere = generateSphere(getSphereParams<S>());

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

  auto sphere = generateSphere(getSphereParams<S>());
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

// Test SIMD ray-triangle intersection directly
TYPED_TEST(TriBvhTest, SimdRayTriangleIntersection) {
  using S = typename TestFixture::Type;

  // Create a simple triangle for testing - make it easier to hit
  Eigen::Vector3<S> v0(-1, -1, 2); // Triangle at z=2
  Eigen::Vector3<S> v1(1, -1, 2);
  Eigen::Vector3<S> v2(0, 1, 2);

  // Create a single triangle mesh
  Eigen::MatrixX3<S> positions(3, 3);
  positions.row(0) = v0;
  positions.row(1) = v1;
  positions.row(2) = v2;

  Eigen::MatrixX3i triangles(1, 3);
  triangles.row(0) = Eigen::Vector3i(0, 1, 2);

  // Create BVH to access SIMD intersection function
  TriBvh<S> bvh(std::move(positions), std::move(triangles), 0.0);

  // Test ray that should definitely hit the triangle
  axel::Ray3<S> hitRay;
  hitRay.origin = Eigen::Vector3<S>(0, 0, 0); // Shoot from origin
  hitRay.direction = Eigen::Vector3<S>(0, 0, 1); // Towards +Z
  hitRay.minT = static_cast<S>(0.001);
  hitRay.maxT = static_cast<S>(10);

  std::cout << "=== Testing SIMD Ray-Triangle Intersection ===" << std::endl;
  std::cout << "Triangle vertices:" << std::endl;
  std::cout << "  v0: " << v0.transpose() << std::endl;
  std::cout << "  v1: " << v1.transpose() << std::endl;
  std::cout << "  v2: " << v2.transpose() << std::endl;
  std::cout << "Ray: origin=" << hitRay.origin.transpose()
            << " dir=" << hitRay.direction.transpose() << " minT=" << hitRay.minT
            << " maxT=" << hitRay.maxT << std::endl;

  // Test with scalar ray-triangle intersection first
  S u, v, t;
  Eigen::Vector3<S> hitPoint;
  bool scalarHit =
      axel::rayTriangleIntersect(hitRay.origin, hitRay.direction, v0, v1, v2, hitPoint, t, u, v);

  std::cout << "Scalar intersection: " << (scalarHit ? "HIT" : "MISS") << std::endl;
  if (scalarHit) {
    std::cout << "  t=" << t << " u=" << u << " v=" << v << std::endl;
    std::cout << "  w=" << (1 - u - v) << std::endl;
    std::cout << "  hitPoint=" << hitPoint.transpose() << std::endl;
    std::cout << "  Expected hitPoint=" << (hitRay.origin + t * hitRay.direction).transpose()
              << std::endl;
  }

  // Now test BVH intersection
  auto bvhResult = bvh.closestHit(hitRay);
  std::cout << "BVH intersection: " << (bvhResult.has_value() ? "HIT" : "MISS") << std::endl;
  if (bvhResult.has_value()) {
    std::cout << "  triangleId=" << bvhResult->triangleId << std::endl;
    std::cout << "  hitDistance=" << bvhResult->hitDistance << std::endl;
    std::cout << "  hitPoint=" << bvhResult->hitPoint.transpose() << std::endl;
    std::cout << "  baryCoords=" << bvhResult->baryCoords.transpose() << std::endl;
  }

  // They should both hit
  EXPECT_TRUE(scalarHit);
  EXPECT_TRUE(bvhResult.has_value());

  if (scalarHit && bvhResult.has_value()) {
    EXPECT_NEAR(t, bvhResult->hitDistance, static_cast<S>(1e-5));
    EXPECT_EIGEN_MATRIX_NEAR(hitPoint, bvhResult->hitPoint, static_cast<S>(1e-5));
  }
}

// Test ray that should miss
TYPED_TEST(TriBvhTest, SimdRayTriangleMiss) {
  using S = typename TestFixture::Type;

  // Create a simple triangle for testing
  Eigen::Vector3<S> v0(-1, 0, -2);
  Eigen::Vector3<S> v1(1, 0, -2);
  Eigen::Vector3<S> v2(0, 1, -2);

  // Create a single triangle mesh
  Eigen::MatrixX3<S> positions(3, 3);
  positions.row(0) = v0;
  positions.row(1) = v1;
  positions.row(2) = v2;

  Eigen::MatrixX3i triangles(1, 3);
  triangles.row(0) = Eigen::Vector3i(0, 1, 2);

  // Create BVH to access SIMD intersection function
  TriBvh<S> bvh(std::move(positions), std::move(triangles), 0.0);

  // Test ray that should miss the triangle (shoots to the side)
  axel::Ray3<S> missRay;
  missRay.origin = Eigen::Vector3<S>(2, 0, 0); // Outside triangle projection
  missRay.direction = Eigen::Vector3<S>(0, 0, -1);
  missRay.minT = static_cast<S>(0.001);
  missRay.maxT = static_cast<S>(10);

  std::cout << "=== Testing SIMD Ray-Triangle Miss ===" << std::endl;
  std::cout << "Ray: origin=" << missRay.origin.transpose()
            << " dir=" << missRay.direction.transpose() << std::endl;

  // Test with scalar ray-triangle intersection first
  S u, v, t;
  Eigen::Vector3<S> hitPoint;
  bool scalarHit =
      axel::rayTriangleIntersect(missRay.origin, missRay.direction, v0, v1, v2, hitPoint, t, u, v);

  std::cout << "Scalar intersection: " << (scalarHit ? "HIT" : "MISS") << std::endl;

  // Now test BVH intersection
  auto bvhResult = bvh.closestHit(missRay);
  std::cout << "BVH intersection: " << (bvhResult.has_value() ? "HIT" : "MISS") << std::endl;

  // Both should miss
  EXPECT_FALSE(scalarHit);
  EXPECT_FALSE(bvhResult.has_value());
}

TYPED_TEST(TriBvhTest, ClosestSurfacePoint_MaxDistanceTest) {
  using S = typename TestFixture::Type;

  // Create a simple triangle mesh - single triangle at known location
  Eigen::MatrixX3<S> positions(3, 3);
  positions.row(0) = Eigen::Vector3<S>(-1, 0, 0);
  positions.row(1) = Eigen::Vector3<S>(1, 0, 0);
  positions.row(2) = Eigen::Vector3<S>(0, 1, 0);

  Eigen::MatrixX3i triangles(1, 3);
  triangles.row(0) = Eigen::Vector3i(0, 1, 2);

  const TriBvh<S> bvh{std::move(positions), std::move(triangles), 0.0};

  // Query point at a known distance from the triangle
  // Place query point above the triangle center at distance = 2.0
  const Eigen::Vector3<S> queryPoint(0, S(1.0 / 3.0), 2);

  // Calculate the expected closest point on the triangle (should be at the centroid)
  const Eigen::Vector3<S> expectedClosestPoint(0, S(1.0 / 3.0), 0);
  const S expectedDistance = 2.0;

  std::cout << "=== Testing maxDist parameter ===" << std::endl;
  std::cout << "Query point: " << queryPoint.transpose() << std::endl;
  std::cout << "Expected closest point: " << expectedClosestPoint.transpose() << std::endl;
  std::cout << "Expected distance: " << expectedDistance << std::endl;

  // Test 1: Unlimited search (should find the point)
  {
    auto result = bvh.closestSurfacePoint(queryPoint);
    std::cout << "Unlimited search result:" << std::endl;
    if (result.triangleIdx != kInvalidTriangleIdx) {
      std::cout << "  Found triangle: " << result.triangleIdx << std::endl;
      std::cout << "  Closest point: " << result.point.transpose() << std::endl;
      std::cout << "  Distance: " << (queryPoint - result.point).norm() << std::endl;
      EXPECT_NE(result.triangleIdx, kInvalidTriangleIdx);
      EXPECT_EIGEN_MATRIX_NEAR(result.point, expectedClosestPoint, 1e-6);
      EXPECT_NEAR((queryPoint - result.point).norm(), expectedDistance, 1e-6);
    } else {
      std::cout << "  No result found" << std::endl;
      FAIL() << "Unlimited search should find the triangle";
    }
  }

  // Test 2: maxDist larger than actual distance (should find the point)
  {
    const S largeMaxDist = expectedDistance + 1.0; // 3.0
    auto result = bvh.closestSurfacePoint(queryPoint, largeMaxDist);
    std::cout << "Large maxDist (" << largeMaxDist << ") result:" << std::endl;
    if (result.triangleIdx != kInvalidTriangleIdx) {
      std::cout << "  Found triangle: " << result.triangleIdx << std::endl;
      std::cout << "  Closest point: " << result.point.transpose() << std::endl;
      std::cout << "  Distance: " << (queryPoint - result.point).norm() << std::endl;
      EXPECT_NE(result.triangleIdx, kInvalidTriangleIdx);
      EXPECT_EIGEN_MATRIX_NEAR(result.point, expectedClosestPoint, 1e-6);
      EXPECT_NEAR((queryPoint - result.point).norm(), expectedDistance, 1e-6);
    } else {
      std::cout << "  No result found" << std::endl;
      FAIL() << "Search with large maxDist should find the triangle";
    }
  }

  // Test 3: maxDist smaller than actual distance (should NOT find the point)
  {
    const S smallMaxDist = expectedDistance - 0.5; // 1.5
    auto result = bvh.closestSurfacePoint(queryPoint, smallMaxDist);
    std::cout << "Small maxDist (" << smallMaxDist << ") result:" << std::endl;
    if (result.triangleIdx != kInvalidTriangleIdx) {
      std::cout << "  Found triangle: " << result.triangleIdx << std::endl;
      std::cout << "  Closest point: " << result.point.transpose() << std::endl;
      std::cout << "  Distance: " << (queryPoint - result.point).norm() << std::endl;
      FAIL() << "Search with small maxDist should NOT find the triangle";
    } else {
      std::cout << "  No result found (as expected)" << std::endl;
      EXPECT_EQ(result.triangleIdx, kInvalidTriangleIdx);
    }
  }

  // Test 4: maxDist exactly equal to actual distance (should find the point)
  {
    const S exactMaxDist = expectedDistance;
    auto result = bvh.closestSurfacePoint(queryPoint, exactMaxDist);
    std::cout << "Exact maxDist (" << exactMaxDist << ") result:" << std::endl;
    if (result.triangleIdx != kInvalidTriangleIdx) {
      std::cout << "  Found triangle: " << result.triangleIdx << std::endl;
      std::cout << "  Closest point: " << result.point.transpose() << std::endl;
      std::cout << "  Distance: " << (queryPoint - result.point).norm() << std::endl;
      EXPECT_NE(result.triangleIdx, kInvalidTriangleIdx);
      EXPECT_EIGEN_MATRIX_NEAR(result.point, expectedClosestPoint, 1e-6);
      EXPECT_NEAR((queryPoint - result.point).norm(), expectedDistance, 1e-6);
    } else {
      std::cout << "  No result found" << std::endl;
      FAIL() << "Search with exact maxDist should find the triangle";
    }
  }

  // Test 5: maxDist very small (should NOT find the point)
  {
    const S tinyMaxDist = 0.1;
    auto result = bvh.closestSurfacePoint(queryPoint, tinyMaxDist);
    std::cout << "Tiny maxDist (" << tinyMaxDist << ") result:" << std::endl;
    if (result.triangleIdx != kInvalidTriangleIdx) {
      std::cout << "  Found triangle: " << result.triangleIdx << std::endl;
      std::cout << "  Closest point: " << result.point.transpose() << std::endl;
      std::cout << "  Distance: " << (queryPoint - result.point).norm() << std::endl;
      FAIL() << "Search with tiny maxDist should NOT find the triangle";
    } else {
      std::cout << "  No result found (as expected)" << std::endl;
      EXPECT_EQ(result.triangleIdx, kInvalidTriangleIdx);
    }
  }
}

} // namespace
} // namespace axel::test
