/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "axel/Bvh.h"
#include "axel/BvhEmbree.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace axel::test {
namespace {

// Components: BVHEmbree + BoundingBoxd
// Alias for convenience, matching existing test style.
using BBox = BoundingBoxd;

using ::testing::SizeIs;
using ::testing::UnorderedElementsAreArray;

// ============================================================================
// Test fixture that sets up both BVHEmbree and Bvh with the same bounding boxes
// to verify cross-component parity.
// Components: BVHEmbree, Bvhd (two real BVH implementations)
// ============================================================================
class BvhEmbreeBvhParityIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a set of non-trivial bounding boxes spread across 3D space.
    boxes_ = {
        BBox(Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(1.0, 1.0, 1.0), 0),
        BBox(Eigen::Vector3d(2.0, 2.0, 2.0), Eigen::Vector3d(4.0, 4.0, 4.0), 1),
        BBox(Eigen::Vector3d(-3.0, -3.0, -3.0), Eigen::Vector3d(-1.0, -1.0, -1.0), 2),
        BBox(Eigen::Vector3d(0.5, 0.5, 0.5), Eigen::Vector3d(2.5, 2.5, 2.5), 3),
        BBox(Eigen::Vector3d(5.0, 5.0, 5.0), Eigen::Vector3d(7.0, 7.0, 7.0), 4),
        BBox(Eigen::Vector3d(-1.0, 0.0, -1.0), Eigen::Vector3d(1.0, 2.0, 1.0), 5),
    };

    bvhEmbree_ = std::make_unique<BVHEmbree>();
    bvhEmbree_->setBoundingBoxes(boxes_);

    bvh_ = std::make_unique<Bvhd>();
    bvh_->setBoundingBoxes(boxes_);
  }

  std::vector<BBox> boxes_;
  std::unique_ptr<BVHEmbree> bvhEmbree_;
  std::unique_ptr<Bvhd> bvh_;
};

// Components: BVHEmbree + Bvhd
// Verify that AABB queries return the same set of hits from both implementations.
TEST_F(BvhEmbreeBvhParityIntegrationTest, AabbQueryParityPartialOverlap) {
  // Query box that overlaps boxes 0, 3, and 5 but not 1, 2, or 4.
  const BBox queryBox(Eigen::Vector3d(-0.5, -0.5, -0.5), Eigen::Vector3d(1.5, 1.5, 1.5));

  const auto embreeHits = bvhEmbree_->query(queryBox);
  const auto bvhHits = bvh_->query(queryBox);

  ASSERT_EQ(embreeHits.size(), bvhHits.size());
  // Both should find the same primitives (order may differ between implementations).
  EXPECT_THAT(embreeHits, UnorderedElementsAreArray(bvhHits));
  // Verify we actually got meaningful hits (not zero).
  EXPECT_GE(embreeHits.size(), 2u);
}

// Components: BVHEmbree + Bvhd
// Large-scale random AABB query parity between the two BVH implementations.
TEST_F(BvhEmbreeBvhParityIntegrationTest, RandomAabbQueryParity) {
  std::default_random_engine eng{123};
  std::uniform_real_distribution<double> dist{-5.0, 10.0};

  // Build larger BVHs with 200 random boxes.
  std::vector<BBox> largeBoxes;
  for (uint32_t i = 0; i < 200; ++i) {
    const Eigen::Vector3d minCorner(dist(eng), dist(eng), dist(eng));
    const Eigen::Vector3d size(
        std::abs(dist(eng)) * 0.5, std::abs(dist(eng)) * 0.5, std::abs(dist(eng)) * 0.5);
    largeBoxes.emplace_back(minCorner, minCorner + size, i);
  }

  BVHEmbree embree;
  embree.setBoundingBoxes(largeBoxes);

  Bvhd bvh;
  bvh.setBoundingBoxes(largeBoxes);

  // Run 50 random AABB queries and verify parity.
  for (int q = 0; q < 50; ++q) {
    const Eigen::Vector3d qMin(dist(eng), dist(eng), dist(eng));
    const Eigen::Vector3d qSize(
        std::abs(dist(eng)) * 2.0, std::abs(dist(eng)) * 2.0, std::abs(dist(eng)) * 2.0);
    const BBox queryBox(qMin, qMin + qSize);

    const auto embreeHits = embree.query(queryBox);
    const auto bvhHits = bvh.query(queryBox);

    ASSERT_EQ(embreeHits.size(), bvhHits.size()) << "Mismatch at query " << q;
    EXPECT_THAT(embreeHits, UnorderedElementsAreArray(bvhHits))
        << "Hit set mismatch at query " << q;
  }
}

// ============================================================================
// Tests for BVHEmbree rebuild and QueryBuffer interactions with BoundingBoxd.
// Components: BVHEmbree + BoundingBoxd
// ============================================================================
class BvhEmbreeBoundingBoxIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    bvh_ = std::make_unique<BVHEmbree>();
  }

  std::unique_ptr<BVHEmbree> bvh_;
};

// Components: BVHEmbree + BoundingBoxd
// Verify that rebuilding the BVH with new boxes produces correct query results.
TEST_F(BvhEmbreeBoundingBoxIntegrationTest, RebuildWithNewBoxesUpdatesQueries) {
  // First build: 3 boxes in positive quadrant.
  std::vector<BBox> firstBoxes = {
      BBox(Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(1.0, 1.0, 1.0), 0),
      BBox(Eigen::Vector3d(2.0, 2.0, 2.0), Eigen::Vector3d(3.0, 3.0, 3.0), 1),
      BBox(Eigen::Vector3d(4.0, 4.0, 4.0), Eigen::Vector3d(5.0, 5.0, 5.0), 2),
  };
  bvh_->setBoundingBoxes(firstBoxes);
  EXPECT_EQ(bvh_->getPrimitiveCount(), 3u);

  // Query that hits box 0 only.
  const BBox queryNearOrigin(Eigen::Vector3d(-0.5, -0.5, -0.5), Eigen::Vector3d(0.5, 0.5, 0.5));
  auto hits = bvh_->query(queryNearOrigin);
  EXPECT_THAT(hits, SizeIs(1));
  EXPECT_EQ(hits[0], 0u);

  // Rebuild with different boxes: now box at origin has ID 10.
  std::vector<BBox> secondBoxes = {
      BBox(Eigen::Vector3d(-1.0, -1.0, -1.0), Eigen::Vector3d(1.0, 1.0, 1.0), 10),
      BBox(Eigen::Vector3d(10.0, 10.0, 10.0), Eigen::Vector3d(11.0, 11.0, 11.0), 11),
  };
  bvh_->setBoundingBoxes(secondBoxes);
  EXPECT_EQ(bvh_->getPrimitiveCount(), 2u);

  // Same query should now hit box with ID 10.
  hits = bvh_->query(queryNearOrigin);
  EXPECT_THAT(hits, SizeIs(1));
  EXPECT_EQ(hits[0], 10u);

  // Old box 2 at (4,4,4)-(5,5,5) should no longer be found.
  const BBox queryOldBox2(Eigen::Vector3d(3.5, 3.5, 3.5), Eigen::Vector3d(5.5, 5.5, 5.5));
  auto oldHits = bvh_->query(queryOldBox2);
  EXPECT_THAT(oldHits, SizeIs(0));
}

// Components: BVHEmbree + BoundingBoxd
// Verify ray and AABB queries agree on which primitives are hit for a
// carefully constructed scenario.
TEST_F(BvhEmbreeBoundingBoxIntegrationTest, RayAndAabbQueryAgreement) {
  // Line up boxes along the X axis.
  std::vector<BBox> boxes = {
      BBox(Eigen::Vector3d(0.0, -0.5, -0.5), Eigen::Vector3d(1.0, 0.5, 0.5), 0),
      BBox(Eigen::Vector3d(3.0, -0.5, -0.5), Eigen::Vector3d(4.0, 0.5, 0.5), 1),
      BBox(Eigen::Vector3d(6.0, -0.5, -0.5), Eigen::Vector3d(7.0, 0.5, 0.5), 2),
      // This box is off-axis, should not be hit by a ray along X.
      BBox(Eigen::Vector3d(0.0, 5.0, 5.0), Eigen::Vector3d(1.0, 6.0, 6.0), 3),
  };
  bvh_->setBoundingBoxes(boxes);

  // Ray along X axis at y=0, z=0 should hit boxes 0, 1, 2 but not 3.
  const Eigen::Vector3d origin(-1.0, 0.0, 0.0);
  const Eigen::Vector3d direction(1.0, 0.0, 0.0);
  const auto rayHits = bvh_->query(origin, direction);

  EXPECT_THAT(rayHits, SizeIs(3));
  // Verify the off-axis box is not in the ray hits.
  EXPECT_THAT(rayHits, ::testing::Not(::testing::Contains(3u)));

  // AABB query covering only the first two boxes along X.
  const BBox aabbQuery(Eigen::Vector3d(-0.5, -1.0, -1.0), Eigen::Vector3d(4.5, 1.0, 1.0));
  const auto aabbHits = bvh_->query(aabbQuery);
  EXPECT_THAT(aabbHits, SizeIs(2));
  EXPECT_THAT(aabbHits, ::testing::UnorderedElementsAre(0u, 1u));
}

} // namespace
} // namespace axel::test
