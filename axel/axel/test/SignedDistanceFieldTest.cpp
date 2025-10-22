/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "axel/SignedDistanceField.h"

#include <gtest/gtest.h>

#include "axel/common/Constants.h"

namespace axel {

class SignedDistanceFieldTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a simple 4x4x4 SDF with bounds from (0,0,0) to (3,3,3)
    // Initialize with 0.0f to maintain existing test behavior
    const BoundingBoxf bounds(Eigen::Vector3f(0.0f, 0.0f, 0.0f), Eigen::Vector3f(3.0f, 3.0f, 3.0f));
    const Eigen::Vector3<Index> resolution(4, 4, 4);
    sdf_ = std::make_unique<SignedDistanceFieldf>(bounds, resolution, 0.0f);
  }

  std::unique_ptr<SignedDistanceFieldf> sdf_;
};

TEST_F(SignedDistanceFieldTest, ConstructorWithData) {
  const BoundingBoxf bounds(Eigen::Vector3f(0.0f, 0.0f, 0.0f), Eigen::Vector3f(1.0f, 1.0f, 1.0f));
  const Eigen::Vector3<Index> resolution(2, 2, 2);
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

  SignedDistanceFieldf sdf(bounds, resolution, std::move(data));

  EXPECT_EQ(sdf.resolution(), resolution);
  EXPECT_EQ(sdf.totalVoxels(), 8);
  EXPECT_EQ(sdf.data().size(), 8);
  EXPECT_EQ(sdf.at(0, 0, 0), 1.0f);
  EXPECT_EQ(sdf.at(1, 0, 0), 2.0f);
  EXPECT_EQ(sdf.at(0, 1, 0), 3.0f);
  EXPECT_EQ(sdf.at(1, 1, 0), 4.0f);
}

TEST_F(SignedDistanceFieldTest, BasicGetSet) {
  sdf_->set(0, 0, 0, 1.5f);
  sdf_->set(3, 3, 3, -2.0f);
  sdf_->set(1, 2, 1, 0.5f);

  EXPECT_EQ(sdf_->at(0, 0, 0), 1.5f);
  EXPECT_EQ(sdf_->at(3, 3, 3), -2.0f);
  EXPECT_EQ(sdf_->at(1, 2, 1), 0.5f);
  EXPECT_EQ(sdf_->at(2, 2, 2), 0.0f); // Should still be zero
}

TEST_F(SignedDistanceFieldTest, WorldGridConversion) {
  // Test coordinate conversion
  const Eigen::Vector3f worldPos(1.5f, 1.5f, 1.5f);
  const Eigen::Vector3f gridPos = sdf_->worldToGrid(worldPos);
  const Eigen::Vector3f backToWorld = sdf_->gridToWorld(gridPos);

  // Grid position should be (1.5, 1.5, 1.5) / (0.75, 0.75, 0.75) = (2, 2, 2)
  EXPECT_NEAR(gridPos.x(), 2.0f, 1e-6f);
  EXPECT_NEAR(gridPos.y(), 2.0f, 1e-6f);
  EXPECT_NEAR(gridPos.z(), 2.0f, 1e-6f);

  // Converting back should give the original position
  EXPECT_NEAR(backToWorld.x(), worldPos.x(), 1e-6f);
  EXPECT_NEAR(backToWorld.y(), worldPos.y(), 1e-6f);
  EXPECT_NEAR(backToWorld.z(), worldPos.z(), 1e-6f);
}

TEST_F(SignedDistanceFieldTest, GridLocationFunction) {
  // Test the new gridLocation function that converts discrete indices to world coordinates

  // Test some specific grid indices
  const Eigen::Vector3f loc_000 = sdf_->gridLocation(0, 0, 0);
  const Eigen::Vector3f loc_111 = sdf_->gridLocation(1, 1, 1);
  const Eigen::Vector3f loc_333 = sdf_->gridLocation(3, 3, 3);

  // Expected world positions based on bounds (0,0,0) to (3,3,3) with resolution (4,4,4)
  // Grid (0,0,0) should map to world (0,0,0)
  EXPECT_NEAR(loc_000.x(), 0.0f, 1e-6f);
  EXPECT_NEAR(loc_000.y(), 0.0f, 1e-6f);
  EXPECT_NEAR(loc_000.z(), 0.0f, 1e-6f);

  // Grid (1,1,1) should map to world (0.75, 0.75, 0.75)
  EXPECT_NEAR(loc_111.x(), 0.75f, 1e-6f);
  EXPECT_NEAR(loc_111.y(), 0.75f, 1e-6f);
  EXPECT_NEAR(loc_111.z(), 0.75f, 1e-6f);

  // Grid (3,3,3) should map to world (2.25, 2.25, 2.25)
  EXPECT_NEAR(loc_333.x(), 2.25f, 1e-6f);
  EXPECT_NEAR(loc_333.y(), 2.25f, 1e-6f);
  EXPECT_NEAR(loc_333.z(), 2.25f, 1e-6f);

  // Test that gridLocation(i,j,k) is equivalent to gridToWorld(Vector3(i,j,k))
  for (Index i = 0; i < sdf_->resolution().x(); ++i) {
    for (Index j = 0; j < sdf_->resolution().y(); ++j) {
      for (Index k = 0; k < sdf_->resolution().z(); ++k) {
        const Eigen::Vector3f locationMethod = sdf_->gridLocation(i, j, k);
        const Eigen::Vector3f gridToWorldMethod = sdf_->gridToWorld(
            Eigen::Vector3f(static_cast<float>(i), static_cast<float>(j), static_cast<float>(k)));

        EXPECT_NEAR(locationMethod.x(), gridToWorldMethod.x(), 1e-6f);
        EXPECT_NEAR(locationMethod.y(), gridToWorldMethod.y(), 1e-6f);
        EXPECT_NEAR(locationMethod.z(), gridToWorldMethod.z(), 1e-6f);
      }
    }
  }
}

TEST_F(SignedDistanceFieldTest, VoxelSize) {
  const Eigen::Vector3f voxelSize = sdf_->voxelSize();
  // Bounds are from (0,0,0) to (3,3,3) with resolution (4,4,4), so voxel size should be 3/4 = 0.75
  EXPECT_NEAR(voxelSize.x(), 0.75f, 1e-6f);
  EXPECT_NEAR(voxelSize.y(), 0.75f, 1e-6f);
  EXPECT_NEAR(voxelSize.z(), 0.75f, 1e-6f);
}

TEST_F(SignedDistanceFieldTest, IsValidIndex) {
  EXPECT_TRUE(sdf_->isValidIndex(0, 0, 0));
  EXPECT_TRUE(sdf_->isValidIndex(3, 3, 3));
  EXPECT_TRUE(sdf_->isValidIndex(1, 2, 1));

  EXPECT_FALSE(sdf_->isValidIndex(-1, 0, 0));
  EXPECT_FALSE(sdf_->isValidIndex(0, -1, 0));
  EXPECT_FALSE(sdf_->isValidIndex(0, 0, -1));
  EXPECT_FALSE(sdf_->isValidIndex(4, 0, 0));
  EXPECT_FALSE(sdf_->isValidIndex(0, 4, 0));
  EXPECT_FALSE(sdf_->isValidIndex(0, 0, 4));
}

TEST_F(SignedDistanceFieldTest, TrilinearInterpolation) {
  // Set up a simple pattern for interpolation testing
  sdf_->set(0, 0, 0, 0.0f);
  sdf_->set(1, 0, 0, 1.0f);
  sdf_->set(0, 1, 0, 2.0f);
  sdf_->set(1, 1, 0, 3.0f);
  sdf_->set(0, 0, 1, 4.0f);
  sdf_->set(1, 0, 1, 5.0f);
  sdf_->set(0, 1, 1, 6.0f);
  sdf_->set(1, 1, 1, 7.0f);

  // Test exact corner values by converting grid coordinates to world coordinates
  EXPECT_NEAR(sdf_->sample(sdf_->gridToWorld(Eigen::Vector3f(0.0f, 0.0f, 0.0f))), 0.0f, 1e-6f);
  EXPECT_NEAR(sdf_->sample(sdf_->gridToWorld(Eigen::Vector3f(1.0f, 0.0f, 0.0f))), 1.0f, 1e-6f);
  EXPECT_NEAR(sdf_->sample(sdf_->gridToWorld(Eigen::Vector3f(0.0f, 1.0f, 1.0f))), 6.0f, 1e-6f);

  // Test interpolation at center of the unit cube (grid coordinates 0.5, 0.5, 0.5)
  const Eigen::Vector3f centerGridPos(0.5f, 0.5f, 0.5f);
  const Eigen::Vector3f centerWorldPos = sdf_->gridToWorld(centerGridPos);
  const float centerValue = sdf_->sample(centerWorldPos);
  const float expectedCenter = (0.0f + 1.0f + 2.0f + 3.0f + 4.0f + 5.0f + 6.0f + 7.0f) / 8.0f;
  EXPECT_NEAR(centerValue, expectedCenter, 1e-6f);
}

TEST_F(SignedDistanceFieldTest, GradientCalculation) {
  // Create a simple linear gradient in world x direction: f(x,y,z) = x
  for (Index i = 0; i < sdf_->resolution().x(); ++i) {
    for (Index j = 0; j < sdf_->resolution().y(); ++j) {
      for (Index k = 0; k < sdf_->resolution().z(); ++k) {
        const Eigen::Vector3f gridPos(
            static_cast<float>(i), static_cast<float>(j), static_cast<float>(k));
        const Eigen::Vector3f worldPos = sdf_->gridToWorld(gridPos);
        sdf_->set(i, j, k, worldPos.x()); // Linear function f(x,y,z) = x
      }
    }
  }

  const Eigen::Vector3f gradient = sdf_->gradient(Eigen::Vector3f(1.5f, 1.5f, 1.5f));

  // Gradient of f(x,y,z) = x should be (1, 0, 0)
  EXPECT_NEAR(gradient.x(), 1.0f, 0.1f);
  EXPECT_NEAR(gradient.y(), 0.0f, 0.1f);
  EXPECT_NEAR(gradient.z(), 0.0f, 0.1f);
}

TEST_F(SignedDistanceFieldTest, FillAndClear) {
  sdf_->fill(3.14f);

  for (const auto& value : sdf_->data()) {
    EXPECT_EQ(value, 3.14f);
  }

  sdf_->clear();

  for (const auto& value : sdf_->data()) {
    EXPECT_EQ(value, 0.0f);
  }
}

TEST_F(SignedDistanceFieldTest, BoundaryHandling) {
  // Set corner values
  sdf_->set(0, 0, 0, -1.0f);
  sdf_->set(3, 3, 3, 1.0f);

  // Sample outside bounds should now return SDF value at clamped point PLUS distance to that point
  const Eigen::Vector3f outsidePos1(-1.0f, -1.0f, -1.0f);
  const float outsideValue1 = sdf_->sample(outsidePos1);
  // Distance from (-1,-1,-1) to clamped point (0,0,0) is sqrt(3) ≈ 1.732
  // SDF value at (0,0,0) is -1.0, so total should be -1.0 + 1.732 ≈ 0.732
  const float expectedValue1 = -1.0f + (outsidePos1 - Eigen::Vector3f(0.0f, 0.0f, 0.0f)).norm();
  EXPECT_NEAR(outsideValue1, expectedValue1, 1e-5f);

  const Eigen::Vector3f outsidePos2(5.0f, 5.0f, 5.0f);
  const float outsideValue2 = sdf_->sample(outsidePos2);
  // Distance from (5,5,5) to clamped point (2.25,2.25,2.25) is sqrt((2.75)²*3) ≈ 4.77
  // SDF value at (3,3,3) corresponds to grid position (3,3,3) -> world pos (2.25,2.25,2.25)
  const Eigen::Vector3f clampedPos2 = sdf_->gridToWorld(Eigen::Vector3f(3.0f, 3.0f, 3.0f));
  const float expectedValue2 = 1.0f + (outsidePos2 - clampedPos2).norm();
  EXPECT_NEAR(outsideValue2, expectedValue2, 1e-5f);
}

TEST_F(SignedDistanceFieldTest, DoubleTypeSDF) {
  const BoundingBoxd bounds(Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(1.0, 1.0, 1.0));
  const Eigen::Vector3<Index> resolution(2, 2, 2);

  SignedDistanceFieldd sdf(bounds, resolution);

  sdf.set(0, 0, 0, 1.23456789);
  sdf.set(1, 1, 1, -2.87654321);

  EXPECT_NEAR(sdf.at(0, 0, 0), 1.23456789, 1e-15);
  EXPECT_NEAR(sdf.at(1, 1, 1), -2.87654321, 1e-15);

  const double sampledValue = sdf.sample(Eigen::Vector3d(0.0, 0.0, 0.0));
  EXPECT_NEAR(sampledValue, 1.23456789, 1e-15);
}

TEST_F(SignedDistanceFieldTest, UnitSphereInterpolation) {
  // Create a high-resolution SDF centered around origin with a unit sphere
  const BoundingBoxf bounds(
      Eigen::Vector3f(-2.0f, -2.0f, -2.0f), Eigen::Vector3f(2.0f, 2.0f, 2.0f));
  const Eigen::Vector3<Index> resolution(16, 16, 16);
  SignedDistanceFieldf sphereSdf(bounds, resolution);

  // Fill the SDF with signed distances to a unit sphere centered at origin
  const Eigen::Vector3f sphereCenter(0.0f, 0.0f, 0.0f);
  const float sphereRadius = 1.0f;

  for (Index i = 0; i < resolution.x(); ++i) {
    for (Index j = 0; j < resolution.y(); ++j) {
      for (Index k = 0; k < resolution.z(); ++k) {
        const Eigen::Vector3f gridPos(
            static_cast<float>(i), static_cast<float>(j), static_cast<float>(k));
        const Eigen::Vector3f worldPos = sphereSdf.gridToWorld(gridPos);
        const float distance = (worldPos - sphereCenter).norm() - sphereRadius;
        sphereSdf.set(i, j, k, distance);
      }
    }
  }

  // Test interpolation at various positions
  std::vector<std::pair<Eigen::Vector3f, float>> testCases = {
      // Points exactly on sphere surface should have distance ~0
      {Eigen::Vector3f(1.0f, 0.0f, 0.0f), 0.0f},
      {Eigen::Vector3f(0.0f, 1.0f, 0.0f), 0.0f},
      {Eigen::Vector3f(0.0f, 0.0f, 1.0f), 0.0f},
      {Eigen::Vector3f(-1.0f, 0.0f, 0.0f), 0.0f},

      // Points inside sphere should have negative distance
      {Eigen::Vector3f(0.0f, 0.0f, 0.0f), -1.0f}, // Center of sphere
      {Eigen::Vector3f(0.5f, 0.0f, 0.0f), -0.5f}, // Halfway to surface

      // Points outside sphere should have positive distance
      {Eigen::Vector3f(2.0f, 0.0f, 0.0f), 1.0f}, // 1 unit beyond surface
      {Eigen::Vector3f(1.5f, 0.0f, 0.0f), 0.5f}, // 0.5 units beyond surface

      // Test some off-axis points
      {Eigen::Vector3f(0.707f, 0.707f, 0.0f), 0.0f}, // On surface at 45 degrees
      {Eigen::Vector3f(0.577f, 0.577f, 0.577f), 0.0f}, // On surface at 45 degrees in 3D
  };

  for (const auto& testCase : testCases) {
    const Eigen::Vector3f& position = testCase.first;
    const float expectedDistance = testCase.second;
    const float sampledDistance = sphereSdf.sample(position);
    const float exactDistance = (position - sphereCenter).norm() - sphereRadius;

    // The sampled distance should be close to the expected/exact distance
    EXPECT_NEAR(sampledDistance, expectedDistance, 0.3f)
        << "Position: [" << position.transpose() << "], " << "Expected: " << expectedDistance
        << ", " << "Sampled: " << sampledDistance << ", " << "Exact: " << exactDistance;

    // The sampled distance should also be reasonably close to the exact analytical distance
    EXPECT_NEAR(sampledDistance, exactDistance, 0.3f)
        << "Position: [" << position.transpose() << "], " << "Analytical: " << exactDistance << ", "
        << "Sampled: " << sampledDistance;
  }
}

TEST_F(SignedDistanceFieldTest, LinearFunctionInterpolation) {
  // Create a small SDF and fill it with a simple linear function: f(x,y,z) = x + 2*y + 3*z
  // This tests that trilinear interpolation works correctly for a known analytical function
  const BoundingBoxf bounds(Eigen::Vector3f(0.0f, 0.0f, 0.0f), Eigen::Vector3f(3.0f, 3.0f, 3.0f));
  const Eigen::Vector3<Index> resolution(4, 4, 4);
  SignedDistanceFieldf linearSdf(bounds, resolution);

  // Fill SDF with linear function: f(x,y,z) = x + 2*y + 3*z
  auto linearFunction = [](const Eigen::Vector3f& pos) -> float {
    return pos.x() + 2.0f * pos.y() + 3.0f * pos.z();
  };

  for (Index i = 0; i < resolution.x(); ++i) {
    for (Index j = 0; j < resolution.y(); ++j) {
      for (Index k = 0; k < resolution.z(); ++k) {
        const Eigen::Vector3f gridPos(
            static_cast<float>(i), static_cast<float>(j), static_cast<float>(k));
        const Eigen::Vector3f worldPos = linearSdf.gridToWorld(gridPos);
        const float value = linearFunction(worldPos);
        linearSdf.set(i, j, k, value);
      }
    }
  }

  // Test interpolation at various non-grid positions
  // For a linear function, trilinear interpolation should be exact
  std::vector<Eigen::Vector3f> testPositions = {
      Eigen::Vector3f(0.5f, 0.5f, 0.5f), // Center of first voxel
      Eigen::Vector3f(1.5f, 1.5f, 1.5f), // Center of grid
      Eigen::Vector3f(2.0f, 2.0f, 2.0f), // Near end but within bounds
      Eigen::Vector3f(0.75f, 1.25f, 2.1f), // Arbitrary position
      Eigen::Vector3f(2.1f, 0.3f, 1.7f), // Another arbitrary position
  };

  for (const auto& position : testPositions) {
    const float sampledValue = linearSdf.sample(position);
    const float exactValue = linearFunction(position);

    // For a linear function, trilinear interpolation should be very accurate
    EXPECT_NEAR(sampledValue, exactValue, 1e-5f)
        << "Position: [" << position.transpose() << "], " << "Expected: " << exactValue << ", "
        << "Sampled: " << sampledValue;
  }
}

TEST_F(SignedDistanceFieldTest, InterpolationContinuity) {
  // Test that interpolation is continuous by sampling along a line and checking
  // that adjacent samples don't have large jumps

  // Create a small SDF with some variation
  const BoundingBoxf bounds(
      Eigen::Vector3f(-1.0f, -1.0f, -1.0f), Eigen::Vector3f(1.0f, 1.0f, 1.0f));
  const Eigen::Vector3<Index> resolution(8, 8, 8);
  SignedDistanceFieldf continuousSdf(bounds, resolution);

  // Fill with a smooth function: distance to origin
  for (Index i = 0; i < resolution.x(); ++i) {
    for (Index j = 0; j < resolution.y(); ++j) {
      for (Index k = 0; k < resolution.z(); ++k) {
        const Eigen::Vector3f gridPos(
            static_cast<float>(i), static_cast<float>(j), static_cast<float>(k));
        const Eigen::Vector3f worldPos = continuousSdf.gridToWorld(gridPos);
        const float distance = worldPos.norm();
        continuousSdf.set(i, j, k, distance);
      }
    }
  }

  // Sample along a line from (-0.5, 0, 0) to (0.5, 0, 0) and check continuity
  const int numSamples = 50;
  float previousValue = continuousSdf.sample(Eigen::Vector3f(-0.5f, 0.0f, 0.0f));

  for (int i = 1; i < numSamples; ++i) {
    const float t = static_cast<float>(i) / static_cast<float>(numSamples - 1);
    const Eigen::Vector3f position = Eigen::Vector3f(-0.5f + t, 0.0f, 0.0f);
    const float currentValue = continuousSdf.sample(position);

    // The change between adjacent samples should be small (continuity test)
    const float valueDifference = std::abs(currentValue - previousValue);
    EXPECT_LT(valueDifference, 0.2f) // Reasonable threshold for smooth function
        << "Large jump detected at position [" << position.transpose() << "], "
        << "Previous: " << previousValue << ", Current: " << currentValue;

    previousValue = currentValue;
  }
}

TEST_F(SignedDistanceFieldTest, SphereSDFGradientAccuracy) {
  // Test analytical gradient accuracy on a proper sphere SDF
  // SDF(p) = |p - center| - radius
  // Gradient(p) = (p - center) / |p - center|  (normalized direction from center)
  const BoundingBoxf bounds(
      Eigen::Vector3f(-2.0f, -2.0f, -2.0f), Eigen::Vector3f(2.0f, 2.0f, 2.0f));
  const Eigen::Vector3<Index> resolution(32, 32, 32);
  SignedDistanceFieldf sphereSdf(bounds, resolution);

  const Eigen::Vector3f sphereCenter(0.0f, 0.0f, 0.0f);
  const float sphereRadius = 1.0f;

  // Fill with sphere SDF
  for (Index i = 0; i < resolution.x(); ++i) {
    for (Index j = 0; j < resolution.y(); ++j) {
      for (Index k = 0; k < resolution.z(); ++k) {
        const Eigen::Vector3f gridPos(
            static_cast<float>(i), static_cast<float>(j), static_cast<float>(k));
        const Eigen::Vector3f worldPos = sphereSdf.gridToWorld(gridPos);
        const float distance = (worldPos - sphereCenter).norm() - sphereRadius;
        sphereSdf.set(i, j, k, distance);
      }
    }
  }

  // Test positions well within the SDF bounds (-2, -2, -2) to (2, 2, 2)
  // but at different distances from the sphere center
  std::vector<Eigen::Vector3f> testPositions = {
      Eigen::Vector3f(1.2f, 0.0f, 0.0f), // Outside sphere, on axis, well within bounds
      Eigen::Vector3f(0.5f, 0.0f, 0.0f), // Inside sphere, on axis
      Eigen::Vector3f(0.3f, 0.4f, 0.0f), // Inside sphere, off-axis (distance = 0.5)
      Eigen::Vector3f(0.4f, 0.3f, 0.2f), // Inside sphere, 3D position
  };

  for (const auto& testPosition : testPositions) {
    const Eigen::Vector3f sdfGradient = sphereSdf.gradient(testPosition);

    // For a sphere SDF, the gradient should point radially outward from center
    const Eigen::Vector3f expectedDirection = (testPosition - sphereCenter).normalized();

    // The gradient should be normalized (magnitude ~1 for well-behaved SDF)
    const float gradientMagnitude = sdfGradient.norm();
    EXPECT_NEAR(gradientMagnitude, 1.0f, 0.2f)
        << "Gradient should be normalized at position [" << testPosition.transpose() << "]"
        << ", magnitude: " << gradientMagnitude;

    // The gradient direction should match the expected radial direction
    const Eigen::Vector3f normalizedGradient = sdfGradient.normalized();
    const float dotProduct = normalizedGradient.dot(expectedDirection);
    EXPECT_GT(dotProduct, 0.9f) // Should be very close to parallel
        << "Gradient direction incorrect at position [" << testPosition.transpose() << "]"
        << ", expected direction: [" << expectedDirection.transpose() << "]" << ", got direction: ["
        << normalizedGradient.transpose() << "]" << ", dot product: " << dotProduct;
  }
}

TEST_F(SignedDistanceFieldTest, AnalyticalGradientVsFiniteDifference) {
  // Test that our new analytical gradient implementation produces results
  // that are very close to finite differences but more accurate

  const BoundingBoxf bounds(
      Eigen::Vector3f(-2.0f, -2.0f, -2.0f), Eigen::Vector3f(2.0f, 2.0f, 2.0f));
  const Eigen::Vector3<Index> resolution(16, 16, 16);
  SignedDistanceFieldf testSdf(bounds, resolution);

  // Fill with a smooth but non-trivial function: f(x,y,z) = x³ + y²*z + sin(x*y)
  auto testFunction = [](const Eigen::Vector3f& pos) -> float {
    return pos.x() * pos.x() * pos.x() + pos.y() * pos.y() * pos.z() + std::sin(pos.x() * pos.y());
  };

  // Analytical gradient: (3x² + y*cos(xy), 2yz + x*cos(xy), y²)
  auto analyticalGradient = [](const Eigen::Vector3f& pos) -> Eigen::Vector3f {
    const float gradX = 3.0f * pos.x() * pos.x() + pos.y() * std::cos(pos.x() * pos.y());
    const float gradY = 2.0f * pos.y() * pos.z() + pos.x() * std::cos(pos.x() * pos.y());
    const float gradZ = pos.y() * pos.y();
    return {gradX, gradY, gradZ};
  };

  // Fill the SDF
  for (Index i = 0; i < resolution.x(); ++i) {
    for (Index j = 0; j < resolution.y(); ++j) {
      for (Index k = 0; k < resolution.z(); ++k) {
        const Eigen::Vector3f gridPos(
            static_cast<float>(i), static_cast<float>(j), static_cast<float>(k));
        const Eigen::Vector3f worldPos = testSdf.gridToWorld(gridPos);
        const float value = testFunction(worldPos);
        testSdf.set(i, j, k, value);
      }
    }
  }

  // Helper function for finite difference gradients
  auto computeFiniteDifferenceGradient = [&](const Eigen::Vector3f& position,
                                             float h) -> Eigen::Vector3f {
    const float gradX = (testSdf.sample<float>(position + Eigen::Vector3f(h, 0, 0)) -
                         testSdf.sample<float>(position - Eigen::Vector3f(h, 0, 0))) /
        (2.0f * h);
    const float gradY = (testSdf.sample<float>(position + Eigen::Vector3f(0, h, 0)) -
                         testSdf.sample<float>(position - Eigen::Vector3f(0, h, 0))) /
        (2.0f * h);
    const float gradZ = (testSdf.sample<float>(position + Eigen::Vector3f(0, 0, h)) -
                         testSdf.sample<float>(position - Eigen::Vector3f(0, 0, h))) /
        (2.0f * h);
    return {gradX, gradY, gradZ};
  };

  // Test positions - use only positions well within bounds (-2, -2, -2) to (2, 2, 2)
  std::vector<Eigen::Vector3f> testPositions = {
      Eigen::Vector3f(0.0f, 0.0f, 0.0f),
      Eigen::Vector3f(0.45f, 0.3f, -0.2f),
      Eigen::Vector3f(-0.7f, 1.2f, 0.8f),
      Eigen::Vector3f(1.1f, -0.9f, -1.3f),
  };

  for (const auto& position : testPositions) {
    // Get our analytical gradient from the SDF
    const Eigen::Vector3f sdfGradient = testSdf.gradient(position);

    // Get the true analytical gradient
    const Eigen::Vector3f trueGradient = analyticalGradient(position);

    // Compare with finite differences
    const float stepSize = 0.01f;
    const Eigen::Vector3f fdGradient = computeFiniteDifferenceGradient(position, stepSize);

    // Our analytical gradient should be reasonably close to the true analytical gradient
    // Note: there will be some discretization error due to the finite grid
    const float analyticalError = (sdfGradient - trueGradient).norm();
    EXPECT_LT(analyticalError, 0.3f) // We have to use a large tolerance here because
                                     // the function is highly non-linear and the grid is coarse
        << "Analytical gradient error too large at position [" << position.transpose() << "]"
        << ", Expected: [" << trueGradient.transpose() << "], Got: [" << sdfGradient.transpose()
        << "], Error: " << analyticalError;

    // Our analytical gradient should generally be at least as accurate as finite differences
    const float fdError = (fdGradient - trueGradient).norm();
    EXPECT_LE(
        analyticalError, fdError * 2.0f) // Allow some margin as both have discretization errors
        << "Analytical gradient should be competitive with finite differences at position ["
        << position.transpose() << "]" << ", Analytical error: " << analyticalError
        << ", FD error: " << fdError;

    // The gradients should also be reasonably close to each other
    const float comparisonError = (sdfGradient - fdGradient).norm();
    EXPECT_LT(comparisonError, 0.05f)
        << "Analytical and finite difference gradients should be close at position ["
        << position.transpose() << "]" << ", Analytical: [" << sdfGradient.transpose() << "], FD: ["
        << fdGradient.transpose() << "], Difference: " << comparisonError;
  }
}

TEST_F(SignedDistanceFieldTest, SampleWithGradientConsistency) {
  // Test that sampleWithGradient() produces identical results to calling
  // sample() and gradient() separately

  const BoundingBoxf bounds(
      Eigen::Vector3f(-1.5f, -1.5f, -1.5f), Eigen::Vector3f(1.5f, 1.5f, 1.5f));
  const Eigen::Vector3<Index> resolution(20, 20, 20);
  SignedDistanceFieldf testSdf(bounds, resolution);

  // Fill with a complex function
  auto complexFunction = [](const Eigen::Vector3f& pos) -> float {
    return pos.x() * pos.y() + pos.y() * pos.z() * pos.z() + std::cos(pos.x() + pos.y() + pos.z());
  };

  for (Index i = 0; i < resolution.x(); ++i) {
    for (Index j = 0; j < resolution.y(); ++j) {
      for (Index k = 0; k < resolution.z(); ++k) {
        const Eigen::Vector3f gridPos(
            static_cast<float>(i), static_cast<float>(j), static_cast<float>(k));
        const Eigen::Vector3f worldPos = testSdf.gridToWorld(gridPos);
        const float value = complexFunction(worldPos);
        testSdf.set(i, j, k, value);
      }
    }
  }

  // Test positions - ensure all positions are within bounds to avoid out-of-bounds behavior changes
  // Bounds are (-1.5, -1.5, -1.5) to (1.5, 1.5, 1.5)
  std::vector<Eigen::Vector3f> testPositions = {
      Eigen::Vector3f(0.0f, 0.0f, 0.0f),
      Eigen::Vector3f(0.5f, -0.3f, 0.7f),
      Eigen::Vector3f(-0.8f, 1.1f, -0.4f),
      Eigen::Vector3f(1.0f, -0.9f, 0.8f), // Changed from (1.2, -1.0, 0.9) to be within bounds
      Eigen::Vector3f(-1.2f, -0.6f, 1.3f), // Changed from (-1.3, -0.6, 1.4) to be within bounds
  };

  for (const auto& position : testPositions) {
    // Get value and gradient separately
    const float separateValue = testSdf.sample(position);
    const Eigen::Vector3f separateGradient = testSdf.gradient(position);

    // Get value and gradient together
    const auto [combinedValue, combinedGradient] = testSdf.sampleWithGradient(position);

    // They should be identical (no tolerance since they use the exact same computation)
    EXPECT_EQ(separateValue, combinedValue)
        << "Value mismatch at position [" << position.transpose() << "]"
        << ", Separate: " << separateValue << ", Combined: " << combinedValue;

    EXPECT_EQ(separateGradient.x(), combinedGradient.x())
        << "X-gradient mismatch at position [" << position.transpose() << "]";
    EXPECT_EQ(separateGradient.y(), combinedGradient.y())
        << "Y-gradient mismatch at position [" << position.transpose() << "]";
    EXPECT_EQ(separateGradient.z(), combinedGradient.z())
        << "Z-gradient mismatch at position [" << position.transpose() << "]";

    // Verify the combined gradient vector is exactly the same
    EXPECT_EQ(separateGradient, combinedGradient)
        << "Gradient vector mismatch at position [" << position.transpose() << "]";
  }
}

} // namespace axel
