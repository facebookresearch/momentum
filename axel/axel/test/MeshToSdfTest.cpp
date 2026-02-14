/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "axel/MeshToSdf.h"

#include <chrono>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>
#include <span>

#include "axel/SignedDistanceField.h"
#include "axel/TriBvh.h"
#include "axel/math/PointTriangleProjection.h"
#include "axel/test/Helper.h"

namespace axel {

class MeshToSdfTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a simple cube mesh for testing
    createCubeMesh();
    createTetrahedronMesh();
  }

  void createCubeMesh() {
    // Define vertices of a unit cube centered at origin as vector
    cubeVertices = {
        Eigen::Vector3f(-0.5f, -0.5f, -0.5f), // 0: ---
        Eigen::Vector3f(0.5f, -0.5f, -0.5f), // 1: +--
        Eigen::Vector3f(0.5f, 0.5f, -0.5f), // 2: ++-
        Eigen::Vector3f(-0.5f, 0.5f, -0.5f), // 3: -+-
        Eigen::Vector3f(-0.5f, -0.5f, 0.5f), // 4: --+
        Eigen::Vector3f(0.5f, -0.5f, 0.5f), // 5: +-+
        Eigen::Vector3f(0.5f, 0.5f, 0.5f), // 6: +++
        Eigen::Vector3f(-0.5f, 0.5f, 0.5f) // 7: -++
    };

    // Define faces (12 triangles, 2 per cube face) as vector
    cubeFaces = {
        // Bottom face (z = -0.5)
        Eigen::Vector3i(0, 1, 2),
        Eigen::Vector3i(0, 2, 3),
        // Top face (z = +0.5)
        Eigen::Vector3i(4, 6, 5),
        Eigen::Vector3i(4, 7, 6),
        // Front face (y = -0.5)
        Eigen::Vector3i(0, 5, 1),
        Eigen::Vector3i(0, 4, 5),
        // Back face (y = +0.5)
        Eigen::Vector3i(2, 7, 3),
        Eigen::Vector3i(2, 6, 7),
        // Left face (x = -0.5)
        Eigen::Vector3i(0, 3, 7),
        Eigen::Vector3i(0, 7, 4),
        // Right face (x = +0.5)
        Eigen::Vector3i(1, 6, 2),
        Eigen::Vector3i(1, 5, 6)};
  }

  void createTetrahedronMesh() {
    // Define vertices of a simple tetrahedron as vector
    tetraVertices = {
        Eigen::Vector3f(0.5f, 0.5f, 0.5f), // Vertex 0
        Eigen::Vector3f(0.5f, -0.5f, -0.5f), // Vertex 1
        Eigen::Vector3f(-0.5f, 0.5f, -0.5f), // Vertex 2
        Eigen::Vector3f(-0.5f, -0.5f, 0.5f) // Vertex 3
    };

    // Define faces (4 triangular faces) as vector
    tetraFaces = {
        Eigen::Vector3i(0, 1, 2), // Face 0
        Eigen::Vector3i(0, 3, 1), // Face 1
        Eigen::Vector3i(0, 2, 3), // Face 2
        Eigen::Vector3i(1, 3, 2) // Face 3
    };
  }

  // Helper function: Compute brute force distance from point to all triangles
  static float bruteForceDistanceToMesh(
      const Eigen::Vector3f& point,
      std::span<const Eigen::Vector3f> vertices,
      std::span<const Eigen::Vector3i> triangles) {
    float minDistance = std::numeric_limits<float>::max();

    for (const auto& triangle : triangles) {
      const auto& v0 = vertices[triangle.x()];
      const auto& v1 = vertices[triangle.y()];
      const auto& v2 = vertices[triangle.z()];

      Eigen::Vector3f closestPoint;
      projectOnTriangle(point, v0, v1, v2, closestPoint);
      const float distance = (point - closestPoint).norm();
      minDistance = std::min(minDistance, distance);
    }

    return minDistance;
  }

  // Helper function: Check if point is inside unit cube
  static bool isInsideUnitCube(const Eigen::Vector3f& point) {
    return point.x() >= -0.5f && point.x() <= 0.5f && point.y() >= -0.5f && point.y() <= 0.5f &&
        point.z() >= -0.5f && point.z() <= 0.5f;
  }

  // Helper function: Exact distance to unit cube surface
  static float exactDistanceToUnitCube(const Eigen::Vector3f& point) {
    // Distance to unit cube centered at origin with half-size 0.5
    Eigen::Vector3f abs_point = point.cwiseAbs();
    Eigen::Vector3f half_size(0.5f, 0.5f, 0.5f);

    if (isInsideUnitCube(point)) {
      // Inside: distance to nearest face
      Eigen::Vector3f dist_to_faces = half_size - abs_point;
      return dist_to_faces.minCoeff();
    } else {
      // Outside: distance to nearest point on cube
      Eigen::Vector3f clamped = abs_point.cwiseMin(half_size);
      return (abs_point - clamped).norm();
    }
  }

  std::vector<Eigen::Vector3f> cubeVertices;
  std::vector<Eigen::Vector3i> cubeFaces;
  std::vector<Eigen::Vector3f> tetraVertices;
  std::vector<Eigen::Vector3i> tetraFaces;
};

// ================================================================================================
// RIGOROUS TESTS FOR EACH ALGORITHM STEP
// ================================================================================================

TEST_F(MeshToSdfTest, Step1_NarrowBandInitialization_ExactDistances) {
  // Test the narrow band initialization with exact distance verification
  const BoundingBoxf bounds(
      Eigen::Vector3f(-1.0f, -1.0f, -1.0f), Eigen::Vector3f(1.0f, 1.0f, 1.0f));
  const Eigen::Vector3<Index> resolution(20, 20, 20);
  SignedDistanceField<float> sdf(bounds, resolution);

  const float bandWidth = 0.3f; // Wide enough to capture several voxel layers
  const float epsilon = 1e-5f; // Machine epsilon tolerance

  // Initialize narrow band
  detail::initializeNarrowBand(
      std::span<const Eigen::Vector3f>(cubeVertices),
      std::span<const Eigen::Vector3i>(cubeFaces),
      sdf,
      bandWidth);

  // Count voxels that have been initialized (values less than max)
  int knownVoxelCount = 0;
  for (Index i = 0; i < resolution.x(); ++i) {
    for (Index j = 0; j < resolution.y(); ++j) {
      for (Index k = 0; k < resolution.z(); ++k) {
        if (sdf.at(i, j, k) < std::numeric_limits<float>::max()) {
          knownVoxelCount++;
        }
      }
    }
  }

  EXPECT_GT(knownVoxelCount, 0) << "Narrow band should contain voxels";

  // Verify every voxel in the narrow band has correct distance
  int verifiedVoxels = 0;
  for (Index i = 0; i < resolution.x(); ++i) {
    for (Index j = 0; j < resolution.y(); ++j) {
      for (Index k = 0; k < resolution.z(); ++k) {
        const float sdfValue = sdf.at(i, j, k);

        // This voxel should be in the narrow band
        const Eigen::Vector3f gridPos(
            static_cast<float>(i), static_cast<float>(j), static_cast<float>(k));
        const Eigen::Vector3f worldPos = sdf.gridToWorld(gridPos);

        // Compute exact distance using brute force
        const float exactDistance = bruteForceDistanceToMesh(
            worldPos,
            std::span<const Eigen::Vector3f>(cubeVertices),
            std::span<const Eigen::Vector3i>(cubeFaces));

        if (exactDistance > bandWidth) {
          continue;
        }

        // Verify the SDF value matches the exact distance
        EXPECT_NEAR(sdfValue, exactDistance, epsilon)
            << "Voxel (" << i << "," << j << "," << k << ") " << "at world pos "
            << worldPos.transpose() << " has SDF value " << sdfValue << " but exact distance is "
            << exactDistance;

        // Verify it's within the narrow band
        EXPECT_LE(exactDistance, bandWidth + epsilon) << "Voxel should be within narrow band width";

        verifiedVoxels++;
      }
    }
  }

  EXPECT_GT(verifiedVoxels, 100) << "Should have verified many narrow band voxels";
  std::cout << "Step 1: Verified " << verifiedVoxels << " narrow band voxels with exact distances"
            << std::endl;
}

TEST_F(MeshToSdfTest, Step2_FastMarchingPropagation_NearSurfaceAccuracy) {
  // Test that areas close to surface still have accurate distance-to-mesh values
  const BoundingBoxf bounds(
      Eigen::Vector3f(-1.0f, -1.0f, -1.0f), Eigen::Vector3f(1.0f, 1.0f, 1.0f));
  const Eigen::Vector3<Index> resolution(20, 20, 20);
  SignedDistanceField<float> sdf(bounds, resolution);

  const float bandWidth = 1.5f * sdf.voxelSize().norm(); // Wide enough to capture several voxel
  const float maxError = 0.05f; // Stricter tolerance for near-surface regions

  // Step 1: Initialize narrow band
  detail::initializeNarrowBand(
      std::span<const Eigen::Vector3f>(cubeVertices),
      std::span<const Eigen::Vector3i>(cubeFaces),
      sdf,
      bandWidth);

  // Step 2: Fast marching propagation
  detail::fastMarchingPropagate(sdf);

  // Verify distances are accurate near the surface
  int nearSurfaceVoxels = 0;
  int accurateNearSurfaceVoxels = 0;

  for (Index i = 0; i < resolution.x(); ++i) {
    for (Index j = 0; j < resolution.y(); ++j) {
      for (Index k = 0; k < resolution.z(); ++k) {
        const float sdfValue = sdf.at(i, j, k);

        // All voxels should now have finite values
        EXPECT_TRUE(std::isfinite(sdfValue))
            << "All voxels should have finite distances after fast marching";

        const Eigen::Vector3f gridPos(
            static_cast<float>(i), static_cast<float>(j), static_cast<float>(k));
        const Eigen::Vector3f worldPos = sdf.gridToWorld(gridPos);

        // Compute exact distance using brute force
        const float exactDistance = bruteForceDistanceToMesh(
            worldPos,
            std::span<const Eigen::Vector3f>(cubeVertices),
            std::span<const Eigen::Vector3i>(cubeFaces));

        // Only test voxels that are near the surface
        if (exactDistance <= bandWidth) {
          nearSurfaceVoxels++;

          // SDF should be very close to exact distance near the surface
          const float error = std::abs(sdfValue - exactDistance);
          if (error <= maxError) {
            accurateNearSurfaceVoxels++;
          }

          EXPECT_NEAR(sdfValue, exactDistance, maxError)
              << "Near-surface voxel at " << worldPos.transpose()
              << " should have accurate distance. SDF: " << sdfValue << ", Exact: " << exactDistance
              << ", Error: " << error;
        }

        // Basic sanity checks for all voxels
        EXPECT_GE(sdfValue, 0.0f) << "Distance should be positive (no signs applied yet)";
        EXPECT_LT(sdfValue, 5.0f) << "Distance should be reasonable for this grid";
      }
    }
  }

  const float nearSurfaceAccuracy =
      static_cast<float>(accurateNearSurfaceVoxels) / nearSurfaceVoxels;
  EXPECT_GT(nearSurfaceAccuracy, 0.95f) << "Near-surface distance accuracy should be very high";
  EXPECT_GT(nearSurfaceVoxels, 50) << "Should have tested many near-surface voxels";

  std::cout << "Step 2 (Near-Surface): Verified " << nearSurfaceVoxels << " near-surface voxels, "
            << nearSurfaceAccuracy * 100.0f << "% accurate within " << maxError << std::endl;
}

TEST_F(MeshToSdfTest, Step2_FastMarchingPropagation_EikonalEquation) {
  // Test that fast marching satisfies the Eikonal equation |grad(f)| ≈ 1 away from surface
  const BoundingBoxf bounds(
      Eigen::Vector3f(-1.2f, -1.2f, -1.2f), Eigen::Vector3f(1.2f, 1.2f, 1.2f));
  const Eigen::Vector3<Index> resolution(24, 24, 24);
  SignedDistanceField<float> sdf(bounds, resolution);

  const float bandWidth = 0.15f;
  const float farFromSurfaceThreshold = 0.4f; // Test Eikonal equation beyond this distance
  const float gradientTolerance = 0.25f; // Allow some deviation from |grad| = 1

  // Step 1: Initialize narrow band
  detail::initializeNarrowBand(
      std::span<const Eigen::Vector3f>(cubeVertices),
      std::span<const Eigen::Vector3i>(cubeFaces),
      sdf,
      bandWidth);

  // Step 2: Fast marching propagation
  detail::fastMarchingPropagate(sdf);

  // Verify Eikonal equation |grad(f)| ≈ 1 away from surface
  int farFromSurfaceVoxels = 0;
  int validGradientVoxels = 0;

  for (Index i = 1; i < resolution.x() - 1; ++i) { // Avoid boundaries for gradient computation
    for (Index j = 1; j < resolution.y() - 1; ++j) {
      for (Index k = 1; k < resolution.z() - 1; ++k) {
        const Eigen::Vector3f gridPos(
            static_cast<float>(i), static_cast<float>(j), static_cast<float>(k));
        const Eigen::Vector3f worldPos = sdf.gridToWorld(gridPos);

        // Compute exact distance to determine if we're far from surface
        const float exactDistance = bruteForceDistanceToMesh(
            worldPos,
            std::span<const Eigen::Vector3f>(cubeVertices),
            std::span<const Eigen::Vector3i>(cubeFaces));

        // Only test Eikonal equation for voxels far from surface
        if (exactDistance >= farFromSurfaceThreshold) {
          farFromSurfaceVoxels++;

          // Compute gradient magnitude using SDF's analytical gradient
          const Eigen::Vector3f gradient = sdf.gradient(worldPos);
          const float gradientMagnitude = gradient.norm();

          // For the Eikonal equation |∇u| = 1, gradient magnitude should be close to 1
          if (std::abs(gradientMagnitude - 1.0f) <= gradientTolerance) {
            validGradientVoxels++;
          }

          // More lenient check - gradient magnitude should be reasonable
          EXPECT_GT(gradientMagnitude, 0.5f)
              << "Gradient magnitude too small at " << worldPos.transpose()
              << ", magnitude: " << gradientMagnitude;
          EXPECT_LT(gradientMagnitude, 2.0f)
              << "Gradient magnitude too large at " << worldPos.transpose()
              << ", magnitude: " << gradientMagnitude;
        }
      }
    }
  }

  const float eikonalAccuracy = static_cast<float>(validGradientVoxels) / farFromSurfaceVoxels;
  EXPECT_GT(eikonalAccuracy, 0.8f)
      << "Eikonal equation should be satisfied for most far-surface voxels";
  EXPECT_GT(farFromSurfaceVoxels, 100) << "Should have tested many far-from-surface voxels";

  std::cout << "Step 2 (Eikonal): Verified " << farFromSurfaceVoxels << " far-surface voxels, "
            << eikonalAccuracy * 100.0f << "% satisfy |grad| ≈ 1 within tolerance "
            << gradientTolerance << std::endl;
}

TEST_F(MeshToSdfTest, Step3_SignDetermination_InsideOutsideAccuracy) {
  // Test sign determination: inside cube should be negative, outside positive
  // Only test voxels that are clearly inside or outside (not on/near boundary)
  const BoundingBoxf bounds(
      Eigen::Vector3f(-1.0f, -1.0f, -1.0f), Eigen::Vector3f(1.0f, 1.0f, 1.0f));
  const Eigen::Vector3<Index> resolution(12, 12, 12);

  MeshToSdfConfigf config;

  // Generate complete SDF
  const auto sdf = meshToSdf<float>(
      std::span<const Eigen::Vector3f>(cubeVertices),
      std::span<const Eigen::Vector3i>(cubeFaces),
      bounds,
      resolution,
      config);

  // Distance threshold to exclude boundary voxels from accuracy calculation
  const float boundaryThreshold = 0.05f; // Exclude voxels within 0.05 units of surface

  int clearlyInsideVoxels = 0;
  int clearlyOutsideVoxels = 0;
  int correctInsideVoxels = 0;
  int correctOutsideVoxels = 0;
  int boundaryVoxels = 0;

  for (Index i = 0; i < resolution.x(); ++i) {
    for (Index j = 0; j < resolution.y(); ++j) {
      for (Index k = 0; k < resolution.z(); ++k) {
        const float sdfValue = sdf.at(i, j, k);

        const Eigen::Vector3f gridPos(
            static_cast<float>(i), static_cast<float>(j), static_cast<float>(k));
        const Eigen::Vector3f worldPos = sdf.gridToWorld(gridPos);

        // Compute exact distance to surface to determine if this is a boundary voxel
        const float exactDistance = exactDistanceToUnitCube(worldPos);
        const bool shouldBeInside = isInsideUnitCube(worldPos);
        const bool sdfSaysInside = sdfValue < 0.0f;

        // Skip voxels that are very close to the boundary
        if (exactDistance < boundaryThreshold) {
          boundaryVoxels++;
          continue;
        }

        // Only test voxels that are clearly inside or outside
        if (shouldBeInside) {
          clearlyInsideVoxels++;
          if (sdfSaysInside) {
            correctInsideVoxels++;
          } else {
            std::cout << "ERROR: Clearly inside voxel " << worldPos.transpose()
                      << " (dist=" << exactDistance << ") has positive SDF value " << sdfValue
                      << std::endl;
          }
        } else {
          clearlyOutsideVoxels++;
          if (!sdfSaysInside) {
            correctOutsideVoxels++;
          } else {
            std::cout << "ERROR: Clearly outside voxel " << worldPos.transpose()
                      << " (dist=" << exactDistance << ") has negative SDF value " << sdfValue
                      << std::endl;
          }
        }
      }
    }
  }

  const float insideAccuracy = clearlyInsideVoxels > 0
      ? static_cast<float>(correctInsideVoxels) / clearlyInsideVoxels
      : 1.0f;
  const float outsideAccuracy = clearlyOutsideVoxels > 0
      ? static_cast<float>(correctOutsideVoxels) / clearlyOutsideVoxels
      : 1.0f;

  std::cout << "Step 3: Clearly inside accuracy: " << insideAccuracy * 100.0f << "% ("
            << correctInsideVoxels << "/" << clearlyInsideVoxels << ")" << std::endl;
  std::cout << "Step 3: Clearly outside accuracy: " << outsideAccuracy * 100.0f << "% ("
            << correctOutsideVoxels << "/" << clearlyOutsideVoxels << ")" << std::endl;
  std::cout << "Step 3: Excluded " << boundaryVoxels << " boundary voxels within "
            << boundaryThreshold << " units of surface" << std::endl;

  // These should be very high accuracy for a simple cube (excluding boundary cases)
  EXPECT_GT(insideAccuracy, 0.95f) << "Clearly inside classification should be very accurate";
  EXPECT_GT(outsideAccuracy, 0.95f) << "Clearly outside classification should be very accurate";

  EXPECT_GT(clearlyInsideVoxels, 0) << "Should have some clearly inside voxels";
  EXPECT_GT(clearlyOutsideVoxels, 0) << "Should have some clearly outside voxels";
}

TEST_F(MeshToSdfTest, Step3_SignDetermination_WindingNumbers) {
  // Test sign determination using winding numbers
  const BoundingBoxf bounds(
      Eigen::Vector3f(-1.0f, -1.0f, -1.0f), Eigen::Vector3f(1.0f, 1.0f, 1.0f));
  const Eigen::Vector3<Index> resolution(10, 10, 10);

  MeshToSdfConfigf config;
  config.narrowBandWidth = 2.0f;

  // Generate complete SDF
  const auto sdf = meshToSdf<float>(
      std::span<const Eigen::Vector3f>(cubeVertices),
      std::span<const Eigen::Vector3i>(cubeFaces),
      bounds,
      resolution,
      config);

  int correctSigns = 0;
  int totalVoxels = 0;

  for (Index i = 0; i < resolution.x(); ++i) {
    for (Index j = 0; j < resolution.y(); ++j) {
      for (Index k = 0; k < resolution.z(); ++k) {
        const float sdfValue = sdf.at(i, j, k);

        const Eigen::Vector3f gridPos(
            static_cast<float>(i), static_cast<float>(j), static_cast<float>(k));
        const Eigen::Vector3f worldPos = sdf.gridToWorld(gridPos);

        const bool shouldBeInside = isInsideUnitCube(worldPos);
        const bool sdfSaysInside = sdfValue < 0.0f;

        if (shouldBeInside == sdfSaysInside) {
          correctSigns++;
        }
        totalVoxels++;
      }
    }
  }

  const float accuracy = static_cast<float>(correctSigns) / totalVoxels;
  std::cout << "Step 3 (Winding): Sign accuracy: " << accuracy * 100.0f << "% (" << correctSigns
            << "/" << totalVoxels << ")" << std::endl;

  EXPECT_GT(accuracy, 0.95f) << "Winding number sign determination should be very accurate";
}

TEST_F(MeshToSdfTest, IntegratedTest_CubeSDFProperties) {
  // Test the complete SDF has the expected mathematical properties
  const BoundingBoxf bounds(
      Eigen::Vector3f(-1.0f, -1.0f, -1.0f), Eigen::Vector3f(1.0f, 1.0f, 1.0f));
  const Eigen::Vector3<Index> resolution(16, 16, 16);

  MeshToSdfConfigf config;
  config.narrowBandWidth = 3.0f;

  const auto sdf = meshToSdf<float>(
      std::span<const Eigen::Vector3f>(cubeVertices),
      std::span<const Eigen::Vector3i>(cubeFaces),
      bounds,
      resolution,
      config);

  // Test center (should be inside with distance ~0.5)
  const float centerValue = sdf.sample(Eigen::Vector3f(0.0f, 0.0f, 0.0f));
  EXPECT_LT(centerValue, 0.0f) << "Center should be inside (negative)";
  EXPECT_NEAR(std::abs(centerValue), 0.5f, 0.1f) << "Center distance should be ~0.5";

  // Test corners (should be outside with distance 0)
  const float cornerValue = sdf.sample(Eigen::Vector3f(0.5f, 0.5f, 0.5f));
  EXPECT_NEAR(cornerValue, 0.0f, 0.05f) << "Corner should be on surface";

  // Test outside point
  const float outsideValue = sdf.sample(Eigen::Vector3f(1.0f, 1.0f, 1.0f));
  EXPECT_GT(outsideValue, 0.0f) << "Outside point should be positive";
  EXPECT_NEAR(outsideValue, std::sqrt(3.0f) * 0.5f, 0.15f) << "Outside distance should be correct";

  // Test face centers (should be exactly 0 or very close)
  const float faceValue = sdf.sample(Eigen::Vector3f(0.5f, 0.0f, 0.0f));
  EXPECT_NEAR(faceValue, 0.0f, 0.05f) << "Face center should be on surface";
}

TEST_F(MeshToSdfTest, VoxelSizeOverload_UniformVoxels) {
  // Test that the voxel-size overload produces uniform (isotropic) voxels
  const float voxelSize = 0.1f;

  const auto sdf = meshToSdf<float>(
      std::span<const Eigen::Vector3f>(cubeVertices),
      std::span<const Eigen::Vector3i>(cubeFaces),
      voxelSize);

  const auto actualVoxelSize = sdf.voxelSize();

  // All three voxel dimensions should be exactly equal to the requested voxel size
  EXPECT_NEAR(actualVoxelSize.x(), voxelSize, 1e-6f) << "X voxel size should match requested size";
  EXPECT_NEAR(actualVoxelSize.y(), voxelSize, 1e-6f) << "Y voxel size should match requested size";
  EXPECT_NEAR(actualVoxelSize.z(), voxelSize, 1e-6f) << "Z voxel size should match requested size";

  // Resolution should be consistent with the voxel size
  const auto& bounds = sdf.bounds();
  const auto& resolution = sdf.resolution();
  const auto extent = bounds.max() - bounds.min();

  for (int i = 0; i < 3; ++i) {
    const float expectedExtent = voxelSize * static_cast<float>(resolution[i]);
    EXPECT_NEAR(extent[i], expectedExtent, 1e-5f)
        << "Extent[" << i << "] should be voxelSize * resolution";
  }

  // The SDF should contain the mesh (test center is inside)
  const float centerValue = sdf.sample(Eigen::Vector3f(0.0f, 0.0f, 0.0f));
  EXPECT_LT(centerValue, 0.0f) << "Center of cube should be inside (negative)";
}

TEST_F(MeshToSdfTest, VoxelSizeOverload_AnisotropicBboxIsotropicVoxels) {
  // Create an elongated mesh (anisotropic bounding box) and verify
  // voxels are still isotropic
  std::vector<Eigen::Vector3f> elongatedVertices;
  elongatedVertices.reserve(cubeVertices.size());
  for (const auto& v : cubeVertices) {
    // Scale x by 3 to create a 3:1:1 aspect ratio
    elongatedVertices.emplace_back(v.x() * 3.0f, v.y(), v.z());
  }

  const float voxelSize = 0.1f;

  const auto sdf = meshToSdf<float>(
      std::span<const Eigen::Vector3f>(elongatedVertices),
      std::span<const Eigen::Vector3i>(cubeFaces),
      voxelSize);

  const auto actualVoxelSize = sdf.voxelSize();

  // All three dimensions should still be the same voxel size
  EXPECT_NEAR(actualVoxelSize.x(), voxelSize, 1e-6f);
  EXPECT_NEAR(actualVoxelSize.y(), voxelSize, 1e-6f);
  EXPECT_NEAR(actualVoxelSize.z(), voxelSize, 1e-6f);

  // X resolution should be approximately 3x larger than Y and Z resolutions
  const auto& resolution = sdf.resolution();
  EXPECT_GT(resolution.x(), resolution.y() * 2)
      << "X resolution should be much larger for elongated mesh";
  EXPECT_NEAR(resolution.y(), resolution.z(), 1)
      << "Y and Z resolutions should be similar for a cube cross-section";

  std::cout << "Anisotropic mesh resolution: " << resolution.transpose() << std::endl;
}

TEST_F(MeshToSdfTest, VoxelSizeOverload_CustomPadding) {
  // Test that custom padding configuration works
  const float voxelSize = 0.1f;

  MeshToSdfPaddingf padding;
  padding.fractional = 0.2f;
  padding.absolute = 0.3f;

  const auto sdf = meshToSdf<float>(
      std::span<const Eigen::Vector3f>(cubeVertices),
      std::span<const Eigen::Vector3i>(cubeFaces),
      voxelSize,
      padding);

  const auto& bounds = sdf.bounds();

  // Mesh extent is 1.0 per axis, so fractional padding = 0.2 * 1.0 = 0.2
  // Absolute padding = 0.3
  // max(0.2, 0.3) = 0.3 per side
  // Total extent should be approximately 1.0 + 2 * 0.3 = 1.6 per axis
  // (then adjusted to fit exact voxel sizes)
  const auto extent = bounds.max() - bounds.min();
  for (int i = 0; i < 3; ++i) {
    EXPECT_GT(extent[i], 1.5f) << "Padded extent[" << i
                               << "] should be > 1.5 with absolute padding 0.3";
  }

  // Voxels should still be isotropic
  const auto actualVoxelSize = sdf.voxelSize();
  EXPECT_NEAR(actualVoxelSize.x(), voxelSize, 1e-6f);
  EXPECT_NEAR(actualVoxelSize.y(), voxelSize, 1e-6f);
  EXPECT_NEAR(actualVoxelSize.z(), voxelSize, 1e-6f);
}

TEST_F(MeshToSdfTest, InPlaceOverload_MatchesReturnOverload) {
  // Test that the in-place overload produces the same result as the return overload
  const BoundingBoxf bounds(
      Eigen::Vector3f(-1.0f, -1.0f, -1.0f), Eigen::Vector3f(1.0f, 1.0f, 1.0f));
  const Eigen::Vector3<Index> resolution(12, 12, 12);

  MeshToSdfConfigf config;

  // Generate SDF using return overload
  const auto sdf1 = meshToSdf<float>(
      std::span<const Eigen::Vector3f>(cubeVertices),
      std::span<const Eigen::Vector3i>(cubeFaces),
      bounds,
      resolution,
      config);

  // Generate SDF using in-place overload
  SignedDistanceField<float> sdf2(bounds, resolution);
  meshToSdf<float>(
      std::span<const Eigen::Vector3f>(cubeVertices),
      std::span<const Eigen::Vector3i>(cubeFaces),
      sdf2,
      config);

  // Compare values at all grid points
  for (Index i = 0; i < resolution.x(); ++i) {
    for (Index j = 0; j < resolution.y(); ++j) {
      for (Index k = 0; k < resolution.z(); ++k) {
        EXPECT_NEAR(sdf1.at(i, j, k), sdf2.at(i, j, k), 1e-6f)
            << "In-place and return overloads should produce identical results at (" << i << ","
            << j << "," << k << ")";
      }
    }
  }
}

} // namespace axel
