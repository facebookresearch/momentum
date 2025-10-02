/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <momentum/rasterizer/camera.h>
#include <momentum/rasterizer/fwd.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

using namespace momentum::rasterizer;

class CameraTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create test intrinsics models (float precision for existing tests)
    pinholeIntrinsics = std::make_shared<PinholeIntrinsicsModel>(640, 480, 500.0f, 500.0f);

    OpenCVDistortionParametersT<float> distortionParams;
    distortionParams.k1 = 0.1f;
    distortionParams.k2 = -0.05f;
    distortionParams.p1 = 0.01f;
    distortionParams.p2 = 0.02f;

    opencvIntrinsics = std::make_shared<OpenCVIntrinsicsModel>(
        640, 480, 500.0f, 500.0f, 320.0f, 240.0f, distortionParams);

    // Create double precision intrinsics models for Jacobian tests
    pinholeIntrinsicsDouble = std::make_shared<PinholeIntrinsicsModeld>(640, 480, 500.0, 500.0);

    OpenCVDistortionParametersT<double> distortionParamsDouble;
    distortionParamsDouble.k1 = 0.1;
    distortionParamsDouble.k2 = -0.05;
    distortionParamsDouble.p1 = 0.01;
    distortionParamsDouble.p2 = 0.02;

    opencvIntrinsicsDouble = std::make_shared<OpenCVIntrinsicsModeld>(
        640, 480, 500.0, 500.0, 320.0, 240.0, distortionParamsDouble);
  }

  // Helper function to create test cameras with different intrinsics models
  struct TestCameraInfo {
    std::shared_ptr<IntrinsicsModel> intrinsics;
    std::string name;
    bool hasDistortion;
  };

  std::vector<TestCameraInfo> createTestCameras() {
    std::vector<TestCameraInfo> cameras;

    // Pinhole camera
    cameras.push_back({pinholeIntrinsics, "Pinhole", false});

    // OpenCV camera with distortion
    cameras.push_back({opencvIntrinsics, "OpenCV_with_distortion", true});

    // OpenCV camera with zero distortion
    OpenCVDistortionParametersT<float> zeroDistortion; // All zeros by default
    auto opencvZeroDistortion = std::make_shared<OpenCVIntrinsicsModel>(
        640, 480, 500.0f, 500.0f, 320.0f, 240.0f, zeroDistortion);
    cameras.push_back({opencvZeroDistortion, "OpenCV_zero_distortion", false});

    return cameras;
  }

  std::shared_ptr<PinholeIntrinsicsModel> pinholeIntrinsics;
  std::shared_ptr<OpenCVIntrinsicsModel> opencvIntrinsics;
  std::shared_ptr<PinholeIntrinsicsModeld> pinholeIntrinsicsDouble;
  std::shared_ptr<OpenCVIntrinsicsModeld> opencvIntrinsicsDouble;
};

// Test PinholeIntrinsicsModel construction and basic properties
TEST_F(CameraTest, PinholeIntrinsicsConstruction) {
  EXPECT_EQ(pinholeIntrinsics->imageWidth(), 640);
  EXPECT_EQ(pinholeIntrinsics->imageHeight(), 480);
  EXPECT_FLOAT_EQ(pinholeIntrinsics->fx(), 500.0f);
  EXPECT_FLOAT_EQ(pinholeIntrinsics->fy(), 500.0f);
  // Cast to concrete type to access cx() and cy() methods
  auto pinholeConcrete = std::dynamic_pointer_cast<const PinholeIntrinsicsModel>(pinholeIntrinsics);
  ASSERT_NE(pinholeConcrete, nullptr);
  EXPECT_FLOAT_EQ(pinholeIntrinsics->cx(), 320.0f); // imageWidth / 2
  EXPECT_FLOAT_EQ(pinholeIntrinsics->cy(), 240.0f); // imageHeight / 2
}

// Test OpenCVIntrinsicsModel construction and basic properties
TEST_F(CameraTest, OpenCVIntrinsicsConstruction) {
  EXPECT_EQ(opencvIntrinsics->imageWidth(), 640);
  EXPECT_EQ(opencvIntrinsics->imageHeight(), 480);
  EXPECT_FLOAT_EQ(opencvIntrinsics->fx(), 500.0f);
  EXPECT_FLOAT_EQ(opencvIntrinsics->fy(), 500.0f);
  EXPECT_FLOAT_EQ(opencvIntrinsics->cx(), 320.0f);
  EXPECT_FLOAT_EQ(opencvIntrinsics->cy(), 240.0f);
}

// Test PinholeIntrinsicsModel upsampling
TEST_F(CameraTest, PinholeIntrinsicsUpsample) {
  auto upsampled = pinholeIntrinsics->upsample(2.0f);
  EXPECT_EQ(upsampled->imageWidth(), 1280);
  EXPECT_EQ(upsampled->imageHeight(), 960);
  EXPECT_FLOAT_EQ(upsampled->fx(), 1000.0f);
  EXPECT_FLOAT_EQ(upsampled->fy(), 1000.0f);
  // Cast to concrete type to access cx() and cy() methods
  auto upsampledConcrete = std::dynamic_pointer_cast<const PinholeIntrinsicsModel>(upsampled);
  ASSERT_NE(upsampledConcrete, nullptr);
  // Using correct camera center formula: cx = (old_cx + 0.5) * scale - 0.5
  // (320.0 + 0.5) * 2.0 - 0.5 = 640.5
  EXPECT_FLOAT_EQ(upsampledConcrete->cx(), 640.5f);
  EXPECT_FLOAT_EQ(upsampledConcrete->cy(), 480.5f);
}

// Test PinholeIntrinsicsModel downsampling
TEST_F(CameraTest, PinholeIntrinsicsDownsample) {
  auto downsampled = pinholeIntrinsics->downsample(2.0f);
  EXPECT_EQ(downsampled->imageWidth(), 320);
  EXPECT_EQ(downsampled->imageHeight(), 240);
  EXPECT_FLOAT_EQ(downsampled->fx(), 250.0f);
  EXPECT_FLOAT_EQ(downsampled->fy(), 250.0f);
  // Cast to concrete type to access cx() and cy() methods
  auto downsampledConcrete = std::dynamic_pointer_cast<const PinholeIntrinsicsModel>(downsampled);
  ASSERT_NE(downsampledConcrete, nullptr);
  // Using correct camera center formula: cx = (old_cx + 0.5) * scale - 0.5
  // (320.0 + 0.5) * 0.5 - 0.5 = 159.75
  EXPECT_FLOAT_EQ(downsampledConcrete->cx(), 159.75f);
  EXPECT_FLOAT_EQ(downsampledConcrete->cy(), 119.75f);
}

// Test OpenCVIntrinsicsModel upsampling
TEST_F(CameraTest, OpenCVIntrinsicsUpsample) {
  auto upsampled = opencvIntrinsics->upsample(2.0f);
  EXPECT_EQ(upsampled->imageWidth(), 1280);
  EXPECT_EQ(upsampled->imageHeight(), 960);
  EXPECT_FLOAT_EQ(upsampled->fx(), 1000.0f);
  EXPECT_FLOAT_EQ(upsampled->fy(), 1000.0f);
  // Cast to concrete type to access cx() and cy() methods
  auto upsampledConcrete = std::dynamic_pointer_cast<const OpenCVIntrinsicsModel>(upsampled);
  ASSERT_NE(upsampledConcrete, nullptr);
  // Using correct camera center formula: cx = (old_cx + 0.5) * scale - 0.5
  // (320.0 + 0.5) * 2.0 - 0.5 = 640.5
  EXPECT_FLOAT_EQ(upsampledConcrete->cx(), 640.5f);
  EXPECT_FLOAT_EQ(upsampledConcrete->cy(), 480.5f);
}

// Test OpenCVIntrinsicsModel downsampling
TEST_F(CameraTest, OpenCVIntrinsicsDownsample) {
  auto downsampled = opencvIntrinsics->downsample(2.0f);
  EXPECT_EQ(downsampled->imageWidth(), 320);
  EXPECT_EQ(downsampled->imageHeight(), 240);
  EXPECT_FLOAT_EQ(downsampled->fx(), 250.0f);
  EXPECT_FLOAT_EQ(downsampled->fy(), 250.0f);
  // Cast to concrete type to access cx() and cy() methods
  auto downsampledConcrete = std::dynamic_pointer_cast<const OpenCVIntrinsicsModel>(downsampled);
  ASSERT_NE(downsampledConcrete, nullptr);
  // Using correct camera center formula: cx = (old_cx + 0.5) * scale - 0.5
  // (320.0 + 0.5) * 0.5 - 0.5 = 159.75
  EXPECT_FLOAT_EQ(downsampledConcrete->cx(), 159.75f);
  EXPECT_FLOAT_EQ(downsampledConcrete->cy(), 119.75f);
}

// Test Camera construction with default parameters
TEST_F(CameraTest, CameraDefaultConstruction) {
  Camera camera;
  // Default camera should have identity transform
  const auto& eyeFromWorld = camera.eyeFromWorld();
  EXPECT_TRUE(eyeFromWorld.matrix().isApprox(Eigen::Matrix4f::Identity()));
}

// Test Camera construction with intrinsics
TEST_F(CameraTest, CameraConstructionWithIntrinsics) {
  Camera camera(pinholeIntrinsics);

  EXPECT_EQ(camera.imageWidth(), 640);
  EXPECT_EQ(camera.imageHeight(), 480);
  EXPECT_FLOAT_EQ(camera.fx(), 500.0f);
  EXPECT_FLOAT_EQ(camera.fy(), 500.0f);
  // Camera class doesn't have cx() and cy() methods - access through intrinsics
  auto pinholeConcrete =
      std::dynamic_pointer_cast<const PinholeIntrinsicsModel>(camera.intrinsicsModel());
  ASSERT_NE(pinholeConcrete, nullptr);
  EXPECT_FLOAT_EQ(pinholeConcrete->cx(), 320.0f);
  EXPECT_FLOAT_EQ(pinholeConcrete->cy(), 240.0f);

  EXPECT_EQ(camera.intrinsicsModel(), pinholeIntrinsics);
}

// Test Camera construction with transform
TEST_F(CameraTest, CameraConstructionWithTransform) {
  Eigen::Transform<float, 3, Eigen::Affine> transform =
      Eigen::Transform<float, 3, Eigen::Affine>(Eigen::Translation3f(1.0f, 2.0f, 3.0f));

  Camera camera(pinholeIntrinsics, transform);

  const auto& eyeFromWorld = camera.eyeFromWorld();
  EXPECT_TRUE(eyeFromWorld.isApprox(transform));

  auto worldFromEye = camera.worldFromEye();
  EXPECT_TRUE(worldFromEye.isApprox(transform.inverse()));
}

// Test Camera setEyeFromWorld
TEST_F(CameraTest, CameraSetEyeFromWorld) {
  Camera camera(pinholeIntrinsics);

  Eigen::Transform<float, 3, Eigen::Affine> newTransform =
      Eigen::Transform<float, 3, Eigen::Affine>(Eigen::Translation3f(5.0f, 6.0f, 7.0f));

  camera.setEyeFromWorld(newTransform);

  EXPECT_TRUE(camera.eyeFromWorld().isApprox(newTransform));
}

// Test Camera lookAt functionality - basic case
TEST_F(CameraTest, CameraLookAtBasic) {
  Camera camera(pinholeIntrinsics);

  Eigen::Vector3f position(0.0f, 0.0f, 5.0f);
  Eigen::Vector3f target(0.0f, 0.0f, 0.0f);
  Eigen::Vector3f up(0.0f, 1.0f, 0.0f);

  Camera newCamera = camera.lookAt(position, target, up);

  // Camera should be positioned at (0, 0, 5) looking towards origin
  Eigen::Vector3f cameraPos = newCamera.worldFromEye().translation();
  EXPECT_TRUE(cameraPos.isApprox(position, 1e-5f));

  // Z-axis should point towards target (negative Z in camera space)
  Eigen::Vector3f zAxis = newCamera.worldFromEye().linear().col(2);
  Eigen::Vector3f expectedZ = (target - position).normalized();
  EXPECT_TRUE(zAxis.isApprox(expectedZ, 1e-5f));
}

// Test Camera lookAt with same position and target
TEST_F(CameraTest, CameraLookAtSamePositionTarget) {
  Camera camera(pinholeIntrinsics);

  Eigen::Vector3f position(1.0f, 2.0f, 3.0f);

  Camera newCamera = camera.lookAt(position, position);

  // Should return the original camera unchanged
  EXPECT_TRUE(newCamera.eyeFromWorld().isApprox(camera.eyeFromWorld()));
}

// Test Camera lookAt with parallel up vector
TEST_F(CameraTest, CameraLookAtParallelUp) {
  Camera camera(pinholeIntrinsics);

  Eigen::Vector3f position(0.0f, 0.0f, 5.0f);
  Eigen::Vector3f target(0.0f, 0.0f, 0.0f);
  Eigen::Vector3f up(0.0f, 0.0f, -1.0f); // Parallel to view direction

  Camera newCamera = camera.lookAt(position, target, up);

  // Should still work and position camera correctly
  Eigen::Vector3f cameraPos = newCamera.worldFromEye().translation();
  EXPECT_TRUE(cameraPos.isApprox(position, 1e-5f));
}

// Test Camera framePoints with empty points
TEST_F(CameraTest, CameraFramePointsEmpty) {
  Camera camera(pinholeIntrinsics);
  std::vector<Eigen::Vector3f> emptyPoints;

  Camera newCamera = camera.framePoints(emptyPoints);

  // Should return the original camera unchanged
  EXPECT_TRUE(newCamera.eyeFromWorld().isApprox(camera.eyeFromWorld()));
}

// Test Camera framePoints with single point - verify proper framing
TEST_F(CameraTest, CameraFramePointsSingle) {
  Camera camera(pinholeIntrinsics);
  std::vector<Eigen::Vector3f> points = {Eigen::Vector3f(1.0f, 2.0f, 3.0f)};

  const float minZ = 0.5f;
  const float edgePadding = 0.1f;
  Camera newCamera = camera.framePoints(points, minZ, edgePadding);

  // Camera should be adjusted to frame the point
  EXPECT_FALSE(newCamera.eyeFromWorld().isApprox(camera.eyeFromWorld()));

  // Verify the point is properly framed
  for (const auto& worldPoint : points) {
    // Transform to eye space
    Eigen::Vector3f eyePoint = newCamera.eyeFromWorld() * worldPoint;

    // Check minimum Z distance
    EXPECT_GE(eyePoint.z(), minZ);

    // Project to image coordinates using pinhole model
    // Camera class doesn't have cx() and cy() methods - use image center calculation
    float cx = newCamera.imageWidth() / 2.0f;
    float cy = newCamera.imageHeight() / 2.0f;
    float u = newCamera.fx() * (eyePoint.x() / eyePoint.z()) + cx;
    float v = newCamera.fy() * (eyePoint.y() / eyePoint.z()) + cy;

    // Check that point is within image bounds with edge padding
    float paddingX = edgePadding * newCamera.imageWidth();
    float paddingY = edgePadding * newCamera.imageHeight();

    EXPECT_GE(u, paddingX);
    EXPECT_LE(u, newCamera.imageWidth() - paddingX);
    EXPECT_GE(v, paddingY);
    EXPECT_LE(v, newCamera.imageHeight() - paddingY);
  }
}

// Helper function to test project/unproject round trip for intrinsics models
template <typename IntrinsicsModelType>
void testIntrinsicsRoundTrip(
    const std::shared_ptr<IntrinsicsModelType>& intrinsics,
    const std::vector<Eigen::Vector3f>& testPoints,
    float tolerance = 1e-4f) {
  for (const auto& originalPoint : testPoints) {
    // Project to image coordinates
    Vector3fP originalPointWide(
        FloatP(originalPoint.x()), FloatP(originalPoint.y()), FloatP(originalPoint.z()));
    auto [projectedPoint, mask] = intrinsics->project(originalPointWide);

    ASSERT_TRUE(mask[0]) << "Original point (" << originalPoint.x() << ", " << originalPoint.y()
                         << ", " << originalPoint.z() << ") should be valid for projection";

    // Unproject back to 3D
    Eigen::Vector3f imagePoint(projectedPoint.x()[0], projectedPoint.y()[0], originalPoint.z());
    auto [unprojectedPoint, unprojectValid] = intrinsics->unproject(imagePoint);

    // Check that unprojection succeeded
    ASSERT_TRUE(unprojectValid) << "Unprojected point should be valid for original point ("
                                << originalPoint.x() << ", " << originalPoint.y() << ", "
                                << originalPoint.z() << ")";

    // Check round-trip accuracy
    EXPECT_NEAR(unprojectedPoint.x(), originalPoint.x(), tolerance)
        << "X coordinate mismatch for point (" << originalPoint.x() << ", " << originalPoint.y()
        << ", " << originalPoint.z() << ")";
    EXPECT_NEAR(unprojectedPoint.y(), originalPoint.y(), tolerance)
        << "Y coordinate mismatch for point (" << originalPoint.x() << ", " << originalPoint.y()
        << ", " << originalPoint.z() << ")";
    EXPECT_NEAR(unprojectedPoint.z(), originalPoint.z(), tolerance)
        << "Z coordinate mismatch for point (" << originalPoint.x() << ", " << originalPoint.y()
        << ", " << originalPoint.z() << ")";
  }
}

// Test PinholeIntrinsicsModel unproject round trip
TEST_F(CameraTest, PinholeIntrinsicsUnprojectRoundTrip) {
  // Test points that should work well with pinhole model
  std::vector<Eigen::Vector3f> testPoints = {
      Eigen::Vector3f(0.0f, 0.0f, 1.0f), // Center
      Eigen::Vector3f(0.5f, 0.3f, 2.0f), // Off-center
      Eigen::Vector3f(-0.8f, -1.2f, 3.0f), // Negative coordinates
      Eigen::Vector3f(1.5f, 1.0f, 5.0f), // Larger coordinates
  };

  // Pinhole should be very accurate
  testIntrinsicsRoundTrip(pinholeIntrinsics, testPoints, 1e-5f);
}

// Test OpenCVIntrinsicsModel unproject round trip (single precision)
TEST_F(CameraTest, OpenCVIntrinsicsUnprojectRoundTripFloat) {
  // Test points that should converge well with OpenCV distortion model
  // Use reasonable coordinates - no need to restrict to tiny values
  std::vector<Eigen::Vector3f> testPoints = {
      Eigen::Vector3f(0.0f, 0.0f, 1.0f), // Center point (always works)
      Eigen::Vector3f(0.2f, 0.1f, 1.5f), // Small off-center
      Eigen::Vector3f(-0.3f, -0.2f, 2.0f), // Negative coordinates
      Eigen::Vector3f(0.4f, 0.3f, 3.0f), // Moderate coordinates
  };

  // Single precision with distortion needs loose tolerance due to iterative solver and floating
  // point precision
  testIntrinsicsRoundTrip(opencvIntrinsics, testPoints, 1e-1f);
}

// Test OpenCVIntrinsicsModel unproject round trip (double precision)
TEST_F(CameraTest, OpenCVIntrinsicsUnprojectRoundTripDouble) {
  // Helper function for double precision testing
  auto testIntrinsicsRoundTripDouble = [](const std::shared_ptr<OpenCVIntrinsicsModeld>& intrinsics,
                                          const std::vector<Eigen::Vector3d>& testPoints,
                                          double tolerance = 1e-6) {
    for (const auto& originalPoint : testPoints) {
      // Project to image coordinates
      auto [projectedPoint, projectedValid] = intrinsics->project(originalPoint);

      ASSERT_TRUE(projectedValid) << "Original point (" << originalPoint.x() << ", "
                                  << originalPoint.y() << ", " << originalPoint.z()
                                  << ") should be valid for projection";

      // Unproject back to 3D
      Eigen::Vector3d imagePoint3d(projectedPoint.x(), projectedPoint.y(), projectedPoint.z());
      auto [unprojectedPoint, unprojectValid] = intrinsics->unproject(imagePoint3d);
      // Check that unprojection succeeded
      ASSERT_TRUE(unprojectValid) << "Unprojected point should be valid for original point ("
                                  << originalPoint.x() << ", " << originalPoint.y() << ", "
                                  << originalPoint.z() << ")";

      auto [reprojectedPoint, reprojectedValid] = intrinsics->project(unprojectedPoint);
      std::cout << "reprojectedPoint: " << reprojectedPoint.transpose()
                << "; projected point: " << projectedPoint.transpose() << std::endl;

      ASSERT_TRUE(reprojectedValid) << "Unprojected point should be valid for re-projection";

      // Check that reprojected point matches the original point within tolerance
      EXPECT_NEAR(reprojectedPoint.x(), projectedPoint.x(), tolerance)
          << "Reprojected X coordinate mismatch for point (" << projectedPoint.x() << ", "
          << projectedPoint.y() << ", " << projectedPoint.z() << ")";
      EXPECT_NEAR(reprojectedPoint.y(), projectedPoint.y(), tolerance)
          << "Reprojected Y coordinate mismatch for point (" << projectedPoint.x() << ", "
          << projectedPoint.y() << ", " << projectedPoint.z() << ")";
      EXPECT_NEAR(reprojectedPoint.z(), projectedPoint.z(), tolerance)
          << "Reprojected Z coordinate mismatch for point (" << projectedPoint.x() << ", "
          << projectedPoint.y() << ", " << projectedPoint.z() << ")";

      // Check round-trip accuracy
      EXPECT_NEAR(unprojectedPoint.x(), originalPoint.x(), tolerance)
          << "X coordinate mismatch for point (" << originalPoint.x() << ", " << originalPoint.y()
          << ", " << originalPoint.z() << ")";
      EXPECT_NEAR(unprojectedPoint.y(), originalPoint.y(), tolerance)
          << "Y coordinate mismatch for point (" << originalPoint.x() << ", " << originalPoint.y()
          << ", " << originalPoint.z() << ")";
      EXPECT_NEAR(unprojectedPoint.z(), originalPoint.z(), tolerance)
          << "Z coordinate mismatch for point (" << originalPoint.x() << ", " << originalPoint.y()
          << ", " << originalPoint.z() << ")";
    }
  };

  // Test points with double precision
  std::vector<Eigen::Vector3d> testPoints = {
      Eigen::Vector3d(0.0, 0.0, 1.0), // Center point
      Eigen::Vector3d(0.1, 0.05, 1.5), // Small off-center
      Eigen::Vector3d(-0.15, -0.1, 2.0), // Small negative coordinates
      Eigen::Vector3d(0.2, 0.15, 3.0), // Moderate coordinates
  };

  // Double precision can have tighter tolerance than single precision, but still needs to account
  // for iterative solver
  testIntrinsicsRoundTripDouble(opencvIntrinsicsDouble, testPoints, 1e-3);
}

// Test Camera unproject round trip with transforms
TEST_F(CameraTest, CameraUnprojectRoundTripWithTransform) {
  // Create camera with non-identity transform
  Eigen::Transform<float, 3, Eigen::Affine> transform =
      Eigen::Transform<float, 3, Eigen::Affine>(Eigen::Translation3f(1.0f, 2.0f, 3.0f)) *
      Eigen::Transform<float, 3, Eigen::Affine>(Eigen::AngleAxisf(0.3f, Eigen::Vector3f::UnitY()));

  Camera transformedCamera(pinholeIntrinsics, transform);

  // Test world points
  std::vector<Eigen::Vector3f> worldPoints = {
      Eigen::Vector3f(2.0f, 1.0f, -1.0f), // Point in front of camera after transform
      Eigen::Vector3f(1.5f, 2.5f, -2.0f), // Another valid point
      Eigen::Vector3f(0.5f, 1.8f, -0.5f), // Close point
  };

  const float tolerance = 1e-3f;

  for (const auto& originalWorldPoint : worldPoints) {
    // Transform to camera space to check if point is in front
    Eigen::Vector3f cameraPoint = transform * originalWorldPoint;

    // Skip if point is behind camera
    if (cameraPoint.z() <= 0.1f) {
      continue;
    }

    // Project world point to image using camera
    Vector3fP worldPointWide(
        FloatP(originalWorldPoint.x()),
        FloatP(originalWorldPoint.y()),
        FloatP(originalWorldPoint.z()));

    // Transform to camera space and project
    Vector3fP cameraPointWide(
        FloatP(cameraPoint.x()), FloatP(cameraPoint.y()), FloatP(cameraPoint.z()));
    auto [projectedPoint, mask] = pinholeIntrinsics->project(cameraPointWide);

    ASSERT_TRUE(mask[0]) << "World point should be valid for projection";

    // Create image points for camera unproject (packet interface)
    Vector3fP imagePointsWide(
        projectedPoint.x(),
        projectedPoint.y(),
        FloatP(cameraPoint.z())); // z component contains depth

    // Unproject using camera method (should return world coordinates)
    auto [unprojectedWorldPoints, unprojectMask] = transformedCamera.unproject(imagePointsWide);

    ASSERT_TRUE(unprojectMask[0]) << "Unprojection should be valid";

    // Check that we get back the original world point
    EXPECT_NEAR(unprojectedWorldPoints.x()[0], originalWorldPoint.x(), tolerance)
        << "X mismatch for world point (" << originalWorldPoint.x() << ", "
        << originalWorldPoint.y() << ", " << originalWorldPoint.z() << ")";
    EXPECT_NEAR(unprojectedWorldPoints.y()[0], originalWorldPoint.y(), tolerance)
        << "Y mismatch for world point (" << originalWorldPoint.x() << ", "
        << originalWorldPoint.y() << ", " << originalWorldPoint.z() << ")";
    EXPECT_NEAR(unprojectedWorldPoints.z()[0], originalWorldPoint.z(), tolerance)
        << "Z mismatch for world point (" << originalWorldPoint.x() << ", "
        << originalWorldPoint.y() << ", " << originalWorldPoint.z() << ")";
  }
}

// Test Camera framePoints with multiple points - comprehensive framing verification
TEST_F(CameraTest, CameraFramePointsMultiple) {
  Camera camera(pinholeIntrinsics);
  std::vector<Eigen::Vector3f> points = {
      Eigen::Vector3f(-2.0f, -1.5f, 1.0f),
      Eigen::Vector3f(2.0f, -1.5f, 1.0f),
      Eigen::Vector3f(2.0f, 1.5f, 1.0f),
      Eigen::Vector3f(-2.0f, 1.5f, 1.0f)};

  const float minZ = 1.0f;
  const float edgePadding = 0.15f;
  Camera newCamera = camera.framePoints(points, minZ, edgePadding);

  // Camera should be adjusted to frame all points
  EXPECT_FALSE(newCamera.eyeFromWorld().isApprox(camera.eyeFromWorld()));

  // Verify all points are properly framed
  Eigen::Vector2f projectedMin(
      std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
  Eigen::Vector2f projectedMax(
      std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());

  for (const auto& worldPoint : points) {
    // Transform to eye space
    Eigen::Vector3f eyePoint = newCamera.eyeFromWorld() * worldPoint;

    // Check minimum Z distance
    EXPECT_GE(eyePoint.z(), minZ);

    // Project to image coordinates using pinhole model
    // Camera class doesn't have cx() and cy() methods - use image center calculation
    float cx = newCamera.imageWidth() / 2.0f;
    float cy = newCamera.imageHeight() / 2.0f;
    float u = newCamera.fx() * (eyePoint.x() / eyePoint.z()) + cx;
    float v = newCamera.fy() * (eyePoint.y() / eyePoint.z()) + cy;

    // Track bounding box of projected points
    projectedMin.x() = std::min(projectedMin.x(), u);
    projectedMin.y() = std::min(projectedMin.y(), v);
    projectedMax.x() = std::max(projectedMax.x(), u);
    projectedMax.y() = std::max(projectedMax.y(), v);

    // Check that each point is within image bounds
    EXPECT_GE(u, 0.0f);
    EXPECT_LE(u, static_cast<float>(newCamera.imageWidth()));
    EXPECT_GE(v, 0.0f);
    EXPECT_LE(v, static_cast<float>(newCamera.imageHeight()));
  }

  // Verify that the projected points span a reasonable portion of the image
  // The framePoints algorithm should position the camera so points use most of the available space
  float projectedWidth = projectedMax.x() - projectedMin.x();
  float projectedHeight = projectedMax.y() - projectedMin.y();

  // Points should span a significant portion of the image (at least 50% in each dimension)
  EXPECT_GT(projectedWidth, 0.5f * newCamera.imageWidth());
  EXPECT_GT(projectedHeight, 0.5f * newCamera.imageHeight());

  // Points should be reasonably centered
  Eigen::Vector2f projectedCenter = (projectedMin + projectedMax) * 0.5f;
  Eigen::Vector2f imageCenter(newCamera.imageWidth() * 0.5f, newCamera.imageHeight() * 0.5f);
  float centerDistance = (projectedCenter - imageCenter).norm();

  // Center should be within 25% of image diagonal from image center
  float imageDiagonal = std::sqrt(
      newCamera.imageWidth() * newCamera.imageWidth() +
      newCamera.imageHeight() * newCamera.imageHeight());
  EXPECT_LT(centerDistance, 0.25f * imageDiagonal);
}

// Test Camera framePoints with points at different depths
TEST_F(CameraTest, CameraFramePointsVariableDepth) {
  Camera camera(pinholeIntrinsics);
  std::vector<Eigen::Vector3f> points = {
      Eigen::Vector3f(-1.0f, -1.0f, 2.0f),
      Eigen::Vector3f(1.0f, -1.0f, 5.0f),
      Eigen::Vector3f(1.0f, 1.0f, 3.0f),
      Eigen::Vector3f(-1.0f, 1.0f, 4.0f)};

  const float minZ = 1.5f;
  const float edgePadding = 0.1f;
  Camera newCamera = camera.framePoints(points, minZ, edgePadding);

  // Verify all points meet minimum Z requirement and are within bounds
  for (const auto& worldPoint : points) {
    Eigen::Vector3f eyePoint = newCamera.eyeFromWorld() * worldPoint;

    EXPECT_GE(eyePoint.z(), minZ);

    // Camera class doesn't have cx() and cy() methods - use image center calculation
    float cx = newCamera.imageWidth() / 2.0f;
    float cy = newCamera.imageHeight() / 2.0f;
    float u = newCamera.fx() * (eyePoint.x() / eyePoint.z()) + cx;
    float v = newCamera.fy() * (eyePoint.y() / eyePoint.z()) + cy;

    EXPECT_GE(u, 0.0f);
    EXPECT_LE(u, static_cast<float>(newCamera.imageWidth()));
    EXPECT_GE(v, 0.0f);
    EXPECT_LE(v, static_cast<float>(newCamera.imageHeight()));
  }
}

// Test Camera framePoints preserves intrinsics
TEST_F(CameraTest, CameraFramePointsPreservesIntrinsics) {
  Camera camera(pinholeIntrinsics);
  std::vector<Eigen::Vector3f> points = {
      Eigen::Vector3f(0.5f, 0.5f, 2.0f), Eigen::Vector3f(-0.5f, -0.5f, 2.0f)};

  Camera newCamera = camera.framePoints(points, 0.1f, 0.1f);

  // Intrinsics should remain unchanged
  EXPECT_EQ(newCamera.imageWidth(), camera.imageWidth());
  EXPECT_EQ(newCamera.imageHeight(), camera.imageHeight());
  EXPECT_FLOAT_EQ(newCamera.fx(), camera.fx());
  EXPECT_FLOAT_EQ(newCamera.fy(), camera.fy());
  // Camera class doesn't have cx() and cy() methods - access through intrinsics
  auto newCameraPinhole =
      std::dynamic_pointer_cast<const PinholeIntrinsicsModel>(newCamera.intrinsicsModel());
  auto cameraPinhole =
      std::dynamic_pointer_cast<const PinholeIntrinsicsModel>(camera.intrinsicsModel());
  ASSERT_NE(newCameraPinhole, nullptr);
  ASSERT_NE(cameraPinhole, nullptr);
  EXPECT_FLOAT_EQ(newCameraPinhole->cx(), cameraPinhole->cx());
  EXPECT_FLOAT_EQ(newCameraPinhole->cy(), cameraPinhole->cy());
  EXPECT_EQ(newCamera.intrinsicsModel(), camera.intrinsicsModel());
}

// Test template instantiations work for double precision
TEST_F(CameraTest, DoublePrecisionCamera) {
  auto doublePinholeIntrinsics = std::make_shared<PinholeIntrinsicsModeld>(640, 480, 500.0, 500.0);
  Camerad doubleCamera(doublePinholeIntrinsics);

  EXPECT_EQ(doubleCamera.imageWidth(), 640);
  EXPECT_EQ(doubleCamera.imageHeight(), 480);
  EXPECT_DOUBLE_EQ(doubleCamera.fx(), 500.0);
  EXPECT_DOUBLE_EQ(doubleCamera.fy(), 500.0);
}

// Test PinholeIntrinsicsModel project method
TEST_F(CameraTest, PinholeIntrinsicsProject) {
  // Create a simple test point in front of the camera
  Vector3fP testPoint(FloatP(1.0f), FloatP(2.0f), FloatP(5.0f));

  auto [projectedPoint, mask] = pinholeIntrinsics->project(testPoint);

  // Expected projection: u = fx * x/z + cx, v = fy * y/z + cy
  // u = 500 * 1/5 + 320 = 100 + 320 = 420
  // v = 500 * 2/5 + 240 = 200 + 240 = 440

  EXPECT_FLOAT_EQ(projectedPoint.x()[0], 420.0f);
  EXPECT_FLOAT_EQ(projectedPoint.y()[0], 440.0f);
  EXPECT_FLOAT_EQ(projectedPoint.z()[0], 5.0f);
  EXPECT_TRUE(mask[0]); // Point should be in front of camera
}

// Test PinholeIntrinsicsModel project with point behind camera
TEST_F(CameraTest, PinholeIntrinsicsProjectBehind) {
  // Create a test point behind the camera (negative Z)
  Vector3fP testPoint(FloatP(1.0f), FloatP(2.0f), FloatP(-5.0f));

  auto [projectedPoint, mask] = pinholeIntrinsics->project(testPoint);

  EXPECT_FALSE(mask[0]); // Point should be masked out (behind camera)
}

// Test OpenCVIntrinsicsModel project method (basic case without distortion)
TEST_F(CameraTest, OpenCVIntrinsicsProjectBasic) {
  // Create OpenCV intrinsics without distortion for easier testing
  OpenCVDistortionParametersT<float> noDistortion; // All zeros by default
  auto simpleOpenCV = std::make_shared<OpenCVIntrinsicsModel>(
      640, 480, 500.0f, 500.0f, 320.0f, 240.0f, noDistortion);

  Vector3fP testPoint(FloatP(1.0f), FloatP(2.0f), FloatP(5.0f));

  auto [projectedPoint, mask] = simpleOpenCV->project(testPoint);

  // Without distortion, should be similar to pinhole model
  // u = fx * x/z + cx, v = fy * y/z + cy
  EXPECT_NEAR(projectedPoint.x()[0], 420.0f, 1e-5f);
  EXPECT_NEAR(projectedPoint.y()[0], 440.0f, 1e-5f);
  EXPECT_FLOAT_EQ(projectedPoint.z()[0], 5.0f);
  EXPECT_TRUE(mask[0]);
}

// Test that Camera properly delegates to intrinsics model
TEST_F(CameraTest, CameraDelegatesIntrinsics) {
  Camera camera(pinholeIntrinsics);

  // Test that camera delegates properly to intrinsics
  EXPECT_EQ(camera.imageWidth(), pinholeIntrinsics->imageWidth());
  EXPECT_EQ(camera.imageHeight(), pinholeIntrinsics->imageHeight());
  EXPECT_FLOAT_EQ(camera.fx(), pinholeIntrinsics->fx());
  EXPECT_FLOAT_EQ(camera.fy(), pinholeIntrinsics->fy());
  // Camera class doesn't have cx() and cy() methods - access through intrinsics
  auto pinholeConcrete =
      std::dynamic_pointer_cast<const PinholeIntrinsicsModel>(camera.intrinsicsModel());
  ASSERT_NE(pinholeConcrete, nullptr);
  EXPECT_FLOAT_EQ(pinholeConcrete->cx(), 320.0f); // Default center point calculation
  EXPECT_FLOAT_EQ(pinholeConcrete->cy(), 240.0f); // Default center point calculation
}

// Test edge cases for framePoints
TEST_F(CameraTest, CameraFramePointsEdgeCases) {
  Camera camera(pinholeIntrinsics);

  // Test with very small edge padding
  std::vector<Eigen::Vector3f> points = {Eigen::Vector3f(0.0f, 0.0f, 1.0f)};
  Camera newCamera1 = camera.framePoints(points, 0.1f, 0.0f);
  EXPECT_FALSE(newCamera1.eyeFromWorld().isApprox(camera.eyeFromWorld()));

  // Test with large edge padding
  Camera newCamera2 = camera.framePoints(points, 0.1f, 0.9f);
  EXPECT_FALSE(newCamera2.eyeFromWorld().isApprox(camera.eyeFromWorld()));

  // Test with different minZ values
  Camera newCamera3 = camera.framePoints(points, 10.0f, 0.1f);
  EXPECT_FALSE(newCamera3.eyeFromWorld().isApprox(camera.eyeFromWorld()));
}

// Test that Eigen vector project function matches vectorized version
TEST_F(CameraTest, EigenProjectMatchesVectorized) {
  std::vector<Eigen::Vector3f> testPoints = {
      Eigen::Vector3f(1.0f, 2.0f, 5.0f),
      Eigen::Vector3f(-1.5f, 0.5f, 3.0f),
      Eigen::Vector3f(0.0f, -2.0f, 4.0f),
      Eigen::Vector3f(2.5f, -1.0f, 6.0f)};

  auto testCameras = createTestCameras();

  for (const auto& cameraInfo : testCameras) {
    for (const auto& eigenPoint : testPoints) {
      // Test Eigen version
      auto [eigenProjected, eigenValid] = cameraInfo.intrinsics->project(eigenPoint);

      // Test vectorized version with single point
      Vector3fP vectorizedPoint(
          FloatP(eigenPoint.x()), FloatP(eigenPoint.y()), FloatP(eigenPoint.z()));
      auto [vectorizedProjected, vectorizedMask] = cameraInfo.intrinsics->project(vectorizedPoint);

      // Choose tolerance based on whether camera has distortion
      float tolerance = cameraInfo.hasDistortion ? 4e-5f : 1e-4f;

      // Compare results
      if (tolerance > 0.0f) {
        EXPECT_NEAR(eigenProjected.x(), vectorizedProjected.x()[0], tolerance)
            << "X projection mismatch for " << cameraInfo.name << " point (" << eigenPoint.x()
            << ", " << eigenPoint.y() << ", " << eigenPoint.z() << ")";
        EXPECT_NEAR(eigenProjected.y(), vectorizedProjected.y()[0], tolerance)
            << "Y projection mismatch for " << cameraInfo.name << " point (" << eigenPoint.x()
            << ", " << eigenPoint.y() << ", " << eigenPoint.z() << ")";
      } else {
        EXPECT_FLOAT_EQ(eigenProjected.x(), vectorizedProjected.x()[0])
            << "X projection mismatch for " << cameraInfo.name << " point (" << eigenPoint.x()
            << ", " << eigenPoint.y() << ", " << eigenPoint.z() << ")";
        EXPECT_FLOAT_EQ(eigenProjected.y(), vectorizedProjected.y()[0])
            << "Y projection mismatch for " << cameraInfo.name << " point (" << eigenPoint.x()
            << ", " << eigenPoint.y() << ", " << eigenPoint.z() << ")";
      }

      EXPECT_FLOAT_EQ(eigenProjected.z(), vectorizedProjected.z()[0])
          << "Z projection mismatch for " << cameraInfo.name << " point (" << eigenPoint.x() << ", "
          << eigenPoint.y() << ", " << eigenPoint.z() << ")";
      EXPECT_EQ(eigenValid, vectorizedMask[0])
          << "Validity mismatch for " << cameraInfo.name << " point (" << eigenPoint.x() << ", "
          << eigenPoint.y() << ", " << eigenPoint.z() << ")";
    }
  }
}

// Test Eigen project with points behind camera
TEST_F(CameraTest, EigenProjectBehindCamera) {
  Eigen::Vector3f behindPoint(-1.0f, 2.0f, -5.0f);

  auto testCameras = createTestCameras();

  for (const auto& cameraInfo : testCameras) {
    auto [projected, valid] = cameraInfo.intrinsics->project(behindPoint);
    EXPECT_FALSE(valid) << "Point behind camera should be invalid for " << cameraInfo.name
                        << " model";
  }
}

// Comprehensive Jacobian testing function for intrinsics models
template <typename IntrinsicsModelType, typename T>
void testIntrinsicsJacobianComprehensive(
    const std::shared_ptr<IntrinsicsModelType>& intrinsics,
    const std::string& modelName,
    const std::vector<Eigen::Vector<T, 3>>& testPoints,
    T finiteDiffTolerance = T(1e-5),
    T consistencyTolerance = T(1e-6)) {
  for (const auto& testPoint : testPoints) {
    // Test 1: Finite difference validation (only for double precision to avoid numerical issues)
    if constexpr (std::is_same_v<T, double>) {
      // Compute analytical Jacobian using Eigen interface
      auto [projectedPoint, jacobian3x3, isValid] = intrinsics->projectJacobian(testPoint);

      EXPECT_TRUE(isValid) << modelName << " test point should be valid for Jacobian";

      // Extract the 2x3 Jacobian matrix from the 3x3 matrix (ignore the third row for homogeneous
      // coordinates)
      Eigen::Matrix<T, 2, 3> analyticalJacobian = jacobian3x3.template topRows<2>();

      // Compute finite difference Jacobian
      const T epsilon = T(1e-8);
      Eigen::Matrix<T, 2, 3> finiteDiffJacobian;

      // Create wide vector for the base point
      using PacketT = PacketType_t<T>;
      Vector3xP<T> basePoint(
          PacketT(testPoint.x()), PacketT(testPoint.y()), PacketT(testPoint.z()));
      auto [baseProjection, baseMask] = intrinsics->project(basePoint);
      EXPECT_TRUE(baseMask[0]) << modelName << " base point should be valid for finite difference";

      T base_u = baseProjection.x()[0];
      T base_v = baseProjection.y()[0];

      // Compute partial derivatives with respect to X, Y, Z
      for (int i = 0; i < 3; ++i) {
        Eigen::Vector<T, 3> perturbedPoint = testPoint;
        perturbedPoint[i] += epsilon;

        auto [perturbedProjection, perturbedMask] = intrinsics->project(perturbedPoint);

        EXPECT_TRUE(perturbedMask)
            << modelName << " perturbed point should be valid for finite difference";

        T perturbed_u = perturbedProjection.x();
        T perturbed_v = perturbedProjection.y();

        // Compute finite difference derivatives
        finiteDiffJacobian(0, i) = (perturbed_u - base_u) / epsilon; // du/d(coordinate_i)
        finiteDiffJacobian(1, i) = (perturbed_v - base_v) / epsilon; // dv/d(coordinate_i)
      }

      // Compare analytical and finite difference Jacobians
      for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 3; ++col) {
          EXPECT_NEAR(
              analyticalJacobian(row, col), finiteDiffJacobian(row, col), finiteDiffTolerance)
              << modelName << " finite difference Jacobian mismatch at (" << row << "," << col
              << ") for point (" << testPoint.x() << "," << testPoint.y() << "," << testPoint.z()
              << ")" << " analytical=" << analyticalJacobian(row, col)
              << " finite_diff=" << finiteDiffJacobian(row, col);
        }
      }
    }

    // Test 2: Consistency between project() and projectJacobian() projected points
    {
      // Get projection from original project() method
      using PacketT = PacketType_t<T>;
      Vector3xP<T> testPointWide(
          PacketT(testPoint.x()), PacketT(testPoint.y()), PacketT(testPoint.z()));
      auto [projectedPoint, projectMask] = intrinsics->project(testPointWide);

      // Get projection from projectJacobian() method
      auto [jacobianProjectedPoint, jacobian, jacobianValid] =
          intrinsics->projectJacobian(testPoint);

      // Both should have same validity
      EXPECT_EQ(projectMask[0], jacobianValid)
          << modelName << " validity mismatch for point (" << testPoint.x() << ", " << testPoint.y()
          << ", " << testPoint.z() << ")";

      if (projectMask[0] && jacobianValid) {
        // Compare projected points - they should be very close
        EXPECT_NEAR(projectedPoint.x()[0], jacobianProjectedPoint(0), consistencyTolerance)
            << modelName << " X projection mismatch for point (" << testPoint.x() << ", "
            << testPoint.y() << ", " << testPoint.z() << ")";
        EXPECT_NEAR(projectedPoint.y()[0], jacobianProjectedPoint(1), consistencyTolerance)
            << modelName << " Y projection mismatch for point (" << testPoint.x() << ", "
            << testPoint.y() << ", " << testPoint.z() << ")";
        EXPECT_NEAR(projectedPoint.z()[0], jacobianProjectedPoint(2), consistencyTolerance)
            << modelName << " Z projection mismatch for point (" << testPoint.x() << ", "
            << testPoint.y() << ", " << testPoint.z() << ")";
      }
    }
  }
}

// Helper function to compute finite difference Jacobian for world coordinates using double
// precision
Eigen::Matrix<double, 2, 3> computeFiniteDifferenceWorldJacobian(
    const Camerad& camera,
    const Eigen::Vector3d& worldPoint,
    double epsilon = 1e-8) {
  Eigen::Matrix<double, 2, 3> jacobian;

  // Compute analytical world Jacobian using the new Eigen-based API
  auto [baseProjectedPoint, baseJacobian, baseValid] = camera.projectJacobian(worldPoint);
  EXPECT_TRUE(baseValid) << "Base point should be valid for world Jacobian computation";

  double base_u = baseProjectedPoint(0);
  double base_v = baseProjectedPoint(1);

  // Compute partial derivatives with respect to world X, Y, Z
  for (int i = 0; i < 3; ++i) {
    Eigen::Vector3d perturbedWorldPoint = worldPoint;
    perturbedWorldPoint[i] += epsilon;

    // Get projection for perturbed point
    auto [perturbedProjectedPoint, perturbedJacobian, perturbedValid] =
        camera.projectJacobian(perturbedWorldPoint);

    EXPECT_TRUE(perturbedValid) << "Perturbed point should be valid for world Jacobian computation";

    double perturbed_u = perturbedProjectedPoint(0);
    double perturbed_v = perturbedProjectedPoint(1);

    // Compute finite difference derivatives
    jacobian(0, i) = (perturbed_u - base_u) / epsilon; // du/d(world_coordinate_i)
    jacobian(1, i) = (perturbed_v - base_v) / epsilon; // dv/d(world_coordinate_i)
  }

  return jacobian;
}

// Test PinholeIntrinsicsModel Jacobian computation using comprehensive testing
TEST_F(CameraTest, PinholeIntrinsicsJacobianValidation) {
  // Test multiple points at different positions and depths
  std::vector<Eigen::Vector3d> testPoints = {
      Eigen::Vector3d(0.0, 0.0, 1.0), // Center point
      Eigen::Vector3d(1.0, 0.5, 2.0), // Off-center point
      Eigen::Vector3d(-0.8, -1.2, 3.0), // Negative coordinates
      Eigen::Vector3d(2.5, 1.8, 5.0), // Larger coordinates
      Eigen::Vector3d(0.1, 0.1, 0.5) // Close to camera
  };

  // Use comprehensive testing function
  testIntrinsicsJacobianComprehensive(
      pinholeIntrinsicsDouble,
      "PinholeDouble",
      testPoints,
      1e-5, // finite difference tolerance
      1e-12 // consistency tolerance
  );
}

// Test OpenCVIntrinsicsModel Jacobian computation using comprehensive testing
TEST_F(CameraTest, OpenCVIntrinsicsJacobianValidation) {
  // Test with both distorted and undistorted models using double precision
  std::vector<std::shared_ptr<OpenCVIntrinsicsModeld>> testModels = {
      opencvIntrinsicsDouble, // With distortion
  };

  // Also test with no distortion
  OpenCVDistortionParametersT<double> noDistortion;
  auto undistortedModel =
      std::make_shared<OpenCVIntrinsicsModeld>(640, 480, 500.0, 500.0, 320.0, 240.0, noDistortion);
  testModels.push_back(undistortedModel);

  std::vector<Eigen::Vector3d> testPoints = {
      Eigen::Vector3d(0.0, 0.0, 1.0), // Center point
      Eigen::Vector3d(0.5, 0.3, 2.0), // Small off-center point
      Eigen::Vector3d(-0.4, -0.6, 3.0), // Negative coordinates
      Eigen::Vector3d(1.2, 0.8, 4.0) // Larger coordinates
  };

  // Test distorted model
  testIntrinsicsJacobianComprehensive(
      opencvIntrinsicsDouble,
      "OpenCVDistortedDouble",
      testPoints,
      1e-4, // finite difference tolerance (looser for distortion)
      1e-12 // consistency tolerance
  );

  // Test undistorted model
  testIntrinsicsJacobianComprehensive(
      undistortedModel,
      "OpenCVUndistortedDouble",
      testPoints,
      1e-5, // finite difference tolerance (tighter for no distortion)
      1e-12 // consistency tolerance
  );
}

// Test Camera world Jacobian computation using finite differences with double precision
TEST_F(CameraTest, CameraWorldJacobianValidation) {
  // Test with different camera transforms using double precision
  std::vector<Eigen::Transform<double, 3, Eigen::Affine>> transforms = {
      Eigen::Transform<double, 3, Eigen::Affine>::Identity(), // Identity transform
      Eigen::Transform<double, 3, Eigen::Affine>(
          Eigen::Translation3d(1.0, 2.0, 3.0)), // Translation only
      Eigen::Transform<double, 3, Eigen::Affine>(
          Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitY())), // Rotation only
  };

  // Combined rotation and translation
  Eigen::Transform<double, 3, Eigen::Affine> combinedTransform =
      Eigen::Transform<double, 3, Eigen::Affine>(Eigen::Translation3d(0.5, -1.0, 2.0)) *
      Eigen::Transform<double, 3, Eigen::Affine>(
          Eigen::AngleAxisd(0.3, Eigen::Vector3d(1.0, 0.5, 0.2).normalized()));
  transforms.push_back(combinedTransform);

  std::vector<Eigen::Vector3d> testWorldPoints = {
      Eigen::Vector3d(0.0, 0.0, 0.0), // Origin
      Eigen::Vector3d(1.0, 0.5, -1.0), // Various coordinates
      Eigen::Vector3d(-2.0, 1.5, -3.0), // Negative coordinates
      Eigen::Vector3d(0.8, -0.6, -2.0) // Mixed signs
  };

  const double tolerance = 2e-5;

  for (const auto& transform : transforms) {
    Camerad camera(pinholeIntrinsicsDouble, transform);

    for (const auto& worldPoint : testWorldPoints) {
      // Check if point will be in front of camera after transformation
      Eigen::Vector3d cameraPoint = transform * worldPoint;
      if (cameraPoint.z() <= 0.1) {
        continue; // Skip points behind or too close to camera
      }

      // Compute analytical world Jacobian
      // Get world space Jacobian from camera (should be same for identity transform)
      auto [projectedPoint, worldJacobian, isValid] = camera.projectJacobian(worldPoint);

      EXPECT_TRUE(isValid) << "World point should be valid";

      // Extract analytical Jacobian (already in the right format)
      Eigen::Matrix<double, 2, 3> analyticalJacobian = worldJacobian;

      // Compute finite difference Jacobian
      Eigen::Matrix<double, 2, 3> finiteDiffJacobian =
          computeFiniteDifferenceWorldJacobian(camera, worldPoint);

      // Compare analytical and finite difference Jacobians
      for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 3; ++col) {
          EXPECT_NEAR(analyticalJacobian(row, col), finiteDiffJacobian(row, col), tolerance)
              << "World Jacobian mismatch at (" << row << "," << col << ") for world point ("
              << worldPoint.x() << "," << worldPoint.y() << "," << worldPoint.z() << ")"
              << " analytical=" << analyticalJacobian(row, col)
              << " finite_diff=" << finiteDiffJacobian(row, col);
        }
      }
    }
  }
}

// Test Jacobian consistency between camera and intrinsics models
TEST_F(CameraTest, JacobianConsistencyTest) {
  // For identity transform, world Jacobian should equal camera Jacobian
  Camera identityCamera(pinholeIntrinsics, Eigen::Transform<float, 3, Eigen::Affine>::Identity());

  std::vector<Eigen::Vector3f> testPoints = {
      Eigen::Vector3f(0.5f, 0.3f, 2.0f),
      Eigen::Vector3f(-0.8f, 1.2f, 3.0f),
      Eigen::Vector3f(1.5f, -0.7f, 1.5f)};

  const float tolerance = 1e-5f;

  for (const auto& point : testPoints) {
    // Get camera space Jacobian from intrinsics model using Eigen API
    auto [intrinsicsProjected, intrinsicsJacobian, intrinsicsValid] =
        pinholeIntrinsics->projectJacobian(point);

    // Get world space Jacobian from camera (should be same for identity transform)
    auto [worldProjected, worldJacobian, worldValid] = identityCamera.projectJacobian(point);

    EXPECT_TRUE(intrinsicsValid && worldValid) << "Both Jacobians should be valid";

    if (intrinsicsValid && worldValid) {
      // Extract the 2x3 Jacobian matrix from the 3x3 intrinsics matrix
      Eigen::Matrix<float, 2, 3> intrinsicsJacobian2x3 = intrinsicsJacobian.template topRows<2>();

      // Compare the Jacobians (world Jacobian is already 2x3)
      for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 3; ++col) {
          EXPECT_NEAR(intrinsicsJacobian2x3(row, col), worldJacobian(row, col), tolerance)
              << "Jacobian mismatch at (" << row << "," << col << ") for point (" << point.x()
              << "," << point.y() << "," << point.z() << ")";
        }
      }
    }
  }
}

// Test projectJacobian consistency for single precision models
TEST_F(CameraTest, ProjectJacobianConsistencyFloat) {
  std::vector<Eigen::Vector3f> testPoints = {
      Eigen::Vector3f(0.0f, 0.0f, 1.0f), // Center point
      Eigen::Vector3f(1.0f, 0.5f, 2.0f), // Off-center point
      Eigen::Vector3f(-0.8f, -1.2f, 3.0f), // Negative coordinates
      Eigen::Vector3f(0.5f, 0.3f, 1.5f) // Small coordinates
  };

  // Test pinhole model (should be very precise, but account for single precision limits)
  testIntrinsicsJacobianComprehensive(
      pinholeIntrinsics,
      "PinholeFloat",
      testPoints,
      1e-5f, // finite difference tolerance (not used for float)
      5e-5f // consistency tolerance (looser for single precision)
  );

  // Test OpenCV model (looser tolerance due to complexity and single precision)
  testIntrinsicsJacobianComprehensive(
      opencvIntrinsics,
      "OpenCVFloat",
      testPoints,
      1e-5f, // finite difference tolerance (not used for float)
      5e-5f // consistency tolerance (looser for OpenCV and single precision)
  );
}

// Test projectJacobian consistency with double precision models
TEST_F(CameraTest, ProjectJacobianConsistencyDouble) {
  std::vector<Eigen::Vector3d> testPoints = {
      Eigen::Vector3d(0.0, 0.0, 1.0), // Center point
      Eigen::Vector3d(1.0, 0.5, 2.0), // Off-center point
      Eigen::Vector3d(-0.8, -1.2, 3.0), // Negative coordinates
      Eigen::Vector3d(0.5, 0.3, 1.5) // Small coordinates
  };

  // Test pinhole model (should be very precise)
  testIntrinsicsJacobianComprehensive(
      pinholeIntrinsicsDouble,
      "PinholeDouble",
      testPoints,
      1e-5, // finite difference tolerance
      1e-12 // consistency tolerance (very tight for double precision)
  );

  // Test OpenCV model (slightly looser tolerance due to complexity)
  testIntrinsicsJacobianComprehensive(
      opencvIntrinsicsDouble,
      "OpenCVDouble",
      testPoints,
      1e-4, // finite difference tolerance (looser for distortion)
      1e-12 // consistency tolerance (still tight for double precision)
  );
}

// Test that SIMD and non-SIMD versions of transformWorldToEye produce identical results
// Since transformWorldToEye is private, we test it indirectly through the project methods
TEST_F(CameraTest, TransformWorldToEyeSimdConsistency) {
  // Create a camera with a non-identity transform to make the test more meaningful
  Eigen::Transform<float, 3, Eigen::Affine> transform =
      Eigen::Transform<float, 3, Eigen::Affine>(Eigen::Translation3f(1.0f, -2.0f, 3.0f)) *
      Eigen::Transform<float, 3, Eigen::Affine>(
          Eigen::AngleAxisf(0.5f, Eigen::Vector3f(0.2f, 1.0f, 0.3f).normalized()));

  Camera camera(pinholeIntrinsics, transform);

  // Test points that will be transformed and then projected
  std::vector<Eigen::Vector3f> testWorldPoints = {
      Eigen::Vector3f(0.0f, 0.0f, -5.0f), // Point that should be in front after transform
      Eigen::Vector3f(1.0f, 2.0f, -4.0f), // Off-center point
      Eigen::Vector3f(-1.5f, 0.5f, -3.0f), // Negative coordinates
      Eigen::Vector3f(2.0f, -1.0f, -6.0f), // Another test point
      Eigen::Vector3f(0.8f, 1.2f, -2.5f), // Close point
  };

  const float tolerance =
      2e-4f; // Reasonable tolerance for SIMD vs scalar floating-point comparison

  for (const auto& worldPoint : testWorldPoints) {
    // Test the scalar (non-SIMD) version by calling the Eigen project method
    auto [scalarProjected, scalarValid] = camera.project(worldPoint);

    // Test the SIMD version by calling the vectorized project method with a single point
    Vector3fP worldPointWide(
        FloatP(worldPoint.x()), FloatP(worldPoint.y()), FloatP(worldPoint.z()));
    auto [simdProjected, simdMask] = camera.project(worldPointWide);

    // Both should have the same validity
    EXPECT_EQ(scalarValid, simdMask[0])
        << "Validity mismatch between SIMD and scalar for world point (" << worldPoint.x() << ", "
        << worldPoint.y() << ", " << worldPoint.z() << ")";

    // If both are valid, compare the projected results
    if (scalarValid && simdMask[0]) {
      EXPECT_NEAR(scalarProjected.x(), simdProjected.x()[0], tolerance)
          << "X projection mismatch between SIMD and scalar for world point (" << worldPoint.x()
          << ", " << worldPoint.y() << ", " << worldPoint.z() << ")";

      EXPECT_NEAR(scalarProjected.y(), simdProjected.y()[0], tolerance)
          << "Y projection mismatch between SIMD and scalar for world point (" << worldPoint.x()
          << ", " << worldPoint.y() << ", " << worldPoint.z() << ")";

      EXPECT_NEAR(scalarProjected.z(), simdProjected.z()[0], tolerance)
          << "Z projection mismatch between SIMD and scalar for world point (" << worldPoint.x()
          << ", " << worldPoint.y() << ", " << worldPoint.z() << ")";
    }
  }
}

// Test transformWorldToEye consistency with different camera transforms
TEST_F(CameraTest, TransformWorldToEyeSimdConsistencyVariousTransforms) {
  std::vector<Eigen::Transform<float, 3, Eigen::Affine>> transforms = {
      Eigen::Transform<float, 3, Eigen::Affine>::Identity(), // Identity transform
      Eigen::Transform<float, 3, Eigen::Affine>(
          Eigen::Translation3f(0.0f, 0.0f, 5.0f)), // Translation only
      Eigen::Transform<float, 3, Eigen::Affine>(
          Eigen::AngleAxisf(0.7f, Eigen::Vector3f::UnitY())), // Rotation only
      Eigen::Transform<float, 3, Eigen::Affine>(Eigen::Translation3f(2.0f, -1.0f, 4.0f)) *
          Eigen::Transform<float, 3, Eigen::Affine>(
              Eigen::AngleAxisf(1.2f, Eigen::Vector3f(1.0f, 0.0f, 1.0f).normalized())), // Combined
  };

  // Test with both Pinhole and OpenCV intrinsics to ensure transform consistency across models
  auto testCameras = createTestCameras();

  std::vector<Eigen::Vector3f> testWorldPoints = {
      Eigen::Vector3f(1.0f, 0.0f, -2.0f),
      Eigen::Vector3f(-0.5f, 1.2f, -3.0f),
      Eigen::Vector3f(2.0f, -1.5f, -1.5f),
      Eigen::Vector3f(0.0f, 0.0f, -4.0f),
  };

  const float tolerance = 2e-4f;

  for (const auto& transform : transforms) {
    for (const auto& cameraInfo : testCameras) {
      Camera camera(cameraInfo.intrinsics, transform);

      for (const auto& worldPoint : testWorldPoints) {
        // Skip points that would be behind the camera after transformation
        Eigen::Vector3f transformedPoint = transform * worldPoint;
        if (transformedPoint.z() <= 0.1f) {
          continue;
        }

        // Test scalar version
        auto [scalarProjected, scalarValid] = camera.project(worldPoint);

        // Test SIMD version
        Vector3fP worldPointWide(
            FloatP(worldPoint.x()), FloatP(worldPoint.y()), FloatP(worldPoint.z()));
        auto [simdProjected, simdMask] = camera.project(worldPointWide);

        // Both should have the same validity
        EXPECT_EQ(scalarValid, simdMask[0])
            << "Validity mismatch for " << cameraInfo.name << " with transform and world point ("
            << worldPoint.x() << ", " << worldPoint.y() << ", " << worldPoint.z() << ")";

        if (scalarValid && simdMask[0]) {
          // Adjust tolerance based on camera type complexity
          float testTolerance = cameraInfo.hasDistortion ? tolerance * 10.0f : tolerance;

          EXPECT_NEAR(scalarProjected.x(), simdProjected.x()[0], testTolerance)
              << "X projection mismatch for " << cameraInfo.name
              << " with transform and world point (" << worldPoint.x() << ", " << worldPoint.y()
              << ", " << worldPoint.z() << ")";

          EXPECT_NEAR(scalarProjected.y(), simdProjected.y()[0], testTolerance)
              << "Y projection mismatch for " << cameraInfo.name
              << " with transform and world point (" << worldPoint.x() << ", " << worldPoint.y()
              << ", " << worldPoint.z() << ")";

          EXPECT_NEAR(scalarProjected.z(), simdProjected.z()[0], testTolerance)
              << "Z projection mismatch for " << cameraInfo.name
              << " with transform and world point (" << worldPoint.x() << ", " << worldPoint.y()
              << ", " << worldPoint.z() << ")";
        }
      }
    }
  }
}
