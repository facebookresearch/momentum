/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <momentum/camera/camera.h>
#include <momentum/camera/projection_utils.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <memory>

using namespace momentum;

namespace {

/// Apply a 3x4 projection matrix to a world point and perform the perspective divide.
template <typename T>
Eigen::Vector2<T> applyProjection(
    const Eigen::Matrix<T, 3, 4>& M,
    const Eigen::Vector3<T>& worldPoint) {
  const Eigen::Vector4<T> ph(worldPoint.x(), worldPoint.y(), worldPoint.z(), T(1));
  const Eigen::Vector3<T> h = M * ph;
  return h.template head<2>() / h.z();
}

/// Apply a 3x4 projection matrix and return the raw homogeneous result (before perspective divide).
template <typename T>
Eigen::Vector3<T> applyProjectionHomogeneous(
    const Eigen::Matrix<T, 3, 4>& M,
    const Eigen::Vector3<T>& worldPoint) {
  const Eigen::Vector4<T> ph(worldPoint.x(), worldPoint.y(), worldPoint.z(), T(1));
  return M * ph;
}

/// Create a non-trivial eye-from-world transform for testing.
Eigen::Transform<double, 3, Eigen::Affine> createTestTransform() {
  Eigen::Transform<double, 3, Eigen::Affine> t =
      Eigen::Transform<double, 3, Eigen::Affine>::Identity();
  t.prerotate(Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitY()));
  t.pretranslate(Eigen::Vector3d(1.0, -0.5, 0.3));
  return t;
}

} // namespace

class ProjectionUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    eyeFromWorld_ = createTestTransform();

    // Pinhole camera
    auto pinholeIntrinsics = std::make_shared<PinholeIntrinsicsModeld>(640, 480, 500.0, 500.0);
    pinholeCamera_ = CameraT<double>(pinholeIntrinsics, eyeFromWorld_);

    // OpenCV camera with moderate distortion
    OpenCVDistortionParametersT<double> distParams;
    distParams.k1 = 0.1;
    distParams.k2 = -0.05;
    distParams.p1 = 0.001;
    distParams.p2 = 0.002;
    auto opencvIntrinsics =
        std::make_shared<OpenCVIntrinsicsModeld>(640, 480, 500.0, 500.0, 320.0, 240.0, distParams);
    opencvCamera_ = CameraT<double>(opencvIntrinsics, eyeFromWorld_);

    // Fisheye camera
    OpenCVFisheyeDistortionParametersT<double> fisheyeParams;
    fisheyeParams.k1 = 0.01;
    fisheyeParams.k2 = -0.005;
    auto fisheyeIntrinsics = std::make_shared<OpenCVFisheyeIntrinsicsModeld>(
        640, 480, 300.0, 300.0, 320.0, 240.0, fisheyeParams);
    fisheyeCamera_ = CameraT<double>(fisheyeIntrinsics, eyeFromWorld_);
  }

  Eigen::Transform<double, 3, Eigen::Affine> eyeFromWorld_;
  CameraT<double> pinholeCamera_;
  CameraT<double> opencvCamera_;
  CameraT<double> fisheyeCamera_;
};

// A world point that projects exactly to the target pixel should yield zero residual
// from the constructed matrix. Tests all three camera models and both variants.
TEST_F(ProjectionUtilsTest, ZeroResidualAtTarget) {
  const std::vector<std::pair<std::string, CameraT<double>*>> cameras = {
      {"Pinhole", &pinholeCamera_},
      {"OpenCV", &opencvCamera_},
      {"Fisheye", &fisheyeCamera_},
  };

  // Place a world point in front of the camera and find where it projects.
  const Eigen::Vector3d worldPoint =
      pinholeCamera_.worldFromEye() * Eigen::Vector3d(0.5, -0.3, 5.0);

  for (const auto& [name, camera] : cameras) {
    const auto [projected, valid] = camera->project(worldPoint);
    ASSERT_TRUE(valid) << name;
    const Eigen::Vector2d targetPixel = projected.head<2>();

    // Pixels variant. Tolerance accounts for iterative unproject in distortion models
    // (default Newton tolerance is 1e-6).
    {
      const auto [M, validM] = projectionMatrixPixels(*camera, targetPixel);
      ASSERT_TRUE(validM) << name << " pixels matrix";
      const Eigen::Vector2d residual = applyProjection(M, worldPoint);
      EXPECT_NEAR(residual.x(), 0.0, 1e-5) << name << " pixels x";
      EXPECT_NEAR(residual.y(), 0.0, 1e-5) << name << " pixels y";
    }

    // Radians variant
    {
      const auto [M, validM] = projectionMatrixRadians(*camera, targetPixel);
      ASSERT_TRUE(validM) << name << " radians matrix";
      const Eigen::Vector2d residual = applyProjection(M, worldPoint);
      EXPECT_NEAR(residual.x(), 0.0, 1e-5) << name << " radians x";
      EXPECT_NEAR(residual.y(), 0.0, 1e-5) << name << " radians y";
    }
  }
}

// For nearby points, the pixel-space matrix residual should match camera.project(point) - target.
// For pinhole (no distortion), the match is exact. For distorted cameras, the linearization
// introduces error proportional to the nonlinearity of the distortion over the offset.
TEST_F(ProjectionUtilsTest, ConsistencyWithCameraProject) {
  // Small perturbations in world space
  const std::vector<Eigen::Vector3d> offsets = {
      {0.01, 0.0, 0.0},
      {0.0, 0.01, 0.0},
      {0.0, 0.0, 0.01},
      {0.02, -0.01, 0.005},
  };

  // --- Pinhole: matrix residual should be exactly equal to camera residual ---
  {
    const Eigen::Vector2d targetPixel(350.0, 260.0);
    const auto [M, validM] = projectionMatrixPixels(pinholeCamera_, targetPixel);
    ASSERT_TRUE(validM);
    const auto [targetWorld, valid] =
        pinholeCamera_.unproject(Eigen::Vector3d(targetPixel.x(), targetPixel.y(), 5.0));
    ASSERT_TRUE(valid);

    for (const auto& offset : offsets) {
      const Eigen::Vector3d testPoint = targetWorld + offset;
      const auto [projectedPt, validPt] = pinholeCamera_.project(testPoint);
      ASSERT_TRUE(validPt);

      const Eigen::Vector2d cameraResidual = projectedPt.head<2>() - targetPixel;
      const Eigen::Vector2d matrixResidual = applyProjection(M, testPoint);

      // Pinhole has no distortion to linearize, so the match should be exact.
      EXPECT_NEAR(matrixResidual.x(), cameraResidual.x(), 1e-8)
          << "pinhole offset: " << offset.transpose();
      EXPECT_NEAR(matrixResidual.y(), cameraResidual.y(), 1e-8)
          << "pinhole offset: " << offset.transpose();
    }
  }

  // --- OpenCV with distortion: linearization is approximate but close for small offsets ---
  {
    const Eigen::Vector2d targetPixel(350.0, 260.0);
    const auto [M, validM] = projectionMatrixPixels(opencvCamera_, targetPixel);
    ASSERT_TRUE(validM);
    const auto [targetWorld, valid] =
        opencvCamera_.unproject(Eigen::Vector3d(targetPixel.x(), targetPixel.y(), 5.0));
    ASSERT_TRUE(valid);

    for (const auto& offset : offsets) {
      const Eigen::Vector3d testPoint = targetWorld + offset;
      const auto [projectedPt, validPt] = opencvCamera_.project(testPoint);
      if (!validPt) {
        continue;
      }

      const Eigen::Vector2d cameraResidual = projectedPt.head<2>() - targetPixel;
      const Eigen::Vector2d matrixResidual = applyProjection(M, testPoint);

      // With distortion the linearization is approximate; 1-pixel tolerance for these
      // small offsets is a reasonable bound on the linearization error.
      EXPECT_NEAR(matrixResidual.x(), cameraResidual.x(), 1.0)
          << "opencv offset: " << offset.transpose();
      EXPECT_NEAR(matrixResidual.y(), cameraResidual.y(), 1.0)
          << "opencv offset: " << offset.transpose();
    }
  }
}

// Points behind the camera or far off-screen should produce finite values.
// This is the key advantage of the linearized projection matrix approach.
TEST_F(ProjectionUtilsTest, OffScreenRobustness) {
  const Eigen::Vector2d targetPixel(320.0, 240.0);

  const std::vector<std::pair<std::string, CameraT<double>*>> cameras = {
      {"Pinhole", &pinholeCamera_},
      {"OpenCV", &opencvCamera_},
      {"Fisheye", &fisheyeCamera_},
  };

  for (const auto& [name, camera] : cameras) {
    const auto [M, validM] = projectionMatrixPixels(*camera, targetPixel);
    ASSERT_TRUE(validM) << name;

    // Point behind the camera (negative z in eye space).
    // The homogeneous z should be negative, and all values should be finite.
    const Eigen::Vector3d behindCamera = camera->worldFromEye() * Eigen::Vector3d(0.0, 0.0, -5.0);
    const Eigen::Vector3d h = applyProjectionHomogeneous(M, behindCamera);
    EXPECT_TRUE(std::isfinite(h.x())) << name;
    EXPECT_TRUE(std::isfinite(h.y())) << name;
    EXPECT_TRUE(std::isfinite(h.z())) << name;
    EXPECT_LT(h.z(), 0.0) << name << " z should be negative for behind-camera points";

    // Far off-screen point: residual should be finite and large.
    const Eigen::Vector3d farOffScreen =
        camera->worldFromEye() * Eigen::Vector3d(100.0, 100.0, 1.0);
    const Eigen::Vector2d residual = applyProjection(M, farOffScreen);
    EXPECT_TRUE(std::isfinite(residual.x())) << name;
    EXPECT_TRUE(std::isfinite(residual.y())) << name;
    EXPECT_GT(residual.norm(), 100.0) << name << " far off-screen should have large residual";
  }
}

// The radians variant's residual magnitude should equal tan(angle) between the point's ray
// and the target ray. For small angles, tan(angle) ~ angle in radians.
TEST_F(ProjectionUtilsTest, RadiansApproximatesAngularError) {
  const Eigen::Vector2d targetPixel(320.0, 240.0);
  const auto [M, validM] = projectionMatrixRadians(pinholeCamera_, targetPixel);
  ASSERT_TRUE(validM);

  // Unproject target to get the reference ray direction in eye-space.
  const auto [targetEye, validTarget] = pinholeCamera_.intrinsicsModel()->unproject(
      Eigen::Vector3d(targetPixel.x(), targetPixel.y(), 1.0));
  ASSERT_TRUE(validTarget);
  const Eigen::Vector3d targetRay = targetEye.normalized();

  // Create a point at a known small angle from the target ray.
  const Eigen::Vector3d perturbedEye = targetEye + Eigen::Vector3d(0.05, 0.03, 0.0);
  const Eigen::Vector3d perturbedRay = perturbedEye.normalized();

  // True angle between the rays.
  const double trueAngle = std::acos(std::min(1.0, std::max(-1.0, targetRay.dot(perturbedRay))));

  // Transform perturbed point to world-space at some depth.
  const Eigen::Vector3d worldPoint =
      pinholeCamera_.worldFromEye() * (perturbedEye.normalized() * 5.0);

  const Eigen::Vector2d residual = applyProjection(M, worldPoint);
  const double residualMagnitude = residual.norm();

  // The residual components are x/z and y/z in the rotated frame, so the magnitude
  // is exactly tan(angle) between the point's ray and the target ray.
  EXPECT_NEAR(residualMagnitude, std::tan(trueAngle), 1e-6);
}

// When the target pixel cannot be unprojected, the functions should return valid=false
// and a zero matrix rather than crashing or producing garbage.
TEST_F(ProjectionUtilsTest, InvalidUnprojectReturnsInvalid) {
  // Use strong distortion + a pixel far outside the image so that the OpenCV
  // Newton-based unproject fails to converge.
  OpenCVDistortionParametersT<double> strongDist;
  strongDist.k1 = 0.5;
  strongDist.k2 = 0.3;
  auto intrinsics =
      std::make_shared<OpenCVIntrinsicsModeld>(640, 480, 500.0, 500.0, 320.0, 240.0, strongDist);
  CameraT<double> camera(intrinsics, eyeFromWorld_);

  const Eigen::Vector2d extremePixel(1e6, 1e6);

  // Verify that the intrinsics model itself fails to unproject this point,
  // confirming our test setup actually triggers the invalid path.
  const auto [eyePt, unprojectValid] =
      intrinsics->unproject(Eigen::Vector3d(extremePixel.x(), extremePixel.y(), 1.0));
  ASSERT_FALSE(unprojectValid) << "Test setup: extreme pixel should fail to unproject";

  const auto [Mpx, validPx] = projectionMatrixPixels(camera, extremePixel);
  EXPECT_FALSE(validPx);
  EXPECT_TRUE(Mpx.isZero()) << "Invalid result should return zero matrix";

  const auto [Mrad, validRad] = projectionMatrixRadians(camera, extremePixel);
  EXPECT_FALSE(validRad);
  EXPECT_TRUE(Mrad.isZero()) << "Invalid result should return zero matrix";
}
