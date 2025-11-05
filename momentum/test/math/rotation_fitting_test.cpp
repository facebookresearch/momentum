/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/math/constants.h"
#include "momentum/math/utility.h"

#include <gtest/gtest.h>

using namespace momentum;

using Types = testing::Types<float, double>;

template <typename T>
struct RotationFittingTest : testing::Test {
  using Type = T;
};

TYPED_TEST_SUITE(RotationFittingTest, Types);

namespace {

/// Helper function to build a single-axis rotation matrix using AngleAxis
template <typename T>
Matrix3<T> createOneAxisRotationMatrix(T angle, int axis) {
  return AngleAxis<T>(angle, Eigen::Vector3<T>::Unit(axis)).toRotationMatrix();
}

/// Helper function to build a two-axis rotation matrix using AngleAxis
/// The rotation is applied as R_axis1 * R_axis0 (axis0 first, then axis1)
template <typename T>
Matrix3<T> createTwoAxisRotationMatrix(Vector2<T> angles, int axis0, int axis1) {
  const Matrix3<T> R0 = AngleAxis<T>(angles[0], Vector3<T>::Unit(axis0)).toRotationMatrix();
  const Matrix3<T> R1 = AngleAxis<T>(angles[1], Vector3<T>::Unit(axis1)).toRotationMatrix();
  return R1 * R0;
}

} // namespace

TYPED_TEST(RotationFittingTest, OneAxisRotationFitting) {
  using T = typename TestFixture::Type;

  // Test exact single-axis rotations using Eigen
  std::vector<T> testAngles = {
      T(0), // Identity
      pi<T>() / 6, // 30 degrees
      pi<T>() / 4, // 45 degrees
      pi<T>() / 3, // 60 degrees
      pi<T>() / 2, // 90 degrees
      -pi<T>() / 4, // -45 degrees
      -pi<T>() / 2 // -90 degrees
  };

  for (int axis = 0; axis < 3; ++axis) {
    for (const T testAngle : testAngles) {
      // Create exact rotation matrix using helper function
      const Matrix3<T> target = createOneAxisRotationMatrix(testAngle, axis);

      const T fittedAngle = rotationMatrixToOneAxisEuler(target, axis);

      // For exact single-axis rotations, we should get machine precision
      EXPECT_NEAR(fittedAngle, testAngle, std::numeric_limits<T>::epsilon() * T(100));

      // Test reconstruction accuracy - should be machine precision
      const Matrix3<T> reconstructed = createOneAxisRotationMatrix(fittedAngle, axis);
      const T reconstructionError = (reconstructed - target).squaredNorm();
      EXPECT_LT(reconstructionError, std::numeric_limits<T>::epsilon() * T(100000));
    }
  }
}

TYPED_TEST(RotationFittingTest, TwoAxisRotationFitting) {
  using T = typename TestFixture::Type;

  // Test exact two-axis rotations using Eigen
  std::vector<std::tuple<int, int, Vector2<T>>> testCases = {
      // {axis0, axis1, {angle0, angle1}}
      {0, 1, Vector2<T>(pi<T>() / 6, pi<T>() / 4)}, // X then Y: 30°, 45°
      {1, 2, Vector2<T>(pi<T>() / 3, pi<T>() / 6)}, // Y then Z: 60°, 30°
      {2, 0, Vector2<T>(pi<T>() / 4, pi<T>() / 3)}, // Z then X: 45°, 60°
      {0, 2, Vector2<T>(pi<T>() / 8, pi<T>() / 5)}, // X then Z: 22.5°, 36°
      {1, 0, Vector2<T>(-pi<T>() / 6, pi<T>() / 3)}, // Y then X: -30°, 60°
      {2, 1, Vector2<T>(pi<T>() / 2, -pi<T>() / 4)} // Z then Y: 90°, -45°
  };

  for (const auto& [axis0, axis1, testAngles] : testCases) {
    std::cout << "Testing two-axis rotation with axes " << axis0 << " and " << axis1 << std::endl;
    // Create exact rotation matrix using helper function
    const Matrix3<T> target = createTwoAxisRotationMatrix(testAngles, axis0, axis1);

    const Vector2<T> fittedAngles = rotationMatrixToTwoAxisEuler(target, axis0, axis1);

    // For exact two-axis rotations, we should get very good precision (within ~50x machine epsilon)
    EXPECT_NEAR(fittedAngles[0], testAngles[0], momentum::Eps<T>(1e-3, 1e-6));
    EXPECT_NEAR(fittedAngles[1], testAngles[1], momentum::Eps<T>(1e-3, 1e-6));

    // Test reconstruction accuracy - should be very precise using helper function
    const Matrix3<T> reconstructed = createTwoAxisRotationMatrix(fittedAngles, axis0, axis1);
    const T reconstructionError = (reconstructed - target).squaredNorm();
    EXPECT_LT(reconstructionError, momentum::Eps<T>(1e-5, 1e-8));
  }

  // Test identity matrix (should return zero angles)
  {
    const Matrix3<T> identity = Matrix3<T>::Identity();
    const Vector2<T> fittedAngles = rotationMatrixToTwoAxisEuler(identity, 0, 1);
    EXPECT_NEAR(fittedAngles[0], T(0), std::numeric_limits<T>::epsilon() * T(10));
    EXPECT_NEAR(fittedAngles[1], T(0), std::numeric_limits<T>::epsilon() * T(10));
  }
}

// Test how the rotation matrix to two-axis Euler conversion handles different order conventions
TYPED_TEST(RotationFittingTest, RotationOrderConventionTest) {
  using T = typename TestFixture::Type;

  // Test various combinations of moderate angles
  std::vector<std::pair<T, T>> testAnglePairs = {
      {T(0.1), T(0.2)}, // Small angles
      {T(0.4), T(0.6)}, // Medium angles (from our failing test)
      {T(0.8), T(0.9)}, // Larger angles
      {T(-0.3), T(0.5)}, // Mixed signs
      {T(0.7), T(-0.4)} // Mixed signs reversed
  };

  for (int iAxis = 0; iAxis < 3; ++iAxis) {
    for (int jAxis = iAxis + 1; jAxis < 3; ++jAxis) {
      for (const auto& [angle1, angle2] : testAnglePairs) {
        Eigen::Vector3<T> xyzAngles = Eigen::Vector3<T>::Zero();
        xyzAngles[iAxis] = angle1;
        xyzAngles[jAxis] = angle2;

        const Eigen::Matrix3<T> rotMat =
            eulerXYZToRotationMatrix(xyzAngles, EulerConvention::Extrinsic);
        const Eigen::Matrix3<T> expectedRotMat =
            Eigen::AngleAxis<T>(angle2, Eigen::Vector3<T>::Unit(jAxis)).toRotationMatrix() *
            Eigen::AngleAxis<T>(angle1, Eigen::Vector3<T>::Unit(iAxis)).toRotationMatrix();
        EXPECT_LT((rotMat - expectedRotMat).squaredNorm(), 1e-4f);

        const Vector2<T> fitted = rotationMatrixToTwoAxisEuler(rotMat, iAxis, jAxis);
        EXPECT_NEAR(fitted.x(), angle1, 1e-4);
        EXPECT_NEAR(fitted.y(), angle2, 1e-4);
      }
    }
  }
}

// Test edge cases that might cause issues in joint parameter conversion
TYPED_TEST(RotationFittingTest, JointParameterConversionEdgeCases) {
  using T = typename TestFixture::Type;

  std::cout << "=== Testing edge cases for joint parameter conversion ===" << std::endl;

  // Test case 1: Very small angles (should be handled precisely)
  {
    const T smallAngle1 = T(1e-6);
    const T smallAngle2 = T(2e-6);

    const auto axis1 = 0;
    const auto axis2 = 2;

    const Matrix3<T> target =
        createTwoAxisRotationMatrix(Eigen::Vector2<T>(smallAngle1, smallAngle2), axis1, axis2);

    const Vector2<T> fitted = rotationMatrixToTwoAxisEuler(target, axis1, axis2);
    std::cout << "Small angles test - Input: [" << smallAngle1 << ", " << smallAngle2
              << "], Fitted: [" << fitted[0] << ", " << fitted[1] << "]" << std::endl;

    EXPECT_NEAR(fitted[0], smallAngle1, T(1e-12)) << "Small angle RX recovery failed";
    EXPECT_NEAR(fitted[1], smallAngle2, T(1e-12)) << "Small angle RZ recovery failed";
  }

  // Test case 2: One axis near zero, other axis moderate
  {
    const T zeroAngle = T(0);
    const T moderateAngle = T(0.5);

    const auto axis1 = 0;
    const auto axis2 = 2;

    const Matrix3<T> target =
        createTwoAxisRotationMatrix(Eigen::Vector2<T>(zeroAngle, moderateAngle), axis1, axis2);

    const Vector2<T> fitted = rotationMatrixToTwoAxisEuler(target, axis1, axis2);
    std::cout << "Zero/moderate angles test - Input: [" << zeroAngle << ", " << moderateAngle
              << "], Fitted: [" << fitted[0] << ", " << fitted[1] << "]" << std::endl;

    EXPECT_NEAR(fitted[0], zeroAngle, T(1e-10)) << "Zero angle RX recovery failed";
    EXPECT_NEAR(fitted[1], moderateAngle, T(1e-6)) << "Moderate angle RZ recovery failed";
  }

  // Test case 3: Angles near pi/2 (potential singularities)
  {
    const T nearPiHalf = T(pi<T>() / 2 - 0.01);
    const T smallAngle = T(0.1);

    const auto axis1 = 0;
    const auto axis2 = 2;

    const Matrix3<T> target =
        createTwoAxisRotationMatrix(Eigen::Vector2<T>(nearPiHalf, smallAngle), axis1, axis2);

    const Vector2<T> fitted = rotationMatrixToTwoAxisEuler(target, axis1, axis2);
    std::cout << "Near pi/2 test - Input: [" << nearPiHalf << ", " << smallAngle << "], Fitted: ["
              << fitted[0] << ", " << fitted[1] << "]" << std::endl;

    // For angles near singularities, we allow more tolerance
    EXPECT_NEAR(fitted[0], nearPiHalf, T(0.01)) << "Near pi/2 angle RX recovery failed";
    EXPECT_NEAR(fitted[1], smallAngle, T(0.01)) << "Small angle RZ recovery failed";
  }
}

TYPED_TEST(RotationFittingTest, LargeAngleReconstructionAccuracy) {
  using T = typename TestFixture::Type;

  // Test with various large angles, focusing on reconstruction accuracy rather than exact angle
  // recovery
  std::vector<T> testAngles = {
      pi<T>() / 2, // 90 degrees
      pi<T>() * 2 / 3, // 120 degrees
      pi<T>() * 3 / 4, // 135 degrees
      pi<T>() - T(0.1), // Close to 180 degrees
      -pi<T>() / 2, // -90 degrees
      -pi<T>() * 2 / 3 // -120 degrees
  };

  for (const T angle : testAngles) {
    for (int axis = 0; axis < 3; ++axis) {
      // Create exact single-axis rotation using helper function
      const Matrix3<T> target = createOneAxisRotationMatrix(angle, axis);

      const T fittedAngle = rotationMatrixToOneAxisEuler(target, axis);

      // Reconstruct the rotation matrix using the helper function
      const Matrix3<T> reconstructed = createOneAxisRotationMatrix(fittedAngle, axis);

      // For exact single-axis rotations, reconstruction should be very accurate
      const T reconstructionError = (reconstructed - target).squaredNorm();
      EXPECT_LT(reconstructionError, T(1e-10)); // Very tight tolerance for exact case
    }
  }
}

TYPED_TEST(RotationFittingTest, ApproximationQuality) {
  using T = typename TestFixture::Type;

  // Test that the fitted rotations provide reasonable approximations
  // when the target cannot be exactly represented
  {
    // Create a complex rotation that cannot be represented by single or two axes
    const Matrix3<T> complexTarget = (AngleAxis<T>(pi<T>() / 4, Vector3<T>::UnitX()) *
                                      AngleAxis<T>(pi<T>() / 3, Vector3<T>::UnitY()) *
                                      AngleAxis<T>(pi<T>() / 6, Vector3<T>::UnitZ()))
                                         .toRotationMatrix();

    // Test single-axis approximations
    for (int axis = 0; axis < 3; ++axis) {
      const T fittedAngle = rotationMatrixToOneAxisEuler(complexTarget, axis);
      const Matrix3<T> singleAxisApprox =
          AngleAxis<T>(fittedAngle, Vector3<T>::Unit(axis)).toRotationMatrix();
      const T singleAxisError = (singleAxisApprox - complexTarget).squaredNorm();

      // Try a bunch of other angles:
      for (int i = 0; i < 100; ++i) {
        const T angle = T(2) * static_cast<T>(i) / T(100) * pi<T>();
        const Matrix3<T> singleAxisApprox2 =
            AngleAxis<T>(angle, Vector3<T>::Unit(axis)).toRotationMatrix();
        const T singleAxisError2 = (singleAxisApprox2 - complexTarget).squaredNorm();
        EXPECT_LE(singleAxisError, singleAxisError2);
      }
    }

    // Test two-axis approximations (should be better than single-axis)
    std::vector<std::pair<int, int>> axisPairs = {{0, 1}, {1, 2}, {2, 0}};
    for (const auto& [axis0, axis1] : axisPairs) {
      const Vector2<T> fittedAngles = rotationMatrixToTwoAxisEuler(complexTarget, axis0, axis1);
      const Matrix3<T> twoAxisApprox = createTwoAxisRotationMatrix(fittedAngles, axis0, axis1);

      const T twoAxisError = (twoAxisApprox - complexTarget).squaredNorm();
      const T identityError = (Matrix3<T>::Identity() - complexTarget).squaredNorm();
      EXPECT_LE(
          twoAxisError,
          (createOneAxisRotationMatrix(rotationMatrixToOneAxisEuler(complexTarget, axis0), axis0) -
           complexTarget)
              .squaredNorm());
      EXPECT_LE(
          twoAxisError,
          (createOneAxisRotationMatrix(rotationMatrixToOneAxisEuler(complexTarget, axis1), axis1) -
           complexTarget)
              .squaredNorm());
      EXPECT_LE(twoAxisError, identityError);
    }
  }
}
