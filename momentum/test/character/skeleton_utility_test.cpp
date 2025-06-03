/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character/skeleton_utility.h"
#include "momentum/character/types.h"
#include "momentum/math/types.h"

#include <gtest/gtest.h>

using namespace momentum;

class SkeletonUtilityTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create test model parameters
    Eigen::VectorXf prevVec(5);
    prevVec << 0.1f, 0.2f, 0.3f, 0.4f, 0.5f;
    previous = prevVec;

    Eigen::VectorXf currVec(5);
    currVec << 0.2f, 0.4f, 0.6f, 0.8f, 1.0f;
    current = currVec;

    // Create active parameters bitset
    activeParams.reset();
    activeParams.set(0); // Only first parameter is active
    activeParams.set(2); // And third parameter is active
    activeParams.set(4); // And fifth parameter is active
  }

  ModelParameters previous;
  ModelParameters current;
  ParameterSet activeParams;
};

// Test the first overload of extrapolateModelParameters with default parameters
TEST_F(SkeletonUtilityTest, ExtrapolateModelParametersDefaultParams) {
  ModelParameters result = extrapolateModelParameters(previous, current);

  // Expected result: current + (current - previous) * kDefaultExtrapolateFactor
  // with clamping to kDefaultExtrapolateMaxDelta
  Eigen::VectorXf expectedVec(5);
  // For each parameter:
  // diff = current - previous = [0.1, 0.2, 0.3, 0.4, 0.5]
  // Clamp diffs to [-kDefaultExtrapolateMaxDelta, kDefaultExtrapolateMaxDelta] = [0.1, 0.2, 0.3,
  // 0.4, 0.4] result = current + clamped_diff * kDefaultExtrapolateFactor = [0.2, 0.4, 0.6,
  // 0.8, 1.0] + [0.1, 0.2, 0.3, 0.4, 0.4] * 0.8 = [0.2, 0.4, 0.6, 0.8, 1.0] + [0.08, 0.16, 0.24,
  // 0.32, 0.32] = [0.28, 0.56, 0.84, 1.12, 1.32]
  expectedVec << 0.28f, 0.56f, 0.84f, 1.12f, 1.32f;
  ModelParameters expected = expectedVec;

  EXPECT_TRUE(result.v.isApprox(expected.v));
}

// Test the first overload of extrapolateModelParameters with custom parameters
TEST_F(SkeletonUtilityTest, ExtrapolateModelParametersCustomParams) {
  float factor = 0.5f;
  float maxDelta = 0.2f;
  ModelParameters result = extrapolateModelParameters(previous, current, factor, maxDelta);

  // Expected result: current + (current - previous) * factor
  // with clamping to maxDelta
  Eigen::VectorXf expectedVec(5);
  // For each parameter:
  // diff = current - previous = [0.1, 0.2, 0.3, 0.4, 0.5]
  // Clamp diffs to maxDelta (0.2): [0.1, 0.2, 0.2, 0.2, 0.2]
  // result = current + clamped_diff * factor
  // = [0.2, 0.4, 0.6, 0.8, 1.0] + [0.1, 0.2, 0.2, 0.2, 0.2] * 0.5
  // = [0.2, 0.4, 0.6, 0.8, 1.0] + [0.05, 0.1, 0.1, 0.1, 0.1]
  // = [0.25, 0.5, 0.7, 0.9, 1.1]
  expectedVec << 0.25f, 0.5f, 0.7f, 0.9f, 1.1f;
  ModelParameters expected = expectedVec;

  EXPECT_TRUE(result.v.isApprox(expected.v));
}

// Test the first overload of extrapolateModelParameters with negative differences
TEST_F(SkeletonUtilityTest, ExtrapolateModelParametersNegativeDiffs) {
  // Swap previous and current to get negative differences
  ModelParameters result = extrapolateModelParameters(current, previous);

  // Expected result: previous + (previous - current) * kDefaultExtrapolateFactor
  // with clamping to kDefaultExtrapolateMaxDelta
  Eigen::VectorXf expectedVec(5);
  // For each parameter:
  // diff = previous - current = [-0.1, -0.2, -0.3, -0.4, -0.5]
  // Clamp diffs to [-kDefaultExtrapolateMaxDelta, kDefaultExtrapolateMaxDelta]
  // = [-0.1, -0.2, -0.3, -0.4, -0.4]
  // result = previous + clamped_diff * kDefaultExtrapolateFactor
  // = [0.1, 0.2, 0.3, 0.4, 0.5] + [-0.1, -0.2, -0.3, -0.4, -0.4] * 0.8
  // = [0.1, 0.2, 0.3, 0.4, 0.5] + [-0.08, -0.16, -0.24, -0.32, -0.32]
  // = [0.02, 0.04, 0.06, 0.08, 0.18]
  expectedVec << 0.02f, 0.04f, 0.06f, 0.08f, 0.18f;
  ModelParameters expected = expectedVec;

  EXPECT_TRUE(result.v.isApprox(expected.v));
}

// Test the first overload of extrapolateModelParameters with size mismatch
TEST_F(SkeletonUtilityTest, ExtrapolateModelParametersSizeMismatch) {
  Eigen::VectorXf smallerPrevVec(3);
  smallerPrevVec << 0.1f, 0.2f, 0.3f;
  ModelParameters smallerPrevious = smallerPrevVec;

  // When sizes mismatch, the function should return current unchanged
  ModelParameters result = extrapolateModelParameters(smallerPrevious, current);
  EXPECT_TRUE(result.v.isApprox(current.v));
}

// Test the second overload of extrapolateModelParameters with default parameters
TEST_F(SkeletonUtilityTest, ExtrapolateModelParametersWithActiveParamsDefault) {
  ModelParameters result = extrapolateModelParameters(previous, current, activeParams);

  // Expected result: current parameters, but only extrapolate active ones
  ModelParameters expected = current;
  // For active parameters (0, 2, 4):
  // diff = current - previous = [0.1, -, 0.3, -, 0.5]
  // All diffs are within kDefaultExtrapolateMaxDelta (0.4), except the last one
  // which is clamped to 0.4
  // result = current + clamped_diff * kDefaultExtrapolateFactor
  // For index 0: 0.2 + 0.1 * 0.8 = 0.28
  // For index 2: 0.6 + 0.3 * 0.8 = 0.84
  // For index 4: 1.0 + 0.4 * 0.8 = 1.32
  expected(0) = 0.28f;
  expected(2) = 0.84f;
  expected(4) = 1.32f;

  EXPECT_TRUE(result.v.isApprox(expected.v));
}

// Test the second overload of extrapolateModelParameters with custom parameters
TEST_F(SkeletonUtilityTest, ExtrapolateModelParametersWithActiveParamsCustom) {
  float factor = 0.5f;
  float maxDelta = 0.2f;
  ModelParameters result =
      extrapolateModelParameters(previous, current, activeParams, factor, maxDelta);

  // Expected result: current parameters, but only extrapolate active ones
  ModelParameters expected = current;
  // For active parameters (0, 2, 4):
  // diff = current - previous = [0.1, -, 0.3, -, 0.5]
  // Clamp diffs to maxDelta (0.2): [0.1, -, 0.2, -, 0.2]
  // result = current + clamped_diff * factor
  // For index 0: 0.2 + 0.1 * 0.5 = 0.25
  // For index 2: 0.6 + 0.2 * 0.5 = 0.7
  // For index 4: 1.0 + 0.2 * 0.5 = 1.1
  expected(0) = 0.25f;
  expected(2) = 0.7f;
  expected(4) = 1.1f;

  EXPECT_TRUE(result.v.isApprox(expected.v));
}

// Test the second overload of extrapolateModelParameters with negative differences
TEST_F(SkeletonUtilityTest, ExtrapolateModelParametersWithActiveParamsNegativeDiffs) {
  // Swap previous and current to get negative differences
  ModelParameters result = extrapolateModelParameters(current, previous, activeParams);

  // Expected result: previous parameters, but only extrapolate active ones
  ModelParameters expected = previous;
  // For active parameters (0, 2, 4):
  // diff = previous - current = [-0.1, -, -0.3, -, -0.5]
  // Clamp diffs to [-kDefaultExtrapolateMaxDelta, kDefaultExtrapolateMaxDelta]
  // = [-0.1, -, -0.3, -, -0.4]
  // result = previous + clamped_diff * kDefaultExtrapolateFactor
  // For index 0: 0.1 + (-0.1) * 0.8 = 0.02
  // For index 2: 0.3 + (-0.3) * 0.8 = 0.06
  // For index 4: 0.5 + (-0.4) * 0.8 = 0.18
  expected(0) = 0.02f;
  expected(2) = 0.06f;
  expected(4) = 0.18f;

  EXPECT_TRUE(result.v.isApprox(expected.v));
}

// Test the second overload of extrapolateModelParameters with size mismatch
TEST_F(SkeletonUtilityTest, ExtrapolateModelParametersWithActiveParamsSizeMismatch) {
  Eigen::VectorXf smallerPrevVec(3);
  smallerPrevVec << 0.1f, 0.2f, 0.3f;
  ModelParameters smallerPrevious = smallerPrevVec;

  // When sizes mismatch, the function should return current unchanged
  ModelParameters result = extrapolateModelParameters(smallerPrevious, current, activeParams);
  EXPECT_TRUE(result.v.isApprox(current.v));
}

// Test the second overload of extrapolateModelParameters with no active parameters
TEST_F(SkeletonUtilityTest, ExtrapolateModelParametersWithNoActiveParams) {
  ParameterSet noActiveParams;
  noActiveParams.reset(); // No active parameters

  // When no parameters are active, the function should return current unchanged
  ModelParameters result = extrapolateModelParameters(previous, current, noActiveParams);
  EXPECT_TRUE(result.v.isApprox(current.v));
}

// Test the second overload of extrapolateModelParameters with all active parameters
TEST_F(SkeletonUtilityTest, ExtrapolateModelParametersWithAllActiveParams) {
  ParameterSet allActiveParams;
  allActiveParams.set(); // All parameters are active

  // When all parameters are active, the result should be the same as the first overload
  ModelParameters result1 = extrapolateModelParameters(previous, current, allActiveParams);
  ModelParameters result2 = extrapolateModelParameters(previous, current);

  EXPECT_TRUE(result1.v.isApprox(result2.v));
}

// Test the extrapolateModelParameters functions with extreme values
TEST_F(SkeletonUtilityTest, ExtrapolateModelParametersExtremeValues) {
  // Create test model parameters with extreme values
  Eigen::VectorXf extremePrevVec(5);
  extremePrevVec << -1000.0f, -100.0f, 0.0f, 100.0f, 1000.0f;
  ModelParameters extremePrevious = extremePrevVec;

  Eigen::VectorXf extremeCurrVec(5);
  extremeCurrVec << -900.0f, 0.0f, 0.0f, 200.0f, 1100.0f;
  ModelParameters extremeCurrent = extremeCurrVec;

  // Test first overload
  ModelParameters result1 = extrapolateModelParameters(extremePrevious, extremeCurrent);

  // Test second overload with all parameters active
  ParameterSet allActiveParams;
  allActiveParams.set();
  ModelParameters result2 =
      extrapolateModelParameters(extremePrevious, extremeCurrent, allActiveParams);

  // The results should be the same
  EXPECT_TRUE(result1.v.isApprox(result2.v));

  // Verify that the extrapolation is working correctly with clamping
  // The differences are [100, 100, 0, 100, 100]
  // After clamping to kDefaultExtrapolateMaxDelta (0.4), they become [0.4, 0.4, 0, 0.4, 0.4]
  // result = current + clamped_diff * kDefaultExtrapolateFactor
  // = [-900, 0, 0, 200, 1100] + [0.4, 0.4, 0, 0.4, 0.4] * 0.8
  // = [-900, 0, 0, 200, 1100] + [0.32, 0.32, 0, 0.32, 0.32]
  // = [-899.68, 0.32, 0, 200.32, 1100.32]
  Eigen::VectorXf expectedVec(5);
  expectedVec << -899.68f, 0.32f, 0.0f, 200.32f, 1100.32f;
  ModelParameters expected = expectedVec;

  EXPECT_TRUE(result1.v.isApprox(expected.v));
}

// Test the extrapolateModelParameters functions with zero factor
TEST_F(SkeletonUtilityTest, ExtrapolateModelParametersZeroFactor) {
  // With zero factor, the result should be the same as current
  ModelParameters result = extrapolateModelParameters(previous, current, 0.0f);
  EXPECT_TRUE(result.v.isApprox(current.v));

  // Same for the second overload
  ModelParameters result2 = extrapolateModelParameters(previous, current, activeParams, 0.0f);
  EXPECT_TRUE(result2.v.isApprox(current.v));
}

// Test the extrapolateModelParameters functions with negative factor
TEST_F(SkeletonUtilityTest, ExtrapolateModelParametersNegativeFactor) {
  // With negative factor, the extrapolation should go in the opposite direction
  float factor = -0.5f;
  ModelParameters result = extrapolateModelParameters(previous, current, factor);

  // Expected result: current + (current - previous) * factor with clamping
  // diff = current - previous = [0.1, 0.2, 0.3, 0.4, 0.5]
  // Clamp diffs to [-kDefaultExtrapolateMaxDelta, kDefaultExtrapolateMaxDelta] = [0.1, 0.2, 0.3,
  // 0.4, 0.4] result = current + clamped_diff * factor = [0.2, 0.4, 0.6, 0.8, 1.0] + [0.1, 0.2,
  // 0.3, 0.4, 0.4] * (-0.5) = [0.2, 0.4, 0.6, 0.8, 1.0] + [-0.05, -0.1, -0.15, -0.2, -0.2] = [0.15,
  // 0.3, 0.45, 0.6, 0.8]
  Eigen::VectorXf expectedVec(5);
  expectedVec << 0.15f, 0.3f, 0.45f, 0.6f, 0.8f;
  ModelParameters expected = expectedVec;

  EXPECT_TRUE(result.v.isApprox(expected.v));
}

// Test the extrapolateModelParameters functions with zero maxDelta
TEST_F(SkeletonUtilityTest, ExtrapolateModelParametersZeroMaxDelta) {
  // With zero maxDelta, all differences should be clamped to zero
  // So the result should be the same as current
  ModelParameters result =
      extrapolateModelParameters(previous, current, kDefaultExtrapolateFactor, 0.0f);
  EXPECT_TRUE(result.v.isApprox(current.v));

  // Same for the second overload
  ModelParameters result2 =
      extrapolateModelParameters(previous, current, activeParams, kDefaultExtrapolateFactor, 0.0f);
  EXPECT_TRUE(result2.v.isApprox(current.v));
}
