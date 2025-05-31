/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/character/parameter_limits.h"
#include "momentum/character/parameter_transform.h"
#include "momentum/character/skeleton.h"
#include "momentum/math/utility.h"

namespace momentum {

// Test toString function for all LimitType enum values
TEST(ParameterLimitsTest, ToString) {
  EXPECT_EQ(toString(LimitType::MinMax), "MinMax");
  EXPECT_EQ(toString(LimitType::MinMaxJoint), "MinMaxJoint");
  EXPECT_EQ(toString(LimitType::MinMaxJointPassive), "MinMaxJointPassive");
  EXPECT_EQ(toString(LimitType::Linear), "Linear");
  EXPECT_EQ(toString(LimitType::Ellipsoid), "Ellipsoid");
  EXPECT_EQ(toString(LimitType::LinearJoint), "LinearJoint");
  EXPECT_EQ(toString(LimitType::HalfPlane), "HalfPlane");
  EXPECT_EQ(LimitType::LimitTypeCount, 7);
}

// Test LimitData constructors and operators
TEST(ParameterLimitsTest, LimitDataConstructorsAndOperators) {
  // Test default constructor
  LimitData data1;

  // Verify default constructor initializes with zeros
  for (unsigned char& i : data1.rawData) {
    EXPECT_EQ(i, 0);
  }

  // Modify data1
  data1.minMax.parameterIndex = 5;
  data1.minMax.limits = Vector2f::Constant(1.0f);
  data1.minMax.limits[1] = 2.0f;

  // Test copy constructor
  LimitData data2(data1);
  EXPECT_EQ(data2.minMax.parameterIndex, 5);
  EXPECT_EQ(data2.minMax.limits[0], 1.0f);
  EXPECT_EQ(data2.minMax.limits[1], 2.0f);

  // Test assignment operator
  LimitData data3;
  data3 = data1;
  EXPECT_EQ(data3.minMax.parameterIndex, 5);
  EXPECT_EQ(data3.minMax.limits[0], 1.0f);
  EXPECT_EQ(data3.minMax.limits[1], 2.0f);

  // Test equality operator
  EXPECT_TRUE(data1 == data2);
  EXPECT_TRUE(data1 == data3);

  // Modify data3 and test inequality
  data3.minMax.parameterIndex = 6;
  EXPECT_FALSE(data1 == data3);
}

// Test ParameterLimit equality operator
TEST(ParameterLimitsTest, ParameterLimitEquality) {
  ParameterLimit limit1;
  limit1.type = LimitType::MinMax;
  limit1.weight = 1.0f;
  limit1.data.minMax.parameterIndex = 5;
  limit1.data.minMax.limits = Vector2f::Constant(1.0f);
  limit1.data.minMax.limits[1] = 2.0f;

  ParameterLimit limit2;
  limit2.type = LimitType::MinMax;
  limit2.weight = 1.0f;
  limit2.data.minMax.parameterIndex = 5;
  limit2.data.minMax.limits = Vector2f::Constant(1.0f);
  limit2.data.minMax.limits[1] = 2.0f;

  EXPECT_TRUE(limit1 == limit2);

  // Test with different type
  ParameterLimit limit3 = limit1;
  limit3.type = LimitType::Linear;
  EXPECT_FALSE(limit1 == limit3);

  // Test with different weight
  ParameterLimit limit4 = limit1;
  limit4.weight = 2.0f;
  EXPECT_FALSE(limit1 == limit4);

  // Test with different data
  ParameterLimit limit5 = limit1;
  limit5.data.minMax.parameterIndex = 6;
  EXPECT_FALSE(limit1 == limit5);

  // Test with very close weights (should be equal due to isApprox)
  ParameterLimit limit6 = limit1;
  limit6.weight = 1.0f + 1e-7f;
  EXPECT_TRUE(limit1 == limit6);
}

// Test isInRange for LimitLinear
TEST(ParameterLimitsTest, IsInRangeLimitLinear) {
  // Test default range (0, 0) which should apply to all values
  LimitLinear limit1{};
  limit1.rangeMin = 0.0f;
  limit1.rangeMax = 0.0f;
  EXPECT_TRUE(isInRange(limit1, -10.0f));
  EXPECT_TRUE(isInRange(limit1, 0.0f));
  EXPECT_TRUE(isInRange(limit1, 10.0f));

  // Test specific range
  LimitLinear limit2{};
  limit2.rangeMin = 1.0f;
  limit2.rangeMax = 5.0f;
  EXPECT_FALSE(isInRange(limit2, 0.0f));
  EXPECT_TRUE(isInRange(limit2, 1.0f));
  EXPECT_TRUE(isInRange(limit2, 3.0f));
  EXPECT_FALSE(isInRange(limit2, 5.0f)); // rangeMax is non-inclusive
  EXPECT_FALSE(isInRange(limit2, 6.0f));
}

// Test isInRange for LimitLinearJoint
TEST(ParameterLimitsTest, IsInRangeLimitLinearJoint) {
  // Test default range (0, 0) which should apply to all values
  LimitLinearJoint limit1{};
  limit1.rangeMin = 0.0f;
  limit1.rangeMax = 0.0f;
  EXPECT_TRUE(isInRange(limit1, -10.0f));
  EXPECT_TRUE(isInRange(limit1, 0.0f));
  EXPECT_TRUE(isInRange(limit1, 10.0f));

  // Test specific range
  LimitLinearJoint limit2{};
  limit2.rangeMin = 1.0f;
  limit2.rangeMax = 5.0f;
  EXPECT_FALSE(isInRange(limit2, 0.0f));
  EXPECT_TRUE(isInRange(limit2, 1.0f));
  EXPECT_TRUE(isInRange(limit2, 3.0f));
  EXPECT_FALSE(isInRange(limit2, 5.0f)); // rangeMax is non-inclusive
  EXPECT_FALSE(isInRange(limit2, 6.0f));
}

// Test applyPassiveJointParameterLimits
TEST(ParameterLimitsTest, ApplyPassiveJointParameterLimits) {
  // Create joint parameters
  JointParameters jointParams = JointParameters::Zero(3 * kParametersPerJoint);

  // Set some values
  jointParams(0) = 0.5f; // Joint 0, parameter 0
  jointParams(1) = 1.5f; // Joint 0, parameter 1
  jointParams(kParametersPerJoint) = 2.5f; // Joint 1, parameter 0
  jointParams(kParametersPerJoint + 1) = 3.5f; // Joint 1, parameter 1

  // Create parameter limits
  ParameterLimits limits;

  // Add a MinMaxJointPassive limit for joint 0, parameter 0
  ParameterLimit limit1;
  limit1.type = LimitType::MinMaxJointPassive;
  limit1.data.minMaxJoint.jointIndex = 0;
  limit1.data.minMaxJoint.jointParameter = 0;
  limit1.data.minMaxJoint.limits = Vector2f::Constant(0.0f);
  limit1.data.minMaxJoint.limits[1] = 1.0f;
  limits.push_back(limit1);

  // Add a MinMaxJointPassive limit for joint 1, parameter 0
  ParameterLimit limit2;
  limit2.type = LimitType::MinMaxJointPassive;
  limit2.data.minMaxJoint.jointIndex = 1;
  limit2.data.minMaxJoint.jointParameter = 0;
  limit2.data.minMaxJoint.limits = Vector2f::Constant(0.0f);
  limit2.data.minMaxJoint.limits[1] = 2.0f;
  limits.push_back(limit2);

  // Add a non-passive limit that should be ignored
  ParameterLimit limit3;
  limit3.type = LimitType::MinMaxJoint;
  limit3.data.minMaxJoint.jointIndex = 0;
  limit3.data.minMaxJoint.jointParameter = 1;
  limit3.data.minMaxJoint.limits = Vector2f::Constant(0.0f);
  limit3.data.minMaxJoint.limits[1] = 1.0f;
  limits.push_back(limit3);

  // Apply limits
  JointParameters result = applyPassiveJointParameterLimits(limits, jointParams);

  // Check results
  EXPECT_EQ(result(0), 0.5f); // Within limits, unchanged
  EXPECT_EQ(result(1), 1.5f); // Non-passive limit, unchanged
  EXPECT_EQ(result(kParametersPerJoint), 2.0f); // Clamped to max
  EXPECT_EQ(result(kParametersPerJoint + 1), 3.5f); // No limit, unchanged

  // Test clamping to min
  jointParams(0) = -1.0f;
  result = applyPassiveJointParameterLimits(limits, jointParams);
  EXPECT_EQ(result(0), 0.0f); // Clamped to min
}

// Test getPoseConstraintParameterLimits
TEST(ParameterLimitsTest, GetPoseConstraintParameterLimits) {
  // Create a parameter transform with pose constraints
  ParameterTransform pt;

  // Add a pose constraint
  PoseConstraint pc;
  pc.parameterIdValue.emplace_back(1, 1.0f);
  pc.parameterIdValue.emplace_back(2, 2.0f);
  pt.poseConstraints["test_pose"] = pc;

  // Get parameter limits with default weight
  ParameterLimits limits1 = getPoseConstraintParameterLimits("test_pose", pt);

  EXPECT_EQ(limits1.size(), 2);
  EXPECT_EQ(limits1[0].type, LimitType::MinMax);
  EXPECT_EQ(limits1[0].data.minMax.parameterIndex, 1);
  EXPECT_EQ(limits1[0].data.minMax.limits[0], 1.0f);
  EXPECT_EQ(limits1[0].data.minMax.limits[1], 1.0f);
  EXPECT_EQ(limits1[0].weight, 1.0f);

  EXPECT_EQ(limits1[1].type, LimitType::MinMax);
  EXPECT_EQ(limits1[1].data.minMax.parameterIndex, 2);
  EXPECT_EQ(limits1[1].data.minMax.limits[0], 2.0f);
  EXPECT_EQ(limits1[1].data.minMax.limits[1], 2.0f);
  EXPECT_EQ(limits1[1].weight, 1.0f);

  // Get parameter limits with custom weight
  ParameterLimits limits2 = getPoseConstraintParameterLimits("test_pose", pt, 2.5f);

  EXPECT_EQ(limits2.size(), 2);
  EXPECT_EQ(limits2[0].weight, 2.5f);
  EXPECT_EQ(limits2[1].weight, 2.5f);

  // Test with non-existent pose constraint
  ParameterLimits limits3 = getPoseConstraintParameterLimits("non_existent", pt);
  EXPECT_TRUE(limits3.empty());
}

// Test edge cases for applyPassiveJointParameterLimits
TEST(ParameterLimitsTest, ApplyPassiveJointParameterLimitsEdgeCases) {
  // Empty limits
  JointParameters jointParams = JointParameters::Zero(3 * kParametersPerJoint);
  jointParams(0) = 0.5f;

  ParameterLimits emptyLimits;
  JointParameters result = applyPassiveJointParameterLimits(emptyLimits, jointParams);

  // Should return unchanged parameters
  EXPECT_EQ(result(0), 0.5f);

  // Test with empty joint parameters
  JointParameters emptyJointParams = JointParameters::Zero(0);
  ParameterLimits limits;

  ParameterLimit limit;
  limit.type = LimitType::MinMaxJointPassive;
  limit.data.minMaxJoint.jointIndex = 0;
  limit.data.minMaxJoint.jointParameter = 0;
  limit.data.minMaxJoint.limits = Vector2f::Constant(0.0f);
  limit.data.minMaxJoint.limits[1] = 1.0f;
  limits.push_back(limit);

  // This should not crash, but we can't test the result directly
  // as it would trigger the MT_CHECK assertion
  // applyPassiveJointParameterLimits(limits, emptyJointParams);
}

// Test self-assignment for LimitData
TEST(ParameterLimitsTest, LimitDataSelfAssignment) {
  LimitData data;
  data.minMax.parameterIndex = 5;
  data.minMax.limits = Vector2f::Constant(1.0f);
  data.minMax.limits[1] = 2.0f;

  // Self-assignment should not change the data
  LimitData& dataRef = data;
  data = dataRef;

  EXPECT_EQ(data.minMax.parameterIndex, 5);
  EXPECT_EQ(data.minMax.limits[0], 1.0f);
  EXPECT_EQ(data.minMax.limits[1], 2.0f);
}

} // namespace momentum
