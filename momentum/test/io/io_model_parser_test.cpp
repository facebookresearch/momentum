/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <momentum/character/character.h>
#include <momentum/character/skeleton.h>
#include <momentum/io/skeleton/parameter_transform_io.h>
#include <momentum/test/character/character_helpers_gtest.h>

using namespace momentum;

TEST(IoModelParserTest, DuplicateSections_Concatenated) {
  const std::string modelContent = R"(Momentum Model Definition V1.0

[ParameterTransform]
# Some parameter transforms
root.tx = 1.0 * param1

[Limits]
# First set of limits
limit param1 minmax [-1.0, 1.0] 1.0

[PoseConstraints]
# Some pose constraints
poseconstraints tpose param1=0.0

[Limits]
# Second set of limits
limit param2 minmax [-2.0, 2.0] 1.0
)";

  const auto sections = loadMomentumModelFromBuffer(
      std::span<const std::byte>(
          reinterpret_cast<const std::byte*>(modelContent.data()), modelContent.size()));

  ASSERT_TRUE(sections.find("Limits") != sections.end());

  const std::string& limitsContent = sections.at("Limits");

  EXPECT_TRUE(limitsContent.find("limit param1 minmax") != std::string::npos)
      << "First limits section should be present. Content:\n"
      << limitsContent;
  EXPECT_TRUE(limitsContent.find("limit param2 minmax") != std::string::npos)
      << "Second limits section should be present. Content:\n"
      << limitsContent;
}

TEST(IoModelParserTest, MultipleDuplicateSections_AllConcatenated) {
  const std::string modelContent = R"(Momentum Model Definition V1.0

[Limits]
limit param1 minmax [-1.0, 1.0] 1.0

[ParameterTransform]
root.tx = 1.0 * param1

[Limits]
limit param2 minmax [-2.0, 2.0] 1.0

[PoseConstraints]
poseconstraints tpose param1=0.0

[Limits]
limit param3 minmax [-3.0, 3.0] 1.0
)";

  const auto sections = loadMomentumModelFromBuffer(
      std::span<const std::byte>(
          reinterpret_cast<const std::byte*>(modelContent.data()), modelContent.size()));

  ASSERT_TRUE(sections.find("Limits") != sections.end());

  const std::string& limitsContent = sections.at("Limits");

  EXPECT_TRUE(limitsContent.find("limit param1 minmax") != std::string::npos)
      << "First limits section should be present";
  EXPECT_TRUE(limitsContent.find("limit param2 minmax") != std::string::npos)
      << "Second limits section should be present";
  EXPECT_TRUE(limitsContent.find("limit param3 minmax") != std::string::npos)
      << "Third limits section should be present";
}

TEST(IoModelParserTest, SingleSection_WorksAsExpected) {
  const std::string modelContent = R"(Momentum Model Definition V1.0

[ParameterTransform]
root.tx = 1.0 * param1

[Limits]
limit param1 minmax [-1.0, 1.0] 1.0
limit param2 minmax [-2.0, 2.0] 1.0
)";

  const auto sections = loadMomentumModelFromBuffer(
      std::span<const std::byte>(
          reinterpret_cast<const std::byte*>(modelContent.data()), modelContent.size()));

  ASSERT_TRUE(sections.find("Limits") != sections.end());

  const std::string& limitsContent = sections.at("Limits");

  EXPECT_TRUE(limitsContent.find("limit param1 minmax") != std::string::npos);
  EXPECT_TRUE(limitsContent.find("limit param2 minmax") != std::string::npos);
}

TEST(IoModelParserTest, WriteModelDefinition_RoundTrip) {
  // Create original character
  Character original;
  original.name = "test_character";

  // Create a simple skeleton
  original.skeleton.joints.resize(2);
  original.skeleton.joints[0].name = "root";
  original.skeleton.joints[0].parent = kInvalidIndex;
  original.skeleton.joints[1].name = "child";
  original.skeleton.joints[1].parent = 0;

  // Create parameter transform
  original.parameterTransform.name = {"tx", "ty", "tz"};
  original.parameterTransform.activeJointParams.setConstant(
      original.skeleton.joints.size() * kParametersPerJoint, false);
  original.parameterTransform.offsets.setZero(
      original.skeleton.joints.size() * kParametersPerJoint);

  // Set up transform matrix (2 joints * 7 params per joint = 14 rows, 3 parameters = 3 cols)
  original.parameterTransform.transform.resize(
      original.skeleton.joints.size() * kParametersPerJoint, 3);
  std::vector<Eigen::Triplet<float>> triplets;

  // root.tx = 1.0 * tx
  original.parameterTransform.activeJointParams[0] = true;
  triplets.emplace_back(0, 0, 1.0f);

  // root.ty = 1.0 * ty
  original.parameterTransform.activeJointParams[1] = true;
  triplets.emplace_back(1, 1, 1.0f);

  // child.tz = 2.0 * tz + 0.5
  original.parameterTransform.activeJointParams[7 + 2] = true;
  triplets.emplace_back(7 + 2, 2, 2.0f);
  original.parameterTransform.offsets[7 + 2] = 0.5f;

  original.parameterTransform.transform.setFromTriplets(triplets.begin(), triplets.end());

  // Add parameter sets
  ParameterSet ps1;
  ps1.set(0, true); // tx
  ps1.set(1, true); // ty
  original.parameterTransform.parameterSets["translation_xy"] = ps1;

  // Add pose constraints
  PoseConstraint pc;
  pc.parameterIdValue.emplace_back(0, 0.0f); // tx = 0
  pc.parameterIdValue.emplace_back(1, 0.0f); // ty = 0
  original.parameterTransform.poseConstraints["bind_pose"] = pc;

  // Create parameter limits
  ParameterLimit limit1;
  limit1.type = LimitType::MinMax;
  limit1.weight = 1.0f;
  limit1.data.minMax.parameterIndex = 0;
  limit1.data.minMax.limits = Eigen::Vector2f(-1.0f, 1.0f);
  original.parameterLimits.push_back(limit1);

  ParameterLimit limit2;
  limit2.type = LimitType::MinMax;
  limit2.weight = 1.0f;
  limit2.data.minMax.parameterIndex = 1;
  limit2.data.minMax.limits = Eigen::Vector2f(-2.0f, 2.0f);
  original.parameterLimits.push_back(limit2);

  // Write model definition
  const std::string written = writeModelDefinition(
      original.skeleton, original.parameterTransform, original.parameterLimits);

  // Verify the output contains expected sections
  EXPECT_TRUE(written.find("Momentum Model Definition V1.0") != std::string::npos);
  EXPECT_TRUE(written.find("[ParameterTransform]") != std::string::npos);
  EXPECT_TRUE(written.find("[ParameterSets]") != std::string::npos);
  EXPECT_TRUE(written.find("[PoseConstraints]") != std::string::npos);
  EXPECT_TRUE(written.find("[ParameterLimits]") != std::string::npos);

  // Verify parameter transform content
  EXPECT_TRUE(written.find("root.tx = 1*tx") != std::string::npos);
  EXPECT_TRUE(written.find("root.ty = 1*ty") != std::string::npos);
  EXPECT_TRUE(written.find("child.tz = 2*tz + 0.5") != std::string::npos);

  // Verify parameter sets
  EXPECT_TRUE(written.find("parameterset translation_xy tx ty") != std::string::npos);

  // Verify pose constraints
  EXPECT_TRUE(written.find("poseconstraints bind_pose tx=0 ty=0") != std::string::npos);

  // Round-trip test: parse what we wrote and build a new character
  Character roundtrip;
  roundtrip.name = "test_character";
  roundtrip.skeleton = original.skeleton;
  std::tie(roundtrip.parameterTransform, roundtrip.parameterLimits) = loadModelDefinition(
      std::span<const std::byte>(
          reinterpret_cast<const std::byte*>(written.data()), written.size()),
      original.skeleton);

  // Use the comprehensive compareChars helper to verify everything matches
  compareChars(original, roundtrip, false);
}
