/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <momentum/io/skeleton/parameter_transform_io.h>

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
