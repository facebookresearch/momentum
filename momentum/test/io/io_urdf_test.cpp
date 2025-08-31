/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/urdf/urdf_io.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character/character_helpers_gtest.h"
#include "momentum/test/io/io_helpers.h"

#include <gtest/gtest.h>

using namespace momentum;

namespace {

std::optional<filesystem::path> getTestFilePath(const std::string& filename) {
  auto envVar = GetEnvVar("TEST_MOMENTUM_MODELS_PATH");
  if (!envVar.has_value()) {
    return std::nullopt;
  }
  return filesystem::path(envVar.value()) / filename;
}

TEST(IoUrdfTest, LoadCharacter) {
  auto urdfPath = getTestFilePath("character.urdf");
  if (!urdfPath.has_value()) {
    GTEST_SKIP() << "Environment variable 'TEST_MOMENTUM_MODELS_PATH' is not set.";
  }

  const auto& character = loadUrdfCharacter(*urdfPath);
  const auto& skeleton = character.skeleton;
  EXPECT_EQ(skeleton.joints.size(), 45);
  EXPECT_EQ(skeleton.joints[0].name, "b_root");
  EXPECT_EQ(skeleton.joints[0].parent, kInvalidIndex);
  EXPECT_TRUE(skeleton.joints[0].preRotation.isApprox(Quaternionf::Identity()));
  EXPECT_TRUE(skeleton.joints[0].translationOffset.isApprox(Vector3f::Zero()));

  const auto& parameterTransform = character.parameterTransform;
  const auto totalDofs = parameterTransform.transform.cols();
  EXPECT_EQ(totalDofs, 6 + 34); // 6 global parameters (root) + 34 joint parameters
  const size_t numJoints = skeleton.joints.size();
  const size_t numJointParameters = numJoints * kParametersPerJoint;
  EXPECT_EQ(parameterTransform.name.size(), totalDofs);
  EXPECT_EQ(parameterTransform.offsets.size(), (numJointParameters));
  EXPECT_EQ(parameterTransform.transform.rows(), numJointParameters);
  EXPECT_EQ(parameterTransform.transform.cols(), totalDofs);

  const auto& parameterLimits = character.parameterLimits;
  EXPECT_EQ(parameterLimits.size(), 34); // the number of revolute and prismatic joints
  EXPECT_EQ(parameterLimits[0].type, MinMax);
  EXPECT_EQ(parameterLimits[0].data.minMax.parameterIndex, 6); // the first joint parameter
  EXPECT_FLOAT_EQ(parameterLimits[0].data.minMax.limits[0], -2.095);
  EXPECT_FLOAT_EQ(parameterLimits[0].data.minMax.limits[1], 0.785);
}

} // namespace
