/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/joint.h>
#include <momentum/io/common/json_utils.h>

#include <gtest/gtest.h>
#include <Eigen/Geometry>

#include <optional>
#include <string>

using namespace momentum;

namespace {

JointPhysicalProperties makeTestPhysicalProperties() {
  JointPhysicalProperties properties;
  properties.jointName = "joint";
  properties.jointIndex = 2;
  properties.mass = 7.5f;
  properties.centerOfMassOffset = Eigen::Vector3f(1.0f, -2.0f, 3.0f);
  properties.inertia << 100.0f, 1.0f, 2.0f, 1.0f, 110.0f, 3.0f, 2.0f, 3.0f, 120.0f;
  properties.inertiaRotation =
      Eigen::Quaternionf(Eigen::AngleAxisf(0.3f, Eigen::Vector3f::UnitY()));
  return properties;
}

} // namespace

TEST(JsonUtilsTest, JointPhysicalPropertiesStringRoundTrip) {
  const JointPhysicalProperties properties = makeTestPhysicalProperties();

  const std::optional<JointPhysicalProperties> loaded = jointPhysicalPropertiesFromJsonString(
      jointPhysicalPropertiesToJsonString(properties), properties.jointName, properties.jointIndex);

  ASSERT_TRUE(loaded.has_value());
  EXPECT_EQ(loaded->jointName, properties.jointName);
  EXPECT_EQ(loaded->jointIndex, properties.jointIndex);
  EXPECT_FLOAT_EQ(loaded->mass, properties.mass);
  EXPECT_TRUE(loaded->centerOfMassOffset.isApprox(properties.centerOfMassOffset, 1e-5f));
  EXPECT_TRUE(loaded->inertia.isApprox(properties.inertia, 1e-5f));
  EXPECT_TRUE(loaded->inertiaRotation.isApprox(properties.inertiaRotation, 1e-5f));
}

TEST(JsonUtilsTest, JointPhysicalPropertiesFromJsonStringSkipsMalformedJson) {
  // Malformed JSON syntax throws nlohmann::json::parse_error and must be skipped, not propagated.
  EXPECT_FALSE(jointPhysicalPropertiesFromJsonString("{not valid json", "joint", 0).has_value());
}

TEST(JsonUtilsTest, JointPhysicalPropertiesFromJsonStringSkipsInvalidContent) {
  // Well-formed JSON that fails semantic validation throws std::runtime_error via MT_THROW. The
  // loader must skip the single bad entry (return std::nullopt) rather than abort the whole load.
  EXPECT_FALSE(jointPhysicalPropertiesFromJsonString(R"({"mass":1.0})", "joint", 0).has_value());
  EXPECT_FALSE(jointPhysicalPropertiesFromJsonString(R"({})", "joint", 0).has_value());
}
