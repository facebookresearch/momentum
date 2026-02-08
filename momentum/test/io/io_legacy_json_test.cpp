/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/legacy_json/legacy_json_io.h"

#include "momentum/character/character.h"
#include "momentum/character/collision_geometry.h"
#include "momentum/character/locator.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skin_weights.h"
#include "momentum/math/mesh.h"
#include "momentum/test/character/character_helpers_gtest.h"
#include "momentum/test/io/io_helpers.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace momentum;

class LegacyJsonIOTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a simple test character
    testCharacter_ = createTestCharacter();
  }

  static Character createTestCharacter() {
    // Create a simple skeleton with 3 joints
    JointList joints;

    // Root joint
    Joint root;
    root.name = "root";
    root.parent = kInvalidIndex;
    root.preRotation = Eigen::Quaternionf::Identity();
    root.translationOffset = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
    joints.push_back(root);

    // Child joint 1
    Joint child1;
    child1.name = "child1";
    child1.parent = 0;
    child1.preRotation = Eigen::Quaternionf::Identity();
    child1.translationOffset = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
    joints.push_back(child1);

    // Child joint 2
    Joint child2;
    child2.name = "child2";
    child2.parent = 1;
    child2.preRotation = Eigen::Quaternionf::Identity();
    child2.translationOffset = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
    joints.push_back(child2);

    Skeleton skeleton(joints);

    // Create a simple mesh
    auto mesh = std::make_unique<Mesh>();
    mesh->vertices = {
        Eigen::Vector3f(0.0f, 0.0f, 0.0f),
        Eigen::Vector3f(1.0f, 0.0f, 0.0f),
        Eigen::Vector3f(0.0f, 1.0f, 0.0f)};
    mesh->faces = {Eigen::Vector3i(0, 1, 2)};
    mesh->normals = {
        Eigen::Vector3f(0.0f, 0.0f, 1.0f),
        Eigen::Vector3f(0.0f, 0.0f, 1.0f),
        Eigen::Vector3f(0.0f, 0.0f, 1.0f)};

    // Create skin weights
    auto skinWeights = std::make_unique<SkinWeights>();
    skinWeights->index = IndexMatrix::Zero(3, kMaxSkinJoints);
    skinWeights->weight = WeightMatrix::Zero(3, kMaxSkinJoints);

    // Simple skinning: each vertex influenced by one joint
    skinWeights->index(0, 0) = 0;
    skinWeights->weight(0, 0) = 1.0f;
    skinWeights->index(1, 0) = 1;
    skinWeights->weight(1, 0) = 1.0f;
    skinWeights->index(2, 0) = 2;
    skinWeights->weight(2, 0) = 1.0f;

    // Create locators
    LocatorList locators;
    Locator locator;
    locator.name = "test_locator";
    locator.parent = 0;
    locator.offset = Eigen::Vector3f(0.5f, 0.5f, 0.5f);
    locator.locked = Eigen::Vector3i::Zero();
    locator.weight = 1.0f;
    locator.limitOrigin = locator.offset;
    locator.limitWeight = Eigen::Vector3f::Zero();
    locators.push_back(locator);

    // Create collision geometry
    auto collision = std::make_unique<CollisionGeometry>();
    TaperedCapsule capsule;
    capsule.transformation = Eigen::Affine3f::Identity();
    capsule.radius = Eigen::Vector2f(0.1f, 0.05f);
    capsule.parent = 0;
    capsule.length = 1.0f;
    collision->push_back(capsule);

    // Create parameter transform
    ParameterTransform parameterTransform =
        ParameterTransform::empty(skeleton.joints.size() * kParametersPerJoint);

    return {
        skeleton,
        parameterTransform,
        ParameterLimits(),
        locators,
        mesh.get(),
        skinWeights.get(),
        collision.get()};
  }

  Character testCharacter_;
};

TEST_F(LegacyJsonIOTest, RoundTripConversion) {
  // Convert character to legacy JSON
  nlohmann::json legacyJson = characterToLegacyJson(testCharacter_);

  // Verify the JSON structure
  EXPECT_TRUE(legacyJson.contains("Skeleton"));
  EXPECT_TRUE(legacyJson.contains("SkinnedModel"));
  EXPECT_TRUE(legacyJson.contains("Collision"));
  EXPECT_TRUE(legacyJson.contains("Locators"));

  // Convert back to character
  Character roundTripCharacter = loadCharacterFromLegacyJsonString(legacyJson.dump());

  // Verify skeleton structure
  EXPECT_EQ(roundTripCharacter.skeleton.joints.size(), testCharacter_.skeleton.joints.size());

  for (size_t i = 0; i < testCharacter_.skeleton.joints.size(); ++i) {
    const auto& originalJoint = testCharacter_.skeleton.joints[i];
    const auto& roundTripJoint = roundTripCharacter.skeleton.joints[i];

    EXPECT_EQ(originalJoint.name, roundTripJoint.name);
    EXPECT_EQ(originalJoint.parent, roundTripJoint.parent);
    EXPECT_TRUE(originalJoint.preRotation.isApprox(roundTripJoint.preRotation, 1e-5f));
    EXPECT_TRUE(originalJoint.translationOffset.isApprox(roundTripJoint.translationOffset, 1e-5f));
  }

  // Verify mesh data
  ASSERT_NE(roundTripCharacter.mesh, nullptr);
  EXPECT_EQ(roundTripCharacter.mesh->vertices.size(), testCharacter_.mesh->vertices.size());
  EXPECT_EQ(roundTripCharacter.mesh->faces.size(), testCharacter_.mesh->faces.size());

  // Verify locators
  EXPECT_EQ(roundTripCharacter.locators.size(), testCharacter_.locators.size());
  if (!roundTripCharacter.locators.empty()) {
    const auto& originalLocator = testCharacter_.locators[0];
    const auto& roundTripLocator = roundTripCharacter.locators[0];

    EXPECT_EQ(originalLocator.name, roundTripLocator.name);
    EXPECT_EQ(originalLocator.parent, roundTripLocator.parent);
    EXPECT_TRUE(originalLocator.offset.isApprox(roundTripLocator.offset, 1e-5f));
  }

  // Verify collision geometry
  ASSERT_NE(roundTripCharacter.collision, nullptr);
  EXPECT_EQ(roundTripCharacter.collision->size(), testCharacter_.collision->size());
}

TEST_F(LegacyJsonIOTest, SkeletonOnlyConversion) {
  // Create a character with only skeleton data
  Character skeletonOnlyCharacter(
      testCharacter_.skeleton,
      testCharacter_.parameterTransform,
      testCharacter_.parameterLimits,
      LocatorList(),
      nullptr,
      nullptr,
      nullptr);

  // Convert to legacy JSON
  nlohmann::json legacyJson = characterToLegacyJson(skeletonOnlyCharacter);

  // Should only contain skeleton
  EXPECT_TRUE(legacyJson.contains("Skeleton"));
  EXPECT_FALSE(legacyJson.contains("SkinnedModel"));
  EXPECT_FALSE(legacyJson.contains("Collision"));
  EXPECT_FALSE(legacyJson.contains("Locators"));

  // Convert back
  Character roundTripCharacter = loadCharacterFromLegacyJsonString(legacyJson.dump());

  // Verify skeleton is preserved
  EXPECT_EQ(
      roundTripCharacter.skeleton.joints.size(), skeletonOnlyCharacter.skeleton.joints.size());
  EXPECT_EQ(roundTripCharacter.mesh, nullptr);
  EXPECT_EQ(roundTripCharacter.skinWeights, nullptr);
  EXPECT_EQ(roundTripCharacter.collision, nullptr);
  EXPECT_TRUE(roundTripCharacter.locators.empty());
}

TEST_F(LegacyJsonIOTest, InvalidJsonHandling) {
  // Test with invalid JSON
  EXPECT_THROW((void)loadCharacterFromLegacyJsonString("invalid json"), std::exception);

  // Test with JSON missing required fields
  nlohmann::json invalidJson;
  invalidJson["not_skeleton"] = "invalid";
  EXPECT_THROW((void)loadCharacterFromLegacyJsonString(invalidJson.dump()), std::runtime_error);
}

TEST_F(LegacyJsonIOTest, StringConversion) {
  // Convert to string
  std::string jsonString = characterToLegacyJsonString(testCharacter_);

  // Verify it's valid JSON
  nlohmann::json parsedJson;
  EXPECT_NO_THROW(parsedJson = nlohmann::json::parse(jsonString));

  // Convert back from string
  Character roundTripCharacter = loadCharacterFromLegacyJsonString(jsonString);

  // Verify structure is preserved
  EXPECT_EQ(roundTripCharacter.skeleton.joints.size(), testCharacter_.skeleton.joints.size());
}

TEST_F(LegacyJsonIOTest, BufferOperations) {
  // Convert to string first
  std::string jsonString = characterToLegacyJsonString(testCharacter_);

  // Create buffer
  std::vector<std::byte> buffer(jsonString.size());
  std::memcpy(buffer.data(), jsonString.data(), jsonString.size());

  // Load from buffer
  Character loadedCharacter = loadCharacterFromLegacyJsonBuffer(std::span<const std::byte>(buffer));

  // Verify loaded character
  EXPECT_EQ(loadedCharacter.skeleton.joints.size(), testCharacter_.skeleton.joints.size());
}

TEST_F(LegacyJsonIOTest, LegacyLocatorFormat) {
  // Test legacy locator format with separate X, Y, Z fields
  nlohmann::json legacyJson;
  legacyJson["skeleton"]["Bones"] = nlohmann::json::array();

  // Add a simple bone
  nlohmann::json bone;
  bone["Name"] = "root";
  bone["Parent"] = SIZE_MAX;
  bone["PreRotation"] = nlohmann::json::array({1.0, 0.0, 0.0, 0.0}); // w, x, y, z
  bone["TranslationOffset"] = nlohmann::json::array({0.0, 0.0, 0.0});
  bone["RestState"] = nlohmann::json{
      {"Rot", nlohmann::json::array({0.0, 0.0, 0.0})},
      {"Trans", nlohmann::json::array({0.0, 0.0, 0.0})},
      {"Scale", 0.0}};
  bone["JointType"] = "Root";
  bone["RotationOrder"] = "XYZ";
  legacyJson["skeleton"]["Bones"].push_back(bone);

  // Add legacy format locator
  legacyJson["locators"] = nlohmann::json::array();
  nlohmann::json locator;
  locator["name"] = "legacy_locator";
  locator["parent"] = 0;
  locator["offsetX"] = 1.0f;
  locator["offsetY"] = 2.0f;
  locator["offsetZ"] = 3.0f;
  legacyJson["locators"].push_back(locator);

  // Load character
  Character character = loadCharacterFromLegacyJsonString(legacyJson.dump());

  // Verify locator was loaded correctly
  EXPECT_EQ(character.locators.size(), 1);
  const auto& loadedLocator = character.locators[0];
  EXPECT_EQ(loadedLocator.name, "legacy_locator");
  EXPECT_EQ(loadedLocator.parent, 0);
  EXPECT_FLOAT_EQ(loadedLocator.offset.x(), 1.0f);
  EXPECT_FLOAT_EQ(loadedLocator.offset.y(), 2.0f);
  EXPECT_FLOAT_EQ(loadedLocator.offset.z(), 3.0f);
}

// Test that empty offsets array throws an appropriate error.
// This tests the size_t underflow fix in legacy_json_io.cpp.
TEST_F(LegacyJsonIOTest, EmptyOffsetsArrayHandling) {
  // Create JSON with empty offsets array (would cause size_t underflow without fix)
  nlohmann::json jsonWithEmptyOffsets;
  jsonWithEmptyOffsets["skeleton"]["Bones"] = nlohmann::json::array();

  // Add a simple bone
  nlohmann::json bone;
  bone["Name"] = "root";
  bone["Parent"] = SIZE_MAX;
  bone["PreRotation"] = nlohmann::json::array({1.0, 0.0, 0.0, 0.0});
  bone["TranslationOffset"] = nlohmann::json::array({0.0, 0.0, 0.0});
  bone["RestState"] = nlohmann::json{
      {"Rot", nlohmann::json::array({0.0, 0.0, 0.0})},
      {"Trans", nlohmann::json::array({0.0, 0.0, 0.0})},
      {"Scale", 0.0}};
  bone["JointType"] = "Root";
  bone["RotationOrder"] = "XYZ";
  jsonWithEmptyOffsets["skeleton"]["Bones"].push_back(bone);

  // Add SkinnedModel with empty offsets arrays (the edge case we fixed)
  jsonWithEmptyOffsets["SkinnedModel"] = nlohmann::json{};
  jsonWithEmptyOffsets["SkinnedModel"]["vertices"] = nlohmann::json::array();
  jsonWithEmptyOffsets["SkinnedModel"]["indices"] = nlohmann::json::array();
  jsonWithEmptyOffsets["SkinnedModel"]["skinningWeights"] = nlohmann::json::array();
  jsonWithEmptyOffsets["SkinnedModel"]["skinningOffsets"] =
      nlohmann::json::array(); // Empty offsets

  // This should either throw or handle gracefully (not crash with underflow)
  // With our fix, empty skinningOffsets is handled gracefully
  EXPECT_NO_THROW({
    Character character = loadCharacterFromLegacyJsonString(jsonWithEmptyOffsets.dump());
    // Verify the character was loaded (skeleton is present)
    EXPECT_EQ(character.skeleton.joints.size(), 1);
  });
}

// Test that JSON with valid single-element offsets works correctly.
// This verifies the boundary case is handled properly.
TEST_F(LegacyJsonIOTest, SingleElementOffsetsArrayHandling) {
  nlohmann::json jsonWithSingleOffset;
  jsonWithSingleOffset["skeleton"]["Bones"] = nlohmann::json::array();

  // Add a simple bone
  nlohmann::json bone;
  bone["Name"] = "root";
  bone["Parent"] = SIZE_MAX;
  bone["PreRotation"] = nlohmann::json::array({1.0, 0.0, 0.0, 0.0});
  bone["TranslationOffset"] = nlohmann::json::array({0.0, 0.0, 0.0});
  bone["RestState"] = nlohmann::json{
      {"Rot", nlohmann::json::array({0.0, 0.0, 0.0})},
      {"Trans", nlohmann::json::array({0.0, 0.0, 0.0})},
      {"Scale", 0.0}};
  bone["JointType"] = "Root";
  bone["RotationOrder"] = "XYZ";
  jsonWithSingleOffset["skeleton"]["Bones"].push_back(bone);

  // Add SkinnedModel with single-element offsets (another edge case)
  // Note: vertices must be an array of [x,y,z] arrays, not a flat array
  jsonWithSingleOffset["SkinnedModel"] = nlohmann::json{};
  jsonWithSingleOffset["SkinnedModel"]["vertices"] =
      nlohmann::json::array({nlohmann::json::array({0.0, 0.0, 0.0})}); // One vertex as [x,y,z]
  jsonWithSingleOffset["SkinnedModel"]["indices"] = nlohmann::json::array({0, 0, 0}); // One face
  jsonWithSingleOffset["SkinnedModel"]["skinningWeights"] = nlohmann::json::array(
      {nlohmann::json::array({0, 1.0})}); // One weight entry as [bone_idx, weight]
  jsonWithSingleOffset["SkinnedModel"]["skinningOffsets"] =
      nlohmann::json::array({0, 1}); // Offsets: vertex 0 starts at 0, ends at 1

  // This should handle gracefully
  EXPECT_NO_THROW({
    Character character = loadCharacterFromLegacyJsonString(jsonWithSingleOffset.dump());
    EXPECT_EQ(character.skeleton.joints.size(), 1);
  });
}
