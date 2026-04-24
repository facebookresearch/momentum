/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character/character.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/io/fbx/fbx_builder.h"
#include "momentum/io/fbx/fbx_io.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character/character_helpers_gtest.h"
#include "momentum/test/io/io_helpers.h"

#include <gtest/gtest.h>

using namespace momentum;

TEST(FbxBuilderTest, AddCharacter_RoundTripsNameAndMetadata) {
  Character character = createTestCharacter<float>(5);

  const auto outFile = temporaryFile("builder_char", "fbx");
  {
    FbxBuilder builder;
    builder.addCharacter(character);
    builder.save(outFile.path());
  }

  const Character loaded = loadFbxCharacter(outFile.path());
  EXPECT_EQ(character.name, loaded.name);
  EXPECT_EQ(character.metadata, loaded.metadata);
  EXPECT_EQ(character.skeleton.joints.size(), loaded.skeleton.joints.size());
}

TEST(FbxBuilderTest, AddCharacter_MatchesSaveFbxModel) {
  auto envVar = GetEnvVar("TEST_MOMENTUM_MODELS_PATH");
  if (!envVar.has_value()) {
    GTEST_SKIP() << "TEST_MOMENTUM_MODELS_PATH not set";
  }

  const filesystem::path fbxPath = filesystem::path(envVar.value()) / "blueman_s0.fbx";
  const Character original = loadFbxCharacter(fbxPath);

  // Save with legacy API
  const auto legacyFile = temporaryFile("legacy", "fbx");
  saveFbxModel(legacyFile.path(), original);

  // Save with FbxBuilder
  const auto builderFile = temporaryFile("builder", "fbx");
  {
    FbxBuilder builder;
    builder.addCharacter(original);
    builder.save(builderFile.path());
  }

  const Character fromLegacy = loadFbxCharacter(legacyFile.path());
  const Character fromBuilder = loadFbxCharacter(builderFile.path());
  compareChars(fromLegacy, fromBuilder);
}

TEST(FbxBuilderTest, AddCharacterWithMotion) {
  Character character = createTestCharacter<float>(5);

  const auto nModelParams = character.parameterTransform.numAllModelParameters();
  const Eigen::Index nFrames = 10;
  MatrixXf motion = MatrixXf::Random(nModelParams, nFrames) * 0.1f;

  const auto outFile = temporaryFile("builder_motion", "fbx");
  {
    FbxBuilder builder;
    builder.addCharacter(character);
    builder.addMotion(character, 30.0f, motion);
    builder.save(outFile.path());
  }

  auto [loaded, motionData, fps] = loadFbxCharacterWithMotion(outFile.path());
  EXPECT_EQ(loaded.skeleton.joints.size(), character.skeleton.joints.size());
  EXPECT_NEAR(fps, 30.0f, 1e-3f);
  // motionData[0] has joint parameters; verify we got the right number of frames
  ASSERT_FALSE(motionData.empty());
  EXPECT_EQ(motionData[0].cols(), nFrames);
}

TEST(FbxBuilderTest, AddRigidBody_HasCorrectJointCount) {
  Character character = createTestCharacter<float>(3);

  const auto outFile = temporaryFile("builder_rigid", "fbx");
  {
    FbxBuilder builder;
    builder.addRigidBody(character, "prop");
    builder.save(outFile.path());
  }

  // Rigid bodies have no skin weights, so load in permissive mode
  const Character loaded = loadFbxCharacter(outFile.path(), KeepLocators::No, Permissive::Yes);
  EXPECT_EQ(loaded.skeleton.joints.size(), character.skeleton.joints.size());
}

TEST(FbxBuilderTest, AddRigidBody_NonRootParentJoint) {
  Character character = createTestCharacter<float>(3);

  const auto outFile = temporaryFile("builder_rigid_joint1", "fbx");
  {
    FbxBuilder builder;
    // Parent mesh under joint1 instead of root
    builder.addRigidBody(character, "prop", 1);
    builder.save(outFile.path());
  }

  const Character loaded = loadFbxCharacter(outFile.path(), KeepLocators::No, Permissive::Yes);
  EXPECT_EQ(loaded.skeleton.joints.size(), character.skeleton.joints.size());
}

TEST(FbxBuilderTest, AddRigidBodyWithMotion) {
  Character character = createTestCharacter<float>(3);

  const auto nJointParams =
      static_cast<Eigen::Index>(character.parameterTransform.numJointParameters());
  const Eigen::Index nFrames = 5;
  MatrixXf jointParams = MatrixXf::Zero(nJointParams, nFrames);
  // Set some translation on the root joint across frames
  for (Eigen::Index f = 0; f < nFrames; f++) {
    jointParams(0, f) = static_cast<float>(f) * 0.1f; // tx
  }

  const auto outFile = temporaryFile("builder_rigid_motion", "fbx");
  {
    FbxBuilder builder;
    builder.addRigidBody(character, "prop");
    builder.addMotionWithJointParams(character, 30.0f, jointParams);
    builder.save(outFile.path());
  }

  // Rigid bodies have no skin weights, so load in permissive mode
  auto [loaded, motionData, fps] =
      loadFbxCharacterWithMotion(outFile.path(), KeepLocators::No, Permissive::Yes);
  EXPECT_EQ(loaded.skeleton.joints.size(), character.skeleton.joints.size());
  EXPECT_NEAR(fps, 30.0f, 1e-3f);
  ASSERT_FALSE(motionData.empty());
  EXPECT_EQ(motionData[0].cols(), nFrames);
}

TEST(FbxBuilderTest, MultipleCharacters) {
  Character body = createTestCharacter<float>(5);
  body.name = "body";
  Character prop = createTestCharacter<float>(3);
  prop.name = "prop";

  const auto outFile = temporaryFile("builder_multi", "fbx");
  {
    FbxBuilder builder;
    builder.addCharacter(body);
    builder.addRigidBody(prop, "prop");
    builder.save(outFile.path());
  }

  // The FBX should load without error; permissive because of the rigid body mesh
  const Character loaded = loadFbxCharacter(outFile.path(), KeepLocators::No, Permissive::Yes);
  EXPECT_FALSE(loaded.skeleton.joints.empty());
}

TEST(FbxBuilderTest, MarkerSequence) {
  Character character = createTestCharacter<float>(3);

  std::vector<std::vector<Marker>> markerSeq(5);
  for (size_t f = 0; f < markerSeq.size(); f++) {
    Marker m;
    m.name = "test_marker";
    m.pos = Vector3d(static_cast<double>(f), 0.0, 0.0);
    m.occluded = false;
    markerSeq[f].push_back(m);
  }

  const auto outFile = temporaryFile("builder_markers", "fbx");
  {
    FbxBuilder builder;
    builder.addCharacter(character);
    builder.addMarkerSequence(30.0f, markerSeq);
    builder.save(outFile.path());
  }

  auto markerResult = loadFbxMarkerSequence(outFile.path());
  EXPECT_FALSE(markerResult.frames.empty());
}

TEST(FbxBuilderTest, SaveConsumesBuilder) {
  Character character = createTestCharacter<float>(3);

  FbxBuilder builder;
  builder.addCharacter(character);

  const auto outFile = temporaryFile("builder_consumed", "fbx");
  builder.save(outFile.path());

  // After save, the builder is consumed — further operations should throw
  EXPECT_THROW(builder.addCharacter(character), std::runtime_error);
  EXPECT_THROW(builder.save(outFile.path()), std::runtime_error);
}
