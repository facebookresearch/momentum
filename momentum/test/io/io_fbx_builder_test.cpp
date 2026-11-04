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

#include <algorithm>

using namespace momentum;

TEST(FbxBuilderTest, AddCharacter_RoundTripsNameAndMetadata) {
  Character character = createTestCharacter(5);

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
  Character character = createTestCharacter(5);

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
  Character character = createTestCharacter(3);

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
  Character character = createTestCharacter(3);

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
  Character character = createTestCharacter(3);

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
  Character body = createTestCharacter(5);
  Character prop = createTestCharacter(3);

  const auto numPropJointParams =
      static_cast<Eigen::Index>(prop.parameterTransform.numJointParameters());
  const MatrixXf propJointParams = MatrixXf::Zero(numPropJointParams, 2);

  const auto outFile = temporaryFile("builder_multi", "fbx");
  {
    FbxBuilder builder;
    const std::string bodyName = builder.addCharacter(body);
    const std::string propName = builder.addRigidBody(prop);
    EXPECT_EQ(bodyName, "test character");
    EXPECT_EQ(propName, "test character_1");
    builder.addMotionWithJointParams(prop, 30.0f, propJointParams, propName);
    builder.save(outFile.path());
  }

  const Character loaded = loadFbxCharacter(outFile.path(), KeepLocators::No, Permissive::Yes);
  EXPECT_EQ(
      loaded.skeleton.joints.size(), body.skeleton.joints.size() + prop.skeleton.joints.size());
  EXPECT_EQ(
      std::count_if(
          loaded.skeleton.joints.begin(),
          loaded.skeleton.joints.end(),
          [](const Joint& joint) { return joint.parent == kInvalidIndex; }),
      2);
}

TEST(FbxBuilderTest, MarkerSequence) {
  Character character = createTestCharacter(3);

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
  Character character = createTestCharacter(3);

  FbxBuilder builder;
  builder.addCharacter(character);

  const auto outFile = temporaryFile("builder_consumed", "fbx");
  builder.save(outFile.path());

  // After save, the builder is consumed — further operations should throw
  EXPECT_THROW(builder.addCharacter(character), std::runtime_error);
  EXPECT_THROW(builder.save(outFile.path()), std::runtime_error);
}

TEST(FbxBuilderTest, DuplicateNamesGetSequentialSuffixes) {
  FbxBuilder builder;
  EXPECT_EQ(builder.addRigidBody(createTestCharacter(3)), "test character");
  EXPECT_EQ(builder.addRigidBody(createTestCharacter(3)), "test character_1");
  EXPECT_EQ(builder.addRigidBody(createTestCharacter(3)), "test character_2");
}

TEST(FbxBuilderTest, DuplicateSkinnedCharacterNamesArePrefixed) {
  Character body = createTestCharacter(4);
  Character other = createTestCharacter(6);

  const auto outFile = temporaryFile("builder_dup_skinned", "fbx");
  std::string name1;
  std::string name2;
  {
    FbxBuilder builder;
    name1 = builder.addCharacter(body);
    name2 = builder.addCharacter(other);
    EXPECT_EQ(name1, "test character");
    EXPECT_EQ(name2, "test character_1");
    builder.save(outFile.path());
  }

  // Load without stripping namespaces so the deduped second character's prefix is visible.
  const Character loaded = loadFbxCharacter(
      outFile.path(), KeepLocators::No, Permissive::Yes, LoadBlendShapes::No, false);
  EXPECT_EQ(
      loaded.skeleton.joints.size(), body.skeleton.joints.size() + other.skeleton.joints.size());
  EXPECT_TRUE(
      std::any_of(
          loaded.skeleton.joints.begin(),
          loaded.skeleton.joints.end(),
          [&name2](const Joint& joint) { return joint.name.rfind(name2 + ":", 0) == 0; }));
}

TEST(FbxBuilderTest, AddMotionWrongExplicitNameThrows) {
  Character body = createTestCharacter(5);
  Character prop = createTestCharacter(3);

  const auto numBodyJointParams =
      static_cast<Eigen::Index>(body.parameterTransform.numJointParameters());
  const auto numPropJointParams =
      static_cast<Eigen::Index>(prop.parameterTransform.numJointParameters());
  const MatrixXf bodyJointParams = MatrixXf::Zero(numBodyJointParams, 2);
  const MatrixXf propJointParams = MatrixXf::Zero(numPropJointParams, 2);

  FbxBuilder builder;
  const std::string bodyName = builder.addCharacter(body);
  const std::string propName = builder.addRigidBody(prop);
  ASSERT_EQ(bodyName, "test character");
  ASSERT_EQ(propName, "test character_1");

  // Passing the explicit name of a different-skeleton character (the rigid body, 3 joints) while
  // animating body (5 joints) must throw instead of indexing out of bounds.
  EXPECT_THROW(
      builder.addMotionWithJointParams(body, 30.0f, bodyJointParams, propName), std::exception);
  // Empty characterName resolves to the character exported under character.name (body), which
  // matches.
  EXPECT_NO_THROW(builder.addMotionWithJointParams(body, 30.0f, bodyJointParams));
  // Passing the exact returned name targets the correct character.
  EXPECT_NO_THROW(builder.addMotionWithJointParams(prop, 30.0f, propJointParams, propName));
}

TEST(FbxBuilderTest, UnnamedCharacterIsReachableAfterUnnamedRigidBody) {
  Character body = createTestCharacter(5);
  body.name = "";
  Character prop = createTestCharacter(3);
  prop.name = "";

  FbxBuilder builder;
  ASSERT_EQ(builder.addRigidBody(prop, "prop"), "prop");
  ASSERT_EQ(builder.addCharacter(body), "");

  // The empty name the body is exported under selects the body, not the earlier unnamed prop
  // (whose joint count differs, so selecting it would throw).
  const MatrixXf bodyJointParams =
      MatrixXf::Zero(static_cast<Eigen::Index>(body.parameterTransform.numJointParameters()), 2);
  EXPECT_NO_THROW(builder.addMotionWithJointParams(body, 30.0f, bodyJointParams, ""));
}

TEST(FbxBuilderTest, AddMotionUnknownCharacterThrows) {
  Character character = createTestCharacter(3);
  const auto numJointParams =
      static_cast<Eigen::Index>(character.parameterTransform.numJointParameters());
  const MatrixXf jointParams = MatrixXf::Zero(numJointParams, 2);

  FbxBuilder builder;
  EXPECT_THROW(builder.addMotionWithJointParams(character, 30.0f, jointParams), std::exception);
  EXPECT_THROW(
      builder.addMotionWithJointParams(character, 30.0f, jointParams, "nonexistent"),
      std::exception);
}

TEST(FbxBuilderTest, DedupeSkipsNamesTakenByExplicitNames) {
  FbxBuilder builder;
  EXPECT_EQ(builder.addRigidBody(createTestCharacter(3), "dup_1"), "dup_1");
  EXPECT_EQ(builder.addRigidBody(createTestCharacter(3), "dup"), "dup");
  // The auto-generated "dup_1" is already taken, so the next free suffix is used.
  EXPECT_EQ(builder.addRigidBody(createTestCharacter(3), "dup"), "dup_2");
  // An explicit name colliding with an existing one is itself suffixed.
  EXPECT_EQ(builder.addRigidBody(createTestCharacter(3), "dup_1"), "dup_1_1");
}

TEST(FbxBuilderTest, LoadMultipleCharacters) {
  Character charA = createTestCharacter(4);
  charA.name = "alpha";
  Character charB = createTestCharacter(6);
  charB.name = "beta";

  const auto outFile = temporaryFile("builder_multi_load", "fbx");
  {
    FbxBuilder builder;
    builder.addCharacter(charA);
    builder.addCharacter(charB);
    builder.save(outFile.path());
  }

  // Permissive since the round trip may produce nonzero rest rotations.
  const std::vector<Character> characters =
      loadFbxCharacters(outFile.path(), KeepLocators::No, Permissive::Yes);
  ASSERT_EQ(characters.size(), 2);
  for (size_t i = 0; i < characters.size(); ++i) {
    const Character& source = i == 0 ? charA : charB;
    const Character& loaded = characters[i];
    EXPECT_EQ(loaded.name, source.name);
    EXPECT_EQ(loaded.skeleton.joints.size(), source.skeleton.joints.size());
    EXPECT_EQ(
        loaded.parameterTransform.numAllModelParameters(),
        loaded.skeleton.joints.size() * kParametersPerJoint);
    ASSERT_NE(loaded.mesh, nullptr);
    EXPECT_EQ(loaded.mesh->vertices.size(), source.mesh->vertices.size());
  }
}

namespace {

// Expect every vertex of `loaded` to be fully weighted on `joint`, and to map back to
// `localVertices` through that joint's inverse bind pose.
void expectRigidlyBound(
    const Character& loaded,
    size_t joint,
    const std::vector<Vector3f>& localVertices) {
  ASSERT_NE(loaded.mesh, nullptr);
  ASSERT_NE(loaded.skinWeights, nullptr);
  ASSERT_EQ(loaded.mesh->vertices.size(), localVertices.size());
  for (size_t v = 0; v < localVertices.size(); ++v) {
    EXPECT_EQ(loaded.skinWeights->index(v, 0), joint);
    EXPECT_FLOAT_EQ(loaded.skinWeights->weight(v, 0), 1.0f);
    EXPECT_TRUE((loaded.inverseBindPose[joint] * loaded.mesh->vertices[v])
                    .isApprox(localVertices[v], 1e-3f));
  }
}

// Three frames of joint parameters for a one-joint object.
MatrixXf objectMotion() {
  MatrixXf jointParams = MatrixXf::Zero(kParametersPerJoint, 3);
  for (Eigen::Index f = 0; f < jointParams.cols(); ++f) {
    jointParams.col(f) << 1.0f + f, 2.0f, -0.5f * f, 0.1f * f, 0.2f, -0.3f * f, 0.0f;
  }
  return jointParams;
}

} // namespace

TEST(FbxBuilderTest, LoadCharactersWithObjects) {
  const Character prop = createTestCharacter(3);
  const Character object = createTestCharacter(3);
  const Vector3f objectOffset(0.5f, 1.0f, -2.0f);
  constexpr size_t kPropMeshJoint = 1;

  const auto outFile = temporaryFile("builder_objects", "fbx");
  {
    FbxBuilder builder;
    builder.addRigidBody(prop, "prop", kPropMeshJoint);
    builder.addAnimatedMesh(*object.mesh, "object", 30.0f, objectMotion(), objectOffset);
    builder.save(outFile.path());
  }

  const std::vector<Character> characters = loadFbxCharacters(outFile.path());
  ASSERT_EQ(characters.size(), 2);

  // A mesh under a joint moves rigidly with that joint.
  EXPECT_EQ(characters[0].name, "prop");
  EXPECT_EQ(characters[0].skeleton.joints.size(), prop.skeleton.joints.size());
  expectRigidlyBound(characters[0], kPropMeshJoint, prop.mesh->vertices);

  // An animated mesh is a one-joint character.
  EXPECT_EQ(characters[1].name, "object");
  ASSERT_EQ(characters[1].skeleton.joints.size(), 1);
  expectRigidlyBound(characters[1], 0, object.mesh->vertices);
}
