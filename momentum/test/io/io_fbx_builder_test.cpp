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

TEST(FbxBuilderTest, MotionWithWrongRowCountThrows) {
  const Character character = createTestCharacter(3);
  FbxBuilder builder;
  builder.addCharacter(character);
  const auto numModelParams = character.parameterTransform.numAllModelParameters();
  const auto numJointParams = character.parameterTransform.numJointParameters();
  EXPECT_THROW(
      builder.addMotion(character, 30.0f, MatrixXf::Zero(numModelParams - 1, 2)), std::exception);
  EXPECT_THROW(
      builder.addMotionWithJointParams(character, 30.0f, MatrixXf::Zero(numJointParams - 1, 2)),
      std::exception);
}

TEST(FbxBuilderTest, AnimatedMeshWithFewerThanSixRowsThrows) {
  const Character object = createTestCharacter(3);
  FbxBuilder builder;
  EXPECT_THROW(
      builder.addAnimatedMesh(*object.mesh, "object", 30.0f, MatrixXf::Zero(5, 2)), std::exception);
}

TEST(FbxBuilderTest, NamespaceFromOptionsPrefixesEveryNode) {
  const Character character = createTestCharacter(3);
  FileSaveOptions options;
  options.fbxNamespace = "ns";

  const auto outFile = temporaryFile("builder_namespace", "fbx");
  {
    FbxBuilder builder;
    builder.addCharacter(character, options);
    builder.save(outFile.path());
  }

  const Character prefixed = loadFbxCharacter(
      outFile.path(), KeepLocators::No, Permissive::Yes, LoadBlendShapes::No, false);
  ASSERT_EQ(prefixed.skeleton.joints.size(), character.skeleton.joints.size());
  for (size_t i = 0; i < character.skeleton.joints.size(); ++i) {
    EXPECT_EQ(prefixed.skeleton.joints[i].name, "ns:" + character.skeleton.joints[i].name);
  }
}

TEST(FbxBuilderTest, FileHasOneNamespace) {
  Character second = createTestCharacter(3);
  second.name = "second";
  FileSaveOptions first;
  first.fbxNamespace = "a";
  FileSaveOptions same;
  same.fbxNamespace = "a:";
  FileSaveOptions other;
  other.fbxNamespace = "b";

  FbxBuilder builder;
  builder.addCharacter(createTestCharacter(3), first);
  EXPECT_THROW(builder.addCharacter(second, other), std::exception);
  EXPECT_THROW(builder.addRigidBody(second, "prop", 0, other), std::exception);
  EXPECT_NO_THROW(builder.addCharacter(second, same));
}

TEST(FbxBuilderTest, FailedAddDoesNotSetTheNamespace) {
  FileSaveOptions first;
  first.fbxNamespace = "a";
  FileSaveOptions other;
  other.fbxNamespace = "b";

  FbxBuilder builder;
  // The parent joint is out of range, so the call fails after it checked the namespace.
  EXPECT_THROW(builder.addRigidBody(createTestCharacter(3), "prop", 99, first), std::exception);
  EXPECT_NO_THROW(builder.addCharacter(createTestCharacter(3), other));
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

TEST(FbxBuilderTest, AnimatedMeshNamesAreUniqueAmongCharacters) {
  Character body = createTestCharacter(3);
  body.name = "body";
  Character later = createTestCharacter(3);
  later.name = "body_1";
  const Character object = createTestCharacter(3);

  const auto outFile = temporaryFile("builder_mesh_names", "fbx");
  {
    FbxBuilder builder;
    EXPECT_EQ(builder.addCharacter(body), "body");
    EXPECT_EQ(builder.addAnimatedMesh(*object.mesh, "body", 30.0f, objectMotion()), "body_1");
    // The mesh keeps its name from a character added after it.
    EXPECT_EQ(builder.addCharacter(later), "body_1_1");
    builder.save(outFile.path());
  }

  std::vector<std::string> names;
  for (const Character& loaded :
       loadFbxCharacters(outFile.path(), KeepLocators::No, Permissive::Yes)) {
    names.push_back(loaded.name);
  }
  std::ranges::sort(names);
  EXPECT_EQ(names, (std::vector<std::string>{"body", "body_1", "body_1_1"}));
}

namespace {

// Expect `motion` on `loaded` to pose every joint as `expectedMotion` poses it on `expected`.
void expectSameJointTransforms(
    const Character& loaded,
    const MatrixXf& motion,
    const Skeleton& expected,
    const MatrixXf& expectedMotion) {
  ASSERT_EQ(motion.cols(), expectedMotion.cols());
  for (Eigen::Index f = 0; f < motion.cols(); ++f) {
    const SkeletonState state(JointParameters(motion.col(f)), loaded.skeleton);
    const SkeletonState expectedState(JointParameters(expectedMotion.col(f)), expected);
    for (size_t j = 0; j < expected.joints.size(); ++j) {
      EXPECT_TRUE(
          state.jointState[j].transform.isApprox(expectedState.jointState[j].transform, 1e-4f))
          << "frame " << f << ", joint " << j;
    }
  }
}

} // namespace

TEST(FbxBuilderTest, AnimatedMeshWithoutScaleRowHasUnitScale) {
  const Character object = createTestCharacter(3);
  // Translation and rotation only, with a constant nonzero rotation about z.
  MatrixXf motion = objectMotion().topRows(6);
  motion.row(5).setConstant(0.4f);

  const auto outFile = temporaryFile("builder_mesh_no_scale", "fbx");
  {
    FbxBuilder builder;
    builder.addAnimatedMesh(*object.mesh, "object", 30.0f, motion);
    builder.save(outFile.path());
  }

  const auto [characters, loadedFps] = loadFbxCharactersWithMotion(outFile.path());
  ASSERT_EQ(characters.size(), 1);
  const auto& [loaded, takes] = characters[0];
  ASSERT_EQ(takes.size(), 1);
  MatrixXf expectedMotion = MatrixXf::Zero(kParametersPerJoint, motion.cols());
  expectedMotion.topRows(6) = motion;
  Skeleton objectSkeleton;
  objectSkeleton.joints.emplace_back();
  expectSameJointTransforms(loaded, takes[0], objectSkeleton, expectedMotion);
}

TEST(FbxBuilderTest, LoadMultipleCharactersWithMotion) {
  const Character prop = createTestCharacter(3);
  const Character object = createTestCharacter(3);
  const Vector3f objectOffset(0.5f, 1.0f, -2.0f);
  MatrixXf propMotion = MatrixXf::Zero(prop.parameterTransform.numJointParameters(), 5);
  for (Eigen::Index f = 0; f < propMotion.cols(); ++f) {
    propMotion(1, f) = 0.2f * f;
  }

  const float fps = 30.0f;
  const auto outFile = temporaryFile("builder_multi_motion", "fbx");
  {
    FbxBuilder builder;
    const std::string propName = builder.addRigidBody(prop, "prop");
    builder.addMotionWithJointParams(prop, fps, propMotion, propName);
    builder.addAnimatedMesh(*object.mesh, "object", fps, objectMotion(), objectOffset);
    builder.save(outFile.path());
  }

  const auto [characters, loadedFps] = loadFbxCharactersWithMotion(outFile.path());
  ASSERT_EQ(characters.size(), 2);
  EXPECT_NEAR(loadedFps, fps, 1e-3f);

  // Each character reads only the curves on its own joints from the shared animation stack.
  const auto& [loadedProp, propTakes] = characters[0];
  ASSERT_EQ(propTakes.size(), 1);
  expectSameJointTransforms(loadedProp, propTakes[0], prop.skeleton, propMotion);

  Skeleton objectSkeleton;
  objectSkeleton.joints.emplace_back();
  objectSkeleton.joints[0].translationOffset = objectOffset;
  const auto& [loadedObject, objectTakes] = characters[1];
  ASSERT_EQ(objectTakes.size(), 1);
  expectSameJointTransforms(loadedObject, objectTakes[0], objectSkeleton, objectMotion());
}

TEST(FbxBuilderTest, CharacterNoStackAnimatesGetsEmptyTakes) {
  const Character body = createTestCharacter(3);
  const Character prop = createTestCharacter(3);
  MatrixXf bodyMotion = MatrixXf::Zero(body.parameterTransform.numJointParameters(), 5);
  for (Eigen::Index f = 0; f < bodyMotion.cols(); ++f) {
    bodyMotion(1, f) = 0.2f * f;
  }

  const auto outFile = temporaryFile("builder_one_animated", "fbx");
  {
    FbxBuilder builder;
    const std::string bodyName = builder.addCharacter(body);
    builder.addMotionWithJointParams(body, 30.0f, bodyMotion, bodyName);
    builder.addRigidBody(prop, "prop");
    builder.save(outFile.path());
  }

  // The takes are index-aligned across characters, so the unanimated prop gets an empty take.
  const auto [characters, fps] = loadFbxCharactersWithMotion(outFile.path());
  ASSERT_EQ(characters.size(), 2);
  const auto& [loadedBody, bodyTakes] = characters[0];
  ASSERT_EQ(bodyTakes.size(), 1);
  EXPECT_EQ(bodyTakes[0].cols(), bodyMotion.cols());
  const auto& [loadedProp, propTakes] = characters[1];
  ASSERT_EQ(propTakes.size(), 1);
  EXPECT_EQ(propTakes[0].cols(), 0);
}

namespace {

// The joint parameters that addMotion() writes for `motion` and `offsets`: PT * pose + offsets.
MatrixXf expectedJointParameters(
    const Character& character,
    const MatrixXf& motion,
    const VectorXf& offsets) {
  MatrixXf result(character.parameterTransform.numJointParameters(), motion.cols());
  for (Eigen::Index f = 0; f < motion.cols(); ++f) {
    result.col(f) =
        character.parameterTransform.apply(ModelParameters(VectorXf(motion.col(f)))).v + offsets;
  }
  return result;
}

// The first joint parameter that no model parameter drives, or -1.
Eigen::Index undrivenJointParameter(const Character& character) {
  const auto& active = character.parameterTransform.activeJointParams;
  for (Eigen::Index p = 0; p < active.size(); ++p) {
    if (!active[p]) {
      return p;
    }
  }
  return -1;
}

} // namespace

TEST(FbxBuilderTest, MotionKeepsTheOffsetOfAnUndrivenJointParameter) {
  const Character character = createTestCharacter(3);
  const Eigen::Index undriven = undrivenJointParameter(character);
  ASSERT_GE(undriven, 0);
  MatrixXf motion = MatrixXf::Zero(character.parameterTransform.numAllModelParameters(), 3);
  for (Eigen::Index f = 0; f < motion.cols(); ++f) {
    motion(0, f) = 0.5f * static_cast<float>(f);
  }
  VectorXf offsets = VectorXf::Zero(character.parameterTransform.numJointParameters());
  offsets[undriven] = 0.25f;

  const auto outFile = temporaryFile("builder_undriven_offset", "fbx");
  {
    FbxBuilder builder;
    const std::string name = builder.addCharacter(character);
    builder.addMotion(character, 30.0, motion, offsets, name);
    builder.save(outFile.path());
  }

  const auto [characters, fps] = loadFbxCharactersWithMotion(outFile.path());
  ASSERT_EQ(characters.size(), 1);
  ASSERT_EQ(characters[0].second.size(), 1);
  const MatrixXf& loaded = characters[0].second[0];
  const MatrixXf expected = expectedJointParameters(character, motion, offsets);
  ASSERT_EQ(loaded.rows(), expected.rows());
  ASSERT_EQ(loaded.cols(), expected.cols());
  EXPECT_LT((loaded - expected).cwiseAbs().maxCoeff(), 1e-4f);
}

TEST(FbxBuilderTest, LaterMotionReplacesTheEarlierOne) {
  const Character character = createTestCharacter(3);
  const Eigen::Index undriven = undrivenJointParameter(character);
  ASSERT_GE(undriven, 0);
  const auto numModelParameters = character.parameterTransform.numAllModelParameters();
  const auto numJointParameters = character.parameterTransform.numJointParameters();
  const MatrixXf first = MatrixXf::Constant(numModelParameters, 5, 0.3f);
  VectorXf firstOffsets = VectorXf::Zero(numJointParameters);
  firstOffsets[undriven] = 0.25f;
  const MatrixXf second = MatrixXf::Constant(numModelParameters, 2, -0.2f);

  const auto outFile = temporaryFile("builder_replaced_motion", "fbx");
  {
    FbxBuilder builder;
    const std::string name = builder.addCharacter(character);
    builder.addMotion(character, 30.0, first, firstOffsets, name);
    // Zero offsets: the undriven parameter's curve is rewritten only because it exists.
    builder.addMotion(character, 30.0, second, VectorXf::Zero(numJointParameters), name);
    builder.save(outFile.path());
  }

  const auto [characters, fps] = loadFbxCharactersWithMotion(outFile.path());
  ASSERT_EQ(characters.size(), 1);
  ASSERT_EQ(characters[0].second.size(), 1);
  const MatrixXf& loaded = characters[0].second[0];
  const MatrixXf expected =
      expectedJointParameters(character, second, VectorXf::Zero(numJointParameters));
  ASSERT_EQ(loaded.rows(), expected.rows());
  ASSERT_EQ(loaded.cols(), expected.cols());
  EXPECT_LT((loaded - expected).cwiseAbs().maxCoeff(), 1e-4f);
}

TEST(FbxBuilderTest, MotionWithoutOffsetsKeepsTheEarlierOffsets) {
  const Character character = createTestCharacter(3);
  const Eigen::Index undriven = undrivenJointParameter(character);
  ASSERT_GE(undriven, 0);
  const auto numModelParameters = character.parameterTransform.numAllModelParameters();
  VectorXf offsets = VectorXf::Zero(character.parameterTransform.numJointParameters());
  offsets[undriven] = 0.25f;
  const MatrixXf second = MatrixXf::Constant(numModelParameters, 2, -0.2f);

  const auto outFile = temporaryFile("builder_kept_offsets", "fbx");
  {
    FbxBuilder builder;
    const std::string name = builder.addCharacter(character);
    builder.addMotion(
        character, 30.0, MatrixXf::Constant(numModelParameters, 3, 0.3f), offsets, name);
    builder.addMotion(character, 30.0, second, {}, name);
    builder.save(outFile.path());
  }

  const auto [characters, fps] = loadFbxCharactersWithMotion(outFile.path());
  ASSERT_EQ(characters.size(), 1);
  ASSERT_EQ(characters[0].second.size(), 1);
  const MatrixXf& loaded = characters[0].second[0];
  const MatrixXf expected = expectedJointParameters(character, second, offsets);
  ASSERT_EQ(loaded.rows(), expected.rows());
  ASSERT_EQ(loaded.cols(), expected.cols());
  EXPECT_LT((loaded - expected).cwiseAbs().maxCoeff(), 1e-4f);
}

TEST(FbxBuilderTest, MotionAt72And96FramesPerSecondIsReadBack) {
  const Character character = createTestCharacter(3);
  const MatrixXf motion =
      MatrixXf::Constant(character.parameterTransform.numAllModelParameters(), 10, 0.2f);
  for (const double fps : {72.0, 96.0}) {
    const auto outFile = temporaryFile("builder_fps", "fbx");
    {
      FbxBuilder builder;
      const std::string name = builder.addCharacter(character);
      builder.addMotion(character, fps, motion, {}, name);
      builder.save(outFile.path());
    }
    const auto [characters, loadedFps] = loadFbxCharactersWithMotion(outFile.path());
    EXPECT_FLOAT_EQ(loadedFps, static_cast<float>(fps));
    ASSERT_EQ(characters.size(), 1);
    ASSERT_EQ(characters[0].second.size(), 1);
    EXPECT_EQ(characters[0].second[0].cols(), motion.cols()) << fps << " fps";
  }
}

TEST(FbxBuilderTest, SkeletonWithTwoRootsKeepsBothTrees) {
  // Joints root, joint1, joint2 and joint3 with parents -, 0, - and 2: two root joints.
  Character character = createTestCharacter(4);
  character.skeleton.joints[2].parent = kInvalidIndex;
  character.initInverseBindPose();

  const auto outFile = temporaryFile("builder_two_roots", "fbx");
  {
    FbxBuilder builder;
    builder.addCharacter(character);
    builder.save(outFile.path());
  }

  // The singular loader reads every skeleton of the scene as one character; the plural loaders
  // read one character per root joint.
  const Character loaded = loadFbxCharacter(outFile.path());
  EXPECT_EQ(loaded.skeleton.getJointNames(), character.skeleton.getJointNames());
  std::vector<size_t> parents(loaded.skeleton.joints.size());
  std::ranges::transform(loaded.skeleton.joints, parents.begin(), &Joint::parent);
  const std::vector<size_t> expectedParents = {kInvalidIndex, 0, kInvalidIndex, 2};
  EXPECT_EQ(parents, expectedParents);
  EXPECT_EQ(loaded.name, character.name);
}

TEST(FbxBuilderTest, MotionAndMarkersKeepTheirFrameCount) {
  const Character character = createTestCharacter(3);
  const auto numModelParameters = character.parameterTransform.numAllModelParameters();
  for (const double fps : {25.0, 29.95, 30.0}) {
    for (Eigen::Index n = 1; n <= 64; ++n) {
      Marker marker;
      marker.name = "m";
      marker.occluded = false;
      const std::vector<std::vector<Marker>> markers(n, {marker});
      const auto outFile = temporaryFile("builder_frame_count", "fbx");
      {
        FbxBuilder builder;
        const std::string name = builder.addCharacter(character);
        builder.addMotion(
            character, fps, MatrixXf::Constant(numModelParameters, n, 0.1f), {}, name);
        builder.addMarkerSequence(fps, markers);
        builder.save(outFile.path());
      }
      const auto [characters, loadedFps] = loadFbxCharactersWithMotion(outFile.path());
      ASSERT_EQ(characters.size(), 1);
      ASSERT_EQ(characters[0].second.size(), 1);
      EXPECT_EQ(characters[0].second[0].cols(), n) << n << " frames at " << fps << " fps";
      EXPECT_EQ(loadFbxMarkerSequence(outFile.path()).frames.size(), n)
          << n << " marker frames at " << fps << " fps";
    }
  }
}

TEST(FbxBuilderTest, AnimatedMeshWithAConstantTransformKeepsItsFrameCount) {
  const Character object = createTestCharacter(3);
  const MatrixXf motion = objectMotion().col(1).replicate(1, 4);

  const auto outFile = temporaryFile("builder_mesh_constant", "fbx");
  {
    FbxBuilder builder;
    builder.addAnimatedMesh(*object.mesh, "object", 30.0f, motion);
    builder.save(outFile.path());
  }

  const auto [characters, loadedFps] = loadFbxCharactersWithMotion(outFile.path());
  ASSERT_EQ(characters.size(), 1);
  const auto& [loaded, takes] = characters[0];
  ASSERT_EQ(takes.size(), 1);
  ASSERT_EQ(takes[0].cols(), motion.cols());
  Skeleton objectSkeleton;
  objectSkeleton.joints.emplace_back();
  expectSameJointTransforms(loaded, takes[0], objectSkeleton, motion);
}
