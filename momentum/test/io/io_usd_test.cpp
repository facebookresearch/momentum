/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/usd/usd_io.h"

#include "momentum/character/blend_shape.h"
#include "momentum/character/character.h"
#include "momentum/character/collision_geometry.h"
#include "momentum/character/locator.h"
#include "momentum/character/marker.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skin_weights.h"
#include "momentum/math/mesh.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/io/io_helpers.h"

#include <pxr/base/gf/half.h>
#include <pxr/base/gf/quatf.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec3h.h>
#include <pxr/base/vt/array.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdSkel/animation.h>
#include <pxr/usd/usdSkel/bindingAPI.h>
#include <pxr/usd/usdSkel/skeleton.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

using namespace momentum;

namespace {

std::optional<filesystem::path> getTestResourcePath(const std::string& filename) {
  auto envVar = GetEnvVar("TEST_RESOURCES_PATH");
  if (!envVar.has_value()) {
    return std::nullopt;
  }
  return filesystem::path(envVar.value()) / "usd" / filename;
}

} // namespace

class UsdIoTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a simple test character
    testCharacter = createTestCharacter();
  }

  Character testCharacter;
};

TEST_F(UsdIoTest, LoadSimpleCharacter) {
  auto usdPath = getTestResourcePath("simple_character.usda");
  if (!usdPath.has_value()) {
    GTEST_SKIP() << "Environment variable 'TEST_RESOURCES_PATH' is not set.";
  }

  if (!filesystem::exists(*usdPath)) {
    GTEST_SKIP() << "Test resource file not found: " << *usdPath;
  }

  const auto character = loadUsdCharacter(*usdPath);

  // Verify skeleton
  EXPECT_GT(character.skeleton.joints.size(), 0);

  // Verify mesh
  EXPECT_GT(character.mesh->vertices.size(), 0);
  EXPECT_GT(character.mesh->faces.size(), 0);
}

TEST_F(UsdIoTest, LoadSimpleMesh) {
  auto usdPath = getTestResourcePath("simple_mesh.usda");
  if (!usdPath.has_value()) {
    GTEST_SKIP() << "Environment variable 'TEST_RESOURCES_PATH' is not set.";
  }

  if (!filesystem::exists(*usdPath)) {
    GTEST_SKIP() << "Test resource file not found: " << *usdPath;
  }

  const auto character = loadUsdCharacter(*usdPath);

  // Verify mesh
  EXPECT_GT(character.mesh->vertices.size(), 0);
  EXPECT_GT(character.mesh->faces.size(), 0);
}

TEST_F(UsdIoTest, LoadCharacterWithMaterials) {
  auto usdPath = getTestResourcePath("character_with_materials.usda");
  if (!usdPath.has_value()) {
    GTEST_SKIP() << "Environment variable 'TEST_RESOURCES_PATH' is not set.";
  }

  if (!filesystem::exists(*usdPath)) {
    GTEST_SKIP() << "Test resource file not found: " << *usdPath;
  }

  const auto character = loadUsdCharacter(*usdPath);

  // Verify skeleton
  EXPECT_GT(character.skeleton.joints.size(), 0);

  // Verify mesh
  EXPECT_GT(character.mesh->vertices.size(), 0);
  EXPECT_GT(character.mesh->faces.size(), 0);
}

TEST_F(UsdIoTest, SaveAndLoadRoundTrip) {
  auto tempFile = temporaryFile("momentum_usd_roundtrip", ".usda");

  saveUsd(tempFile.path(), testCharacter);

  ASSERT_TRUE(filesystem::exists(tempFile.path()))
      << "USD file was not created at: " << tempFile.path();

  const auto loadedCharacter = loadUsdCharacter(tempFile.path());

  ASSERT_EQ(loadedCharacter.skeleton.joints.size(), testCharacter.skeleton.joints.size());
  ASSERT_EQ(loadedCharacter.mesh->vertices.size(), testCharacter.mesh->vertices.size());
  ASSERT_EQ(loadedCharacter.mesh->faces.size(), testCharacter.mesh->faces.size());

  for (size_t i = 0; i < testCharacter.skeleton.joints.size(); ++i) {
    const auto& original = testCharacter.skeleton.joints[i];
    const auto& loaded = loadedCharacter.skeleton.joints[i];

    EXPECT_EQ(loaded.name, original.name) << "Joint name mismatch at index " << i;

    EXPECT_TRUE(loaded.preRotation.toRotationMatrix().isApprox(
        original.preRotation.toRotationMatrix(), 1e-4f))
        << "preRotation mismatch at joint " << i << " (" << original.name << ")";

    EXPECT_TRUE(loaded.translationOffset.isApprox(original.translationOffset, 1e-4f))
        << "translationOffset mismatch at joint " << i << " (" << original.name << ")";
  }
}

TEST_F(UsdIoTest, SaveAndLoadRoundTrip_JointTopology) {
  auto tempFile = temporaryFile("momentum_usd_topology", ".usda");

  saveUsd(tempFile.path(), testCharacter);

  const auto loadedCharacter = loadUsdCharacter(tempFile.path());

  ASSERT_EQ(loadedCharacter.skeleton.joints.size(), testCharacter.skeleton.joints.size());

  for (size_t i = 0; i < testCharacter.skeleton.joints.size(); ++i) {
    EXPECT_EQ(loadedCharacter.skeleton.joints[i].parent, testCharacter.skeleton.joints[i].parent)
        << "Parent index mismatch at joint " << i << " (" << testCharacter.skeleton.joints[i].name
        << ")";
  }
}

TEST_F(UsdIoTest, SaveAndLoadRoundTrip_SkinWeights) {
  auto tempFile = temporaryFile("momentum_usd_skinweights", ".usda");

  saveUsd(tempFile.path(), testCharacter);

  const auto loadedCharacter = loadUsdCharacter(tempFile.path());

  ASSERT_TRUE(testCharacter.skinWeights);
  ASSERT_TRUE(loadedCharacter.skinWeights);

  const auto& origSkin = *testCharacter.skinWeights;
  const auto& loadedSkin = *loadedCharacter.skinWeights;

  ASSERT_EQ(loadedSkin.index.rows(), origSkin.index.rows());

  // Compare that nonzero weights survive roundtrip
  for (Eigen::Index v = 0; v < origSkin.index.rows(); ++v) {
    for (Eigen::Index j = 0; j < origSkin.weight.cols(); ++j) {
      if (origSkin.weight(v, j) > 0.0f) {
        EXPECT_EQ(loadedSkin.index(v, j), origSkin.index(v, j))
            << "Joint index mismatch at vertex " << v << ", influence " << j;
        EXPECT_NEAR(loadedSkin.weight(v, j), origSkin.weight(v, j), 1e-5f)
            << "Weight mismatch at vertex " << v << ", influence " << j;
      }
    }
  }
}

TEST_F(UsdIoTest, SaveAndLoadRoundTrip_NonTrivialPreRotation) {
  // Create a character with non-identity preRotations to verify the bug fix
  auto character = createTestCharacter();

  // Set non-trivial preRotations on joints
  character.skeleton.joints[0].preRotation =
      Eigen::Quaternionf(Eigen::AngleAxisf(0.3f, Eigen::Vector3f::UnitX()));
  character.skeleton.joints[1].preRotation =
      Eigen::Quaternionf(Eigen::AngleAxisf(-0.5f, Eigen::Vector3f::UnitZ()));
  character.skeleton.joints[2].preRotation =
      Eigen::Quaternionf(Eigen::AngleAxisf(0.7f, Eigen::Vector3f(1.0f, 1.0f, 0.0f).normalized()));

  auto tempFile = temporaryFile("momentum_usd_prerot", ".usda");
  saveUsd(tempFile.path(), character);

  const auto loadedCharacter = loadUsdCharacter(tempFile.path());

  ASSERT_EQ(loadedCharacter.skeleton.joints.size(), character.skeleton.joints.size());

  for (size_t i = 0; i < character.skeleton.joints.size(); ++i) {
    const auto& original = character.skeleton.joints[i];
    const auto& loaded = loadedCharacter.skeleton.joints[i];

    EXPECT_TRUE(loaded.preRotation.toRotationMatrix().isApprox(
        original.preRotation.toRotationMatrix(), 1e-4f))
        << "preRotation mismatch at joint " << i << " (" << original.name << ")";

    EXPECT_TRUE(loaded.translationOffset.isApprox(original.translationOffset, 1e-4f))
        << "translationOffset mismatch at joint " << i << " (" << original.name << ")";

    EXPECT_EQ(loaded.parent, original.parent) << "Parent index mismatch at joint " << i;
  }
}

TEST_F(UsdIoTest, LoadFromBuffer) {
  auto usdPath = getTestResourcePath("simple_character.usda");
  if (!usdPath.has_value()) {
    GTEST_SKIP() << "Environment variable 'TEST_RESOURCES_PATH' is not set.";
  }

  if (!filesystem::exists(*usdPath)) {
    GTEST_SKIP() << "Test resource file not found: " << *usdPath;
  }

  // Read file into buffer
  std::ifstream file(*usdPath, std::ios::binary | std::ios::ate);
  ASSERT_TRUE(file.is_open());

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<std::byte> buffer(size);
  file.read(reinterpret_cast<char*>(buffer.data()), size);
  file.close();

  // Load from buffer
  const auto character = loadUsdCharacter(std::span<const std::byte>(buffer));

  // Verify character was loaded correctly
  EXPECT_GT(character.skeleton.joints.size(), 0);
  EXPECT_GT(character.mesh->vertices.size(), 0);
  EXPECT_GT(character.mesh->faces.size(), 0);
}

TEST_F(UsdIoTest, SaveAndLoadRoundTrip_Normals) {
  // The test character mesh has normals computed via updateNormals()
  ASSERT_TRUE(testCharacter.mesh);
  ASSERT_FALSE(testCharacter.mesh->normals.empty());

  auto tempFile = temporaryFile("momentum_usd_normals", ".usda");
  saveUsd(tempFile.path(), testCharacter);

  const auto loadedCharacter = loadUsdCharacter(tempFile.path());

  ASSERT_TRUE(loadedCharacter.mesh);
  ASSERT_EQ(loadedCharacter.mesh->normals.size(), testCharacter.mesh->normals.size());

  for (size_t i = 0; i < testCharacter.mesh->normals.size(); ++i) {
    EXPECT_TRUE(loadedCharacter.mesh->normals[i].isApprox(testCharacter.mesh->normals[i], 1e-4f))
        << "Normal mismatch at vertex " << i;
  }
}

TEST_F(UsdIoTest, SaveAndLoadRoundTrip_VertexColors) {
  // The test character mesh has colors
  ASSERT_TRUE(testCharacter.mesh);
  ASSERT_FALSE(testCharacter.mesh->colors.empty());

  auto tempFile = temporaryFile("momentum_usd_colors", ".usda");
  saveUsd(tempFile.path(), testCharacter);

  const auto loadedCharacter = loadUsdCharacter(tempFile.path());

  ASSERT_TRUE(loadedCharacter.mesh);
  ASSERT_EQ(loadedCharacter.mesh->colors.size(), testCharacter.mesh->colors.size());

  for (size_t i = 0; i < testCharacter.mesh->colors.size(); ++i) {
    // Allow +-1 tolerance due to float->byte->float->byte roundtrip
    for (int c = 0; c < 3; ++c) {
      EXPECT_NEAR(loadedCharacter.mesh->colors[i][c], testCharacter.mesh->colors[i][c], 1)
          << "Color mismatch at vertex " << i << ", channel " << c;
    }
  }
}

TEST_F(UsdIoTest, SaveAndLoadRoundTrip_BlendShapes) {
  auto character = withTestBlendShapes(testCharacter);

  ASSERT_TRUE(character.blendShape);
  const auto numShapes = character.blendShape->shapeSize();
  ASSERT_GT(numShapes, 0);

  auto tempFile = temporaryFile("momentum_usd_blendshapes", ".usda");
  saveUsd(tempFile.path(), character);

  const auto loadedCharacter = loadUsdCharacter(tempFile.path());

  ASSERT_TRUE(loadedCharacter.blendShape);
  EXPECT_EQ(loadedCharacter.blendShape->shapeSize(), numShapes);
  EXPECT_EQ(loadedCharacter.blendShape->modelSize(), character.blendShape->modelSize());

  // Compare shape vectors
  const auto& origVectors = character.blendShape->getShapeVectors();
  const auto& loadedVectors = loadedCharacter.blendShape->getShapeVectors();

  ASSERT_EQ(loadedVectors.rows(), origVectors.rows());
  ASSERT_EQ(loadedVectors.cols(), origVectors.cols());

  EXPECT_TRUE(loadedVectors.isApprox(origVectors, 1e-4f)) << "Shape vectors mismatch";
}

TEST_F(UsdIoTest, SaveAndLoadRoundTrip_MomentumMetadata) {
  auto tempFile = temporaryFile("momentum_usd_metadata", ".usda");

  saveUsd(tempFile.path(), testCharacter);

  const auto loadedCharacter = loadUsdCharacter(tempFile.path());

  // Verify character name
  EXPECT_EQ(loadedCharacter.name, testCharacter.name);

  // Verify parameter transform names
  ASSERT_EQ(
      loadedCharacter.parameterTransform.name.size(), testCharacter.parameterTransform.name.size());

  for (size_t i = 0; i < testCharacter.parameterTransform.name.size(); ++i) {
    EXPECT_EQ(loadedCharacter.parameterTransform.name[i], testCharacter.parameterTransform.name[i])
        << "Parameter name mismatch at index " << i;
  }

  // Verify transform matrix dimensions
  EXPECT_EQ(
      loadedCharacter.parameterTransform.transform.rows(),
      testCharacter.parameterTransform.transform.rows());
  EXPECT_EQ(
      loadedCharacter.parameterTransform.transform.cols(),
      testCharacter.parameterTransform.transform.cols());

  // Verify parameter limits
  ASSERT_EQ(loadedCharacter.parameterLimits.size(), testCharacter.parameterLimits.size());

  for (size_t i = 0; i < testCharacter.parameterLimits.size(); ++i) {
    EXPECT_EQ(loadedCharacter.parameterLimits[i].type, testCharacter.parameterLimits[i].type)
        << "Parameter limit type mismatch at index " << i;
    EXPECT_FLOAT_EQ(
        loadedCharacter.parameterLimits[i].weight, testCharacter.parameterLimits[i].weight)
        << "Parameter limit weight mismatch at index " << i;
  }
}

TEST_F(UsdIoTest, SaveAndLoadRoundTrip_SkeletonStates) {
  const auto& skeleton = testCharacter.skeleton;
  const auto numJoints = skeleton.joints.size();
  const float fps = 30.0f;

  // Create a few skeleton states with varying joint parameters
  std::vector<SkeletonState> states;
  for (int frame = 0; frame < 5; ++frame) {
    JointParameters params = JointParameters::Zero(numJoints * kParametersPerJoint);
    // Vary the root translation and joint1 rotation per frame
    params[0] = static_cast<float>(frame) * 0.1f; // root_tx
    params[1] = static_cast<float>(frame) * 0.05f; // root_ty
    params[kParametersPerJoint + 3] = static_cast<float>(frame) * 0.2f; // joint1_rx
    states.emplace_back(params, skeleton, false);
  }

  auto tempFile = temporaryFile("momentum_usd_skelstates", ".usda");
  saveUsdCharacter(tempFile.path(), testCharacter, fps, states);

  auto [loadedCharacter, loadedStates, frameTimes] =
      loadUsdCharacterWithSkeletonStates(tempFile.path());

  ASSERT_EQ(loadedStates.size(), states.size());
  ASSERT_EQ(frameTimes.size(), states.size());

  // Compare global transforms (more reliable than comparing euler angles)
  for (size_t f = 0; f < states.size(); ++f) {
    ASSERT_EQ(loadedStates[f].jointState.size(), numJoints);

    for (size_t j = 0; j < numJoints; ++j) {
      const auto& origTransform = states[f].jointState[j].transform;
      const auto& loadedTransform = loadedStates[f].jointState[j].transform;

      EXPECT_TRUE(origTransform.translation.isApprox(loadedTransform.translation, 0.01f))
          << "Translation mismatch at frame " << f << ", joint " << j;

      EXPECT_TRUE(origTransform.rotation.isApprox(loadedTransform.rotation, 0.01f))
          << "Rotation mismatch at frame " << f << ", joint " << j;

      EXPECT_NEAR(origTransform.scale, loadedTransform.scale, 0.01f)
          << "Scale mismatch at frame " << f << ", joint " << j;
    }
  }
}

TEST_F(UsdIoTest, SaveAndLoadRoundTrip_CollisionGeometry) {
  ASSERT_TRUE(testCharacter.collision);
  ASSERT_GT(testCharacter.collision->size(), 0);

  auto tempFile = temporaryFile("momentum_usd_collision", ".usda");
  saveUsd(tempFile.path(), testCharacter);

  const auto loadedCharacter = loadUsdCharacter(tempFile.path());

  ASSERT_TRUE(loadedCharacter.collision);
  ASSERT_EQ(loadedCharacter.collision->size(), testCharacter.collision->size());

  for (size_t i = 0; i < testCharacter.collision->size(); ++i) {
    const auto& orig = (*testCharacter.collision)[i];
    const auto& loaded = (*loadedCharacter.collision)[i];

    EXPECT_EQ(loaded.parent, orig.parent) << "Capsule parent mismatch at index " << i;
    EXPECT_NEAR(loaded.length, orig.length, 1e-5f) << "Capsule length mismatch at index " << i;
    EXPECT_TRUE(loaded.radius.isApprox(orig.radius, 1e-5f))
        << "Capsule radius mismatch at index " << i;
    EXPECT_TRUE(loaded.transformation.translation.isApprox(orig.transformation.translation, 1e-4f))
        << "Capsule translation mismatch at index " << i;
    EXPECT_TRUE(loaded.transformation.rotation.toRotationMatrix().isApprox(
        orig.transformation.rotation.toRotationMatrix(), 1e-4f))
        << "Capsule rotation mismatch at index " << i;
  }
}

TEST_F(UsdIoTest, SaveAndLoadRoundTrip_Locators) {
  ASSERT_GT(testCharacter.locators.size(), 0);

  // Set non-default values on the first locator so the roundtrip is non-trivial
  auto& loc0 = testCharacter.locators[0];
  loc0.locked = Eigen::Vector3i(1, 0, 1);
  loc0.weight = 0.75f;
  loc0.limitOrigin = Eigen::Vector3f(0.1f, 0.2f, 0.3f);
  loc0.limitWeight = Eigen::Vector3f(0.5f, 0.6f, 0.7f);
  loc0.attachedToSkin = true;
  loc0.skinOffset = 0.05f;

  auto tempFile = temporaryFile("momentum_usd_locators", ".usda");
  saveUsd(tempFile.path(), testCharacter);

  const auto loadedCharacter = loadUsdCharacter(tempFile.path());

  ASSERT_EQ(loadedCharacter.locators.size(), testCharacter.locators.size());

  for (size_t i = 0; i < testCharacter.locators.size(); ++i) {
    const auto& orig = testCharacter.locators[i];
    const auto& loaded = loadedCharacter.locators[i];

    EXPECT_EQ(loaded.name, orig.name) << "Locator name mismatch at index " << i;
    EXPECT_EQ(loaded.parent, orig.parent) << "Locator parent mismatch at index " << i;
    EXPECT_TRUE(loaded.offset.isApprox(orig.offset, 1e-4f))
        << "Locator offset mismatch at index " << i;
    EXPECT_FLOAT_EQ(loaded.weight, orig.weight) << "Locator weight mismatch at index " << i;
    EXPECT_EQ(loaded.locked, orig.locked) << "Locator locked mismatch at index " << i;
    EXPECT_TRUE(loaded.limitOrigin.isApprox(orig.limitOrigin, 1e-4f))
        << "Locator limitOrigin mismatch at index " << i;
    EXPECT_TRUE(loaded.limitWeight.isApprox(orig.limitWeight, 1e-4f))
        << "Locator limitWeight mismatch at index " << i;
    EXPECT_EQ(loaded.attachedToSkin, orig.attachedToSkin)
        << "Locator attachedToSkin mismatch at index " << i;
    EXPECT_FLOAT_EQ(loaded.skinOffset, orig.skinOffset)
        << "Locator skinOffset mismatch at index " << i;
  }
}

TEST_F(UsdIoTest, LoadSkeletonStates_NonIntegerTimeCodes) {
  const auto& skeleton = testCharacter.skeleton;
  const auto numJoints = skeleton.joints.size();

  // First, save a character to get a valid USD stage with a skeleton prim
  auto tempFile = temporaryFile("momentum_usd_fractional_time", ".usda");
  saveUsd(tempFile.path(), testCharacter);

  // Re-open the stage and manually add a UsdSkelAnimation with non-integer
  // time samples to simulate a USD file authored by an external tool
  auto stage = UsdStage::Open(tempFile.path().string());
  ASSERT_TRUE(stage);

  // Find the skeleton prim
  UsdSkelSkeleton skelPrim;
  for (const auto& prim : stage->Traverse()) {
    if (prim.IsA<UsdSkelSkeleton>()) {
      skelPrim = UsdSkelSkeleton(prim);
      break;
    }
  }
  ASSERT_TRUE(skelPrim);

  // Create animation prim with non-integer time samples
  auto animPath = skelPrim.GetPath().AppendChild(TfToken("Animation"));
  auto animPrim = UsdSkelAnimation::Define(stage, animPath);

  VtArray<TfToken> jointTokens;
  for (const auto& joint : skeleton.joints) {
    jointTokens.push_back(TfToken(joint.name));
  }
  animPrim.GetJointsAttr().Set(jointTokens);

  // Bind animation to skeleton
  UsdSkelBindingAPI bindingAPI = UsdSkelBindingAPI::Apply(skelPrim.GetPrim());
  bindingAPI.GetAnimationSourceRel().SetTargets({animPath});

  // Write time samples at non-integer time codes: 0.0, 0.5, 1.0, 1.5, 2.0
  const std::vector<double> timeCodes = {0.0, 0.5, 1.0, 1.5, 2.0};
  const double tps = 2.0; // 2 time codes per second
  stage->SetStartTimeCode(0.0);
  stage->SetEndTimeCode(2.0);
  stage->SetTimeCodesPerSecond(tps);

  for (const double tc : timeCodes) {
    VtArray<GfVec3f> translations;
    VtArray<GfQuatf> rotations;
    VtArray<GfVec3h> scales;

    for (size_t j = 0; j < numJoints; ++j) {
      const auto& joint = skeleton.joints[j];
      // Vary translation by time code so we can verify frame identity
      translations.push_back(GfVec3f(
          joint.translationOffset.x() + static_cast<float>(tc) * 0.1f,
          joint.translationOffset.y(),
          joint.translationOffset.z()));
      rotations.push_back(GfQuatf(
          joint.preRotation.w(),
          joint.preRotation.x(),
          joint.preRotation.y(),
          joint.preRotation.z()));
      scales.push_back(GfVec3h(GfHalf(1.0f), GfHalf(1.0f), GfHalf(1.0f)));
    }

    animPrim.GetTranslationsAttr().Set(translations, UsdTimeCode(tc));
    animPrim.GetRotationsAttr().Set(rotations, UsdTimeCode(tc));
    animPrim.GetScalesAttr().Set(scales, UsdTimeCode(tc));
  }

  stage->Save();

  // Now load via the public API and verify all 5 frames are read
  auto [loadedCharacter, states, frameTimes] = loadUsdCharacterWithSkeletonStates(tempFile.path());

  // Verify we have expected number of frames
  ASSERT_EQ(states.size(), timeCodes.size())
      << "Expected " << timeCodes.size() << " frames for non-integer time codes";
  ASSERT_EQ(frameTimes.size(), timeCodes.size());

  // Verify frame times are correct (timeCode / timeCodesPerSecond)
  for (size_t f = 0; f < timeCodes.size(); ++f) {
    const auto expectedTime = static_cast<float>(timeCodes[f] / tps);
    EXPECT_NEAR(frameTimes[f], expectedTime, 1e-5f) << "Frame time mismatch at index " << f;
  }

  // Verify per-frame translation varies as expected
  const auto& loadedSkeleton = loadedCharacter.skeleton;
  for (size_t f = 0; f < timeCodes.size(); ++f) {
    const float expectedDelta = static_cast<float>(timeCodes[f]) * 0.1f;
    const auto& rootTranslation = states[f].jointState[0].localTransform.translation;
    EXPECT_NEAR(
        rootTranslation.x() - loadedSkeleton.joints[0].translationOffset.x(), expectedDelta, 0.01f)
        << "Translation delta mismatch at frame " << f;
  }
}

TEST_F(UsdIoTest, SaveAndLoadRoundTrip_ModelParameterMotion) {
  const auto numModelParams = testCharacter.parameterTransform.numAllModelParameters();
  const int numFrames = 5;
  const float fps = 60.0f;

  // Create motion data with non-zero values in every entry to catch index/sign bugs
  MatrixXf poses(numModelParams, numFrames);
  for (int f = 0; f < numFrames; ++f) {
    for (Eigen::Index p = 0; p < numModelParams; ++p) {
      poses(p, f) = static_cast<float>(p + 1) * 0.1f + static_cast<float>(f) * 0.01f;
    }
  }

  std::vector<std::string> paramNames = testCharacter.parameterTransform.name;
  MotionParameters motion = {paramNames, poses};

  // Create identity offsets with distinct non-zero values per entry
  std::vector<std::string> jointNames;
  for (const auto& joint : testCharacter.skeleton.joints) {
    jointNames.push_back(joint.name);
  }
  const auto numOffsets = testCharacter.skeleton.joints.size() * kParametersPerJoint;
  JointParameters offsetValues(numOffsets);
  for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(numOffsets); ++i) {
    offsetValues[i] = static_cast<float>(i + 1) * 0.01f;
  }
  IdentityParameters offsets = {jointNames, offsetValues};

  auto tempFile = temporaryFile("momentum_usd_motion", ".usda");
  saveUsdCharacterWithMotion(tempFile.path(), testCharacter, fps, motion, offsets);

  auto [loadedCharacter, loadedMotion, loadedIdentity, loadedFps] =
      loadUsdCharacterWithMotion(tempFile.path());

  EXPECT_FLOAT_EQ(loadedFps, fps);

  // Verify motion parameter names
  const auto& [loadedParamNames, loadedPoses] = loadedMotion;
  ASSERT_EQ(loadedParamNames.size(), paramNames.size());
  for (size_t i = 0; i < paramNames.size(); ++i) {
    EXPECT_EQ(loadedParamNames[i], paramNames[i]) << "Parameter name mismatch at index " << i;
  }

  // Verify poses matrix
  ASSERT_EQ(loadedPoses.rows(), poses.rows());
  ASSERT_EQ(loadedPoses.cols(), poses.cols());
  EXPECT_TRUE(loadedPoses.isApprox(poses, 1e-5f)) << "Poses matrix mismatch";

  // Verify identity joint names
  const auto& [loadedJointNames, loadedOffsetValues] = loadedIdentity;
  ASSERT_EQ(loadedJointNames.size(), jointNames.size());
  for (size_t i = 0; i < jointNames.size(); ++i) {
    EXPECT_EQ(loadedJointNames[i], jointNames[i]) << "Joint name mismatch at index " << i;
  }

  // Verify identity offset values
  ASSERT_EQ(loadedOffsetValues.size(), offsetValues.size());
  for (Eigen::Index i = 0; i < offsetValues.size(); ++i) {
    EXPECT_NEAR(loadedOffsetValues[i], offsetValues[i], 1e-5f)
        << "Identity offset mismatch at index " << i;
  }
}

TEST_F(UsdIoTest, SaveAndLoadRoundTrip_MarkerSequence) {
  const float fps = 60.0f;
  const int numFrames = 5;
  const int numMarkers = 3;

  // Create marker sequence
  std::vector<std::vector<Marker>> markerSequence(numFrames);
  for (int f = 0; f < numFrames; ++f) {
    markerSequence[f].resize(numMarkers);
    for (int m = 0; m < numMarkers; ++m) {
      markerSequence[f][m].name = "Marker_" + std::to_string(m);
      markerSequence[f][m].pos = Vector3d(
          static_cast<double>(f) * 0.1 + m,
          static_cast<double>(f) * 0.2 + m * 2,
          static_cast<double>(f) * 0.3 + m * 3);
      // Make some markers occluded in some frames
      markerSequence[f][m].occluded = (f + m) % 3 == 0;
    }
  }

  auto tempFile = temporaryFile("momentum_usd_markers", ".usda");
  saveUsdCharacterWithMotion(tempFile.path(), testCharacter, fps, {}, {}, markerSequence);

  auto loadedMarkers = loadUsdMarkerSequence(tempFile.path());

  ASSERT_EQ(loadedMarkers.frames.size(), static_cast<size_t>(numFrames));
  EXPECT_FLOAT_EQ(loadedMarkers.fps, fps);

  for (int f = 0; f < numFrames; ++f) {
    ASSERT_EQ(loadedMarkers.frames[f].size(), static_cast<size_t>(numMarkers));
    for (int m = 0; m < numMarkers; ++m) {
      const auto& orig = markerSequence[f][m];
      const auto& loaded = loadedMarkers.frames[f][m];

      EXPECT_EQ(loaded.name, orig.name)
          << "Marker name mismatch at frame " << f << ", marker " << m;
      EXPECT_NEAR(loaded.pos.x(), orig.pos.x(), 1e-6)
          << "Marker pos.x mismatch at frame " << f << ", marker " << m;
      EXPECT_NEAR(loaded.pos.y(), orig.pos.y(), 1e-6)
          << "Marker pos.y mismatch at frame " << f << ", marker " << m;
      EXPECT_NEAR(loaded.pos.z(), orig.pos.z(), 1e-6)
          << "Marker pos.z mismatch at frame " << f << ", marker " << m;
      EXPECT_EQ(loaded.occluded, orig.occluded)
          << "Marker occlusion mismatch at frame " << f << ", marker " << m;
    }
  }
}
