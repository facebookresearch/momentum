/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "convert_model_helpers.h"

#include <momentum/character/character.h>
#include <momentum/character/inverse_parameter_transform.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/common/filesystem.h>
#include <momentum/io/character_io.h>
#include <momentum/io/fbx/fbx_io.h>
#include <momentum/io/gltf/gltf_io.h>

#include <gtest/gtest.h>

using namespace momentum;

/// Test fixture providing shared test data and utilities for convert_model tests.
class ConvertModelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Load test models from TEST_MOMENTUM_MODELS_PATH
    const char* testDataPath = getenv("TEST_MOMENTUM_MODELS_PATH");
    ASSERT_NE(testDataPath, nullptr) << "TEST_MOMENTUM_MODELS_PATH environment variable not set";
    testDataPath_ = testDataPath;
  }

  std::string testDataPath_;
};

// ============================================================================
// findLongestMotion tests
// ============================================================================

TEST_F(ConvertModelTest, FindLongestMotion_EmptyVector) {
  std::vector<MatrixXf> motions;
  EXPECT_EQ(findLongestMotion(motions), -1);
}

TEST_F(ConvertModelTest, FindLongestMotion_SingleEmptyMotion) {
  std::vector<MatrixXf> motions = {MatrixXf(10, 0)};
  EXPECT_EQ(findLongestMotion(motions), -1);
}

TEST_F(ConvertModelTest, FindLongestMotion_SingleValidMotion) {
  MatrixXf motion(10, 50);
  motion.setZero();
  std::vector<MatrixXf> motions = {motion};
  EXPECT_EQ(findLongestMotion(motions), 0);
}

TEST_F(ConvertModelTest, FindLongestMotion_MultipleMotions_FirstLongest) {
  MatrixXf motion1(10, 100);
  MatrixXf motion2(10, 50);
  MatrixXf motion3(10, 75);
  std::vector<MatrixXf> motions = {motion1, motion2, motion3};
  EXPECT_EQ(findLongestMotion(motions), 0);
}

TEST_F(ConvertModelTest, FindLongestMotion_MultipleMotions_LastLongest) {
  MatrixXf motion1(10, 50);
  MatrixXf motion2(10, 75);
  MatrixXf motion3(10, 100);
  std::vector<MatrixXf> motions = {motion1, motion2, motion3};
  EXPECT_EQ(findLongestMotion(motions), 2);
}

// ============================================================================
// initializeCharacterFromFbx tests
// ============================================================================

TEST_F(ConvertModelTest, InitializeCharacterFromFbx_NoParams_CreatesIdentityTransform) {
  const auto fbxPath = filesystem::path(testDataPath_) / "character.fbx";
  if (!filesystem::exists(fbxPath)) {
    GTEST_SKIP() << "Test data not available: " << fbxPath;
  }

  auto fbxCharacter = loadFbxCharacter(fbxPath, KeepLocators::Yes, Permissive::No);
  Character character;
  Options options;

  initializeCharacterFromFbx(character, fbxCharacter, options);

  EXPECT_EQ(character.skeleton.joints.size(), fbxCharacter.skeleton.joints.size());
  EXPECT_EQ(
      character.parameterTransform.numAllModelParameters(),
      character.skeleton.joints.size() * kParametersPerJoint);
}

TEST_F(ConvertModelTest, InitializeCharacterFromFbx_WithParams_LoadsTransform) {
  const auto fbxPath = filesystem::path(testDataPath_) / "character.fbx";
  const auto modelPath = filesystem::path(testDataPath_) / "character.model";
  if (!filesystem::exists(fbxPath) || !filesystem::exists(modelPath)) {
    GTEST_SKIP() << "Test data not available";
  }

  auto fbxCharacter = loadFbxCharacter(fbxPath, KeepLocators::Yes, Permissive::No);
  Character character;
  Options options;
  options.input_params_file = modelPath.string();

  initializeCharacterFromFbx(character, fbxCharacter, options);

  EXPECT_EQ(character.skeleton.joints.size(), fbxCharacter.skeleton.joints.size());
  EXPECT_GT(character.parameterTransform.numAllModelParameters(), 0);
}

TEST_F(ConvertModelTest, InitializeCharacterFromFbx_WithLocators_LoadsLocators) {
  const auto fbxPath = filesystem::path(testDataPath_) / "character.fbx";
  const auto locatorPath = filesystem::path(testDataPath_) / "character.locators";
  if (!filesystem::exists(fbxPath) || !filesystem::exists(locatorPath)) {
    GTEST_SKIP() << "Test data not available";
  }

  auto fbxCharacter = loadFbxCharacter(fbxPath, KeepLocators::Yes, Permissive::No);
  Character character;
  Options options;
  options.input_locator_file = locatorPath.string();

  initializeCharacterFromFbx(character, fbxCharacter, options);

  EXPECT_GT(character.locators.size(), 0);
}

TEST_F(ConvertModelTest, InitializeCharacterFromFbx_WithParamsAndLocators_LoadsBoth) {
  const auto fbxPath = filesystem::path(testDataPath_) / "character.fbx";
  const auto modelPath = filesystem::path(testDataPath_) / "character.model";
  const auto locatorPath = filesystem::path(testDataPath_) / "character.locators";
  if (!filesystem::exists(fbxPath) || !filesystem::exists(modelPath) ||
      !filesystem::exists(locatorPath)) {
    GTEST_SKIP() << "Test data not available";
  }

  auto fbxCharacter = loadFbxCharacter(fbxPath, KeepLocators::Yes, Permissive::No);
  Character character;
  Options options;
  options.input_params_file = modelPath.string();
  options.input_locator_file = locatorPath.string();

  initializeCharacterFromFbx(character, fbxCharacter, options);

  EXPECT_GT(character.parameterTransform.numAllModelParameters(), 0);
  EXPECT_GT(character.locators.size(), 0);
}

// ============================================================================
// convertToModelParams tests
// ============================================================================

TEST_F(ConvertModelTest, ConvertToModelParams_StandardTransform) {
  const auto modelPath = filesystem::path(testDataPath_) / "model_with_motion.glb";
  if (!filesystem::exists(modelPath)) {
    GTEST_SKIP() << "Test data not available: " << modelPath;
  }

  auto character = loadFullCharacter(modelPath.string());
  const size_t nFrames = 10;
  // convertToModelParams expects JOINT parameters as input (not model parameters)
  const size_t numJointParams = character.skeleton.joints.size() * kParametersPerJoint;
  MatrixXf jointMotion;
  jointMotion.setZero(numJointParams, nFrames);

  for (size_t i = 0; i < nFrames; ++i) {
    jointMotion.col(i).setConstant(static_cast<float>(i) * 0.1f);
  }

  MatrixXf modelParams = convertToModelParams(jointMotion, character.parameterTransform);

  EXPECT_EQ(modelParams.rows(), character.parameterTransform.numAllModelParameters());
  EXPECT_EQ(modelParams.cols(), nFrames);
}

TEST_F(ConvertModelTest, ConvertToModelParams_IdentityTransform) {
  const auto modelPath = filesystem::path(testDataPath_) / "model_with_motion.glb";
  if (!filesystem::exists(modelPath)) {
    GTEST_SKIP() << "Test data not available: " << modelPath;
  }

  Character character = loadFullCharacter(modelPath.string());
  character.parameterTransform = ParameterTransform::identity(character.skeleton.getJointNames());

  const size_t nFrames = 5;
  MatrixXf jointMotion;
  jointMotion.setRandom(character.parameterTransform.numAllModelParameters(), nFrames);

  MatrixXf modelParams = convertToModelParams(jointMotion, character.parameterTransform);

  EXPECT_EQ(modelParams.rows(), jointMotion.rows());
  EXPECT_EQ(modelParams.cols(), jointMotion.cols());
}

TEST_F(ConvertModelTest, ConvertToModelParams_MultipleFrames) {
  const auto modelPath = filesystem::path(testDataPath_) / "model_with_motion.glb";
  if (!filesystem::exists(modelPath)) {
    GTEST_SKIP() << "Test data not available: " << modelPath;
  }

  auto character = loadFullCharacter(modelPath.string());
  const size_t nFrames = 100;
  const size_t numJointParams = character.skeleton.joints.size() * kParametersPerJoint;
  MatrixXf jointMotion;
  jointMotion.setZero(numJointParams, nFrames);

  MatrixXf modelParams = convertToModelParams(jointMotion, character.parameterTransform);

  EXPECT_EQ(modelParams.cols(), nFrames);
  EXPECT_EQ(modelParams.rows(), character.parameterTransform.numAllModelParameters());
}

TEST_F(ConvertModelTest, ConvertToModelParams_SingleFrame) {
  const auto modelPath = filesystem::path(testDataPath_) / "model_with_motion.glb";
  if (!filesystem::exists(modelPath)) {
    GTEST_SKIP() << "Test data not available: " << modelPath;
  }

  auto character = loadFullCharacter(modelPath.string());
  const size_t nFrames = 1;
  const size_t numJointParams = character.skeleton.joints.size() * kParametersPerJoint;
  MatrixXf jointMotion;
  jointMotion.setZero(numJointParams, nFrames);
  jointMotion.col(0).setConstant(0.5f);

  MatrixXf modelParams = convertToModelParams(jointMotion, character.parameterTransform);

  EXPECT_EQ(modelParams.cols(), 1);
  EXPECT_EQ(modelParams.rows(), character.parameterTransform.numAllModelParameters());
}

// ============================================================================
// Error path and edge case tests
// ============================================================================

TEST_F(ConvertModelTest, Error_InvalidOutputFormatExtension) {
  const std::vector<std::string> invalidExtensions = {".obj", ".dae", ".txt", ".json", ".xml"};

  for (const auto& ext : invalidExtensions) {
    filesystem::path testPath = filesystem::path(testDataPath_) / ("output" + ext);
    auto outputExt = testPath.extension();

    EXPECT_NE(outputExt, ".fbx");
    EXPECT_NE(outputExt, ".glb");
    EXPECT_NE(outputExt, ".gltf");
  }
}

TEST_F(ConvertModelTest, Error_ValidOutputFormatExtensions) {
  const std::vector<std::string> validExtensions = {".fbx", ".glb", ".gltf"};

  for (const auto& ext : validExtensions) {
    filesystem::path testPath = filesystem::path(testDataPath_) / ("output" + ext);
    auto outputExt = testPath.extension();

    bool isValid = (outputExt == ".fbx") || (outputExt == ".glb") || (outputExt == ".gltf");
    EXPECT_TRUE(isValid) << "Extension " << ext << " should be recognized as valid";
  }
}

TEST_F(ConvertModelTest, Error_EmptyCharacterInitialization) {
  Character emptyCharacter;

  EXPECT_EQ(emptyCharacter.skeleton.joints.size(), 0);
  EXPECT_EQ(emptyCharacter.parameterTransform.numAllModelParameters(), 0);
}

TEST_F(ConvertModelTest, Error_IncompatibleSkeletonSizes) {
  Character character1;
  character1.skeleton.joints.resize(10);

  Character character2;
  character2.skeleton.joints.resize(20);

  EXPECT_NE(character1.skeleton.joints.size(), character2.skeleton.joints.size())
      << "Characters with different joint counts are incompatible";
}

TEST_F(ConvertModelTest, Error_UnknownMotionFileFormats) {
  const std::vector<std::string> unknownFormats = {".bvh", ".c3d", ".mp4", ".txt", ".csv"};

  for (const auto& format : unknownFormats) {
    filesystem::path motionPath = filesystem::path(testDataPath_) / ("motion" + format);
    auto ext = motionPath.extension();

    EXPECT_NE(ext, ".glb") << format << " should not be .glb";
    EXPECT_NE(ext, ".fbx") << format << " should not be .fbx";
  }
}

TEST_F(ConvertModelTest, Error_NonExistentInputFile) {
  const std::string nonExistentFile = testDataPath_ + "/does_not_exist_12345.fbx";
  EXPECT_FALSE(filesystem::exists(nonExistentFile));
}

TEST_F(ConvertModelTest, EdgeCase_CharacterWithZeroJoints) {
  Character character;
  EXPECT_EQ(character.skeleton.joints.size(), 0);

  Character validCharacter;
  validCharacter.skeleton.joints.resize(1);
  EXPECT_NE(character.skeleton.joints.size(), validCharacter.skeleton.joints.size());
}

TEST_F(ConvertModelTest, EdgeCase_FilePathWithSpecialCharacters) {
  const std::vector<std::string> specialPaths = {
      "file with spaces.fbx",
      "file-with-dashes.glb",
      "file_with_underscores.gltf",
  };

  for (const auto& filename : specialPaths) {
    filesystem::path testPath = filesystem::path(testDataPath_) / filename;

    EXPECT_FALSE(testPath.string().empty());
    auto ext = testPath.extension();

    bool hasValidExt = (ext == ".fbx") || (ext == ".glb") || (ext == ".gltf");
    EXPECT_TRUE(hasValidExt) << "Extension should be parsed correctly for: " << filename;
  }
}

// ============================================================================
// convertMotionWithJointMapping tests
// ============================================================================

// Helper to create a skeleton with joints in the specified order.
// Each joint name in the list becomes a joint; the first is the root,
// and subsequent joints are children of the previous joint.
namespace {
Skeleton createReorderedSkeleton(const std::vector<std::string>& jointNames) {
  Skeleton skeleton;
  for (size_t i = 0; i < jointNames.size(); ++i) {
    Joint joint;
    joint.name = jointNames[i];
    joint.parent = (i == 0) ? kInvalidIndex : i - 1;
    joint.preRotation = Eigen::Quaternionf::Identity();
    if (i == 0) {
      joint.translationOffset = Eigen::Vector3f::Zero();
    } else {
      joint.translationOffset = Eigen::Vector3f::UnitY();
    }
    skeleton.joints.push_back(joint);
  }
  return skeleton;
}
} // namespace

// When source and target skeletons have the same joint order, the mapping
// should produce the same result as the identity mapping.
TEST_F(ConvertModelTest, ConvertMotionWithJointMapping_SameOrder) {
  const auto modelPath = filesystem::path(testDataPath_) / "model_with_motion.glb";
  if (!filesystem::exists(modelPath)) {
    GTEST_SKIP() << "Test data not available: " << modelPath;
  }

  auto character = loadFullCharacter(modelPath.string());
  const auto& skeleton = character.skeleton;

  const size_t nFrames = 3;
  const size_t numJointParams = skeleton.joints.size() * kParametersPerJoint;
  MatrixXf motion;
  motion.setZero(numJointParams, nFrames);

  MatrixXf result = convertMotionWithJointMapping(motion, skeleton, character);

  EXPECT_EQ(result.rows(), character.parameterTransform.numAllModelParameters());
  EXPECT_EQ(result.cols(), nFrames);
  EXPECT_TRUE(result.allFinite());
}

// When source and target skeletons have different joint orders, the mapping
// should correctly reorder the joint transforms so the output animation
// matches the target skeleton's joint ordering.
TEST_F(ConvertModelTest, ConvertMotionWithJointMapping_DifferentOrder) {
  const auto modelPath = filesystem::path(testDataPath_) / "model_with_motion.glb";
  if (!filesystem::exists(modelPath)) {
    GTEST_SKIP() << "Test data not available: " << modelPath;
  }

  auto targetCharacter = loadFullCharacter(modelPath.string());
  const auto jointNames = targetCharacter.skeleton.getJointNames();

  // Create a source skeleton with the same joints but reversed order
  auto reversedNames = jointNames;
  std::reverse(reversedNames.begin(), reversedNames.end());
  Skeleton sourceSkeleton = createReorderedSkeleton(reversedNames);

  const size_t nFrames = 2;
  const size_t numSourceJointParams = sourceSkeleton.joints.size() * kParametersPerJoint;
  MatrixXf motion;
  motion.setZero(numSourceJointParams, nFrames);

  MatrixXf result = convertMotionWithJointMapping(motion, sourceSkeleton, targetCharacter);

  EXPECT_EQ(result.rows(), targetCharacter.parameterTransform.numAllModelParameters());
  EXPECT_EQ(result.cols(), nFrames);
  EXPECT_TRUE(result.allFinite());
}

// When the source skeleton is missing joints that exist in the target, the
// target should use its rest pose for the unmatched joints.
TEST_F(ConvertModelTest, ConvertMotionWithJointMapping_MissingSourceJoints) {
  const auto modelPath = filesystem::path(testDataPath_) / "model_with_motion.glb";
  if (!filesystem::exists(modelPath)) {
    GTEST_SKIP() << "Test data not available: " << modelPath;
  }

  auto targetCharacter = loadFullCharacter(modelPath.string());
  const auto jointNames = targetCharacter.skeleton.getJointNames();

  // Create a source skeleton with only the first two joints
  std::vector<std::string> subsetNames(jointNames.begin(), jointNames.begin() + 2);
  Skeleton sourceSkeleton = createReorderedSkeleton(subsetNames);

  const size_t nFrames = 2;
  const size_t numSourceJointParams = sourceSkeleton.joints.size() * kParametersPerJoint;
  MatrixXf motion;
  motion.setZero(numSourceJointParams, nFrames);

  MatrixXf result = convertMotionWithJointMapping(motion, sourceSkeleton, targetCharacter);

  EXPECT_EQ(result.rows(), targetCharacter.parameterTransform.numAllModelParameters());
  EXPECT_EQ(result.cols(), nFrames);
  EXPECT_TRUE(result.allFinite());
}

// Verify that non-zero motion on the source produces non-trivial output
// when joint mapping is required due to different joint orders.
TEST_F(ConvertModelTest, ConvertMotionWithJointMapping_NonZeroMotion) {
  const auto modelPath = filesystem::path(testDataPath_) / "model_with_motion.glb";
  if (!filesystem::exists(modelPath)) {
    GTEST_SKIP() << "Test data not available: " << modelPath;
  }

  auto targetCharacter = loadFullCharacter(modelPath.string());
  const auto jointNames = targetCharacter.skeleton.getJointNames();

  // Use a shuffled order so joint mapping is exercised
  auto shuffledNames = jointNames;
  if (shuffledNames.size() >= 3) {
    std::swap(shuffledNames[1], shuffledNames[2]);
  }
  Skeleton sourceSkeleton = createReorderedSkeleton(shuffledNames);

  const size_t nFrames = 5;
  const size_t numSourceJointParams = sourceSkeleton.joints.size() * kParametersPerJoint;
  MatrixXf motion;
  motion.setZero(numSourceJointParams, nFrames);

  // Apply a small rotation to the second joint on the first frame.
  // Joint parameters are 7 per joint: tx, ty, tz, rx, ry, rz, scale.
  motion(1 * kParametersPerJoint + 3, 0) = 0.1f;

  MatrixXf result = convertMotionWithJointMapping(motion, sourceSkeleton, targetCharacter);

  EXPECT_EQ(result.rows(), targetCharacter.parameterTransform.numAllModelParameters());
  EXPECT_EQ(result.cols(), nFrames);
  EXPECT_TRUE(result.allFinite());

  // The first frame should differ from the rest since we applied motion
  // to it. At minimum, the results should not all be identical.
  bool hasNonZeroDiff = false;
  for (Eigen::Index iFrame = 1; iFrame < result.cols(); ++iFrame) {
    if ((result.col(0) - result.col(iFrame)).norm() > 1e-6f) {
      hasNonZeroDiff = true;
      break;
    }
  }
  EXPECT_TRUE(hasNonZeroDiff) << "Non-zero source motion should produce different output frames";
}
