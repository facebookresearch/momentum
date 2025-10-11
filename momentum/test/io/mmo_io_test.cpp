/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/motion/mmo_io.h"

#include "momentum/character/character.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/io/io_helpers.h"

#include <gtest/gtest.h>

#include <fstream>

using namespace momentum;

class MmoIOTest : public ::testing::Test {
 protected:
  void SetUp() override {
    character_ = createTestCharacter();

    // Create test poses matrix (3 frames, matching character parameters)
    const size_t numParams = character_.parameterTransform.numAllModelParameters();
    poses_ = MatrixXf::Random(numParams, 3);

    // Create test scale vector (7 parameters per joint)
    const size_t numJoints = character_.skeleton.joints.size();
    scale_ = VectorXf::Random(numJoints * kParametersPerJoint);

    // Create parameter names
    parameterNames_.reserve(numParams);
    for (const auto& name : character_.parameterTransform.name) {
      parameterNames_.push_back(name);
    }

    // Create joint names
    jointNames_.reserve(numJoints);
    for (const auto& joint : character_.skeleton.joints) {
      jointNames_.push_back(joint.name);
    }
  }

  Character character_;
  MatrixXf poses_;
  VectorXf scale_;
  std::vector<std::string> parameterNames_;
  std::vector<std::string> jointNames_;
};

TEST_F(MmoIOTest, BasicSaveLoad) {
  auto tempFile = temporaryFile("mmo_test", ".mmo");

  saveMmo(tempFile.path().string(), poses_, scale_, character_);

  auto [loadedPoses, loadedScale, loadedParamNames, loadedJointNames] =
      loadMmo(tempFile.path().string());

  EXPECT_TRUE(loadedPoses.isApprox(poses_));
  EXPECT_TRUE(loadedScale.isApprox(scale_));
  EXPECT_EQ(loadedParamNames, parameterNames_);
  EXPECT_EQ(loadedJointNames, jointNames_);

  // Test character mapping
  auto [mappedPoses, mappedScale] = loadMmo(tempFile.path().string(), character_);
  EXPECT_TRUE(mappedPoses.isApprox(poses_));
  EXPECT_TRUE(mappedScale.isApprox(scale_));

  // Test span overload
  std::vector<VectorXf> poseVectors;
  poseVectors.reserve(poses_.cols());
  for (int i = 0; i < poses_.cols(); ++i) {
    poseVectors.emplace_back(poses_.col(i));
  }
  saveMmo(tempFile.path().string(), poseVectors, scale_, character_);
  auto [spanPoses, spanScale, _, __] = loadMmo(tempFile.path().string());
  EXPECT_TRUE(spanPoses.isApprox(poses_));
}

TEST_F(MmoIOTest, AdditionalParameters) {
  auto tempFile = temporaryFile("mmo_test", ".mmo");

  MatrixXf additionalParams = MatrixXf::Random(2, 3);
  std::vector<std::string> additionalNames = {"param1", "param2"};

  saveMmo(tempFile.path().string(), poses_, scale_, character_, additionalParams, additionalNames);

  auto [loadedPoses, loadedScale, loadedParamNames, loadedJointNames] =
      loadMmo(tempFile.path().string());

  EXPECT_EQ(loadedPoses.rows(), poses_.rows() + additionalParams.rows());
  EXPECT_EQ(loadedParamNames[loadedParamNames.size() - 2], "__param1__");
  EXPECT_EQ(loadedParamNames[loadedParamNames.size() - 1], "__param2__");
  EXPECT_TRUE(loadedPoses.topRows(poses_.rows()).isApprox(poses_));
  EXPECT_TRUE(loadedPoses.bottomRows(additionalParams.rows()).isApprox(additionalParams));
}

TEST_F(MmoIOTest, DirectParameterNames) {
  auto tempFile = temporaryFile("mmo_test", ".mmo");

  saveMmo(tempFile.path().string(), poses_, scale_, parameterNames_, jointNames_);

  auto [loadedPoses, loadedScale, loadedParamNames, loadedJointNames] =
      loadMmo(tempFile.path().string());

  EXPECT_TRUE(loadedPoses.isApprox(poses_));
  EXPECT_TRUE(loadedScale.isApprox(scale_));
  EXPECT_EQ(loadedParamNames, parameterNames_);
  EXPECT_EQ(loadedJointNames, jointNames_);
}

TEST_F(MmoIOTest, AuxiliaryDataAndMapping) {
  // Test auxiliary data extraction
  std::vector<std::string> paramNames = {"joint1_tx", "__aux1__", "joint2_ty", "__aux2__"};
  MatrixXf testPoses = MatrixXf::Random(4, 3);

  auto [auxData, auxNames] = getAuxiliaryDataFromMotion(testPoses, paramNames);

  EXPECT_EQ(auxData.rows(), 2);
  EXPECT_EQ(auxNames.size(), 2);
  EXPECT_EQ(auxNames[0], "aux1");
  EXPECT_EQ(auxNames[1], "aux2");
  EXPECT_TRUE(auxData.row(0).isApprox(testPoses.row(1)));
  EXPECT_TRUE(auxData.row(1).isApprox(testPoses.row(3)));

  // Test motion mapping
  std::vector<std::string> motionParamNames = {
      character_.parameterTransform.name[0], "unknown_param"};
  std::vector<std::string> motionJointNames = {character_.skeleton.joints[0].name, "unknown_joint"};

  MatrixXf motionPoses = MatrixXf::Random(2, 2);
  VectorXf motionOffsets = VectorXf::Random(2 * kParametersPerJoint);

  auto [mappedPoses, mappedOffsets] = mapMotionToCharacter(
      motionPoses, motionOffsets, motionParamNames, motionJointNames, character_);

  EXPECT_EQ(mappedPoses.rows(), character_.parameterTransform.numAllModelParameters());
  EXPECT_TRUE(mappedPoses.row(0).isApprox(motionPoses.row(0)));
  EXPECT_TRUE(mappedPoses.row(1).isZero()); // Unknown parameter should be zero
}

TEST_F(MmoIOTest, ErrorConditions) {
  auto tempFile = temporaryFile("mmo_test", ".mmo");
  auto expectNoFile = [&]() {
    std::ifstream file(tempFile.path().string());
    EXPECT_TRUE(!file.good() || (file.good() && file.peek() == std::ifstream::traits_type::eof()));
  };

  // Empty poses
  std::vector<VectorXf> emptyPoses;
  saveMmo(tempFile.path().string(), emptyPoses, scale_, character_);
  expectNoFile();

  // Wrong dimension poses
  MatrixXf wrongPoses = MatrixXf::Random(poses_.rows() + 1, poses_.cols());
  saveMmo(tempFile.path().string(), wrongPoses, scale_, character_);
  expectNoFile();

  // Wrong dimension scale
  VectorXf wrongScale = VectorXf::Random(scale_.size() + 1);
  saveMmo(tempFile.path().string(), poses_, wrongScale, character_);
  expectNoFile();

  // Mismatched additional parameters
  MatrixXf additionalParams = MatrixXf::Random(2, 3);
  std::vector<std::string> additionalNames = {"param1"};
  saveMmo(tempFile.path().string(), poses_, scale_, character_, additionalParams, additionalNames);
  expectNoFile();

  // Mismatched additional parameter columns
  additionalParams = MatrixXf::Random(2, poses_.cols() + 1);
  additionalNames = {"param1", "param2"};
  saveMmo(tempFile.path().string(), poses_, scale_, character_, additionalParams, additionalNames);
  expectNoFile();

  // Inconsistent pose dimensions
  std::vector<VectorXf> inconsistentPoses;
  inconsistentPoses.emplace_back(VectorXf::Random(poses_.rows()));
  inconsistentPoses.emplace_back(VectorXf::Random(poses_.rows() + 1));
  saveMmo(tempFile.path().string(), inconsistentPoses, scale_, character_);
  expectNoFile();

  // Mismatched parameter names
  std::vector<std::string> wrongParamNames = {"param1", "param2"};
  MatrixXf testPoses = MatrixXf::Random(3, 2);
  saveMmo(tempFile.path().string(), testPoses, scale_, wrongParamNames, jointNames_);
  expectNoFile();

  // Mismatched joint names
  std::vector<std::string> wrongJointNames = {"joint1", "joint2"};
  VectorXf testScale = VectorXf::Random(3 * kParametersPerJoint);
  saveMmo(tempFile.path().string(), poses_, testScale, parameterNames_, wrongJointNames);
  expectNoFile();

  // Non-existent file
  auto [poses, scale, paramNames, jointNames] = loadMmo("nonexistent.mmo");
  EXPECT_EQ(poses.rows(), 0);
  EXPECT_TRUE(paramNames.empty());
}
