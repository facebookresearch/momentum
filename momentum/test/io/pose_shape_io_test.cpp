/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/shape/pose_shape_io.h"

#include "momentum/character/character.h"
#include "momentum/character/pose_shape.h"
#include "momentum/math/mesh.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/io/io_helpers.h"

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

using namespace momentum;

class PoseShapeIOTest : public ::testing::Test {
 protected:
  void SetUp() override {
    character_ = createTestCharacter();
  }

  std::string createPoseShapeData(size_t numJoints = 2) {
    std::ostringstream oss(std::ios::binary);
    uint64_t numRows = character_.mesh->vertices.size() * 3;
    uint64_t joints = numJoints;

    oss.write(reinterpret_cast<const char*>(&numRows), sizeof(numRows));
    oss.write(reinterpret_cast<const char*>(&joints), sizeof(joints));

    // Base joint name
    std::string baseName = character_.skeleton.joints[0].name;
    uint64_t nameLen = baseName.size();
    oss.write(reinterpret_cast<const char*>(&nameLen), sizeof(nameLen));
    oss.write(baseName.data(), nameLen);

    // Joint names
    for (size_t i = 0; i < numJoints; ++i) {
      std::string name = character_.skeleton.joints[i].name;
      nameLen = name.size();
      oss.write(reinterpret_cast<const char*>(&nameLen), sizeof(nameLen));
      oss.write(name.data(), nameLen);
    }

    // Base shape and vectors
    std::vector<float> baseShape(numRows, 0.1f);
    oss.write(reinterpret_cast<const char*>(baseShape.data()), sizeof(float) * numRows);

    std::vector<float> shapeVectors(numRows * numJoints * 4, 0.05f);
    oss.write(
        reinterpret_cast<const char*>(shapeVectors.data()),
        sizeof(float) * numRows * numJoints * 4);

    return oss.str();
  }

  Character character_;
};

TEST_F(PoseShapeIOTest, LoadValid) {
  auto tempFile = temporaryFile("test", ".bin");
  std::ofstream ofs(tempFile.path(), std::ios::binary);
  std::string data = createPoseShapeData();
  ofs.write(data.data(), data.size());
  ofs.close();

  PoseShape result = loadPoseShape(tempFile.path().string(), character_);

  EXPECT_EQ(result.baseJoint, 0);
  EXPECT_EQ(result.jointMap.size(), 2);
  EXPECT_EQ(result.shapeVectors.cols(), 8); // 2 joints * 4 quaternion components
}

TEST_F(PoseShapeIOTest, LoadNonExistent) {
  PoseShape result = loadPoseShape("nonexistent.bin", character_);
  EXPECT_TRUE(result.jointMap.empty());
  EXPECT_EQ(result.baseShape.size(), 0);
}

TEST_F(PoseShapeIOTest, LoadZeroJoints) {
  auto tempFile = temporaryFile("test", ".bin");
  std::ofstream ofs(tempFile.path(), std::ios::binary);
  std::string data = createPoseShapeData(0);
  ofs.write(data.data(), data.size());
  ofs.close();

  PoseShape result = loadPoseShape(tempFile.path().string(), character_);
  EXPECT_TRUE(result.jointMap.empty());
  EXPECT_EQ(result.shapeVectors.cols(), 0);
}
