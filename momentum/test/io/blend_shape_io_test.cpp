/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/shape/blend_shape_io.h"

#include "momentum/character/blend_shape.h"
#include "momentum/character/blend_shape_base.h"
#include "momentum/test/io/io_helpers.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

using namespace momentum;

class BlendShapeIOTest : public ::testing::Test {
 protected:
  void SetUp() override {
    baseShape_ = {
        Vector3f(1.0f, 2.0f, 3.0f), Vector3f(4.0f, 5.0f, 6.0f), Vector3f(7.0f, 8.0f, 9.0f)};
    shapeVectors_.resize(9, 2);
    shapeVectors_ << 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f,
        1.4f, 1.5f, 1.6f, 1.7f, 1.8f;
  }

  std::string createBlendShapeBaseData() {
    std::ostringstream oss(std::ios::binary);
    uint64_t rows = 9, cols = 2;
    oss.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    oss.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    oss.write(reinterpret_cast<const char*>(shapeVectors_.data()), sizeof(float) * 18);
    return oss.str();
  }

  std::string createBlendShapeData() {
    std::ostringstream oss(std::ios::binary);
    uint64_t rows = 9, cols = 2;
    oss.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    oss.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    oss.write(reinterpret_cast<const char*>(baseShape_.data()), sizeof(float) * 9);
    oss.write(reinterpret_cast<const char*>(shapeVectors_.data()), sizeof(float) * 18);
    return oss.str();
  }

  std::vector<Vector3f> baseShape_;
  MatrixXf shapeVectors_;
};

TEST_F(BlendShapeIOTest, LoadBlendShapeBase) {
  std::string data = createBlendShapeBaseData();
  std::istringstream iss(data, std::ios::binary);
  BlendShapeBase result = loadBlendShapeBase(iss);
  EXPECT_EQ(result.modelSize(), 3);
  EXPECT_EQ(result.shapeSize(), 2);
  EXPECT_TRUE(result.getShapeVectors().isApprox(shapeVectors_));
}

TEST_F(BlendShapeIOTest, LoadBlendShapeBaseWithLimits) {
  std::string data = createBlendShapeBaseData();
  std::istringstream iss(data, std::ios::binary);
  BlendShapeBase result = loadBlendShapeBase(iss, 1, 2);
  EXPECT_EQ(result.modelSize(), 2);
  EXPECT_EQ(result.shapeSize(), 1);
}

TEST_F(BlendShapeIOTest, LoadBlendShapeBaseFromFile) {
  auto tempFile = temporaryFile("test", ".bin");
  std::ofstream ofs(tempFile.path(), std::ios::binary);
  std::string data = createBlendShapeBaseData();
  ofs.write(data.data(), data.size());
  ofs.close();

  BlendShapeBase result = loadBlendShapeBase(tempFile.path());
  EXPECT_EQ(result.modelSize(), 3);
  EXPECT_EQ(result.shapeSize(), 2);
}

TEST_F(BlendShapeIOTest, LoadBlendShapeBaseFileNotFound) {
  EXPECT_THROW(loadBlendShapeBase("nonexistent.bin"), std::runtime_error);
}

TEST_F(BlendShapeIOTest, LoadBlendShape) {
  std::string data = createBlendShapeData();
  std::istringstream iss(data, std::ios::binary);
  BlendShape result = loadBlendShape(iss);
  EXPECT_EQ(result.modelSize(), 3);
  EXPECT_EQ(result.shapeSize(), 2);
  EXPECT_EQ(result.getBaseShape().size(), 3);
  EXPECT_TRUE(result.getShapeVectors().isApprox(shapeVectors_));
}

TEST_F(BlendShapeIOTest, LoadBlendShapeWithLimits) {
  std::string data = createBlendShapeData();
  std::istringstream iss(data, std::ios::binary);
  BlendShape result = loadBlendShape(iss, 1, 2);
  EXPECT_EQ(result.modelSize(), 2);
  EXPECT_EQ(result.shapeSize(), 1);
  EXPECT_EQ(result.getBaseShape().size(), 2);
}

TEST_F(BlendShapeIOTest, LoadBlendShapeFromFile) {
  auto tempFile = temporaryFile("test", ".bin");
  std::ofstream ofs(tempFile.path(), std::ios::binary);
  std::string data = createBlendShapeData();
  ofs.write(data.data(), data.size());
  ofs.close();

  BlendShape result = loadBlendShape(tempFile.path());
  EXPECT_EQ(result.modelSize(), 3);
  EXPECT_EQ(result.shapeSize(), 2);
}

TEST_F(BlendShapeIOTest, LoadBlendShapeFileNotFound) {
  EXPECT_THROW(loadBlendShape("nonexistent.bin"), std::runtime_error);
}

TEST_F(BlendShapeIOTest, BlendShapeBaseSaveAndLoadRoundTrip) {
  BlendShapeBase original(shapeVectors_.rows(), 2);
  original.setShapeVectors(shapeVectors_);

  auto tempFile = temporaryFile("test", ".bin");
  saveBlendShapeBase(tempFile.path(), original);
  BlendShapeBase loaded = loadBlendShapeBase(tempFile.path());

  EXPECT_TRUE(loaded.getShapeVectors().isApprox(original.getShapeVectors()));
}

TEST_F(BlendShapeIOTest, SaveAndLoadRoundTrip) {
  BlendShape original(baseShape_, 2);
  original.setShapeVectors(shapeVectors_);

  auto tempFile = temporaryFile("test", ".bin");
  saveBlendShape(tempFile.path(), original);
  BlendShape loaded = loadBlendShape(tempFile.path());

  EXPECT_TRUE(loaded.isApprox(original));
}

TEST_F(BlendShapeIOTest, SaveToStream) {
  BlendShape blendShape(baseShape_, 2);
  blendShape.setShapeVectors(shapeVectors_);

  std::ostringstream oss(std::ios::binary);
  saveBlendShape(oss, blendShape);

  std::istringstream iss(oss.str(), std::ios::binary);
  BlendShape loaded = loadBlendShape(iss);
  EXPECT_TRUE(loaded.isApprox(blendShape));
}

TEST_F(BlendShapeIOTest, SaveToInvalidPath) {
  BlendShape blendShape(baseShape_, 2);
  EXPECT_NO_THROW(saveBlendShape("/invalid/path/file.bin", blendShape));
}

TEST_F(BlendShapeIOTest, EdgeCases) {
  // Zero shapes
  std::ostringstream oss(std::ios::binary);
  uint64_t rows = 9, cols = 0;
  oss.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
  oss.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
  oss.write(reinterpret_cast<const char*>(baseShape_.data()), sizeof(float) * 9);

  std::istringstream iss(oss.str(), std::ios::binary);
  BlendShape result = loadBlendShape(iss);
  EXPECT_EQ(result.shapeSize(), 0);

  // Large expected values
  std::string data = createBlendShapeData();
  std::istringstream iss2(data, std::ios::binary);
  BlendShape result2 = loadBlendShape(iss2, 100, 100);
  EXPECT_EQ(result2.modelSize(), 3);
  EXPECT_EQ(result2.shapeSize(), 2);
}
