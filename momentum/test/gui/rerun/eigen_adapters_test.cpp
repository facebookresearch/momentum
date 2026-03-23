/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/gui/rerun/eigen_adapters.h>

#include <gtest/gtest.h>

// --- Vector3f adapter tests ---

TEST(EigenAdaptersTest, Vector3fBorrow) {
  std::vector<Eigen::Vector3f> positions = {
      {1.0f, 2.0f, 3.0f},
      {4.0f, 5.0f, 6.0f},
      {7.0f, 8.0f, 9.0f},
  };

  auto collection =
      rerun::CollectionAdapter<rerun::Position3D, std::vector<Eigen::Vector3f>>{}(positions);

  ASSERT_EQ(collection.size(), 3);
  EXPECT_FLOAT_EQ(collection[0].x(), 1.0f);
  EXPECT_FLOAT_EQ(collection[0].y(), 2.0f);
  EXPECT_FLOAT_EQ(collection[0].z(), 3.0f);
  EXPECT_FLOAT_EQ(collection[1].x(), 4.0f);
  EXPECT_FLOAT_EQ(collection[1].y(), 5.0f);
  EXPECT_FLOAT_EQ(collection[1].z(), 6.0f);
  EXPECT_FLOAT_EQ(collection[2].x(), 7.0f);
  EXPECT_FLOAT_EQ(collection[2].y(), 8.0f);
  EXPECT_FLOAT_EQ(collection[2].z(), 9.0f);
}

TEST(EigenAdaptersTest, Vector3fMove) {
  std::vector<Eigen::Vector3f> positions = {
      {1.0f, 2.0f, 3.0f},
      {4.0f, 5.0f, 6.0f},
  };

  auto collection = rerun::CollectionAdapter<rerun::Position3D, std::vector<Eigen::Vector3f>>{}(
      std::move(positions));

  ASSERT_EQ(collection.size(), 2);
  EXPECT_FLOAT_EQ(collection[0].x(), 1.0f);
  EXPECT_FLOAT_EQ(collection[0].y(), 2.0f);
  EXPECT_FLOAT_EQ(collection[0].z(), 3.0f);
  EXPECT_FLOAT_EQ(collection[1].x(), 4.0f);
  EXPECT_FLOAT_EQ(collection[1].y(), 5.0f);
  EXPECT_FLOAT_EQ(collection[1].z(), 6.0f);
}

TEST(EigenAdaptersTest, EmptyVector3f) {
  std::vector<Eigen::Vector3f> empty;

  auto collection =
      rerun::CollectionAdapter<rerun::Position3D, std::vector<Eigen::Vector3f>>{}(empty);

  EXPECT_EQ(collection.size(), 0);
}

// --- Color adapter tests ---

TEST(EigenAdaptersTest, Vector3UintToColor) {
  std::vector<Eigen::Vector3<uint8_t>> vertexColors = {
      {255, 0, 0},
      {0, 255, 0},
      {0, 0, 255},
  };

  auto collection =
      rerun::CollectionAdapter<rerun::Color, std::vector<Eigen::Vector3<uint8_t>>>{}(vertexColors);

  ASSERT_EQ(collection.size(), 3);
  EXPECT_EQ(collection[0].r(), 255);
  EXPECT_EQ(collection[0].g(), 0);
  EXPECT_EQ(collection[0].b(), 0);
  EXPECT_EQ(collection[0].a(), 255);
  EXPECT_EQ(collection[1].r(), 0);
  EXPECT_EQ(collection[1].g(), 255);
  EXPECT_EQ(collection[1].b(), 0);
  EXPECT_EQ(collection[2].r(), 0);
  EXPECT_EQ(collection[2].g(), 0);
  EXPECT_EQ(collection[2].b(), 255);
}

TEST(EigenAdaptersTest, EmptyColorVector) {
  std::vector<Eigen::Vector3<uint8_t>> empty;

  auto collection =
      rerun::CollectionAdapter<rerun::Color, std::vector<Eigen::Vector3<uint8_t>>>{}(empty);

  EXPECT_EQ(collection.size(), 0);
}
