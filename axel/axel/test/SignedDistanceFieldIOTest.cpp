/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <filesystem>

#include <gtest/gtest.h>

#include "axel/SignedDistanceFieldIO.h"

namespace axel {

namespace {

std::string getTempFilePath(const std::string& suffix) {
  return (std::filesystem::temp_directory_path() / ("sdf_io_test_" + suffix + ".msgpack")).string();
}

} // namespace

TEST(SignedDistanceFieldIOTest, RoundtripFloat) {
  const BoundingBoxf bounds(
      Eigen::Vector3f(-1.0f, -2.0f, -3.0f), Eigen::Vector3f(4.0f, 5.0f, 6.0f));
  const Eigen::Vector3<Index> resolution(3, 4, 5);

  SignedDistanceFieldf sdf(bounds, resolution);
  for (Index k = 0; k < resolution.z(); ++k) {
    for (Index j = 0; j < resolution.y(); ++j) {
      for (Index i = 0; i < resolution.x(); ++i) {
        sdf.set(i, j, k, static_cast<float>(i * 100 + j * 10 + k) * 0.1f);
      }
    }
  }

  const auto path = getTempFilePath("float");
  saveSdfToMsgpack(sdf, path);

  const auto loaded = loadSdfFromMsgpack<float>(path);

  EXPECT_EQ(loaded.resolution(), sdf.resolution());
  EXPECT_NEAR(loaded.bounds().min().x(), sdf.bounds().min().x(), 1e-6f);
  EXPECT_NEAR(loaded.bounds().min().y(), sdf.bounds().min().y(), 1e-6f);
  EXPECT_NEAR(loaded.bounds().min().z(), sdf.bounds().min().z(), 1e-6f);
  EXPECT_NEAR(loaded.bounds().max().x(), sdf.bounds().max().x(), 1e-6f);
  EXPECT_NEAR(loaded.bounds().max().y(), sdf.bounds().max().y(), 1e-6f);
  EXPECT_NEAR(loaded.bounds().max().z(), sdf.bounds().max().z(), 1e-6f);

  ASSERT_EQ(loaded.data().size(), sdf.data().size());
  for (size_t i = 0; i < sdf.data().size(); ++i) {
    EXPECT_NEAR(loaded.data()[i], sdf.data()[i], 1e-6f) << "Mismatch at index " << i;
  }

  std::remove(path.c_str());
}

TEST(SignedDistanceFieldIOTest, RoundtripDouble) {
  const BoundingBoxd bounds(Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(1.0, 1.0, 1.0));
  const Eigen::Vector3<Index> resolution(2, 2, 2);
  std::vector<double> data = {1.23456789, -2.87654321, 0.5, -0.5, 3.14, -3.14, 0.0, 1.0};

  SignedDistanceFieldd sdf(bounds, resolution, std::move(data));

  const auto path = getTempFilePath("double");
  saveSdfToMsgpack(sdf, path);

  const auto loaded = loadSdfFromMsgpack<double>(path);

  EXPECT_EQ(loaded.resolution(), sdf.resolution());
  ASSERT_EQ(loaded.data().size(), sdf.data().size());
  for (size_t i = 0; i < sdf.data().size(); ++i) {
    EXPECT_NEAR(loaded.data()[i], sdf.data()[i], 1e-12) << "Mismatch at index " << i;
  }

  std::remove(path.c_str());
}

TEST(SignedDistanceFieldIOTest, SampleConsistencyAfterRoundtrip) {
  auto sdf = SignedDistanceFieldf::createSphere(2.0f, Eigen::Vector3<Index>(8, 8, 8));

  const auto path = getTempFilePath("sample");
  saveSdfToMsgpack(sdf, path);
  const auto loaded = loadSdfFromMsgpack<float>(path);

  const std::vector<Eigen::Vector3f> testPositions = {
      {0.0f, 0.0f, 0.0f},
      {1.0f, 0.0f, 0.0f},
      {0.5f, 0.5f, 0.5f},
      {-1.0f, 0.5f, -0.3f},
  };

  for (const auto& pos : testPositions) {
    EXPECT_NEAR(loaded.sample(pos), sdf.sample(pos), 1e-6f)
        << "Sample mismatch at [" << pos.transpose() << "]";
  }

  std::remove(path.c_str());
}

TEST(SignedDistanceFieldIOTest, RoundtripMultipleSdfs) {
  const Eigen::Vector3<Index> res(3, 3, 3);

  auto sphere1 = SignedDistanceFieldf::createSphere(1.0f, res);
  auto sphere2 = SignedDistanceFieldf::createSphere(2.0f, res);
  auto sphere3 = SignedDistanceFieldf::createSphere(0.5f, res);

  std::unordered_map<std::string, SignedDistanceFieldf> sdfs;
  sdfs.emplace("chest", std::move(sphere1));
  sdfs.emplace("r_upperarm", std::move(sphere2));
  sdfs.emplace("head", std::move(sphere3));

  const auto path = getTempFilePath("multi");
  saveSdfsToMsgpack(sdfs, path);

  const auto loaded = loadSdfsFromMsgpack<float>(path);

  ASSERT_EQ(loaded.size(), 3);
  ASSERT_TRUE(loaded.count("chest"));
  ASSERT_TRUE(loaded.count("r_upperarm"));
  ASSERT_TRUE(loaded.count("head"));

  for (const auto& [name, original] : sdfs) {
    const auto& restored = loaded.at(name);
    EXPECT_EQ(restored.resolution(), original.resolution());
    ASSERT_EQ(restored.data().size(), original.data().size());
    for (size_t i = 0; i < original.data().size(); ++i) {
      EXPECT_NEAR(restored.data()[i], original.data()[i], 1e-6f)
          << "Mismatch in '" << name << "' at index " << i;
    }
  }

  std::remove(path.c_str());
}

TEST(SignedDistanceFieldIOTest, RoundtripMultipleSdfsWithParentJoint) {
  const Eigen::Vector3<Index> res(3, 3, 3);

  std::unordered_map<std::string, SignedDistanceFieldf> sdfs;
  auto chest = SignedDistanceFieldf::createSphere(1.0f, res);
  chest.setParentJoint("c_spine2");
  auto pelvis = SignedDistanceFieldf::createSphere(0.5f, res);
  pelvis.setParentJoint("root");
  auto head = SignedDistanceFieldf::createSphere(0.3f, res);
  sdfs.emplace("chest", std::move(chest));
  sdfs.emplace("pelvis", std::move(pelvis));
  sdfs.emplace("head", std::move(head));

  const auto path = getTempFilePath("parent_joint");
  saveSdfsToMsgpack(sdfs, path);

  const auto loaded = loadSdfsFromMsgpack<float>(path);

  ASSERT_EQ(loaded.size(), 3);
  EXPECT_EQ(loaded.at("chest").parentJoint(), "c_spine2");
  EXPECT_EQ(loaded.at("pelvis").parentJoint(), "root");
  EXPECT_EQ(loaded.at("head").parentJoint(), "");

  for (const auto& [name, original] : sdfs) {
    const auto& restored = loaded.at(name);
    EXPECT_EQ(restored.resolution(), original.resolution());
    ASSERT_EQ(restored.data().size(), original.data().size());
  }

  std::remove(path.c_str());
}

TEST(SignedDistanceFieldIOTest, LoadNonexistentFileThrows) {
  EXPECT_THROW(loadSdfFromMsgpack<float>("/nonexistent/path/sdf.msgpack"), std::runtime_error);
}

} // namespace axel
