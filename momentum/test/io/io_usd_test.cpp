/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/usd/usd_io.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skin_weights.h"
#include "momentum/math/mesh.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/io/io_helpers.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>
#include <vector>

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
