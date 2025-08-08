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
    return;
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
    return;
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
    return;
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

  EXPECT_EQ(loadedCharacter.skeleton.joints.size(), testCharacter.skeleton.joints.size());
  EXPECT_EQ(loadedCharacter.mesh->vertices.size(), testCharacter.mesh->vertices.size());
  EXPECT_EQ(loadedCharacter.mesh->faces.size(), testCharacter.mesh->faces.size());

  for (size_t i = 0; i < testCharacter.skeleton.joints.size(); ++i) {
    EXPECT_EQ(loadedCharacter.skeleton.joints[i].name, testCharacter.skeleton.joints[i].name);
  }
}

TEST_F(UsdIoTest, LoadFromBuffer) {
  auto usdPath = getTestResourcePath("simple_character.usda");
  if (!usdPath.has_value()) {
    GTEST_SKIP() << "Environment variable 'TEST_RESOURCES_PATH' is not set.";
    return;
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
  const auto character = loadUsdCharacter(gsl::span<const std::byte>(buffer));

  // Verify character was loaded correctly
  EXPECT_GT(character.skeleton.joints.size(), 0);
  EXPECT_GT(character.mesh->vertices.size(), 0);
  EXPECT_GT(character.mesh->faces.size(), 0);
}
