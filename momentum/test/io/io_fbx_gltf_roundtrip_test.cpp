/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/blend_shape.h>
#include <momentum/character/character.h>
#include <momentum/io/fbx/fbx_io.h>
#include <momentum/io/gltf/gltf_io.h>
#include <momentum/io/skeleton/parameter_transform_io.h>
#include <momentum/test/character/character_helpers_gtest.h>
#include <momentum/test/io/io_helpers.h>

#include <gtest/gtest.h>
#include <filesystem>

using namespace momentum;

TEST(IoFbxGltfRoundtripTest, BlendShapes) {
  auto resourcesEnvVar = GetEnvVar("TEST_MOMENTUM_MODELS_PATH");
  if (!resourcesEnvVar.has_value()) {
    GTEST_SKIP() << "Environment variable 'TEST_MOMENTUM_MODELS_PATH' is not set.";
  }

  const KeepLocators keepLocators = KeepLocators::No;
  const bool permissive = true;
  const LoadBlendShapes loadBlendShapes = LoadBlendShapes::Yes;

  const filesystem::path fbxPath =
      filesystem::path(resourcesEnvVar.value()) / "trinity_v4/trinity_basesewn_lod3_tris_v4.0.fbx";

  // Load character from FBX
  Character fbxCharacter;
  EXPECT_NO_THROW(
      fbxCharacter = loadFbxCharacter(fbxPath, keepLocators, permissive, loadBlendShapes));
  EXPECT_TRUE(fbxCharacter.blendShape != nullptr) << "FBX character should have blendshapes";
  EXPECT_EQ(fbxCharacter.blendShape->shapeSize(), 72) << "Expected 72 blendshapes from FBX";

  const filesystem::path modelPath =
      filesystem::path(resourcesEnvVar.value()) / "compact_v6_1.model";
  std::tie(fbxCharacter.parameterTransform, fbxCharacter.parameterLimits) =
      loadModelDefinition(modelPath, fbxCharacter.skeleton);

  // Save to GLTF
  const auto gltfFile = temporaryFile("roundtrip_blendshapes", "glb");
  const filesystem::path& gltfPath = gltfFile.path();
  EXPECT_NO_THROW(saveCharacter(gltfPath, fbxCharacter));
  EXPECT_TRUE(filesystem::exists(gltfPath)) << "GLTF file should be saved";

  // Load back from GLTF
  Character gltfCharacter;
  EXPECT_NO_THROW(gltfCharacter = loadGltfCharacter(gltfPath));

  // Compare characters (including blendshapes)
  compareChars(fbxCharacter, gltfCharacter, false);
}
