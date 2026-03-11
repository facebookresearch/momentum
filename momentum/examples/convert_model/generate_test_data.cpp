/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/character.h>
#include <momentum/character/skeleton.h>
#include <momentum/io/fbx/fbx_io.h>
#include <momentum/io/gltf/gltf_io.h>
#include <momentum/io/skeleton/locator_io.h>
#include <momentum/io/skeleton/parameter_transform_io.h>
#include <momentum/test/character/character_helpers.h>

#include <filesystem>
#include <fstream>
#include <iostream>

using namespace momentum;

// Create a simple motion (10 frames of T-pose) in MODEL parameter space
MatrixXf createSimpleMotion(const Character& character, int numFrames = 10) {
  const Eigen::Index numParams = character.parameterTransform.numAllModelParameters();
  MatrixXf motion = MatrixXf::Zero(numParams, numFrames);

  // Create simple animation (slight rotation on first parameter if it exists)
  for (int frame = 0; frame < numFrames; ++frame) {
    if (numParams > 7) {
      motion(7, frame) = std::sin(frame * 0.3f) * 0.1f; // joint1_rx
    }
  }

  return motion;
}

int main() {
  const std::filesystem::path outputDir =
      std::filesystem::path(__FILE__).parent_path() / "test_data";

  // Create output directory
  std::filesystem::create_directories(outputDir);

  std::cout << "Generating minimal test data files...\n";

  // Create character
  Character character = createTestCharacter(3);
  const float fps = 30.0f;

  // 1. character_with_motion.glb
  std::cout << "  Creating character_with_motion.glb...\n";
  MatrixXf motion = createSimpleMotion(character, 10);
  JointParameters offsets = character.parameterTransform.zero();
  saveGltfCharacter(
      (outputDir / "character_with_motion.glb").string(),
      character,
      fps,
      {character.parameterTransform.name, motion},
      {character.skeleton.getJointNames(), offsets},
      {});

  // 2. model_with_motion.glb (same as above)
  std::cout << "  Creating model_with_motion.glb...\n";
  saveGltfCharacter(
      (outputDir / "model_with_motion.glb").string(),
      character,
      fps,
      {character.parameterTransform.name, motion},
      {character.skeleton.getJointNames(), offsets},
      {});

  // 3. motion.fbx (character + motion) - enable permissive mode for mesh without skinning
  std::cout << "  Creating motion.fbx...\n";
  FileSaveOptions fbxOptions;
  // No permissive mode needed for skeleton-only
  saveFbx((outputDir / "motion.fbx").string(), character, motion, offsets.v, fps, {}, fbxOptions);

  // 4. character.fbx (character without motion)
  std::cout << "  Creating character.fbx...\n";
  MatrixXf emptyMotion(character.parameterTransform.numAllModelParameters(), 0);
  saveFbx(
      (outputDir / "character.fbx").string(),
      character,
      emptyMotion,
      offsets.v,
      fps,
      {},
      fbxOptions);

  // 5. character_no_motion.fbx (skeleton only)
  std::cout << "  Creating character_no_motion.fbx...\n";
  saveFbx(
      (outputDir / "character_no_motion.fbx").string(),
      character,
      emptyMotion,
      offsets.v,
      fps,
      {},
      fbxOptions);

  // 6. multiple_motions.fbx (longer motion sequence)
  std::cout << "  Creating multiple_motions.fbx...\n";
  MatrixXf longMotion = createSimpleMotion(character, 30);
  saveFbx(
      (outputDir / "multiple_motions.fbx").string(),
      character,
      longMotion,
      offsets.v,
      fps,
      {},
      fbxOptions);

  // 7. character.model - write parameter transform in .model format
  std::cout << "  Creating character.model...\n";
  {
    const auto filepath = (outputDir / "character.model").string();
    std::ofstream ofs(filepath);
    ofs << writeParameterTransform(character.parameterTransform, character.skeleton);
    ofs.close();
  }

  // 8. character.locators - save locators from character
  std::cout << "  Creating character.locators...\n";
  {
    const auto filepath = (outputDir / "character.locators").string();
    saveLocators(filepath, character.locators, character.skeleton, LocatorSpace::Local);
  }

  std::cout << "\nTest data files created in: " << outputDir << "\n";

  return 0;
}
