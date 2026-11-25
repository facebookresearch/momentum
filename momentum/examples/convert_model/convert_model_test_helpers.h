/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/common/filesystem.h>
#include <momentum/io/fbx/fbx_builder.h>
#include <momentum/io/gltf/gltf_builder.h>
#include <momentum/test/character/character_helpers.h>

#include <cmath>
#include <string>
#include <vector>

namespace momentum {

/// A character with the motion it is saved with.
struct TestSceneCharacter {
  Character character;
  /// Model parameters (numAllModelParameters x numFrames).
  MatrixXf motion;
};

/// A test character named `name` with `numJoints` joints, physical properties on its first two
/// joints, metadata, and a motion of `numFrames` frames, a sine wave with the phase `phase`.
inline TestSceneCharacter createTestSceneCharacter(
    const std::string& name,
    size_t numJoints,
    Eigen::Index numFrames,
    float phase) {
  TestSceneCharacter result{
      .character = withTestPhysicalProperties(createTestCharacter(numJoints)), .motion = {}};
  result.character.name = name;
  result.character.metadata = R"({"source": ")" + name + R"("})";
  const auto numParameters = result.character.parameterTransform.numAllModelParameters();
  result.motion.resize(numParameters, numFrames);
  for (Eigen::Index frame = 0; frame < numFrames; ++frame) {
    for (Eigen::Index parameter = 0; parameter < numParameters; ++parameter) {
      result.motion(parameter, frame) = 0.1f *
          std::sin(phase + 0.3f * static_cast<float>(frame) + 0.7f * static_cast<float>(parameter));
    }
  }
  return result;
}

/// Two test characters, "a" and "b", with different skeletons and motions.
inline std::vector<TestSceneCharacter> createTwoCharacterScene() {
  return {createTestSceneCharacter("a", 3, 4, 0.0f), createTestSceneCharacter("b", 5, 6, 1.0f)};
}

/// Saves the characters with their motions to one glTF file.
inline void saveTestSceneAsGltf(
    const filesystem::path& path,
    const std::vector<TestSceneCharacter>& characters,
    float fps) {
  GltfBuilder builder;
  builder.setFps(fps);
  for (const auto& [character, motion] : characters) {
    const std::string name = builder.addCharacter(character);
    builder.addMotion(
        character, fps, {character.parameterTransform.name, motion}, {}, true, "default", {}, name);
  }
  builder.save(path);
}

/// Saves the characters with their motions to one FBX file.
inline void saveTestSceneAsFbx(
    const filesystem::path& path,
    const std::vector<TestSceneCharacter>& characters,
    float fps) {
  FbxBuilder builder;
  for (const auto& [character, motion] : characters) {
    const std::string name = builder.addCharacter(character);
    builder.addMotion(character, fps, motion, {}, name);
  }
  builder.save(path);
}

/// The joint parameters (numJointParameters x numFrames) of a model-parameter motion.
inline MatrixXf toJointParameters(const Character& character, const MatrixXf& motion) {
  MatrixXf result(character.parameterTransform.numJointParameters(), motion.cols());
  for (Eigen::Index frame = 0; frame < motion.cols(); ++frame) {
    result.col(frame) = character.parameterTransform.apply(ModelParameters(motion.col(frame))).v;
  }
  return result;
}

} // namespace momentum
