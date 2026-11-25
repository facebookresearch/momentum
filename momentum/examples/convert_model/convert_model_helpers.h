/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/common/filesystem.h>

#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace momentum {

/// Command-line options for the convert_model tool.
struct Options {
  std::string input_model_file;
  std::string input_params_file;
  std::string input_locator_file;
  std::string input_motion_file;
  std::string output_model_file;
  std::string output_locator_local;
  std::string output_locator_global;
  std::string output_params_file;
  std::string fbx_namespace;
  /// Name of the one character to convert; empty converts every character of the input.
  std::string character_name;
  bool save_markers = false;
  bool character_mesh_save = false;
};

/// Bundles the result of loading motion data from a file.
struct MotionData {
  MatrixXf poses;
  JointParameters offsets;
  float fps = 120.0f;
};

/// A character to convert, with its motion.
struct SceneCharacter {
  Character character;
  /// Model parameters (numAllModelParameters x numFrames); no columns without motion.
  MatrixXf poses;
  /// Identity joint parameters; empty or zero when the file has none.
  JointParameters offsets;
};

/// The characters to convert, one or several, and the frame rate of their motion.
struct Scene {
  std::vector<SceneCharacter> characters;
  float fps = 120.0f;
};

/// Finds the index of the longest motion in a list of motions.
/// Returns -1 if no valid motion is found.
int findLongestMotion(const std::vector<MatrixXf>& motions);

/// Finds the FBX take (animation stack) to convert for every character: the take in which some
/// character has the longest motion.
///
/// @param characters The characters with their takes, index-aligned across characters, as
/// loadFbxCharactersWithMotion() returns them.
/// @return The take index, or -1 if no take has frames.
int findLongestTake(std::span<const std::pair<Character, std::vector<MatrixXf>>> characters);

/// Finds the FBX take to convert for the characters named in `names`, which get the motion of the
/// file's characters with their names: the take in which one of these characters has the longest
/// motion, as findLongestTake() finds it. When no character is named in `names`, every character
/// counts.
///
/// @param characters The characters with their takes, as findLongestTake() takes them.
/// @param names The names of the characters that get motion.
/// @return The take index, or -1 if no take has frames.
int findLongestTake(
    std::span<const std::pair<Character, std::vector<MatrixXf>>> characters,
    std::span<const std::string> names);

/// Loads the characters to convert and their motion.
///
/// When the input has several characters or options.character_name selects one, the characters
/// are those of the model file, or of the motion file when there is no model file, as
/// loadGltfCharacters(), loadGltfCharactersWithMotion(), loadFbxCharacters() and
/// loadFbxCharactersWithMotion() read them; only the one named options.character_name when it is
/// not empty. Without a model file, each character keeps the motion stored with it in the motion
/// file. With a model file, each character gets the motion of the motion-file character with the
/// same name.
///
/// Otherwise the input converts as one character, as loadModelCharacter(), loadGlbMotion(),
/// loadFbxMotion() and the BVH loaders read it: a file with one character, a file of another
/// format, a file that the multi-character loaders cannot read (after a warning), or no input file
/// at all (an empty character). A BVH file holds one character; its motion is mapped onto the
/// model character by joint name.
///
/// The parameter and locator files, and the locator and parameter outputs, apply to one character,
/// so this throws when they are given for several characters. It also throws when no character has
/// the selected name, when a name matches several characters, when no character of the model file
/// has a motion-file character with the same name, when a BVH motion file comes with several model
/// characters, and when --character selects a character of a BVH motion file without a model
/// file.
///
/// @param options The command-line options.
/// @return The characters with their motion, and its frame rate.
Scene loadScene(const Options& options);

/// Loads motion data from a glTF/GLB file. If no separate model was provided,
/// the character is also loaded from the GLB file.
MotionData loadGlbMotion(
    const filesystem::path& motionPath,
    Character& character,
    bool hasModel,
    const Options& options);

/// Loads motion data from an FBX file. If no separate model was provided,
/// the character is also initialized from the FBX file.
MotionData loadFbxMotion(
    const filesystem::path& motionPath,
    Character& character,
    bool hasModel,
    const Options& options);

/// Whether `path` names a BVH file.
///
/// @param path A file path.
/// @return Whether the extension of `path` is .bvh, in any letter case.
bool isBvhFile(const filesystem::path& path);

/// Loads the input model character, applying optional parameter and locator files.
Character loadModelCharacter(const Options& options);

/// Initializes the character from the character of an FBX or BVH motion file when no separate
/// model was provided. Applies parameter transform and locators if specified in the options.
///
/// @param character The character to initialize.
/// @param motionCharacter The character of the motion file.
/// @param options The command-line options: the parameter and locator files.
void initializeCharacterFromMotionFile(
    Character& character,
    const Character& motionCharacter,
    const Options& options);

/// Converts joint-parameter motion data to model parameters using
/// inverse parameter transform. May lose information.
MatrixXf convertToModelParams(const MatrixXf& motion, const ParameterTransform& parameterTransform);

/// Converts joint-parameter motion from a source skeleton to model parameters
/// on a target character by mapping global joint transforms based on joint names,
/// then converting back to joint parameters based on the target skeleton's bind
/// pose. This handles cases where the source and target skeletons have different
/// joint orders or rest/bind poses.
MatrixXf convertMotionWithJointMapping(
    const MatrixXf& motion,
    const Skeleton& sourceSkeleton,
    const Character& targetCharacter);

} // namespace momentum
