/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define DEFAULT_LOG_CHANNEL "convert_model"

#include "convert_model_helpers.h"

#include <momentum/character/character_utility.h>
#include <momentum/character/inverse_parameter_transform.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/common/exception.h>
#include <momentum/common/log.h>
#include <momentum/io/character_io.h>
#include <momentum/io/fbx/fbx_io.h>
#include <momentum/io/gltf/gltf_io.h>
#include <momentum/io/skeleton/locator_io.h>
#include <momentum/io/skeleton/parameter_transform_io.h>
#include <momentum/io/skeleton/parameters_io.h>
#include <momentum/io/urdf/urdf_io.h>

#include <algorithm>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace {

void applyOptionalInputs(momentum::Character& character, const momentum::Options& options) {
  if (!options.input_params_file.empty()) {
    auto def = momentum::loadMomentumModel(options.input_params_file);
    momentum::loadParameters(def, character);
  }
  if (!options.input_locator_file.empty()) {
    character.locators = momentum::loadLocators(
        options.input_locator_file, character.skeleton, character.parameterTransform);
  }
}

// A character read from a glTF motion file without a model file comes with its own parameter
// transform and locators, so the parameter and locator files do not apply.
void warnIgnoredInputs(const momentum::Options& options) {
  if (!options.input_params_file.empty()) {
    MT_LOGW("Ignoring input parameter transform {}.", options.input_params_file);
  }
  if (!options.input_locator_file.empty()) {
    MT_LOGW("Ignoring input locators {}.", options.input_locator_file);
  }
}

// The index of the longest of `lengths` (frame counts), as findLongestMotion() and
// findLongestTake() choose. Returns -1 when no length is positive, and warns when there is no
// length or a single zero length. Warns when it chooses one of several lengths.
int findLongestLength(std::span<const Eigen::Index> lengths) {
  if (lengths.empty() || (lengths.size() == 1 && lengths.front() == 0)) {
    MT_LOGW("No motion loaded from file");
    return -1;
  }
  int longest = -1;
  Eigen::Index maxLength = 0;
  int index = 0;
  for (const Eigen::Index length : lengths) {
    if (length > maxLength) {
      maxLength = length;
      longest = index;
    }
    ++index;
  }
  if (maxLength > 0 && lengths.size() > 1) {
    MT_LOGW("More than one motion found; only taking the longest one");
  }
  return longest;
}

// Whether characters read from a file convert as a scene: when there are several, or when one is
// selected by name.
bool convertsAsScene(size_t numCharacters, const momentum::Options& options) {
  return numCharacters > 1 || !options.character_name.empty();
}

// Reads the characters of a file with `load`. When no character is selected and `load` throws,
// returns nullopt after a warning, so that the file converts as one character as before: the
// multi-character loaders reject some files that the single-character loaders read, such as an
// FBX mesh skinned to the joints of several skeletons.
template <typename Load>
auto loadCharacters(const filesystem::path& path, const momentum::Options& options, Load load)
    -> std::optional<decltype(load())> {
  if (!options.character_name.empty()) {
    return load();
  }
  try {
    return load();
  } catch (const std::runtime_error& e) {
    MT_LOGW("Converting {} as one character: {}", path.string(), e.what());
    return std::nullopt;
  }
}

std::string joinNames(std::span<const std::string> names) {
  std::string result;
  for (const auto& name : names) {
    if (!result.empty()) {
      result += ", ";
    }
    result += name;
  }
  return result;
}

// The entries named `name`, or every entry when `name` is empty. Throws when no entry, or more than
// one, has that name.
template <typename T, typename NameOf>
std::vector<T> selectByName(
    std::vector<T> entries,
    const std::string& name,
    const filesystem::path& file,
    NameOf nameOf) {
  if (name.empty()) {
    return entries;
  }
  std::vector<std::string> names;
  std::vector<T> selected;
  for (auto& entry : entries) {
    names.push_back(nameOf(entry));
    if (names.back() == name) {
      selected.push_back(std::move(entry));
    }
  }
  MT_THROW_IF(
      selected.empty(),
      "{} has no character named '{}'; its characters are: {}",
      file.string(),
      name,
      joinNames(names));
  MT_THROW_IF(
      selected.size() > 1, "{} has {} characters named '{}'", file.string(), selected.size(), name);
  return selected;
}

// Throws when an option that applies to one character is given for several characters.
void checkOneCharacterOptions(
    const momentum::Options& options,
    std::span<const std::string> characterNames,
    const filesystem::path& file) {
  if (characterNames.size() <= 1) {
    return;
  }
  std::vector<std::string> given;
  if (!options.input_params_file.empty()) {
    given.emplace_back("--parameters");
  }
  if (!options.input_locator_file.empty()) {
    given.emplace_back("--locator");
  }
  if (!options.output_locator_local.empty()) {
    given.emplace_back("--out-locators-local");
  }
  if (!options.output_locator_global.empty()) {
    given.emplace_back("--out-locators-global");
  }
  if (!options.output_params_file.empty()) {
    given.emplace_back("--out-parameters");
  }
  MT_THROW_IF(
      !given.empty(),
      "{} apply to one character, but {} has {} characters ({}); select one with --character",
      joinNames(given),
      file.string(),
      characterNames.size(),
      joinNames(characterNames));
}

template <typename T, typename NameOf>
std::vector<std::string> namesOf(std::span<const T> entries, NameOf nameOf) {
  std::vector<std::string> names;
  names.reserve(entries.size());
  std::ranges::transform(entries, std::back_inserter(names), nameOf);
  return names;
}

const std::string& characterName(const momentum::Character& character) {
  return character.name;
}

// The characters of the model file, each read as loadModelCharacter() reads a single character,
// before the parameter and locator files.
std::vector<momentum::Character> loadModelCharacters(const filesystem::path& modelPath) {
  switch (momentum::parseCharacterFormat(modelPath)) {
    case momentum::CharacterFormat::Gltf: {
      auto characters = momentum::loadGltfCharacters(modelPath);
      // loadFullCharacter() reads glTF characters without their blend shapes.
      for (auto& character : characters) {
        character.blendShape.reset();
        character.faceExpressionBlendShape.reset();
      }
      return characters;
    }
    case momentum::CharacterFormat::Fbx:
      return momentum::loadFbxCharacters(
          modelPath,
          momentum::KeepLocators::Yes,
          momentum::Permissive::No,
          momentum::LoadBlendShapes::No);
    case momentum::CharacterFormat::Usd:
    case momentum::CharacterFormat::Unknown:
      break;
  }
  momentum::Options modelOnly;
  modelOnly.input_model_file = modelPath.string();
  return {momentum::loadModelCharacter(modelOnly)};
}

// Gives each scene character the motion of the source with the same name, with `assign`. Warns
// about scene characters without a source and, if `warnUnused`, about sources that no scene
// character uses. Throws when a name matches several sources, and when no scene character has a
// source.
template <typename Source, typename NameOf, typename Assign>
void assignMotionByName(
    momentum::Scene& scene,
    std::span<const Source> sources,
    const filesystem::path& motionPath,
    bool warnUnused,
    NameOf nameOf,
    Assign assign) {
  std::vector<bool> used(sources.size(), false);
  size_t matched = 0;
  for (auto& target : scene.characters) {
    const Source* match = nullptr;
    size_t matchIndex = 0;
    size_t index = 0;
    for (const auto& source : sources) {
      if (nameOf(source) == target.character.name) {
        MT_THROW_IF(
            match != nullptr,
            "{} has several characters named '{}'",
            motionPath.string(),
            target.character.name);
        match = &source;
        matchIndex = index;
      }
      ++index;
    }
    if (match == nullptr) {
      MT_LOGW(
          "{} has no character named '{}'; converting that character without motion.",
          motionPath.string(),
          target.character.name);
      continue;
    }
    assign(target, *match);
    used.at(matchIndex) = true;
    ++matched;
  }
  size_t index = 0;
  for (const auto& source : sources) {
    MT_LOGW_IF(
        warnUnused && !used.at(index),
        "Ignoring the motion of character '{}' in {}: no character to convert has that name.",
        nameOf(source),
        motionPath.string());
    ++index;
  }
  MT_THROW_IF(
      matched == 0,
      "No character of {} has the name of a character to convert",
      motionPath.string());
}

// Model parameters and identity of a glTF motion-file character, mapped by name onto `target`, as
// loadMotionOnCharacter() maps them.
void assignGltfMotion(momentum::SceneCharacter& target, const momentum::CharacterMotion& source) {
  target.poses = momentum::mapMotionToCharacter(
      {source.character.parameterTransform.name, source.motion}, target.character);
  if (source.identity.size() != 0) {
    target.offsets = momentum::mapIdentityToCharacter(
        {source.character.skeleton.getJointNames(), source.identity}, target.character);
  }
}

const std::string& gltfSourceName(const momentum::CharacterMotion& source) {
  return source.character.name;
}

using FbxCharacterTakes = std::pair<momentum::Character, std::vector<momentum::MatrixXf>>;

const std::string& fbxSourceName(const FbxCharacterTakes& source) {
  return source.first.name;
}

// The motion of an FBX character in `take`, mapped by joint name onto `target`, as loadFbxMotion()
// maps a single character.
void assignFbxTake(
    momentum::SceneCharacter& target,
    const momentum::Skeleton& sourceSkeleton,
    std::span<const momentum::MatrixXf> takes,
    size_t take) {
  if (take < takes.size()) {
    target.poses =
        momentum::convertMotionWithJointMapping(takes[take], sourceSkeleton, target.character);
  }
  target.offsets = target.character.parameterTransform.zero();
}

// Gives each character of the model file the motion of the motion-file character with the same
// name. Warns about motion-file characters that no character uses if `warnUnused`.
void addMotionByName(momentum::Scene& scene, const filesystem::path& motionPath, bool warnUnused) {
  switch (momentum::parseCharacterFormat(motionPath)) {
    case momentum::CharacterFormat::Gltf: {
      auto [sources, fps] = momentum::loadGltfCharactersWithMotion(motionPath);
      scene.fps = fps;
      assignMotionByName(
          scene,
          std::span<const momentum::CharacterMotion>(sources),
          motionPath,
          warnUnused,
          gltfSourceName,
          assignGltfMotion);
      return;
    }
    case momentum::CharacterFormat::Fbx: {
      auto [sources, fps] = momentum::loadFbxCharactersWithMotion(
          motionPath, momentum::KeepLocators::Yes, momentum::Permissive::No);
      // The take is chosen among the characters that get motion, as loadFbxMotionScene() chooses
      // it among the selected characters.
      std::vector<std::string> names(scene.characters.size());
      std::ranges::transform(
          scene.characters, names.begin(), [](const momentum::SceneCharacter& target) {
            return target.character.name;
          });
      const int take = momentum::findLongestTake(sources, names);
      if (take < 0) {
        return;
      }
      scene.fps = fps;
      assignMotionByName(
          scene,
          std::span<const FbxCharacterTakes>(sources),
          motionPath,
          warnUnused,
          fbxSourceName,
          [take](momentum::SceneCharacter& target, const FbxCharacterTakes& source) {
            assignFbxTake(target, source.first.skeleton, source.second, static_cast<size_t>(take));
          });
      return;
    }
    case momentum::CharacterFormat::Usd:
    case momentum::CharacterFormat::Unknown:
      break;
  }
  MT_LOGW("Unknown motion file format: {}. Exporting without motion.", motionPath.string());
}

// The characters of a motion file read with `load` and its frame rate, when they convert as a scene
// (see loadScene()): only the one selected by name, after a check of the one-character options.
template <typename Load, typename NameOf>
auto loadSelectedCharacters(
    const filesystem::path& motionPath,
    const momentum::Options& options,
    Load load,
    NameOf nameOf) -> std::optional<decltype(load())> {
  auto loaded = loadCharacters(motionPath, options, load);
  if (!loaded.has_value() || !convertsAsScene(std::get<0>(*loaded).size(), options)) {
    return std::nullopt;
  }
  auto& characters = std::get<0>(*loaded);
  characters = selectByName(std::move(characters), options.character_name, motionPath, nameOf);
  using Entry = typename std::decay_t<decltype(characters)>::value_type;
  checkOneCharacterOptions(
      options, namesOf(std::span<const Entry>(characters), nameOf), motionPath);
  return loaded;
}

// The characters of a glTF motion file with the motion stored with them, as loadGlbMotion() reads
// a single character; see loadScene().
std::optional<momentum::Scene> loadGltfMotionScene(
    const filesystem::path& motionPath,
    const momentum::Options& options) {
  auto loaded = loadSelectedCharacters(
      motionPath,
      options,
      [&] { return momentum::loadGltfCharactersWithMotion(motionPath); },
      gltfSourceName);
  if (!loaded.has_value()) {
    return std::nullopt;
  }
  auto& [selected, fps] = *loaded;
  warnIgnoredInputs(options);
  momentum::Scene scene;
  scene.fps = fps;
  for (auto& entry : selected) {
    scene.characters.push_back(
        {std::move(entry.character), std::move(entry.motion), std::move(entry.identity)});
  }
  return scene;
}

// The characters of an FBX motion file with their motion from one take, as loadFbxMotion() reads a
// single character; see loadScene().
std::optional<momentum::Scene> loadFbxMotionScene(
    const filesystem::path& motionPath,
    const momentum::Options& options) {
  auto loaded = loadSelectedCharacters(
      motionPath,
      options,
      [&] {
        return momentum::loadFbxCharactersWithMotion(
            motionPath, momentum::KeepLocators::Yes, momentum::Permissive::No);
      },
      fbxSourceName);
  if (!loaded.has_value()) {
    return std::nullopt;
  }
  auto& [selected, fps] = *loaded;
  const int take = momentum::findLongestTake(selected);
  momentum::Scene scene;
  if (take >= 0) {
    scene.fps = fps;
  }
  for (const auto& [fbxCharacter, takes] : selected) {
    momentum::SceneCharacter entry;
    momentum::initializeCharacterFromFbx(entry.character, fbxCharacter, options);
    if (take >= 0) {
      assignFbxTake(entry, fbxCharacter.skeleton, takes, static_cast<size_t>(take));
    }
    scene.characters.push_back(std::move(entry));
  }
  return scene;
}

} // namespace

namespace momentum {

int findLongestTake(std::span<const std::pair<Character, std::vector<MatrixXf>>> characters) {
  return findLongestTake(characters, {});
}

int findLongestTake(
    std::span<const std::pair<Character, std::vector<MatrixXf>>> characters,
    std::span<const std::string> names) {
  const auto isNamed = [names](const std::pair<Character, std::vector<MatrixXf>>& entry) {
    return std::ranges::find(names, entry.first.name) != names.end();
  };
  const bool someAreNamed = std::ranges::any_of(characters, isNamed);
  // The longest motion of any character that counts in each take.
  std::vector<Eigen::Index> takeLengths;
  for (const auto& entry : characters) {
    if (someAreNamed && !isNamed(entry)) {
      continue;
    }
    const auto& takes = entry.second;
    takeLengths.resize(std::max(takeLengths.size(), takes.size()), 0);
    for (size_t take = 0; take < takes.size(); ++take) {
      auto& length = takeLengths.at(take);
      length = std::max(length, takes.at(take).cols());
    }
  }
  return findLongestLength(takeLengths);
}

std::optional<Scene> loadScene(const Options& options) {
  const bool selectsCharacter = !options.character_name.empty();
  const bool hasMotion = !options.input_motion_file.empty();
  if (options.input_model_file.empty()) {
    if (!hasMotion) {
      MT_THROW_IF(selectsCharacter, "--character needs a model file or a motion file");
      return std::nullopt;
    }
    const filesystem::path motionPath(options.input_motion_file);
    switch (parseCharacterFormat(motionPath)) {
      case CharacterFormat::Gltf:
        return loadGltfMotionScene(motionPath, options);
      case CharacterFormat::Fbx:
        return loadFbxMotionScene(motionPath, options);
      case CharacterFormat::Usd:
      case CharacterFormat::Unknown:
        break;
    }
    MT_THROW_IF(
        selectsCharacter,
        "Cannot read characters from {}: unknown motion file format",
        motionPath.string());
    return std::nullopt;
  }

  const filesystem::path modelPath(options.input_model_file);
  const CharacterFormat modelFormat = parseCharacterFormat(modelPath);
  if (modelFormat != CharacterFormat::Gltf && modelFormat != CharacterFormat::Fbx &&
      !selectsCharacter) {
    // A file of another format has one character.
    return std::nullopt;
  }
  auto loaded = loadCharacters(modelPath, options, [&] { return loadModelCharacters(modelPath); });
  if (!loaded.has_value() || !convertsAsScene(loaded->size(), options)) {
    return std::nullopt;
  }
  auto characters =
      selectByName(std::move(*loaded), options.character_name, modelPath, characterName);
  checkOneCharacterOptions(
      options, namesOf(std::span<const Character>(characters), characterName), modelPath);
  if (characters.size() == 1) {
    // As loadFullCharacter() and loadModelCharacter().
    applyOptionalInputs(characters.front(), options);
  }
  Scene scene;
  for (auto& character : characters) {
    scene.characters.push_back({std::move(character), {}, {}});
  }
  if (hasMotion) {
    // With a selected character, the other motion-file characters are left out on purpose.
    addMotionByName(scene, filesystem::path(options.input_motion_file), !selectsCharacter);
  }
  return scene;
}

int findLongestMotion(const std::vector<MatrixXf>& motions) {
  std::vector<Eigen::Index> lengths;
  lengths.reserve(motions.size());
  for (const auto& motion : motions) {
    lengths.push_back(motion.cols());
  }
  return findLongestLength(lengths);
}

MotionData loadGlbMotion(
    const filesystem::path& motionPath,
    Character& character,
    bool hasModel,
    const Options& options) {
  MT_LOGI("Loading motion from glb...");
  MotionData result;
  if (hasModel) {
    std::tie(result.poses, result.offsets, result.fps) =
        loadMotionOnCharacter(motionPath, character);
    return result;
  }

  std::tie(character, result.poses, result.offsets, result.fps) =
      loadCharacterWithMotion(motionPath);
  warnIgnoredInputs(options);
  return result;
}

void initializeCharacterFromFbx(
    Character& character,
    const Character& fbxCharacter,
    const Options& options) {
  character = fbxCharacter;
  if (options.input_params_file.empty()) {
    character.parameterTransform = ParameterTransform::identity(character.skeleton.getJointNames());
  }
  applyOptionalInputs(character, options);
}

Character loadModelCharacter(const Options& options) {
  const filesystem::path modelPath(options.input_model_file);
  if (modelPath.extension() != ".urdf") {
    return loadFullCharacter(
        options.input_model_file, options.input_params_file, options.input_locator_file);
  }

  MT_LOGI("Loading URDF model...");
  Character character = loadUrdfCharacter(modelPath);
  applyOptionalInputs(character, options);
  return character;
}

MatrixXf convertToModelParams(
    const MatrixXf& motion,
    const ParameterTransform& parameterTransform) {
  const size_t nFrames = motion.cols();
  MatrixXf poses;
  poses.setZero(parameterTransform.numAllModelParameters(), nFrames);
  InverseParameterTransform inversePt(parameterTransform);
  for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
    poses.col(iFrame) = inversePt.apply(motion.col(iFrame)).pose.v;
  }
  return poses;
}

MatrixXf convertMotionWithJointMapping(
    const MatrixXf& motion,
    const Skeleton& sourceSkeleton,
    const Character& targetCharacter) {
  const size_t nFrames = motion.cols();
  const size_t nJoints = targetCharacter.skeleton.joints.size();
  MatrixXf poses;
  poses.setZero(targetCharacter.parameterTransform.numAllModelParameters(), nFrames);
  InverseParameterTransform inversePt(targetCharacter.parameterTransform);
  SkeletonState sourceState;
  TransformList transforms(nJoints);

  // Build name mapping: indexMap[target_index] = source_index
  const auto jointNames = targetCharacter.skeleton.getJointNames();
  std::vector<size_t> indexMap(nJoints, kInvalidIndex);
  for (size_t iJoint = 0; iJoint < nJoints; ++iJoint) {
    indexMap[iJoint] = sourceSkeleton.getJointIdByName(jointNames[iJoint]);
  }

  for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
    sourceState.set(motion.col(iFrame), sourceSkeleton);

    for (size_t jJoint = 0; jJoint < nJoints; ++jJoint) {
      if (indexMap[jJoint] == kInvalidIndex) {
        // No correspondence in source; use rest pose
        const Joint& joint = targetCharacter.skeleton.joints[jJoint];
        const Transform local(joint.translationOffset, joint.preRotation);
        const size_t parentIndex = targetCharacter.skeleton.joints[jJoint].parent;
        if (parentIndex == kInvalidIndex) {
          transforms[jJoint] = local;
        } else {
          transforms[jJoint] = transforms[parentIndex] * local;
        }
      } else {
        transforms[jJoint] = sourceState.jointState[indexMap[jJoint]].transform;
      }
    }
    const auto jointParams = skeletonStateToJointParameters(transforms, targetCharacter.skeleton);
    poses.col(iFrame) = inversePt.apply(jointParams).pose.v;
  }
  return poses;
}

MotionData loadFbxMotion(
    const filesystem::path& motionPath,
    Character& character,
    bool hasModel,
    const Options& options) {
  MT_LOGI("Loading motion from fbx...");
  MotionData result;

  auto [c, motions, framerate] =
      loadFbxCharacterWithMotion(motionPath, KeepLocators::Yes, Permissive::No);

  const int motionIndex = findLongestMotion(motions);

  if (!hasModel) {
    initializeCharacterFromFbx(character, c, options);
  }

  if (motionIndex < 0) {
    return result;
  }

  // To account for differences in joint order and rest pose between the source
  // and target skeletons, map global joint transforms by name then convert back
  // to joint parameters based on the target skeleton's bind pose.
  result.poses = convertMotionWithJointMapping(motions.at(motionIndex), c.skeleton, character);
  result.fps = framerate;
  result.offsets = character.parameterTransform.zero();
  return result;
}

} // namespace momentum
