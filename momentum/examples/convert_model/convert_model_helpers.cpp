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
#include <momentum/io/bvh/bvh_io.h>
#include <momentum/io/character_io.h>
#include <momentum/io/fbx/fbx_builder.h>
#include <momentum/io/fbx/fbx_io.h>
#include <momentum/io/gltf/gltf_builder.h>
#include <momentum/io/gltf/gltf_io.h>
#include <momentum/io/marker/marker_io.h>
#include <momentum/io/skeleton/locator_io.h>
#include <momentum/io/skeleton/parameter_transform_io.h>
#include <momentum/io/skeleton/parameters_io.h>
#include <momentum/io/urdf/urdf_io.h>

#include <algorithm>
#include <cctype>
#include <fstream>
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

// The extension of `path` in lower case, such as ".bvh".
std::string lowerCaseExtension(const filesystem::path& path) {
  std::string extension = path.extension().string();
  std::ranges::transform(extension, extension.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return extension;
}

// The joint-parameter motion of a character with the skeleton `source`, mapped by joint name onto
// `target` (see convertMotionWithJointMapping()). The mapping subtracts the offsets of the
// parameter transform from the joint parameters, and the identity offsets add them back.
momentum::MotionData mappedMotion(
    const momentum::MatrixXf& jointMotion,
    const momentum::Skeleton& source,
    const momentum::Character& target,
    float fps) {
  momentum::MotionData result;
  result.poses = momentum::convertMotionWithJointMapping(jointMotion, source, target);
  result.offsets = target.parameterTransform.zero();
  result.fps = fps;
  return result;
}

// Whether the model characters come from an FBX file without a parameter file. An FBX file stores
// no parameter transform, so these characters get the identity transform, and the motion of every
// motion file is mapped onto them through the joint transforms.
bool isFbxModelWithoutParameters(const momentum::Options& options) {
  return !options.input_model_file.empty() && options.input_params_file.empty() &&
      momentum::parseCharacterFormat(options.input_model_file) == momentum::CharacterFormat::Fbx;
}

// The joint parameters (numJointParameters x frames) of glTF motion: the model parameters `motion`
// of `character` with the identity offsets, as the glTF writer computes the joints.
momentum::MatrixXf gltfJointMotion(
    const momentum::Character& character,
    const momentum::MatrixXf& motion,
    const momentum::JointParameters& identity) {
  momentum::MatrixXf result = character.parameterTransform.transform * motion;
  if (identity.size() > 0) {
    result.colwise() += identity.v;
  }
  return result;
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

// The motion of a glTF motion-file character for `target`: mapped through the joint transforms
// when `byJoints` and the source has frames, as FBX motion is; otherwise the model parameters and
// the identity are mapped by name, as loadMotionOnCharacter() maps them.
void assignGltfMotion(
    momentum::SceneCharacter& target,
    const momentum::CharacterMotion& source,
    bool byJoints) {
  if (byJoints && source.motion.cols() > 0) {
    target.poses = momentum::convertMotionWithJointMapping(
        gltfJointMotion(source.character, source.motion, source.identity),
        source.character.skeleton,
        target.character);
    target.offsets = target.character.parameterTransform.zero();
    return;
  }
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
// name. Warns about motion-file characters that no character uses if `warnUnused`. glTF motion is
// mapped through the joint transforms if `gltfByJoints` (see assignGltfMotion()).
void addMotionByName(
    momentum::Scene& scene,
    const filesystem::path& motionPath,
    bool warnUnused,
    bool gltfByJoints) {
  if (momentum::isBvhFile(motionPath)) {
    // A BVH file holds one character without a name.
    MT_THROW_IF(
        scene.characters.size() != 1,
        "{} holds one character: select the model character that gets its motion with --character",
        motionPath.string());
    auto& target = scene.characters.front();
    const auto [bvhCharacter, bvhMotion, fps] = momentum::loadBvhCharacterWithMotion(motionPath);
    const auto& jointMotion = std::get<1>(bvhMotion);
    if (jointMotion.cols() == 0) {
      return;
    }
    auto motion = mappedMotion(jointMotion, bvhCharacter.skeleton, target.character, fps);
    target.poses = std::move(motion.poses);
    target.offsets = std::move(motion.offsets);
    scene.fps = motion.fps;
    return;
  }
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
          [gltfByJoints](
              momentum::SceneCharacter& target, const momentum::CharacterMotion& source) {
            assignGltfMotion(target, source, gltfByJoints);
          });
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
    momentum::initializeCharacterFromMotionFile(entry.character, fbxCharacter, options);
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

namespace {

// Saves the locators of the character to options.output_locator_local (local space) and
// options.output_locator_global (global space), for each of them that is not empty.
void saveLocatorFiles(const Options& options, const Character& character) {
  if (!options.output_locator_local.empty()) {
    saveLocators(
        options.output_locator_local, character.locators, character.skeleton, LocatorSpace::Local);
  }
  if (!options.output_locator_global.empty()) {
    saveLocators(
        options.output_locator_global,
        character.locators,
        character.skeleton,
        LocatorSpace::Global);
  }
}

// Saves the parameter transform and parameter limits of the character to a .model file at
// options.output_params_file, when it is not empty.
void saveParameterFile(const Options& options, const Character& character) {
  if (options.output_params_file.empty()) {
    return;
  }

  MT_LOGI("Saving model parameter file...");
  std::ofstream ofs(options.output_params_file);
  MT_THROW_IF(
      !ofs.is_open(),
      "Cannot open output parameter file for writing: {}",
      options.output_params_file);
  ofs << writeModelDefinition(
      character.skeleton, character.parameterTransform, character.parameterLimits);
}

// The motion of a BVH file for one character. Without a model file, the character is the BVH
// character, which keeps the motion as it is, unless a parameter file changes its parameters;
// otherwise the motion is mapped onto the character by joint name, as loadFbxMotion() maps it.
MotionData loadBvhMotion(
    const filesystem::path& motionPath,
    Character& character,
    bool hasModel,
    const Options& options) {
  MT_LOGI("Loading motion from bvh...");
  // The BVH character has an identity parameter transform, so its motion holds joint parameters.
  auto [bvhCharacter, bvhMotion, fps] = loadBvhCharacterWithMotion(motionPath);
  if (!hasModel) {
    initializeCharacterFromMotionFile(character, bvhCharacter, options);
  }
  MatrixXf& jointMotion = std::get<1>(bvhMotion);
  if (jointMotion.cols() == 0) {
    return {};
  }
  if (!hasModel && options.input_params_file.empty()) {
    return {
        .poses = std::move(jointMotion),
        .offsets = character.parameterTransform.zero(),
        .fps = fps};
  }
  return mappedMotion(jointMotion, bvhCharacter.skeleton, character, fps);
}

// The motion of the motion file for one character, read with the loader of its format. Without a
// model file, the character is read from the motion file too.
MotionData loadMotion(const Options& options, Character& character, bool hasModel) {
  if (options.input_motion_file.empty()) {
    return {};
  }

  const auto motionPath = filesystem::path(options.input_motion_file);
  if (isBvhFile(motionPath)) {
    return loadBvhMotion(motionPath, character, hasModel, options);
  }
  switch (parseCharacterFormat(motionPath)) {
    case CharacterFormat::Gltf:
      return loadGlbMotion(motionPath, character, hasModel, options);
    case CharacterFormat::Fbx:
      return loadFbxMotion(motionPath, character, hasModel, options);
    case CharacterFormat::Usd:
    case CharacterFormat::Unknown:
      break;
  }

  MT_LOGW("Unknown motion file format: {}. Exporting without motion.", options.input_motion_file);
  return {};
}

// The input converted as one character (see loadScene()).
Scene loadOneCharacterScene(const Options& options) {
  const bool hasModel = !options.input_model_file.empty();
  SceneCharacter entry;
  if (hasModel) {
    entry.character = loadModelCharacter(options);
  }
  MotionData motion = loadMotion(options, entry.character, hasModel);
  entry.poses = std::move(motion.poses);
  entry.offsets = std::move(motion.offsets);
  Scene scene;
  scene.fps = motion.fps;
  scene.characters.push_back(std::move(entry));
  return scene;
}

// The characters when the input has several or one is selected by name (see loadScene()); nullopt
// when the input converts as one character.
std::optional<Scene> loadMultiCharacterScene(const Options& options) {
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
        selectsCharacter && isBvhFile(motionPath),
        "--character needs a model file with a BVH motion file: {} holds one character",
        motionPath.string());
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
    addMotionByName(
        scene,
        filesystem::path(options.input_motion_file),
        !selectsCharacter,
        isFbxModelWithoutParameters(options));
  }
  return scene;
}

// The first marker sequence of `markerFile`; no frames if it has none.
MarkerSequence loadMarkerData(const std::string& markerFile) {
  auto markerData = loadMarkers(markerFile);
  if (!markerData.empty()) {
    return std::move(markerData[0]);
  }
  return {};
}

// The marker sequence that the options ask for; no frames without options.saveMarkers.
MarkerSequence loadRequestedMarkers(const SaveSceneOptions& options) {
  if (!options.saveMarkers) {
    return {};
  }
  return loadMarkerData(options.markerFile);
}

// The identity offsets of `character` for the glTF writer.
IdentityParameters toIdentityParameters(
    const Character& character,
    const JointParameters& offsets) {
  // Empty offsets mean "the loaded motion has no per-joint identity vector"; passing the
  // skeleton's joint names paired with an empty identity vector would violate
  // mapIdentityToCharacter's size precondition. Use a fully empty IdentityParameters so the
  // glTF writer sees "no identity data" rather than a mismatched tuple.
  return offsets.size() == 0 ? IdentityParameters{}
                             : IdentityParameters{character.skeleton.getJointNames(), offsets};
}

// Saves the characters with their motion and the marker sequence to an FBX file.
void saveFbxScene(
    const filesystem::path& path,
    const Scene& scene,
    const MarkerSequence& markerSequence,
    const SaveSceneOptions& options) {
  MT_LOGI("Saving fbx file...");
  FileSaveOptions fbxOptions;
  fbxOptions.mesh = options.saveMeshes;
  fbxOptions.fbxNamespace = options.fbxNamespace;
  FbxBuilder builder;
  for (const auto& entry : scene.characters) {
    const std::string name = builder.addCharacter(entry.character, fbxOptions);
    builder.addMotion(entry.character, scene.fps, entry.poses, entry.offsets.v, name);
  }
  builder.addMarkerSequence(scene.fps, markerSequence.frames);
  builder.save(path);
}

// Saves the characters with their motion and the marker sequence to a glTF/GLB file.
void saveGltfScene(
    const filesystem::path& path,
    const Scene& scene,
    const MarkerSequence& markerSequence) {
  MT_LOGI("Saving gltf/glb file...");
  GltfBuilder builder;
  for (const auto& entry : scene.characters) {
    // As saveGltfCharacter(), an empty character (from no input file) adds nothing.
    if (entry.character.skeleton.joints.empty() && entry.character.mesh == nullptr) {
      continue;
    }
    const std::string name = builder.addCharacter(entry.character);
    // In a scene with several characters, a character without motion, such as a model character
    // that no motion-file character matches, gets no animation. One character gets the motion of
    // the motion file even without frames, as saveGltfCharacter() adds it.
    const bool hasOwnMotion = entry.poses.cols() > 0 || entry.offsets.size() > 0;
    if (!scene.hasMotionFile || (scene.characters.size() > 1 && !hasOwnMotion)) {
      continue;
    }
    const auto& character = entry.character;
    builder.addMotion(
        character,
        scene.fps,
        {character.parameterTransform.name, entry.poses},
        toIdentityParameters(character, entry.offsets),
        /*addExtensions=*/true,
        /*customName=*/"default",
        /*timestamps=*/{},
        name);
  }
  if (!markerSequence.frames.empty()) {
    builder.addMarkerSequence(scene.fps, markerSequence.frames);
  }
  builder.save(path);
}

// Saves the one character of the scene with its motion to a BVH file, which stores only a
// skeleton and its motion: saveBvhCharacterWithMotion() warns about the other data of the
// character.
void saveBvhScene(const filesystem::path& path, const Scene& scene, bool saveMarkers) {
  MT_LOGI("Saving bvh file...");
  MT_THROW_IF(
      scene.characters.size() != 1,
      "A BVH file holds one character, but the input has {}: select one with --character",
      scene.characters.size());
  if (saveMarkers) {
    MT_LOGW("A BVH file cannot store markers: --save-markers is ignored");
  }
  const auto& entry = scene.characters.front();
  const Character& character = entry.character;
  if (entry.poses.cols() == 0) {
    saveBvhCharacter(path, character);
    return;
  }
  saveBvhCharacterWithMotion(
      path,
      character,
      {character.parameterTransform.name, entry.poses},
      scene.fps,
      entry.offsets.v);
}

// The options with which loadScene() reads the input. Without a model file, a glTF motion file is
// also the model file when a parameter or locator file is given: these files then apply to its
// characters, as they apply to the characters of FBX and BVH motion files (see
// initializeCharacterFromMotionFile()), and its motion moves to the new parameter transform by
// parameter name.
Options sceneInputOptions(const Options& options) {
  Options result = options;
  const bool hasRigInputs =
      !options.input_params_file.empty() || !options.input_locator_file.empty();
  if (options.input_model_file.empty() && hasRigInputs &&
      parseCharacterFormat(options.input_motion_file) == CharacterFormat::Gltf) {
    result.input_model_file = options.input_motion_file;
  }
  return result;
}

} // namespace

Scene loadScene(const Options& options) {
  const Options input = sceneInputOptions(options);
  std::optional<Scene> scene = loadMultiCharacterScene(input);
  if (!scene) {
    scene = loadOneCharacterScene(input);
  }
  scene->hasMotionFile = !options.input_motion_file.empty();
  return std::move(*scene);
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
  if (hasModel && isFbxModelWithoutParameters(options)) {
    auto [source, motion, identity, fps] = loadCharacterWithMotion(motionPath);
    if (motion.cols() > 0) {
      return mappedMotion(
          gltfJointMotion(source, motion, identity), source.skeleton, character, fps);
    }
  }
  if (hasModel) {
    std::tie(result.poses, result.offsets, result.fps) =
        loadMotionOnCharacter(motionPath, character);
    return result;
  }

  std::tie(character, result.poses, result.offsets, result.fps) =
      loadCharacterWithMotion(motionPath);
  return result;
}

void initializeCharacterFromMotionFile(
    Character& character,
    const Character& motionCharacter,
    const Options& options) {
  character = motionCharacter;
  if (options.input_params_file.empty()) {
    character.parameterTransform = ParameterTransform::identity(character.skeleton.getJointNames());
  }
  applyOptionalInputs(character, options);
}

bool hasRigOutputs(const Options& options) {
  return !options.output_locator_local.empty() || !options.output_locator_global.empty() ||
      !options.output_params_file.empty();
}

void saveRigOutputs(const Options& options, const Scene& scene) {
  if (!hasRigOutputs(options)) {
    return;
  }
  MT_THROW_IF(
      scene.characters.size() != 1,
      "The locator and parameter outputs apply to one character, but the input has {}",
      scene.characters.size());
  const Character& character = scene.characters.front().character;
  saveLocatorFiles(options, character);
  saveParameterFile(options, character);
}

bool isBvhFile(const filesystem::path& path) {
  return lowerCaseExtension(path) == ".bvh";
}

bool isSceneOutputFile(const filesystem::path& path) {
  const CharacterFormat format = parseCharacterFormat(path);
  return isBvhFile(path) || format == CharacterFormat::Fbx || format == CharacterFormat::Gltf;
}

bool isMotionInputFile(const filesystem::path& path) {
  const CharacterFormat format = parseCharacterFormat(path);
  return isBvhFile(path) || format == CharacterFormat::Gltf || format == CharacterFormat::Fbx;
}

SaveSceneOptions saveSceneOptions(const Options& options) {
  SaveSceneOptions result;
  result.fbxNamespace = options.fbx_namespace;
  result.saveMeshes = options.character_mesh_save;
  result.saveMarkers = options.save_markers;
  result.markerFile = options.input_motion_file;
  return result;
}

void saveScene(const filesystem::path& path, const Scene& scene, const SaveSceneOptions& options) {
  if (isBvhFile(path)) {
    saveBvhScene(path, scene, options.saveMarkers);
    return;
  }
  const CharacterFormat format = parseCharacterFormat(path);
  MT_THROW_IF(
      format != CharacterFormat::Fbx && format != CharacterFormat::Gltf,
      "Unknown output file format: {}",
      path.string());
  if (format == CharacterFormat::Fbx) {
    saveFbxScene(path, scene, loadRequestedMarkers(options), options);
  } else {
    saveGltfScene(path, scene, loadRequestedMarkers(options));
  }
}

Character loadModelCharacter(const Options& options) {
  const filesystem::path modelPath(options.input_model_file);
  const std::string extension = lowerCaseExtension(modelPath);
  if (isFbxModelWithoutParameters(options)) {
    Character character;
    initializeCharacterFromMotionFile(
        character, loadFullCharacter(options.input_model_file), options);
    return character;
  }
  if (extension != ".bvh" && extension != ".urdf") {
    return loadFullCharacter(
        options.input_model_file, options.input_params_file, options.input_locator_file);
  }

  const bool bvh = extension == ".bvh";
  MT_LOGI("Loading {} model...", bvh ? "BVH" : "URDF");
  Character character = bvh ? loadBvhCharacter(modelPath) : loadUrdfCharacter(modelPath);
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
  auto [c, motions, framerate] =
      loadFbxCharacterWithMotion(motionPath, KeepLocators::Yes, Permissive::No);

  const int motionIndex = findLongestMotion(motions);

  if (!hasModel) {
    initializeCharacterFromMotionFile(character, c, options);
  }

  if (motionIndex < 0) {
    return {};
  }

  // To account for differences in joint order and rest pose between the source
  // and target skeletons, map global joint transforms by name then convert back
  // to joint parameters based on the target skeleton's bind pose.
  return mappedMotion(motions.at(motionIndex), c.skeleton, character, framerate);
}

} // namespace momentum
