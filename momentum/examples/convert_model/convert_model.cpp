/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define DEFAULT_LOG_CHANNEL "convert_model"

#include <momentum/character/character.h>
#include <momentum/character/inverse_parameter_transform.h>
#include <momentum/common/log.h>
#include <momentum/io/character_io.h>
#include <momentum/io/fbx/fbx_io.h>
#include <momentum/io/gltf/gltf_io.h>
#include <momentum/io/marker/marker_io.h>
#include <momentum/io/skeleton/locator_io.h>
#include <momentum/io/skeleton/parameter_transform_io.h>
#include <momentum/io/skeleton/parameters_io.h>

#include <CLI/CLI.hpp>

using namespace momentum;

namespace {

struct Options {
  std::string input_model_file;
  std::string input_params_file;
  std::string input_locator_file;
  std::string input_motion_file;
  std::string output_model_file;
  std::string output_locator_local;
  std::string output_locator_global;
  std::string fbx_namespace;
  bool save_markers = false;
  bool character_mesh_save = false;
};

// Bundles the result of loading motion data from a file.
struct MotionData {
  MatrixXf poses;
  JointParameters offsets;
  float fps = 120.0f;
};

std::shared_ptr<Options> setupOptions(CLI::App& app) {
  auto opt = std::make_shared<Options>();

  // add options and flags
  app.add_option(
         "-m,--model",
         opt->input_model_file,
         "Input model (.fbx/.glb); not required if reading animation from glb or fbx")
      ->check(CLI::ExistingFile);
  app.add_option("-p,--parameters", opt->input_params_file, "Input model parameter file (.model)")
      ->check(CLI::ExistingFile);
  app.add_option("-l,--locator", opt->input_locator_file, "Input locator file (.locators)")
      ->check(CLI::ExistingFile);

  app.add_option("-d,--motion", opt->input_motion_file, "Input motion data file (.glb/.fbx)")
      ->check(CLI::ExistingFile);

  app.add_option("-o,--out", opt->output_model_file, "Output file (.fbx/.glb)")->required();

  app.add_option(
      "--out-locator-local",
      opt->output_locator_local,
      "Output a locator file (.locators) in local space for transferring between identities");
  app.add_option(
      "--out-locator-global",
      opt->output_locator_global,
      "Output a locator file (.locators) in global space for authoring a template");
  app.add_option("--fbx-namespace", opt->fbx_namespace, "Namespace in output fbx file");

  app.add_flag(
      "--save-markers", opt->save_markers, "Save marker data from input motion file in the output");
  app.add_flag(
      "-c,--character-mesh",
      opt->character_mesh_save,
      "(FBX Output file only) Saves the Character Mesh to the output file.");

  return opt;
}

// Loads motion data from a GLB file. If no separate model was provided,
// the character is also loaded from the GLB file.
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
  if (!options.input_params_file.empty()) {
    MT_LOGW("Ignoring input parameter transform {}.", options.input_params_file);
  }
  if (!options.input_locator_file.empty()) {
    MT_LOGW("Ignoring input locators {}.", options.input_locator_file);
  }
  return result;
}

// Finds the index of the longest motion in a list of motions.
// Returns -1 if no valid motion is found.
int findLongestMotion(const std::vector<MatrixXf>& motions) {
  if (motions.empty() || (motions.size() == 1 && motions.at(0).cols() == 0)) {
    MT_LOGW("No motion loaded from file");
    return -1;
  }

  int motionIndex = -1;
  size_t nFrames = 0;
  for (size_t iMotion = 0; iMotion < motions.size(); ++iMotion) {
    const size_t length = motions.at(iMotion).cols();
    if (length > nFrames) {
      nFrames = length;
      motionIndex = static_cast<int>(iMotion);
    }
  }
  if (nFrames > 0 && motions.size() > 1) {
    MT_LOGW("More than one motion found; only taking the longest one");
  }
  return motionIndex;
}

// Initializes the character from FBX data when no separate model was provided.
// Applies parameter transform and locators if specified in the options.
void initializeCharacterFromFbx(
    Character& character,
    const Character& fbxCharacter,
    const Options& options) {
  character = fbxCharacter;
  if (!options.input_params_file.empty()) {
    auto def = loadMomentumModel(options.input_params_file);
    loadParameters(def, character);
  } else {
    character.parameterTransform = ParameterTransform::identity(character.skeleton.getJointNames());
  }
  if (!options.input_locator_file.empty()) {
    character.locators =
        loadLocators(options.input_locator_file, character.skeleton, character.parameterTransform);
  }
}

// Converts joint-parameter motion data to model parameters using
// inverse parameter transform. May lose information.
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

// Loads motion data from an FBX file. If no separate model was provided,
// the character is also initialized from the FBX file.
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

  if (c.skeleton.joints.size() != character.skeleton.joints.size()) {
    MT_LOGE("The motion is not on a compatible character");
    return result;
  }

  if (motionIndex < 0) {
    return result;
  }

  if (character.parameterTransform.numAllModelParameters() == motions.at(0).rows()) {
    result.poses = std::move(motions.at(motionIndex));
  } else {
    result.poses = convertToModelParams(motions.at(motionIndex), character.parameterTransform);
  }
  result.fps = framerate;
  result.offsets = character.parameterTransform.zero();
  return result;
}

// Loads motion data from the specified file. Dispatches to format-specific
// loading functions based on the file extension.
MotionData loadMotion(const Options& options, Character& character, bool hasModel) {
  if (options.input_motion_file.empty()) {
    return {};
  }

  const auto motionPath = filesystem::path(options.input_motion_file);
  const auto motionExt = motionPath.extension();

  if (motionExt == ".glb") {
    return loadGlbMotion(motionPath, character, hasModel, options);
  }
  if (motionExt == ".fbx") {
    return loadFbxMotion(motionPath, character, hasModel, options);
  }

  MT_LOGW("Unknown motion file format: {}. Exporting without motion.", options.input_motion_file);
  return {};
}

// Loads marker sequence data from a motion file.
MarkerSequence loadMarkerData(const std::string& motionFile) {
  auto markerData = loadMarkers(motionFile);
  if (!markerData.empty()) {
    return std::move(markerData[0]);
  }
  return {};
}

// Saves the character and motion data to an FBX output file.
void saveFbxOutput(
    const std::string& outputFile,
    const Character& character,
    const MotionData& motionData,
    const MarkerSequence& markerSequence,
    const Options& options) {
  MT_LOGI("Saving fbx file...");
  FileSaveOptions fbxOptions;
  fbxOptions.mesh = options.character_mesh_save;
  fbxOptions.fbxNamespace = options.fbx_namespace;
  saveFbx(
      outputFile,
      character,
      motionData.poses,
      motionData.offsets.v,
      motionData.fps,
      markerSequence.frames,
      fbxOptions);
}

// Saves the character and motion data to a glTF/GLB output file.
void saveGltfOutput(
    const std::string& outputFile,
    const Character& character,
    const MotionData& motionData,
    const MarkerSequence& markerSequence,
    bool hasMotion) {
  MT_LOGI("Saving gltf/glb file...");
  if (hasMotion) {
    saveGltfCharacter(
        outputFile,
        character,
        motionData.fps,
        {character.parameterTransform.name, motionData.poses},
        {character.skeleton.getJointNames(), motionData.offsets},
        markerSequence.frames);
  } else {
    saveGltfCharacter(outputFile, character);
  }
}

// Saves locator files in local and/or global space if output paths are specified.
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

} // namespace

int main(int argc, char** argv) {
  try {
    const std::string appName(argv[0]);
    CLI::App app(appName);
    auto options = setupOptions(app);
    CLI11_PARSE(app, argc, argv);

    // Validate output file extension
    const filesystem::path output(options->output_model_file);
    const auto oextension = output.extension();
    if (oextension != ".fbx" && oextension != ".glb" && oextension != ".gltf") {
      MT_LOGE("Unknown output file format: {}", options->output_model_file);
      return EXIT_FAILURE;
    }

    // Load character model
    const bool hasModel = !options->input_model_file.empty();
    Character character;
    if (hasModel) {
      character = loadFullCharacter(
          options->input_model_file, options->input_params_file, options->input_locator_file);
    }

    // Load motion data
    const bool hasMotion = !options->input_motion_file.empty();
    const auto motionData = loadMotion(*options, character, hasModel);

    // Load markers if requested
    MarkerSequence markerSequence;
    if (options->save_markers) {
      markerSequence = loadMarkerData(options->input_motion_file);
    }

    // Save output file
    if (oextension == ".fbx") {
      saveFbxOutput(options->output_model_file, character, motionData, markerSequence, *options);
    } else if (oextension == ".glb" || oextension == ".gltf") {
      saveGltfOutput(options->output_model_file, character, motionData, markerSequence, hasMotion);
    }

    // Save locator files
    saveLocatorFiles(*options, character);
  } catch (std::exception& e) {
    MT_LOGE("Failed to convert model. Error: {}", e.what());
    return EXIT_FAILURE;
  } catch (...) {
    MT_LOGE("Unknown error encountered.");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
