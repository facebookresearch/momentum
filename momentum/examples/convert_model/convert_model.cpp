/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define DEFAULT_LOG_CHANNEL "convert_model"

#include "convert_model_helpers.h"

#include <momentum/character/character.h>
#include <momentum/common/log.h>
#include <momentum/io/character_io.h>
#include <momentum/io/fbx/fbx_io.h>
#include <momentum/io/gltf/gltf_io.h>
#include <momentum/io/marker/marker_io.h>
#include <momentum/io/skeleton/locator_io.h>

#include <CLI/CLI.hpp>

using namespace momentum;

namespace {

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
