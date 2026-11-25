/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define DEFAULT_LOG_CHANNEL "convert_model"

#include "convert_model_helpers.h"

#include <momentum/character/character.h>
#include <momentum/common/exception.h>
#include <momentum/common/log.h>
#include <momentum/io/character_io.h>
#include <momentum/io/fbx/fbx_builder.h>
#include <momentum/io/fbx/fbx_io.h>
#include <momentum/io/gltf/gltf_builder.h>
#include <momentum/io/gltf/gltf_io.h>
#include <momentum/io/marker/marker_io.h>
#include <momentum/io/skeleton/locator_io.h>
#include <momentum/io/skeleton/parameter_transform_io.h>

#include <CLI/CLI.hpp>

#include <fstream>

using namespace momentum;

namespace {

std::shared_ptr<Options> setupOptions(CLI::App& app) {
  auto opt = std::make_shared<Options>();

  // add options and flags
  app.add_option(
         "-m,--model",
         opt->input_model_file,
         "Input model (.fbx/.glb/.gltf/.urdf); not required if reading animation from glb/gltf or fbx. "
         "Every character of the file is converted, unless --character selects one. With several "
         "characters or with --character, each character gets the motion of the motion-file "
         "character with the same name")
      ->check(CLI::ExistingFile);
  app.add_option("-p,--parameters", opt->input_params_file, "Input model parameter file (.model)")
      ->check(CLI::ExistingFile);
  app.add_option("-l,--locator", opt->input_locator_file, "Input locator file (.locators)")
      ->check(CLI::ExistingFile);

  app.add_option(
         "-d,--motion",
         opt->input_motion_file,
         "Input motion data file (.glb/.gltf/.fbx); without --model, every character of the file "
         "is converted with its motion")
      ->check(CLI::ExistingFile);
  app.add_option(
      "--character",
      opt->character_name,
      "Convert only the character with this name, from the model file and the motion file; "
      "required for --parameters, --locator, --out-locators-local, --out-locators-global and "
      "--out-parameters when the input has several characters");

  app.add_option("-o,--out", opt->output_model_file, "Output file (.fbx/.glb/.gltf)")->required();

  app.add_option(
      "--out-locators-local",
      opt->output_locator_local,
      "Output a locator file (.locators) in local space for transferring between identities");
  app.add_option(
      "--out-locators-global",
      opt->output_locator_global,
      "Output a locator file (.locators) in global space for authoring a template");
  app.add_option(
      "--out-parameters",
      opt->output_params_file,
      "Output a model parameter file (.model) containing the parameter transform and limits");
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

// Loads marker sequence data from a motion file.
MarkerSequence loadMarkerData(const std::string& motionFile) {
  auto markerData = loadMarkers(motionFile);
  if (!markerData.empty()) {
    return std::move(markerData[0]);
  }
  return {};
}

// The FBX save options that the command-line options select.
FileSaveOptions fbxSaveOptions(const Options& options) {
  FileSaveOptions fbxOptions;
  fbxOptions.mesh = options.character_mesh_save;
  fbxOptions.fbxNamespace = options.fbx_namespace;
  return fbxOptions;
}

// Saves the character and motion data to an FBX output file.
void saveFbxOutput(
    const std::string& outputFile,
    const Character& character,
    const MotionData& motionData,
    const MarkerSequence& markerSequence,
    const Options& options) {
  MT_LOGI("Saving fbx file...");
  saveFbx(
      outputFile,
      character,
      motionData.poses,
      motionData.offsets.v,
      motionData.fps,
      markerSequence.frames,
      fbxSaveOptions(options));
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
        toIdentityParameters(character, motionData.offsets),
        markerSequence.frames);
  } else {
    saveGltfCharacter(outputFile, character);
  }
}

// Saves the characters with their motion and the marker sequence to an FBX output file, as
// saveFbxOutput() saves a single character.
void saveFbxScene(
    const std::string& outputFile,
    const Scene& scene,
    const MarkerSequence& markerSequence,
    const Options& options) {
  MT_LOGI("Saving fbx file...");
  const FileSaveOptions fbxOptions = fbxSaveOptions(options);
  FbxBuilder builder;
  for (const auto& entry : scene.characters) {
    const std::string name = builder.addCharacter(entry.character, fbxOptions);
    builder.addMotion(entry.character, scene.fps, entry.poses, entry.offsets.v, name);
  }
  builder.addMarkerSequence(scene.fps, markerSequence.frames);
  builder.save(outputFile);
}

// Saves the characters with their motion and the marker sequence to a glTF/GLB output file, as
// saveGltfOutput() saves a single character.
void saveGltfScene(
    const std::string& outputFile,
    const Scene& scene,
    const MarkerSequence& markerSequence,
    bool hasMotion) {
  MT_LOGI("Saving gltf/glb file...");
  GltfBuilder builder;
  for (const auto& entry : scene.characters) {
    const std::string name = builder.addCharacter(entry.character);
    // A character without motion, such as a model character that no motion-file character
    // matches, gets no animation.
    if (!hasMotion || (entry.poses.cols() == 0 && entry.offsets.size() == 0)) {
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
  builder.save(outputFile);
}

// The marker sequence of the motion file when --save-markers is given.
MarkerSequence loadRequestedMarkers(const Options& options) {
  if (!options.save_markers) {
    return {};
  }
  return loadMarkerData(options.input_motion_file);
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

// Saves the character's parameter transform and limits to a .model file if an
// output path is specified.
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

} // namespace

int main(int argc, char** argv) {
  try {
    const std::string appName(argv[0]);
    CLI::App app(appName);
    auto options = setupOptions(app);
    CLI11_PARSE(app, argc, argv);

    // Validate output file extension
    const CharacterFormat outputFormat = parseCharacterFormat(options->output_model_file);
    if (outputFormat != CharacterFormat::Fbx && outputFormat != CharacterFormat::Gltf) {
      MT_LOGE("Unknown output file format: {}", options->output_model_file);
      return EXIT_FAILURE;
    }

    if (const auto scene = loadScene(*options)) {
      const MarkerSequence markerSequence = loadRequestedMarkers(*options);
      if (outputFormat == CharacterFormat::Fbx) {
        saveFbxScene(options->output_model_file, *scene, markerSequence, *options);
      } else {
        saveGltfScene(
            options->output_model_file,
            *scene,
            markerSequence,
            !options->input_motion_file.empty());
      }
      // loadScene() rejects these outputs for several characters.
      if (scene->characters.size() == 1) {
        saveLocatorFiles(*options, scene->characters.front().character);
        saveParameterFile(*options, scene->characters.front().character);
      }
      return EXIT_SUCCESS;
    }

    // Load character model
    const bool hasModel = !options->input_model_file.empty();
    Character character;
    if (hasModel) {
      character = loadModelCharacter(*options);
    }

    // Load motion data
    const bool hasMotion = !options->input_motion_file.empty();
    const auto motionData = loadMotion(*options, character, hasModel);

    // Load markers if requested
    const MarkerSequence markerSequence = loadRequestedMarkers(*options);

    // Save output file
    if (outputFormat == CharacterFormat::Fbx) {
      saveFbxOutput(options->output_model_file, character, motionData, markerSequence, *options);
    } else {
      saveGltfOutput(options->output_model_file, character, motionData, markerSequence, hasMotion);
    }

    // Save locator files
    saveLocatorFiles(*options, character);

    // Save model parameter file
    saveParameterFile(*options, character);
  } catch (std::exception& e) {
    MT_LOGE("Failed to convert model. Error: {}", e.what());
    return EXIT_FAILURE;
  } catch (...) {
    MT_LOGE("Unknown error encountered.");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
