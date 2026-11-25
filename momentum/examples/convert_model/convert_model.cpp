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
#include <momentum/io/bvh/bvh_io.h>
#include <momentum/io/character_io.h>
#include <momentum/io/fbx/fbx_builder.h>
#include <momentum/io/fbx/fbx_io.h>
#include <momentum/io/gltf/gltf_builder.h>
#include <momentum/io/gltf/gltf_io.h>
#include <momentum/io/marker/marker_io.h>

#include <CLI/CLI.hpp>

using namespace momentum;

namespace {

std::shared_ptr<Options> setupOptions(CLI::App& app) {
  auto opt = std::make_shared<Options>();

  // add options and flags
  app.add_option(
         "-m,--model",
         opt->input_model_file,
         "Input model (.fbx/.glb/.gltf/.urdf/.bvh); not required if reading animation from glb/gltf, fbx or bvh. "
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
         "Input motion data file (.glb/.gltf/.fbx/.bvh); without --model, every character of the file "
         "is converted with its motion")
      ->check(CLI::ExistingFile);
  app.add_option(
      "--character",
      opt->character_name,
      "Convert only the character with this name, from the model file and the motion file; "
      "required for --parameters, --locator, --out-locators-local, --out-locators-global and "
      "--out-parameters when the input has several characters");

  app.add_option(
         "-o,--out",
         opt->output_model_file,
         "Output file (.fbx/.glb/.gltf/.bvh); a .bvh file holds one character's skeleton and motion")
      ->required();

  app.add_option(
      "--out-locators-local",
      opt->output_locator_local,
      "(Deprecated: use export_rig) Output a locator file (.locators) in local space for "
      "transferring between identities");
  app.add_option(
      "--out-locators-global",
      opt->output_locator_global,
      "(Deprecated: use export_rig) Output a locator file (.locators) in global space for "
      "authoring a template");
  app.add_option(
      "--out-parameters",
      opt->output_params_file,
      "(Deprecated: use export_rig) Output a model parameter file (.model) containing the "
      "parameter transform and limits");
  app.add_option("--fbx-namespace", opt->fbx_namespace, "Namespace in output fbx file");

  app.add_flag(
      "--save-markers", opt->save_markers, "Save marker data from input motion file in the output");
  app.add_flag(
      "-c,--character-mesh",
      opt->character_mesh_save,
      "(FBX Output file only) Saves the Character Mesh to the output file.");

  return opt;
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

// Saves the characters with their motion and the marker sequence to an FBX output file.
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

// Saves the characters with their motion and the marker sequence to a glTF/GLB output file.
void saveGltfScene(
    const std::string& outputFile,
    const Scene& scene,
    const MarkerSequence& markerSequence,
    bool hasMotion) {
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
    if (!hasMotion || (scene.characters.size() > 1 && !hasOwnMotion)) {
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

// Saves the one character of the scene with its motion to a BVH output file, which stores only a
// skeleton and its motion: saveBvhCharacterWithMotion() warns about the other data of the
// character.
void saveBvhScene(const std::string& outputFile, const Scene& scene, bool saveMarkers) {
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
    saveBvhCharacter(outputFile, character);
    return;
  }
  saveBvhCharacterWithMotion(
      outputFile,
      character,
      {character.parameterTransform.name, entry.poses},
      scene.fps,
      entry.offsets.v);
}

// The marker sequence of the motion file when --save-markers is given.
MarkerSequence loadRequestedMarkers(const Options& options) {
  if (!options.save_markers) {
    return {};
  }
  return loadMarkerData(options.input_motion_file);
}

// Warns that the rig outputs of convert_model moved to export_rig.
void warnAboutDeprecatedRigOutputs(const Options& options) {
  if (!hasRigOutputs(options)) {
    return;
  }
  MT_LOGW(
      "--out-locators-local, --out-locators-global and --out-parameters of convert_model are "
      "deprecated: use export_rig");
}

} // namespace

int main(int argc, char** argv) {
  try {
    const std::string appName(argv[0]);
    CLI::App app(appName);
    auto options = setupOptions(app);
    CLI11_PARSE(app, argc, argv);
    warnAboutDeprecatedRigOutputs(*options);

    // Validate output file extension
    const CharacterFormat outputFormat = parseCharacterFormat(options->output_model_file);
    const bool bvhOutput = isBvhFile(options->output_model_file);
    if (!bvhOutput && outputFormat != CharacterFormat::Fbx &&
        outputFormat != CharacterFormat::Gltf) {
      MT_LOGE("Unknown output file format: {}", options->output_model_file);
      return EXIT_FAILURE;
    }

    const Scene scene = loadScene(*options);
    if (bvhOutput) {
      saveBvhScene(options->output_model_file, scene, options->save_markers);
    } else if (outputFormat == CharacterFormat::Fbx) {
      saveFbxScene(options->output_model_file, scene, loadRequestedMarkers(*options), *options);
    } else {
      saveGltfScene(
          options->output_model_file,
          scene,
          loadRequestedMarkers(*options),
          !options->input_motion_file.empty());
    }
    saveRigOutputs(*options, scene);
  } catch (std::exception& e) {
    MT_LOGE("Failed to convert model. Error: {}", e.what());
    return EXIT_FAILURE;
  } catch (...) {
    MT_LOGE("Unknown error encountered.");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
