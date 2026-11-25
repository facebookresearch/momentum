/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define DEFAULT_LOG_CHANNEL "apply_motion"

#include "convert_model_helpers.h"

#include <momentum/common/log.h>

#include <CLI/CLI.hpp>

#include <memory>

using namespace momentum;

namespace {

// apply_motion reads the model and motion inputs and writes the output of the convert_model
// options.
std::shared_ptr<Options> setupOptions(CLI::App& app) {
  auto opt = std::make_shared<Options>();

  app.add_option(
         "-m,--model",
         opt->input_model_file,
         "Input character file (.fbx/.glb/.gltf/.urdf/.bvh) that gets the motion. With several "
         "characters or with --character, each character gets the motion of the motion-file "
         "character with the same name")
      ->required()
      ->check(CLI::ExistingFile);
  app.add_option("-p,--parameters", opt->input_params_file, "Input model parameter file (.model)")
      ->check(CLI::ExistingFile);
  app.add_option("-l,--locator", opt->input_locator_file, "Input locator file (.locators)")
      ->check(CLI::ExistingFile);
  app.add_option(
         "-d,--motion", opt->input_motion_file, "Input motion data file (.glb/.gltf/.fbx/.bvh)")
      ->required()
      ->check(CLI::ExistingFile);
  app.add_option(
      "--character",
      opt->character_name,
      "Apply the motion only to the character with this name, from the model file and the motion "
      "file; required for --parameters and --locator when the model file has several characters");

  app.add_option(
         "-o,--out",
         opt->output_model_file,
         "Output file (.fbx/.glb/.gltf/.bvh); a .bvh file holds one character's skeleton and motion")
      ->required();
  app.add_option("--fbx-namespace", opt->fbx_namespace, "Namespace in output fbx file");
  app.add_flag(
      "--save-markers", opt->save_markers, "Save marker data from input motion file in the output");
  app.add_flag(
      "-c,--character-mesh",
      opt->character_mesh_save,
      "(FBX Output file only) Saves the Character Mesh to the output file.");

  return opt;
}

} // namespace

int main(int argc, char** argv) {
  try {
    CLI::App app(
        "Applies the motion of a motion file to the characters of a model file, and saves them");
    auto options = setupOptions(app);
    CLI11_PARSE(app, argc, argv);

    if (!isSceneOutputFile(options->output_model_file)) {
      MT_LOGE("Unknown output file format: {}", options->output_model_file);
      return EXIT_FAILURE;
    }
    if (!isMotionInputFile(options->input_motion_file)) {
      MT_LOGE("Unknown motion file format: {}", options->input_motion_file);
      return EXIT_FAILURE;
    }

    const Scene scene = loadScene(*options);
    saveScene(options->output_model_file, scene, saveSceneOptions(*options));
  } catch (std::exception& e) {
    MT_LOGE("Failed to apply the motion. Error: {}", e.what());
    return EXIT_FAILURE;
  } catch (...) {
    MT_LOGE("Unknown error encountered.");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
