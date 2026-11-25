/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define DEFAULT_LOG_CHANNEL "convert_model"

#include "convert_model_helpers.h"

#include <momentum/common/log.h>

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
         "Every character of the file is converted, unless --character selects one. (Deprecated "
         "with --motion: use apply_motion) With several characters or with --character, each "
         "character gets the motion of the motion-file character with the same name")
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

// Warns that the rig outputs of convert_model moved to export_rig.
void warnAboutDeprecatedRigOutputs(const Options& options) {
  if (!hasRigOutputs(options)) {
    return;
  }
  MT_LOGW(
      "--out-locators-local, --out-locators-global and --out-parameters of convert_model are "
      "deprecated: use export_rig");
}

// Warns that convert_model --model with --motion moved to apply_motion.
void warnAboutDeprecatedModelWithMotion(const Options& options) {
  if (options.input_model_file.empty() || options.input_motion_file.empty()) {
    return;
  }
  MT_LOGW("--model with --motion of convert_model is deprecated: use apply_motion");
}

} // namespace

int main(int argc, char** argv) {
  try {
    const std::string appName(argv[0]);
    CLI::App app(appName);
    auto options = setupOptions(app);
    CLI11_PARSE(app, argc, argv);
    warnAboutDeprecatedRigOutputs(*options);
    warnAboutDeprecatedModelWithMotion(*options);

    if (!isSceneOutputFile(options->output_model_file)) {
      MT_LOGE("Unknown output file format: {}", options->output_model_file);
      return EXIT_FAILURE;
    }

    const Scene scene = loadScene(*options);
    saveScene(options->output_model_file, scene, saveSceneOptions(*options));
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
