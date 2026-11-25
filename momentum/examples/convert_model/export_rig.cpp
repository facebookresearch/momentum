/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define DEFAULT_LOG_CHANNEL "export_rig"

#include "convert_model_helpers.h"

#include <momentum/common/log.h>

#include <CLI/CLI.hpp>

#include <memory>

using namespace momentum;

namespace {

// export_rig reads the model inputs and writes the rig outputs of the convert_model options.
std::shared_ptr<Options> setupOptions(CLI::App& app) {
  auto opt = std::make_shared<Options>();

  app.add_option(
         "-m,--model", opt->input_model_file, "Input character file (.fbx/.glb/.gltf/.urdf/.bvh)")
      ->required()
      ->check(CLI::ExistingFile);
  app.add_option("-p,--parameters", opt->input_params_file, "Input model parameter file (.model)")
      ->check(CLI::ExistingFile);
  app.add_option("-l,--locator", opt->input_locator_file, "Input locator file (.locators)")
      ->check(CLI::ExistingFile);
  app.add_option(
      "--character",
      opt->character_name,
      "Export the character with this name; required when the input has several characters");

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

  return opt;
}

} // namespace

int main(int argc, char** argv) {
  try {
    CLI::App app(
        "Saves the parameter transform, parameter limits and locators of a character to .model "
        "and .locators files");
    auto options = setupOptions(app);
    CLI11_PARSE(app, argc, argv);

    if (!hasRigOutputs(*options)) {
      MT_LOGE(
          "Nothing to export: give --out-parameters, --out-locators-local or --out-locators-global");
      return EXIT_FAILURE;
    }

    // loadScene() throws for an input with several characters unless --character selects one.
    saveRigOutputs(*options, loadScene(*options));
  } catch (std::exception& e) {
    MT_LOGE("Failed to export the rig. Error: {}", e.what());
    return EXIT_FAILURE;
  } catch (...) {
    MT_LOGE("Unknown error encountered.");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
