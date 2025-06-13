/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/character_utility.h>
#include <momentum/character/inverse_parameter_transform.h>
#include <momentum/io/gltf/gltf_io.h>
#include <momentum/io/marker/marker_io.h>
#include <momentum/io/skeleton/parameter_transform_io.h>
#include <momentum/io/skeleton/parameters_io.h>
#include <momentum/marker_tracking/app_utils.h>
#include <momentum/marker_tracking/marker_tracker.h>
#include <momentum/marker_tracking/tracker_utils.h>

#include <CLI/CLI.hpp>

#define DEFAULT_LOG_CHANNEL "refine_motion"
#include <momentum/common/log.h>

using namespace momentum;

int main(int argc, char* argv[]) {
  try {
    const std::string appName("Refine Motion");
    CLI::App app(appName);

    // set up command line options
    app.set_config("-c", "", "Input configuration", false);
    auto ioOpt = std::make_shared<IOOptions>();
    addIOOptions(app, ioOpt);

    auto config = std::make_shared<RefineConfig>();
    addRefineOptions(app, config);

    std::string paramFile;
    app.add_option(
           "-p,--parameters",
           paramFile,
           "A new model parameter config for the character. Use at your own risk!")
        ->check(CLI::ExistingFile);

    // Only doing motion smoothing for now. We can also fine tune marker offsets or solve everything
    // together again.
    CLI11_PARSE(app, argc, argv);

    // Load character and motion
    auto [character, motion, id, fps] = loadCharacterWithMotion(ioOpt->inputFile);
    // Do some surgery with a different param config.
    if (!paramFile.empty()) {
      // save current parameter names for name-based mapping.
      std::vector<std::string> oldNames = character.parameterTransform.name;

      // replace the old param on the character with the new one
      auto loadedParams = loadMomentumModel(paramFile);
      loadParameters(loadedParams, character);

      // map motion to new pt; new params will be zero
      motion = mapMotionToCharacter({oldNames, motion}, character);
    }
    // Use the new params to recover id model parameters.
    ModelParameters identity =
        InverseParameterTransform(character.parameterTransform).apply(id).pose;

    // Load input marker data
    const auto actor = loadMarkersForMainSubject(ioOpt->inputFile);
    std::vector<std::vector<Marker>> markerData;
    if (actor) {
      markerData = actor->frames;
    } else {
      MT_LOGE("Failed to load data from {}", ioOpt->inputFile);
      return 0;
    }

    // the loaded motion matrix has zeros in identity fields; we need to fill them in for
    // trackSequence, which assumes complete parameters in the initial value
    const ParameterSet idParamSet = character.parameterTransform.getScalingParameters();
    fillIdentity(idParamSet, identity, motion);

    MatrixXf finalMotion = refineMotion(markerData, motion, *config, character);
    // save results
    saveMotion(
        ioOpt->outputFile,
        character,
        finalMotion.col(0) /*contains newly solved id*/,
        finalMotion,
        markerData,
        fps);
    MT_LOGI("{} saved", ioOpt->outputFile);
  } catch (std::exception& e) {
    MT_LOGE("{}", e.what());
  } catch (...) {
    // Handle any other exceptions to prevent them from escaping main
    MT_LOGE("Unknown error occurred");
  }

  return 0;
}
