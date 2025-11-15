/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/character/marker.h>
#include <momentum/marker_tracking/marker_tracker.h>

#include <CLI/CLI.hpp>

namespace momentum {

struct IOOptions {
  std::string inputFile;
  std::string outputFile;
};

struct ModelOptions {
  std::string model;
  std::string parameters;
  std::string locators;
};

void addIOOptions(CLI::App& app, std::shared_ptr<IOOptions> ioOptions);
void addModelOptions(CLI::App& app, std::shared_ptr<ModelOptions> modelOptions);
void addCalibrationOptions(CLI::App& app, std::shared_ptr<CalibrationConfig> config);
void addTrackingOptions(CLI::App& app, std::shared_ptr<TrackingConfig> config);
void addRefineOptions(CLI::App& app, std::shared_ptr<RefineConfig> config);

std::tuple<momentum::Character, momentum::ModelParameters> loadCalibratedModel(
    const std::string& modelFile);

std::tuple<momentum::Character, momentum::ModelParameters> loadCharacterWithIdentity(
    const ModelOptions& modelFiles);

} // namespace momentum
