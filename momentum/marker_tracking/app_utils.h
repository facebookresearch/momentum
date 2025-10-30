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

/// Save the given character and motion to a GLB or FBX file.
///
/// @param[in] outFile The GLB/FBX file to save to
/// @param[in] character The GLB/FBX file to save to
/// @param[in] identity The identity parameters used for the character
/// @param[in] finalMotion The motion save to the file. (Note: this may be modified to remove
/// scaling parameters if saveScaleToMotion is false)
/// @param[in] markerData Marker data to save to the file
/// @param[in] fps Framerate of the motion
/// @param[in] saveMarkerMesh (optional) Whether to save a visible cube mesh for the markers
/// @param[in] saveScaleToMotion (optional) Whether to save the scale parameters to the motion or
/// identity parameter vectors (saving to motion is preferred)
void saveMotion(
    const std::string& outFile,
    const momentum::Character& character,
    const momentum::ModelParameters& identity,
    Eigen::MatrixXf& finalMotion,
    std::span<const std::vector<momentum::Marker>> markerData,
    double fps,
    bool saveMarkerMesh = true);

} // namespace momentum
