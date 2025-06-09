/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/types.h>
#include <momentum/marker_tracking/app_utils.h>
#include <momentum/marker_tracking/marker_tracker.h>

#include <string>

namespace momentum {

/// Processes marker data for a character model.
///
/// It can calibrate the model and locators based on the calibrationConfig, and track the motion of
/// the markers.
///
/// @param[in,out] character The input character definition with locators. It also holds the output
/// locators if locators are to be calibrated from the calibrationConfig.
/// @param[in,out] identity The input identity used for tracking. It also holds the solved identify
/// if identity is to be calibrated from the calibrationConfig.
/// @param[in] markerData The per-frame position targets to be tracked.
/// @param[in] trackingConfig The config for per-frame tracking after the correct identity and
/// locators are solved for (or from input).
/// @param[in] calibrationConfig The config for running calibration if calibrate is set to true.
/// @param[in] calibrate True to run calibration according to calibrationConfig.
/// @param[in] firstFrame The first frame to start solving. We pass in the first frame instead of
/// triming markerData to avoid data copy.
/// @param[in] maxFrames The maximum number of frames to process starting from firstFrame.
/// @returns The final motion matrix after tracking.
Eigen::MatrixXf processMarkers(
    momentum::Character& character,
    momentum::ModelParameters& identity,
    const std::vector<std::vector<momentum::Marker>>& markerData,
    const TrackingConfig& trackingConfig,
    const CalibrationConfig& calibrationConfig,
    bool calibrate = true,
    size_t firstFrame = 0,
    size_t maxFrames = 0);

/// Runs marker tracking on an input marker file (e.g. c3d) and writes the output motion (e.g. glb)
/// to outputFile
void processMarkerFile(
    const std::string& inputMarkerFile,
    const std::string& outputFile,
    const TrackingConfig& trackingConfig,
    const CalibrationConfig& calibrationConfig,
    const ModelOptions& modelOptions,
    bool calibrate,
    size_t firstFrame = 0,
    size_t maxFrames = 0);

} // namespace momentum
