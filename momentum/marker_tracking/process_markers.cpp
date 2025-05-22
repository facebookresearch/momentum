/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/marker_tracking/process_markers.h"

#include "momentum/character/marker.h"
#include "momentum/character/types.h"
#include "momentum/common/checks.h"
#include "momentum/common/filesystem.h"
#include "momentum/common/log.h"
#include "momentum/io/marker/marker_io.h"
#include "momentum/marker_tracking/app_utils.h"
#include "momentum/marker_tracking/tracker_utils.h"

#include <tuple>

namespace marker_tracking {

Eigen::MatrixXf processMarkers(
    momentum::Character& character,
    momentum::ModelParameters& identity,
    const std::vector<std::vector<momentum::Marker>>& markerData,
    const TrackingConfig& trackingConfig,
    const CalibrationConfig& calibrationConfig,
    bool calibrate,
    size_t firstFrame,
    size_t maxFrames) {
  MT_THROW_IF(
      firstFrame > markerData.size(),
      "First frame {} can't exceed total frames {}",
      firstFrame,
      markerData.size());
  const size_t lastFrame =
      maxFrames > 0 ? std::min(firstFrame + maxFrames, markerData.size()) : markerData.size();
  const gsl::span<const std::vector<momentum::Marker>> inputData(
      markerData.data() + firstFrame, lastFrame - firstFrame);
  // calibrate model and locators
  if (calibrate) {
    MT_CHECK(
        !(calibrationConfig.globalScaleOnly & calibrationConfig.locatorsOnly),
        "globalScaleOnly and locatorsOnly are exclusive; they cannot both be true.");

    if (calibrationConfig.locatorsOnly) {
      // The output locators will be written to character.
      calibrateLocators(inputData, calibrationConfig, identity, character);
    } else {
      // The output locators will be written to character. The output identity will be saved in the
      // identity variable.
      calibrateModel(inputData, calibrationConfig, character, identity);
    }
  }

  // track motion; identity parameters will be repeated for every frame in finalMotion.
  Eigen::MatrixXf finalMotion = trackPosesPerframe(inputData, character, identity, trackingConfig);

  if (trackingConfig.debug) {
    auto errors = getLocatorError(markerData, finalMotion, character);
    MT_LOGI("Average marker error: {}", errors.first);
    MT_LOGI("Max marker error: {}", errors.second);
  }
  return finalMotion;
}

void processMarkerFile(
    const std::string& inputMarkerFile,
    const std::string& outputFile,
    const TrackingConfig& trackingConfig,
    const CalibrationConfig& calibrationConfig,
    const ModelOptions& modelOptions,
    bool calibrate,
    size_t firstFrame,
    size_t maxFrames) {
  // validate output file extension and quit early
  const auto outExtension = filesystem::path(outputFile).extension();
  if (outExtension != ".fbx" && outExtension != ".gltf" && outExtension != ".glb") {
    MT_LOGE(
        "Invalid output file type {}; supported types are glb, gltf, and fbx",
        outExtension.string());
    return;
  }

  // read marker data
  const auto actor = momentum::loadMarkersForMainSubject(inputMarkerFile);
  if (!actor) {
    MT_LOGE("No marker sequence found in the marker file: {}", inputMarkerFile);
    return;
  }
  // read input character and optionally its identity parameters
  auto [character, identity] = loadCharacterWithIdentity(modelOptions);
  try {
    Eigen::MatrixXf finalMotion = processMarkers(
        character,
        identity,
        actor->frames,
        trackingConfig,
        calibrationConfig,
        calibrate,
        firstFrame,
        maxFrames);

    // save results
    const size_t lastFrame = maxFrames > 0 ? std::min(firstFrame + maxFrames, actor->frames.size())
                                           : actor->frames.size();
    saveMotion(
        outputFile,
        character,
        identity,
        finalMotion,
        gsl::span<const std::vector<momentum::Marker>>(
            actor->frames.data() + firstFrame, actor->frames.data() + lastFrame),
        actor->fps);
    MT_LOGI("{} saved", outputFile);
  } catch (std::exception& e) {
    MT_LOGE("Failed to track the markers from file {}, exception: {}", inputMarkerFile, e.what());
  }
}

} // namespace marker_tracking
