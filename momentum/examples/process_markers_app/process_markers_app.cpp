/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/common/log.h>
#include <momentum/marker_tracking/app_utils.h>
#include <momentum/marker_tracking/process_markers.h>

#include <CLI/CLI.hpp>

using namespace momentum;

int main(int argc, char* argv[]) {
  const std::string appName("Process Markers");
  CLI::App app(appName);

  // set up command line options
  app.set_config("-c", "", "Input configuration", false);
  auto ioOpt = std::make_shared<IOOptions>();
  addIOOptions(app, ioOpt);
  auto modelOpt = std::make_shared<ModelOptions>();
  addModelOptions(app, modelOpt);

  // max frames
  size_t maxFrames = 0;
  auto* maxFramesOption =
      app.add_option("--max-frames", maxFrames, "Max frames to solve for; 0 means full length");
  maxFramesOption->default_val(maxFrames)->check(CLI::NonNegativeNumber);

  // first frame
  size_t firstFrame = 0;
  auto* firstFrameOption =
      app.add_option("--first-frame", firstFrame, "First frame to start solving");
  firstFrameOption->default_val(firstFrame)->check(CLI::NonNegativeNumber);

  // calibration
  bool calibrate = true;
  auto* calibOption = app.add_option("--calibrate", calibrate, "Calibrate model");
  calibOption->default_val(calibrate);

  // tracker options
  auto calibConfig = std::make_shared<CalibrationConfig>();
  addCalibrationOptions(app, calibConfig);
  auto trackingConfig = std::make_shared<TrackingConfig>();
  addTrackingOptions(app, trackingConfig);

  try {
    CLI11_PARSE(app, argc, argv);
    if (calibConfig->debug || trackingConfig->debug) {
      setLogLevel(LogLevel::Debug);
    }
    processMarkerFile(
        ioOpt->inputFile,
        ioOpt->outputFile,
        *trackingConfig,
        *calibConfig,
        *modelOpt,
        calibrate,
        firstFrame,
        maxFrames);
  } catch (std::exception& e) {
    MT_LOGE("{}", e.what());
  }

  return 0;
}
