/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/common/filesystem.h>
#include <momentum/common/log.h>
#include <momentum/gui/rerun/logger.h>
#include <momentum/gui/rerun/logging_redirect.h>
#include <momentum/gui/rerun/rerun_compat.h>
#include <momentum/io/marker/c3d_io.h>

#include <CLI/CLI.hpp>
#include <rerun.hpp>

using namespace rerun;
using namespace momentum;

namespace {

struct Options {
  std::string c3dFile;
  bool plot = false;
};

std::shared_ptr<Options> setupOptions(CLI::App& app) {
  auto opt = std::make_shared<Options>();
  app.add_option("-i,--input", opt->c3dFile, "Path to the c3d file")
      ->required()
      ->check(CLI::ExistingFile);
  app.add_flag("--plot", opt->plot, "Draw 2D plots of marker trajectories (very slow)")
      ->default_val(opt->plot);
  return opt;
}

} // namespace

int main(int argc, char* argv[]) {
  try {
    CLI::App app("C3D Viewer");
    auto options = setupOptions(app);
    CLI11_PARSE(app, argc, argv);

    const filesystem::path filePath(options->c3dFile);
    const std::string fileName = filePath.filename().string();
    if (filePath.extension() != ".c3d") {
      MT_LOGE("{} is not a supported format.", fileName);
      return 0;
    }
    const auto sequences = loadC3d(options->c3dFile);

    const auto rec = RecordingStream(fileName);
    rec.spawn().exit_on_failure();
    redirectLogsToRerun(rec);

    rec.log_static("world", ViewCoordinates(::components::ViewCoordinates::RUB)); // Set an up-axis

    for (const auto& actor : sequences) {
      if (actor.frames.empty()) {
        continue;
      }

      const size_t nFrames = actor.frames.size();
      const std::string streamName = actor.name.empty() ? "Character" : actor.name;
      if (options->plot) {
        for (const auto& marker : actor.frames[0]) {
          rec.log_static(
              streamName + "/" + marker.name + "/x",
              momentum::makeSeriesLineWithName(marker.name + ".x"));
          rec.log_static(
              streamName + "/" + marker.name + "/y",
              momentum::makeSeriesLineWithName(marker.name + ".y"));
          rec.log_static(
              streamName + "/" + marker.name + "/z",
              momentum::makeSeriesLineWithName(marker.name + ".z"));
        }
      }

      for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
        rec.set_time_sequence("frame_index", iFrame);
        momentum::setTimeSeconds(rec, "log_time", (float)iFrame / actor.fps);
        logMarkers(rec, "world/" + streamName, actor.frames.at(iFrame));

        if (options->plot) {
          for (const auto& marker : actor.frames.at(iFrame)) {
            if (!marker.occluded) {
              rec.log(streamName + "/" + marker.name + "/x", momentum::makeScalar(marker.pos.x()));
              rec.log(streamName + "/" + marker.name + "/y", momentum::makeScalar(marker.pos.y()));
              rec.log(streamName + "/" + marker.name + "/z", momentum::makeScalar(marker.pos.z()));
            }
          }
        }
      }
    }
  } catch (const std::exception& e) {
    MT_LOGE("Exception thrown. Error: {}", e.what());
    return EXIT_FAILURE;
  } catch (...) {
    MT_LOGE("Exception thrown. Unknown error.");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
