/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/character.h>
#include <momentum/character/character_state.h>
#include <momentum/common/filesystem.h>
#include <momentum/common/log.h>
#include <momentum/gui/rerun/logger.h>
#include <momentum/gui/rerun/logging_redirect.h>
#include <momentum/gui/rerun/rerun_compat.h>
#include <momentum/io/urdf/urdf_io.h>

#include <CLI/CLI.hpp>
#include <rerun.hpp>

#include <string>

using namespace rerun;
using namespace momentum;

namespace {

struct Options {
  std::string urdfFile;
  LogLevel logLevel = LogLevel::Info;
  std::string title;
  bool logJoints = false;
};

std::shared_ptr<Options> setupOptions(CLI::App& app) {
  auto opt = std::make_shared<Options>();
  app.add_option("--title", opt->title, "Title in viewer (default to be filename)");
  app.add_option("-i,--input", opt->urdfFile, "Path to the URDF file")
      ->required()
      ->check(CLI::ExistingFile);
  app.add_option("-l,--loglevel", opt->logLevel, "Set the log level")
      ->transform(CLI::CheckedTransformer(logLevelMap(), CLI::ignore_case))
      ->default_val(opt->logLevel);
  app.add_flag("--log-joints", opt->logJoints, "Log joint parameters (very slow)")
      ->default_val(opt->logJoints);
  return opt;
}

} // namespace

int main(int argc, char* argv[]) {
  try {
    CLI::App app("URDF Viewer");
    auto options = setupOptions(app);
    CLI11_PARSE(app, argc, argv);

    setLogLevel(options->logLevel);

    // Extract the file name from the path
    const filesystem::path filePath(options->urdfFile);
    const std::string fileName = filePath.filename().string();
    if (filePath.extension() != ".urdf") {
      MT_LOGE("{} is not a supported format.", fileName);
      return 0;
    }

    const auto character = loadUrdfCharacter(options->urdfFile);

    const auto kNumModelParams = character.parameterTransform.numAllModelParameters();

    // Sinusoidal motion for each joint, one at a time
    const int framesPerDoF = 100;
    const auto totalDoFs = character.parameterTransform.transform.cols();
    const auto kNumFrames = framesPerDoF * totalDoFs;
    MatrixXf motion = MatrixXf::Zero(kNumModelParams, kNumFrames);
    const float fps = 30.0f;
    const float frequency = 1.0f;
    const float amplitude = 1.0f;
    for (auto j = 0; j < totalDoFs; ++j) {
      for (auto i = 0; i < framesPerDoF; ++i) {
        int frameIndex = j * framesPerDoF + i;
        if (frameIndex >= kNumFrames) {
          break; // Prevent exceeding total frames
        }
        const float time = i / fps;
        motion(j, frameIndex) = amplitude * std::sin(2.0f * pi() * frequency * time);
      }
    }

    const std::string title = options->title.empty() ? fileName : options->title;
    const auto rec = RecordingStream(title);
    rec.spawn().exit_on_failure();

    redirectLogsToRerun(rec);

    rec.log_static(
        "world", ViewCoordinates(components::ViewCoordinates::RIGHT_HAND_Z_UP)); // Set an up-axis

    CharacterState charState;
    CharacterParameters charParams;

    for (auto i = 0; i < kNumFrames; ++i) {
      // log timeline
      rec.set_time_sequence("frame_index", i);
      momentum::setTimeSeconds(rec, "log_time", (float)i / fps);

      charParams.pose = motion.col(i);
      charState.set(
          charParams,
          character,
          true /*updateMesh*/,
          true /*updateCollision*/,
          false /*applyLimits*/);

      logCharacter(rec, "world/character", character, charState);
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
