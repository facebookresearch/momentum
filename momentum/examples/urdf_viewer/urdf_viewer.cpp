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
#include <unordered_map>

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

    // Build a lookup from model parameter index to its min/max limits.
    std::unordered_map<Eigen::Index, std::pair<float, float>> paramLimitsMap;
    for (const auto& limit : character.parameterLimits) {
      if (limit.type == MinMax) {
        paramLimitsMap[limit.data.minMax.parameterIndex] = {
            limit.data.minMax.limits[0], limit.data.minMax.limits[1]};
      }
    }

    // Sinusoidal motion for each joint, one at a time.
    // Each DOF sweeps within its parameter limits (or a default range if no limits).
    const int framesPerDoF = 100;
    const auto totalDoFs = character.parameterTransform.transform.cols();
    const auto kNumFrames = framesPerDoF * totalDoFs;
    MatrixXf motion = MatrixXf::Zero(kNumModelParams, kNumFrames);
    const float fps = 30.0f;
    const float frequency = 1.0f;
    const float defaultAmplitude = 0.5f; // radians, for DOFs without limits
    for (Eigen::Index j = 0; j < totalDoFs; ++j) {
      float lower = -defaultAmplitude;
      float upper = defaultAmplitude;
      auto it = paramLimitsMap.find(j);
      if (it != paramLimitsMap.end()) {
        lower = it->second.first;
        upper = it->second.second;
      }

      // Sweep from lower to upper: midpoint + half_range * sin(...)
      const float midpoint = (lower + upper) * 0.5f;
      const float halfRange = (upper - lower) * 0.5f;

      for (int i = 0; i < framesPerDoF; ++i) {
        const auto frameIndex = static_cast<int>(j) * framesPerDoF + i;
        if (frameIndex >= kNumFrames) {
          break;
        }
        const float time = static_cast<float>(i) / fps;
        motion(j, frameIndex) = midpoint + halfRange * std::sin(twopi<float>() * frequency * time);
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
      momentum::setTimeSeconds(rec, "time", (float)i / fps);

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
