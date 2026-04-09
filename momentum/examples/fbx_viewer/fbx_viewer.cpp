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
#include <momentum/io/fbx/fbx_io.h>

#include <CLI/CLI.hpp>
#include <rerun.hpp>

#include <string>
#include <vector>

using namespace rerun;
using namespace momentum;

namespace {

struct Options {
  std::string fbxFile;
  LogLevel logLevel = LogLevel::Info;
  bool logJoints = false;
  bool permissive = false;
};

std::shared_ptr<Options> setupOptions(CLI::App& app) {
  auto opt = std::make_shared<Options>();
  app.add_option("-i,--input", opt->fbxFile, "Path to the fbx file")
      ->required()
      ->check(CLI::ExistingFile);
  app.add_option("-l,--loglevel", opt->logLevel, "Set the log level")
      ->transform(CLI::CheckedTransformer(logLevelMap(), CLI::ignore_case))
      ->default_val(opt->logLevel);
  app.add_flag("--log-joints", opt->logJoints, "Log joint parameters")->default_val(opt->logJoints);
  app.add_flag("--permissive", opt->permissive, "Allow nonstandard file")
      ->default_val(opt->permissive);
  return opt;
}

} // namespace

int main(int argc, char* argv[]) {
  try {
    CLI::App app("FBX Viewer");
    auto options = setupOptions(app);
    CLI11_PARSE(app, argc, argv);

    setLogLevel(options->logLevel);

    // Extract the file name from the path
    const filesystem::path filePath(options->fbxFile);
    const std::string fileName = filePath.filename().string();
    if (filePath.extension() != ".fbx") {
      MT_LOGE("{} is not a supported format.", fileName);
      return 0;
    }

    const auto rec = RecordingStream(fileName);
    rec.spawn().exit_on_failure();

    redirectLogsToRerun(rec);

    rec.log_static("world", ViewCoordinates(::components::ViewCoordinates::RUB)); // Set an up-axis

    const auto [character, motions, fps] = loadFbxCharacterWithMotion(
        options->fbxFile,
        KeepLocators::Yes,
        options->permissive ? Permissive::Yes : Permissive::No);
    // Validate the loaded motion
    if (motions.empty() || (motions.size() == 1 && motions.at(0).cols() == 0)) {
      MT_LOGW("No motion loaded from file");
    }

    CharacterState charState;
    CharacterParameters charParams;
    // fbx motion is represented as JointParameters. We will do a hack here to set it to offsets and
    // have zero pose.
    charParams.pose.v.setZero(character.parameterTransform.numAllModelParameters());
    const std::vector<std::string> jointNames = character.skeleton.getJointNames();

    for (size_t iMotion = 0; iMotion < motions.size(); ++iMotion) {
      const std::string motionIndex = std::to_string(iMotion);
      const MatrixXf& motion = motions.at(iMotion);
      if (options->logJoints) {
        logJointParamNames(
            rec, "world_params/" + motionIndex, "joint_params/" + motionIndex, jointNames);
      }

      const size_t nFrames = motion.cols();

      // Pre-compute time columns for batch scalar logging
      std::vector<int64_t> frameIndices(nFrames);
      std::vector<double> logTimes(nFrames);
      for (size_t i = 0; i < nFrames; ++i) {
        frameIndices[i] = static_cast<int64_t>(i);
        logTimes[i] = static_cast<double>(i) / fps;
      }

      // Prepare matrix for collecting joint parameters during the frame loop
      Eigen::MatrixXf allJointParams;

      for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
        // log timeline
        rec.set_time_sequence("frame_index", iFrame);
        momentum::setTimeSeconds(rec, "time", (float)iFrame / fps);

        // log character info
        charParams.offsets = motion.col(iFrame);
        charState.set(
            charParams,
            character,
            true /*updateMesh*/,
            true /*updateCollision*/,
            false /*updateLimits*/);
        logCharacter(rec, "world/character/" + motionIndex, character, charState);

        // Collect joint params for batch logging after the loop
        if (options->logJoints) {
          const auto& jp = charState.skeletonState.jointParameters.v;
          if (allJointParams.cols() == 0) {
            allJointParams.resize(jp.size(), nFrames);
          }
          allJointParams.col(iFrame) = jp;
        }
      }

      // Batch log joint parameters using send_columns (after the frame loop)
      if (options->logJoints && nFrames > 0) {
        logJointParamsColumns(
            rec,
            "world_params/" + motionIndex,
            "joint_params/" + motionIndex,
            jointNames,
            allJointParams,
            frameIndices,
            logTimes);
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
