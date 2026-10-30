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
#include <momentum/gui/rerun/rerun_compat.h>
#include <momentum/gui/rerun/rerun_sink.h>
#include <momentum/io/fbx/fbx_io.h>

#include <CLI/CLI.hpp>
#include <rerun.hpp>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace rerun;
using namespace momentum;

namespace {

struct Options {
  RerunSinkOptions rerunSink;
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
  addRerunSinkOptions(app, opt->rerunSink);
  return opt;
}

// Log one animation take of a character; a take without frames shows the character's rest pose.
// FBX motion is joint parameters, which are passed as offsets on a zero pose.
void logTake(
    const RecordingStream& rec,
    const Character& character,
    const MatrixXf& motion,
    float fps,
    const std::string& entityPath,
    const std::string& paramsPrefix,
    bool logJoints) {
  CharacterParameters charParams;
  charParams.pose.v.setZero(character.parameterTransform.numAllModelParameters());
  charParams.offsets.v.setZero(character.parameterTransform.numJointParameters());
  CharacterState charState;
  const std::vector<std::string> jointNames = character.skeleton.getJointNames();
  if (logJoints) {
    logJointParamNames(rec, paramsPrefix, "joint_params", jointNames);
  }

  // A file without animation can have a zero frame rate.
  const double frameDuration = fps > 0.0f ? 1.0 / fps : 0.0;
  const auto nFrames = std::max<size_t>(motion.cols(), 1);
  Eigen::MatrixXf allJointParams;
  std::vector<int64_t> frameIndices;
  std::vector<double> logTimes;
  if (logJoints) {
    allJointParams.resize(charParams.offsets.size(), nFrames);
    frameIndices.reserve(nFrames);
    logTimes.reserve(nFrames);
  }
  for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
    const double time = static_cast<double>(iFrame) * frameDuration;
    rec.set_time_sequence("frame_index", iFrame);
    momentum::setTimeSeconds(rec, "time", static_cast<float>(time));

    if (motion.cols() > 0) {
      charParams.offsets = motion.col(iFrame);
    }
    charState.set(
        charParams,
        character,
        true /*updateMesh*/,
        true /*updateCollision*/,
        false /*updateLimits*/);
    logCharacter(rec, entityPath, character, charState);
    if (logJoints) {
      allJointParams.col(iFrame) = charState.skeletonState.jointParameters.v;
      frameIndices.push_back(static_cast<int64_t>(iFrame));
      logTimes.push_back(time);
    }
  }

  if (logJoints) {
    logJointParamsColumns(
        rec, paramsPrefix, "joint_params", jointNames, allJointParams, frameIndices, logTimes);
  }
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
    openRerunSink(rec, options->rerunSink);

    rec.log_static("world", ViewCoordinates(::components::ViewCoordinates::RUB)); // Set an up-axis

    const auto [characters, fps] = loadFbxCharactersWithMotion(
        options->fbxFile,
        KeepLocators::Yes,
        options->permissive ? Permissive::Yes : Permissive::No);
    if (characters.empty()) {
      MT_LOGW("No characters loaded from file");
    }

    // Each take of each character is logged under its own entity, starting at frame 0. A take that
    // does not animate the character is skipped; a character that no take animates shows its rest
    // pose once.
    for (size_t iChar = 0; iChar < characters.size(); ++iChar) {
      const auto& [character, takes] = characters[iChar];
      const auto logCharacterTake = [&rec, &character, fps, iChar, &options](
                                        size_t iTake, const MatrixXf& motion) {
        const std::string take = "take_" + std::to_string(iTake);
        logTake(
            rec,
            character,
            motion,
            fps,
            "world/character/" + std::to_string(iChar) + "/" + take,
            "world_params/character_" + std::to_string(iChar) + "/" + take,
            options->logJoints);
      };
      if (std::ranges::none_of(takes, [](const MatrixXf& take) { return take.cols() > 0; })) {
        logCharacterTake(0, MatrixXf());
        continue;
      }
      for (size_t iTake = 0; iTake < takes.size(); ++iTake) {
        if (takes[iTake].cols() > 0) {
          logCharacterTake(iTake, takes[iTake]);
        }
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "Exception thrown. Error: " << e.what() << '\n';
    return EXIT_FAILURE;
  } catch (...) {
    std::cerr << "Exception thrown. Unknown error.\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
