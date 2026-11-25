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

// Log the motion of a character; a motion without frames shows the character's rest pose. FBX
// motion is joint parameters, which are passed as offsets on a zero pose.
void logMotion(
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
  JointParamsRecorder jointParams;
  if (logJoints) {
    jointParams.reserve(charParams.offsets.size(), nFrames);
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
      jointParams.add(
          static_cast<int64_t>(iFrame), time, charState.skeletonState.jointParameters.v);
    }
  }
  jointParams.send(rec, paramsPrefix, jointNames);
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

    auto [characters, fps] = loadFbxCharactersWithMotion(
        options->fbxFile,
        KeepLocators::Yes,
        options->permissive ? Permissive::Yes : Permissive::No);
    if (characters.empty()) {
      MT_LOGW("No characters loaded from file");
    }

    // Every character is logged once, with its takes one after another on a shared timeline; a
    // character that no take animates shows its rest pose once.
    // The takes are freed once joined, so all motion data is not held twice while logging.
    const std::vector<MatrixXf> motions = [&characters] {
      std::vector<std::vector<MatrixXf>> takes;
      takes.reserve(characters.size());
      for (auto& [character, characterTakes] : characters) {
        takes.push_back(std::move(characterTakes));
      }
      return joinTakesOnSharedTimeline(takes);
    }();
    for (size_t iChar = 0; iChar < characters.size(); ++iChar) {
      logMotion(
          rec,
          characters[iChar].first,
          motions[iChar],
          fps,
          "world/character/" + std::to_string(iChar),
          "world_params/character_" + std::to_string(iChar),
          options->logJoints);
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
