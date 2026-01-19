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
#include <momentum/io/gltf/gltf_io.h>

#include <CLI/CLI.hpp>
#include <rerun.hpp>

#include <string>

using namespace rerun;
using namespace momentum;

namespace {

struct Options {
  std::string glbFile;
  LogLevel logLevel = LogLevel::Info;
  std::string title;
  bool logParams = false;
  bool logJoints = false;
  size_t stride = 1;
  size_t firstFrame = 0;
  size_t maxFrames = 0;
};

std::shared_ptr<Options> setupOptions(CLI::App& app) {
  auto opt = std::make_shared<Options>();
  app.add_option("-i,--input", opt->glbFile, "Path to the GLB file")
      ->required()
      ->check(CLI::ExistingFile);
  app.add_option("--title", opt->title, "Title in viewer (default to be filename)");
  app.add_option(
         "--stride", opt->stride, "Stride to subsample the motion for high frequency captures")
      ->default_val(opt->stride)
      ->check(CLI::PositiveNumber);
  app.add_option("--first-frame", opt->firstFrame, "First frame to play")
      ->default_val(opt->firstFrame)
      ->check(CLI::NonNegativeNumber);
  app.add_option("--max-frames", opt->maxFrames, "Max number of frames to play (0 means all)")
      ->default_val(opt->maxFrames)
      ->check(CLI::NonNegativeNumber);
  // TODO: use enum
  app.add_option("-l,--loglevel", opt->logLevel, "Set the log level")
      ->transform(CLI::CheckedTransformer(logLevelMap(), CLI::ignore_case))
      ->default_val(opt->logLevel);
  app.add_flag("--log-params", opt->logParams, "Log model parameters (slow)")
      ->default_val(opt->logParams);
  app.add_flag("--log-joints", opt->logJoints, "Log joint parameters (very slow)")
      ->default_val(opt->logJoints);
  return opt;
}

} // namespace

int main(int argc, char* argv[]) {
  try {
    CLI::App app("GLB Viewer");
    auto options = setupOptions(app);
    CLI11_PARSE(app, argc, argv);

    setLogLevel(options->logLevel);

    // Extract the file name from the path
    const filesystem::path filePath(options->glbFile);
    const std::string fileName = filePath.filename().string();
    if (filePath.extension() != ".glb" && filePath.extension() != ".gltf") {
      MT_LOGE("{} is not a supported format.", fileName);
      return 0;
    }

    const std::string title = options->title.empty() ? fileName : options->title;
    const auto rec = RecordingStream(title);
    rec.spawn().exit_on_failure();

    redirectLogsToRerun(rec);

    rec.log_static("world", ViewCoordinates(::components::ViewCoordinates::RUB)); // Set an up-axis

    const auto [character, motion, offsets, cFps] = loadCharacterWithMotion(options->glbFile);
    const size_t nFrames = motion.cols();
    const auto kHasCharacterMotion = nFrames > 0;
    auto fps = cFps;

    // Special case when we want to view the template with no motion
    if (!kHasCharacterMotion) {
      CharacterParameters param;
      param.pose = Eigen::VectorXf::Zero(character.parameterTransform.numAllModelParameters());
      CharacterState charState(
          param, character, true /*updateMesh*/, true /*updateCollision*/, false /*applyLimits*/);
      logCharacter(rec, "world/character", character, charState);
    }

    // Load marker sequence from the same file. There can be cases where we only have marker
    // sequence.
    const auto markers = loadMarkerSequence(options->glbFile);
    const size_t nMarkerFrames = markers.frames.size();
    const auto kHasMarkerSequence = nMarkerFrames > 0;
    std::string markerStreamName;
    if (kHasMarkerSequence) {
      markerStreamName = "world/markers/" + (markers.name.empty() ? "positions" : markers.name);

      if (!kHasCharacterMotion) {
        MT_LOGE("No character motion in the file. Using fps from the marker sequence.");
        fps = markers.fps;
      } else {
        // Validate whether marker sequence and motion sequence matches.
        if (nMarkerFrames != nFrames) {
          MT_LOGW("Has {} marker frames but {} motion frames.", nMarkerFrames, nFrames);
        }
        if (fps != markers.fps) {
          MT_LOGW(
              "Marker sequence has {} fps but motion sequence has {} fps. Using motion's.",
              markers.fps,
              fps);
        }
      }
    }

    // Logging static mesh
    if ((!kHasCharacterMotion) && (!kHasMarkerSequence)) {
      return EXIT_SUCCESS;
    }

    const std::vector<std::string> modelParamNames = character.parameterTransform.name;
    const std::vector<std::string> jointNames = character.skeleton.getJointNames();

    if (options->logParams && kHasCharacterMotion) {
      logModelParamNames(rec, "world_params", "model_params", modelParamNames);
    }
    if (options->logJoints && kHasCharacterMotion) {
      logJointParamNames(rec, "world_params", "joint_params", jointNames);
    }

    std::map<std::string, size_t> locatorLookup;
    for (size_t i = 0; i < character.locators.size(); i++) {
      locatorLookup[character.locators[i].name] = i;
    }

    // Validate frame range
    size_t firstFrame = 0;
    if ((options->firstFrame >= nFrames) && (options->firstFrame >= nMarkerFrames)) {
      MT_LOGW(
          "Requested first frame {} is larger than the total number of frames and marker frames{}; argument ignored.",
          options->firstFrame,
          nFrames);
    } else {
      firstFrame = options->firstFrame;
    }

    size_t lastFrame = std::max(nFrames, nMarkerFrames);
    lastFrame = options->maxFrames > 0 ? options->firstFrame + options->maxFrames * options->stride
                                       : lastFrame;

    CharacterState charState;
    CharacterParameters charParams;
    charParams.offsets = offsets;

    for (size_t iFrame = firstFrame; iFrame < lastFrame; iFrame += options->stride) {
      // log timeline
      rec.set_time_sequence("frame_index", iFrame);
      momentum::setTimeSeconds(rec, "log_time", (float)iFrame / fps);

      // log character info
      if (iFrame < nFrames) {
        charParams.pose = motion.col(iFrame);
        charState.set(
            charParams,
            character,
            true /*updateMesh*/,
            true /*updateCollision*/,
            false /*applyLimits*/);
        logCharacter(rec, "world/character", character, charState);
        // XXX 2D plots in rerun are not scalable at the moment
        if (options->logParams) {
          logModelParams(rec, "world_params", "model_params", modelParamNames, motion.col(iFrame));
        }
        if (options->logJoints) {
          logJointParams(
              rec,
              "world_params",
              "joint_params",
              jointNames,
              charState.skeletonState.jointParameters.v);
        }
      }

      // log marker info
      if (iFrame < nMarkerFrames) {
        logMarkers(rec, markerStreamName, markers.frames.at(iFrame));
        logMarkerLocatorCorrespondence(
            rec,
            "world/markers/correspondence",
            locatorLookup,
            charState.locatorState,
            markers.frames.at(iFrame),
            3.0f);
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
