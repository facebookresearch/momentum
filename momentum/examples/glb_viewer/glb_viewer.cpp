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
#include <vector>

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
  app.add_flag("--log-params", opt->logParams, "Log model parameters")->default_val(opt->logParams);
  app.add_flag("--log-joints", opt->logJoints, "Log joint parameters")->default_val(opt->logJoints);
  return opt;
}

/// Compute the frame range [firstFrame, lastFrame) to iterate over based on user options.
std::pair<size_t, size_t>
computeFrameRange(const Options& options, size_t nFrames, size_t nMarkerFrames) {
  size_t firstFrame = 0;
  if (options.firstFrame >= nFrames && options.firstFrame >= nMarkerFrames) {
    MT_LOGW(
        "Requested first frame {} is larger than the total number of frames and marker frames{}; argument ignored.",
        options.firstFrame,
        nFrames);
  } else {
    firstFrame = options.firstFrame;
  }

  size_t lastFrame = std::max(nFrames, nMarkerFrames);
  if (options.maxFrames > 0) {
    lastFrame = options.firstFrame + options.maxFrames * options.stride;
  }
  return {firstFrame, lastFrame};
}

/// Validate marker sequence consistency with character motion and return the effective fps.
float validateMarkerSequence(
    const MarkerSequence& markers,
    bool hasMotion,
    size_t nFrames,
    float motionFps) {
  if (!hasMotion) {
    MT_LOGE("No character motion in the file. Using fps from the marker sequence.");
    return markers.fps;
  }

  if (markers.frames.size() != nFrames) {
    MT_LOGW("Has {} marker frames but {} motion frames.", markers.frames.size(), nFrames);
  }
  if (motionFps != markers.fps) {
    MT_LOGW(
        "Marker sequence has {} fps but motion sequence has {} fps. Using motion's.",
        markers.fps,
        motionFps);
  }
  return motionFps;
}

/// Build a lookup map from locator name to index.
std::map<std::string, size_t> buildLocatorLookup(const Character& character) {
  std::map<std::string, size_t> lookup;
  for (size_t i = 0; i < character.locators.size(); i++) {
    lookup[character.locators[i].name] = i;
  }
  return lookup;
}

/// Selected frames for batch scalar parameter logging.
struct FrameSelection {
  std::vector<int64_t> indices;
  std::vector<double> logTimes;
  std::vector<Eigen::Index> cols;
};

/// Pre-compute which frames in the range contain character motion data.
FrameSelection
selectMotionFrames(size_t firstFrame, size_t lastFrame, size_t stride, size_t nFrames, float fps) {
  FrameSelection sel;
  for (size_t iFrame = firstFrame; iFrame < lastFrame; iFrame += stride) {
    if (iFrame < nFrames) {
      sel.indices.push_back(static_cast<int64_t>(iFrame));
      sel.logTimes.push_back(static_cast<double>(iFrame) / fps);
      sel.cols.push_back(static_cast<Eigen::Index>(iFrame));
    }
  }
  return sel;
}

/// Batch log model parameters using send_columns for all selected frames.
void batchLogModelParams(
    const RecordingStream& rec,
    std::span<const std::string> paramNames,
    const Eigen::MatrixXf& motion,
    const FrameSelection& sel) {
  if (sel.cols.empty()) {
    return;
  }
  Eigen::MatrixXf selectedParams(motion.rows(), sel.cols.size());
  for (size_t i = 0; i < sel.cols.size(); ++i) {
    selectedParams.col(i) = motion.col(sel.cols[i]);
  }
  logModelParamsColumns(
      rec, "world_params", "model_params", paramNames, selectedParams, sel.indices, sel.logTimes);
}

/// Batch log joint parameters collected during the frame loop.
void batchLogJointParams(
    const RecordingStream& rec,
    std::span<const std::string> jointNames,
    const Eigen::MatrixXf& allJointParams,
    size_t nCollected,
    const FrameSelection& sel) {
  if (nCollected == 0) {
    return;
  }
  logJointParamsColumns(
      rec,
      "world_params",
      "joint_params",
      jointNames,
      allJointParams.leftCols(nCollected),
      std::span<const int64_t>(sel.indices.data(), nCollected),
      std::span<const double>(sel.logTimes.data(), nCollected));
}

/// Run the GLB viewer with the given options.
int run(const Options& options) {
  // Validate file extension
  const filesystem::path filePath(options.glbFile);
  const std::string fileName = filePath.filename().string();
  if (filePath.extension() != ".glb" && filePath.extension() != ".gltf") {
    MT_LOGE("{} is not a supported format.", fileName);
    return 0;
  }

  // Set up Rerun recording stream
  const std::string title = options.title.empty() ? fileName : options.title;
  const auto rec = RecordingStream(title);
  rec.spawn().exit_on_failure();
  redirectLogsToRerun(rec);
  rec.log_static("world", ViewCoordinates(::components::ViewCoordinates::RUB)); // Set an up-axis

  // Load character and motion data
  const auto [character, motion, offsets, cFps] = loadCharacterWithMotion(options.glbFile);
  const size_t nFrames = motion.cols();
  const bool hasMotion = nFrames > 0;
  auto fps = cFps;

  // Log the static template when there is no motion
  if (!hasMotion) {
    CharacterParameters param;
    param.pose = Eigen::VectorXf::Zero(character.parameterTransform.numAllModelParameters());
    CharacterState charState(
        param, character, true /*updateMesh*/, true /*updateCollision*/, false /*applyLimits*/);
    logCharacter(rec, "world/character", character, charState);
  }

  // Load and validate marker sequence
  const auto markers = loadMarkerSequence(options.glbFile);
  const size_t nMarkerFrames = markers.frames.size();
  const bool hasMarkers = nMarkerFrames > 0;

  std::string markerStreamName;
  if (hasMarkers) {
    markerStreamName = "world/markers/" + (markers.name.empty() ? "positions" : markers.name);
    fps = validateMarkerSequence(markers, hasMotion, nFrames, fps);
  }

  if (!hasMotion && !hasMarkers) {
    return EXIT_SUCCESS;
  }

  // Log parameter names and pre-compute batch frame selection
  const auto& modelParamNames = character.parameterTransform.name;
  const auto jointNames = character.skeleton.getJointNames();
  if (options.logParams && hasMotion) {
    logModelParamNames(rec, "world_params", "model_params", modelParamNames);
  }
  if (options.logJoints && hasMotion) {
    logJointParamNames(rec, "world_params", "joint_params", jointNames);
  }

  const auto locatorLookup = buildLocatorLookup(character);
  const auto [firstFrame, lastFrame] = computeFrameRange(options, nFrames, nMarkerFrames);
  const auto sel = selectMotionFrames(firstFrame, lastFrame, options.stride, nFrames, fps);

  if (options.logParams) {
    batchLogModelParams(rec, modelParamNames, motion, sel);
  }

  // Frame loop: log character and marker data per frame, collect joint params
  CharacterState charState;
  CharacterParameters charParams;
  charParams.offsets = offsets;
  Eigen::MatrixXf allJointParams;
  size_t jointParamFrameIdx = 0;

  for (size_t iFrame = firstFrame; iFrame < lastFrame; iFrame += options.stride) {
    rec.set_time_sequence("frame_index", iFrame);
    momentum::setTimeSeconds(rec, "log_time", (float)iFrame / fps);

    if (iFrame < nFrames) {
      charParams.pose = motion.col(iFrame);
      charState.set(
          charParams,
          character,
          true /*updateMesh*/,
          true /*updateCollision*/,
          false /*applyLimits*/);
      logCharacter(rec, "world/character", character, charState);

      if (options.logJoints) {
        const auto& jp = charState.skeletonState.jointParameters.v;
        if (allJointParams.cols() == 0) {
          allJointParams.resize(jp.size(), sel.cols.size());
        }
        allJointParams.col(jointParamFrameIdx++) = jp;
      }
    }

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

  if (options.logJoints) {
    batchLogJointParams(rec, jointNames, allJointParams, jointParamFrameIdx, sel);
  }

  return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char* argv[]) {
  try {
    CLI::App app("GLB Viewer");
    auto options = setupOptions(app);
    CLI11_PARSE(app, argc, argv);
    setLogLevel(options->logLevel);
    return run(*options);
  } catch (const std::exception& e) {
    MT_LOGE("Exception thrown. Error: {}", e.what());
    return EXIT_FAILURE;
  } catch (...) {
    MT_LOGE("Exception thrown. Unknown error.");
    return EXIT_FAILURE;
  }
}
