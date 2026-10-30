/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/character.h>
#include <momentum/character/character_state.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/common/exception.h>
#include <momentum/common/filesystem.h>
#include <momentum/common/log.h>
#include <momentum/gui/rerun/logger.h>
#include <momentum/gui/rerun/rerun_compat.h>
#include <momentum/gui/rerun/rerun_sink.h>
#include <momentum/io/gltf/gltf_io.h>

#include <CLI/CLI.hpp>
#include <rerun.hpp>

#include <algorithm>
#include <iostream>
#include <map>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace rerun;
using namespace momentum;

namespace {

struct Options {
  std::string glbFile;
  LogLevel logLevel = LogLevel::Info;
  std::string title;
  RerunSinkOptions rerunSink;
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
  addRerunSinkOptions(app, opt->rerunSink);
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
    float motionFps,
    size_t numCharacters) {
  if (!hasMotion) {
    MT_LOGE("No character motion in the file. Using fps from the marker sequence.");
    return markers.fps;
  }

  // With several characters nFrames is the longest motion, so a mismatch says nothing about any
  // particular character; only compare when there is one character to compare against.
  if (numCharacters == 1 && markers.frames.size() != nFrames) {
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

/// Batch log model parameters using send_columns for all selected frames, under the per-character
/// stream prefixes `worldPrefix` (root parameters, e.g. "world_params/character_0") and
/// `posePrefix` (the other parameters, e.g. "model_params/character_0").
void batchLogModelParams(
    const RecordingStream& rec,
    const std::string& worldPrefix,
    const std::string& posePrefix,
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
      rec, worldPrefix, posePrefix, paramNames, selectedParams, sel.indices, sel.logTimes);
}

/// Everything the frame loop needs for one character.
struct CharacterPlayback {
  std::string entityPath;
  // Stream prefixes of the character's parameters: world (root) parameters, and the other joint
  // and model parameters.
  std::string worldParamsPrefix;
  std::string jointParamsPrefix;
  std::string modelParamsPrefix;
  std::string correspondencePath;
  std::vector<std::string> jointNames;
  std::map<std::string, size_t> locatorLookup;
  CharacterParameters params;
  // One state per character: CharacterState caches the posed mesh of the character it last posed.
  CharacterState state;
  // Joint parameters per frame for a character played from its glTF animation channels.
  Eigen::MatrixXf jointMotion;
  // Frames of momentum motion or channel motion; the character holds its last pose after them.
  size_t numFrames = 0;
  JointParamsRecorder jointParams;
};

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
  openRerunSink(rec, options.rerunSink);
  rec.log_static("world", ViewCoordinates(::components::ViewCoordinates::RUB)); // Set an up-axis

  // Load every character in the scene together with its own motion.
  auto [characters, cFps] = loadGltfCharactersWithMotion(options.glbFile);
  if (characters.empty()) {
    // Not fatal: a file may hold only a marker sequence, which is still worth displaying.
    MT_LOGW("No character found in {}.", fileName);
  }
  auto fps = cFps;

  // A file not written by momentum keeps its animation in glTF animation channels rather than in
  // momentum motion. Play those channels, as joint parameters, for the characters without momentum
  // motion.
  std::vector<CharacterPlayback> playback(characters.size());
  if (std::ranges::any_of(
          characters, [](const CharacterMotion& cm) { return cm.motion.cols() == 0; })) {
    try {
      auto jointMotion = loadGltfCharactersChannelMotion(filesystem::path(options.glbFile));
      // Both loaders find the characters of a file in the same order.
      MT_THROW_IF(
          jointMotion.size() != characters.size(),
          "Loaded {} characters with animation channels but {} with motion",
          jointMotion.size(),
          characters.size());
      for (size_t i = 0; i < playback.size(); ++i) {
        if (characters[i].motion.cols() == 0) {
          playback.at(i).jointMotion = std::move(jointMotion.at(i));
        }
      }
    } catch (const std::exception& e) {
      MT_LOGW(
          "Unable to play the glTF animation channels of {}: {}. Characters without momentum "
          "motion show their identity pose.",
          fileName,
          e.what());
    }
  }

  // The frame range spans the longest motion.
  size_t nFrames = 0;
  for (size_t i = 0; i < characters.size(); ++i) {
    auto& pc = playback.at(i);
    pc.numFrames =
        static_cast<size_t>(std::max(characters[i].motion.cols(), pc.jointMotion.cols()));
    nFrames = std::max(nFrames, pc.numFrames);
  }
  const bool hasMotion = nFrames > 0;

  // Start every character at its rest pose. That is all a character without motion shows, and it
  // gives the marker correspondence a locator state before a character's motion starts.
  for (size_t i = 0; i < characters.size(); ++i) {
    const auto& cm = characters[i];
    auto& pc = playback.at(i);
    pc.entityPath = "world/character/" + std::to_string(i);
    pc.worldParamsPrefix = "world_params/character_" + std::to_string(i);
    pc.jointParamsPrefix = "joint_params/character_" + std::to_string(i);
    pc.modelParamsPrefix = "model_params/character_" + std::to_string(i);
    pc.correspondencePath = "world/markers/correspondence/character_" + std::to_string(i);
    pc.params.offsets = cm.identity;
    pc.params.pose = Eigen::VectorXf::Zero(cm.character.parameterTransform.numAllModelParameters());
    pc.state.set(
        pc.params,
        cm.character,
        true /*updateMesh*/,
        true /*updateCollision*/,
        false /*applyLimits*/);
    if (pc.numFrames == 0) {
      logCharacter(rec, pc.entityPath, cm.character, pc.state);
    }
  }

  // Load and validate the marker sequence, which belongs to the scene rather than to a character.
  const auto markers = loadMarkerSequence(options.glbFile);
  const size_t nMarkerFrames = markers.frames.size();
  const bool hasMarkers = nMarkerFrames > 0;

  std::string markerStreamName;
  if (hasMarkers) {
    markerStreamName = "world/markers/" + (markers.name.empty() ? "positions" : markers.name);
    fps = validateMarkerSequence(markers, hasMotion, nFrames, fps, characters.size());
  }

  if (!hasMotion && !hasMarkers) {
    return EXIT_SUCCESS;
  }

  // Log parameter names and pre-compute batch frame selection
  const auto [firstFrame, lastFrame] = computeFrameRange(options, nFrames, nMarkerFrames);
  for (size_t i = 0; i < characters.size(); ++i) {
    const auto& cm = characters[i];
    auto& pc = playback.at(i);
    pc.jointNames = cm.character.skeleton.getJointNames();
    pc.locatorLookup = buildLocatorLookup(cm.character);
    if (pc.numFrames == 0) {
      continue;
    }
    const FrameSelection sel =
        selectMotionFrames(firstFrame, lastFrame, options.stride, pc.numFrames, fps);
    const auto& modelParamNames = cm.character.parameterTransform.name;
    if (options.logParams && cm.motion.cols() > 0) {
      logModelParamNames(rec, pc.worldParamsPrefix, pc.modelParamsPrefix, modelParamNames);
      batchLogModelParams(
          rec, pc.worldParamsPrefix, pc.modelParamsPrefix, modelParamNames, cm.motion, sel);
    }
    if (options.logJoints) {
      logJointParamNames(rec, pc.worldParamsPrefix, pc.jointParamsPrefix, pc.jointNames);
      pc.jointParams.reserve(cm.character.parameterTransform.numJointParameters(), sel.cols.size());
    }
  }

  // Frame loop: log the marker frame, then each character's pose and its marker correspondence.
  for (size_t iFrame = firstFrame; iFrame < lastFrame; iFrame += options.stride) {
    rec.set_time_sequence("frame_index", iFrame);
    momentum::setTimeSeconds(rec, "time", (float)iFrame / fps);

    if (iFrame < nMarkerFrames) {
      logMarkers(rec, markerStreamName, markers.frames.at(iFrame));
    }

    for (size_t i = 0; i < characters.size(); ++i) {
      const auto& cm = characters[i];
      auto& pc = playback.at(i);

      // A character keeps its last pose after its motion ends. A character whose motion ends
      // before the first frame shows its last pose from the first frame on.
      const bool inMotion = iFrame < pc.numFrames;
      if (inMotion || (iFrame == firstFrame && pc.numFrames > 0)) {
        const auto frame = static_cast<Eigen::Index>(std::min(iFrame, pc.numFrames - 1));
        // Joint motion goes through the offsets, on top of the zero pose set up above.
        if (cm.motion.cols() > 0) {
          pc.params.pose = cm.motion.col(frame);
        } else {
          pc.params.offsets = pc.jointMotion.col(frame);
        }
        pc.state.set(
            pc.params,
            cm.character,
            true /*updateMesh*/,
            true /*updateCollision*/,
            false /*applyLimits*/);
        logCharacter(rec, pc.entityPath, cm.character, pc.state);

        if (options.logJoints && inMotion) {
          pc.jointParams.add(
              static_cast<int64_t>(iFrame),
              static_cast<double>(iFrame) / fps,
              pc.state.skeletonState.jointParameters.v);
        }
      }

      if (iFrame < nMarkerFrames) {
        logMarkerLocatorCorrespondence(
            rec,
            pc.correspondencePath,
            pc.locatorLookup,
            pc.state.locatorState,
            markers.frames.at(iFrame),
            3.0f);
      }
    }
  }

  for (const auto& pc : playback) {
    pc.jointParams.send(rec, pc.worldParamsPrefix, pc.jointParamsPrefix, pc.jointNames);
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
    std::cerr << "Exception thrown. Error: " << e.what() << '\n';
    return EXIT_FAILURE;
  } catch (...) {
    std::cerr << "Exception thrown. Unknown error.\n";
    return EXIT_FAILURE;
  }
}
