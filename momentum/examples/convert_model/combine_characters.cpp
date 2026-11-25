/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define DEFAULT_LOG_CHANNEL "combine_characters"

#include "convert_model_helpers.h"

#include <momentum/common/exception.h>
#include <momentum/common/log.h>
#include <momentum/io/marker/marker_io.h>

#include <CLI/CLI.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace momentum;

namespace {

struct CombineOptions {
  std::vector<std::string> inputs;
  std::string output;
  // A file whose first marker sequence the output gets; empty for no markers.
  std::string markerFile;
  bool resample = false;
  // The frame rate of the output with `resample`; 0 for the frame rate of the first input with
  // motion.
  float fps = 0.0f;
  std::string fbxNamespace;
  bool saveMeshes = false;
};

std::shared_ptr<CombineOptions> setupOptions(CLI::App& app) {
  auto opt = std::make_shared<CombineOptions>();

  app.add_option(
         "-i,--input",
         opt->inputs,
         "Input files (.glb/.gltf/.fbx/.bvh/.urdf); every character of each file is saved with its "
         "motion, which starts at the first frame of the output")
      ->required()
      ->check(CLI::ExistingFile);
  app.add_option(
         "-o,--out",
         opt->output,
         "Output file (.fbx/.glb/.gltf/.bvh); a .bvh file holds one character's skeleton and motion")
      ->required();
  app.add_option(
         "--markers",
         opt->markerFile,
         "Marker file (.c3d/.trc/.glb/.gltf/.fbx) whose markers are saved in the output; its frame "
         "rate must be the frame rate of the output")
      ->check(CLI::ExistingFile);
  auto* resample = app.add_flag(
      "--resample",
      opt->resample,
      "Resample the motion of the inputs with another frame rate than the output, by linear "
      "interpolation of the model parameters; without it, inputs with different frame rates are "
      "an error");
  app.add_option(
         "--fps",
         opt->fps,
         "Frame rate of the output with --resample (default: the frame rate of the first input "
         "with motion)")
      ->check(CLI::PositiveNumber)
      ->needs(resample);
  app.add_option("--fbx-namespace", opt->fbxNamespace, "Namespace in output fbx file");
  app.add_flag(
      "-c,--character-mesh",
      opt->saveMeshes,
      "(FBX Output file only) Saves the Character Mesh to the output file.");

  return opt;
}

// The characters of one input file, read as convert_model reads a motion file, or a URDF model.
struct Input {
  std::string file;
  Scene scene;
};

Input loadInput(const std::string& file) {
  Options options;
  if (isMotionInputFile(file)) {
    options.input_motion_file = file;
  } else {
    options.input_model_file = file;
  }
  return {.file = file, .scene = loadScene(options)};
}

bool hasMotion(const Scene& scene) {
  return std::ranges::any_of(
      scene.characters, [](const SceneCharacter& entry) { return entry.poses.cols() > 0; });
}

bool sameFps(float fps1, float fps2) {
  return std::abs(fps1 - fps2) <= 1e-4f * std::max(fps1, fps2);
}

// The frame rate of the output: options.fps with --resample, otherwise the frame rate of the first
// input with motion, or of the markers when no input has motion. Throws when an input with motion
// or the markers have another frame rate and --resample is not given.
float outputFps(
    const std::vector<Input>& inputs,
    const std::optional<MarkerSequence>& markers,
    const CombineOptions& options) {
  std::vector<const Input*> withMotion;
  for (const auto& input : inputs) {
    if (hasMotion(input.scene)) {
      withMotion.push_back(&input);
    }
  }
  float fps = 120.0f;
  if (options.fps > 0.0f) {
    fps = options.fps;
  } else if (!withMotion.empty()) {
    fps = withMotion.front()->scene.fps;
  } else if (markers.has_value()) {
    fps = markers->fps;
  }
  if (!options.resample) {
    for (const Input* input : withMotion) {
      MT_THROW_IF(
          !sameFps(input->scene.fps, fps),
          "{} has the frame rate {}, but {} has {}: give --resample to resample the motion to the "
          "frame rate of the first input with motion, or to the frame rate given with --fps",
          input->file,
          input->scene.fps,
          withMotion.front()->file,
          fps);
    }
  }
  MT_THROW_IF(
      markers.has_value() && !sameFps(markers->fps, fps),
      "The markers of {} have the frame rate {}, but the output has {}: give --resample --fps {} to "
      "resample the motion to the frame rate of the markers",
      options.markerFile,
      markers->fps,
      fps,
      markers->fps);
  return fps;
}

// The characters of all inputs in one scene at `fps`. The motion of an input with another frame
// rate is resampled, with a warning. saveScene() saves a character with the name of an earlier
// character with a numeric suffix, with a warning.
Scene combine(std::vector<Input> inputs, float fps) {
  Scene result;
  result.fps = fps;
  for (auto& [file, scene] : inputs) {
    const bool resample = hasMotion(scene) && !sameFps(scene.fps, fps);
    if (resample) {
      MT_LOGW(
          "Resampling the motion of {} from {} fps to {} fps: each frame of the output is at the "
          "same time from the start, and its model parameters are linearly interpolated between "
          "the two nearest frames of the input",
          file,
          scene.fps,
          fps);
    }
    result.hasMotionFile = result.hasMotionFile || scene.hasMotionFile;
    for (auto& entry : scene.characters) {
      if (resample) {
        entry.poses = resampleMotion(entry.poses, scene.fps, fps);
      }
      result.characters.push_back(std::move(entry));
    }
  }
  return result;
}

} // namespace

int main(int argc, char** argv) {
  try {
    CLI::App app("Saves the characters of several files with their motion to one file");
    auto options = setupOptions(app);
    CLI11_PARSE(app, argc, argv);

    if (!isSceneOutputFile(options->output)) {
      MT_LOGE("Unknown output file format: {}", options->output);
      return EXIT_FAILURE;
    }
    const bool bvhOutput = isBvhFile(options->output);
    if (bvhOutput && !options->markerFile.empty()) {
      MT_LOGE("A BVH file cannot store markers: {}", options->output);
      return EXIT_FAILURE;
    }

    std::vector<Input> inputs;
    inputs.reserve(options->inputs.size());
    for (const auto& file : options->inputs) {
      inputs.push_back(loadInput(file));
    }
    std::optional<MarkerSequence> markers;
    if (!options->markerFile.empty()) {
      auto sequences = loadMarkers(options->markerFile);
      MT_THROW_IF(sequences.empty(), "No markers in {}", options->markerFile);
      markers = std::move(sequences.front());
    }

    const float fps = outputFps(inputs, markers, *options);
    const Scene scene = combine(std::move(inputs), fps);
    MT_THROW_IF(
        bvhOutput && scene.characters.size() != 1,
        "A BVH file holds one character, but the inputs have {}",
        scene.characters.size());

    SaveSceneOptions saveOptions;
    saveOptions.fbxNamespace = options->fbxNamespace;
    saveOptions.saveMeshes = options->saveMeshes;
    saveOptions.saveMarkers = markers.has_value();
    saveOptions.markerFile = options->markerFile;
    saveScene(options->output, scene, saveOptions);
  } catch (std::exception& e) {
    MT_LOGE("Failed to combine the characters. Error: {}", e.what());
    return EXIT_FAILURE;
  } catch (...) {
    MT_LOGE("Unknown error encountered.");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
