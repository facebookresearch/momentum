/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Export OBJs Example
 *
 * This example demonstrates how to export animation data from GLB/GLTF or FBX files
 * as per-frame OBJ mesh files. It loads a character with animation and exports each
 * frame as a separate OBJ file containing the deformed mesh geometry.
 *
 * Supported formats:
 * - GLB/GLTF: Loaded using loadCharacterWithMotion()
 * - FBX: Loaded using loadFbxCharacterWithMotion()
 *
 * Usage:
 *   export_objs -i <input_file> -o <output_folder> [options]
 *
 * See README.md for detailed documentation.
 */

#include <momentum/character/character.h>
#include <momentum/character/character_state.h>
#include <momentum/common/log.h>
#include <momentum/io/fbx/fbx_io.h>
#include <momentum/io/gltf/gltf_io.h>
#include <momentum/math/mesh.h>

#include <CLI/CLI.hpp>
#include <fmt/format.h>

#include <filesystem>

using namespace momentum;

namespace {

struct Options {
  std::string inputFile;
  std::string outputFolder;
  size_t firstFrame = 0;
  int lastFrame = -1;
  size_t stride = 1;
};

std::shared_ptr<Options> setupOptions(CLI::App& app) {
  auto opt = std::make_shared<Options>();
  app.add_option("-i,--input", opt->inputFile, "Path to the input animation file (.fbx/.glb).")
      ->required()
      ->check(CLI::ExistingFile);
  app.add_option("-o,--output", opt->outputFolder, "Path to the output folder.")->required();
  app.add_option("--first", opt->firstFrame, "First frame in the motion to start obj export.")
      ->default_val(opt->firstFrame)
      ->check(CLI::NonNegativeNumber);
  app.add_option(
         "--last",
         opt->lastFrame,
         "Last frame in the motion to export (inclusive). -1 to indicate the last frame in the motion.")
      ->default_val(opt->lastFrame);
  app.add_option("--stride", opt->stride, "Frame stride when exporting data.")
      ->default_val(opt->stride)
      ->check(CLI::PositiveNumber);

  return opt;
}

// A simple obj export function as an example to avoid external deps.
int saveObj(const std::string& filename, const Mesh* mesh) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    MT_LOGE("Failed to open {} for writing", filename);
    return -1;
  }

  // Simple info
  file << "# " << mesh->vertices.size() << " vertices; " << mesh->faces.size() << " faces"
       << std::endl;

  // Vertex positions
  for (const auto& v : mesh->vertices) {
    file << "v " << v(0) << " " << v(1) << " " << v(2) << std::endl;
  }
  file << std::endl;

  // Faces -- our mesh is triangle only.
  // NOTE: obj vertex indices is 1-based not 0-based.
  for (const auto& f : mesh->faces) {
    file << "f " << 1 + f(0) << " " << 1 + f(1) << " " << 1 + f(2) << std::endl;
  }
  file.close();
  return 0;
}

} // namespace

int main(int argc, char* argv[]) try {
  CLI::App app("Export objs app");
  auto opts = setupOptions(app);
  CLI11_PARSE(app, argc, argv);

  // Determine file type by extension
  const std::string extension = std::filesystem::path(opts->inputFile).extension().string();
  const bool isGlb = (extension == ".glb" || extension == ".gltf");
  const bool isFbx = (extension == ".fbx");

  if (!isGlb && !isFbx) {
    MT_LOGE("Unsupported file format: {}. Only .glb, .gltf, and .fbx are supported.", extension);
    return EXIT_FAILURE;
  }

  Character character;
  MatrixXf motion; // For GLB: single motion matrix
  std::vector<MatrixXf> motions; // For FBX: vector of motion matrices
  VectorXf id; // Identity parameters (GLB only)
  float fps = 0.0f;

  // Load character and motion based on file type
  if (isGlb) {
    MT_LOGI("Loading GLB/GLTF file: {}", opts->inputFile);
    std::tie(character, motion, id, fps) = loadCharacterWithMotion(opts->inputFile);
  } else {
    MT_LOGI("Loading FBX file: {}", opts->inputFile);
    std::tie(character, motions, fps) = loadFbxCharacterWithMotion(opts->inputFile);
    // Convert vector of motions to single motion matrix if needed
    if (!motions.empty() && motions[0].cols() > 0) {
      motion = motions[0]; // Use first motion
      if (motions.size() > 1) {
        MT_LOGW("FBX file contains {} motions. Using only the first one.", motions.size());
      }
    }
  }

  if (character.mesh == nullptr) {
    MT_LOGW("No mesh found in the input; exit without saving.");
    return EXIT_SUCCESS;
  }

  // Check and create the output folder if needed
  if (!std::filesystem::is_directory(opts->outputFolder)) {
    MT_LOGI("Create output folder {}", opts->outputFolder);
    std::filesystem::create_directories(opts->outputFolder);
  }

  const size_t numFrames = motion.cols();

  if (numFrames == 0) {
    // Export the template mesh (no animation)
    MT_LOGI("No animation data found. Exporting template mesh.");
    const std::string outFile = fmt::format(
        "{}/{}.obj",
        opts->outputFolder,
        std::filesystem::path(opts->inputFile).filename().stem().string());
    if (saveObj(outFile, character.mesh.get()) == 0) {
      return EXIT_SUCCESS;
    } else {
      return EXIT_FAILURE;
    }
  }

  // Export animation sequence
  MT_LOGI("Exporting animation with {} frames at {} fps", numFrames, fps);

  // Apply frame range options
  const size_t startFrame = opts->firstFrame;
  const size_t endFrame = (opts->lastFrame < 0)
      ? numFrames
      : std::min(static_cast<size_t>(opts->lastFrame + 1), numFrames);

  if (startFrame >= numFrames) {
    MT_LOGE("First frame ({}) is out of range (total frames: {})", startFrame, numFrames);
    return EXIT_FAILURE;
  }

  CharacterState state(character);
  CharacterParameters params;
  if (isGlb && id.size() > 0) {
    // id is in JointParameters. It is constant for the character and not time-varying.
    params.offsets = id;
  } else if (isFbx) {
    params.pose.v.setZero(character.parameterTransform.numAllModelParameters()); // should be zero
  }

  size_t exportCount = 0;
  for (size_t iFrame = startFrame; iFrame < endFrame; iFrame += opts->stride) {
    // Here is the tricky part: for glb files, we store model parameters in a custom plugin and
    // read it back in. So the motion matrix we read from glb is of ModelParameters type. For fbx
    // files, we do not store custom information, so we can only read back joint parameters. The
    // motion matrix we read from fbx is of JointParameters type. We will need to handle them
    // differently.
    if (isGlb) {
      params.pose = motion.col(iFrame);
    } else {
      params.offsets = motion.col(iFrame);
    }
    state.set(params, character, true, false, false);
    const std::string outFile = fmt::format("{}/{:05}.obj", opts->outputFolder, exportCount);
    if (saveObj(outFile, state.meshState.get()) == 0) {
      exportCount++;
    }
  }
  MT_LOGI("Exported {} frames to {}", exportCount, opts->outputFolder);
  return EXIT_SUCCESS;
} catch (std::exception& e) {
  MT_LOGE("Exception thrown {}", e.what());
  return EXIT_FAILURE;
} catch (...) {
  MT_LOGE("Unknown exception.");
  return EXIT_FAILURE;
}
