/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define DEFAULT_LOG_CHANNEL "split_characters"

#include "convert_model_helpers.h"

#include <momentum/common/filesystem.h>
#include <momentum/common/log.h>
#include <momentum/io/common/name_utils.h>

#include <CLI/CLI.hpp>

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

using namespace momentum;

namespace {

struct SplitOptions {
  // The input file is read as the motion file of convert_model, so every character keeps its
  // motion; the FBX options apply to every output file.
  Options options;
  std::string outputDirectory;
  // The extension of the output files, without the dot; empty for the extension of the input.
  std::string format;
};

std::shared_ptr<SplitOptions> setupOptions(CLI::App& app) {
  auto opt = std::make_shared<SplitOptions>();

  app.add_option(
         "-i,--input",
         opt->options.input_motion_file,
         "Input file (.glb/.gltf/.fbx/.bvh); a .bvh file holds one character")
      ->required()
      ->check(CLI::ExistingFile);
  app.add_option(
         "--out-dir",
         opt->outputDirectory,
         "Output directory; each character is saved with its motion to <character name>.<format> "
         "in it")
      ->required();
  app.add_option(
      "--format",
      opt->format,
      "Format of the output files: fbx, glb, gltf or bvh (default: the format of the input file)");
  app.add_option("--fbx-namespace", opt->options.fbx_namespace, "Namespace in output fbx files");
  app.add_flag(
      "-c,--character-mesh",
      opt->options.character_mesh_save,
      "(FBX Output files only) Saves the Character Mesh to the output files.");

  return opt;
}

std::string toLowerCase(std::string text) {
  std::ranges::transform(
      text, text.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return text;
}

// The names in lower case, for makeUniqueName(): some file systems ignore the letter case.
struct LowerCaseNames {
  std::unordered_set<std::string> names;

  [[nodiscard]] auto find(const std::string& name) const {
    return names.find(toLowerCase(name));
  }

  [[nodiscard]] auto end() const {
    return names.end();
  }
};

// Whether Windows uses `baseName`, a file name before its first '.', in any letter case, for a
// device, so that no file can have it.
bool isWindowsDeviceName(const std::string& baseName) {
  const std::string name = toLowerCase(baseName);
  if (name == "con" || name == "prn" || name == "aux" || name == "nul") {
    return true;
  }
  return name.size() == 4 && (name.starts_with("com") || name.starts_with("lpt")) &&
      std::isdigit(static_cast<unsigned char>(name[3])) != 0;
}

// The file name, without the extension, of each character of `scene`: its name, or
// "character_<index>" for a character without a name, with '_' in place of the characters other
// than letters, digits, '-', '_' and '.', '_' after a name that Windows uses for a device (such
// as "NUL" or "COM1"), and a numeric suffix when an earlier character has the same file name in
// any letter case.
std::vector<std::string> fileNames(const Scene& scene) {
  LowerCaseNames used;
  std::vector<std::string> result;
  for (size_t index = 0; index < scene.characters.size(); ++index) {
    std::string name = scene.characters[index].character.name;
    if (name.empty()) {
      name = "character_" + std::to_string(index);
    }
    std::ranges::replace_if(
        name,
        [](unsigned char c) { return std::isalnum(c) == 0 && c != '-' && c != '_' && c != '.'; },
        '_');
    const size_t baseNameEnd = std::min(name.find('.'), name.size());
    if (isWindowsDeviceName(name.substr(0, baseNameEnd))) {
      name.insert(baseNameEnd, "_");
    }
    name = makeUniqueName(name, used);
    used.names.insert(toLowerCase(name));
    result.push_back(std::move(name));
  }
  return result;
}

// The extension of the output files, with the dot, in lower case. The format can be given with
// or without the dot.
std::string outputExtension(const SplitOptions& options) {
  if (options.format.empty()) {
    return toLowerCase(filesystem::path(options.options.input_motion_file).extension().string());
  }
  const std::string format = toLowerCase(options.format);
  return format.starts_with('.') ? format : "." + format;
}

} // namespace

int main(int argc, char** argv) {
  try {
    CLI::App app("Saves each character of a file with its motion to a file of its own");
    auto options = setupOptions(app);
    CLI11_PARSE(app, argc, argv);

    if (!isMotionInputFile(options->options.input_motion_file)) {
      MT_LOGE("Unknown input file format: {}", options->options.input_motion_file);
      return EXIT_FAILURE;
    }
    const std::string extension = outputExtension(*options);
    if (!isSceneOutputFile("output" + extension)) {
      MT_LOGE("Unknown output file format: {}", extension);
      return EXIT_FAILURE;
    }

    Scene scene = loadScene(options->options);
    const std::vector<std::string> names = fileNames(scene);
    const filesystem::path directory(options->outputDirectory);
    filesystem::create_directories(directory);
    const SaveSceneOptions saveOptions = saveSceneOptions(options->options);
    for (size_t index = 0; index < scene.characters.size(); ++index) {
      const filesystem::path path = directory / (names[index] + extension);
      MT_LOGI("Character '{}': {}", scene.characters[index].character.name, path.string());
      Scene single;
      single.fps = scene.fps;
      single.hasMotionFile = scene.hasMotionFile;
      single.characters.push_back(std::move(scene.characters[index]));
      saveScene(path, single, saveOptions);
    }
  } catch (std::exception& e) {
    MT_LOGE("Failed to split the characters. Error: {}", e.what());
    return EXIT_FAILURE;
  } catch (...) {
    MT_LOGE("Unknown error encountered.");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
