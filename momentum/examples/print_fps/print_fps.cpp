/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define DEFAULT_LOG_CHANNEL "print_fps"

#include <momentum/common/log.h>
#include <momentum/io/gltf/gltf_io.h>

#include <CLI/CLI.hpp>

using namespace momentum;

namespace {

struct Options {
  std::string input_file;
};

std::shared_ptr<Options> setupOptions(CLI::App& app) {
  auto opt = std::make_shared<Options>();

  app.add_option("input", opt->input_file, "Input GLB/GLTF file")
      ->required()
      ->check(CLI::ExistingFile);

  return opt;
}

} // namespace

int main(int argc, char* argv[]) {
  try {
    CLI::App app("Print FPS from GLB/GLTF file");
    auto options = setupOptions(app);
    CLI11_PARSE(app, argc, argv);

    // Load motion data and extract FPS
    const auto [motion, identity, fps] = loadMotion(options->input_file);

    if (fps > 0.0f) {
      std::cout << "FPS: " << fps << std::endl;
    } else {
      MT_LOGW("No FPS information found in file");
      return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
  } catch (std::exception& e) {
    MT_LOGE("Failed to read FPS from file. Error: {}", e.what());
    return EXIT_FAILURE;
  } catch (...) {
    MT_LOGE("Unknown error encountered.");
    return EXIT_FAILURE;
  }
}
