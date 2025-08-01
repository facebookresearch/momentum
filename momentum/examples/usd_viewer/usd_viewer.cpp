/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/character.h>
#include <momentum/character/character_state.h>
#include <momentum/character/skin_weights.h>
#include <momentum/common/filesystem.h>
#include <momentum/common/log.h>
#include <momentum/gui/rerun/logger.h>
#include <momentum/gui/rerun/logging_redirect.h>
#include <momentum/io/usd/usd_io.h>
#include <momentum/math/mesh.h>

#include <CLI/CLI.hpp>
#include <rerun.hpp>

#include <string>

using namespace rerun;
using namespace momentum;

namespace {

struct Options {
  std::string usdFile;
  LogLevel logLevel = LogLevel::Info;
  std::string title;
  bool logJoints = false;
};

std::shared_ptr<Options> setupOptions(CLI::App& app) {
  auto opt = std::make_shared<Options>();
  app.add_option("--title", opt->title, "Title in viewer (default to be filename)");
  app.add_option("-i,--input", opt->usdFile, "Path to the USD file")->required();
  app.add_option("-l,--loglevel", opt->logLevel, "Set the log level")
      ->transform(CLI::CheckedTransformer(logLevelMap(), CLI::ignore_case))
      ->default_val(opt->logLevel);
  app.add_flag("--log-joints", opt->logJoints, "Log joint parameters (very slow)")
      ->default_val(opt->logJoints);
  return opt;
}

} // namespace

int main(int argc, char* argv[]) {
  try {
    CLI::App app("USD Viewer");
    auto options = setupOptions(app);
    CLI11_PARSE(app, argc, argv);

    setLogLevel(options->logLevel);

    filesystem::path filePath(options->usdFile);
    if (!filePath.is_absolute()) {
      filesystem::path currentPath = filesystem::current_path();
      filesystem::path fbsourcePath;

      for (const auto& part : currentPath) {
        fbsourcePath /= part;
        if (part == "fbsource") {
          break;
        }
      }

      if (!fbsourcePath.empty() && filesystem::exists(fbsourcePath)) {
        filePath = fbsourcePath / options->usdFile;
      }
    }

    if (!filesystem::exists(filePath)) {
      MT_LOGE("--input: File does not exist: {}", filePath.string());
      return EXIT_FAILURE;
    }

    const std::string fileName = filePath.filename().string();
    const auto extension = filePath.extension().string();
    if (extension != ".usd" && extension != ".usda" && extension != ".usdc" &&
        extension != ".usdz") {
      MT_LOGE("{} is not a supported USD format.", fileName);
      return EXIT_FAILURE;
    }

    const auto character = loadUsdCharacter(filePath.string());

    MT_LOGI(
        "Loaded character with {} joints, {} vertices, {} faces",
        character.skeleton.joints.size(),
        character.mesh ? character.mesh->vertices.size() : 0,
        character.mesh ? character.mesh->faces.size() : 0);

    const std::string title = options->title.empty() ? fileName : options->title;
    const auto rec = RecordingStream(title);
    rec.spawn().exit_on_failure();

    redirectLogsToRerun(rec);

    rec.log_static("world", ViewCoordinates(components::ViewCoordinates::RIGHT_HAND_Z_UP));

    try {
      rec.set_time_sequence("frame_index", 0);
      rec.set_time_seconds("log_time", 0.0f);

      if (character.mesh && !character.mesh->vertices.empty()) {
        if (character.mesh->normals.empty()) {
          character.mesh->updateNormals();
        }
        logMesh(rec, "world/character/mesh", *character.mesh);
      }

      if (!character.skeleton.joints.empty()) {
        std::vector<rerun::Position3D> jointPositions;
        std::vector<std::string> jointLabels;

        for (const auto& joint : character.skeleton.joints) {
          jointPositions.emplace_back(
              joint.translationOffset.x(),
              joint.translationOffset.y(),
              joint.translationOffset.z());
          jointLabels.push_back(joint.name);
        }

        rec.log(
            "world/character/joints",
            rerun::Points3D(jointPositions)
                .with_radii(0.02f)
                .with_colors(rerun::Color(255, 100, 100))
                .with_labels(jointLabels));
      }

    } catch (const std::exception& e) {
      MT_LOGE("Error logging character data: {}", e.what());
      return EXIT_FAILURE;
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
