/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define DEFAULT_LOG_CHANNEL "convert_model"

#include <momentum/character/character.h>
#include <momentum/character/inverse_parameter_transform.h>
#include <momentum/common/log.h>
#include <momentum/io/character_io.h>
#include <momentum/io/fbx/fbx_io.h>
#include <momentum/io/gltf/gltf_io.h>
#include <momentum/io/motion/mmo_io.h>
#include <momentum/io/skeleton/locator_io.h>
#include <momentum/io/skeleton/parameter_transform_io.h>
#include <momentum/io/skeleton/parameters_io.h>

#include <CLI/CLI.hpp>

using namespace momentum;

namespace {

struct Options {
  std::string input_model_file;
  std::string input_params_file;
  std::string input_locator_file;
  std::string input_motion_file;
  std::string output_model_file;
  std::string output_locator_local;
  std::string output_locator_global;
  bool save_markers = false;
  bool character_mesh_save = false;
};

std::shared_ptr<Options> setupOptions(CLI::App& app) {
  auto opt = std::make_shared<Options>();

  // add options and flags
  app.add_option(
         "-m,--model",
         opt->input_model_file,
         "Input model (.fbx/.glb); not required if reading animation from glb or fbx")
      ->check(CLI::ExistingFile);
  app.add_option("-p,--parameters", opt->input_params_file, "Input model parameter file (.model)")
      ->check(CLI::ExistingFile);
  app.add_option("-l,--locator", opt->input_locator_file, "Input locator file (.locators)")
      ->check(CLI::ExistingFile);

  app.add_option("-d,--motion", opt->input_motion_file, "Input motion data file (.mmo/.glb/.fbx)")
      ->check(CLI::ExistingFile);

  app.add_option("-o,--out", opt->output_model_file, "Output file (.fbx/.glb)")->required();

  app.add_option(
      "--out-locator-local",
      opt->output_locator_local,
      "Output a locator file (.locators) in local space for transfering between identities");
  app.add_option(
      "--out-locator-global",
      opt->output_locator_global,
      "Output a locator file (.locators) in global space for authoring a template");

  app.add_flag(
      "--save-markers",
      opt->save_markers,
      "Save marker data from motion file in output (glb only)");
  app.add_flag(
      "-c,--character-mesh",
      opt->character_mesh_save,
      "(FBX Output file only) Saves the Character Mesh to the output file.");

  return opt;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const std::string appName(argv[0]);
    CLI::App app(appName);
    auto options = setupOptions(app);
    CLI11_PARSE(app, argc, argv);

    // Verify output file extension is supported
    const filesystem::path output(options->output_model_file);
    const auto oextension = output.extension();
    if (oextension != ".fbx" && oextension != ".glb" && oextension != ".gltf") {
      MT_LOGE("Unknown output file format: {}", options->output_model_file);
      return EXIT_FAILURE;
    }

    // load character and markers
    Character character;
    const bool hasModel = !options->input_model_file.empty();
    if (hasModel) {
      character = loadFullCharacter(
          options->input_model_file, options->input_params_file, options->input_locator_file);
    }

    // load motion file if it exists
    MatrixXf poses;
    VectorXf offsets;
    float fps = 120.0;
    const bool hasMotion = !options->input_motion_file.empty();

    MarkerSequence markerSequence;
    bool saveMarkers = options->save_markers;
    if (saveMarkers && oextension == ".fbx") {
      MT_LOGW("We cannot save marker data in .fbx yet, sorry!");
      saveMarkers = false;
    }

    if (hasMotion) {
      const auto motionPath = filesystem::path(options->input_motion_file);
      const auto motionExt = motionPath.extension();
      if (motionExt == ".mmo") {
        MT_LOGI("Loading motion from mmo...");
        MT_THROW_IF(!hasModel, "mmo file requires an input character.");
        std::tie(poses, offsets) = loadMmo(motionPath.string(), character);

        if (saveMarkers) {
          MT_LOGW("No marker data in .mmo file {}", motionPath.string());
        }
      } else if (motionExt == ".glb") {
        MT_LOGI("Loading motion from glb...");
        if (hasModel) {
          std::tie(poses, offsets, fps) = loadMotionOnCharacter(motionPath, character);
        } else {
          std::tie(character, poses, offsets, fps) = loadCharacterWithMotion(motionPath);
          if (!options->input_params_file.empty()) {
            MT_LOGW("Ignoring input parameter transform {}.", options->input_params_file);
          }
          if (!options->input_locator_file.empty()) {
            MT_LOGW("Ignoring input locators {}.", options->input_locator_file);
          }
        }

        if (saveMarkers) {
          markerSequence = loadMarkerSequence(motionPath);
        }
      } else if (motionExt == ".fbx") {
        MT_LOGI("Loading motion from fbx...");
        int motionIndex = -1;
        auto [c, motions, framerate] =
            loadFbxCharacterWithMotion(motionPath, KeepLocators::Yes, false);
        // Validate the motion
        if (motions.empty() || (motions.size() == 1 && motions.at(0).cols() == 0)) {
          MT_LOGW("No motion loaded from file");
        } else if (!motions.empty()) {
          size_t nFrames = 0;
          for (size_t iMotion = 0; iMotion < motions.size(); ++iMotion) {
            const size_t length = motions.at(iMotion).cols();
            if (length > nFrames) {
              nFrames = length;
              motionIndex = iMotion;
            }
          }
          if (nFrames > 0 && motions.size() > 1) {
            MT_LOGW("More than one motion found; only taking the longest one");
          }
        }
        // Initialize the character properly
        if (!hasModel) {
          character = c;
          // create parameter transform
          if (!options->input_params_file.empty()) {
            auto def = loadMomentumModel(options->input_params_file);
            loadParameters(def, character);
          } else {
            character.parameterTransform =
                ParameterTransform::identity(character.skeleton.getJointNames());
          }
          // create locators
          if (!options->input_locator_file.empty()) {
            character.locators = loadLocators(
                options->input_locator_file, character.skeleton, character.parameterTransform);
          }
        }
        // Validate model compatibility
        if (c.skeleton.joints.size() != character.skeleton.joints.size()) {
          MT_LOGE("The motion is not on a compatible character");
        } else if (motionIndex >= 0) {
          if (character.parameterTransform.numAllModelParameters() == motions.at(0).rows()) {
            poses = std::move(motions.at(motionIndex));
          } else {
            // Use inverse parameter transform to convet from joint params to model params; may lose
            // info.
            const auto motion = motions.at(motionIndex);
            const size_t nFrames = motion.cols();
            poses.setZero(character.parameterTransform.numAllModelParameters(), nFrames);
            InverseParameterTransform inversePt(character.parameterTransform);
            for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
              poses.col(iFrame) = inversePt.apply(motion.col(iFrame)).pose.v;
            }
          }
          fps = framerate;
          offsets = character.parameterTransform.zero().v;
        }

        if (saveMarkers) {
          MT_LOGW("No marker data in .fbx file {}", motionPath.string());
        }
      } else {
        MT_LOGW(
            "Unknown motion file format: {}. Exporting without motion.",
            options->input_motion_file);
      }
    } else if (saveMarkers) {
      MT_LOGW("No motion file to read marker data from");
    }

    // save output
    if (oextension == ".fbx") {
      MT_LOGI("Saving fbx file...");
      saveFbx(
          options->output_model_file, character, poses, offsets, fps, options->character_mesh_save);
    } else if (oextension == ".glb" || oextension == ".gltf") {
      MT_LOGI("Saving gltf/glb file...");
      if (hasMotion) {
        saveGltfCharacter(
            options->output_model_file,
            character,
            fps,
            {character.parameterTransform.name, poses},
            {character.skeleton.getJointNames(), offsets},
            markerSequence.frames);
      } else {
        saveGltfCharacter(options->output_model_file, character);
      }
    }
    if (!options->output_locator_local.empty()) {
      saveLocators(
          options->output_locator_local,
          character.locators,
          character.skeleton,
          LocatorSpace::Local);
    }
    if (!options->output_locator_global.empty()) {
      saveLocators(
          options->output_locator_global,
          character.locators,
          character.skeleton,
          LocatorSpace::Global);
    }
  } catch (std::exception& e) {
    MT_LOGE("Failed to convert model. Error: {}", e.what());
    return EXIT_FAILURE;
  } catch (...) {
    MT_LOGE("Unknown error encountered.");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
