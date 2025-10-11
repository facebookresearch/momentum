/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/marker_tracking/app_utils.h"

#include "momentum/character/character.h"
#include "momentum/character/inverse_parameter_transform.h"
#include "momentum/io/character_io.h"
#include "momentum/io/fbx/fbx_io.h"
#include "momentum/io/gltf/gltf_builder.h"
#include "momentum/io/gltf/gltf_io.h"
#include "momentum/io/skeleton/locator_io.h"
#include "momentum/marker_tracking/marker_tracker.h"
#include "momentum/marker_tracking/tracker_utils.h"

#include <CLI/CLI.hpp>

using namespace momentum;

namespace momentum {

void addIOOptions(CLI::App& app, std::shared_ptr<IOOptions> ioOptions) {
  auto* inputOption =
      app.add_option("-i,--input", ioOptions->inputFile, "Marker data file (.kbd/.glb/.c3d/.trc)");
  inputOption->required()->check(CLI::ExistingFile);

  auto* outputOption = app.add_option("-o,--output", ioOptions->outputFile, "Output glb file");
  outputOption->required();
}

void addModelOptions(CLI::App& app, std::shared_ptr<ModelOptions> modelOptions) {
  auto* modelOption = app.add_option(
      "-m,--model", modelOptions->model, "Model file (.fbx/.glb -- may contain calibrated model)");
  modelOption->required()->check(CLI::ExistingFile);

  app.add_option("-p,--parameters", modelOptions->parameters, "Model parameter file (.model)");
  app.add_option("-l,--locators", modelOptions->locators, "Marker definitions (.locators)");
}

void addCalibrationOptions(CLI::App& app, std::shared_ptr<CalibrationConfig> config) {
  auto* debugOption =
      app.add_option("--calib-debug", config->debug, "Save debug info for calibration");
  debugOption->default_val(config->debug);

  auto* calibFramesOption = app.add_option(
      "--calib-frames", config->calibFrames, "Number of frames used for model calibration");
  calibFramesOption->default_val(config->calibFrames)->check(CLI::PositiveNumber);

  auto* majorIterOption =
      app.add_option("--major-iter", config->majorIter, "Number of calibration loops to run");
  majorIterOption->default_val(config->majorIter)->check(CLI::PositiveNumber);

  auto* maxCalibIterOption = app.add_option(
      "--max-calib-iter", config->maxIter, "Max (minor) iterations in one calibration solve");
  maxCalibIterOption->default_val(config->maxIter)->check(CLI::PositiveNumber);

  auto* minVisPercentOption = app.add_option(
      "--min-calib-vis-percent",
      config->minVisPercent,
      "Minimum percentage of visible markers used for calibration");
  minVisPercentOption->default_val(config->minVisPercent)->check(CLI::Range(0.0, 1.0));

  auto* alphaOption =
      app.add_option("--calib-loss-alpha", config->lossAlpha, "Alpha value of calibration loss");
  alphaOption->default_val(config->lossAlpha);

  auto* scaleOnlyOption = app.add_option(
      "--global-scale-only",
      config->globalScaleOnly,
      "Calibrate only the global scale and not all proportions");
  scaleOnlyOption->default_val(config->globalScaleOnly);

  auto* locatorsOnlyOption =
      app.add_option("--locators-only", config->locatorsOnly, "Calibrate only the locator offsets");
  locatorsOnlyOption->default_val(config->locatorsOnly);

  auto* greedySamplingOption = app.add_option(
      "--greedy-sampling",
      config->greedySampling,
      "Use greedy sampling to select calibration frames");
  greedySamplingOption->default_val(config->greedySampling)->check(CLI::NonNegativeNumber);

  auto* firstFrameFloorContactOption = app.add_flag(
      "--init-floor-contact",
      config->enforceFloorInFirstFrame,
      "Enforce floor contact constraints in first frame");
  firstFrameFloorContactOption->default_val(config->enforceFloorInFirstFrame);

  auto* poseOption = app.add_option(
      "--init-pose",
      config->firstFramePoseConstraintSet,
      "Enforce pose constraints from named pose constraint set in first frame");
  poseOption->default_val(config->firstFramePoseConstraintSet);
}

void addTrackingOptions(CLI::App& app, std::shared_ptr<TrackingConfig> config) {
  auto* debugOption =
      app.add_option("--tracking-debug", config->debug, "Save debug info for tracking");
  debugOption->default_val(config->debug);

  auto* alphaOption =
      app.add_option("--tracking-loss-alpha", config->lossAlpha, "Alpha value of tracking loss");
  alphaOption->default_val(config->lossAlpha);

  auto* smoothOption =
      app.add_option("--smoothing", config->smoothing, "Smoothing weight; 0 to disable");
  smoothOption->default_val(config->smoothing)->check(CLI::NonNegativeNumber);

  auto* collisionErrorWeightOption = app.add_option(
      "--collision-error-weight",
      config->collisionErrorWeight,
      "Collision error weight, default is 0.0");
  collisionErrorWeightOption->default_val(config->collisionErrorWeight)
      ->check(CLI::NonNegativeNumber);

  auto* maxTrackingIterOption = app.add_option(
      "--max-tracking-iter", config->maxIter, "Max iterations for motion tracking solve");
  maxTrackingIterOption->default_val(config->maxIter)->check(CLI::PositiveNumber);

  auto* minVisPercentOption = app.add_option(
      "--min-tracking-vis-percent",
      config->minVisPercent,
      "Minimum percentage of visible markers used for tracking");
  minVisPercentOption->default_val(config->minVisPercent)->check(CLI::Range(0.0, 1.0));
}

void addRefineOptions(CLI::App& app, std::shared_ptr<RefineConfig> config) {
  addTrackingOptions(app, config);

  auto* regOption = app.add_option(
      "--regularizer",
      config->regularizer,
      "Regularize time-invariant parameters to prevent large change");
  regOption->default_val(config->regularizer)->check(CLI::NonNegativeNumber);

  auto* calibIdOption =
      app.add_option("--calib-id", config->calibId, "Calibrate identity parameters");
  calibIdOption->default_val(config->calibId);

  auto* calibLocatorsOption =
      app.add_option("--calib-locators", config->calibLocators, "Calibrate locator offsets");
  calibLocatorsOption->default_val(config->calibLocators);
}

std::tuple<momentum::Character, momentum::ModelParameters> loadCalibratedModel(
    const std::string& modelFile) {
  auto [c, m, id, fps] = loadCharacterWithMotion(modelFile);

  // make sure we actually only convert joint id back to scale id parameters
  const ModelParameters identity =
      jointIdentityToModelIdentity(c, c.parameterTransform.getScalingParameters(), id);

  return {c, identity};
}

std::tuple<momentum::Character, momentum::ModelParameters> loadCharacterWithIdentity(
    const ModelOptions& modelFiles) {
  Character character;
  ModelParameters identity;

  // TODO: Very hacky!
  if (modelFiles.model.find(".glb") != std::string::npos &&
      modelFiles.parameters
          .empty()) { // Assume input has precomputed proportions and marker offsets etc.
    auto [c, m, id, fps] = loadCharacterWithMotion(modelFiles.model);
    character = c;
    if (id.size() > 0) {
      identity = jointIdentityToModelIdentity(c, c.parameterTransform.getScalingParameters(), id);
    } else {
      identity = ModelParameters::Zero(character.parameterTransform.numAllModelParameters());
    }

    if (!modelFiles.locators.empty()) {
      character.locators =
          loadLocators(modelFiles.locators, character.skeleton, character.parameterTransform);
    }
  } else { // Assume input is the template
    character = loadFullCharacter(modelFiles.model, modelFiles.parameters, modelFiles.locators);
    identity = ModelParameters::Zero(character.parameterTransform.numAllModelParameters());
  }

  return {character, identity};
}

void saveMotion(
    const std::string& outFile,
    const momentum::Character& character,
    const momentum::ModelParameters& identity,
    Eigen::MatrixXf& finalMotion,
    gsl::span<const std::vector<momentum::Marker>> markerData,
    const double fps,
    const bool saveMarkerMesh) {
  const filesystem::path output(outFile);
  const auto ext = output.extension();

  ModelParameters id =
      extractParameters(identity, character.parameterTransform.getScalingParameters());

  // gltf io assumes the identity info is removed from the motion matrix
  removeIdentity(character.parameterTransform.getScalingParameters(), id, finalMotion);
  const VectorXf idVec = character.parameterTransform.apply(id).v;

  if (ext == ".fbx") {
    saveFbx(output, character, finalMotion, idVec, fps, saveMarkerMesh);
  } else if (ext == ".glb" || ext == ".gltf") {
    GltfBuilder fileBuilder;
    fileBuilder.addMotion(
        character,
        fps,
        std::make_tuple(character.parameterTransform.name, finalMotion),
        std::make_tuple(character.skeleton.getJointNames(), idVec));
    fileBuilder.addMarkerSequence(
        fps,
        markerData,
        saveMarkerMesh ? GltfBuilder::MarkerMesh::UnitCube : GltfBuilder::MarkerMesh::None);
    fileBuilder.save(outFile);
  }
}

} // namespace momentum
