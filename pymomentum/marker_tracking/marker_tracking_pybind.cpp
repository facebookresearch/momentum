/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/marker.h>
#include <momentum/marker_tracking/marker_tracker.h>
#include <momentum/marker_tracking/process_markers.h>
#include <momentum/marker_tracking/tracker_utils.h>
#include <momentum/math/mesh.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fmt/format.h>
#include <sstream>
#include <string>

namespace py = pybind11;

namespace {

// Helper function to convert a vector of floats to string representation
std::string vectorToString(const Eigen::VectorXf& vec) {
  std::ostringstream ss;
  if (vec.size() > 0) {
    ss << "[";
    for (int i = 0; i < std::min(3, (int)vec.size()); i++) {
      ss << vec[i];
      if (i < std::min(2, (int)vec.size() - 1)) {
        ss << ", ";
      }
    }
    if (vec.size() > 3) {
      ss << ", ... (" << vec.size() << " total)";
    }
    ss << "]";
  } else {
    ss << "[]";
  }
  return ss.str();
}

// Helper function to convert a boolean to Python-style string representation
std::string boolToString(bool value) {
  return value ? "True" : "False";
}

} // namespace

// Python bindings for marker tracking APIs defined under:
// //arvr/libraries/momentum/marker_tracking

// @dep=fbsource//arvr/libraries/dispenso:dispenso

PYBIND11_MODULE(marker_tracking, m) {
  m.doc() = "Module for exposing the C++ APIs of the marker tracking pipeline ";
  m.attr("__name__") = "pymomentum.marker_tracking";

  pybind11::module_::import(
      "pymomentum.geometry"); // @dep=fbcode//pymomentum:geometry

  // Bindings for types defined in marker_tracking/marker_tracker.h
  auto baseConfig = py::class_<momentum::BaseConfig>(
      m, "BaseConfig", "Represents base config class");

  baseConfig.def(py::init<>())
      .def(
          "__repr__",
          [](const momentum::BaseConfig& self) {
            return fmt::format(
                "BaseConfig(min_vis_percent={}, loss_alpha={}, max_iter={}, debug={})",
                self.minVisPercent,
                self.lossAlpha,
                self.maxIter,
                boolToString(self.debug));
          })
      .def(
          py::init<float, float, size_t, bool>(),
          py::arg("min_vis_percent") = 0.0,
          py::arg("loss_alpha") = 2.0,
          py::arg("max_iter") = 30,
          py::arg("debug") = false)
      .def_readwrite(
          "min_vis_percent",
          &momentum::BaseConfig::minVisPercent,
          "Minimum percentage of visible markers to be used")
      .def_readwrite(
          "loss_alpha",
          &momentum::BaseConfig::lossAlpha,
          "Parameter to control the loss function")
      .def_readwrite(
          "max_iter", &momentum::BaseConfig::maxIter, "Max iterations")
      .def_readwrite(
          "debug",
          &momentum::BaseConfig::debug,
          "Whether to output debugging info");

  auto calibrationConfig =
      py::class_<momentum::CalibrationConfig, momentum::BaseConfig>(
          m, "CalibrationConfig", "Config for the body scale calibration step");

  // Default values are set from the configured values in marker_tracker.h
  calibrationConfig.def(py::init<>())
      .def(
          "__repr__",
          [](const momentum::CalibrationConfig& self) {
            return fmt::format(
                "CalibrationConfig(min_vis_percent={}, loss_alpha={}, max_iter={}, debug={}, calib_frames={}, major_iter={}, global_scale_only={}, locators_only={}, greedy_sampling={}, enforce_floor_in_first_frame={}, first_frame_pose_constraint_set=\"{}\")",
                self.minVisPercent,
                self.lossAlpha,
                self.maxIter,
                boolToString(self.debug),
                self.calibFrames,
                self.majorIter,
                boolToString(self.globalScaleOnly),
                boolToString(self.locatorsOnly),
                self.greedySampling,
                boolToString(self.enforceFloorInFirstFrame),
                self.firstFramePoseConstraintSet);
          })
      .def(
          py::init<
              float,
              float,
              size_t,
              bool,
              size_t,
              size_t,
              bool,
              bool,
              size_t,
              bool,
              std::string>(),
          py::arg("min_vis_percent") = 0.0,
          py::arg("loss_alpha") = 2.0,
          py::arg("max_iter") = 30,
          py::arg("debug") = false,
          py::arg("calib_frames") = 100,
          py::arg("major_iter") = 3,
          py::arg("global_scale_only") = false,
          py::arg("locators_only") = false,
          py::arg("greedy_sampling") = 0,
          py::arg("enforce_floor_in_first_frame") = false,
          py::arg("first_frame_pose_constraint_set") = "")
      .def_readwrite(
          "calib_frames",
          &momentum::CalibrationConfig::calibFrames,
          "Number of frames used for model calibration")
      .def_readwrite(
          "greedy_sampling",
          &momentum::CalibrationConfig::greedySampling,
          "Enable greedy frame sampling with the given stride")
      .def_readwrite(
          "major_iter",
          &momentum::CalibrationConfig::majorIter,
          "Number of calibration loops to run")
      .def_readwrite(
          "global_scale_only",
          &momentum::CalibrationConfig::globalScaleOnly,
          "Calibrate only the global scale and not all proportions")
      .def_readwrite(
          "locators_only",
          &momentum::CalibrationConfig::locatorsOnly,
          "Calibrate only the locator offsets")
      .def_readwrite(
          "enforce_floor_in_first_frame",
          &momentum::CalibrationConfig::enforceFloorInFirstFrame,
          "Force floor contact in first frame")
      .def_readwrite(
          "first_frame_pose_constraint_set",
          &momentum::CalibrationConfig::firstFramePoseConstraintSet,
          "Name of pose constraint set to use in first frame");

  auto trackingConfig =
      py::class_<momentum::TrackingConfig, momentum::BaseConfig>(
          m, "TrackingConfig", "Config for the tracking optimization step");

  trackingConfig.def(py::init<>())
      .def(
          "__repr__",
          [](const momentum::TrackingConfig& self) {
            return fmt::format(
                "TrackingConfig(min_vis_percent={}, loss_alpha={}, max_iter={}, debug={}, smoothing={}, collision_error_weight={}, smoothing_weights={})",
                self.minVisPercent,
                self.lossAlpha,
                self.maxIter,
                boolToString(self.debug),
                self.smoothing,
                self.collisionErrorWeight,
                vectorToString(self.smoothingWeights));
          })
      .def(
          py::init<float, float, size_t, bool, float, float, Eigen::VectorXf>(),
          py::arg("min_vis_percent") = 0.0,
          py::arg("loss_alpha") = 2.0,
          py::arg("max_iter") = 30,
          py::arg("debug") = false,
          py::arg("smoothing") = 0.0,
          py::arg("collision_error_weight") = 0.0,
          py::arg("smoothing_weights") = Eigen::VectorXf())
      .def_readwrite(
          "smoothing",
          &momentum::TrackingConfig::smoothing,
          "Smoothing weight; 0 to disable")
      .def_readwrite(
          "collision_error_weight",
          &momentum::TrackingConfig::collisionErrorWeight,
          "Collision error weight; 0 to disable")
      .def_readwrite(
          "smoothing_weights",
          &momentum::TrackingConfig::smoothingWeights,
          R"(Smoothing weights per model parameter. The size of this vector should be
            equal to number of model parameters and this overrides the value specific in smoothing)");

  auto refineConfig =
      py::class_<momentum::RefineConfig, momentum::TrackingConfig>(
          m, "RefineConfig", "Config for refining a tracked motion.");

  refineConfig.def(py::init<>())
      .def(
          "__repr__",
          [](const momentum::RefineConfig& self) {
            return fmt::format(
                "RefineConfig(min_vis_percent={}, loss_alpha={}, max_iter={}, debug={}, smoothing={}, collision_error_weight={}, smoothing_weights={}, regularizer={}, calib_id={}, calib_locators={})",
                self.minVisPercent,
                self.lossAlpha,
                self.maxIter,
                boolToString(self.debug),
                self.smoothing,
                self.collisionErrorWeight,
                vectorToString(self.smoothingWeights),
                self.regularizer,
                boolToString(self.calibId),
                boolToString(self.calibLocators));
          })
      .def(
          py::init<
              float,
              float,
              size_t,
              bool,
              float,
              float,
              Eigen::VectorXf,
              float,
              bool,
              bool>(),
          py::arg("min_vis_percent") = 0.0,
          py::arg("loss_alpha") = 2.0,
          py::arg("max_iter") = 30,
          py::arg("debug") = false,
          py::arg("smoothing") = 0.0,
          py::arg("collision_error_weight") = 0.0,
          py::arg("smoothing_weights") = Eigen::VectorXf(),
          py::arg("regularizer") = 0.0,
          py::arg("calib_id") = false,
          py::arg("calib_locators") = false)
      .def_readwrite(
          "regularizer",
          &momentum::RefineConfig::regularizer,
          "Regularize the time-invariant parameters to prevent large changes.")
      .def_readwrite(
          "calib_id",
          &momentum::RefineConfig::calibId,
          "Calibrate identity parameters; default to False.")
      .def_readwrite(
          "calib_locators",
          &momentum::RefineConfig::calibLocators,
          "Calibrate locator offsets; default to False.");

  auto modelOptions = py::class_<momentum::ModelOptions>(
      m,
      "ModelOptions",
      "Model options to specify the template model, parameter transform and locator mappings");

  modelOptions.def(py::init<>())
      .def(
          "__repr__",
          [](const momentum::ModelOptions& self) {
            return fmt::format(
                "ModelOptions(model=\"{}\", parameters=\"{}\", locators=\"{}\")",
                self.model,
                self.parameters,
                self.locators);
          })
      .def(
          py::init<
              const std::string&,
              const std::string&,
              const std::string&>(),
          py::arg("model"),
          py::arg("parameters"),
          py::arg("locators"))
      .def_readwrite(
          "model",
          &momentum::ModelOptions::model,
          "Path to template model file with locators e.g. character.glb")
      .def_readwrite(
          "parameters",
          &momentum::ModelOptions::parameters,
          "Path of parameter transform model file e.g. character.model")
      .def_readwrite(
          "locators",
          &momentum::ModelOptions::locators,
          "Path to locator mapping file e.g. character.locators");
  m.def(
      "process_marker_file",
      &momentum::processMarkerFile,
      py::arg("input_marker_file"),
      py::arg("output_file"),
      py::arg("tracking_config"),
      py::arg("calibration_config"),
      py::arg("model_options"),
      py::arg("calibrate"),
      py::arg("first_frame") = 0,
      py::arg("max_frames") = 0);

  m.def(
      "process_markers",
      [](momentum::Character& character,
         const Eigen::VectorXf& identity,
         const std::vector<std::vector<momentum::Marker>>& markerData,
         const momentum::TrackingConfig& trackingConfig,
         const momentum::CalibrationConfig& calibrationConfig,
         bool calibrate = true,
         size_t firstFrame = 0,
         size_t maxFrames = 0) {
        momentum::ModelParameters params(identity);

        if (params.size() == 0) { // If no identity is passed in, use default
          params = momentum::ModelParameters::Zero(
              character.parameterTransform.name.size());
        }

        Eigen::MatrixXf motion = momentum::processMarkers(
            character,
            params,
            markerData,
            trackingConfig,
            calibrationConfig,
            calibrate,
            firstFrame,
            maxFrames);

        // python and cpp have the motion matrix transposed from each other:
        // python (#frames, #params) vs. cpp (#params, #frames)
        return motion.transpose().eval();
      },
      R"(process markers given character and identity.

:parameter character: Character to be used for tracking
:parameter identity: Identity parameters, pass in empty array for default identity
:parameter marker_data: A list of marker data for each frame
:parameter tracking_config: Tracking config to be used for tracking
:parameter calibration_config: Calibration config to be used for calibration
:parameter calibrate: Whether to calibrate the model
:parameter first_frame: First frame to be processed
:parameter max_frames: Max number of frames to be processed
:return: Transform parameters for each frame)",
      py::arg("character"),
      py::arg("identity"),
      py::arg("marker_data"),
      py::arg("tracking_config"),
      py::arg("calibration_config"),
      py::arg("calibrate") = true,
      py::arg("first_frame") = 0,
      py::arg("max_frames") = 0);

  m.def(
      "save_motion",
      [](const std::string& outFile,
         const momentum::Character& character,
         const Eigen::VectorXf& identity,
         Eigen::MatrixXf& motion,
         const std::vector<std::vector<momentum::Marker>>& markerData,
         const float fps,
         const bool saveMarkerMesh = true) {
        momentum::ModelParameters params(identity);

        if (params.size() == 0) { // If no identity is passed in, use default
          params = momentum::ModelParameters::Zero(
              character.parameterTransform.name.size());
        }

        // python and cpp have the motion matrix transposed from each other:
        // python (#frames, #params) vs. cpp (#params, #frames)
        if (motion.cols() ==
            character.parameterTransform.numAllModelParameters()) {
          // we need to transpose the matrix before passing it to the cpp
          Eigen::MatrixXf finalMotion(motion.transpose());
          // note: saveMotion removes identity from the motion matrix
          momentum::saveMotion(
              outFile,
              character,
              params,
              finalMotion,
              markerData,
              fps,
              saveMarkerMesh);
          // and transpose it back since motion is passed by reference
          motion = finalMotion.transpose();
        } else if (
            motion.rows() ==
            character.parameterTransform.numAllModelParameters()) {
          // motion matrix is already in cpp format
          // keeping this branch for backward compatibility
          // note: saveMotion removes identity from the motion matrix
          momentum::saveMotion(
              outFile,
              character,
              params,
              motion,
              markerData,
              fps,
              saveMarkerMesh);
        } else {
          throw std::runtime_error(
              "Inconsistent number of parameters in motion matrix with the character parameter transform");
        }
      },
      py::arg("out_file"),
      py::arg("character"),
      py::arg("identity"),
      py::arg("motion"),
      py::arg("marker_data"),
      py::arg("fps"),
      py::arg("save_marker_mesh") = true);

  m.def(
      "refine_motion",
      [](momentum::Character& character,
         const Eigen::VectorXf& identity,
         const Eigen::MatrixXf& motion,
         const std::vector<std::vector<momentum::Marker>>& markerData,
         const momentum::RefineConfig& refineConfig) {
        // python and cpp have the motion matrix transposed from each other.
        // Let's do that on the way in and out here so it's consistent for both
        // languages.
        Eigen::MatrixXf inputMotion(motion.transpose());

        // If input identity is not empty, it means the motion is stripped of
        // identity field (eg. read from a glb file), so we need to fill it in.
        // If the input identity is empty, we assume the identity fields already
        // exist in the motion matrix.
        if (identity.size() > 0) {
          momentum::ParameterSet idParamSet =
              character.parameterTransform.getScalingParameters();
          momentum::fillIdentity(idParamSet, identity, inputMotion);
        }
        Eigen::MatrixXf finalMotion = momentum::refineMotion(
            markerData, inputMotion, refineConfig, character);
        auto finalMotionTransposed = Eigen::MatrixXf(finalMotion.transpose());
        return finalMotionTransposed;
      },
      py::arg("character"),
      py::arg("identity"),
      py::arg("motion"),
      py::arg("marker_data"),
      py::arg("refine_config"));

  m.def(
      "convert_locators_to_skinned_locators",
      &momentum::locatorsToSkinnedLocators,
      R"(Convert regular locators to skinned locators based on mesh proximity.

This function converts locators attached to specific joints into skinned locators
that are weighted across multiple joints based on the underlying mesh skin weights.
For each locator, it:

1. Computes the locator's world space position using the rest skeleton state
2. Finds the closest point on the character's mesh surface that is skinned to the same
   bone as the locator (this is to avoid skinning the locator to the wrong bone)
3. If the distance is within max_distance, converts the locator to a skinned locator
   with bone weights interpolated from the closest mesh triangle
4. Otherwise, keeps the original locator unchanged

The resulting skinned locators maintain the same world space position but are now
influenced by multiple joints through skin weights.

:param character: Character with mesh, skin weights, and locators to convert
:param max_distance: Maximum distance from mesh surface to convert a locator (default: 3.0)
:return: New character with converted skinned locators and remaining regular locators)",
      py::arg("character"),
      py::arg("max_distance") = 3.0f);
}
