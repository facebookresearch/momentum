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

#include <string>

namespace py = pybind11;

// Python bindings for marker tracking APIs defined under:
// //arvr/libraries/momentum/marker_tracking

// @dep=fbsource//arvr/libraries/dispenso:dispenso

PYBIND11_MODULE(marker_tracking, m) {
  m.doc() = "Module for exposing the C++ APIs of the marker tracking pipeline ";
  m.attr("__name__") = "pymomentum.marker_tracking";

  pybind11::module_::import(
      "pymomentum.geometry"); // @dep=fbcode//pymomentum:geometry

  // Bindings for types defined in marker_tracking/marker_tracker.h
  auto baseConfig = py::class_<marker_tracking::BaseConfig>(
      m, "BaseConfig", "Represents base config class");

  baseConfig.def(py::init<>())
      .def(
          py::init<float, float, size_t, bool>(),
          py::arg("min_vis_percent") = 0.0,
          py::arg("loss_alpha") = 2.0,
          py::arg("max_iter") = 30,
          py::arg("debug") = false)
      .def_readwrite(
          "min_vis_percent",
          &marker_tracking::BaseConfig::minVisPercent,
          "Minimum percentage of visible markers to be used")
      .def_readwrite(
          "loss_alpha",
          &marker_tracking::BaseConfig::lossAlpha,
          "Parameter to control the loss function")
      .def_readwrite(
          "max_iter", &marker_tracking::BaseConfig::maxIter, "Max iterations")
      .def_readwrite(
          "debug",
          &marker_tracking::BaseConfig::debug,
          "Whether to output debugging info");

  auto calibrationConfig = py::
      class_<marker_tracking::CalibrationConfig, marker_tracking::BaseConfig>(
          m, "CalibrationConfig", "Config for the body scale calibration step");

  // Default values are set from the configured values in marker_tracker.h
  calibrationConfig.def(py::init<>())
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
          &marker_tracking::CalibrationConfig::calibFrames,
          "Number of frames used for model calibration")
      .def_readwrite(
          "greedy_sampling",
          &marker_tracking::CalibrationConfig::greedySampling,
          "Enable greedy frame sampling with the given stride")
      .def_readwrite(
          "major_iter",
          &marker_tracking::CalibrationConfig::majorIter,
          "Number of calibration loops to run")
      .def_readwrite(
          "global_scale_only",
          &marker_tracking::CalibrationConfig::globalScaleOnly,
          "Calibrate only the global scale and not all proportions")
      .def_readwrite(
          "locators_only",
          &marker_tracking::CalibrationConfig::locatorsOnly,
          "Calibrate only the locator offsets")
      .def_readwrite(
          "enforce_floor_in_first_frame",
          &marker_tracking::CalibrationConfig::enforceFloorInFirstFrame,
          "Force floor contact in first frame")
      .def_readwrite(
          "first_frame_pose_constraint_set",
          &marker_tracking::CalibrationConfig::firstFramePoseConstraintSet,
          "Name of pose constraint set to use in first frame");

  auto trackingConfig =
      py::class_<marker_tracking::TrackingConfig, marker_tracking::BaseConfig>(
          m, "TrackingConfig", "Config for the tracking optimization step");

  trackingConfig.def(py::init<>())
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
          &marker_tracking::TrackingConfig::smoothing,
          "Smoothing weight; 0 to disable")
      .def_readwrite(
          "collision_error_weight",
          &marker_tracking::TrackingConfig::collisionErrorWeight,
          "Collision error weight; 0 to disable")
      .def_readwrite(
          "smoothing_weights",
          &marker_tracking::TrackingConfig::smoothingWeights,
          R"(Smoothing weights per model parameter. The size of this vector should be
            equal to number of model parameters and this overrides the value specific in smoothing)");

  auto refineConfig = py::
      class_<marker_tracking::RefineConfig, marker_tracking::TrackingConfig>(
          m, "RefineConfig", "Config for refining a tracked motion.");

  refineConfig.def(py::init<>())
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
          &marker_tracking::RefineConfig::regularizer,
          "Regularize the time-invariant parameters to prevent large changes.")
      .def_readwrite(
          "calib_id",
          &marker_tracking::RefineConfig::calibId,
          "Calibrate identity parameters; default to False.")
      .def_readwrite(
          "calib_locators",
          &marker_tracking::RefineConfig::calibLocators,
          "Calibrate locator offsets; default to False.");

  auto modelOptions = py::class_<marker_tracking::ModelOptions>(
      m,
      "ModelOptions",
      "Model options to specify the template model, parameter transform and locator mappings");

  modelOptions.def(py::init<>())
      .def(py::init<
           const std::string&,
           const std::string&,
           const std::string&>())
      .def_readwrite(
          "model",
          &marker_tracking::ModelOptions::model,
          "Path to template model file with locators e.g. character.glb")
      .def_readwrite(
          "parameters",
          &marker_tracking::ModelOptions::parameters,
          "Path of parameter transform model file e.g. character.model")
      .def_readwrite(
          "locators",
          &marker_tracking::ModelOptions::locators,
          "Path to locator mapping file e.g. character.locators");
  m.def(
      "process_marker_file",
      &marker_tracking::processMarkerFile,
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
         const marker_tracking::TrackingConfig& trackingConfig,
         const marker_tracking::CalibrationConfig& calibrationConfig,
         bool calibrate = true,
         size_t firstFrame = 0,
         size_t maxFrames = 0) {
        momentum::ModelParameters params(identity);

        if (params.size() == 0) { // If no identity is passed in, use default
          params = momentum::ModelParameters::Zero(
              character.parameterTransform.name.size());
        }

        Eigen::MatrixXf motion = marker_tracking::processMarkers(
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
          marker_tracking::saveMotion(
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
          marker_tracking::saveMotion(
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
         const marker_tracking::RefineConfig& refineConfig) {
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
          marker_tracking::fillIdentity(idParamSet, identity, inputMotion);
        }
        Eigen::MatrixXf finalMotion = marker_tracking::refineMotion(
            markerData, inputMotion, refineConfig, character);
        auto finalMotionTransposed = Eigen::MatrixXf(finalMotion.transpose());
        return finalMotionTransposed;
      },
      py::arg("character"),
      py::arg("identity"),
      py::arg("motion"),
      py::arg("marker_data"),
      py::arg("refine_config"));
}
