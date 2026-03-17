/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/marker.h>
#include <momentum/marker_tracking/glove_utils.h>
#include <momentum/marker_tracking/marker_tracker.h>
#include <momentum/marker_tracking/process_markers.h>
#include <momentum/marker_tracking/tracker_utils.h>
#include <momentum/math/mesh.h>

#include <pymomentum/python_utility/eigen_quaternion.h>

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

#ifdef PYMOMENTUM_LIMITED_TORCH_API
  m.attr("AUTOGRAD_ENABLED") = false;
#else
  m.attr("AUTOGRAD_ENABLED") = true;
#endif

  pybind11::module_::import(
      "pymomentum.geometry"); // @dep=fbsource//arvr/libraries/pymomentum:geometry

  // Bindings for types defined in marker_tracking/marker_tracker.h
  auto baseConfig =
      py::class_<momentum::BaseConfig>(m, "BaseConfig", "Represents base config class");

  baseConfig.def(py::init<>())
      .def(
          "__repr__",
          [](const momentum::BaseConfig& self) {
            return fmt::format(
                "BaseConfig(min_vis_percent={}, loss_alpha={}, max_iter={}, regularization={}, debug={})",
                self.minVisPercent,
                self.lossAlpha,
                self.maxIter,
                self.regularization,
                boolToString(self.debug));
          })
      .def(
          py::init<float, float, size_t, float, bool>(),
          R"(Create a BaseConfig with specified parameters.

          :param min_vis_percent: Minimum percentage of visible markers to be used
          :param loss_alpha: Parameter to control the loss function
          :param max_iter: Maximum number of iterations
          :param debug: Whether to output debugging info
          )",
          py::arg("min_vis_percent") = 0.0,
          py::arg("loss_alpha") = 2.0,
          py::arg("max_iter") = 30,
          py::arg("regularization") = 0.05f,
          py::arg("debug") = false)
      .def_readwrite(
          "min_vis_percent",
          &momentum::BaseConfig::minVisPercent,
          "Minimum percentage of visible markers to be used")
      .def_readwrite(
          "loss_alpha", &momentum::BaseConfig::lossAlpha, "Parameter to control the loss function")
      .def_readwrite("max_iter", &momentum::BaseConfig::maxIter, "Max iterations")
      .def_readwrite("debug", &momentum::BaseConfig::debug, "Whether to output debugging info");

  auto calibrationConfig = py::class_<momentum::CalibrationConfig, momentum::BaseConfig>(
      m, "CalibrationConfig", "Config for the body scale calibration step");

  // Default values are set from the configured values in marker_tracker.h
  calibrationConfig.def(py::init<>())
      .def(
          "__repr__",
          [](const momentum::CalibrationConfig& self) {
            return fmt::format(
                "CalibrationConfig(min_vis_percent={}, loss_alpha={}, max_iter={}, regularization={}, debug={}, calib_frames={}, major_iter={}, global_scale_only={}, locators_only={}, greedy_sampling={}, enforce_floor_in_first_frame={}, first_frame_pose_constraint_set=\"{}\", calib_shape={}, target_height_cm={})",
                self.minVisPercent,
                self.lossAlpha,
                self.maxIter,
                self.regularization,
                boolToString(self.debug),
                self.calibFrames,
                self.majorIter,
                boolToString(self.globalScaleOnly),
                boolToString(self.locatorsOnly),
                self.greedySampling,
                boolToString(self.enforceFloorInFirstFrame),
                self.firstFramePoseConstraintSet,
                boolToString(self.calibShape),
                self.targetHeightCm);
          })
      .def(
          py::init<
              float,
              float,
              size_t,
              float,
              bool,
              size_t,
              size_t,
              bool,
              bool,
              size_t,
              bool,
              std::string,
              bool,
              float>(),
          R"(Create a CalibrationConfig with specified parameters.

          :param min_vis_percent: Minimum percentage of visible markers to be used
          :param loss_alpha: Parameter to control the loss function
          :param max_iter: Maximum number of iterations
          :param debug: Whether to output debugging info
          :param calib_frames: Number of frames used for model calibration
          :param major_iter: Number of calibration loops to run
          :param global_scale_only: Calibrate only the global scale and not all proportions
          :param locators_only: Calibrate only the locator offsets
          :param greedy_sampling: Enable greedy frame sampling with the given stride
          :param enforce_floor_in_first_frame: Force floor contact in first frame
          :param first_frame_pose_constraint_set: Name of pose constraint set to use in first frame
          :param target_height_cm: Target height for character in cm.  Defaults to 0 (unspecified).
          )",
          py::arg("min_vis_percent") = 0.0,
          py::arg("loss_alpha") = 2.0,
          py::arg("max_iter") = 30,
          py::arg("regularization") = 0.05f,
          py::arg("debug") = false,
          py::arg("calib_frames") = 100,
          py::arg("major_iter") = 3,
          py::arg("global_scale_only") = false,
          py::arg("locators_only") = false,
          py::arg("greedy_sampling") = 0,
          py::arg("enforce_floor_in_first_frame") = false,
          py::arg("first_frame_pose_constraint_set") = "",
          py::arg("calib_shape") = false,
          py::arg("target_height_cm") = 0.0)
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
          "Name of pose constraint set to use in first frame")
      .def_readwrite(
          "calib_shape", &momentum::CalibrationConfig::calibShape, "Calibrate shape parameters")
      .def_readwrite(
          "target_height_cm",
          &momentum::CalibrationConfig::targetHeightCm,
          "Target height for the character in cm (0 means no target height specified)");

  auto trackingConfig = py::class_<momentum::TrackingConfig, momentum::BaseConfig>(
      m, "TrackingConfig", "Config for the tracking optimization step");

  trackingConfig.def(py::init<>())
      .def(
          "__repr__",
          [](const momentum::TrackingConfig& self) {
            return fmt::format(
                "TrackingConfig(min_vis_percent={}, loss_alpha={}, max_iter={}, regularization={}, debug={}, smoothing={}, collision_error_weight={}, marker_weight={}, smoothing_weights={})",
                self.minVisPercent,
                self.lossAlpha,
                self.maxIter,
                self.regularization,
                boolToString(self.debug),
                self.smoothing,
                self.collisionErrorWeight,
                self.markerWeight,
                vectorToString(self.smoothingWeights));
          })
      .def(
          py::init<float, float, size_t, float, bool, float, float, float, Eigen::VectorXf>(),
          R"(Create a TrackingConfig with specified parameters.

          :param min_vis_percent: Minimum percentage of visible markers to be used
          :param loss_alpha: Parameter to control the loss function
          :param max_iter: Maximum number of iterations
          :param regularization: Regularization parameter (lambda) for solver
          :param debug: Whether to output debugging info
          :param smoothing: Smoothing weight; 0 to disable
          :param collision_error_weight: Collision error weight; 0 to disable
          :param marker_weight: Multiplier for marker position constraint weight; 0 to disable markers
          :param smoothing_weights: Smoothing weights per model parameter
          )",
          py::arg("min_vis_percent") = 0.0,
          py::arg("loss_alpha") = 2.0,
          py::arg("max_iter") = 30,
          py::arg("regularization") = 0.05f,
          py::arg("debug") = false,
          py::arg("smoothing") = 0.0,
          py::arg("collision_error_weight") = 0.0,
          py::arg("marker_weight") = 1.0f,
          py::arg("smoothing_weights") = Eigen::VectorXf())
      .def_readwrite(
          "smoothing", &momentum::TrackingConfig::smoothing, "Smoothing weight; 0 to disable")
      .def_readwrite(
          "collision_error_weight",
          &momentum::TrackingConfig::collisionErrorWeight,
          "Collision error weight; 0 to disable")
      .def_readwrite(
          "marker_weight",
          &momentum::TrackingConfig::markerWeight,
          "Multiplier for marker position constraint weight; 0 to disable markers")
      .def_readwrite(
          "smoothing_weights",
          &momentum::TrackingConfig::smoothingWeights,
          R"(Smoothing weights per model parameter. The size of this vector should be
            equal to number of model parameters and this overrides the value specific in smoothing)");

  auto refineConfig = py::class_<momentum::RefineConfig, momentum::TrackingConfig>(
      m, "RefineConfig", "Config for refining a tracked motion.");

  refineConfig.def(py::init<>())
      .def(
          "__repr__",
          [](const momentum::RefineConfig& self) {
            return fmt::format(
                "RefineConfig(min_vis_percent={}, loss_alpha={}, max_iter={}, regularization={}, debug={}, smoothing={}, collision_error_weight={}, marker_weight={}, smoothing_weights={}, regularizer={}, calib_id={}, calib_locators={})",
                self.minVisPercent,
                self.lossAlpha,
                self.maxIter,
                self.regularization,
                boolToString(self.debug),
                self.smoothing,
                self.collisionErrorWeight,
                self.markerWeight,
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
              float,
              bool,
              float,
              float,
              float,
              Eigen::VectorXf,
              float,
              bool,
              bool>(),
          R"(Create a RefineConfig with specified parameters.

          :param min_vis_percent: Minimum percentage of visible markers to be used
          :param loss_alpha: Parameter to control the loss function
          :param max_iter: Maximum number of iterations
          :param debug: Whether to output debugging info
          :param smoothing: Smoothing weight; 0 to disable
          :param collision_error_weight: Collision error weight; 0 to disable
          :param marker_weight: Multiplier for marker position constraint weight; 0 to disable markers
          :param smoothing_weights: Smoothing weights per model parameter
          :param regularizer: Regularize the time-invariant parameters to prevent large changes
          :param calib_id: Calibrate identity parameters
          :param calib_locators: Calibrate locator offsets
          )",
          py::arg("min_vis_percent") = 0.0,
          py::arg("loss_alpha") = 2.0,
          py::arg("max_iter") = 30,
          py::arg("regularization") = 0.05f,
          py::arg("debug") = false,
          py::arg("smoothing") = 0.0,
          py::arg("collision_error_weight") = 0.0,
          py::arg("marker_weight") = 1.0f,
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
                R"(ModelOptions(model="{}", parameters="{}", locators="{}"))",
                self.model,
                self.parameters,
                self.locators);
          })
      .def(
          py::init<const std::string&, const std::string&, const std::string&>(),
          R"(Create ModelOptions with specified file paths.

          :param model: Path to template model file with locators e.g. character.glb
          :param parameters: Path of parameter transform model file e.g. character.model
          :param locators: Path to locator mapping file e.g. character.locators
          )",
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

  // Bindings for glove types defined in marker_tracking/glove_utils.h
  auto gloveConfig = py::class_<momentum::GloveConfig>(
      m,
      "GloveConfig",
      R"(Configuration for glove constraints in marker tracking.

Controls how data glove observations are integrated into the marker tracking
solver, including constraint weights and which wrist joints to attach glove
bones to.)");

  gloveConfig.def(py::init<>())
      .def(
          "__repr__",
          [](const momentum::GloveConfig& self) {
            return fmt::format(
                R"(GloveConfig(position_weight={}, orientation_weight={}, wrist_joint_names=["{}", "{}"]))",
                self.positionWeight,
                self.orientationWeight,
                self.wristJointNames[0],
                self.wristJointNames[1]);
          })
      .def_readwrite(
          "position_weight",
          &momentum::GloveConfig::positionWeight,
          "Weight for position constraints between glove and finger joints.")
      .def_readwrite(
          "orientation_weight",
          &momentum::GloveConfig::orientationWeight,
          "Weight for orientation constraints between glove and finger joints.")
      .def_readwrite(
          "wrist_joint_names",
          &momentum::GloveConfig::wristJointNames,
          "Names of the left and right wrist joints in the skeleton.");

  auto gloveSensorObservation = py::class_<momentum::GloveSensorObservation>(
      m,
      "GloveSensorObservation",
      R"(Single glove sensor observation for one finger joint in one frame.

Represents a measurement from a data glove sensor, providing position
and orientation of a finger joint in the glove's local coordinate frame.)");

  gloveSensorObservation.def(py::init<>())
      .def(
          py::init([](const std::string& jointName,
                      const Eigen::Vector3f& position,
                      const Eigen::Quaternionf& orientation,
                      bool valid) {
            momentum::GloveSensorObservation obs;
            obs.jointName = jointName;
            obs.position = position;
            obs.orientation = orientation;
            obs.valid = valid;
            return obs;
          }),
          R"(Create a GloveSensorObservation with specified parameters.

:param joint_name: Skeleton joint name (e.g. "b_l_thumb0").
:param position: Position in glove-local frame as a 3D vector.
:param orientation: Orientation in glove-local frame as a quaternion [x, y, z, w].
:param valid: Whether this observation is valid (False if sensor data is missing/occluded).
)",
          py::arg("joint_name"),
          py::arg("position") = Eigen::Vector3f::Zero(),
          py::arg("orientation") = Eigen::Quaternionf::Identity(),
          py::arg("valid") = true)
      .def_readwrite(
          "joint_name",
          &momentum::GloveSensorObservation::jointName,
          "Skeleton joint name (e.g. \"b_l_thumb0\").")
      .def_readwrite(
          "position",
          &momentum::GloveSensorObservation::position,
          "Position in glove-local frame as a 3D vector.")
      .def_property(
          "orientation",
          [](const momentum::GloveSensorObservation& self) { return self.orientation; },
          [](momentum::GloveSensorObservation& self, const Eigen::Quaternionf& q) {
            self.orientation = q;
          },
          "Orientation in glove-local frame as a quaternion [x, y, z, w].")
      .def_readwrite(
          "valid",
          &momentum::GloveSensorObservation::valid,
          "Whether this observation is valid (False if sensor data is missing/occluded).");

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
      "calibrate_markers",
      [](momentum::Character& character,
         const Eigen::VectorXf& identity,
         const std::vector<std::vector<momentum::Marker>>& markerData,
         const momentum::CalibrationConfig& calibrationConfig,
         size_t firstFrame,
         size_t maxFrames,
         const std::vector<momentum::GloveFrameData>& leftGloveData,
         const std::vector<momentum::GloveFrameData>& rightGloveData,
         const std::optional<momentum::GloveConfig>& gloveConfig) {
        momentum::ModelParameters params(identity);

        if (params.size() == 0) { // If no identity is passed in, use default
          params = momentum::ModelParameters::Zero(character.parameterTransform.name.size());
        }

        momentum::calibrateMarkers(
            character,
            params,
            markerData,
            calibrationConfig,
            firstFrame,
            maxFrames,
            leftGloveData,
            rightGloveData,
            gloveConfig);

        // Return the identity parameters (scaling params only)
        // Use .v to get the underlying Eigen::VectorXf from ModelParameters
        return params.v;
      },
      R"(Calibrate a character model using marker data without running full tracking.

This function performs only the calibration step (scaling, locator offsets, and optionally
shape parameters) without running per-frame tracking on all frames. This is useful when you
want to calibrate on a ROM (range of motion) sequence but don't need the tracked motion for
that sequence.

The calibration modifies the character in-place (updating locators if configured) and returns
the calibrated identity parameters. These can then be used with :func:`process_markers` or
:func:`track_poses` for tracking other sequences.

When glove data is provided via ``left_glove_data`` / ``right_glove_data`` and a
:class:`GloveConfig`, the solver adds glove-to-finger constraints during calibration.
This improves locator calibration by providing additional constraint information from the
data glove sensors.

:param character: Character to be calibrated. Will be modified in-place if locator
    calibration is enabled.
:param identity: Identity parameters, pass in empty array for default identity.
:param marker_data: A list of marker data for each frame (from ROM/calibration sequence).
:param calibration_config: Calibration config specifying number of frames, iterations, etc.
:param first_frame: First frame to be used for calibration.
:param max_frames: Max number of frames to be used for calibration (0 for all).
:param left_glove_data: Per-frame glove sensor observations for the left hand.
    Each element is a list of :class:`GloveSensorObservation` for that frame.
:param right_glove_data: Per-frame glove sensor observations for the right hand.
:param glove_config: Optional :class:`GloveConfig` controlling constraint weights and
    wrist joint names. Must be provided if glove data is non-empty.
:return: Calibrated identity parameters (scaling parameters).)",
      py::arg("character"),
      py::arg("identity"),
      py::arg("marker_data"),
      py::arg("calibration_config"),
      py::arg("first_frame") = 0,
      py::arg("max_frames") = 0,
      py::arg("left_glove_data") = std::vector<momentum::GloveFrameData>{},
      py::arg("right_glove_data") = std::vector<momentum::GloveFrameData>{},
      py::arg("glove_config") = std::nullopt);

  m.def(
      "process_markers",
      [](momentum::Character& character,
         const Eigen::VectorXf& identity,
         const std::vector<std::vector<momentum::Marker>>& markerData,
         const momentum::TrackingConfig& trackingConfig,
         const momentum::CalibrationConfig& calibrationConfig,
         bool calibrate,
         size_t firstFrame,
         size_t maxFrames,
         const std::vector<momentum::GloveFrameData>& leftGloveData,
         const std::vector<momentum::GloveFrameData>& rightGloveData,
         const std::optional<momentum::GloveConfig>& gloveConfig) {
        momentum::ModelParameters params(identity);

        if (params.size() == 0) { // If no identity is passed in, use default
          params = momentum::ModelParameters::Zero(character.parameterTransform.name.size());
        }

        Eigen::MatrixXf motion = momentum::processMarkers(
            character,
            params,
            markerData,
            trackingConfig,
            calibrationConfig,
            calibrate,
            firstFrame,
            maxFrames,
            leftGloveData,
            rightGloveData,
            gloveConfig);

        // python and cpp have the motion matrix transposed from each other:
        // python (#frames, #params) vs. cpp (#params, #frames)
        return motion.transpose().eval();
      },
      R"(Process markers given character and identity.

Calibrates the character model (if enabled) and tracks per-frame poses from marker data.

When glove data is provided via ``left_glove_data`` / ``right_glove_data`` and a
:class:`GloveConfig`, the solver adds glove-to-finger constraints during both
calibration and per-frame tracking. This produces more accurate results for
characters wearing data gloves alongside optical markers.

:param character: Character to be used for tracking.
:param identity: Identity parameters, pass in empty array for default identity.
:param marker_data: A list of marker data for each frame.
:param tracking_config: Tracking config to be used for tracking.
:param calibration_config: Calibration config to be used for calibration.
:param calibrate: Whether to calibrate the model.
:param first_frame: First frame to be processed.
:param max_frames: Max number of frames to be processed.
:param left_glove_data: Per-frame glove sensor observations for the left hand.
    Each element is a list of :class:`GloveSensorObservation` for that frame.
:param right_glove_data: Per-frame glove sensor observations for the right hand.
:param glove_config: Optional :class:`GloveConfig` controlling constraint weights and
    wrist joint names. Must be provided if glove data is non-empty.
:return: Transform parameters for each frame.)",
      py::arg("character"),
      py::arg("identity"),
      py::arg("marker_data"),
      py::arg("tracking_config"),
      py::arg("calibration_config"),
      py::arg("calibrate") = true,
      py::arg("first_frame") = 0,
      py::arg("max_frames") = 0,
      py::arg("left_glove_data") = std::vector<momentum::GloveFrameData>{},
      py::arg("right_glove_data") = std::vector<momentum::GloveFrameData>{},
      py::arg("glove_config") = std::nullopt);

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
          params = momentum::ModelParameters::Zero(character.parameterTransform.name.size());
        }

        // python and cpp have the motion matrix transposed from each other:
        // python (#frames, #params) vs. cpp (#params, #frames)
        if (motion.cols() == character.parameterTransform.numAllModelParameters()) {
          // we need to transpose the matrix before passing it to the cpp
          Eigen::MatrixXf finalMotion(motion.transpose());
          // note: saveMotion removes identity from the motion matrix
          momentum::saveMotion(
              outFile, character, params, finalMotion, markerData, fps, saveMarkerMesh);
          // and transpose it back since motion is passed by reference
          motion = finalMotion.transpose();
        } else if (motion.rows() == character.parameterTransform.numAllModelParameters()) {
          // motion matrix is already in cpp format
          // keeping this branch for backward compatibility
          // note: saveMotion removes identity from the motion matrix
          momentum::saveMotion(outFile, character, params, motion, markerData, fps, saveMarkerMesh);
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
          momentum::ParameterSet idParamSet = character.parameterTransform.getScalingParameters();
          momentum::fillIdentity(idParamSet, identity, inputMotion);
        }
        Eigen::MatrixXf finalMotion =
            momentum::refineMotion(markerData, inputMotion, refineConfig, character);
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
:param min_skin_weight: Minimum skin weight threshold for considering a mesh triangle as
    belonging to the same bone as the locator (default: 0.03)
:param verbose: If True, print diagnostic messages about locators that could not be
    converted (default: False)
:return: New character with converted skinned locators and remaining regular locators)",
      py::arg("character"),
      py::arg("max_distance") = 3.0f,
      py::arg("min_skin_weight") = 0.03f,
      py::arg("verbose") = false);

  m.def(
      "convert_skinned_locators_to_locators",
      &momentum::skinnedLocatorsToLocators,
      R"(Convert skinned locators to regular locators by selecting the bone with highest weight.

This function is useful when exporting to file formats that don't support skinned locators
(e.g., Maya). Each skinned locator is converted to a regular locator by:

1. Finding the bone with the highest skin weight from the skinned locator's bone influences
2. Transforming the locator's position from rest pose space to the local coordinate space
   of the selected bone
3. Creating a regular locator attached to that bone with the computed offset

The resulting locators can be exported to formats like Maya that only support single-parent
attachments. Any existing regular locators in the character are preserved.

:param character: Character with skinned locators to convert
:return: New character with skinned locators converted to regular locators)",
      py::arg("character"));

  m.def(
      "get_locator_error",
      [](const std::vector<std::vector<momentum::Marker>>& markerData,
         const Eigen::MatrixXf& motion,
         momentum::Character& character) {
        // Python uses (frames, params) layout, C++ uses (params, frames)
        Eigen::MatrixXf motionTransposed;
        if (motion.cols() == character.parameterTransform.numAllModelParameters()) {
          motionTransposed = motion.transpose();
        } else {
          motionTransposed = motion;
        }
        return momentum::getLocatorError(markerData, motionTransposed, character);
      },
      R"(Compute average and maximum marker tracking errors across all frames.

This is a utility function for evaluating tracking quality by measuring the
Euclidean distance between observed marker positions and their corresponding
locator positions on the character. It provides both average error per frame
and the maximum error encountered across all markers and frames.

:param marker_data: List of marker observations for each frame. Each frame contains
    a list of Marker objects with position and occlusion information.
:param motion: Solved motion parameters matrix. Can be in Python layout (frames, params)
    or C++ layout (params, frames) - the function handles both.
:param character: Character model with locators that correspond to the markers.
:return: Tuple of (average_error, max_error) in world units (typically meters or cm
    depending on your character scale). average_error is the mean error per frame
    across all frames, max_error is the single largest marker error found.)",
      py::arg("marker_data"),
      py::arg("motion"),
      py::arg("character"));
}
