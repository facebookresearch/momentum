/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Separate pybind11 module for USD I/O support.
// This is split from the main geometry module to avoid forcing the large USD
// static libraries (linked with link_whole=True) on downstream consumers that
// don't need USD support.

#include <momentum/character/character.h>
#include <momentum/character/marker.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character/types.h>
#include <momentum/common/checks.h>
#include <momentum/io/skeleton/parameters_io.h>
#include <momentum/io/usd/usd_io.h>
#include <cmath>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Core>

#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <vector>

namespace py = pybind11;
namespace mm = momentum;

namespace {

using RowMatrixf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

mm::MotionParameters transpose(const mm::MotionParameters& motionParameters) {
  const auto& [parameters, poses] = motionParameters;
  return {parameters, poses.transpose()};
}

py::array_t<float> skelStatesToTensor(
    std::span<const mm::SkeletonState> states,
    const mm::Character& character) {
  py::array_t<float> result({(int)states.size(), (int)character.skeleton.joints.size(), 8});

  auto r = result.mutable_unchecked<3>();

  for (size_t iFrame = 0; iFrame < states.size(); ++iFrame) {
    const auto& state = states[iFrame];
    MT_THROW_IF(
        state.jointState.size() != static_cast<size_t>(r.shape(1)),
        "Mismatch between number of joints in skeleton state and tensor");

    for (size_t iJoint = 0; iJoint < state.jointState.size(); ++iJoint) {
      const auto& js = state.jointState.at(iJoint);
      r(iFrame, iJoint, 0) = js.transform.translation.x();
      r(iFrame, iJoint, 1) = js.transform.translation.y();
      r(iFrame, iJoint, 2) = js.transform.translation.z();
      r(iFrame, iJoint, 3) = js.transform.rotation.x();
      r(iFrame, iJoint, 4) = js.transform.rotation.y();
      r(iFrame, iJoint, 5) = js.transform.rotation.z();
      r(iFrame, iJoint, 6) = js.transform.rotation.w();
      r(iFrame, iJoint, 7) = js.transform.scale;
    }
  }
  return result;
}

std::vector<mm::SkeletonState> arrayToSkeletonStates(
    const py::array_t<float>& skelStates,
    const mm::Character& character) {
  MT_THROW_IF(
      skelStates.ndim() != 3,
      "Expected skelStates to have 3 dimensions (n_frames x n_joints x 8), but got ndim={}",
      skelStates.ndim());

  const auto numFrames = static_cast<int>(skelStates.shape(0));
  const auto numJoints = static_cast<int>(skelStates.shape(1));
  const auto numElements = static_cast<int>(skelStates.shape(2));

  MT_THROW_IF(
      numElements != 8,
      "Expecting size 8 (3 translation + 4 rotation + scale) for last dimension, but got {}",
      numElements);
  MT_THROW_IF(
      numJoints != static_cast<int>(character.skeleton.joints.size()),
      "Expecting {} joints in the skeleton states, but got {}",
      character.skeleton.joints.size(),
      numJoints);

  std::vector<mm::SkeletonState> skeletonStates(numFrames);

  auto acc = skelStates.unchecked<3>();

  for (int iFrame = 0; iFrame < numFrames; ++iFrame) {
    auto& skelStateCur = skeletonStates[iFrame];
    skelStateCur.jointState.resize(numJoints);
    for (int jJoint = 0; jJoint < numJoints; ++jJoint) {
      const Eigen::Vector3f translation{
          acc(iFrame, jJoint, 0),
          acc(iFrame, jJoint, 1),
          acc(iFrame, jJoint, 2),
      };

      // Skel_state has order rx, ry, rz, rw
      // Quaternion constructor takes order w, x, y, z
      const Eigen::Quaternionf rotation{
          acc(iFrame, jJoint, 6),
          acc(iFrame, jJoint, 3),
          acc(iFrame, jJoint, 4),
          acc(iFrame, jJoint, 5)};
      const float scale = acc(iFrame, jJoint, 7);

      const mm::Transform transform{translation, rotation, scale};

      mm::Transform parentTransform;
      const auto parent = character.skeleton.joints[jJoint].parent;
      if (parent != mm::kInvalidIndex) {
        parentTransform = skelStateCur.jointState[parent].transform;
      }

      skelStateCur.jointState[jJoint].transform = transform;
      skelStateCur.jointState[jJoint].localRotation() =
          parentTransform.rotation.inverse() * transform.rotation;
      skelStateCur.jointState[jJoint].localTranslation() = parentTransform.rotation.inverse() *
          (transform.translation - parentTransform.translation) / parentTransform.scale;
      skelStateCur.jointState[jJoint].localScale() = transform.scale / parentTransform.scale;
    }
  }

  return skeletonStates;
}

// ---------------------------------------------------------------------------
// Load functions
// ---------------------------------------------------------------------------

mm::Character loadCharacter(const std::string& path) {
  return mm::loadUsdCharacter(path);
}

mm::Character loadCharacterFromBytes(const py::bytes& bytes) {
  py::buffer_info info(py::buffer(bytes).request());
  const auto* data = reinterpret_cast<const std::byte*>(info.ptr);
  const auto length = static_cast<size_t>(info.size);
  MT_THROW_IF(data == nullptr, "Unable to extract contents from bytes.");

  py::gil_scoped_release release;
  return mm::loadUsdCharacter(std::span<const std::byte>(data, length));
}

std::tuple<mm::Character, RowMatrixf, Eigen::VectorXf, float> loadCharacterWithMotion(
    const std::string& path) {
  auto [character, motionParams, identityParams, fps] = mm::loadUsdCharacterWithMotion(path);
  const auto& motion = std::get<1>(motionParams);
  const auto& identity = std::get<1>(identityParams);
  return {std::move(character), motion.transpose(), identity.v, fps};
}

std::tuple<mm::Character, RowMatrixf, Eigen::VectorXf, float> loadCharacterWithMotionFromBytes(
    const py::bytes& bytes) {
  py::buffer_info info(py::buffer(bytes).request());
  const auto* data = reinterpret_cast<const std::byte*>(info.ptr);
  const auto length = static_cast<size_t>(info.size);
  MT_THROW_IF(data == nullptr, "Unable to extract contents from bytes.");

  mm::Character character;
  mm::MotionParameters motionParams;
  mm::IdentityParameters identityParams;
  float fps = NAN;
  {
    py::gil_scoped_release release;
    std::tie(character, motionParams, identityParams, fps) =
        mm::loadUsdCharacterWithMotion(std::span<const std::byte>(data, length));
  }
  const auto& motion = std::get<1>(motionParams);
  const auto& identity = std::get<1>(identityParams);
  return {std::move(character), motion.transpose(), identity.v, fps};
}

std::tuple<mm::Character, RowMatrixf, Eigen::VectorXf, float>
loadCharacterWithMotionModelParameterScales(const std::string& path) {
  auto [character, motion, modelIdentity, fps] =
      mm::loadUsdCharacterWithMotionModelParameterScales(path);
  return {std::move(character), motion.transpose(), modelIdentity.v, fps};
}

std::tuple<mm::Character, RowMatrixf, Eigen::VectorXf, float>
loadCharacterWithMotionModelParameterScalesFromBytes(const py::bytes& bytes) {
  py::buffer_info info(py::buffer(bytes).request());
  const auto* data = reinterpret_cast<const std::byte*>(info.ptr);
  const auto length = static_cast<size_t>(info.size);
  MT_THROW_IF(data == nullptr, "Unable to extract contents from bytes.");

  mm::Character character;
  Eigen::MatrixXf motion;
  mm::ModelParameters modelIdentity;
  float fps = NAN;
  {
    py::gil_scoped_release release;
    std::tie(character, motion, modelIdentity, fps) =
        mm::loadUsdCharacterWithMotionModelParameterScales(
            std::span<const std::byte>(data, length));
  }
  return {std::move(character), motion.transpose(), modelIdentity.v, fps};
}

std::tuple<mm::Character, py::array_t<float>, std::vector<float>> loadCharacterWithSkelStates(
    const std::string& path) {
  mm::Character character;
  std::vector<mm::SkeletonState> states;
  std::vector<float> timestamps;
  {
    py::gil_scoped_release release;
    std::tie(character, states, timestamps) = mm::loadUsdCharacterWithSkeletonStates(path);
  }
  auto tensor = skelStatesToTensor(states, character);
  return std::make_tuple(std::move(character), std::move(tensor), std::move(timestamps));
}

std::tuple<mm::Character, py::array_t<float>, std::vector<float>>
loadCharacterWithSkelStatesFromBytes(const py::bytes& bytes) {
  py::buffer_info info(py::buffer(bytes).request());
  const auto* data = reinterpret_cast<const std::byte*>(info.ptr);
  const auto length = static_cast<size_t>(info.size);
  MT_THROW_IF(data == nullptr, "Unable to extract contents from bytes.");

  mm::Character character;
  std::vector<mm::SkeletonState> states;
  std::vector<float> timestamps;
  {
    py::gil_scoped_release release;
    std::tie(character, states, timestamps) =
        mm::loadUsdCharacterWithSkeletonStates(std::span<const std::byte>(data, length));
  }
  auto tensor = skelStatesToTensor(states, character);
  return std::make_tuple(std::move(character), std::move(tensor), std::move(timestamps));
}

// ---------------------------------------------------------------------------
// Save functions
// ---------------------------------------------------------------------------

void saveCharacter(
    const std::string& path,
    const mm::Character& character,
    const float fps,
    const std::optional<const mm::MotionParameters>& motion,
    const std::optional<const std::tuple<std::vector<std::string>, Eigen::VectorXf>>& offsets,
    const std::optional<const std::vector<std::vector<mm::Marker>>>& markers,
    const std::optional<const mm::FileSaveOptions>& options) {
  if (motion.has_value()) {
    const auto& [parameters, poses] = motion.value();
    MT_THROW_IF(
        poses.cols() != static_cast<Eigen::Index>(parameters.size()),
        "Expected motion parameters to be n_frames x {}, but got {} x {}",
        parameters.size(),
        poses.rows(),
        poses.cols());
  }

  mm::IdentityParameters identityParams;
  if (offsets.has_value()) {
    const auto& [names, params] = offsets.value();
    identityParams = {names, params};
  }

  static const std::vector<std::vector<mm::Marker>> kEmptyMarkers;
  const auto& markersRef = markers.has_value() ? *markers : kEmptyMarkers;
  auto transposedMotion = motion.has_value() ? transpose(*motion) : mm::MotionParameters{};

  mm::saveUsdCharacterWithMotion(
      path,
      character,
      fps,
      transposedMotion,
      identityParams,
      markersRef,
      options.value_or(mm::FileSaveOptions{}));
}

void saveCharacterFromSkelStates(
    const std::string& path,
    const mm::Character& character,
    const float fps,
    const py::array_t<float>& skelStates,
    const std::optional<const std::vector<std::vector<mm::Marker>>>& markers,
    const std::optional<const mm::FileSaveOptions>& options) {
  const auto numFrames = skelStates.shape(0);

  MT_THROW_IF(
      markers.has_value() && static_cast<Eigen::Index>(markers->size()) != numFrames,
      "The number of frames of the skeleton states array {} does not coincide with the number of frames of the markers {}",
      numFrames,
      markers->size());

  std::vector<mm::SkeletonState> skeletonStates = arrayToSkeletonStates(skelStates, character);

  static const std::vector<std::vector<mm::Marker>> kEmptyMarkers;
  const auto& markersRef = markers.has_value() ? *markers : kEmptyMarkers;

  mm::saveUsdCharacter(
      path, character, fps, skeletonStates, markersRef, options.value_or(mm::FileSaveOptions{}));
}

mm::MarkerSequence loadMarkerSequence(const std::string& path) {
  return mm::loadUsdMarkerSequence(path);
}

} // namespace

// NOLINTNEXTLINE(facebook-hte-NonConstFunctionCalls)
PYBIND11_MODULE(io_usd, m) {
  py::module_::import("pymomentum.geometry"); // @dep=//arvr/libraries/pymomentum:geometry
  m.doc() = R"(USD I/O support for pymomentum.

This module provides functions for loading and saving characters, motion data,
and markers in USD format (.usd, .usda, .usdc, .usdz).

This is a separate module from pymomentum.geometry to avoid forcing the large
USD static libraries on downstream consumers that don't need USD support.
)";

  m.def(
      "is_usd_available",
      []() { return true; },
      R"(Returns True, indicating that USD support is available.

When this module is importable, USD support is available.)");

  // -------------------------------------------------------------------------
  // Load functions
  // -------------------------------------------------------------------------

  m.def(
      "load_character",
      &loadCharacter,
      py::call_guard<py::gil_scoped_release>(),
      R"(Load a character from a USD file.

Supports .usd, .usda, .usdc, and .usdz file formats.

:param path: Path to the USD file.
:return: A Character object containing the loaded skeleton, mesh, and skin weights.
      )",
      py::arg("path"));

  m.def(
      "load_character_from_bytes",
      &loadCharacterFromBytes,
      R"(Load a character from USD bytes.

:param bytes: Bytes array containing the USD data.
:return: A Character object containing the loaded skeleton, mesh, and skin weights.
      )",
      py::arg("bytes"));

  m.def(
      "load_character_with_motion",
      &loadCharacterWithMotion,
      py::call_guard<py::gil_scoped_release>(),
      R"(Load both character (skeleton/mesh) and motion data from a USD file in a single call.

The motion is stored as model parameters per frame and identity as joint parameters (bone offsets/scales).

:param path: Path to the USD file.
:return: A tuple (character, motion, identity, fps), where character is the loaded Character object,
         motion is a numpy array of shape (n_frames, n_params), identity is a numpy array of joint parameters,
         and fps is the frames per second.
      )",
      py::arg("path"));

  m.def(
      "load_character_with_motion_from_bytes",
      &loadCharacterWithMotionFromBytes,
      R"(Load both character and motion data from USD bytes.

This is the byte array version of load_character_with_motion.

:param bytes: Bytes array containing the USD data.
:return: A tuple (character, motion, identity, fps).
      )",
      py::arg("bytes"));

  m.def(
      "load_character_with_motion_model_parameter_scales",
      &loadCharacterWithMotionModelParameterScales,
      py::call_guard<py::gil_scoped_release>(),
      R"(Load both character and motion data from a USD file, with model parameter scales.

This function differs from load_character_with_motion by returning identity parameters
as model parameters (n_params,) instead of joint parameters (n_joints * 7,).
Additionally, the motion has the identity parameters added back to the scale parameters.

:param path: Path to the USD file.
:return: A tuple (character, motion, model_identity, fps).
      )",
      py::arg("path"));

  m.def(
      "load_character_with_motion_model_parameter_scales_from_bytes",
      &loadCharacterWithMotionModelParameterScalesFromBytes,
      R"(Load both character and motion data from USD bytes, with model parameter scales.

This is the byte array version of load_character_with_motion_model_parameter_scales.

:param bytes: Bytes array containing the USD data.
:return: A tuple (character, motion, model_identity, fps).
      )",
      py::arg("bytes"));

  m.def(
      "load_character_with_skel_states",
      &loadCharacterWithSkelStates,
      R"(Load a character and skeleton state motion sequence from a USD file.

Unlike load_character_with_motion, this function reads raw joint transforms and does not
require a valid parameter transform. Use this for USD files authored by any application.

:param path: Path to the USD file.
:return: A tuple (character, skel_states, timestamps), where skel_states is a numpy array
         of shape (n_frames, n_joints, 8) and timestamps is a list of frame times in seconds.
      )",
      py::arg("path"));

  m.def(
      "load_character_with_skel_states_from_bytes",
      &loadCharacterWithSkelStatesFromBytes,
      R"(Load a character and skeleton state motion sequence from USD bytes.

This is the byte array version of load_character_with_skel_states.

:param bytes: Bytes array containing the USD data.
:return: A tuple (character, skel_states, timestamps), where skel_states is a numpy array
         of shape (n_frames, n_joints, 8) and timestamps is a list of frame times in seconds.
      )",
      py::arg("bytes"));

  // -------------------------------------------------------------------------
  // Save functions
  // -------------------------------------------------------------------------

  m.def(
      "save_character",
      &saveCharacter,
      py::call_guard<py::gil_scoped_release>(),
      R"(Save a character to a USD file.

:param path: Path to save the USD file.
:param character: The Character object to save.
:param fps: Frequency in frames per second.
:param motion: Pose data as a tuple of (parameter_names, poses_array) where poses_array has shape (n_frames, n_params).
:param offsets: Identity/offset data as a tuple of (joint_names, offsets_array).
:param markers: Additional marker (3D positions) data as a list of lists of Marker.
:param options: FileSaveOptions for controlling output (mesh, locators, collisions, etc.)
      )",
      py::arg("path"),
      py::arg("character"),
      py::arg("fps") = 120.f,
      py::arg("motion") = std::nullopt,
      py::arg("offsets") = std::nullopt,
      py::arg("markers") = std::nullopt,
      py::arg("options") = std::nullopt);

  m.def(
      "save_character_from_skel_states",
      &saveCharacterFromSkelStates,
      py::call_guard<py::gil_scoped_release>(),
      R"(Save a character with skeleton state animation to a USD file.

:param path: Path to save the USD file.
:param character: The Character object to save.
:param fps: Frequency in frames per second.
:param skel_states: Skeleton states as a numpy array of shape (n_frames, n_joints, 8).
:param markers: Additional marker data.
:param options: FileSaveOptions for controlling output (mesh, locators, collisions, etc.)
      )",
      py::arg("path"),
      py::arg("character"),
      py::arg("fps"),
      py::arg("skel_states"),
      py::arg("markers") = std::optional<const std::vector<std::vector<mm::Marker>>>{},
      py::arg("options") = std::optional<mm::FileSaveOptions>{});

  // -------------------------------------------------------------------------
  // Marker functions
  // -------------------------------------------------------------------------

  m.def(
      "load_marker_sequence",
      &loadMarkerSequence,
      py::call_guard<py::gil_scoped_release>(),
      R"(Load a marker sequence from a USD file.

:param path: Path to the USD file.
:return: A MarkerSequence containing the loaded marker data.
      )",
      py::arg("path"));
}
