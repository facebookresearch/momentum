/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/geometry/momentum_io.h"
#include "pymomentum/geometry/momentum_geometry.h"

#include <momentum/character/character.h>
#include <momentum/character/joint_state.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character/types.h>
#include <momentum/common/checks.h>
#include <momentum/io/fbx/fbx_io.h>
#include <momentum/io/gltf/gltf_io.h>
#include <momentum/io/marker/marker_io.h>

namespace pymomentum {

momentum::Character loadGLTFCharacterFromFile(const std::string& path) {
  return momentum::loadGltfCharacter(path);
}

momentum::Character loadGLTFCharacterFromBytes(const pybind11::bytes& bytes) {
  pybind11::buffer_info info(pybind11::buffer(bytes).request());
  const std::byte* data = reinterpret_cast<const std::byte*>(info.ptr);
  const size_t length = static_cast<size_t>(info.size);

  MT_THROW_IF(data == nullptr, "Unable to extract contents from bytes.");

  return momentum::loadGltfCharacter(
      gsl::make_span<const std::byte>(data, length));
}

momentum::MotionParameters transpose(
    const momentum::MotionParameters& motionParameters) {
  const auto& [parameters, poses] = motionParameters;
  return {parameters, poses.transpose()};
}

std::vector<momentum::SkeletonState> arrayToSkeletonStates(
    const pybind11::array_t<float>& skel_states,
    const momentum::Character& character) {
  MT_THROW_IF(
      skel_states.ndim() != 3,
      "Expected skel_states to have size n_frames x n_joints x 8, but got {}",
      formatDimensions(skel_states));

  const int numFrames = skel_states.shape(0);
  const int numJoints = skel_states.shape(1);
  const int numElements = skel_states.shape(2);

  MT_THROW_IF(
      numElements != 8,
      "Expecting size 8 (3 translation + 4 rotation + scale) for last dimension of the skel_states, but got {}",
      numElements);
  MT_THROW_IF(
      numJoints != character.skeleton.joints.size(),
      "Expecting {} joints in the skeleton states, but got {}",
      character.skeleton.joints.size(),
      numJoints);

  std::vector<momentum::SkeletonState> skeletonStates(numFrames);

  auto skelStatesAccess = skel_states.unchecked<3>();

  for (int iFrame = 0; iFrame < numFrames; ++iFrame) {
    auto& skelStateCur = skeletonStates[iFrame];
    skelStateCur.jointState.resize(numJoints);
    for (int jJoint = 0; jJoint < numJoints; ++jJoint) {
      const Eigen::Vector3f translation{
          skelStatesAccess(iFrame, jJoint, 0),
          skelStatesAccess(iFrame, jJoint, 1),
          skelStatesAccess(iFrame, jJoint, 2),
      };

      // Skel_state has order rx, ry, rz, rw
      // Quaternion constructor takes order w, x, y, z
      const Eigen::Quaternionf rotation{
          skelStatesAccess(iFrame, jJoint, 6),
          skelStatesAccess(iFrame, jJoint, 3),
          skelStatesAccess(iFrame, jJoint, 4),
          skelStatesAccess(iFrame, jJoint, 5)};
      const float scale = skelStatesAccess(iFrame, jJoint, 7);

      const momentum::Transform transform{translation, rotation, scale};

      momentum::Transform parentTransform;
      const auto parent = character.skeleton.joints[jJoint].parent;
      if (parent != momentum::kInvalidIndex) {
        parentTransform = skelStateCur.jointState[parent].transform;
      }

      // transform = parentTransform * localTransform
      // localTransform = parentTransform.inverse() * transform
      skelStateCur.jointState[jJoint].transform = transform;
      skelStateCur.jointState[jJoint].localTransform =
          parentTransform.inverse() * transform;
    }
  }
  return skeletonStates;
}

void saveGLTFCharacterToFile(
    const std::string& path,
    const momentum::Character& character,
    const float fps,
    const std::optional<const momentum::MotionParameters>& motion,
    const std::optional<const momentum::IdentityParameters>& offsets,
    const std::optional<const std::vector<std::vector<momentum::Marker>>>&
        markers) {
  if (motion.has_value()) {
    const auto& [parameters, poses] = motion.value();
    MT_THROW_IF(
        poses.cols() != parameters.size(),
        "Expected motion parameters to be n_frames x {}, but got {} x {}",
        parameters.size(),
        poses.rows(),
        poses.cols());
  }
  momentum::saveCharacter(
      path,
      character,
      fps,
      transpose(motion.value_or(momentum::MotionParameters{})),
      offsets.value_or(momentum::IdentityParameters{}),
      markers.value_or(std::vector<std::vector<momentum::Marker>>{}));
}

void saveGLTFCharacterToFileFromSkelStates(
    const std::string& path,
    const momentum::Character& character,
    const float fps,
    const pybind11::array_t<float>& skel_states,
    const std::optional<const std::vector<std::vector<momentum::Marker>>>&
        markers) {
  const auto numFrames = skel_states.shape(0);

  MT_THROW_IF(
      markers.has_value() && markers->size() != numFrames,
      "The number of frames of the skeleton states array {} does not coincide with the number of frames of the markers {}",
      skel_states.size(),
      markers->size());

  // Use the shared utility function for conversion
  std::vector<momentum::SkeletonState> skeletonStates =
      arrayToSkeletonStates(skel_states, character);

  momentum::saveCharacter(
      path,
      character,
      fps,
      skeletonStates,
      markers.value_or(std::vector<std::vector<momentum::Marker>>{}));
}

void saveFBXCharacterToFile(
    const std::string& path,
    const momentum::Character& character,
    const float fps,
    std::optional<const Eigen::MatrixXf> motion,
    std::optional<const Eigen::VectorXf> offsets,
    std::optional<const momentum::FBXCoordSystemInfo> coordSystemInfo) {
  if (motion.has_value() && offsets.has_value()) {
    momentum::saveFbx(
        path,
        character,
        motion.value().transpose(),
        offsets.value(),
        fps,
        true, /*saveMesh*/
        coordSystemInfo.has_value() ? coordSystemInfo.value()
                                    : momentum::FBXCoordSystemInfo());
  } else {
    momentum::saveFbxModel(path, character);
  }
}

void saveFBXCharacterToFileWithJointParams(
    const std::string& path,
    const momentum::Character& character,
    const float fps,
    std::optional<const Eigen::MatrixXf> jointParams,
    std::optional<const momentum::FBXCoordSystemInfo> coordSystemInfo) {
  if (jointParams.has_value()) {
    momentum::saveFbxWithJointParams(
        path,
        character,
        jointParams.value().transpose(),
        fps,
        true, /*saveMesh*/
        coordSystemInfo.has_value() ? coordSystemInfo.value()
                                    : momentum::FBXCoordSystemInfo());
  } else {
    momentum::saveFbxModel(path, character);
  }
}

std::tuple<momentum::Character, RowMatrixf, Eigen::VectorXf, float>
loadGLTFCharacterWithMotion(const std::string& gltfFilename) {
  const auto [character, motion, identity, fps] =
      momentum::loadCharacterWithMotion(gltfFilename);
  return std::make_tuple(character, motion.transpose(), identity, fps);
}

std::tuple<momentum::Character, RowMatrixf, Eigen::VectorXf, float>
loadGLTFCharacterWithMotionFromBytes(const pybind11::bytes& gltfBytes) {
  pybind11::buffer_info info(pybind11::buffer(gltfBytes).request());
  const std::byte* data = reinterpret_cast<const std::byte*>(info.ptr);
  const size_t length = static_cast<size_t>(info.size);

  MT_THROW_IF(data == nullptr, "Unable to extract contents from bytes.");

  const auto [character, motion, identity, fps] =
      momentum::loadCharacterWithMotion(
          gsl::make_span<const std::byte>(data, length));
  return std::make_tuple(character, motion.transpose(), identity, fps);
}

pybind11::array_t<float> skelStatesToTensor(
    gsl::span<const momentum::SkeletonState> states,
    const momentum::Character& character) {
  pybind11::array_t<float> result(
      {(int)states.size(), (int)character.skeleton.joints.size(), 8});

  auto r = result.mutable_unchecked<3>();

  for (size_t iFrame = 0; iFrame < states.size(); ++iFrame) {
    const auto& state = states[iFrame];
    if (state.jointState.size() != r.shape(1)) {
      throw std::runtime_error(
          "Mismatch between number of joints in skeleton state and tensor");
    }

    for (size_t iJoint = 0; iJoint < state.jointState.size(); ++iJoint) {
      const auto& jointState = state.jointState.at(iJoint);
      r(iFrame, iJoint, 0) = jointState.transform.translation.x();
      r(iFrame, iJoint, 1) = jointState.transform.translation.y();
      r(iFrame, iJoint, 2) = jointState.transform.translation.z();
      r(iFrame, iJoint, 3) = jointState.transform.rotation.x();
      r(iFrame, iJoint, 4) = jointState.transform.rotation.y();
      r(iFrame, iJoint, 5) = jointState.transform.rotation.z();
      r(iFrame, iJoint, 6) = jointState.transform.rotation.w();
      r(iFrame, iJoint, 7) = jointState.transform.scale;
    }
  }
  return result;
}

std::tuple<momentum::Character, pybind11::array_t<float>, std::vector<float>>
loadGLTFCharacterWithSkelStatesFromBytes(const pybind11::bytes& gltfBytes) {
  pybind11::buffer_info info(pybind11::buffer(gltfBytes).request());
  const std::byte* data = reinterpret_cast<const std::byte*>(info.ptr);
  const size_t length = static_cast<size_t>(info.size);

  MT_THROW_IF(data == nullptr, "Unable to extract contents from bytes.");

  momentum::Character character;
  std::vector<momentum::SkeletonState> states;
  std::vector<float> timestamps;

  {
    pybind11::gil_scoped_release release;
    std::tie(character, states, timestamps) =
        momentum::loadCharacterWithSkeletonStates(
            gsl::make_span<const std::byte>(data, length));
  }

  return std::make_tuple(
      character, skelStatesToTensor(states, character), timestamps);
}

std::tuple<momentum::Character, pybind11::array_t<float>, std::vector<float>>
loadGLTFCharacterWithSkelStates(const std::string& gltfFilename) {
  momentum::Character character;
  std::vector<momentum::SkeletonState> states;
  std::vector<float> timestamps;
  {
    pybind11::gil_scoped_release release;
    std::tie(character, states, timestamps) =
        momentum::loadCharacterWithSkeletonStates(gltfFilename);
  }

  return std::make_tuple(
      character, skelStatesToTensor(states, character), timestamps);
}

std::string toGLTF(const momentum::Character& character) {
  nlohmann::json j = momentum::makeCharacterDocument(character);
  return j.dump();
}

std::tuple<
    RowMatrixf,
    std::vector<std::string>,
    Eigen::VectorXf,
    std::vector<std::string>>
loadMotion(const std::string& gltfFilename) {
  const auto [motion, identity, fps] = momentum::loadMotion(gltfFilename);

  return {
      std::get<1>(motion).transpose(),
      std::get<0>(motion),
      std::get<1>(identity),
      std::get<0>(identity)};
}

std::vector<momentum::MarkerSequence> loadMarkersFromFile(
    const std::string& path,
    const bool mainSubjectOnly) {
  if (mainSubjectOnly) {
    const auto markerSequence = momentum::loadMarkersForMainSubject(path);
    if (markerSequence.has_value()) {
      return {markerSequence.value()};
    } else {
      return {};
    }
  } else {
    return momentum::loadMarkers(path);
  }
}

} // namespace pymomentum
