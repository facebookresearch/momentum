/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/renderer/momentum_render.h"

#include <momentum/camera/camera.h>
#include <momentum/character/character.h>
#include <momentum/character/linear_skinning.h>
#include <momentum/character/parameter_transform.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character/skin_weights.h>
#include <momentum/common/exception.h>
#include <momentum/math/mesh.h>

#include <dispenso/parallel_for.h> // @manual
#include <pybind11/numpy.h>
#include <Eigen/Core>

#include <array>
#include <cstring>
#include <optional>
#include <span>
#include <sstream>
#include <string_view>
#include <utility>

namespace py = pybind11;

namespace pymomentum {

// Build a camera to look at an object:
// cameraUpWorld: world-space camera up-vector.
// cameraLookAtWorld: world-space direction where camera looks at.
// aimCenterWorld: world-space point the camera looks at
// distanceToAimCenter: distance of the camera to the location of the aim,
//   aimCenterWorld.
// Note: if cameraUpWorld is not orthogonal to cameraLookAtWorld, the function
// will fix it, as long as cameraUpWorld is not parallel to cameraLookAtWorld.
momentum::Camera makeOutsideInCamera(
    const Eigen::Vector3f& cameraUpWorld,
    const Eigen::Vector3f& cameraLookAtWorld,
    const Eigen::Vector3f& aimCenterWorld,
    const float distanceToAimCenter,
    const int imageHeight_pixels,
    const int imageWidth_pixels,
    const float focal_length_mm = 50) {
  // In openCV eye coords,
  //    z points _into_ the frame
  //    x points to frame right
  //    y points _down_
  Eigen::Matrix3f eyeToWorldRotation = Eigen::Matrix3f::Identity();
  // To ensure the matrix is a rotation, we need to orthogonalize the two input
  // vectors:
  const Eigen::Vector3f sideDirection = cameraLookAtWorld.cross(cameraUpWorld);
  const Eigen::Vector3f cameraUpWorldOrtho = sideDirection.cross(cameraLookAtWorld);
  eyeToWorldRotation.col(1) = -cameraUpWorldOrtho.normalized();
  eyeToWorldRotation.col(2) = cameraLookAtWorld.normalized();
  eyeToWorldRotation.col(0) = eyeToWorldRotation.col(1).cross(eyeToWorldRotation.col(2));
  // make sure I got the ordering right
  assert(eyeToWorldRotation.determinant() > 0);

  Eigen::Affine3f worldToEyeXF = Eigen::Affine3f::Identity();

  // To go from world to eye, we
  //   1. translate the object to the origin.
  //   2. rotate the object according to the inverse of eye-to-world rotation.
  //   3. translate the object back (in eye space) along negative z.
  worldToEyeXF.translate(distanceToAimCenter * Eigen::Vector3f::UnitZ());
  worldToEyeXF.rotate(eyeToWorldRotation.transpose());
  worldToEyeXF.translate(-aimCenterWorld);

  // "normal" lens is a 50mm lens on a 35mm camera body:
  const float film_width_mm = 36.0f;
  const float focal_length_pixels = (focal_length_mm / film_width_mm) * (double)imageWidth_pixels;

  // Create a PinholeIntrinsicsModel
  auto intrinsicsModel = std::make_shared<momentum::PinholeIntrinsicsModel>(
      imageWidth_pixels, imageHeight_pixels, focal_length_pixels, focal_length_pixels);

  // Create and return the camera with the intrinsics model and transform
  return momentum::Camera(intrinsicsModel, worldToEyeXF);
}

namespace {

// The joints that a camera for a body faces: the mid-spine, to get a good view of the upper body
// rather than the root (which is down in the pelvis), or the wrist for hand models.
constexpr std::array<const char*, 7> kBodyCameraJoints =
    {"b_spine3", "c_spine3", "spineUpper_joint", "b_l_wrist", "b_r_wrist", "l_wrist", "r_wrist"};

std::optional<size_t> findBodyCameraJoint(const momentum::Character& character) {
  for (const auto* name : kBodyCameraJoints) {
    const auto id = character.skeleton.getJointIdByName(name);
    if (id != momentum::kInvalidIndex) {
      return id;
    }
  }
  return std::nullopt;
}

// Reads skeleton states with shape (nJoints, 8) or (nFrames, nJoints, 8); `context` starts the
// error messages.
std::vector<momentum::SkeletonState> toSkeletonStates(
    const momentum::Character& character,
    const py::array_t<float>& skeletonStates,
    std::string_view context) {
  const size_t nJoints = character.skeleton.joints.size();

  MT_THROW_IF(
      skeletonStates.ndim() < 2 || skeletonStates.ndim() > 3,
      "{}: skeleton_states must be 2D (nJoints x 8), 3D (nFrames x nJoints x 8). Got {} dimensions.",
      context,
      skeletonStates.ndim());

  // Verify the last dimension is 8 (tx, ty, tz, rx, ry, rz, rw, s)
  const size_t lastDim = skeletonStates.shape(skeletonStates.ndim() - 1);
  MT_THROW_IF(
      lastDim != 8,
      "{}: Expected last dimension to be 8 (tx, ty, tz, rx, ry, rz, rw, s), but got {}.",
      context,
      lastDim);

  // Verify the second-to-last dimension matches the number of joints
  const size_t jointsDim = skeletonStates.shape(skeletonStates.ndim() - 2);
  MT_THROW_IF(
      jointsDim != nJoints,
      "{}: Expected {} joints (second-to-last dimension), but got {}.",
      context,
      nJoints,
      jointsDim);

  // `value(iJoint, i)` reads value i of joint iJoint in one frame.
  const auto readFrame = [nJoints](const auto& value) {
    momentum::SkeletonState skelState;
    skelState.jointState.resize(nJoints);
    for (size_t iJoint = 0; iJoint < nJoints; ++iJoint) {
      auto& transform = skelState.jointState[iJoint].transform;
      transform.translation = Eigen::Vector3f(value(iJoint, 0), value(iJoint, 1), value(iJoint, 2));
      transform.rotation = Eigen::Quaternionf(
          value(iJoint, 6), // rw
          value(iJoint, 3), // rx
          value(iJoint, 4), // ry
          value(iJoint, 5)); // rz
      transform.scale = value(iJoint, 7);
    }
    return skelState;
  };

  std::vector<momentum::SkeletonState> result;
  if (skeletonStates.ndim() == 2) {
    const auto accessor = skeletonStates.unchecked<2>();
    result.push_back(
        readFrame([&accessor](size_t iJoint, size_t i) { return accessor(iJoint, i); }));
    return result;
  }

  const auto accessor = skeletonStates.unchecked<3>();
  result.reserve(accessor.shape(0));
  for (py::ssize_t iFrame = 0; iFrame < accessor.shape(0); ++iFrame) {
    result.push_back(readFrame(
        [&accessor, iFrame](size_t iJoint, size_t i) { return accessor(iFrame, iJoint, i); }));
  }
  return result;
}

// Appends the points that keep a character in view: its skinned mesh vertices if it has a mesh and
// skin weights, otherwise its joint positions.
void appendFramedPoints(
    const momentum::Character& character,
    std::span<const momentum::SkeletonState> skelStates,
    std::vector<Eigen::Vector3f>& points) {
  for (const auto& s : skelStates) {
    if (character.mesh && character.skinWeights) {
      for (const auto& p : momentum::applySSD(
               character.inverseBindPose, *character.skinWeights, character.mesh->vertices, s)) {
        points.push_back(p);
      }
    } else {
      for (const auto& js : s.jointState) {
        points.push_back(js.translation());
      }
    }
  }
}

// Moves the camera, keeping its orientation, so that every point is in view.
momentum::Camera framePoints(
    const momentum::Camera& camera,
    const std::vector<Eigen::Vector3f>& points) {
  const float min_z = 5;
  return camera.framePoints(points, min_z, 0.05f);
}

// Where a camera looks from, before framePoints() moves it.
struct CameraAim {
  Eigen::Vector3f up;
  Eigen::Vector3f lookAt;
  Eigen::Vector3f center;
};

// Faces the front of the body at `joint`, averaged over the frames.
CameraAim aimAtBody(
    std::span<const momentum::SkeletonState> skelStates,
    size_t joint,
    bool horizontal,
    float cameraAngle) {
  std::vector<momentum::TransformT<double>> spineLocalToWorldTransforms;
  for (const auto& skelState : skelStates) {
    const auto& jointState = skelState.jointState.at(joint);
    spineLocalToWorldTransforms.emplace_back(
        jointState.translation().cast<double>(), jointState.rotation().cast<double>());
  }
  const std::vector<double> weights(skelStates.size(), 1.0 / skelStates.size());
  const auto blendedTransform = momentum::blendTransforms(
      std::span<const momentum::TransformT<double>>(spineLocalToWorldTransforms),
      std::span<const double>(weights));
  const Eigen::Affine3f spineLocalToWorldXF = blendedTransform.toAffine3().cast<float>();

  const Eigen::Vector3f body_center_world_cm = spineLocalToWorldXF.translation();

  // in spine-local coords,
  //   x points up
  //   y points forward
  //   z points to the body's left

  Eigen::Vector3f body_forward_world = spineLocalToWorldXF.linear() * Eigen::Vector3f::UnitY();
  Eigen::Vector3f camera_forward_world = -body_forward_world;

  // If `horizontal`, we place the camera horizontally, with camera up vector
  // facing upward. We assume the world Y axis points upward.
  // Otherwise, the camera is placed perpendicular to the spine direction.
  Eigen::Vector3f camera_up_world = Eigen::Vector3f::Zero();

  if (horizontal) {
    camera_up_world = Eigen::Vector3f::UnitY();
    camera_forward_world[1] = 0.0;
    camera_forward_world = camera_forward_world.stableNormalized();
    if (camera_forward_world.norm() < 1e-5) {
      // The horizontal camera placement is degenerate. Revert back to the
      // original placement.
      camera_forward_world = -body_forward_world;
      camera_up_world = spineLocalToWorldXF.linear() * Eigen::Vector3f::UnitX();
      camera_up_world = camera_up_world.stableNormalized();
    }
  } else {
    camera_up_world = spineLocalToWorldXF.linear() * Eigen::Vector3f::UnitX();
    camera_up_world = camera_up_world.stableNormalized();
  }

  if (cameraAngle != 0.0) {
    // Rotate camera_forward_world around camera_up_world by `cameraAngle` (rad
    // unit)
    camera_forward_world = Eigen::AngleAxisf(cameraAngle, camera_up_world) * camera_forward_world;
  }

  return {.up = camera_up_world, .lookAt = camera_forward_world, .center = body_center_world_cm};
}

// Looks along the world -Z axis with the world Y axis up, which faces the front of a character
// that faces +Z, rotated by `cameraAngle` about the up axis.
CameraAim aimAlongWorldAxes(std::span<const Eigen::Vector3f> points, float cameraAngle) {
  Eigen::Vector3f center = Eigen::Vector3f::Zero();
  for (const auto& p : points) {
    center += p;
  }
  center /= static_cast<float>(points.size());
  const Eigen::Vector3f up = Eigen::Vector3f::UnitY();
  return {
      .up = up,
      .lookAt = Eigen::AngleAxisf(cameraAngle, up) * -Eigen::Vector3f::UnitZ(),
      .center = center};
}

momentum::Camera makeOutsideInCameraForAim(
    const CameraAim& aim,
    const int imageHeight_pixels,
    const int imageWidth_pixels,
    const float focalLength_mm) {
  const float cameraDistance_cm = 2.5f * 100; // 2.5 meters away.
  return makeOutsideInCamera(
      aim.up,
      aim.lookAt,
      aim.center,
      cameraDistance_cm,
      imageHeight_pixels,
      imageWidth_pixels,
      focalLength_mm);
}

} // namespace

momentum::Camera createCameraForBody(
    const momentum::Character& character,
    const py::array_t<float>& skeletonStates,
    int imageHeight,
    int imageWidth,
    float focalLength_mm,
    bool horizontal,
    float cameraAngle) {
  const auto skelStates = toSkeletonStates(character, skeletonStates, "create_camera_for_body");

  const auto joint = findBodyCameraJoint(character);
  if (!joint.has_value()) {
    std::ostringstream oss;
    oss << "Unable to locate appropriate root joint.  Options are {";
    for (const auto& r : kBodyCameraJoints) {
      oss << r << ", ";
    }
    oss << "}";
    throw std::runtime_error(oss.str());
  }

  std::vector<Eigen::Vector3f> points;
  appendFramedPoints(character, skelStates, points);
  return framePoints(
      makeOutsideInCameraForAim(
          aimAtBody(skelStates, *joint, horizontal, cameraAngle),
          imageHeight,
          imageWidth,
          focalLength_mm),
      points);
}

momentum::Camera createCameraForCharacters(
    const std::vector<const momentum::Character*>& characters,
    const std::vector<py::array_t<float>>& skeletonStates,
    int imageHeight,
    int imageWidth,
    float focalLength_mm,
    bool horizontal,
    float cameraAngle) {
  MT_THROW_IF(characters.empty(), "create_camera_for_characters: characters is empty");
  MT_THROW_IF(
      skeletonStates.size() != characters.size(),
      "create_camera_for_characters: got {} skeleton states for {} characters",
      skeletonStates.size(),
      characters.size());

  // The camera faces the first character with frames that a camera for a body can face: the
  // index of that character, and of the joint it faces.
  std::optional<std::pair<size_t, size_t>> aimTarget;
  for (size_t iCharacter = 0; iCharacter < characters.size(); ++iCharacter) {
    const momentum::Character* character = characters.at(iCharacter);
    MT_THROW_IF(
        character == nullptr, "create_camera_for_characters: character {} is None", iCharacter);
    const auto& states = skeletonStates.at(iCharacter);
    const bool hasFrames = states.ndim() == 2 || (states.ndim() == 3 && states.shape(0) > 0);
    const auto joint = findBodyCameraJoint(*character);
    if (!aimTarget.has_value() && joint.has_value() && hasFrames) {
      aimTarget = std::pair(iCharacter, *joint);
    }
  }

  std::vector<Eigen::Vector3f> points;
  std::optional<CameraAim> aim;
  for (size_t iCharacter = 0; iCharacter < characters.size(); ++iCharacter) {
    const momentum::Character& character = *characters.at(iCharacter);
    const auto states = toSkeletonStates(
        character,
        skeletonStates.at(iCharacter),
        fmt::format("create_camera_for_characters: character {}", iCharacter));
    const size_t pointsPerFrame = character.mesh && character.skinWeights
        ? character.mesh->vertices.size()
        : character.skeleton.joints.size();
    points.reserve(points.size() + states.size() * pointsPerFrame);

    // Skinning reads no Python objects.
    py::gil_scoped_release release;
    appendFramedPoints(character, states, points);
    if (aimTarget.has_value() && aimTarget->first == iCharacter) {
      aim = aimAtBody(states, aimTarget->second, horizontal, cameraAngle);
    }
  }
  MT_THROW_IF(points.empty(), "create_camera_for_characters: skeleton_states have no frames");
  if (!aim.has_value()) {
    aim = aimAlongWorldAxes(points, cameraAngle);
  }

  return framePoints(
      makeOutsideInCameraForAim(*aim, imageHeight, imageWidth, focalLength_mm), points);
}

Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> triangulate(
    const Eigen::VectorXi& faceIndices,
    const Eigen::VectorXi& faceOffsets) {
  std::vector<int32_t> triangleIndices;
  const size_t numPolygons = faceOffsets.size() - 1;
  for (size_t iFace = 0; iFace < numPolygons; ++iFace) {
    const auto faceBegin = faceOffsets(iFace);
    const auto faceEnd = faceOffsets(iFace + 1);
    const auto nv = faceEnd - faceBegin;
    if (nv < 3) {
      throw std::runtime_error(
          (fmt::format("Invalid face with {} indices; expected at least 3.", nv)));
    }
    for (size_t j = 1; j < (nv - 1); ++j) {
      triangleIndices.push_back(faceIndices(faceBegin));
      triangleIndices.push_back(faceIndices(faceBegin + j));
      triangleIndices.push_back(faceIndices(faceBegin + j + 1));
    }
  }

  const size_t numTris = triangleIndices.size() / 3;
  Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> result(numTris, 3);
  for (size_t iTri = 0; iTri < numTris; iTri++) {
    const size_t baseIdx = iTri * 3;
    result(iTri, 0) = triangleIndices.at(baseIdx + 0);
    result(iTri, 1) = triangleIndices.at(baseIdx + 1);
    result(iTri, 2) = triangleIndices.at(baseIdx + 2);
  }
  return result;
}

momentum::Camera createCameraForHand(
    const py::array_t<float>& wristTransformation,
    int imageHeight,
    int imageWidth) {
  if (wristTransformation.ndim() != 2 || wristTransformation.shape(0) != 4 ||
      wristTransformation.shape(1) != 4) {
    throw std::runtime_error(
        fmt::format(
            "create_camera_for_hand: wrist_transformation must be a 4x4 matrix, got shape ({}, {})",
            wristTransformation.shape(0),
            wristTransformation.shape(1)));
  }

  auto accessor = wristTransformation.unchecked<2>();

  // Extract the wrist translation (column 3 of the 4x4 matrix) and convert mm → cm
  const Eigen::Vector3f hand_center_world_cm(
      accessor(0, 3) * 0.1f, accessor(1, 3) * 0.1f, accessor(2, 3) * 0.1f);

  // Camera looks inward from the front of the hand
  // Using fixed Y-up and Z-forward conventions
  const Eigen::Vector3f camera_up_world = Eigen::Vector3f::UnitY();
  const Eigen::Vector3f camera_forward_world = Eigen::Vector3f::UnitZ();

  const float cameraDistanceToHand_cm = 0.5f * 100; // 0.5 meters away

  return makeOutsideInCamera(
      camera_up_world,
      camera_forward_world,
      hand_center_world_cm,
      cameraDistanceToHand_cm,
      imageHeight,
      imageWidth);
}

} // namespace pymomentum
