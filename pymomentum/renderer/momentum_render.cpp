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
#include <momentum/math/mesh.h>

#include <dispenso/parallel_for.h> // @manual
#include <pybind11/numpy.h>
#include <Eigen/Core>

#include <cstring>

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

momentum::Camera frameMesh(
    const momentum::Camera& cam_in,
    const momentum::Character& character,
    std::span<const momentum::SkeletonState> skelStates) {
  const float min_z = 5;
  std::vector<Eigen::Vector3f> positions;
  for (const auto& s : skelStates) {
    if (character.mesh && character.skinWeights) {
      for (const auto& p : momentum::applySSD(
               character.inverseBindPose, *character.skinWeights, character.mesh->vertices, s)) {
        positions.push_back(p);
      }
    } else {
      for (const auto& js : s.jointState) {
        positions.push_back(js.translation());
      }
    }
  }

  return cam_in.framePoints(positions, min_z, 0.05f);
}

momentum::Camera makeOutsideInCameraForBody(
    const momentum::Character& character,
    std::span<const momentum::SkeletonState> skelStates,
    const int imageHeight_pixels,
    const int imageWidth_pixels,
    const float focalLength_mm,
    bool horizontal,
    float cameraAngle) {
  // Center on the mid-spine (to get a good view of the upper body) rather than
  // the root (which is down in the pelvis).
  // For hand models, center on the wrist.
  std::vector<const char*> possibleRoots = {
      "b_spine3", "c_spine3", "spineUpper_joint", "b_l_wrist", "b_r_wrist", "l_wrist", "r_wrist"};

  const auto spineJoint = [&]() -> size_t {
    for (const auto& r : possibleRoots) {
      auto id = character.skeleton.getJointIdByName(r);
      if (id != momentum::kInvalidIndex) {
        return id;
      }
    }

    std::ostringstream oss;
    oss << "Unable to locate appropriate root joint.  Options are {";
    for (const auto& r : possibleRoots) {
      oss << r << ", ";
    }
    oss << "}";
    throw std::runtime_error(oss.str());
  }();

  std::vector<momentum::TransformT<double>> spineLocalToWorldTransforms;
  for (const auto& skelState : skelStates) {
    const auto& jointState = skelState.jointState.at(spineJoint);
    spineLocalToWorldTransforms.emplace_back(
        jointState.translation().cast<double>(), jointState.rotation().cast<double>());
  }
  const std::vector<double> weights(skelStates.size(), 1.0 / skelStates.size());
  const auto blendedTransform = momentum::blendTransforms(
      std::span<const momentum::TransformT<double>>(spineLocalToWorldTransforms),
      std::span<const double>(weights));
  const Eigen::Affine3f spineLocalToWorldXF = blendedTransform.toAffine3().cast<float>();

  const Eigen::Vector3f body_center_world_cm = spineLocalToWorldXF.translation();
  const float cameraDistanceToBody_cm = 2.5f * 100; // 2.5 meters away.

  // in spine-local coords,
  //   x points up
  //   y points forward
  //   z points to the body's left

  Eigen::Vector3f body_forward_world = spineLocalToWorldXF.linear() * Eigen::Vector3f::UnitY();
  Eigen::Vector3f camera_forward_world = -body_forward_world;

  // If `horizontal`, we place the camera horizontally, with camera up vector
  // facing upward. We assume the world Y axis points upward.
  // Otherwise, the camera is placed perpendicular to the spine direction.
  Eigen::Vector3f camera_up_world;

  if (horizontal) {
    camera_up_world = Eigen::Vector3f::UnitY();
    camera_forward_world[1] = 0.0;
    camera_forward_world = camera_forward_world.normalized();
    if (camera_forward_world.norm() < 1e-5) {
      // The horizontal camera placement is degenerate. Revert back to the
      // original placement.
      camera_forward_world = -body_forward_world;
      camera_up_world = spineLocalToWorldXF.linear() * Eigen::Vector3f::UnitX();
      camera_up_world = camera_up_world.normalized();
    }
  } else {
    camera_up_world = spineLocalToWorldXF.linear() * Eigen::Vector3f::UnitX();
    camera_up_world = camera_up_world.normalized();
  }

  if (cameraAngle != 0.0) {
    // Rotate camera_forward_world around camera_up_world by `cameraAngle` (rad
    // unit)
    camera_forward_world = Eigen::AngleAxisf(cameraAngle, camera_up_world) * camera_forward_world;
  }

  auto result = makeOutsideInCamera(
      camera_up_world,
      camera_forward_world,
      body_center_world_cm,
      cameraDistanceToBody_cm,
      imageHeight_pixels,
      imageWidth_pixels,
      focalLength_mm);

  return frameMesh(result, character, skelStates);
}

momentum::Camera createCameraForBody(
    const momentum::Character& character,
    const py::array_t<float>& skeletonStates,
    int imageHeight,
    int imageWidth,
    float focalLength_mm,
    bool horizontal,
    float cameraAngle) {
  const size_t nJoints = character.skeleton.joints.size();

  if (skeletonStates.ndim() < 2 || skeletonStates.ndim() > 3) {
    throw std::runtime_error(
        fmt::format(
            "create_camera_for_body: skeleton_states must be 2D (nJoints x 8), 3D (nFrames x nJoints x 8). Got {} dimensions.",
            skeletonStates.ndim()));
  }

  // Verify the last dimension is 8 (tx, ty, tz, rx, ry, rz, rw, s)
  const size_t lastDim = skeletonStates.shape(skeletonStates.ndim() - 1);
  if (lastDim != 8) {
    throw std::runtime_error(
        fmt::format(
            "create_camera_for_body: Expected last dimension to be 8 (tx, ty, tz, rx, ry, rz, rw, s), but got {}.",
            lastDim));
  }

  // Verify the second-to-last dimension matches the number of joints
  const size_t jointsDim = skeletonStates.shape(skeletonStates.ndim() - 2);
  if (jointsDim != nJoints) {
    throw std::runtime_error(
        fmt::format(
            "create_camera_for_body: Expected {} joints (second-to-last dimension), but got {}.",
            nJoints,
            jointsDim));
  }

  // Handle different dimensionalities with appropriate accessors
  if (skeletonStates.ndim() == 2) {
    // Unbatched: (nJoints, 8)
    auto accessor = skeletonStates.unchecked<2>();

    std::vector<momentum::SkeletonState> skelStates;
    skelStates.reserve(1);

    momentum::SkeletonState skelState;
    skelState.jointState.resize(nJoints);

    for (size_t iJoint = 0; iJoint < nJoints; ++iJoint) {
      Eigen::Vector3f translation(accessor(iJoint, 0), accessor(iJoint, 1), accessor(iJoint, 2));

      Eigen::Quaternionf rotation(
          accessor(iJoint, 6), // rw
          accessor(iJoint, 3), // rx
          accessor(iJoint, 4), // ry
          accessor(iJoint, 5)); // rz

      float scale = accessor(iJoint, 7);

      skelState.jointState[iJoint].transform.translation = translation;
      skelState.jointState[iJoint].transform.rotation = rotation;
      skelState.jointState[iJoint].transform.scale = scale;
    }

    skelStates.push_back(skelState);
    return makeOutsideInCameraForBody(
        character, skelStates, imageHeight, imageWidth, focalLength_mm, horizontal, cameraAngle);

  } else { // skeletonStates.ndim() == 3
    // Batched: (nBatch, nJoints, 8)
    auto accessor = skeletonStates.unchecked<3>();

    std::vector<momentum::SkeletonState> skelStates;
    skelStates.reserve(accessor.shape(0));
    for (size_t iPose = 0; iPose < accessor.shape(0); ++iPose) {
      momentum::SkeletonState skelState;
      skelState.jointState.resize(nJoints);

      for (size_t iJoint = 0; iJoint < nJoints; ++iJoint) {
        Eigen::Vector3f translation(
            accessor(iPose, iJoint, 0), accessor(iPose, iJoint, 1), accessor(iPose, iJoint, 2));

        Eigen::Quaternionf rotation(
            accessor(iPose, iJoint, 6), // rw
            accessor(iPose, iJoint, 3), // rx
            accessor(iPose, iJoint, 4), // ry
            accessor(iPose, iJoint, 5)); // rz

        float scale = accessor(iPose, iJoint, 7);

        skelState.jointState[iJoint].transform.translation = translation;
        skelState.jointState[iJoint].transform.rotation = rotation;
        skelState.jointState[iJoint].transform.scale = scale;
      }

      skelStates.push_back(skelState);
    }
    return makeOutsideInCameraForBody(
        character, skelStates, imageHeight, imageWidth, focalLength_mm, horizontal, cameraAngle);
  }
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
