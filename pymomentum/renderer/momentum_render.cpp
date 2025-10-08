/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/renderer/momentum_render.h"

#include "pymomentum/tensor_momentum/tensor_parameter_transform.h"
#include "pymomentum/tensor_utility/tensor_utility.h"

#include <momentum/character/character.h>
#include <momentum/character/linear_skinning.h>
#include <momentum/character/parameter_transform.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character/skin_weights.h>
#include <momentum/math/mesh.h>

#include <momentum/rasterizer/camera.h>

#include <dispenso/parallel_for.h> // @manual
#include <Eigen/Core>

#include <string.h>

namespace pymomentum {

// Build a camera to look at an object:
// cameraUpWorld: world-space camera up-vector.
// cameraLookAtWorld: world-space direction where camera looks at.
// aimCenterWorld: world-space point the camera looks at
// distanceToAimCenter: distance of the camera to the location of the aim,
//   aimCenterWorld.
// Note: if cameraUpWorld is not orthogonal to cameraLookAtWorld, the function
// will fix it, as long as cameraUpWorld is not parallel to cameraLookAtWorld.
momentum::rasterizer::Camera makeOutsideInCamera(
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

  std::cout << "focal length pixels: " << focal_length_pixels << std::endl;

  // Create a PinholeIntrinsicsModel
  auto intrinsicsModel = std::make_shared<momentum::rasterizer::PinholeIntrinsicsModel>(
      imageWidth_pixels, imageHeight_pixels, focal_length_pixels, focal_length_pixels);

  std::cout << "fx: " << intrinsicsModel->fx() << std::endl;

  // Create and return the camera with the intrinsics model and transform
  return momentum::rasterizer::Camera(intrinsicsModel, worldToEyeXF);
}

momentum::rasterizer::Camera frameMesh(
    const momentum::rasterizer::Camera& cam_in,
    const momentum::Character& character,
    gsl::span<const momentum::SkeletonState> skelStates) {
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

momentum::rasterizer::Camera makeOutsideInCameraForBody(
    const momentum::Character& character,
    gsl::span<const momentum::SkeletonState> skelStates,
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
  for (size_t i = 0; i < skelStates.size(); ++i) {
    const auto& jointState = skelStates[i].jointState.at(spineJoint);
    spineLocalToWorldTransforms.emplace_back(
        jointState.translation().cast<double>(), jointState.rotation().cast<double>());
  }
  const std::vector<double> weights(skelStates.size(), 1.0 / skelStates.size());
  const auto blendedTransform = momentum::blendTransforms(
      gsl::span<const momentum::TransformT<double>>(spineLocalToWorldTransforms),
      gsl::span<const double>(weights));
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

std::vector<momentum::rasterizer::Camera> buildCamerasForBody(
    const momentum::Character& character,
    at::Tensor jointParameters,
    int imageHeight,
    int imageWidth,
    float focalLength_mm,
    bool horizontal,
    float cameraAngle) {
  jointParameters = flattenJointParameters(character, jointParameters);

  // If the user passes in a tensor of dimension [n x nJointParams] we assume
  // the n refers to the batch dimension, and add the nFrames dimension of 1
  // so it becomes [batchSize x 1 x nJointParams].
  // If the tensor is [nJointParams], we will add the nFrames dimension of 1
  // so it becomes [1 x nJointParams].
  if (jointParameters.ndimension() < 3) {
    jointParameters = jointParameters.unsqueeze(-2);
  }

  const int nFramesBinding = -1;

  TensorChecker checker("buildCamerasForBody");
  jointParameters = checker.validateAndFixTensor(
      jointParameters,
      "jointParameters",
      {nFramesBinding, (int)character.parameterTransform.numJointParameters()},
      {"nFrames", "nJointParams"},
      at::kFloat,
      true,
      false);

  const auto nBatch = checker.getBatchSize();
  std::vector<momentum::rasterizer::Camera> result(nBatch);

  for (int64_t iBatch = 0; iBatch < nBatch; ++iBatch) {
    at::Tensor sequence_cur = jointParameters.select(0, iBatch);

    const auto nFrames = sequence_cur.size(0);
    std::vector<momentum::SkeletonState> skelStates;
    skelStates.reserve(nFrames);
    for (int64_t iFrame = 0; iFrame < nFrames; ++iFrame) {
      at::Tensor jointParameters_cur = sequence_cur.select(0, iFrame);
      skelStates.emplace_back(toEigenMap<float>(jointParameters_cur), character.skeleton);
    }

    result[iBatch] = makeOutsideInCameraForBody(
        character, skelStates, imageHeight, imageWidth, focalLength_mm, horizontal, cameraAngle);
  } // end for iBatch

  return result;
}

template <typename T2, typename T1>
std::vector<Eigen::Vector3<T2>> castVector(const std::vector<Eigen::Vector3<T1>>& vec) {
  std::vector<Eigen::Vector3<T2>> result;
  result.reserve(vec.size());
  for (const auto& v : vec) {
    result.push_back(v.template cast<T2>());
  }
  return result;
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
    result(iTri, 0) = triangleIndices[iTri * 3 + 0];
    result(iTri, 1) = triangleIndices[iTri * 3 + 1];
    result(iTri, 2) = triangleIndices[iTri * 3 + 2];
  }
  return result;
}

std::vector<momentum::rasterizer::Camera>
buildCamerasForHand(at::Tensor wristTransformation, int imageHeight, int imageWidth) {
  TensorChecker checker("buildCamerasForHand");
  wristTransformation = checker.validateAndFixTensor(
      wristTransformation,
      "wristTransformation",
      {4, 4},
      {"nTransformMatrixRows", "nTransformMatrixCols"},
      at::kFloat,
      true,
      false);

  const auto nBatch = checker.getBatchSize();
  std::vector<momentum::rasterizer::Camera> result(nBatch);
  momentum::SkeletonState skelState;

  for (int64_t iBatch = 0; iBatch < nBatch; ++iBatch) {
    // in spine-local coords,
    //   x points up
    //   y points forward
    //   z points to the hand's left
    Eigen::Affine3f wristLocalToWorldXF = Eigen::Affine3f::Identity();
    const Eigen::Vector3f hand_up_world = wristLocalToWorldXF.rotation() * Eigen::Vector3f::UnitY();
    const Eigen::Vector3f hand_forward_world =
        -1.0 * wristLocalToWorldXF.rotation() * Eigen::Vector3f::UnitZ();

    at::Tensor wristtranslation_mm =
        wristTransformation.select(0, iBatch).select(1, iBatch).slice(0, 0, 3).contiguous();
    const Eigen::Vector3f hand_center_world_cm = toEigenMap<float>(wristtranslation_mm) * 0.1;

    const float cameraDistanceToHand_cm = 0.5f * 100; // 0.5 meters away.
    const Eigen::Vector3f& camera_up_world = hand_up_world;
    const Eigen::Vector3f camera_forward_world = -hand_forward_world;

    result[iBatch] = makeOutsideInCamera(
        camera_up_world,
        camera_forward_world,
        hand_center_world_cm,
        cameraDistanceToHand_cm,
        imageHeight,
        imageWidth);
  } // end for iBatch

  return result;
}

std::vector<momentum::rasterizer::Camera>
buildCamerasForHandSurface(at::Tensor wristTransformation, int imageHeight, int imageWidth) {
  TensorChecker checker("buildCamerasForHand");
  wristTransformation = checker.validateAndFixTensor(
      wristTransformation,
      "wristTransformation",
      {4, 4},
      {"nTransformMatrixRows", "nTransformMatrixCols"},
      at::kFloat,
      true,
      false);

  const auto nBatch = checker.getBatchSize();
  std::vector<momentum::rasterizer::Camera> result(nBatch);
  momentum::SkeletonState skelState;

  for (int64_t iBatch = 0; iBatch < nBatch; ++iBatch) {
    // in spine-local coords,
    //   x points up
    //   y points forward
    //   z points to the hand's left
    Eigen::Affine3f wristLocalToWorldXF = Eigen::Affine3f::Identity();
    const Eigen::Vector3f hand_up_world = wristLocalToWorldXF.rotation() * Eigen::Vector3f::UnitY();
    const Eigen::Vector3f hand_forward_world =
        -1.0 * wristLocalToWorldXF.rotation() * Eigen::Vector3f::UnitZ();

    const Eigen::Vector3f hand_center_world_cm = wristLocalToWorldXF.translation();
    const float cameraDistanceToHand_cm = 0.5f * 100; // 0.5 meters away.
    const Eigen::Vector3f& camera_up_world = hand_up_world;
    const Eigen::Vector3f camera_forward_world = -hand_forward_world;

    result[iBatch] = makeOutsideInCamera(
        camera_up_world,
        camera_forward_world,
        hand_center_world_cm,
        cameraDistanceToHand_cm,
        imageHeight,
        imageWidth);
  } // end for iBatch

  return result;
}

} // namespace pymomentum
