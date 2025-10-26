/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/rasterizer/camera.h>

#include <ATen/ATen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <memory>
#include <vector>

namespace pymomentum {

// Returned camera is in cm unit.
std::vector<momentum::rasterizer::Camera> buildCamerasForBodyJoints(
    at::Tensor jointLocalToWorldTransforms,
    int spineJointIndexToOrientCamera,
    int imageHeight,
    int imageWidth);

// Build a vector of cameras that roughly face the body (default: facing the
// front of the body). If horizontal is true, the cameras are placed
// horizontally, assuming the Y axis is the world up direction. Returned camera
// is in cm unit.
// cameraAngle: control from what angle the camera looks at the body. Default:
// 0, looking at the front. Pi/2: looking at the body left side.
std::vector<momentum::rasterizer::Camera> buildCamerasForBody(
    const momentum::Character& character,
    at::Tensor jointParameters,
    int imageHeight,
    int imageWidth,
    float focalLength_mm,
    bool horizontal,
    float cameraAngle = 0.0f);

momentum::rasterizer::Camera createCameraForBody(
    const momentum::Character& character,
    const pybind11::array_t<float>& skeletonStates,
    int imageHeight,
    int imageWidth,
    float focalLength_mm,
    bool horizontal,
    float cameraAngle = 0.0f);

Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> triangulate(
    const Eigen::VectorXi& faceIndices,
    const Eigen::VectorXi& faceOffsets);

std::vector<momentum::rasterizer::Camera>
buildCamerasForHand(at::Tensor wristTransformation, int imageHeight, int imageWidth);

std::vector<momentum::rasterizer::Camera>
buildCamerasForHandSurface(at::Tensor wristTransformation, int imageHeight, int imageWidth);

} // namespace pymomentum
