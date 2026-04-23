/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/camera/camera.h>
#include <momentum/character/character.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <memory>
#include <vector>

namespace pymomentum {

momentum::Camera createCameraForBody(
    const momentum::Character& character,
    const pybind11::array_t<float>& skeletonStates,
    int imageHeight,
    int imageWidth,
    float focalLength_mm,
    bool horizontal,
    float cameraAngle = 0.0f);

momentum::Camera createCameraForHand(
    const pybind11::array_t<float>& wristTransformation,
    int imageHeight,
    int imageWidth);

Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> triangulate(
    const Eigen::VectorXi& faceIndices,
    const Eigen::VectorXi& faceOffsets);

} // namespace pymomentum
