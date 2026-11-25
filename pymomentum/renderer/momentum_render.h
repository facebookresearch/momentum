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

/// Creates a camera that keeps several characters in view, e.g. every character of a
/// multi-character file.
///
/// The camera faces the first character that has one of the joints that createCameraForBody()
/// faces, as createCameraForBody() does. If no character has one, the camera looks along the world
/// -Z axis with the world Y axis up, rotated by `cameraAngle` about the up axis. The camera then
/// moves so that every frame of every character is in view: the skinned mesh of a character with a
/// mesh and skin weights, otherwise its joints.
///
/// @param characters Characters to keep in view.
/// @param skeletonStates Skeleton states per character, (nJoints x 8) or (nFrames x nJoints x 8);
///   the number of frames can differ between characters.
/// @param imageHeight Image height in pixels.
/// @param imageWidth Image width in pixels.
/// @param focalLength_mm 35mm-equivalent focal length.
/// @param horizontal Whether to keep the camera horizontal (world Y up) when it faces a body.
/// @param cameraAngle Rotation (radians) of the view direction about the camera up axis.
momentum::Camera createCameraForCharacters(
    const std::vector<const momentum::Character*>& characters,
    const std::vector<pybind11::array_t<float>>& skeletonStates,
    int imageHeight,
    int imageWidth,
    float focalLength_mm,
    bool horizontal,
    float cameraAngle);

momentum::Camera createCameraForHand(
    const pybind11::array_t<float>& wristTransformation,
    int imageHeight,
    int imageWidth);

Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> triangulate(
    const Eigen::VectorXi& faceIndices,
    const Eigen::VectorXi& faceOffsets);

} // namespace pymomentum
