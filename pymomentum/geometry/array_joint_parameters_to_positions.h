/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace pymomentum {

namespace py = pybind11;

// Convert joint parameters to 3D positions relative to skeleton joints using forward kinematics.
// This is useful for supervising a model to produce the correct 3D ground truth.
//
// Input shapes:
//   joint_parameters: [..., nJoints * 7] (flat) or [..., nJoints, 7] (structured)
//   parents: [nPoints] - joint parent index for each target position
//   offsets: [nPoints, 3] - 3D offset in each joint's local space
//
// Output shape: [..., nPoints, 3] - world-space position of each point
py::array jointParametersToPositionsArray(
    const momentum::Character& character,
    const py::buffer& jointParameters,
    const py::buffer& parents,
    const py::buffer& offsets);

// Convert model parameters to 3D positions relative to skeleton joints using forward kinematics.
// This first applies the parameter transform to convert model parameters to joint parameters,
// then computes the positions using forward kinematics.
//
// Working directly from model parameters is preferable to mapping to joint parameters first
// because it does a better job exploiting the sparsity in the model and therefore can be
// made somewhat faster.
//
// Input shapes:
//   model_parameters: [..., nModelParams]
//   parents: [nPoints] - joint parent index for each target position
//   offsets: [nPoints, 3] - 3D offset in each joint's local space
//
// Output shape: [..., nPoints, 3] - world-space position of each point
py::array modelParametersToPositionsArray(
    const momentum::Character& character,
    const py::buffer& modelParameters,
    const py::buffer& parents,
    const py::buffer& offsets);

} // namespace pymomentum
