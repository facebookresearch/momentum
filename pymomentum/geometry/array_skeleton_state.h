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

// Convert joint parameters to skeleton state (global transforms).
// The skeleton state is stored as (tx, ty, tz, rx, ry, rz, rw, scale) per joint,
// where each represents the transform from the joint's local space to world space.
// Quaternions are in (x, y, z, w) format.
// Accepts any buffer-protocol object (numpy arrays, torch tensors, etc.)
// for backward compatibility.
//
// Input shape: [..., numJointParams] or [..., numJoints, 7]
// Output shape: [..., numJoints, 8]
//
// Supports both float32 and float64 dtypes (auto-detected from input).
py::array jointParametersToSkeletonStateArray(
    const momentum::Character& character,
    const py::buffer& jointParams);

// Convert model parameters to skeleton state (global transforms).
// This is equivalent to calling applyParameterTransform followed by
// jointParametersToSkeletonState.
// Accepts any buffer-protocol object (numpy arrays, torch tensors, etc.)
// for backward compatibility.
//
// Input shape: [..., numModelParams]
// Output shape: [..., numJoints, 8]
//
// Supports both float32 and float64 dtypes (auto-detected from input).
py::array modelParametersToSkeletonStateArray(
    const momentum::Character& character,
    const py::buffer& modelParams);

// Convert joint parameters to local skeleton state (local transforms).
// The local skeleton state is stored as (tx, ty, tz, rx, ry, rz, rw, scale) per joint,
// where each represents the local transform of the joint.
// Quaternions are in (x, y, z, w) format.
// Accepts any buffer-protocol object (numpy arrays, torch tensors, etc.)
// for backward compatibility.
//
// Input shape: [..., numJointParams] or [..., numJoints, 7]
// Output shape: [..., numJoints, 8]
//
// Supports both float32 and float64 dtypes (auto-detected from input).
py::array jointParametersToLocalSkeletonStateArray(
    const momentum::Character& character,
    const py::buffer& jointParams);

// Convert model parameters to local skeleton state (local transforms).
// Accepts any buffer-protocol object (numpy arrays, torch tensors, etc.)
// for backward compatibility.
//
// Input shape: [..., numModelParams]
// Output shape: [..., numJoints, 8]
//
// Supports both float32 and float64 dtypes (auto-detected from input).
py::array modelParametersToLocalSkeletonStateArray(
    const momentum::Character& character,
    const py::buffer& modelParams);

// Convert skeleton state (global transforms) back to joint parameters.
// Note that this conversion is not unique due to the non-uniqueness of Euler angle conversion.
// Accepts any buffer-protocol object (numpy arrays, torch tensors, etc.)
// for backward compatibility.
//
// Input shape: [..., numJoints, 8]
// Output shape: [..., numJoints, 7]
//
// Supports both float32 and float64 dtypes (auto-detected from input).
py::array skeletonStateToJointParametersArray(
    const momentum::Character& character,
    const py::buffer& skeletonState);

// Convert local skeleton state (local transforms) back to joint parameters.
// Accepts any buffer-protocol object (numpy arrays, torch tensors, etc.)
// for backward compatibility.
//
// Input shape: [..., numJoints, 8]
// Output shape: [..., numJoints, 7]
//
// Supports both float32 and float64 dtypes (auto-detected from input).
py::array localSkeletonStateToJointParametersArray(
    const momentum::Character& character,
    const py::buffer& localSkeletonState);

} // namespace pymomentum
