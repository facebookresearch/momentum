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

#include <optional>

namespace pymomentum {

namespace py = pybind11;

/// Apply linear blend skinning to compute vertex positions from skeleton state or transforms.
///
/// Transforms rest-pose vertices to world space using the character's skin weights
/// and the provided skeleton state or transformation matrices. The function
/// auto-detects the input format and converts as needed.
///
/// @param character The character containing skin weights and inverse bind pose.
/// @param skelState Either:
///                  - Skeleton state with shape [..., nJoints, 8] where each joint
///                    has [tx, ty, tz, rx, ry, rz, rw, scale], OR
///                  - Transform matrices with shape [..., nJoints, 4, 4] containing
///                    4x4 transformation matrices.
/// @param restVertices Optional rest pose vertices with shape [..., nVertices, 3].
///                     If not provided, uses character's mesh vertices.
/// @return World-space vertex positions with shape [..., nVertices, 3].
py::array skinPointsArray(
    const momentum::Character& character,
    py::buffer skelState,
    std::optional<py::buffer> restVertices);

/// Apply linear blend skinning to compute world-space positions of skinned locators.
///
/// Uses the character's built-in skinned locators and applies linear blend skinning
/// to compute their world-space positions given a skeleton state.
///
/// @param character The character containing skinned locators and inverse bind pose.
/// @param skelState Skeleton state with shape [..., nJoints, 8] where each joint
///                  has [tx, ty, tz, rx, ry, rz, rw, scale].
/// @param restPositions Optional rest pose positions with shape [..., nLocators, 3].
///                      If not provided, uses positions from character's skinned locators.
/// @return World-space locator positions with shape [..., nLocators, 3].
py::array skinSkinnedLocatorsArray(
    const momentum::Character& character,
    py::buffer skelState,
    std::optional<py::buffer> restPositions);

} // namespace pymomentum
