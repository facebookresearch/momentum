/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/character/inverse_parameter_transform.h>
#include <momentum/character/parameter_transform.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace pymomentum {

namespace py = pybind11;

// Apply parameter transform to model parameters, returning joint parameters.
// Supports arbitrary leading dimensions and both float32 and float64 dtypes.
// Accepts any buffer-protocol object (numpy arrays, torch tensors, etc.)
// for backward compatibility.
//
// Input shape: [..., numModelParams]
// Output shape: [..., numJoints * 7] if flatten=true, [..., numJoints, 7] if flatten=false
py::array applyParameterTransformArray(
    const momentum::ParameterTransform& paramTransform,
    const py::buffer& modelParams,
    bool flatten = true);

// Apply inverse parameter transform to joint parameters, returning model parameters.
// Supports arbitrary leading dimensions and both float32 and float64 dtypes.
// Accepts any buffer-protocol object (numpy arrays, torch tensors, etc.)
// for backward compatibility.
//
// Input shape: [..., numJoints, 7]
// Output shape: [..., numModelParams]
py::array applyInverseParameterTransformArray(
    const momentum::InverseParameterTransform& invParamTransform,
    const py::buffer& jointParams);

// Gets the parameter sets specified in the model file as a dictionary
// mapping strings to boolean numpy arrays.
std::unordered_map<std::string, py::array_t<bool>> getParameterSetsArray(
    const momentum::ParameterTransform& parameterTransform);

// Convert a ParameterSet to a boolean numpy array
py::array_t<bool> parameterSetToArray(
    const momentum::ParameterTransform& parameterTransform,
    const momentum::ParameterSet& paramSet);

py::array_t<bool> getScalingParametersArray(const momentum::ParameterTransform& parameterTransform);

py::array_t<bool> getRigidParametersArray(const momentum::ParameterTransform& parameterTransform);

py::array_t<bool> getAllParametersArray(const momentum::ParameterTransform& parameterTransform);

py::array_t<bool> getBlendShapeParametersArray(
    const momentum::ParameterTransform& parameterTransform);

py::array_t<bool> getFaceExpressionParametersArray(
    const momentum::ParameterTransform& parameterTransform);

py::array_t<bool> getPoseParametersArray(const momentum::ParameterTransform& parameterTransform);

void addParameterSetArray(
    momentum::ParameterTransform& parameterTransform,
    const std::string& paramSetName,
    const py::array_t<bool>& paramSet);

py::array_t<bool> getParametersForJointsArray(
    const momentum::ParameterTransform& parameterTransform,
    const std::vector<size_t>& jointIndices);

py::array_t<bool> findParametersArray(
    const momentum::ParameterTransform& parameterTransform,
    const std::vector<std::string>& parameterNames,
    bool allowMissing = false);

// Returns the parameter transform matrix as a numpy array
py::array_t<float> getParameterTransformMatrix(
    const momentum::ParameterTransform& parameterTransform);

} // namespace pymomentum
