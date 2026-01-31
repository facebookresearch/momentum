/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <pymomentum/array_utility/default_parameter_set.h>

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

// Convert a boolean numpy array to a ParameterSet
// If the array is empty and defaultParamSet is not NO_DEFAULT, return the default set
// Note: DefaultParameterSet enum is defined in tensor_momentum/tensor_parameter_transform.h
momentum::ParameterSet arrayToParameterSet(
    const momentum::ParameterTransform& parameterTransform,
    const py::array_t<bool>& paramSet,
    DefaultParameterSet defaultParamSet = DefaultParameterSet::NoDefault);

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

// Convert uniform random values [0, 1] to valid model parameters.
// Maps uniform noise to appropriate ranges based on parameter type:
// - Scale parameters: [-0.5, 0.5]
// - Translation parameters: [-2.5, 2.5]
// - Rotation parameters: [-pi/8, pi/8]
//
// Input shape: [..., numModelParams]
// Output shape: [..., numModelParams]
py::array uniformRandomToModelParametersArray(
    const momentum::Character& character,
    py::buffer unifNoise);

// Extract blend shape coefficients from model parameters.
// Input: (..., nModelParams) - model parameters
// Output: (..., nBlendShapes) - blend shape coefficients
py::array modelParametersToBlendShapeCoefficientsArray(
    const momentum::Character& character,
    const py::buffer& modelParameters);

// Extract face expression blend shape coefficients from model parameters.
// Input: (..., nModelParams) - model parameters
// Output: (..., nFaceExpressionBlendShapes) - face expression coefficients
py::array modelParametersToFaceExpressionCoefficientsArray(
    const momentum::Character& character,
    const py::buffer& modelParameters);

// Map model parameters from one character to another by matching parameter names.
// Parameters that don't exist in the target character are dropped.
// Parameters that exist in the target but not source are set to zero.
//
// Input shape: [..., numSourceModelParams]
// Output shape: [..., numTargetModelParams]
py::array mapModelParametersArray(
    py::buffer motionData,
    const momentum::Character& srcCharacter,
    const momentum::Character& tgtCharacter,
    bool verbose = true);

// Map model parameters using explicit parameter names (source names -> target character).
// This overload is useful when you have parameter names from a file but not the full character.
//
// Input shape: [..., len(sourceParameterNames)]
// Output shape: [..., numTargetModelParams]
py::array mapModelParametersNamesArray(
    py::buffer motionData,
    const std::vector<std::string>& sourceParameterNames,
    const momentum::Character& targetCharacter,
    bool verbose = false);

// Map joint parameters from one character to another by matching joint names.
// Joints that don't exist in the target character are dropped.
// Joints that exist in the target but not source are set to zero.
//
// Input shape: [..., numSourceJoints * 7] (flat) or [..., numSourceJoints, 7] (structured)
// Output shape: Same format as input, with target joint count
py::array mapJointParametersArray(
    py::buffer motionData,
    const momentum::Character& srcCharacter,
    const momentum::Character& tgtCharacter);

} // namespace pymomentum
