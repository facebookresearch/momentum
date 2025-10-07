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

#include <ATen/ATen.h>
#include <pybind11/pybind11.h>

namespace pymomentum {

at::Tensor applyParamTransform(
    const momentum::ParameterTransform* paramTransform,
    at::Tensor modelParams);

at::Tensor applyParamTransform(pybind11::object characters, at::Tensor modelParams);

// Gets the parameter sets specified in the model file as a dictionary
// mapping strings to boolean tensors.
std::unordered_map<std::string, at::Tensor> getParameterSets(
    const momentum::ParameterTransform& parameterTransform);

at::Tensor getScalingParameters(const momentum::ParameterTransform& parameterTransform);

at::Tensor getRigidParameters(const momentum::ParameterTransform& parameterTransform);

at::Tensor getAllParameters(const momentum::ParameterTransform& parameterTransform);

at::Tensor getBlendShapeParameters(const momentum::ParameterTransform& parameterTransform);

at::Tensor getPoseParameters(const momentum::ParameterTransform& parameterTransform);

at::Tensor getParametersForJoints(
    const momentum::ParameterTransform& parameterTransform,
    const std::vector<size_t>& jointIndices);

at::Tensor findParameters(
    const momentum::ParameterTransform& parameterTransform,
    const std::vector<std::string>& parameterNames,
    bool allowMissing = false);

at::Tensor parameterSetToTensor(
    const momentum::ParameterTransform& parameterTransform,
    const momentum::ParameterSet& paramSet);

at::Tensor applyInverseParamTransform(
    const momentum::InverseParameterTransform* invParamTransform,
    at::Tensor jointParams);

std::unique_ptr<momentum::InverseParameterTransform> createInverseParameterTransform(
    const momentum::ParameterTransform& transform);

// If the user passes an empty tensor for a parameter set, what kind of
// value to return.  This is different for different cases: sometimes we
// should include all parameters, sometimes none, and sometimes no reasonable
// default is possible.
enum class DefaultParameterSet { ALL_ONES, ALL_ZEROS, NO_DEFAULT };

momentum::ParameterSet tensorToParameterSet(
    const momentum::ParameterTransform& parameterTransform,
    at::Tensor paramSet,
    DefaultParameterSet defaultParamSet = DefaultParameterSet::NO_DEFAULT);

// Change flattened joint parameter motion tensor with shape: [... x
// (n_joint_params x 7)] to shape [... x n_joint_params x 7]. Return the same
// tensor if the input is already unflattened. If `unflattened` is not nullptr,
// assign true only if the function does the shape conversion.
at::Tensor unflattenJointParameters(
    const momentum::Character& character,
    at::Tensor tensor_in,
    bool* unflattened = nullptr);

// Change unflattened joint parameter motion tensor with shape [... x
// n_joint_params x 7)] to shape [... x (n_joint_params x 7)]. Return the same
// tensor if the input is already flattened. If `flattened` is not nullptr,
// assign true only if the function does the shape conversion.
at::Tensor flattenJointParameters(
    const momentum::Character& character,
    at::Tensor tensor_in,
    bool* flattened = nullptr);

at::Tensor modelParametersToBlendShapeCoefficients(
    const momentum::Character& character,
    at::Tensor modelParameters);

at::Tensor getParameterTransformTensor(const momentum::ParameterTransform& parameterTransform);

} // namespace pymomentum
