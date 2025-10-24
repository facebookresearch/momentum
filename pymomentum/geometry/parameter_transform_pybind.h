/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/inverse_parameter_transform.h>
#include <momentum/character/parameter_transform.h>

#include <pybind11/pybind11.h>

#include <ATen/ATen.h>

namespace pymomentum {

void registerParameterTransformBindings(
    pybind11::class_<momentum::ParameterTransform>& parameterTransformClass);

void registerInverseParameterTransformBindings(
    pybind11::class_<momentum::InverseParameterTransform>& inverseParameterTransformClass);

} // namespace pymomentum
