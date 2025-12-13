/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>

#include <ATen/ATen.h>
#include <pybind11/pybind11.h>

namespace pymomentum {

at::Tensor modelParametersToSkeletonState(pybind11::object characters, at::Tensor modelParams);

at::Tensor modelParametersToLocalSkeletonState(pybind11::object characters, at::Tensor modelParams);

at::Tensor jointParametersToSkeletonState(pybind11::object characters, at::Tensor jointParams);

at::Tensor jointParametersToLocalSkeletonState(pybind11::object characters, at::Tensor jointParams);

at::Tensor skeletonStateToJointParameters(
    const momentum::Character& character,
    at::Tensor skelState);

at::Tensor localSkeletonStateToJointParameters(
    const momentum::Character& character,
    at::Tensor localSkelState);

at::Tensor matricesToSkeletonStates(at::Tensor matrices);

} // namespace pymomentum
