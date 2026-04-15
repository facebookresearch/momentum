/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace pymomentum {

void addErrorFunctions(pybind11::module_& m);

void addAimAxisErrorFunctions(pybind11::module_& m);

void addProjectionErrorFunctions(pybind11::module_& m);

void addDistanceErrorFunctions(pybind11::module_& m);

void addSDFErrorFunctions(pybind11::module_& m);

} // namespace pymomentum
