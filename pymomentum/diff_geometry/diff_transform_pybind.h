/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declaration for the functions defined in diff_transform_pybind.cpp
void defineParameterTransforms(py::module_& m);
