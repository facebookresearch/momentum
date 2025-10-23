/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace pymomentum {

/**
 * Register Python bindings for axel::TriBvh and related types.
 */
void registerTriBvhBindings(py::module& m);

} // namespace pymomentum
