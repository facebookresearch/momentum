/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <pybind11/pybind11.h>

namespace pymomentum {

/// Registers Python bindings for logging control and redirection.
/// @param m The pybind11 module to register bindings on.
void registerLoggingBindings(pybind11::module& m);

} // namespace pymomentum
