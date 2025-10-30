/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/math/mesh.h>

#include <pybind11/pybind11.h>

namespace pymomentum {

void registerMeshBindings(pybind11::class_<momentum::Mesh>& meshClass);

} // namespace pymomentum
