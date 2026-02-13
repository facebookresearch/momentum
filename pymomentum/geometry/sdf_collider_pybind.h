/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/sdf_collision_geometry.h>

#include <pybind11/pybind11.h>

namespace pymomentum {

void registerSDFColliderBindings(pybind11::class_<momentum::SDFColliderT<float>>& sdfColliderClass);

} // namespace pymomentum
