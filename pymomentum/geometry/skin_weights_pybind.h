/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/skin_weights.h>
#include <pybind11/pybind11.h>

namespace pymomentum {

void registerSkinWeightsBindings(pybind11::class_<momentum::SkinWeights>& skinWeightsClass);

}
