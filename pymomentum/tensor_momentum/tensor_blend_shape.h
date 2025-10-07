/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <pybind11/pybind11.h>

namespace pymomentum {

at::Tensor applyBlendShapeCoefficients(pybind11::object blendShape, at::Tensor coeffs);

}
