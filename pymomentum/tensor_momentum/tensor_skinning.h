/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <momentum/character/fwd.h>
#include <pybind11/pybind11.h>

#include <optional>

namespace pymomentum {

at::Tensor
skinPoints(pybind11::object character, at::Tensor skel_state, std::optional<at::Tensor> restPoints);

at::Tensor computeVertexNormals(at::Tensor positions, at::Tensor triangles);

at::Tensor skinSkinnedLocators(
    const momentum::Character& character,
    at::Tensor skel_state,
    const std::optional<at::Tensor>& restPositions = std::nullopt);

} // namespace pymomentum
