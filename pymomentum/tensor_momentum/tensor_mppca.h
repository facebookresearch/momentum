/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/parameter_transform.h>
#include <momentum/math/fwd.h>

#include <ATen/ATen.h>

#include <optional>

namespace pymomentum {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> mppcaToTensors(
    const momentum::Mppca& mppca,
    std::optional<const momentum::ParameterTransform*> paramTransform);

}
