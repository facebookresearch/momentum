/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>

#include <optional>

namespace pymomentum {

at::Tensor skeletonStateToTransforms(at::Tensor skeletonState);
at::Tensor multiplySkeletonStates(at::Tensor skelState1, at::Tensor skelState2);
at::Tensor transformPointsWithSkeletonState(at::Tensor skelState, at::Tensor p);
at::Tensor inverseSkeletonStates(at::Tensor skelState);

at::Tensor quaternionToSkeletonState(at::Tensor q);
at::Tensor translationToSkeletonState(at::Tensor t);
at::Tensor scaleToSkeletonState(at::Tensor s);

std::tuple<at::Tensor, at::Tensor, at::Tensor> splitSkeletonState(at::Tensor skelState);

at::Tensor identitySkeletonState();

at::Tensor blendSkeletonStates(at::Tensor skel_states, std::optional<at::Tensor> weights);

} // namespace pymomentum
