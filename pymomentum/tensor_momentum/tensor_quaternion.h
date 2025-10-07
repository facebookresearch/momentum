/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <pybind11/pybind11.h>

#include <optional>

namespace pymomentum {

at::Tensor quaternionMultiply(at::Tensor q1, at::Tensor q2);
at::Tensor quaternionNormalize(at::Tensor q);
at::Tensor quaternionConjugate(at::Tensor q);
at::Tensor quaternionInverse(at::Tensor q);
at::Tensor quaternionRotateVector(at::Tensor q, at::Tensor v);

// Euler conversion:
at::Tensor xyzEulerToQuaternion(at::Tensor xyzEuler);
at::Tensor quaternionToXYZEuler(at::Tensor quat);

at::Tensor quaternionIdentity();

void checkQuaternion(at::Tensor q);
std::tuple<at::Tensor, at::Tensor> splitQuaternion(at::Tensor q);

at::Tensor quaternionToRotationMatrix(at::Tensor q);
at::Tensor rotationMatrixToQuaternion(at::Tensor q);

at::Tensor blendQuaternions(at::Tensor quaternions, std::optional<at::Tensor> weights);

at::Tensor checkAndNormalizeWeights(at::Tensor quaternions, std::optional<at::Tensor> weights);

} // namespace pymomentum
