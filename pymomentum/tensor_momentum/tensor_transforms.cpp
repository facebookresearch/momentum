/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/tensor_momentum/tensor_transforms.h"

#include "pymomentum/tensor_momentum/tensor_quaternion.h"
#include "pymomentum/tensor_utility/tensor_utility.h"

#include <ATen/Functions.h>
#include <ceres/jet.h>
#include <dispenso/parallel_for.h> // @manual
#include <momentum/common/exception.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace pymomentum {

void checkSkelState(at::Tensor skelState) {
  MT_THROW_IF(
      skelState.size(-1) != 8,
      "Expected skeleton state to have last dimension 8 (tx, ty, tz, rx, ry, rz, rw, s)");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> splitSkeletonState(at::Tensor skelState) {
  checkSkelState(skelState);
  return {skelState.narrow(-1, 0, 3), skelState.narrow(-1, 3, 4), skelState.narrow(-1, 7, 1)};
}

at::Tensor skeletonStateToTransforms(at::Tensor skeletonState) {
  checkSkelState(skeletonState);
  auto [t, q, s] = splitSkeletonState(skeletonState);
  const at::Tensor rotMat = quaternionToRotationMatrix(q);
  const at::Tensor linear = rotMat * s.unsqueeze(-2).expand_as(rotMat);
  const at::Tensor affine = at::cat({linear, t.unsqueeze(-1)}, -1);

  const at::Tensor lastRow = [&]() {
    at::Tensor result =
        to1DTensor(Eigen::Vector4f(0, 0, 0, 1)).to(skeletonState.device(), skeletonState.dtype());
    std::vector<int64_t> result_shape;
    while (result.dim() < affine.dim()) {
      result = result.unsqueeze(0);
    }

    for (int64_t i = 0; i < affine.dim(); ++i) {
      result_shape.push_back(affine.size(i));
    }
    result_shape[result_shape.size() - 2] = 1;
    return result.expand(result_shape);
  }();

  at::Tensor result = at::cat({affine, lastRow}, -2);
  return result;
}

// Expands the left tensor to match the right tensor.  This supports the
// common use case of transforming a whole bunch of points or a whole bunch of
// skeleton state by a single transform.
at::Tensor matchLeadingDimensions(at::Tensor tLeft, at::Tensor tRight) {
  MT_THROW_IF(
      tRight.dim() < tLeft.dim(), "First tensor can't have larger dimensionality than the second.");

  while (tLeft.dim() < tRight.dim()) {
    tLeft = tLeft.unsqueeze(0);
  }

  std::vector<int64_t> new_dim;
  new_dim.reserve(tLeft.dim());
  for (int64_t iDim = 0; iDim < tLeft.dim() - 1; ++iDim) {
    if (tLeft.size(iDim) == 1) {
      new_dim.push_back(tRight.size(iDim));
    } else if (tLeft.size(iDim) == tRight.size(iDim)) {
      new_dim.push_back(tLeft.size(iDim));
    } else {
      MT_THROW("Tensors should match in all nonsingular dimensions.");
    }
  }

  new_dim.push_back(-1);
  return tLeft.expand(new_dim);
}

at::Tensor multiplySkeletonStates(at::Tensor skelState1, at::Tensor skelState2) {
  checkSkelState(skelState1);
  checkSkelState(skelState2);
  skelState1 = matchLeadingDimensions(skelState1, skelState2);

  auto [t1, q1, s1] = splitSkeletonState(skelState1);
  auto [t2, q2, s2] = splitSkeletonState(skelState2);

  auto tRes = t1 + quaternionRotateVector(q1, s1.expand_as(t2) * t2);
  auto sRes = s1 * s2;
  auto qRes = quaternionMultiply(q1, q2);

  return at::cat({tRes, qRes, sRes}, -1);
}

at::Tensor quaternionToSkeletonState(at::Tensor q) {
  checkQuaternion(q);
  const auto sz = q.sizes();
  MT_THROW_IF(sz.empty(), "Empty quaternion tensor");

  std::vector<int64_t> sz_trans(std::begin(sz), std::end(sz));
  assert(sz_trans.size() > 0); // guaranteed by the sz.empty() check above.
  sz_trans.back() = 3;

  std::vector<int64_t> sz_scale(std::begin(sz), std::end(sz));
  assert(sz_scale.size() > 0);
  sz_scale.back() = 1;
  return at::cat({at::zeros(sz_trans), q, at::ones(sz_scale)}, -1);
}

at::Tensor translationToSkeletonState(at::Tensor t) {
  MT_THROW_IF(t.size(-1) != 3, "Expected 3-dimensional translation vector.");

  const auto sz = t.sizes();
  MT_THROW_IF(sz.empty(), "Empty quaternion tensor");

  std::vector<int64_t> sz_scale(std::begin(sz), std::end(sz));
  assert(sz_scale.size() > 0); // guaranteed by the sz.empty() check above.
  sz_scale.back() = 1;

  std::vector<int64_t> sz_rot(std::begin(sz), std::end(sz));
  assert(sz_rot.size() > 0); // guaranteed by the sz.empty() check above.
  sz_rot.back() = 4;

  return at::cat({t, quaternionIdentity().expand(sz_rot), at::ones(sz_scale)}, -1);
}

at::Tensor scaleToSkeletonState(at::Tensor s) {
  const auto sz = s.sizes();
  MT_THROW_IF(sz.empty(), "Empty quaternion tensor");

  std::vector<int64_t> sz_trans(std::begin(sz), std::end(sz));
  assert(sz_trans.size() > 0); // guaranteed by the sz.empty() check above.
  sz_trans.back() = 3;

  std::vector<int64_t> sz_rot(std::begin(sz), std::end(sz));
  assert(sz_rot.size() > 0); // guaranteed by the sz.empty() check above.
  sz_rot.back() = 4;

  return at::cat({at::zeros(sz_trans), quaternionIdentity().expand(sz_rot), s}, -1);
}

at::Tensor transformPointsWithSkeletonState(at::Tensor skelState, at::Tensor p) {
  checkSkelState(skelState);
  MT_THROW_IF(p.dim() < 1 || p.size(-1) != 3, "Points tensor should have last dimension 3.");
  skelState = matchLeadingDimensions(skelState, p);

  auto [t, q, s] = splitSkeletonState(skelState);
  return t + quaternionRotateVector(q, s.expand_as(p) * p);
}

at::Tensor inverseSkeletonStates(at::Tensor skelState) {
  auto [t, q, s] = splitSkeletonState(skelState);
  auto qInv = quaternionInverse(q);
  auto sInv = at::reciprocal(s);

  return at::cat(
      {-sInv * quaternionRotateVector(qInv, t), quaternionInverse(q), at::reciprocal(s)}, -1);
}

at::Tensor identitySkeletonState() {
  return at::cat(
      {at::zeros(std::vector<int64_t>{3}), quaternionIdentity(), at::ones(std::vector<int64_t>{1})},
      -1);
}

at::Tensor blendSkeletonStates(at::Tensor skel_states, std::optional<at::Tensor> weights_in) {
  auto [t, q, s] = splitSkeletonState(skel_states);
  at::Tensor weights = checkAndNormalizeWeights(q, weights_in);
  at::Tensor t_blend = (weights.unsqueeze(-1).expand_as(t) * t).sum(-2);
  at::Tensor q_blend = blendQuaternions(q, weights);
  at::Tensor s_blend = (weights.unsqueeze(-1).expand_as(s) * s).sum(-2);
  return at::cat({t_blend, q_blend, s_blend}, -1);
}

} // namespace pymomentum
