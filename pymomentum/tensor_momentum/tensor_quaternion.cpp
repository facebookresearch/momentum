/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/tensor_momentum/tensor_quaternion.h"

#include "pymomentum/tensor_utility/autograd_utility.h"
#include "pymomentum/tensor_utility/tensor_utility.h"

#include <momentum/common/exception.h>
#include <momentum/diff_ik/ceres_utility.h>

#include <ATen/Functions.h>
#include <ceres/jet.h>
#include <dispenso/parallel_for.h> // @manual
#ifndef PYMOMENTUM_LIMITED_TORCH_API
#include <torch/csrc/jit/python/python_ivalue.h>
#endif
#include <Eigen/Core>

namespace pymomentum {

using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

namespace {

template <typename T>
struct XYZEulerToQuaternionFunction
    : public torch::autograd::Function<XYZEulerToQuaternionFunction<T>> {
 public:
  static variable_list forward(AutogradContext* ctx, at::Tensor xyzEuler);

  static variable_list backward(AutogradContext* ctx, variable_list grad_rotationMatrices);
};

template <typename T>
Eigen::Quaternion<T> xyzEulerToQuaternion(const Eigen::Vector3<T>& eulerAngles) {
  Eigen::Quaternion<T> result = Eigen::Quaternion<T>::Identity();
  for (Eigen::Index l = 0; l < 3; ++l) {
    result = Eigen::AngleAxis<T>(eulerAngles[l], Eigen::Vector3<T>::Unit(l)) * result;
  }
  return result;
}

template <typename T>
variable_list XYZEulerToQuaternionFunction<T>::forward(AutogradContext* ctx, at::Tensor xyzEuler) {
  const auto nEuler_index = -1;

  ctx->save_for_backward({xyzEuler});

  TensorChecker checker("euler_xyz_to_quaternion");

  const auto input_device = xyzEuler.device();

  bool squeeze = false;
  xyzEuler = checker.validateAndFixTensor(
      xyzEuler,
      "xyzEuler",
      {nEuler_index, 3},
      {"nEuler", "xyz"},
      toScalarType<T>(),
      true,
      false,
      &squeeze);

  const auto nEuler = checker.getBoundValue(nEuler_index);
  const auto nBatch = checker.getBatchSize();

  at::Tensor result = at::zeros({nBatch, nEuler, 4}, at::CPU(toScalarType<T>()));

  dispenso::parallel_for(0, nBatch, [&](size_t iBatch) {
    at::Tensor xyzEuler_cur = xyzEuler.select(0, iBatch);
    at::Tensor result_cur = result.select(0, iBatch);

    Eigen::Map<Eigen::VectorX<T>> xyzEuler_map = toEigenMap<T>(xyzEuler_cur);
    Eigen::Map<Eigen::VectorX<T>> result_map = toEigenMap<T>(result_cur);

    for (Eigen::Index k = 0; k < nEuler; ++k) {
      result_map.template segment<4>(4 * k) =
          xyzEulerToQuaternion<T>(xyzEuler_map.template segment<3>(3 * k)).coeffs();
    }
  });

  if (squeeze) {
    result = result.squeeze(0);
  }

  return {result.to(input_device)};
}

template <typename T>
variable_list XYZEulerToQuaternionFunction<T>::backward(
    AutogradContext* ctx,
    variable_list grad_outputs) {
  TensorChecker checker("euler_xyz_to_quaternion");
  const auto nEuler_index = -1;

  MT_THROW_IF(
      grad_outputs.size() != 1,
      "Invalid grad_outputs in ApplyParameterTransformFunction::backward");

  bool squeeze = false;
  const auto input_device = grad_outputs[0].device();

  at::Tensor dLoss_dQuat = checker.validateAndFixTensor(
      grad_outputs[0],
      "dLoss_dQuat",
      {nEuler_index, 4},
      {"nEuler", "xyzw"},
      toScalarType<T>(),
      true,
      true,
      &squeeze);

  const auto saved = ctx->get_saved_variables();
  auto savedItr = std::begin(saved);
  at::Tensor xyzEuler = checker.validateAndFixTensor(
      *savedItr++,
      "xyzEuler",
      {nEuler_index, 3},
      {"nEuler", "xyz"},
      toScalarType<T>(),
      true,
      false,
      nullptr);

  const auto nBatch = checker.getBatchSize();
  const auto nEuler = checker.getBoundValue(nEuler_index);
  at::Tensor d_xyzEuler = at::zeros({nBatch, nEuler, 3}, at::CPU(toScalarType<T>()));

  dispenso::parallel_for(0, nBatch, [&](size_t iBatch) {
    at::Tensor xyzEuler_cur = xyzEuler.select(0, iBatch);
    at::Tensor dLoss_dQuat_cur = dLoss_dQuat.select(0, iBatch);
    at::Tensor d_xyzEuler_cur = d_xyzEuler.select(0, iBatch);

    Eigen::Map<Eigen::VectorX<T>> xyzEuler_map = toEigenMap<T>(xyzEuler_cur);
    Eigen::Map<Eigen::VectorX<T>> dLoss_dQuat_map = toEigenMap<T>(dLoss_dQuat_cur);
    Eigen::Map<Eigen::VectorX<T>> d_xyzEuler_map = toEigenMap<T>(d_xyzEuler_cur);

    using JetType = ceres::Jet<T, 3>;

    for (Eigen::Index k = 0; k < nEuler; ++k) {
      d_xyzEuler_map.template segment<3>(3 * k) =
          xyzEulerToQuaternion<JetType>(
              momentum::buildJetVec<T, 3>(xyzEuler_map.template segment<3>(3 * k)))
              .coeffs()
              .dot(dLoss_dQuat_map.template segment<4>(4 * k))
              .v;
    }
  });

  if (squeeze) {
    d_xyzEuler = d_xyzEuler.sum(0);
  }

  return {d_xyzEuler.to(input_device)};
}

at::Tensor sqr(at::Tensor val) {
  return val * val;
}

} // namespace

void checkQuaternion(at::Tensor q) {
  MT_THROW_IF(q.size(-1) != 4, "Quaternion should have last dimension equal to 4.");
}

std::tuple<at::Tensor, at::Tensor> splitQuaternion(at::Tensor q) {
  checkQuaternion(q);

  return {q.narrow(-1, 3, 1), q.narrow(-1, 0, 3)};
}

at::Tensor quaternionMultiply(at::Tensor q1, at::Tensor q2) {
  checkQuaternion(q1);
  checkQuaternion(q2);

  auto [r1, v1] = splitQuaternion(q1);
  auto [r2, v2] = splitQuaternion(q2);

  MT_THROW_IF(q1.sizes() != q2.sizes(), "Expected matching quaternion dimensions.");

  // (r1*v1 + r2*v2 + v1 x v2, r1*r2 - v1.v2)
  // Dot product here is a product followed by a sum because I can't figure out
  // what 'tensordot' is actually supposed to do.
  at::Tensor r_res = r1 * r2 - (v1 * v2).sum(-1, true);
  at::Tensor v_res = r1.expand_as(v2) * v2 + r2.expand_as(v1) * v1 + at::cross(v1, v2, -1);

  return at::cat({v_res, r_res}, -1);
}

at::Tensor quaternionNormalize(at::Tensor q) {
  checkQuaternion(q);
  return q / q.norm(2, {-1}, true).expand_as(q);
}

at::Tensor quaternionConjugate(at::Tensor q) {
  checkQuaternion(q);
  auto [scalar, vec] = splitQuaternion(q);

  return at::concat({-vec, scalar}, -1);
}

at::Tensor quaternionInverse(at::Tensor q) {
  return quaternionConjugate(q) / at::sum(q * q, -1, true).expand_as(q);
}

at::Tensor quaternionToXYZEuler(at::Tensor q) {
  checkQuaternion(q);
  at::Tensor qx = q.select(-1, 0);
  at::Tensor qy = q.select(-1, 1);
  at::Tensor qz = q.select(-1, 2);
  at::Tensor qw = q.select(-1, 3);

  at::Tensor rx = at::atan2(2 * (qw * qx + qy * qz), 1 - 2 * (sqr(qx) + sqr(qy)));
  at::Tensor ry = at::asin(2 * (qw * qy - qz * qx));
  at::Tensor rz = at::atan2(2 * (qw * qz + qx * qy), 1 - 2 * (sqr(qy) + sqr(qz)));
  return at::stack({rx, ry, rz}, -1);
}

at::Tensor quaternionRotateVector(at::Tensor q, at::Tensor v) {
  auto [r, axis] = splitQuaternion(q);
  at::Tensor av = at::cross(axis, v, -1);
  at::Tensor aav = at::cross(axis, av, -1);
  return v + 2 * (av * r + aav);
}

at::Tensor xyzEulerToQuaternion(at::Tensor xyzEuler) {
  return applyTemplatedAutogradFunction<XYZEulerToQuaternionFunction>(xyzEuler)[0];
}

at::Tensor quaternionIdentity() {
  const Eigen::Vector4f tmp = Eigen::Quaternionf::Identity().coeffs();
  return to1DTensor(tmp);
}

at::Tensor quaternionToRotationMatrix(at::Tensor q) {
  MT_THROW_IF(q.size(-1) != 4, "Expected quaternion tensor (last dimension=4).");

  const at::Tensor qx = q.select(-1, 0).unsqueeze(-1);
  const at::Tensor qy = q.select(-1, 1).unsqueeze(-1);
  const at::Tensor qz = q.select(-1, 2).unsqueeze(-1);
  const at::Tensor qw = q.select(-1, 3).unsqueeze(-1);

  const at::Tensor qx2 = at::square(qx);
  const at::Tensor qy2 = at::square(qy);
  const at::Tensor qz2 = at::square(qz);

  const at::Tensor qxqy = qx * qy;
  const at::Tensor qxqz = qx * qz;
  const at::Tensor qxqw = qx * qw;
  const at::Tensor qyqz = qy * qz;
  const at::Tensor qyqw = qy * qw;
  const at::Tensor qzqw = qz * qw;
  const at::Tensor one = at::ones_like(qx);

  const auto sizes_init = q.sizes();
  std::vector<int64_t> result_size(sizes_init.begin(), sizes_init.end() - 1);
  result_size.push_back(3);
  result_size.push_back(3);
  at::Tensor result = at::cat(
                          {one - 2 * (qy2 + qz2),
                           2 * (qxqy - qzqw),
                           2 * (qxqz + qyqw),
                           2 * (qxqy + qzqw),
                           one - 2 * (qx2 + qz2),
                           2 * (qyqz - qxqw),
                           2 * (qxqz - qyqw),
                           2 * (qyqz + qxqw),
                           one - 2 * (qx2 + qy2)},
                          -1)
                          .reshape(result_size);
  return result;
}

at::Tensor rotationMatrixToQuaternion(at::Tensor matrices) {
  // Convert a rotation matrix to a quaternion using the method described here:
  // https://math.stackexchange.com/questions/893984/conversion-of-rotation-matrix-to-quaternion
  // Assumes that the input is a rotation matrix, will return the wrong result
  // if not.

  /*
  # Python version of the following code:
  eigenvalues, eigenvectors = torch.linalg.eig(matrices)
  max_eig, max_eig_ind = torch.max(eigenvalues.real, -1)

  trace_m : torch.Tensor = matrices[..., 0, 0] + matrices[..., 1, 1] +
  matrices[..., 2, 2] cos_theta = (trace_m - 1) / 2 cos_half_theta =
  torch.sqrt(torch.clamp((1 + cos_theta) / 2.0, min=0)) sin_half_theta =
  torch.sqrt(torch.clamp((1 - cos_theta) / 2.0, min=0))
  qv : torch.Tensor = (
       torch.gather(
           eigenvectors,
           -1,
           max_eig_ind.unsqueeze(-1).unsqueeze(-1).expand(*max_eig_ind.shape, 3,
  1),
       )
       .squeeze(-1)
       .real
   )
   qv = sin_half_theta.unsqueeze(-1).expand_as(qv) * qv

   qx : torch.Tensor = qv[..., 0].unsqueeze(-1)
   qy : torch.Tensor = qv[..., 1].unsqueeze(-1)
   qz : torch.Tensor = qv[..., 2].unsqueeze(-1)
   qw : torch.Tensor = cos_half_theta.unsqueeze(-1)

   symmetric_part_diff : torch.Tensor = torch.stack(
       [
           2 * qx * qy - matrices[..., 0, 1].unsqueeze(-1),
           2 * qx * qz - matrices[..., 0, 2].unsqueeze(-1),
           2 * qy * qz - matrices[..., 1, 2].unsqueeze(-1),
       ],
       -1,
   )
   skew_symmetric_part = torch.stack([-2 * qz, 2 * qy, -2 * qx], -1)
   diff_positive = symmetric_part_diff +
   qw.unsqueeze(-1).expand_as(skew_symmetric_part) * skew_symmetric_part
   diff_negative = symmetric_part_diff -
   qw.unsqueeze(-1).expand_as(skew_symmetric_part) * skew_symmetric_part

   qw = torch.where(
       torch.linalg.norm(diff_positive, dim=-1) <
  torch.linalg.norm(diff_negative,  dim=-1), qw, -qw)

   return torch.cat([qv, qw], -1)
  */

  const auto [eigenvalues, eigenvectors] = at::linalg_eig(matrices);

  using at::indexing::Ellipsis;

  // Angle can be read off the trace, as described in the SO post:
  const at::Tensor trace_m = matrices.index({Ellipsis, 0, 0}) + matrices.index({Ellipsis, 1, 1}) +
      matrices.index({Ellipsis, 2, 2});
  const at::Tensor cos_theta = (trace_m - 1) / 2;

  // For quaternion, we need cos(theta/2) and sin(theta/2):
  // TODO is use of half-angle formula here bad for precision?
  const at::Tensor cos_half_theta = at::sqrt(at::clamp((1 + cos_theta) / 2.0, 0));
  const at::Tensor sin_half_theta = at::sqrt(at::clamp((1 - cos_theta) / 2.0, 0));

  // There is one vector for which R * v = v; that vector must be the axis of
  // rotation and has an eigenvalue of 1.  Because the eigenvalues aren't sorted
  // we'll need to find it by looking for eigenvalue with the largest real
  // component (which should be 1).
  const auto [max_eig, max_eig_ind] = at::max(at::real(eigenvalues), -1);

  // Extract the eigenvector matching the real eigenvalue 1.  This requires a
  // torch.gather because we're extracting a different eigenvector for each
  // input (the eigenvalues are not sorted and anyway they all have norm 1).
  std::vector<int64_t> max_eig_ind_sizes(max_eig_ind.dim(), -1);
  max_eig_ind_sizes.push_back(3);
  max_eig_ind_sizes.push_back(1);
  at::Tensor qv = at::real(
      eigenvectors.gather(-1, max_eig_ind.unsqueeze(-1).unsqueeze(-1).expand(max_eig_ind_sizes))
          .squeeze(-1));
  qv = sin_half_theta.unsqueeze(-1).expand_as(qv) * qv;

  at::Tensor qx = qv.index({Ellipsis, 0}).unsqueeze(-1);
  at::Tensor qy = qv.index({Ellipsis, 1}).unsqueeze(-1);
  at::Tensor qz = qv.index({Ellipsis, 2}).unsqueeze(-1);
  at::Tensor qw = cos_half_theta.unsqueeze(-1);

  // Because of the way we derived qw, we don't know if we have the sign correct
  // (this depends also on which way v is pointing).  To make sure we use the
  // correct one, reconstruct the upper triangular part of the matrix from the
  // quaternion and pick the sign that produces the target matrix.

  // Quaternion-to-matrix is:
  //   [1 0 0]     [-qy^2-qz^2    qx*qy       qx*qz     ]        [ 0  -qz  qy]
  //   [0 1 0] + 2*[qx*qy        -qx^2-qz^2   qy*qz     ] + 2*qw*[ qz  0  -qx]
  //   [0 0 1]     [qx*qz         qy*qz       -qx^2-qy^2]        [-qy  qx  0 ]
  //   identity    symmetric matrix                            skew-symmetric
  //   matrix
  // We can ignore the diagonal entries since they don't depend on w and focus
  // on the upper off-diagonal entries.
  //     2*qx*qy - 2*w*qz = m12
  //     2*qx*qz + 2*w*qy = m13
  //     2*qy*qz - 2*w*qx = m23
  at::Tensor symmetric_part_diff = at::stack(
      {
          2 * qx * qy - matrices.index({Ellipsis, 0, 1}).unsqueeze(-1),
          2 * qx * qz - matrices.index({Ellipsis, 0, 2}).unsqueeze(-1),
          2 * qy * qz - matrices.index({Ellipsis, 1, 2}).unsqueeze(-1),
      },
      -1);

  at::Tensor skew_symmetric_part = at::stack({-2 * qz, 2 * qy, -2 * qx}, -1);
  // The difference if w is positive:
  at::Tensor diff_w_positive =
      symmetric_part_diff + qw.unsqueeze(-1).expand_as(skew_symmetric_part) * skew_symmetric_part;
  // The difference if w is negative:
  at::Tensor diff_w_negative =
      symmetric_part_diff - qw.unsqueeze(-1).expand_as(skew_symmetric_part) * skew_symmetric_part;

  // Select the one that matches the target matrix (we are flipping v here
  // instead of w because -q = q):
  qv = at::where(at::norm(diff_w_positive, 2, -1) < at::norm(diff_w_negative, 2, -1), qv, -qv);

  return at::cat({qv, qw}, -1);
}

at::Tensor checkAndNormalizeWeights(at::Tensor quaternions, std::optional<at::Tensor> weights_in) {
  at::Tensor weights;
  if (weights_in) {
    weights = *weights_in;
  } else {
    weights = at::ones_like(quaternions.select(-1, 0));
  }

  if (weights.dim() == quaternions.dim()) {
    weights = weights.squeeze(-1);
  }

  MT_THROW_IF(
      weights.dim() + 1 != quaternions.dim(),
      "Expected weights vector to match quaternion vector in all dimensions except the last; got weights={} and quaternions={}",
      formatTensorSizes(weights),
      formatTensorSizes(quaternions));

  for (int64_t i = 0; i < weights.dim(); ++i) {
    MT_THROW_IF(
        weights.size(i) != quaternions.size(i),
        "Expected weights vector to match quaternion vector in all dimensions except the last; got weights={} and quaternions={}",
        formatTensorSizes(weights),
        formatTensorSizes(quaternions));
  }

  // Normalize the weights
  weights = weights.clamp(0);
  at::Tensor weight_sum = weights.sum(-1);
  return weights / weight_sum.unsqueeze(-1).expand_as(weights);
}

at::Tensor blendQuaternions(at::Tensor quaternions, std::optional<at::Tensor> weights_in) {
  // If no weights, then assume evenly weighted:
  at::Tensor weights = checkAndNormalizeWeights(quaternions, weights_in);

  // Find average rotation by means described in
  // https://stackoverflow.com/questions/12374087/average-of-multiple-quaternions
  // http://www.acsu.buffalo.edu/~johnc/ave_quat07.pdf
  //
  // i.e. Stack quaternion coeffs in Q, compute M = Q^T x Q, and yield the
  // eigenvector corresponding to the largest eigenvalue as the average rotation
  checkQuaternion(quaternions);
  at::Tensor outer_prod = at::einsum("...i,...k->...ik", {quaternions, quaternions});
  at::Tensor QtQ = (weights.unsqueeze(-1).unsqueeze(-1) * outer_prod).sum(-3);
  const auto [eigenvalues, eigenvectors] = at::linalg_eigh(QtQ);
  at::Tensor result = eigenvectors.select(-1, 3);
  return result;
}

} // namespace pymomentum
