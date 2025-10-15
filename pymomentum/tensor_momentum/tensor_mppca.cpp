/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/tensor_momentum/tensor_mppca.h"

#include "pymomentum/tensor_utility/tensor_utility.h"

#include <momentum/common/exception.h>
#include <momentum/math/constants.h>
#include <momentum/math/mppca.h>

#include <Eigen/Core>

namespace pymomentum {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> mppcaToTensors(
    const momentum::Mppca& mppca,
    std::optional<const momentum::ParameterTransform*> paramTransform) {
  at::Tensor mu = to2DTensor(mppca.mu);
  at::Tensor W;
  const auto nMixtures = mppca.p;
  Eigen::VectorXf pi(nMixtures);
  Eigen::VectorXf sigma(nMixtures);
  MT_THROW_IF(mppca.Cinv.size() != nMixtures, "Invalid Mppca");
  for (Eigen::Index iMix = 0; iMix < nMixtures; ++iMix) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> Cinv_eigs(mppca.Cinv[iMix]);

    // Eigenvalues of the inverse are the inverse of the eigenvalues:
    Eigen::VectorXf C_eigenvalues = Cinv_eigs.eigenvalues().cwiseInverse();

    // Assume that it's not full rank and hence the last eigenvalue is sigma^2.
    const float sigma2 = C_eigenvalues(C_eigenvalues.size() - 1);

    assert(sigma2 >= 0);
    sigma[iMix] = std::sqrt(sigma2);

    // (sigma^2*I + W^T*W) has eigenvalues (sigma^2 + lambda)
    // where the lambda are the eigenvalues for W^T*W (which we want):
    C_eigenvalues.array() -= sigma2;

    // Find the rank of W:
    int W_rank = C_eigenvalues.size();
    for (Eigen::Index i = 0; i < C_eigenvalues.size(); ++i) {
      if (C_eigenvalues(i) < 0.0001) {
        W_rank = i;
        break;
      }
    }

    if (iMix == 0) {
      W = at::zeros({(int)nMixtures, (int)W_rank, (int)mppca.d});
    }

    for (Eigen::Index jComponent = 0; jComponent < W_rank && jComponent < W.size(1); ++jComponent) {
      toEigenMap<float>(W.select(0, iMix).select(0, jComponent)) =
          std::sqrt(C_eigenvalues(jComponent)) * Cinv_eigs.eigenvectors().col(jComponent);
    }

    const float C_logDeterminant = -Cinv_eigs.eigenvalues().array().log().sum();

    // We have:
    //   Rpre(c) = std::log(pi(c))
    //       - 0.5 * C_logDeterminant
    //       - 0.5 * static_cast<double>(d) * std::log(2.0 * PI));
    // so std::log(pi(c)) = Rpre(c) + 0.5 * C_logDeterminant + 0.5 *
    //      d * std::log(2.0 * PI));
    const float log_pi = mppca.Rpre(iMix) + 0.5f * C_logDeterminant +
        0.5f * static_cast<float>(mppca.d) * std::log(2.0 * momentum::pi<float>());
    pi[iMix] = exp(log_pi);
  }

  Eigen::VectorXi parameterIndices = Eigen::VectorXi::Constant(mppca.d, -1);
  if (paramTransform.has_value()) {
    for (Eigen::Index i = 0; i < mppca.names.size() && i < mppca.d; ++i) {
      auto paramIdx = (*paramTransform)->getParameterIdByName(mppca.names[i]);
      if (paramIdx != momentum::kInvalidIndex) {
        parameterIndices[i] = (int)paramIdx;
      }
    }
  }

  return {
      to1DTensor<float>(pi), mu, W, to1DTensor<float>(sigma), to1DTensor<int>(parameterIndices)};
}

} // namespace pymomentum
