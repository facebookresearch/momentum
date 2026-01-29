/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/geometry/array_mppca.h"

#include <momentum/common/exception.h>
#include <momentum/math/constants.h>
#include <momentum/math/mppca.h>

#include <Eigen/Core>

namespace py = pybind11;

namespace pymomentum {

std::tuple<
    py::array_t<float>,
    py::array_t<float>,
    py::array_t<float>,
    py::array_t<float>,
    py::array_t<int>>
mppcaToArrays(
    const momentum::Mppca& mppca,
    std::optional<const momentum::ParameterTransform*> paramTransform) {
  const auto nMixtures = mppca.p;
  const auto dimension = mppca.d;

  // Create output arrays
  py::array_t<float> pi_array(static_cast<py::ssize_t>(nMixtures));
  py::array_t<float> mu_array(
      {static_cast<py::ssize_t>(nMixtures), static_cast<py::ssize_t>(dimension)});

  auto pi = pi_array.mutable_unchecked<1>();
  auto mu = mu_array.mutable_unchecked<2>();

  // Copy mu values
  for (py::ssize_t iMix = 0; iMix < nMixtures; ++iMix) {
    for (py::ssize_t d = 0; d < dimension; ++d) {
      mu(iMix, d) = mppca.mu(iMix, d);
    }
  }

  // Process each mixture component
  MT_THROW_IF(mppca.Cinv.size() != nMixtures, "Invalid Mppca");

  Eigen::VectorXf sigma_vec(nMixtures);
  int W_rank = 0; // Will be determined from first mixture

  // First pass: determine W rank from first mixture
  {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> Cinv_eigs(mppca.Cinv[0]);
    Eigen::VectorXf C_eigenvalues = Cinv_eigs.eigenvalues().cwiseInverse();
    const float sigma2 = C_eigenvalues(C_eigenvalues.size() - 1);
    C_eigenvalues.array() -= sigma2;

    W_rank = C_eigenvalues.size();
    for (Eigen::Index i = 0; i < C_eigenvalues.size(); ++i) {
      if (C_eigenvalues(i) < 0.0001) {
        W_rank = i;
        break;
      }
    }
  }

  // Create W array with determined rank
  py::array_t<float> W_array(
      {static_cast<py::ssize_t>(nMixtures),
       static_cast<py::ssize_t>(W_rank),
       static_cast<py::ssize_t>(dimension)});
  auto W = W_array.mutable_unchecked<3>();

  // Second pass: fill in all values
  for (Eigen::Index iMix = 0; iMix < nMixtures; ++iMix) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> Cinv_eigs(mppca.Cinv[iMix]);

    // Eigenvalues of the inverse are the inverse of the eigenvalues:
    Eigen::VectorXf C_eigenvalues = Cinv_eigs.eigenvalues().cwiseInverse();

    // Assume that it's not full rank and hence the last eigenvalue is sigma^2.
    const float sigma2 = C_eigenvalues(C_eigenvalues.size() - 1);
    assert(sigma2 >= 0);
    sigma_vec[iMix] = std::sqrt(sigma2);

    // (sigma^2*I + W^T*W) has eigenvalues (sigma^2 + lambda)
    // where the lambda are the eigenvalues for W^T*W (which we want):
    C_eigenvalues.array() -= sigma2;

    // Fill W for this mixture
    for (Eigen::Index jComponent = 0; jComponent < W_rank; ++jComponent) {
      const float scale = std::sqrt(C_eigenvalues(jComponent));
      for (Eigen::Index d = 0; d < dimension; ++d) {
        W(iMix, jComponent, d) = scale * Cinv_eigs.eigenvectors()(d, jComponent);
      }
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
    pi(iMix) = std::exp(log_pi);
  }

  // Create sigma array
  py::array_t<float> sigma_array(static_cast<py::ssize_t>(nMixtures));
  auto sigma = sigma_array.mutable_unchecked<1>();
  for (py::ssize_t iMix = 0; iMix < nMixtures; ++iMix) {
    sigma(iMix) = sigma_vec[iMix];
  }

  // Create parameter indices array
  py::array_t<int> param_indices_array(static_cast<py::ssize_t>(dimension));
  auto param_indices = param_indices_array.mutable_unchecked<1>();

  for (py::ssize_t i = 0; i < dimension; ++i) {
    param_indices(i) = -1; // Default to -1
  }

  if (paramTransform.has_value()) {
    for (Eigen::Index i = 0; i < mppca.names.size() && i < dimension; ++i) {
      auto paramIdx = (*paramTransform)->getParameterIdByName(mppca.names[i]);
      if (paramIdx != momentum::kInvalidIndex) {
        param_indices(i) = static_cast<int>(paramIdx);
      }
    }
  }

  return {pi_array, mu_array, W_array, sigma_array, param_indices_array};
}

} // namespace pymomentum
