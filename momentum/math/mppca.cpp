/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/math/mppca.h"

#include "momentum/math/constants.h"
#include "momentum/math/covariance_matrix.h"

namespace momentum {

template <typename T>
void MppcaT<T>::set(
    const VectorX<T>& inPi,
    const MatrixX<T>& mmu,
    momentum::span<const MatrixX<T>> W,
    const VectorX<T>& sigma2) {
  using CovarianceMatrix = LowRankCovarianceMatrixT<T>;

  d = mmu.cols();
  p = sigma2.rows();

  // TODO: Validate input dimensions: inPi.rows() == p, mmu.rows() == p, W.size() == p,
  //       and each W[c] has d columns. Currently mismatches will trigger out-of-bounds
  //       access or silent corruption rather than a clear error.
  Rpre.setZero(p);
  Cinv.resize(p);
  L.resize(p);

  CovarianceMatrix covariance;

  // For each mixture component c, PPCA models the data covariance as
  //     C_c = sigma_c^2 * I + W_c * W_c^T,
  // i.e. low-rank signal (W_c) plus isotropic noise (sigma_c^2). The log of the
  // Gaussian normalizer used in the per-component log-density is
  //     -0.5 * d * log(2*pi)   (the half_d_log_2pi term below).
  const T half_d_log_2pi = T(0.5) * static_cast<T>(d) * std::log(twopi<T>());

  for (size_t c = 0; c < p; c++) {
    // LowRankCovarianceMatrix takes A such that C = sigma^2*I + A^T*A; here A = W_c^T,
    // so A^T*A = W_c * W_c^T as required by the PPCA model. It produces the QR
    // factorization R with R^T * R = sigma^2*I + W_c * W_c^T = C_c.
    covariance.reset(std::sqrt(sigma2(c)), W[c].transpose());

    // R is upper-triangular, so back-substituting the identity yields R^{-1}.
    // L_tmp = R^{-T} (the lower-triangular factor of C^{-1} = R^{-1} * R^{-T}).
    const Eigen::MatrixX<T> L_tmp = covariance.R()
                                        .template triangularView<Eigen::Upper>()
                                        .solve(Eigen::MatrixX<T>::Identity(d, d))
                                        .transpose();
    // Cinv = R^{-1} * R^{-T} = (R^T * R)^{-1} = C^{-1}.
    Cinv[c] = covariance.R().template triangularView<Eigen::Upper>().solve(L_tmp);
    L[c] = L_tmp;

    const T C_logDeterminant = covariance.logDeterminant();

    // Per-component log responsibility for input x is
    //     log r_c(x) = log pi_c - 0.5*log|C_c| - 0.5*d*log(2*pi) - 0.5*(x-mu_c)^T C_c^{-1}
    //     (x-mu_c).
    // Rpre stores the data-independent prefactor (everything but the quadratic term),
    // so callers can add the cached -0.5 * Mahalanobis distance at evaluation time
    // and apply log-sum-exp across components for a numerically stable mixture log-likelihood.
    Rpre(c) = std::log(inPi(c)) - T(0.5) * C_logDeterminant - half_d_log_2pi;
  }
  mu = mmu;

  // Reset names to the data-space dimension; caller is expected to populate them
  // afterwards if needed. Note: this discards any names previously set on the model.
  names = std::vector<std::string>(d);
}

template <typename T, typename T2>
std::vector<Eigen::MatrixX<T2>> castMatrices(momentum::span<const Eigen::MatrixX<T>> m_in) {
  std::vector<Eigen::MatrixX<T2>> result;
  result.reserve(m_in.size());
  for (const auto& m : m_in) {
    result.emplace_back(m.template cast<T2>());
  }
  return result;
}

template <typename T>
template <typename T2>
MppcaT<T2> MppcaT<T>::cast() const {
  MppcaT<T2> result;
  result.d = this->d;
  result.p = this->p;
  result.names = this->names;

  result.mu = this->mu.template cast<T2>();
  result.Cinv = castMatrices<T, T2>(this->Cinv);
  result.L = castMatrices<T, T2>(this->L);
  result.Rpre = this->Rpre.template cast<T2>();

  return result;
}

template <typename T>
bool MppcaT<T>::isApprox(const MppcaT<T>& mppcaT) const {
  auto vecIsApprox = [](auto& vL, auto& vR) {
    if (vL.size() != vR.size()) {
      return false;
    }
    for (size_t i = 0; i < vL.size(); i++) {
      if (!vL[i].isApprox(vR[i])) {
        return false;
      }
    }
    return true;
  };
  return (
      (d == mppcaT.d) && (p == mppcaT.p) && (names == mppcaT.names) && mu.isApprox(mppcaT.mu) &&
      vecIsApprox(Cinv, mppcaT.Cinv) && vecIsApprox(L, mppcaT.L) && Rpre.isApprox(mppcaT.Rpre));
}

template struct MppcaT<float>;
template struct MppcaT<double>;

template MppcaT<float> MppcaT<float>::cast() const;
template MppcaT<float> MppcaT<double>::cast() const;
template MppcaT<double> MppcaT<float>::cast() const;
template MppcaT<double> MppcaT<double>::cast() const;

} // namespace momentum
