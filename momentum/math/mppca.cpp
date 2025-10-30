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
    std::span<const MatrixX<T>> W,
    const VectorX<T>& sigma2) {
  using CovarianceMatrix = LowRankCovarianceMatrixT<T>;

  d = mmu.cols();
  p = sigma2.rows();

  // set the data
  Rpre.setZero(p);
  Cinv.resize(p);
  L.resize(p);

  CovarianceMatrix covariance;

  const T half_d_log_2pi = 0.5 * static_cast<T>(d) * std::log(twopi<T>());

  for (size_t c = 0; c < p; c++) {
    covariance.reset(std::sqrt(sigma2(c)), W[c].transpose());

    // Back-substitution of the identity matrix gives the inverse:
    const Eigen::MatrixX<T> L_tmp = covariance.R()
                                        .template triangularView<Eigen::Upper>()
                                        .solve(Eigen::MatrixX<T>::Identity(d, d))
                                        .transpose();
    Cinv[c] = covariance.R().template triangularView<Eigen::Upper>().solve(L_tmp);
    L[c] = L_tmp;

    const T C_logDeterminant = covariance.logDeterminant();

    // Rpre is the constant part of R that does not depend on the parameters, so we can precalculate
    // it
    Rpre(c) = std::log(inPi(c)) - 0.5 * C_logDeterminant - half_d_log_2pi;
  }
  mu = mmu;

  names = std::vector<std::string>(d);
}

template <typename T, typename T2>
std::vector<Eigen::MatrixX<T2>> castMatrices(std::span<const Eigen::MatrixX<T>> m_in) {
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
