/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/math/covariance_matrix.h"

#include "momentum/common/checks.h"

namespace momentum {

template <typename T>
void LowRankCovarianceMatrixT<T>::reset(T sigma, Eigen::Ref<const Eigen::MatrixX<T>> A) {
  A_ = A;
  sigma_ = sigma;

  const Eigen::Index n = A_.cols();
  const Eigen::Index m = A_.rows();

  // Initialize the online QR with a sigma*I diagonal, then stream in the rows of A.
  // The resulting R is the upper-triangular factor of the augmented matrix
  //   A_aug = [ sigma*I ; A ]    so that   A_aug^T * A_aug = sigma^2*I + A^T*A = C.
  // Equivalently, R is (up to sign of diagonals) the Cholesky factor of C.
  // The dummy zero rhs is required by the online QR API but unused here.
  qrFactorization_.reset(n, sigma);
  Eigen::VectorX<T> b = Eigen::VectorX<T>::Zero(m);
  qrFactorization_.add(A_, b);
}

template <typename T>
Eigen::Index LowRankCovarianceMatrixT<T>::dimension() const {
  return A_.cols();
}

template <typename T>
const Eigen::MatrixX<T>& LowRankCovarianceMatrixT<T>::basis() const {
  return A_;
}

template <typename T>
T LowRankCovarianceMatrixT<T>::sigma() const {
  return sigma_;
}

template <typename T>
Eigen::VectorX<T> LowRankCovarianceMatrixT<T>::inverse_times_vec(
    Eigen::Ref<const Eigen::VectorX<T>> rhs) const {
  const auto& R = qrFactorization_.R();
  MT_CHECK(R.rows() == rhs.size());

  // C^{-1} * rhs = (R^T * R)^{-1} * rhs = R^{-1} * R^{-T} * rhs.
  // Implemented as two triangular back-solves: first R^T y = rhs, then R x = y.
  return R.template triangularView<Eigen::Upper>().solve(
      R.template triangularView<Eigen::Upper>().transpose().solve(rhs));
}

template <typename T>
Eigen::VectorX<T> LowRankCovarianceMatrixT<T>::times_vec(
    Eigen::Ref<const Eigen::VectorX<T>> rhs) const {
  // C * rhs = (sigma^2 * I + A^T * A) * rhs computed directly from the stored
  // basis A_; avoids forming the dense [n x n] matrix and exploits the low-rank
  // structure (cost O(m*n) instead of O(n^2)).
  Eigen::VectorX<T> result = (sigma_ * sigma_) * rhs;
  result.noalias() += A_.transpose() * (A_ * rhs);
  return result;
}

template <typename T>
Eigen::MatrixX<T> LowRankCovarianceMatrixT<T>::inverse_times_mat(
    Eigen::Ref<const Eigen::MatrixX<T>> rhs) const {
  const auto& R = qrFactorization_.R();
  MT_CHECK(rhs.rows() == dimension());

  // Same C^{-1} = R^{-1} * R^{-T} solve as inverse_times_vec, applied column-wise.
  return R.template triangularView<Eigen::Upper>().solve(
      R.template triangularView<Eigen::Upper>().transpose().solve(rhs));
}

template <typename T>
Eigen::MatrixX<T> LowRankCovarianceMatrixT<T>::times_mat(
    Eigen::Ref<const Eigen::MatrixX<T>> rhs) const {
  Eigen::MatrixX<T> result = (sigma_ * sigma_) * rhs;
  result.noalias() += A_.transpose() * (A_ * rhs);
  return result;
}

template <typename T>
const Eigen::MatrixX<T>& LowRankCovarianceMatrixT<T>::R() const {
  return qrFactorization_.R();
}

template <typename T>
T LowRankCovarianceMatrixT<T>::logDeterminant() const {
  const auto& R = qrFactorization_.R();

  // log|R| = sum of log of diagonal entries (R is upper triangular).
  // The MT_CHECK enforces strictly positive diagonals, which the
  // OnlineHouseholderQR is expected to produce because the augmented matrix
  // [sigma*I; A] has full column rank for any sigma > 0 (so R is non-singular)
  // and the implementation chooses positive diagonal signs.
  T result = 0;
  for (Eigen::Index i = 0; i < dimension(); ++i) {
    MT_CHECK(R(i, i) > 0);
    result += std::log(R(i, i));
  }

  // C = R^T * R, so log|C| = log|R^T| + log|R| = 2 * log|R|.
  return T(2) * result;
}

template <typename T>
T LowRankCovarianceMatrixT<T>::inverse_logDeterminant() const {
  return -logDeterminant();
}

template class LowRankCovarianceMatrixT<float>;
template class LowRankCovarianceMatrixT<double>;

} // namespace momentum
