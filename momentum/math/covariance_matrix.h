/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/math/online_householder_qr.h>
#include <momentum/math/types.h>

namespace momentum {

/// Efficient and numerically stable representation of a low-rank-plus-diagonal
/// Gaussian covariance matrix
///   C = sigma^2 * I + A^T * A
/// where C and I are [n x n] and the basis A is [m x n] (m may be larger or
/// smaller than n). Access to C^{-1} is computed via the QR factorization of
/// the augmented matrix [sigma*I; A], which keeps the construction
/// numerically stable and avoids ever forming A^T * A explicitly.
///
/// @pre sigma > 0 (required for C to be strictly positive-definite and for
///      the QR-based inverse to exist; the implementation does not check).
/// @note C is symmetric positive-definite by construction.
template <typename T>
class LowRankCovarianceMatrixT {
 public:
  LowRankCovarianceMatrixT() = default;

  /// (Re)build the covariance from a regularizer sigma and basis A.
  /// Computes the QR factorization of [sigma*I; A] from scratch (batch
  /// update; not an incremental/online refinement of any previous state).
  /// @param sigma Diagonal regularizer; must be > 0.
  /// @param A Low-rank basis with n columns and m rows.
  void reset(T sigma, Eigen::Ref<const Eigen::MatrixX<T>> A);

  /// Returns n, the dimension of the covariance (number of columns of A).
  [[nodiscard]] Eigen::Index dimension() const;

  /// Returns the stored basis A passed to reset().
  [[nodiscard]] const Eigen::MatrixX<T>& basis() const;

  /// Returns the diagonal regularizer sigma passed to reset().
  [[nodiscard]] T sigma() const;

  /// Returns C^{-1} * rhs, computed via two triangular solves on R.
  [[nodiscard]] Eigen::VectorX<T> inverse_times_vec(Eigen::Ref<const Eigen::VectorX<T>> rhs) const;
  /// Returns C * rhs = sigma^2 * rhs + A^T * (A * rhs).
  [[nodiscard]] Eigen::VectorX<T> times_vec(Eigen::Ref<const Eigen::VectorX<T>> rhs) const;

  /// Column-wise application of C^{-1} to a matrix rhs.
  [[nodiscard]] Eigen::MatrixX<T> inverse_times_mat(Eigen::Ref<const Eigen::MatrixX<T>> rhs) const;
  /// Column-wise application of C to a matrix rhs.
  [[nodiscard]] Eigen::MatrixX<T> times_mat(Eigen::Ref<const Eigen::MatrixX<T>> rhs) const;

  /// Returns the upper-triangular R factor of the QR factorization of
  /// [sigma*I; A], so that C = R^T * R.
  [[nodiscard]] const Eigen::MatrixX<T>& R() const;

  /// Returns log(det(C)), computed stably as 2 * sum(log|diag(R)|).
  [[nodiscard]] T logDeterminant() const;

  /// Returns log(det(C^{-1})) = -log(det(C)).
  [[nodiscard]] T inverse_logDeterminant() const;

 private:
  /// QR factorization of the augmented matrix [sigma*I; A] = Q*R, where Q is
  /// orthonormal and R is upper triangular. OnlineHouseholderQR is used
  /// because it can initialize R from a diagonal (sigma*I) and then only
  /// process the m rows of A, giving an O(m*n^2) construction that is
  /// cheaper than a fully general O(n^3) QR/Cholesky when m << n.
  ///
  /// Given [sigma*I; A] = Q*R we have C = sigma^2*I + A^T*A = R^T*R, so
  /// C^{-1} = R^{-1} * R^{-T} via two triangular solves (Q is never needed
  /// and OnlineHouseholderQR does not store it; see online_householder_qr.h
  /// for why -- it is intended for least-squares solves that apply Q to b
  /// on the fly).
  OnlineHouseholderQR<T> qrFactorization_{0};

  /// Low-rank basis A passed to reset(); shape [m x n].
  Eigen::MatrixX<T> A_;

  /// Diagonal regularizer; assumed > 0.
  T sigma_ = 0;
};

} // namespace momentum
