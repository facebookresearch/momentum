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

//
// Efficient and numerically stable representation of the generalized
// Gaussian covariance
//   C = (sigma^2 * I + A^T * A)
// providing access to the inverse covariance matrix (computed using
// QR factorization).
//
// Note the dimensionality here: C and I are [n x n], A is [m x n]
// where m can be either larger or smaller than n.
template <typename T>
class LowRankCovarianceMatrixT {
 public:
  LowRankCovarianceMatrixT() = default;

  void reset(T sigma, Eigen::Ref<const Eigen::MatrixX<T>> A);

  [[nodiscard]] Eigen::Index dimension() const;

  [[nodiscard]] const Eigen::MatrixX<T>& basis() const;

  [[nodiscard]] T sigma() const;

  [[nodiscard]] Eigen::VectorX<T> inverse_times_vec(Eigen::Ref<const Eigen::VectorX<T>> rhs) const;
  [[nodiscard]] Eigen::VectorX<T> times_vec(Eigen::Ref<const Eigen::VectorX<T>> rhs) const;

  [[nodiscard]] Eigen::MatrixX<T> inverse_times_mat(Eigen::Ref<const Eigen::MatrixX<T>> rhs) const;
  [[nodiscard]] Eigen::MatrixX<T> times_mat(Eigen::Ref<const Eigen::MatrixX<T>> rhs) const;

  [[nodiscard]] const Eigen::MatrixX<T>& R() const;

  // The log of the determinant of the covariance matrix
  [[nodiscard]] T logDeterminant() const;

  // The log of the determinant of the inverse covariance matrix
  [[nodiscard]] T inverse_logDeterminant() const;

 private:
  // The QR factorization of A is defined by:
  //   A = Q*R
  // where Q is orthonormal and R is upper triangular.  This online QR solver
  // used here happens to be particularly efficient for the case where the A
  // matrix looks like:
  //    A = [ lambda*I ]
  //        [    B     ]
  // The reason for this is that it allows initialization of the R matrix by
  // a diagonal, and hence only needs to process the rows in W.  The
  // resulting factorization is O(m*n^2) to compute, which may (for m << n)
  // be a touch cheaper than the fully general O(n^3) that a "standard" QR
  // factorization or Cholesky would require.
  //
  // Once we have A = Q*R, we can compute (A^T*A)^{-1} by
  //     (A^T * A)^{-1} = (R^T * Q^T * Q * R)^{-1}
  //                    = (R^T * R)^{-1}            by orthonormality
  //                    = (R^{-1}) * R^{-T}         distributing the inverse
  //
  // Note that OnlineHouseholderQR does _not_ store the Q matrix (or the
  // information required to reconstruct it, as is more typical) because it
  // assumes that it will be used to solve least squares problems of the form
  // min |Ax - b|_2 and hence can apply Q to b as it is being computed.  In
  // our case, however, as noted above we won't be using the Q matrix so this
  // is not a limitation.
  OnlineHouseholderQR<T> qrFactorization_{0};

  // Low-rank basis:
  Eigen::MatrixX<T> A_;

  T sigma_ = 0;
};

} // namespace momentum
