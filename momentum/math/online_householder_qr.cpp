/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/math/online_householder_qr.h"

#include "momentum/common/profile.h"

namespace momentum {

namespace {

template <typename T>
T sqr(T val) {
  return val * val;
}

// Apply the Householder transformation H*y, where
//   H = (I - beta*v*v^T)
// We use v_2m and y_2m to indicate that these vectors contain the
// 2...m entries of the vector.
//
// Thus, the y vector is divided up:
//   y = [ y_1 y_2m... ]^T
// and v is implicitly assumed to have 1 stored in its first entry,
//   v = [ 1 v_2m ]^T
template <typename T>
void applyHouseholderTransformation(
    T beta,
    const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& v_2m,
    T& y_1,
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> y_2m) {
  MT_CHECK(v_2m.size() == y_2m.size());

  // Compute v^T * y.  Note that v_1 is assumed to be 1 here.
  const T dotProd = y_1 + v_2m.dot(y_2m);
  const T scalar = dotProd * beta;

  // Subtract off (beta * v * v^T * A)
  y_1 -= scalar;
  y_2m.noalias() -= scalar * v_2m;
}

// Computes the Householder reflection vector for the column [x1 x2...xm]^T
// (x1 is passed in separately because it comes from the R matrix rather than A).
// Uses the stable algorithm from Golub and van Loan.
// The result of applying the Householder transform (I - beta*v*v^T) is a vector of the form
//     [mu 0 0 0 ... 0]^T
// Returns the pair [beta, mu].
template <typename T>
std::pair<T, T> computeHouseholderVec(
    T x1,
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>>
        vec) // NOLINT(facebook-hte-ConstantArgumentPassByValue)
{
  const T sigma = vec.squaredNorm();
  if (sigma == 0) {
    // No need to apply a transform if the vector is already zeroed out.
    return {0, x1};
  }

  const T mu = std::sqrt(sqr(x1) + sigma);
  const T v1 = (x1 <= 0) ? (x1 - mu) : (-sigma / (x1 + mu));
  const T beta = T(2) * sqr(v1) / (sigma + sqr(v1));
  vec /= v1;

  return std::make_pair(beta, mu);
}

} // namespace

template <typename T>
OnlineHouseholderQR<T>::OnlineHouseholderQR(const Eigen::Index n, T lambda) {
  reset(n, lambda);
}

template <typename T>
void OnlineHouseholderQR<T>::reset() {
  reset(R_.cols());
}

template <typename T>
void OnlineHouseholderQR<T>::reset(const Eigen::Index n, T lambda) {
  R_.resize(n, n);
  R_.setZero();
  if (lambda != 0) {
    R_.diagonal().setConstant(lambda);
  }

  y_.resize(n);
  y_.setZero();
}

void validateColumnIndices(gsl::span<const Eigen::Index> colIndices, Eigen::Index maxEntry) {
#ifndef NDEBUG
  {
    for (const auto& idx : colIndices) {
      MT_CHECK(idx < maxEntry);
    }

    // check for duplicates:
    std::vector<Eigen::Index> colsCheckDups(std::begin(colIndices), std::end(colIndices));
    std::sort(colsCheckDups.begin(), colsCheckDups.end());
    MT_CHECK(std::unique(colsCheckDups.begin(), colsCheckDups.end()) == colsCheckDups.end());
  }
#else
  (void)colIndices;
  (void)maxEntry;
#endif
}

template <typename T>
void OnlineHouseholderQR<T>::add(MatrixType A, VectorType b) {
  addMutating(A, b);
}

template <typename T>
void OnlineHouseholderQR<T>::addMutating(
    ColumnIndexedMatrix<MatrixType> A,
    Eigen::Ref<VectorType> b) {
  MT_PROFILE_FUNCTION();

  const Eigen::Index n = R_.rows();
  MT_CHECK(A.rows() == b.rows());
  MT_CHECK(A.cols() == n);
  for (Eigen::Index iCol = 0; iCol < n; ++iCol) {
    // At this point, we have zeroed out all the columns to the
    // left of iCol.  The current matrix looks like this.
    //          iCol
    //          \|/
    //     [ r r r r ]
    //     [ 0 r r r ]
    //     [ 0 0 r r ]
    //     [ 0 0 0 r ]
    //     [ 0 0 0 0 ]
    //     [ ...     ]
    //     [ 0 0 a a ]
    //     [ 0 0 a a ]
    // We will apply a Householder reflection (I - beta*v*v^T) * A
    // This will zero out the ith column.
    const auto [beta, mu] = computeHouseholderVec<T>(R_(iCol, iCol), A.col(iCol));
    if (beta == 0) {
      continue;
    }

    // computeHouseholderVec guarantees that applying the transform to iCol
    // results in a vector [mu 0 ... 0].
    R_(iCol, iCol) = mu;

    // Apply the Householder vector to the remaining columns.
    const auto& v2m = A.col(iCol);
    for (Eigen::Index jCol = iCol + 1; jCol < n; ++jCol) {
      applyHouseholderTransformation<T>(beta, v2m, R_(iCol, jCol), A.col(jCol));
    }

    // Apply the Householder vector to the rhs.
    applyHouseholderTransformation<T>(beta, v2m, y_(iCol), b);
  }
}

template <typename T>
typename OnlineHouseholderQR<T>::VectorType OnlineHouseholderQR<T>::At_times_b() const {
  // We defined R and y as
  //   R = Q*A     y = Q*b
  // Therefore
  //   R^T*y = (Q*A)^T*(Q*b)
  //         = A^T * Q^T * Q * b =
  //         = A^T*b
  return R_.transpose() * y_;
}

template <typename T>
typename OnlineHouseholderQR<T>::VectorType OnlineHouseholderQR<T>::result() const {
  // TODO consider checking that the back-substitution is okay here, and
  // "fixing it up" if we encounter zeroes on the diagonal.
  return R_.template triangularView<Eigen::Upper>().solve(y_);
}

template class OnlineHouseholderQR<float>;
template class OnlineHouseholderQR<double>;

template <typename T>
OnlineBlockHouseholderQR<T>::OnlineBlockHouseholderQR(const Eigen::Index n_common, T lambda)
    : lambda_(lambda),
      R_nn_(MatrixType::Zero(n_common, n_common)),
      y_n_(VectorType::Zero(n_common)) {
  R_nn_.diagonal().setConstant(lambda);
}

template <typename T>
void OnlineBlockHouseholderQR<T>::addMutating(
    size_t iBlock,
    ColumnIndexedMatrix<MatrixType> A_diag,
    ColumnIndexedMatrix<MatrixType> A_common,
    Eigen::Ref<VectorType> b) {
  const Eigen::Index n_diag = A_diag.cols();
  const Eigen::Index n_common = R_nn_.cols();
  MT_CHECK(A_diag.rows() == b.rows());
  MT_CHECK(A_common.rows() == b.rows());
  MT_CHECK(A_common.cols() == n_common);

  while (R_ii_.size() <= iBlock) {
    R_ii_.push_back(MatrixType());
  }
  while (y_i_.size() <= iBlock) {
    y_i_.push_back(VectorType());
  }
  while (R_in_.size() <= iBlock) {
    R_in_.push_back(MatrixType());
  }

  if (R_ii_[iBlock].rows() == 0) {
    R_ii_[iBlock] = MatrixType::Zero(n_diag, n_diag);
    R_ii_[iBlock].diagonal().setConstant(lambda_);

    y_i_[iBlock] = VectorType::Zero(n_diag);

    R_in_[iBlock] = MatrixType::Zero(n_diag, n_common);
  } else {
    MT_CHECK(R_ii_[iBlock].cols() == A_diag.cols());
    MT_CHECK(y_i_[iBlock].rows() == A_diag.cols());
  }

  auto& R_ii = R_ii_[iBlock];
  auto& y_i = y_i_[iBlock];
  auto& R_in = R_in_[iBlock];
  auto& R_nn = R_nn_;

  for (Eigen::Index iCol = 0; iCol < n_diag; ++iCol) {
    if (A_diag.col(iCol).isZero()) {
      continue;
    }

    // At this point, we have zeroed out all the columns to the
    // left of iCol.  The current matrix looks like this.
    //          iCol
    //          \|/
    //     [ r r r r ]
    //     [ 0 r r r ]
    //     [ 0 0 r r ]
    //     [ 0 0 0 r ]
    //     [ 0 0 0 0 ]
    //     [ ...     ]
    //     [ 0 0 a a ]
    //     [ 0 0 a a ]
    // We will apply a Householder reflection (I - beta*v*v^T) * A
    // This will zero out the ith column.
    const auto [beta, mu] = computeHouseholderVec<T>(R_ii(iCol, iCol), A_diag.col(iCol));
    if (beta == 0) {
      continue;
    }

    // computeHouseholderVec guarantees that applying the transform to iCol
    // results in a vector [mu 0 ... 0].
    R_ii(iCol, iCol) = mu;

    // Apply the Householder vector to the remaining columns.
    const auto& v2m = A_diag.col(iCol);
    for (Eigen::Index jCol_diag = iCol + 1; jCol_diag < n_diag; ++jCol_diag) {
      applyHouseholderTransformation<T>(beta, v2m, R_ii(iCol, jCol_diag), A_diag.col(jCol_diag));
    }

    for (Eigen::Index jCol_common = 0; jCol_common < n_common; ++jCol_common) {
      applyHouseholderTransformation<T>(
          beta, v2m, R_in(iCol, jCol_common), A_common.col(jCol_common));
    }

    // Apply the Householder vector to the rhs.
    applyHouseholderTransformation<T>(beta, v2m, y_i(iCol), b);
  }

  // Now for the common parameters:
  for (Eigen::Index iCol = 0; iCol < n_common; ++iCol) {
    const auto [beta, mu] = computeHouseholderVec<T>(R_nn(iCol, iCol), A_common.col(iCol));
    if (beta == 0) {
      continue;
    }

    R_nn(iCol, iCol) = mu;

    const auto& v2m = A_common.col(iCol);
    for (Eigen::Index jCol_common = iCol + 1; jCol_common < n_common; ++jCol_common) {
      applyHouseholderTransformation<T>(
          beta, v2m, R_nn_(iCol, jCol_common), A_common.col(jCol_common));
    }

    // Apply the Householder vector to the rhs.
    applyHouseholderTransformation<T>(beta, v2m, y_n_(iCol), b);
  }
}

template <typename T>
typename OnlineBlockHouseholderQR<T>::MatrixType OnlineBlockHouseholderQR<T>::R_dense() const {
  const auto nCol_common = R_nn_.cols();

  Eigen::Index nCol_diag = 0;
  for (const auto& R_ii : R_ii_) {
    nCol_diag += R_ii.cols();
  }

  const Eigen::Index nCol_total = nCol_diag + nCol_common;

  MatrixType result = MatrixType::Zero(nCol_total, nCol_total);
  auto offset_cur = 0;
  for (size_t iBlock = 0; iBlock < R_ii_.size(); ++iBlock) {
    const auto& R_ii = R_ii_[iBlock];
    const auto& R_in = R_in_[iBlock];
    MT_CHECK(R_ii.rows() == R_in.rows());
    MT_CHECK(R_in.cols() == nCol_common);
    result.block(offset_cur, offset_cur, R_ii.rows(), R_ii.cols()) = R_ii;
    result.block(offset_cur, nCol_diag, R_in.rows(), nCol_common) = R_in;
    offset_cur += R_ii.rows();
  }
  result.block(offset_cur, offset_cur, nCol_common, nCol_common) = R_nn_;
  return result;
}

template <typename T>
typename OnlineBlockHouseholderQR<T>::MatrixType OnlineBlockHouseholderQR<T>::y_dense() const {
  const auto nRow_common = y_n_.rows();

  Eigen::Index nRow_diag = 0;
  for (const auto& y_i : y_i_) {
    nRow_diag += y_i.rows();
  }

  const Eigen::Index nRow_total = nRow_diag + nRow_common;

  auto offset_cur = 0;
  VectorType result = VectorType::Zero(nRow_total);
  for (size_t iBlock = 0; iBlock < R_ii_.size(); ++iBlock) {
    const auto& y_i = y_i_[iBlock];
    result.segment(offset_cur, y_i.rows()) = y_i;
    offset_cur += y_i.rows();
  }

  result.segment(offset_cur, nRow_common) = y_n_;
  return result;
}

template <typename T>
typename OnlineBlockHouseholderQR<T>::VectorType OnlineBlockHouseholderQR<T>::x_dense() const {
  const auto nBlocks = R_ii_.size();
  Eigen::Index nRows_diag = 0;
  for (const auto& block : R_ii_) {
    nRows_diag += block.cols();
  }
  const Eigen::Index nRows_common = R_nn_.cols();
  const Eigen::Index nRows_total = nRows_diag + nRows_common;

  VectorType result(nRows_total);
  result.segment(nRows_diag, nRows_common) =
      R_nn_.template triangularView<Eigen::Upper>().solve(y_n_);

  const auto& x_n = result.segment(nRows_diag, nRows_common);
  Eigen::Index offset = 0;
  for (size_t i = 0; i < nBlocks; ++i) {
    const auto nRows_cur = R_ii_[i].cols();
    result.segment(offset, nRows_cur) =
        R_ii_[i].template triangularView<Eigen::Upper>().solve(y_i_[i] - R_in_[i] * x_n);
    offset += nRows_cur;
  }

  return result;
}

template <typename T>
typename OnlineBlockHouseholderQR<T>::VectorType OnlineBlockHouseholderQR<T>::x_n() const {
  return R_nn_.template triangularView<Eigen::Upper>().solve(y_n_);
}

template <typename T>
typename OnlineBlockHouseholderQR<T>::VectorType OnlineBlockHouseholderQR<T>::x_i(
    size_t iBlock) const {
  return R_ii_[iBlock].template triangularView<Eigen::Upper>().solve(
      y_i_[iBlock] - R_in_[iBlock] * (R_nn_.template triangularView<Eigen::Upper>().solve(y_n_)));
}

template <typename T>
typename OnlineBlockHouseholderQR<T>::VectorType OnlineBlockHouseholderQR<T>::At_times_b_i(
    size_t iBlock) const {
  // As noted above, A^T*b = R^T*y
  return R_ii_[iBlock].transpose() * y_i_[iBlock];
}

template <typename T>
typename OnlineBlockHouseholderQR<T>::VectorType OnlineBlockHouseholderQR<T>::At_times_b_n() const {
  VectorType result = R_nn_.transpose() * y_n_;
  for (size_t iBlock = 0; iBlock < R_ii_.size(); ++iBlock) {
    result.noalias() += R_in_[iBlock].transpose() * y_i_[iBlock];
  }
  return result;
}

template <typename T>
typename OnlineBlockHouseholderQR<T>::VectorType OnlineBlockHouseholderQR<T>::At_times_b_dense()
    const {
  Eigen::Index nRows_total = y_n_.rows();
  for (const auto& y_i : y_i_) {
    nRows_total += y_i.rows();
  }

  VectorType result(nRows_total);
  Eigen::Index offset = 0;
  for (size_t iBlock = 0; iBlock < R_ii_.size(); ++iBlock) {
    const auto nRows_cur = y_i_[iBlock].rows();
    result.segment(offset, nRows_cur).noalias() = R_ii_[iBlock].transpose() * y_i_[iBlock];
    offset += nRows_cur;
  }

  result.segment(offset, y_n_.rows()).noalias() = R_nn_.transpose() * y_n_;
  for (size_t iBlock = 0; iBlock < R_ii_.size(); ++iBlock) {
    result.segment(offset, y_n_.rows()).noalias() += R_in_[iBlock].transpose() * y_i_[iBlock];
  }

  return result;
}

template <typename T>
T OnlineBlockHouseholderQR<T>::At_times_b_dot(Eigen::Ref<const VectorType> rhs) const {
  return At_times_b_dense().dot(rhs);
}

template <typename T>
void OnlineBlockHouseholderQR<T>::reset() {
  R_ii_.clear();
  R_in_.clear();
  y_i_.clear();

  R_nn_.setZero();
  R_nn_.diagonal().setConstant(lambda_);
  y_n_.setZero();
}

template class OnlineBlockHouseholderQR<float>;
template class OnlineBlockHouseholderQR<double>;

template <typename T>
OnlineBandedHouseholderQR<T>::OnlineBandedHouseholderQR(
    const Eigen::Index n_band,
    Eigen::Index n_common,
    const Eigen::Index bandwidth,
    T lambda)
    : lambda_(lambda),
      R_common_{Eigen::MatrixX<T>::Zero(n_band + n_common, n_common)},
      R_band_{Eigen::MatrixX<T>::Zero(bandwidth, n_band)},
      y_{Eigen::VectorX<T>::Zero(n_band + n_common)} {
  // Bandwidth is only allowed to be zero if there are no parameters in the banded part,
  // in which case it falls back to behaving the same as a regular Householder QR (for
  // the common part).
  MT_CHECK(n_band == 0 || bandwidth > 0);

  initializeDiagonal();
}

template <typename T>
void OnlineBandedHouseholderQR<T>::initializeDiagonal() {
  const auto n_common = R_common_.cols();

  if (n_common != 0) {
    R_common_.bottomRightCorner(n_common, n_common).diagonal().setConstant(lambda_);
  }

  if (R_band_.rows() > 0 && R_band_.cols() > 0) {
    R_band_.template bottomRows<1>().setConstant(lambda_);
  }
}

template <typename T>
void OnlineBandedHouseholderQR<T>::reset() {
  R_common_.setZero();
  R_band_.setZero();
  y_.setZero();

  initializeDiagonal();
}

template <typename T>
void OnlineBandedHouseholderQR<T>::addMutating(
    const Eigen::Index iCol_offset,
    ColumnIndexedMatrix<MatrixType> A_band,
    ColumnIndexedMatrix<MatrixType> A_common,
    Eigen::Ref<VectorType> b) {
  zeroBandedPart(iCol_offset, A_band, A_common, b);
  addMutating(A_common, b);
}

template <typename T>
void OnlineBandedHouseholderQR<T>::zeroBandedPart(
    const Eigen::Index iCol_offset,
    ColumnIndexedMatrix<MatrixType> A_band,
    ColumnIndexedMatrix<MatrixType> A_common,
    Eigen::Ref<VectorType> b) {
  const auto n_common = R_common_.cols();

  MT_CHECK(A_band.cols() <= R_band_.rows());
  MT_CHECK(iCol_offset + A_band.cols() <= R_band_.cols());

  // TODO what to assert about A_band?
  MT_CHECK(A_common.cols() == n_common);

  for (Eigen::Index iCol_local = 0; iCol_local < A_band.cols(); ++iCol_local) {
    const Eigen::Index iCol_global = iCol_local + iCol_offset;

    // At this point, we have zeroed out all the columns to the
    // left of iCol.  The current matrix looks like this.
    //          iCol
    //          \|/
    //     [ r r r r ]
    //     [ 0 r r r ]
    //     [ 0 0 r r ]
    //     [ 0 0 0 r ]
    //     [ 0 0 0 0 ]
    //     [ ...     ]
    //     [ 0 0 a a ]
    //     [ 0 0 a a ]
    // We will apply a Householder reflection (I - beta*v*v^T) * A
    // This will zero out the ith column.
    const auto [beta, mu] =
        computeHouseholderVec<T>(R_band_entry(iCol_global, iCol_global), A_band.col(iCol_local));
    if (beta == 0) {
      continue;
    }

    // computeHouseholderVec guarantees that applying the transform to iCol
    // results in a vector [mu 0 ... 0].
    R_band_entry(iCol_global, iCol_global) = mu;

    // Apply the Householder vector to the remaining columns.
    const auto& v2m = A_band.col(iCol_local);
    for (Eigen::Index jCol_local = iCol_local + 1; jCol_local < A_band.cols(); ++jCol_local) {
      const auto jCol_global = jCol_local + iCol_offset;

      applyHouseholderTransformation<T>(
          beta, v2m, R_band_entry(iCol_global, jCol_global), A_band.col(jCol_local));
    }

    for (Eigen::Index jCol_common = 0; jCol_common < n_common; ++jCol_common) {
      applyHouseholderTransformation<T>(
          beta, v2m, R_common_(iCol_global, jCol_common), A_common.col(jCol_common));
    }

    applyHouseholderTransformation<T>(beta, v2m, y_(iCol_global), b);
  }
}

template <typename T>
void OnlineBandedHouseholderQR<T>::addMutating(
    ColumnIndexedMatrix<MatrixType> A_common,
    Eigen::Ref<VectorType> b) {
  const auto n_common = R_common_.cols();
  const auto n_band = R_band_.cols();

  // TODO what to assert about A_band?
  MT_CHECK(A_common.cols() == n_common);

  // Now for the common parameters:
  for (Eigen::Index iCol = 0; iCol < n_common; ++iCol) {
    const auto [beta, mu] =
        computeHouseholderVec<T>(R_common_(iCol + n_band, iCol), A_common.col(iCol));
    if (beta == 0) {
      continue;
    }

    R_common_(iCol + n_band, iCol) = mu;

    const auto& v2m = A_common.col(iCol);
    for (Eigen::Index jCol_common = iCol + 1; jCol_common < n_common; ++jCol_common) {
      applyHouseholderTransformation<T>(
          beta, v2m, R_common_(iCol + n_band, jCol_common), A_common.col(jCol_common));
    }

    applyHouseholderTransformation<T>(beta, v2m, y_(n_band + iCol), b);
  }
}

template <typename T>
typename OnlineBandedHouseholderQR<T>::MatrixType OnlineBandedHouseholderQR<T>::R_dense() const {
  const auto n_common = R_common_.cols();
  const auto n_band = R_band_.cols();
  const auto bandwidth = R_band_.rows();

  const auto n_total = n_band + n_common;

  MatrixType result = MatrixType::Zero(n_total, n_total);

  for (Eigen::Index iCol_band = 0; iCol_band < n_band; ++iCol_band) {
    for (Eigen::Index jRow = iCol_band; (iCol_band - jRow) < bandwidth && jRow >= 0; --jRow) {
      result(jRow, iCol_band) = R_band_entry(jRow, iCol_band);
    }
  }

  for (Eigen::Index jRow = 0; jRow < n_total; ++jRow) {
    for (Eigen::Index iCol_common = 0; iCol_common < n_common; ++iCol_common) {
      result(jRow, n_band + iCol_common) = R_common_(jRow, iCol_common);
    }
  }

  return result;
}

template <typename T>
typename OnlineBandedHouseholderQR<T>::VectorType OnlineBandedHouseholderQR<T>::x_dense() const {
  VectorType result = VectorType::Zero(R_band_.cols() + R_common_.cols());

  const auto n_common = R_common_.cols();
  const auto n_band = R_band_.cols();
  const auto bandwidth = R_band_.rows();

  if (n_common > 0) {
    result.tail(n_common) =
        R_common_.bottomRows(n_common).template triangularView<Eigen::Upper>().solve(
            y_.tail(n_common));
  }

  if (n_band > 0) {
    for (Eigen::Index iRow = n_band - 1; iRow >= 0; --iRow) {
      T dotProd = n_common > 0 ? R_common_.row(iRow).dot(result.tail(n_common)) : T(0);

      const Eigen::Index bandwidth_cur = std::min(bandwidth, n_band - iRow);
      for (Eigen::Index jColOffset = 1; jColOffset < bandwidth_cur; ++jColOffset) {
        dotProd += R_band_entry(iRow, iRow + jColOffset) * result(iRow + jColOffset);
      }
      result(iRow) = (y_(iRow) - dotProd) / R_band_entry(iRow, iRow);
    }
  }

  return result;
}

template <typename T>
typename OnlineBandedHouseholderQR<T>::VectorType OnlineBandedHouseholderQR<T>::At_times_b() const {
  // As noted above, A^T*b = R^T*y
  VectorType result = VectorType::Zero(R_band_.cols() + R_common_.cols());

  const auto n_common = R_common_.cols();
  const auto n_band = R_band_.cols();
  const auto bandwidth = R_band_.rows();

  for (Eigen::Index iRow = 0; iRow < n_band; ++iRow) {
    const Eigen::Index bandwidth_cur = std::min(bandwidth, n_band - iRow);
    for (Eigen::Index jColOffset = 0; jColOffset < bandwidth_cur; ++jColOffset) {
      const Eigen::Index jCol = iRow + jColOffset;
      result(jCol) += R_band_entry(iRow, jCol) * y_(iRow);
    }
  }

  result.tail(n_common).noalias() = R_common_.transpose() * y_;

  return result;
}

template class OnlineBandedHouseholderQR<float>;
template class OnlineBandedHouseholderQR<double>;

} // namespace momentum
