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
// Splitting v_1=1 out of storage saves one multiply per dot product and lets
// computeHouseholderVec reuse the input column to store v_2m in place.
template <typename T>
void applyHouseholderTransformation(
    T beta,
    const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& v_2m,
    T& y_1,
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> y_2m) {
  MT_CHECK(v_2m.size() == y_2m.size());

  // dotProd = v^T * y, exploiting v_1 = 1 implicitly.
  const T dotProd = y_1 + v_2m.dot(y_2m);
  const T scalar = dotProd * beta;

  // y <- y - beta * v * (v^T * y)
  y_1 -= scalar;
  y_2m.noalias() -= scalar * v_2m;
}

// Apply the Householder transformation to columns [colStart, colEnd) of a
// ColumnIndexedMatrix. Y_1 holds the leading row entries (full width); Y_2m
// holds the trailing rows used to form v_2m.
//
// TODO: this currently only iterates column-by-column. The "Block" name and
// the loop structure suggest that a true blocked variant (e.g. WY-form
// updates) was intended but never implemented.
template <typename T, typename MatrixType, typename RowType>
void applyHouseholderTransformationBlock(
    T beta,
    const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& v_2m,
    RowType Y_1,
    ColumnIndexedMatrix<MatrixType> Y_2m,
    Eigen::Index colStart,
    Eigen::Index colEnd) {
  [[maybe_unused]] const auto nCols = Y_2m.cols();
  MT_CHECK(v_2m.rows() == Y_2m.rows());
  MT_CHECK(colStart <= colEnd);
  MT_CHECK(colEnd <= nCols);
  MT_CHECK(Y_1.cols() == nCols);

  [[maybe_unused]] const auto colIndices = Y_2m.columnIndices();
  MT_CHECK(colIndices.empty() || colIndices.size() >= colEnd);
  auto jCol = colStart;

  while (jCol < colEnd) {
    applyHouseholderTransformation<T>(beta, v_2m, Y_1(jCol), Y_2m.col(jCol));
    ++jCol;
  }
}

// Computes the Householder reflection vector for the column [x1 x2...xm]^T
// (x1 is passed in separately because it comes from the R matrix rather than A).
// Uses the stable algorithm from Golub and van Loan, "Matrix Computations" 4e,
// Algorithm 5.1.1.
// The result of applying the Householder transform (I - beta*v*v^T) is a vector of the form
//     [mu 0 0 0 ... 0]^T
// Returns the pair [beta, mu]. Side effect: `vec` is overwritten in place with
// v_2m (the implicit v_1 = 1 is not stored). Caller must treat `vec` as the
// reflector vector after this call, not the original column data.
//
// Numerical-stability notes:
//   - When x1 > 0, computing v1 = x1 - mu would suffer cancellation
//     (mu ~ |x1| when sigma is small), so we use the algebraically-equivalent
//     -sigma / (x1 + mu) form. The x1 <= 0 branch is already cancellation-free.
//   - sigma == 0 means the trailing entries are exactly zero, so x1 already
//     sits in the [mu 0 ... 0] form; we return beta = 0 to signal "skip".
//     Callers must check beta == 0 before applying the transform; otherwise
//     `vec` would be divided by an undefined v1.
//   - mu has the sign opposite to v1 by construction, so the diagonal entry of
//     R may be negative. This is standard and downstream solves only depend on
//     R^T R, so the sign is irrelevant.
template <typename T>
std::pair<T, T> computeHouseholderVec(
    T x1,
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>>
        vec) // NOLINT(facebook-hte-ConstantArgumentPassByValue)
{
  const T sigma = vec.squaredNorm();
  if (sigma == 0) {
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
    // Seed the diagonal with `lambda` so the first add() effectively appends
    // a Tikhonov regularizer block (lambda*I, 0) to (A, b).
    R_.diagonal().setConstant(lambda);
  }

  y_.resize(n);
  y_.setZero();
}

// Definition of the public helper declared in online_householder_qr.h; called from
// `ColumnIndexedMatrix`'s constructor and from a small number of tests.
void validateColumnIndices(momentum::span<const Eigen::Index> colIndices, Eigen::Index maxEntry) {
#ifndef NDEBUG
  {
    for (const auto& idx : colIndices) {
      MT_CHECK(idx < maxEntry);
    }

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

  // Conceptually we form the stacked matrix [R; A] and reduce it back to upper
  // triangular by sweeping a Householder reflector down each column. R already
  // satisfies the upper-triangular invariant from prior calls, so for column
  // iCol we only need to zero the new rows in A.col(iCol) below R_(iCol, iCol).
  // Side effect: A and b are overwritten in place — A.col(iCol) is replaced
  // with v_2m (the reflector for column iCol), and rhs entries are mixed into
  // y_. Caller must not reuse A/b after this call.
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
      // A.col(iCol) was already zero below R_(iCol,iCol); R is unchanged for
      // this column and nothing needs to be applied to the trailing columns.
      continue;
    }

    // computeHouseholderVec guarantees that applying the transform to iCol
    // results in a vector [mu 0 ... 0].
    R_(iCol, iCol) = mu;

    // v2m aliases A.col(iCol), which now stores the reflector tail (see
    // computeHouseholderVec side effect).
    const auto& v2m = A.col(iCol);
    for (Eigen::Index jCol = iCol + 1; jCol < n; ++jCol) {
      applyHouseholderTransformation<T>(beta, v2m, R_(iCol, jCol), A.col(jCol));
    }

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
  // Note on rank deficiency: when `lambda == 0` and a column never received any contribution
  // from `add()`, the corresponding diagonal of R is zero. The triangular back-substitution
  // below then produces 0/0 for that row's update, which Eigen's TriangularView solver returns
  // as 0 (rather than NaN) — this matches the documented "empty QR returns zero" behavior the
  // existing tests rely on. Callers that need a meaningful solution under rank deficiency
  // should construct with a non-zero `lambda` to add Tikhonov damping on the diagonal.
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

  // Phase 1: zero out the per-block diagonal columns A_diag. Each reflector
  // also touches A_common (cross-block fill) so that the running R has the
  // arrowhead structure [R_ii R_in; 0 R_nn].
  for (Eigen::Index iCol = 0; iCol < n_diag; ++iCol) {
    if (A_diag.col(iCol).isZero()) {
      // Fast skip for empty Jacobian columns. Note that this is stricter than
      // the beta == 0 check below: a column whose squared norm underflows to 0
      // would be caught there too, but isZero() avoids the squaredNorm() cost.
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

    const auto& v2m = A_diag.col(iCol);
    if (iCol + 1 < n_diag) {
      applyHouseholderTransformationBlock<T>(beta, v2m, R_ii.row(iCol), A_diag, iCol + 1, n_diag);
    }

    if (n_common > 0) {
      // Couple the per-block rotation into the common columns; this is what
      // lets the block solve later subtract R_in * x_n.
      applyHouseholderTransformationBlock<T>(beta, v2m, R_in.row(iCol), A_common, 0, n_common);
    }

    applyHouseholderTransformation<T>(beta, v2m, y_i(iCol), b);
  }

  // Phase 2: with A_diag now zero in this block, fold the residual A_common
  // and b into the shared (R_nn, y_n) tail. This is the same algorithm as
  // OnlineHouseholderQR::addMutating but operating on the bottom-right block.
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
  // Solve the arrowhead system in two stages, exploiting that R has structure
  //   [R_ii   R_in] [x_i]   [y_i]
  //   [  0    R_nn] [x_n] = [y_n]
  // 1) Back-solve R_nn * x_n = y_n once for the shared parameters.
  // 2) For each block: back-solve R_ii * x_i = y_i - R_in * x_n.
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
  // A^T*b = R^T*y because Q is orthonormal (see OnlineHouseholderQR::At_times_b
  // for the derivation). For block i this restricts to R_ii^T*y_i since R_ii
  // is the only block that interacts with the per-block parameters.
  return R_ii_[iBlock].transpose() * y_i_[iBlock];
}

template <typename T>
typename OnlineBlockHouseholderQR<T>::VectorType OnlineBlockHouseholderQR<T>::At_times_b_n() const {
  // For the common parameters every block contributes via its R_in coupling
  // term, so we sum R_in_[i]^T * y_i_[i] across blocks alongside R_nn^T * y_n.
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
  // R_band_ uses a packed banded layout: R_band_(i, j) stores the (i, j+k)
  // entry of the dense R for some shift k determined by R_band_entry. Rows of
  // the storage matrix correspond to diagonals (the last row is the main
  // diagonal), columns correspond to the global column index.
  //
  // Bandwidth is only allowed to be zero if there are no parameters in the banded part,
  // in which case it falls back to behaving the same as a regular Householder QR (for
  // the common part).
  MT_CHECK(n_band == 0 || bandwidth > 0);

  initializeDiagonal();
}

template <typename T>
void OnlineBandedHouseholderQR<T>::initializeDiagonal() {
  // Seeds the Tikhonov regularizer lambda*I on R's diagonal in both storage
  // layouts: the bottom-right square of R_common_ (regular dense storage), and
  // the bottom row of R_band_ (which holds the main diagonal in banded
  // storage).
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

  // A_band.cols() bounds bandwidth because each new column can only fill in
  // rows up to (bandwidth - 1) above its diagonal entry.
  MT_CHECK(A_band.cols() <= R_band_.rows());
  MT_CHECK(iCol_offset + A_band.cols() <= R_band_.cols());

  MT_CHECK(A_band.rows() == b.rows());
  MT_CHECK(A_common.rows() == b.rows());
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

    const auto& v2m = A_band.col(iCol_local);

    // R_band_ uses packed banded storage, so its row(iCol_global) cannot be
    // expressed as a contiguous Eigen row expression — we must address each
    // (iCol_global, jCol_global) entry individually via R_band_entry.
    for (Eigen::Index jCol_local = iCol_local + 1; jCol_local < A_band.cols(); ++jCol_local) {
      const auto jCol_global = jCol_local + iCol_offset;
      applyHouseholderTransformation<T>(
          beta, v2m, R_band_entry(iCol_global, jCol_global), A_band.col(jCol_local));
    }

    // R_common_ is a dense matrix, so the block-friendly helper applies.
    if (n_common > 0) {
      applyHouseholderTransformationBlock<T>(
          beta, v2m, R_common_.row(iCol_global), A_common, 0, n_common);
    }

    applyHouseholderTransformation<T>(beta, v2m, y_(iCol_global), b);
  }
}

template <typename T>
void OnlineBandedHouseholderQR<T>::addMutating(
    ColumnIndexedMatrix<MatrixType> A_common,
    Eigen::Ref<VectorType> b) {
  // Reduce A_common into the bottom-right (n_common x n_common) block of
  // R_common_. R_common_ stores the common columns of R as a dense matrix
  // whose top n_band rows hold the cross-coupling against the banded part;
  // the diagonal of R_common itself begins at row n_band, which is why every
  // row index below is shifted by n_band.
  const auto n_common = R_common_.cols();
  const auto n_band = R_band_.cols();

  MT_CHECK(A_common.rows() == b.rows());
  MT_CHECK(A_common.cols() == n_common);

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
  // Unpack the banded + common storage into a single dense upper-triangular
  // matrix. Row of the dense result corresponds to a row of R; the banded
  // block occupies rows/cols [0, n_band) and the common block occupies
  // [n_band, n_band+n_common).
  const auto n_common = R_common_.cols();
  const auto n_band = R_band_.cols();
  const auto bandwidth = R_band_.rows();

  const auto n_total = n_band + n_common;

  MatrixType result = MatrixType::Zero(n_total, n_total);

  // Walk each banded column upward at most `bandwidth` rows; rows above are
  // structurally zero in band storage.
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
  // Two-stage back-substitution analogous to OnlineBlockHouseholderQR::x_dense:
  //   1) Solve the dense common block first to obtain x_n.
  //   2) Sweep the banded block from bottom to top, subtracting R_in coupling
  //      via R_common_.row(iRow).dot(x_n) and the in-band tail via R_band_entry.
  // The inner loop's bandwidth_cur clamp avoids reading past the rightmost
  // banded column (n_band - 1).
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
    // Note on rank deficiency: when `lambda == 0` and a column never received any contribution,
    // the corresponding banded diagonal entry is zero. We pin that row's result to 0 instead of
    // dividing — the same observable behavior as the dense `OnlineHouseholderQR::result` path,
    // and what the existing edge-case tests rely on (an empty / zero-bandwidth solver returns
    // a zero vector). Callers wanting a meaningful damped solution under genuine rank
    // deficiency should construct with a non-zero `lambda`.
    for (Eigen::Index iRow = n_band - 1; iRow >= 0; --iRow) {
      const T diag = R_band_entry(iRow, iRow);
      if (diag == T(0)) {
        result(iRow) = T(0);
        continue;
      }

      T dotProd = n_common > 0 ? R_common_.row(iRow).dot(result.tail(n_common)) : T(0);

      const Eigen::Index bandwidth_cur = std::min(bandwidth, n_band - iRow);
      for (Eigen::Index jColOffset = 1; jColOffset < bandwidth_cur; ++jColOffset) {
        dotProd += R_band_entry(iRow, iRow + jColOffset) * result(iRow + jColOffset);
      }
      result(iRow) = (y_(iRow) - dotProd) / diag;
    }
  }

  return result;
}

template <typename T>
typename OnlineBandedHouseholderQR<T>::VectorType OnlineBandedHouseholderQR<T>::At_times_b() const {
  // A^T*b = R^T*y (Q is orthonormal). Computed by accumulating column sums:
  // each row iRow of R contributes R(iRow, jCol)*y(iRow) into result(jCol).
  // The banded loop walks only the in-band entries; the common columns are
  // dense and folded in by the final R_common_.transpose() * y_.
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
