/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/common/checks.h>

#include <Eigen/Core>
#include <span>

#include <deque>
#include <vector>

namespace momentum {

void validateColumnIndices(std::span<const Eigen::Index> colIndices, Eigen::Index maxEntry);

// Remaps the matrix such that A_new.col(i) = A_orig.col(colIndices[i]).
// This can be used to perform QR on a subset of the matrix columns.
template <typename MatrixType>
class ColumnIndexedMatrix {
 public:
  using VectorType = Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, 1>;

  explicit ColumnIndexedMatrix(Eigen::Ref<MatrixType> mat) : mat_(mat), useColumnIndices_(false) {}

  explicit ColumnIndexedMatrix(Eigen::Ref<MatrixType> mat, std::span<const Eigen::Index> colIndices)
      : mat_(mat), columnIndices_(colIndices), useColumnIndices_(true) {
    validateColumnIndices(colIndices, mat.cols());
  }

  [[nodiscard]] auto col(Eigen::Index colIdx) {
    if (useColumnIndices_) {
      return mat_.col(columnIndices_[colIdx]);
    } else {
      return mat_.col(colIdx);
    }
  }

  [[nodiscard]] auto rows() const {
    return mat_.rows();
  }

  [[nodiscard]] auto cols() const {
    if (useColumnIndices_) {
      return (Eigen::Index)columnIndices_.size();
    } else {
      return mat_.cols();
    }
  }

  [[nodiscard]] std::span<const Eigen::Index> columnIndices() const {
    return columnIndices_;
  }

  [[nodiscard]] VectorType transpose_times(Eigen::Ref<VectorType> b) const {
    if (useColumnIndices_) {
      VectorType result(columnIndices_.size());
      for (Eigen::Index iCol = 0; iCol < (Eigen::Index)columnIndices_.size(); ++iCol) {
        result[iCol] = mat_.col(iCol).dot(b);
      }
      return result;
    } else {
      return mat_.transpose() * b;
    }
  }

 private:
  Eigen::Ref<MatrixType> mat_;
  std::span<const Eigen::Index> columnIndices_;
  bool useColumnIndices_;
};

// The native Eigen matrix type reallocates the matrix every time you call
// either resize() or conservativeResize(), which means there is no way to
// have a matrix that reuses memory.  This is inconvenient, because there are
// definitely cases where we want to be able to reuse the same memory over
// and over for slightly different matrix sizes (for example, if we are computing
// our Jacobian blockwise).
template <typename T>
class ResizeableMatrix {
 public:
  using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  ResizeableMatrix() = default;
  explicit ResizeableMatrix(Eigen::Index rows, Eigen::Index cols = 1) {
    resizeAndSetZero(rows, cols);
  }

  void resizeAndSetZero(Eigen::Index rows, Eigen::Index cols = 1) {
    rows_ = rows;
    cols_ = cols;

    // This will not reallocate unless needed.
    data_.resize(rows * cols);
    std::fill(std::begin(data_), std::end(data_), T(0));
  }

  [[nodiscard]] Eigen::Index rows() const {
    return rows_;
  }
  [[nodiscard]] Eigen::Index cols() const {
    return cols_;
  }

  Eigen::Map<MatrixType, Eigen::Aligned16> mat() {
    return Eigen::Map<MatrixType, Eigen::Aligned16>(data_.data(), rows_, cols_);
  }
  Eigen::Map<const MatrixType, Eigen::Aligned16> mat() const {
    return Eigen::Map<const MatrixType, Eigen::Aligned16>(data_.data(), rows_, cols_);
  }

 private:
  Eigen::Index rows_ = 0;
  Eigen::Index cols_ = 0;
  std::vector<T> data_;
};

// This class is designed to solve least squares problems of the form:
//   min || A * x - b ||_2
// where
//   a. n is small-ish,
//   b. A is tall and _sparse_ (m >> n), BUT
//   c. (A^T * A) is _dense_.
//
// This turns out to be the common case for hand model IK, because:
//   a. The hand has only ~22 DOF,
//   b. Each constraint touches only a few DOFs (typically the 3-4 on a single finger), but
//   c. Constraints that touch the global DOFs tend to densify A^T*A.
// From the perspective of a QR factorization: since the R matrix
// has the same sparsity pattern as the L matrix in the Cholesky factorization of
// A^T*A, there is no point trying to exploit sparsity in the computation of R
// (e.g. by reordering columns).  So instead, we'll accept the O(n^2) cost of
// having a dense R and try to exploit sparsity in A.
//
// It turns out that it is possible to exploit sparsity in A if you do the factorization
// _row-wise_ instead of column-wise as is generally done.  One paper that uses Givens
// rotations to zero out one row at a time is this one:
//   Solution of Sparse Linear Squares Problems using Givens Rotations (1980)
//   by Alan George and Michael T. Heath
//
// However, Givens rotations aren't really ideal for our case because you have
// to compute a c/s pair for each entry of the matrix, and it involves a square
// root.  Also, row-wise operations are kind of slow in general due to poor
// cache locality on Eigen matrices.  What we can do instead is to apply
// Householder reflections to (k x n) blocks of the matrix; computing a
// Householder vector for a whole column only requires one square root, and
// it can be applied as a level-2 BLAS operation.  Moreover, this maps
// perfectly to our problem; a Jacobian matrix from a single constraint has
// the sparsity pattern where entire columns are dropped, and these zero
// columns can just be skipped entirely by the Householder process provided
// we are careful about the order we process them in.  And by processing the
// J submatrices one at a time, we need never assemble the full A matrix,
// eliminating one major disadvantage of QR relative to the normal equations
// (hence the name "OnlineHouseholderQR").
//
// Even better, we can do the whole operation in-place: the caller passes in
// the J and r matrix, and the columns of the J matrix are rotated out one by
// one, revealing the R matrix.  At the end, a simple back-substitution using
// R gives the x vector.  (in fact, the answer for the current submatrix of
// A,b is always available, as a consequence of doing the factorization sort-of
// row-wise).
//
// Note that because we apply the transforms as we go, this method is not currently
// suitable for multiple right-hand-sides.
template <typename T>
class OnlineHouseholderQR {
 public:
  using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  // The lambda here lets you pass in a lambda for a Gauss-Newton/Levenberg-Marquardt solver.
  // Basically this emulates solving the following system, but without needing to append the
  // extra matrix:
  //   | [ lambda * I ] [ x ] - [ 0 ] |^2
  //   | [     A      ]         [ b ] |
  explicit OnlineHouseholderQR(Eigen::Index n, T lambda = T(0));

  void addMutating(Eigen::Ref<MatrixType> A, Eigen::Ref<VectorType> b) {
    addMutating(ColumnIndexedMatrix<MatrixType>(A), b);
  }

  // We require passing in both A and rhs so that we can multiply rhs by Q
  // as R is being constructed.
  // This is an 'in-place' version which computes the factorization inside the
  // passed-in matrix; it is here so that use cases that don't need to re-use
  // the matrix can avoid copying.
  void addMutating(ColumnIndexedMatrix<MatrixType> A, Eigen::Ref<VectorType> b);

  // This version can be made efficient if you std::move() the passed-in matrices.
  void add(MatrixType A, VectorType b);

  [[nodiscard]] VectorType result() const;

  [[nodiscard]] const MatrixType& R() const {
    return R_;
  }
  [[nodiscard]] const VectorType& y() const {
    return y_;
  }
  [[nodiscard]] VectorType At_times_b() const;

  void reset();
  void reset(Eigen::Index n, T lambda = T(0));

 private:
  // R is defined by
  //   Q*A = ( R )
  //         ( 0 )
  MatrixType R_;

  // We define y as
  //   Q*b = y
  VectorType y_;
};

// This class is designed to solve least squares problems of the form:
//   min || A * x - b ||_2
// where A has the block structure:
//   [ A_11                    A_1n ]
//   [       A_22              A_2n ]
//   [              A_33       A_3n ]
//   [                     ... ...  ]
//
// Basically, you have these blocks along the diagonal where each set of
// x parameters is disjoint and then you have a set of x's that are common
// to all blocks.
//
// This is basically the case for calibration problems, where each hand
// pose is independent but you have a set of common calibration parameters
// that connect them.
//
// Besides the blockwise structure, the behavior of this solver should be
// identical to the OnlineHouseholderQR solver above, so please see those
// note for more details.  The key takeaway is that this can be used in an
// _online_ fashion, so you can pass in the individual poses one at a time
// and then throw away the Jacobians (in fact, a future version of this class
// could even consider throwing away the intermediate R matrices if memory
// becomes an issue).
template <typename T>
class OnlineBlockHouseholderQR {
 public:
  using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  // The lambda here lets you pass in a lambda for a Gauss-Newton/Levenberg-Marquardt solver.
  // Basically this emulates solving the following system, but without needing to append the
  // extra matrix:
  //   | [ lambda * I ] [ x ] - [ 0 ] |^2
  //   | [     A      ]         [ b ] |
  explicit OnlineBlockHouseholderQR(Eigen::Index n_common, T lambda = T(0));

  void add(size_t iBlock, MatrixType A_diag, MatrixType A_common, VectorType b) {
    addMutating(iBlock, A_diag, A_common, b);
  }

  // We require passing in both A and rhs so that we can multiply rhs by Q
  // as R is being constructed.
  // This is an 'in-place' version which computes the factorization inside the
  // passed-in matrix; it is here so that use cases that don't need to re-use
  // the matrix can avoid copying.
  void addMutating(
      size_t iBlock,
      Eigen::Ref<MatrixType> A_diag,
      Eigen::Ref<MatrixType> A_common,
      Eigen::Ref<VectorType> b) {
    addMutating(
        iBlock,
        ColumnIndexedMatrix<MatrixType>(A_diag),
        ColumnIndexedMatrix<MatrixType>(A_common),
        b);
  }

  // Mutating addition operator that also remaps the columns, allowing passing in arbitrary subsets
  // of the columns in A_diag and A_common.
  void addMutating(
      size_t iBlock,
      ColumnIndexedMatrix<MatrixType> A_diag,
      ColumnIndexedMatrix<MatrixType> A_common,
      Eigen::Ref<VectorType> b);

  // You can retrieve the result x in one of two ways:
  //   1. As a single, dense vector, which solves everything simultaneously, or
  //   2. A block at a time.
  // The latter is a little bit slower because it re-does the back-substitution
  // for the final rows every time, but for the common case where the last set
  // of variables is very small this should not be an issue.

  // Retrieve the full dense result:
  [[nodiscard]] VectorType x_dense() const;

  // Retrieve the answer for a single block:
  [[nodiscard]] VectorType x_i(size_t iBlock) const;

  // Retrieve the answer for just the final "common" rows.
  [[nodiscard]] VectorType x_n() const;

  [[nodiscard]] const MatrixType& R_ii(size_t iBlock) const {
    return R_ii_[iBlock];
  }

  [[nodiscard]] const VectorType& y_i(size_t iBlock) const {
    return y_i_[iBlock];
  }

  [[nodiscard]] MatrixType R_dense() const;
  [[nodiscard]] MatrixType y_dense() const;

  [[nodiscard]] VectorType At_times_b_i(size_t iBlock) const;
  [[nodiscard]] VectorType At_times_b_n() const;
  [[nodiscard]] VectorType At_times_b_dense() const;
  [[nodiscard]] T At_times_b_dot(Eigen::Ref<const VectorType> rhs) const;

  void reset();

 private:
  const double lambda_;

  // R is defined by
  //   Q*A = ( R )
  //         ( 0 )
  // R looks like:
  //   [ R_ii[0]                          R_in[0] ]
  //   [            R_ii[1]               R_in[1] ]
  //   [                         R_ii[2]  R_in[2] ]
  //   [                                  R_nn    ]
  std::deque<MatrixType> R_ii_;
  std::deque<MatrixType> R_in_;
  MatrixType R_nn_;

  // We define y as
  //   Q*b = y
  // y looks like:
  //   [ y_i ]
  //   [ ... ]
  //   [ y_n ]
  std::deque<VectorType> y_i_;
  VectorType y_n_;
};

// Another online QR solver, but for matrices that have a particular
// type of band structure.  The left-hand part of the matrix is
// required to have a limited bandwidth, and we permit a "common"
// section at the right-hand side of the matrix, like this:
//      ...b a n d e d...     common
//   [ a_11                   a_15 ]
//   [ a_21  a_22             a_25 ]
//   [ a_31  a_32             a_35 ]
//   [       a_42  a_43       a_45 ]
//   [             a_53  a_54 a_55 ]
//   [                   a_64 a_65 ]
// Here, the "bandwidth" is in _columns_; that is, the above matrix
// has a bandwidth of 2 because no row spans more than two columns (excepting
// the "common" section).
// We also allow a "common" section at the right-hand side of the matrix
// which contains parameters that are allowed to be shared between all rows.
// This might seem very specialized, but it maps to an extremely common
// set of problems, where the band structure maps to the time dimension
// (where smoothness constraints tend to connect adjacent timepoints) while
// common parameters map to a set of common optimized parameters.  For example:
//   body tracking: band structure maps to smoothness constraints between adjacent frames,
//     while common parameters control body scale
//   camera calibration: band structure maps to smoothness constraints between
//     adjacent frames of the extrinsics, while common parameters refer to
//     intrinsics.
template <typename T>
class OnlineBandedHouseholderQR {
 public:
  using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  // The lambda here lets you pass in a lambda for a Gauss-Newton/Levenberg-Marquardt solver.
  // Basically this emulates solving the following system, but without needing to append the
  // extra matrix:
  //   | [ lambda * I ] [ x ] - [ 0 ] |^2
  //   | [     A      ]         [ b ] |
  explicit OnlineBandedHouseholderQR(
      Eigen::Index n_band,
      Eigen::Index n_common,
      Eigen::Index bandwidth,
      T lambda = T(0));

  void add(size_t iCol_offset, MatrixType A_band, MatrixType A_common, VectorType b) {
    addMutating(iCol_offset, A_band, A_common, b);
  }

  void add(MatrixType A_common, VectorType b) {
    addMutating(A_common, b);
  }

  // We require passing in both A and rhs so that we can multiply rhs by Q
  // as R is being constructed.
  // This is an 'in-place' version which computes the factorization inside the
  // passed-in matrix; it is here so that use cases that don't need to re-use
  // the matrix can avoid copying.
  void addMutating(
      const Eigen::Index iCol_offset,
      Eigen::Ref<MatrixType> A_band,
      Eigen::Ref<MatrixType> A_common,
      Eigen::Ref<VectorType> b) {
    addMutating(
        iCol_offset,
        ColumnIndexedMatrix<MatrixType>(A_band),
        ColumnIndexedMatrix<MatrixType>(A_common),
        b);
  }

  void addMutating(Eigen::Ref<MatrixType> A_common, Eigen::Ref<VectorType> b) {
    addMutating(ColumnIndexedMatrix<MatrixType>(A_common), b);
  }

  // Mutating addition operator that also remaps the columns, allowing passing in arbitrary subsets
  // of the columns in A_diag and A_common.
  void addMutating(
      Eigen::Index iCol_offset,
      ColumnIndexedMatrix<MatrixType> A_band,
      ColumnIndexedMatrix<MatrixType> A_common,
      Eigen::Ref<VectorType> b);

  void addMutating(ColumnIndexedMatrix<MatrixType> A_common, Eigen::Ref<VectorType> b);

  // Zeroes out just the banded part of the matrix; applies Householder rotations that
  // zero out everything in A_band.  The A_common will still have nonzero entries so
  // after calling this you should call addMutating(A_common, b) to complete adding the
  // columns to the matrix.
  //
  // Why is this useful?  If you are careful about it, you can call zeroBandedPart on
  // multiple non-overlapping segments of the banded matrix at a time, but processing
  // A_common must be serialized.  Since in many problems the number of common variables
  // is much smaller than the number of banded variables this can provide a substantial
  // speedup in practice.
  void zeroBandedPart(
      Eigen::Index iCol_offset,
      ColumnIndexedMatrix<MatrixType> A_band,
      ColumnIndexedMatrix<MatrixType> A_common,
      Eigen::Ref<VectorType> b);

  // You can retrieve the result x in one of two ways:
  //   1. As a single, dense vector, which solves everything simultaneously, or
  //   2. A block at a time.
  // The latter is a little bit slower because it re-does the back-substitution
  // for the final rows every time, but for the common case where the last set
  // of variables is very small this should not be an issue.

  // Retrieve the full dense result:
  [[nodiscard]] VectorType x_dense() const;
  [[nodiscard]] MatrixType R_dense() const;
  [[nodiscard]] MatrixType y_dense() const {
    return y_;
  }

  [[nodiscard]] VectorType At_times_b() const;

  void reset();

  [[nodiscard]] Eigen::Index bandwidth() const {
    return R_band_.rows();
  }

  [[nodiscard]] Eigen::Index n_band() const {
    return R_band_.cols();
  }

  [[nodiscard]] Eigen::Index n_common() const {
    return R_common_.cols();
  }

 private:
  // Sets the diagonal of the R matrix to lambda.
  void initializeDiagonal();

  [[nodiscard]] T R_band_entry(Eigen::Index iRow, Eigen::Index jCol) const {
    const auto bandwidth = R_band_.rows();
    MT_CHECK(iRow <= jCol);
    MT_CHECK(jCol - iRow <= bandwidth);
    return R_band_(bandwidth + iRow - jCol - 1, jCol);
  }

  [[nodiscard]] T& R_band_entry(Eigen::Index iRow, Eigen::Index jCol) {
    const auto bandwidth = R_band_.rows();
    MT_CHECK(iRow <= jCol);
    MT_CHECK(jCol - iRow <= bandwidth);
    return R_band_(bandwidth + iRow - jCol - 1, jCol);
  }

  const double lambda_;

  // R is defined by
  //   Q*A = ( R )
  //         ( 0 )
  // R looks like:
  //   [ *  *  *        *  * ]
  //   [    *  *  *     *  * ]
  //   [       *  *  *  *  * ]
  //   [          *  *  *  * ]
  //   [             *  *  * ]
  //   [                *  * ]
  //   [                   * ]
  //     ^^^^^^^^^      ^^^^
  //     R_band_      R_common_
  MatrixType R_common_;

  // Uses "banded" storage, which looks like this:
  //  [ r_11 r_12           ]
  //  [      r_22 r_23      ]
  //  [           r_33 r_34 ]
  MatrixType R_band_;

  VectorType y_;
};

} // namespace momentum
