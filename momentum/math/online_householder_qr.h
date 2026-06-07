/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/common/aligned.h>
#include <momentum/common/checks.h>
#include <momentum/math/resizeable_matrix.h>
#include <momentum/math/span_compat.h>

#include <Eigen/Core>

#include <deque>
#include <vector>

namespace momentum {

/// Validates that every entry in @p colIndices is in [0, maxEntry); throws otherwise.
/// @param colIndices Column indices to validate.
/// @param maxEntry Exclusive upper bound (typically the underlying matrix's `cols()`).
void validateColumnIndices(momentum::span<const Eigen::Index> colIndices, Eigen::Index maxEntry);

/// View over a matrix that exposes a (possibly reordered) subset of its columns.
///
/// When constructed with a `colIndices` span, accessing column @c i of this view returns
/// the underlying matrix's column @c colIndices[i]. This lets callers run QR on a column
/// subset without copying the matrix. When constructed without indices, this is a thin
/// pass-through to the wrapped Eigen reference.
///
/// @note Holds an `Eigen::Ref` and a non-owning `span` — the underlying matrix and index
///   buffer must outlive this view. The view is mutable: mutations through `col()` write
///   back into the wrapped matrix.
template <typename MatrixType>
class ColumnIndexedMatrix {
 public:
  using VectorType = Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, 1>;

  explicit ColumnIndexedMatrix(Eigen::Ref<MatrixType> mat) : mat_(mat), useColumnIndices_(false) {}

  explicit ColumnIndexedMatrix(
      Eigen::Ref<MatrixType> mat,
      momentum::span<const Eigen::Index> colIndices)
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

  [[nodiscard]] momentum::span<const Eigen::Index> columnIndices() const {
    return columnIndices_;
  }

  /// Computes `this->transpose() * b`, returning a vector of length `cols()`.
  /// @param b Vector with `rows()` entries.
  /// @return Vector with `cols()` entries; entry @c i equals `view.col(i).dot(b)`.
  [[nodiscard]] VectorType transpose_times(Eigen::Ref<VectorType> b) const {
    if (useColumnIndices_) {
      VectorType result(columnIndices_.size());
      for (Eigen::Index iCol = 0; iCol < (Eigen::Index)columnIndices_.size(); ++iCol) {
        result[iCol] = mat_.col(columnIndices_[iCol]).dot(b);
      }
      return result;
    } else {
      return mat_.transpose() * b;
    }
  }

 private:
  Eigen::Ref<MatrixType> mat_;
  momentum::span<const Eigen::Index> columnIndices_;
  bool useColumnIndices_;
};

/// Online (incremental) Householder QR solver for tall, column-sparse least-squares
/// problems.
///
/// Solves
/// \code
///   min || A * x - b ||_2
/// \endcode
/// where
///   a. n (the number of unknowns) is small-ish,
///   b. A is tall and _sparse_ (m >> n), BUT
///   c. (A^T * A) is _dense_.
///
/// This turns out to be the common case for hand-model IK, because:
///   a. The hand has only ~22 DOF,
///   b. Each constraint touches only a few DOFs (typically the 3-4 on a single finger), but
///   c. Constraints that touch the global DOFs tend to densify A^T*A.
/// From the perspective of a QR factorization: since the R matrix has the same sparsity
/// pattern as the L matrix in the Cholesky factorization of A^T*A, there is no point
/// trying to exploit sparsity in the computation of R (e.g. by reordering columns). So
/// instead, we accept the O(n^2) cost of a dense R and try to exploit sparsity in A.
///
/// It is possible to exploit sparsity in A if you do the factorization _row-wise_
/// instead of column-wise as is generally done. One paper that uses Givens rotations to
/// zero out one row at a time is:
///   Solution of Sparse Linear Squares Problems using Givens Rotations (1980)
///   by Alan George and Michael T. Heath
///
/// However, Givens rotations aren't really ideal for our case because you have to
/// compute a c/s pair for each entry of the matrix, and it involves a square root. Also,
/// row-wise operations are slow in general due to poor cache locality on Eigen matrices.
/// What we do instead is apply Householder reflections to (k x n) blocks of the matrix;
/// computing a Householder vector for a whole column only requires one square root, and
/// it can be applied as a level-2 BLAS operation. Moreover, this maps perfectly to our
/// problem; a Jacobian matrix from a single constraint has the sparsity pattern where
/// entire columns are dropped, and these zero columns can just be skipped entirely by
/// the Householder process provided we are careful about the order we process them in.
/// And by processing the J submatrices one at a time, we never assemble the full A
/// matrix, eliminating one major disadvantage of QR relative to the normal equations
/// (hence the name "OnlineHouseholderQR").
///
/// Even better, we can do the whole operation in-place: the caller passes in the J and r
/// matrix, and the columns of the J matrix are rotated out one by one, revealing the R
/// matrix. At the end, a simple back-substitution using R gives the x vector. (In fact,
/// the answer for the current submatrix of A,b is always available, as a consequence of
/// doing the factorization sort-of row-wise.)
///
/// @note Because the transforms are applied as we go, this method is not currently
///   suitable for multiple right-hand-sides.
///
/// ### Usage contract
/// 1. Construct with the number of unknowns @c n and (optionally) a Levenberg-Marquardt
///    damping @c lambda.
/// 2. Call `add()`/`addMutating()` once per Jacobian/residual block. Internal state
///    (`R_`, `y_`) is mutated incrementally on each call.
/// 3. Call `result()` to back-substitute and recover @c x. `R()`, `y()`, and
///    `At_times_b()` are also valid at any point and reflect the data accumulated so far.
/// 4. Call `reset()` to discard accumulated state and reuse the solver.
template <typename T>
class OnlineHouseholderQR {
 public:
  using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  /// Constructs a solver for problems with @p n unknowns.
  ///
  /// The @p lambda parameter implements Gauss-Newton/Levenberg-Marquardt damping,
  /// emulating the augmented least-squares system
  /// \code
  ///   | [ lambda * I ] [ x ] - [ 0 ] |^2
  ///   | [     A      ]         [ b ] |
  /// \endcode
  /// without materializing the extra `lambda * I` rows. R is initialized with
  /// `lambda` on its diagonal so the first `add()` call sees the regularization.
  explicit OnlineHouseholderQR(Eigen::Index n, T lambda = T(0));

  /// In-place overload: factorizes @p A and @p b directly, mutating both buffers.
  /// Convenience wrapper that wraps @p A in a `ColumnIndexedMatrix`.
  void addMutating(Eigen::Ref<MatrixType> A, Eigen::Ref<VectorType> b) {
    addMutating(ColumnIndexedMatrix<MatrixType>(A), b);
  }

  /// In-place addition of a Jacobian/residual block to the running QR factorization.
  ///
  /// Both @p A and @p b are required so we can apply Q to the right-hand side as R is
  /// being constructed. The factorization is computed inside the passed-in buffers, so
  /// callers who do not need the original data can avoid a copy.
  ///
  /// @pre `A.cols()` equals the @c n passed to the constructor.
  /// @pre `A.rows()` equals `b.size()`.
  /// @post Internal `R_` and `y_` are updated to reflect the combined (prior + new) data.
  ///   The contents of @p A and @p b are arbitrary on return.
  void addMutating(ColumnIndexedMatrix<MatrixType> A, Eigen::Ref<VectorType> b);

  /// Copying overload of `addMutating()`.
  /// @note Pass the arguments via `std::move` to avoid the internal copy when the caller
  ///   no longer needs them.
  void add(MatrixType A, VectorType b);

  /// Solves `R * x = y` by back-substitution using the currently accumulated state and
  /// returns @c x.
  /// @return Vector of length @c n containing the least-squares solution for the data
  ///   added so far.
  [[nodiscard]] VectorType result() const;

  /// @return Upper-triangular R factor of the current accumulated factorization.
  [[nodiscard]] const MatrixType& R() const {
    return R_;
  }
  /// @return The transformed right-hand side `Q * b` (top @c n entries).
  [[nodiscard]] const VectorType& y() const {
    return y_;
  }
  /// @return `A^T * b` for the data accumulated so far, computed from `R^T * y` without
  ///   needing the original A.
  [[nodiscard]] VectorType At_times_b() const;

  /// Discards accumulated state, keeping the existing @c n and @c lambda.
  void reset();
  /// Discards accumulated state and reconfigures the solver with new @p n and @p lambda.
  void reset(Eigen::Index n, T lambda = T(0));

 private:
  /// Upper-triangular factor defined by `Q*A = (R; 0)`.
  MatrixType R_;

  /// Transformed right-hand side defined by `Q*b = y` (top @c n entries).
  VectorType y_;
};

/// Online Householder QR solver for least-squares problems with block-arrowhead structure.
///
/// Solves
/// \code
///   min || A * x - b ||_2
/// \endcode
/// where A has the block structure
/// \code
///   [ A_11                    A_1n ]
///   [       A_22              A_2n ]
///   [              A_33       A_3n ]
///   [                     ... ...  ]
/// \endcode
/// i.e. independent diagonal blocks `A_ii` (one set of disjoint `x_i` per block) plus a
/// shared set of "common" columns `A_in` that are used by every block.
///
/// This is the case for calibration problems, where each hand pose is independent but a
/// shared set of calibration parameters connects them.
///
/// Besides the blockwise structure, the behavior of this solver is identical to
/// OnlineHouseholderQR — see that class's notes for the algorithmic details. The key
/// takeaway is that this can be used in an _online_ fashion: poses are passed in one at
/// a time and the Jacobians can then be thrown away.
///
/// @note A future version could also discard the intermediate `R_ii_` and `R_in_`
///   blocks if memory becomes an issue (only `R_nn_` is required to compute `x_n()`).
///
/// ### Usage contract
/// 1. Construct with the number of common unknowns @c n_common and (optionally)
///    @c lambda damping. Diagonal block sizes are inferred from each `add()` call.
/// 2. Call `add()`/`addMutating()` once per block, supplying a unique @c iBlock index.
///    Internal `R_ii_`, `R_in_`, `R_nn_`, `y_i_`, and `y_n_` are mutated incrementally.
/// 3. Retrieve results via `x_dense()`, `x_i(iBlock)`, or `x_n()`.
/// 4. Call `reset()` to discard accumulated state and reuse the solver.
template <typename T>
class OnlineBlockHouseholderQR {
 public:
  using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  /// Constructs a solver for problems with @p n_common shared unknowns.
  ///
  /// Per-block unknown counts are determined from the first `add()` call for each block.
  /// @p lambda implements LM-style damping, emulating
  /// \code
  ///   | [ lambda * I ] [ x ] - [ 0 ] |^2
  ///   | [     A      ]         [ b ] |
  /// \endcode
  /// without materializing the extra rows.
  explicit OnlineBlockHouseholderQR(Eigen::Index n_common, T lambda = T(0));

  /// Copying overload of `addMutating()`.
  void add(size_t iBlock, MatrixType A_diag, MatrixType A_common, VectorType b) {
    addMutating(iBlock, A_diag, A_common, b);
  }

  /// In-place addition; convenience wrapper that wraps the matrices in
  /// `ColumnIndexedMatrix` views without remapping.
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

  /// In-place addition that also accepts column-remapped views, so callers can pass
  /// arbitrary subsets of the columns in @p A_diag and @p A_common.
  ///
  /// @param iBlock Index identifying the diagonal block this contribution belongs to.
  ///   The first call with a given @p iBlock fixes the size of `R_ii_[iBlock]`; later
  ///   calls must provide a matching `A_diag.cols()`.
  /// @param A_diag Block-local Jacobian columns for this @c iBlock. `A_diag.cols()`
  ///   defines the per-block unknown count.
  /// @param A_common Jacobian columns for the shared unknowns. Must have
  ///   `n_common` columns.
  /// @param b Residual vector with `A_diag.rows()` entries.
  /// @post `R_ii_[iBlock]`, `R_in_[iBlock]`, `R_nn_`, `y_i_[iBlock]`, and `y_n_` are
  ///   updated. The contents of @p A_diag, @p A_common, and @p b are arbitrary on return.
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

  /// Retrieves the full dense result `[x_0; x_1; ...; x_{B-1}; x_n]`.
  [[nodiscard]] VectorType x_dense() const;

  /// Retrieves the answer for a single diagonal block @p iBlock.
  /// @note Re-runs the back-substitution for the final ("common") rows on each call;
  ///   batch-retrieve via `x_dense()` if calling for many blocks.
  [[nodiscard]] VectorType x_i(size_t iBlock) const;

  /// Retrieves the answer for just the final "common" rows.
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

  /// Components of R defined by `Q*A = (R; 0)` with the block-arrowhead layout
  /// \code
  ///   [ R_ii[0]                          R_in[0] ]
  ///   [            R_ii[1]               R_in[1] ]
  ///   [                         R_ii[2]  R_in[2] ]
  ///   [                                  R_nn    ]
  /// \endcode
  std::deque<MatrixType> R_ii_;
  std::deque<MatrixType> R_in_;
  MatrixType R_nn_;

  /// Transformed right-hand side defined by `Q*b = y`, laid out as
  /// `[ y_i_[0]; y_i_[1]; ...; y_n_ ]`.
  std::deque<VectorType> y_i_;
  VectorType y_n_;
};

/// Online Householder QR solver for band-plus-arrowhead matrices.
///
/// The left-hand part of the matrix is required to have a limited column-bandwidth, and
/// a "common" section is permitted on the right, like this:
/// \code
///        ...b a n d e d...     common
///     [ a_11                   a_15 ]
///     [ a_21  a_22             a_25 ]
///     [ a_31  a_32             a_35 ]
///     [       a_42  a_43       a_45 ]
///     [             a_53  a_54 a_55 ]
///     [                   a_64 a_65 ]
/// \endcode
/// Here "bandwidth" is in _columns_; the matrix above has a bandwidth of 2 because no
/// row spans more than two columns (excluding the common section). The common section
/// holds parameters shared by every row.
///
/// While this seems specialized, it maps to a very common class of problems where the
/// band structure represents a time dimension (smoothness constraints tend to connect
/// adjacent timepoints) and the common parameters are shared across time. For example:
///   - body tracking: band structure maps to smoothness constraints between adjacent
///     frames, while common parameters control body scale.
///   - camera calibration: band structure maps to smoothness constraints between
///     adjacent frames of the extrinsics, while common parameters refer to intrinsics.
///
/// ### Usage contract
/// 1. Construct with `n_band` (number of banded unknowns), `n_common` (shared unknowns),
///    `bandwidth`, and optional `lambda` damping.
/// 2. Add contributions via `add()`/`addMutating()`. Pass an `iCol_offset` indicating
///    where the contribution's left-most banded column lives in the global numbering.
///    Contributions touching only the common columns use the no-offset overload.
/// 3. Optionally use `zeroBandedPart()` to parallelize the banded sweep over disjoint
///    column ranges; the matching `addMutating(A_common, b)` call must be serialized.
/// 4. Retrieve results via `x_dense()` (and inspect via `R_dense()`/`y_dense()`).
/// 5. Call `reset()` to discard accumulated state and reuse the solver.
template <typename T>
class OnlineBandedHouseholderQR {
 public:
  using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  /// Constructs a banded solver.
  /// @param n_band Number of banded (per-time) unknowns.
  /// @param n_common Number of common unknowns (the right-hand "arrowhead").
  /// @param bandwidth Maximum number of banded columns any single row may touch.
  /// @param lambda LM-style damping; emulates the augmented system
  ///   `| [lambda*I; A] [x] - [0; b] |^2` without materializing the extra rows.
  explicit OnlineBandedHouseholderQR(
      Eigen::Index n_band,
      Eigen::Index n_common,
      Eigen::Index bandwidth,
      T lambda = T(0));

  /// Copying overload of `addMutating()` that takes both banded and common contributions.
  void add(size_t iCol_offset, MatrixType A_band, MatrixType A_common, VectorType b) {
    addMutating(iCol_offset, A_band, A_common, b);
  }

  /// Copying overload for contributions that only touch the common columns.
  void add(MatrixType A_common, VectorType b) {
    addMutating(A_common, b);
  }

  /// In-place addition; convenience wrapper that wraps the matrices in
  /// `ColumnIndexedMatrix` views without remapping.
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

  /// In-place addition for contributions that only touch the common columns.
  void addMutating(Eigen::Ref<MatrixType> A_common, Eigen::Ref<VectorType> b) {
    addMutating(ColumnIndexedMatrix<MatrixType>(A_common), b);
  }

  /// In-place addition that accepts column-remapped views, so callers can pass arbitrary
  /// subsets of the columns in @p A_band and @p A_common.
  ///
  /// @param iCol_offset Global column index of `A_band`'s left-most column. Must satisfy
  ///   `iCol_offset + A_band.cols() <= n_band()`.
  /// @param A_band Banded contribution. `A_band.cols()` must be at most `bandwidth()`.
  /// @param A_common Common contribution. `A_common.cols()` must equal `n_common()`.
  /// @param b Residual vector with `A_band.rows()` entries.
  /// @post Internal `R_band_`, `R_common_`, and `y_` are updated. The contents of
  ///   @p A_band, @p A_common, and @p b are arbitrary on return.
  void addMutating(
      Eigen::Index iCol_offset,
      ColumnIndexedMatrix<MatrixType> A_band,
      ColumnIndexedMatrix<MatrixType> A_common,
      Eigen::Ref<VectorType> b);

  /// Common-only addition overload. Use when a contribution touches no banded columns.
  void addMutating(ColumnIndexedMatrix<MatrixType> A_common, Eigen::Ref<VectorType> b);

  /// Zeros out just the banded part of @p A_band by applying Householder rotations.
  /// @p A_common is updated in place but will still have nonzero entries on return, so
  /// after calling this you should follow up with `addMutating(A_common, b)` to fold the
  /// remaining common contribution into the factorization.
  ///
  /// Why is this useful? If you are careful, you can call `zeroBandedPart` on multiple
  /// non-overlapping segments of the banded matrix in parallel; only the `A_common`
  /// processing must be serialized. Because the number of common variables is usually
  /// much smaller than the number of banded variables, this can provide a substantial
  /// speedup in practice.
  ///
  /// @pre @p iCol_offset, @p A_band, @p A_common, @p b satisfy the same shape rules as
  ///   `addMutating()`.
  /// @pre The column range `[iCol_offset, iCol_offset + A_band.cols())` does not overlap
  ///   any concurrently in-flight `zeroBandedPart` call on the same solver.
  /// @post `R_band_` is updated; @p A_band is zeroed; @p A_common and @p b are mutated
  ///   and must be passed to the common-only `addMutating()` to finish the contribution.
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

  /// Retrieves the full dense result vector of length `n_band() + n_common()`.
  [[nodiscard]] VectorType x_dense() const;
  /// @return R as a dense matrix of size `(n_band+n_common) x (n_band+n_common)`,
  ///   reconstructed from the banded and common storage.
  [[nodiscard]] MatrixType R_dense() const;
  /// @return The transformed right-hand side `Q * b`.
  [[nodiscard]] MatrixType y_dense() const {
    return y_;
  }

  /// @return `A^T * b` for the data accumulated so far.
  [[nodiscard]] VectorType At_times_b() const;

  /// Discards accumulated state, keeping the existing dimensions and damping.
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
  /// Initializes the diagonal of R with @c lambda_ to bake LM damping into the
  /// factorization without materializing the `lambda*I` augmentation.
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

  /// R defined by `Q*A = (R; 0)`, split between the banded and common columns:
  /// \code
  ///   [ *  *  *        *  * ]
  ///   [    *  *  *     *  * ]
  ///   [       *  *  *  *  * ]
  ///   [          *  *  *  * ]
  ///   [             *  *  * ]
  ///   [                *  * ]
  ///   [                   * ]
  ///     ^^^^^^^^^      ^^^^
  ///     R_band_      R_common_
  /// \endcode
  MatrixType R_common_;

  /// Banded part of R using compact diagonal storage of shape `(bandwidth, n_band)`,
  /// representing the upper-triangular band:
  /// \code
  ///   [ r_11 r_12           ]
  ///   [      r_22 r_23      ]
  ///   [           r_33 r_34 ]
  /// \endcode
  /// See `R_band_entry()` for the row/col -> storage index mapping.
  MatrixType R_band_;

  /// Transformed right-hand side `Q * b`.
  VectorType y_;
};

} // namespace momentum
