/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/common/aligned.h>

#include <Eigen/Core>

#include <vector>

namespace momentum {

/// A dense column-major matrix whose underlying storage capacity grows monotonically and is
/// reused across resizes, unlike `Eigen::Matrix<T, Dynamic, Dynamic>` which reallocates on
/// every `resize()` / `conservativeResize()` call.
///
/// Motivation: solver inner loops repeatedly build matrices (e.g. blockwise Jacobians) whose
/// logical dimensions vary slightly from iteration to iteration. Using a plain Eigen matrix
/// triggers a heap allocation each time the size changes; this class amortizes those costs by
/// only growing the backing buffer when the requested element count exceeds the current
/// capacity. Storage shrinks only when the object is destroyed.
///
/// Capacity vs. logical size: `rows()` / `cols()` report the logical size requested by the
/// last `resizeAndSetZero()`. The backing `std::vector` capacity may be larger and is never
/// returned to the allocator while the object lives.
///
/// Layout: data is stored contiguously in column-major order (matching Eigen's default) and
/// is aligned to `kAlignment` bytes so that the returned `Eigen::Map<..., Eigen::Aligned128>`
/// can use aligned SIMD loads/stores.
///
/// Thread safety: not thread-safe. Concurrent calls to any non-const method, or a non-const
/// method concurrent with any other access, are data races. Multiple readers via `const`
/// accessors are safe provided no writer is active.
template <typename T>
class ResizeableMatrix {
 public:
  /// Alignment in bytes for the underlying storage. Chosen to satisfy `Eigen::Aligned128`
  /// so the `Eigen::Map` accessors below can perform aligned SIMD loads/stores.
  static constexpr std::size_t kAlignment = 128;

  using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  ResizeableMatrix() = default;

  /// Constructs a matrix of the given logical size and zero-initializes all entries.
  /// @param rows Number of rows in the logical matrix.
  /// @param cols Number of columns in the logical matrix; defaults to 1 (column vector).
  explicit ResizeableMatrix(Eigen::Index rows, Eigen::Index cols = 1) {
    resizeAndSetZero(rows, cols);
  }

  /// Sets the logical size to `rows x cols` and zeroes every element.
  ///
  /// The backing buffer is reused if its current capacity is sufficient; otherwise it grows.
  /// Capacity never shrinks. Any `Eigen::Map` previously returned by `mat()` / `vec()` is
  /// invalidated by this call (storage may have been reallocated, and existing contents are
  /// overwritten with zeros regardless).
  ///
  /// @param rows Number of rows in the new logical matrix.
  /// @param cols Number of columns in the new logical matrix; defaults to 1.
  /// @post `this->rows() == rows && this->cols() == cols` and every entry is `T(0)`.
  void resizeAndSetZero(Eigen::Index rows, Eigen::Index cols = 1) {
    rows_ = rows;
    cols_ = cols;

    // clear() + resize() relies on std::vector preserving capacity across clear(); the
    // resize() therefore only reallocates when rows*cols exceeds the previous capacity.
    // TODO: this default-inserts (zero-initializes) `rows*cols` elements every call, which
    // is wasted work when the caller is about to overwrite all of them. Consider adding a
    // separate `resizeWithoutZero()` for that case.
    data_.clear();
    data_.resize(rows * cols);
  }

  [[nodiscard]] Eigen::Index rows() const {
    return rows_;
  }
  [[nodiscard]] Eigen::Index cols() const {
    return cols_;
  }

  /// Returns an aligned, mutable `Eigen::Map` view of the logical `rows() x cols()` matrix.
  /// The map is invalidated by any subsequent call to `resizeAndSetZero()`.
  Eigen::Map<MatrixType, Eigen::Aligned128> mat() {
    return Eigen::Map<MatrixType, Eigen::Aligned128>(data_.data(), rows_, cols_);
  }

  /// Const overload of `mat()`.
  Eigen::Map<const MatrixType, Eigen::Aligned128> mat() const {
    return Eigen::Map<const MatrixType, Eigen::Aligned128>(data_.data(), rows_, cols_);
  }

  /// Returns an aligned, mutable `Eigen::Map` view interpreting the storage as a length-`rows()`
  /// column vector. The `cols()` value is ignored; intended for the single-column case.
  /// The map is invalidated by any subsequent call to `resizeAndSetZero()`.
  /// TODO: `vec()` ignores `cols_` and only exposes the first column when `cols_ > 1`. Either
  /// document this contract loudly or assert `cols_ == 1`.
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>, Eigen::Aligned128> vec() {
    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>, Eigen::Aligned128>(data_.data(), rows_);
  }

  /// Const overload of `vec()`.
  Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, Eigen::Aligned128> vec() const {
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, Eigen::Aligned128>(
        data_.data(), rows_);
  }

 private:
  Eigen::Index rows_ = 0;
  Eigen::Index cols_ = 0;
  std::vector<T, AlignedAllocator<T, kAlignment>> data_;
};

} // namespace momentum
