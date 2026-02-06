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

/// A matrix class that can be resized without reallocating memory when shrinking.
///
/// The native Eigen matrix type reallocates the matrix every time you call
/// either resize() or conservativeResize(), which means there is no way to
/// have a matrix that reuses memory. This is inconvenient, because there are
/// definitely cases where we want to be able to reuse the same memory over
/// and over for slightly different matrix sizes (for example, if we are computing
/// our Jacobian blockwise).
template <typename T>
class ResizeableMatrix {
 public:
  /// Alignment in bytes for SIMD and cache efficiency
  static constexpr std::size_t kAlignment = 128;

  using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  ResizeableMatrix() = default;
  explicit ResizeableMatrix(Eigen::Index rows, Eigen::Index cols = 1) {
    resizeAndSetZero(rows, cols);
  }

  void resizeAndSetZero(Eigen::Index rows, Eigen::Index cols = 1) {
    rows_ = rows;
    cols_ = cols;

    // This will not reallocate unless needed.
    data_.clear();
    data_.resize(rows * cols);
  }

  [[nodiscard]] Eigen::Index rows() const {
    return rows_;
  }
  [[nodiscard]] Eigen::Index cols() const {
    return cols_;
  }

  Eigen::Map<MatrixType, Eigen::Aligned128> mat() {
    return Eigen::Map<MatrixType, Eigen::Aligned128>(data_.data(), rows_, cols_);
  }
  Eigen::Map<const MatrixType, Eigen::Aligned128> mat() const {
    return Eigen::Map<const MatrixType, Eigen::Aligned128>(data_.data(), rows_, cols_);
  }

  /// Vector accessor for single-column matrices
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>, Eigen::Aligned128> vec() {
    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>, Eigen::Aligned128>(data_.data(), rows_);
  }
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
