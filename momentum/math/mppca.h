/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/math/utility.h>

namespace momentum {

/// Mixture of Probabilistic Principal Component Analysis (MPPCA)
/// Based on: http://www.miketipping.com/papers/met-mppca.pdf
///
/// @tparam T Scalar type (float or double)
template <typename T>
struct MppcaT {
  /// Dimension of data space
  size_t d = 0;

  /// Number of mixture components
  size_t p = 0;

  /// Parameter names (should match dimension d)
  std::vector<std::string> names;

  /// Mean vectors (p×d matrix)
  Eigen::MatrixX<T> mu;

  /// Inverse covariance matrices
  std::vector<Eigen::MatrixX<T>> Cinv;

  /// Matrices for efficient computation
  std::vector<Eigen::MatrixX<T>> L;

  /// Precomputed responsibility terms
  Eigen::VectorX<T> Rpre;

  /// Sets model parameters and precomputes matrices
  ///
  /// @param[in] pi Mixture weights
  /// @param[in] mmu Mean vectors
  /// @param[in] W Principal axes matrices
  /// @param[in] sigma2 Noise variances
  void set(
      const VectorX<T>& pi,
      const MatrixX<T>& mmu,
      std::span<const MatrixX<T>> W,
      const VectorX<T>& sigma2);

  /// Converts to a different scalar type
  ///
  /// @tparam T2 Target scalar type
  /// @return Converted MPPCA model
  template <typename T2>
  [[nodiscard]] MppcaT<T2> cast() const;

  /// Checks approximate equality with another model
  ///
  /// @param[in] mppcaT Model to compare with
  /// @return true if approximately equal
  [[nodiscard]] bool isApprox(const MppcaT<T>& mppcaT) const;
};

} // namespace momentum
