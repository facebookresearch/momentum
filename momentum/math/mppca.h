/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/math/utility.h>

namespace momentum {

/// Mixture of Probabilistic Principal Component Analyzers (MPPCA).
///
/// Each component models the data as `x = W_c z + mu_c + epsilon`, where `z`
/// is a low-dimensional latent vector and `epsilon ~ N(0, sigma_c^2 I)` is
/// isotropic noise. The full mixture density is
/// `p(x) = sum_c pi_c N(x | mu_c, C_c)` with `C_c = W_c W_c^T + sigma_c^2 I`.
///
/// After `set()`, the model stores precomputed quantities so that the
/// per-component log-likelihood of a sample `x` can be evaluated as
/// `Rpre(c) - 0.5 * (x - mu.row(c))^T * Cinv[c] * (x - mu.row(c))`.
///
/// Based on: http://www.miketipping.com/papers/met-mppca.pdf
///
/// @tparam T Scalar type (float or double).
template <typename T>
struct MppcaT {
  /// Dimensionality of the data space `x`.
  size_t d = 0;

  /// Number of mixture components.
  size_t p = 0;

  /// Optional human-readable names for each of the `d` data dimensions.
  /// `set()` resizes this to `d` empty strings; callers may overwrite entries.
  std::vector<std::string> names;

  /// Per-component mean vectors stored as a `p x d` matrix; row `c` is `mu_c`.
  Eigen::MatrixX<T> mu;

  /// Per-component inverse covariance matrices `C_c^{-1}` of size `d x d`.
  /// `Cinv.size() == p` after `set()`.
  std::vector<Eigen::MatrixX<T>> Cinv;

  /// Per-component lower-triangular factors derived from the Cholesky-like
  /// decomposition of `C_c` (used as a whitening transform when sampling or
  /// evaluating the Mahalanobis term efficiently). `L.size() == p` after
  /// `set()`.
  std::vector<Eigen::MatrixX<T>> L;

  /// Per-component constant term of the log-likelihood:
  /// `Rpre(c) = log(pi_c) - 0.5 * log|C_c| - (d/2) * log(2*pi)`.
  /// Add the `-0.5 * Mahalanobis` term to obtain the log-density of `x` under
  /// component `c`. Length is `p` after `set()`.
  Eigen::VectorX<T> Rpre;

  /// Initializes the model from raw mixture parameters and precomputes
  /// `Cinv`, `L`, and `Rpre` for fast log-likelihood evaluation.
  ///
  /// @param[in] pi Mixture weights, length `p`. Must sum to 1 and be > 0.
  /// @param[in] mmu Per-component means as a `p x d` matrix (row `c` is `mu_c`).
  /// @param[in] W Per-component factor-loading matrices `W_c`; expected to be
  ///   of size `d x q_c` where `q_c < d` is the latent dimension. Length `p`.
  /// @param[in] sigma2 Per-component isotropic noise variances `sigma_c^2`,
  ///   length `p`. Must be strictly positive.
  /// @pre `pi.size() == p`, `mmu.rows() == p`, `W.size() == p`,
  ///   `sigma2.size() == p`, and `W[c].rows() == mmu.cols()` for all `c`.
  /// @post `d == mmu.cols()`, `p == sigma2.rows()`, and `Cinv`, `L`, `Rpre`,
  ///   `mu`, `names` are sized consistently.
  // TODO: Parameter is named `pi` here but `inPi` in the .cpp definition;
  //   align the names for clarity.
  // TODO: `pi` is consumed only to compute `Rpre` and is not retained as a
  //   member. Document this or expose `pi` directly so callers can recover
  //   mixture weights without inverting the precomputation.
  void set(
      const VectorX<T>& pi,
      const MatrixX<T>& mmu,
      momentum::span<const MatrixX<T>> W,
      const VectorX<T>& sigma2);

  /// Returns a copy of this model with all numeric data converted to scalar
  /// type `T2`. Names and dimensions are copied unchanged.
  ///
  /// @tparam T2 Target scalar type.
  /// @return Converted MPPCA model.
  template <typename T2>
  [[nodiscard]] MppcaT<T2> cast() const;

  /// Returns true if every member matches `mppcaT` exactly (`d`, `p`, `names`)
  /// or approximately (`mu`, `Cinv`, `L`, `Rpre`) using Eigen's `isApprox`
  /// default tolerance.
  ///
  /// @param[in] mppcaT Model to compare with.
  /// @return True if approximately equal.
  [[nodiscard]] bool isApprox(const MppcaT<T>& mppcaT) const;
};

} // namespace momentum
