/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace momentum {

/// Implementation of "A General and Adaptive Robust Loss Function"
/// (https://arxiv.org/abs/1701.03077).
///
/// This loss transforms a squared error value with parameters @p alpha and @p c.
/// @p alpha can be approximately thought of as the norm: alpha=2 is the squared L2 norm loss;
/// alpha=1 is a soft L1 norm or the L2-L1/pseudo-Huber loss. @p alpha controls the effect of
/// "outliers" (i.e., "robust"). Smaller @p alpha reduces the contribution of large errors so they
/// don't affect the result as much. @p c scales the gradient with respect to the squared error
/// term inversely.
///
/// This does not implement solving for @p alpha. It assumes @p alpha and @p c are given as input.
///
/// @par Alpha value reference
/// - `alpha = 2.0` (`kL2`): L2 (quadratic). Standard least-squares. Fastest convergence, sensitive
///   to outliers.
/// - `alpha = 1.0` (`kL1`): Pseudo-Huber. Smooth approximation to Huber. Robust to moderate
///   outliers.
/// - `alpha = 0.0` (`kCauchy`): Cauchy (Lorentzian). Very robust to outliers. Much slower
///   convergence.
/// - `alpha = -2.0`: Geman-McClure. Extremely robust. Bounded influence — large outliers are
///   essentially ignored.
/// - `alpha = -inf` (`kWelsch`): Welsch. Limit case. Maximum robustness, slowest convergence.
///
/// The @p c parameter controls the transition scale: residuals << c behave like L2, residuals >> c
/// are downweighted according to the loss shape.
///
/// In the Gauss-Newton solver, the loss function modifies both the residual and Jacobian via the
/// chain rule. The `SimdGeneralizedLoss` class handles this vectorized computation for SIMD error
/// functions. Generic error functions apply the loss via the `JointErrorFunctionT` base class
/// automatically — subclasses implement `evalFunction()` with raw residuals.
template <typename T>
class GeneralizedLossT {
 public:
  /// Special-case alpha values that are either at a singularity in the general formula or simplify
  /// to a closed-form computation.
  static constexpr T kL2 = T(2);
  static constexpr T kL1 = T(1);
  static constexpr T kCauchy = T(0);
  /// Sentinel for the Welsch (alpha = -inf) limit; any alpha less than this is treated as -inf.
  static constexpr T kWelsch = -1e-9;

  /// Constructs a generalized loss with shape @p a and scale @p c.
  ///
  /// @param a Shape parameter alpha. Theoretically in [-inf, inf], but values with very large
  ///   magnitude (e.g., > 100) are numerically unstable and not needed in practice.
  /// @param c Scale parameter. Must be > 0. Very small values are unstable because the gradient
  ///   scales with 1/c^2.
  ///
  /// @note Internal calculations involving `log`/`exp` are performed in double precision for
  ///   numerical stability.
  GeneralizedLossT(const T& a = kL2, const T& c = T(1));

  /// Returns the loss value for a given squared error.
  /// @param sqrError Non-negative squared residual.
  [[nodiscard]] T value(const T& sqrError) const;

  /// Derivative of the loss with respect to the input squared error.
  /// This effectively scales the gradient of what's being squared.
  [[nodiscard]] T deriv(const T& sqrError) const;

  /// Returns true if this loss is an L2 loss (alpha ~= 2).
  /// Useful for fast-path optimizations that bypass value()/deriv() calls.
  [[nodiscard]] bool isL2() const {
    return lossType_ == LossType::L2;
  }

  /// Returns the stored 1/c^2 value. For default c=1, this is 1.0.
  [[nodiscard]] T invC2() const {
    return invC2_;
  }

 protected:
  enum class LossType {
    L1,
    L2,
    Cauchy,
    Welsch,
    General,
  };

  /// Small threshold used to detect proximity to special-case alpha values.
  static constexpr T kEps = 1e-9;

  const T alpha_;

  /// Stored as 1/c^2 (instead of c) as a small optimization, since the loss formulas use this
  /// quantity directly.
  const T invC2_;

  LossType lossType_;
};
} // namespace momentum
