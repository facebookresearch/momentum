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
/// This loss transforms a squared error value with parameters alpha and c.
/// Alpha can be approximately thought of as the norm: alpha=2 is the squared L2 norm loss; alpha=1
/// is a soft L1 norm or the L2-L1/pseudo-Huber loss. Alpha controls the effect of "outliers" (ie.
/// "robust"). Smaller alpha reduces the contribution of large errors so they don't affect the
/// result as much. c scales the gradient with respect to the squared error term inversely.
/// This does not implement solving for alpha. It assumes alpha and c are given as input.
template <typename T>
class GeneralizedLossT {
 public:
  // Special case alpha values that are either at singularity or would simplify computation
  static constexpr T kL2 = T(2);
  static constexpr T kL1 = T(1);
  static constexpr T kCauchy = T(0);
  static constexpr T kWelsch = -1e-9; // any number smaller is considered -inf

  // Alpha can be [-inf, inf] in theory, but it will be numerically unstable if
  // alpha is too large, eg. >100 (large alpha not needed in practice).
  // Similarly, c scales the gradient inversely and quadratically. So too small a value for c will
  // lead to instability.
  // Calculations with log and exp usually require double precision.
  GeneralizedLossT(const T& a = kL2, const T& c = T(1));

  [[nodiscard]] T value(const T& sqrError) const;

  /// Derivative of the loss with respective to the input squared error.
  /// This effectively scales the gradient of what's being squared.
  [[nodiscard]] T deriv(const T& sqrError) const;

 protected:
  enum class LossType {
    L1,
    L2,
    Cauchy,
    Welsch,
    General,
  };

  // A small threshold for special values
  static constexpr T kEps = 1e-9;

  const T alpha_;

  // store 1/c^2 instead of c as a small optimization
  const T invC2_;

  LossType lossType_;
};
} // namespace momentum
