/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/math/generalized_loss.h>
#include <momentum/simd/simd.h>

namespace momentum {

/// SIMD (vectorized) variant of `GeneralizedLossT` that operates on DrJit `Packet<T>` lanes
/// instead of individual scalars. See `GeneralizedLossT` for the full mathematical description,
/// the meaning of @p alpha and @p c, and the alpha-value reference (L2, L1/pseudo-Huber, Cauchy,
/// Geman-McClure, Welsch).
///
/// Inherits the scalar `value()` / `deriv()` overloads and the cached `lossType_` / `invC2_`
/// state from the base class. Branching on the loss type is uniform across all packet lanes
/// (alpha is fixed at construction), so no per-lane masking is needed in the implementation.
///
/// @par When to use
/// Use this class inside SIMD error functions that process multiple constraints in a single
/// packet (e.g., `Simd*ErrorFunction`). For scalar error functions or one-off evaluations,
/// prefer the base `GeneralizedLossT` directly.
///
/// @par Thread safety
/// Instances are immutable after construction (all state is `const` in the base class), so
/// concurrent reads from multiple threads are safe. Each evaluation is stateless.
template <typename T>
class SimdGeneralizedLossT : public GeneralizedLossT<T> {
 public:
  using Base = GeneralizedLossT<T>;

  /// Constructs a SIMD generalized loss with shape @p a and scale @p c.
  ///
  /// @param a Shape parameter alpha. See `GeneralizedLossT` for the alpha-value reference and
  ///   the special-case constants (`kL2`, `kL1`, `kCauchy`, `kWelsch`).
  /// @param c Scale parameter. Must be > 0.
  explicit SimdGeneralizedLossT(const T& a = Base::kL2, const T& c = T(1));

  /// Returns the loss value for each lane of @p sqrError.
  /// @param sqrError Packet of non-negative squared residuals (one per lane).
  /// @return Packet of loss values, one per input lane.
  [[nodiscard]] Packet<T> value(const Packet<T>& sqrError) const;

  /// Returns d(loss)/d(sqrError) for each lane of @p sqrError. Used to chain-rule the loss
  /// into the Jacobian of the underlying residual.
  /// @param sqrError Packet of non-negative squared residuals (one per lane).
  /// @return Packet of derivatives, one per input lane.
  [[nodiscard]] Packet<T> deriv(const Packet<T>& sqrError) const;
};

} // namespace momentum
