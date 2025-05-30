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

/// SIMD version of the generalized loss function.
template <typename T>
class SimdGeneralizedLossT : public GeneralizedLossT<T> {
 public:
  using Base = GeneralizedLossT<T>;

  /// Constructs a SIMD generalized loss function
  ///
  /// @param a Alpha parameter controlling the shape of the loss function
  /// @param c Scale parameter controlling the width of the quadratic region
  explicit SimdGeneralizedLossT(const T& a = Base::kL2, const T& c = T(1));

  /// Computes the loss value for a given squared error
  ///
  /// @param sqrError Squared error term (SIMD packet)
  /// @return Loss value as a SIMD packet
  [[nodiscard]] Packet<T> value(const Packet<T>& sqrError) const;

  /// Computes the derivative of the loss function
  ///
  /// @param sqrError Squared error term (SIMD packet)
  /// @return Derivative value as a SIMD packet
  [[nodiscard]] Packet<T> deriv(const Packet<T>& sqrError) const;
};

} // namespace momentum
