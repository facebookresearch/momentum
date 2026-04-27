/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/math/simd_generalized_loss.h"

#include "momentum/common/log.h"

#include <drjit/math.h>

#include <cmath>

namespace momentum {

namespace {

// SIMD mirrors of the scalar helpers in generalized_loss.cpp. Each takes a Packet of squared
// errors and broadcasts the scalar `invC2` over the lanes via DrJit's overloaded operators.
// Math intrinsics are dispatched through `drjit::sqrt/log/exp/pow` to vectorize across the packet
// instead of falling back to per-lane `std::*` calls.

// L2: returns invC2 * sqrError (no 0.5 factor — kept separate so the result can be reused as the
// inner term of the L1/Cauchy/Welsch/General formulas).
template <typename T>
Packet<T> L2Loss(const Packet<T>& sqrError, T invC2) {
  return sqrError * invC2;
}

// d/dsqrError of L2Loss is the constant invC2 — broadcast implicitly to a Packet at the call site.
template <typename T>
Packet<T> derivL2Loss(const Packet<T>& /* sqrError */, T invC2) {
  return invC2;
}

// L1 (pseudo-Huber, alpha = 1): sqrt(invC2 * sqrError + 1) - 1.
// The trailing `- 1` is intentionally kept so value() at sqrError=0 is exactly 0.
template <typename T>
Packet<T> L1Loss(const Packet<T>& sqrError, T invC2) {
  return drjit::sqrt(L2Loss(sqrError, invC2) + T(1)) - T(1);
}

template <typename T>
Packet<T> derivL1Loss(const Packet<T>& sqrError, T invC2) {
  return T(0.5) * invC2 / drjit::sqrt(L2Loss(sqrError, invC2) + T(1));
}

// Cauchy / Lorentzian (alpha = 0): log(0.5 * invC2 * sqrError + 1).
template <typename T>
Packet<T> CauchyLoss(const Packet<T>& sqrError, T invC2) {
  return drjit::log(T(0.5) * L2Loss(sqrError, invC2) + T(1));
}

template <typename T>
Packet<T> derivCauchyLoss(const Packet<T>& sqrError, T invC2) {
  return invC2 / (invC2 * sqrError + T(2));
}

// Welsch (alpha -> -inf limit): 1 - exp(-0.5 * invC2 * sqrError). Bounded in [0, 1).
template <typename T>
Packet<T> WelschLoss(const Packet<T>& sqrError, T invC2) {
  return T(1) - drjit::exp(T(-0.5) * L2Loss(sqrError, invC2));
}

template <typename T>
Packet<T> derivWelschLoss(const Packet<T>& sqrError, T invC2) {
  return (T(0.5) * invC2) * drjit::exp(T(-0.5) * L2Loss(sqrError, invC2));
}

} // namespace

template <typename T>
SimdGeneralizedLossT<T>::SimdGeneralizedLossT(const T& a, const T& c) : Base(a, c) {}

// Branching is on the scalar `lossType_` cached at construction (Base classifies alpha into one
// of L2/L1/Cauchy/Welsch/General). All packet lanes follow the same branch — no per-lane mask or
// `drjit::select` is needed because alpha is uniform across the packet.
template <typename T>
Packet<T> SimdGeneralizedLossT<T>::value(const Packet<T>& sqrError) const {
  switch (this->lossType_) {
    case Base::LossType::L2:
      return L2Loss(sqrError, this->invC2_);
    case Base::LossType::L1:
      return L1Loss(sqrError, this->invC2_);
    case Base::LossType::Cauchy:
      return CauchyLoss(sqrError, this->invC2_);
    case Base::LossType::Welsch:
      // Welsch limit is taken when alpha <= kWelsch (alpha == -inf is handled by the same branch
      // since Base classifies any sufficiently negative alpha as Welsch).
      return WelschLoss(sqrError, this->invC2_);
    case Base::LossType::General:
      // General Barron loss for alpha not near 0, 1, 2, or -inf. The packet `pow` is the slow
      // path; the special cases above exist precisely to avoid it. The scalar prefactor
      // `|alpha - 2| / alpha` is computed once outside the packet expression.
      return (drjit::pow(
                  L2Loss(sqrError, this->invC2_) / std::fabs(this->alpha_ - T(2)) + T(1),
                  T(0.5) * this->alpha_) -
              T(1)) *
          (std::fabs(this->alpha_ - T(2)) / this->alpha_);
    default: {
      // TODO: returning a default-constructed Packet here may yield uninitialized lanes; consider
      // returning drjit::zeros<Packet<T>>() or asserting in debug builds so callers don't silently
      // propagate garbage on a corrupted lossType_.
      MT_LOGE("Invalid lossType_ ({})", static_cast<int>(this->lossType_));
      return {};
    }
  }
}

// Same lane-uniform switch structure as value(); see notes there. Returns d(loss)/d(sqrError) so
// callers can chain-rule into the Jacobian of the underlying residual.
template <typename T>
Packet<T> SimdGeneralizedLossT<T>::deriv(const Packet<T>& sqrError) const {
  switch (this->lossType_) {
    case Base::LossType::L2:
      return derivL2Loss(sqrError, this->invC2_);
    case Base::LossType::L1:
      return derivL1Loss(sqrError, this->invC2_);
    case Base::LossType::Cauchy:
      return derivCauchyLoss(sqrError, this->invC2_);
    case Base::LossType::Welsch:
      return derivWelschLoss(sqrError, this->invC2_);
    case Base::LossType::General:
      // General Barron derivative; uses packet `pow` and is the slow path.
      return (T(0.5) * this->invC2_) *
          drjit::pow(
                 L2Loss(sqrError, this->invC2_) / std::fabs(this->alpha_ - T(2)) + T(1),
                 T(0.5) * this->alpha_ - T(1));
    default: {
      // TODO: see value() — same uninitialized-lane concern on the default branch.
      MT_LOGE("Invalid lossType_ ({})", static_cast<int>(this->lossType_));
      return {};
    }
  }
}

template class SimdGeneralizedLossT<float>;
template class SimdGeneralizedLossT<double>;

} // namespace momentum
