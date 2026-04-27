/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/math/generalized_loss.h"

#include "momentum/common/checks.h"
#include "momentum/common/log.h"

#include <cmath>

namespace momentum {

namespace {

// L2 loss (alpha = 2):
//   f(s) = s / c^2
//   f'(s) = 1 / c^2
// Note: the canonical Barron form has a 1/2 factor; it is intentionally omitted here so this
// helper can be reused inside the other closed forms (L1/Cauchy/Welsch) without re-multiplying.
template <typename T>
T L2Loss(const T& sqrError, const T& invC2) {
  return sqrError * invC2;
}

template <typename T>
T derivL2Loss(const T& /* sqrError */, const T& invC2) {
  return invC2;
}

// L1 / Pseudo-Huber loss (alpha = 1):
//   f(s) = sqrt(s/c^2 + 1) - 1
//   f'(s) = (1/2) * (1/c^2) / sqrt(s/c^2 + 1)
// The "-1" offset is kept (rather than dropped as a constant) so that f(0) = 0 and the value
// itself is interpretable as a loss magnitude.
template <typename T>
T L1Loss(const T& sqrError, const T& invC2) {
  return std::sqrt(L2Loss(sqrError, invC2) + T(1)) - T(1);
}

template <typename T>
T derivL1Loss(const T& sqrError, const T& invC2) {
  return T(0.5) * invC2 / std::sqrt(L2Loss(sqrError, invC2) + T(1));
}

// Cauchy / Lorentzian loss (alpha = 0, the limit):
//   f(s) = log(1 + (1/2) * s/c^2)
//   f'(s) = 1 / (s + 2*c^2) = (1/c^2) / (s/c^2 + 2)
// The argument to log is always >= 1 for sqrError >= 0, so log is well-defined.
template <typename T>
T CauchyLoss(const T& sqrError, const T& invC2) {
  return std::log(T(0.5) * L2Loss(sqrError, invC2) + T(1));
}

template <typename T>
T derivCauchyLoss(const T& sqrError, const T& invC2) {
  return invC2 / (invC2 * sqrError + T(2));
}

// Welsch / Leclerc loss (alpha -> -inf, the limit):
//   f(s) = 1 - exp(-(1/2) * s/c^2)
//   f'(s) = (1/2) * (1/c^2) * exp(-(1/2) * s/c^2)
// Bounded in [0, 1); derivative decays to 0 for large residuals — outliers contribute essentially
// nothing to the gradient. exp argument is <= 0, so no overflow.
template <typename T>
T WelschLoss(const T& sqrError, const T& invC2) {
  return T(1) - std::exp(T(-0.5) * L2Loss(sqrError, invC2));
}

template <typename T>
T derivWelschLoss(const T& sqrError, const T& invC2) {
  return T(0.5) * invC2 * std::exp(T(-0.5) * L2Loss(sqrError, invC2));
}

} // namespace

template <typename T>
GeneralizedLossT<T>::GeneralizedLossT(const T& a, const T& c) : alpha_(a), invC2_(T(1) / (c * c)) {
  MT_CHECK(c > 0, "Parameter c should be positive but received {}", c);

  // Snap alpha to a closed-form branch when it is within kEps of a special value, both for
  // numerical stability (the General formula has a removable singularity at alpha=2 from the
  // |alpha - 2| factor, and a 0/0 form at alpha=0) and for speed (closed forms avoid pow/log).
  // Note: kWelsch = -1e-9 is a sentinel for the alpha -> -inf limit; the branch below routes
  // *any* alpha <= kWelsch to the Welsch closed form.
  // TODO: This Welsch branch is too greedy — finite negative alphas like -2 (Geman-McClure),
  // which the header documents as a supported case, are silently replaced by the alpha = -inf
  // Welsch limit instead of being evaluated with the General formula. Tighten the predicate to
  // something like `alpha_ < some_large_negative_threshold` (or check for -infinity explicitly)
  // so that finite negative alphas fall through to LossType::General.
  if (alpha_ >= kL2 - kEps && alpha_ <= kL2 + kEps) {
    lossType_ = LossType::L2;
  } else if (alpha_ >= kL1 - kEps && alpha_ <= kL1 + kEps) {
    lossType_ = LossType::L1;
  } else if (alpha_ >= kCauchy - kEps && alpha_ <= kCauchy + kEps) {
    lossType_ = LossType::Cauchy;
  } else if (alpha_ <= kWelsch) {
    lossType_ = LossType::Welsch;
  } else {
    lossType_ = LossType::General;
  }
}

template <typename T>
T GeneralizedLossT<T>::value(const T& sqrError) const {
  switch (lossType_) {
    case LossType::L2:
      return L2Loss(sqrError, invC2_);
    case LossType::L1:
      return L1Loss(sqrError, invC2_);
    case LossType::Cauchy:
      return CauchyLoss(sqrError, invC2_);
    case LossType::Welsch:
      return WelschLoss(sqrError, invC2_);
    case LossType::General:
      // Barron's general form (eq. 1 of arXiv:1701.03077), valid for finite alpha not in
      // {2, 0}; the |alpha - 2| / alpha factor recovers the L2/Cauchy/Welsch limits as
      // alpha -> 2, 0, -inf respectively. Slower than the closed forms above due to pow().
      //   f(s) = (|alpha-2| / alpha) * ( (s/(c^2 * |alpha-2|) + 1)^(alpha/2) - 1 )
      // Singularities at alpha == 2 and alpha == 0 are avoided by the kEps snap in the ctor.
      return (std::pow(L2Loss(sqrError, invC2_) / std::abs(alpha_ - T(2)) + T(1), T(0.5) * alpha_) -
              T(1)) *
          std::abs(alpha_ - T(2)) / alpha_;
    default: {
      MT_LOGE("Invalid lossType_ ({})", static_cast<int>(lossType_));
      return {};
    }
  }
}

template <typename T>
T GeneralizedLossT<T>::deriv(const T& sqrError) const {
  switch (lossType_) {
    case LossType::L2:
      return derivL2Loss(sqrError, invC2_);
    case LossType::L1:
      return derivL1Loss(sqrError, invC2_);
    case LossType::Cauchy:
      return derivCauchyLoss(sqrError, invC2_);
    case LossType::Welsch:
      return derivWelschLoss(sqrError, invC2_);
    case LossType::General:
      // Derivative of Barron's general form w.r.t. s (the squared error):
      //   f'(s) = (1 / (2 c^2)) * ( s / (c^2 * |alpha-2|) + 1 )^(alpha/2 - 1)
      // The |alpha-2| / alpha prefactor in f(s) cancels under differentiation, so it does not
      // appear here. Slower than the closed forms above due to pow().
      return T(0.5) * invC2_ *
          std::pow(
                 L2Loss(sqrError, invC2_) / std::abs(alpha_ - T(2)) + T(1), T(0.5) * alpha_ - T(1));
    default: {
      MT_LOGE("Invalid lossType_ ({})", static_cast<int>(lossType_));
      return {};
    }
  }
}

template class GeneralizedLossT<float>;
template class GeneralizedLossT<double>;

} // namespace momentum
