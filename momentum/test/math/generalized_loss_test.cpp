/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/math/constants.h"
#include "momentum/math/fwd.h"
#include "momentum/math/generalized_loss.h"
#include "momentum/math/random.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>

using namespace momentum;

using Types = testing::Types<float, double>;

template <typename T>
struct GeneralizedLossTest : testing::Test {
  using Type = T;
  Random<> rng{12345};
};

TYPED_TEST_SUITE(GeneralizedLossTest, Types);

template <typename T>
T expectedGeneralBarronValue(T alpha, T c, T sqrError) {
  const T invC2 = T(1) / (c * c);
  const T absAlphaMinusTwo = std::abs(alpha - T(2));
  return (std::pow(sqrError * invC2 / absAlphaMinusTwo + T(1), T(0.5) * alpha) - T(1)) *
      absAlphaMinusTwo / alpha;
}

template <typename T>
T expectedGeneralBarronDeriv(T alpha, T c, T sqrError) {
  const T invC2 = T(1) / (c * c);
  return T(0.5) * invC2 *
      std::pow(sqrError * invC2 / std::abs(alpha - T(2)) + T(1), T(0.5) * alpha - T(1));
}

// Validates that `loss.deriv(s)` matches the finite-difference of `loss.value(s)` with tight
// tolerance.
//
// Numerical stability strategy:
//
// 1. **Deterministic seed.** All randomness flows from the fixture-owned seeded RNG so flakes are
//    impossible — the test either passes every run or fails every run.
//
// 2. **Argument range.** `sqrError` is sampled on a log-scale so that the dimensionless argument
//    `s/c^2` lands in [0.1, 5] — the regime where every closed form (L2/L1/Cauchy/Welsch/General)
//    is well-conditioned. Outside this range the closed-form values saturate (Welsch -> 1), are
//    dominated by an additive `+1` (L1, Cauchy), or `pow()` loses precision (General with large
//    |alpha|), all of which destroy the relative precision of the finite difference even when the
//    analytic derivative itself is accurate.
//
// 3. **Step size.** `stepSize` is proportional to `sqrError` (relative perturbation, eps^(1/5) of
//    the argument), so the test stays in the same well-conditioned regime regardless of where
//    `sqrError` was sampled.
//
// 4. **Higher-order finite difference.** A 4th-order central difference is used:
//      f'(x) ~= (8 (f(x+h) - f(x-h)) - (f(x+2h) - f(x-2h))) / (12 h)
//    Truncation error is O(h^4) instead of O(h^2), so the tolerance can be tight (1e-3 relative)
//    rather than the loose 0.5 absolute the original 2nd-order test required.
//
// 5. **Tolerance.** Relative comparison with an absolute floor proportional to the natural
//    derivative scale (~1/c^2). The floor handles cases where `refDeriv` is small enough that
//    relative comparison would multiply roundoff into spurious failures.
template <typename T>
void testGeneralizedLoss(Random<>& rng, T alpha, T c, T relTol) {
  const GeneralizedLossT<T> loss(alpha, c);

  // Sample `sqrError` so that `s/c^2 ∈ [0.1, 5]`, log-uniformly to give equal coverage to both
  // ends of the well-conditioned regime.
  const T cSq = c * c;
  const T logSqrError = rng.template uniform<T>(std::log(cSq / T(10)), std::log(T(5) * cSq));
  const T sqrError = std::exp(logSqrError);
  const T refDeriv = loss.deriv(sqrError);

  // 4th-order central difference. eps^(1/5) is the optimal relative step for a 4th-order
  // central difference balancing truncation against roundoff: ~5e-3 (float), ~8e-4 (double).
  const T stepSize = sqrError * Eps<T>(T(5e-3f), T(8e-4));
  const T fp1 = loss.value(sqrError + stepSize);
  const T fm1 = loss.value(sqrError - stepSize);
  const T fp2 = loss.value(sqrError + T(2) * stepSize);
  const T fm2 = loss.value(sqrError - T(2) * stepSize);
  const T testDeriv = (T(8) * (fp1 - fm1) - (fp2 - fm2)) / (T(12) * stepSize);

  // Relative tolerance + absolute floor scaled by 1/c^2 (the natural derivative scale).
  const T absFloor = Eps<T>(T(5e-5f), T(1e-10)) / cSq;
  const T tol = std::max(relTol * std::abs(refDeriv), absFloor);
  EXPECT_NEAR(testDeriv, refDeriv, tol)
      // clang-format off
      << "Failure in testGeneralizedLoss. Local variables are:"
      << "\n - alpha    : " << alpha
      << "\n - c        : " << c
      << "\n - relTol   : " << relTol
      << "\n - tol      : " << tol
      << "\n - stepSize : " << stepSize
      << "\n - sqrError : " << sqrError
      << "\n - refDeriv : " << refDeriv
      << "\n - testDeriv: " << testDeriv << std::endl;
  // clang-format on
}

TYPED_TEST(GeneralizedLossTest, SpecialCaseTest) {
  using T = typename TestFixture::Type;

  const size_t nTrials = 20;
  // 0.1% relative tolerance is comfortably above the 4th-order central-difference error for the
  // L2/L1/Cauchy/Welsch closed forms in the sampled regime.
  const T relTol = T(1e-3);
  for (size_t i = 0; i < nTrials; ++i) {
    const T c = this->rng.template uniform<T>(T(0.5), T(10));
    testGeneralizedLoss<T>(this->rng, GeneralizedLossT<T>::kL2, c, relTol);
    testGeneralizedLoss<T>(this->rng, GeneralizedLossT<T>::kL1, c, relTol);
    testGeneralizedLoss<T>(this->rng, GeneralizedLossT<T>::kCauchy, c, relTol);
    testGeneralizedLoss<T>(this->rng, GeneralizedLossT<T>::kWelsch, c, relTol);
  }
}

TYPED_TEST(GeneralizedLossTest, WelschSentinelUsesFiniteLowestValue) {
  using T = typename TestFixture::Type;

  EXPECT_EQ(GeneralizedLossT<T>::kWelsch, std::numeric_limits<T>::lowest());
}

TYPED_TEST(GeneralizedLossTest, GeneralCaseTest) {
  using T = typename TestFixture::Type;

  const size_t nTrials = 100;
  // Looser tolerance than the SpecialCase test because the General Barron formula evaluates
  // `pow(1 + small, alpha/2)` which loses precision as |alpha| grows (the base approaches 1
  // while the exponent grows). 1% covers the worst float case in the sampled alpha range.
  const T relTol = Eps<T>(T(1e-2f), T(1e-5));

  // Spot-check the upper alpha boundary explicitly.
  testGeneralizedLoss<T>(this->rng, T(10), T(1), relTol);

  // - alpha in [-30, 10] keeps the General Barron `pow(1+small, alpha/2)` reasonably accurate in
  //   float (|alpha/2| <= 15 keeps the relative error below ~1%). Beyond that the formula's own
  //   roundoff dominates the finite-difference noise we are trying to verify against. Users who
  //   want the Welsch limit should pass `GeneralizedLossT::kWelsch` explicitly so the closed-form
  //   Welsch branch is taken.
  // - c in [0.5, 10] avoids `c == 0` (forbidden by the ctor's `MT_CHECK(c > 0)`) and tiny c that
  //   blows up `invC2 = 1/c^2`.
  for (size_t i = 0; i < nTrials; ++i) {
    testGeneralizedLoss<T>(
        this->rng,
        this->rng.template uniform<T>(T(-30), T(10)),
        this->rng.template uniform<T>(T(0.5), T(10)),
        relTol);
  }
}

TYPED_TEST(GeneralizedLossTest, FiniteNegativeAlphaUsesGeneralFormula) {
  using T = typename TestFixture::Type;

  const T alpha = T(-2);
  const T c = T(1);
  const T sqrError = T(10);
  const GeneralizedLossT<T> loss(alpha, c);

  EXPECT_NEAR(
      loss.value(sqrError),
      expectedGeneralBarronValue(alpha, c, sqrError),
      Eps<T>(T(1e-6f), T(1e-12)));
  EXPECT_NEAR(
      loss.deriv(sqrError),
      expectedGeneralBarronDeriv(alpha, c, sqrError),
      Eps<T>(T(1e-6f), T(1e-12)));
}
