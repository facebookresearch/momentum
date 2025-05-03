/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/math/constants.h"
#include "momentum/math/fwd.h"
#include "momentum/math/random.h"
#include "momentum/math/simd_generalized_loss.h"

#include <drjit/math.h>
#include <gtest/gtest.h>

using namespace momentum;

namespace {

template <typename T>
[[nodiscard]] auto AreAlmostEqualRelative(const Packet<T>& a, const Packet<T>& b, T relTol) {
  return drjit::abs(a - b) <= (relTol * drjit::maximum(drjit::abs(a), drjit::abs(b)));
}

} // namespace

using Types = testing::Types<float, double>;

template <typename T>
struct SimdGeneralizedLossTest : testing::Test {
  using Type = T;
};

TYPED_TEST_SUITE(SimdGeneralizedLossTest, Types);

template <typename T>
void testSimdGeneralizedLoss(T alpha, T c, T absTol, T relTol) {
  Random rand;
  const SimdGeneralizedLossT<T> loss(alpha, c);

  const T stepSize = Eps<T>(5e-3f, 5e-5);
  // make sure the value is greater than stepSize for finite difference test
  const Packet<T> sqrError = rand.uniform<T>(stepSize, 1e3);
  const Packet<T> refDeriv = loss.deriv(sqrError);

  // finite difference test
  const Packet<T> val1 = loss.value(sqrError - stepSize);
  const Packet<T> val2 = loss.value(sqrError + stepSize);
  Packet<T> testDeriv = (val2 - val1) / (T(2) * stepSize);

  // use an esp as input because larger alpha needs larger threshold
  auto result = (drjit::abs(testDeriv - refDeriv) <= absTol);
  if (!drjit::all(result)) {
    result = AreAlmostEqualRelative(testDeriv, refDeriv, relTol);
  }
  EXPECT_TRUE(drjit::all(result))
      // clang-format off
      << "Failure in testSimdGeneralizedLoss. Local variables are:"
      << "\n - alpha    : " << alpha
      << "\n - c        : " << c
      << "\n - absTol   : " << absTol
      << "\n - relTol   : " << relTol
      << "\n - stepSize : " << stepSize
      << "\n - sqrError : " << drjit::string(sqrError).c_str()
      << "\n - val1     : " << drjit::string(val1).c_str()
      << "\n - val2     : " << drjit::string(val2).c_str()
      << "\n - refDeriv : " << drjit::string(refDeriv).c_str()
      << "\n - testDeriv: " << drjit::string(testDeriv).c_str()
      << std::endl;
      // clang-format off
}

TYPED_TEST(SimdGeneralizedLossTest, SpecialCaseTest) {
  using T = typename TestFixture::Type;

  Random rand;
  const size_t nTrials = 20;
  const T absTol = Eps<T>(5e-1f, 5e-5);
  const T relTol = 0.01; // 1% relative tolerance
  for (size_t i = 0; i < nTrials; ++i) {
    {
      SCOPED_TRACE("L2");
      testSimdGeneralizedLoss<T>(SimdGeneralizedLossd::kL2, rand.uniform<T>(1, 10), absTol, relTol);
    }

    {
      SCOPED_TRACE("L1");
      testSimdGeneralizedLoss<T>(SimdGeneralizedLossd::kL1, rand.uniform<T>(1, 10), absTol, relTol);
    }

    {
      SCOPED_TRACE("Cauchy");
      testSimdGeneralizedLoss<T>(
          SimdGeneralizedLossd::kCauchy, rand.uniform<T>(1, 10), absTol, relTol);
    }

    {
      SCOPED_TRACE("Welsch");
      testSimdGeneralizedLoss<T>(
          SimdGeneralizedLossd::kWelsch, rand.uniform<T>(1, 10), absTol, relTol);
    }
  }
}

TYPED_TEST(SimdGeneralizedLossTest, GeneralCaseTest) {
  using T = typename TestFixture::Type;

  Random rand;
  const size_t nTrials = 100;
  const T absTol = Eps<T>(1e-3f, 2e-6);
  const T relTol = 0.02;  // 2% relative tolerance
  testSimdGeneralizedLoss<T>(10, 10, absTol, relTol);  // most extreme case
  for (size_t i = 0; i < nTrials; ++i) {
    testSimdGeneralizedLoss<T>(rand.uniform<T>(-1e6, 10), rand.uniform<T>(0, 10), absTol, relTol);
  }
}
