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
#include "momentum/math/simd_generalized_loss.h"

#include <drjit/math.h>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

using namespace momentum;

namespace {

template <typename T>
[[nodiscard]] auto AreAlmostEqualRelative(const Packet<T>& a, const Packet<T>& b, T relTol) {
  return drjit::abs(a - b) <= (relTol * drjit::maximum(drjit::abs(a), drjit::abs(b)));
}

template <typename T>
[[nodiscard]] Packet<T> expectedGeneralBarronValue(const Packet<T>& sqrError, T alpha, T c) {
  const T invC2 = T(1) / (c * c);
  const T absAlphaMinusTwo = std::fabs(alpha - T(2));
  return (drjit::pow(sqrError * invC2 / absAlphaMinusTwo + T(1), T(0.5) * alpha) - T(1)) *
      (absAlphaMinusTwo / alpha);
}

template <typename T>
[[nodiscard]] Packet<T> expectedGeneralBarronDeriv(const Packet<T>& sqrError, T alpha, T c) {
  const T invC2 = T(1) / (c * c);
  return (T(0.5) * invC2) *
      drjit::pow(sqrError * invC2 / std::fabs(alpha - T(2)) + T(1), T(0.5) * alpha - T(1));
}

// Helper function to compare SIMD loss value with scalar implementation
template <typename T>
void compareWithScalar(
    const SimdGeneralizedLossT<T>& simdLoss,
    const GeneralizedLossT<T>& scalarLoss,
    T sqrError,
    T absTol,
    T relTol) {
  // Get scalar result
  T scalarValue = scalarLoss.value(sqrError);

  // Create a packet with the same value for all elements
  Packet<T> sqrErrorPacket(sqrError);

  // Get SIMD result
  Packet<T> simdValuePacket = simdLoss.value(sqrErrorPacket);

  // For each element in the packet, the value should be close to the scalar value
  // We can't easily extract individual elements, so we'll check if the SIMD packet
  // is close to a packet filled with the scalar result
  Packet<T> scalarValuePacket(scalarValue);

  bool equal = drjit::all(
      (drjit::abs(simdValuePacket - scalarValuePacket) <= absTol) ||
      (drjit::abs(simdValuePacket - scalarValuePacket) <=
       relTol * drjit::maximum(drjit::abs(simdValuePacket), drjit::abs(scalarValuePacket))));

  EXPECT_TRUE(equal) << "SIMD and scalar implementations differ for sqrError = " << sqrError
                     << "\nScalar value: " << scalarValue;
}

} // namespace

using Types = testing::Types<float, double>;

template <typename T>
struct SimdGeneralizedLossTest : testing::Test {
  using Type = T;
  Random<> rng{12345};
};

TYPED_TEST_SUITE(SimdGeneralizedLossTest, Types);

template <typename T>
void testSimdGeneralizedLoss(Random<>& rng, T alpha, T c, T relTol) {
  const SimdGeneralizedLossT<T> loss(alpha, c);

  const T cSq = c * c;
  const T logSqrError = rng.template uniform<T>(std::log(cSq / T(10)), std::log(T(5) * cSq));
  const Packet<T> sqrError(std::exp(logSqrError));
  const Packet<T> refDeriv = loss.deriv(sqrError);

  const T stepSize = std::exp(logSqrError) * Eps<T>(T(5e-3f), T(8e-4));
  const Packet<T> fp1 = loss.value(sqrError + stepSize);
  const Packet<T> fm1 = loss.value(sqrError - stepSize);
  const Packet<T> fp2 = loss.value(sqrError + T(2) * stepSize);
  const Packet<T> fm2 = loss.value(sqrError - T(2) * stepSize);
  Packet<T> testDeriv = (T(8) * (fp1 - fm1) - (fp2 - fm2)) / (T(12) * stepSize);

  const T absFloor = Eps<T>(T(5e-5f), T(1e-10)) / cSq;
  const Packet<T> tol =
      drjit::maximum(Packet<T>(relTol) * drjit::abs(refDeriv), Packet<T>(absFloor));
  EXPECT_TRUE(drjit::all(drjit::abs(testDeriv - refDeriv) <= tol))
      // clang-format off
      << "Failure in testSimdGeneralizedLoss. Local variables are:"
      << "\n - alpha    : " << alpha
      << "\n - c        : " << c
      << "\n - relTol   : " << relTol
      << "\n - tol      : " << drjit::string(tol).c_str()
      << "\n - stepSize : " << stepSize
      << "\n - sqrError : " << drjit::string(sqrError).c_str()
      << "\n - refDeriv : " << drjit::string(refDeriv).c_str()
      << "\n - testDeriv: " << drjit::string(testDeriv).c_str()
      << std::endl;
  // clang-format on
}

TYPED_TEST(SimdGeneralizedLossTest, ValueFunctionTest) {
  using T = typename TestFixture::Type;

  const T relTol = 1e-5;

  // Test L2 loss
  {
    SCOPED_TRACE("L2 Value");
    SimdGeneralizedLossT<T> loss(GeneralizedLossT<T>::kL2, T(2));
    Packet<T> sqrError = this->rng.template uniform<T>(0, 10);
    Packet<T> expected = sqrError * T(0.25); // invC2 = 1/(2*2) = 0.25
    Packet<T> actual = loss.value(sqrError);
    EXPECT_TRUE(drjit::all(AreAlmostEqualRelative(actual, expected, relTol)));
  }

  // Test L1 loss
  {
    SCOPED_TRACE("L1 Value");
    SimdGeneralizedLossT<T> loss(GeneralizedLossT<T>::kL1, T(2));
    Packet<T> sqrError = this->rng.template uniform<T>(0, 10);
    Packet<T> expected = drjit::sqrt(sqrError * T(0.25) + T(1)) - T(1);
    Packet<T> actual = loss.value(sqrError);
    EXPECT_TRUE(drjit::all(AreAlmostEqualRelative(actual, expected, relTol)));
  }

  // Test Cauchy loss
  {
    SCOPED_TRACE("Cauchy Value");
    SimdGeneralizedLossT<T> loss(GeneralizedLossT<T>::kCauchy, T(2));
    Packet<T> sqrError = this->rng.template uniform<T>(0, 10);
    Packet<T> expected = drjit::log(T(0.5) * sqrError * T(0.25) + T(1));
    Packet<T> actual = loss.value(sqrError);
    EXPECT_TRUE(drjit::all(AreAlmostEqualRelative(actual, expected, relTol)));
  }

  // Skip direct test for Welsch loss as it's sensitive to numerical differences
  // We still test it indirectly through CompareWithScalarTest

  // Test General loss
  {
    SCOPED_TRACE("General Value");
    const T alpha = T(4);
    SimdGeneralizedLossT<T> loss(alpha, T(2));
    Packet<T> sqrError = this->rng.template uniform<T>(0, 10);
    // General case formula from the implementation
    Packet<T> expected =
        (drjit::pow(sqrError * T(0.25) / std::fabs(alpha - T(2)) + T(1), T(0.5) * alpha) - T(1)) *
        (std::fabs(alpha - T(2)) / alpha);
    Packet<T> actual = loss.value(sqrError);
    EXPECT_TRUE(drjit::all(AreAlmostEqualRelative(actual, expected, relTol)));
  }
}

TYPED_TEST(SimdGeneralizedLossTest, FiniteNegativeAlphaUsesGeneralFormula) {
  using T = typename TestFixture::Type;

  const T alpha = T(-2);
  const T c = T(1);
  const Packet<T> sqrError(T(10));
  const SimdGeneralizedLossT<T> loss(alpha, c);

  const T relTol = T(1e-5);
  const Packet<T> expectedValue = expectedGeneralBarronValue(sqrError, alpha, c);
  const Packet<T> expectedDeriv = expectedGeneralBarronDeriv(sqrError, alpha, c);
  EXPECT_TRUE(drjit::all(AreAlmostEqualRelative(loss.value(sqrError), expectedValue, relTol)));
  EXPECT_TRUE(drjit::all(AreAlmostEqualRelative(loss.deriv(sqrError), expectedDeriv, relTol)));
}

TYPED_TEST(SimdGeneralizedLossTest, WelschSentinelUsesFiniteLowestValue) {
  using T = typename TestFixture::Type;

  EXPECT_EQ(SimdGeneralizedLossT<T>::kWelsch, std::numeric_limits<T>::lowest());
}

TYPED_TEST(SimdGeneralizedLossTest, EdgeCaseTest) {
  using T = typename TestFixture::Type;

  const T relTol = 1e-5;

  // Test with zero squared error
  {
    SCOPED_TRACE("Zero Error");
    SimdGeneralizedLossT<T> l2Loss(GeneralizedLossT<T>::kL2, T(1));
    SimdGeneralizedLossT<T> l1Loss(GeneralizedLossT<T>::kL1, T(1));
    SimdGeneralizedLossT<T> cauchyLoss(GeneralizedLossT<T>::kCauchy, T(1));
    SimdGeneralizedLossT<T> welschLoss(GeneralizedLossT<T>::kWelsch, T(1));

    auto zeroError = Packet<T>(T(0));

    // For zero error, L2 should be 0
    EXPECT_TRUE(
        drjit::all(AreAlmostEqualRelative(l2Loss.value(zeroError), Packet<T>(T(0)), relTol)));

    // For zero error, L1 should be 0
    EXPECT_TRUE(
        drjit::all(AreAlmostEqualRelative(l1Loss.value(zeroError), Packet<T>(T(0)), relTol)));

    // For zero error, Cauchy should be 0
    EXPECT_TRUE(
        drjit::all(AreAlmostEqualRelative(cauchyLoss.value(zeroError), Packet<T>(T(0)), relTol)));

    // For zero error, Welsch should be 0
    EXPECT_TRUE(
        drjit::all(AreAlmostEqualRelative(welschLoss.value(zeroError), Packet<T>(T(0)), relTol)));
  }

  // Test with very large squared error
  {
    SCOPED_TRACE("Large Error");
    SimdGeneralizedLossT<T> loss(GeneralizedLossT<T>::kL2, T(1));
    auto largeError = Packet<T>(T(1e6));
    Packet<T> expected = largeError; // For L2, value should be sqrError
    Packet<T> actual = loss.value(largeError);
    EXPECT_TRUE(drjit::all(AreAlmostEqualRelative(actual, expected, relTol)));
  }
}

TYPED_TEST(SimdGeneralizedLossTest, CompareWithScalarTest) {
  using T = typename TestFixture::Type;

  const T absTol = Eps<T>(1e-5f, 1e-10);
  const T relTol = 1e-5;

  // Test all loss types
  const std::vector<T> alphaValues = {
      GeneralizedLossT<T>::kL2,
      GeneralizedLossT<T>::kL1,
      GeneralizedLossT<T>::kCauchy,
      GeneralizedLossT<T>::kWelsch,
      T(3.5) // General case
  };

  for (const T& alpha : alphaValues) {
    SCOPED_TRACE("Alpha = " + std::to_string(alpha));

    const T c = this->rng.template uniform<T>(0.5, 5);
    SimdGeneralizedLossT<T> simdLoss(alpha, c);
    GeneralizedLossT<T> scalarLoss(alpha, c);

    // Test with various squared errors
    for (int i = 0; i < 10; ++i) {
      T sqrError = this->rng.template uniform<T>(0, 10);
      compareWithScalar(simdLoss, scalarLoss, sqrError, absTol, relTol);
    }

    // Test with zero error
    compareWithScalar(simdLoss, scalarLoss, T(0), absTol, relTol);

    // Test with large error
    compareWithScalar(simdLoss, scalarLoss, T(100), absTol, relTol);
  }
}

TYPED_TEST(SimdGeneralizedLossTest, SpecialCaseTest) {
  using T = typename TestFixture::Type;

  const size_t nTrials = 20;
  const T relTol = T(1e-3);
  for (size_t i = 0; i < nTrials; ++i) {
    {
      SCOPED_TRACE("L2");
      testSimdGeneralizedLoss<T>(
          this->rng,
          SimdGeneralizedLossT<T>::kL2,
          this->rng.template uniform<T>(T(0.5), T(10)),
          relTol);
    }

    {
      SCOPED_TRACE("L1");
      testSimdGeneralizedLoss<T>(
          this->rng,
          SimdGeneralizedLossT<T>::kL1,
          this->rng.template uniform<T>(T(0.5), T(10)),
          relTol);
    }

    {
      SCOPED_TRACE("Cauchy");
      testSimdGeneralizedLoss<T>(
          this->rng,
          SimdGeneralizedLossT<T>::kCauchy,
          this->rng.template uniform<T>(T(0.5), T(10)),
          relTol);
    }

    {
      SCOPED_TRACE("Welsch");
      testSimdGeneralizedLoss<T>(
          this->rng,
          SimdGeneralizedLossT<T>::kWelsch,
          this->rng.template uniform<T>(T(0.5), T(10)),
          relTol);
    }
  }
}

TYPED_TEST(SimdGeneralizedLossTest, GeneralCaseTest) {
  using T = typename TestFixture::Type;

  const size_t nTrials = 100;
  const T relTol = Eps<T>(T(1e-2f), T(1e-5));

  testSimdGeneralizedLoss<T>(this->rng, T(10), T(1), relTol);

  for (size_t i = 0; i < nTrials; ++i) {
    // Draw alpha and c into named locals in a fixed order; the order of evaluation of function
    // arguments is unspecified in C++, so drawing both inline would make the values
    // compiler-dependent and defeat cross-platform reproducibility.
    const T alpha = this->rng.template uniform<T>(T(-30), T(10));
    const T c = this->rng.template uniform<T>(T(0.5), T(10));
    testSimdGeneralizedLoss<T>(this->rng, alpha, c, relTol);
  }
}
