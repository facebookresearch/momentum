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

using namespace momentum;

namespace {

template <typename T>
[[nodiscard]] auto AreAlmostEqualRelative(const Packet<T>& a, const Packet<T>& b, T relTol) {
  return drjit::abs(a - b) <= (relTol * drjit::maximum(drjit::abs(a), drjit::abs(b)));
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

TYPED_TEST(SimdGeneralizedLossTest, ValueFunctionTest) {
  using T = typename TestFixture::Type;

  Random rand;
  const T relTol = 1e-5;

  // Test L2 loss
  {
    SCOPED_TRACE("L2 Value");
    SimdGeneralizedLossT<T> loss(GeneralizedLossT<T>::kL2, T(2));
    Packet<T> sqrError = rand.uniform<T>(0, 10);
    Packet<T> expected = sqrError * T(0.25); // invC2 = 1/(2*2) = 0.25
    Packet<T> actual = loss.value(sqrError);
    EXPECT_TRUE(drjit::all(AreAlmostEqualRelative(actual, expected, relTol)));
  }

  // Test L1 loss
  {
    SCOPED_TRACE("L1 Value");
    SimdGeneralizedLossT<T> loss(GeneralizedLossT<T>::kL1, T(2));
    Packet<T> sqrError = rand.uniform<T>(0, 10);
    Packet<T> expected = drjit::sqrt(sqrError * T(0.25) + T(1)) - T(1);
    Packet<T> actual = loss.value(sqrError);
    EXPECT_TRUE(drjit::all(AreAlmostEqualRelative(actual, expected, relTol)));
  }

  // Test Cauchy loss
  {
    SCOPED_TRACE("Cauchy Value");
    SimdGeneralizedLossT<T> loss(GeneralizedLossT<T>::kCauchy, T(2));
    Packet<T> sqrError = rand.uniform<T>(0, 10);
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
    Packet<T> sqrError = rand.uniform<T>(0, 10);
    // General case formula from the implementation
    Packet<T> expected = (drjit::pow(
                            sqrError * T(0.25) / std::fabs(alpha - T(2)) + T(1),
                            T(0.5) * alpha) -
                        T(1)) *
                    (std::fabs(alpha - T(2)) / alpha);
    Packet<T> actual = loss.value(sqrError);
    EXPECT_TRUE(drjit::all(AreAlmostEqualRelative(actual, expected, relTol)));
  }
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
    EXPECT_TRUE(drjit::all(AreAlmostEqualRelative(l2Loss.value(zeroError), Packet<T>(T(0)), relTol)));

    // For zero error, L1 should be 0
    EXPECT_TRUE(drjit::all(AreAlmostEqualRelative(l1Loss.value(zeroError), Packet<T>(T(0)), relTol)));

    // For zero error, Cauchy should be 0
    EXPECT_TRUE(drjit::all(AreAlmostEqualRelative(cauchyLoss.value(zeroError), Packet<T>(T(0)), relTol)));

    // For zero error, Welsch should be 0
    EXPECT_TRUE(drjit::all(AreAlmostEqualRelative(welschLoss.value(zeroError), Packet<T>(T(0)), relTol)));
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

  Random rand;
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

    const T c = rand.uniform<T>(0.5, 5);
    SimdGeneralizedLossT<T> simdLoss(alpha, c);
    GeneralizedLossT<T> scalarLoss(alpha, c);

    // Test with various squared errors
    for (int i = 0; i < 10; ++i) {
      T sqrError = rand.uniform<T>(0, 10);
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

  // Test an extreme case with relaxed tolerances due to numerical precision limits
  testSimdGeneralizedLoss<T>(10, 10, Eps<T>(5e-3f, 1e-5), 0.05);  // 5% relative tolerance

  for (size_t i = 0; i < nTrials; ++i) {
    testSimdGeneralizedLoss<T>(rand.uniform<T>(-1e6, 9), rand.uniform<T>(0, 9), absTol, relTol);
  }
}
