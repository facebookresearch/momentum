/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/math/constants.h"
#include "momentum/math/random.h"

#include <gtest/gtest.h>

#include <vector>

using namespace momentum;

namespace {
constexpr auto kMaxAllowedAbsError = 1e-6f;
}

using Types = testing::Types<float, double>;

template <typename T>
struct RandomTest : testing::Test {
  using Type = T;
};

TYPED_TEST_SUITE(RandomTest, Types);

TEST(RandomTest, DeterministicBySeed) {
  const auto numTests = 100;
  const float fmin = -1.2345f;
  const float fmax = +5.6789f;
  const std::vector<unsigned> seeds = {0, 12, 123, 1234, 12345};

  for (const auto& seed : seeds) {
    // Create two random number generator with the same seed
    Random r0(seed);
    Random r1(seed);

    EXPECT_EQ(r0.getSeed(), seed);
    EXPECT_EQ(r1.getSeed(), seed);

    // Expects to draw the same random numbers in sequence
    for (auto i = 0u; i < numTests; ++i) {
      EXPECT_NEAR(r0.uniform(fmin, fmax), r1.uniform(fmin, fmax), kMaxAllowedAbsError);
    }
  }

  // Create two random number generator with a random seed
  Random r0;
  Random r1;
  for (const auto& seed : seeds) {
    r0.setSeed(seed);
    r1.setSeed(seed);

    EXPECT_EQ(r0.getSeed(), seed);
    EXPECT_EQ(r1.getSeed(), seed);

    // Expects to draw the same random numbers in sequence
    for (auto i = 0u; i < numTests; ++i) {
      EXPECT_NEAR(r0.uniform(fmin, fmax), r1.uniform(fmin, fmax), kMaxAllowedAbsError);
    }
  }
}

TEST(RandomTest, ScalarUniform) {
  const auto numTests = 1000;

  for (auto i = 0u; i < numTests; ++i) {
    const float fmin = -1.234f;
    const float fmax = +5.678f;
    EXPECT_GT(uniform<float>(fmin, fmax), fmin);
    EXPECT_LE(uniform<float>(fmin, fmax), fmax);

    const double dmin = -1.234;
    const double dmax = +5.678;
    EXPECT_GT(uniform<double>(dmin, dmax), dmin);
    EXPECT_LE(uniform<double>(dmin, dmax), dmax);

    const int imin = -123;
    const int imax = 456;
    EXPECT_GE(uniform<int>(imin, imax), imin);
    EXPECT_LE(uniform<int>(imin, imax), imax);

    const unsigned int umin = 123u;
    const unsigned int umax = 456u;
    EXPECT_GE(uniform<unsigned int>(umin, umax), umin);
    EXPECT_LE(uniform<unsigned int>(umin, umax), umax);
  }
}

template <typename T, typename Scalar>
void testUniformDynamicMatrixScalarBounds(int rows, int cols, Scalar min, Scalar max) {
  const auto rand = uniform<T>(rows, cols, min, max);
  EXPECT_EQ(rand.rows(), rows) << "type: " << typeid(T).name();
  EXPECT_EQ(rand.cols(), cols) << "type: " << typeid(T).name();
  EXPECT_TRUE((rand.array() >= min).all())
      << "type: " << typeid(T).name() << "\nrand: " << rand.transpose() << "\nmin: " << min
      << "\nmax: " << max;
  if constexpr (std::is_floating_point_v<Scalar>) {
#if defined(MOMENTUM_TEST_FAST_MATH)
    EXPECT_TRUE((rand.array() <= max).all())
        << "type: " << typeid(T).name() << "\nrand: " << rand.transpose() << "\nmin: " << min
        << "\nmax: " << max;
#else
    EXPECT_TRUE((rand.array() < max).all())
        << "type: " << typeid(T).name() << "\nrand: " << rand.transpose() << "\nmin: " << min
        << "\nmax: " << max;
#endif
  } else {
    EXPECT_TRUE((rand.array() <= max).all())
        << "type: " << typeid(T).name() << "\nrand: " << rand.transpose() << "\nmin: " << min
        << "\nmax: " << max;
  }
}

template <typename T, typename Scalar>
void testUniformDynamicVectorScalarBounds(int size, Scalar min, Scalar max) {
  const auto rand = uniform<T>(size, min, max);
  EXPECT_EQ(rand.size(), size) << "type: " << typeid(T).name();
  EXPECT_TRUE((rand.array() >= min).all())
      << "type: " << typeid(T).name() << "\nrand: " << rand.transpose() << "\nmin: " << min
      << "\nmax: " << max;
  if constexpr (std::is_floating_point_v<Scalar>) {
#if defined(MOMENTUM_TEST_FAST_MATH)
    EXPECT_TRUE((rand.array() <= max).all())
        << "type: " << typeid(T).name() << "\nrand: " << rand.transpose() << "\nmin: " << min
        << "\nmax: " << max;
#else
    EXPECT_TRUE((rand.array() < max).all())
        << "type: " << typeid(T).name() << "\nrand: " << rand.transpose() << "\nmin: " << min
        << "\nmax: " << max;
#endif
  } else {
    EXPECT_TRUE((rand.array() <= max).all())
        << "type: " << typeid(T).name() << "\nrand: " << rand.transpose() << "\nmin: " << min
        << "\nmax: " << max;
  }
}

template <typename T, typename Scalar>
void testUniformFixedScalarBounds(Scalar min, Scalar max) {
  const auto rand = uniform<T>(min, max);
  EXPECT_TRUE((rand.array() >= min).all())
      << "type: " << typeid(T).name() << "\nrand: " << rand.transpose() << "\nmin: " << min
      << "\nmax: " << max;
  EXPECT_TRUE((rand.array() <= max).all())
      << "type: " << typeid(T).name() << "\nrand: " << rand.transpose() << "\nmin: " << min
      << "\nmax: " << max;
}

template <typename T>
void testUniform(const T& min, const T& max) {
  const auto rand = uniform(min, max);
  EXPECT_TRUE((rand.array() >= min.array()).all())
      << "type: " << typeid(T).name() << "\nrand: " << rand.transpose()
      << "\nmin: " << min.transpose() << "\nmax: " << max.transpose();
  EXPECT_TRUE((rand.array() <= max.array()).all())
      << "type: " << typeid(T).name() << "\nrand: " << rand.transpose()
      << "\nmin: " << min.transpose() << "\nmax: " << max.transpose();
}

TEST(RandomTest, VectorMatrixUniform) {
  const auto numTests = 1000;

  const float minf = -1.234f;
  const float maxf = +1.234f;

  const double mind = -1.234;
  const double maxd = +1.234;

  const int mini = -1234;
  const int maxi = +1234;

  const unsigned int minu = 1234;
  const unsigned int maxu = 5678;

  auto minVecXf = VectorXf(2);
  minVecXf << -1.234f, -5.678f;
  auto maxVecXf = VectorXf(2);
  maxVecXf << +1.234f, +5.678f;

  auto minVecXd = VectorXd(2);
  minVecXd << -1.234, -5.678;
  auto maxVecXd = VectorXd(2);
  maxVecXd << +1.234, +5.678;

  auto minVecXi = VectorXi(2);
  minVecXi << -1234, -5678;
  auto maxVecXi = VectorXi(2);
  maxVecXi << +1234, +5678;

  auto minVecXu = VectorXu(2);
  minVecXu << 123, 1234;
  auto maxVecXu = VectorXu(2);
  maxVecXu << 456, 5678;

  const auto minVec2f = Vector2f(-1.234f, -5.678f);
  const auto maxVec2f = Vector2f(+1.234f, +5.678f);

  const auto minVec2d = Vector2d(-1.234, -5.678);
  const auto maxVec2d = Vector2d(+1.234, +5.678);

  const auto minVec2i = Vector2i(-1234, -5678);
  const auto maxVec2i = Vector2i(+1234, +5678);

  const auto minVec2u = Vector2u(123, 1234);
  const auto maxVec2u = Vector2u(456, 5678);

  auto minMatXf = MatrixXf(2, 3);
  minMatXf << -1.23f, -4.56f, -7.89f, -9.87f, -6.54f, -3.21f;
  auto maxMatXf = MatrixXf(2, 3);
  maxMatXf << +1.23f, +4.56f, +7.89f, +9.87f, +6.54f, +3.21f;

  auto minMatXd = MatrixXd(2, 3);
  minMatXd << -1.23, -4.56, -7.89, -9.87, -6.54, -3.21;
  auto maxMatXd = MatrixXd(2, 3);
  maxMatXd << +1.23, +4.56, +7.89, +9.87, +6.54, +3.21;

  auto minMatXi = MatrixXi(2, 3);
  minMatXi << -123, -456, -789, -987, -654, -321;
  auto maxMatXi = MatrixXi(2, 3);
  maxMatXi << +123, +456, +789, +987, +654, +321;

  auto minMatXu = MatrixXu(2, 3);
  minMatXu << +123, +456, +789, +789, +456, +123;
  auto maxMatXu = MatrixXu(2, 3);
  maxMatXu << +321, +654, +987, +987, +654, +321;

  auto minMat2f = Matrix2f();
  minMat2f << -1.23f, -4.56f, -7.89f, -9.87f;
  auto maxMat2f = Matrix2f();
  maxMat2f << +1.23f, +4.56f, +7.89f, +9.87f;

  auto minMat2d = Matrix2d();
  minMat2d << -1.23, -4.56, -7.89, -9.87;
  auto maxMat2d = Matrix2d();
  maxMat2d << +1.23, +4.56, +7.89, +9.87;

  auto minMat2i = Matrix2i();
  minMat2i << -123, -456, -789, -987;
  auto maxMat2i = Matrix2i();
  maxMat2i << +123, +456, +789, +987;

  auto minMat2u = Matrix2u();
  minMat2u << +123, +456, +789, +789;
  auto maxMat2u = Matrix2u();
  maxMat2u << +321, +654, +987, +987;

  // Scalar bounds
  for (auto i = 0u; i < numTests; ++i) {
    // Dynamic float vector
    testUniformDynamicVectorScalarBounds<VectorXf>(0, minf, maxf);
    testUniformDynamicVectorScalarBounds<VectorXf>(1, minf, maxf);
    testUniformDynamicVectorScalarBounds<VectorXf>(2, minf, maxf);
    testUniformDynamicVectorScalarBounds<VectorXf>(6, minf, maxf);

    // Dynamic double vector
    testUniformDynamicVectorScalarBounds<VectorXd>(0, mind, maxd);
    testUniformDynamicVectorScalarBounds<VectorXd>(1, mind, maxd);
    testUniformDynamicVectorScalarBounds<VectorXd>(2, mind, maxd);
    testUniformDynamicVectorScalarBounds<VectorXd>(6, mind, maxd);

    // Dynamic int vector
    testUniformDynamicVectorScalarBounds<VectorXi>(0, mini, maxi);
    testUniformDynamicVectorScalarBounds<VectorXi>(1, mini, maxi);
    testUniformDynamicVectorScalarBounds<VectorXi>(2, mini, maxi);
    testUniformDynamicVectorScalarBounds<VectorXi>(6, mini, maxi);

    // Dynamic unsigned int vector
    testUniformDynamicVectorScalarBounds<VectorXu>(0, minu, maxu);
    testUniformDynamicVectorScalarBounds<VectorXu>(1, minu, maxu);
    testUniformDynamicVectorScalarBounds<VectorXu>(2, minu, maxu);
    testUniformDynamicVectorScalarBounds<VectorXu>(6, minu, maxu);

    // Fixed float vector
    testUniformFixedScalarBounds<Vector0f>(minf, maxf);
    testUniformFixedScalarBounds<Vector1f>(minf, maxf);
    testUniformFixedScalarBounds<Vector2f>(minf, maxf);
    testUniformFixedScalarBounds<Vector6f>(minf, maxf);

    // Fixed double vector
    testUniformFixedScalarBounds<Vector0d>(mind, maxd);
    testUniformFixedScalarBounds<Vector1d>(mind, maxd);
    testUniformFixedScalarBounds<Vector2d>(mind, maxd);
    testUniformFixedScalarBounds<Vector6d>(mind, maxd);

    // Fixed int vector
    testUniformFixedScalarBounds<Vector0i>(mini, maxi);
    testUniformFixedScalarBounds<Vector1i>(mini, maxi);
    testUniformFixedScalarBounds<Vector2i>(mini, maxi);
    testUniformFixedScalarBounds<Vector6i>(mini, maxi);

    // Fixed unsigned int vector
    testUniformFixedScalarBounds<Vector0u>(minu, maxu);
    testUniformFixedScalarBounds<Vector1u>(minu, maxu);
    testUniformFixedScalarBounds<Vector2u>(minu, maxu);
    testUniformFixedScalarBounds<Vector6u>(minu, maxu);

    // Dynamic float matrix
    testUniformDynamicMatrixScalarBounds<MatrixXf>(0, 2, minf, maxf);
    testUniformDynamicMatrixScalarBounds<MatrixXf>(1, 2, minf, maxf);
    testUniformDynamicMatrixScalarBounds<MatrixXf>(2, 2, minf, maxf);
    testUniformDynamicMatrixScalarBounds<MatrixXf>(6, 2, minf, maxf);

    // Dynamic double matrix
    testUniformDynamicMatrixScalarBounds<MatrixXd>(0, 2, mind, maxd);
    testUniformDynamicMatrixScalarBounds<MatrixXd>(1, 2, mind, maxd);
    testUniformDynamicMatrixScalarBounds<MatrixXd>(2, 2, mind, maxd);
    testUniformDynamicMatrixScalarBounds<MatrixXd>(6, 2, mind, maxd);

    // Dynamic int matrix
    testUniformDynamicMatrixScalarBounds<MatrixXi>(0, 2, mini, maxi);
    testUniformDynamicMatrixScalarBounds<MatrixXi>(1, 2, mini, maxi);
    testUniformDynamicMatrixScalarBounds<MatrixXi>(2, 2, mini, maxi);
    testUniformDynamicMatrixScalarBounds<MatrixXi>(6, 2, mini, maxi);

    // Dynamic unsigned int matrix
    testUniformDynamicMatrixScalarBounds<MatrixXu>(0, 2, minu, maxu);
    testUniformDynamicMatrixScalarBounds<MatrixXu>(1, 2, minu, maxu);
    testUniformDynamicMatrixScalarBounds<MatrixXu>(2, 2, minu, maxu);
    testUniformDynamicMatrixScalarBounds<MatrixXu>(6, 2, minu, maxu);

    // Fixed float matrix
    testUniformFixedScalarBounds<Matrix0f>(minf, maxf);
    testUniformFixedScalarBounds<Matrix1f>(minf, maxf);
    testUniformFixedScalarBounds<Matrix2f>(minf, maxf);
    testUniformFixedScalarBounds<Matrix6f>(minf, maxf);

    // Fixed double matrix
    testUniformFixedScalarBounds<Matrix0d>(mind, maxd);
    testUniformFixedScalarBounds<Matrix1d>(mind, maxd);
    testUniformFixedScalarBounds<Matrix2d>(mind, maxd);
    testUniformFixedScalarBounds<Matrix6d>(mind, maxd);

    // Fixed int matrix
    testUniformFixedScalarBounds<Matrix0i>(mini, maxi);
    testUniformFixedScalarBounds<Matrix1i>(mini, maxi);
    testUniformFixedScalarBounds<Matrix2i>(mini, maxi);
    testUniformFixedScalarBounds<Matrix6i>(mini, maxi);

    // Fixed unsigned int matrix
    testUniformFixedScalarBounds<Matrix0u>(minu, maxu);
    testUniformFixedScalarBounds<Matrix1u>(minu, maxu);
    testUniformFixedScalarBounds<Matrix2u>(minu, maxu);
    testUniformFixedScalarBounds<Matrix6u>(minu, maxu);
  }

  // Vector bounds
  for (auto i = 0u; i < numTests; ++i) {
    // Dynamic float vector
    testUniform(minVecXf, maxVecXf);

    // Dynamic double vector
    testUniform(minVecXd, maxVecXd);

    // Dynamic int vector
    testUniform(minVecXi, maxVecXi);

    // Dynamic unsigned int vector
    testUniform(minVecXu, maxVecXu);

    // Fixed float vector
    testUniform(minVec2f, maxVec2f);

    // Fixed double vector
    testUniform(minVec2d, maxVec2d);

    // Fixed int vector
    testUniform(minVec2i, maxVec2i);

    // Fixed unsigned int vector
    testUniform(minVec2u, maxVec2u);

    // Dynamic float matrix
    testUniform(minMatXf, maxMatXf);

    // Dynamic double matrix
    testUniform(minMatXd, maxMatXd);

    // Dynamic int matrix
    testUniform(minMatXi, maxMatXi);

    // Dynamic unsigned int matrix
    testUniform(minMatXu, maxMatXu);

    // Fixed float matrix
    testUniform(minMat2f, maxMat2f);

    // Fixed double matrix
    testUniform(minMat2d, maxMat2d);

    // Fixed int matrix
    testUniform(minMat2i, maxMat2i);

    // Fixed unsigned int matrix
    testUniform(minMat2u, maxMat2u);
  }
}

TYPED_TEST(RandomTest, UniformQuaternion) {
  using T = typename TestFixture::Type;

  // Run the test multiple times to check the randomness
  for (int i = 0; i < 1000; ++i) {
    Quaternion<T> q = uniformQuaternion<T>();

    // Check each component is in the range [-1, 1]
    EXPECT_GE(1.0, std::abs(q.x()));
    EXPECT_GE(1.0, std::abs(q.y()));
    EXPECT_GE(1.0, std::abs(q.z()));
    EXPECT_GE(1.0, std::abs(q.w()));

    // Check if the quaternion is normalized (i.e., its magnitude is 1)
    T magnitude = std::sqrt(q.x() * q.x() + q.y() * q.y() + q.z() * q.z() + q.w() * q.w());
    EXPECT_NEAR(1.0, magnitude, Eps<T>(1e-6, 1e-15));
  }
}

// TODO: Add tests for normal distributions
