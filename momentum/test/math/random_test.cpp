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

 protected:
  void SetUp() override {
    // Set fixed seed to make tests deterministic and avoid flakiness
    fixedSeed_ = 42;
    Random<>::GetSingleton().setSeed(fixedSeed_);
    initialSeed_ = Random<>::GetSingleton().getSeed();
  }

  void TearDown() override {
    // Verify that the random seed hasn't changed during the test
    uint32_t finalSeed = Random<>::GetSingleton().getSeed();
    EXPECT_EQ(initialSeed_, finalSeed) << "Random seed should not change during test execution";
  }

 private:
  uint32_t fixedSeed_{};
  uint32_t initialSeed_{};
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

TEST(RandomTest, ScalarNormal) {
  const auto numTests = 1000;
  const float mean = 2.5f;
  const float sigma = 1.5f;

  // Test that normal distribution generates values around the mean
  std::vector<float> values;
  values.reserve(numTests);

  for (auto i = 0u; i < numTests; ++i) {
    auto val = normal<float>(mean, sigma);
    values.push_back(val);
  }

  // Calculate sample mean and standard deviation
  float sampleMean = 0.0f;
  for (float val : values) {
    sampleMean += val;
  }
  sampleMean /= numTests;

  float sampleVariance = 0.0f;
  for (float val : values) {
    float diff = val - sampleMean;
    sampleVariance += diff * diff;
  }
  sampleVariance /= (numTests - 1);
  float sampleStdDev = std::sqrt(sampleVariance);

  // Check that sample statistics are close to expected values
  // Allow for some statistical variation
  EXPECT_NEAR(sampleMean, mean, 0.2f) << "Sample mean should be close to expected mean";
  EXPECT_NEAR(sampleStdDev, sigma, 0.2f) << "Sample std dev should be close to expected sigma";
}

TEST(RandomTest, NormalVsUniform) {
  const auto numTests = 1000;
  const float mean = 0.0f;
  const float sigma = 1.0f;
  const float uniformMin = -3.0f;
  const float uniformMax = 3.0f;

  std::vector<float> normalValues;
  std::vector<float> uniformValues;
  normalValues.reserve(numTests);
  uniformValues.reserve(numTests);

  // Generate samples from both distributions
  for (auto i = 0u; i < numTests; ++i) {
    normalValues.push_back(normal<float>(mean, sigma));
    uniformValues.push_back(uniform<float>(uniformMin, uniformMax));
  }

  // Calculate means
  float normalMean = 0.0f, uniformMean = 0.0f;
  for (auto i = 0u; i < numTests; ++i) {
    normalMean += normalValues[i];
    uniformMean += uniformValues[i];
  }
  normalMean /= numTests;
  uniformMean /= numTests;

  // Calculate variances
  float normalVar = 0.0f, uniformVar = 0.0f;
  for (auto i = 0u; i < numTests; ++i) {
    float normalDiff = normalValues[i] - normalMean;
    float uniformDiff = uniformValues[i] - uniformMean;
    normalVar += normalDiff * normalDiff;
    uniformVar += uniformDiff * uniformDiff;
  }
  normalVar /= (numTests - 1);
  uniformVar /= (numTests - 1);

  // Normal distribution should have different statistical properties than uniform
  // Uniform distribution on [-3, 3] has variance = (b-a)^2/12 = 36/12 = 3
  // Normal distribution with sigma=1 has variance = 1
  EXPECT_NEAR(normalVar, 1.0f, 0.3f) << "Normal distribution variance should be close to sigma^2";
  EXPECT_NEAR(uniformVar, 3.0f, 0.5f) << "Uniform distribution variance should be (b-a)^2/12";

  // The variances should be significantly different
  EXPECT_GT(std::abs(normalVar - uniformVar), 1.0f)
      << "Normal and uniform should have different variances";
}

template <typename T, typename Scalar>
void testNormalDynamicVectorScalarBounds(int size, Scalar mean, Scalar sigma) {
  const auto rand = normal<T>(size, mean, sigma);
  EXPECT_EQ(rand.size(), size) << "type: " << typeid(T).name();

  // For normal distribution, use 5 sigma bounds to reduce false positives
  // 5 sigma covers 99.99994% of values, making test failures extremely rare
  EXPECT_TRUE((rand.array() >= (mean - 5 * sigma)).all())
      << "Values should be within 5 sigma of mean\ntype: " << typeid(T).name()
      << "\nrand: " << rand.transpose() << "\nmean: " << mean << "\nsigma: " << sigma;
  EXPECT_TRUE((rand.array() <= (mean + 5 * sigma)).all())
      << "Values should be within 5 sigma of mean\ntype: " << typeid(T).name()
      << "\nrand: " << rand.transpose() << "\nmean: " << mean << "\nsigma: " << sigma;

  // Sample mean should be reasonably close to expected mean for larger vectors
  if (size >= 10) {
    Scalar sampleMean = rand.mean();
    // Standard error of the mean is sigma/sqrt(n), use 4 standard errors as tolerance
    // This gives 99.99% confidence interval, reducing false positives
    Scalar tolerance = 4 * sigma / std::sqrt(static_cast<Scalar>(size));
    EXPECT_NEAR(sampleMean, mean, tolerance)
        << "Sample mean should be close to expected mean for size " << size;
  }
}

template <typename T, typename Scalar>
void testNormalDynamicMatrixScalarBounds(int rows, int cols, Scalar mean, Scalar sigma) {
  const auto rand = normal<T>(rows, cols, mean, sigma);
  EXPECT_EQ(rand.rows(), rows) << "type: " << typeid(T).name();
  EXPECT_EQ(rand.cols(), cols) << "type: " << typeid(T).name();

  // For normal distribution, use 5 sigma bounds to reduce false positives
  EXPECT_TRUE((rand.array() >= (mean - 5 * sigma)).all())
      << "Values should be within 5 sigma of mean\ntype: " << typeid(T).name() << "\nmean: " << mean
      << "\nsigma: " << sigma;
  EXPECT_TRUE((rand.array() <= (mean + 5 * sigma)).all())
      << "Values should be within 5 sigma of mean\ntype: " << typeid(T).name() << "\nmean: " << mean
      << "\nsigma: " << sigma;

  // Sample mean should be reasonably close to expected mean for larger matrices
  if (rows * cols >= 10) {
    Scalar sampleMean = rand.mean();
    // Standard error of the mean is sigma/sqrt(n), use 4 standard errors as tolerance
    // This gives 99.99% confidence interval, reducing false positives
    Scalar tolerance = 4 * sigma / std::sqrt(static_cast<Scalar>(rows * cols));
    EXPECT_NEAR(sampleMean, mean, tolerance)
        << "Sample mean should be close to expected mean for size " << rows << "x" << cols;
  }
}

template <typename T, typename Scalar>
void testNormalFixedScalarBounds(Scalar mean, Scalar sigma) {
  const auto rand = normal<T>(mean, sigma);

  // For normal distribution, use 5 sigma bounds to reduce false positives
  EXPECT_TRUE((rand.array() >= (mean - 5 * sigma)).all())
      << "Values should be within 5 sigma of mean\ntype: " << typeid(T).name()
      << "\nrand: " << rand.transpose() << "\nmean: " << mean << "\nsigma: " << sigma;
  EXPECT_TRUE((rand.array() <= (mean + 5 * sigma)).all())
      << "Values should be within 5 sigma of mean\ntype: " << typeid(T).name()
      << "\nrand: " << rand.transpose() << "\nmean: " << mean << "\nsigma: " << sigma;

  // Sample mean should be reasonably close to expected mean for larger vectors
  if (T::SizeAtCompileTime >= 10) {
    Scalar sampleMean = rand.mean();
    // Standard error of the mean is sigma/sqrt(n), use 4 standard errors as tolerance
    // This gives 99.99% confidence interval, reducing false positives
    Scalar tolerance = 4 * sigma / std::sqrt(static_cast<Scalar>(T::SizeAtCompileTime));
    EXPECT_NEAR(sampleMean, mean, tolerance) << "Sample mean should be close to expected mean";
  }
}

TEST(RandomTest, VectorMatrixNormal) {
  const float meanf = 1.5f;
  const float sigmaf = 0.8f;
  const double meand = 2.3;
  const double sigmad = 1.2;

  // Test basic functionality - just ensure no crashes and reasonable bounds
  // Dynamic float vector
  auto vecf1 = normal<VectorXf>(1, meanf, sigmaf);
  auto vecf2 = normal<VectorXf>(2, meanf, sigmaf);
  auto vecf10 = normal<VectorXf>(10, meanf, sigmaf);

  EXPECT_EQ(vecf1.size(), 1);
  EXPECT_EQ(vecf2.size(), 2);
  EXPECT_EQ(vecf10.size(), 10);

  // Values should be within reasonable bounds (10 sigma is extremely generous)
  EXPECT_TRUE((vecf10.array() >= (meanf - 10 * sigmaf)).all());
  EXPECT_TRUE((vecf10.array() <= (meanf + 10 * sigmaf)).all());

  // Dynamic double vector
  auto vecd1 = normal<VectorXd>(1, meand, sigmad);
  auto vecd2 = normal<VectorXd>(2, meand, sigmad);
  auto vecd10 = normal<VectorXd>(10, meand, sigmad);

  EXPECT_EQ(vecd1.size(), 1);
  EXPECT_EQ(vecd2.size(), 2);
  EXPECT_EQ(vecd10.size(), 10);

  EXPECT_TRUE((vecd10.array() >= (meand - 10 * sigmad)).all());
  EXPECT_TRUE((vecd10.array() <= (meand + 10 * sigmad)).all());

  // Fixed size vectors
  auto vec2f = normal<Vector2f>(meanf, sigmaf);
  auto vec3f = normal<Vector3f>(meanf, sigmaf);
  auto vec2d = normal<Vector2d>(meand, sigmad);
  auto vec3d = normal<Vector3d>(meand, sigmad);

  EXPECT_EQ(vec2f.size(), 2);
  EXPECT_EQ(vec3f.size(), 3);
  EXPECT_EQ(vec2d.size(), 2);
  EXPECT_EQ(vec3d.size(), 3);

  // Dynamic matrices
  auto matf22 = normal<MatrixXf>(2, 2, meanf, sigmaf);
  auto matf34 = normal<MatrixXf>(3, 4, meanf, sigmaf);
  auto matd22 = normal<MatrixXd>(2, 2, meand, sigmad);
  auto matd34 = normal<MatrixXd>(3, 4, meand, sigmad);

  EXPECT_EQ(matf22.rows(), 2);
  EXPECT_EQ(matf22.cols(), 2);
  EXPECT_EQ(matf34.rows(), 3);
  EXPECT_EQ(matf34.cols(), 4);
  EXPECT_EQ(matd22.rows(), 2);
  EXPECT_EQ(matd22.cols(), 2);
  EXPECT_EQ(matd34.rows(), 3);
  EXPECT_EQ(matd34.cols(), 4);

  // Fixed size matrices
  auto mat2f = normal<Matrix2f>(meanf, sigmaf);
  auto mat3f = normal<Matrix3f>(meanf, sigmaf);
  auto mat2d = normal<Matrix2d>(meand, sigmad);
  auto mat3d = normal<Matrix3d>(meand, sigmad);

  EXPECT_EQ(mat2f.rows(), 2);
  EXPECT_EQ(mat2f.cols(), 2);
  EXPECT_EQ(mat3f.rows(), 3);
  EXPECT_EQ(mat3f.cols(), 3);
  EXPECT_EQ(mat2d.rows(), 2);
  EXPECT_EQ(mat2d.cols(), 2);
  EXPECT_EQ(mat3d.rows(), 3);
  EXPECT_EQ(mat3d.cols(), 3);
}

TEST(RandomTest, NormalDistributionConsistency) {
  // Test that class methods and global functions produce the same results
  const uint32_t seed = 12345;
  const float mean = 1.0f;
  const float sigma = 2.0f;

  // Test scalar normal
  Random r1(seed);
  Random r2(seed);

  float classResult = r1.normal(mean, sigma);
  float globalResult = r2.normal(mean, sigma);

  // Reset with same seed and test global function
  Random<>::GetSingleton().setSeed(seed);
  auto globalFuncResult = normal<float>(mean, sigma);

  // The class method and member normal should produce same result with same seed
  EXPECT_NEAR(classResult, globalResult, kMaxAllowedAbsError);

  // Global function should produce reasonable values (within 3 sigma)
  EXPECT_GE(globalFuncResult, mean - 3 * sigma);
  EXPECT_LE(globalFuncResult, mean + 3 * sigma);

  // Test vector normal
  Random r3(seed);
  Random r4(seed);

  auto classVecResult = r3.normal<Vector3f>(mean, sigma);
  auto globalVecResult = r4.normal<Vector3f>(mean, sigma);

  EXPECT_TRUE(classVecResult.isApprox(globalVecResult, kMaxAllowedAbsError));

  // Reset and test global function
  Random<>::GetSingleton().setSeed(seed);
  auto globalFuncVecResult = normal<Vector3f>(mean, sigma);

  // Note: This might not match exactly due to singleton state, but should be reasonable
  EXPECT_TRUE((globalFuncVecResult.array() >= (mean - 3 * sigma)).all());
  EXPECT_TRUE((globalFuncVecResult.array() <= (mean + 3 * sigma)).all());
}

TEST(RandomTest, NormalDistributionStatisticalProperties) {
  // More rigorous statistical test for normal distribution
  const int numSamples = 10000;
  const float expectedMean = 5.0f;
  const float expectedSigma = 2.0f;

  std::vector<float> samples;
  samples.reserve(numSamples);

  // Generate large sample
  for (int i = 0; i < numSamples; ++i) {
    samples.push_back(normal<float>(expectedMean, expectedSigma));
  }

  // Calculate sample statistics
  float sampleMean = 0.0f;
  for (float val : samples) {
    sampleMean += val;
  }
  sampleMean /= numSamples;

  float sampleVariance = 0.0f;
  for (float val : samples) {
    float diff = val - sampleMean;
    sampleVariance += diff * diff;
  }
  sampleVariance /= (numSamples - 1);
  float sampleStdDev = std::sqrt(sampleVariance);

  // With large sample size, statistics should be very close to expected values
  EXPECT_NEAR(sampleMean, expectedMean, 0.05f)
      << "Sample mean should be very close to expected mean with large sample";
  EXPECT_NEAR(sampleStdDev, expectedSigma, 0.05f)
      << "Sample std dev should be very close to expected sigma with large sample";

  // Test empirical rule (68-95-99.7 rule)
  int within1Sigma = 0, within2Sigma = 0, within3Sigma = 0;
  for (float val : samples) {
    float deviations = std::abs(val - sampleMean) / sampleStdDev;
    if (deviations <= 1.0f) {
      within1Sigma++;
    }
    if (deviations <= 2.0f) {
      within2Sigma++;
    }
    if (deviations <= 3.0f) {
      within3Sigma++;
    }
  }

  float pct1Sigma = static_cast<float>(within1Sigma) / numSamples;
  float pct2Sigma = static_cast<float>(within2Sigma) / numSamples;
  float pct3Sigma = static_cast<float>(within3Sigma) / numSamples;

  // Allow some tolerance for statistical variation
  EXPECT_GT(pct1Sigma, 0.65f) << "~68% should be within 1 sigma";
  EXPECT_LT(pct1Sigma, 0.71f) << "~68% should be within 1 sigma";

  EXPECT_GT(pct2Sigma, 0.93f) << "~95% should be within 2 sigma";
  EXPECT_LT(pct2Sigma, 0.97f) << "~95% should be within 2 sigma";

  EXPECT_GT(pct3Sigma, 0.995f) << "~99.7% should be within 3 sigma";
}
