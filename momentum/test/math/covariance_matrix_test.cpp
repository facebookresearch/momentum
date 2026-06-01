/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/math/constants.h"
#include "momentum/math/covariance_matrix.h"
#include "momentum/math/random.h"

using namespace momentum;

using Types = testing::Types<float, double>;

template <typename T>
struct CovarianceMatrixTest : testing::Test {
  using Type = T;
  Random<> rng{12345};
};

TYPED_TEST_SUITE(CovarianceMatrixTest, Types);

TYPED_TEST(CovarianceMatrixTest, Inverse) {
  using T = typename TestFixture::Type;

  const int d = 10;
  const int q = 4;

  const T sigma = 0.1;
  const auto A = this->rng.template uniform<Eigen::MatrixX<T>>(q, d, T(-1), T(1));

  LowRankCovarianceMatrixT<T> cov;
  cov.reset(sigma, A);

  const Eigen::MatrixX<T> cov2 =
      sigma * sigma * Eigen::MatrixX<T>::Identity(d, d) + A.transpose() * A;
  const Eigen::MatrixX<T> cov2_inverse = cov2.inverse();

  const auto testVec = this->rng.template uniform<Eigen::VectorX<T>>(d, T(-1), T(1));
  const auto testMat = this->rng.template uniform<Eigen::MatrixX<T>>(d, 5, T(-1), T(1));

  EXPECT_LT((cov2 * testVec - cov.times_vec(testVec)).norm(), Eps<T>(1e-6f, 5e-15));
  EXPECT_LT((cov2 * testMat - cov.times_mat(testMat)).norm(), Eps<T>(5e-6f, 5e-15));
  // The inverse-based comparisons are conditioning-sensitive: sigma^2 = 0.01 makes the smallest
  // eigenvalue ~0.01, so the inverse amplifies float error ~100x. The float residuals therefore
  // vary across platforms/std libs and seeded draws. Observed worst cases: inv_mat ~4.3e-3 on Linux
  // (almost no margin under 5e-3), and inverse_logDeterminant 5.9e-5 on GitHub macOS (over the 5e-5
  // threshold). Use generous float tolerances so the checks stay meaningful but robust; the double
  // path is unaffected.
  EXPECT_LT((cov2_inverse * testVec - cov.inverse_times_vec(testVec)).norm(), Eps<T>(2e-2f, 5e-11));
  EXPECT_LT((cov2_inverse * testMat - cov.inverse_times_mat(testMat)).norm(), Eps<T>(2e-2f, 5e-11));
  EXPECT_NEAR(std::log(cov2.determinant()), cov.logDeterminant(), Eps<T>(5e-4f, 5e-13));
  EXPECT_NEAR(
      std::log(cov2_inverse.determinant()), cov.inverse_logDeterminant(), Eps<T>(5e-4f, 5e-13));
}
