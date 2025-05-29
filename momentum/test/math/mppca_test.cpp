/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <cstddef>

#include "momentum/math/constants.h"
#include "momentum/math/mppca.h"

namespace {

using namespace momentum;

using Types = testing::Types<float, double>;

template <typename T>
struct MppcaTest : testing::Test {
  using Type = T;
};

TYPED_TEST_SUITE(MppcaTest, Types);

// Test default constructor and initial values
TYPED_TEST(MppcaTest, DefaultConstructor) {
  using T = typename TestFixture::Type;
  MppcaT<T> mppca;

  EXPECT_EQ(mppca.d, 0);
  EXPECT_EQ(mppca.p, 0);
  EXPECT_TRUE(mppca.names.empty());
  EXPECT_EQ(mppca.mu.size(), 0);
  EXPECT_TRUE(mppca.Cinv.empty());
  EXPECT_TRUE(mppca.L.empty());
  EXPECT_EQ(mppca.Rpre.size(), 0);
}

// Test the set() method with valid inputs
TYPED_TEST(MppcaTest, Set) {
  using T = typename TestFixture::Type;

  const size_t d = 3; // dimension of data
  const size_t p = 2; // number of mixture components
  const size_t q = 2; // dimension of latent space

  // Create test data
  VectorX<T> pi(p);
  pi << 0.6, 0.4;

  MatrixX<T> mu(p, d);
  mu << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;

  std::vector<MatrixX<T>> W;
  MatrixX<T> W1 = MatrixX<T>::Random(d, q);
  MatrixX<T> W2 = MatrixX<T>::Random(d, q);
  W.push_back(W1);
  W.push_back(W2);

  VectorX<T> sigma2(p);
  sigma2 << 0.1, 0.2;

  // Create and set MPPCA
  MppcaT<T> mppca;
  mppca.set(pi, mu, W, sigma2);

  // Check dimensions
  EXPECT_EQ(mppca.d, d);
  EXPECT_EQ(mppca.p, p);

  // Check mu
  EXPECT_EQ(mppca.mu.rows(), p);
  EXPECT_EQ(mppca.mu.cols(), d);
  EXPECT_TRUE(mppca.mu.isApprox(mu));

  // Check sizes of other members
  EXPECT_EQ(mppca.Cinv.size(), p);
  EXPECT_EQ(mppca.L.size(), p);
  EXPECT_EQ(mppca.Rpre.size(), p);

  // Check that Rpre values are computed (not checking exact values as they depend on internal
  // calculations)
  EXPECT_FALSE(std::isnan(mppca.Rpre(0)));
  EXPECT_FALSE(std::isnan(mppca.Rpre(1)));
}

// Test the cast() method between different types
TYPED_TEST(MppcaTest, Cast) {
  using T = typename TestFixture::Type;
  using OtherT = typename std::conditional<std::is_same<T, float>::value, double, float>::type;

  const size_t d = 3; // dimension of data
  const size_t p = 2; // number of mixture components
  const size_t q = 2; // dimension of latent space

  // Create test data
  VectorX<T> pi(p);
  pi << 0.6, 0.4;

  MatrixX<T> mu(p, d);
  mu << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;

  std::vector<MatrixX<T>> W;
  MatrixX<T> W1 = MatrixX<T>::Random(d, q);
  MatrixX<T> W2 = MatrixX<T>::Random(d, q);
  W.push_back(W1);
  W.push_back(W2);

  VectorX<T> sigma2(p);
  sigma2 << 0.1, 0.2;

  // Create and set MPPCA
  MppcaT<T> mppca;
  mppca.set(pi, mu, W, sigma2);

  // Cast to other type
  MppcaT<OtherT> mppcaCasted = mppca.template cast<OtherT>();

  // Check dimensions
  EXPECT_EQ(mppcaCasted.d, mppca.d);
  EXPECT_EQ(mppcaCasted.p, mppca.p);

  // Check mu (with appropriate epsilon for type conversion)
  for (int i = 0; i < mu.rows(); ++i) {
    for (int j = 0; j < mu.cols(); ++j) {
      EXPECT_NEAR(
          static_cast<double>(mppcaCasted.mu(i, j)), static_cast<double>(mppca.mu(i, j)), 1e-5);
    }
  }

  // Check sizes of other members
  EXPECT_EQ(mppcaCasted.Cinv.size(), mppca.Cinv.size());
  EXPECT_EQ(mppcaCasted.L.size(), mppca.L.size());
  EXPECT_EQ(mppcaCasted.Rpre.size(), mppca.Rpre.size());

  // Cast back to original type
  MppcaT<T> mppcaDoubleCasted = mppcaCasted.template cast<T>();

  // Check that double casting preserves dimensions and names
  EXPECT_EQ(mppcaDoubleCasted.d, mppca.d);
  EXPECT_EQ(mppcaDoubleCasted.p, mppca.p);
  EXPECT_EQ(mppcaDoubleCasted.names, mppca.names);

  // Check that matrices are approximately equal (with appropriate epsilon for double casting)
  EXPECT_TRUE(mppcaDoubleCasted.mu.isApprox(mppca.mu, 1e-4));

  // Check Cinv matrices
  EXPECT_EQ(mppcaDoubleCasted.Cinv.size(), mppca.Cinv.size());
  for (size_t i = 0; i < mppca.Cinv.size(); ++i) {
    EXPECT_TRUE(mppcaDoubleCasted.Cinv[i].isApprox(mppca.Cinv[i], 1e-4));
  }

  // Check L matrices
  EXPECT_EQ(mppcaDoubleCasted.L.size(), mppca.L.size());
  for (size_t i = 0; i < mppca.L.size(); ++i) {
    EXPECT_TRUE(mppcaDoubleCasted.L[i].isApprox(mppca.L[i], 1e-4));
  }

  // Check Rpre vector
  EXPECT_TRUE(mppcaDoubleCasted.Rpre.isApprox(mppca.Rpre, 1e-4));
}

// Test the isApprox() method with identical and slightly different instances
TYPED_TEST(MppcaTest, IsApprox) {
  using T = typename TestFixture::Type;

  const size_t d = 3; // dimension of data
  const size_t p = 2; // number of mixture components
  const size_t q = 2; // dimension of latent space

  // Create test data
  VectorX<T> pi(p);
  pi << 0.6, 0.4;

  MatrixX<T> mu(p, d);
  mu << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;

  std::vector<MatrixX<T>> W;
  MatrixX<T> W1 = MatrixX<T>::Random(d, q);
  MatrixX<T> W2 = MatrixX<T>::Random(d, q);
  W.push_back(W1);
  W.push_back(W2);

  VectorX<T> sigma2(p);
  sigma2 << 0.1, 0.2;

  // Create and set first MPPCA
  MppcaT<T> mppca1;
  mppca1.set(pi, mu, W, sigma2);

  // Create identical MPPCA
  MppcaT<T> mppca2;
  mppca2.set(pi, mu, W, sigma2);

  // Check that identical MPPCAs are approximately equal
  EXPECT_TRUE(mppca1.isApprox(mppca2));
  EXPECT_TRUE(mppca2.isApprox(mppca1));

  // Create slightly different MPPCA
  MatrixX<T> muDiff = mu;
  muDiff(0, 0) += static_cast<T>(0.1); // Small change to first element

  MppcaT<T> mppca3;
  mppca3.set(pi, muDiff, W, sigma2);

  // Check that different MPPCAs are not approximately equal
  EXPECT_FALSE(mppca1.isApprox(mppca3));
  EXPECT_FALSE(mppca3.isApprox(mppca1));
}

// Test with different dimensions
TYPED_TEST(MppcaTest, DifferentDimensions) {
  using T = typename TestFixture::Type;

  // Test with different dimensions
  const size_t d1 = 3; // dimension of data
  const size_t p1 = 2; // number of mixture components
  const size_t q1 = 2; // dimension of latent space

  // Create test data for first MPPCA
  VectorX<T> pi1(p1);
  pi1 << 0.6, 0.4;

  MatrixX<T> mu1(p1, d1);
  mu1 << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;

  std::vector<MatrixX<T>> W1;
  W1.push_back(MatrixX<T>::Random(d1, q1));
  W1.push_back(MatrixX<T>::Random(d1, q1));

  VectorX<T> sigma2_1(p1);
  sigma2_1 << 0.1, 0.2;

  // Create and set first MPPCA
  MppcaT<T> mppca1;
  mppca1.set(pi1, mu1, W1, sigma2_1);

  // Different dimensions
  const size_t d2 = 4; // different data dimension
  const size_t p2 = 3; // different number of components
  const size_t q2 = 2; // same latent dimension

  // Create test data for second MPPCA
  VectorX<T> pi2(p2);
  pi2 << 0.3, 0.3, 0.4;

  MatrixX<T> mu2(p2, d2);
  mu2 << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0;

  std::vector<MatrixX<T>> W2;
  W2.push_back(MatrixX<T>::Random(d2, q2));
  W2.push_back(MatrixX<T>::Random(d2, q2));
  W2.push_back(MatrixX<T>::Random(d2, q2));

  VectorX<T> sigma2_2(p2);
  sigma2_2 << 0.1, 0.2, 0.3;

  // Create and set second MPPCA
  MppcaT<T> mppca2;
  mppca2.set(pi2, mu2, W2, sigma2_2);

  // Check dimensions
  EXPECT_EQ(mppca1.d, d1);
  EXPECT_EQ(mppca1.p, p1);
  EXPECT_EQ(mppca2.d, d2);
  EXPECT_EQ(mppca2.p, p2);

  // Check that MPPCAs with different dimensions are not approximately equal
  EXPECT_FALSE(mppca1.isApprox(mppca2));
  EXPECT_FALSE(mppca2.isApprox(mppca1));
}

// Test with edge cases
TYPED_TEST(MppcaTest, EdgeCases) {
  using T = typename TestFixture::Type;

  // Test with minimal dimensions
  const size_t d = 1; // minimal dimension of data
  const size_t p = 1; // minimal number of mixture components
  const size_t q = 1; // minimal dimension of latent space

  // Create test data
  VectorX<T> pi(p);
  pi << 1.0;

  MatrixX<T> mu(p, d);
  mu << 1.0;

  std::vector<MatrixX<T>> W;
  MatrixX<T> W1(d, q);
  W1 << 0.5;
  W.push_back(W1);

  VectorX<T> sigma2(p);
  sigma2 << 0.1;

  // Create and set MPPCA
  MppcaT<T> mppca;
  mppca.set(pi, mu, W, sigma2);

  // Check dimensions
  EXPECT_EQ(mppca.d, d);
  EXPECT_EQ(mppca.p, p);

  // Check mu
  EXPECT_EQ(mppca.mu.rows(), p);
  EXPECT_EQ(mppca.mu.cols(), d);
  EXPECT_TRUE(mppca.mu.isApprox(mu));

  // Check sizes of other members
  EXPECT_EQ(mppca.Cinv.size(), p);
  EXPECT_EQ(mppca.L.size(), p);
  EXPECT_EQ(mppca.Rpre.size(), p);
}

// Test with names
TYPED_TEST(MppcaTest, Names) {
  using T = typename TestFixture::Type;

  const size_t d = 3; // dimension of data
  const size_t p = 2; // number of mixture components
  const size_t q = 2; // dimension of latent space

  // Create test data
  VectorX<T> pi(p);
  pi << 0.6, 0.4;

  MatrixX<T> mu(p, d);
  mu << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;

  std::vector<MatrixX<T>> W;
  MatrixX<T> W1 = MatrixX<T>::Random(d, q);
  MatrixX<T> W2 = MatrixX<T>::Random(d, q);
  W.push_back(W1);
  W.push_back(W2);

  VectorX<T> sigma2(p);
  sigma2 << 0.1, 0.2;

  // Create and set MPPCA
  MppcaT<T> mppca;
  mppca.set(pi, mu, W, sigma2);

  // Check that names are initialized with the correct size
  EXPECT_EQ(mppca.names.size(), d);

  // Create a copy and modify names
  MppcaT<T> mppcaCopy = mppca;
  mppcaCopy.names[0] = "X";
  mppcaCopy.names[1] = "Y";
  mppcaCopy.names[2] = "Z";

  // Check that names are different
  EXPECT_NE(mppca.names, mppcaCopy.names);

  // Check that isApprox returns false when names are different
  // The isApprox method compares names directly with ==
  EXPECT_FALSE(mppca.isApprox(mppcaCopy));
}

// Test the vecIsApprox lambda in isApprox method
TYPED_TEST(MppcaTest, VecIsApprox) {
  using T = typename TestFixture::Type;

  const size_t d = 3; // dimension of data
  const size_t p = 2; // number of mixture components
  const size_t q = 2; // dimension of latent space

  // Create test data
  VectorX<T> pi(p);
  pi << 0.6, 0.4;

  MatrixX<T> mu(p, d);
  mu << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;

  std::vector<MatrixX<T>> W;
  MatrixX<T> W1 = MatrixX<T>::Random(d, q);
  MatrixX<T> W2 = MatrixX<T>::Random(d, q);
  W.push_back(W1);
  W.push_back(W2);

  VectorX<T> sigma2(p);
  sigma2 << 0.1, 0.2;

  // Create and set first MPPCA
  MppcaT<T> mppca1;
  mppca1.set(pi, mu, W, sigma2);

  // Create a copy with different size Cinv vector
  MppcaT<T> mppcaDiffSize = mppca1;
  mppcaDiffSize.Cinv.resize(1); // Resize to different size

  // Check that isApprox returns false when vector sizes are different
  EXPECT_FALSE(mppca1.isApprox(mppcaDiffSize));

  // Create a copy with same size Cinv vector but different matrix content
  MppcaT<T> mppcaDiffContent = mppca1;
  if (!mppcaDiffContent.Cinv.empty()) {
    // Modify the first matrix in Cinv to be different
    mppcaDiffContent.Cinv[0] = mppcaDiffContent.Cinv[0] * 2.0;
  }

  // Check that isApprox returns false when matrices are not approximately equal
  EXPECT_FALSE(mppca1.isApprox(mppcaDiffContent));
}

} // namespace
