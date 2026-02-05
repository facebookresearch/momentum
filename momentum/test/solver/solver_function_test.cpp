/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/math/types.h>
#include <momentum/solver/solver_function.h>
#include <momentum/test/helpers/expect_throw.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {

using namespace momentum;

// A mock implementation of SolverFunctionT for testing
template <typename T>
class MockSolverFunction : public SolverFunctionT<T> {
 public:
  explicit MockSolverFunction(size_t numParameters) {
    this->numParameters_ = numParameters;
    this->actualParameters_ = numParameters;
  }

  double getError(const VectorX<T>& parameters) override {
    return 0.5 * parameters.squaredNorm();
  }

  double getGradient(const VectorX<T>& parameters, VectorX<T>& gradient) override {
    gradient = parameters;
    return 0.5 * parameters.squaredNorm();
  }

  // Block-wise Jacobian interface implementation
  void initializeJacobianComputation(const VectorX<T>& /* parameters */) override {
    // Nothing to initialize for this simple mock
  }

  [[nodiscard]] size_t getJacobianBlockCount() const override {
    // Single block containing all residuals
    return 1;
  }

  [[nodiscard]] size_t getJacobianBlockSize(size_t blockIndex) const override {
    if (blockIndex >= 1) {
      return 0;
    }
    return this->numParameters_;
  }

  double computeJacobianBlock(
      const VectorX<T>& parameters,
      size_t blockIndex,
      Eigen::Ref<MatrixX<T>> jacobianBlock,
      Eigen::Ref<VectorX<T>> residualBlock,
      size_t& actualRows) override {
    if (blockIndex >= 1) {
      actualRows = 0;
      return 0.0;
    }

    // For a quadratic function, the Jacobian is the identity matrix
    // and the residual is the parameters.
    // Note: The passed block may be larger than numParameters_ due to SIMD padding,
    // so we only write to the first numParameters_ rows.
    const auto n = static_cast<Eigen::Index>(this->numParameters_);
    jacobianBlock.topRows(n).setIdentity();
    residualBlock.head(n) = parameters;

    actualRows = this->numParameters_;
    return 0.5 * parameters.squaredNorm();
  }

  void finalizeJacobianComputation() override {
    // Nothing to clean up
  }

  void updateParameters(VectorX<T>& parameters, const VectorX<T>& delta) override {
    parameters -= delta;
  }

  // Override getJtJR_Sparse to test our implementation
  double getJtJR_Sparse(const VectorX<T>& parameters, SparseMatrix<T>& jtj, VectorX<T>& jtr)
      override {
    // Call the base class implementation
    return SolverFunctionT<T>::getJtJR_Sparse(parameters, jtj, jtr);
  }

  // Override getSolverDerivatives to test our implementation
  double getSolverDerivatives(const VectorX<T>& parameters, MatrixX<T>& hess, VectorX<T>& grad)
      override {
    // Call the base class implementation
    return SolverFunctionT<T>::getSolverDerivatives(parameters, hess, grad);
  }

  // Override setEnabledParameters to update the actualParameters_ member variable
  void setEnabledParameters(const ParameterSet& parameterSet) override {
    // Update the actualParameters_ member variable
    this->actualParameters_ = 0;
    for (size_t i = 0; i < this->numParameters_; ++i) {
      if (parameterSet.test(i)) {
        this->actualParameters_++;
      }
    }
  }

  // Override storeHistory to test our implementation
  void storeHistory(
      std::unordered_map<std::string, MatrixX<T>>& history,
      size_t iteration,
      size_t maxIterations) override {
    // Call the base class implementation
    SolverFunctionT<T>::storeHistory(history, iteration, maxIterations);

    // Add some custom history data for testing
    if (history.find("custom") == history.end()) {
      history["custom"].resize(1, 1);
    }
    history["custom"](0, 0) = static_cast<T>(iteration);
  }
};

// Test fixture for SolverFunctionT tests
template <typename T>
struct SolverFunctionTest : public testing::Test {
  using Type = T;
};

using Types = testing::Types<float, double>;
TYPED_TEST_SUITE(SolverFunctionTest, Types);

// Test getHessian method
TYPED_TEST(SolverFunctionTest, GetHessian) {
  using T = typename TestFixture::Type;

  const size_t numParameters = 10;

  auto function = std::make_unique<MockSolverFunction<T>>(numParameters);

  // Create parameters
  VectorX<T> parameters = VectorX<T>::Ones(numParameters);

  // Create hessian matrix
  MatrixX<T> hessian;

  // getHessian should throw an exception since it's not implemented
  EXPECT_THROW_WITH_MESSAGE(
      [&]() { function->getHessian(parameters, hessian); },
      std::runtime_error,
      testing::HasSubstr("SolverFunctionT::getHessian() is not implemented"));
}

// Test getJtJR method
TYPED_TEST(SolverFunctionTest, GetJtJR) {
  using T = typename TestFixture::Type;

  const size_t numParameters = 10;

  auto function = std::make_unique<MockSolverFunction<T>>(numParameters);

  // Create parameters
  VectorX<T> parameters = VectorX<T>::Ones(numParameters);

  // Create JtJ matrix and JtR vector
  MatrixX<T> jtj;
  VectorX<T> jtr;

  // Call getJtJR
  double error = function->getJtJR(parameters, jtj, jtr);

  // Check that the error is correct
  EXPECT_DOUBLE_EQ(error, 0.5 * numParameters);

  // Check that JtJ is the identity matrix
  EXPECT_EQ(jtj.rows(), numParameters);
  EXPECT_EQ(jtj.cols(), numParameters);
  for (size_t i = 0; i < numParameters; ++i) {
    for (size_t j = 0; j < numParameters; ++j) {
      if (i == j) {
        EXPECT_DOUBLE_EQ(jtj(i, j), 1.0);
      } else {
        EXPECT_DOUBLE_EQ(jtj(i, j), 0.0);
      }
    }
  }

  // Check that JtR is the parameters
  EXPECT_EQ(jtr.size(), numParameters);
  for (size_t i = 0; i < numParameters; ++i) {
    EXPECT_DOUBLE_EQ(jtr(i), 1.0);
  }
}

// Test getJtJR_Sparse method
TYPED_TEST(SolverFunctionTest, GetJtJR_Sparse) {
  using T = typename TestFixture::Type;

  const size_t numParameters = 10;

  auto function = std::make_unique<MockSolverFunction<T>>(numParameters);

  // Create parameters
  VectorX<T> parameters = VectorX<T>::Ones(numParameters);

  // Create JtJ matrix and JtR vector
  SparseMatrix<T> jtj;
  VectorX<T> jtr;

  // Call getJtJR_Sparse
  double error = function->getJtJR_Sparse(parameters, jtj, jtr);

  // Check that the error is 0.0 (default implementation returns 0.0)
  EXPECT_DOUBLE_EQ(error, 0.0);
}

// Test getSolverDerivatives method
TYPED_TEST(SolverFunctionTest, GetSolverDerivatives) {
  using T = typename TestFixture::Type;

  const size_t numParameters = 10;

  auto function = std::make_unique<MockSolverFunction<T>>(numParameters);

  // Create parameters
  VectorX<T> parameters = VectorX<T>::Ones(numParameters);

  // Create hessian matrix and gradient vector
  MatrixX<T> hessian;
  VectorX<T> gradient;

  // Call getSolverDerivatives
  double error = function->getSolverDerivatives(parameters, hessian, gradient);

  // Check that the error is correct
  EXPECT_DOUBLE_EQ(error, 0.5 * numParameters);

  // Check that hessian is the identity matrix
  EXPECT_EQ(hessian.rows(), numParameters);
  EXPECT_EQ(hessian.cols(), numParameters);
  for (size_t i = 0; i < numParameters; ++i) {
    for (size_t j = 0; j < numParameters; ++j) {
      if (i == j) {
        EXPECT_DOUBLE_EQ(hessian(i, j), 1.0);
      } else {
        EXPECT_DOUBLE_EQ(hessian(i, j), 0.0);
      }
    }
  }

  // Check that gradient is the parameters
  EXPECT_EQ(gradient.size(), numParameters);
  for (size_t i = 0; i < numParameters; ++i) {
    EXPECT_DOUBLE_EQ(gradient(i), 1.0);
  }
}

// Test setEnabledParameters method
TYPED_TEST(SolverFunctionTest, SetEnabledParameters) {
  using T = typename TestFixture::Type;

  const size_t numParameters = 10;

  auto function = std::make_unique<MockSolverFunction<T>>(numParameters);

  // Initially all parameters should be enabled
  EXPECT_EQ(function->getActualParameters(), numParameters);

  // Enable only the first half of the parameters
  ParameterSet enabledParams;
  for (size_t i = 0; i < numParameters / 2; ++i) {
    enabledParams.set(i);
  }
  function->setEnabledParameters(enabledParams);

  // Check that only the first half of the parameters are enabled
  EXPECT_EQ(function->getActualParameters(), numParameters / 2);
}

// Test storeHistory method
TYPED_TEST(SolverFunctionTest, StoreHistory) {
  using T = typename TestFixture::Type;

  const size_t numParameters = 10;

  auto function = std::make_unique<MockSolverFunction<T>>(numParameters);

  // Create history
  std::unordered_map<std::string, MatrixX<T>> history;

  // Call storeHistory
  function->storeHistory(history, 5, 10);

  // Check that custom history was stored
  EXPECT_TRUE(history.find("custom") != history.end());
  EXPECT_EQ(history["custom"].rows(), 1);
  EXPECT_EQ(history["custom"].cols(), 1);
  EXPECT_DOUBLE_EQ(history["custom"](0, 0), 5.0);
}

// Test getNumParameters method
TYPED_TEST(SolverFunctionTest, GetNumParameters) {
  using T = typename TestFixture::Type;

  const size_t numParameters = 10;

  auto function = std::make_unique<MockSolverFunction<T>>(numParameters);

  // Check that getNumParameters returns the correct value
  EXPECT_EQ(function->getNumParameters(), numParameters);
}

// Test getActualParameters method
TYPED_TEST(SolverFunctionTest, GetActualParameters) {
  using T = typename TestFixture::Type;

  const size_t numParameters = 10;

  auto function = std::make_unique<MockSolverFunction<T>>(numParameters);

  // Initially all parameters should be enabled
  EXPECT_EQ(function->getActualParameters(), numParameters);

  // Enable only the first half of the parameters
  ParameterSet enabledParams;
  for (size_t i = 0; i < numParameters / 2; ++i) {
    enabledParams.set(i);
  }
  function->setEnabledParameters(enabledParams);

  // Check that getActualParameters returns the correct value
  EXPECT_EQ(function->getActualParameters(), numParameters / 2);
}

} // namespace
