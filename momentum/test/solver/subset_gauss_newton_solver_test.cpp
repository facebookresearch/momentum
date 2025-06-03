/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <momentum/solver/solver_function.h>
#include <momentum/solver/subset_gauss_newton_solver.h>

namespace {

using namespace momentum;

// A simple mock solver function for testing
class MockSolverFunction : public SolverFunctionT<float> {
 public:
  explicit MockSolverFunction(size_t numParameters) {
    // Initialize the base class's numParameters_ member variable
    this->numParameters_ = numParameters;
    this->actualParameters_ = numParameters;
  }

  double getError(const VectorX<float>& parameters) override {
    // Simple quadratic error function: 0.5 * ||parameters||^2
    return 0.5 * parameters.squaredNorm();
  }

  double getGradient(const VectorX<float>& parameters, VectorX<float>& gradient) override {
    // Gradient of 0.5 * ||parameters||^2 is parameters
    gradient = parameters;
    return 0.5 * parameters.squaredNorm();
  }

  double getJacobian(
      const VectorX<float>& parameters,
      MatrixX<float>& jacobian,
      VectorX<float>& residual,
      size_t& actualRows) override {
    // For a quadratic function, the Jacobian is the identity matrix
    // and the residual is the parameters
    if (jacobian.rows() != numParameters_ || jacobian.cols() != numParameters_) {
      jacobian.resize(numParameters_, numParameters_);
    }
    jacobian.setIdentity();

    if (residual.size() != numParameters_) {
      residual.resize(numParameters_);
    }
    residual = parameters;

    actualRows = numParameters_;
    return 0.5 * parameters.squaredNorm();
  }

  void updateParameters(VectorX<float>& parameters, const VectorX<float>& delta) override {
    // Update parameters by subtracting delta
    parameters -= delta;
  }

  void setEnabledParameters(const ParameterSet& parameterSet) override {
    // Count the number of enabled parameters
    actualParameters_ = 0;
    for (size_t i = 0; i < numParameters_; ++i) {
      if (parameterSet.test(i)) {
        actualParameters_++;
      }
    }
  }
};

// A mock solver function that causes line search to fail
class FailingLineSearchFunction : public SolverFunctionT<float> {
 public:
  explicit FailingLineSearchFunction(size_t numParameters) {
    // Initialize the base class's numParameters_ member variable
    this->numParameters_ = numParameters;
    this->actualParameters_ = numParameters;
  }

  double getError(const VectorX<float>& /*parameters*/) override {
    // Return a constant error that doesn't decrease with parameter updates
    return 1.0;
  }

  double getGradient(const VectorX<float>& parameters, VectorX<float>& gradient) override {
    gradient = VectorX<float>::Ones(parameters.size());
    return 1.0;
  }

  double getJacobian(
      const VectorX<float>& parameters,
      MatrixX<float>& jacobian,
      VectorX<float>& residual,
      size_t& actualRows) override {
    if (jacobian.rows() != numParameters_ || jacobian.cols() != numParameters_) {
      jacobian.resize(numParameters_, numParameters_);
    }
    jacobian.setIdentity();

    if (residual.size() != numParameters_) {
      residual.resize(numParameters_);
    }
    residual = VectorX<float>::Ones(parameters.size());

    actualRows = numParameters_;
    return 1.0;
  }

  void updateParameters(VectorX<float>& parameters, const VectorX<float>& delta) override {
    // Don't actually update the parameters, which will cause line search to fail
    // because the error doesn't decrease
    (void)parameters;
    (void)delta;
  }

  void setEnabledParameters(const ParameterSet& parameterSet) override {
    actualParameters_ = 0;
    for (size_t i = 0; i < numParameters_; ++i) {
      if (parameterSet.test(i)) {
        actualParameters_++;
      }
    }
  }
};

// Simple test to verify that the Subset Gauss-Newton solver exists
TEST(SubsetGaussNewtonSolverTest, SolverExists) {
  // This test doesn't actually test any functionality,
  // it just verifies that the Subset Gauss-Newton solver can be compiled
  EXPECT_TRUE(true);
}

// Test getName method
TEST(SubsetGaussNewtonSolverTest, GetName) {
  // Create a mock solver function
  MockSolverFunction mockFunction(10);

  // Create a solver with default options
  SolverOptions options;
  SubsetGaussNewtonSolverT<float> solver(options, &mockFunction);

  // Check that the name is correct
  EXPECT_EQ(solver.getName(), "SubsetGaussNewton");
}

// Test solver options
TEST(SubsetGaussNewtonSolverTest, Options) {
  // Create default options
  SubsetGaussNewtonSolverOptions options;

  // Check default values
  EXPECT_FLOAT_EQ(options.regularization, 0.05f);
  EXPECT_FALSE(options.doLineSearch);

  // Create options from base options
  SolverOptions baseOptions;
  baseOptions.maxIterations = 100;
  baseOptions.minIterations = 10;
  baseOptions.threshold = 1e-8;
  baseOptions.verbose = true;

  SubsetGaussNewtonSolverOptions derivedOptions(baseOptions);

  // Check that base options were copied
  EXPECT_EQ(derivedOptions.maxIterations, 100);
  EXPECT_EQ(derivedOptions.minIterations, 10);
  EXPECT_FLOAT_EQ(derivedOptions.threshold, 1e-8);
  EXPECT_TRUE(derivedOptions.verbose);

  // Check that derived options have default values
  EXPECT_FLOAT_EQ(derivedOptions.regularization, 0.05f);
  EXPECT_FALSE(derivedOptions.doLineSearch);
}

// Test basic solving functionality
TEST(SubsetGaussNewtonSolverTest, BasicSolve) {
  // Create a mock solver function
  MockSolverFunction mockFunction(10);

  // Create a solver with default options
  SolverOptions options;
  options.maxIterations = 10;
  options.minIterations = 1;
  options.threshold = 1e-6;
  options.verbose = false;

  SubsetGaussNewtonSolverT<float> solver(options, &mockFunction);

  // Create initial parameters
  VectorX<float> parameters = VectorX<float>::Ones(10);

  // Solve
  double finalError = solver.solve(parameters);

  // Check that the solution is close to zero (the minimum of the quadratic function)
  EXPECT_LE(parameters.norm(), 1e-4);
  EXPECT_LE(finalError, 1e-8);
}

// Test with line search enabled
TEST(SubsetGaussNewtonSolverTest, LineSearch) {
  // Create a mock solver function
  MockSolverFunction mockFunction(10);

  // Create a solver with line search enabled
  SubsetGaussNewtonSolverOptions options;
  options.maxIterations = 10;
  options.minIterations = 1;
  options.threshold = 1e-6;
  options.verbose = false;
  options.doLineSearch = true;

  SubsetGaussNewtonSolverT<float> solver(options, &mockFunction);

  // Create initial parameters
  VectorX<float> parameters = VectorX<float>::Ones(10);

  // Solve
  double finalError = solver.solve(parameters);

  // Check that the solution is close to zero (the minimum of the quadratic function)
  EXPECT_LE(parameters.norm(), 1e-4);
  EXPECT_LE(finalError, 1e-8);
}

// Test with line search failure
TEST(SubsetGaussNewtonSolverTest, LineSearchFailure) {
  // Create a mock solver function that causes line search to fail
  FailingLineSearchFunction mockFunction(10);

  // Create a solver with line search enabled
  SubsetGaussNewtonSolverOptions options;
  options.maxIterations = 10;
  options.minIterations = 1;
  options.threshold = 1e-6;
  options.verbose = false;
  options.doLineSearch = true;

  SubsetGaussNewtonSolverT<float> solver(options, &mockFunction);

  // Create initial parameters
  VectorX<float> parameters = VectorX<float>::Ones(10);

  // Solve
  solver.solve(parameters);

  // No assertions needed, we just want to make sure the line search code path is executed
}

// Test with history storage
TEST(SubsetGaussNewtonSolverTest, HistoryStorage) {
  // Create a mock solver function
  MockSolverFunction mockFunction(10);

  // Create a solver with default options
  SolverOptions options;
  options.maxIterations = 10;
  options.minIterations = 1;
  options.threshold = 1e-6;
  options.verbose = false;

  SubsetGaussNewtonSolverT<float> solver(options, &mockFunction);

  // Enable history storage
  solver.setStoreHistory(true);

  // Create initial parameters
  VectorX<float> parameters = VectorX<float>::Ones(10);

  // Solve
  double finalError = solver.solve(parameters);
  EXPECT_LE(finalError, 1e-8);

  // Check that history was stored
  const auto& history = solver.getHistory();
  EXPECT_FALSE(history.empty());

  // Check that error history exists and is decreasing
  const auto errorIter = history.find("error");
  ASSERT_NE(errorIter, history.end());

  const Eigen::MatrixX<float>& errorHistory = errorIter->second;
  EXPECT_GT(errorHistory.rows(), 0);

  // Check that error decreases monotonically
  for (int i = 1; i < errorHistory.rows(); ++i) {
    if (errorHistory(i, 0) > 0) { // Only check valid iterations
      EXPECT_LE(errorHistory(i, 0), errorHistory(i - 1, 0));
    }
  }
}

// Test with enabled parameters subset
TEST(SubsetGaussNewtonSolverTest, EnabledParametersSubset) {
  // Create a mock solver function
  MockSolverFunction mockFunction(10);

  // Create a solver with default options
  SolverOptions options;
  options.maxIterations = 10;
  options.minIterations = 1;
  options.threshold = 1e-6;
  options.verbose = false;

  SubsetGaussNewtonSolverT<float> solver(options, &mockFunction);

  // Create initial parameters
  VectorX<float> parameters = VectorX<float>::Ones(10);
  VectorX<float> originalParameters = parameters;

  // Enable only the first half of the parameters
  ParameterSet enabledParams;
  for (size_t i = 0; i < 5; ++i) {
    enabledParams.set(i);
  }
  solver.setEnabledParameters(enabledParams);

  // Solve
  solver.solve(parameters);

  // Check that only the enabled parameters were modified
  for (size_t i = 5; i < 10; ++i) {
    EXPECT_EQ(parameters(i), originalParameters(i));
  }
}

// Test with non-contiguous enabled parameters
TEST(SubsetGaussNewtonSolverTest, NonContiguousEnabledParameters) {
  // Create a mock solver function
  MockSolverFunction mockFunction(10);

  // Create a solver with default options
  SolverOptions options;
  options.maxIterations = 10;
  options.minIterations = 1;
  options.threshold = 1e-6;
  options.verbose = false;

  SubsetGaussNewtonSolverT<float> solver(options, &mockFunction);

  // Create initial parameters
  VectorX<float> parameters = VectorX<float>::Ones(10);
  VectorX<float> originalParameters = parameters;

  // Enable non-contiguous parameters (e.g., parameters 0, 2, 4, 6, 8)
  ParameterSet enabledParams;
  for (size_t i = 0; i < 10; i += 2) {
    enabledParams.set(i);
  }
  solver.setEnabledParameters(enabledParams);

  // Solve
  solver.solve(parameters);

  // Check that only the enabled parameters were modified
  for (size_t i = 1; i < 10; i += 2) {
    EXPECT_EQ(parameters(i), originalParameters(i));
  }
}

// A simple mock solver function for testing with double precision
class MockSolverFunctionDouble : public SolverFunctionT<double> {
 public:
  explicit MockSolverFunctionDouble(size_t numParameters) {
    // Initialize the base class's numParameters_ member variable
    this->numParameters_ = numParameters;
    this->actualParameters_ = numParameters;
  }

  double getError(const VectorX<double>& parameters) override {
    // Simple quadratic error function: 0.5 * ||parameters||^2
    return 0.5 * parameters.squaredNorm();
  }

  double getGradient(const VectorX<double>& parameters, VectorX<double>& gradient) override {
    // Gradient of 0.5 * ||parameters||^2 is parameters
    gradient = parameters;
    return 0.5 * parameters.squaredNorm();
  }

  double getJacobian(
      const VectorX<double>& parameters,
      MatrixX<double>& jacobian,
      VectorX<double>& residual,
      size_t& actualRows) override {
    // For a quadratic function, the Jacobian is the identity matrix
    // and the residual is the parameters
    if (jacobian.rows() != numParameters_ || jacobian.cols() != numParameters_) {
      jacobian.resize(numParameters_, numParameters_);
    }
    jacobian.setIdentity();

    if (residual.size() != numParameters_) {
      residual.resize(numParameters_);
    }
    residual = parameters;

    actualRows = numParameters_;
    return 0.5 * parameters.squaredNorm();
  }

  void updateParameters(VectorX<double>& parameters, const VectorX<double>& delta) override {
    // Update parameters by subtracting delta
    parameters -= delta;
  }

  void setEnabledParameters(const ParameterSet& parameterSet) override {
    // Count the number of enabled parameters
    actualParameters_ = 0;
    for (size_t i = 0; i < numParameters_; ++i) {
      if (parameterSet.test(i)) {
        actualParameters_++;
      }
    }
  }
};

// Test with double precision
TEST(SubsetGaussNewtonSolverTest, DoublePrecision) {
  // Create a mock solver function with double precision
  auto mockFunction = std::make_unique<MockSolverFunctionDouble>(10);

  // Create a solver with default options
  SolverOptions options;
  options.maxIterations = 10;
  options.minIterations = 1;
  options.threshold = 1e-6;
  options.verbose = false;

  SubsetGaussNewtonSolverT<double> solver(options, mockFunction.get());

  // Create initial parameters
  VectorX<double> parameters = VectorX<double>::Ones(10);

  // Solve
  double finalError = solver.solve(parameters);

  // Check that the solution is close to zero (the minimum of the quadratic function)
  EXPECT_LE(parameters.norm(), 1e-4);
  EXPECT_LE(finalError, 1e-8);
}

} // namespace
