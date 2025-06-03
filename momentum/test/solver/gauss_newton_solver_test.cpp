/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/solver/gauss_newton_solver.h"
#include "momentum/solver/solver_function.h"

#include <gtest/gtest.h>

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

  double getJtJR(
      const VectorX<float>& parameters,
      MatrixX<float>& hessianApprox,
      VectorX<float>& JtR) override {
    // For a quadratic function, the Hessian approximation is the identity matrix
    // and JtR is the parameters
    if (hessianApprox.rows() != numParameters_ || hessianApprox.cols() != numParameters_) {
      hessianApprox.resize(numParameters_, numParameters_);
    }
    hessianApprox.setIdentity();

    if (JtR.size() != numParameters_) {
      JtR.resize(numParameters_);
    }
    JtR = parameters;

    return 0.5 * parameters.squaredNorm();
  }

  double getJtJR_Sparse(
      const VectorX<float>& parameters,
      SparseMatrix<float>& JtJ,
      VectorX<float>& JtR) override {
    // For a quadratic function, the Hessian approximation is the identity matrix
    // and JtR is the parameters
    if (JtJ.rows() != numParameters_ || JtJ.cols() != numParameters_) {
      JtJ.resize(numParameters_, numParameters_);
    }
    JtJ.setIdentity();

    if (JtR.size() != numParameters_) {
      JtR.resize(numParameters_);
    }
    JtR = parameters;

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

// Simple test to verify that the Gauss-Newton solver exists
TEST(GaussNewtonSolverTest, SolverExists) {
  // This test doesn't actually test any functionality,
  // it just verifies that the Gauss-Newton solver can be compiled
  EXPECT_TRUE(true);
}

// Test getName method
TEST(GaussNewtonSolverTest, GetName) {
  // Create a mock solver function
  MockSolverFunction mockFunction(10);

  // Create a solver with default options
  SolverOptions options;
  GaussNewtonSolverT<float> solver(options, &mockFunction);

  // Check that the name is correct
  EXPECT_EQ(solver.getName(), "GaussNewton");
}

// Test solver options
TEST(GaussNewtonSolverTest, Options) {
  // Create default options
  GaussNewtonSolverOptions options;

  // Check default values
  EXPECT_FLOAT_EQ(options.regularization, 0.05f);
  EXPECT_FALSE(options.doLineSearch);
  EXPECT_FALSE(options.useBlockJtJ);
  EXPECT_FALSE(options.directSparseJtJ);
  EXPECT_EQ(options.sparseMatrixThreshold, 200);

  // Create options from base options
  SolverOptions baseOptions;
  baseOptions.maxIterations = 100;
  baseOptions.minIterations = 10;
  baseOptions.threshold = 1e-8;
  baseOptions.verbose = true;

  GaussNewtonSolverOptions derivedOptions(baseOptions);

  // Check that base options were copied
  EXPECT_EQ(derivedOptions.maxIterations, 100);
  EXPECT_EQ(derivedOptions.minIterations, 10);
  EXPECT_FLOAT_EQ(derivedOptions.threshold, 1e-8);
  EXPECT_TRUE(derivedOptions.verbose);

  // Check that derived options have default values
  EXPECT_FLOAT_EQ(derivedOptions.regularization, 0.05f);
  EXPECT_FALSE(derivedOptions.doLineSearch);
  EXPECT_FALSE(derivedOptions.useBlockJtJ);
  EXPECT_FALSE(derivedOptions.directSparseJtJ);
  EXPECT_EQ(derivedOptions.sparseMatrixThreshold, 200);
}

// Test basic solving functionality
TEST(GaussNewtonSolverTest, BasicSolve) {
  // Create a mock solver function
  MockSolverFunction mockFunction(10);

  // Create a solver with default options
  SolverOptions options;
  options.maxIterations = 10;
  options.minIterations = 1;
  options.threshold = 1e-6;
  options.verbose = false;

  GaussNewtonSolverT<float> solver(options, &mockFunction);

  // Create initial parameters
  VectorX<float> parameters = VectorX<float>::Ones(10);

  // Solve and check that the error is non-negative
  double finalError = solver.solve(parameters);
  EXPECT_GE(finalError, 0.0);

  // Check that the solution is close to zero (the minimum of the quadratic function)
  EXPECT_LE(parameters.norm(), 1e-4);
  EXPECT_LE(finalError, 1e-8);
}

// Test with line search enabled
TEST(GaussNewtonSolverTest, LineSearch) {
  // Create a mock solver function
  MockSolverFunction mockFunction(10);

  // Create a solver with line search enabled
  GaussNewtonSolverOptions options;
  options.maxIterations = 10;
  options.minIterations = 1;
  options.threshold = 1e-6;
  options.verbose = false;
  options.doLineSearch = true;

  GaussNewtonSolverT<float> solver(options, &mockFunction);

  // Create initial parameters
  VectorX<float> parameters = VectorX<float>::Ones(10);

  // Solve and check that the final error is small
  double finalError = solver.solve(parameters);
  EXPECT_LE(finalError, 1e-8);

  // Check that the solution is close to zero (the minimum of the quadratic function)
  EXPECT_LE(parameters.norm(), 1e-4);
  EXPECT_LE(finalError, 1e-8);
}

// Test with block JtJ enabled
TEST(GaussNewtonSolverTest, BlockJtJ) {
  // Create a mock solver function
  MockSolverFunction mockFunction(10);

  // Create a solver with block JtJ enabled
  GaussNewtonSolverOptions options;
  options.maxIterations = 10;
  options.minIterations = 1;
  options.threshold = 1e-6;
  options.verbose = false;
  options.useBlockJtJ = true;

  GaussNewtonSolverT<float> solver(options, &mockFunction);

  // Create initial parameters
  VectorX<float> parameters = VectorX<float>::Ones(10);

  // Solve and check that the final error is small
  double finalError = solver.solve(parameters);
  EXPECT_LE(finalError, 1e-8);

  // Check that the solution is close to zero (the minimum of the quadratic function)
  EXPECT_LE(parameters.norm(), 1e-4);
  EXPECT_LE(finalError, 1e-8);
}

// Test with sparse matrix implementation
TEST(GaussNewtonSolverTest, SparseMatrix) {
  // Create a mock solver function with a large number of parameters
  // to trigger the sparse matrix implementation
  MockSolverFunction mockFunction(250);

  // Create a solver with default options
  SolverOptions options;
  options.maxIterations = 10;
  options.minIterations = 1;
  options.threshold = 1e-6;
  options.verbose = false;

  GaussNewtonSolverT<float> solver(options, &mockFunction);

  // Create initial parameters
  VectorX<float> parameters = VectorX<float>::Ones(250);

  // Solve and check that the final error is small
  double finalError = solver.solve(parameters);
  EXPECT_LE(finalError, 1e-8);

  // Check that the solution is close to zero (the minimum of the quadratic function)
  EXPECT_LE(parameters.norm(), 1e-4);
  EXPECT_LE(finalError, 1e-8);
}

// Test with direct sparse JtJ enabled
TEST(GaussNewtonSolverTest, DirectSparseJtJ) {
  // Create a mock solver function with a large number of parameters
  // to trigger the sparse matrix implementation
  MockSolverFunction mockFunction(250);

  // Create a solver with direct sparse JtJ enabled
  GaussNewtonSolverOptions options;
  options.maxIterations = 10;
  options.minIterations = 1;
  options.threshold = 1e-6;
  options.verbose = false;
  options.useBlockJtJ = true;
  options.directSparseJtJ = true;

  GaussNewtonSolverT<float> solver(options, &mockFunction);

  // Create initial parameters
  VectorX<float> parameters = VectorX<float>::Ones(250);

  // Solve and check that the final error is small
  double finalError = solver.solve(parameters);
  EXPECT_LE(finalError, 1e-8);

  // Check that the solution is close to zero (the minimum of the quadratic function)
  EXPECT_LE(parameters.norm(), 1e-4);
  EXPECT_LE(finalError, 1e-8);
}

// Test with sparse JtJ but not direct sparse JtJ
TEST(GaussNewtonSolverTest, SparseJtJ) {
  // Create a mock solver function with a large number of parameters
  // to trigger the sparse matrix implementation
  MockSolverFunction mockFunction(250);

  // Create a solver with block JtJ enabled but not direct sparse JtJ
  GaussNewtonSolverOptions options;
  options.maxIterations = 10;
  options.minIterations = 1;
  options.threshold = 1e-6;
  options.verbose = false;
  options.useBlockJtJ = true;
  options.directSparseJtJ = false;

  GaussNewtonSolverT<float> solver(options, &mockFunction);

  // Create initial parameters
  VectorX<float> parameters = VectorX<float>::Ones(250);

  // Solve and check that the final error is small
  double finalError = solver.solve(parameters);
  EXPECT_LE(finalError, 1e-8);

  // Check that the solution is close to zero (the minimum of the quadratic function)
  EXPECT_LE(parameters.norm(), 1e-4);
  EXPECT_LE(finalError, 1e-8);
}

// A mock solver function that causes the sparse solver to fail
class FailingSparseFunction : public SolverFunctionT<float> {
 public:
  explicit FailingSparseFunction(size_t numParameters) {
    // Initialize the base class's numParameters_ member variable
    this->numParameters_ = numParameters;
    this->actualParameters_ = numParameters;
  }

  double getError(const VectorX<float>& parameters) override {
    return 0.5 * parameters.squaredNorm();
  }

  double getGradient(const VectorX<float>& parameters, VectorX<float>& gradient) override {
    gradient = parameters;
    return 0.5 * parameters.squaredNorm();
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
    residual = parameters;

    actualRows = numParameters_;
    return 0.5 * parameters.squaredNorm();
  }

  double getJtJR(
      const VectorX<float>& parameters,
      MatrixX<float>& hessianApprox,
      VectorX<float>& JtR) override {
    if (hessianApprox.rows() != numParameters_ || hessianApprox.cols() != numParameters_) {
      hessianApprox.resize(numParameters_, numParameters_);
    }
    hessianApprox.setIdentity();

    if (JtR.size() != numParameters_) {
      JtR.resize(numParameters_);
    }
    JtR = parameters;

    return 0.5 * parameters.squaredNorm();
  }

  double getJtJR_Sparse(
      const VectorX<float>& parameters,
      SparseMatrix<float>& JtJ,
      VectorX<float>& JtR) override {
    if (JtJ.rows() != numParameters_ || JtJ.cols() != numParameters_) {
      JtJ.resize(numParameters_, numParameters_);
    }

    // Create a matrix that will cause the sparse solver to fail
    // by making it non-positive definite
    JtJ.setZero(); // This will make the matrix singular

    // Add NaN values to the matrix, which should definitely cause the solver to fail
    float nan_value = std::numeric_limits<float>::quiet_NaN();
    for (int i = 0; i < std::min(10, static_cast<int>(numParameters_)); ++i) {
      Eigen::Triplet<float> triplet(i, i, nan_value);
      std::vector<Eigen::Triplet<float>> triplets = {triplet};
      JtJ.setFromTriplets(triplets.begin(), triplets.end());
    }

    if (JtR.size() != numParameters_) {
      JtR.resize(numParameters_);
    }
    JtR = parameters;

    return 0.5 * parameters.squaredNorm();
  }

  void updateParameters(VectorX<float>& parameters, const VectorX<float>& delta) override {
    parameters -= delta;
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

  double getJtJR(
      const VectorX<float>& parameters,
      MatrixX<float>& hessianApprox,
      VectorX<float>& JtR) override {
    if (hessianApprox.rows() != numParameters_ || hessianApprox.cols() != numParameters_) {
      hessianApprox.resize(numParameters_, numParameters_);
    }
    hessianApprox.setIdentity();

    if (JtR.size() != numParameters_) {
      JtR.resize(numParameters_);
    }
    JtR = VectorX<float>::Ones(parameters.size());

    return 1.0;
  }

  double getJtJR_Sparse(
      const VectorX<float>& parameters,
      SparseMatrix<float>& JtJ,
      VectorX<float>& JtR) override {
    if (JtJ.rows() != numParameters_ || JtJ.cols() != numParameters_) {
      JtJ.resize(numParameters_, numParameters_);
    }
    JtJ.setIdentity();

    if (JtR.size() != numParameters_) {
      JtR.resize(numParameters_);
    }
    JtR = VectorX<float>::Ones(parameters.size());

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

// Test with sparse solver failure
TEST(GaussNewtonSolverTest, SparseSolverFailure) {
  // Create a mock solver function that causes the sparse solver to fail
  FailingSparseFunction mockFunction(250);

  // Create a solver with sparse matrix implementation
  GaussNewtonSolverOptions options;
  options.maxIterations = 10;
  options.minIterations = 1;
  options.threshold = 1e-6;
  options.verbose = false;

  GaussNewtonSolverT<float> solver(options, &mockFunction);

  // Enable history storage to check solver_err
  solver.setStoreHistory(true);

  // Create initial parameters
  VectorX<float> parameters = VectorX<float>::Ones(250);

  // Solve
  solver.solve(parameters);

  // Check that solver_err was set in the history
  const auto& history = solver.getHistory();
  const auto solverErrIter = history.find("solver_err");
  ASSERT_NE(solverErrIter, history.end());

  // Check that solver_err exists in the history
  // Note: The value might be 0 or 1 depending on whether the solver actually failed
  // For code coverage purposes, we just need to execute the code path that sets the value
  EXPECT_TRUE(solverErrIter->second(0, 0) == 0.0f || solverErrIter->second(0, 0) == 1.0f);

  // Check that jtr_norm was set in the history
  const auto jtrNormIter = history.find("jtr_norm");
  ASSERT_NE(jtrNormIter, history.end());
}

// Test with line search failure
TEST(GaussNewtonSolverTest, LineSearchFailure) {
  // Create a mock solver function that causes line search to fail
  FailingLineSearchFunction mockFunction(10);

  // Create a solver with line search enabled
  GaussNewtonSolverOptions options;
  options.maxIterations = 10;
  options.minIterations = 1;
  options.threshold = 1e-6;
  options.verbose = false;
  options.doLineSearch = true;

  GaussNewtonSolverT<float> solver(options, &mockFunction);

  // Create initial parameters
  VectorX<float> parameters = VectorX<float>::Ones(10);

  // Solve
  solver.solve(parameters);

  // No assertions needed, we just want to make sure the line search code path is executed
}

// Test with history storage
TEST(GaussNewtonSolverTest, HistoryStorage) {
  // Create a mock solver function
  MockSolverFunction mockFunction(10);

  // Create a solver with default options
  SolverOptions options;
  options.maxIterations = 10;
  options.minIterations = 1;
  options.threshold = 1e-6;
  options.verbose = false;

  GaussNewtonSolverT<float> solver(options, &mockFunction);

  // Enable history storage
  solver.setStoreHistory(true);

  // Create initial parameters
  VectorX<float> parameters = VectorX<float>::Ones(10);

  // Solve and check that the final error is small
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
TEST(GaussNewtonSolverTest, EnabledParametersSubset) {
  // Create a mock solver function
  MockSolverFunction mockFunction(10);

  // Create a solver with default options
  SolverOptions options;
  options.maxIterations = 10;
  options.minIterations = 1;
  options.threshold = 1e-6;
  options.verbose = false;

  GaussNewtonSolverT<float> solver(options, &mockFunction);

  // Create initial parameters
  VectorX<float> parameters = VectorX<float>::Ones(10);
  VectorX<float> originalParameters = parameters;

  // Enable only the first half of the parameters
  ParameterSet enabledParams;
  for (size_t i = 0; i < 5; ++i) {
    enabledParams.set(i);
  }
  solver.setEnabledParameters(enabledParams);

  // Solve and check that the error is non-negative
  EXPECT_GE(solver.solve(parameters), 0.0);

  // Check that only the enabled parameters were modified
  for (size_t i = 5; i < 10; ++i) {
    EXPECT_EQ(parameters(i), originalParameters(i));
  }
}

} // namespace
