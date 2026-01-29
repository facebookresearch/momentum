/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/math/types.h>
#include <momentum/solver/solver.h>
#include <momentum/solver/solver_function.h>
#include <momentum/test/helpers/expect_throw.h>

#include <gtest/gtest.h>

#include <cstddef>

namespace {

using namespace momentum;

// A mock implementation of SolverT for testing
template <typename T>
class MockSolver : public SolverT<T> {
 public:
  MockSolver(const SolverOptions& options, SolverFunctionT<T>* solver)
      : SolverT<T>(options, solver) {}

  [[nodiscard]] std::string_view getName() const override {
    return "MockSolver";
  }

  void initializeSolver() override {
    initializeCalled = true;
  }

  void doIteration() override {
    iterationCalled = true;
    this->error_ = 0.0; // Set error to 0 to simulate convergence
  }

  bool initializeCalled = false;
  bool iterationCalled = false;
};

// A mock implementation of SolverFunctionT for testing
template <typename T>
class MockSolverFunction : public SolverFunctionT<T> {
 public:
  explicit MockSolverFunction(size_t numParameters) {
    this->numParameters_ = numParameters;
    this->actualParameters_ = numParameters;
  }

  double getError(const Eigen::VectorX<T>& /*parameters*/) override {
    return 0.0;
  }

  double getGradient(const Eigen::VectorX<T>& parameters, Eigen::VectorX<T>& gradient) override {
    gradient.setZero(parameters.size());
    return 0.0;
  }

  // Block-wise Jacobian interface implementation
  void initializeJacobianComputation(const Eigen::VectorX<T>& /* parameters */) override {}

  [[nodiscard]] size_t getJacobianBlockCount() const override {
    return 1;
  }

  [[nodiscard]] size_t getJacobianBlockSize(size_t blockIndex) const override {
    return blockIndex == 0 ? this->numParameters_ : 0;
  }

  double computeJacobianBlock(
      const Eigen::VectorX<T>& /* parameters */,
      size_t blockIndex,
      Eigen::Ref<Eigen::MatrixX<T>> jacobianBlock,
      Eigen::Ref<Eigen::VectorX<T>> residualBlock,
      size_t& actualRows) override {
    if (blockIndex >= 1) {
      actualRows = 0;
      return 0.0;
    }
    jacobianBlock.setZero();
    residualBlock.setZero();
    actualRows = this->numParameters_;
    return 0.0;
  }

  void finalizeJacobianComputation() override {}

  void updateParameters(Eigen::VectorX<T>& parameters, const Eigen::VectorX<T>& delta) override {
    parameters -= delta;
  }

  void setEnabledParameters(const ParameterSet& parameterSet) override {
    this->actualParameters_ = 0;
    for (size_t i = 0; i < this->numParameters_; ++i) {
      if (parameterSet.test(i)) {
        this->actualParameters_++;
      }
    }
  }
};

// Test fixture for SolverT tests
template <typename T>
struct SolverTest : public testing::Test {
  using Type = T;
};

using Types = testing::Types<float, double>;
TYPED_TEST_SUITE(SolverTest, Types);

// Test constructor and initialization
TYPED_TEST(SolverTest, Constructor) {
  using T = typename TestFixture::Type;

  const size_t numParameters = 10;

  auto function = std::make_unique<MockSolverFunction<T>>(numParameters);

  SolverOptions options;
  options.minIterations = 5;
  options.maxIterations = 10;
  options.threshold = 1e-6;
  options.verbose = true;

  MockSolver<T> solver(options, function.get());

  // Check that options were set correctly
  EXPECT_EQ(solver.getMinIterations(), 5);
  EXPECT_EQ(solver.getMaxIterations(), 10);

  // Check that the name is correct
  EXPECT_EQ(solver.getName(), "MockSolver");
}

// Test setOptions method
TYPED_TEST(SolverTest, SetOptions) {
  using T = typename TestFixture::Type;

  const size_t numParameters = 10;

  auto function = std::make_unique<MockSolverFunction<T>>(numParameters);

  SolverOptions options;
  options.minIterations = 5;
  options.maxIterations = 10;
  options.threshold = 1e-6;
  options.verbose = true;

  MockSolver<T> solver(options, function.get());

  // Check initial options
  EXPECT_EQ(solver.getMinIterations(), 5);
  EXPECT_EQ(solver.getMaxIterations(), 10);

  // Set new options
  SolverOptions newOptions;
  newOptions.minIterations = 2;
  newOptions.maxIterations = 20;
  newOptions.threshold = 1e-8;
  newOptions.verbose = false;

  solver.setOptions(newOptions);

  // Check that options were updated
  EXPECT_EQ(solver.getMinIterations(), 2);
  EXPECT_EQ(solver.getMaxIterations(), 20);
}

// Test setEnabledParameters method
TYPED_TEST(SolverTest, SetEnabledParameters) {
  using T = typename TestFixture::Type;

  const size_t numParameters = 10;

  auto function = std::make_unique<MockSolverFunction<T>>(numParameters);

  SolverOptions options;
  MockSolver<T> solver(options, function.get());

  // Initially all parameters should be enabled
  const auto& initialActiveParams = solver.getActiveParameters();
  for (size_t i = 0; i < numParameters; ++i) {
    EXPECT_TRUE(initialActiveParams.test(i));
  }

  // Enable only the first half of the parameters
  ParameterSet enabledParams;
  for (size_t i = 0; i < numParameters / 2; ++i) {
    enabledParams.set(i);
  }
  solver.setEnabledParameters(enabledParams);

  // Check that only the first half of the parameters are enabled
  const auto& activeParams = solver.getActiveParameters();
  for (size_t i = 0; i < numParameters; ++i) {
    if (i < numParameters / 2) {
      EXPECT_TRUE(activeParams.test(i));
    } else {
      EXPECT_FALSE(activeParams.test(i));
    }
  }
}

// Test solve method
TYPED_TEST(SolverTest, Solve) {
  using T = typename TestFixture::Type;

  const size_t numParameters = 10;

  auto function = std::make_unique<MockSolverFunction<T>>(numParameters);

  SolverOptions options;
  options.minIterations = 1;
  options.maxIterations = 10;
  options.threshold = 1e-6;
  options.verbose = false;

  MockSolver<T> solver(options, function.get());

  // Create initial parameters
  Eigen::VectorX<T> parameters = Eigen::VectorX<T>::Ones(numParameters);

  // Solve
  double finalError = solver.solve(parameters);

  // Check that initializeSolver and doIteration were called
  EXPECT_TRUE(solver.initializeCalled);
  EXPECT_TRUE(solver.iterationCalled);

  // Check that the error is 0 (as set in the mock solver)
  EXPECT_EQ(finalError, 0.0);
}

// Test solve method with history storage
TYPED_TEST(SolverTest, SolveWithHistory) {
  using T = typename TestFixture::Type;

  const size_t numParameters = 10;

  auto function = std::make_unique<MockSolverFunction<T>>(numParameters);

  SolverOptions options;
  options.minIterations = 1;
  options.maxIterations = 10;
  options.threshold = 1e-6;
  options.verbose = false;

  MockSolver<T> solver(options, function.get());

  // Enable history storage
  solver.setStoreHistory(true);

  // Create initial parameters
  Eigen::VectorX<T> parameters = Eigen::VectorX<T>::Ones(numParameters);

  // Solve
  solver.solve(parameters);

  // Check that history was stored
  const auto& history = solver.getHistory();
  EXPECT_FALSE(history.empty());

  // Check that parameters history exists
  const auto paramsIter = history.find("parameters");
  ASSERT_NE(paramsIter, history.end());

  // Check that error history exists
  const auto errorIter = history.find("error");
  ASSERT_NE(errorIter, history.end());

  // Check that iteration count exists
  const auto iterCountIter = history.find("iterations");
  ASSERT_NE(iterCountIter, history.end());
  EXPECT_EQ(iterCountIter->second(0, 0), 1.0f); // We should have 1 iteration
}

// Test setParameters method
TYPED_TEST(SolverTest, SetParameters) {
  using T = typename TestFixture::Type;

  const size_t numParameters = 10;

  auto function = std::make_unique<MockSolverFunction<T>>(numParameters);

  SolverOptions options;
  MockSolver<T> solver(options, function.get());

  // Create parameters
  Eigen::VectorX<T> parameters = Eigen::VectorX<T>::Ones(numParameters);

  // Set parameters
  solver.setParameters(parameters);

  // Solve (this will use the parameters we set)
  solver.solve(parameters);

  // Check that initializeSolver and doIteration were called
  EXPECT_TRUE(solver.initializeCalled);
  EXPECT_TRUE(solver.iterationCalled);
}

// Test convergence
TYPED_TEST(SolverTest, Convergence) {
  using T = typename TestFixture::Type;

  const size_t numParameters = 10;

  auto function = std::make_unique<MockSolverFunction<T>>(numParameters);

  SolverOptions options;
  options.minIterations = 1;
  options.maxIterations = 10;
  options.threshold = 1e-6;
  options.verbose = false;

  // Create a custom mock solver that simulates convergence
  class ConvergenceMockSolver : public MockSolver<T> {
   public:
    ConvergenceMockSolver(const SolverOptions& options, SolverFunctionT<T>* solver)
        : MockSolver<T>(options, solver) {}

    void doIteration() override {
      this->iterationCalled = true;

      // Simulate convergence by setting error to a small value
      if (this->iteration_ == 0) {
        this->error_ = 1.0;
      } else {
        this->error_ = this->lastError_ * 0.1; // Reduce error by 90% each iteration
      }
    }
  };

  ConvergenceMockSolver solver(options, function.get());

  // Enable history storage
  solver.setStoreHistory(true);

  // Create initial parameters
  Eigen::VectorX<T> parameters = Eigen::VectorX<T>::Ones(numParameters);

  // Solve
  solver.solve(parameters);

  // Check that history was stored
  const auto& history = solver.getHistory();

  // Check that error history exists
  const auto errorIter = history.find("error");
  ASSERT_NE(errorIter, history.end());

  // Check that iteration count exists
  const auto iterCountIter = history.find("iterations");
  ASSERT_NE(iterCountIter, history.end());

  // We should have more than 1 iteration due to the convergence criteria
  EXPECT_GT(iterCountIter->second(0, 0), 1.0f);
}

// Test with invalid parameters size
TYPED_TEST(SolverTest, InvalidParametersSize) {
  using T = typename TestFixture::Type;

  const size_t numParameters = 10;

  auto function = std::make_unique<MockSolverFunction<T>>(numParameters);

  SolverOptions options;
  MockSolver<T> solver(options, function.get());

  // Create parameters with wrong size
  Eigen::VectorX<T> parameters = Eigen::VectorX<T>::Ones(numParameters + 1);

  // Solve should throw an exception
  MOMENTUM_EXPECT_DEATH(solver.solve(parameters), "");
}

} // namespace
