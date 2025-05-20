/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/math/types.h>
#include <momentum/solver/fwd.h>

#include <string>
#include <unordered_map>

namespace momentum {

/// Common options for solvers.
struct SolverOptions {
  /// Minimum number of iterations for the solver.
  size_t minIterations = 1;

  /// Maximum number of iterations for the solver.
  size_t maxIterations = 2;

  /// Convergence threshold for the solver.
  float threshold = 1.0f;

  /// Flag to enable verbose logging during the solver's execution.
  bool verbose = true;

  /// Virtual destructor to make this struct polymorphic.
  virtual ~SolverOptions() = default;
};

/// Base Solver class.
template <typename T>
class SolverT {
 public:
  SolverT(const SolverOptions& options, SolverFunctionT<T>* solver);

  virtual ~SolverT() = default;

  [[nodiscard]] virtual std::string_view getName() const = 0;

  virtual void setOptions(const SolverOptions& options);

  /// Solves the optimization problem.
  ///
  /// @param[in,out] params Vector of initial guess, updated with the optimized solution.
  /// @return Final objective function value after optimization.
  double solve(Eigen::VectorX<T>& params);

  virtual void setEnabledParameters(const ParameterSet& parameters);

  const ParameterSet& getActiveParameters() const;

  void setParameters(const Eigen::VectorX<T>& params);

  /// Sets the flag whether to store iteration history.
  void setStoreHistory(bool b);

  /// Returns the history of the solver's iterations.
  const std::unordered_map<std::string, Eigen::MatrixX<T>>& getHistory() const;

  size_t getMinIterations() const;

  size_t getMaxIterations() const;

 protected:
  /// Initialize the solver.
  virtual void initializeSolver() = 0;

  /// Perform a single iteration of the solver.
  virtual void doIteration() = 0;

 protected:
  /// Number of parameters in the solver.
  size_t numParameters_;

  SolverFunctionT<T>* solverFunction_;

  /// Current parameter state.
  Eigen::VectorX<T> parameters_;

  /// Enabled parameters.
  ParameterSet activeParameters_;

  /// The first n parameters are enabled.
  int actualParameters_;

  /// The parameters have been changed.
  bool newParameterPattern_;

  /// Current iteration.
  size_t iteration_{};

  /// Current iteration error.
  double error_{};

  /// Last iteration error.
  double lastError_{};

  /// Flag to store iteration history.
  bool storeHistory = false;

  /// Iteration history data.
  std::unordered_map<std::string, Eigen::MatrixX<T>> iterationHistory_;

  /// Flag to enable verbose logging during the solver's execution.
  bool verbose_;

 private:
  /// Minimum number of iterations.
  size_t minIterations_{};

  /// Maximum number of iterations.
  size_t maxIterations_{};

  /// Convergence threshold.
  float threshold_{};
};

} // namespace momentum
