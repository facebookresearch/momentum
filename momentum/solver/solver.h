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

/// Common options for numerical optimization solvers.
struct SolverOptions {
  /// Minimum number of iterations before checking convergence criteria
  size_t minIterations = 1;

  /// Maximum number of iterations before terminating
  size_t maxIterations = 2;

  /// Convergence threshold for relative error change
  float threshold = 1.0f;

  /// Enable detailed logging during optimization
  bool verbose = false;

  /// Virtual destructor for polymorphic behavior
  virtual ~SolverOptions() = default;
};

/// Base class for numerical optimization solvers.
///
/// Provides common functionality for iterative optimization algorithms
/// including parameter management, convergence checking, and history tracking.
template <typename T>
class SolverT {
 public:
  /// Constructor with solver options and function to optimize
  SolverT(const SolverOptions& options, SolverFunctionT<T>* solver);

  virtual ~SolverT() = default;

  /// Returns the name of the solver implementation
  [[nodiscard]] virtual std::string_view getName() const = 0;

  /// Updates solver configuration with new options
  virtual void setOptions(const SolverOptions& options);

  /// Solves the optimization problem
  ///
  /// @param[in,out] params Vector of initial guess, updated with the optimized solution
  /// @return Final objective function value after optimization
  double solve(Eigen::VectorX<T>& params);

  /// Specifies which parameters should be optimized
  ///
  /// @param parameters Bitset where each bit indicates if the corresponding parameter is enabled
  virtual void setEnabledParameters(const ParameterSet& parameters);

  /// Returns the current set of enabled parameters
  [[nodiscard]] const ParameterSet& getActiveParameters() const;

  /// Sets the current parameter values without solving
  void setParameters(const Eigen::VectorX<T>& params);

  /// Controls whether to store iteration history for debugging and analysis
  void setStoreHistory(bool b);

  /// Returns the history of the solver's iterations
  ///
  /// The history contains matrices for parameters, errors, and other solver-specific data
  [[nodiscard]] const std::unordered_map<std::string, Eigen::MatrixX<T>>& getHistory() const;

  /// Returns the minimum number of iterations before checking convergence
  [[nodiscard]] size_t getMinIterations() const;

  /// Returns the maximum number of iterations before terminating
  [[nodiscard]] size_t getMaxIterations() const;

  [[nodiscard]] size_t getNumParameters() const {
    return numParameters_;
  }

  /// Returns the history of objective function values over the iterations of the solve
  [[nodiscard]] const std::vector<double>& getErrorHistory() const {
    return errorHistory_;
  }

 protected:
  /// Initializes solver state before optimization begins
  virtual void initializeSolver() = 0;

  /// Performs a single iteration of the optimization algorithm
  virtual void doIteration() = 0;

 protected:
  /// Total number of parameters in the optimization problem
  size_t numParameters_;

  /// Function to be optimized
  SolverFunctionT<T>* solverFunction_;

  /// Current parameter values
  Eigen::VectorX<T> parameters_;

  /// Bitset indicating which parameters are enabled for optimization
  ParameterSet activeParameters_;

  /// Number of parameters currently enabled for optimization
  int actualParameters_;

  /// Flag indicating parameter structure has changed
  ///
  /// Used to trigger reinitialization of solver data structures
  bool newParameterPattern_;

  /// Current iteration count
  size_t iteration_{};

  /// Current objective function value
  double error_{};

  /// Previous iteration's objective function value
  double lastError_{};

  /// Whether to record optimization progress for analysis
  bool storeHistory = false;

  /// Recorded data from optimization process
  ///
  /// Contains matrices for parameters, errors, and other solver-specific data
  std::unordered_map<std::string, Eigen::MatrixX<T>> iterationHistory_;
  /// History of objective function values over the iterations
  std::vector<double> errorHistory_;

  /// Whether to output detailed progress information
  bool verbose_ = false;

 private:
  /// Minimum iterations before checking convergence criteria
  size_t minIterations_{};

  /// Maximum iterations before terminating optimization
  size_t maxIterations_{};

  /// Relative error change threshold for convergence
  float threshold_{};
};

} // namespace momentum
