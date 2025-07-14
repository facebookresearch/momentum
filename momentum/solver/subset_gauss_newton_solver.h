/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/solver/fwd.h>
#include <momentum/solver/solver.h>

namespace momentum {

/// Extended options for the Subset Gauss-Newton optimization algorithm
///
/// Provides configuration specific to the subset-based Gauss-Newton solver
/// that optimizes only a selected subset of parameters
struct SubsetGaussNewtonSolverOptions : SolverOptions {
  /// Damping parameter added to Hessian diagonal for numerical stability; see
  /// https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
  ///
  /// Higher values improve stability but may slow convergence
  float regularization = 0.05f;

  /// Enables backtracking line search to ensure error reduction at each step
  bool doLineSearch = false;

  /// Default constructor
  SubsetGaussNewtonSolverOptions() = default;

  /// Construct from base solver options while preserving Subset Gauss-Newton defaults
  /* implicit */ SubsetGaussNewtonSolverOptions(const SolverOptions& baseOptions)
      : SolverOptions(baseOptions) {}
};

/// Gauss-Newton solver that optimizes only a selected subset of parameters
///
/// This specialized solver efficiently handles cases where only some parameters
/// need to be optimized while others remain fixed. It remaps the active parameters
/// to a smaller optimization problem, reducing computational cost.
template <typename T>
class SubsetGaussNewtonSolverT : public SolverT<T> {
 public:
  /// Creates a solver with the specified options and function to optimize
  SubsetGaussNewtonSolverT(const SolverOptions& options, SolverFunctionT<T>* solver);

  /// Returns "SubsetGaussNewton" as the solver name
  [[nodiscard]] std::string_view getName() const override;

  /// Updates solver configuration, handling both base and Subset Gauss-Newton specific options
  void setOptions(const SolverOptions& options) final;

  /// Specifies which parameters should be optimized
  ///
  /// Updates the internal mapping between full parameter space and optimization subset
  void setEnabledParameters(const ParameterSet& parameters) final;

 protected:
  /// Performs one iteration of the Subset Gauss-Newton algorithm
  ///
  /// Computes the Jacobian, forms a reduced optimization problem for the enabled parameters,
  /// solves it, and updates the full parameter vector
  void doIteration() final;

  /// Initializes solver state before optimization begins
  void initializeSolver() final;

 private:
  /// Whether the solver has been initialized
  bool initialized_{};

  /// Jacobian matrix for the full parameter space
  Eigen::MatrixX<T> jacobian_;

  /// Residual vector
  Eigen::VectorX<T> residual_;

  /// Hessian approximation (JᵀJ) for the subset of enabled parameters
  Eigen::MatrixX<T> subsetHessian_;

  /// Gradient vector (JᵀR) for the subset of enabled parameters
  Eigen::VectorX<T> subsetGradient_;

  /// Cholesky factorization solver for the subset problem
  Eigen::LLT<Eigen::MatrixX<T>> llt_;

  /// Regularization parameter for numerical stability
  float regularization_;

  /// Whether to perform line search during parameter updates
  bool doLineSearch_;

  /// Step direction in the full parameter space
  Eigen::VectorX<T> delta_;

  /// Step direction in the reduced parameter space
  Eigen::VectorX<T> subsetDelta_;

  /// Mapping from subset parameter indices to full parameter indices
  std::vector<int> enabledParameters_;

  /// Updates the mapping between full and subset parameter spaces
  ///
  /// Called when the set of enabled parameters changes
  void updateEnabledParameters();
};

} // namespace momentum
