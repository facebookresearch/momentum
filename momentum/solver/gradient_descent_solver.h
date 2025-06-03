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

/// Extended options for the Gradient Descent optimization algorithm
///
/// Provides configuration specific to the first-order gradient descent method
struct GradientDescentSolverOptions : SolverOptions {
  /// Step size for parameter updates during optimization
  ///
  /// Controls how far to move in the direction of the negative gradient
  float learningRate = 0.01f;

  /// Default constructor
  GradientDescentSolverOptions() = default;

  /// Construct from base solver options while preserving Gradient Descent defaults
  /* implicit */ GradientDescentSolverOptions(const SolverOptions& baseOptions)
      : SolverOptions(baseOptions) {}
};

/// First-order optimization algorithm that follows the negative gradient
///
/// Implements the standard gradient descent method which iteratively
/// updates parameters by moving in the direction of steepest descent
template <typename T>
class GradientDescentSolverT : public SolverT<T> {
 public:
  /// Creates a solver with the specified options and function to optimize
  GradientDescentSolverT(const SolverOptions& options, SolverFunctionT<T>* solver);

  /// Returns "GradientDescent" as the solver name
  [[nodiscard]] std::string_view getName() const override;

  /// Updates solver configuration, handling both base and Gradient Descent specific options
  void setOptions(const SolverOptions& options) final;

 protected:
  /// Performs one iteration of the gradient descent algorithm
  ///
  /// Computes the gradient and updates parameters by moving in the negative gradient direction
  void doIteration() final;

  /// Initializes solver state before optimization begins
  void initializeSolver() final;

 private:
  /// Gradient vector at current parameter values
  Eigen::VectorX<T> gradient_;

  /// Step size for parameter updates
  float learningRate_;
};

} // namespace momentum
