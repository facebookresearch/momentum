/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/math/types.h>
#include <momentum/solver/fwd.h>

#include <unordered_map>

namespace momentum {

/// Abstract base class for optimization objective functions
///
/// Provides the interface for computing objective function values, gradients,
/// and other derivatives needed by numerical optimization algorithms.
template <typename T>
class SolverFunctionT {
 public:
  virtual ~SolverFunctionT() = default;

  /// Evaluates the objective function at the given parameter values
  ///
  /// @param parameters Current parameter values
  /// @return Objective function value (typically sum of squared errors)
  virtual double getError(const VectorX<T>& parameters) = 0;

  /// Computes the gradient of the objective function
  ///
  /// @param parameters Current parameter values
  /// @param[out] gradient Computed gradient vector
  /// @return Objective function value
  virtual double getGradient(const VectorX<T>& parameters, VectorX<T>& gradient) = 0;

  /// Computes the Jacobian matrix for least squares problems
  ///
  /// @param parameters Current parameter values
  /// @param[out] jacobian Jacobian matrix (m×n for m residuals and n parameters)
  /// @param[out] residual Vector of residual values
  /// @param[out] actualRows Number of active residual rows
  /// @return Objective function value
  virtual double getJacobian(
      const VectorX<T>& parameters,
      MatrixX<T>& jacobian,
      VectorX<T>& residual,
      size_t& actualRows) = 0;

  /// Computes the Hessian matrix of second derivatives
  ///
  /// Default implementation throws an exception as this is rarely needed
  /// @param parameters Current parameter values
  /// @param[out] hessian Computed Hessian matrix
  virtual void getHessian(const VectorX<T>& parameters, MatrixX<T>& hessian);

  /// Computes JᵀJ and JᵀR for Gauss-Newton optimization
  ///
  /// Default implementation computes these from the Jacobian
  /// @param parameters Current parameter values
  /// @param[out] jtj Approximated Hessian matrix (JᵀJ)
  /// @param[out] jtr Gradient vector (JᵀR)
  /// @return Objective function value
  virtual double getJtJR(const VectorX<T>& parameters, MatrixX<T>& jtj, VectorX<T>& jtr);

  /// Computes sparse JᵀJ and JᵀR for large-scale problems
  ///
  /// Default implementation returns 0.0 and must be overridden for sparse optimization
  /// @param parameters Current parameter values
  /// @param[out] jtj Sparse approximated Hessian matrix
  /// @param[out] jtr Gradient vector
  /// @return Objective function value
  virtual double
  getJtJR_Sparse(const VectorX<T>& parameters, SparseMatrix<T>& jtj, VectorX<T>& jtr);

  /// Computes derivatives needed by the solver
  ///
  /// Default implementation calls getJtJR
  /// @param parameters Current parameter values
  /// @param[out] hess Hessian matrix or approximation
  /// @param[out] grad Gradient vector
  /// @return Objective function value
  virtual double
  getSolverDerivatives(const VectorX<T>& parameters, MatrixX<T>& hess, VectorX<T>& grad);

  /// Updates parameters using the computed step direction
  ///
  /// @param[in,out] parameters Current parameters, updated in-place
  /// @param gradient Step direction (typically the negative gradient)
  virtual void updateParameters(VectorX<T>& parameters, const VectorX<T>& gradient) = 0;

  /// Specifies which parameters should be optimized
  ///
  /// Default implementation does nothing
  /// @param parameterSet Bitset where each bit indicates if the corresponding parameter is enabled
  virtual void setEnabledParameters(const ParameterSet& parameterSet);

  /// Returns the total number of parameters in the optimization problem
  [[nodiscard]] size_t getNumParameters() const;

  /// Returns the number of parameters currently enabled for optimization
  [[nodiscard]] size_t getActualParameters() const;

  /// Records solver state for debugging and analysis
  ///
  /// @param[in,out] history Map to store iteration data
  /// @param iteration Current iteration number
  /// @param maxIterations Maximum number of iterations
  virtual void storeHistory(
      std::unordered_map<std::string, MatrixX<T>>& history,
      size_t iteration,
      size_t maxIterations_);

 protected:
  /// Total number of parameters in the optimization problem
  size_t numParameters_{};

  /// Number of parameters currently enabled for optimization
  ///
  /// Always less than or equal to numParameters_
  size_t actualParameters_{};
};

} // namespace momentum
