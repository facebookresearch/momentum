/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/math/online_householder_qr.h>
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
  /// Default implementation assembles the full Jacobian from blocks using the
  /// block-wise interface methods.
  /// @param parameters Current parameter values
  /// @param[out] jacobian Jacobian matrix (m×n for m residuals and n parameters)
  /// @param[out] residual Vector of residual values
  /// @param[out] actualRows Number of active residual rows
  /// @return Objective function value
  double getJacobian(
      const VectorX<T>& parameters,
      MatrixX<T>& jacobian,
      VectorX<T>& residual,
      size_t& actualRows);

  /// Computes the Hessian matrix of second derivatives
  ///
  /// Default implementation throws an exception as this is rarely needed
  /// @param parameters Current parameter values
  /// @param[out] hessian Computed Hessian matrix
  virtual void getHessian(const VectorX<T>& parameters, MatrixX<T>& hessian);

  /// Computes JᵀJ and JᵀR for Gauss-Newton optimization
  ///
  /// Default implementation computes these from the Jacobian using the block-wise interface.
  /// Derived classes that don't support the block-wise Jacobian interface should override
  /// this method to compute JᵀJ and JᵀR directly.
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

  /// Initializes Jacobian computation for the given parameters
  ///
  /// Handles expensive one-time setup that can be reused across multiple blocks
  /// (e.g., constructing SkeletonState). Must be called before iterating over blocks.
  /// @param parameters Current parameter values
  virtual void initializeJacobianComputation(const VectorX<T>& parameters) = 0;

  /// Returns the number of Jacobian blocks available
  ///
  /// Each block corresponds to one error function or constraint group
  /// @return Number of blocks that can be computed via computeJacobianBlock
  [[nodiscard]] virtual size_t getJacobianBlockCount() const = 0;

  /// Returns the maximum number of rows for a specific Jacobian block
  ///
  /// This is the maximum size; actual rows may be fewer (returned via actualRows)
  /// @param blockIndex Index of the block (0 to getJacobianBlockCount()-1)
  /// @return Maximum number of rows for this block
  [[nodiscard]] virtual size_t getJacobianBlockSize(size_t blockIndex) const = 0;

  /// Computes a specific Jacobian block
  ///
  /// Must be called after initializeJacobianComputation. The jacobianBlock and
  /// residualBlock parameters should be preallocated to the size returned by
  /// getJacobianBlockSize.
  /// @param parameters Current parameter values
  /// @param blockIndex Index of the block to compute (0 to getJacobianBlockCount()-1)
  /// @param[out] jacobianBlock Jacobian for this block (preallocated)
  /// @param[out] residualBlock Residual for this block (preallocated)
  /// @param[out] actualRows Number of active rows in this block
  /// @return Error contribution from this block
  virtual double computeJacobianBlock(
      const VectorX<T>& parameters,
      size_t blockIndex,
      Eigen::Ref<MatrixX<T>> jacobianBlock,
      Eigen::Ref<VectorX<T>> residualBlock,
      size_t& actualRows) = 0;

  /// Optional cleanup after all Jacobian blocks have been computed
  ///
  /// Called after all blocks have been processed.
  /// Default implementation does nothing.
  virtual void finalizeJacobianComputation();

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

  /// Pre-allocated temporary storage for block-wise JtJ computation
  ///
  /// Uses ResizeableMatrix to avoid reallocations when size fluctuates
  ResizeableMatrix<T> tJacobian_;
  ResizeableMatrix<T> tResidual_;
};

} // namespace momentum
