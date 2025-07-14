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

/// Extended options specific to the Gauss-Newton optimization algorithm
struct GaussNewtonSolverOptions : SolverOptions {
  /// Damping parameter added to Hessian diagonal for numerical stability; see
  /// https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
  ///
  /// Higher values improve stability but may slow convergence
  float regularization = 0.05f;

  /// Enables backtracking line search to ensure error reduction at each step
  bool doLineSearch = false;

  /// Uses pre-computed JᵀJ and JᵀR from the solver function
  ///
  /// Can improve performance for problems with specialized structure
  bool useBlockJtJ = false;

  /// Directly computes sparse JᵀJ without dense intermediate representation
  ///
  /// Only effective when useBlockJtJ is true
  bool directSparseJtJ = false;

  /// Parameter count threshold for switching to sparse matrix operations
  ///
  /// Problems with more parameters than this threshold will use sparse solvers
  size_t sparseMatrixThreshold = 200;

  /// Default constructor
  GaussNewtonSolverOptions() = default;

  /// Construct from base solver options while preserving Gauss-Newton defaults
  /* implicit */ GaussNewtonSolverOptions(const SolverOptions& baseOptions)
      : SolverOptions(baseOptions) {}
};

/// Implementation of the Gauss-Newton optimization algorithm
///
/// Minimizes non-linear least squares problems by iteratively approximating
/// the objective function with a quadratic model based on first derivatives.
/// Supports both dense and sparse matrix operations depending on problem size.
template <typename T>
class GaussNewtonSolverT : public SolverT<T> {
 public:
  /// Creates a solver with the specified options and function to optimize
  GaussNewtonSolverT(const SolverOptions& options, SolverFunctionT<T>* solver);

  /// Returns "GaussNewton" as the solver name
  [[nodiscard]] std::string_view getName() const override;

  /// Updates solver configuration, handling both base and Gauss-Newton specific options
  void setOptions(const SolverOptions& options) final;

 protected:
  /// Performs one iteration of the Gauss-Newton algorithm
  ///
  /// Delegates to either doIterationDense or doIterationSparse based on problem size
  void doIteration() final;

  /// Initializes solver state before optimization begins
  void initializeSolver() final;

 private:
  /// Performs one iteration using dense matrix operations
  void doIterationDense();

  /// Performs one iteration using sparse matrix operations
  void doIterationSparse();

  /// Updates parameters using the computed step direction
  ///
  /// Optionally performs line search if enabled
  void updateParameters(Eigen::VectorX<T>& delta);

  /// Whether to use pre-computed JᵀJ and JᵀR from solver function
  bool useBlockJtJ_{};

  /// Whether to directly compute sparse JᵀJ without dense intermediate
  bool directSparseJtJ_{};

  /// Parameter count threshold for switching to sparse operations
  size_t sparseMatrixThreshold_{200};

  /// Whether the solver has been initialized
  bool initialized_;

  /// Whether to perform line search during parameter updates
  bool doLineSearch_;

  /// Sparse Cholesky factorization solver
  Eigen::SimplicialLLT<Eigen::SparseMatrix<T>, Eigen::Lower> lltSolver_;

  /// Sparse approximation of Hessian matrix (JᵀJ)
  Eigen::SparseMatrix<T> JtJ_;

  /// Sparse identity matrix for regularization
  Eigen::SparseMatrix<T> D_;

  /// Jacobian matrix for dense operations
  Eigen::MatrixX<T> jacobian_;

  /// Dense approximation of Hessian matrix (JᵀJ)
  Eigen::MatrixX<T> hessianApprox_;

  /// Gradient vector (JᵀR)
  Eigen::VectorX<T> JtR_;

  /// Residual vector
  Eigen::VectorX<T> residual_;

  /// Dense Cholesky factorization solver
  Eigen::LLT<Eigen::MatrixX<T>> llt_;

  /// Base regularization parameter
  T regularization_;

  /// Current regularization value (may be adjusted during iterations)
  T alpha_;

  /// Whether the current iteration uses dense operations
  bool denseIteration_{};
};

} // namespace momentum
