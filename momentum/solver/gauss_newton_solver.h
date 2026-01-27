/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/math/online_householder_qr.h>
#include <momentum/solver/fwd.h>
#include <momentum/solver/solver.h>

namespace momentum {

/// Base options shared by all dense Gauss-Newton solver variants
struct GaussNewtonSolverBaseOptions : SolverOptions {
  /// Damping parameter added to Hessian diagonal for numerical stability; see
  /// https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
  ///
  /// Higher values improve stability but may slow convergence
  float regularization = 0.05f;

  /// Enables backtracking line search to ensure error reduction at each step
  bool doLineSearch = false;

  /// Default constructor
  GaussNewtonSolverBaseOptions() = default;

  /// Construct from base solver options while preserving Gauss-Newton defaults
  /* implicit */ GaussNewtonSolverBaseOptions(const SolverOptions& baseOptions)
      : SolverOptions(baseOptions) {}
};

/// Extended options specific to the Gauss-Newton optimization algorithm
struct GaussNewtonSolverOptions : GaussNewtonSolverBaseOptions {
  /// Uses pre-computed JᵀJ and JᵀR from the solver function
  ///
  /// Can improve performance for problems with specialized structure
  bool useBlockJtJ = false;

  /// Target number of Jacobian rows to accumulate before computing JᵀJ
  ///
  /// Controls the trade-off between memory usage and computational efficiency:
  /// - SIZE_MAX (default): Build full Jacobian, single JᵀJ multiplication (fastest, most memory)
  /// - 0: Process each error function block separately (slowest, minimum memory)
  /// - Other values: Group blocks via bin-packing up to this row count
  ///
  /// Note: The effective minimum is the largest single block size, since we can't
  /// split individual error function blocks.
  size_t targetRowsPerChunk = SIZE_MAX;

  /// Default constructor
  GaussNewtonSolverOptions() = default;

  /// Construct from base solver options while preserving Gauss-Newton defaults
  /* implicit */ GaussNewtonSolverOptions(const SolverOptions& baseOptions)
      : GaussNewtonSolverBaseOptions(baseOptions) {}
};

/// Dense implementation of the Gauss-Newton optimization algorithm
///
/// Minimizes non-linear least squares problems by iteratively approximating
/// the objective function with a quadratic model based on first derivatives.
/// Uses dense matrix operations for small to medium-sized parameter spaces.
template <typename T>
class GaussNewtonSolverT : public SolverT<T> {
 public:
  /// Creates a solver with the specified options and function to optimize
  GaussNewtonSolverT(const SolverOptions& options, SolverFunctionT<T>* solver);

  /// Returns "GaussNewton" as the solver name
  [[nodiscard]] std::string_view getName() const override;

  /// Updates solver configuration, handling both base and Gauss-Newton specific options
  void setOptions(const SolverOptions& options) final;

  // Testing/debug accessors
  const Eigen::MatrixX<T>& getLastJtJ() const {
    return hessianApprox_;
  }
  const Eigen::VectorX<T>& getLastJtR() const {
    return JtR_;
  }

 protected:
  /// Performs one iteration of the Gauss-Newton algorithm
  void doIteration() final;

  /// Initializes solver state before optimization begins
  void initializeSolver() final;

 private:
  /// Updates parameters using the computed step direction
  ///
  /// Optionally performs line search if enabled
  void updateParameters(Eigen::VectorX<T>& delta);

  /// Builds enabled parameters mapping from activeParameters_ bitset
  void updateEnabledParameters();

  /// Computes JtJ and JtR using pre-computed JtJ from solver function
  void computeJtJFromBlockJtJ();

  /// Computes JtJ and JtR from Jacobian blocks
  void computeJtJFromJacobianBlocks();

  /// Whether to use pre-computed JᵀJ and JᵀR from solver function
  bool useBlockJtJ_{};

  /// Whether to perform line search during parameter updates
  bool doLineSearch_;

  /// Jacobian matrix storage (uses ResizeableMatrix to avoid reallocations)
  ResizeableMatrix<T> jacobian_;

  /// Residual vector storage (uses ResizeableMatrix to avoid reallocations)
  ResizeableMatrix<T> residual_;

  /// Target rows per chunk for Jacobian processing
  size_t targetRowsPerChunk_;

  /// Dense approximation of Hessian matrix (JᵀJ)
  Eigen::MatrixX<T> hessianApprox_;

  /// Gradient vector (JᵀR)
  Eigen::VectorX<T> JtR_;

  /// Dense Cholesky factorization solver
  Eigen::LLT<Eigen::MatrixX<T>> llt_;

  /// Regularization parameter
  T regularization_;

  /// Mapping from subset parameter indices to full parameter indices
  std::vector<int> enabledParameters_;
};

} // namespace momentum
