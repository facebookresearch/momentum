/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/skeleton_state.h>
#include <momentum/character_solver/fwd.h>
#include <momentum/character_solver/skeleton_solver_function.h>
#include <momentum/math/online_householder_qr.h>
#include <momentum/solver/solver.h>

#include <Eigen/Core>

namespace momentum {

/// Gauss-Newton solver with QR decomposition specific options
struct GaussNewtonSolverQROptions : SolverOptions {
  /// Damping parameter added to Hessian diagonal for numerical stability; see
  /// https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
  float regularization = 0.05f;

  /// Flag to enable line search during optimization.
  bool doLineSearch = false;

  GaussNewtonSolverQROptions() = default;

  /* implicit */ GaussNewtonSolverQROptions(const SolverOptions& baseOptions)
      : SolverOptions(baseOptions) {
    // Empty
  }
};

template <typename T>
class GaussNewtonSolverQRT : public SolverT<T> {
 public:
  GaussNewtonSolverQRT(const SolverOptions& options, SkeletonSolverFunctionT<T>* solver);
  ~GaussNewtonSolverQRT() override;

  [[nodiscard]] std::string_view getName() const override;

  void setOptions(const SolverOptions& options) final;

 protected:
  void doIteration() final;
  void initializeSolver() final;

 private:
  std::unique_ptr<SkeletonStateT<T>> skeletonState_;
  std::unique_ptr<MeshStateT<T>> meshState_;

  ResizeableMatrix<T> jacobian_;
  ResizeableMatrix<T> residual_;

  OnlineHouseholderQR<T> qrSolver_;

  float regularization_;
  bool doLineSearch_;
};

} // namespace momentum
