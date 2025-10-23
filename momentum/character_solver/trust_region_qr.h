/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/mesh_state.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character_solver/fwd.h>
#include <momentum/character_solver/skeleton_solver_function.h>
#include <momentum/math/online_householder_qr.h>
#include <momentum/solver/solver.h>

#include <Eigen/Core>

namespace momentum {

/// Trust Region with QR decomposition specific options
struct TrustRegionQROptions : SolverOptions {
  /// Trust region radius parameter for the Trust Region solver with QR decomposition.
  float trustRegionRadius_ = 1.0f;

  TrustRegionQROptions() = default;

  /* implicit */ TrustRegionQROptions(const SolverOptions& baseOptions)
      : SolverOptions(baseOptions) {
    // Empty
  }
};

template <typename T>
class TrustRegionQRT : public SolverT<T> {
 public:
  TrustRegionQRT(const SolverOptions& options, SkeletonSolverFunctionT<T>* solverFun);

  [[nodiscard]] std::string_view getName() const override;

  void setOptions(const SolverOptions& options) final;

  void iter() {
    doIteration();
    this->iteration_++;
  }

  void init() {
    this->iteration_ = 0;
    initializeSolver();
  }

  Eigen::VectorX<T> getP() {
    return this->parameters_;
  }

 protected:
  void doIteration() final;
  void initializeSolver() final;

 private:
  ResizeableMatrix<T> jacobian_;
  ResizeableMatrix<T> residual_;

  Eigen::VectorX<T> gradientSub_;
  Eigen::MatrixX<T> lambdaDiag_;
  Eigen::VectorX<T> lambdaZero_;

  OnlineHouseholderQR<T> qrSolver_;

  SkeletonStateT<T> skeletonState_;
  MeshStateT<T> meshState_;

  T trustRegionRadius_;
  T maxTrustRegionRadius_ = 10.0;
  T curTrustRegionRadius_ = 1.0;
  Eigen::MatrixX<T> Rmatrix_;
};

} // namespace momentum
