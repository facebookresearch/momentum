/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character_sequence_solver/fwd.h>
#include <momentum/character_sequence_solver/multipose_solver_function.h>
#include <momentum/solver/solver.h>

namespace momentum {

/// Multipose solver specific options.
struct MultiposeSolverOptions : SolverOptions {
  /// Regularization parameter for QR decomposition.
  float regularization = 0.05f;

  MultiposeSolverOptions() = default;

  /* implicit */ MultiposeSolverOptions(const SolverOptions& baseOptions)
      : SolverOptions(baseOptions) {
    // Empty
  }
};

template <typename T>
class MultiposeSolverT : public SolverT<T> {
 public:
  MultiposeSolverT(const SolverOptions& options, MultiposeSolverFunctionT<T>* function);
  ~MultiposeSolverT() override;

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
  Eigen::MatrixX<T> jacobianBlock_;
  Eigen::VectorX<T> residualBlock_;
  std::unique_ptr<MeshStateT<T>> meshState_;

  float regularization_;
};

} // namespace momentum
