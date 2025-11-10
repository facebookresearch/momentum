/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character_sequence_solver/fwd.h>
#include <momentum/common/fwd.h>
#include <momentum/math/fwd.h>
#include <momentum/math/online_householder_qr.h>
#include <momentum/solver/solver.h>

#include <functional>
#include <span>

namespace momentum {

/// Sequence solver specific options
struct SequenceSolverOptions : SolverOptions {
  /// Regularization parameter for QR decomposition.
  float regularization = 0.05f;

  /// Flag to enable line search during optimization.
  bool doLineSearch = false;

  /// Flag to enable multithreading for the Sequence solver.
  bool multithreaded = false;

  /// Flag to enable a progress bar during optimization.
  bool progressBar = false;

  SequenceSolverOptions() = default;

  /* implicit */ SequenceSolverOptions(const SolverOptions& baseOptions)
      : SolverOptions(baseOptions) {
    // Empty
  }
};

template <typename T>
class SequenceSolverT : public SolverT<T> {
 public:
  SequenceSolverT(const SolverOptions& options, SequenceSolverFunctionT<T>* function);

  [[nodiscard]] std::string_view getName() const override;

  void setOptions(const SolverOptions& options) final;

  // "numParameters" has a different meaning here than in Solver.
  // For Solver class, numParameters = ParameterTransform.numAllModelParameters() * numFrames; but
  // it means #variables here. To hack around this discrepancy, we update numParameters when a new
  // set of variables are defined.
  // XXX To be fixed as part of T118191244
  void setEnabledParameters(const ParameterSet& parameters) final {
    SolverT<T>::setEnabledParameters(parameters);
    this->numParameters_ = this->solverFunction_->getNumParameters();
  }

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
  static double processPerFrameErrors_parallel(
      SequenceSolverFunctionT<T>* fn,
      OnlineBandedHouseholderQR<T>& qrSolver,
      ProgressBar* progress);
  static double processPerFrameErrors_serial(
      SequenceSolverFunctionT<T>* fn,
      OnlineBandedHouseholderQR<T>& qrSolver,
      ProgressBar* progress);

  static double processSequenceErrors_serial(
      SequenceSolverFunctionT<T>* fn,
      OnlineBandedHouseholderQR<T>& qrSolver,
      ProgressBar* progress);

  struct JacobianResidual {
    size_t frameIndex = SIZE_MAX;
    Eigen::MatrixX<T> jacobian;
    Eigen::VectorX<T> residual;
    double error;
    size_t nFunctions;
    Eigen::Index usedRows = 0;

    bool operator<(const JacobianResidual& rhs) const {
      return frameIndex < rhs.frameIndex;
    }

    bool operator>(const JacobianResidual& rhs) const {
      return frameIndex > rhs.frameIndex;
    }
  };

  // Here, the processJac function computes the Jacobian/residual, applies it to the banded
  // part of the matrix, and then returns the universal part.
  static double processErrorFunctions_parallel(
      const std::function<JacobianResidual(
          size_t,
          SequenceSolverFunctionT<T>*,
          OnlineBandedHouseholderQR<T>&,
          SkeletonStateT<T> skelState,
          MeshStateT<T>& meshState)>& processJac,
      SequenceSolverFunctionT<T>* fn,
      OnlineBandedHouseholderQR<T>& qrSolver,
      ProgressBar* progress);

  // Returns the [Jacobian, residual, error] for all the error functions applying to a single frame:
  static JacobianResidual computePerFrameJacobian(
      SequenceSolverFunctionT<T>* fn,
      size_t iFrame,
      SkeletonStateT<T>& skelState,
      MeshStateT<T>& meshState);

  // Returns the [Jacobian, residual, error] for all the sequence error functions starting from a
  // single frame:
  static JacobianResidual computeSequenceJacobian(
      SequenceSolverFunctionT<T>* fn,
      size_t iFrame,
      size_t bandwidth,
      std::span<const SkeletonStateT<T>> skelStates,
      std::span<const MeshStateT<T>> meshStates);

  static std::vector<Eigen::Index> buildSequenceColumnIndices(
      const SequenceSolverFunctionT<T>* fn,
      size_t bandwidth);

  float regularization_ = 0.05f;
  bool doLineSearch_ = false;
  bool multithreaded_ = false;
  bool progressBar_ = false;

  // bandwidth in frames:
  size_t bandwidth_ = 0;
};

} // namespace momentum
