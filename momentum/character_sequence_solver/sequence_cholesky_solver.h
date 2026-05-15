/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character_sequence_solver/fwd.h>
#include <momentum/character_sequence_solver/sequence_solver_base.h>

namespace momentum {

/// Sequence Cholesky solver specific options.
struct SequenceCholeskySolverOptions : SequenceSolverOptionsBase {
  /// Number of frames processed by one independent accumulation chunk.
  size_t chunkSize = 20;

  /// Target number of Jacobian rows per local J^T J multiplication.
  size_t targetRowsPerJtJChunk = 1024;

  /// Use the block LDLT factorization path when the band layout is block-compatible.
  bool useBlockLdlt = true;

  /// Accumulate compact normal equations in double precision.
  ///
  /// This is useful for float error functions where the Jacobian/residual evaluation should remain
  /// single precision, but forming `J^T J` and `J^T r` in float loses too much precision before the
  /// LDLT solve.
  bool useDoublePrecisionNormalEquations = true;

  SequenceCholeskySolverOptions() = default;

  /* implicit */ SequenceCholeskySolverOptions(const SolverOptions& baseOptions)
      : SequenceSolverOptionsBase(baseOptions) {
    // Empty
  }
};

/// Sequence solver based on banded normal equations and banded factorization.
///
/// This solver evaluates each residual/Jacobian block once, immediately folds it into compact
/// `J^T J` and `J^T r` storage, and then solves the resulting band-plus-universal system. It uses a
/// block LDLT fast path when possible and falls back to scalar banded LDLT for better numerical
/// robustness. In multithreaded mode, frame chunks accumulate local per-frame and intra-chunk
/// sequence normal equations independently. Sequence error functions that cross chunk boundaries
/// are accumulated afterward in deterministic frame order.
template <typename T>
class SequenceCholeskySolverT : public SequenceSolverBaseT<T> {
 public:
  SequenceCholeskySolverT(const SolverOptions& options, SequenceSolverFunctionT<T>* function);

  [[nodiscard]] std::string_view getName() const override;

  [[nodiscard]] double getLastNormalEquationTimeMs() const {
    return lastNormalEquationTimeMs_;
  }

  [[nodiscard]] double getLastLinearSolveTimeMs() const {
    return lastLinearSolveTimeMs_;
  }

  void setOptions(const SolverOptions& options) final;

 protected:
  void doIteration() final;
  void initializeSolver() final;

 private:
  template <typename NormalEquationScalar>
  void doIterationWithNormalEquationScalar();

  size_t chunkSize_ = 20;
  size_t targetRowsPerJtJChunk_ = 1024;
  bool useBlockLdlt_ = true;
  bool useDoublePrecisionNormalEquations_ = true;
  double lastNormalEquationTimeMs_ = 0.0;
  double lastLinearSolveTimeMs_ = 0.0;
};

} // namespace momentum
