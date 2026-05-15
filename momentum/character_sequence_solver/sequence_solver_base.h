/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character_sequence_solver/fwd.h>
#include <momentum/character_sequence_solver/sequence_solver_function.h>
#include <momentum/solver/solver.h>

#include <algorithm>
#include <memory>
#include <vector>

namespace momentum {

namespace detail {

template <typename ErrorFunctionType>
size_t computeJacobianSize(const std::vector<std::shared_ptr<ErrorFunctionType>>& functions) {
  size_t result = 0;
  for (const auto& f : functions) {
    if (f->getWeight() <= 0) {
      continue;
    }
    result += f->getJacobianSize();
  }
  return result;
}

template <typename T>
std::vector<Eigen::Index> buildSequenceColumnIndices(
    const SequenceSolverFunctionT<T>* fn,
    size_t bandwidth) {
  std::vector<Eigen::Index> result;
  result.reserve(bandwidth * fn->getPerFrameParameterIndices().size());
  for (size_t iSubFrame = 0; iSubFrame < bandwidth; ++iSubFrame) {
    std::transform(
        fn->getPerFrameParameterIndices().begin(),
        fn->getPerFrameParameterIndices().end(),
        std::back_inserter(result),
        [offset = iSubFrame * (*fn->getParameterTransform()).numAllModelParameters()](
            Eigen::Index kParam) -> Eigen::Index { return kParam + offset; });
  }

  return result;
}

} // namespace detail

/// Shared options for sequence solvers.
struct SequenceSolverOptionsBase : SolverOptions {
  /// Regularization parameter used by the linearized sequence solve.
  float regularization = 0.05f;

  /// Flag to enable line search during optimization.
  bool doLineSearch = false;

  /// Flag to enable multithreaded sequence processing.
  bool multithreaded = false;

  /// Flag to enable a progress bar during optimization.
  bool progressBar = false;

  SequenceSolverOptionsBase() = default;

  /* implicit */ SequenceSolverOptionsBase(const SolverOptions& baseOptions)
      : SolverOptions(baseOptions) {
    // Empty
  }
};

/// Shared base class for sequence solvers.
template <typename T>
class SequenceSolverBaseT : public SolverT<T> {
 public:
  SequenceSolverBaseT(const SolverOptions& options, SequenceSolverFunctionT<T>* function)
      : SolverT<T>(options, function) {
    setOptions(options);
  }

  void setOptions(const SolverOptions& options) override {
    SolverT<T>::setOptions(options);

    if (const auto* sequenceOptions = dynamic_cast<const SequenceSolverOptionsBase*>(&options)) {
      regularization_ = sequenceOptions->regularization;
      doLineSearch_ = sequenceOptions->doLineSearch;
      multithreaded_ = sequenceOptions->multithreaded;
      progressBar_ = sequenceOptions->progressBar;
    }
  }

  // "numParameters" has a different meaning here than in Solver.
  // For Solver class, numParameters = ParameterTransform.numAllModelParameters() * numFrames; but
  // it means #variables here. To hack around this discrepancy, we update numParameters when a new
  // set of variables are defined.
  // XXX To be fixed as part of T118191244
  void setEnabledParameters(const ParameterSet& parameters) final {
    SolverT<T>::setEnabledParameters(parameters);
    this->numParameters_ = this->solverFunction_->getNumParameters();
  }

 protected:
  float regularization_ = 0.05f;
  bool doLineSearch_ = false;
  bool multithreaded_ = false;
  bool progressBar_ = false;

  // bandwidth in frames:
  size_t bandwidth_ = 0;
};

using SequenceSolverBase = SequenceSolverBaseT<float>;
using SequenceSolverBased = SequenceSolverBaseT<double>;

} // namespace momentum
