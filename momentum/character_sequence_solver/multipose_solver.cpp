/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_sequence_solver/multipose_solver.h"

#include "momentum/character/skeleton_state.h"
#include "momentum/character_sequence_solver/multipose_solver_function.h"
#include "momentum/character_solver/skeleton_error_function.h"
#include "momentum/common/profile.h"
#include "momentum/math/online_householder_qr.h"

namespace momentum {

template <typename T>
MultiposeSolverT<T>::MultiposeSolverT(
    const SolverOptions& options,
    MultiposeSolverFunctionT<T>* function)
    : SolverT<T>(options, function) {
  // Set default values from MultiposeSolverOptions
  regularization_ = MultiposeSolverOptions().regularization;

  // Update values based on provided options
  setOptions(options);
}

template <typename T>
std::string_view MultiposeSolverT<T>::getName() const {
  return "MultiposeSolver";
}

template <typename T>
void MultiposeSolverT<T>::setOptions(const SolverOptions& options) {
  SolverT<T>::setOptions(options);

  if (const auto* derivedOptions = dynamic_cast<const MultiposeSolverOptions*>(&options)) {
    regularization_ = derivedOptions->regularization;
  }
}

template <typename T>
void MultiposeSolverT<T>::initializeSolver() {}

template <typename T>
void MultiposeSolverT<T>::doIteration() {
  MT_PROFILE_FUNCTION();

  MultiposeSolverFunctionT<T>* fn =
      dynamic_cast<MultiposeSolverFunctionT<T>*>(this->solverFunction_);

  const std::vector<Eigen::Index> genericParameters(
      fn->genericParameters_.begin(), fn->genericParameters_.end());
  const std::vector<Eigen::Index> universalParameters(
      fn->universalParameters_.begin(), fn->universalParameters_.end());

  fn->setFrameParametersFromJoinedParameterVector(this->parameters_);

  OnlineBlockHouseholderQR<T> qrSolver(fn->universalParameters_.size(), std::sqrt(regularization_));
  const auto nFullParameters = fn->parameterTransform_->numAllModelParameters();

  this->error_ = 0;
  for (size_t f = 0; f < fn->getNumFrames(); ++f) {
    const auto& frameParameters = fn->frameParameters_[f];
    auto& skelState = *fn->states_[f];

    skelState.set(fn->parameterTransform_->apply(frameParameters), *fn->skeleton_);
    for (auto&& solvable : fn->errorFunctions_[f]) {
      if (solvable->getWeight() <= 0.0f) {
        continue;
      }

      const size_t n = solvable->getJacobianSize();
      const size_t jacobianSize = n + 8 - (n % 8);

      jacobianBlock_.resize(jacobianSize, nFullParameters);
      jacobianBlock_.setZero();

      residualBlock_.resize(jacobianSize);
      residualBlock_.setZero();
      int rows = 0;
      this->error_ +=
          solvable->getJacobian(frameParameters, skelState, jacobianBlock_, residualBlock_, rows);

      qrSolver.addMutating(
          f,
          ColumnIndexedMatrix<Eigen::MatrixX<T>>(jacobianBlock_, genericParameters),
          ColumnIndexedMatrix<Eigen::MatrixX<T>>(jacobianBlock_, universalParameters),
          residualBlock_);
    }
  }

  // Now, extract the delta:
  // TODO line search?
  const Eigen::VectorX<T> delta = qrSolver.x_dense();
  this->solverFunction_->updateParameters(this->parameters_, delta);
}

template class MultiposeSolverT<float>;
template class MultiposeSolverT<double>;

} // namespace momentum
