/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/solver/solver.h"

#include "momentum/common/checks.h"
#include "momentum/common/log.h"
#include "momentum/common/profile.h"
#include "momentum/solver/solver_function.h"

namespace momentum {

template <typename T>
SolverT<T>::SolverT(const SolverOptions& options, SolverFunctionT<T>* solver)
    : solverFunction_(solver) {
  numParameters_ = solverFunction_->getNumParameters();
  actualParameters_ = gsl::narrow<int>(numParameters_);
  parameters_.setZero(numParameters_);
  activeParameters_.flip();

  newParameterPattern_ = true;
  setOptions(options);
}

template <typename T>
void SolverT<T>::setOptions(const SolverOptions& options) {
  minIterations_ = options.minIterations;
  maxIterations_ = options.maxIterations;
  threshold_ = options.threshold;
  verbose_ = options.verbose;
}

template <typename T>
void SolverT<T>::setEnabledParameters(const ParameterSet& parameterSet) {
  activeParameters_ = parameterSet;

  newParameterPattern_ = true;

  solverFunction_->setEnabledParameters(parameterSet);
  actualParameters_ = solverFunction_->getActualParameters();
}

template <typename T>
double SolverT<T>::solve(Eigen::VectorX<T>& params) {
  MT_PROFILE_FUNCTION();

  if (storeHistory) {
    auto& parameterHistory = iterationHistory_["parameters"];
    auto& errorHistory = iterationHistory_["error"];
    auto& iterationCount = iterationHistory_["iterations"];
    if (parameterHistory.rows() != gsl::narrow_cast<Eigen::Index>(numParameters_) ||
        parameterHistory.cols() != gsl::narrow_cast<Eigen::Index>(maxIterations_)) {
      parameterHistory.resize(numParameters_, maxIterations_);
    }
    if (errorHistory.rows() != gsl::narrow_cast<Eigen::Index>(maxIterations_) ||
        errorHistory.cols() != 1) {
      errorHistory.resize(maxIterations_, 1);
    }
    if (iterationCount.rows() != 1 || iterationCount.cols() != 1) {
      iterationCount.resize(1, 1);
    }

    parameterHistory.setZero();
    errorHistory.setZero();
    iterationCount.setZero();
  }

  MT_CHECK(params.size() == gsl::narrow<Eigen::DenseIndex>(numParameters_));
  parameters_ = params;

  static_assert(
      std::is_same<decltype(error_), decltype(lastError_)>::value,
      "error and lastError_ should be of the same type");
  error_ = std::numeric_limits<decltype(error_)>::max();
  lastError_ = std::numeric_limits<decltype(lastError_)>::max();

  initializeSolver();

  iteration_ = 0;
  for (; iteration_ < maxIterations_; iteration_++) {
    // do actual iteration (iterations should update the error value)
    doIteration();

    MT_LOGI_IF(this->verbose_, "Iteration: {}, error: {}", this->iteration_, error_);

    // check for convergence
    bool converged = false;
    if (std::fabs(lastError_ - error_) / (std::fabs(error_) + std::numeric_limits<float>::min()) <=
        threshold_ * std::numeric_limits<float>::epsilon()) {
      converged = true;
    }

    // set history
    if (storeHistory) {
      auto& parameterHistory = iterationHistory_["parameters"];
      auto& errorHistory = iterationHistory_["error"];
      parameterHistory.col(iteration_) = parameters_;
      errorHistory(iteration_, 0) = gsl::narrow_cast<float>(error_);

      solverFunction_->storeHistory(iterationHistory_, iteration_, maxIterations_);
    }

    if (iteration_ >= minIterations_ && converged) {
      break;
    }

    // update last error
    lastError_ = error_;
  }

  if (storeHistory) {
    auto& iterationCount = iterationHistory_["iterations"];
    iterationCount(0, 0) = gsl::narrow_cast<float>(iteration_);
  }

  params = parameters_;
  return error_;
}

template <typename T>
const ParameterSet& SolverT<T>::getActiveParameters() const {
  return activeParameters_;
}

template <typename T>
void SolverT<T>::setParameters(const Eigen::VectorX<T>& params) {
  parameters_ = params;
}

template <typename T>
void SolverT<T>::setStoreHistory(const bool b) {
  storeHistory = b;
}

template <typename T>
const std::unordered_map<std::string, Eigen::MatrixX<T>>& SolverT<T>::getHistory() const {
  return iterationHistory_;
}

template <typename T>
size_t SolverT<T>::getMinIterations() const {
  return minIterations_;
}

template <typename T>
size_t SolverT<T>::getMaxIterations() const {
  return maxIterations_;
}

template class SolverT<float>;
template class SolverT<double>;

} // namespace momentum
