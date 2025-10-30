/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/solver/subset_gauss_newton_solver.h"

#include "momentum/common/profile.h"
#include "momentum/solver/solver_function.h"

#include <gsl/narrow>

namespace momentum {

template <typename T>
SubsetGaussNewtonSolverT<T>::SubsetGaussNewtonSolverT(
    const SolverOptions& options,
    SolverFunctionT<T>* solver)
    : SolverT<T>(options, solver), delta_(this->parameters_.size()) {
  this->initialized_ = false;

  // Set default values from SubsetGaussNewtonSolverOptions
  doLineSearch_ = SubsetGaussNewtonSolverOptions().doLineSearch;
  regularization_ = SubsetGaussNewtonSolverOptions().regularization;

  // Update values based on provided options
  setOptions(options);
  updateEnabledParameters();
}

template <typename T>
std::string_view SubsetGaussNewtonSolverT<T>::getName() const {
  return "SubsetGaussNewton";
}

template <typename T>
void SubsetGaussNewtonSolverT<T>::setOptions(const SolverOptions& options) {
  SolverT<T>::setOptions(options);

  if (const auto* derivedOptions = dynamic_cast<const SubsetGaussNewtonSolverOptions*>(&options)) {
    regularization_ = derivedOptions->regularization;
    doLineSearch_ = derivedOptions->doLineSearch;
  }
}

template <typename T>
void SubsetGaussNewtonSolverT<T>::updateEnabledParameters() {
  const auto nFullParams = this->parameters_.size();
  this->enabledParameters_.clear();
  this->enabledParameters_.reserve(nFullParams);
  for (int iFullParam = 0; iFullParam < nFullParams; ++iFullParam) {
    if (this->activeParameters_.test(iFullParam)) {
      this->enabledParameters_.push_back(iFullParam);
    }
  }
  delta_.setZero();

  const int nSubsetParams = gsl::narrow_cast<int>(this->enabledParameters_.size());
  this->subsetHessian_.resize(nSubsetParams, nSubsetParams);
  subsetDelta_.resize(nSubsetParams);
}

template <typename T>
void SubsetGaussNewtonSolverT<T>::setEnabledParameters(const ParameterSet& parameterSet) {
  SolverT<T>::setEnabledParameters(parameterSet);
  updateEnabledParameters();
}

template <typename T>
void SubsetGaussNewtonSolverT<T>::doIteration() {
  MT_PROFILE_FUNCTION();

  // get the jacobian and residual
  size_t jacobianRows = 0;
  this->error_ = this->solverFunction_->getJacobian(
      this->parameters_, this->jacobian_, this->residual_, jacobianRows);
  const double error_orig = this->error_;

  const int nSubsetParams = gsl::narrow_cast<int>(this->enabledParameters_.size());

  // Fill in J for this subset

  // Move the solved columns of the Jacobian into place.
  // Doing this in-place in the jacobian minimizes the total work.
  // For the common case where all the "dropped" parameters at the end,
  // this will effectively be a no-op, since enabledParameters_[iSubsetParam] = iFullParam
  // for all parameters prior to the dropped ones.
  for (int iSubsetParam = 0; iSubsetParam < nSubsetParams; ++iSubsetParam) {
    // Only move columns that need it:
    if (this->enabledParameters_[iSubsetParam] > iSubsetParam) {
      this->jacobian_.col(iSubsetParam) =
          this->jacobian_.col(this->enabledParameters_[iSubsetParam]);
    }
  }

  const auto JBlock = this->jacobian_.topLeftCorner(jacobianRows, nSubsetParams);

  // Compute grad = (J^T * r) and hess ~= (J^T * J) for the actual solved subset:
  // The gradient is just Jt * r but only keeping `jacobianRows` which can be smaller than the
  // number of rows of the `jacobian` matrix.
  this->subsetGradient_.noalias() = this->residual_.head(jacobianRows).transpose() * JBlock;

  // The Hessian takes a regularizer on top of Jt * J
  this->subsetHessian_.template triangularView<Eigen::Lower>() = JBlock.transpose() * JBlock;
  this->subsetHessian_.diagonal().array() += this->regularization_;

  // LLT only reads the lower triangular part
  subsetDelta_.noalias() = llt_.compute(this->subsetHessian_).solve(subsetGradient_);

  // We don't need to set this->delta_ to zero because it is already set to zero when
  // enabledParameters_ is updated.
  for (int iSubsetParam = 0; iSubsetParam < nSubsetParams; ++iSubsetParam) {
    const int iFullParam = this->enabledParameters_[iSubsetParam];
    delta_(iFullParam) = subsetDelta_(iSubsetParam);
  }
  // update the parameters
  if (doLineSearch_) {
    const double innerProd = -this->subsetGradient_.dot(subsetDelta_);

    // Line search:
    const float c_1 = 1e-4F;
    const float tau = 0.5F;
    float alpha = 1.0F;

    const Eigen::VectorX<T> parameters_orig = this->parameters_;
    for (size_t kStep = 0; kStep < 10 && std::fpclassify(alpha) == FP_NORMAL; ++kStep) {
      // update the this->parameters_
      this->parameters_ = parameters_orig;
      this->solverFunction_->updateParameters(this->parameters_, alpha * delta_);

      const double error_new = this->solverFunction_->getError(this->parameters_);

      if ((error_orig - error_new) >= c_1 * alpha * -innerProd) {
        break;
      }

      // Reduce step size:
      alpha = alpha * tau;
    }
  } else {
    this->solverFunction_->updateParameters(this->parameters_, delta_);
  }
}

template <typename T>
void SubsetGaussNewtonSolverT<T>::initializeSolver() {}

template class SubsetGaussNewtonSolverT<float>;
template class SubsetGaussNewtonSolverT<double>;

} // namespace momentum
