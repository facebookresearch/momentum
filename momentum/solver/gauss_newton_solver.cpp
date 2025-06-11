/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/solver/gauss_newton_solver.h"

#include "momentum/common/profile.h"
#include "momentum/solver/solver_function.h"

namespace momentum {

template <typename T>
GaussNewtonSolverT<T>::GaussNewtonSolverT(const SolverOptions& options, SolverFunctionT<T>* solver)
    : SolverT<T>(options, solver) {
  initialized_ = false;

  // Set default values from GaussNewtonSolverOptions
  doLineSearch_ = GaussNewtonSolverOptions().doLineSearch;
  regularization_ = GaussNewtonSolverOptions().regularization;

  // Update values based on provided options
  setOptions(options);
}

template <typename T>
std::string_view GaussNewtonSolverT<T>::getName() const {
  return "GaussNewton";
}

template <typename T>
void GaussNewtonSolverT<T>::setOptions(const SolverOptions& options) {
  SolverT<T>::setOptions(options);

  if (const auto* derivedOptions = dynamic_cast<const GaussNewtonSolverOptions*>(&options)) {
    regularization_ = derivedOptions->regularization;
    doLineSearch_ = derivedOptions->doLineSearch;
    useBlockJtJ_ = derivedOptions->useBlockJtJ;
    directSparseJtJ_ = derivedOptions->directSparseJtJ;
    sparseMatrixThreshold_ = derivedOptions->sparseMatrixThreshold;
  }
}

template <typename T>
void GaussNewtonSolverT<T>::initializeSolver() {
  // This is called from the solver base class .solve()
  alpha_ = regularization_;
  this->newParameterPattern_ = true;
  this->lastError_ = std::numeric_limits<double>::max();

  denseIteration_ = this->numParameters_ < sparseMatrixThreshold_;
}

template <typename T>
void GaussNewtonSolverT<T>::doIteration() {
  MT_PROFILE_FUNCTION();
  if (denseIteration_) {
    doIterationDense();
  } else {
    doIterationSparse();
  }
}

template <typename T>
void GaussNewtonSolverT<T>::doIterationDense() {
  MT_PROFILE_FUNCTION();

  if (useBlockJtJ_) {
    // get JtJ and JtR pre-computed
    MT_PROFILE_EVENT("Get JtJ and JtR");
    this->error_ = this->solverFunction_->getJtJR(this->parameters_, hessianApprox_, JtR_);
  } else {
    MT_PROFILE_EVENT("Get Jacobian and compute JtJ and JtR");
    // Get the jacobian and compute JtJ and JtR here
    size_t actualRows = 0;
    this->error_ =
        this->solverFunction_->getJacobian(this->parameters_, jacobian_, residual_, actualRows);

    if (hessianApprox_.rows() != this->actualParameters_) {
      hessianApprox_.resize(this->actualParameters_, this->actualParameters_);
    }

    const auto JBlock = jacobian_.topLeftCorner(actualRows, this->actualParameters_);

    hessianApprox_.template triangularView<Eigen::Lower>() = JBlock.transpose() * JBlock;

    JtR_.noalias() = JBlock.transpose() * residual_.head(actualRows);
  }

  // calculate the step direction according to the gauss newton update
  Eigen::VectorX<T> delta = Eigen::VectorX<T>::Zero(this->numParameters_);
  {
    MT_PROFILE_EVENT("Dense gauss newton step");

    // delta = (Jt*J)^-1*Jt*r ...
    // - add some regularization to make sure the system is never unstable and explodes in weird
    // ways.
    hessianApprox_.diagonal().array() += alpha_;

    // - llt solve
    delta.head(this->actualParameters_) = llt_.compute(hessianApprox_).solve(JtR_);
  }

  updateParameters(delta);

  {
    MT_PROFILE_EVENT("Store history");
    if (this->storeHistory) {
      auto& jtjHist = this->iterationHistory_["jtj"];
      if (jtjHist.rows() !=
              gsl::narrow_cast<Eigen::Index>(this->actualParameters_ * this->getMaxIterations()) ||
          jtjHist.cols() != gsl::narrow_cast<Eigen::Index>(this->actualParameters_)) {
        jtjHist.resize(this->actualParameters_ * this->getMaxIterations(), this->actualParameters_);
      }
      jtjHist.block(
          this->iteration_ * this->actualParameters_,
          0,
          this->actualParameters_,
          this->actualParameters_) = hessianApprox_;
    }
  }
}

template <typename T>
void GaussNewtonSolverT<T>::doIterationSparse() {
  MT_PROFILE_FUNCTION();

  Eigen::VectorX<T> delta;

  if (!directSparseJtJ_) { // make a dense matrix and sparsify later
    if (useBlockJtJ_) {
      // get JtJ and JtR pre-computed
      {
        MT_PROFILE_EVENT("Get sparse JtJ and JtR");
        this->error_ = this->solverFunction_->getJtJR(this->parameters_, hessianApprox_, JtR_);
      }

      {
        MT_PROFILE_EVENT("Get sparse JtJ and JtR");
        // sparsify the system
        JtJ_ = hessianApprox_.sparseView();
      }
    } else {
      MT_PROFILE_EVENT("Sparse gauss newton step");

      // get the jacobian and residual
      size_t size = 0;
      {
        MT_PROFILE_EVENT("get Jacobian");
        this->error_ =
            this->solverFunction_->getJacobian(this->parameters_, jacobian_, residual_, size);
      }

      // sparsify the system
      const Eigen::SparseMatrix<T> sjac =
          jacobian_.topLeftCorner(size, this->actualParameters_).sparseView();
      JtR_.noalias() = sjac.transpose() * residual_.head(size);

      JtJ_ = sjac.transpose() * sjac;
    }
  } else { // directly sparse JtJ
    MT_PROFILE_EVENT("Get sparse JtJ and JtR");
    this->error_ = this->solverFunction_->getJtJR_Sparse(this->parameters_, JtJ_, JtR_);
  }

  if (D_.innerSize() != this->actualParameters_) {
    D_.resize(this->actualParameters_, this->actualParameters_);
    D_.setIdentity();
  }
  JtJ_ += D_ * alpha_;

  // Symbolic decomposition, only needed if the params pattern changed
  if (this->newParameterPattern_) {
    MT_PROFILE_EVENT("Sparse analyze");
    lltSolver_.analyzePattern(JtJ_);
    this->newParameterPattern_ = !directSparseJtJ_; // works fine for sparse matrix with explicit 0s
  }

  // Numerical update with the new coefficients
  {
    MT_PROFILE_EVENT("Sparse factorization");
    lltSolver_.factorize(JtJ_);
  }

  // Solve, compute the gauss-newton step
  {
    MT_PROFILE_EVENT("Sparse solve");
    delta.setZero(this->numParameters_);
    delta.head(this->actualParameters_) = lltSolver_.solve(JtR_);
  }

  updateParameters(delta);

  {
    MT_PROFILE_EVENT("Store history");
    if (this->storeHistory) {
      this->iterationHistory_["solver_err"].setZero(1, 1);
      if (lltSolver_.info() != Eigen::Success) {
        this->iterationHistory_["solver_err"](0, 0) = 1.0;
      }

      this->iterationHistory_["jtr_norm"].resize(1, 1);
      this->iterationHistory_["jtr_norm"](0, 0) = JtR_.norm();
    }
  }
}

template <typename T>
void GaussNewtonSolverT<T>::updateParameters(Eigen::VectorX<T>& delta) {
  if (!doLineSearch_) {
    MT_PROFILE_EVENT("Update params");
    this->solverFunction_->updateParameters(this->parameters_, delta);
    return;
  }

  MT_PROFILE_EVENT("Line search");

  static constexpr T kC1 = 1e-3;
  static constexpr T kTau = 0.5;
  static constexpr size_t kMaxLineSearchSteps = 10;

  const T scaledError = kC1 * this->error_;
  const Eigen::VectorX<T> parametersOrig = this->parameters_;
  T lineSearchScale = 1.0;

  for (size_t i = 0; i < kMaxLineSearchSteps && std::isnormal(lineSearchScale); ++i) {
    this->parameters_ = parametersOrig;

    this->solverFunction_->updateParameters(this->parameters_, lineSearchScale * delta);

    const double errorNew = this->solverFunction_->getError(this->parameters_);
    if ((this->error_ - errorNew) >= lineSearchScale * scaledError) {
      break;
    }

    // Reduce step size:
    lineSearchScale *= kTau;
  }
}

template class GaussNewtonSolverT<float>;
template class GaussNewtonSolverT<double>;

} // namespace momentum
