/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/trust_region_qr.h"

#include "momentum/character/mesh_state.h"
#include "momentum/character_solver/skeleton_error_function.h"
#include "momentum/common/log.h"
#include "momentum/common/profile.h"

#include <cfloat>

namespace momentum {

template <typename T>
TrustRegionQRT<T>::TrustRegionQRT(
    const SolverOptions& options,
    SkeletonSolverFunctionT<T>* solverFun)
    : SolverT<T>(options, solverFun), qrSolver_(0) {
  // Set default values from TrustRegionQROptions
  trustRegionRadius_ = TrustRegionQROptions().trustRegionRadius_;

  // Update values based on provided options
  setOptions(options);
}

template <typename T>
std::string_view TrustRegionQRT<T>::getName() const {
  return "TrustRegionQR";
}

template <typename T>
void TrustRegionQRT<T>::initializeSolver() {
  this->curTrustRegionRadius_ = this->trustRegionRadius_;
}

template <typename T>
void TrustRegionQRT<T>::setOptions(const SolverOptions& options) {
  SolverT<T>::setOptions(options);

  if (const auto* derivedOptions = dynamic_cast<const TrustRegionQROptions*>(&options)) {
    this->trustRegionRadius_ = derivedOptions->trustRegionRadius_;
  }
}

template <typename T>
void TrustRegionQRT<T>::doIteration() {
  MT_PROFILE_FUNCTION();

  const auto sf = static_cast<SkeletonSolverFunctionT<T>*>(this->solverFunction_);

  const auto skeleton = sf->getSkeleton();
  const auto parameterTransform = sf->getParameterTransform();

  {
    MT_PROFILE_EVENT("JtJR - update state");
    skeletonState_.set(parameterTransform->apply(this->parameters_), *skeleton);
  }

  if (sf->needsMeshState()) {
    MT_PROFILE_EVENT("JtJR - update mesh state");
    meshState_.update(this->parameters_, skeletonState_, sf->getCharacter());
  }

  const auto nFullParams = gsl::narrow_cast<Eigen::Index>(this->parameters_.size());

  std::vector<Eigen::Index> enabledParameters;
  enabledParameters.reserve(nFullParams);
  for (Eigen::Index iFullParam = 0; iFullParam < nFullParams; ++iFullParam) {
    if (this->activeParameters_.test(iFullParam)) {
      enabledParameters.push_back(iFullParam);
    }
  }

  const Eigen::Index nSubsetParams = gsl::narrow_cast<Eigen::Index>(enabledParameters.size());

  // Add a tiny lambda just to make sure we don't divide by zero when back-substituting into R,
  // this isn't intended to regularize the problem at all (searching for the actual lambda parameter
  // that respects the trust region radius happens below).
  T lambda = 1e-10;
  this->qrSolver_.reset(nSubsetParams, lambda);

  // Compute the QR factorization of the Jacobian:
  double error_orig = 0.0;
  for (auto errorFunction : sf->getErrorFunctions()) {
    if (errorFunction->getWeight() <= 0) {
      continue;
    }

    auto rows = errorFunction->getJacobianSize();

    jacobian_.resizeAndSetZero(rows, nFullParams);
    residual_.resizeAndSetZero(rows);

    int usedRows = 0;
    error_orig += errorFunction->getJacobian(
        this->parameters_, skeletonState_, meshState_, jacobian_.mat(), residual_.mat(), usedRows);
    if (usedRows == 0) {
      continue;
    }

    this->qrSolver_.addMutating(
        ColumnIndexedMatrix<Eigen::MatrixX<T>>(
            jacobian_.mat().topRows(usedRows), enabledParameters),
        residual_.mat().topRows(usedRows));
  }

  this->error_ = error_orig;

  // Because the error function is computed r^t*r, the gradient is
  // 2*J^T*r.
  this->gradientSub_ = T(2) * qrSolver_.At_times_b();

  // Save the R matrix before we start adding lambdas:
  this->Rmatrix_ = this->qrSolver_.R();

  // The Hessian is approximated by 2*J^T*J.
  // In the QR solver, we compute J = Q*R.
  //    2*J^T*J = (Q*R)^T * (Q*R)
  //            = R^T * Q^T * Q * R
  //            = 2 * R^T * R
  // To compute v^T * H * v, we can then compute
  //            2 * ||R * v||^2

  // This is the local approximation to the function used in the trust region solve:
  auto evalQuadraticModel = [&](Eigen::Ref<const Eigen::VectorX<T>> p) -> T {
    // MT_LOGI("In evalQuadraticModel(); orig error: {}", this->error_);
    T result = this->error_;
    result -= this->gradientSub_.dot(p);
    // Normally in textbooks you'd see this multiplied by 1/2 for the proper Taylor series
    // expansion but remember that our Hessian is actually 2*J^T*J:
    result += (this->Rmatrix_.template triangularView<Eigen::Upper>() * p).squaredNorm();
    return result;
  };

  auto subsetToFullVector = [&](Eigen::Ref<const Eigen::VectorX<T>> v_subset) -> Eigen::VectorX<T> {
    Eigen::VectorX<T> result = Eigen::VectorX<T>::Zero(nFullParams);
    for (Eigen::Index iSubsetParam = 0; iSubsetParam < nSubsetParams; ++iSubsetParam) {
      const Eigen::Index iFullParam = enabledParameters[iSubsetParam];
      result[iFullParam] = v_subset[iSubsetParam];
    }
    return result;
  };

  // TODO what is a good value for nu?
  const T nu = 0;

  for (size_t iTrustStep = 0; iTrustStep < 10; ++iTrustStep) {
    Eigen::VectorX<T> searchDir_sub = this->qrSolver_.result();

    // If even our most optimistic guess of how much we can decrease the error
    // suggests that it's not going to change by a measurable amount (FLT_EPS times its current
    // value), then stop trying to iterate.  If we don't do that we run into issues below where the
    // rho value is just capturing noise and our trust region will drop to 0.
    if (searchDir_sub.dot(gradientSub_) < FLT_EPSILON * (T(1.0) + this->error_)) {
      break;
    }

    // We use a standard Newton approach for quickly converging to the lambda that limits
    // the step size to the trust region.  This is described in section 4.3 of Nocedal/Wright,
    // "Iterative solution of the subproblem."
    //
    // We will only allow the value of lambda to _increase_, which corresponds to shrinking the
    // trust region.  That is, we start with a trust region that is too large, and then we
    // iteratively shrink it until it's close enough to the desired delta.  This is because (1) in
    // practice, this is what seems to happen anyways, given that we start from 0 and (2) it's a lot
    // easier to add to lambda than subtract from lambda; the former can always be done by just
    // adding some additional rows to the Jacobian while the latter requires "backing out" to a
    // previous version of the QR solver.  This is okay because it's always more acceptable (within
    // reason) to take a too-small step than a too-large one:
    for (size_t iIter = 0; iIter < 3; ++iIter) {
      if (searchDir_sub.norm() < T(1.05) * this->curTrustRegionRadius_) {
        break;
      }

      const Eigen::MatrixX<T>& Rmat = this->qrSolver_.R();
      // This is equation 4.44 in Nocedal/Wright.  Note that our error function is actually
      // r^T*r which means gradientSub_ is 2*J^T*r, but the Hessian approximated inside the QR
      // solver is J^T*J.  Since we're applying the lambda within the QR solver also, we'll stay
      // consistent with the J^T*J/J^T*r version of the world and multiply our gradientSub_ by 0.5
      // on its way in.
      Eigen::VectorX<T> p_l = Rmat.template triangularView<Eigen::Upper>().solve(
          Rmat.template triangularView<Eigen::Upper>().transpose().solve(-T(0.5) * gradientSub_));
      Eigen::VectorX<T> q_l = Rmat.template triangularView<Eigen::Upper>().transpose().solve(p_l);

      const T p_l_norm2 = p_l.squaredNorm();
      const T q_l_norm2 = q_l.squaredNorm();

      if (q_l_norm2 < FLT_EPSILON) {
        break;
      }

      const T p_l_norm = std::sqrt(p_l_norm2);
      const T deltaLambda = (p_l_norm2 / q_l_norm2) *
          ((p_l_norm - this->curTrustRegionRadius_) / (this->curTrustRegionRadius_));

      // We can only easily _increase_ the lambda value, so stop if the change is negative:
      if (deltaLambda <= 0) {
        break;
      }

      {
        // We want to find a y such that:
        //   lambda + y^2 = (lambda + deltaLambda)
        //   y = sqrt((lambda + deltaLambda) - lambda)
        const T lambdaNew = lambda + deltaLambda;
        const T y = std::sqrt(lambdaNew - lambda);

        // Add the lambda terms:
        // TODO make this faster than O(n^2):
        this->lambdaDiag_.resize(nSubsetParams, nSubsetParams);
        this->lambdaDiag_.setZero();
        this->lambdaDiag_.diagonal().setConstant(y);

        this->lambdaZero_.resize(nSubsetParams);
        this->lambdaZero_.setZero();

        this->qrSolver_.addMutating(this->lambdaDiag_, this->lambdaZero_);

        lambda = lambdaNew;
      }

      searchDir_sub = this->qrSolver_.result();

      MT_LOGI_IF(
          this->verbose_,
          "    Iter: {}; Search dir cur norm: {}; delta: {}; lambda: {}",
          iIter,
          searchDir_sub.norm(),
          this->curTrustRegionRadius_,
          lambda);
    }

    const Eigen::VectorX<T> parameters_orig = this->parameters_;
    this->solverFunction_->updateParameters(this->parameters_, subsetToFullVector(searchDir_sub));
    const double error_new = this->solverFunction_->getError(this->parameters_);

    // This is the decrease of the function relative to what we _expected_ the function to
    // decrease by.  It is eq. 4.4 in Nocedal and Wright.
    const T quadraticModelEval = evalQuadraticModel(searchDir_sub);
    const T rho = (this->error_ - error_new) / (this->error_ - quadraticModelEval);
    MT_LOGI_IF(
        this->verbose_,
        "Error orig: {}; error new: {}; quadratic model: {}; rho: {}",
        this->error_,
        error_new,
        quadraticModelEval,
        rho);

    if (rho < T(0.25)) {
      // Model is no good, decrease the radius:
      this->curTrustRegionRadius_ = T(0.25) * this->curTrustRegionRadius_;
    } else if (rho > T(0.75) && lambda > 0) {
      // Model seems good and we're pushing up against the trust region boundary, so increase it:
      this->curTrustRegionRadius_ =
          std::min(T(2) * this->curTrustRegionRadius_, this->maxTrustRegionRadius_);
    }

    if (rho > nu) {
      break;
    } else {
      // Reject the step, reduce the trust region size and try again:
      this->parameters_ = parameters_orig;
    }
  }
}

template class TrustRegionQRT<float>;
template class TrustRegionQRT<double>;

} // namespace momentum
