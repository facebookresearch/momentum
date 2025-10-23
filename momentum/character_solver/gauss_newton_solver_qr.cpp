/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/gauss_newton_solver_qr.h"

#include "momentum/character/mesh_state.h"
#include "momentum/character_solver/skeleton_error_function.h"
#include "momentum/common/profile.h"

#include <gsl/util>

namespace momentum {

template <typename T>
GaussNewtonSolverQRT<T>::GaussNewtonSolverQRT(
    const SolverOptions& options,
    SkeletonSolverFunctionT<T>* solverFun)
    : SolverT<T>(options, solverFun),
      skeletonState_(std::make_unique<SkeletonStateT<T>>()),
      meshState_(std::make_unique<MeshStateT<T>>()),
      qrSolver_(0) {
  // Set default values from GaussNewtonSolverQROptions
  regularization_ = GaussNewtonSolverQROptions().regularization;
  doLineSearch_ = GaussNewtonSolverQROptions().doLineSearch;

  // Update values based on provided options
  setOptions(options);
}

template <typename T>
GaussNewtonSolverQRT<T>::~GaussNewtonSolverQRT() = default;

template <typename T>
std::string_view GaussNewtonSolverQRT<T>::getName() const {
  return "GaussNewtonQR";
}

template <typename T>
void GaussNewtonSolverQRT<T>::initializeSolver() {}

template <typename T>
void GaussNewtonSolverQRT<T>::setOptions(const SolverOptions& options) {
  SolverT<T>::setOptions(options);

  if (const auto* derivedOptions = dynamic_cast<const GaussNewtonSolverQROptions*>(&options)) {
    regularization_ = derivedOptions->regularization;
    doLineSearch_ = derivedOptions->doLineSearch;
  }
}

template <typename T>
void GaussNewtonSolverQRT<T>::doIteration() {
  MT_PROFILE_FUNCTION();

  const auto sf = static_cast<SkeletonSolverFunctionT<T>*>(this->solverFunction_);

  const auto skeleton = sf->getSkeleton();
  const auto parameterTransform = sf->getParameterTransform();

  {
    MT_PROFILE_EVENT("JtJR - update state");
    skeletonState_->set(parameterTransform->apply(this->parameters_), *skeleton);
  }

  if (sf->needsMeshState()) {
    MT_PROFILE_EVENT("JtJR - update mesh state");
    meshState_->update(this->parameters_, *skeletonState_, sf->getCharacter());
  }

  const auto nFullParams = gsl::narrow<Eigen::Index>(this->parameters_.size());

  std::vector<Eigen::Index> enabledParameters;
  enabledParameters.reserve(nFullParams);
  for (int iFullParam = 0; iFullParam < nFullParams; ++iFullParam) {
    if (this->activeParameters_.test(iFullParam)) {
      enabledParameters.push_back(iFullParam);
    }
  }

  const auto nSubsetParams = gsl::narrow_cast<int>(enabledParameters.size());

  // momentum solves the problem (J^T*J + lambda*I) x = J^T*r;
  // the QR solver wants the square root of that lambda.
  qrSolver_.reset(nSubsetParams, std::sqrt(regularization_));

  double error_orig = 0.0;
  for (auto errorFunction : sf->getErrorFunctions()) {
    if (errorFunction->getWeight() <= 0) {
      continue;
    }

    const auto rows = gsl::narrow<Eigen::Index>(errorFunction->getJacobianSize());

    jacobian_.resizeAndSetZero(rows, nFullParams);
    residual_.resizeAndSetZero(rows);

    int usedRows = 0;
    error_orig += errorFunction->getJacobian(
        this->parameters_,
        *skeletonState_,
        *meshState_,
        jacobian_.mat(),
        residual_.mat(),
        usedRows);
    if (usedRows == 0) {
      continue;
    }

    qrSolver_.addMutating(
        ColumnIndexedMatrix<Eigen::MatrixX<T>>(
            jacobian_.mat().topRows(usedRows), enabledParameters),
        residual_.mat().topRows(usedRows));
  }

  this->error_ = error_orig;

  const Eigen::VectorX<T> subsetDelta = qrSolver_.result();

  Eigen::VectorX<T> searchDir = Eigen::VectorX<T>::Zero(nFullParams);
  for (int iSubsetParam = 0; iSubsetParam < nSubsetParams; ++iSubsetParam) {
    const auto iFullParam = enabledParameters[iSubsetParam];
    searchDir(iFullParam) = subsetDelta(iSubsetParam);
  }

  if (doLineSearch_) {
    const double innerProd = -qrSolver_.At_times_b().dot(subsetDelta);

    // Line search:
    const float c_1 = 1e-4f;
    const float tau = 0.5f;
    float alpha = 1.0f;

    const Eigen::VectorX<T> parameters_orig = this->parameters_;
    for (size_t kStep = 0; kStep < 10 && std::fpclassify(alpha) == FP_NORMAL; ++kStep) {
      // update the this->parameters_
      this->parameters_ = parameters_orig;
      this->solverFunction_->updateParameters(this->parameters_, alpha * searchDir);

      const double error_new = this->solverFunction_->getError(this->parameters_);

      if ((error_orig - error_new) >= c_1 * alpha * -innerProd) {
        break;
      }

      // Reduce step size:
      alpha = alpha * tau;
    }
  } else {
    this->solverFunction_->updateParameters(this->parameters_, searchDir);
  }
}

template class GaussNewtonSolverQRT<float>;
template class GaussNewtonSolverQRT<double>;

} // namespace momentum
