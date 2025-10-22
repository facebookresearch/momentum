/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/diff_ik/fully_differentiable_body_ik.h"

#include "momentum/character/mesh_state.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/common/log.h"
#include "momentum/diff_ik/fully_differentiable_skeleton_error_function.h"
#include "momentum/math/fmt_eigen.h"

#include <Eigen/Eigenvalues>

namespace momentum {

template <typename T>
std::vector<Eigen::Index> computeSubsetParameterIndices(
    const ParameterTransformT<T>& parameterTransform,
    const ParameterSet& activeParams) {
  std::vector<Eigen::Index> result;
  for (size_t i = 0; i < parameterTransform.numAllModelParameters(); ++i) {
    if (activeParams.test(i)) {
      result.push_back(i);
    }
  }

  return result;
}

template <typename T>
Eigen::VectorX<T> extractSubsetVec(
    Eigen::Ref<const Eigen::VectorX<T>> fullVec,
    const std::vector<Eigen::Index>& subsetIndexToFullParamIndex) {
  const Eigen::Index nSubset = subsetIndexToFullParamIndex.size();
  Eigen::VectorX<T> result(nSubset);
  for (Eigen::Index iSubset = 0; iSubset < nSubset; ++iSubset) {
    result(iSubset) = fullVec(subsetIndexToFullParamIndex[iSubset]);
  }
  return result;
}

template <typename T>
Eigen::MatrixX<T> extractSubsetMat(
    Eigen::Ref<const Eigen::MatrixX<T>> fullMat,
    const std::vector<Eigen::Index>& subsetIndexToFullParamIndex) {
  const Eigen::Index nSubset = subsetIndexToFullParamIndex.size();
  Eigen::MatrixX<T> result(fullMat.rows(), nSubset);
  for (Eigen::Index iSubset = 0; iSubset < nSubset; ++iSubset) {
    result.col(iSubset) = fullMat.col(subsetIndexToFullParamIndex[iSubset]);
  }
  return result;
}

template <typename T>
Eigen::VectorX<T> subsetVecToFullVec(
    Eigen::Ref<const Eigen::VectorX<T>> subsetVec,
    const std::vector<Eigen::Index>& subsetIndexToFullParamIndex,
    Eigen::Index fullVecSize) {
  Eigen::VectorX<T> result = Eigen::VectorX<T>::Zero(fullVecSize);
  for (size_t iSubset = 0; iSubset < subsetIndexToFullParamIndex.size(); ++iSubset) {
    result[subsetIndexToFullParamIndex[iSubset]] = subsetVec[iSubset];
  }
  return result;
}

float sqr(float val) {
  return val * val;
}

template <typename T>
Eigen::VectorX<T> hessianInverseTimes(
    Eigen::Ref<const Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<const Eigen::VectorX<T>> v) {
  // Use approximate Hessian:
  //   Hess ~ J^T * J
  // We will use the SVD on J as follows:
  //   J = U * S * V^T
  //   Hess = 2 * J^T * J = (U * S * V^T)^T * (U * S * V^T)
  //        = 2 * V * S * U^T * U * S * V^T
  //        = 2 * V * S^2 * V^T
  // This is equivalent to performing an eigenvalue decomposition
  // on the Hessian, and lets us ensure that the Hessian is actually
  // invertible.
  //
  // Note that the particular SVD used here is not terribly fast for large n;
  // we should consider switching to the one in LAPACK (the divide-and-conquer
  // version in Eigen is apparently not reliable with -ffast-math).
  Eigen::JacobiSVD<Eigen::MatrixX<T>> svd(jacobian, Eigen::ComputeThinV);

  // To get the Hess^{-1} * v, we need:
  //    2 * V * S^2 * V^T * w = v
  //                        w = 0.5 * V^T * v)
  //                  V^T * w = 0.5 * (S^2)^{-1} * V^T * v
  //                        w = 0.5 * V * (S^2)^{-1} * V^T * v
  Eigen::VectorX<T> tmp = svd.matrixV().transpose() * v;
  for (Eigen::Index i = 0; i < tmp.size(); ++i) {
    const float sSqr = sqr(svd.singularValues()(i));
    if (sSqr < 1e-5f) {
      tmp(i) = 0;
    } else {
      tmp(i) /= sSqr;
    }
  }
  return 0.5f * svd.matrixV() * tmp;
}

template <typename T>
std::vector<ErrorFunctionDerivativesT<T>> d_modelParams_d_inputs(
    const Skeleton& skeleton,
    const ParameterTransformT<T>& parameterTransform,
    const ParameterSet& activeParams,
    const ModelParametersT<T>& modelParameters,
    SkeletonSolverFunctionT<T>& solverFunction,
    Eigen::Ref<const Eigen::VectorX<T>> dLoss_dModelParams,
    T* gradientRmse) {
  std::vector<ErrorFunctionDerivativesT<T>> result(solverFunction.getErrorFunctions().size());

  const std::vector<Eigen::Index> subsetIndexToFullParamIndex =
      computeSubsetParameterIndices(parameterTransform, activeParams);
  const Eigen::Index nFullParams = parameterTransform.numAllModelParameters();

  Eigen::MatrixX<T> jacobian_full;
  Eigen::VectorX<T> residual_full;

  // Some basic definitions:
  // theta_j are the model parameters, basically the skeleton pose.
  // w_j are the "other" inputs that affect the error function values.  Here I
  //    am thinking of  things like constraint weights, etc.
  // E(theta_i, w_j) is the energy function our IK solver is minimizing.
  //
  // At the minimum, we have the gradient equal to zero:
  //   grad E(theta, w) = dE/dTheta = 0
  //
  // Let f(theta, w) = dE/dTheta.
  // By the implicit function theorem, under certain assumptions about
  // continuity etc., there exists a function g(w) such that f(theta = g(w), w)
  // = 0; that is, the energy function is still minimized.  Basically, g(w) maps
  // from any new (but sufficiently close) value of w to a new value of theta.
  //
  // Now, what we want is the _gradient_ of g, this will tell us how theta
  // changes as w varies.  By the IFT again:
  //   dg/dw = (-df/dTheta)^{-1} * df/dw
  // The first term (df/dTheta) is just the Hessian, but as is typical we will
  // approximate it using the Gauss-Newton approximation.
  // df/dw is how the gradient changes as a function of w,
  //    d/dw(dE/dTheta) = d^2E/(dTheta dw)
  // This is going to be a big, very sparse matrix; typical values are like n ~
  // 50 for the pose vector and m ~ 200 is the number of constraints, and it's
  // likely that most constraints only affect a few of the pose components.
  // So then we'd need to multiply that by the inverse Hessian.
  //
  // However, at this point it's worth remembering that we actually want to
  // compute
  //    dLoss/dTheta * dTheta/dw
  // for some dLoss_dTheta that was passed into this function (this is due to
  // reverse-mode auto-differentiation).  Thus, we can write
  //    dLoss/dTheta * (-df/dTheta)^{-1} * df/dw
  // We can group the multiplications,
  //    (dLoss/dTheta * (-df/dTheta)^{-1}) * df/dw
  // This first part is just (Hess^{-1} * dLoss/dTheta^T)^T.  The result is a
  // n-dimensional vector, and we can then multiply this into the df/dw
  // directly. Thus no need to store any big sparse matrices.
  size_t actualRows = 0;
  solverFunction.getJacobian(modelParameters.v, jacobian_full, residual_full, actualRows);

  const Eigen::MatrixX<T> jacobian_subset =
      extractSubsetMat<T>(jacobian_full, subsetIndexToFullParamIndex);
  const Eigen::VectorX<T> gradient_subset = 2.0f * jacobian_subset.transpose() * residual_full;
  if (gradientRmse != nullptr) {
    *gradientRmse = std::sqrt(gradient_subset.squaredNorm() / gradient_subset.size());
  }

  const Eigen::VectorX<T> Jinv_times_dLoss_dModelParams_subset = hessianInverseTimes<T>(
      jacobian_subset, extractSubsetVec<T>(dLoss_dModelParams, subsetIndexToFullParamIndex));

  const Eigen::VectorX<T> Jinv_times_dLoss_dModelParams_full = subsetVecToFullVec<T>(
      Jinv_times_dLoss_dModelParams_subset, subsetIndexToFullParamIndex, nFullParams);

  MT_LOGD("Jinv_times_dLoss_dModelParams = {}", Jinv_times_dLoss_dModelParams_full.transpose());

  const JointParametersT<T> jointParameters = parameterTransform.apply(modelParameters);
  SkeletonStateT<T> skelState(jointParameters, skeleton);
  MeshStateT<T> meshState;
  if (solverFunction.needsMeshState()) {
    meshState.update(modelParameters, skelState, solverFunction.getCharacter());
  }

  const auto& errorFunctions = solverFunction.getErrorFunctions();
  for (size_t iErr = 0; iErr < errorFunctions.size(); ++iErr) {
    const auto errf = errorFunctions[iErr];
    Eigen::VectorX<T> dGrad_dWeight =
        Eigen::VectorX<T>::Zero(parameterTransform.numAllModelParameters());
    errf->getGradient(modelParameters, skelState, meshState, dGrad_dWeight);
    if (errf->getWeight() != 0) {
      dGrad_dWeight /= errf->getWeight();
    }
    MT_LOGD("dGrad_dWeight = {}", dGrad_dWeight.transpose());

    /*
    {
      const auto w_init = errf->getWeight();

      const float eps = 1e-3f;
      Eigen::VectorX<T> grad_init =
    Eigen::VectorX<T>::Zero(parameterTransform.numAllModelParameters());
      errf->getGradient(modelParameters, skelState, grad_init);

      errf->setWeight(w_init + eps);
      Eigen::VectorX<T> grad_plus =
    Eigen::VectorX<T>::Zero(parameterTransform.numAllModelParameters());
      errf->getGradient(modelParameters, skelState, grad_plus);
      errf->setWeight(w_init);

      Eigen::VectorX<T> diff = (grad_plus - grad_init) / eps;
      MT_LOGI("dGrad_dWeight (est) = {}", diff.transpose());
    }
    */

    result[iErr].errorFunction = errf;
    result[iErr].gradWeight = -dGrad_dWeight.dot(Jinv_times_dLoss_dModelParams_full);

    auto differentiableErr =
        std::dynamic_pointer_cast<FullyDifferentiableSkeletonErrorFunctionT<T>>(errf);
    if (differentiableErr != nullptr) {
      for (const auto& inputName : differentiableErr->inputs()) {
        result[iErr].gradInputs.emplace(
            inputName,
            differentiableErr->d_gradient_d_input_dot(
                inputName, modelParameters, skelState, -Jinv_times_dLoss_dModelParams_full));
      }
    }
  }

  return result;
}

template std::vector<ErrorFunctionDerivativesT<float>> d_modelParams_d_inputs<float>(
    const Skeleton& skeleton,
    const ParameterTransformT<float>& parameterTransform,
    const ParameterSet& activeParams,
    const ModelParametersT<float>& modelParameters,
    SkeletonSolverFunctionT<float>& solverFunction,
    Eigen::Ref<const Eigen::VectorX<float>> dLoss_dModelParams,
    float* gradientRmse);

template std::vector<ErrorFunctionDerivativesT<double>> d_modelParams_d_inputs<double>(
    const Skeleton& skeleton,
    const ParameterTransformT<double>& parameterTransform,
    const ParameterSet& activeParams,
    const ModelParametersT<double>& modelParameters,
    SkeletonSolverFunctionT<double>& solverFunction,
    Eigen::Ref<const Eigen::VectorX<double>> dLoss_dModelParams,
    double* gradientRmse);

} // namespace momentum
