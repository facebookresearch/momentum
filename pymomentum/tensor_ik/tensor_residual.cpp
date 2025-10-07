/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/tensor_ik/tensor_residual.h"

#include "pymomentum/tensor_ik/tensor_error_function.h"
#include "pymomentum/tensor_ik/tensor_ik_utility.h"
#include "pymomentum/tensor_utility/tensor_utility.h"

#include <dispenso/parallel_for.h> // @manual
#include <momentum/character/skeleton_state.h>
#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/character_solver/skeleton_solver_function.h>
#include <momentum/solver/gauss_newton_solver.h>

namespace pymomentum {

using namespace pymomentum::detail;

template <typename T>
std::tuple<at::Tensor, at::Tensor> computeResidual(
    const std::vector<const momentum::Character*>& characters,
    at::Tensor modelParams,
    const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
    at::Tensor errorFunctionWeights,
    size_t numActiveErrorFunctions,
    const std::vector<int>& weightsMap) {
  std::tie(modelParams, errorFunctionWeights) = checkIKInputs<T>(
      characters, modelParams, errorFunctionWeights, numActiveErrorFunctions, "computeGradient()");
  assert(!characters.empty()); // checked in checkIKInputs()
  const auto nParams = characters.front()->parameterTransform.numAllModelParameters();
  const auto nBatch = modelParams.size(0);

  if (nBatch == 0) {
    return {at::Tensor(), at::Tensor()};
  }

  std::vector<Eigen::MatrixX<T>> jacobians(nBatch);
  std::vector<Eigen::VectorX<T>> residuals(nBatch);
  std::vector<size_t> actualRows(nBatch, 0);

  dispenso::parallel_for(0, nBatch, [&](size_t iBatch) {
    const auto& character = *characters[iBatch];
    const momentum::ParameterTransformT<T> parameterTransform =
        character.parameterTransform.cast<T>();

    const std::vector<std::shared_ptr<momentum::SkeletonErrorFunctionT<T>>> errorFunctions_cur =
        buildMomentumErrorFunctions(
            characters, errorFunctions, errorFunctionWeights, weightsMap, iBatch);
    momentum::SkeletonSolverFunctionT<T> solverFunction =
        buildSolverFunction(*characters[iBatch], parameterTransform, errorFunctions_cur);
    const momentum::ModelParametersT<T> modelParameters_cur =
        toEigenMap<T>(modelParams.select(0, iBatch));

    solverFunction.getJacobian(
        modelParameters_cur.v, jacobians[iBatch], residuals[iBatch], actualRows[iBatch]);
  });

  // Need to be able to handle the largest Jacobian (we don't assume the
  // number of residual terms is identical across the batch, since some
  // error functions may choose to drop terms with zero weights etc.).
  const size_t maxRows = *std::max_element(actualRows.begin(), actualRows.end());

  at::Tensor jacobian = at::zeros({nBatch, (int)maxRows, (int)nParams}, toScalarType<T>());
  at::Tensor residual = at::zeros({nBatch, (int)maxRows}, toScalarType<T>());

  dispenso::parallel_for(0, nBatch, [&](size_t iBatch) {
    const size_t actualRows_cur = actualRows[iBatch];

    // All we need to do here is to copy the Jacobian/residuals back into the
    // tensors.
    jacobian.select(0, iBatch).narrow(0, 0, actualRows_cur) =
        to2DTensor(jacobians[iBatch]).narrow(0, 0, actualRows_cur);
    residual.select(0, iBatch).narrow(0, 0, actualRows_cur) =
        to1DTensor(residuals[iBatch]).narrow(0, 0, actualRows_cur);
  });

  return {residual, jacobian};
}

template std::tuple<at::Tensor, at::Tensor> computeResidual<float>(
    const std::vector<const momentum::Character*>& characters,
    at::Tensor modelParams,
    const std::vector<std::unique_ptr<TensorErrorFunction<float>>>& errorFunctions,
    at::Tensor errorFunctionWeights,
    size_t numActiveErrorFunctions,
    const std::vector<int>& weightsMap);

template std::tuple<at::Tensor, at::Tensor> computeResidual<double>(
    const std::vector<const momentum::Character*>& characters,
    at::Tensor modelParams,
    const std::vector<std::unique_ptr<TensorErrorFunction<double>>>& errorFunctions,
    at::Tensor errorFunctionWeights,
    size_t numActiveErrorFunctions,
    const std::vector<int>& weightsMap);

} // namespace pymomentum
