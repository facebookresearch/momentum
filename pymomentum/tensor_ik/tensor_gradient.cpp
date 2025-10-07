/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/tensor_ik/tensor_gradient.h"

#include "pymomentum/tensor_ik/tensor_error_function.h"
#include "pymomentum/tensor_ik/tensor_ik_utility.h"
#include "pymomentum/tensor_utility/tensor_utility.h"

#include <dispenso/parallel_for.h> // @manual
#include <momentum/character/skeleton_state.h>
#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/character_solver/skeleton_solver_function.h>
#include <momentum/diff_ik/fully_differentiable_skeleton_error_function.h>
#include <momentum/solver/gauss_newton_solver.h>

namespace pymomentum {

using namespace pymomentum::detail;

template <typename T>
at::Tensor computeGradient(
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

  at::Tensor gradient = at::zeros({nBatch, (int)nParams}, toScalarType<T>());

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

    Eigen::VectorX<T> grad = Eigen::VectorX<T>::Zero(nParams);
    solverFunction.getGradient(modelParameters_cur.v, grad);

    gradient.select(0, iBatch) = to1DTensor(grad);
  });

  return gradient;
}

template <typename T>
std::tuple<at::Tensor, at::Tensor, std::vector<at::Tensor>> d_computeGradient(
    const std::vector<const momentum::Character*>& characters,
    at::Tensor modelParams,
    at::Tensor d_loss_d_gradient,
    const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
    at::Tensor errorFunctionWeights,
    size_t numActiveErrorFunctions,
    const std::vector<int>& weightsMap) {
  bool squeezeErrorFunctionWeights;
  std::tie(modelParams, errorFunctionWeights) = checkIKInputs<T>(
      characters,
      modelParams,
      errorFunctionWeights,
      numActiveErrorFunctions,
      "d_computeGradient()",
      &squeezeErrorFunctionWeights);
  assert(!characters.empty()); // checked in checkIKInputs()
  const auto nParams = characters.front()->parameterTransform.numAllModelParameters();
  const auto nBatch = modelParams.size(0);

  at::Tensor d_loss_d_modelParams = at::zeros({nBatch, (int)nParams}, toScalarType<T>());

  std::vector<ErrorFunctionInput<T>> grad_inputs =
      buildErrorFunctionInputs(errorFunctions, weightsMap);

  at::Tensor grad_errorFunctionWeights = at::zeros(errorFunctionWeights.sizes(), toScalarType<T>());

  dispenso::parallel_for(0, nBatch, [&](size_t iBatch) {
    const auto& character = *characters[iBatch];
    const momentum::ParameterTransformT<T> parameterTransform =
        character.parameterTransform.cast<T>();

    const std::vector<std::shared_ptr<momentum::SkeletonErrorFunctionT<T>>> errorFunctions_cur =
        buildMomentumErrorFunctions(
            characters, errorFunctions, errorFunctionWeights, weightsMap, iBatch);
    assert(errorFunctions_cur.size() == errorFunctions.size());
    momentum::SkeletonSolverFunctionT<T> solverFunction =
        buildSolverFunction(*characters[iBatch], parameterTransform, errorFunctions_cur);
    const momentum::ModelParametersT<T> modelParameters_cur =
        toEigenMap<T>(modelParams.select(0, iBatch));

    at::Tensor d_loss_d_grad_cur = d_loss_d_gradient.select(0, iBatch);

    Eigen::VectorX<T> residual;
    Eigen::MatrixX<T> jacobian;
    size_t actualRows;
    solverFunction.getJacobian(modelParameters_cur.v, jacobian, residual, actualRows);

    // The gradient is grad_ik(theta) = 2 * J^T(theta) * r(theta).  We want to
    // know the derivative of the ML loss wrt the model parameters.  We compute:
    //     dL/d_theta = dL/dGrad_ik dGrad_ik/dTheta
    //                = dL/dGrad_ik [2 * dJ^T/dTheta * r(theta) + 2 * J^T * J]
    // We can drop that first higher order term because it's hard to compute and
    // use
    //     dL/d_theta = dL/dGrad_ik [2 * J^T * J].
    // If we transpose the whole thing, we get
    //     dL/d_theta^T = 2 * J^T * J * dL/dGrad_ik:
    at::Tensor d_loss_d_modelParams_cur = d_loss_d_modelParams.select(0, iBatch);
    toEigenMap<T>(d_loss_d_modelParams_cur) =
        T(2) * jacobian.transpose() * (jacobian * toEigenMap<T>(d_loss_d_grad_cur));

    const momentum::SkeletonStateT<T> skelState(
        parameterTransform.apply(modelParameters_cur), character.skeleton);

    at::Tensor grad_errorFunctionWeights_cur = grad_errorFunctionWeights.select(0, iBatch);

    // Derivative of the gradient wrt the weight is just the gradient with the
    // weight divided out:
    for (size_t iErr = 0; iErr < errorFunctions.size(); ++iErr) {
      if (weightsMap[iErr] < 0) {
        continue;
      }

      Eigen::VectorX<T> gradient = Eigen::VectorX<T>::Zero(nParams);
      errorFunctions_cur[iErr]->getGradient(modelParameters_cur, skelState, gradient);
      const T w = errorFunctions_cur[iErr]->getWeight();
      if (w == 0) { // avoid divide-by-zero
        continue;
      }

      toEigenMap<T>(grad_errorFunctionWeights_cur)(weightsMap[iErr]) =
          (gradient.dot(toEigenMap<T>(d_loss_d_grad_cur))) / w;
    }

    // For each differentiable input, collect the derivative of the IK gradient
    // wrt the input (conveniently we already know how to calculate this since
    // it's the thing FullyDifferentiableSkeletonErrorFunctions know how to
    // compute).
    for (size_t iGlobalInput = 0; iGlobalInput < grad_inputs.size(); ++iGlobalInput) {
      if (grad_inputs[iGlobalInput].dLoss_dInput.empty()) {
        continue;
      }

      const auto iErrorFunction = grad_inputs[iGlobalInput].iErrorFunction;
      const auto jInput = grad_inputs[iGlobalInput].jInput;
      const auto& tensorErrf = errorFunctions[iErrorFunction];
      const auto& input = tensorErrf->tensorInputs()[jInput];
      const auto& inputName = input.inputName;
      const auto errf = errorFunctions_cur[iErrorFunction].get();

      auto differentiableErr =
          dynamic_cast<momentum::FullyDifferentiableSkeletonErrorFunctionT<T>*>(errf);
      if (differentiableErr != nullptr) {
        const Eigen::VectorX<T> dLoss_dInput = differentiableErr->d_gradient_d_input_dot(
            inputName, modelParameters_cur, skelState, toEigenMap<T>(d_loss_d_grad_cur));
        grad_inputs[iGlobalInput].dLoss_dInput[iBatch] = std::move(dLoss_dInput);
      }
    }
  });

  // Decide if we need to de-batch each gradient (by summing along the
  // batch dimension):
  if (squeezeErrorFunctionWeights) {
    grad_errorFunctionWeights = grad_errorFunctionWeights.sum(0);
  }

  // Map the input derivatives back to tensors:
  return {d_loss_d_modelParams, grad_errorFunctionWeights, toTensors(errorFunctions, grad_inputs)};
}

template at::Tensor computeGradient<float>(
    const std::vector<const momentum::Character*>& characters,
    at::Tensor modelParams,
    const std::vector<std::unique_ptr<TensorErrorFunction<float>>>& errorFunctions,
    at::Tensor errorFunctionWeights,
    size_t numActiveErrorFunctions,
    const std::vector<int>& weightsMap);

template at::Tensor computeGradient<double>(
    const std::vector<const momentum::Character*>& characters,
    at::Tensor modelParams,
    const std::vector<std::unique_ptr<TensorErrorFunction<double>>>& errorFunctions,
    at::Tensor errorFunctionWeights,
    size_t numActiveErrorFunctions,
    const std::vector<int>& weightsMap);

template std::tuple<at::Tensor, at::Tensor, std::vector<at::Tensor>> d_computeGradient<float>(
    const std::vector<const momentum::Character*>& characters,
    at::Tensor modelParams,
    at::Tensor d_loss_d_gradient,
    const std::vector<std::unique_ptr<TensorErrorFunction<float>>>& errorFunctions,
    at::Tensor errorFunctionWeights,
    size_t numActiveErrorFunctions,
    const std::vector<int>& weightsMap);

template std::tuple<at::Tensor, at::Tensor, std::vector<at::Tensor>> d_computeGradient<double>(
    const std::vector<const momentum::Character*>& characters,
    at::Tensor modelParams,
    at::Tensor d_loss_d_gradient,
    const std::vector<std::unique_ptr<TensorErrorFunction<double>>>& errorFunctions,
    at::Tensor errorFunctionWeights,
    size_t numActiveErrorFunctions,
    const std::vector<int>& weightsMap);

} // namespace pymomentum
