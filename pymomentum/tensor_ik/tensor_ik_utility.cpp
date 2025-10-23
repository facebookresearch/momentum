/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/tensor_ik/tensor_ik_utility.h"

#include "pymomentum/tensor_ik/tensor_error_function.h"
#include "pymomentum/tensor_utility/tensor_utility.h"

#include <dispenso/parallel_for.h> // @manual
#include <momentum/character/skeleton_state.h>
#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/character_solver/skeleton_solver_function.h>
#include <momentum/solver/gauss_newton_solver.h>

namespace pymomentum::detail {

size_t checkNumParams(
    const std::vector<const momentum::Character*>& characters,
    const char* context) {
  MT_THROW_IF(characters.empty(), "Empty character list in {}", context);

  const auto nParams = characters.front()->parameterTransform.numAllModelParameters();
  MT_THROW_IF(
      nParams == 0,
      "In {}, parameter transform is empty (has no parameters) and hence is not suitable for optimization.",
      context);

  for (const auto& character : characters) {
    MT_THROW_IF(
        character->parameterTransform.name != characters.front()->parameterTransform.name,
        "In {}, mismatch between the parameter transforms of the passed-in characters.  For batch-mode solve, all parameters must be the same (although the skeletons can vary).",
        context);
  }

  return nParams;
}

void maybeSet(bool* ptr, bool value) {
  if (ptr != nullptr) {
    *ptr = value;
  }
}

template <typename T>
std::tuple<at::Tensor, at::Tensor> checkIKInputs(
    const std::vector<const momentum::Character*>& characters,
    at::Tensor modelParams,
    at::Tensor errorFunctionWeights,
    size_t numActiveErrorFunctions,
    const char* context,
    bool* squeezeErrorFunctionWeights) {
  const auto nParams = checkNumParams(characters, context);

  modelParams = modelParams.contiguous().to(at::DeviceType::CPU, toScalarType<T>());

  throwIfNaNOrINF(modelParams, context, "model params");

  MT_THROW_IF(
      modelParams.size(-1) != nParams,
      "Mismatch between modelParameters_init size {} and parameter transform size {} in {}",
      formatTensorSizes(modelParams),
      nParams,
      context);

  // Code in momentumIK is supposed to enforce this:
  MT_THROW_IF(modelParams.ndimension() != 2, "Expected batched modelParameters vector.");

  const auto nBatch = modelParams.size(0);

  if (errorFunctionWeights.ndimension() == 1) {
    errorFunctionWeights = errorFunctionWeights.unsqueeze(0).expand({nBatch, -1});
    maybeSet(squeezeErrorFunctionWeights, true);
  } else {
    maybeSet(squeezeErrorFunctionWeights, false);
  }

  MT_THROW_IF(
      errorFunctionWeights.size(0) != nBatch ||
          errorFunctionWeights.size(1) != numActiveErrorFunctions,
      "In {}: mismatch in error function weights sizes ({}, {}) and batch size {} or active error function size {}",
      context,
      errorFunctionWeights.size(0),
      errorFunctionWeights.size(1),
      nBatch,
      numActiveErrorFunctions);

  return {
      modelParams, errorFunctionWeights.contiguous().to(at::DeviceType::CPU, toScalarType<T>())};
}

template <typename T>
std::tuple<at::Tensor, at::Tensor> checkSequenceIKInputs(
    const std::vector<const momentum::Character*>& characters,
    at::Tensor modelParams,
    at::Tensor errorFunctionWeights,
    size_t numActiveErrorFunctions,
    const char* context) {
  const auto nParams = checkNumParams(characters, context);

  modelParams = modelParams.contiguous().to(at::DeviceType::CPU, toScalarType<T>());
  throwIfNaNOrINF(modelParams, context, "model params");

  // Caller is supposed to ensure this:
  MT_THROW_IF(
      modelParams.ndimension() != 3,
      "Expected modelParams with (nBatch, nFrames, nParams) dimensions");

  const int64_t nBatch = modelParams.size(0);
  const int64_t nFrames = modelParams.size(1);

  MT_THROW_IF(
      modelParams.size(-1) != nParams,
      "Mismatch between modelParameters_init size {} and parameter transform size {} in {}",
      formatTensorSizes(modelParams),
      nParams,
      context);

  if (errorFunctionWeights.ndimension() == 1) {
    errorFunctionWeights =
        errorFunctionWeights.unsqueeze(0).unsqueeze(0).expand({nBatch, nFrames, -1});
  } else if (errorFunctionWeights.ndimension() == 2) {
    MT_THROW_IF(
        nBatch != 1,
        "Ambiguous error function weights dimension.  Expected either _both_ nBatch and nFrames dimensions (nBatch x nFrames x nErrorFunctions) or neither. Got {}",
        formatTensorSizes(errorFunctionWeights));

    errorFunctionWeights = errorFunctionWeights.unsqueeze(0);
  }

  MT_THROW_IF(
      errorFunctionWeights.ndimension() != 3 || errorFunctionWeights.size(0) != nBatch ||
          errorFunctionWeights.size(1) != nFrames ||
          errorFunctionWeights.size(2) != numActiveErrorFunctions,
      "In {}: mismatch in error function weights sizes. Expected [opt. nBatch={} x opt. nFrames={} x nErrorFunctions={}] but got {}",
      context,
      nBatch,
      nFrames,
      numActiveErrorFunctions,
      formatTensorSizes(errorFunctionWeights));

  return {
      modelParams, errorFunctionWeights.contiguous().to(at::DeviceType::CPU, toScalarType<T>())};
}

template <typename T>
std::vector<std::shared_ptr<momentum::SkeletonErrorFunctionT<T>>> buildMomentumErrorFunctions(
    const std::vector<const momentum::Character*>& characters,
    const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
    at::Tensor errorFunctionWeights,
    const std::vector<int>& weightsMap,
    const int64_t iBatch) {
  const auto& character = *characters[iBatch];
  at::Tensor weights_cur = errorFunctionWeights.select(0, iBatch);

  std::vector<std::shared_ptr<momentum::SkeletonErrorFunctionT<T>>> result;
  const size_t jFrame = SIZE_MAX;
  for (size_t iErr = 0; iErr < errorFunctions.size(); ++iErr) {
    const auto& errf = errorFunctions[iErr];
    result.push_back(errf->createErrorFunction(character, iBatch, jFrame));
    // weightsMap maps error type order in the enum to order in input
    // errorFunctionWeights.
    T weight = weightsMap[iErr] < 0 ? T(0) : toEigenMap<T>(weights_cur)[weightsMap[iErr]];
    result.back()->setWeight(weight);
  }

  return result;
}

template <typename T>
momentum::SkeletonSolverFunctionT<T> buildSolverFunction(
    const momentum::Character& character,
    const momentum::ParameterTransformT<T>& parameterTransform,
    const std::vector<std::shared_ptr<momentum::SkeletonErrorFunctionT<T>>>& errorFunctions) {
  return {character, parameterTransform, errorFunctions};
}

template <typename T>
std::unique_ptr<momentum::SequenceSolverFunctionT<T>> buildSequenceSolverFunction(
    const momentum::Character& character,
    const momentum::ParameterTransformT<T>& parameterTransform,
    at::Tensor modelParams_init,
    const momentum::ParameterSet& sharedParams,
    const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
    at::Tensor errorFunctionWeights,
    const std::vector<int>& weightsMap,
    const int64_t iBatch) {
  assert(modelParams_init.ndimension() == 2);
  assert(modelParams_init.size(-1) == parameterTransform.numAllModelParameters());
  const auto nFrames = modelParams_init.size(0);
  auto result = std::make_unique<momentum::SequenceSolverFunctionT<T>>(
      character, parameterTransform, sharedParams, nFrames);

  assert(modelParams_init.ndimension() == 2);
  assert(errorFunctionWeights.ndimension() == 2);

  // We're allowed to add the error functions in parallel, which will be an
  // advantage on larger problems.
  dispenso::parallel_for(0, nFrames, [&](int64_t jFrame) {
    at::Tensor weights_cur = errorFunctionWeights.select(0, jFrame);

    for (size_t iErr = 0; iErr < errorFunctions.size(); ++iErr) {
      const auto& errf = errorFunctions[iErr];
      auto errf_momentum = errf->createErrorFunction(character, iBatch, jFrame);
      // weightsMap maps error type order in the enum to order in input
      // errorFunctionWeights.
      T weight = weightsMap[iErr] < 0 ? T(0) : toEigenMap<T>(weights_cur)[weightsMap[iErr]];
      errf_momentum->setWeight(weight);

      result->addErrorFunction(jFrame, errf_momentum);
    }
  });

  dispenso::parallel_for(0, nFrames, [&](size_t jFrame) {
    at::Tensor frameParams = modelParams_init.select(0, jFrame);

    result->setFrameParameters(jFrame, toEigenMap<T>(frameParams));
  });

  return result;
}

template <typename T>
std::vector<ErrorFunctionInput<T>> buildErrorFunctionInputs(
    const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
    const std::vector<int>& weightsMap) {
  std::vector<ErrorFunctionInput<T>> result;
  for (size_t iErrorFunction = 0; iErrorFunction < errorFunctions.size(); ++iErrorFunction) {
    const auto& errf = errorFunctions[iErrorFunction];
    const auto& inputs_cur = errf->tensorInputs();
    for (size_t jInput = 0; jInput < inputs_cur.size(); ++jInput) {
      result.emplace_back(iErrorFunction, jInput);

      if (errf->requiredTensorEmpty()) {
        continue;
      }

      if (weightsMap[iErrorFunction] < 0) {
        continue;
      }

      if (inputs_cur[jInput].differentiability == TensorInput::NON_DIFFERENTIABLE ||
          isEmpty(inputs_cur[jInput].tensor)) {
        continue;
      }

      result.back().dLoss_dInput.resize(errf->batchSize());
    }
  }

  return result;
}

template <typename T>
torch::autograd::variable_list toTensors(
    const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
    const std::vector<ErrorFunctionInput<T>>& inputs_all) {
  const auto nGlobalInputs = inputs_all.size();
  torch::autograd::variable_list result(nGlobalInputs);

  for (size_t iGlobalInput = 0; iGlobalInput < nGlobalInputs; ++iGlobalInput) {
    if (inputs_all[iGlobalInput].dLoss_dInput.empty()) {
      // No derivatives for this input:
      continue;
    }

    const auto iErrorFunction = inputs_all[iGlobalInput].iErrorFunction;
    const auto jInput = inputs_all[iGlobalInput].jInput;
    const auto& errf = errorFunctions[iErrorFunction];
    const auto& input = errf->tensorInputs()[jInput];
    const auto nBatch = errf->batchSize();

    auto derivs = at::zeros(input.tensor.sizes(), toScalarType<T>());
    if (derivs.ndimension() == input.targetSizes.size()) {
      // Un-batched input, so sum across the whole batch.
      for (size_t jBatch = 0; jBatch < nBatch; ++jBatch) {
        if (inputs_all[iGlobalInput].dLoss_dInput[jBatch].size() == 0) {
          continue;
        }
        toEigenMap<T>(derivs) += inputs_all[iGlobalInput].dLoss_dInput[jBatch];
      }
    } else {
      // Batched
      for (size_t jBatch = 0; jBatch < nBatch; ++jBatch) {
        auto derivs_cur = derivs.select(0, jBatch);
        if (inputs_all[iGlobalInput].dLoss_dInput[jBatch].size() == 0) {
          continue;
        }
        toEigenMap<T>(derivs_cur) = inputs_all[iGlobalInput].dLoss_dInput[jBatch];
      }
    }

    result[iGlobalInput] = derivs;
  }

  return result;
}

template std::tuple<at::Tensor, at::Tensor> checkIKInputs<float>(
    const std::vector<const momentum::Character*>& characters,
    at::Tensor modelParams,
    at::Tensor errorFunctionWeights,
    size_t numActiveErrorFunctions,
    const char* context,
    bool* squeezeErrorFunctionWeights);
template std::tuple<at::Tensor, at::Tensor> checkIKInputs<double>(
    const std::vector<const momentum::Character*>& characters,
    at::Tensor modelParams,
    at::Tensor errorFunctionWeights,
    size_t numActiveErrorFunctions,
    const char* context,
    bool* squeezeErrorFunctionWeights);

template std::tuple<at::Tensor, at::Tensor> checkSequenceIKInputs<float>(
    const std::vector<const momentum::Character*>& characters,
    at::Tensor modelParams,
    at::Tensor errorFunctionWeights,
    size_t numActiveErrorFunctions,
    const char* context);
template std::tuple<at::Tensor, at::Tensor> checkSequenceIKInputs<double>(
    const std::vector<const momentum::Character*>& characters,
    at::Tensor modelParams,
    at::Tensor errorFunctionWeights,
    size_t numActiveErrorFunctions,
    const char* context);

template std::vector<std::shared_ptr<momentum::SkeletonErrorFunctionT<float>>>
buildMomentumErrorFunctions<float>(
    const std::vector<const momentum::Character*>& characters,
    const std::vector<std::unique_ptr<TensorErrorFunction<float>>>& errorFunctions,
    at::Tensor errorFunctionWeights,
    const std::vector<int>& weightsMap,
    const int64_t iBatch);
template std::vector<std::shared_ptr<momentum::SkeletonErrorFunctionT<double>>>
buildMomentumErrorFunctions<double>(
    const std::vector<const momentum::Character*>& characters,
    const std::vector<std::unique_ptr<TensorErrorFunction<double>>>& errorFunctions,
    at::Tensor errorFunctionWeights,
    const std::vector<int>& weightsMap,
    const int64_t iBatch);

template momentum::SkeletonSolverFunctionT<float> buildSolverFunction<float>(
    const momentum::Character& character,
    const momentum::ParameterTransformT<float>& parameterTransform,
    const std::vector<std::shared_ptr<momentum::SkeletonErrorFunctionT<float>>>& errorFunctions);
template momentum::SkeletonSolverFunctionT<double> buildSolverFunction<double>(
    const momentum::Character& character,
    const momentum::ParameterTransformT<double>& parameterTransform,
    const std::vector<std::shared_ptr<momentum::SkeletonErrorFunctionT<double>>>& errorFunctions);

template std::unique_ptr<momentum::SequenceSolverFunctionT<float>>
buildSequenceSolverFunction<float>(
    const momentum::Character& character,
    const momentum::ParameterTransformT<float>& parameterTransform,
    at::Tensor modelParams_init,
    const momentum::ParameterSet& sharedParams,
    const std::vector<std::unique_ptr<TensorErrorFunction<float>>>& errorFunctions,
    at::Tensor errorFunctionWeights,
    const std::vector<int>& weightsMap,
    const int64_t iBatch);
template std::unique_ptr<momentum::SequenceSolverFunctionT<double>>
buildSequenceSolverFunction<double>(
    const momentum::Character& character,
    const momentum::ParameterTransformT<double>& parameterTransform,
    at::Tensor modelParams_init,
    const momentum::ParameterSet& sharedParams,
    const std::vector<std::unique_ptr<TensorErrorFunction<double>>>& errorFunctions,
    at::Tensor errorFunctionWeights,
    const std::vector<int>& weightsMap,
    const int64_t iBatch);

template std::vector<ErrorFunctionInput<float>> buildErrorFunctionInputs(
    const std::vector<std::unique_ptr<TensorErrorFunction<float>>>& errorFunctions,
    const std::vector<int>& weightsMap);
template std::vector<ErrorFunctionInput<double>> buildErrorFunctionInputs(
    const std::vector<std::unique_ptr<TensorErrorFunction<double>>>& errorFunctions,
    const std::vector<int>& weightsMap);

template std::vector<at::Tensor> toTensors<float>(
    const std::vector<std::unique_ptr<TensorErrorFunction<float>>>& errorFunctions,
    const std::vector<ErrorFunctionInput<float>>& inputs_all);
template std::vector<at::Tensor> toTensors<double>(
    const std::vector<std::unique_ptr<TensorErrorFunction<double>>>& errorFunctions,
    const std::vector<ErrorFunctionInput<double>>& inputs_all);

} // namespace pymomentum::detail
