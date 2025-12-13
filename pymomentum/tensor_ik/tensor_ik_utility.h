/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <pymomentum/tensor_ik/tensor_error_function.h>

#include <momentum/character/character.h>
#include <momentum/character_sequence_solver/sequence_solver_function.h>
#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/character_solver/skeleton_solver_function.h>

#include <ATen/ATen.h>
#include <torch/torch.h>

#include <memory>
#include <vector>

// Utility functions that can be reused for different IK-related functions (e.g.
// computeGradient, computeResidual, etc.).
// These functions should only be used internally to the TensorIK library.
namespace pymomentum::detail {

// Utility function to check if it's a valid IK problem.  Returns the "fixed"
// modelParams and errorFunctionWeights.
template <typename T>
std::tuple<at::Tensor, at::Tensor> checkIKInputs(
    const std::vector<const momentum::Character*>& characters,
    at::Tensor modelParams,
    at::Tensor errorFunctionWeights,
    size_t numActiveErrorFunctions,
    const char* context,
    bool* squeezeErrorFunctionWeights = nullptr);

template <typename T>
std::tuple<at::Tensor, at::Tensor> checkSequenceIKInputs(
    const std::vector<const momentum::Character*>& characters,
    at::Tensor modelParams,
    at::Tensor errorFunctionWeights,
    size_t numActiveErrorFunctions,
    const char* context);

template <typename T>
std::vector<std::shared_ptr<momentum::SkeletonErrorFunctionT<T>>> buildMomentumErrorFunctions(
    const std::vector<const momentum::Character*>& characters,
    const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
    at::Tensor errorFunctionWeights,
    const std::vector<int>& weightsMap,
    int64_t iBatch);

template <typename T>
momentum::SkeletonSolverFunctionT<T> buildSolverFunction(
    const momentum::Character& character,
    const momentum::ParameterTransformT<T>& parameterTransform,
    const std::vector<std::shared_ptr<momentum::SkeletonErrorFunctionT<T>>>& errorFunctions);

template <typename T>
std::unique_ptr<momentum::SequenceSolverFunctionT<T>> buildSequenceSolverFunction(
    const momentum::Character& character,
    const momentum::ParameterTransformT<T>& parameterTransform,
    at::Tensor modelParams_init,
    const momentum::ParameterSet& sharedParams,
    const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
    at::Tensor errorFunctionWeights,
    const std::vector<int>& weightsMap,
    int64_t iBatch);

// This struct is to keep track of where the individual inputs are wrt the
// single longer list of inputs that we return from the ::backward() function.
template <typename T>
struct ErrorFunctionInput {
  ErrorFunctionInput(size_t iErrFunction, size_t jInput)
      : iErrorFunction(iErrFunction), jInput(jInput) {}

  // Which error function this input refers to
  size_t iErrorFunction = SIZE_MAX;

  // Which of the inputs of that particular error function
  size_t jInput = SIZE_MAX;

  // derivative of the loss wrt this input:
  std::vector<Eigen::VectorX<T>> dLoss_dInput;
};

template <typename T>
std::vector<ErrorFunctionInput<T>> buildErrorFunctionInputs(
    const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
    const std::vector<int>& weightsMap);

template <typename T>
torch::autograd::variable_list toTensors(
    const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
    const std::vector<ErrorFunctionInput<T>>& inputs_all);

} // namespace pymomentum::detail
