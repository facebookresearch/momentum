/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <pymomentum/tensor_ik/solver_options.h>
#include <pymomentum/tensor_ik/tensor_error_function.h>

#include <momentum/character/character.h>
#include <momentum/character/types.h>
#include <momentum/character_solver/vertex_error_function.h>
#include <momentum/math/mppca.h>

#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/autograd/autograd.h>

#include <memory>
#include <vector>

namespace pymomentum {

enum class ErrorFunctionType;

struct TensorMppcaModel {
  at::Tensor pi;
  at::Tensor mu;
  at::Tensor W;
  at::Tensor sigma;
  at::Tensor parameterIndices;

  pybind11::object mppca;
};

TensorMppcaModel extractMppcaModel(pybind11::object model_in);

template <typename T>
std::pair<std::vector<std::unique_ptr<TensorErrorFunction<T>>>, std::vector<int>> createIKProblem(
    const std::vector<const momentum::Character*>& characters,
    int64_t nBatch,
    int64_t nFrames,
    const std::vector<ErrorFunctionType>& activeErrorFunctions,
    at::Tensor positionCons_parents,
    at::Tensor positionCons_offsets,
    at::Tensor positionCons_weights,
    at::Tensor positionCons_targets,
    at::Tensor orientation_parents,
    at::Tensor orientation_offsets,
    at::Tensor orientation_weights,
    at::Tensor orientation_targets,
    at::Tensor posePrior_pi,
    at::Tensor posePrior_mu,
    at::Tensor posePrior_W,
    at::Tensor posePrior_sigma,
    at::Tensor posePrior_parameterIndices,
    const momentum::Mppca* posePrior_model,
    at::Tensor motion_targets,
    at::Tensor motion_weights,
    at::Tensor projectionCons_projections,
    at::Tensor projectionCons_parents,
    at::Tensor projectionCons_offsets,
    at::Tensor projectionCons_weights,
    at::Tensor projectionCons_targets,
    at::Tensor distanceCons_origins,
    at::Tensor distanceCons_parents,
    at::Tensor distanceCons_offsets,
    at::Tensor distanceCons_weights,
    at::Tensor distanceCons_targets,
    at::Tensor vertexCons_vertices,
    at::Tensor vertexCons_weights,
    at::Tensor vertexCons_target_positions,
    at::Tensor vertexCons_target_normals,
    momentum::VertexConstraintType vertexCons_type,
    at::Tensor vertexProjCons_vertices,
    at::Tensor vertexProjCons_weights,
    at::Tensor vertexProjCons_target_positions,
    at::Tensor vertexProjCons_projections);

template <typename T>
class IKSolveFunction {
 public:
  static constexpr int N_RESULTS = 1;
  static constexpr bool SEQUENCE = false;

  static std::vector<at::Tensor> forward(
      const std::vector<const momentum::Character*>& characters,
      const momentum::ParameterSet& activeParams,
      const momentum::ParameterSet& sharedParams,
      const SolverOptions& options,
      at::Tensor modelParams_init,
      const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
      at::Tensor errorFunctionWeights,
      size_t numActiveErrorFunctions,
      const std::vector<int>& weightsMap);

  static std::tuple<at::Tensor, at::Tensor, std::vector<at::Tensor>> backward(
      const std::vector<const momentum::Character*>& characters,
      const momentum::ParameterSet& activeParams,
      const momentum::ParameterSet& sharedParams,
      const at::Tensor& modelParams_init,
      const std::vector<at::Tensor>& results,
      const std::vector<at::Tensor>& dLoss_dResults,
      const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
      at::Tensor errorFunctionWeights,
      size_t numActiveErrorFunctions,
      const std::vector<int>& weightsMap);
};

template <typename T>
class GradientFunction {
 public:
  static constexpr int N_RESULTS = 1;
  static constexpr bool SEQUENCE = false;

  static std::vector<at::Tensor> forward(
      const std::vector<const momentum::Character*>& characters,
      const momentum::ParameterSet& activeParams,
      const momentum::ParameterSet& sharedParams,
      const SolverOptions& options,
      at::Tensor modelParams_init,
      const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
      at::Tensor errorFunctionWeights,
      size_t numActiveErrorFunctions,
      const std::vector<int>& weightsMap);

  static std::tuple<at::Tensor, at::Tensor, std::vector<at::Tensor>> backward(
      const std::vector<const momentum::Character*>& characters,
      const momentum::ParameterSet& activeParams,
      const momentum::ParameterSet& sharedParams,
      at::Tensor modelParams_init,
      const std::vector<at::Tensor>& results,
      const std::vector<at::Tensor>& dLoss_dResults,
      const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
      at::Tensor errorFunctionWeights,
      size_t numActiveErrorFunctions,
      const std::vector<int>& weightsMap);
};

template <typename T>
class ResidualFunction {
 public:
  static constexpr int N_RESULTS = 2;
  static constexpr bool SEQUENCE = false;

  static std::vector<at::Tensor> forward(
      const std::vector<const momentum::Character*>& characters,
      const momentum::ParameterSet& activeParams,
      const momentum::ParameterSet& sharedParams,
      const SolverOptions& options,
      at::Tensor modelParams_init,
      const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
      at::Tensor errorFunctionWeights,
      size_t numActiveErrorFunctions,
      const std::vector<int>& weightsMap);

  static std::tuple<at::Tensor, at::Tensor, std::vector<at::Tensor>> backward(
      const std::vector<const momentum::Character*>& characters,
      const momentum::ParameterSet& activeParams,
      const momentum::ParameterSet& sharedParams,
      at::Tensor modelParams_init,
      const std::vector<at::Tensor>& results,
      const std::vector<at::Tensor>& dLoss_dResults,
      const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
      at::Tensor errorFunctionWeights,
      size_t numActiveErrorFunctions,
      const std::vector<int>& weightsMap);
};

template <typename T>
class SequenceIKSolveFunction {
 public:
  static constexpr int N_RESULTS = 1;
  static constexpr bool SEQUENCE = true;

  static std::vector<at::Tensor> forward(
      const std::vector<const momentum::Character*>& characters,
      const momentum::ParameterSet& activeParams,
      const momentum::ParameterSet& sharedParams,
      const SolverOptions& options,
      at::Tensor modelParams_init,
      const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
      at::Tensor errorFunctionWeights,
      size_t numActiveErrorFunctions,
      const std::vector<int>& weightsMap);

  static std::tuple<at::Tensor, at::Tensor, std::vector<at::Tensor>> backward(
      const std::vector<const momentum::Character*>& characters,
      const momentum::ParameterSet& activeParams,
      const momentum::ParameterSet& sharedParams,
      const at::Tensor& modelParams_init,
      const std::vector<at::Tensor>& results,
      const std::vector<at::Tensor>& dLoss_dResults,
      const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
      const at::Tensor& errorFunctionWeights,
      size_t numActiveErrorFunctions,
      const std::vector<int>& weightsMap);
};

// This has been templated on both the scalar type and the "IKFunction",
// where an IKFunction is something that takes an IKProblem as input and returns
// a quantity such as the modelParameters where the error is minimized or the
// gradient/Jacobian/residual.  Note that the IKFunction needs to have both a
// forward function, which returns the results of the function as a list of
// tensors, and a backward() function, which takes the gradient wrt the
// forward() outputs and returns the tuple:
//   (grad_modelParams, grad_errorFunctionWeights, [grad_errorFunctionInputs])
template <typename T, template <typename> class IKFunction>
struct IKProblemAutogradFunction
    : public torch::autograd::Function<IKProblemAutogradFunction<T, IKFunction>> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      PyObject* characters,
      at::Tensor activeParams_in,
      at::Tensor sharedParams_in,
      SolverOptions solverOptions,
      at::Tensor modelParameters_init,
      const std::vector<ErrorFunctionType>& activeErrorFunctions,
      at::Tensor errorFunctionWeights,
      at::Tensor positionCons_parents,
      at::Tensor positionCons_offsets,
      at::Tensor positionCons_weights,
      at::Tensor positionCons_targets,
      at::Tensor orientation_parents,
      at::Tensor orientation_offsets,
      at::Tensor orientation_weights,
      at::Tensor orientation_targets,
      at::Tensor posePrior_pi,
      at::Tensor posePrior_mu,
      at::Tensor posePrior_W,
      at::Tensor posePrior_sigma,
      at::Tensor posePrior_parameterIndices,
      PyObject* posePrior_model,
      at::Tensor motion_targets,
      at::Tensor motion_weights,
      at::Tensor projectionCons_projections,
      at::Tensor projectionCons_parents,
      at::Tensor projectionCons_offsets,
      at::Tensor projectionCons_weights,
      at::Tensor projectionCons_targets,
      at::Tensor distanceCons_origins,
      at::Tensor distanceCons_parents,
      at::Tensor distanceCons_offsets,
      at::Tensor distanceCons_weights,
      at::Tensor distanceCons_targets,
      at::Tensor vertexCons_vertices,
      at::Tensor vertexCons_weights,
      at::Tensor vertexCons_target_positions,
      at::Tensor vertexCons_target_normals,
      momentum::VertexConstraintType vertexCons_type,
      at::Tensor vertexProjCons_vertices,
      at::Tensor vertexProjCons_weights,
      at::Tensor vertexProjCons_target_positions,
      at::Tensor vertexProjCons_projections);

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_modelParameters);
};

} // namespace pymomentum
