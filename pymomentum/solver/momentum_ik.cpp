/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/solver/momentum_ik.h"

#include "momentum/character/types.h"
#include "momentum/character_solver/transform_pose.h"
#include "momentum/math/types.h"
#include "pymomentum/python_utility/python_utility.h"
#include "pymomentum/tensor_ik/tensor_collision_error_function.h"
#include "pymomentum/tensor_ik/tensor_diff_pose_prior_error_function.h"
#include "pymomentum/tensor_ik/tensor_distance_error_function.h"
#include "pymomentum/tensor_ik/tensor_error_function.h"
#include "pymomentum/tensor_ik/tensor_gradient.h"
#include "pymomentum/tensor_ik/tensor_ik.h"
#include "pymomentum/tensor_ik/tensor_limit_error_function.h"
#include "pymomentum/tensor_ik/tensor_marker_error_function.h"
#include "pymomentum/tensor_ik/tensor_motion_error_function.h"
#include "pymomentum/tensor_ik/tensor_pose_prior_error_function.h"
#include "pymomentum/tensor_ik/tensor_projection_error_function.h"
#include "pymomentum/tensor_ik/tensor_residual.h"
#include "pymomentum/tensor_ik/tensor_vertex_error_function.h"
#include "pymomentum/tensor_ik/tensor_vertex_projection_error_function.h"
#include "pymomentum/tensor_momentum/tensor_momentum_utility.h"
#include "pymomentum/tensor_momentum/tensor_parameter_transform.h"
#include "pymomentum/tensor_utility/autograd_utility.h"
#include "pymomentum/tensor_utility/tensor_utility.h"

#ifndef PYMOMENTUM_LIMITED_TORCH_API
#include <torch/csrc/jit/python/python_ivalue.h>
#endif

// Details on how PyTorch works for building a custom differentiable module:
//
// For official introduction, see:
// https://pytorch.org/docs/stable/notes/extending.html and
// https://pytorch.org/tutorials/advanced/cpp_autograd.html#using-custom-autograd-function-in-c
//
// More details as follows:
//
// torch.autograd.Function:
// https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function
// Function models a math operation done on tensors. It records operation
// history and defines formulas for forward and backward operations.
//
// We first inherit torch::autograd::Function to define our custom operation in
// C++.
//
// autograd:
// https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-tensors-and-autograd
// The automatic differentation in PyTorch works by defining a computational
// graph, where each node is a Tensor and each edge is a Function. If x is a
// Tensor that has x.requires_grad=True then x.grad is another Tensor holding
// the gradient of x with respect to some scalar value.
// Default creation of a Tensor has requires_grad set to false.
//
// To implement the forward() in custom Function, one can use the first function
// parameter, AutogradContext to save input tensor values for computation in
// backward(), by calling ctx->save_for_backward(...). For additional non-tensor
// data to be saved in forward(), use ctx->saved_data to store them. See below
// for brief introduction:
// https://pytorch.org/cppdocs/api/structtorch_1_1autograd_1_1_function.html#struct-documentation
// Each entry in ctx->saved_data is a <std::string, at::IValue> pair.
// at::IValue is Interpreter Value, a wrapper for any value. According to:
// https://pytorch.org/cppdocs/api/structc10_1_1_i_value.html
//

using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

namespace py = pybind11;

namespace pymomentum {

namespace {

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
  static variable_list forward(
      AutogradContext* ctx,
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

  static variable_list backward(AutogradContext* ctx, variable_list grad_modelParameters);
};

// Create a vector of TensorErrorFunction from given error function weights.
template <typename T>
std::pair<std::vector<std::unique_ptr<TensorErrorFunction<T>>>, std::vector<int>> createIKProblem(
    const std::vector<const momentum::Character*>& characters,
    const int64_t nBatch,
    const int64_t nFrames, // Number of frames, for a sequence/multi-frame
                           // IK problem; should be 0 otherwise
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
    at::Tensor vertexProjCons_projections) {
  MT_THROW_IF(characters.size() != nBatch, "Mismatch between size of Character list and nBatch.");

  std::vector<std::unique_ptr<TensorErrorFunction<T>>> errorFunctions;

  auto numErrorTypes = static_cast<size_t>(ErrorFunctionType::NumTypes);

  // To make the gradients computed from the backward pass to have the correct
  // order as the order of the input parameter for IK solve, we need to maintain
  // the order of the error function passed to the backward solve.

  // So we need a mapping between input types (with arbitrary type order) and
  // ordered types defined in enum (the same order as the function parameter
  // definition).

  // errorFunctionMap: map indices from enum order to input active type order.
  // Value of -1 means this type is not active.
  std::vector<int> errorFunctionMap(numErrorTypes, -1);
  for (size_t iErr = 0; iErr < activeErrorFunctions.size(); ++iErr) {
    auto typeIndex = static_cast<size_t>(activeErrorFunctions[iErr]);
    // Check whether we have duplicated error function types here:
    MT_THROW_IF(
        errorFunctionMap[typeIndex] != -1,
        "solveBodyIKProblem(): active_error_functions have duplicate types")
    errorFunctionMap[typeIndex] = iErr;
  }

  // All characters are guaranteed to have the same skeleton sizes so it's safe
  // to compare against just the first.
  checkValidBoneIndex(positionCons_parents, *characters.front(), "position_cons_parents");
  checkValidBoneIndex(orientation_parents, *characters.front(), "orientation_cons_parents");
  checkValidBoneIndex(projectionCons_parents, *characters.front(), "projection_cons_parents");
  checkValidBoneIndex(distanceCons_parents, *characters.front(), "distance_cons_parents");
  checkValidParameterIndex(
      posePrior_parameterIndices, *characters.front(), "posePrior_parameterIndices", true);
  checkValidVertexIndex(vertexCons_vertices, *characters.front(), "vertex_cons_vertices", false);
  checkValidVertexIndex(
      vertexProjCons_vertices, *characters.front(), "vertex_cons_proj_vertices", false);

  // Build each type of error function according to enum order:
  errorFunctions.push_back(
      createPositionErrorFunction<T>(
          nBatch,
          nFrames,
          positionCons_parents,
          positionCons_offsets,
          positionCons_weights,
          positionCons_targets));
  errorFunctions.push_back(
      createOrientationErrorFunction<T>(
          nBatch,
          nFrames,
          orientation_parents,
          orientation_offsets,
          orientation_weights,
          orientation_targets));
  errorFunctions.push_back(createLimitErrorFunction<T>(nBatch, nFrames));
  errorFunctions.push_back(createCollisionErrorFunction<T>(nBatch, nFrames));
  if (posePrior_model != nullptr) {
    errorFunctions.push_back(createPosePriorErrorFunction<T>(nBatch, nFrames, posePrior_model));
  } else {
    errorFunctions.push_back(
        createDiffPosePriorErrorFunction<T>(
            nBatch,
            nFrames,
            posePrior_pi,
            posePrior_mu,
            posePrior_W,
            posePrior_sigma,
            posePrior_parameterIndices));
  }
  errorFunctions.push_back(
      createMotionErrorFunction<T>(
          nBatch, nFrames, characters[0]->parameterTransform, motion_targets, motion_weights));
  errorFunctions.push_back(
      createProjectionErrorFunction<T>(
          nBatch,
          nFrames,
          projectionCons_projections,
          projectionCons_parents,
          projectionCons_offsets,
          projectionCons_weights,
          projectionCons_targets));
  errorFunctions.push_back(
      createDistanceErrorFunction<T>(
          nBatch,
          nFrames,
          distanceCons_origins,
          distanceCons_parents,
          distanceCons_offsets,
          distanceCons_weights,
          distanceCons_targets));
  errorFunctions.push_back(
      createVertexErrorFunction<T>(
          nBatch,
          nFrames,
          vertexCons_vertices,
          vertexCons_weights,
          vertexCons_target_positions,
          vertexCons_target_normals,
          vertexCons_type));
  errorFunctions.push_back(
      createVertexProjectionErrorFunction<T>(
          nBatch,
          nFrames,
          vertexProjCons_vertices,
          vertexProjCons_weights,
          vertexProjCons_target_positions,
          vertexProjCons_projections));

  return std::make_pair(std::move(errorFunctions), std::move(errorFunctionMap));
}

template <typename T, template <typename> class IKFunction>
variable_list IKProblemAutogradFunction<T, IKFunction>::forward(
    AutogradContext* ctx,
    PyObject* characters_in,
    at::Tensor activeParams, // Model parameters that are actually solved for.
    at::Tensor sharedParams, // Model parameters that are "shared" across frames
                             // in the case of a sequence/multi-frame IK problem
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
    at::Tensor vertexProjCons_projections) {
  // Input modelParameters_init can be batched or unbatched.
  // If unbatched, unsqueeze it by adding a batch dim of 1.
  // If input modelParameters_init is unbatched, remember this
  // in bool squeeze so that the output is also unbatched.
  bool squeeze = false;
  int64_t nFrames = 0;
  if (IKFunction<T>::SEQUENCE) {
    if (modelParameters_init.ndimension() == 1 || modelParameters_init.ndimension() > 3) {
      MT_THROW(
          "In IKProblemAutogradFunction<T, IKFunction>::forward(), multi-frame problem expects a model_parameters tensor of size [nBatch] x nFrames x nModelParameters; got {}",
          formatTensorSizes(modelParameters_init));
    } else if (modelParameters_init.ndimension() == 2) {
      modelParameters_init = modelParameters_init.unsqueeze(0);
      squeeze = true;
    }

    nFrames = modelParameters_init.size(1);
  } else {
    if (modelParameters_init.ndimension() == 1) {
      modelParameters_init = modelParameters_init.unsqueeze(0);
      squeeze = true;
    }

    nFrames = 0;
  }

  const int64_t nBatch = modelParameters_init.size(0);

  const auto characters = toCharacterList(characters_in, nBatch, "solveBodyIKProblem()");

  const momentum::Mppca* mppca = nullptr;
  if (posePrior_model) {
    mppca = py::cast<const momentum::Mppca*>(posePrior_model);
  }

  // Create a vector of pymomentum::TensorErrorFunctions:
  const auto [errorFunctions, errorWeightsMap] = createIKProblem<T>(
      characters,
      nBatch,
      nFrames,
      activeErrorFunctions,
      positionCons_parents,
      positionCons_offsets,
      positionCons_weights,
      positionCons_targets,
      orientation_parents,
      orientation_offsets,
      orientation_weights,
      orientation_targets,
      posePrior_pi,
      posePrior_mu,
      posePrior_W,
      posePrior_sigma,
      posePrior_parameterIndices,
      mppca,
      motion_targets,
      motion_weights,
      projectionCons_projections,
      projectionCons_parents,
      projectionCons_offsets,
      projectionCons_weights,
      projectionCons_targets,
      distanceCons_origins,
      distanceCons_parents,
      distanceCons_offsets,
      distanceCons_weights,
      distanceCons_targets,
      vertexCons_vertices,
      vertexCons_weights,
      vertexCons_target_positions,
      vertexCons_target_normals,
      vertexCons_type,
      vertexProjCons_vertices,
      vertexProjCons_weights,
      vertexProjCons_target_positions,
      vertexProjCons_projections);

  // Do the actual IK/FK operation to get the solved model parameters:
  std::vector<at::Tensor> results = IKFunction<T>::forward(
      characters,
      tensorToParameterSet(
          characters[0]->parameterTransform, activeParams, DefaultParameterSet::ALL_ONES),
      tensorToParameterSet(
          characters[0]->parameterTransform, sharedParams, DefaultParameterSet::ALL_ZEROS),
      solverOptions,
      modelParameters_init,
      errorFunctions,
      errorFunctionWeights,
      activeErrorFunctions.size(),
      errorWeightsMap);

  // Check for nan or inf:
  for (const auto& tensor : results) {
    throwIfNaNOrINF(
        tensor,
        "IKProblemAutogradFunction<T, IKFunction>::forward()",
        "result from IKFunction<T>::forward()");
  }

  // Recover the unbatched form of model parameters as output, to be
  // consistent with input model parameters.
  if (squeeze) {
    modelParameters_init = modelParameters_init.squeeze(0);
    for (auto& r : results) {
      r = r.squeeze(0);
    }
  }

  // Save non-Tensor data for gradient computation in backward():
  // First cast pointer to momentum's class to py::object, then
  // use ConcretePyObjectHolder::create() to cast it to
  // c0::intrusive_ptr<PyObject>, which is then assigned to c10::IValue.
  //
  // Note: according to https://github.com/pytorch/pytorch/pull/30136
  // letting a PyObject to be held by IValue seems not a part of public
  // API and should not be used by user.

#ifndef PYMOMENTUM_LIMITED_TORCH_API
  ctx->saved_data["characters"] = c10::ivalue::ConcretePyObjectHolder::create(characters_in);

  if (posePrior_model) {
    ctx->saved_data["posePrior"] = c10::ivalue::ConcretePyObjectHolder::create(posePrior_model);
  }
#else
  // When LIMITED_TORCH_API is defined, ConcretePyObjectHolder is not available.
  // Note: backward pass will fail in this build configuration.
  (void)characters_in; // Suppress unused variable warning
  (void)posePrior_model; // Suppress unused variable warning
#endif

  // We convert vector<ErrorFunctionType> to vector<int> because pytorch api
  // cannot save a vector of custom enum type.
  std::vector<int64_t> activatedErrorTypes(activeErrorFunctions.size());
  for (size_t i = 0; i < activeErrorFunctions.size(); ++i) {
    MT_THROW_IF(
        i >= activatedErrorTypes.size(),
        "Index {} exceeds activatedErrorTypes size {}",
        i,
        activatedErrorTypes.size());
    activatedErrorTypes[i] = static_cast<int>(activeErrorFunctions[i]);
  }
  ctx->saved_data["activeErrorFunctions"] = activatedErrorTypes;

  ctx->saved_data["vertexCons_type"] = std::vector<int64_t>(1, (int64_t)vertexCons_type);

  // Save Tensors for gradient computation in backward():
  torch::autograd::variable_list to_save = {
      activeParams,
      sharedParams,
      modelParameters_init,
      errorFunctionWeights,
      positionCons_parents,
      positionCons_offsets,
      positionCons_weights,
      positionCons_targets,
      orientation_parents,
      orientation_offsets,
      orientation_weights,
      orientation_targets,
      posePrior_pi,
      posePrior_mu,
      posePrior_W,
      posePrior_sigma,
      posePrior_parameterIndices,
      motion_targets,
      motion_weights,
      projectionCons_projections,
      projectionCons_parents,
      projectionCons_offsets,
      projectionCons_weights,
      projectionCons_targets,
      distanceCons_origins,
      distanceCons_parents,
      distanceCons_offsets,
      distanceCons_weights,
      distanceCons_targets,
      vertexCons_vertices,
      vertexCons_weights,
      vertexCons_target_positions,
      vertexCons_target_normals,
      vertexProjCons_vertices,
      vertexProjCons_weights,
      vertexProjCons_target_positions,
      vertexProjCons_projections};
  std::copy(std::begin(results), std::end(results), std::back_inserter(to_save));
  ctx->save_for_backward(to_save);

  return results;
}

template <typename T, template <typename> class IKFunction>
variable_list IKProblemAutogradFunction<T, IKFunction>::backward(
    AutogradContext* ctx,
    variable_list dLoss_dResults) {
  MT_THROW_IF(
      dLoss_dResults.size() != IKFunction<T>::N_RESULTS,
      "Invalid grad_outputs in IKProblemAutogradFunction::backward.")

  const momentum::Mppca* mppca = nullptr;
#ifndef PYMOMENTUM_LIMITED_TORCH_API
  if (ctx->saved_data.find("posePrior") != ctx->saved_data.end()) {
    mppca = py::cast<const momentum::Mppca*>(ctx->saved_data["posePrior"].toPyObject());
  }
#endif

  const auto activatedErrorTypes = ctx->saved_data["activeErrorFunctions"].toIntVector();
  std::vector<ErrorFunctionType> activeErrorFunctions(activatedErrorTypes.size());
  for (size_t i = 0; i < activatedErrorTypes.size(); ++i) {
    activeErrorFunctions[i] = static_cast<ErrorFunctionType>(activatedErrorTypes[i]);
  }

  const auto vertexCons_type_vec = ctx->saved_data["vertexCons_type"].toIntVector();
  momentum::VertexConstraintType vertexCons_type = momentum::VertexConstraintType::Position;
  if (vertexCons_type_vec.size() == 1) {
    vertexCons_type = (momentum::VertexConstraintType)vertexCons_type_vec.front();
  } else {
    // Can't throw here, it causes Pytorch to crash.
    std::cerr << "WARNING: vertex constraint type improperly saved.\n";
  }

  // Get all saved tensors:
  const auto saved = ctx->get_saved_variables();
  auto savedItr = std::begin(saved);
  auto checkAndIncrementSavedItr = [&]() -> at::Tensor {
    MT_THROW_IF(savedItr == std::end(saved), "Not enough saved variables.");
    return *savedItr++;
  };
  auto activeParams = checkAndIncrementSavedItr();
  auto sharedParams = checkAndIncrementSavedItr();
  auto modelParameters_init = checkAndIncrementSavedItr();
  auto errorFunctionWeights = checkAndIncrementSavedItr();
  auto positionCons_parents = checkAndIncrementSavedItr();
  auto positionCons_offsets = checkAndIncrementSavedItr();
  auto positionCons_weights = checkAndIncrementSavedItr();
  auto positionCons_targets = checkAndIncrementSavedItr();
  auto orientation_parents = checkAndIncrementSavedItr();
  auto orientation_offsets = checkAndIncrementSavedItr();
  auto orientation_weights = checkAndIncrementSavedItr();
  auto orientation_targets = checkAndIncrementSavedItr();
  auto posePrior_pi = checkAndIncrementSavedItr();
  auto posePrior_mu = checkAndIncrementSavedItr();
  auto posePrior_W = checkAndIncrementSavedItr();
  auto posePrior_sigma = checkAndIncrementSavedItr();
  auto posePrior_parameterIndices = checkAndIncrementSavedItr();
  auto motion_targets = checkAndIncrementSavedItr();
  auto motion_weights = checkAndIncrementSavedItr();
  auto projectionCons_projections = checkAndIncrementSavedItr();
  auto projectionCons_parents = checkAndIncrementSavedItr();
  auto projectionCons_offsets = checkAndIncrementSavedItr();
  auto projectionCons_weights = checkAndIncrementSavedItr();
  auto projectionCons_targets = checkAndIncrementSavedItr();
  auto distanceCons_origins = checkAndIncrementSavedItr();
  auto distanceCons_parents = checkAndIncrementSavedItr();
  auto distanceCons_offsets = checkAndIncrementSavedItr();
  auto distanceCons_weights = checkAndIncrementSavedItr();
  auto distanceCons_targets = checkAndIncrementSavedItr();
  auto vertexCons_vertices = checkAndIncrementSavedItr();
  auto vertexCons_weights = checkAndIncrementSavedItr();
  auto vertexCons_target_positions = checkAndIncrementSavedItr();
  auto vertexCons_target_normals = checkAndIncrementSavedItr();
  auto vertexProjCons_vertices = checkAndIncrementSavedItr();
  auto vertexProjCons_weights = checkAndIncrementSavedItr();
  auto vertexProjCons_target_positions = checkAndIncrementSavedItr();
  auto vertexProjCons_projections = checkAndIncrementSavedItr();

  // Get stored results from the forward pass, which contains the solved body
  // model params of the IK problem.
  std::vector<at::Tensor> results;
  while (savedItr != std::end(saved)) {
    results.push_back(*savedItr++);
  }

  MT_THROW_IF(
      results.size() != IKFunction<T>::N_RESULTS,
      "Mismatch in saved variable counts in IKProblemAutogradFunction::backward");

  for (auto& t : dLoss_dResults) {
    t = t.contiguous().to(at::DeviceType::CPU, toScalarType<T>());
  }

  int expectedDim = 2;
  int64_t nFrames = 0;
  if (IKFunction<T>::SEQUENCE) {
    expectedDim = 3;
    nFrames = modelParameters_init.size(-2);
  }

  // Add batch dimension if needed:
  if (modelParameters_init.ndimension() < expectedDim) {
    modelParameters_init = modelParameters_init.unsqueeze(0);
    for (auto& t : dLoss_dResults) {
      t = t.unsqueeze(0);
    }
    for (auto& res : results) {
      res = res.unsqueeze(0);
    }
  }
  const auto nBatch = modelParameters_init.size(0);

  // Restore variables:
#ifndef PYMOMENTUM_LIMITED_TORCH_API
  const auto characters =
      toCharacterList(ctx->saved_data["characters"].toPyObject(), nBatch, "solveBodyIKProblem()");
#else
  // When LIMITED_TORCH_API is defined, toPyObject() is not available.
  MT_THROW("Backward pass is not supported when PYMOMENTUM_LIMITED_TORCH_API is defined");
  const std::vector<const momentum::Character*> characters;
#endif

  // Recreate the IK problem:
  const auto [errorFunctions, errorWeightsMap] = createIKProblem<T>(
      characters,
      nBatch,
      nFrames,
      activeErrorFunctions,
      positionCons_parents,
      positionCons_offsets,
      positionCons_weights,
      positionCons_targets,
      orientation_parents,
      orientation_offsets,
      orientation_weights,
      orientation_targets,
      posePrior_pi,
      posePrior_mu,
      posePrior_W,
      posePrior_sigma,
      posePrior_parameterIndices,
      mppca,
      motion_targets,
      motion_weights,
      projectionCons_projections,
      projectionCons_parents,
      projectionCons_offsets,
      projectionCons_weights,
      projectionCons_targets,
      distanceCons_origins,
      distanceCons_parents,
      distanceCons_offsets,
      distanceCons_weights,
      distanceCons_targets,
      vertexCons_vertices,
      vertexCons_weights,
      vertexCons_target_positions,
      vertexCons_target_normals,
      vertexCons_type,
      vertexProjCons_vertices,
      vertexProjCons_weights,
      vertexProjCons_target_positions,
      vertexProjCons_projections);

  // Compute the gradient of the IK problem:
  const auto [grad_modelParams, grad_errorFunctionWeights, grad_inputs] = IKFunction<T>::backward(
      characters,
      tensorToParameterSet(
          characters[0]->parameterTransform, activeParams, DefaultParameterSet::ALL_ONES),
      tensorToParameterSet(
          characters[0]->parameterTransform, sharedParams, DefaultParameterSet::ALL_ZEROS),
      modelParameters_init,
      results,
      dLoss_dResults,
      errorFunctions,
      errorFunctionWeights,
      activeErrorFunctions.size(),
      errorWeightsMap);

  // Check for nan or inf:
  const char* context = "IKProblemAutogradFunction<T, IKFunction>::backward()";
  throwIfNaNOrINF(grad_modelParams, context, "grad_modelParams from IKFunction<T>::backward()");
  throwIfNaNOrINF(
      grad_errorFunctionWeights,
      context,
      "grad_errorFunctionWeights fromIKFunction<T>::backward() ");
  for (const auto& tensor : grad_inputs) {
    throwIfNaNOrINF(tensor, context, "one grad_input from IKFunction<T>::backward()");
  }

  // Insert 2 at the beginning of the list for:
  //    const momentum::Character* character,
  //    at::Tensor activeParams,

  // Then comes dLoss_dModelParams_init, then insert 1 for:
  //    const std::vector<ErrorFunctionType>& activeErrorFunctions,
  variable_list result;
  result.emplace_back(); // character
  result.emplace_back(); // activeParams
  result.emplace_back(); // sharedParams
  result.emplace_back(); // solverOptions
  result.push_back(grad_modelParams);
  result.emplace_back(); // activeErrorFunctions
  result.push_back(grad_errorFunctionWeights);
  std::copy(grad_inputs.begin(), grad_inputs.end(), std::back_inserter(result));

  return result;
}

template <typename T>
class IKSolveFunction {
 public:
  static constexpr int N_RESULTS = 1;
  static constexpr bool SEQUENCE = false;

  static std::vector<at::Tensor> forward(
      const std::vector<const momentum::Character*>& characters,
      [[maybe_unused]] const momentum::ParameterSet& activeParams,
      [[maybe_unused]] const momentum::ParameterSet& sharedParams,
      [[maybe_unused]] const SolverOptions& options,
      at::Tensor modelParams_init,
      const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
      at::Tensor errorFunctionWeights,
      size_t numActiveErrorFunctions,
      const std::vector<int>& weightsMap) {
    return std::vector<at::Tensor>{solveTensorIKProblem<T>(
        characters,
        activeParams,
        modelParams_init,
        errorFunctions,
        errorFunctionWeights,
        numActiveErrorFunctions,
        weightsMap,
        options)};
  }

  // Returns the tuple (grad_modelParams, grad_errorFunctionWeights,
  // [grad_input1, grad_input2, ...])
  static std::tuple<at::Tensor, at::Tensor, std::vector<at::Tensor>> backward(
      [[maybe_unused]] const std::vector<const momentum::Character*>& characters,
      [[maybe_unused]] const momentum::ParameterSet& activeParams,
      [[maybe_unused]] const momentum::ParameterSet& sharedParams,
      [[maybe_unused]] const at::Tensor& modelParams_init,
      const std::vector<at::Tensor>& results,
      const std::vector<at::Tensor>& dLoss_dResults,
      [[maybe_unused]] const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
      [[maybe_unused]] at::Tensor errorFunctionWeights,
      [[maybe_unused]] size_t numActiveErrorFunctions,
      [[maybe_unused]] const std::vector<int>& weightsMap) {
    MT_THROW_IF(
        results.size() != 1 || dLoss_dResults.size() != 1,
        "Mismatch in length of results list in IKSolveFunction::backward.");
    const at::Tensor& modelParams_final = results[0];
    const at::Tensor& dLoss_dModelParams = dLoss_dResults[0];

    auto [grad_errorFunctionWeights, grad_inputs] = d_solveTensorIKProblem<T>(
        characters,
        activeParams,
        modelParams_final,
        dLoss_dModelParams,
        errorFunctions,
        errorFunctionWeights,
        numActiveErrorFunctions,
        weightsMap);
    return {at::Tensor(), grad_errorFunctionWeights, grad_inputs};
  }
};

template <typename T>
class GradientFunction {
 public:
  static constexpr int N_RESULTS = 1;
  static constexpr bool SEQUENCE = false;

  static std::vector<at::Tensor> forward(
      const std::vector<const momentum::Character*>& characters,
      [[maybe_unused]] const momentum::ParameterSet& activeParams,
      [[maybe_unused]] const momentum::ParameterSet& sharedParams,
      [[maybe_unused]] const SolverOptions& options,
      at::Tensor modelParams_init,
      const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
      at::Tensor errorFunctionWeights,
      size_t numActiveErrorFunctions,
      const std::vector<int>& weightsMap) {
    return {computeGradient<T>(
        characters,
        modelParams_init,
        errorFunctions,
        errorFunctionWeights,
        numActiveErrorFunctions,
        weightsMap)};
  }

  static std::tuple<at::Tensor, at::Tensor, std::vector<at::Tensor>> backward(
      const std::vector<const momentum::Character*>& characters,
      [[maybe_unused]] const momentum::ParameterSet& activeParams,
      [[maybe_unused]] const momentum::ParameterSet& sharedParams,
      at::Tensor modelParams_init,
      const std::vector<at::Tensor>& results,
      const std::vector<at::Tensor>& dLoss_dResults,
      const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
      at::Tensor errorFunctionWeights,
      size_t numActiveErrorFunctions,
      const std::vector<int>& weightsMap) {
    MT_THROW_IF(
        results.size() != 1 || dLoss_dResults.size() != 1,
        "Mismatch in length of results list in IKSolveFunction::backward.")

    const at::Tensor& dLoss_dGradient = dLoss_dResults[0];
    return d_computeGradient<T>(
        characters,
        modelParams_init,
        dLoss_dGradient,
        errorFunctions,
        errorFunctionWeights,
        numActiveErrorFunctions,
        weightsMap);
  }
};

template <typename T>
class ResidualFunction {
 public:
  static constexpr int N_RESULTS = 2;
  static constexpr bool SEQUENCE = false;

  static std::vector<at::Tensor> forward(
      const std::vector<const momentum::Character*>& characters,
      [[maybe_unused]] const momentum::ParameterSet& activeParams,
      [[maybe_unused]] const momentum::ParameterSet& sharedParams,
      [[maybe_unused]] const SolverOptions& options,
      at::Tensor modelParams_init,
      const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
      at::Tensor errorFunctionWeights,
      size_t numActiveErrorFunctions,
      const std::vector<int>& weightsMap) {
    auto res = computeResidual<T>(
        characters,
        modelParams_init,
        errorFunctions,
        errorFunctionWeights,
        numActiveErrorFunctions,
        weightsMap);
    return {std::get<0>(res), std::get<1>(res)};
  }

  static std::tuple<at::Tensor, at::Tensor, std::vector<at::Tensor>> backward(
      const std::vector<const momentum::Character*>& characters,
      [[maybe_unused]] const momentum::ParameterSet& activeParams,
      [[maybe_unused]] const momentum::ParameterSet& sharedParams,
      at::Tensor modelParams_init,
      const std::vector<at::Tensor>& results,
      const std::vector<at::Tensor>& dLoss_dResults,
      const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
      at::Tensor errorFunctionWeights,
      size_t numActiveErrorFunctions,
      const std::vector<int>& weightsMap) {
    const at::Tensor& grad_resid = dLoss_dResults[0];
    const at::Tensor& jacobian = results[1];

    // Conveniently, the derivative of the residual wrt the model parameters is
    // just the Jacobian. J = ([nBatch] x m x nParam) grad_resid = ([nBatch] x m
    // x 1) We want to compute J^T*grad_resid = (grad_resid^T*J)^T
    at::Tensor grad_modelParams = at::bmm(grad_resid.unsqueeze(-2), jacobian).squeeze(-2);

    size_t nInputs = 0;
    for (const auto& e : errorFunctions) {
      nInputs += e->tensorInputs().size();
    }

    // We don't know how to take the derivative of the residual wrt anything
    // except the model params.
    return {grad_modelParams, at::Tensor(), std::vector<at::Tensor>(nInputs, at::Tensor())};
  }
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
      const SolverOptions options,
      at::Tensor modelParams_init,
      const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
      at::Tensor errorFunctionWeights,
      size_t numActiveErrorFunctions,
      const std::vector<int>& weightsMap) {
    return std::vector<at::Tensor>{solveTensorSequenceIKProblem<T>(
        characters,
        activeParams,
        sharedParams,
        modelParams_init,
        errorFunctions,
        errorFunctionWeights,
        numActiveErrorFunctions,
        weightsMap,
        options)};
  }

  // Returns the tuple (grad_modelParams, grad_errorFunctionWeights,
  // [grad_input1, grad_input2, ...])
  static std::tuple<at::Tensor, at::Tensor, std::vector<at::Tensor>> backward(
      [[maybe_unused]] const std::vector<const momentum::Character*>& characters,
      [[maybe_unused]] const momentum::ParameterSet& activeParams,
      [[maybe_unused]] const momentum::ParameterSet& sharedParams,
      [[maybe_unused]] const at::Tensor& modelParams_init,
      const std::vector<at::Tensor>& results,
      const std::vector<at::Tensor>& dLoss_dResults,
      [[maybe_unused]] const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
      [[maybe_unused]] const at::Tensor& errorFunctionWeights,
      [[maybe_unused]] size_t numActiveErrorFunctions,
      [[maybe_unused]] const std::vector<int>& weightsMap) {
    MT_THROW_IF(
        results.size() != 1 || dLoss_dResults.size() != 1,
        "Mismatch in length of results list in IKSolveFunction::backward.");

    return {at::Tensor(), at::Tensor(), std::vector<at::Tensor>()};
  }
};

// Version of applyTemplatedAutogradFunction that has been specialized for the
// IK functionality.
template <template <class> class IKFunction, class... Args>
inline torch::autograd::variable_list applyIKProblemAutogradFunction(Args... args) {
  pybind11::gil_scoped_release release;
  if (hasFloat64(args...)) {
    return IKProblemAutogradFunction<double, IKFunction>::apply(args...);
  } else {
    return IKProblemAutogradFunction<float, IKFunction>::apply(args...);
  }
}

struct TensorMppcaModel {
  at::Tensor pi;
  at::Tensor mu;
  at::Tensor W;
  at::Tensor sigma;
  at::Tensor parameterIndices;

  py::object mppca;
};

TensorMppcaModel extractMppcaModel(py::object model_in) {
  TensorMppcaModel result;
  if (!model_in) {
    // No model, just return immediately.
    return result;
  }

  if (PyList_Check(model_in.ptr())) {
    MT_THROW_IF(
        PyList_Size(model_in.ptr()) != 5,
        "For the pose prior Mppca model, expected a tuple of 5 tensors [pi, mu, W, sigma, parameterIndices].");

    result.pi = py::cast<at::Tensor>(PyList_GetItem(model_in.ptr(), 0));
    result.mu = py::cast<at::Tensor>(PyList_GetItem(model_in.ptr(), 1));
    result.W = py::cast<at::Tensor>(PyList_GetItem(model_in.ptr(), 2));
    result.sigma = py::cast<at::Tensor>(PyList_GetItem(model_in.ptr(), 3));
    result.parameterIndices = py::cast<at::Tensor>(PyList_GetItem(model_in.ptr(), 4));
  } else {
    result.pi = at::empty({0});
    result.mu = at::empty({0});
    result.W = at::empty({0});
    result.sigma = at::empty({0});
    result.parameterIndices = at::empty({0});

    result.mppca = model_in;
  }
  return result;
}

} // namespace

torch::Tensor solveBodyIKProblem(
    py::object characters,
    at::Tensor activeParams,
    at::Tensor modelParameters_init,
    const std::vector<ErrorFunctionType>& activeErrorFunctions,
    at::Tensor errorFunctionWeights,
    SolverOptions options,
    std::optional<at::Tensor> positionCons_parents,
    std::optional<at::Tensor> positionCons_offsets,
    std::optional<at::Tensor> positionCons_weights,
    std::optional<at::Tensor> positionCons_targets,
    std::optional<at::Tensor> orientation_parents,
    std::optional<at::Tensor> orientation_offsets,
    std::optional<at::Tensor> orientation_weights,
    std::optional<at::Tensor> orientation_targets,
    pybind11::object posePrior_model,
    std::optional<at::Tensor> motion_targets,
    std::optional<at::Tensor> motion_weights,
    std::optional<at::Tensor> projectionCons_projections,
    std::optional<at::Tensor> projectionCons_parents,
    std::optional<at::Tensor> projectionCons_offsets,
    std::optional<at::Tensor> projectionCons_weights,
    std::optional<at::Tensor> projectionCons_targets,
    std::optional<at::Tensor> distanceCons_origins,
    std::optional<at::Tensor> distanceCons_parents,
    std::optional<at::Tensor> distanceCons_offsets,
    std::optional<at::Tensor> distanceCons_weights,
    std::optional<at::Tensor> distanceCons_targets,
    std::optional<at::Tensor> vertexCons_vertices,
    std::optional<at::Tensor> vertexCons_weights,
    std::optional<at::Tensor> vertexCons_target_positions,
    std::optional<at::Tensor> vertexCons_target_normals,
    momentum::VertexConstraintType vertexCons_type,
    std::optional<at::Tensor> vertexProjCons_vertices,
    std::optional<at::Tensor> vertexProjCons_weights,
    std::optional<at::Tensor> vertexProjCons_target_positions,
    std::optional<at::Tensor> vertexProjCons_projections) {
  TensorMppcaModel mppcaModel = extractMppcaModel(posePrior_model);
  return applyIKProblemAutogradFunction<IKSolveFunction>(
      characters.ptr(),
      activeParams,
      at::empty({0}),
      options,
      modelParameters_init,
      activeErrorFunctions,
      errorFunctionWeights,
      // Convert empty optional<at::Tensor> into empty tensor:
      denullify(positionCons_parents),
      denullify(positionCons_offsets),
      denullify(positionCons_weights),
      denullify(positionCons_targets),
      denullify(orientation_parents),
      denullify(orientation_offsets),
      denullify(orientation_weights),
      denullify(orientation_targets),
      mppcaModel.pi,
      mppcaModel.mu,
      mppcaModel.W,
      mppcaModel.sigma,
      mppcaModel.parameterIndices,
      mppcaModel.mppca.is_none() ? nullptr : mppcaModel.mppca.ptr(),
      denullify(motion_targets),
      denullify(motion_weights),
      denullify(std::move(projectionCons_projections)),
      denullify(projectionCons_parents),
      denullify(projectionCons_offsets),
      denullify(projectionCons_weights),
      denullify(projectionCons_targets),
      denullify(std::move(distanceCons_origins)),
      denullify(distanceCons_parents),
      denullify(distanceCons_offsets),
      denullify(distanceCons_weights),
      denullify(distanceCons_targets),
      denullify(vertexCons_vertices),
      denullify(vertexCons_weights),
      denullify(vertexCons_target_positions),
      denullify(vertexCons_target_normals),
      vertexCons_type,
      denullify(vertexProjCons_vertices),
      denullify(vertexProjCons_weights),
      denullify(vertexProjCons_target_positions),
      denullify(vertexProjCons_projections))[0];
}

torch::Tensor computeGradient(
    py::object characters,
    at::Tensor modelParameters,
    const std::vector<ErrorFunctionType>& activeErrorFunctions,
    at::Tensor errorFunctionWeights,
    std::optional<at::Tensor> positionCons_parents,
    std::optional<at::Tensor> positionCons_offsets,
    std::optional<at::Tensor> positionCons_weights,
    std::optional<at::Tensor> positionCons_targets,
    std::optional<at::Tensor> orientation_parents,
    std::optional<at::Tensor> orientation_offsets,
    std::optional<at::Tensor> orientation_weights,
    std::optional<at::Tensor> orientation_targets,
    pybind11::object posePrior_model,
    std::optional<at::Tensor> motion_targets,
    std::optional<at::Tensor> motion_weights,
    std::optional<at::Tensor> projectionCons_projections,
    std::optional<at::Tensor> projectionCons_parents,
    std::optional<at::Tensor> projectionCons_offsets,
    std::optional<at::Tensor> projectionCons_weights,
    std::optional<at::Tensor> projectionCons_targets,
    std::optional<at::Tensor> distanceCons_origins,
    std::optional<at::Tensor> distanceCons_parents,
    std::optional<at::Tensor> distanceCons_offsets,
    std::optional<at::Tensor> distanceCons_weights,
    std::optional<at::Tensor> distanceCons_targets,
    std::optional<at::Tensor> vertexCons_vertices,
    std::optional<at::Tensor> vertexCons_weights,
    std::optional<at::Tensor> vertexCons_target_positions,
    std::optional<at::Tensor> vertexCons_target_normals,
    momentum::VertexConstraintType vertexCons_type,
    std::optional<at::Tensor> vertexProjCons_vertices,
    std::optional<at::Tensor> vertexProjCons_weights,
    std::optional<at::Tensor> vertexProjCons_target_positions,
    std::optional<at::Tensor> vertexProjCons_projections) {
  auto activeParameters = at::empty({0});
  TensorMppcaModel mppcaModel = extractMppcaModel(posePrior_model);
  at::Tensor result = applyIKProblemAutogradFunction<GradientFunction>(
      characters.ptr(),
      activeParameters,
      at::empty({0}),
      SolverOptions(),
      modelParameters,
      activeErrorFunctions,
      errorFunctionWeights,
      // Convert empty optional<at::Tensor> into empty tensor:
      denullify(positionCons_parents),
      denullify(positionCons_offsets),
      denullify(positionCons_weights),
      denullify(positionCons_targets),
      denullify(orientation_parents),
      denullify(orientation_offsets),
      denullify(orientation_weights),
      denullify(orientation_targets),
      mppcaModel.pi,
      mppcaModel.mu,
      mppcaModel.W,
      mppcaModel.sigma,
      mppcaModel.parameterIndices,
      mppcaModel.mppca.is_none() ? nullptr : mppcaModel.mppca.ptr(),
      denullify(motion_targets),
      denullify(motion_weights),
      denullify(std::move(projectionCons_projections)),
      denullify(projectionCons_parents),
      denullify(projectionCons_offsets),
      denullify(projectionCons_weights),
      denullify(projectionCons_targets),
      denullify(std::move(distanceCons_origins)),
      denullify(distanceCons_parents),
      denullify(distanceCons_offsets),
      denullify(distanceCons_weights),
      denullify(distanceCons_targets),
      denullify(vertexCons_vertices),
      denullify(vertexCons_weights),
      denullify(vertexCons_target_positions),
      denullify(vertexCons_target_normals),
      vertexCons_type,
      denullify(vertexProjCons_vertices),
      denullify(vertexProjCons_weights),
      denullify(vertexProjCons_target_positions),
      denullify(vertexProjCons_projections))[0];
  return result;
}

torch::Tensor computeResidual(
    py::object characters,
    at::Tensor modelParameters,
    const std::vector<ErrorFunctionType>& activeErrorFunctions,
    at::Tensor errorFunctionWeights,
    std::optional<at::Tensor> positionCons_parents,
    std::optional<at::Tensor> positionCons_offsets,
    std::optional<at::Tensor> positionCons_weights,
    std::optional<at::Tensor> positionCons_targets,
    std::optional<at::Tensor> orientation_parents,
    std::optional<at::Tensor> orientation_offsets,
    std::optional<at::Tensor> orientation_weights,
    std::optional<at::Tensor> orientation_targets,
    pybind11::object posePrior_model,
    std::optional<at::Tensor> motion_targets,
    std::optional<at::Tensor> motion_weights,
    std::optional<at::Tensor> projectionCons_projections,
    std::optional<at::Tensor> projectionCons_parents,
    std::optional<at::Tensor> projectionCons_offsets,
    std::optional<at::Tensor> projectionCons_weights,
    std::optional<at::Tensor> projectionCons_targets,
    std::optional<at::Tensor> distanceCons_origins,
    std::optional<at::Tensor> distanceCons_parents,
    std::optional<at::Tensor> distanceCons_offsets,
    std::optional<at::Tensor> distanceCons_weights,
    std::optional<at::Tensor> distanceCons_targets,
    std::optional<at::Tensor> vertexCons_vertices,
    std::optional<at::Tensor> vertexCons_weights,
    std::optional<at::Tensor> vertexCons_target_positions,
    std::optional<at::Tensor> vertexCons_target_normals,
    momentum::VertexConstraintType vertexCons_type,
    std::optional<at::Tensor> vertexProjCons_vertices,
    std::optional<at::Tensor> vertexProjCons_weights,
    std::optional<at::Tensor> vertexProjCons_target_positions,
    std::optional<at::Tensor> vertexProjCons_projections) {
  auto activeParameters = at::empty({0});
  TensorMppcaModel mppcaModel = extractMppcaModel(posePrior_model);
  auto res = applyIKProblemAutogradFunction<ResidualFunction>(
      characters.ptr(),
      activeParameters,
      at::empty({0}),
      SolverOptions(),
      modelParameters,
      activeErrorFunctions,
      errorFunctionWeights,
      // Convert empty optional<at::Tensor> into empty tensor:
      denullify(positionCons_parents),
      denullify(positionCons_offsets),
      denullify(positionCons_weights),
      denullify(positionCons_targets),
      denullify(orientation_parents),
      denullify(orientation_offsets),
      denullify(orientation_weights),
      denullify(orientation_targets),
      mppcaModel.pi,
      mppcaModel.mu,
      mppcaModel.W,
      mppcaModel.sigma,
      mppcaModel.parameterIndices,
      mppcaModel.mppca.is_none() ? nullptr : mppcaModel.mppca.ptr(),
      denullify(motion_targets),
      denullify(motion_weights),
      denullify(std::move(projectionCons_projections)),
      denullify(projectionCons_parents),
      denullify(projectionCons_offsets),
      denullify(projectionCons_weights),
      denullify(projectionCons_targets),
      denullify(std::move(distanceCons_origins)),
      denullify(distanceCons_parents),
      denullify(distanceCons_offsets),
      denullify(distanceCons_weights),
      denullify(distanceCons_targets),
      denullify(vertexCons_vertices),
      denullify(vertexCons_weights),
      denullify(vertexCons_target_positions),
      denullify(vertexCons_target_normals),
      vertexCons_type,
      denullify(vertexProjCons_vertices),
      denullify(vertexProjCons_weights),
      denullify(vertexProjCons_target_positions),
      denullify(vertexProjCons_projections));
  assert(res.size() == 2);
  return res[0];
}

std::tuple<torch::Tensor, torch::Tensor> computeJacobian(
    py::object characters,
    at::Tensor modelParameters,
    const std::vector<ErrorFunctionType>& activeErrorFunctions,
    at::Tensor errorFunctionWeights,
    std::optional<at::Tensor> positionCons_parents,
    std::optional<at::Tensor> positionCons_offsets,
    std::optional<at::Tensor> positionCons_weights,
    std::optional<at::Tensor> positionCons_targets,
    std::optional<at::Tensor> orientation_parents,
    std::optional<at::Tensor> orientation_offsets,
    std::optional<at::Tensor> orientation_weights,
    std::optional<at::Tensor> orientation_targets,
    pybind11::object posePrior_model,
    std::optional<at::Tensor> motion_targets,
    std::optional<at::Tensor> motion_weights,
    std::optional<at::Tensor> projectionCons_projections,
    std::optional<at::Tensor> projectionCons_parents,
    std::optional<at::Tensor> projectionCons_offsets,
    std::optional<at::Tensor> projectionCons_weights,
    std::optional<at::Tensor> projectionCons_targets,
    std::optional<at::Tensor> distanceCons_origins,
    std::optional<at::Tensor> distanceCons_parents,
    std::optional<at::Tensor> distanceCons_offsets,
    std::optional<at::Tensor> distanceCons_weights,
    std::optional<at::Tensor> distanceCons_targets,
    std::optional<at::Tensor> vertexCons_vertices,
    std::optional<at::Tensor> vertexCons_weights,
    std::optional<at::Tensor> vertexCons_target_positions,
    std::optional<at::Tensor> vertexCons_target_normals,
    momentum::VertexConstraintType vertexCons_type,
    std::optional<at::Tensor> vertexProjCons_vertices,
    std::optional<at::Tensor> vertexProjCons_weights,
    std::optional<at::Tensor> vertexProjCons_target_positions,
    std::optional<at::Tensor> vertexProjCons_projections) {
  auto activeParameters = at::empty({0});
  TensorMppcaModel mppcaModel = extractMppcaModel(posePrior_model);
  auto res = applyIKProblemAutogradFunction<ResidualFunction>(
      characters.ptr(),
      activeParameters,
      at::empty({0}),
      SolverOptions(),
      modelParameters,
      activeErrorFunctions,
      errorFunctionWeights,
      // Convert empty optional<at::Tensor> into empty tensor:
      denullify(positionCons_parents),
      denullify(positionCons_offsets),
      denullify(positionCons_weights),
      denullify(positionCons_targets),
      denullify(orientation_parents),
      denullify(orientation_offsets),
      denullify(orientation_weights),
      denullify(orientation_targets),
      mppcaModel.pi,
      mppcaModel.mu,
      mppcaModel.W,
      mppcaModel.sigma,
      mppcaModel.parameterIndices,
      mppcaModel.mppca.is_none() ? nullptr : mppcaModel.mppca.ptr(),
      denullify(motion_targets),
      denullify(motion_weights),
      denullify(std::move(projectionCons_projections)),
      denullify(projectionCons_parents),
      denullify(projectionCons_offsets),
      denullify(projectionCons_weights),
      denullify(projectionCons_targets),
      denullify(std::move(distanceCons_origins)),
      denullify(distanceCons_parents),
      denullify(distanceCons_offsets),
      denullify(distanceCons_weights),
      denullify(distanceCons_targets),
      denullify(vertexCons_vertices),
      denullify(vertexCons_weights),
      denullify(vertexCons_target_positions),
      denullify(vertexCons_target_normals),
      vertexCons_type,
      denullify(vertexProjCons_vertices),
      denullify(vertexProjCons_weights),
      denullify(vertexProjCons_target_positions),
      denullify(vertexProjCons_projections));
  assert(res.size() == 2);
  return {res[0], res[1]};
}

at::Tensor solveBodySequenceIKProblem(
    pybind11::object characters,
    at::Tensor activeParameters,
    at::Tensor sharedParameters,
    at::Tensor modelParameters_init,
    const std::vector<ErrorFunctionType>& activeErrorFunctions,
    at::Tensor errorFunctionWeights,
    SolverOptions options,
    std::optional<at::Tensor> positionCons_parents,
    std::optional<at::Tensor> positionCons_offsets,
    std::optional<at::Tensor> positionCons_weights,
    std::optional<at::Tensor> positionCons_targets,
    std::optional<at::Tensor> orientation_parents,
    std::optional<at::Tensor> orientation_offsets,
    std::optional<at::Tensor> orientation_weights,
    std::optional<at::Tensor> orientation_targets,
    pybind11::object posePrior_model,
    std::optional<at::Tensor> motion_targets,
    std::optional<at::Tensor> motion_weights,
    std::optional<at::Tensor> projectionCons_projections,
    std::optional<at::Tensor> projectionCons_parents,
    std::optional<at::Tensor> projectionCons_offsets,
    std::optional<at::Tensor> projectionCons_weights,
    std::optional<at::Tensor> projectionCons_targets,
    std::optional<at::Tensor> distanceCons_origins,
    std::optional<at::Tensor> distanceCons_parents,
    std::optional<at::Tensor> distanceCons_offsets,
    std::optional<at::Tensor> distanceCons_weights,
    std::optional<at::Tensor> distanceCons_targets,
    std::optional<at::Tensor> vertexCons_vertices,
    std::optional<at::Tensor> vertexCons_weights,
    std::optional<at::Tensor> vertexCons_target_positions,
    std::optional<at::Tensor> vertexCons_target_normals,
    momentum::VertexConstraintType vertexCons_type,
    std::optional<at::Tensor> vertexProjCons_vertices,
    std::optional<at::Tensor> vertexProjCons_weights,
    std::optional<at::Tensor> vertexProjCons_target_positions,
    std::optional<at::Tensor> vertexProjCons_projections) {
  TensorMppcaModel mppcaModel = extractMppcaModel(posePrior_model);
  return applyIKProblemAutogradFunction<SequenceIKSolveFunction>(
      characters.ptr(),
      activeParameters,
      sharedParameters,
      options,
      modelParameters_init,
      activeErrorFunctions,
      errorFunctionWeights,
      // Convert empty optional<at::Tensor> into empty tensor:
      denullify(positionCons_parents),
      denullify(positionCons_offsets),
      denullify(positionCons_weights),
      denullify(positionCons_targets),
      denullify(orientation_parents),
      denullify(orientation_offsets),
      denullify(orientation_weights),
      denullify(orientation_targets),
      mppcaModel.pi,
      mppcaModel.mu,
      mppcaModel.W,
      mppcaModel.sigma,
      mppcaModel.parameterIndices,
      mppcaModel.mppca.is_none() ? nullptr : mppcaModel.mppca.ptr(),
      denullify(motion_targets),
      denullify(motion_weights),
      denullify(std::move(projectionCons_projections)),
      denullify(projectionCons_parents),
      denullify(projectionCons_offsets),
      denullify(projectionCons_weights),
      denullify(projectionCons_targets),
      denullify(std::move(distanceCons_origins)),
      denullify(distanceCons_parents),
      denullify(distanceCons_offsets),
      denullify(distanceCons_weights),
      denullify(distanceCons_targets),
      denullify(vertexCons_vertices),
      denullify(vertexCons_weights),
      denullify(vertexCons_target_positions),
      denullify(vertexCons_target_normals),
      vertexCons_type,
      denullify(vertexProjCons_vertices),
      denullify(vertexProjCons_weights),
      denullify(vertexProjCons_target_positions),
      denullify(vertexProjCons_projections))[0];
}

template <typename T>
at::Tensor transformPoseImp(
    const momentum::Character& character,
    at::Tensor modelParams,
    at::Tensor transforms,
    bool ensureContinuousOutput) {
  TensorChecker checker("ParameterTransform.apply");
  bool squeeze = false;

  const int nModelParam = character.parameterTransform.numAllModelParameters();

  modelParams = checker.validateAndFixTensor(
      modelParams,
      "modelParameters",
      {nModelParam},
      {"nModelParams"},
      toScalarType<T>(),
      true,
      false,
      &squeeze);

  transforms = checker.validateAndFixTensor(
      transforms, "transforms", {8}, {"trans_rot"}, toScalarType<T>(), true, false, &squeeze);

  const int nBatch = checker.getBatchSize();

  std::vector<momentum::ModelParametersT<T>> modelParamsVec(nBatch);
  std::vector<momentum::TransformT<T>> transformsVec(nBatch);

  bool hasScale = false;
  for (size_t iBatch = 0; iBatch < nBatch; ++iBatch) {
    modelParamsVec[iBatch] = toEigenMap<T>(modelParams.select(0, iBatch));
    auto skelStateVec = toEigenMap<T>(transforms.select(0, iBatch));
    // momentum order is rx, ry, rz, rw
    // Eigen order is w, x, y, z
    transformsVec[iBatch] = momentum::TransformT<T>(
        skelStateVec.template head<3>(),
        Eigen::Quaternion<T>(skelStateVec[6], skelStateVec[3], skelStateVec[4], skelStateVec[5])
            .normalized());
    if (std::abs(skelStateVec[7] - 1.0) > 1e-3) {
      hasScale = true;
    }
  }

  if (hasScale) {
    py::print(
        "Warning: scaling detected in transform passed to transform pose but will have no effect.");
  }

  const std::vector<momentum::ModelParametersT<T>> resultVec =
      momentum::transformPose(character, modelParamsVec, transformsVec, ensureContinuousOutput);

  at::Tensor result = at::zeros({nBatch, (int)nModelParam}, toScalarType<T>());
  for (size_t iBatch = 0; iBatch < nBatch; ++iBatch) {
    toEigenMap<T>(result.select(0, iBatch)) = resultVec.at(iBatch).v;
  }

  if (squeeze) {
    result = result.squeeze(0);
  }
  return result;
}

at::Tensor transformPose(
    const momentum::Character& character,
    at::Tensor modelParams,
    at::Tensor transforms,
    bool ensureContinuousOutput) {
  if (hasFloat64(modelParams, transforms)) {
    return transformPoseImp<double>(character, modelParams, transforms, ensureContinuousOutput);
  } else {
    return transformPoseImp<float>(character, modelParams, transforms, ensureContinuousOutput);
  }
}

} // namespace pymomentum
