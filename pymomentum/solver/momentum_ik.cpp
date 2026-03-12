/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/solver/momentum_ik.h"

#include "pymomentum/solver/momentum_ik_autograd.h"

#include "momentum/character/types.h"
#include "momentum/character_solver/transform_pose.h"
#include "momentum/common/log.h"
#include "momentum/math/types.h"
#include "pymomentum/python_utility/python_utility.h"
#include "pymomentum/tensor_momentum/tensor_momentum_utility.h"
#include "pymomentum/tensor_momentum/tensor_parameter_transform.h"
#include "pymomentum/tensor_utility/autograd_utility.h"
#include "pymomentum/tensor_utility/tensor_utility.h"

namespace py = pybind11;

namespace pymomentum {

namespace {

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
    MT_LOGW("Scaling detected in transform passed to transform pose but will have no effect.");
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
