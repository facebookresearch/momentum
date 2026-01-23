/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/geometry/array_parameter_transform.h"

#include <pymomentum/array_utility/array_utility.h>
#include <pymomentum/array_utility/batch_accessor.h>
#include <pymomentum/array_utility/geometry_accessors.h>

#include <momentum/character/inverse_parameter_transform.h>
#include <momentum/character/joint.h>
#include <momentum/character/parameter_transform.h>

#include <dispenso/parallel_for.h>
#include <fmt/format.h>

namespace pymomentum {

namespace {

template <typename T>
py::array_t<T> applyParameterTransformImpl(
    const momentum::ParameterTransform& paramTransform,
    const py::array& modelParams,
    const LeadingDimensions& leadingDims,
    bool flatten) {
  const auto nModelParams = static_cast<py::ssize_t>(paramTransform.numAllModelParameters());
  const auto nJointParams = static_cast<py::ssize_t>(paramTransform.numJointParameters());
  const auto nJoints = nJointParams / 7;
  const auto nBatch = leadingDims.totalBatchElements();

  // Create output array with shape [..., nJointParams] (flat) or [..., nJoints, 7] (structured)
  py::array_t<T> result;
  JointParamsShape outputShape;
  if (flatten) {
    result = createOutputArray<T>(leadingDims, {nJointParams});
    outputShape = JointParamsShape::Flat;
  } else {
    result = createOutputArray<T>(leadingDims, {nJoints, static_cast<py::ssize_t>(7)});
    outputShape = JointParamsShape::Structured;
  }

  // Create batch indexer for converting flat indices to multi-dimensional indices
  BatchIndexer indexer(leadingDims);

  // Create accessors using geometry accessors
  ModelParametersAccessor<T> inputAcc(modelParams, leadingDims, nModelParams);
  JointParametersAccessor<T> outputAcc(result, leadingDims, nJoints, outputShape);

  // Get typed parameter transform
  auto ptT = paramTransform.cast<T>();

  // Release GIL for parallel computation
  {
    py::gil_scoped_release release;
    dispenso::parallel_for(0, static_cast<int64_t>(nBatch), [&](int64_t iBatch) {
      // Convert flat batch index to multi-dimensional indices
      auto indices = indexer.decompose(iBatch);

      // Get model parameters as ModelParametersT<T>
      auto modelParamsT = inputAcc.get(indices);

      // Apply parameter transform - returns JointParametersT<T>
      auto jointParamsT = ptT.apply(modelParamsT);

      // Write joint parameters directly
      outputAcc.set(indices, jointParamsT);
    });
  }

  return result;
}

// InverseParameterTransform is always float-based, so we only support float
py::array_t<float> applyInverseParameterTransformImpl(
    const momentum::InverseParameterTransform& invParamTransform,
    const py::array& jointParams,
    const LeadingDimensions& leadingDims,
    JointParamsShape shape) {
  const auto nJointParams = static_cast<py::ssize_t>(invParamTransform.numJointParameters());
  const auto nJoints = nJointParams / 7;
  const auto nModelParams = static_cast<py::ssize_t>(invParamTransform.numAllModelParameters());
  const auto nBatch = leadingDims.totalBatchElements();

  // Create output array
  auto result = createOutputArray<float>(leadingDims, {nModelParams});

  // Create batch indexer for converting flat indices to multi-dimensional indices
  BatchIndexer indexer(leadingDims);

  // Use geometry accessors for both input and output
  JointParametersAccessor<float> inputAcc(jointParams, leadingDims, nJoints, shape);
  ModelParametersAccessor<float> outputAcc(result, leadingDims, nModelParams);

  // Release GIL for parallel computation
  {
    py::gil_scoped_release release;
    // InverseParameterTransform.apply() takes JointParametersT and returns CharacterParameters
    // Eigen SparseQR is triggering a data race for some reason even though the solve() function is
    // declared const, so do this single-threaded.
    for (py::ssize_t iBatch = 0; iBatch < nBatch; ++iBatch) {
      // Convert flat batch index to multi-dimensional indices
      auto indices = indexer.decompose(iBatch);

      // Get joint parameters as JointParametersT<float>
      auto jointParamsT = inputAcc.get(indices);

      // Apply inverse transform
      auto charParams = invParamTransform.apply(jointParamsT);

      // Write model parameters directly
      outputAcc.set(indices, charParams.pose);
    }
  }

  return result;
}

} // namespace

py::array applyParameterTransformArray(
    const momentum::ParameterTransform& paramTransform,
    const py::buffer& modelParams,
    bool flatten) {
  const auto nModelParams = static_cast<int>(paramTransform.numAllModelParameters());

  ArrayChecker checker("ParameterTransform.apply");
  checker.validateBuffer(modelParams, "model_parameters", {nModelParams}, {"numModelParams"});

  if (checker.isFloat64()) {
    return applyParameterTransformImpl<double>(
        paramTransform, modelParams, checker.getLeadingDimensions(), flatten);
  } else {
    return applyParameterTransformImpl<float>(
        paramTransform, modelParams, checker.getLeadingDimensions(), flatten);
  }
}

py::array applyInverseParameterTransformArray(
    const momentum::InverseParameterTransform& invParamTransform,
    const py::buffer& jointParams) {
  const auto nJointParams = static_cast<int>(invParamTransform.numJointParameters());
  const auto nJoints = nJointParams / 7;

  ArrayChecker checker("InverseParameterTransform.apply");
  // Accept both structured and flat formats
  JointParamsShape shape;
  py::buffer_info bufInfo = jointParams.request();
  if (bufInfo.ndim >= 2 && bufInfo.shape[bufInfo.ndim - 1] == 7 &&
      bufInfo.shape[bufInfo.ndim - 2] == nJoints) {
    // Structured format: (..., nJoints, 7)
    checker.validateBuffer(jointParams, "joint_parameters", {nJoints, 7}, {"numJoints", "7"});
    shape = JointParamsShape::Structured;
  } else {
    // Flat format: (..., nJointParams)
    checker.validateBuffer(jointParams, "joint_parameters", {nJointParams}, {"numJointParams"});
    shape = JointParamsShape::Flat;
  }

  return applyInverseParameterTransformImpl(
      invParamTransform, jointParams, checker.getLeadingDimensions(), shape);
}

py::array_t<bool> parameterSetToArray(
    const momentum::ParameterTransform& parameterTransform,
    const momentum::ParameterSet& paramSet) {
  const auto nParams = parameterTransform.numAllModelParameters();
  py::array_t<bool> result(nParams);
  auto accessor = result.mutable_unchecked<1>();
  for (size_t i = 0; i < nParams; ++i) {
    accessor(i) = paramSet.test(i);
  }
  return result;
}

std::unordered_map<std::string, py::array_t<bool>> getParameterSetsArray(
    const momentum::ParameterTransform& parameterTransform) {
  std::unordered_map<std::string, py::array_t<bool>> result;
  for (const auto& [name, paramSet] : parameterTransform.parameterSets) {
    result[name] = parameterSetToArray(parameterTransform, paramSet);
  }
  return result;
}

py::array_t<bool> getScalingParametersArray(
    const momentum::ParameterTransform& parameterTransform) {
  return parameterSetToArray(parameterTransform, parameterTransform.getScalingParameters());
}

py::array_t<bool> getRigidParametersArray(const momentum::ParameterTransform& parameterTransform) {
  return parameterSetToArray(parameterTransform, parameterTransform.getRigidParameters());
}

py::array_t<bool> getAllParametersArray(const momentum::ParameterTransform& parameterTransform) {
  const auto nParams = parameterTransform.numAllModelParameters();
  py::array_t<bool> result(nParams);
  auto accessor = result.mutable_unchecked<1>();
  for (size_t i = 0; i < nParams; ++i) {
    accessor(i) = true;
  }
  return result;
}

py::array_t<bool> getBlendShapeParametersArray(
    const momentum::ParameterTransform& parameterTransform) {
  return parameterSetToArray(parameterTransform, parameterTransform.getBlendShapeParameters());
}

py::array_t<bool> getFaceExpressionParametersArray(
    const momentum::ParameterTransform& parameterTransform) {
  return parameterSetToArray(parameterTransform, parameterTransform.getFaceExpressionParameters());
}

py::array_t<bool> getPoseParametersArray(const momentum::ParameterTransform& parameterTransform) {
  return parameterSetToArray(parameterTransform, parameterTransform.getPoseParameters());
}

void addParameterSetArray(
    momentum::ParameterTransform& parameterTransform,
    const std::string& paramSetName,
    const py::array_t<bool>& paramSet) {
  const auto nParams = parameterTransform.numAllModelParameters();

  MT_THROW_IF(
      paramSet.ndim() != 1,
      "add_parameter_set: parameter_set must be a 1D array, got {}D",
      paramSet.ndim());

  MT_THROW_IF(
      static_cast<size_t>(paramSet.shape(0)) != nParams,
      "add_parameter_set: parameter_set has {} elements but expected {}",
      paramSet.shape(0),
      nParams);

  momentum::ParameterSet ps;
  auto accessor = paramSet.unchecked<1>();
  for (size_t i = 0; i < nParams; ++i) {
    ps.set(i, accessor(i));
  }
  parameterTransform.parameterSets[paramSetName] = ps;
}

py::array_t<bool> getParametersForJointsArray(
    const momentum::ParameterTransform& parameterTransform,
    const std::vector<size_t>& jointIndices) {
  const auto nJoints = parameterTransform.numJointParameters() / momentum::kParametersPerJoint;
  std::vector<bool> activeJoints(nJoints);
  for (const auto& idx : jointIndices) {
    MT_THROW_IF(idx >= nJoints, "getParametersForJoints: joint index {} out of bounds.", idx);
    activeJoints[idx] = true;
  }

  momentum::ParameterSet result;

  // Iterate over all non-zero entries of the sparse row-major matrix
  for (int k = 0; k < parameterTransform.transform.outerSize(); ++k) {
    for (momentum::SparseRowMatrixf::InnerIterator it(parameterTransform.transform, k); it; ++it) {
      const auto globalParam = it.row();
      const auto kParam = it.col();
      assert(kParam < parameterTransform.numAllModelParameters());
      const auto jointIndex = globalParam / momentum::kParametersPerJoint;
      assert(jointIndex < static_cast<Eigen::Index>(nJoints));

      if (activeJoints[jointIndex]) {
        result.set(kParam, true);
      }
    }
  }

  return parameterSetToArray(parameterTransform, result);
}

py::array_t<bool> findParametersArray(
    const momentum::ParameterTransform& parameterTransform,
    const std::vector<std::string>& parameterNames,
    bool allowMissing) {
  const auto nParams = parameterTransform.numAllModelParameters();
  py::array_t<bool> result(nParams);
  auto accessor = result.mutable_unchecked<1>();

  // Initialize to false
  for (size_t i = 0; i < nParams; ++i) {
    accessor(i) = false;
  }

  // Find each parameter
  for (const auto& name : parameterNames) {
    auto it = std::find(parameterTransform.name.begin(), parameterTransform.name.end(), name);
    if (it != parameterTransform.name.end()) {
      accessor(std::distance(parameterTransform.name.begin(), it)) = true;
    } else if (!allowMissing) {
      MT_THROW("find_parameters: parameter '{}' not found", name);
    }
  }

  return result;
}

py::array_t<float> getParameterTransformMatrix(
    const momentum::ParameterTransform& parameterTransform) {
  const auto nJointParams = parameterTransform.numJointParameters();
  const auto nModelParams = parameterTransform.numAllModelParameters();

  py::array_t<float> result(
      {static_cast<py::ssize_t>(nJointParams), static_cast<py::ssize_t>(nModelParams)});
  auto accessor = result.mutable_unchecked<2>();

  // Initialize to zero
  for (size_t i = 0; i < nJointParams; ++i) {
    for (size_t j = 0; j < nModelParams; ++j) {
      accessor(i, j) = 0.0f;
    }
  }

  // Copy sparse matrix values using the correct row-major iterator
  for (int k = 0; k < parameterTransform.transform.outerSize(); ++k) {
    for (momentum::SparseRowMatrixf::InnerIterator it(parameterTransform.transform, k); it; ++it) {
      accessor(it.row(), it.col()) = it.value();
    }
  }

  return result;
}

} // namespace pymomentum
