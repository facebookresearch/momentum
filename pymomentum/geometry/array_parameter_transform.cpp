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

#include <momentum/character/blend_shape_skinning.h>
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
  // Create a dummy character to use validateJointParameters
  // We only need the skeleton for joint count validation
  const auto nJointParams = static_cast<int>(invParamTransform.numJointParameters());
  const auto nJoints = nJointParams / 7;

  // Create a temporary skeleton with the right number of joints
  momentum::Skeleton tempSkeleton;
  tempSkeleton.joints.resize(nJoints);
  momentum::Character tempCharacter;
  tempCharacter.skeleton = tempSkeleton;

  ArrayChecker checker("InverseParameterTransform.apply");
  JointParamsShape shape =
      checker.validateJointParameters(jointParams, "joint_parameters", tempCharacter);

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

momentum::ParameterSet arrayToParameterSet(
    const momentum::ParameterTransform& parameterTransform,
    const py::array_t<bool>& paramSet,
    DefaultParameterSet defaultParamSet) {
  // Handle empty array with defaults
  if (paramSet.size() == 0) {
    switch (defaultParamSet) {
      case DefaultParameterSet::AllZeros: {
        momentum::ParameterSet result;
        return result;
      }
      case DefaultParameterSet::AllOnes: {
        momentum::ParameterSet result;
        result.set();
        return result;
      }
      case DefaultParameterSet::NoDefault:
      default:
        // Fall through to validation below
        break;
    }
  }

  const auto nParams = parameterTransform.numAllModelParameters();

  MT_THROW_IF(
      paramSet.size() == 0 || paramSet.ndim() != 1 ||
          paramSet.shape(0) != static_cast<py::ssize_t>(nParams),
      "Mismatch between active parameters size ({}) and parameter transform size ({})",
      paramSet.size() > 0 ? paramSet.shape(0) : 0,
      nParams);

  momentum::ParameterSet result;
  auto accessor = paramSet.unchecked<1>();
  for (size_t i = 0; i < nParams; ++i) {
    if (accessor(i)) {
      result.set(i);
    }
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

py::array uniformRandomToModelParametersArray(
    const momentum::Character& character,
    py::buffer unifNoise) {
  const auto& paramTransform = character.parameterTransform;
  const auto nModelParams = static_cast<int>(paramTransform.numAllModelParameters());

  ArrayChecker checker("uniform_random_to_model_parameters");
  checker.validateBuffer(unifNoise, "unif_noise", {nModelParams}, {"numModelParams"});

  const auto& leadingDims = checker.getLeadingDimensions();
  const auto nBatch = leadingDims.totalBatchElements();

  // Create output array with same shape as input
  auto result = createOutputArray<float>(leadingDims, {static_cast<py::ssize_t>(nModelParams)});

  // Create batch indexer
  BatchIndexer indexer(leadingDims);

  // Create accessors
  ModelParametersAccessor<float> inputAcc(unifNoise, leadingDims, nModelParams);
  ModelParametersAccessor<float> outputAcc(result, leadingDims, nModelParams);

  // Release GIL for parallel computation
  {
    py::gil_scoped_release release;
    dispenso::parallel_for(0, static_cast<int64_t>(nBatch), [&](int64_t iBatch) {
      // Convert flat batch index to multi-dimensional indices
      auto indices = indexer.decompose(iBatch);

      // Get input model parameters
      auto unifParams = inputAcc.get(indices);

      // Create output model parameters
      momentum::ModelParametersT<float> outParams(nModelParams);

      // Process each parameter
      for (int iParam = 0; iParam < nModelParams; ++iParam) {
        const float unifValue = unifParams[iParam];
        const auto& name = paramTransform.name[iParam];

        float result_val = NAN;
        if (name.find("scale_") != std::string::npos) {
          result_val = 1.0f * (unifValue - 0.5f);
        } else if (
            name.find("_tx") != std::string::npos || name.find("_ty") != std::string::npos ||
            name.find("_tz") != std::string::npos) {
          result_val = 5.0f * (unifValue - 0.5f);
        } else {
          // Assume it's a rotation parameter
          result_val = (momentum::pi<float>() / 4.0f) * (unifValue - 0.5f);
        }
        outParams[iParam] = result_val;
      }

      // Write output
      outputAcc.set(indices, outParams);
    });
  }

  return result;
}

namespace {

// Helper template for extracting coefficients from model parameters
// using a vector of parameter indices (blendShapeParameters or faceExpressionParameters)
template <typename T>
py::array_t<T> extractCoefficientsImpl(
    const Eigen::VectorXi& parameterIndices,
    const py::array& modelParams,
    const LeadingDimensions& leadingDims,
    py::ssize_t nModelParams) {
  const auto nCoeffs = static_cast<py::ssize_t>(parameterIndices.size());
  const auto nBatch = leadingDims.totalBatchElements();

  // Create output array
  auto result = createOutputArray<T>(leadingDims, {nCoeffs});

  // Create batch indexer
  BatchIndexer indexer(leadingDims);

  // Create accessors
  ModelParametersAccessor<T> inputAcc(modelParams, leadingDims, nModelParams);
  BlendWeightsAccessor<T> outputAcc(result, leadingDims, nCoeffs);

  // Release GIL for parallel computation
  {
    py::gil_scoped_release release;
    dispenso::parallel_for(0, static_cast<int64_t>(nBatch), [&](int64_t iBatch) {
      // Convert flat batch index to multi-dimensional indices
      auto indices = indexer.decompose(iBatch);

      // Get input model parameters
      auto modelParamsT = inputAcc.get(indices);

      // Extract coefficients
      momentum::BlendWeightsT<T> coeffs = Eigen::VectorX<T>::Zero(nCoeffs);
      for (Eigen::Index iCoeff = 0; iCoeff < nCoeffs; ++iCoeff) {
        const auto paramIdx = parameterIndices[iCoeff];
        if (paramIdx >= 0 && paramIdx < modelParamsT.size()) {
          coeffs(iCoeff) = modelParamsT(paramIdx);
        }
      }

      // Write output
      outputAcc.set(indices, coeffs);
    });
  }

  return result;
}

} // namespace

py::array modelParametersToBlendShapeCoefficientsArray(
    const momentum::Character& character,
    const py::buffer& modelParameters) {
  const auto& paramTransform = character.parameterTransform;
  const auto nModelParams = static_cast<int>(paramTransform.numAllModelParameters());

  ArrayChecker checker("model_parameters_to_blend_shape_coefficients");
  checker.validateBuffer(modelParameters, "model_parameters", {nModelParams}, {"numModelParams"});

  const auto& leadingDims = checker.getLeadingDimensions();

  if (checker.isFloat64()) {
    return extractCoefficientsImpl<double>(
        paramTransform.blendShapeParameters, modelParameters, leadingDims, nModelParams);
  } else {
    return extractCoefficientsImpl<float>(
        paramTransform.blendShapeParameters, modelParameters, leadingDims, nModelParams);
  }
}

py::array modelParametersToFaceExpressionCoefficientsArray(
    const momentum::Character& character,
    const py::buffer& modelParameters) {
  const auto& paramTransform = character.parameterTransform;
  const auto nModelParams = static_cast<int>(paramTransform.numAllModelParameters());

  ArrayChecker checker("model_parameters_to_face_expression_coefficients");
  checker.validateBuffer(modelParameters, "model_parameters", {nModelParams}, {"numModelParams"});

  const auto& leadingDims = checker.getLeadingDimensions();

  if (checker.isFloat64()) {
    return extractCoefficientsImpl<double>(
        paramTransform.faceExpressionParameters, modelParameters, leadingDims, nModelParams);
  } else {
    return extractCoefficientsImpl<float>(
        paramTransform.faceExpressionParameters, modelParameters, leadingDims, nModelParams);
  }
}

namespace {

// Helper to build a mapping from source names to target names.
// Returns two vectors: srcIndices and tgtIndices, where
// srcIndices[i] maps to tgtIndices[i].
std::pair<std::vector<py::ssize_t>, std::vector<py::ssize_t>> buildNameMapping(
    const std::vector<std::string>& srcNames,
    const std::vector<std::string>& tgtNames) {
  // Build a map from target names to indices for O(1) lookup
  std::unordered_map<std::string, size_t> tgtNameMap;
  for (size_t iTgt = 0; iTgt < tgtNames.size(); ++iTgt) {
    tgtNameMap.emplace(tgtNames[iTgt], iTgt);
  }

  const size_t expectedSize = std::min(srcNames.size(), tgtNames.size());

  std::vector<py::ssize_t> srcIndices;
  std::vector<py::ssize_t> tgtIndices;
  srcIndices.reserve(expectedSize);
  tgtIndices.reserve(expectedSize);

  for (size_t iSrc = 0; iSrc < srcNames.size(); ++iSrc) {
    auto itr = tgtNameMap.find(srcNames[iSrc]);
    if (itr != tgtNameMap.end()) {
      srcIndices.push_back(static_cast<py::ssize_t>(iSrc));
      tgtIndices.push_back(static_cast<py::ssize_t>(itr->second));
    }
  }

  return {srcIndices, tgtIndices};
}

template <typename T>
py::array_t<T> mapModelParametersImpl(
    const py::array& motionData,
    const std::vector<std::string>& srcNames,
    const std::vector<std::string>& tgtNames,
    const LeadingDimensions& leadingDims) {
  const auto nSrcParams = static_cast<py::ssize_t>(srcNames.size());
  const auto nTgtParams = static_cast<py::ssize_t>(tgtNames.size());
  const auto nBatch = leadingDims.totalBatchElements();

  // Build the mapping between source and target parameter indices
  const auto indexMapping = buildNameMapping(srcNames, tgtNames);
  const auto& srcIndices = indexMapping.first;
  const auto& tgtIndices = indexMapping.second;

  // Create output array
  auto result = createOutputArray<T>(leadingDims, {nTgtParams});

  // Create batch indexer
  BatchIndexer indexer(leadingDims);

  // Create accessors
  ModelParametersAccessor<T> inputAcc(motionData, leadingDims, nSrcParams);
  ModelParametersAccessor<T> outputAcc(result, leadingDims, nTgtParams);

  // Release GIL for parallel computation
  {
    py::gil_scoped_release release;
    dispenso::parallel_for(0, static_cast<int64_t>(nBatch), [&](int64_t iBatch) {
      // Convert flat batch index to multi-dimensional indices
      auto indices = indexer.decompose(iBatch);

      // Get source model parameters
      auto srcParams = inputAcc.get(indices);

      // Create output model parameters (initialized to zero)
      momentum::ModelParametersT<T> tgtParams = Eigen::VectorX<T>::Zero(nTgtParams);

      // Copy matching parameters
      for (size_t i = 0; i < srcIndices.size(); ++i) {
        tgtParams[tgtIndices[i]] = srcParams[srcIndices[i]];
      }

      // Write output
      outputAcc.set(indices, tgtParams);
    });
  }

  return result;
}

template <typename T>
py::array_t<T> mapJointParametersImpl(
    const py::array& motionData,
    const std::vector<std::string>& srcJointNames,
    const std::vector<std::string>& tgtJointNames,
    const LeadingDimensions& leadingDims,
    py::ssize_t nSrcJoints,
    py::ssize_t nTgtJoints,
    JointParamsShape inputShape) {
  const auto nBatch = leadingDims.totalBatchElements();

  // Build the mapping between source and target joint indices
  const auto jointIndexMapping = buildNameMapping(srcJointNames, tgtJointNames);
  const auto& srcJointIndices = jointIndexMapping.first;
  const auto& tgtJointIndices = jointIndexMapping.second;

  // Create output array with same format as input
  py::array_t<T> result;
  if (inputShape == JointParamsShape::Structured) {
    result = createOutputArray<T>(leadingDims, {nTgtJoints, static_cast<py::ssize_t>(7)});
  } else {
    result = createOutputArray<T>(leadingDims, {nTgtJoints * 7});
  }

  // Create batch indexer
  BatchIndexer indexer(leadingDims);

  // Create accessors
  JointParametersAccessor<T> inputAcc(motionData, leadingDims, nSrcJoints, inputShape);
  JointParametersAccessor<T> outputAcc(result, leadingDims, nTgtJoints, inputShape);

  // Release GIL for parallel computation
  {
    py::gil_scoped_release release;
    dispenso::parallel_for(0, static_cast<int64_t>(nBatch), [&](int64_t iBatch) {
      // Convert flat batch index to multi-dimensional indices
      auto indices = indexer.decompose(iBatch);

      // Get source joint parameters
      auto srcParams = inputAcc.get(indices);

      // Create output joint parameters (initialized to zero)
      momentum::JointParametersT<T> tgtParams =
          Eigen::VectorX<T>::Zero(nTgtJoints * momentum::kParametersPerJoint);

      // Copy matching joints (each joint has 7 parameters)
      for (size_t i = 0; i < srcJointIndices.size(); ++i) {
        const auto srcJoint = srcJointIndices[i];
        const auto tgtJoint = tgtJointIndices[i];
        for (int p = 0; p < momentum::kParametersPerJoint; ++p) {
          tgtParams[tgtJoint * momentum::kParametersPerJoint + p] =
              srcParams[srcJoint * momentum::kParametersPerJoint + p];
        }
      }

      // Write output
      outputAcc.set(indices, tgtParams);
    });
  }

  return result;
}

} // namespace

py::array mapModelParametersNamesArray(
    py::buffer motionData,
    const std::vector<std::string>& sourceParameterNames,
    const momentum::Character& targetCharacter,
    bool verbose) {
  const auto nSrcParams = static_cast<int>(sourceParameterNames.size());
  const auto& tgtNames = targetCharacter.parameterTransform.name;

  ArrayChecker checker("map_model_parameters");
  checker.validateBuffer(motionData, "motion_data", {nSrcParams}, {"numSourceParams"});

  // Check for missing parameters and warn if verbose
  if (verbose) {
    std::vector<std::string> missingParams;
    for (const auto& srcName : sourceParameterNames) {
      auto iParam = targetCharacter.parameterTransform.getParameterIdByName(srcName);
      if (iParam == momentum::kInvalidIndex) {
        missingParams.push_back(srcName);
      }
    }
    if (!missingParams.empty()) {
      // Convert to Python list for printing
      py::list pyMissingParams;
      for (const auto& name : missingParams) {
        pyMissingParams.append(name);
      }
      py::print("WARNING: missing parameters found during map_model_parameters: ", pyMissingParams);
    }
  }

  const auto& leadingDims = checker.getLeadingDimensions();

  if (checker.isFloat64()) {
    return mapModelParametersImpl<double>(motionData, sourceParameterNames, tgtNames, leadingDims);
  } else {
    return mapModelParametersImpl<float>(motionData, sourceParameterNames, tgtNames, leadingDims);
  }
}

py::array mapModelParametersArray(
    py::buffer motionData,
    const momentum::Character& srcCharacter,
    const momentum::Character& tgtCharacter,
    bool verbose) {
  return mapModelParametersNamesArray(
      motionData, srcCharacter.parameterTransform.name, tgtCharacter, verbose);
}

py::array mapJointParametersArray(
    py::buffer motionData,
    const momentum::Character& srcCharacter,
    const momentum::Character& tgtCharacter) {
  const auto& srcSkeleton = srcCharacter.skeleton;
  const auto& tgtSkeleton = tgtCharacter.skeleton;
  const auto nSrcJoints = static_cast<py::ssize_t>(srcSkeleton.joints.size());
  const auto nTgtJoints = static_cast<py::ssize_t>(tgtSkeleton.joints.size());

  // Use ArrayChecker to detect and validate the input format
  ArrayChecker checker("map_joint_parameters");
  JointParamsShape inputShape =
      checker.validateJointParameters(motionData, "motion_data", srcCharacter);

  const auto& leadingDims = checker.getLeadingDimensions();

  if (checker.isFloat64()) {
    return mapJointParametersImpl<double>(
        motionData,
        srcSkeleton.getJointNames(),
        tgtSkeleton.getJointNames(),
        leadingDims,
        nSrcJoints,
        nTgtJoints,
        inputShape);
  } else {
    return mapJointParametersImpl<float>(
        motionData,
        srcSkeleton.getJointNames(),
        tgtSkeleton.getJointNames(),
        leadingDims,
        nSrcJoints,
        nTgtJoints,
        inputShape);
  }
}

} // namespace pymomentum
