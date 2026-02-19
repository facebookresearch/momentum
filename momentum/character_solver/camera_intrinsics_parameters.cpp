/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/camera_intrinsics_parameters.h"

#include <momentum/common/checks.h>

#include <sstream>

namespace momentum {

namespace {

std::string makePrefix(const std::string& cameraName) {
  return std::string(kIntrinsicsParamPrefix) + cameraName + "_";
}

// Find the model parameter index for a given full parameter name, or -1 if not present.
Eigen::Index findParamIndex(const ParameterTransform& paramTransform, const std::string& fullName) {
  for (size_t i = 0; i < paramTransform.name.size(); ++i) {
    if (paramTransform.name[i] == fullName) {
      return static_cast<Eigen::Index>(i);
    }
  }
  return -1;
}

} // namespace

std::tuple<ParameterTransform, ParameterLimits> addCameraIntrinsicsParameters(
    ParameterTransform paramTransform,
    ParameterLimits paramLimits,
    const IntrinsicsModel& intrinsicsModel) {
  const std::string& effectiveName = intrinsicsModel.name();
  MT_CHECK(
      !effectiveName.empty(),
      "Camera name must not be empty; call setName() on the intrinsics model first.");

  const std::string prefix = makePrefix(effectiveName);
  const std::string paramSetName = std::string(kIntrinsicsParamPrefix) + effectiveName;

  // Strip existing params for same camera name (idempotency)
  ParameterSet existingParams = getCameraIntrinsicsParameterSet(paramTransform, effectiveName);
  if (existingParams.any()) {
    std::tie(paramTransform, paramLimits) =
        subsetParameterTransform(paramTransform, paramLimits, ~existingParams);
  }

  // Get parameter names from the intrinsics model
  const auto paramNames = intrinsicsModel.getParameterNames();

  // Build the parameter set for this camera
  ParameterSet paramSet;

  // Append parameters
  paramTransform.name.reserve(paramTransform.name.size() + paramNames.size());
  for (const auto& paramName : paramNames) {
    paramSet.set(paramTransform.name.size());
    paramTransform.name.push_back(prefix + paramName);
  }

  // Store the named parameter set
  paramTransform.parameterSets[paramSetName] = paramSet;

  // Resize the transform matrix with zero columns for the new parameters
  paramTransform.transform.conservativeResize(
      paramTransform.transform.rows(), paramTransform.name.size());
  paramTransform.transform.makeCompressed();

  return {paramTransform, paramLimits};
}

Eigen::VectorXf extractCameraIntrinsics(
    const ParameterTransform& paramTransform,
    const ModelParameters& modelParams,
    const IntrinsicsModel& intrinsicsModel) {
  // Start with the model's current values as defaults for any missing params
  Eigen::VectorXf result = intrinsicsModel.getIntrinsicParameters();
  const std::string prefix = makePrefix(intrinsicsModel.name());
  const auto paramNames = intrinsicsModel.getParameterNames();

  for (size_t i = 0; i < paramNames.size(); ++i) {
    const auto paramIdx = findParamIndex(paramTransform, prefix + paramNames[i]);
    if (paramIdx >= 0 && paramIdx < modelParams.size()) {
      result(static_cast<Eigen::Index>(i)) = modelParams(paramIdx);
    }
  }
  return result;
}

void setCameraIntrinsics(
    const ParameterTransform& paramTransform,
    const IntrinsicsModel& intrinsicsModel,
    ModelParameters& modelParams) {
  const std::string prefix = makePrefix(intrinsicsModel.name());
  const auto paramNames = intrinsicsModel.getParameterNames();
  const Eigen::VectorXf intrinsicValues = intrinsicsModel.getIntrinsicParameters();

  for (size_t i = 0; i < paramNames.size(); ++i) {
    const auto paramIdx = findParamIndex(paramTransform, prefix + paramNames[i]);
    if (paramIdx >= 0 && paramIdx < modelParams.size()) {
      modelParams(paramIdx) = intrinsicValues(static_cast<Eigen::Index>(i));
    }
  }
}

ParameterSet getCameraIntrinsicsParameterSet(
    const ParameterTransform& paramTransform,
    const std::string& cameraName) {
  const std::string prefix = makePrefix(cameraName);
  ParameterSet result;
  for (size_t i = 0; i < paramTransform.name.size(); ++i) {
    if (paramTransform.name[i].compare(0, prefix.size(), prefix) == 0) {
      result.set(i);
    }
  }
  return result;
}

ParameterSet getAllCameraIntrinsicsParameterSet(const ParameterTransform& paramTransform) {
  const std::string prefix(kIntrinsicsParamPrefix);
  ParameterSet result;
  for (size_t i = 0; i < paramTransform.name.size(); ++i) {
    if (paramTransform.name[i].compare(0, prefix.size(), prefix) == 0) {
      result.set(i);
    }
  }
  return result;
}

} // namespace momentum
