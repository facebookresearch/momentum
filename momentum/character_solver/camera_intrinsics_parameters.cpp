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
    const IntrinsicsModel& intrinsicsModel,
    bool tieFocalLength) {
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

  // When tieFocalLength is true, replace separate "fx"/"fy" with a single "f"
  std::vector<std::string> addedNames;
  bool fAdded = false;
  for (const auto& paramName : paramNames) {
    if (tieFocalLength && (paramName == "fx" || paramName == "fy")) {
      if (!fAdded) {
        addedNames.emplace_back(kTiedFocalLengthSuffix);
        fAdded = true;
      }
      // skip fy (already covered by "f")
    } else {
      addedNames.push_back(paramName);
    }
  }

  // Append parameters
  paramTransform.name.reserve(paramTransform.name.size() + addedNames.size());
  for (const auto& name : addedNames) {
    paramSet.set(paramTransform.name.size());
    paramTransform.name.push_back(prefix + name);
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
  const Eigen::Index tiedFIdx = findParamIndex(paramTransform, prefix + kTiedFocalLengthSuffix);

  for (size_t i = 0; i < paramNames.size(); ++i) {
    auto paramIdx = findParamIndex(paramTransform, prefix + paramNames[i]);
    if (paramIdx < 0 && tiedFIdx >= 0 && (paramNames[i] == "fx" || paramNames[i] == "fy")) {
      paramIdx = tiedFIdx;
    }
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
  const Eigen::Index tiedFIdx = findParamIndex(paramTransform, prefix + kTiedFocalLengthSuffix);

  for (size_t i = 0; i < paramNames.size(); ++i) {
    auto paramIdx = findParamIndex(paramTransform, prefix + paramNames[i]);
    if (paramIdx < 0 && tiedFIdx >= 0 && (paramNames[i] == "fx" || paramNames[i] == "fy")) {
      paramIdx = tiedFIdx;
    }
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

template <typename T>
CameraIntrinsicsMapping<T>::CameraIntrinsicsMapping(
    const ParameterTransform& paramTransform,
    const IntrinsicsModelT<T>& intrinsicsModel)
    : mutableIntrinsics(intrinsicsModel.clone()) {
  const std::string prefix = makePrefix(intrinsicsModel.name());
  const auto paramNames = intrinsicsModel.getParameterNames();
  modelParamIndices.resize(paramNames.size());

  // Check for tied focal length: a single "f" parameter that controls both fx and fy.
  const Eigen::Index tiedFIdx = findParamIndex(paramTransform, prefix + kTiedFocalLengthSuffix);

  for (size_t i = 0; i < paramNames.size(); ++i) {
    modelParamIndices[i] = findParamIndex(paramTransform, prefix + paramNames[i]);
    // Fall back to tied "f" for fx/fy if separate params not found
    if (modelParamIndices[i] < 0 && tiedFIdx >= 0 &&
        (paramNames[i] == "fx" || paramNames[i] == "fy")) {
      modelParamIndices[i] = tiedFIdx;
    }
  }
}

template <typename T>
bool CameraIntrinsicsMapping<T>::hasActiveParams() const {
  for (const auto idx : modelParamIndices) {
    if (idx >= 0) {
      return true;
    }
  }
  return false;
}

template <typename T>
const IntrinsicsModelT<T>& CameraIntrinsicsMapping<T>::updateIntrinsics(
    const ModelParametersT<T>& modelParams) {
  Eigen::VectorX<T> params = mutableIntrinsics->getIntrinsicParameters();
  for (size_t i = 0; i < modelParamIndices.size(); ++i) {
    const auto idx = modelParamIndices[i];
    if (idx >= 0 && idx < modelParams.size()) {
      params(static_cast<Eigen::Index>(i)) = modelParams(idx);
    }
  }
  mutableIntrinsics->setIntrinsicParameters(params);
  return *mutableIntrinsics;
}

template <typename T>
void CameraIntrinsicsMapping<T>::addGradient(
    const Eigen::Matrix<T, 3, Eigen::Dynamic>& J_intrinsics,
    const Eigen::Vector2<T>& residual,
    T weight,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  for (size_t i = 0; i < modelParamIndices.size(); ++i) {
    const auto idx = modelParamIndices[i];
    if (idx >= 0) {
      const T gradVal = J_intrinsics.col(i).template head<2>().dot(residual);
      gradient[idx] += weight * gradVal;
    }
  }
}

template <typename T>
void CameraIntrinsicsMapping<T>::addJacobian(
    const Eigen::Matrix<T, 3, Eigen::Dynamic>& J_intrinsics,
    T weight,
    Eigen::Index rowOffset,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian) const {
  for (size_t i = 0; i < modelParamIndices.size(); ++i) {
    const auto idx = modelParamIndices[i];
    if (idx >= 0) {
      jacobian.template block<2, 1>(rowOffset, idx) +=
          weight * J_intrinsics.col(i).template head<2>();
    }
  }
}

template struct CameraIntrinsicsMapping<float>;
template struct CameraIntrinsicsMapping<double>;

} // namespace momentum
