/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/camera/camera.h>
#include <momentum/character/parameter_limits.h>
#include <momentum/character/parameter_transform.h>
#include <momentum/character/types.h>

#include <string>
#include <tuple>

namespace momentum {

inline constexpr const char* kIntrinsicsParamPrefix = "intrinsics_";

/// Add camera intrinsics parameters to the parameter transform.
/// Names follow "intrinsics_{cameraName}_{paramName}" convention.
/// Idempotent: strips existing params for same camera name first.
/// New columns in transform matrix are zero (not mapped to joints).
///
/// @param paramTransform The parameter transform to augment
/// @param paramLimits The parameter limits to augment
/// @param intrinsicsModel The intrinsics model whose parameters to add (must have a non-empty name)
/// @return Tuple of updated (paramTransform, paramLimits)
/// @throws if intrinsicsModel.name() is empty
[[nodiscard]] std::tuple<ParameterTransform, ParameterLimits> addCameraIntrinsicsParameters(
    ParameterTransform paramTransform,
    ParameterLimits paramLimits,
    const IntrinsicsModel& intrinsicsModel);

/// Extract camera intrinsic parameter values from model parameters.
///
/// Reads the values at the parameter indices corresponding to the camera's
/// intrinsics (looked up by name in paramTransform). Parameters not present
/// in paramTransform retain the current values from the intrinsics model.
/// This mirrors the extractBlendWeights() pattern.
///
/// @param paramTransform The parameter transform with camera intrinsics parameters
/// @param modelParams The model parameters to read from
/// @param intrinsicsModel The intrinsics model (provides name, parameter names, and default values)
/// @return Intrinsic parameter vector suitable for IntrinsicsModel::setIntrinsicParameters()
[[nodiscard]] Eigen::VectorXf extractCameraIntrinsics(
    const ParameterTransform& paramTransform,
    const ModelParameters& modelParams,
    const IntrinsicsModel& intrinsicsModel);

/// Write camera intrinsic parameter values from an IntrinsicsModel into model parameters.
///
/// Copies the intrinsics model's current parameter values into the corresponding
/// slots of modelParameters. Parameters not present in paramTransform are
/// silently skipped.
///
/// @param paramTransform The parameter transform with camera intrinsics parameters
/// @param intrinsicsModel The intrinsics model to read values from
/// @param[in,out] modelParams The model parameters to write into
void setCameraIntrinsics(
    const ParameterTransform& paramTransform,
    const IntrinsicsModel& intrinsicsModel,
    ModelParameters& modelParams);

/// Get ParameterSet for a specific camera's intrinsics parameters.
///
/// @param paramTransform The parameter transform to search
/// @param cameraName The camera name to look up
/// @return ParameterSet with bits set for the camera's intrinsics parameters
[[nodiscard]] ParameterSet getCameraIntrinsicsParameterSet(
    const ParameterTransform& paramTransform,
    const std::string& cameraName);

/// Get ParameterSet for ALL camera intrinsics parameters.
///
/// @param paramTransform The parameter transform to search
/// @return ParameterSet with bits set for all intrinsics parameters
[[nodiscard]] ParameterSet getAllCameraIntrinsicsParameterSet(
    const ParameterTransform& paramTransform);

} // namespace momentum
