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

/// Cached mapping between intrinsic parameters and model parameter indices.
///
/// This struct is constructed once (at error function setup time) and caches the
/// index mapping so that solve-time operations avoid string comparisons. It also
/// holds a mutable clone of the intrinsics model that can be updated from model
/// parameters each iteration.
template <typename T>
struct CameraIntrinsicsMapping {
  /// For each intrinsic parameter i, the model parameter index, or -1 if absent.
  std::vector<Eigen::Index> modelParamIndices;

  /// Mutable clone of the intrinsics model, updated each iteration via updateIntrinsics().
  std::shared_ptr<IntrinsicsModelT<T>> mutableIntrinsics;

  /// Build the mapping from a parameter transform and intrinsics model.
  /// The intrinsics model must have a non-empty name.
  CameraIntrinsicsMapping(
      const ParameterTransform& paramTransform,
      const IntrinsicsModelT<T>& intrinsicsModel);

  /// Are any intrinsic parameters being optimized?
  [[nodiscard]] bool hasActiveParams() const;

  /// Update the mutable intrinsics model from current model parameters.
  /// Only modifies parameters that have active mappings; others retain their
  /// current values. Returns a reference to the updated model.
  const IntrinsicsModelT<T>& updateIntrinsics(const ModelParametersT<T>& modelParams);

  /// Accumulate intrinsics Jacobian contributions into the gradient vector.
  ///
  /// @param J_intrinsics The 3xN intrinsics Jacobian from projectIntrinsicsJacobian()
  /// @param residual The 2D residual (projected - target)
  /// @param weight Combined weight (2 * constraint_weight * error_function_weight)
  /// @param gradient The gradient vector to accumulate into
  void addGradient(
      const Eigen::Matrix<T, 3, Eigen::Dynamic>& J_intrinsics,
      const Eigen::Vector2<T>& residual,
      T weight,
      Eigen::Ref<Eigen::VectorX<T>> gradient) const;

  /// Accumulate intrinsics Jacobian contributions into the Jacobian matrix.
  ///
  /// @param J_intrinsics The 3xN intrinsics Jacobian from projectIntrinsicsJacobian()
  /// @param weight Combined weight (sqrt(constraint_weight * error_function_weight))
  /// @param rowOffset Row offset into the Jacobian matrix (2 * constraint_index)
  /// @param jacobian The Jacobian matrix to accumulate into
  void addJacobian(
      const Eigen::Matrix<T, 3, Eigen::Dynamic>& J_intrinsics,
      T weight,
      Eigen::Index rowOffset,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian) const;
};

} // namespace momentum
