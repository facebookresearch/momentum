/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/camera/camera.h>
#include <momentum/character_solver/camera_intrinsics_parameters.h>
#include <momentum/character_solver/skeleton_derivative.h>
#include <momentum/character_solver/skeleton_error_function.h>

#include <memory>
#include <optional>
#include <vector>

namespace momentum {

/// Per-constraint data for CameraVertexProjectionErrorFunctionT.
///
/// Each constraint specifies a mesh vertex (by index) and a target 2D pixel
/// location. The error function penalizes the reprojection error between the
/// vertex's projected position and the target pixel.
template <typename T>
struct CameraVertexProjectionDataT {
  size_t vertexIndex = kInvalidIndex; ///< Mesh vertex index
  Eigen::Vector2<T> target = Eigen::Vector2<T>::Zero(); ///< Target pixel coordinates
  T weight = T(1); ///< Constraint weight
};

/// Error function that projects skinned mesh vertices through a bone-parented
/// camera using an IntrinsicsModelT<T> and penalizes reprojection error.
///
/// Combines vertex-based skinning derivatives (with blend shape support) and
/// camera bone extrinsics optimization with optional intrinsics optimization.
/// This enables dense mesh keypoints to contribute to camera parameter
/// optimization, unlike VertexProjectionErrorFunctionT which uses a fixed
/// pre-baked projection matrix.
///
/// The camera is rigidly attached to a skeleton joint (cameraParent) with a
/// fixed offset (cameraOffset). If cameraParent is kInvalidIndex, the camera
/// is static in world space.
template <typename T>
class CameraVertexProjectionErrorFunctionT : public SkeletonErrorFunctionT<T> {
 public:
  CameraVertexProjectionErrorFunctionT(
      const Character& character,
      const ParameterTransform& pt,
      std::shared_ptr<const IntrinsicsModelT<T>> intrinsicsModel,
      size_t cameraParent,
      const Eigen::Transform<T, 3, Eigen::Affine>& cameraOffset =
          Eigen::Transform<T, 3, Eigen::Affine>::Identity());

  /// Convenience constructor that extracts intrinsics and extrinsics from a Camera.
  CameraVertexProjectionErrorFunctionT(
      const Character& character,
      const ParameterTransform& pt,
      const CameraT<T>& camera,
      size_t cameraParent = kInvalidIndex);

  [[nodiscard]] const Character* getCharacter() const final;
  [[nodiscard]] bool needsMesh() const final;

  [[nodiscard]] double getError(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState) final;
  double getGradient(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Eigen::Ref<Eigen::VectorX<T>> gradient) final;
  double getJacobian(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      Eigen::Ref<Eigen::VectorX<T>> residual,
      int& usedRows) final;
  [[nodiscard]] size_t getJacobianSize() const final;

  void addConstraint(const CameraVertexProjectionDataT<T>& constraint);

  void clearConstraints() {
    constraints_.clear();
  }

  void setConstraints(std::vector<CameraVertexProjectionDataT<T>> constraints) {
    constraints_ = std::move(constraints);
  }

  [[nodiscard]] const std::vector<CameraVertexProjectionDataT<T>>& getConstraints() const {
    return constraints_;
  }

  [[nodiscard]] size_t getNumConstraints() const {
    return constraints_.size();
  }

  [[nodiscard]] bool empty() const {
    return constraints_.empty();
  }

  void setMaxThreads(size_t maxThreads) {
    maxThreads_ = maxThreads;
  }

 protected:
  const Character& character_;
  std::vector<CameraVertexProjectionDataT<T>> constraints_;
  std::shared_ptr<const IntrinsicsModelT<T>> intrinsicsModel_;
  std::optional<CameraIntrinsicsMapping<T>> intrinsicsMapping_;
  size_t cameraParent_;
  Eigen::Transform<T, 3, Eigen::Affine> cameraOffset_;
  SkeletonDerivativeT<T> skeletonDerivative_;
  size_t maxThreads_ = 0;
};

} // namespace momentum
