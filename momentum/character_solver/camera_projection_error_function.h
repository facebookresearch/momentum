/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/camera/camera.h>
#include <momentum/character_solver/skeleton_error_function.h>

#include <memory>

namespace momentum {

/// Per-constraint data for CameraProjectionErrorFunctionT.
///
/// Each constraint specifies a 3D point (attached to a skeleton joint via parent + offset)
/// and a target 2D pixel location. The error function penalizes the reprojection error
/// between the projected 3D point and the target pixel.
template <typename T>
struct ProjectionConstraintT {
  size_t parent{}; ///< Parent joint of the 3D point
  Eigen::Vector3<T> offset = Eigen::Vector3<T>::Zero(); ///< Offset from parent joint
  T weight = 1; ///< Constraint weight
  Eigen::Vector2<T> target = Eigen::Vector2<T>::Zero(); ///< Target pixel coordinates
};

/// Error function that projects 3D skeleton points through a bone-parented camera
/// using an IntrinsicsModelT<T> and penalizes reprojection error in pixel space.
///
/// The camera is rigidly attached to a skeleton joint (cameraParent) with a fixed
/// offset (cameraOffset). If cameraParent is kInvalidIndex, the camera is static
/// in world space (cameraOffset is used directly as eyeFromWorld).
///
/// Each constraint specifies a 3D point attached to a joint and a target 2D pixel.
/// The error is the squared distance between the projected point and the target.
template <typename T>
class CameraProjectionErrorFunctionT : public SkeletonErrorFunctionT<T> {
 public:
  CameraProjectionErrorFunctionT(
      const Skeleton& skel,
      const ParameterTransform& pt,
      std::shared_ptr<const IntrinsicsModelT<T>> intrinsicsModel,
      size_t cameraParent,
      const Eigen::Transform<T, 3, Eigen::Affine>& cameraOffset =
          Eigen::Transform<T, 3, Eigen::Affine>::Identity());

  /// Convenience constructor that extracts intrinsics and extrinsics from a Camera.
  CameraProjectionErrorFunctionT(
      const Skeleton& skel,
      const ParameterTransform& pt,
      const CameraT<T>& camera,
      size_t cameraParent = kInvalidIndex);

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

  void addConstraint(const ProjectionConstraintT<T>& constraint);

  void clearConstraints() {
    constraints_.clear();
  }

  void setConstraints(std::vector<ProjectionConstraintT<T>> constraints) {
    constraints_ = std::move(constraints);
  }

  [[nodiscard]] const std::vector<ProjectionConstraintT<T>>& getConstraints() const {
    return constraints_;
  }

  [[nodiscard]] size_t numConstraints() const {
    return constraints_.size();
  }

  [[nodiscard]] bool empty() const {
    return constraints_.empty();
  }

 protected:
  std::vector<ProjectionConstraintT<T>> constraints_;
  std::shared_ptr<const IntrinsicsModelT<T>> intrinsicsModel_;
  size_t cameraParent_;
  Eigen::Transform<T, 3, Eigen::Affine> cameraOffset_;
};

} // namespace momentum
