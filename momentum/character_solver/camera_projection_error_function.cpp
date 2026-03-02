/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/camera_projection_error_function.h"

#include "momentum/character/skeleton_state.h"
#include "momentum/character_solver/skeleton_derivative.h"

namespace momentum {

template <typename T>
CameraProjectionErrorFunctionT<T>::CameraProjectionErrorFunctionT(
    const Skeleton& skel,
    const ParameterTransform& pt,
    std::shared_ptr<const IntrinsicsModelT<T>> intrinsicsModel,
    size_t cameraParent,
    const Eigen::Transform<T, 3, Eigen::Affine>& cameraOffset)
    : SkeletonErrorFunctionT<T>(skel, pt),
      intrinsicsModel_(std::move(intrinsicsModel)),
      cameraParent_(cameraParent),
      cameraOffset_(cameraOffset),
      skeletonDerivative_(skel, pt, this->activeJointParams_, this->enabledParameters_) {}

template <typename T>
CameraProjectionErrorFunctionT<T>::CameraProjectionErrorFunctionT(
    const Skeleton& skel,
    const ParameterTransform& pt,
    const CameraT<T>& camera,
    size_t cameraParent)
    : SkeletonErrorFunctionT<T>(skel, pt),
      intrinsicsModel_(camera.intrinsicsModel()),
      cameraParent_(cameraParent),
      cameraOffset_(camera.eyeFromWorld()),
      skeletonDerivative_(skel, pt, this->activeJointParams_, this->enabledParameters_) {}

template <typename T>
void CameraProjectionErrorFunctionT<T>::addConstraint(const ProjectionConstraintT<T>& constraint) {
  constraints_.push_back(constraint);
}

template <typename T>
double CameraProjectionErrorFunctionT<T>::getError(
    const ModelParametersT<T>& /*params*/,
    const SkeletonStateT<T>& skeletonState,
    const MeshStateT<T>& /*meshState*/) {
  double error = 0;

  const auto& jointState = skeletonState.jointState;

  // Compute eyeFromWorld transform (loop-invariant)
  Eigen::Transform<T, 3, Eigen::Affine> eyeFromWorld;
  if (cameraParent_ == kInvalidIndex) {
    eyeFromWorld = cameraOffset_;
  } else {
    const auto worldFromEye = jointState[cameraParent_].transform * cameraOffset_;
    eyeFromWorld = worldFromEye.inverse();
  }

  for (size_t iCons = 0; iCons < constraints_.size(); ++iCons) {
    const auto& cons = constraints_[iCons];

    // Compute world position of the constraint point
    const Eigen::Vector3<T> p_world = jointState[cons.parent].transform * cons.offset;

    // Transform to eye space
    const Eigen::Vector3<T> p_eye = eyeFromWorld * p_world;

    // Project through intrinsics
    const auto [projected, valid] = intrinsicsModel_->project(p_eye);
    if (!valid) {
      continue;
    }

    const Eigen::Vector2<T> residual = projected.template head<2>() - cons.target;
    error += cons.weight * this->weight_ * residual.squaredNorm();
  }

  return error;
}

template <typename T>
double CameraProjectionErrorFunctionT<T>::getGradient(
    const ModelParametersT<T>& /*params*/,
    const SkeletonStateT<T>& skeletonState,
    const MeshStateT<T>& /*meshState*/,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  double error = 0;

  const auto& jointState = skeletonState.jointState;

  // Compute eyeFromWorld transform (loop-invariant)
  Eigen::Transform<T, 3, Eigen::Affine> eyeFromWorld;
  if (cameraParent_ == kInvalidIndex) {
    eyeFromWorld = cameraOffset_;
  } else {
    const auto worldFromEye = jointState[cameraParent_].transform * cameraOffset_;
    eyeFromWorld = worldFromEye.inverse();
  }
  const Eigen::Matrix<T, 3, 3> R_eyeFromWorld = eyeFromWorld.linear();

  // Pre-allocate joint gradient storage
  Eigen::VectorX<T> jointGrad =
      Eigen::VectorX<T>::Zero(this->parameterTransform_.numAllModelParameters());

  for (size_t iCons = 0; iCons < constraints_.size(); ++iCons) {
    const auto& cons = constraints_[iCons];

    // Compute world position of the constraint point
    const Eigen::Vector3<T> p_world = jointState[cons.parent].transform * cons.offset;

    // Transform to eye space
    const Eigen::Vector3<T> p_eye = eyeFromWorld * p_world;

    // Project through intrinsics with Jacobian
    const auto [projected, J_eye_3x3, valid] = intrinsicsModel_->projectJacobian(p_eye);
    if (!valid) {
      continue;
    }

    const Eigen::Vector2<T> residual = projected.template head<2>() - cons.target;
    error += cons.weight * this->weight_ * residual.squaredNorm();

    const T wgt = T(2) * cons.weight * this->weight_;

    // J_pixel_world = J_eye_3x3.topRows<2>() * R_eyeFromWorld
    const Eigen::Matrix<T, 2, 3> J_pixel_world = J_eye_3x3.template topRows<2>() * R_eyeFromWorld;

    // Walk 1: Constraint-point parent chain (positive sign)
    // p_world is a point (isPoint=1) attached to cons.parent
    {
      const Eigen::Vector2<T> weightedResidual = wgt * residual;
      std::array<Eigen::Vector3<T>, 1> worldVecs = {p_world};
      std::array<Eigen::Matrix<T, 2, 3>, 1> dfdvArr = {J_pixel_world};
      std::array<uint8_t, 1> isPoints = {1};

      skeletonDerivative_.template accumulateJointGradient<2>(
          cons.parent,
          std::span<const Eigen::Vector3<T>>(worldVecs.data(), 1),
          std::span<const Eigen::Matrix<T, 2, 3>>(dfdvArr.data(), 1),
          std::span<const uint8_t>(isPoints.data(), 1),
          weightedResidual,
          jointState,
          jointGrad);
    }

    // Walk 2: Camera-parent chain (negative sign)
    // The lever arm uses p_world because we differentiate eyeFromWorld * p_world
    // w.r.t. q through eyeFromWorld. The negative sign is because increasing
    // camera joint parameters moves the camera, which has the opposite effect
    // on the projected point.
    if (cameraParent_ != kInvalidIndex) {
      const Eigen::Vector2<T> weightedResidual = -wgt * residual;
      std::array<Eigen::Vector3<T>, 1> worldVecs = {p_world};
      std::array<Eigen::Matrix<T, 2, 3>, 1> dfdvArr = {J_pixel_world};
      std::array<uint8_t, 1> isPoints = {1};

      skeletonDerivative_.template accumulateJointGradient<2>(
          cameraParent_,
          std::span<const Eigen::Vector3<T>>(worldVecs.data(), 1),
          std::span<const Eigen::Matrix<T, 2, 3>>(dfdvArr.data(), 1),
          std::span<const uint8_t>(isPoints.data(), 1),
          weightedResidual,
          jointState,
          jointGrad);
    }
  }

  gradient += jointGrad;
  return error;
}

template <typename T>
double CameraProjectionErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& /*params*/,
    const SkeletonStateT<T>& skeletonState,
    const MeshStateT<T>& /*meshState*/,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    int& usedRows) {
  double error = 0;

  const auto& jointState = skeletonState.jointState;

  // Compute eyeFromWorld transform (loop-invariant)
  Eigen::Transform<T, 3, Eigen::Affine> eyeFromWorld;
  if (cameraParent_ == kInvalidIndex) {
    eyeFromWorld = cameraOffset_;
  } else {
    const auto worldFromEye = jointState[cameraParent_].transform * cameraOffset_;
    eyeFromWorld = worldFromEye.inverse();
  }
  const Eigen::Matrix<T, 3, 3> R_eyeFromWorld = eyeFromWorld.linear();

  for (size_t iCons = 0; iCons < constraints_.size(); ++iCons) {
    const auto& cons = constraints_[iCons];

    // Compute world position of the constraint point
    const Eigen::Vector3<T> p_world = jointState[cons.parent].transform * cons.offset;

    // Transform to eye space
    const Eigen::Vector3<T> p_eye = eyeFromWorld * p_world;

    // Project through intrinsics with Jacobian
    const auto [projected, J_eye_3x3, valid] = intrinsicsModel_->projectJacobian(p_eye);
    if (!valid) {
      continue;
    }

    const Eigen::Vector2<T> res = projected.template head<2>() - cons.target;
    error += cons.weight * this->weight_ * res.squaredNorm();

    const T wgt = std::sqrt(cons.weight * this->weight_);
    const size_t rowIndex = 2 * iCons;
    residual.template segment<2>(rowIndex) = wgt * res;

    // J_pixel_world = J_eye_3x3.topRows<2>() * R_eyeFromWorld
    const Eigen::Matrix<T, 2, 3> J_pixel_world = J_eye_3x3.template topRows<2>() * R_eyeFromWorld;

    // Walk 1: Constraint-point parent chain (positive sign)
    {
      std::array<Eigen::Vector3<T>, 1> worldVecs = {p_world};
      std::array<Eigen::Matrix<T, 2, 3>, 1> dfdvArr = {J_pixel_world};
      std::array<uint8_t, 1> isPoints = {1};

      skeletonDerivative_.template accumulateJointJacobian<2>(
          cons.parent,
          std::span<const Eigen::Vector3<T>>(worldVecs.data(), 1),
          std::span<const Eigen::Matrix<T, 2, 3>>(dfdvArr.data(), 1),
          std::span<const uint8_t>(isPoints.data(), 1),
          wgt,
          jointState,
          jacobian,
          rowIndex);
    }

    // Walk 2: Camera-parent chain (negative sign)
    if (cameraParent_ != kInvalidIndex) {
      std::array<Eigen::Vector3<T>, 1> worldVecs = {p_world};
      std::array<Eigen::Matrix<T, 2, 3>, 1> dfdvArr = {J_pixel_world};
      std::array<uint8_t, 1> isPoints = {1};

      skeletonDerivative_.template accumulateJointJacobian<2>(
          cameraParent_,
          std::span<const Eigen::Vector3<T>>(worldVecs.data(), 1),
          std::span<const Eigen::Matrix<T, 2, 3>>(dfdvArr.data(), 1),
          std::span<const uint8_t>(isPoints.data(), 1),
          -wgt,
          jointState,
          jacobian,
          rowIndex);
    }
  }

  usedRows = static_cast<int>(jacobian.rows());

  return error;
}

template <typename T>
size_t CameraProjectionErrorFunctionT<T>::getJacobianSize() const {
  return 2 * constraints_.size();
}

template class CameraProjectionErrorFunctionT<float>;
template class CameraProjectionErrorFunctionT<double>;

} // namespace momentum
