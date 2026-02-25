/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/camera_projection_error_function.h"

#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"

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
      cameraOffset_(cameraOffset) {
  if (!intrinsicsModel_->name().empty()) {
    CameraIntrinsicsMapping<T> mapping(pt, *intrinsicsModel_);
    if (mapping.hasActiveParams()) {
      intrinsicsMapping_.emplace(std::move(mapping));
    }
  }
}

template <typename T>
CameraProjectionErrorFunctionT<T>::CameraProjectionErrorFunctionT(
    const Skeleton& skel,
    const ParameterTransform& pt,
    const CameraT<T>& camera,
    size_t cameraParent)
    : SkeletonErrorFunctionT<T>(skel, pt),
      intrinsicsModel_(camera.intrinsicsModel()),
      cameraParent_(cameraParent),
      cameraOffset_(camera.eyeFromWorld()) {
  if (!intrinsicsModel_->name().empty()) {
    CameraIntrinsicsMapping<T> mapping(pt, *intrinsicsModel_);
    if (mapping.hasActiveParams()) {
      intrinsicsMapping_.emplace(std::move(mapping));
    }
  }
}

template <typename T>
void CameraProjectionErrorFunctionT<T>::addConstraint(const ProjectionConstraintT<T>& constraint) {
  constraints_.push_back(constraint);
}

template <typename T>
double CameraProjectionErrorFunctionT<T>::getError(
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& skeletonState,
    const MeshStateT<T>& /*meshState*/) {
  double error = 0;

  // If intrinsics are being optimized, update from current model parameters
  const auto& intrinsics =
      intrinsicsMapping_ ? intrinsicsMapping_->updateIntrinsics(params) : *intrinsicsModel_;

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
    const auto [projected, valid] = intrinsics.project(p_eye);
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
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& skeletonState,
    const MeshStateT<T>& /*meshState*/,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  double error = 0;

  // If intrinsics are being optimized, update from current model parameters
  const auto& intrinsics =
      intrinsicsMapping_ ? intrinsicsMapping_->updateIntrinsics(params) : *intrinsicsModel_;

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
    const auto [projected, J_eye_3x3, valid] = intrinsics.projectJacobian(p_eye);
    if (!valid) {
      continue;
    }

    const Eigen::Vector2<T> residual = projected.template head<2>() - cons.target;
    error += cons.weight * this->weight_ * residual.squaredNorm();

    const T wgt = T(2) * cons.weight * this->weight_;

    // J_pixel_world = J_eye_3x3.topRows<2>() * R_eyeFromWorld
    const Eigen::Matrix<T, 2, 3> J_pixel_world = J_eye_3x3.template topRows<2>() * R_eyeFromWorld;

    // Helper to accumulate gradient for a given full-body DOF
    auto addGradient = [&](const size_t jFullBodyDOF, const Eigen::Vector2<T>& d_pixel) {
      const T gradVal = d_pixel.dot(residual);
      for (auto index = this->parameterTransform_.transform.outerIndexPtr()[jFullBodyDOF];
           index < this->parameterTransform_.transform.outerIndexPtr()[jFullBodyDOF + 1];
           ++index) {
        gradient[this->parameterTransform_.transform.innerIndexPtr()[index]] +=
            this->parameterTransform_.transform.valuePtr()[index] * wgt * gradVal;
      }
    };

    // For joints that are ancestors of both the constraint point and the camera,
    // the Walk 1 (+) and Walk 2 (-) contributions cancel exactly (same lever arm,
    // opposite signs). We skip them by stopping both walks at the LCA.
    // For static cameras (kInvalidIndex), commonAncestor returns kInvalidIndex,
    // so the walks naturally go all the way to the root.
    const size_t lca = this->skeleton_.commonAncestor(cons.parent, cameraParent_);

    // Walk 1: Constraint-point parent chain (positive sign)
    {
      size_t jointIndex = cons.parent;
      while (jointIndex != lca) {
        const auto& js = jointState[jointIndex];
        const size_t paramIndex = jointIndex * kParametersPerJoint;
        const Eigen::Vector3<T> posd = p_world - js.translation();

        for (size_t d = 0; d < 3; d++) {
          if (this->activeJointParams_[paramIndex + d]) {
            const Eigen::Vector2<T> dp = J_pixel_world * js.getTranslationDerivative(d);
            addGradient(paramIndex + d, dp);
          }
          if (this->activeJointParams_[paramIndex + 3 + d]) {
            const Eigen::Vector2<T> dp = J_pixel_world * js.getRotationDerivative(d, posd);
            addGradient(paramIndex + 3 + d, dp);
          }
        }
        if (this->activeJointParams_[paramIndex + 6]) {
          const Eigen::Vector2<T> dp = J_pixel_world * js.getScaleDerivative(posd);
          addGradient(paramIndex + 6, dp);
        }

        jointIndex = this->skeleton_.joints[jointIndex].parent;
      }
    }

    // Walk 2: Camera-parent chain (negative sign)
    // The lever arm uses p_world because we differentiate eyeFromWorld * p_world
    // w.r.t. q through eyeFromWorld.
    if (cameraParent_ != kInvalidIndex) {
      size_t jointIndex = cameraParent_;
      while (jointIndex != lca) {
        const auto& js = jointState[jointIndex];
        const size_t paramIndex = jointIndex * kParametersPerJoint;
        const Eigen::Vector3<T> posd = p_world - js.translation();

        for (size_t d = 0; d < 3; d++) {
          if (this->activeJointParams_[paramIndex + d]) {
            const Eigen::Vector2<T> dp = -J_pixel_world * js.getTranslationDerivative(d);
            addGradient(paramIndex + d, dp);
          }
          if (this->activeJointParams_[paramIndex + 3 + d]) {
            const Eigen::Vector2<T> dp = -J_pixel_world * js.getRotationDerivative(d, posd);
            addGradient(paramIndex + 3 + d, dp);
          }
        }
        if (this->activeJointParams_[paramIndex + 6]) {
          const Eigen::Vector2<T> dp = -J_pixel_world * js.getScaleDerivative(posd);
          addGradient(paramIndex + 6, dp);
        }

        jointIndex = this->skeleton_.joints[jointIndex].parent;
      }
    }

    // Intrinsics parameters gradient
    if (intrinsicsMapping_) {
      const auto [projected_i, J_intrinsics, valid_i] = intrinsics.projectIntrinsicsJacobian(p_eye);
      if (valid_i) {
        intrinsicsMapping_->addGradient(J_intrinsics, residual, wgt, gradient);
      }
    }
  }

  return error;
}

template <typename T>
double CameraProjectionErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& skeletonState,
    const MeshStateT<T>& /*meshState*/,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    int& usedRows) {
  double error = 0;

  // If intrinsics are being optimized, update from current model parameters
  const auto& intrinsics =
      intrinsicsMapping_ ? intrinsicsMapping_->updateIntrinsics(params) : *intrinsicsModel_;

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
    const auto [projected, J_eye_3x3, valid] = intrinsics.projectJacobian(p_eye);
    if (!valid) {
      continue;
    }

    const Eigen::Vector2<T> res = projected.template head<2>() - cons.target;
    error += cons.weight * this->weight_ * res.squaredNorm();

    const T wgt = std::sqrt(cons.weight * this->weight_);
    residual.template segment<2>(2 * iCons) = wgt * res;

    // J_pixel_world = J_eye_3x3.topRows<2>() * R_eyeFromWorld
    const Eigen::Matrix<T, 2, 3> J_pixel_world = J_eye_3x3.template topRows<2>() * R_eyeFromWorld;

    // Helper to accumulate Jacobian for a given full-body DOF
    auto addJacobian = [&](const size_t jFullBodyDOF, const Eigen::Vector2<T>& d_pixel) {
      for (auto index = this->parameterTransform_.transform.outerIndexPtr()[jFullBodyDOF];
           index < this->parameterTransform_.transform.outerIndexPtr()[jFullBodyDOF + 1];
           ++index) {
        const auto jReducedDOF = this->parameterTransform_.transform.innerIndexPtr()[index];
        jacobian.template block<2, 1>(2 * iCons, jReducedDOF) +=
            this->parameterTransform_.transform.valuePtr()[index] * wgt * d_pixel;
      }
    };

    // For joints that are ancestors of both the constraint point and the camera,
    // the Walk 1 (+) and Walk 2 (-) contributions cancel exactly (same lever arm,
    // opposite signs). We skip them by stopping both walks at the LCA.
    const size_t lca = this->skeleton_.commonAncestor(cons.parent, cameraParent_);

    // Walk 1: Constraint-point parent chain (positive sign)
    {
      size_t jointIndex = cons.parent;
      while (jointIndex != lca) {
        const auto& js = jointState[jointIndex];
        const size_t paramIndex = jointIndex * kParametersPerJoint;
        const Eigen::Vector3<T> posd = p_world - js.translation();

        for (size_t d = 0; d < 3; d++) {
          if (this->activeJointParams_[paramIndex + d]) {
            const Eigen::Vector2<T> dp = J_pixel_world * js.getTranslationDerivative(d);
            addJacobian(paramIndex + d, dp);
          }
          if (this->activeJointParams_[paramIndex + 3 + d]) {
            const Eigen::Vector2<T> dp = J_pixel_world * js.getRotationDerivative(d, posd);
            addJacobian(paramIndex + 3 + d, dp);
          }
        }
        if (this->activeJointParams_[paramIndex + 6]) {
          const Eigen::Vector2<T> dp = J_pixel_world * js.getScaleDerivative(posd);
          addJacobian(paramIndex + 6, dp);
        }

        jointIndex = this->skeleton_.joints[jointIndex].parent;
      }
    }

    // Walk 2: Camera-parent chain (negative sign)
    // The lever arm uses p_world because we differentiate eyeFromWorld * p_world
    // w.r.t. q through eyeFromWorld.
    if (cameraParent_ != kInvalidIndex) {
      size_t jointIndex = cameraParent_;
      while (jointIndex != lca) {
        const auto& js = jointState[jointIndex];
        const size_t paramIndex = jointIndex * kParametersPerJoint;
        const Eigen::Vector3<T> posd = p_world - js.translation();

        for (size_t d = 0; d < 3; d++) {
          if (this->activeJointParams_[paramIndex + d]) {
            const Eigen::Vector2<T> dp = -J_pixel_world * js.getTranslationDerivative(d);
            addJacobian(paramIndex + d, dp);
          }
          if (this->activeJointParams_[paramIndex + 3 + d]) {
            const Eigen::Vector2<T> dp = -J_pixel_world * js.getRotationDerivative(d, posd);
            addJacobian(paramIndex + 3 + d, dp);
          }
        }
        if (this->activeJointParams_[paramIndex + 6]) {
          const Eigen::Vector2<T> dp = -J_pixel_world * js.getScaleDerivative(posd);
          addJacobian(paramIndex + 6, dp);
        }

        jointIndex = this->skeleton_.joints[jointIndex].parent;
      }
    }

    // Intrinsics parameters Jacobian
    if (intrinsicsMapping_) {
      const auto [projected_i, J_intrinsics, valid_i] = intrinsics.projectIntrinsicsJacobian(p_eye);
      if (valid_i) {
        intrinsicsMapping_->addJacobian(J_intrinsics, wgt, 2 * iCons, jacobian);
      }
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
