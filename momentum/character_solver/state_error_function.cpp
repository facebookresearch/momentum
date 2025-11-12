/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/state_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character_solver/error_function_utils.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/math/utility.h"

namespace momentum {

namespace {

/// Computes the derivative of the logarithmic map of a relative rotation for q1 perturbation.
///
/// Given two quaternions q1 and q2, computes d/dt [log(q2^{-1} * exp(t*v) * q1)]|_{t=0}
/// for a single direction v.
///
/// @tparam T The scalar type
/// @param q1 The first quaternion
/// @param q2 The second quaternion
/// @param dir Rotation direction for q1 (3D vector)
/// @param dLogDq Precomputed derivative of logmap w.r.t. quaternion components (3x4 matrix)
///               This should be quaternionLogMapDerivative(q2^{-1} * q1)
/// @return 3D vector: d(log(q2^{-1} * q1))/d(q1 perturbation along dir)
template <typename T>
Vector3<T> quaternionLogMapRelativeDerivativeQ1(
    const Quaternion<T>& q1,
    const Quaternion<T>& q2,
    const Vector3<T>& dir,
    const Eigen::Matrix<T, 3, 4>& dLogDq) {
  // For q1: The perturbation is exp(t*v) * q1, so q_rel = q2^{-1} * (exp(t*v) * q1)
  // We need d/dt [q2^{-1} * (exp(t*v) * q1)]|_{t=0}
  //
  // The derivative of exp(t*v) at t=0 in the direction v is a quaternion [0, v/2]
  // Multiplying on the left gives: d/dt [exp(t*v) * q]|_{t=0} = [0, v/2] * q
  //
  // For a quaternion delta_q = [0, v/2] and q = [w, vec]:
  // delta_q * q = [-(v/2)·vec, (v/2)*w + (v/2)×vec]

  const Vector3<T> vHalf = dir / T(2);

  // Compute d/dt [exp(t*v) * q1]|_{t=0} = [0, v/2] * q1
  const Vector3<T> q1_vec(q1.x(), q1.y(), q1.z());
  const T dq1_w = -vHalf.dot(q1_vec);
  const Vector3<T> dq1_vec = vHalf * q1.w() + vHalf.cross(q1_vec);

  // Then multiply by q2^{-1} on the left: q2^{-1} * [dq1_w, dq1_vec]
  // Since q2^{-1} = [q2.w, -q2.vec]:
  const Vector3<T> q2_vec(q2.x(), q2.y(), q2.z());
  const T dqRel_w = q2.w() * dq1_w - (-q2_vec).dot(dq1_vec);
  const Vector3<T> dqRel_vec = q2.w() * dq1_vec + dq1_w * (-q2_vec) + (-q2_vec).cross(dq1_vec);

  // Apply chain rule: d(log)/d(t) = d(log)/d(q_rel) * d(q_rel)/d(t)
  // q_rel.coeffs() = [x, y, z, w], so we need to arrange as [dqRel_vec, dqRel_w]
  Eigen::Vector4<T> dqRelDt;
  dqRelDt << dqRel_vec, dqRel_w;
  return dLogDq * dqRelDt;
}

/// Computes rotation gradient using quaternion logarithmic map for a single rotation axis.
/// Returns the gradient value: d(||log(target^{-1} * rot)||^2)/d(rotation_axis)
template <typename T>
T computeLogMapRotationGradient(
    const Quaternion<T>& rot,
    const Quaternion<T>& target,
    const Vector3<T>& axis,
    const Eigen::Matrix<T, 3, 4>& dLogDq) {
  // Compute the relative rotation: q_rel = target^{-1} * rot
  const Quaternion<T> qRel = target.conjugate() * rot;

  // Compute the logmap of the relative rotation
  const Vector3<T> logmapVec = quaternionLogMap(qRel);

  // Compute the derivative using quaternionLogMapRelativeDerivativeQ1
  const Vector3<T> jacobian1 = quaternionLogMapRelativeDerivativeQ1(rot, target, axis, dLogDq);

  // Apply chain rule for squared norm: d(||v||^2)/dparams = 2 * v^T * d(v)/dparams
  return T(2) * logmapVec.dot(jacobian1);
}

/// Computes rotation gradient using rotation matrix difference for a single rotation axis.
/// Returns the gradient value: d(||R_rot - R_target||_F^2)/d(rotation_axis)
template <typename T>
T computeMatrixDiffRotationGradient(
    const Quaternion<T>& rot,
    const Quaternion<T>& target,
    const Vector3<T>& axis) {
  const auto rotD = crossProductMatrix(axis) * rot;
  const Eigen::Matrix3<T> rotDiff = rot.toRotationMatrix() - target.toRotationMatrix();
  return T(2) * rotD.cwiseProduct(rotDiff).sum();
}

/// Computes rotation Jacobian using quaternion logarithmic map for a single rotation axis.
/// Returns the Jacobian as a 3D vector.
template <typename T>
Eigen::Matrix<T, 3, 1> computeLogMapRotationJacobian(
    const Quaternion<T>& rot,
    const Quaternion<T>& target,
    const Vector3<T>& axis,
    T weight,
    const Eigen::Matrix<T, 3, 4>& dLogDq) {
  // Compute the derivative using quaternionLogMapRelativeDerivativeQ1
  return weight * quaternionLogMapRelativeDerivativeQ1(rot, target, axis, dLogDq);
}

/// Computes rotation Jacobian using rotation matrix difference for a single rotation axis.
/// Returns the Jacobian as a 9D vector.
template <typename T>
Eigen::Matrix<T, 9, 1>
computeMatrixDiffRotationJacobian(const Quaternion<T>& rot, const Vector3<T>& axis, T weight) {
  const Eigen::Matrix3<T> rotD = crossProductMatrix(axis) * rot * weight;
  return Map<const Eigen::Matrix<T, 9, 1>>(rotD.data(), rotD.size());
}

} // namespace

template <typename T>
StateErrorFunctionT<T>::StateErrorFunctionT(
    const Skeleton& skel,
    const ParameterTransform& pt,
    RotationErrorType rotationErrorType)
    : SkeletonErrorFunctionT<T>(skel, pt), rotationErrorType_(rotationErrorType) {
  targetPositionWeights_.setOnes(this->skeleton_.joints.size());
  targetRotationWeights_.setOnes(this->skeleton_.joints.size());
  posWgt_ = T(1);
  rotWgt_ = T(1);
}

template <typename T>
StateErrorFunctionT<T>::StateErrorFunctionT(
    const Character& character,
    RotationErrorType rotationErrorType)
    : StateErrorFunctionT<T>(character.skeleton, character.parameterTransform, rotationErrorType) {}

template <typename T>
void StateErrorFunctionT<T>::reset() {
  targetPositionWeights_.setOnes(this->skeleton_.joints.size());
  targetRotationWeights_.setOnes(this->skeleton_.joints.size());
}

template <typename T>
void StateErrorFunctionT<T>::setTargetState(const SkeletonStateT<T>* target) {
  MT_CHECK(target != nullptr);
  setTargetState(*target);
}

template <typename T>
void StateErrorFunctionT<T>::setTargetState(const SkeletonStateT<T>& target) {
  MT_CHECK(target.jointState.size() == this->skeleton_.joints.size());
  setTargetState(target.toTransforms());
}

template <typename T>
void StateErrorFunctionT<T>::setTargetState(TransformListT<T> target) {
  MT_CHECK(target.size() == this->skeleton_.joints.size());
  this->targetState_ = std::move(target);
}

template <typename T>
void StateErrorFunctionT<T>::setTargetWeight(const Eigen::VectorX<T>& weights) {
  MT_CHECK(weights.size() == static_cast<Eigen::Index>(this->skeleton_.joints.size()));

  targetPositionWeights_ = weights;
  targetRotationWeights_ = weights;
}

template <typename T>
void StateErrorFunctionT<T>::setTargetWeights(
    const Eigen::VectorX<T>& posWeight,
    const Eigen::VectorX<T>& rotWeight) {
  MT_CHECK(posWeight.size() == static_cast<Eigen::Index>(this->skeleton_.joints.size()));
  MT_CHECK(rotWeight.size() == static_cast<Eigen::Index>(this->skeleton_.joints.size()));

  targetPositionWeights_ = posWeight;
  targetRotationWeights_ = rotWeight;
}

template <typename T>
void StateErrorFunctionT<T>::setPositionTargetWeights(const Eigen::VectorX<T>& posWeight) {
  MT_CHECK(posWeight.size() == static_cast<Eigen::Index>(this->skeleton_.joints.size()));
  targetPositionWeights_ = posWeight;
}

template <typename T>
void StateErrorFunctionT<T>::setRotationTargetWeights(const Eigen::VectorX<T>& rotWeight) {
  MT_CHECK(rotWeight.size() == static_cast<Eigen::Index>(this->skeleton_.joints.size()));
  targetRotationWeights_ = rotWeight;
}

template <typename T>
double StateErrorFunctionT<T>::getError(
    const ModelParametersT<T>& /* parameters */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */) {
  MT_PROFILE_FUNCTION();

  if (state.jointState.empty()) {
    return 0.0;
  }

  // check all is valid
  MT_CHECK(
      state.jointParameters.size() ==
      gsl::narrow<Eigen::Index>(this->skeleton_.joints.size() * kParametersPerJoint));

  // loop over all joints and check for smoothness
  double error = 0.0;

  // ignore if we don't have any reasonable data
  if (targetState_.size() != state.jointState.size()) {
    return error;
  }

  // go over the joint states and calculate the rotation difference contribution on joints that are
  // at the end
  for (size_t i = 0; i < this->skeleton_.joints.size(); ++i) {
    // calculate orientation error based on the selected error type
    T rotationError = T(0);

    const Eigen::Quaternion<T>& target = targetState_[i].rotation.normalized();
    const Eigen::Quaternion<T>& rot = state.jointState[i].rotation();

    switch (rotationErrorType_) {
      case RotationErrorType::QuaternionLogMap: {
        // Compute the relative rotation: q_rel = target^{-1} * rot
        const Eigen::Quaternion<T> qRel = target.conjugate() * rot;

        // Compute the logmap of the relative rotation
        const Eigen::Vector3<T> logmapVec = quaternionLogMap(qRel);

        // The error is the squared norm of the logmap
        rotationError = logmapVec.squaredNorm();
        break;
      }
      case RotationErrorType::RotationMatrixDifference: {
        // Note: |R0 - RT|² is a valid norm on SO3, it doesn't have the same slope as the squared
        // angle difference but its derivative doesn't have a singularity at the minimum, so is more
        // stable
        const Eigen::Matrix3<T> rotDiff = rot.toRotationMatrix() - target.toRotationMatrix();
        rotationError = rotDiff.squaredNorm();
        break;
      }
    }

    error += rotationError * kOrientationWeight * rotWgt_ * targetRotationWeights_[i];

    // calculate position error
    const Eigen::Vector3<T> diff = state.jointState[i].translation() - targetState_[i].translation;
    error += diff.squaredNorm() * kPositionWeight * posWgt_ * targetPositionWeights_[i];
  }

  // return error
  return error * this->weight_;
}

template <typename T>
double StateErrorFunctionT<T>::getGradient(
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */,
    Ref<Eigen::VectorX<T>> gradient) {
  MT_PROFILE_FUNCTION();

  if (state.jointState.empty()) {
    return 0.0;
  }

  // loop over all joints and check for smoothness
  double error = 0.0;

  // check all is valid
  MT_CHECK(
      state.jointParameters.size() ==
      gsl::narrow<Eigen::Index>(this->skeleton_.joints.size() * kParametersPerJoint));

  // ignore if we don't have any reasonable data
  if (targetState_.size() != state.jointState.size()) {
    return error;
  }

  // also go over the joint states and calculate the rotation difference contribution on joints that
  // are at the end
  for (size_t i = 0; i < this->skeleton_.joints.size(); ++i) {
    if (targetRotationWeights_[i] == 0 && targetPositionWeights_[i] == 0) {
      continue;
    }

    // calculate orientation gradient
    const Eigen::Quaternion<T>& target = targetState_[i].rotation.normalized();
    const Eigen::Quaternion<T>& rot = state.jointState[i].rotation();

    T rotationError = T(0);
    Eigen::Matrix<T, 3, 4> dLogDq; // Precomputed derivative (only needed for logmap)

    switch (rotationErrorType_) {
      case RotationErrorType::QuaternionLogMap: {
        // Compute the relative rotation: q_rel = target^{-1} * rot
        const Eigen::Quaternion<T> qRel = target.conjugate() * rot;

        // Precompute the derivative of logmap w.r.t. quaternion components
        // This will be reused for all 3 rotation axes
        dLogDq = quaternionLogMapDerivative(qRel);

        // Compute the logmap of the relative rotation
        const Eigen::Vector3<T> logmapVec = quaternionLogMap(qRel);

        // The error is the squared norm of the logmap
        rotationError = logmapVec.squaredNorm();
        break;
      }
      case RotationErrorType::RotationMatrixDifference: {
        const Eigen::Matrix3<T> rotDiff = rot.toRotationMatrix() - target.toRotationMatrix();
        rotationError = rotDiff.squaredNorm();
        break;
      }
    }

    const T rwgt = kOrientationWeight * rotWgt_ * this->weight_ * targetRotationWeights_[i];
    error += rotationError * rwgt;

    // calculate position gradient
    const Eigen::Vector3<T> diff = state.jointState[i].translation() - targetState_[i].translation;
    const T pwgt = kPositionWeight * posWgt_ * this->weight_ * targetPositionWeights_[i];
    error += diff.squaredNorm() * pwgt;

    // loop over all joints the constraint is attached to and calculate gradient
    size_t jointIndex = i;
    while (jointIndex != kInvalidIndex) {
      // check for valid index
      MT_CHECK(jointIndex < this->skeleton_.joints.size());

      const size_t paramIndex = jointIndex * kParametersPerJoint;

      // precalculate some more data for position gradient
      const Eigen::Vector3<T> posd =
          state.jointState[i].translation() - state.jointState[jointIndex].translation();

      for (size_t d = 0; d < 3; d++) {
        // position gradient
        if (this->activeJointParams_[paramIndex + d]) {
          // calculate joint gradient
          const T val =
              T(2) * diff.dot(state.jointState[jointIndex].getTranslationDerivative(d)) * pwgt;
          gradient_jointParams_to_modelParams(
              val, paramIndex + d, this->parameterTransform_, gradient);
        }
        if (this->activeJointParams_[paramIndex + 3 + d]) {
          // calculate joint gradient consisting of position gradient and orientation gradient
          const Eigen::Vector3<T> axis = state.jointState[jointIndex].rotationAxis.col(d);

          // Compute rotation gradient based on error type
          T rotGradient;
          switch (rotationErrorType_) {
            case RotationErrorType::QuaternionLogMap:
              rotGradient = computeLogMapRotationGradient(rot, target, axis, dLogDq);
              break;
            case RotationErrorType::RotationMatrixDifference:
              rotGradient = computeMatrixDiffRotationGradient(rot, target, axis);
              break;
          }

          const T val =
              T(2) * diff.dot(state.jointState[jointIndex].getRotationDerivative(d, posd)) * pwgt +
              rwgt * rotGradient;
          gradient_jointParams_to_modelParams(
              val, paramIndex + 3 + d, this->parameterTransform_, gradient);
        }
      }
      if (this->activeJointParams_[paramIndex + 6]) {
        // calculate joint gradient
        const T val = T(2) * diff.dot(state.jointState[jointIndex].getScaleDerivative(posd)) * pwgt;
        gradient_jointParams_to_modelParams(
            val, paramIndex + 6, this->parameterTransform_, gradient);
      }

      // go to the next joint
      jointIndex = this->skeleton_.joints[jointIndex].parent;
    }
  }

  // return error
  return error;
}

template <typename T>
size_t StateErrorFunctionT<T>::getJacobianSize() const {
  const auto nActiveJoints =
      (targetPositionWeights_.array() != 0 || targetRotationWeights_.array() != 0).count();
  if (rotationErrorType_ == RotationErrorType::QuaternionLogMap) {
    // 3 (translation) + 3 (logmap) = 6 per joint
    return nActiveJoints * 6;
  } else {
    // 3 (translation) + 9 (matrix) = 12 per joint
    return nActiveJoints * 12;
  }
}

template <typename T>
double StateErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& /* parameters */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    int& usedRows) {
  MT_PROFILE_FUNCTION();

  // loop over all constraints and calculate the error
  double error = 0.0;

  // ignore if we don't have any reasonable data
  if (targetState_.size() != state.jointState.size()) {
    return error;
  }

  Eigen::Index offset = 0;

  // calculate state difference jacobians
  for (size_t i = 0; i < this->skeleton_.joints.size(); ++i) {
    if (targetRotationWeights_[i] == 0 && targetPositionWeights_[i] == 0) {
      continue;
    }

    // Determine block size based on rotation error type
    // QuaternionLogMap: 3 (translation) + 3 (logmap) = 6
    // RotationMatrixDifference: 3 (translation) + 9 (matrix) = 12
    const size_t rotationResidualSize =
        (rotationErrorType_ == RotationErrorType::QuaternionLogMap) ? 3 : 9;
    const size_t blockSize = 3 + rotationResidualSize;

    // calculate translation gradient
    const auto transDiff = (state.jointState[i].translation() - targetState_[i].translation).eval();

    // calculate orientation gradient and error
    const Eigen::Quaternion<T>& target = targetState_[i].rotation.normalized();
    const Eigen::Quaternion<T>& rot = state.jointState[i].rotation();

    // calculate the difference between target and position and error
    const T pwgt = kPositionWeight * posWgt_ * this->weight_ * targetPositionWeights_[i];
    const T rwgt = kOrientationWeight * rotWgt_ * this->weight_ * targetRotationWeights_[i];
    const T wgt = std::sqrt(pwgt);
    const T awgt = std::sqrt(rwgt);
    error += transDiff.squaredNorm() * pwgt;
    residual.template segment<3>(offset).noalias() = transDiff * wgt;

    Eigen::Matrix<T, 3, 4> dLogDq; // Precomputed derivative (only needed for logmap)
    switch (rotationErrorType_) {
      case RotationErrorType::QuaternionLogMap: {
        // Compute the relative rotation: q_rel = target^{-1} * rot
        const Eigen::Quaternion<T> qRel = target.conjugate() * rot;

        // Precompute the derivative of logmap w.r.t. quaternion components
        // This will be reused for all 3 rotation axes
        dLogDq = quaternionLogMapDerivative(qRel);

        // Compute the logmap of the relative rotation
        const Eigen::Vector3<T> logmapVec = quaternionLogMap(qRel);

        // The error is the squared norm of the logmap
        error += logmapVec.squaredNorm() * rwgt;

        // The residual for Jacobian is the logmap vector itself (not squared)
        // because we'll compute d(||log(q)||^2)/dparams = 2 * log(q)^T * d(log(q))/dparams
        residual.template segment<3>(offset + 3).noalias() = logmapVec * awgt;
        break;
      }
      case RotationErrorType::RotationMatrixDifference: {
        // 9D RotationMatrixDifference
        const Eigen::Matrix3<T> rotDiff = rot.toRotationMatrix() - target.toRotationMatrix();
        error += rotDiff.squaredNorm() * rwgt;
        residual.template segment<9>(offset + 3).noalias() =
            Map<const Eigen::Matrix<T, 9, 1>>(rotDiff.data()) * awgt;
        break;
      }
    }

    // loop over all joints the constraint is attached to and calculate jacobian
    size_t jointIndex = i;
    while (jointIndex != kInvalidIndex) {
      // check for valid index
      MT_CHECK(jointIndex < this->skeleton_.joints.size());

      const auto& jointState = state.jointState[jointIndex];
      const size_t paramIndex = jointIndex * kParametersPerJoint;

      // precalculate some more data for position gradient
      const Eigen::Vector3<T> posd = state.jointState[i].translation() - jointState.translation();

      // calculate derivatives based on active joints
      for (size_t d = 0; d < 3; d++) {
        if (this->activeJointParams_[paramIndex + d]) {
          jacobian_jointParams_to_modelParams<T>(
              jointState.getTranslationDerivative(d) * wgt,
              paramIndex + d,
              this->parameterTransform_,
              jacobian.template middleRows<3>(offset));
        }

        if (this->activeJointParams_[paramIndex + 3 + d]) {
          // Translation part of rotation parameter jacobian
          const Eigen::Vector3<T> jc = jointState.getRotationDerivative(d, posd) * wgt;
          jacobian_jointParams_to_modelParams<T>(
              jc,
              paramIndex + d + 3,
              this->parameterTransform_,
              jacobian.template middleRows<3>(offset));

          // The rotation direction (axis)
          const Eigen::Vector3<T> axis = state.jointState[jointIndex].rotationAxis.col(d);

          // Rotation part of rotation parameter jacobian
          switch (rotationErrorType_) {
            case RotationErrorType::QuaternionLogMap:
              jacobian_jointParams_to_modelParams<T>(
                  computeLogMapRotationJacobian(rot, target, axis, awgt, dLogDq),
                  paramIndex + d + 3,
                  this->parameterTransform_,
                  jacobian.middleRows(offset + 3, rotationResidualSize));
              break;
            case RotationErrorType::RotationMatrixDifference:
              jacobian_jointParams_to_modelParams<T>(
                  computeMatrixDiffRotationJacobian(rot, axis, awgt),
                  paramIndex + d + 3,
                  this->parameterTransform_,
                  jacobian.middleRows(offset + 3, rotationResidualSize));
              break;
          }
        }
      }

      if (this->activeJointParams_[paramIndex + 6]) {
        // Scale implicitly affects translation
        const Eigen::Vector3<T> jc = jointState.getScaleDerivative(posd) * wgt;
        jacobian_jointParams_to_modelParams<T>(
            jc, paramIndex + 6, this->parameterTransform_, jacobian.template middleRows<3>(offset));
      }

      // go to the next joint
      jointIndex = this->skeleton_.joints[jointIndex].parent;
    }

    // Move to next block
    offset += blockSize;
  }

  usedRows = gsl::narrow_cast<int>(offset);

  // return error
  return error;
}

template class StateErrorFunctionT<float>;
template class StateErrorFunctionT<double>;

} // namespace momentum
