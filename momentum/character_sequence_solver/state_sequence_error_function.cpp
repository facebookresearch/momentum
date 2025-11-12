/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_sequence_solver/state_sequence_error_function.h"

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
  const Vector3<T> vHalf = dir / T(2);

  // Compute d/dt [exp(t*v) * q1]|_{t=0} = [0, v/2] * q1
  const Vector3<T> q1_vec(q1.x(), q1.y(), q1.z());
  const T dq1_w = -vHalf.dot(q1_vec);
  const Vector3<T> dq1_vec = vHalf * q1.w() + vHalf.cross(q1_vec);

  // Then multiply by q2^{-1} on the left: q2^{-1} * [dq1_w, dq1_vec]
  const Vector3<T> q2_vec(q2.x(), q2.y(), q2.z());
  const T dqRel_w = q2.w() * dq1_w - (-q2_vec).dot(dq1_vec);
  const Vector3<T> dqRel_vec = q2.w() * dq1_vec + dq1_w * (-q2_vec) + (-q2_vec).cross(dq1_vec);

  // Apply chain rule: d(log)/d(t) = d(log)/d(q_rel) * d(q_rel)/d(t)
  Eigen::Vector4<T> dqRelDt;
  dqRelDt << dqRel_vec, dqRel_w;
  return dLogDq * dqRelDt;
}

/// Computes the derivative of the logarithmic map of a relative rotation for q2 perturbation.
template <typename T>
Vector3<T> quaternionLogMapRelativeDerivativeQ2(
    const Quaternion<T>& q1,
    const Quaternion<T>& q2,
    const Vector3<T>& dir,
    const Eigen::Matrix<T, 3, 4>& dLogDq) {
  const Vector3<T> vHalf = dir / T(2);

  // Compute q2^{-1} * [0, v/2]
  const Vector3<T> q2_vec(q2.x(), q2.y(), q2.z());
  const Vector3<T> q2inv_vec = -q2_vec;
  const T temp_w = -q2inv_vec.dot(vHalf);
  const Vector3<T> temp_vec = q2.w() * vHalf + q2inv_vec.cross(vHalf);

  // Then multiply by q1 on the right: [temp_w, temp_vec] * q1
  const Vector3<T> q1_vec(q1.x(), q1.y(), q1.z());
  const T dqRel_w = temp_w * q1.w() - temp_vec.dot(q1_vec);
  const Vector3<T> dqRel_vec = temp_w * q1_vec + q1.w() * temp_vec + temp_vec.cross(q1_vec);

  // Apply chain rule with negative sign from derivative of inverse
  Eigen::Vector4<T> dqRelDt;
  dqRelDt << -dqRel_vec, -dqRel_w;
  return dLogDq * dqRelDt;
}

/// Computes rotation gradient using quaternion logarithmic map for prevRot
/// @param prevRot The rotation of joint i in the previous frame (untransformed)
/// @param nextRot The rotation of joint i in the next frame
/// @param targetRot The target transform rotation to apply to prevRot
/// @param axis The rotation axis (from the perturbed joint, in its local frame)
/// @param dLogDq Precomputed logmap derivative (for nextRot^-1 * targetRot * prevRot)
///
/// Note: We compute the derivative of log(nextRot^{-1} * targetRot * prevRot) w.r.t. prevRot.
/// When prevRot is perturbed as exp(t*axis) * prevRot, we get:
///   targetRot * exp(t*axis) * prevRot = exp(t*(targetRot*axis)) * (targetRot * prevRot)
/// by the conjugation property: R * exp(t*v) = exp(t*(R*v)) * R.
/// This justifies passing (targetRot*prevRot, targetRot*axis) to
/// quaternionLogMapRelativeDerivativeQ1.
template <typename T>
T computeLogMapRotationGradientPrev(
    const Quaternion<T>& prevRot,
    const Quaternion<T>& nextRot,
    const Quaternion<T>& targetRot,
    const Vector3<T>& axis,
    const Eigen::Matrix<T, 3, 4>& dLogDq) {
  const Quaternion<T> transformedPrevRot = targetRot * prevRot;
  const Quaternion<T> qRel = nextRot.conjugate() * transformedPrevRot;
  const Vector3<T> logmapVec = quaternionLogMap(qRel);
  const Vector3<T> transformedAxis = targetRot * axis;
  const Vector3<T> jacobian =
      quaternionLogMapRelativeDerivativeQ1(transformedPrevRot, nextRot, transformedAxis, dLogDq);
  return T(2) * logmapVec.dot(jacobian);
}

/// Computes rotation gradient using quaternion logarithmic map for nextRot
/// @param transformedPrevRot The rotation of joint i after applying targetTransform (already
/// transformed)
/// @param nextRot The rotation of joint i in the next frame
/// @param axis The rotation axis (from the perturbed joint, in its local frame)
/// @param dLogDq Precomputed logmap derivative
template <typename T>
T computeLogMapRotationGradientNext(
    const Quaternion<T>& transformedPrevRot,
    const Quaternion<T>& nextRot,
    const Vector3<T>& axis,
    const Eigen::Matrix<T, 3, 4>& dLogDq) {
  const Quaternion<T> qRel = nextRot.conjugate() * transformedPrevRot;
  const Vector3<T> logmapVec = quaternionLogMap(qRel);
  const Vector3<T> jacobian =
      quaternionLogMapRelativeDerivativeQ2(transformedPrevRot, nextRot, axis, dLogDq);
  return T(2) * logmapVec.dot(jacobian);
}

/// Computes rotation gradient using rotation matrix difference for prevRot
/// Returns the signed gradient: d(||nextRot - targetRot * prevRot||^2) / d(prevRot along axis)
/// The negative sign is included because rotDiff = nextRot - targetRot * prevRot
template <typename T>
T computeMatrixDiffRotationGradientPrev(
    const Quaternion<T>& prevRot,
    const Quaternion<T>& targetRot,
    const Vector3<T>& axis,
    const Matrix3<T>& rotDiff) {
  const auto rotD = targetRot * crossProductMatrix(axis) * prevRot;
  return -T(2) * rotD.cwiseProduct(rotDiff).sum();
}

/// Computes rotation gradient using rotation matrix difference for nextRot
/// Returns the signed gradient: d(||nextRot - targetRot * prevRot||^2) / d(nextRot along axis)
template <typename T>
T computeMatrixDiffRotationGradientNext(
    const Quaternion<T>& nextRot,
    const Vector3<T>& axis,
    const Matrix3<T>& rotDiff) {
  const auto rotD = crossProductMatrix(axis) * nextRot;
  return T(2) * rotD.cwiseProduct(rotDiff).sum();
}

/// Computes rotation Jacobian using quaternion logarithmic map for prevRot
/// @param prevRot The rotation of joint i in the previous frame (untransformed)
/// @param nextRot The rotation of joint i in the next frame
/// @param targetRot The target transform rotation to apply to prevRot
/// @param axis The rotation axis (from the perturbed joint, in its local frame)
/// @param weight The weight to apply
/// @param dLogDq Precomputed logmap derivative (for nextRot^-1 * targetRot * prevRot)
template <typename T>
Eigen::Matrix<T, 3, 1> computeLogMapRotationJacobianPrev(
    const Quaternion<T>& prevRot,
    const Quaternion<T>& nextRot,
    const Quaternion<T>& targetRot,
    const Vector3<T>& axis,
    T weight,
    const Eigen::Matrix<T, 3, 4>& dLogDq) {
  const Quaternion<T> transformedPrevRot = targetRot * prevRot;
  const Vector3<T> transformedAxis = targetRot * axis;
  return weight *
      quaternionLogMapRelativeDerivativeQ1(transformedPrevRot, nextRot, transformedAxis, dLogDq);
}

/// Computes rotation Jacobian using quaternion logarithmic map for nextRot
/// @param transformedPrevRot The rotation of joint i after applying targetTransform (already
/// transformed)
/// @param nextRot The rotation of joint i in the next frame
/// @param axis The rotation axis (from the perturbed joint, in its local frame)
/// @param weight The weight to apply
/// @param dLogDq Precomputed logmap derivative
template <typename T>
Eigen::Matrix<T, 3, 1> computeLogMapRotationJacobianNext(
    const Quaternion<T>& transformedPrevRot,
    const Quaternion<T>& nextRot,
    const Vector3<T>& axis,
    T weight,
    const Eigen::Matrix<T, 3, 4>& dLogDq) {
  return weight * quaternionLogMapRelativeDerivativeQ2(transformedPrevRot, nextRot, axis, dLogDq);
}

/// Computes rotation Jacobian using rotation matrix difference for prevRot
/// Returns the signed Jacobian: d(||nextRot - targetRot * prevRot||^2) / d(prevRot along axis)
/// The negative sign is included because rotDiff = nextRot - targetRot * prevRot
template <typename T>
Eigen::Matrix<T, 9, 1> computeMatrixDiffRotationJacobianPrev(
    const Quaternion<T>& prevRot,
    const Quaternion<T>& targetRot,
    const Vector3<T>& axis,
    T weight) {
  const Eigen::Matrix3<T> rotD = -(targetRot * crossProductMatrix(axis) * prevRot) * weight;
  return Map<const Eigen::Matrix<T, 9, 1>>(rotD.data(), rotD.size());
}

/// Computes rotation Jacobian using rotation matrix difference for nextRot
template <typename T>
Eigen::Matrix<T, 9, 1>
computeMatrixDiffRotationJacobianNext(const Quaternion<T>& rot, const Vector3<T>& axis, T weight) {
  const Eigen::Matrix3<T> rotD = crossProductMatrix(axis) * rot * weight;
  return Map<const Eigen::Matrix<T, 9, 1>>(rotD.data(), rotD.size());
}

} // namespace

template <typename T>
StateSequenceErrorFunctionT<T>::StateSequenceErrorFunctionT(
    const Skeleton& skel,
    const ParameterTransform& pt,
    RotationErrorType rotationErrorType)
    : SequenceErrorFunctionT<T>(skel, pt), rotationErrorType_(rotationErrorType) {
  targetPositionWeights_.setOnes(this->skeleton_.joints.size());
  targetRotationWeights_.setOnes(this->skeleton_.joints.size());
  posWgt_ = T(1);
  rotWgt_ = T(1);
}

template <typename T>
StateSequenceErrorFunctionT<T>::StateSequenceErrorFunctionT(
    const Character& character,
    RotationErrorType rotationErrorType)
    : StateSequenceErrorFunctionT<T>(
          character.skeleton,
          character.parameterTransform,
          rotationErrorType) {}

template <typename T>
void StateSequenceErrorFunctionT<T>::reset() {
  targetPositionWeights_.setOnes(this->skeleton_.joints.size());
  targetRotationWeights_.setOnes(this->skeleton_.joints.size());
}

template <typename T>
void StateSequenceErrorFunctionT<T>::setTargetState(TransformListT<T> target) {
  MT_CHECK(target.size() == this->skeleton_.joints.size());
  targetState_ = target;
}

template <typename T>
void StateSequenceErrorFunctionT<T>::setTargetWeights(
    const Eigen::VectorX<T>& posWeight,
    const Eigen::VectorX<T>& rotWeight) {
  setPositionTargetWeights(posWeight);
  setRotationTargetWeights(rotWeight);
}

template <typename T>
void StateSequenceErrorFunctionT<T>::setPositionTargetWeights(const Eigen::VectorX<T>& posWeight) {
  MT_CHECK(posWeight.size() == static_cast<Eigen::Index>(this->skeleton_.joints.size()));
  targetPositionWeights_ = posWeight;
}

template <typename T>
void StateSequenceErrorFunctionT<T>::setRotationTargetWeights(const Eigen::VectorX<T>& rotWeight) {
  MT_CHECK(rotWeight.size() == static_cast<Eigen::Index>(this->skeleton_.joints.size()));
  targetRotationWeights_ = rotWeight;
}

template <typename T>
double StateSequenceErrorFunctionT<T>::getError(
    std::span<const ModelParametersT<T>> /* modelParameters */,
    std::span<const SkeletonStateT<T>> skelStates,
    std::span<const MeshStateT<T>> /* meshStates */) const {
  MT_PROFILE_FUNCTION();

  // loop over all joints and check for smoothness
  double error = 0.0;

  MT_CHECK(skelStates.size() == 2);

  const auto& prevState = skelStates[0];
  const auto& nextState = skelStates[1];

  // go over the joint states and calculate the rotation difference contribution on joints that are
  // at the end
  for (size_t i = 0; i < this->skeleton_.joints.size(); ++i) {
    // calculate orientation error based on the selected error type
    T rotationError = T(0);

    const TransformT<T> targetTransform =
        (i < targetState_.size()) ? targetState_[i] : TransformT<T>();
    const TransformT<T> transformedPrev = targetTransform * prevState.jointState[i].transform;
    const Eigen::Quaternion<T>& prevRot = transformedPrev.rotation;
    const Eigen::Quaternion<T>& nextRot = nextState.jointState[i].rotation();

    switch (rotationErrorType_) {
      case RotationErrorType::QuaternionLogMap: {
        // Compute the relative rotation: q_rel = nextRot^{-1} * prevRot
        const Eigen::Quaternion<T> qRel = nextRot.conjugate() * prevRot;

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
        const Eigen::Matrix3<T> rotDiff = nextRot.toRotationMatrix() - prevRot.toRotationMatrix();
        rotationError = rotDiff.squaredNorm();
        break;
      }
    }

    error += rotationError * kOrientationWeight * rotWgt_ * targetRotationWeights_[i];

    // calculate position error
    const Eigen::Vector3<T> diff =
        nextState.jointState[i].translation() - transformedPrev.translation;
    error += diff.squaredNorm() * kPositionWeight * posWgt_ * targetPositionWeights_[i];
  }

  // return error
  return error * this->weight_;
}

template <typename T>
double StateSequenceErrorFunctionT<T>::getGradient(
    std::span<const ModelParametersT<T>> /* modelParameters */,
    std::span<const SkeletonStateT<T>> skelStates,
    std::span<const MeshStateT<T>> /* meshStates */,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  MT_PROFILE_FUNCTION();

  MT_CHECK(skelStates.size() == 2);
  const auto& prevState = skelStates[0];
  const auto& nextState = skelStates[1];

  const Eigen::Index nParam = this->parameterTransform_.numAllModelParameters();
  MT_CHECK(gradient.size() == 2 * nParam);

  // loop over all joints and check for smoothness
  double error = 0.0;

  // also go over the joint states and calculate the rotation difference contribution on joints that
  // are at the end
  for (size_t i = 0; i < this->skeleton_.joints.size(); ++i) {
    if (targetRotationWeights_[i] == 0 && targetPositionWeights_[i] == 0) {
      continue;
    }

    // calculate orientation gradient
    const TransformT<T> targetTransform =
        (i < targetState_.size()) ? targetState_[i] : TransformT<T>();
    const TransformT<T> transformedPrev = targetTransform * prevState.jointState[i].transform;
    const Eigen::Quaternion<T>& prevRot = transformedPrev.rotation;
    const Eigen::Quaternion<T>& nextRot = nextState.jointState[i].rotation();

    T rotationError = T(0);
    Eigen::Matrix3<T> rotDiff; // Only needed for RotationMatrixDifference
    Eigen::Matrix<T, 3, 4> dLogDq; // Only needed for QuaternionLogMap

    switch (rotationErrorType_) {
      case RotationErrorType::QuaternionLogMap: {
        // Compute the relative rotation: q_rel = nextRot^{-1} * prevRot
        const Eigen::Quaternion<T> qRel = nextRot.conjugate() * prevRot;

        // Precompute the derivative of logmap w.r.t. quaternion components
        dLogDq = quaternionLogMapDerivative(qRel);

        // Compute the logmap of the relative rotation
        const Eigen::Vector3<T> logmapVec = quaternionLogMap(qRel);

        // The error is the squared norm of the logmap
        rotationError = logmapVec.squaredNorm();
        break;
      }
      case RotationErrorType::RotationMatrixDifference: {
        rotDiff = nextRot.toRotationMatrix() - prevRot.toRotationMatrix();
        rotationError = rotDiff.squaredNorm();
        break;
      }
    }

    const T rwgt = kOrientationWeight * rotWgt_ * this->weight_ * targetRotationWeights_[i];
    error += rotationError * rwgt;

    // calculate position gradient
    const Eigen::Vector3<T> diff =
        nextState.jointState[i].translation() - transformedPrev.translation;
    const T pwgt = kPositionWeight * posWgt_ * this->weight_ * targetPositionWeights_[i];
    error += diff.squaredNorm() * pwgt;
    // Gradient for prevState
    {
      size_t ancestorJointIndex = i;
      while (ancestorJointIndex != kInvalidIndex) {
        MT_CHECK(ancestorJointIndex < this->skeleton_.joints.size());

        const size_t paramIndex = ancestorJointIndex * kParametersPerJoint;
        const Eigen::Vector3<T> posd = prevState.jointState[i].translation() -
            prevState.jointState[ancestorJointIndex].translation();

        for (size_t d = 0; d < 3; d++) {
          // position gradient
          if (this->activeJointParams_[paramIndex + d]) {
            const T val = -T(2) *
                diff.dot(targetTransform.rotate(
                    prevState.jointState[ancestorJointIndex].getTranslationDerivative(d))) *
                pwgt;
            gradient_jointParams_to_modelParams<T>(
                val, paramIndex + d, this->parameterTransform_, gradient.segment(0, nParam));
          }
          if (this->activeJointParams_[paramIndex + 3 + d]) {
            const Eigen::Vector3<T> axis =
                prevState.jointState[ancestorJointIndex].rotationAxis.col(d);

            // (1) Changes in position error due to rotation of ancestor joint
            const T positionErrorGradient = -T(2) *
                diff.dot(targetTransform.rotate(
                    prevState.jointState[ancestorJointIndex].getRotationDerivative(d, posd)));

            // (2) Changes in rotation error due to rotation of ancestor joint
            T rotationErrorGradient = 0;
            switch (rotationErrorType_) {
              case RotationErrorType::QuaternionLogMap:
                rotationErrorGradient = computeLogMapRotationGradientPrev(
                    prevState.jointState[i].rotation(),
                    nextRot,
                    targetTransform.rotation,
                    axis,
                    dLogDq);
                break;
              case RotationErrorType::RotationMatrixDifference:
                rotationErrorGradient = computeMatrixDiffRotationGradientPrev(
                    prevState.jointState[i].rotation(), targetTransform.rotation, axis, rotDiff);
                break;
            }

            const T val = pwgt * positionErrorGradient + rwgt * rotationErrorGradient;
            gradient_jointParams_to_modelParams<T>(
                val, paramIndex + 3 + d, this->parameterTransform_, gradient.segment(0, nParam));
          }
        }
        if (this->activeJointParams_[paramIndex + 6]) {
          const T val = -T(2) *
              diff.dot(targetTransform.rotate(
                  prevState.jointState[ancestorJointIndex].getScaleDerivative(posd))) *
              pwgt;
          gradient_jointParams_to_modelParams<T>(
              val, paramIndex + 6, this->parameterTransform_, gradient.segment(0, nParam));
        }

        ancestorJointIndex = this->skeleton_.joints[ancestorJointIndex].parent;
      }
    }

    // Gradient for nextState
    {
      size_t ancestorJointIndex = i;
      while (ancestorJointIndex != kInvalidIndex) {
        MT_CHECK(ancestorJointIndex < this->skeleton_.joints.size());

        const size_t paramIndex = ancestorJointIndex * kParametersPerJoint;
        const Eigen::Vector3<T> posd = nextState.jointState[i].translation() -
            nextState.jointState[ancestorJointIndex].translation();

        for (size_t d = 0; d < 3; d++) {
          // position gradient
          if (this->activeJointParams_[paramIndex + d]) {
            const T val = T(2) *
                diff.dot(nextState.jointState[ancestorJointIndex].getTranslationDerivative(d)) *
                pwgt;
            gradient_jointParams_to_modelParams<T>(
                val, paramIndex + d, this->parameterTransform_, gradient.segment(nParam, nParam));
          }
          if (this->activeJointParams_[paramIndex + 3 + d]) {
            const Eigen::Vector3<T> axis =
                nextState.jointState[ancestorJointIndex].rotationAxis.col(d);

            // (1) Changes in position error due to rotation of ancestor joint
            const T positionErrorGradient = T(2) *
                diff.dot(nextState.jointState[ancestorJointIndex].getRotationDerivative(d, posd)) *
                pwgt;

            // (2) Changes in rotation error due to rotation of ancestor joint
            T rotationErrorGradient;
            switch (rotationErrorType_) {
              case RotationErrorType::QuaternionLogMap:
                rotationErrorGradient =
                    computeLogMapRotationGradientNext(prevRot, nextRot, axis, dLogDq);
                break;
              case RotationErrorType::RotationMatrixDifference:
                rotationErrorGradient = computeMatrixDiffRotationGradientNext(
                    nextState.jointState[i].rotation(), axis, rotDiff);
                break;
            }

            const T val = positionErrorGradient + rwgt * rotationErrorGradient;
            gradient_jointParams_to_modelParams<T>(
                val,
                paramIndex + 3 + d,
                this->parameterTransform_,
                gradient.segment(nParam, nParam));
          }
        }
        if (this->activeJointParams_[paramIndex + 6]) {
          const T val = T(2) *
              diff.dot(nextState.jointState[ancestorJointIndex].getScaleDerivative(posd)) * pwgt;
          gradient_jointParams_to_modelParams<T>(
              val, paramIndex + 6, this->parameterTransform_, gradient.segment(nParam, nParam));
        }

        ancestorJointIndex = this->skeleton_.joints[ancestorJointIndex].parent;
      }
    }
  }

  // return error
  return error;
}

template <typename T>
size_t StateSequenceErrorFunctionT<T>::getJacobianSize() const {
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
double StateSequenceErrorFunctionT<T>::getJacobian(
    std::span<const ModelParametersT<T>> /* modelParameters */,
    std::span<const SkeletonStateT<T>> skelStates,
    std::span<const MeshStateT<T>> /* meshStates */,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian_full,
    Eigen::Ref<Eigen::VectorX<T>> residual_full,
    int& usedRows) const {
  MT_PROFILE_FUNCTION();

  MT_CHECK(skelStates.size() == 2);
  const auto& prevState = skelStates[0];
  const auto& nextState = skelStates[1];

  const Eigen::Index nParam = this->parameterTransform_.numAllModelParameters();
  MT_CHECK(jacobian_full.cols() == 2 * nParam);

  double error = 0.0;

  Eigen::Index offset = 0;

  // calculate state difference jacobians
  for (size_t i = 0; i < this->skeleton_.joints.size(); ++i) {
    if (targetRotationWeights_[i] == 0 && targetPositionWeights_[i] == 0) {
      continue;
    }

    const TransformT<T> targetTransform =
        (i < targetState_.size()) ? targetState_[i] : TransformT<T>();
    const TransformT<T> transformedPrev = targetTransform * prevState.jointState[i].transform;

    // Determine block size based on rotation error type
    // QuaternionLogMap: 3 (translation) + 3 (logmap) = 6
    // RotationMatrixDifference: 3 (translation) + 9 (matrix) = 12
    const size_t rotationResidualSize =
        (rotationErrorType_ == RotationErrorType::QuaternionLogMap) ? 3 : 9;
    const size_t blockSize = 3 + rotationResidualSize;

    // calculate translation gradient
    const auto transDiff =
        (nextState.jointState[i].translation() - transformedPrev.translation).eval();

    // calculate orientation gradient and error
    const Eigen::Quaternion<T>& prevRot = transformedPrev.rotation;
    const Eigen::Quaternion<T>& nextRot = nextState.jointState[i].rotation();

    // calculate the difference between target and position and error
    const T pwgt = kPositionWeight * posWgt_ * this->weight_ * targetPositionWeights_[i];
    const T wgt = std::sqrt(pwgt);
    error += transDiff.squaredNorm() * pwgt;
    residual_full.template segment<3>(offset).noalias() = transDiff * wgt;

    const T rwgt = kOrientationWeight * rotWgt_ * this->weight_ * targetRotationWeights_[i];
    const T awgt = std::sqrt(rwgt);

    Eigen::Matrix<T, 3, 4> dLogDq; // Precomputed derivative (only needed for logmap)
    switch (rotationErrorType_) {
      case RotationErrorType::QuaternionLogMap: {
        // Compute the relative rotation: q_rel = nextRot^{-1} * prevRot
        const Eigen::Quaternion<T> qRel = nextRot.conjugate() * prevRot;

        // Precompute the derivative of logmap w.r.t. quaternion components
        // This will be reused for all 3 rotation axes
        dLogDq = quaternionLogMapDerivative(qRel);

        // Compute the logmap of the relative rotation
        const Eigen::Vector3<T> logmapVec = quaternionLogMap(qRel);

        // The error is the squared norm of the logmap
        error += logmapVec.squaredNorm() * rwgt;

        // The residual for Jacobian is the logmap vector itself (not squared)
        residual_full.template segment<3>(offset + 3).noalias() = logmapVec * awgt;
        break;
      }
      case RotationErrorType::RotationMatrixDifference: {
        // 9D RotationMatrixDifference
        const Eigen::Matrix3<T> rotDiff = nextRot.toRotationMatrix() - prevRot.toRotationMatrix();
        error += rotDiff.squaredNorm() * rwgt;
        residual_full.template segment<9>(offset + 3).noalias() =
            Map<const Eigen::Matrix<T, 9, 1>>(rotDiff.data(), rotDiff.size()) * awgt;
        break;
      }
    }

    // Jacobian for prevState
    {
      size_t ancestorJointIndex = i;
      while (ancestorJointIndex != kInvalidIndex) {
        MT_CHECK(ancestorJointIndex < this->skeleton_.joints.size());

        const auto& jointState = prevState.jointState[ancestorJointIndex];
        const size_t paramIndex = ancestorJointIndex * kParametersPerJoint;
        const Eigen::Vector3<T> posd =
            prevState.jointState[i].translation() - jointState.translation();

        for (size_t d = 0; d < 3; d++) {
          if (this->activeJointParams_[paramIndex + d]) {
            const Eigen::Vector3<T> jc =
                -(targetTransform.rotate(jointState.getTranslationDerivative(d))) * wgt;
            jacobian_jointParams_to_modelParams<T>(
                jc,
                paramIndex + d,
                this->parameterTransform_,
                jacobian_full.block(offset, 0, blockSize, nParam).template topRows<3>());
          }

          if (this->activeJointParams_[paramIndex + 3 + d]) {
            const Eigen::Vector3<T> axis =
                prevState.jointState[ancestorJointIndex].rotationAxis.col(d);

            // (1) Translation part: Changes in position error due to rotation of ancestor joint
            const Eigen::Vector3<T> jc =
                -(targetTransform.rotate(jointState.getRotationDerivative(d, posd))) * wgt;
            jacobian_jointParams_to_modelParams<T>(
                jc,
                paramIndex + d + 3,
                this->parameterTransform_,
                jacobian_full.block(offset, 0, blockSize, nParam).template topRows<3>());

            // (2) Rotation part: Changes in rotation error due to rotation of ancestor joint
            switch (rotationErrorType_) {
              case RotationErrorType::QuaternionLogMap: {
                const Eigen::Matrix<T, 3, 1> rotJac = computeLogMapRotationJacobianPrev(
                    prevState.jointState[i].rotation(),
                    nextRot,
                    targetTransform.rotation,
                    axis,
                    awgt,
                    dLogDq);
                jacobian_jointParams_to_modelParams<T>(
                    rotJac,
                    paramIndex + d + 3,
                    this->parameterTransform_,
                    jacobian_full.block(offset, 0, blockSize, nParam)
                        .middleRows(3, rotationResidualSize));
                break;
              }
              case RotationErrorType::RotationMatrixDifference: {
                const Eigen::Matrix<T, 9, 1> rotJac = computeMatrixDiffRotationJacobianPrev(
                    prevState.jointState[i].rotation(), targetTransform.rotation, axis, awgt);
                jacobian_jointParams_to_modelParams<T>(
                    rotJac,
                    paramIndex + d + 3,
                    this->parameterTransform_,
                    jacobian_full.block(offset, 0, blockSize, nParam)
                        .middleRows(3, rotationResidualSize));
                break;
              }
            }
          }
        }

        if (this->activeJointParams_[paramIndex + 6]) {
          const Eigen::Vector3<T> jc =
              -(targetTransform.rotate(jointState.getScaleDerivative(posd))) * wgt;
          jacobian_jointParams_to_modelParams<T>(
              jc,
              paramIndex + 6,
              this->parameterTransform_,
              jacobian_full.block(offset, 0, blockSize, nParam).template topRows<3>());
        }

        ancestorJointIndex = this->skeleton_.joints[ancestorJointIndex].parent;
      }
    }

    // Jacobian for nextState
    {
      size_t ancestorJointIndex = i;
      while (ancestorJointIndex != kInvalidIndex) {
        MT_CHECK(ancestorJointIndex < this->skeleton_.joints.size());

        const auto& jointState = nextState.jointState[ancestorJointIndex];
        const size_t paramIndex = ancestorJointIndex * kParametersPerJoint;
        const Eigen::Vector3<T> posd =
            nextState.jointState[i].translation() - jointState.translation();

        for (size_t d = 0; d < 3; d++) {
          if (this->activeJointParams_[paramIndex + d]) {
            const Eigen::Vector3<T> jc = jointState.getTranslationDerivative(d) * wgt;
            jacobian_jointParams_to_modelParams<T>(
                jc,
                paramIndex + d,
                this->parameterTransform_,
                jacobian_full.block(offset, nParam, blockSize, nParam).template topRows<3>());
          }

          if (this->activeJointParams_[paramIndex + 3 + d]) {
            const Eigen::Vector3<T> axis =
                nextState.jointState[ancestorJointIndex].rotationAxis.col(d);

            // (1) Translation part: Changes in position error due to rotation of ancestor joint
            const Eigen::Vector3<T> jc = jointState.getRotationDerivative(d, posd) * wgt;
            jacobian_jointParams_to_modelParams<T>(
                jc,
                paramIndex + d + 3,
                this->parameterTransform_,
                jacobian_full.block(offset, nParam, blockSize, nParam).template topRows<3>());

            // (2) Rotation part: Changes in rotation error due to rotation of ancestor joint
            switch (rotationErrorType_) {
              case RotationErrorType::QuaternionLogMap: {
                const Eigen::Matrix<T, 3, 1> rotJac =
                    computeLogMapRotationJacobianNext(prevRot, nextRot, axis, awgt, dLogDq);
                jacobian_jointParams_to_modelParams<T>(
                    rotJac,
                    paramIndex + d + 3,
                    this->parameterTransform_,
                    jacobian_full.block(offset, nParam, blockSize, nParam)
                        .middleRows(3, rotationResidualSize));
                break;
              }
              case RotationErrorType::RotationMatrixDifference: {
                const Eigen::Matrix<T, 9, 1> rotJac = computeMatrixDiffRotationJacobianNext(
                    nextState.jointState[i].rotation(), axis, awgt);
                jacobian_jointParams_to_modelParams<T>(
                    rotJac,
                    paramIndex + d + 3,
                    this->parameterTransform_,
                    jacobian_full.block(offset, nParam, blockSize, nParam)
                        .middleRows(3, rotationResidualSize));
                break;
              }
            }
          }
        }

        if (this->activeJointParams_[paramIndex + 6]) {
          const Eigen::Vector3<T> jc = jointState.getScaleDerivative(posd) * wgt;
          jacobian_jointParams_to_modelParams<T>(
              jc,
              paramIndex + 6,
              this->parameterTransform_,
              jacobian_full.block(offset, nParam, blockSize, nParam).template topRows<3>());
        }

        ancestorJointIndex = this->skeleton_.joints[ancestorJointIndex].parent;
      }
    }

    offset += blockSize;
  }

  usedRows = gsl::narrow_cast<int>(offset);

  // return error
  return error;
}

template class StateSequenceErrorFunctionT<float>;
template class StateSequenceErrorFunctionT<double>;

} // namespace momentum
