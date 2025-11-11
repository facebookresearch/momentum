/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/joint_to_joint_distance_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/math/constants.h"
#include "momentum/math/utility.h"

namespace momentum {

template <typename T>
JointToJointDistanceErrorFunctionT<T>::JointToJointDistanceErrorFunctionT(
    const Skeleton& skel,
    const ParameterTransform& pt)
    : SkeletonErrorFunctionT<T>(skel, pt) {}

template <typename T>
JointToJointDistanceErrorFunctionT<T>::JointToJointDistanceErrorFunctionT(
    const Character& character)
    : JointToJointDistanceErrorFunctionT(character.skeleton, character.parameterTransform) {}

template <typename T>
void JointToJointDistanceErrorFunctionT<T>::addConstraint(
    size_t joint1,
    const Vector3<T>& offset1,
    size_t joint2,
    const Vector3<T>& offset2,
    T targetDistance,
    T weight) {
  MT_CHECK(joint1 < this->skeleton_.joints.size(), "Invalid joint1 index: {}", joint1);
  MT_CHECK(joint2 < this->skeleton_.joints.size(), "Invalid joint2 index: {}", joint2);
  MT_CHECK(joint1 != joint2, "joint1 and joint2 must be different");
  MT_CHECK(targetDistance >= 0, "targetDistance must be non-negative");

  constraints_.push_back(
      JointToJointDistanceConstraintT<T>{joint1, offset1, joint2, offset2, targetDistance, weight});
}

template <typename T>
void JointToJointDistanceErrorFunctionT<T>::clearConstraints() {
  constraints_.clear();
}

template <typename T>
double JointToJointDistanceErrorFunctionT<T>::getError(
    const ModelParametersT<T>& /* params */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */) {
  if (state.jointState.empty() || constraints_.empty()) {
    return 0.0;
  }

  double error = 0.0;

  for (const auto& constr : constraints_) {
    const auto& jointState1 = state.jointState[constr.joint1];
    const auto& jointState2 = state.jointState[constr.joint2];

    const Vector3<T> worldPos1 = jointState1.transform * constr.offset1;
    const Vector3<T> worldPos2 = jointState2.transform * constr.offset2;

    const Vector3<T> diff = worldPos1 - worldPos2;
    const T actualDistance = diff.norm();
    const T distanceError = actualDistance - constr.targetDistance;

    error += constr.weight * sqr(distanceError) * kDistanceWeight * this->weight_;
  }

  return error;
}

template <typename T>
double JointToJointDistanceErrorFunctionT<T>::getGradient(
    const ModelParametersT<T>& /* params */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */,
    Ref<VectorX<T>> gradient) {
  if (state.jointState.empty() || constraints_.empty()) {
    return 0.0;
  }

  double error = 0.0;

  for (const auto& constr : constraints_) {
    const auto& jointState1 = state.jointState[constr.joint1];
    const auto& jointState2 = state.jointState[constr.joint2];

    const Vector3<T> worldPos1 = jointState1.transform * constr.offset1;
    const Vector3<T> worldPos2 = jointState2.transform * constr.offset2;

    const Vector3<T> diff = worldPos1 - worldPos2;
    const T actualDistance = diff.norm();

    if (actualDistance < Eps<T>(1e-8f, 1e-14)) {
      continue;
    }

    const T distanceError = actualDistance - constr.targetDistance;
    error += constr.weight * sqr(distanceError) * kDistanceWeight * this->weight_;

    const Vector3<T> direction = diff / actualDistance;
    const T wgt = T(2) * constr.weight * distanceError * kDistanceWeight * this->weight_;

    auto processJoint = [&](size_t jointIndex, const Vector3<T>& worldPos, T sign) {
      size_t currentJoint = jointIndex;
      while (currentJoint != kInvalidIndex) {
        const auto& currentJointState = state.jointState[currentJoint];
        const size_t paramIndex = currentJoint * kParametersPerJoint;
        const Vector3<T> posd = worldPos - currentJointState.translation();

        for (size_t d = 0; d < 3; d++) {
          if (this->activeJointParams_[paramIndex + d]) {
            const T val = sign * direction.dot(currentJointState.getTranslationDerivative(d)) * wgt;
            for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
                 index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
                 ++index) {
              gradient[this->parameterTransform_.transform.innerIndexPtr()[index]] +=
                  val * this->parameterTransform_.transform.valuePtr()[index];
            }
          }
          if (this->activeJointParams_[paramIndex + 3 + d]) {
            const T val =
                sign * direction.dot(currentJointState.getRotationDerivative(d, posd)) * wgt;
            for (auto index =
                     this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d];
                 index <
                 this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d + 1];
                 ++index) {
              gradient[this->parameterTransform_.transform.innerIndexPtr()[index]] +=
                  val * this->parameterTransform_.transform.valuePtr()[index];
            }
          }
        }
        if (this->activeJointParams_[paramIndex + 6]) {
          const T val = sign * direction.dot(currentJointState.getScaleDerivative(posd)) * wgt;
          for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6];
               index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6 + 1];
               ++index) {
            gradient[this->parameterTransform_.transform.innerIndexPtr()[index]] +=
                val * this->parameterTransform_.transform.valuePtr()[index];
          }
        }

        currentJoint = this->skeleton_.joints[currentJoint].parent;
      }
    };

    processJoint(constr.joint1, worldPos1, T(1));
    processJoint(constr.joint2, worldPos2, T(-1));
  }

  return error;
}

template <typename T>
double JointToJointDistanceErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& /* params */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */,
    Ref<MatrixX<T>> jacobian,
    Ref<VectorX<T>> residual,
    int& usedRows) {
  MT_PROFILE_FUNCTION();
  MT_CHECK(
      jacobian.rows() >= gsl::narrow<Eigen::Index>(constraints_.size()),
      "Jacobian rows mismatch: Actual {}, Expected {}",
      jacobian.rows(),
      gsl::narrow<Eigen::Index>(constraints_.size()));

  MT_CHECK(
      residual.rows() >= gsl::narrow<Eigen::Index>(constraints_.size()),
      "Residual rows mismatch: Actual {}, Expected {}",
      residual.rows(),
      gsl::narrow<Eigen::Index>(constraints_.size()));

  if (state.jointState.empty() || constraints_.empty()) {
    usedRows = 0;
    return 0.0;
  }

  const T wgt = std::sqrt(kDistanceWeight * this->weight_);

  int pos = 0;
  double error = 0.0;

  for (const auto& constr : constraints_) {
    const auto& jointState1 = state.jointState[constr.joint1];
    const auto& jointState2 = state.jointState[constr.joint2];

    const Vector3<T> worldPos1 = jointState1.transform * constr.offset1;
    const Vector3<T> worldPos2 = jointState2.transform * constr.offset2;

    const Vector3<T> diff = worldPos1 - worldPos2;
    const T actualDistance = diff.norm();

    if (actualDistance < Eps<T>(1e-8f, 1e-14)) {
      continue;
    }

    const T distanceError = actualDistance - constr.targetDistance;
    error += constr.weight * sqr(distanceError) * kDistanceWeight * this->weight_;

    const Vector3<T> direction = diff / actualDistance;
    const T fac = std::sqrt(constr.weight) * wgt;

    const int row = pos++;

    auto processJoint = [&](size_t jointIndex, const Vector3<T>& worldPos, T sign) {
      size_t currentJoint = jointIndex;
      while (currentJoint != kInvalidIndex) {
        const auto& currentJointState = state.jointState[currentJoint];
        const size_t paramIndex = currentJoint * kParametersPerJoint;
        const Vector3<T> posd = worldPos - currentJointState.translation();

        for (size_t d = 0; d < 3; d++) {
          if (this->activeJointParams_[paramIndex + d]) {
            const T val = sign * direction.dot(currentJointState.getTranslationDerivative(d)) * fac;
            for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
                 index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
                 ++index) {
              jacobian(row, this->parameterTransform_.transform.innerIndexPtr()[index]) +=
                  val * this->parameterTransform_.transform.valuePtr()[index];
            }
          }
          if (this->activeJointParams_[paramIndex + 3 + d]) {
            const T val =
                sign * direction.dot(currentJointState.getRotationDerivative(d, posd)) * fac;
            for (auto index =
                     this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d];
                 index <
                 this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d + 1];
                 ++index) {
              jacobian(row, this->parameterTransform_.transform.innerIndexPtr()[index]) +=
                  val * this->parameterTransform_.transform.valuePtr()[index];
            }
          }
        }
        if (this->activeJointParams_[paramIndex + 6]) {
          const T val = sign * direction.dot(currentJointState.getScaleDerivative(posd)) * fac;
          for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6];
               index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6 + 1];
               ++index) {
            jacobian(row, this->parameterTransform_.transform.innerIndexPtr()[index]) +=
                val * this->parameterTransform_.transform.valuePtr()[index];
          }
        }

        currentJoint = this->skeleton_.joints[currentJoint].parent;
      }
    };

    processJoint(constr.joint1, worldPos1, T(1));
    processJoint(constr.joint2, worldPos2, T(-1));

    residual(row) = distanceError * fac;
  }

  usedRows = pos;

  return error;
}

template <typename T>
size_t JointToJointDistanceErrorFunctionT<T>::getJacobianSize() const {
  return constraints_.size();
}

template class JointToJointDistanceErrorFunctionT<float>;
template class JointToJointDistanceErrorFunctionT<double>;

} // namespace momentum
