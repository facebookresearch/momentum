/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/state_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/math/utility.h"

namespace momentum {

template <typename T>
StateErrorFunctionT<T>::StateErrorFunctionT(const Skeleton& skel, const ParameterTransform& pt)
    : SkeletonErrorFunctionT<T>(skel, pt) {
  targetPositionWeights_.setOnes(this->skeleton_.joints.size());
  targetRotationWeights_.setOnes(this->skeleton_.joints.size());
  posWgt_ = T(1);
  rotWgt_ = T(1);
}

template <typename T>
StateErrorFunctionT<T>::StateErrorFunctionT(const Character& character)
    : StateErrorFunctionT<T>(character.skeleton, character.parameterTransform) {}

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
double StateErrorFunctionT<T>::getError(
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& state) {
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
    // calculate orientation error
    // Note: |R0 - RT|² is a valid norm on SO3, it doesn't have the same slope as the squared angle
    // difference
    //       but it's derivative doesn't have a singularity at the minimum, so is more stable
    const Eigen::Quaternion<T>& target = targetState_[i].rotation.normalized();
    const Eigen::Quaternion<T>& rot = state.jointState[i].rotation();
    const Eigen::Matrix3<T> rotDiff = rot.toRotationMatrix() - target.toRotationMatrix();
    error += rotDiff.squaredNorm() * kOrientationWeight * rotWgt_ * targetRotationWeights_[i];

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
    const Eigen::Matrix3<T> rotDiff = rot.toRotationMatrix() - target.toRotationMatrix();
    const T rwgt = kOrientationWeight * rotWgt_ * this->weight_ * targetRotationWeights_[i];
    error += rotDiff.squaredNorm() * rwgt;

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
          // explicitly multiply with the parameter transform to generate parameter space gradients
          for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
               index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
               ++index) {
            gradient[this->parameterTransform_.transform.innerIndexPtr()[index]] +=
                val * this->parameterTransform_.transform.valuePtr()[index];
          }
        }
        if (this->activeJointParams_[paramIndex + 3 + d]) {
          // calculate joint gradient consisting of position gradient and orientation gradient
          const Eigen::Vector3<T> axis = state.jointState[jointIndex].rotationAxis.col(d);
          const auto rotD = crossProductMatrix(axis) * rot;
          const T val =
              T(2) * diff.dot(state.jointState[jointIndex].getRotationDerivative(d, posd)) * pwgt +
              T(2) * rwgt * rotD.cwiseProduct(rotDiff).sum();
          // explicitly multiply with the parameter transform to generate parameter space gradients
          for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d];
               index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d + 1];
               ++index) {
            gradient[this->parameterTransform_.transform.innerIndexPtr()[index]] +=
                val * this->parameterTransform_.transform.valuePtr()[index];
          }
        }
      }
      if (this->activeJointParams_[paramIndex + 6]) {
        // calculate joint gradient
        const T val = T(2) * diff.dot(state.jointState[jointIndex].getScaleDerivative(posd)) * pwgt;
        // explicitly multiply with the parameter transform to generate parameter space gradients
        for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6];
             index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6 + 1];
             ++index) {
          gradient[this->parameterTransform_.transform.innerIndexPtr()[index]] +=
              val * this->parameterTransform_.transform.valuePtr()[index];
        }
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
  auto result =
      (targetPositionWeights_.array() != 0 || targetRotationWeights_.array() != 0).count() * 12;
  return result;
}

template <typename T>
double StateErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& state,
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

    Ref<Eigen::MatrixX<T>> jac =
        jacobian.block(offset, 0, 12, this->parameterTransform_.transform.cols());
    Ref<Eigen::VectorX<T>> res = residual.template middleRows<12>(offset);
    offset += 12;

    // calculate translation gradient
    const auto transDiff = (state.jointState[i].translation() - targetState_[i].translation).eval();

    // calculate orientation gradient
    const Eigen::Quaternion<T>& target = targetState_[i].rotation.normalized();
    const Eigen::Quaternion<T>& rot = state.jointState[i].rotation();
    const Eigen::Matrix3<T> rotDiff = rot.toRotationMatrix() - target.toRotationMatrix();
    const T rwgt = kOrientationWeight * rotWgt_ * this->weight_ * targetRotationWeights_[i];
    error += rotDiff.squaredNorm() * rwgt;
    const T awgt = std::sqrt(rwgt);

    // calculate the difference between target and position and error
    const T pwgt = kPositionWeight * posWgt_ * this->weight_ * targetPositionWeights_[i];
    const T wgt = std::sqrt(pwgt);
    error += transDiff.squaredNorm() * pwgt;

    // update the residue
    res.template topRows<3>().noalias() = transDiff * wgt;
    res.template bottomRows<9>().noalias() =
        Map<const Eigen::Matrix<T, 9, 1>>(rotDiff.data()) * awgt;

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
          const Eigen::Vector3<T> jc = jointState.getTranslationDerivative(d) * wgt;
          for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
               index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
               ++index) {
            jac.col(this->parameterTransform_.transform.innerIndexPtr()[index])
                .template topRows<3>()
                .noalias() += jc * this->parameterTransform_.transform.valuePtr()[index];
          }
        }

        if (this->activeJointParams_[paramIndex + 3 + d]) {
          const Eigen::Vector3<T> jc = jointState.getRotationDerivative(d, posd) * wgt;

          const Eigen::Vector3<T> axis = state.jointState[jointIndex].rotationAxis.col(d);
          const Eigen::Matrix3<T> rotD = crossProductMatrix(axis) * rot * awgt;
          const auto ja = Map<const Eigen::Matrix<T, 9, 1>>(rotD.data(), rotD.size());

          for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 3];
               index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 3 + 1];
               ++index) {
            jac.col(this->parameterTransform_.transform.innerIndexPtr()[index])
                .template topRows<3>()
                .noalias() += jc * this->parameterTransform_.transform.valuePtr()[index];
            jac.col(this->parameterTransform_.transform.innerIndexPtr()[index])
                .template bottomRows<9>()
                .noalias() += ja * this->parameterTransform_.transform.valuePtr()[index];
          }
        }
      }

      if (this->activeJointParams_[paramIndex + 6]) {
        const Eigen::Vector3<T> jc = jointState.getScaleDerivative(posd) * wgt;
        for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6];
             index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6 + 1];
             ++index) {
          jac.col(this->parameterTransform_.transform.innerIndexPtr()[index])
              .template topRows<3>()
              .noalias() += jc * this->parameterTransform_.transform.valuePtr()[index];
        }
      }

      // go to the next joint
      jointIndex = this->skeleton_.joints[jointIndex].parent;
    }
  }

  usedRows = gsl::narrow_cast<int>(offset);

  // return error
  return error;
}

template class StateErrorFunctionT<float>;
template class StateErrorFunctionT<double>;

} // namespace momentum
