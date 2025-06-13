/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/diff_ik/fully_differentiable_state_error_function.h>

#include "momentum/math/utility.h"

#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/diff_ik/ceres_utility.h>

#include <algorithm>

namespace momentum {

template <typename T>
FullyDifferentiableStateErrorFunctionT<T>::FullyDifferentiableStateErrorFunctionT(
    const Skeleton& skel,
    const ParameterTransform& pt)
    : StateErrorFunctionT<T>(skel, pt) {}

template <typename T>
FullyDifferentiableStateErrorFunctionT<T>::~FullyDifferentiableStateErrorFunctionT() = default;

template <typename T>
std::vector<std::string> FullyDifferentiableStateErrorFunctionT<T>::inputs() const {
  return {"targetPositionWeights", "targetRotationWeights", "targetState"};
}

template <typename T>
Eigen::Index FullyDifferentiableStateErrorFunctionT<T>::getInputSize(
    const std::string& name) const {
  if (name == "targetPositionWeights" || name == "targetRotationWeights") {
    return this->skeleton_.joints.size();
  } else if (name == "targetState") {
    return 8 * this->skeleton_.joints.size();
  } else {
    MT_THROW("Unknown input to FullyDifferentiableMotionErrorFunctionT: {}", name);
  }
}

template <typename T>
void FullyDifferentiableStateErrorFunctionT<T>::getInputImp(
    const std::string& name,
    Eigen::Ref<Eigen::VectorX<T>> result) const {
  if (name == "targetPositionWeights") {
    result = this->getPositionWeights();
  } else if (name == "targetRotationWeights") {
    result = this->getRotationWeights();
  } else if (name == "targetState") {
    const TransformListT<T>& targetState = this->getTargetState();

    for (size_t iJoint = 0; iJoint < this->skeleton_.joints.size(); ++iJoint) {
      result.template segment<3>(8 * iJoint + 0) = targetState[iJoint].translation;
      result.template segment<4>(8 * iJoint + 3) =
          targetState[iJoint].rotation.normalized().coeffs();
      result(8 * iJoint + 7) = targetState[iJoint].scale;
    }
  } else {
    MT_THROW("Unknown input to FullyDifferentiableMotionErrorFunctionT: {}", name);
  }
}

template <typename T>
void FullyDifferentiableStateErrorFunctionT<T>::setInputImp(
    const std::string& name,
    Eigen::Ref<const Eigen::VectorX<T>> value) {
  if (name == "targetPositionWeights") {
    this->setTargetWeights(value, this->getRotationWeights());
  } else if (name == "targetRotationWeights") {
    this->setTargetWeights(this->getPositionWeights(), value);
  } else if (name == "targetState") {
    TransformListT<T> transforms(this->skeleton_.joints.size());
    for (size_t iJoint = 0; iJoint < this->skeleton_.joints.size(); ++iJoint) {
      const Eigen::Vector3<T> pos = value.template segment<3>(8 * iJoint + 0);
      const Eigen::Quaternion<T> rot =
          Eigen::Quaternion<T>(value.template segment<4>(8 * iJoint + 3)).normalized();
      const T scale = value(8 * iJoint + 7);
      transforms[iJoint] = TransformT<T>(pos, rot, scale);
    }
    this->setTargetState(transforms);
  } else {
    MT_THROW("Unknown input to FullyDifferentiableMotionErrorFunctionT: {}", name);
  }
}

template <typename T>
template <typename JetType>
JetType FullyDifferentiableStateErrorFunctionT<T>::calculateGradient_dot(
    const SkeletonStateT<T>& state,
    size_t iJoint,
    const Eigen::Vector3<JetType>& targetTranslation,
    const Eigen::Quaternion<JetType>& targetRotation,
    const JetType& targetPositionWeight,
    const JetType& targetRotationWeight,
    Eigen::Ref<const Eigen::VectorX<T>> vec) {
  // check all is valid
  Expects(state.jointState.size() == gsl::narrow<Eigen::Index>(this->skeleton_.joints.size()));

  JetType result;

  // calculate orientation gradient
  const Eigen::Quaternion<T>& rot = state.jointState[iJoint].rotation();
  const Eigen::Matrix3<JetType> rotDiff =
      rot.toRotationMatrix() - targetRotation.toRotationMatrix();
  const JetType rwgt = StateErrorFunctionT<T>::kOrientationWeight * this->getRotationWeight() *
      this->weight_ * targetRotationWeight;

  // calculate position gradient
  const Eigen::Vector3<JetType> diff = state.jointState[iJoint].translation() - targetTranslation;
  const JetType pwgt = StateErrorFunctionT<T>::kPositionWeight * this->getPositionWeight() *
      this->weight_ * targetPositionWeight;

  // loop over all joints the constraint is attached to and calculate gradient
  size_t jointIndex = iJoint;
  while (jointIndex != kInvalidIndex) {
    // check for valid index
    Ensures(jointIndex < this->skeleton_.joints.size());

    const size_t paramIndex = jointIndex * kParametersPerJoint;

    // precalculate some more data for position gradient
    const Eigen::Vector3<T> posd =
        state.jointState[iJoint].translation() - state.jointState[jointIndex].translation();

    for (size_t d = 0; d < 3; d++) {
      // position gradient
      if (this->activeJointParams_[paramIndex + d]) {
        result += times_parameterTransform_times_v(
            T(2) * diff.dot(state.jointState[jointIndex].getTranslationDerivative(d)) * pwgt,
            paramIndex + d,
            this->parameterTransform_,
            vec);
      }
      if (this->activeJointParams_[paramIndex + 3 + d]) {
        // calculate joint gradient consisting of position gradient and orientation gradient
        const Eigen::Vector3<T> axis = state.jointState[jointIndex].rotationAxis.col(d);
        const auto rotD = crossProductMatrix(axis) * rot;
        const JetType val =
            T(2) * diff.dot(state.jointState[jointIndex].getRotationDerivative(d, posd)) * pwgt +
            T(2) * rwgt * rotD.cwiseProduct(rotDiff).sum();
        result += times_parameterTransform_times_v(
            val, paramIndex + 3 + d, this->parameterTransform_, vec);
      }
    }
    if (this->activeJointParams_[paramIndex + 6]) {
      // calculate joint gradient
      const JetType val =
          T(2) * diff.dot(state.jointState[jointIndex].getScaleDerivative(posd)) * pwgt;
      result +=
          times_parameterTransform_times_v(val, paramIndex + 6, this->parameterTransform_, vec);
    }

    // go to the next joint
    jointIndex = this->skeleton_.joints[jointIndex].parent;
  }

  // return error
  return result;
}

template <typename T>
Eigen::VectorX<T> FullyDifferentiableStateErrorFunctionT<T>::d_gradient_d_input_dot(
    const std::string& inputName,
    const ModelParametersT<T>& /* params */,
    const SkeletonStateT<T>& state,
    Eigen::Ref<const Eigen::VectorX<T>> inputVec) {
  const auto& targetState = this->getTargetState();
  const auto& targetPositionWeights = this->getPositionWeights();
  const auto& targetRotationWeights = this->getRotationWeights();

  if (targetState.empty()) {
    return Eigen::VectorX<T>::Zero(getInputSize(inputName));
  }

  // Use automatic differentiation on the gradient function:
  if (inputName == "targetPositionWeights") {
    Eigen::VectorX<T> result = Eigen::VectorX<T>::Zero(this->skeleton_.joints.size());
    using JetType = ceres::Jet<T, 1>;
    for (size_t iJoint = 0; iJoint < this->skeleton_.joints.size(); ++iJoint) {
      result(iJoint) = calculateGradient_dot<JetType>(
                           state,
                           iJoint,
                           targetState[iJoint].translation.template cast<JetType>(),
                           targetState[iJoint].rotation.normalized().template cast<JetType>(),
                           JetType(targetPositionWeights[iJoint], 0),
                           JetType(targetRotationWeights[iJoint]),
                           inputVec)
                           .v[0];
    }
    return result;
  } else if (inputName == "targetRotationWeights") {
    Eigen::VectorX<T> result = Eigen::VectorX<T>::Zero(this->skeleton_.joints.size());
    using JetType = ceres::Jet<T, 1>;
    for (size_t iJoint = 0; iJoint < this->skeleton_.joints.size(); ++iJoint) {
      result(iJoint) = calculateGradient_dot<JetType>(
                           state,
                           iJoint,
                           targetState[iJoint].translation.template cast<JetType>(),
                           targetState[iJoint].rotation.normalized().template cast<JetType>(),
                           JetType(targetPositionWeights[iJoint]),
                           JetType(targetRotationWeights[iJoint], 0),
                           inputVec)
                           .v[0];
    }
    return result;
  } else if (inputName == "targetState") {
    Eigen::VectorX<T> result = Eigen::VectorX<T>::Zero(8 * this->skeleton_.joints.size());
    for (size_t iJoint = 0; iJoint < this->skeleton_.joints.size(); ++iJoint) {
      {
        using JetType = ceres::Jet<T, 3>;
        result.template segment<3>(8 * iJoint + 0) =
            calculateGradient_dot<JetType>(
                state,
                iJoint,
                buildJetVec<T, 3>(targetState[iJoint].translation),
                targetState[iJoint].rotation.normalized().template cast<JetType>(),
                JetType(targetPositionWeights[iJoint]),
                JetType(targetRotationWeights[iJoint]),
                inputVec)
                .v;
      }
      {
        using JetType = ceres::Jet<T, 4>;
        result.template segment<4>(8 * iJoint + 3) =
            calculateGradient_dot<JetType>(
                state,
                iJoint,
                targetState[iJoint].translation.template cast<JetType>(),
                Eigen::Quaternion<JetType>(
                    buildJetVec<T, 4>(targetState[iJoint].rotation.normalized().coeffs()))
                    .normalized(),
                JetType(targetPositionWeights[iJoint]),
                JetType(targetRotationWeights[iJoint]),
                inputVec)
                .v;
      }
    }
    return result;
  } else {
    MT_THROW("Unknown input to FullyDifferentiableMotionErrorFunctionT: {}", inputName);
  }
}

template class FullyDifferentiableStateErrorFunctionT<float>;
template class FullyDifferentiableStateErrorFunctionT<double>;

} // namespace momentum
