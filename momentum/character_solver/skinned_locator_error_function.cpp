/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/skinned_locator_error_function.h"
#include "momentum/character_solver/skinning_weight_iterator.h"

#include "momentum/character/blend_shape.h"
#include "momentum/character/blend_shape_skinning.h"
#include "momentum/character/character.h"
#include "momentum/character/linear_skinning.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character/skin_weights.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"

#include <numeric>

namespace momentum {

template <typename T>
SkinnedLocatorErrorFunctionT<T>::SkinnedLocatorErrorFunctionT(const Character& character_in)
    : SkeletonErrorFunctionT<T>(character_in.skeleton, character_in.parameterTransform),
      character_(character_in) {}

template <typename T>
SkinnedLocatorErrorFunctionT<T>::~SkinnedLocatorErrorFunctionT() = default;

template <typename T>
void SkinnedLocatorErrorFunctionT<T>::clearConstraints() {
  constraints_.clear();
}

template <typename T>
void SkinnedLocatorErrorFunctionT<T>::addConstraint(
    int locatorIndex,
    T weight,
    const Eigen::Vector3<T>& targetPosition) {
  MT_CHECK(locatorIndex >= 0 && ((size_t)locatorIndex) < character_.skinnedLocators.size());
  constraints_.push_back(SkinnedLocatorConstraintT<T>{locatorIndex, weight, targetPosition});
}

template <typename T>
void SkinnedLocatorErrorFunctionT<T>::setConstraints(
    const std::vector<SkinnedLocatorConstraintT<T>>& constraints) {
  constraints_ = constraints;
}

template <typename T>
Eigen::Vector3<T> SkinnedLocatorErrorFunctionT<T>::getLocatorRestPosition(
    const ModelParametersT<T>& modelParams,
    int locatorIndex) const {
  MT_CHECK(locatorIndex >= 0 && locatorIndex < static_cast<int>(character_.skinnedLocators.size()));
  const auto& locator = character_.skinnedLocators[locatorIndex];

  Vector3<T> result = locator.position.template cast<T>();
  int locatorParameterIndex = -1;
  if (locatorIndex < character_.parameterTransform.skinnedLocatorParameters.size()) {
    locatorParameterIndex = character_.parameterTransform.skinnedLocatorParameters[locatorIndex];
  }

  if (locatorParameterIndex >= 0) {
    result += modelParams.v.template segment<3>(locatorParameterIndex).template cast<T>();
  }

  return result;
}

template <typename T>
Eigen::Vector3<T> SkinnedLocatorErrorFunctionT<T>::calculateSkinnedLocatorPosition(
    const SkeletonStateT<T>& state,
    int locatorIndex,
    const Eigen::Vector3<T>& locatorRestPos) const {
  MT_CHECK(locatorIndex >= 0 && locatorIndex < static_cast<int>(character_.skinnedLocators.size()));
  const auto& locator = character_.skinnedLocators[locatorIndex];

  Eigen::Vector3<T> worldPos = Eigen::Vector3<T>::Zero();
  T weightSum = 0;
  for (int k = 0; k < locator.skinWeights.size(); ++k) {
    const auto& weight = locator.skinWeights[k];
    const auto boneIndex = locator.parents[k];
    const auto& jointState = state.jointState[boneIndex];

    worldPos += weight *
        (jointState.transform *
         (character_.inverseBindPose[boneIndex].template cast<T>() * locatorRestPos))
            .template cast<T>();
    weightSum += weight;
  }

  return worldPos / weightSum;
}

template <typename T>
double SkinnedLocatorErrorFunctionT<T>::getError(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& state) {
  MT_PROFILE_FUNCTION();

  // loop over all constraints and calculate the error
  double error = 0.0;

  for (size_t i = 0; i < constraints_.size(); ++i) {
    const SkinnedLocatorConstraintT<T>& constr = constraints_[i];

    const Eigen::Vector3<T> locatorRestPos =
        getLocatorRestPosition(modelParameters, constr.locatorIndex);
    const Eigen::Vector3<T> locatorWorldPos =
        calculateSkinnedLocatorPosition(state, constr.locatorIndex, locatorRestPos);
    const Eigen::Vector3<T> diff = locatorWorldPos - constr.targetPosition;
    error += constr.weight * diff.squaredNorm();
  }

  // return error
  return error * this->weight_;
}

template <typename T>
void SkinnedLocatorErrorFunctionT<T>::calculateDWorldPos(
    const SkeletonStateT<T>& state,
    const SkinnedLocatorConstraintT<T>& constr,
    const Eigen::Vector3<T>& d_restPos,
    Eigen::Vector3<T>& d_worldPos) const {
  const auto& locator = character_.skinnedLocators[constr.locatorIndex];

  for (int k = 0; k < locator.skinWeights.size(); ++k) {
    const auto& weight = locator.skinWeights[k];
    const auto boneIndex = locator.parents[k];
    const auto& jointState = state.jointState[boneIndex];

    if (weight > 0) {
      // Use the full transformation matrix to be consistent with calculateSkinnedLocatorPosition
      d_worldPos += weight *
          ((jointState.transform * character_.inverseBindPose[boneIndex].template cast<T>())
               .linear() *
           d_restPos);
    }
  }
}

template <typename T>
double SkinnedLocatorErrorFunctionT<T>::calculatePositionGradient(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& state,
    const SkinnedLocatorConstraintT<T>& constr,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  const Eigen::Vector3<T> locatorRestPos =
      getLocatorRestPosition(modelParameters, constr.locatorIndex);
  const Eigen::Vector3<T> locatorWorldPos =
      calculateSkinnedLocatorPosition(state, constr.locatorIndex, locatorRestPos);
  const Eigen::Vector3<T> diff = locatorWorldPos - constr.targetPosition;

  // calculate the difference between target and position and error
  const T wgt = constr.weight * 2.0f * this->weight_;

  // Handle derivatives wrt skinnedLocatorParameters if this locator is parameterized
  int locatorParameterIndex = getSkinnedLocatorParameterIndex(constr.locatorIndex);
  if (locatorParameterIndex >= 0) {
    // For each coordinate (x, y, z), calculate the derivative
    for (int d = 0; d < 3; ++d) {
      // Create a unit vector in the direction of the coordinate
      Eigen::Vector3<T> d_restPos = Eigen::Vector3<T>::Zero();
      d_restPos[d] = 1.0;

      // Calculate the derivative of the world position with respect to this coordinate
      Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();
      calculateDWorldPos(state, constr, d_restPos, d_worldPos);

      // Add contribution to the gradient
      gradient(locatorParameterIndex + d) += diff.dot(d_worldPos) * wgt;
    }
  }

  SkinningWeightIteratorT<T> skinningIter(
      this->character_, character_.skinnedLocators[constr.locatorIndex], locatorRestPos, state);

  // Handle derivatives wrt jointParameters
  while (!skinningIter.finished()) {
    const auto [jointIndex, boneWeight, pos] = skinningIter.next();

    // check for valid index
    MT_CHECK(jointIndex < this->skeleton_.joints.size());

    const auto& jointState = state.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> posd = pos - jointState.translation();

    // calculate derivatives based on active joints
    for (size_t d = 0; d < 3; d++) {
      if (this->activeJointParams_[paramIndex + d]) {
        // Gradient wrt translation:
        gradient_jointParams_to_modelParams(
            boneWeight * diff.dot(jointState.getTranslationDerivative(d)) * wgt,
            paramIndex + d,
            this->parameterTransform_,
            gradient);
      }
      if (this->activeJointParams_[paramIndex + 3 + d]) {
        // Gradient wrt rotation:
        gradient_jointParams_to_modelParams(
            boneWeight * diff.dot(jointState.getRotationDerivative(d, posd)) * wgt,
            paramIndex + 3 + d,
            this->parameterTransform_,
            gradient);
      }
    }
    if (this->activeJointParams_[paramIndex + 6]) {
      // Gradient wrt scale:
      gradient_jointParams_to_modelParams(
          boneWeight * diff.dot(jointState.getScaleDerivative(posd)) * wgt,
          paramIndex + 6,
          this->parameterTransform_,
          gradient);
    }
  }

  return constr.weight * diff.squaredNorm();
}

template <typename T>
double SkinnedLocatorErrorFunctionT<T>::calculatePositionJacobian(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& state,
    const SkinnedLocatorConstraintT<T>& constr,
    Ref<Eigen::MatrixX<T>> jac,
    Ref<Eigen::VectorX<T>> res) const {
  const Eigen::Vector3<T> locatorRestPos =
      getLocatorRestPosition(modelParameters, constr.locatorIndex);
  const Eigen::Vector3<T> locatorWorldPos =
      calculateSkinnedLocatorPosition(state, constr.locatorIndex, locatorRestPos);
  const Eigen::Vector3<T> diff = locatorWorldPos - constr.targetPosition;

  const T wgt = std::sqrt(constr.weight * this->weight_);

  // Handle derivatives wrt skinnedLocatorParameters if this locator is parameterized
  int locatorParameterIndex = getSkinnedLocatorParameterIndex(constr.locatorIndex);
  if (locatorParameterIndex >= 0) {
    // For each coordinate (x, y, z), calculate the derivative
    for (int d = 0; d < 3; ++d) {
      // Create a unit vector in the direction of the coordinate
      Eigen::Vector3<T> d_restPos = Eigen::Vector3<T>::Zero();
      d_restPos[d] = 1.0;

      // Calculate the derivative of the world position with respect to this coordinate
      Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();
      calculateDWorldPos(state, constr, d_restPos, d_worldPos);

      // Add contribution to the jacobian
      jac.col(locatorParameterIndex + d).template segment<3>(0) += d_worldPos * wgt;
    }
  }

  SkinningWeightIteratorT<T> skinningIter(
      this->character_, character_.skinnedLocators[constr.locatorIndex], locatorRestPos, state);

  // Handle derivatives wrt jointParameters
  while (!skinningIter.finished()) {
    const auto [jointIndex, boneWeight, pos] = skinningIter.next();

    // check for valid index
    MT_CHECK(jointIndex < this->skeleton_.joints.size());

    const auto& jointState = state.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> posd = pos - jointState.translation();

    // calculate derivatives based on active joints
    for (size_t d = 0; d < 3; d++) {
      if (this->activeJointParams_[paramIndex + d]) {
        jacobian_jointParams_to_modelParams<T>(
            boneWeight * wgt * jointState.getTranslationDerivative(d),
            paramIndex + d,
            this->parameterTransform_,
            jac);
      }
      if (this->activeJointParams_[paramIndex + 3 + d]) {
        jacobian_jointParams_to_modelParams<T>(
            boneWeight * wgt * jointState.getRotationDerivative(d, posd),
            paramIndex + d + 3,
            this->parameterTransform_,
            jac);
      }
    }
    if (this->activeJointParams_[paramIndex + 6]) {
      jacobian_jointParams_to_modelParams<T>(
          boneWeight * wgt * jointState.getScaleDerivative(posd),
          paramIndex + 6,
          this->parameterTransform_,
          jac);
    }
  }

  res = diff * wgt;

  return wgt * wgt * diff.squaredNorm();
}

template <typename T>
double SkinnedLocatorErrorFunctionT<T>::getGradient(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& state,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  double error = 0;

  // Process each constraint
  for (const auto& constraint : constraints_) {
    error += calculatePositionGradient(modelParameters, state, constraint, gradient);
  }

  return this->weight_ * error;
}

template <typename T>
double SkinnedLocatorErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& state,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    int& usedRows) {
  MT_CHECK(
      jacobian.cols() == static_cast<Eigen::Index>(this->parameterTransform_.transform.cols()));
  MT_CHECK(jacobian.rows() >= (Eigen::Index)(3 * constraints_.size()));
  MT_CHECK(residual.rows() >= (Eigen::Index)(3 * constraints_.size()));

  double error = 0;

  for (size_t iCons = 0; iCons < constraints_.size(); ++iCons) {
    const auto& constraint = constraints_[iCons];
    error += calculatePositionJacobian(
        modelParameters,
        state,
        constraint,
        jacobian.block(3 * iCons, 0, 3, modelParameters.size()),
        residual.middleRows(3 * iCons, 3));
    usedRows += 3;
  }

  return error;
}

template <typename T>
size_t SkinnedLocatorErrorFunctionT<T>::getJacobianSize() const {
  return 3 * constraints_.size();
}

template class SkinnedLocatorErrorFunctionT<float>;
template class SkinnedLocatorErrorFunctionT<double>;

} // namespace momentum
