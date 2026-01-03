/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_sequence_solver/finite_difference_sequence_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character_solver/error_function_utils.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"

namespace momentum {

template <typename T>
FiniteDifferenceSequenceErrorFunctionT<T>::FiniteDifferenceSequenceErrorFunctionT(
    const Skeleton& skel,
    const ParameterTransform& pt,
    const std::vector<T>& stencilCoefficients)
    : SequenceErrorFunctionT<T>(skel, pt), stencilCoefficients_(stencilCoefficients) {
  MT_CHECK(!stencilCoefficients_.empty(), "Stencil coefficients cannot be empty");
  targetWeights_.setOnes(skel.joints.size());
  targetValues_.resize(skel.joints.size(), Eigen::Vector3<T>::Zero());
}

template <typename T>
FiniteDifferenceSequenceErrorFunctionT<T>::FiniteDifferenceSequenceErrorFunctionT(
    const Character& character,
    const std::vector<T>& stencilCoefficients)
    : FiniteDifferenceSequenceErrorFunctionT<T>(
          character.skeleton,
          character.parameterTransform,
          stencilCoefficients) {}

template <typename T>
void FiniteDifferenceSequenceErrorFunctionT<T>::reset() {
  targetWeights_.setOnes(this->skeleton_.joints.size());
  targetValues_.assign(this->skeleton_.joints.size(), Eigen::Vector3<T>::Zero());
}

template <typename T>
void FiniteDifferenceSequenceErrorFunctionT<T>::setTargetWeights(const Eigen::VectorX<T>& weights) {
  MT_CHECK(weights.size() == static_cast<Eigen::Index>(this->skeleton_.joints.size()));
  targetWeights_ = weights;
}

template <typename T>
void FiniteDifferenceSequenceErrorFunctionT<T>::setTargetValue(
    const Eigen::Vector3<T>& targetValue) {
  targetValues_.assign(this->skeleton_.joints.size(), targetValue);
}

template <typename T>
void FiniteDifferenceSequenceErrorFunctionT<T>::setTargetValues(
    const std::vector<Eigen::Vector3<T>>& targetValues) {
  MT_CHECK(targetValues.size() == this->skeleton_.joints.size());
  targetValues_ = targetValues;
}

template <typename T>
double FiniteDifferenceSequenceErrorFunctionT<T>::getError(
    std::span<const ModelParametersT<T>> /* modelParameters */,
    std::span<const SkeletonStateT<T>> skelStates,
    std::span<const MeshStateT<T>> /* meshStates */) const {
  MT_PROFILE_FUNCTION();

  MT_CHECK(skelStates.size() == stencilCoefficients_.size());

  double error = 0.0;

  for (size_t i = 0; i < this->skeleton_.joints.size(); ++i) {
    if (targetWeights_[i] == 0) {
      continue;
    }

    // Compute derivative using stencil: sum(coeff[k] * pos[k])
    Eigen::Vector3<T> derivative = Eigen::Vector3<T>::Zero();
    for (size_t k = 0; k < stencilCoefficients_.size(); ++k) {
      derivative += stencilCoefficients_[k] * skelStates[k].jointState[i].translation();
    }

    // Compute residual: derivative - target
    const Eigen::Vector3<T> residual = derivative - targetValues_[i];

    error += residual.squaredNorm() * targetWeights_[i];
  }

  return error * this->weight_;
}

template <typename T>
double FiniteDifferenceSequenceErrorFunctionT<T>::getGradient(
    std::span<const ModelParametersT<T>> /* modelParameters */,
    std::span<const SkeletonStateT<T>> skelStates,
    std::span<const MeshStateT<T>> /* meshStates */,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  MT_PROFILE_FUNCTION();

  MT_CHECK(skelStates.size() == stencilCoefficients_.size());

  const Eigen::Index nParam = this->parameterTransform_.numAllModelParameters();
  MT_CHECK(gradient.size() == static_cast<Eigen::Index>(stencilCoefficients_.size()) * nParam);

  double error = 0.0;

  for (size_t i = 0; i < this->skeleton_.joints.size(); ++i) {
    if (targetWeights_[i] == 0) {
      continue;
    }

    // Compute derivative and residual
    Eigen::Vector3<T> derivative = Eigen::Vector3<T>::Zero();
    for (size_t k = 0; k < stencilCoefficients_.size(); ++k) {
      derivative += stencilCoefficients_[k] * skelStates[k].jointState[i].translation();
    }
    const Eigen::Vector3<T> residual = derivative - targetValues_[i];

    const T wgt = targetWeights_[i] * this->weight_;
    error += residual.squaredNorm() * wgt;

    // Gradient contribution from each frame
    // d/dp ||residual||^2 = 2 * residual^T * d(residual)/dp
    // d(residual)/dp = d(derivative)/dp = coeff * d(pos)/dp

    for (size_t frameIdx = 0; frameIdx < stencilCoefficients_.size(); ++frameIdx) {
      const auto& state = skelStates[frameIdx];
      const T coeff = stencilCoefficients_[frameIdx];

      // Walk up the kinematic chain for joint i
      size_t ancestorJointIndex = i;
      while (ancestorJointIndex != kInvalidIndex) {
        MT_CHECK(ancestorJointIndex < this->skeleton_.joints.size());

        const size_t paramIndex = ancestorJointIndex * kParametersPerJoint;
        const Eigen::Vector3<T> posd =
            state.jointState[i].translation() - state.jointState[ancestorJointIndex].translation();

        // Translation parameters (3)
        for (size_t d = 0; d < 3; d++) {
          if (this->activeJointParams_[paramIndex + d]) {
            const T val = T(2) * wgt * coeff *
                residual.dot(state.jointState[ancestorJointIndex].getTranslationDerivative(d));
            gradient_jointParams_to_modelParams<T>(
                val,
                paramIndex + d,
                this->parameterTransform_,
                gradient.segment(frameIdx * nParam, nParam));
          }
        }

        // Rotation parameters (3)
        for (size_t d = 0; d < 3; d++) {
          if (this->activeJointParams_[paramIndex + 3 + d]) {
            const T val = T(2) * wgt * coeff *
                residual.dot(state.jointState[ancestorJointIndex].getRotationDerivative(d, posd));
            gradient_jointParams_to_modelParams<T>(
                val,
                paramIndex + 3 + d,
                this->parameterTransform_,
                gradient.segment(frameIdx * nParam, nParam));
          }
        }

        // Scale parameter (1)
        if (this->activeJointParams_[paramIndex + 6]) {
          const T val = T(2) * wgt * coeff *
              residual.dot(state.jointState[ancestorJointIndex].getScaleDerivative(posd));
          gradient_jointParams_to_modelParams<T>(
              val,
              paramIndex + 6,
              this->parameterTransform_,
              gradient.segment(frameIdx * nParam, nParam));
        }

        ancestorJointIndex = this->skeleton_.joints[ancestorJointIndex].parent;
      }
    }
  }

  return error;
}

template <typename T>
size_t FiniteDifferenceSequenceErrorFunctionT<T>::getJacobianSize() const {
  // 3 rows per active joint (x, y, z derivative components)
  const auto nActiveJoints = (targetWeights_.array() != 0).count();
  return nActiveJoints * 3;
}

template <typename T>
double FiniteDifferenceSequenceErrorFunctionT<T>::getJacobian(
    std::span<const ModelParametersT<T>> /* modelParameters */,
    std::span<const SkeletonStateT<T>> skelStates,
    std::span<const MeshStateT<T>> /* meshStates */,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    int& usedRows) const {
  MT_PROFILE_FUNCTION();

  MT_CHECK(skelStates.size() == stencilCoefficients_.size());

  const Eigen::Index nParam = this->parameterTransform_.numAllModelParameters();
  MT_CHECK(jacobian.cols() == static_cast<Eigen::Index>(stencilCoefficients_.size()) * nParam);

  double error = 0.0;
  Eigen::Index offset = 0;

  for (size_t i = 0; i < this->skeleton_.joints.size(); ++i) {
    if (targetWeights_[i] == 0) {
      continue;
    }

    // Compute derivative and residual
    Eigen::Vector3<T> derivative = Eigen::Vector3<T>::Zero();
    for (size_t k = 0; k < stencilCoefficients_.size(); ++k) {
      derivative += stencilCoefficients_[k] * skelStates[k].jointState[i].translation();
    }
    const Eigen::Vector3<T> res = derivative - targetValues_[i];

    const T wgt = targetWeights_[i] * this->weight_;
    const T sqrtWgt = std::sqrt(wgt);

    error += res.squaredNorm() * wgt;
    residual.template segment<3>(offset).noalias() = res * sqrtWgt;

    // Jacobian for each frame
    for (size_t frameIdx = 0; frameIdx < stencilCoefficients_.size(); ++frameIdx) {
      const auto& state = skelStates[frameIdx];
      const T coeff = stencilCoefficients_[frameIdx] * sqrtWgt;

      // Walk up the kinematic chain for joint i
      size_t ancestorJointIndex = i;
      while (ancestorJointIndex != kInvalidIndex) {
        MT_CHECK(ancestorJointIndex < this->skeleton_.joints.size());

        const auto& jointState = state.jointState[ancestorJointIndex];
        const size_t paramIndex = ancestorJointIndex * kParametersPerJoint;
        const Eigen::Vector3<T> posd = state.jointState[i].translation() - jointState.translation();

        // Translation parameters (3)
        for (size_t d = 0; d < 3; d++) {
          if (this->activeJointParams_[paramIndex + d]) {
            const Eigen::Vector3<T> jc = jointState.getTranslationDerivative(d) * coeff;
            jacobian_jointParams_to_modelParams<T>(
                jc,
                paramIndex + d,
                this->parameterTransform_,
                jacobian.block(offset, frameIdx * nParam, 3, nParam));
          }
        }

        // Rotation parameters (3)
        for (size_t d = 0; d < 3; d++) {
          if (this->activeJointParams_[paramIndex + 3 + d]) {
            const Eigen::Vector3<T> jc = jointState.getRotationDerivative(d, posd) * coeff;
            jacobian_jointParams_to_modelParams<T>(
                jc,
                paramIndex + d + 3,
                this->parameterTransform_,
                jacobian.block(offset, frameIdx * nParam, 3, nParam));
          }
        }

        // Scale parameter (1)
        if (this->activeJointParams_[paramIndex + 6]) {
          const Eigen::Vector3<T> jc = jointState.getScaleDerivative(posd) * coeff;
          jacobian_jointParams_to_modelParams<T>(
              jc,
              paramIndex + 6,
              this->parameterTransform_,
              jacobian.block(offset, frameIdx * nParam, 3, nParam));
        }

        ancestorJointIndex = this->skeleton_.joints[ancestorJointIndex].parent;
      }
    }

    offset += 3;
  }

  usedRows = gsl::narrow_cast<int>(offset);

  return error;
}

template class FiniteDifferenceSequenceErrorFunctionT<float>;
template class FiniteDifferenceSequenceErrorFunctionT<double>;

} // namespace momentum
