/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_sequence_solver/acceleration_sequence_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character_solver/error_function_utils.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"

namespace momentum {

template <typename T>
AccelerationSequenceErrorFunctionT<T>::AccelerationSequenceErrorFunctionT(
    const Skeleton& skel,
    const ParameterTransform& pt)
    : SequenceErrorFunctionT<T>(skel, pt) {
  targetWeights_.setOnes(skel.joints.size());
  targetAccelerations_.resize(skel.joints.size(), Eigen::Vector3<T>::Zero());
}

template <typename T>
AccelerationSequenceErrorFunctionT<T>::AccelerationSequenceErrorFunctionT(
    const Character& character)
    : AccelerationSequenceErrorFunctionT<T>(character.skeleton, character.parameterTransform) {}

template <typename T>
void AccelerationSequenceErrorFunctionT<T>::reset() {
  targetWeights_.setOnes(this->skeleton_.joints.size());
  targetAccelerations_.assign(this->skeleton_.joints.size(), Eigen::Vector3<T>::Zero());
}

template <typename T>
void AccelerationSequenceErrorFunctionT<T>::setTargetWeights(const Eigen::VectorX<T>& weights) {
  MT_CHECK(weights.size() == static_cast<Eigen::Index>(this->skeleton_.joints.size()));
  targetWeights_ = weights;
}

template <typename T>
void AccelerationSequenceErrorFunctionT<T>::setTargetAcceleration(
    const Eigen::Vector3<T>& acceleration) {
  targetAccelerations_.assign(this->skeleton_.joints.size(), acceleration);
}

template <typename T>
void AccelerationSequenceErrorFunctionT<T>::setTargetAccelerations(
    const std::vector<Eigen::Vector3<T>>& accelerations) {
  MT_CHECK(accelerations.size() == this->skeleton_.joints.size());
  targetAccelerations_ = accelerations;
}

template <typename T>
double AccelerationSequenceErrorFunctionT<T>::getError(
    std::span<const ModelParametersT<T>> /* modelParameters */,
    std::span<const SkeletonStateT<T>> skelStates,
    std::span<const MeshStateT<T>> /* meshStates */) const {
  MT_PROFILE_FUNCTION();

  MT_CHECK(skelStates.size() == 3);

  const auto& prevState = skelStates[0]; // t-1
  const auto& currState = skelStates[1]; // t
  const auto& nextState = skelStates[2]; // t+1

  double error = 0.0;

  for (size_t i = 0; i < this->skeleton_.joints.size(); ++i) {
    if (targetWeights_[i] == 0) {
      continue;
    }

    // Compute acceleration: pos[t+1] - 2*pos[t] + pos[t-1]
    const Eigen::Vector3<T> acceleration = nextState.jointState[i].translation() -
        T(2) * currState.jointState[i].translation() + prevState.jointState[i].translation();

    // Compute residual: acceleration - target
    const Eigen::Vector3<T> residual = acceleration - targetAccelerations_[i];

    error += residual.squaredNorm() * targetWeights_[i];
  }

  return error * this->weight_;
}

template <typename T>
double AccelerationSequenceErrorFunctionT<T>::getGradient(
    std::span<const ModelParametersT<T>> /* modelParameters */,
    std::span<const SkeletonStateT<T>> skelStates,
    std::span<const MeshStateT<T>> /* meshStates */,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  MT_PROFILE_FUNCTION();

  MT_CHECK(skelStates.size() == 3);

  const auto& prevState = skelStates[0]; // t-1
  const auto& currState = skelStates[1]; // t
  const auto& nextState = skelStates[2]; // t+1

  const Eigen::Index nParam = this->parameterTransform_.numAllModelParameters();
  MT_CHECK(gradient.size() == 3 * nParam);

  double error = 0.0;

  // Coefficients for the finite difference stencil [-1, 2, -1]
  // Frame 0 (prevState): coefficient = 1 (contributes +1 to acceleration)
  // Frame 1 (currState): coefficient = -2 (contributes -2 to acceleration)
  // Frame 2 (nextState): coefficient = 1 (contributes +1 to acceleration)
  const std::array<T, 3> coefficients{T(1), T(-2), T(1)};

  for (size_t i = 0; i < this->skeleton_.joints.size(); ++i) {
    if (targetWeights_[i] == 0) {
      continue;
    }

    // Compute acceleration and residual
    const Eigen::Vector3<T> acceleration = nextState.jointState[i].translation() -
        T(2) * currState.jointState[i].translation() + prevState.jointState[i].translation();
    const Eigen::Vector3<T> residual = acceleration - targetAccelerations_[i];

    const T wgt = targetWeights_[i] * this->weight_;
    error += residual.squaredNorm() * wgt;

    // Gradient contribution from each frame
    // d/dp ||residual||^2 = 2 * residual^T * d(residual)/dp
    // d(residual)/dp = d(acceleration)/dp = coeff * d(pos)/dp

    const std::array<const SkeletonStateT<T>*, 3> states = {&prevState, &currState, &nextState};

    for (size_t frameIdx = 0; frameIdx < 3; ++frameIdx) {
      const auto& state = *states[frameIdx];
      const T coeff = coefficients[frameIdx];

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
size_t AccelerationSequenceErrorFunctionT<T>::getJacobianSize() const {
  // 3 rows per active joint (x, y, z acceleration components)
  const auto nActiveJoints = (targetWeights_.array() != 0).count();
  return nActiveJoints * 3;
}

template <typename T>
double AccelerationSequenceErrorFunctionT<T>::getJacobian(
    std::span<const ModelParametersT<T>> /* modelParameters */,
    std::span<const SkeletonStateT<T>> skelStates,
    std::span<const MeshStateT<T>> /* meshStates */,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    int& usedRows) const {
  MT_PROFILE_FUNCTION();

  MT_CHECK(skelStates.size() == 3);

  const auto& prevState = skelStates[0]; // t-1
  const auto& currState = skelStates[1]; // t
  const auto& nextState = skelStates[2]; // t+1

  const Eigen::Index nParam = this->parameterTransform_.numAllModelParameters();
  MT_CHECK(jacobian.cols() == 3 * nParam);

  double error = 0.0;
  Eigen::Index offset = 0;

  // Coefficients for the finite difference stencil
  const std::array<T, 3> coefficients{T(1), T(-2), T(1)};

  for (size_t i = 0; i < this->skeleton_.joints.size(); ++i) {
    if (targetWeights_[i] == 0) {
      continue;
    }

    // Compute acceleration and residual
    const Eigen::Vector3<T> acceleration = nextState.jointState[i].translation() -
        T(2) * currState.jointState[i].translation() + prevState.jointState[i].translation();
    const Eigen::Vector3<T> res = acceleration - targetAccelerations_[i];

    const T wgt = targetWeights_[i] * this->weight_;
    const T sqrtWgt = std::sqrt(wgt);

    error += res.squaredNorm() * wgt;
    residual.template segment<3>(offset).noalias() = res * sqrtWgt;

    const std::array<const SkeletonStateT<T>*, 3> states = {&prevState, &currState, &nextState};

    // Jacobian for each frame
    for (size_t frameIdx = 0; frameIdx < 3; ++frameIdx) {
      const auto& state = *states[frameIdx];
      const T coeff = coefficients[frameIdx] * sqrtWgt;

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

template class AccelerationSequenceErrorFunctionT<float>;
template class AccelerationSequenceErrorFunctionT<double>;

} // namespace momentum
