/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_sequence_solver/velocity_magnitude_sequence_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character_solver/error_function_utils.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"

namespace momentum {

namespace {

// Small epsilon to avoid division by zero when velocity is very small
template <typename T>
constexpr T kVelocityEpsilon = T(1e-8);

} // namespace

template <typename T>
VelocityMagnitudeSequenceErrorFunctionT<T>::VelocityMagnitudeSequenceErrorFunctionT(
    const Skeleton& skel,
    const ParameterTransform& pt)
    : SequenceErrorFunctionT<T>(skel, pt) {
  targetWeights_.setOnes(skel.joints.size());
  targetSpeeds_.setZero(skel.joints.size());
}

template <typename T>
VelocityMagnitudeSequenceErrorFunctionT<T>::VelocityMagnitudeSequenceErrorFunctionT(
    const Character& character)
    : VelocityMagnitudeSequenceErrorFunctionT<T>(character.skeleton, character.parameterTransform) {
}

template <typename T>
void VelocityMagnitudeSequenceErrorFunctionT<T>::reset() {
  targetWeights_.setOnes(this->skeleton_.joints.size());
  targetSpeeds_.setZero(this->skeleton_.joints.size());
}

template <typename T>
void VelocityMagnitudeSequenceErrorFunctionT<T>::setTargetWeights(
    const Eigen::VectorX<T>& weights) {
  MT_CHECK(weights.size() == static_cast<Eigen::Index>(this->skeleton_.joints.size()));
  targetWeights_ = weights;
}

template <typename T>
void VelocityMagnitudeSequenceErrorFunctionT<T>::setTargetSpeed(T speed) {
  targetSpeeds_.setConstant(this->skeleton_.joints.size(), speed);
}

template <typename T>
void VelocityMagnitudeSequenceErrorFunctionT<T>::setTargetSpeeds(const Eigen::VectorX<T>& speeds) {
  MT_CHECK(speeds.size() == static_cast<Eigen::Index>(this->skeleton_.joints.size()));
  targetSpeeds_ = speeds;
}

template <typename T>
double VelocityMagnitudeSequenceErrorFunctionT<T>::getError(
    std::span<const ModelParametersT<T>> /* modelParameters */,
    std::span<const SkeletonStateT<T>> skelStates,
    std::span<const MeshStateT<T>> /* meshStates */) const {
  MT_PROFILE_FUNCTION();

  MT_CHECK(skelStates.size() == 2);

  const auto& prevState = skelStates[0];
  const auto& nextState = skelStates[1];

  double error = 0.0;

  for (size_t i = 0; i < this->skeleton_.joints.size(); ++i) {
    if (targetWeights_[i] == 0) {
      continue;
    }

    // Compute velocity: v = pos[t+1] - pos[t]
    const Eigen::Vector3<T> velocity =
        nextState.jointState[i].translation() - prevState.jointState[i].translation();

    // Compute velocity magnitude
    const T speed = velocity.norm();

    // Compute residual: speed - targetSpeed
    const T residual = speed - targetSpeeds_[i];

    error += residual * residual * targetWeights_[i];
  }

  return error * this->weight_;
}

template <typename T>
double VelocityMagnitudeSequenceErrorFunctionT<T>::getGradient(
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

  double error = 0.0;

  for (size_t i = 0; i < this->skeleton_.joints.size(); ++i) {
    if (targetWeights_[i] == 0) {
      continue;
    }

    // Compute velocity: v = pos[t+1] - pos[t]
    const Eigen::Vector3<T> velocity =
        nextState.jointState[i].translation() - prevState.jointState[i].translation();

    // Compute velocity magnitude
    const T speed = velocity.norm();

    // Compute residual: speed - targetSpeed
    const T residual = speed - targetSpeeds_[i];

    const T wgt = targetWeights_[i] * this->weight_;
    error += residual * residual * wgt;

    // Compute gradient direction: d(||v||)/d(v) = v / ||v|| (unit direction)
    // When speed is very small, the gradient direction is undefined, so we skip
    if (speed < kVelocityEpsilon<T>) {
      continue;
    }

    // d(error)/d(pos) = 2 * (||v|| - s) * (v / ||v||) * d(v)/d(pos)
    // d(v)/d(pos0) = -I, d(v)/d(pos1) = I
    const Eigen::Vector3<T> gradDir = velocity / speed;
    const T gradScale = T(2) * wgt * residual;

    // Gradient for prevState (frame 0) - negative because d(v)/d(pos0) = -I
    {
      size_t ancestorJointIndex = i;
      while (ancestorJointIndex != kInvalidIndex) {
        MT_CHECK(ancestorJointIndex < this->skeleton_.joints.size());

        const size_t paramIndex = ancestorJointIndex * kParametersPerJoint;
        const Eigen::Vector3<T> posd = prevState.jointState[i].translation() -
            prevState.jointState[ancestorJointIndex].translation();

        // Translation parameters (3)
        for (size_t d = 0; d < 3; d++) {
          if (this->activeJointParams_[paramIndex + d]) {
            const T val = -gradScale *
                gradDir.dot(prevState.jointState[ancestorJointIndex].getTranslationDerivative(d));
            gradient_jointParams_to_modelParams<T>(
                val, paramIndex + d, this->parameterTransform_, gradient.segment(0, nParam));
          }
        }

        // Rotation parameters (3)
        for (size_t d = 0; d < 3; d++) {
          if (this->activeJointParams_[paramIndex + 3 + d]) {
            const T val = -gradScale *
                gradDir.dot(
                    prevState.jointState[ancestorJointIndex].getRotationDerivative(d, posd));
            gradient_jointParams_to_modelParams<T>(
                val, paramIndex + 3 + d, this->parameterTransform_, gradient.segment(0, nParam));
          }
        }

        // Scale parameter (1)
        if (this->activeJointParams_[paramIndex + 6]) {
          const T val = -gradScale *
              gradDir.dot(prevState.jointState[ancestorJointIndex].getScaleDerivative(posd));
          gradient_jointParams_to_modelParams<T>(
              val, paramIndex + 6, this->parameterTransform_, gradient.segment(0, nParam));
        }

        ancestorJointIndex = this->skeleton_.joints[ancestorJointIndex].parent;
      }
    }

    // Gradient for nextState (frame 1) - positive because d(v)/d(pos1) = I
    {
      size_t ancestorJointIndex = i;
      while (ancestorJointIndex != kInvalidIndex) {
        MT_CHECK(ancestorJointIndex < this->skeleton_.joints.size());

        const size_t paramIndex = ancestorJointIndex * kParametersPerJoint;
        const Eigen::Vector3<T> posd = nextState.jointState[i].translation() -
            nextState.jointState[ancestorJointIndex].translation();

        // Translation parameters (3)
        for (size_t d = 0; d < 3; d++) {
          if (this->activeJointParams_[paramIndex + d]) {
            const T val = gradScale *
                gradDir.dot(nextState.jointState[ancestorJointIndex].getTranslationDerivative(d));
            gradient_jointParams_to_modelParams<T>(
                val, paramIndex + d, this->parameterTransform_, gradient.segment(nParam, nParam));
          }
        }

        // Rotation parameters (3)
        for (size_t d = 0; d < 3; d++) {
          if (this->activeJointParams_[paramIndex + 3 + d]) {
            const T val = gradScale *
                gradDir.dot(
                    nextState.jointState[ancestorJointIndex].getRotationDerivative(d, posd));
            gradient_jointParams_to_modelParams<T>(
                val,
                paramIndex + 3 + d,
                this->parameterTransform_,
                gradient.segment(nParam, nParam));
          }
        }

        // Scale parameter (1)
        if (this->activeJointParams_[paramIndex + 6]) {
          const T val = gradScale *
              gradDir.dot(nextState.jointState[ancestorJointIndex].getScaleDerivative(posd));
          gradient_jointParams_to_modelParams<T>(
              val, paramIndex + 6, this->parameterTransform_, gradient.segment(nParam, nParam));
        }

        ancestorJointIndex = this->skeleton_.joints[ancestorJointIndex].parent;
      }
    }
  }

  return error;
}

template <typename T>
size_t VelocityMagnitudeSequenceErrorFunctionT<T>::getJacobianSize() const {
  // 1 row per active joint (scalar speed residual)
  const auto nActiveJoints = (targetWeights_.array() != 0).count();
  return nActiveJoints;
}

template <typename T>
double VelocityMagnitudeSequenceErrorFunctionT<T>::getJacobian(
    std::span<const ModelParametersT<T>> /* modelParameters */,
    std::span<const SkeletonStateT<T>> skelStates,
    std::span<const MeshStateT<T>> /* meshStates */,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    int& usedRows) const {
  MT_PROFILE_FUNCTION();

  MT_CHECK(skelStates.size() == 2);

  const auto& prevState = skelStates[0];
  const auto& nextState = skelStates[1];

  const Eigen::Index nParam = this->parameterTransform_.numAllModelParameters();
  MT_CHECK(jacobian.cols() == 2 * nParam);

  double error = 0.0;
  Eigen::Index offset = 0;

  // Helper lambda to accumulate jacobian value for a given joint parameter
  const auto accumulateJacobian = [&](size_t paramIdx, Eigen::Index colOffset, T jc) {
    const auto& transform = this->parameterTransform_.transform;
    for (auto index = transform.outerIndexPtr()[paramIdx];
         index < transform.outerIndexPtr()[paramIdx + 1];
         ++index) {
      jacobian(offset, colOffset + transform.innerIndexPtr()[index]) +=
          jc * transform.valuePtr()[index];
    }
  };

  // Helper lambda to process jacobian contributions for one frame
  const auto processFrame = [&](const SkeletonStateT<T>& state,
                                size_t jointIdx,
                                Eigen::Index colOffset,
                                T sign,
                                T sqrtWgt,
                                const Eigen::Vector3<T>& gradDir) {
    size_t ancestorJointIndex = jointIdx;
    while (ancestorJointIndex != kInvalidIndex) {
      MT_CHECK(ancestorJointIndex < this->skeleton_.joints.size());

      const auto& jointState = state.jointState[ancestorJointIndex];
      const size_t paramIndex = ancestorJointIndex * kParametersPerJoint;
      const Eigen::Vector3<T> posd =
          state.jointState[jointIdx].translation() - jointState.translation();

      // Translation parameters (3)
      for (size_t d = 0; d < 3; d++) {
        if (this->activeJointParams_[paramIndex + d]) {
          const T jc = sign * sqrtWgt * gradDir.dot(jointState.getTranslationDerivative(d));
          accumulateJacobian(paramIndex + d, colOffset, jc);
        }
      }

      // Rotation parameters (3)
      for (size_t d = 0; d < 3; d++) {
        if (this->activeJointParams_[paramIndex + 3 + d]) {
          const T jc = sign * sqrtWgt * gradDir.dot(jointState.getRotationDerivative(d, posd));
          accumulateJacobian(paramIndex + 3 + d, colOffset, jc);
        }
      }

      // Scale parameter (1)
      if (this->activeJointParams_[paramIndex + 6]) {
        const T jc = sign * sqrtWgt * gradDir.dot(jointState.getScaleDerivative(posd));
        accumulateJacobian(paramIndex + 6, colOffset, jc);
      }

      ancestorJointIndex = this->skeleton_.joints[ancestorJointIndex].parent;
    }
  };

  for (size_t i = 0; i < this->skeleton_.joints.size(); ++i) {
    if (targetWeights_[i] == 0) {
      continue;
    }

    // Compute velocity: v = pos[t+1] - pos[t]
    const Eigen::Vector3<T> velocity =
        nextState.jointState[i].translation() - prevState.jointState[i].translation();

    // Compute velocity magnitude
    const T speed = velocity.norm();

    // Compute residual: speed - targetSpeed
    const T res = speed - targetSpeeds_[i];

    const T wgt = targetWeights_[i] * this->weight_;
    const T sqrtWgt = std::sqrt(wgt);

    error += res * res * wgt;
    residual(offset) = res * sqrtWgt;

    // Compute Jacobian direction: d(||v||)/d(v) = v / ||v||
    // When speed is very small, the Jacobian is zero (or undefined)
    Eigen::Vector3<T> gradDir = Eigen::Vector3<T>::Zero();
    if (speed >= kVelocityEpsilon<T>) {
      gradDir = velocity / speed;
    }

    // Jacobian for prevState (frame 0) - negative because d(v)/d(pos0) = -I
    processFrame(prevState, i, 0, T(-1), sqrtWgt, gradDir);

    // Jacobian for nextState (frame 1) - positive because d(v)/d(pos1) = I
    processFrame(nextState, i, nParam, T(1), sqrtWgt, gradDir);

    offset += 1;
  }

  usedRows = gsl::narrow_cast<int>(offset);

  return error;
}

template class VelocityMagnitudeSequenceErrorFunctionT<float>;
template class VelocityMagnitudeSequenceErrorFunctionT<double>;

} // namespace momentum
