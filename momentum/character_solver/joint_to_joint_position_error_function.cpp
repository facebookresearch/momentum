/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/joint_to_joint_position_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/math/constants.h"

namespace momentum {

template <typename T>
JointToJointPositionErrorFunctionT<T>::JointToJointPositionErrorFunctionT(
    const Skeleton& skel,
    const ParameterTransform& pt)
    : SkeletonErrorFunctionT<T>(skel, pt) {}

template <typename T>
JointToJointPositionErrorFunctionT<T>::JointToJointPositionErrorFunctionT(
    const Character& character)
    : JointToJointPositionErrorFunctionT(character.skeleton, character.parameterTransform) {}

template <typename T>
void JointToJointPositionErrorFunctionT<T>::addConstraint(
    size_t sourceJoint,
    const Vector3<T>& sourceOffset,
    size_t referenceJoint,
    const Vector3<T>& referenceOffset,
    const Vector3<T>& target,
    T weight,
    const std::string& name) {
  MT_CHECK(
      sourceJoint < this->skeleton_.joints.size(), "Invalid sourceJoint index: {}", sourceJoint);
  MT_CHECK(
      referenceJoint < this->skeleton_.joints.size(),
      "Invalid referenceJoint index: {}",
      referenceJoint);

  JointToJointPositionDataT<T> constraint;
  constraint.sourceJoint = sourceJoint;
  constraint.sourceOffset = sourceOffset;
  constraint.referenceJoint = referenceJoint;
  constraint.referenceOffset = referenceOffset;
  constraint.target = target;
  constraint.weight = weight;
  constraint.name = name;
  constraints_.push_back(constraint);
}

template <typename T>
void JointToJointPositionErrorFunctionT<T>::addConstraint(
    const JointToJointPositionDataT<T>& constraint) {
  MT_CHECK(
      constraint.sourceJoint < this->skeleton_.joints.size(),
      "Invalid sourceJoint index: {}",
      constraint.sourceJoint);
  MT_CHECK(
      constraint.referenceJoint < this->skeleton_.joints.size(),
      "Invalid referenceJoint index: {}",
      constraint.referenceJoint);
  constraints_.push_back(constraint);
}

template <typename T>
void JointToJointPositionErrorFunctionT<T>::clearConstraints() {
  constraints_.clear();
}

template <typename T>
double JointToJointPositionErrorFunctionT<T>::getError(
    const ModelParametersT<T>& /* params */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */) {
  if (state.jointState.empty() || constraints_.empty()) {
    return 0.0;
  }

  double error = 0.0;

  for (const auto& constr : constraints_) {
    const auto& srcJointState = state.jointState[constr.sourceJoint];
    const auto& refJointState = state.jointState[constr.referenceJoint];

    // Compute world positions
    const Vector3<T> srcWorldPos = srcJointState.transform * constr.sourceOffset;
    const Vector3<T> refWorldPos = refJointState.transform * constr.referenceOffset;

    // Transform source position to reference frame
    // relativePos = R_ref^T * (srcWorldPos - refWorldPos)
    const Vector3<T> diff = srcWorldPos - refWorldPos;
    const Vector3<T> relativePos = refJointState.transform.toRotationMatrix().transpose() * diff;

    // Compute error: difference between relative position and target
    const Vector3<T> posError = relativePos - constr.target;

    error += constr.weight * posError.squaredNorm() * this->weight_;
  }

  return error;
}

template <typename T>
double JointToJointPositionErrorFunctionT<T>::getGradient(
    const ModelParametersT<T>& /* params */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */,
    Ref<VectorX<T>> gradient) {
  MT_PROFILE_FUNCTION();

  if (state.jointState.empty() || constraints_.empty()) {
    return 0.0;
  }

  double error = 0.0;

  for (const auto& constr : constraints_) {
    const auto& srcJointState = state.jointState[constr.sourceJoint];
    const auto& refJointState = state.jointState[constr.referenceJoint];

    // Compute world positions
    const Vector3<T> srcWorldPos = srcJointState.transform * constr.sourceOffset;
    const Vector3<T> refWorldPos = refJointState.transform * constr.referenceOffset;

    // Transform source position to reference frame
    const Vector3<T> diff = srcWorldPos - refWorldPos;
    const Matrix3<T> refRotT = refJointState.transform.toRotationMatrix().transpose();
    const Vector3<T> relativePos = refRotT * diff;

    // Compute error
    const Vector3<T> posError = relativePos - constr.target;
    error += constr.weight * posError.squaredNorm() * this->weight_;

    // Gradient computation
    // d(error)/dq = 2 * w * posError^T * d(relativePos)/dq
    // relativePos = R_ref^T * (srcWorldPos - refWorldPos)
    // d(relativePos)/dq = R_ref^T * d(srcWorldPos)/dq
    //                   - R_ref^T * d(refWorldPos)/dq
    //                   + d(R_ref^T)/dq * (srcWorldPos - refWorldPos)

    const T wgt = T(2) * constr.weight * this->weight_;
    // Transform error back to world frame for gradient computation
    const Vector3<T> worldGrad = refJointState.transform.toRotationMatrix() * posError;

    // Process source joint chain (positive contribution)
    auto processSourceJoint = [&](size_t jointIndex, const Vector3<T>& worldPos) {
      size_t currentJoint = jointIndex;
      while (currentJoint != kInvalidIndex) {
        const auto& currentJointState = state.jointState[currentJoint];
        const size_t paramIndex = currentJoint * kParametersPerJoint;
        const Vector3<T> posd = worldPos - currentJointState.translation();

        // Translation derivatives
        for (size_t d = 0; d < 3; d++) {
          if (this->activeJointParams_[paramIndex + d]) {
            const T val = worldGrad.dot(currentJointState.getTranslationDerivative(d)) * wgt;
            for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
                 index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
                 ++index) {
              gradient[this->parameterTransform_.transform.innerIndexPtr()[index]] +=
                  val * this->parameterTransform_.transform.valuePtr()[index];
            }
          }
          // Rotation derivatives
          if (this->activeJointParams_[paramIndex + 3 + d]) {
            const T val = worldGrad.dot(currentJointState.getRotationDerivative(d, posd)) * wgt;
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
        // Scale derivative
        if (this->activeJointParams_[paramIndex + 6]) {
          const T val = worldGrad.dot(currentJointState.getScaleDerivative(posd)) * wgt;
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

    // Process reference joint chain (negative contribution for translation, plus rotation effect)
    auto processReferenceJoint = [&](size_t jointIndex, const Vector3<T>& worldPos) {
      size_t currentJoint = jointIndex;
      while (currentJoint != kInvalidIndex) {
        const auto& currentJointState = state.jointState[currentJoint];
        const size_t paramIndex = currentJoint * kParametersPerJoint;
        const Vector3<T> posd = worldPos - currentJointState.translation();

        // Translation derivatives (negative contribution)
        for (size_t d = 0; d < 3; d++) {
          if (this->activeJointParams_[paramIndex + d]) {
            const T val = -worldGrad.dot(currentJointState.getTranslationDerivative(d)) * wgt;
            for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
                 index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
                 ++index) {
              gradient[this->parameterTransform_.transform.innerIndexPtr()[index]] +=
                  val * this->parameterTransform_.transform.valuePtr()[index];
            }
          }
          // Rotation derivatives (negative contribution from translation part)
          if (this->activeJointParams_[paramIndex + 3 + d]) {
            const T val = -worldGrad.dot(currentJointState.getRotationDerivative(d, posd)) * wgt;
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
        // Scale derivative (negative contribution)
        if (this->activeJointParams_[paramIndex + 6]) {
          const T val = -worldGrad.dot(currentJointState.getScaleDerivative(posd)) * wgt;
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

    // Additional gradient from rotation of reference frame
    // d(R_ref^T * diff)/dR_ref contributes to joints in reference chain
    auto processReferenceRotation = [&](size_t jointIndex) {
      size_t currentJoint = jointIndex;
      while (currentJoint != kInvalidIndex) {
        const auto& currentJointState = state.jointState[currentJoint];
        const size_t paramIndex = currentJoint * kParametersPerJoint;

        // Only rotation derivatives affect R_ref
        for (size_t d = 0; d < 3; d++) {
          if (this->activeJointParams_[paramIndex + 3 + d]) {
            // d(R_ref^T)/d(rot_angle_i) * diff
            // = ([axis_i]_x R_ref)^T * diff
            // = -R_ref^T * [axis_i]_x * diff
            // where [axis_i]_x is the skew-symmetric matrix of the rotation axis i
            const T val = -worldGrad.dot(currentJointState.getRotationDerivative(d, diff)) * wgt;
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

        currentJoint = this->skeleton_.joints[currentJoint].parent;
      }
    };

    processSourceJoint(constr.sourceJoint, srcWorldPos);
    processReferenceJoint(constr.referenceJoint, refWorldPos);
    processReferenceRotation(constr.referenceJoint);
  }

  return error;
}

template <typename T>
double JointToJointPositionErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& /* params */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */,
    Ref<MatrixX<T>> jacobian,
    Ref<VectorX<T>> residual,
    int& usedRows) {
  MT_PROFILE_FUNCTION();

  MT_CHECK(
      jacobian.rows() >= gsl::narrow<Eigen::Index>(3 * constraints_.size()),
      "Jacobian rows mismatch: Actual {}, Expected at least {}",
      jacobian.rows(),
      3 * constraints_.size());

  MT_CHECK(
      residual.rows() >= gsl::narrow<Eigen::Index>(3 * constraints_.size()),
      "Residual rows mismatch: Actual {}, Expected at least {}",
      residual.rows(),
      3 * constraints_.size());

  if (state.jointState.empty() || constraints_.empty()) {
    usedRows = 0;
    return 0.0;
  }

  const T wgt = std::sqrt(this->weight_);
  int pos = 0;
  double error = 0.0;

  for (const auto& constr : constraints_) {
    const auto& srcJointState = state.jointState[constr.sourceJoint];
    const auto& refJointState = state.jointState[constr.referenceJoint];

    // Compute world positions
    const Vector3<T> srcWorldPos = srcJointState.transform * constr.sourceOffset;
    const Vector3<T> refWorldPos = refJointState.transform * constr.referenceOffset;

    // Transform source position to reference frame
    const Vector3<T> diff = srcWorldPos - refWorldPos;
    const Matrix3<T> refRotT = refJointState.transform.toRotationMatrix().transpose();
    const Vector3<T> relativePos = refRotT * diff;

    // Compute error
    const Vector3<T> posError = relativePos - constr.target;
    error += constr.weight * posError.squaredNorm() * this->weight_;

    const T fac = std::sqrt(constr.weight) * wgt;
    const int baseRow = pos;
    pos += 3;

    // Set residual
    for (int d = 0; d < 3; d++) {
      residual(baseRow + d) = posError(d) * fac;
    }

    // Jacobian from source joint chain
    auto processSourceJoint = [&](size_t jointIndex, const Vector3<T>& worldPos) {
      size_t currentJoint = jointIndex;
      while (currentJoint != kInvalidIndex) {
        const auto& currentJointState = state.jointState[currentJoint];
        const size_t paramIndex = currentJoint * kParametersPerJoint;
        const Vector3<T> posd = worldPos - currentJointState.translation();

        for (size_t d = 0; d < 3; d++) {
          if (this->activeJointParams_[paramIndex + d]) {
            // d(relativePos)/d(trans) = R_ref^T * d(srcWorldPos)/d(trans)
            const Vector3<T> dWorldPos = currentJointState.getTranslationDerivative(d);
            const Vector3<T> dRelativePos = refRotT * dWorldPos;
            for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
                 index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
                 ++index) {
              const auto paramCol = this->parameterTransform_.transform.innerIndexPtr()[index];
              const T coeff = this->parameterTransform_.transform.valuePtr()[index] * fac;
              for (int row = 0; row < 3; row++) {
                jacobian(baseRow + row, paramCol) += dRelativePos(row) * coeff;
              }
            }
          }
          if (this->activeJointParams_[paramIndex + 3 + d]) {
            const Vector3<T> dWorldPos = currentJointState.getRotationDerivative(d, posd);
            const Vector3<T> dRelativePos = refRotT * dWorldPos;
            for (auto index =
                     this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d];
                 index <
                 this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d + 1];
                 ++index) {
              const auto paramCol = this->parameterTransform_.transform.innerIndexPtr()[index];
              const T coeff = this->parameterTransform_.transform.valuePtr()[index] * fac;
              for (int row = 0; row < 3; row++) {
                jacobian(baseRow + row, paramCol) += dRelativePos(row) * coeff;
              }
            }
          }
        }
        if (this->activeJointParams_[paramIndex + 6]) {
          const Vector3<T> dWorldPos = currentJointState.getScaleDerivative(posd);
          const Vector3<T> dRelativePos = refRotT * dWorldPos;
          for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6];
               index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6 + 1];
               ++index) {
            const auto paramCol = this->parameterTransform_.transform.innerIndexPtr()[index];
            const T coeff = this->parameterTransform_.transform.valuePtr()[index] * fac;
            for (int row = 0; row < 3; row++) {
              jacobian(baseRow + row, paramCol) += dRelativePos(row) * coeff;
            }
          }
        }

        currentJoint = this->skeleton_.joints[currentJoint].parent;
      }
    };

    // Jacobian from reference joint chain (translation part - negative)
    auto processReferenceJoint = [&](size_t jointIndex, const Vector3<T>& worldPos) {
      size_t currentJoint = jointIndex;
      while (currentJoint != kInvalidIndex) {
        const auto& currentJointState = state.jointState[currentJoint];
        const size_t paramIndex = currentJoint * kParametersPerJoint;
        const Vector3<T> posd = worldPos - currentJointState.translation();

        for (size_t d = 0; d < 3; d++) {
          if (this->activeJointParams_[paramIndex + d]) {
            const Vector3<T> dWorldPos = currentJointState.getTranslationDerivative(d);
            const Vector3<T> dRelativePos = -refRotT * dWorldPos;
            for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
                 index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
                 ++index) {
              const auto paramCol = this->parameterTransform_.transform.innerIndexPtr()[index];
              const T coeff = this->parameterTransform_.transform.valuePtr()[index] * fac;
              for (int row = 0; row < 3; row++) {
                jacobian(baseRow + row, paramCol) += dRelativePos(row) * coeff;
              }
            }
          }
          if (this->activeJointParams_[paramIndex + 3 + d]) {
            const Vector3<T> dWorldPos = currentJointState.getRotationDerivative(d, posd);
            const Vector3<T> dRelativePos = -refRotT * dWorldPos;
            for (auto index =
                     this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d];
                 index <
                 this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d + 1];
                 ++index) {
              const auto paramCol = this->parameterTransform_.transform.innerIndexPtr()[index];
              const T coeff = this->parameterTransform_.transform.valuePtr()[index] * fac;
              for (int row = 0; row < 3; row++) {
                jacobian(baseRow + row, paramCol) += dRelativePos(row) * coeff;
              }
            }
          }
        }
        if (this->activeJointParams_[paramIndex + 6]) {
          const Vector3<T> dWorldPos = currentJointState.getScaleDerivative(posd);
          const Vector3<T> dRelativePos = -refRotT * dWorldPos;
          for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6];
               index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6 + 1];
               ++index) {
            const auto paramCol = this->parameterTransform_.transform.innerIndexPtr()[index];
            const T coeff = this->parameterTransform_.transform.valuePtr()[index] * fac;
            for (int row = 0; row < 3; row++) {
              jacobian(baseRow + row, paramCol) += dRelativePos(row) * coeff;
            }
          }
        }

        currentJoint = this->skeleton_.joints[currentJoint].parent;
      }
    };

    // Jacobian from reference joint rotation (affects R_ref^T)
    auto processReferenceRotation = [&](size_t jointIndex) {
      size_t currentJoint = jointIndex;
      while (currentJoint != kInvalidIndex) {
        const auto& currentJointState = state.jointState[currentJoint];
        const size_t paramIndex = currentJoint * kParametersPerJoint;

        for (size_t d = 0; d < 3; d++) {
          if (this->activeJointParams_[paramIndex + 3 + d]) {
            // d(R_ref^T)/d(rot_angle_i) * diff
            // = ([axis_i]_x R_ref)^T * diff
            // = -R_ref^T * [axis_i]_x * diff
            // where [axis_i]_x is the skew-symmetric matrix of the rotation axis i
            const Vector3<T> dRotDiff = currentJointState.getRotationDerivative(d, diff);
            const Vector3<T> dRelativePos = -refRotT * dRotDiff;
            for (auto index =
                     this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d];
                 index <
                 this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d + 1];
                 ++index) {
              const auto paramCol = this->parameterTransform_.transform.innerIndexPtr()[index];
              const T coeff = this->parameterTransform_.transform.valuePtr()[index] * fac;
              for (int row = 0; row < 3; row++) {
                jacobian(baseRow + row, paramCol) += dRelativePos(row) * coeff;
              }
            }
          }
        }

        currentJoint = this->skeleton_.joints[currentJoint].parent;
      }
    };

    processSourceJoint(constr.sourceJoint, srcWorldPos);
    processReferenceJoint(constr.referenceJoint, refWorldPos);
    processReferenceRotation(constr.referenceJoint);
  }

  usedRows = pos;
  return error;
}

template <typename T>
size_t JointToJointPositionErrorFunctionT<T>::getJacobianSize() const {
  return 3 * constraints_.size();
}

template class JointToJointPositionErrorFunctionT<float>;
template class JointToJointPositionErrorFunctionT<double>;

} // namespace momentum
