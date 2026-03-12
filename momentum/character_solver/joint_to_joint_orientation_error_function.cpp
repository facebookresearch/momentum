/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/joint_to_joint_orientation_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/math/constants.h"

namespace momentum {

template <typename T>
JointToJointOrientationErrorFunctionT<T>::JointToJointOrientationErrorFunctionT(
    const Skeleton& skel,
    const ParameterTransform& pt)
    : SkeletonErrorFunctionT<T>(skel, pt) {}

template <typename T>
JointToJointOrientationErrorFunctionT<T>::JointToJointOrientationErrorFunctionT(
    const Character& character)
    : JointToJointOrientationErrorFunctionT(character.skeleton, character.parameterTransform) {}

template <typename T>
void JointToJointOrientationErrorFunctionT<T>::addConstraint(
    size_t sourceJoint,
    size_t referenceJoint,
    const Eigen::Quaternion<T>& target,
    T weight,
    const std::string& name) {
  MT_CHECK(
      sourceJoint < this->skeleton_.joints.size(), "Invalid sourceJoint index: {}", sourceJoint);
  MT_CHECK(
      referenceJoint < this->skeleton_.joints.size(),
      "Invalid referenceJoint index: {}",
      referenceJoint);

  JointToJointOrientationDataT<T> constraint;
  constraint.sourceJoint = sourceJoint;
  constraint.referenceJoint = referenceJoint;
  constraint.target = target.normalized();
  constraint.weight = weight;
  constraint.name = name;
  constraints_.push_back(constraint);
}

template <typename T>
void JointToJointOrientationErrorFunctionT<T>::addConstraint(
    const JointToJointOrientationDataT<T>& constraint) {
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
void JointToJointOrientationErrorFunctionT<T>::clearConstraints() {
  constraints_.clear();
}

template <typename T>
double JointToJointOrientationErrorFunctionT<T>::getError(
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

    // Compute relative rotation: R_rel = R_ref^T * R_src
    const Matrix3<T> srcRot = srcJointState.rotation().toRotationMatrix();
    const Matrix3<T> refRot = refJointState.rotation().toRotationMatrix();
    const Matrix3<T> relativeRot = refRot.transpose() * srcRot;

    // Compute error: difference between relative rotation and target rotation
    const Matrix3<T> targetRot = constr.target.toRotationMatrix();
    const Matrix3<T> rotError = relativeRot - targetRot;

    error += constr.weight * rotError.squaredNorm() * this->weight_;
  }

  return error;
}

template <typename T>
double JointToJointOrientationErrorFunctionT<T>::getGradient(
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

    // Compute relative rotation: R_rel = R_ref^T * R_src
    const Matrix3<T> srcRot = srcJointState.rotation().toRotationMatrix();
    const Matrix3<T> refRot = refJointState.rotation().toRotationMatrix();
    const Matrix3<T> relativeRot = refRot.transpose() * srcRot;

    // Compute error
    const Matrix3<T> targetRot = constr.target.toRotationMatrix();
    const Matrix3<T> rotError = relativeRot - targetRot;
    error += constr.weight * rotError.squaredNorm() * this->weight_;

    const T wgt = T(2) * constr.weight * this->weight_;

    // The error is ||R_ref^T * R_src - R_target||_F^2
    // d(error)/dq = 2 * sum_ij (R_rel - R_target)_ij * d(R_rel)_ij/dq
    //
    // R_rel = R_ref^T * R_src, so each column of R_rel is:
    //   R_rel[:,k] = R_ref^T * R_src[:,k]
    //
    // This has the same structure as the position case but with 3 columns
    // of the rotation matrix instead of a single position vector.

    // Process source joint chain: d(R_ref^T * R_src)/d(src params)
    // = R_ref^T * d(R_src)/d(src params)
    // For each column k of R_src, the derivative with respect to rotation parameters
    // gives cross product with the column vector.
    for (int col = 0; col < 3; col++) {
      const Vector3<T> srcCol = srcRot.col(col);
      const Vector3<T> errCol = rotError.col(col);
      // worldGrad = R_ref * errCol transforms back to world frame
      const Vector3<T> worldGrad = refRot * errCol;

      // Process source joint chain (positive contribution)
      size_t currentJoint = constr.sourceJoint;
      while (currentJoint != kInvalidIndex) {
        const auto& currentJointState = state.jointState[currentJoint];
        const size_t paramIndex = currentJoint * kParametersPerJoint;

        // Only rotation derivatives affect the rotation matrix columns
        for (size_t d = 0; d < 3; d++) {
          if (this->activeJointParams_[paramIndex + 3 + d]) {
            const T val = worldGrad.dot(currentJointState.getRotationDerivative(d, srcCol)) * wgt;
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

      // Process reference joint chain: d(R_ref^T * R_src[:,col])/d(ref params)
      // This has two parts:
      // 1. d(R_ref^T)/d(ref params) * R_src[:,col] (rotation of reference frame)
      currentJoint = constr.referenceJoint;
      while (currentJoint != kInvalidIndex) {
        const auto& currentJointState = state.jointState[currentJoint];
        const size_t paramIndex = currentJoint * kParametersPerJoint;

        for (size_t d = 0; d < 3; d++) {
          if (this->activeJointParams_[paramIndex + 3 + d]) {
            // d(R_ref^T)/d(rot_i) * srcCol = -R_ref^T * [axis_i]_x * srcCol
            // The gradient contribution is: errCol^T * (-R_ref^T * [axis_i]_x * srcCol)
            // = -worldGrad^T * getRotationDerivative(d, srcCol)
            const T val = -worldGrad.dot(currentJointState.getRotationDerivative(d, srcCol)) * wgt;
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
    }
  }

  return error;
}

template <typename T>
double JointToJointOrientationErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& /* params */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */,
    Ref<MatrixX<T>> jacobian,
    Ref<VectorX<T>> residual,
    int& usedRows) {
  MT_PROFILE_FUNCTION();

  MT_CHECK(
      jacobian.rows() >= gsl::narrow<Eigen::Index>(9 * constraints_.size()),
      "Jacobian rows mismatch: Actual {}, Expected at least {}",
      jacobian.rows(),
      9 * constraints_.size());

  MT_CHECK(
      residual.rows() >= gsl::narrow<Eigen::Index>(9 * constraints_.size()),
      "Residual rows mismatch: Actual {}, Expected at least {}",
      residual.rows(),
      9 * constraints_.size());

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

    // Compute relative rotation
    const Matrix3<T> srcRot = srcJointState.rotation().toRotationMatrix();
    const Matrix3<T> refRot = refJointState.rotation().toRotationMatrix();
    const Matrix3<T> refRotT = refRot.transpose();
    const Matrix3<T> relativeRot = refRotT * srcRot;

    // Compute error
    const Matrix3<T> targetRot = constr.target.toRotationMatrix();
    const Matrix3<T> rotError = relativeRot - targetRot;
    error += constr.weight * rotError.squaredNorm() * this->weight_;

    const T fac = std::sqrt(constr.weight) * wgt;
    const int baseRow = pos;
    pos += 9;

    // Set residual: 9 elements from the rotation error matrix (column-major)
    for (int col = 0; col < 3; col++) {
      for (int row = 0; row < 3; row++) {
        residual(baseRow + col * 3 + row) = rotError(row, col) * fac;
      }
    }

    // Jacobian: for each column of the rotation matrix
    for (int col = 0; col < 3; col++) {
      const Vector3<T> srcCol = srcRot.col(col);

      // Source joint chain: d(R_ref^T * R_src[:,col])/d(src params)
      // = R_ref^T * d(R_src[:,col])/d(src params)
      size_t currentJoint = constr.sourceJoint;
      while (currentJoint != kInvalidIndex) {
        const auto& currentJointState = state.jointState[currentJoint];
        const size_t paramIndex = currentJoint * kParametersPerJoint;

        for (size_t d = 0; d < 3; d++) {
          if (this->activeJointParams_[paramIndex + 3 + d]) {
            // d(R_src[:,col])/d(rot_i) = getRotationDerivative(d, srcCol)
            const Vector3<T> dWorldCol = currentJointState.getRotationDerivative(d, srcCol);
            const Vector3<T> dRelativeCol = refRotT * dWorldCol;
            for (auto index =
                     this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d];
                 index <
                 this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d + 1];
                 ++index) {
              const auto paramCol = this->parameterTransform_.transform.innerIndexPtr()[index];
              const T coeff = this->parameterTransform_.transform.valuePtr()[index] * fac;
              for (int row = 0; row < 3; row++) {
                jacobian(baseRow + col * 3 + row, paramCol) += dRelativeCol(row) * coeff;
              }
            }
          }
        }

        currentJoint = this->skeleton_.joints[currentJoint].parent;
      }

      // Reference joint chain: d(R_ref^T * srcCol)/d(ref params)
      // = d(R_ref^T)/d(ref params) * srcCol
      currentJoint = constr.referenceJoint;
      while (currentJoint != kInvalidIndex) {
        const auto& currentJointState = state.jointState[currentJoint];
        const size_t paramIndex = currentJoint * kParametersPerJoint;

        for (size_t d = 0; d < 3; d++) {
          if (this->activeJointParams_[paramIndex + 3 + d]) {
            // d(R_ref^T)/d(rot_i) * srcCol = -R_ref^T * [axis_i]_x * srcCol
            const Vector3<T> dRotCol = currentJointState.getRotationDerivative(d, srcCol);
            const Vector3<T> dRelativeCol = -refRotT * dRotCol;
            for (auto index =
                     this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d];
                 index <
                 this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d + 1];
                 ++index) {
              const auto paramCol = this->parameterTransform_.transform.innerIndexPtr()[index];
              const T coeff = this->parameterTransform_.transform.valuePtr()[index] * fac;
              for (int row = 0; row < 3; row++) {
                jacobian(baseRow + col * 3 + row, paramCol) += dRelativeCol(row) * coeff;
              }
            }
          }
        }

        currentJoint = this->skeleton_.joints[currentJoint].parent;
      }
    }
  }

  usedRows = pos;
  return error;
}

template <typename T>
size_t JointToJointOrientationErrorFunctionT<T>::getJacobianSize() const {
  return 9 * constraints_.size();
}

template class JointToJointOrientationErrorFunctionT<float>;
template class JointToJointOrientationErrorFunctionT<double>;

} // namespace momentum
