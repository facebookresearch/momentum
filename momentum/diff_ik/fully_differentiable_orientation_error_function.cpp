/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/diff_ik/fully_differentiable_orientation_error_function.h"

#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/common/checks.h>
#include <momentum/diff_ik/ceres_utility.h>
#include <momentum/math/utility.h>

#include <algorithm>

namespace momentum {

template <typename T>
FullyDifferentiableOrientationErrorFunctionT<T>::FullyDifferentiableOrientationErrorFunctionT(
    const Skeleton& skel,
    const ParameterTransform& pt)
    : SkeletonErrorFunctionT<T>(skel, pt) {
  // Empty
}

template <typename T>
std::vector<std::string> FullyDifferentiableOrientationErrorFunctionT<T>::inputs() const {
  static std::vector<std::string> result = {kWeights, kOffsets, kTargets};
  return result;
}

template <typename T>
Eigen::Index FullyDifferentiableOrientationErrorFunctionT<T>::getInputSize(
    const std::string& name) const {
  if (name == kWeights) {
    return constraints_.size();
  }

  if (name == kOffsets) {
    return 4 * constraints_.size();
  }

  if (name == kTargets) {
    return 4 * constraints_.size();
  }

  MT_THROW("Unknown input to FullyDifferentiableOrientationErrorFunctionT<T>::getInput: {}", name);
}

template <typename T>
double FullyDifferentiableOrientationErrorFunctionT<T>::getError(
    const ModelParametersT<T>& /*params*/,
    const SkeletonStateT<T>& state) {
  // loop over all constraints and calculate the error
  double error = 0.0;

  for (size_t i = 0; i < constraints_.size(); ++i) {
    const OrientationConstraintT<T>& ct = constraints_[i];

    // calculate orientation error
    // Note: |R0 - RT|² is a valid norm on SO3, it doesn't have the same slope as the squared
    // angle difference
    //       but it's derivative doesn't have a singularity at the minimum, so is more stable
    const Eigen::Quaternion<T> rot = state.jointState[ct.parent].rotation() * ct.offset;
    const Eigen::Matrix3<T> rotDiff = rot.toRotationMatrix() - ct.target.toRotationMatrix();
    error += ct.weight * rotDiff.squaredNorm() * kOrientationWeight;
  }

  // return error
  return error * this->weight_;
}

template <typename T>
double FullyDifferentiableOrientationErrorFunctionT<T>::getGradient(
    const ModelParametersT<T>& /*params*/,
    const SkeletonStateT<T>& state,
    Ref<Eigen::VectorX<T>> gradient) {
  const auto& parameterTransform = this->parameterTransform_;

  // storage for joint gradients
  jointGrad_.setZero(parameterTransform.numAllModelParameters());

  // loop over all constraints and calculate the error
  double error = 0.0;

  for (size_t i = 0; i < constraints_.size(); ++i) {
    const OrientationConstraintT<T>& ct = constraints_[i];
    if (ct.weight == 0) {
      continue;
    }

    error += calculateOrientationGradient(state, constraints_[i], jointGrad_);
  }

  // convert joint gradient to parameter gradient and return
  gradient.noalias() += jointGrad_ * this->weight_;

  // return error
  return error * this->weight_;
}

template <typename T>
double FullyDifferentiableOrientationErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& state,
    Ref<Eigen::MatrixX<T>> jacobian,
    Ref<Eigen::VectorX<T>> residual,
    int& usedRows) {
  // loop over all constraints and calculate the error
  double error = 0.0;
  std::atomic_int offset(0);

  for (size_t i = 0; i < constraints_.size(); ++i) {
    const OrientationConstraintT<T>& ct = constraints_[i];
    if (ct.weight == 0) {
      continue;
    }

    const auto currentOffset = offset.fetch_add(9, std::memory_order_relaxed);
    error += calculateOrientationJacobian(
        state,
        constraints_[i],
        jacobian.block(currentOffset, 0, 9, params.size()),
        residual.middleRows(currentOffset, 9));
  }

  usedRows = gsl::narrow_cast<int>(jacobian.rows());

  return error;
}

template <typename T>
T FullyDifferentiableOrientationErrorFunctionT<T>::calculateOrientationJacobian(
    const SkeletonStateT<T>& state,
    const OrientationConstraintT<T>& constr,
    Ref<Eigen::MatrixX<T>> jac,
    Ref<Eigen::VectorX<T>> res) const {
  const auto& parameterTransform = this->parameterTransform_;

  // calculate orientation error
  const Eigen::Quaternion<T> rot = state.jointState[constr.parent].rotation() * constr.offset;
  const Eigen::Quaternion<T>& target = constr.target;
  const Eigen::Matrix3<T> rotDiff = rot.toRotationMatrix() - target.toRotationMatrix();
  const T wgt = kOrientationWeight * constr.weight * this->weight_;
  const T swgt = std::sqrt(wgt);

  // loop over all joints the constraint is attached to and calculate gradient
  size_t jointIndex = constr.parent;
  while (jointIndex != kInvalidIndex) {
    // check for valid index
    MT_CHECK(jointIndex < this->skeleton_.joints.size());

    const size_t paramIndex = jointIndex * kParametersPerJoint;

    // calculate derivatives based on active joints
    for (size_t d = 0; d < 3; d++) {
      if (this->activeJointParams_[paramIndex + 3 + d]) {
        const Eigen::Vector3<T> axis = state.jointState[jointIndex].rotationAxis.col(d);
        const Eigen::Matrix3<T> rotD = crossProductMatrix(axis) * rot * swgt;
        const auto ja = Map<const Eigen::VectorX<T>>(rotD.data(), rotD.size());

        // explicitly multiply with the parameter transform to generate parameter space gradients
        for (auto index = parameterTransform.transform.outerIndexPtr()[paramIndex + 3 + d];
             index < parameterTransform.transform.outerIndexPtr()[paramIndex + d + 3 + 1];
             ++index) {
          jac.col(parameterTransform.transform.innerIndexPtr()[index]) +=
              ja * parameterTransform.transform.valuePtr()[index];
        }
      }
    }

    // go to the next joint
    jointIndex = this->skeleton_.joints[jointIndex].parent;
  }

  res = Map<const Eigen::VectorX<T>>(rotDiff.data(), rotDiff.size()) * swgt;

  return rotDiff.squaredNorm() * wgt;
}

template <typename T>
size_t FullyDifferentiableOrientationErrorFunctionT<T>::getJacobianSize() const {
  return constraints_.size() * 9;
}

template <typename T>
void FullyDifferentiableOrientationErrorFunctionT<T>::addConstraint(
    const OrientationConstraintT<T>& constraint) {
  constraints_.push_back(constraint);
}

template <typename T>
void FullyDifferentiableOrientationErrorFunctionT<T>::getInputImp(
    const std::string& name,
    Eigen::Ref<Eigen::VectorX<T>> value) const {
  if (name == kWeights) {
    for (size_t i = 0; i < constraints_.size(); ++i) {
      value[i] = static_cast<T>(constraints_[i].weight);
    }
  } else if (name == kOffsets) {
    for (size_t i = 0; i < constraints_.size(); ++i) {
      value.template segment<4>(4 * i) = constraints_[i].offset.coeffs();
    }
  } else if (name == kTargets) {
    for (size_t i = 0; i < constraints_.size(); ++i) {
      value.template segment<4>(4 * i) = constraints_[i].target.coeffs();
    }
  } else {
    MT_THROW("Unknown input to FullyDifferentiableOrientationFunctionT<T>::getInput: {}", name);
  }
}

template <typename T>
void FullyDifferentiableOrientationErrorFunctionT<T>::setInputImp(
    const std::string& name,
    Eigen::Ref<const Eigen::VectorX<T>> value) {
  // Normalization is a tricky topic here.  When we set the values, we must normalize normals and
  // quaternions etc because the rest of the code won't work correctly without this (for example
  // computing the angle between unnormalized vectors requires normalizing them first).  However,
  // if we do normalize the values that are passed in, then the derivatives of those values are
  // no longer correct.  For example, imagine that we pass in v = (1000, 0, 0) as the normal to a
  // plane. Clearly we must turn this into (1, 0, 0) so the IK solve does something reasonable, but
  // when we do we find that our dv/dTheta is off by a factor of 1000.  It's hard to address that
  // inside this code, the right way to do it is to require that the inputs be normalized inside
  // Pytorch before getting passed in, which will ensure all the derivatives work out correctly.
  if (name == kWeights) {
    for (size_t i = 0; i < constraints_.size(); ++i) {
      constraints_[i].weight = static_cast<float>(value[i]);
    }
  } else if (name == kOffsets) {
    for (size_t i = 0; i < constraints_.size(); ++i) {
      constraints_[i].offset = Eigen::Quaternion<T>(value.template segment<4>(4 * i)).normalized();
    }
  } else if (name == kTargets) {
    for (size_t i = 0; i < constraints_.size(); ++i) {
      constraints_[i].target = Eigen::Quaternion<T>(value.template segment<4>(4 * i)).normalized();
    }
  } else {
    MT_THROW(
        "Unknown input to FullyDifferentiableOrientationErrorFunctionT<T>::getInput: {}", name);
  }
}

template <typename T>
T FullyDifferentiableOrientationErrorFunctionT<T>::calculateOrientationGradient(
    const SkeletonStateT<T>& state,
    const OrientationConstraintT<T>& constr,
    Eigen::VectorX<T>& jGrad) const {
  const auto& parameterTransform = this->parameterTransform_;

  // calculate orientation error
  const Eigen::Quaternion<T> rot = state.jointState[constr.parent].rotation() * constr.offset;
  const Eigen::Matrix3<T> rotDiff = rot.toRotationMatrix() - constr.target.toRotationMatrix();
  const T wgt = kOrientationWeight * constr.weight;

  // loop over all joints the constraint is attached to and calculate gradient
  size_t jointIndex = constr.parent;
  while (jointIndex != kInvalidIndex) {
    // check for valid index
    MT_CHECK(jointIndex < this->skeleton_.joints.size());

    const size_t paramIndex = jointIndex * kParametersPerJoint;

    // calculate derivatives based on active joints
    for (size_t d = 0; d < 3; d++) {
      if (this->activeJointParams_[paramIndex + 3 + d]) {
        // calculate joint gradient
        const Eigen::Vector3<T> axis = state.jointState[jointIndex].rotationAxis.col(d);
        const auto rotD = crossProductMatrix(axis) * rot;
        const T val = 2.0f * wgt * rotD.cwiseProduct(rotDiff).sum();
        // explicitly multiply with the parameter transform to generate parameter space gradients
        for (auto index = parameterTransform.transform.outerIndexPtr()[paramIndex + 3 + d];
             index < parameterTransform.transform.outerIndexPtr()[paramIndex + d + 3 + 1];
             ++index) {
          jGrad[parameterTransform.transform.innerIndexPtr()[index]] +=
              val * parameterTransform.transform.valuePtr()[index];
        }
      }
    }

    // go to the next joint
    jointIndex = this->skeleton_.joints[jointIndex].parent;
  }

  return rotDiff.squaredNorm() * wgt;
}

template <typename T>
template <typename JetType>
[[nodiscard]] JetType
FullyDifferentiableOrientationErrorFunctionT<T>::calculateOrientationGradient_dot(
    const SkeletonStateT<T>& state,
    const size_t constrParent,
    const JetType& constrWeight,
    const Eigen::Quaternion<JetType>& constrOrientationOffset,
    const Eigen::Quaternion<JetType>& constrOrientationTarget,
    Eigen::Ref<const Eigen::VectorX<T>> vec) const {
  JetType result;

  // calculate orientation error
  const Eigen::Quaternion<JetType> rot =
      (state.jointState[constrParent].rotation().template cast<JetType>() *
       constrOrientationOffset);
  const Eigen::Matrix3<JetType> rotDiff =
      rot.toRotationMatrix() - constrOrientationTarget.toRotationMatrix();
  const JetType wgt = this->weight_ * this->kOrientationWeight * constrWeight;

  // loop over all joints the constraint is attached to and calculate gradient
  size_t jointIndex = constrParent;
  while (jointIndex != kInvalidIndex) {
    // check for valid index
    assert(jointIndex < this->skeleton_.joints.size());

    const size_t paramIndex = jointIndex * kParametersPerJoint;

    // calculate derivatives based on active joints
    for (size_t d = 0; d < 3; d++) {
      if (this->activeJointParams_[paramIndex + 3 + d]) {
        // calculate joint gradient
        const Eigen::Vector3<T> axis = state.jointState[jointIndex].rotationAxis.col(d);
        const Eigen::Matrix3<JetType> rotD = crossProductMatrix(axis) * rot;
        const JetType val = T(2) * wgt * rotD.cwiseProduct(rotDiff).sum();
        result += times_parameterTransform_times_v(
            val, paramIndex + 3 + d, this->parameterTransform_, vec);
      }
    }

    // go to the next joint
    jointIndex = this->skeleton_.joints[jointIndex].parent;
  }

  return result;
}

template <typename T>
Eigen::VectorX<T> FullyDifferentiableOrientationErrorFunctionT<T>::d_gradient_d_input_dot(
    const std::string& inputName,
    const ModelParametersT<T>& /*modelParams*/,
    const SkeletonStateT<T>& state,
    Eigen::Ref<const Eigen::VectorX<T>> inputVec) {
  // Use automatic differentiation on the gradient function:
  if (inputName == kWeights) {
    Eigen::VectorX<T> result = Eigen::VectorX<T>::Zero(constraints_.size());
    using JetType = ceres::Jet<T, 1>;
    for (size_t iCons = 0; iCons < constraints_.size(); ++iCons) {
      const auto& cons = constraints_[iCons];
      result(iCons) = calculateOrientationGradient_dot<JetType>(
                          state,
                          cons.parent,
                          JetType(cons.weight, 0),
                          cons.offset.template cast<JetType>(),
                          cons.target.template cast<JetType>(),
                          inputVec)
                          .v[0];
    }
    return result;
  } else if (inputName == kOffsets) {
    Eigen::VectorX<T> result = Eigen::VectorX<T>::Zero(4 * constraints_.size());
    using JetType = ceres::Jet<T, 4>;
    for (size_t iCons = 0; iCons < constraints_.size(); ++iCons) {
      const auto& cons = constraints_[iCons];
      const auto dotProd = calculateOrientationGradient_dot<JetType>(
          state,
          cons.parent,
          JetType(cons.weight),
          Eigen::Quaternion<JetType>(buildJetVec<T, 4>(cons.offset.coeffs())).normalized(),
          cons.target.template cast<JetType>(),
          inputVec);
      result.template segment<4>(4 * iCons) = dotProd.v;
    }
    return result;
  } else if (inputName == kTargets) {
    Eigen::VectorX<T> result = Eigen::VectorX<T>::Zero(4 * constraints_.size());
    using JetType = ceres::Jet<T, 4>;
    for (size_t iCons = 0; iCons < constraints_.size(); ++iCons) {
      const auto& cons = constraints_[iCons];
      const auto dotProd = calculateOrientationGradient_dot<JetType>(
          state,
          cons.parent,
          JetType(cons.weight),
          cons.offset.template cast<JetType>(),
          Eigen::Quaternion<JetType>(buildJetVec<T, 4>(cons.target.coeffs())).normalized(),
          inputVec);
      result.template segment<4>(4 * iCons) = dotProd.v;
    }
    return result;
  } else {
    MT_THROW(
        "Unknown input name in FullyDifferentiableOrientationErrorFunctionT<T>::d_gradient_d_input_dot: {}",
        inputName);
  }
}

template class FullyDifferentiableOrientationErrorFunctionT<float>;
template class FullyDifferentiableOrientationErrorFunctionT<double>;

} // namespace momentum
