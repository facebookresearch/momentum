/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/diff_ik/fully_differentiable_position_error_function.h"

#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/common/checks.h>
#include <momentum/diff_ik/ceres_utility.h>
#include <momentum/math/utility.h>

#include <algorithm>

namespace momentum {

template <typename T>
void PositionConstraintStateT<T>::update(
    const SkeletonStateT<T>& skeletonState,
    const std::vector<PositionConstraintT<T>>& referenceConstraints) {
  const size_t numConstraints = referenceConstraints.size();

  // resize output arrays
  position.resize(numConstraints);

  // get joint state
  const auto& jointState = skeletonState.jointState;

  // go over all locators
  for (size_t constraintID = 0; constraintID < numConstraints; constraintID++) {
    // reference for quick access
    const PositionConstraintT<T>& constraint = referenceConstraints[constraintID];

    // get parent id
    const size_t& parentId = constraint.parent;

    // transform each locator by its parents transformation and store it in the locator state
    position[constraintID] =
        jointState[parentId].transformation * constraint.offset.template cast<T>();
  }
}

template struct PositionConstraintStateT<float>;
template struct PositionConstraintStateT<double>;

template <typename T>
FullyDifferentiablePositionErrorFunctionT<T>::FullyDifferentiablePositionErrorFunctionT(
    const Skeleton& skel,
    const ParameterTransform& pt)
    : SkeletonErrorFunctionT<T>(skel, pt) {
  // Empty
}

template <typename T>
std::vector<std::string> FullyDifferentiablePositionErrorFunctionT<T>::inputs() const {
  static std::vector<std::string> result = {kWeights, kOffsets, kTargets};
  return result;
}

template <typename T>
Eigen::Index FullyDifferentiablePositionErrorFunctionT<T>::getInputSize(
    const std::string& name) const {
  if (name == kWeights) {
    return constraints_.size();
  }

  if (name == kOffsets) {
    return 3 * constraints_.size();
  }

  if (name == kTargets) {
    return 3 * constraints_.size();
  }

  MT_THROW("Unknown input to FullyDifferentiablePositionErrorFunctionT<T>::getInput: {}", name);
}

template <typename T>
double FullyDifferentiablePositionErrorFunctionT<T>::getError(
    const ModelParametersT<T>& /*params*/,
    const SkeletonStateT<T>& state) {
  // calculate skeleton- and constraints-state
  constraintsState_.update(state, constraints_);

  // loop over all constraints and calculate the error
  double error = 0.0;

  for (size_t i = 0; i < constraints_.size(); ++i) {
    const PositionConstraintT<T>& ct = constraints_[i];
    const Eigen::Vector3<T>& position = constraintsState_.position[i];

    // calculate the difference between target and position and error
    const Eigen::Vector3<T> diff = position - ct.target.template cast<T>();
    error += ct.weight * diff.squaredNorm() * kPositionWeight;
  }

  // return error
  return error * this->weight_;
}

template <typename T>
double FullyDifferentiablePositionErrorFunctionT<T>::getGradient(
    const ModelParametersT<T>& /*params*/,
    const SkeletonStateT<T>& state,
    Ref<Eigen::VectorX<T>> gradient) {
  const auto& parameterTransform = this->parameterTransform_;

  // calculate skeleton- and constraints-state
  constraintsState_.update(state, constraints_);

  // storage for joint gradients
  jointGrad_.setZero(parameterTransform.numAllModelParameters());

  // loop over all constraints and calculate the error
  double error = 0.0;

  for (size_t i = 0; i < constraints_.size(); ++i) {
    const PositionConstraintT<T>& ct = constraints_[i];
    const Eigen::Vector3<T>& position = constraintsState_.position[i];
    if (ct.weight == 0) {
      continue;
    }

    error += calculatePositionGradient(state, constraints_[i], position, jointGrad_);
  }

  // convert joint gradient to parameter gradient and return
  gradient.noalias() += jointGrad_ * this->weight_;

  // return error
  return error * this->weight_;
}

template <typename T>
double FullyDifferentiablePositionErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& state,
    Ref<Eigen::MatrixX<T>> jacobian,
    Ref<Eigen::VectorX<T>> residual,
    int& usedRows) {
  // calculate skeleton- and constraints-state
  constraintsState_.update(state, constraints_);

  // loop over all constraints and calculate the error
  double error = 0.0;
  std::atomic_int offset(0);

  for (size_t i = 0; i < constraints_.size(); ++i) {
    const PositionConstraintT<T>& ct = constraints_[i];
    const Eigen::Vector3<T>& position = constraintsState_.position[i];
    if (ct.weight == 0) {
      continue;
    }

    const auto currentOffset = offset.fetch_add(3, std::memory_order_relaxed);
    error += calculatePositionJacobian(
        state,
        constraints_[i],
        position,
        jacobian.block(currentOffset, 0, 3, params.size()),
        residual.middleRows(currentOffset, 3));
  }

  usedRows = gsl::narrow_cast<int>(jacobian.rows());

  return error;
}

template <typename T>
T FullyDifferentiablePositionErrorFunctionT<T>::calculatePositionJacobian(
    const SkeletonStateT<T>& state,
    const PositionConstraintT<T>& constr,
    const Eigen::Vector3<T>& pos,
    Ref<Eigen::MatrixX<T>> jac,
    Ref<Eigen::VectorX<T>> res) const {
  const auto& parameterTransform = this->parameterTransform_;

  MT_CHECK(jac.cols() == static_cast<Eigen::Index>(parameterTransform.transform.cols()));
  MT_CHECK(jac.rows() == 3);

  // calculate the difference between target and position and error
  const Eigen::Vector3<T> diff = pos - constr.target;
  const T wgt = std::sqrt(constr.weight * kPositionWeight * this->weight_);

  // loop over all joints the constraint is attached to and calculate gradient
  size_t jointIndex = constr.parent;
  while (jointIndex != kInvalidIndex) {
    // check for valid index
    MT_CHECK(jointIndex < this->skeleton_.joints.size());

    const auto& jointState = state.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> posd = pos - jointState.translation();

    // calculate derivatives based on active joints
    for (size_t d = 0; d < 3; d++) {
      if (this->activeJointParams_[paramIndex + d]) {
        const Eigen::Vector3<T> jc = jointState.getTranslationDerivative(d) * wgt;
        for (auto index = parameterTransform.transform.outerIndexPtr()[paramIndex + d];
             index < parameterTransform.transform.outerIndexPtr()[paramIndex + d + 1];
             ++index) {
          jac.col(parameterTransform.transform.innerIndexPtr()[index]) +=
              jc * parameterTransform.transform.valuePtr()[index];
        }
      }
      if (this->activeJointParams_[paramIndex + 3 + d]) {
        const Eigen::Vector3<T> jc = jointState.getRotationDerivative(d, posd) * wgt;
        for (auto index = parameterTransform.transform.outerIndexPtr()[paramIndex + d + 3];
             index < parameterTransform.transform.outerIndexPtr()[paramIndex + d + 3 + 1];
             ++index) {
          jac.col(parameterTransform.transform.innerIndexPtr()[index]) +=
              jc * parameterTransform.transform.valuePtr()[index];
        }
      }
    }
    if (this->activeJointParams_[paramIndex + 6]) {
      const Eigen::Vector3<T> jc = jointState.getScaleDerivative(posd) * wgt;
      for (auto index = parameterTransform.transform.outerIndexPtr()[paramIndex + 6];
           index < parameterTransform.transform.outerIndexPtr()[paramIndex + 6 + 1];
           ++index) {
        jac.col(parameterTransform.transform.innerIndexPtr()[index]) +=
            jc * parameterTransform.transform.valuePtr()[index];
      }
    }

    // go to the next joint
    jointIndex = this->skeleton_.joints[jointIndex].parent;
  }

  res = diff * wgt;

  return wgt * wgt * diff.squaredNorm();
}

template <typename T>
size_t FullyDifferentiablePositionErrorFunctionT<T>::getJacobianSize() const {
  return constraints_.size() * 3;
}

template <typename T>
void FullyDifferentiablePositionErrorFunctionT<T>::addConstraint(
    const PositionConstraintT<T>& constraint) {
  constraints_.push_back(constraint);
}

template <typename T>
const std::vector<PositionConstraintT<T>>&
FullyDifferentiablePositionErrorFunctionT<T>::getConstraints() const {
  return constraints_;
}

template <typename T>
void FullyDifferentiablePositionErrorFunctionT<T>::getInputImp(
    const std::string& name,
    Eigen::Ref<Eigen::VectorX<T>> value) const {
  if (name == kWeights) {
    for (size_t i = 0; i < constraints_.size(); ++i) {
      value[i] = static_cast<T>(constraints_[i].weight);
    }
  } else if (name == kOffsets) {
    for (size_t i = 0; i < constraints_.size(); ++i) {
      value.template segment<3>(3 * i) = constraints_[i].offset;
    }
  } else if (name == kTargets) {
    for (size_t i = 0; i < constraints_.size(); ++i) {
      value.template segment<3>(3 * i) = constraints_[i].target;
    }
  } else {
    MT_THROW("Unknown input to FullyDifferentiablePositionErrorFunctionT<T>::getInput: {}", name);
  }
}

template <typename T>
void FullyDifferentiablePositionErrorFunctionT<T>::setInputImp(
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
      constraints_[i].offset = value.template segment<3>(3 * i);
    }
  } else if (name == kTargets) {
    for (size_t i = 0; i < constraints_.size(); ++i) {
      constraints_[i].target = value.template segment<3>(3 * i);
    }
  } else {
    MT_THROW("Unknown input to FullyDifferentiablePositionErrorFunctionT<T>::getInput: {}", name);
  }
}

template <typename T>
T FullyDifferentiablePositionErrorFunctionT<T>::calculatePositionGradient(
    const SkeletonStateT<T>& state,
    const PositionConstraintT<T>& constr,
    const Eigen::Vector3<T>& pos,
    Eigen::VectorX<T>& jGrad) const {
  const auto& parameterTransform = this->parameterTransform_;

  // calculate the difference between target and position and error
  const Eigen::Vector3<T> diff = pos - constr.target;
  const T wgt = constr.weight * 2.0f * kPositionWeight;

  // loop over all joints the constraint is attached to and calculate gradient
  size_t jointIndex = constr.parent;
  while (jointIndex != kInvalidIndex) {
    // check for valid index
    MT_CHECK(jointIndex < this->skeleton_.joints.size());

    const auto& jointState = state.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> posd = pos - jointState.translation();

    // calculate derivatives based on active joints
    for (size_t d = 0; d < 3; d++) {
      if (this->activeJointParams_[paramIndex + d]) {
        // calculate joint gradient
        const T val = diff.dot(jointState.getTranslationDerivative(d)) * wgt;
        // explicitly multiply with the parameter transform to generate parameter space gradients
        for (auto index = parameterTransform.transform.outerIndexPtr()[paramIndex + d];
             index < parameterTransform.transform.outerIndexPtr()[paramIndex + d + 1];
             ++index) {
          jGrad[parameterTransform.transform.innerIndexPtr()[index]] +=
              val * parameterTransform.transform.valuePtr()[index];
        }
      }
      if (this->activeJointParams_[paramIndex + 3 + d]) {
        // calculate joint gradient
        const T val = diff.dot(jointState.getRotationDerivative(d, posd)) * wgt;
        // explicitly multiply with the parameter transform to generate parameter space gradients
        for (auto index = parameterTransform.transform.outerIndexPtr()[paramIndex + 3 + d];
             index < parameterTransform.transform.outerIndexPtr()[paramIndex + d + 3 + 1];
             ++index) {
          jGrad[parameterTransform.transform.innerIndexPtr()[index]] +=
              val * parameterTransform.transform.valuePtr()[index];
        }
      }
    }
    if (this->activeJointParams_[paramIndex + 6]) {
      // calculate joint gradient
      const T val = diff.dot(jointState.getScaleDerivative(posd)) * wgt;
      // explicitly multiply with the parameter transform to generate parameter space gradients
      for (auto index = parameterTransform.transform.outerIndexPtr()[paramIndex + 6];
           index < parameterTransform.transform.outerIndexPtr()[paramIndex + 6 + 1];
           ++index) {
        jGrad[parameterTransform.transform.innerIndexPtr()[index]] +=
            val * parameterTransform.transform.valuePtr()[index];
      }
    }

    // go to the next joint
    jointIndex = this->skeleton_.joints[jointIndex].parent;
  }

  return constr.weight * diff.squaredNorm() * kPositionWeight;
}

template <typename T>
template <typename JetType>
JetType FullyDifferentiablePositionErrorFunctionT<T>::calculatePositionGradient_dot(
    const SkeletonStateT<T>& state,
    const size_t /*iConstr*/,
    const size_t constrParent,
    const JetType& constrWeight,
    const Eigen::Vector3<JetType>& constr_offset,
    const Eigen::Vector3<JetType>& constr_target,
    Eigen::Ref<const Eigen::VectorX<T>> vec) const {
  JetType result;

  const Eigen::Vector3<JetType> pos =
      state.jointState[constrParent].transformation.template cast<T>() * constr_offset;

  // calculate the difference between target and position and error
  const Eigen::Vector3<JetType> diff = pos - constr_target;
  const JetType wgt = this->weight_ * constrWeight * T(2) * kPositionWeight;

  // loop over all joints the constraint is attached to and calculate gradient
  size_t jointIndex = constrParent;
  while (jointIndex != kInvalidIndex) {
    // check for valid index
    assert(jointIndex < this->skeleton_.joints.size());

    const auto& jointState = state.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<JetType> posd = pos - jointState.translation();

    // calculate derivatives based on active joints
    for (size_t d = 0; d < 3; d++) {
      if (this->activeJointParams_[paramIndex + d]) {
        result += times_parameterTransform_times_v(
            diff.dot(jointState.getTranslationDerivative(d)) * wgt,
            paramIndex + d,
            this->parameterTransform_,
            vec);
      }
      if (this->activeJointParams_[paramIndex + 3 + d]) {
        result += times_parameterTransform_times_v(
            diff.dot(getRotationDerivative(jointState, d, posd)) * wgt,
            paramIndex + 3 + d,
            this->parameterTransform_,
            vec);
      }
    }
    if (this->activeJointParams_[paramIndex + 6]) {
      result += times_parameterTransform_times_v(
          diff.dot(getScaleDerivative(jointState, posd)) * wgt,
          paramIndex + 6,
          this->parameterTransform_,
          vec);
    }

    // go to the next joint
    jointIndex = this->skeleton_.joints[jointIndex].parent;
  }

  return result;
}

template <typename T>
Eigen::VectorX<T> FullyDifferentiablePositionErrorFunctionT<T>::d_gradient_d_input_dot(
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
      result(iCons) = calculatePositionGradient_dot<JetType>(
                          state,
                          iCons,
                          cons.parent,
                          JetType(cons.weight, 0),
                          cons.offset.template cast<JetType>(),
                          cons.target.template cast<JetType>(),
                          inputVec)
                          .v[0];
    }
    return result;
  } else if (inputName == kOffsets) {
    Eigen::VectorX<T> result = Eigen::VectorX<T>::Zero(3 * constraints_.size());
    using JetType = ceres::Jet<T, 3>;
    for (size_t iCons = 0; iCons < constraints_.size(); ++iCons) {
      const auto& cons = constraints_[iCons];
      result.template segment<3>(3 * iCons) = calculatePositionGradient_dot<JetType>(
                                                  state,
                                                  iCons,
                                                  cons.parent,
                                                  JetType(cons.weight),
                                                  buildJetVec<T, 3>(cons.offset.template cast<T>()),
                                                  cons.target.template cast<JetType>(),
                                                  inputVec)
                                                  .v;
    }
    return result;
  } else if (inputName == kTargets) {
    Eigen::VectorX<T> result = Eigen::VectorX<T>::Zero(3 * constraints_.size());
    using JetType = ceres::Jet<T, 3>;
    for (size_t iCons = 0; iCons < constraints_.size(); ++iCons) {
      const auto& cons = constraints_[iCons];
      const auto dotProd = calculatePositionGradient_dot<JetType>(
          state,
          iCons,
          cons.parent,
          JetType(cons.weight),
          cons.offset.template cast<JetType>(),
          buildJetVec<T, 3>(cons.target.template cast<T>()),
          inputVec);
      result.template segment<3>(3 * iCons) = dotProd.v;
    }
    return result;
  } else {
    MT_THROW(
        "Unknown input name in FullyDifferentiablePositionErrorFunctionT<T>::d_gradient_d_input_dot: {}",
        inputName);
  }
}

template class FullyDifferentiablePositionErrorFunctionT<float>;
template class FullyDifferentiablePositionErrorFunctionT<double>;

} // namespace momentum
