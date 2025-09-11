/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/diff_ik/fully_differentiable_distance_error_function.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/common/log.h"
#include "momentum/diff_ik/ceres_utility.h"

namespace momentum {

template <typename T>
FullyDifferentiableDistanceErrorFunctionT<T>::FullyDifferentiableDistanceErrorFunctionT(
    const momentum::Skeleton& skel,
    const momentum::ParameterTransform& pt)
    : DistanceErrorFunctionT<T>(skel, pt) {}

template <typename T>
template <typename JetType>
JetType FullyDifferentiableDistanceErrorFunctionT<T>::constraintGradient_dot(
    const momentum::ModelParametersT<T>& /*modelParams*/,
    const momentum::SkeletonStateT<T>& skelState,
    const Eigen::Vector3<JetType>& origin_cons,
    const int parentJointIndex_cons,
    const Eigen::Vector3<JetType>& offset_cons,
    const JetType& weight_cons,
    const JetType& target_cons,
    Eigen::Ref<const Eigen::VectorX<T>> vec) {
  JetType result;

  const auto& jsCons = skelState.jointState[parentJointIndex_cons];
  const Eigen::Vector3<JetType> p_world_cm =
      jsCons.transform.toAffine3().template cast<T>() * offset_cons;

  const Eigen::Vector3<JetType> diff_vec = p_world_cm - origin_cons;
  const JetType distance = diff_vec.norm();
  const Eigen::Vector3<JetType> diff_distance_diff_p_world = diff_vec / distance;
  const JetType diff = distance - target_cons;

  // Complete weight is the product of the per-constraint weight,
  // the global error function weight, and the constant-valued distanceWeight.
  // The 2.0 here is because the error function is (d - d_target)^2 so the
  // derivative of that brings down a 2 in front.
  const JetType wgt = (2.0f * kDistanceWeight * this->weight_) * weight_cons;

  // Compute
  //    dError_dModelParams^JetType * v =
  //      (dError/dJointParameter[jFullBodyDOF] *
  //       dJointParameter[jFullBodyDOF]/dModelParams)^JetType * v
  auto add_gradient_dot = [&](const size_t jFullBodyDOF,
                              const Eigen::Vector3<JetType>& d_p_world_cm) {
    const JetType diff_p_distance = diff_distance_diff_p_world.dot(d_p_world_cm);
    const auto gradFull = diff_p_distance * diff;
    result += times_parameterTransform_times_v(
        gradFull * wgt, jFullBodyDOF, this->parameterTransform_, vec);
  };

  // loop over all joints the constraint is attached to and calculate gradient
  size_t jointIndex = parentJointIndex_cons;
  while (jointIndex != kInvalidIndex) {
    // check for valid index
    assert(jointIndex < skelState.jointState.size());

    const auto& jointState = skelState.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const auto posd_cm = (p_world_cm - jointState.translation()).eval();

    // calculate derivatives based on active joints
    for (size_t d = 0; d < 3; d++) {
      if (this->activeJointParams_[paramIndex + d]) {
        add_gradient_dot(
            paramIndex + d, jointState.getTranslationDerivative(d).template cast<JetType>());
      }
      if (this->activeJointParams_[paramIndex + 3 + d]) {
        add_gradient_dot(
            paramIndex + 3 + d,
            getRotationDerivative(jointState, d, posd_cm).template cast<JetType>());
      }
    }
    if (this->activeJointParams_[paramIndex + 6]) {
      add_gradient_dot(
          paramIndex + 6, getScaleDerivative(jointState, posd_cm).template cast<JetType>());
    }

    // go to the next joint
    jointIndex = this->skeleton_.joints[jointIndex].parent;
  }

  return result;
}

template <typename T>
Eigen::VectorX<T> FullyDifferentiableDistanceErrorFunctionT<T>::d_gradient_d_input_dot(
    const std::string& inputName,
    const ModelParametersT<T>& modelParams,
    const SkeletonStateT<T>& state,
    Eigen::Ref<const Eigen::VectorX<T>> inputVec) {
  // Use automatic differentiation on the gradient function:
  const auto nCons = constraints_.size();
  if (inputName == kWeights) {
    Eigen::VectorX<T> result = Eigen::VectorX<T>::Zero(nCons);
    using JetType = ceres::Jet<T, 1>;
    for (int iCons = 0; iCons < nCons; ++iCons) {
      const auto dotProd = constraintGradient_dot<JetType>(
          modelParams,
          state,
          constraints_[iCons].origin.template cast<JetType>(),
          constraints_[iCons].parent,
          constraints_[iCons].offset.template cast<JetType>(),
          JetType(constraints_[iCons].weight, 0),
          JetType(constraints_[iCons].target),
          inputVec);
      result(iCons) = dotProd.v[0];
    }
    return result;
  } else if (inputName == kOffsets) {
    Eigen::VectorX<T> result = Eigen::VectorX<T>::Zero(3 * nCons);
    using JetType = ceres::Jet<T, 3>;
    for (int iCons = 0; iCons < nCons; ++iCons) {
      const auto dotProd = constraintGradient_dot<JetType>(
          modelParams,
          state,
          constraints_[iCons].origin.template cast<JetType>(),
          constraints_[iCons].parent,
          momentum::buildJetVec<T, 3>(constraints_[iCons].offset),
          JetType(constraints_[iCons].weight),
          JetType(constraints_[iCons].target),
          inputVec);
      result.template segment<3>(3 * iCons) = dotProd.v;
    }
    return result;
  } else if (inputName == kTargets) {
    Eigen::VectorX<T> result = Eigen::VectorX<T>::Zero(nCons);

    using JetType = ceres::Jet<T, 1>;
    for (int iCons = 0; iCons < nCons; ++iCons) {
      const auto dotProd = constraintGradient_dot<JetType>(
          modelParams,
          state,
          constraints_[iCons].origin.template cast<JetType>(),
          constraints_[iCons].parent,
          constraints_[iCons].offset.template cast<JetType>(),
          JetType(constraints_[iCons].weight),
          JetType(constraints_[iCons].target, 0),
          inputVec);
      result(iCons) = dotProd.v[0];
    }
    return result;
  } else if (inputName == kOrigins) {
    using JetType = ceres::Jet<T, 3>;
    Eigen::VectorX<T> result = Eigen::VectorX<T>::Zero(3 * nCons);
    for (int iCons = 0; iCons < nCons; ++iCons) {
      const auto dotProd = constraintGradient_dot<JetType>(
          modelParams,
          state,
          buildJetVec<T, 3>(constraints_[iCons].origin),
          constraints_[iCons].parent,
          constraints_[iCons].offset.template cast<JetType>(),
          JetType(constraints_[iCons].weight),
          JetType(constraints_[iCons].target),
          inputVec);
      result.template segment<3>(3 * iCons) = dotProd.v;
    }
    return result;
  } else {
    throw std::runtime_error(
        "Unknown input name in FullyDifferentiablePositionErrorFunction::d_gradient_d_input_dot: " +
        inputName);
  }
}

template <typename T>
std::vector<std::string> FullyDifferentiableDistanceErrorFunctionT<T>::inputs() const {
  return {kOrigins, kWeights, kOffsets, kTargets};
}

template <typename T>
Eigen::Index FullyDifferentiableDistanceErrorFunctionT<T>::getInputSize(
    const std::string& name) const {
  if (name == kWeights || name == kTargets) {
    return constraints_.size();
  } else if (name == kOffsets || name == kOrigins) {
    return 3 * constraints_.size();
  } else {
    throw std::runtime_error("Unknown input for DistanceErrorFunction: " + name);
  }
}

template <typename T>
void FullyDifferentiableDistanceErrorFunctionT<T>::getInputImp(
    const std::string& name,
    Eigen::Ref<Eigen::VectorX<T>> result) const {
  if (name == kWeights) {
    for (size_t i = 0; i < constraints_.size(); ++i) {
      result(i) = constraints_[i].weight;
    }
  } else if (name == kOffsets) {
    for (size_t i = 0; i < constraints_.size(); ++i) {
      result.template segment<3>(3 * i) = constraints_[i].offset;
    }
  } else if (name == kTargets) {
    for (size_t i = 0; i < constraints_.size(); ++i) {
      result(i) = constraints_[i].target;
    }
  } else if (name == kOrigins) {
    for (size_t i = 0; i < constraints_.size(); ++i) {
      result.template segment<3>(3 * i) = constraints_[i].origin;
    }
  } else {
    throw std::runtime_error("Unknown input for DistanceErrorFunction: " + name);
  }
}

template <typename T>
void FullyDifferentiableDistanceErrorFunctionT<T>::setInputImp(
    const std::string& name,
    Eigen::Ref<const Eigen::VectorX<T>> value) {
  auto checkSize = [&](Eigen::Index expectedSize) {
    if (value.size() != expectedSize) {
      throw std::runtime_error(
          "In DistanceErrorFunction::setInput, " + name + " expects size " +
          std::to_string(expectedSize) + "; got " + std::to_string(value.size()));
    }
  };

  if (name == kWeights) {
    checkSize(constraints_.size());
    for (size_t i = 0; i < constraints_.size(); ++i) {
      constraints_[i].weight = value(i);
    }
  } else if (name == kOffsets) {
    checkSize(3 * constraints_.size());
    for (size_t i = 0; i < constraints_.size(); ++i) {
      constraints_[i].offset = value.template segment<3>(3 * i);
    }
  } else if (name == kTargets) {
    checkSize(constraints_.size());
    for (size_t i = 0; i < constraints_.size(); ++i) {
      constraints_[i].target = value(i);
    }
  } else if (name == kOrigins) {
    checkSize(3 * constraints_.size());
    for (size_t i = 0; i < constraints_.size(); ++i) {
      constraints_[i].origin = value.template segment<3>(3 * i);
    }
  } else {
    throw std::runtime_error("Unknown input for DistanceErrorFunction: " + name);
  }
}

template class FullyDifferentiableDistanceErrorFunctionT<float>;
template class FullyDifferentiableDistanceErrorFunctionT<double>;

} // namespace momentum
