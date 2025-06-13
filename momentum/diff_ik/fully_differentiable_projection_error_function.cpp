/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/diff_ik/fully_differentiable_projection_error_function.h"
#include "momentum/character_solver/projection_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/diff_ik/ceres_utility.h"

namespace momentum {

template <typename T>
FullyDifferentiableProjectionErrorFunctionT<T>::FullyDifferentiableProjectionErrorFunctionT(
    const momentum::Skeleton& skel,
    const momentum::ParameterTransform& pt,
    T nearClip)
    : ProjectionErrorFunctionT<T>(skel, pt, nearClip) {}

template <typename T>
template <typename JetType>
[[nodiscard]] JetType FullyDifferentiableProjectionErrorFunctionT<T>::constraintGradient_dot(
    const momentum::ModelParametersT<T>& /*modelParams*/,
    const momentum::SkeletonStateT<T>& skelState,
    const int parentJointIndex_cons,
    const Eigen::Vector3<JetType>& offset_cons,
    const JetType& weight_cons,
    const Eigen::Matrix<JetType, 3, 4>& projection_cons,
    const Eigen::Vector2<JetType>& target_cons,
    Eigen::Ref<const Eigen::VectorX<T>> vec) const {
  auto result = JetType();

  const auto& jsCons = skelState.jointState[parentJointIndex_cons];
  const Eigen::Vector3<JetType> p_world_cm = (jsCons.transformation * offset_cons).eval();
  const Eigen::Vector3<JetType> p_projected_cm =
      (projection_cons * p_world_cm.homogeneous()).eval();

  // Behind camera:
  if (p_projected_cm.z().a < _nearClip) {
    return result;
  }

  const Eigen::Vector2<JetType> p_res = p_projected_cm.hnormalized() - target_cons;

  const JetType wgt = T(2) * weight_cons * kProjectionWeight * this->weight_;

  const JetType& z = p_projected_cm(2);
  const JetType zSqr = z * z;
  const JetType x_zz = p_projected_cm(0) / zSqr;
  const JetType y_zz = p_projected_cm(1) / zSqr;

  // Add dError_dModelParametersT<T>^JetType * v =
  //    (dError/dJointParameter[jFullBodyDOF] *
  //       dJointParameter[jFullBodyDOF]/dModelParametersT<T>)^JetType * v
  auto add_gradient_dot = [&](const size_t jFullBodyDOF,
                              const Eigen::Matrix<JetType, 3, 1>& d_p_world_cm) {
    const Eigen::Vector3<JetType> d_p_projected =
        (projection_cons.template topLeftCorner<3, 3>() * d_p_world_cm).eval();
    const JetType& dx = d_p_projected(0);
    const JetType& dy = d_p_projected(1);
    const JetType& dz = d_p_projected(2);

    const Eigen::Vector2<JetType> d_p_res(dx / z - x_zz * dz, dy / z - y_zz * dz);

    // Error is E = w * (p_projected.xy / p_projected.z)^2
    // Gradient of error is:
    //    dE/dJointParameter[jFullBodyDOF] = 2w*(p_window - p_target) *
    //    d(p_window)/ddJointParameter[jFullBodyDOF].
    const JetType gradFull = d_p_res.dot(p_res);

    // Multiply by the parameter transform, take the dot product with v, and add to the result:
    result += times_parameterTransform_times_v(
        gradFull * wgt, jFullBodyDOF, this->parameterTransform_, vec);
  };

  // loop over all joints the constraint is attached to and calculate gradient
  size_t jointIndex = parentJointIndex_cons;
  while (jointIndex != kInvalidIndex) {
    // check for valid index
    assert(jointIndex < this->skeleton_.joints.size());

    const auto& jointState = skelState.jointState[jointIndex];
    const size_t paramIndex = jointIndex * 7;
    const Eigen::Vector3<JetType> posd_cm = p_world_cm - jointState.translation();

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
Eigen::VectorX<T> FullyDifferentiableProjectionErrorFunctionT<T>::d_gradient_d_input_dot(
    const std::string& inputName,
    const ModelParametersT<T>& modelParams,
    const SkeletonStateT<T>& state,
    Eigen::Ref<const Eigen::VectorX<T>> inputVec) {
  // Use automatic differentiation on the gradient function:
  const auto nCons = this->constraints_.size();
  if (inputName == kWeights) {
    Eigen::VectorX<T> result = Eigen::VectorX<T>::Zero(nCons);
    using JetType = ceres::Jet<T, 1>;
    for (int iCons = 0; iCons < nCons; ++iCons) {
      const auto dotProd = constraintGradient_dot<JetType>(
          modelParams,
          state,
          this->constraints_[iCons].parent,
          this->constraints_[iCons].offset.template cast<JetType>(),
          JetType(this->constraints_[iCons].weight, 0),
          this->constraints_[iCons].projection.template cast<JetType>(),
          this->constraints_[iCons].target.template cast<JetType>(),
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
          this->constraints_[iCons].parent,
          momentum::buildJetVec<T, 3>(this->constraints_[iCons].offset),
          JetType(this->constraints_[iCons].weight),
          this->constraints_[iCons].projection.template cast<JetType>(),
          this->constraints_[iCons].target.template cast<JetType>(),
          inputVec);
      result.template segment<3>(3 * iCons) = dotProd.v;
    }
    return result;
  } else if (inputName == kTargets) {
    Eigen::VectorX<T> result = Eigen::VectorX<T>::Zero(2 * nCons);
    using JetType = ceres::Jet<T, 2>;
    for (int iCons = 0; iCons < nCons; ++iCons) {
      const auto dotProd = constraintGradient_dot<JetType>(
          modelParams,
          state,
          this->constraints_[iCons].parent,
          this->constraints_[iCons].offset.template cast<JetType>(),
          JetType(this->constraints_[iCons].weight),
          this->constraints_[iCons].projection.template cast<JetType>(),
          buildJetVec<T, 2>(this->constraints_[iCons].target),
          inputVec);
      result.template segment<2>(2 * iCons) = dotProd.v;
    }
    return result;
  } else if (inputName == kProjections) {
    Eigen::VectorX<T> result = Eigen::VectorX<T>::Zero(12 * nCons);
    using JetType = ceres::Jet<T, 12>;
    for (int iCons = 0; iCons < nCons; ++iCons) {
      const auto dotProd = constraintGradient_dot<JetType>(
          modelParams,
          state,
          this->constraints_[iCons].parent,
          this->constraints_[iCons].offset.template cast<JetType>(),
          JetType(this->constraints_[iCons].weight),
          momentum::buildJetMat<T, 3, 4>(this->constraints_[iCons].projection),
          this->constraints_[iCons].target.template cast<JetType>(),
          inputVec);
      result.template segment<12>(12 * iCons) = dotProd.v;
    }
    return result;
  } else {
    throw std::runtime_error(
        "Unknown input name in FullyDifferentiablePositionErrorFunction::d_gradient_d_input_dot: " +
        inputName);
  }
} // namespace momentum

template <typename T>
std::vector<std::string> FullyDifferentiableProjectionErrorFunctionT<T>::inputs() const {
  // offsets not currently supported because we can't differentiate through
  // the camera without a lot of trouble.
  return {kWeights, kOffsets, kProjections, kTargets};
}

template <typename T>
Eigen::Index FullyDifferentiableProjectionErrorFunctionT<T>::getInputSize(
    const std::string& name) const {
  if (name == kWeights) {
    return this->constraints_.size();
  } else if (name == kOffsets) {
    return 3 * this->constraints_.size();
  } else if (name == kTargets) {
    return 2 * this->constraints_.size();
  } else if (name == kProjections) {
    return 3 * 4 * this->constraints_.size();
  } else {
    throw std::runtime_error("Unknown input for ProjectionErrorFunction: " + name);
  }
}

template <typename T>
void FullyDifferentiableProjectionErrorFunctionT<T>::getInputImp(
    const std::string& name,
    Eigen::Ref<Eigen::VectorX<T>> result) const {
  if (name == kWeights) {
    for (size_t i = 0; i < this->constraints_.size(); ++i) {
      result(i) = this->constraints_[i].weight;
    }
  } else if (name == kOffsets) {
    for (size_t i = 0; i < this->constraints_.size(); ++i) {
      result.template segment<3>(3 * i) = this->constraints_[i].offset;
    }
  } else if (name == kTargets) {
    for (size_t i = 0; i < this->constraints_.size(); ++i) {
      result.template segment<2>(2 * i) = this->constraints_[i].target;
    }
  } else if (name == kProjections) {
    for (size_t i = 0; i < this->constraints_.size(); ++i) {
      const auto& proj = this->constraints_[i].projection;

      // Ensure row-major for compatibility with Torch:
      for (int jRow = 0; jRow < 3; ++jRow) {
        for (int kCol = 0; kCol < 4; ++kCol) {
          result(12 * i + 4 * jRow + kCol) = proj(jRow, kCol);
        }
      }
    }
  } else {
    throw std::runtime_error("Unknown input for ProjectionErrorFunction: " + name);
  }
}

template <typename T>
void FullyDifferentiableProjectionErrorFunctionT<T>::setInputImp(
    const std::string& name,
    Eigen::Ref<const Eigen::VectorX<T>> value) {
  if (name == kWeights) {
    for (size_t i = 0; i < this->constraints_.size(); ++i) {
      this->constraints_[i].weight = value(i);
    }
  } else if (name == kOffsets) {
    for (size_t i = 0; i < this->constraints_.size(); ++i) {
      this->constraints_[i].offset = value.template segment<3>(3 * i);
    }
  } else if (name == kTargets) {
    for (size_t i = 0; i < this->constraints_.size(); ++i) {
      const Eigen::Vector2<T> target = value.template segment<2>(2 * i);
      this->constraints_[i].target = target;
    }
  } else if (name == kProjections) {
    for (size_t i = 0; i < this->constraints_.size(); ++i) {
      auto& proj = this->constraints_[i].projection;

      // Ensure row-major for compatibility with Torch:
      for (int jRow = 0; jRow < 3; ++jRow) {
        for (int kCol = 0; kCol < 4; ++kCol) {
          proj(jRow, kCol) = value(12 * i + 4 * jRow + kCol);
        }
      }
    }
  } else {
    throw std::runtime_error("Unknown input for ProjectionErrorFunction: " + name);
  }
}

template class FullyDifferentiableProjectionErrorFunctionT<float>;
template class FullyDifferentiableProjectionErrorFunctionT<double>;

} // namespace momentum
