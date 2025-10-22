/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/skeleton_state.h>
#include <momentum/common/checks.h>
#include <momentum/common/profile.h>

namespace momentum {

template <typename T, class Data, size_t FuncDim, size_t NumVec, size_t NumPos>
ConstraintErrorFunctionT<T, Data, FuncDim, NumVec, NumPos>::ConstraintErrorFunctionT(
    const Skeleton& skel,
    const ParameterTransform& pt,
    const T& lossAlpha,
    const T& lossC)
    : SkeletonErrorFunctionT<T>(skel, pt),
      loss_(lossAlpha, lossC),
      jointGrad_(pt.numAllModelParameters()) {
  static_assert(FuncDim > 0, "The error function cannot be empty.");
  static_assert(NumVec > 0, "At least one vector is required in the constraint.");
  static_assert(
      NumVec >= NumPos, "The number of points cannot be more than the total number of vectors.");
}

template <typename T, class Data, size_t FuncDim, size_t NumVec, size_t NumPos>
double ConstraintErrorFunctionT<T, Data, FuncDim, NumVec, NumPos>::getError(
    const ModelParametersT<T>& /* params */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */) {
  MT_PROFILE_FUNCTION();

  // loop over all constraints and calculate the error
  FuncType f;
  double error = 0.0;
  for (size_t iConstr = 0; iConstr < constraints_.size(); ++iConstr) {
    const Data& constr = constraints_[iConstr];
    if (constr.weight != 0) {
      evalFunction(iConstr, state.jointState.at(constr.parent), f);
      error += constr.weight * loss_.value(f.squaredNorm());
    }
  }
  return this->weight_ * error;
}

template <typename T, class Data, size_t FuncDim, size_t NumVec, size_t NumPos>
double ConstraintErrorFunctionT<T, Data, FuncDim, NumVec, NumPos>::getJacobianForSingleConstraint(
    const JointStateListT<T>& jointStates,
    const size_t iConstr,
    Ref<Eigen::MatrixX<T>> jacobian,
    Ref<Eigen::VectorX<T>> residual) {
  const Data& constr = this->constraints_[iConstr];
  if (constr.weight == 0) {
    return 0;
  }

  FuncType f;
  std::array<VType, NumVec> v;
  std::array<DfdvType, NumVec> dfdv;
  evalFunction(iConstr, jointStates.at(constr.parent), f, v, dfdv);
  const T sqrError = f.squaredNorm();
  const T w = constr.weight * this->weight_;
  const double error = w * this->loss_.value(sqrError);
  const T deriv = std::sqrt(w * this->loss_.deriv(sqrError));

  // The input jacobian is a subblock from the full jacobian tailored for this error function, so
  // the rows start from zero.
  const size_t rowIndex = FuncDim * iConstr;
  // The gradient of the loss function needs to be split between J and r, so both J'J and
  // J'r are scaled correctly. The factor 2 from the square term is accounted for in the base
  // class function.
  residual.template middleRows<FuncDim>(rowIndex).noalias() = deriv * f;

  // Small optimization for early termination
  if (isApprox<T>(deriv, T(0), Eps<T>(1e-9, 1e-16))) {
    return error;
  }
  const bool zeroDeriv =
      !std::any_of(dfdv.begin(), dfdv.end(), [](const auto& vec) { return !vec.isZero(); });
  if (zeroDeriv) {
    return error;
  }

  // shorthands for code readability
  const auto outerIndexPtr = this->parameterTransform_.transform.outerIndexPtr();
  const auto innerIndexPtr = this->parameterTransform_.transform.innerIndexPtr();
  const auto valuePtr = this->parameterTransform_.transform.valuePtr();
  auto&& jac = jacobian.template middleRows<FuncDim>(rowIndex);

  // Fill in the jacobian by walking up the joint hierarchy for each constraint.
  // Need to consider both active joints and enabled model parameters.
  size_t jntIndex = constr.parent;
  while (jntIndex != kInvalidIndex) {
    // check for valid index
    MT_CHECK(jntIndex < this->skeleton_.joints.size());

    const auto& jntState = jointStates[jntIndex];
    const size_t jntParamIndex = jntIndex * kParametersPerJoint;

    for (size_t jVec = 0; jVec < NumVec; ++jVec) {
      if (dfdv[jVec].isZero()) {
        continue;
      }

      Eigen::Vector3<T> offset;
      if (jVec < NumPos) {
        offset.noalias() = v[jVec] - jntState.translation();
      } else {
        offset = v[jVec];
      }

      // Translational dofs -- only affect POINT
      if (jVec < NumPos) {
        for (size_t d = 0; d < 3; ++d) {
          if (!this->activeJointParams_[jntParamIndex + d]) {
            continue;
          }

          const FuncType jc = deriv * dfdv[jVec] * jntState.getTranslationDerivative(d);
          for (auto index = outerIndexPtr[jntParamIndex + d];
               index < outerIndexPtr[jntParamIndex + d + 1];
               ++index) {
            if (this->enabledParameters_.test(innerIndexPtr[index])) {
              jac.col(innerIndexPtr[index]).noalias() += jc * valuePtr[index];
            }
          }
        }
      }

      // Rotational dofs -- affect both POINT and AXIS
      for (size_t d = 0; d < 3; ++d) {
        if (!this->activeJointParams_[jntParamIndex + 3 + d]) {
          continue;
        }

        const FuncType jc = deriv * dfdv[jVec] * jntState.getRotationDerivative(d, offset);
        for (auto index = outerIndexPtr[jntParamIndex + 3 + d];
             index < outerIndexPtr[jntParamIndex + 3 + d + 1];
             ++index) {
          if (this->enabledParameters_.test(innerIndexPtr[index])) {
            jac.col(innerIndexPtr[index]).noalias() += jc * valuePtr[index];
          }
        }
      }

      // Scale dof -- only affect POINT
      if (jVec < NumPos) {
        if (this->activeJointParams_[jntParamIndex + 6]) {
          const FuncType jc = deriv * dfdv[jVec] * jntState.getScaleDerivative(offset);
          for (auto index = outerIndexPtr[jntParamIndex + 6];
               index < outerIndexPtr[jntParamIndex + 6 + 1];
               ++index) {
            if (this->enabledParameters_.test(innerIndexPtr[index])) {
              jac.col(innerIndexPtr[index]).noalias() += jc * valuePtr[index];
            }
          }
        }
      }
    }
    // go up the hierarchy to the parent
    jntIndex = this->skeleton_.joints[jntIndex].parent;
  }

  return error;
}

template <typename T, class Data, size_t FuncDim, size_t NumVec, size_t NumPos>
double ConstraintErrorFunctionT<T, Data, FuncDim, NumVec, NumPos>::getJacobian(
    const ModelParametersT<T>& /*params*/,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */,
    Ref<Eigen::MatrixX<T>> jacobian,
    Ref<Eigen::VectorX<T>> residual,
    int& usedRows) {
  MT_PROFILE_FUNCTION();
  usedRows = getJacobianSize();

  double error = 0.0;
  for (size_t iConstr = 0; iConstr < this->constraints_.size(); ++iConstr) {
    error += getJacobianForSingleConstraint(state.jointState, iConstr, jacobian, residual);
  }
  return error;
}

template <typename T, class Data, size_t FuncDim, size_t NumVec, size_t NumPos>
double ConstraintErrorFunctionT<T, Data, FuncDim, NumVec, NumPos>::getGradientForSingleConstraint(
    const JointStateListT<T>& jointStates,
    size_t iConstr,
    Ref<VectorX<T>> /*gradient*/) {
  const Data& constr = this->constraints_[iConstr];
  if (constr.weight == 0) {
    return 0;
  }

  FuncType f;
  std::array<VType, NumVec> v;
  std::array<DfdvType, NumVec> dfdv;
  evalFunction(iConstr, jointStates.at(constr.parent), f, v, dfdv);
  if (f.isZero()) {
    return 0;
  }

  const T sqrError = f.squaredNorm();
  const T w = constr.weight * this->weight_;
  const double error = w * this->loss_.value(sqrError);
  // The gradient is scaled by the gradient of the loss function; factor 2 falls out of the
  // square.
  const FuncType deriv = T(2) * w * this->loss_.deriv(sqrError) * f;

  const auto outerIndexPtr = this->parameterTransform_.transform.outerIndexPtr();
  const auto innerIndexPtr = this->parameterTransform_.transform.innerIndexPtr();
  const auto valuePtr = this->parameterTransform_.transform.valuePtr();
  size_t jntIndex = constr.parent;
  while (jntIndex != kInvalidIndex) {
    // check for valid index
    MT_CHECK(jntIndex < this->skeleton_.joints.size());

    const auto& jntState = jointStates[jntIndex];
    const size_t jntParamIndex = jntIndex * kParametersPerJoint;

    for (size_t jVec = 0; jVec < NumVec; ++jVec) {
      if (dfdv[jVec].isZero()) {
        continue;
      }

      Eigen::Vector3<T> offset;
      if (jVec < NumPos) {
        offset.noalias() = v[jVec] - jntState.translation();
      } else {
        offset = v[jVec];
      }

      // Translational dofs -- only affect POINT
      if (jVec < NumPos) {
        for (size_t d = 0; d < 3; d++) {
          if (!this->activeJointParams_[jntParamIndex + d]) {
            continue;
          }

          // joint space gradient
          const T val = deriv.dot(dfdv[jVec] * jntState.getTranslationDerivative(d));
          // multiply with parameter transform to get model parameter gradient
          for (auto index = outerIndexPtr[jntParamIndex + d];
               index < outerIndexPtr[jntParamIndex + d + 1];
               ++index) {
            if (this->enabledParameters_.test(innerIndexPtr[index])) {
              jointGrad_[innerIndexPtr[index]] += val * valuePtr[index];
            }
          }
        }
      }

      // Rotational dofs -- affect both POINT and AXIS
      for (size_t d = 0; d < 3; d++) {
        if (!this->activeJointParams_[jntParamIndex + 3 + d]) {
          continue;
        }

        // joint space gradient
        const T val = deriv.dot(dfdv[jVec] * jntState.getRotationDerivative(d, offset));
        // multiply with parameter transform to get model parameter gradient
        for (auto index = outerIndexPtr[jntParamIndex + 3 + d];
             index < outerIndexPtr[jntParamIndex + 3 + d + 1];
             ++index) {
          if (this->enabledParameters_.test(innerIndexPtr[index])) {
            jointGrad_[innerIndexPtr[index]] += val * valuePtr[index];
          }
        }
      }

      // Scale dof -- only affect POINT
      if (jVec < NumPos) {
        if (this->activeJointParams_[jntParamIndex + 6]) {
          // joint space gradient
          const T val = deriv.dot(dfdv[jVec] * jntState.getScaleDerivative(offset));
          // multiply with parameter transform to get model parameter gradient
          for (auto index = outerIndexPtr[jntParamIndex + 6];
               index < outerIndexPtr[jntParamIndex + 6 + 1];
               ++index) {
            if (this->enabledParameters_.test(innerIndexPtr[index])) {
              jointGrad_[innerIndexPtr[index]] += val * valuePtr[index];
            }
          }
        }
      }
    }
    // go up the skeleton hierarchy to the parent joint
    jntIndex = this->skeleton_.joints[jntIndex].parent;
  }

  return error;
}

template <typename T, class Data, size_t FuncDim, size_t NumVec, size_t NumPos>
double ConstraintErrorFunctionT<T, Data, FuncDim, NumVec, NumPos>::getGradient(
    const ModelParametersT<T>& /*unused*/,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */,
    Ref<Eigen::VectorX<T>> gradient) {
  MT_PROFILE_FUNCTION();

  // initialize joint gradients storage
  jointGrad_.setZero();

  double error = 0.0;
  for (size_t iConstr = 0; iConstr < this->constraints_.size(); ++iConstr) {
    error += getGradientForSingleConstraint(state.jointState, iConstr, gradient);
  }

  gradient += jointGrad_;
  return error;
}

template <typename T, class Data, size_t FuncDim, size_t NumVec, size_t NumPos>
size_t ConstraintErrorFunctionT<T, Data, FuncDim, NumVec, NumPos>::getJacobianSize() const {
  return FuncDim * constraints_.size();
}

} // namespace momentum
