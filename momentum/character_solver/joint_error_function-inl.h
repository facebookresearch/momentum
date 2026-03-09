/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "momentum/character_solver/joint_error_function.h"

#include "momentum/character/skeleton_state.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/math/utility.h"

namespace momentum {

template <typename T, class Data, size_t FuncDim, size_t NumVec, size_t NumPos>
JointErrorFunctionT<T, Data, FuncDim, NumVec, NumPos>::JointErrorFunctionT(
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
double JointErrorFunctionT<T, Data, FuncDim, NumVec, NumPos>::getError(
    const ModelParametersT<T>& /* params */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */) {
  MT_PROFILE_FUNCTION();

  FuncType f;
  std::array<VType, NumVec> v;
  std::array<DfdvType, NumVec> dfdv;
  double error = 0.0;
  for (size_t iConstr = 0; iConstr < constraints_.size(); ++iConstr) {
    const Data& constr = constraints_[iConstr];
    if (constr.weight == 0) {
      continue;
    }
    evalFunction(iConstr, state.jointState.at(constr.parent), f, v, dfdv);
    error += constr.weight * loss_.value(f.squaredNorm());
  }
  return this->weight_ * error;
}

template <typename T, class Data, size_t FuncDim, size_t NumVec, size_t NumPos>
double JointErrorFunctionT<T, Data, FuncDim, NumVec, NumPos>::getGradient(
    const ModelParametersT<T>& /*unused*/,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */,
    Ref<Eigen::VectorX<T>> gradient) {
  MT_PROFILE_FUNCTION();

  // Initialize joint gradients storage
  jointGrad_.setZero();

  // Shorthands for parameter transform
  const auto outerIndexPtr = this->parameterTransform_.transform.outerIndexPtr();
  const auto innerIndexPtr = this->parameterTransform_.transform.innerIndexPtr();
  const auto valuePtr = this->parameterTransform_.transform.valuePtr();

  double error = 0.0;
  for (size_t iConstr = 0; iConstr < this->constraints_.size(); ++iConstr) {
    const Data& constr = constraints_[iConstr];
    if (constr.weight == 0) {
      continue;
    }

    FuncType f;
    std::array<VType, NumVec> v;
    std::array<DfdvType, NumVec> dfdv;
    evalFunction(iConstr, state.jointState.at(constr.parent), f, v, dfdv);

    if (f.isZero()) {
      continue;
    }

    const T sqrError = f.squaredNorm();
    const T w = constr.weight * this->weight_;
    error += w * loss_.value(sqrError);
    const FuncType deriv = T(2) * w * loss_.deriv(sqrError) * f;

    // Precompute wR[jVec] = dfdv[jVec]^T * deriv once per constraint.
    // Converts per-DOF cost from FuncDim x 3 matmul + FuncDim dot to 3-element dot.
    std::array<Eigen::Vector3<T>, NumVec> wR;
    for (size_t jVec = 0; jVec < NumVec; ++jVec) {
      if (dfdv[jVec].isZero()) {
        wR[jVec].setZero();
      } else {
        wR[jVec].noalias() = dfdv[jVec].transpose() * deriv;
      }
    }

    // Inline hierarchy walk: walk from parent joint to root
    size_t jntIndex = constr.parent;
    while (jntIndex != kInvalidIndex) {
      MT_CHECK(jntIndex < this->skeleton_.joints.size());

      const auto& jntState = state.jointState[jntIndex];
      const size_t jntParamIndex = jntIndex * kParametersPerJoint;

      for (size_t jVec = 0; jVec < NumVec; ++jVec) {
        if (wR[jVec].isZero()) {
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
            const T val = wR[jVec].dot(jntState.getTranslationDerivative(d));
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
          const T val = wR[jVec].dot(jntState.getRotationDerivative(d, offset));
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
            const T val = wR[jVec].dot(jntState.getScaleDerivative(offset));
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
      jntIndex = this->skeleton_.joints[jntIndex].parent;
    }
  }

  gradient += jointGrad_;
  return error;
}

template <typename T, class Data, size_t FuncDim, size_t NumVec, size_t NumPos>
double JointErrorFunctionT<T, Data, FuncDim, NumVec, NumPos>::getJacobian(
    const ModelParametersT<T>& /*params*/,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */,
    Ref<Eigen::MatrixX<T>> jacobian,
    Ref<Eigen::VectorX<T>> residual,
    int& usedRows) {
  MT_PROFILE_FUNCTION();
  usedRows = static_cast<int>(getJacobianSize());

  // Shorthands for parameter transform
  const auto outerIndexPtr = this->parameterTransform_.transform.outerIndexPtr();
  const auto innerIndexPtr = this->parameterTransform_.transform.innerIndexPtr();
  const auto valuePtr = this->parameterTransform_.transform.valuePtr();

  double error = 0.0;
  for (size_t iConstr = 0; iConstr < this->constraints_.size(); ++iConstr) {
    const Data& constr = constraints_[iConstr];
    if (constr.weight == 0) {
      continue;
    }

    FuncType f;
    std::array<VType, NumVec> v;
    std::array<DfdvType, NumVec> dfdv;
    evalFunction(iConstr, state.jointState.at(constr.parent), f, v, dfdv);

    const T sqrError = f.squaredNorm();
    const T w = constr.weight * this->weight_;
    error += w * this->loss_.value(sqrError);
    const T derivScale = std::sqrt(w * this->loss_.deriv(sqrError));

    // Write scaled residual
    const size_t rowIndex = FuncDim * iConstr;
    residual.template middleRows<FuncDim>(rowIndex).noalias() = derivScale * f;

    // Early termination
    if (isApprox<T>(derivScale, T(0), Eps<T>(1e-9, 1e-16))) {
      continue;
    }
    const bool zeroDfdv =
        !std::any_of(dfdv.begin(), dfdv.end(), [](const auto& m) { return !m.isZero(); });
    if (zeroDfdv) {
      continue;
    }

    auto&& jac = jacobian.template middleRows<FuncDim>(rowIndex);

    // Inline hierarchy walk: walk from parent joint to root
    size_t jntIndex = constr.parent;
    while (jntIndex != kInvalidIndex) {
      MT_CHECK(jntIndex < this->skeleton_.joints.size());

      const auto& jntState = state.jointState[jntIndex];
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
            const FuncType jc = derivScale * dfdv[jVec] * jntState.getTranslationDerivative(d);
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
          const FuncType jc = derivScale * dfdv[jVec] * jntState.getRotationDerivative(d, offset);
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
            const FuncType jc = derivScale * dfdv[jVec] * jntState.getScaleDerivative(offset);
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
      jntIndex = this->skeleton_.joints[jntIndex].parent;
    }
  }
  return error;
}

template <typename T, class Data, size_t FuncDim, size_t NumVec, size_t NumPos>
size_t JointErrorFunctionT<T, Data, FuncDim, NumVec, NumPos>::getJacobianSize() const {
  return FuncDim * constraints_.size();
}

} // namespace momentum
