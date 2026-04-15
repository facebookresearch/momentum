/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/limit_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_solver/error_function_utils.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"

namespace momentum {

namespace {

constexpr float kPositionWeight = 1e-4f;

template <typename T>
T computeMinMaxError(
    const LimitMinMax& data,
    const ModelParametersT<T>& params,
    T limitWeight,
    const ParameterSet& enabledParameters) {
  if (!enabledParameters.test(data.parameterIndex)) {
    return T(0);
  }
  T error = T(0);
  if (params(data.parameterIndex) < data.limits[0]) {
    const T val = data.limits[0] - params(data.parameterIndex);
    error = val * val * limitWeight;
  }
  if (params(data.parameterIndex) > data.limits[1]) {
    const T val = data.limits[1] - params(data.parameterIndex);
    error = val * val * limitWeight;
  }
  return error;
}

template <typename T>
T computeMinMaxJointError(
    const LimitMinMaxJoint& data,
    const SkeletonStateT<T>& state,
    T limitWeight,
    const Eigen::VectorX<bool>& activeJointParams) {
  const size_t parameterIndex = data.jointIndex * kParametersPerJoint + data.jointParameter;
  MT_CHECK(parameterIndex < static_cast<size_t>(state.jointParameters.size()));
  if (!activeJointParams[parameterIndex]) {
    return T(0);
  }
  T error = T(0);
  if (state.jointParameters(parameterIndex) < data.limits[0]) {
    const T val = data.limits[0] - state.jointParameters[parameterIndex];
    error = val * val * limitWeight;
  }
  if (state.jointParameters(parameterIndex) > data.limits[1]) {
    const T val = data.limits[1] - state.jointParameters[parameterIndex];
    error = val * val * limitWeight;
  }
  return error;
}

template <typename T>
T computeLinearError(
    const LimitLinear& data,
    const ModelParametersT<T>& params,
    T limitWeight,
    const ParameterSet& enabledParameters) {
  MT_CHECK(data.referenceIndex < static_cast<size_t>(params.size()));
  MT_CHECK(data.targetIndex < static_cast<size_t>(params.size()));
  if ((!enabledParameters.test(data.targetIndex) && !enabledParameters.test(data.referenceIndex)) ||
      !isInRange(data, params(data.targetIndex))) {
    return T(0);
  }
  const T residual =
      params(data.targetIndex) * data.scale - data.offset - params(data.referenceIndex);
  return residual * residual * limitWeight;
}

template <typename T>
T computeLinearJointError(
    const LimitLinearJoint& data,
    const SkeletonStateT<T>& state,
    T limitWeight,
    const Eigen::VectorX<bool>& activeJointParams) {
  const size_t referenceParameterIndex =
      data.referenceJointIndex * kParametersPerJoint + data.referenceJointParameter;
  const size_t targetParameterIndex =
      data.targetJointIndex * kParametersPerJoint + data.targetJointParameter;
  MT_CHECK(referenceParameterIndex < static_cast<size_t>(state.jointParameters.size()));
  MT_CHECK(targetParameterIndex < static_cast<size_t>(state.jointParameters.size()));
  if ((!activeJointParams[referenceParameterIndex] && !activeJointParams[targetParameterIndex]) ||
      !isInRange(data, state.jointParameters[targetParameterIndex])) {
    return T(0);
  }
  const T residual = state.jointParameters(targetParameterIndex) * data.scale - data.offset -
      state.jointParameters(referenceParameterIndex);
  return residual * residual * limitWeight;
}

template <typename T>
T computeHalfPlaneError(
    const LimitHalfPlane& data,
    const ModelParametersT<T>& params,
    T limitWeight,
    const ParameterSet& enabledParameters) {
  MT_CHECK(data.param1 < static_cast<size_t>(params.size()));
  MT_CHECK(data.param2 < static_cast<size_t>(params.size()));
  if (!enabledParameters.test(data.param1) && !enabledParameters.test(data.param2)) {
    return T(0);
  }
  const Eigen::Vector2<T> p(params(data.param1), params(data.param2));
  const T residual = p.dot(data.normal.template cast<T>()) - data.offset;
  if (residual < 0) {
    return residual * residual * limitWeight;
  }
  return T(0);
}

template <typename T>
T computeEllipsoidError(const LimitEllipsoid& ct, const SkeletonStateT<T>& state, T limitWeight) {
  const Eigen::Vector3<T> position =
      state.jointState[ct.parent].transform * ct.offset.template cast<T>();
  const Eigen::Vector3<T> localPosition =
      state.jointState[ct.ellipsoidParent].transform.inverse() * position;
  const Eigen::Vector3<T> ellipsoidPosition = ct.ellipsoidInv.template cast<T>() * localPosition;
  const Eigen::Vector3<T> normalizedPosition = ellipsoidPosition.normalized();
  const Eigen::Vector3<T> projectedPosition = ct.ellipsoid.template cast<T>() * normalizedPosition;
  const Eigen::Vector3<T> diff =
      position - state.jointState[ct.ellipsoidParent].transform * projectedPosition;
  return diff.squaredNorm() * T(kPositionWeight) * limitWeight;
}

template <typename T>
T computeMinMaxGradient(
    const LimitMinMax& data,
    const ModelParametersT<T>& params,
    T tWeight,
    T limitWeight,
    const ParameterSet& enabledParameters,
    Ref<Eigen::VectorX<T>> gradient) {
  if (!enabledParameters.test(data.parameterIndex)) {
    return T(0);
  }
  T error = T(0);
  if (params(data.parameterIndex) < data.limits[0]) {
    const T val = params(data.parameterIndex) - data.limits[0];
    error = val * val * tWeight * limitWeight;
    gradient[data.parameterIndex] += T(2) * val * tWeight * limitWeight;
  }
  if (params(data.parameterIndex) > data.limits[1]) {
    const T val = params(data.parameterIndex) - data.limits[1];
    error = val * val * tWeight * limitWeight;
    gradient[data.parameterIndex] += T(2) * val * tWeight * limitWeight;
  }
  return error;
}

template <typename T>
T computeMinMaxJointGradient(
    const LimitMinMaxJoint& data,
    const SkeletonStateT<T>& state,
    T tWeight,
    T limitWeight,
    const Eigen::VectorX<bool>& activeJointParams,
    const ParameterTransform& parameterTransform,
    Ref<Eigen::VectorX<T>> gradient) {
  const size_t parameterIndex = data.jointIndex * kParametersPerJoint + data.jointParameter;
  MT_CHECK(parameterIndex < static_cast<size_t>(state.jointParameters.size()));
  if (!activeJointParams[parameterIndex]) {
    return T(0);
  }

  T error = T(0);
  if (state.jointParameters(parameterIndex) < data.limits[0]) {
    const T val = state.jointParameters[parameterIndex] - data.limits[0];
    error = val * val * tWeight * limitWeight;
    const T jGrad = T(2) * val * tWeight * limitWeight;
    gradient_jointParams_to_modelParams(jGrad, parameterIndex, parameterTransform, gradient);
  }
  if (state.jointParameters(parameterIndex) > data.limits[1]) {
    const T val = state.jointParameters[parameterIndex] - data.limits[1];
    error = val * val * tWeight * limitWeight;
    const T jGrad = T(2) * val * tWeight * limitWeight;
    gradient_jointParams_to_modelParams(jGrad, parameterIndex, parameterTransform, gradient);
  }
  return error;
}

template <typename T>
T computeLinearGradient(
    const LimitLinear& data,
    const ModelParametersT<T>& params,
    T tWeight,
    T limitWeight,
    const ParameterSet& enabledParameters,
    Ref<Eigen::VectorX<T>> gradient) {
  MT_CHECK(data.referenceIndex < static_cast<size_t>(params.size()));
  MT_CHECK(data.targetIndex < static_cast<size_t>(params.size()));
  if (!isInRange(data, params(data.targetIndex))) {
    return T(0);
  }
  const T residual =
      params(data.targetIndex) * data.scale - data.offset - params(data.referenceIndex);
  const T error = residual * residual * limitWeight * tWeight;
  if (enabledParameters.test(data.targetIndex)) {
    gradient[data.targetIndex] += T(2) * residual * T(data.scale) * limitWeight * tWeight;
  }
  if (enabledParameters.test(data.referenceIndex)) {
    gradient[data.referenceIndex] -= T(2) * residual * limitWeight * tWeight;
  }
  return error;
}

template <typename T>
T computeLinearJointGradient(
    const LimitLinearJoint& data,
    const SkeletonStateT<T>& state,
    T tWeight,
    T limitWeight,
    const Eigen::VectorX<bool>& activeJointParams,
    const ParameterTransform& parameterTransform,
    Ref<Eigen::VectorX<T>> gradient) {
  const size_t referenceParameterIndex =
      data.referenceJointIndex * kParametersPerJoint + data.referenceJointParameter;
  const size_t targetParameterIndex =
      data.targetJointIndex * kParametersPerJoint + data.targetJointParameter;
  MT_CHECK(referenceParameterIndex < static_cast<size_t>(state.jointParameters.size()));
  MT_CHECK(targetParameterIndex < static_cast<size_t>(state.jointParameters.size()));

  if ((!activeJointParams[referenceParameterIndex] && !activeJointParams[targetParameterIndex]) ||
      !isInRange(data, state.jointParameters[targetParameterIndex])) {
    return T(0);
  }

  const T residual = state.jointParameters(targetParameterIndex) * data.scale - data.offset -
      state.jointParameters(referenceParameterIndex);
  const T error = residual * residual * limitWeight * tWeight;

  const T targetGrad = T(2) * residual * T(data.scale) * limitWeight * tWeight;
  gradient_jointParams_to_modelParams(
      targetGrad, targetParameterIndex, parameterTransform, gradient);

  const T referenceGrad = T(-2) * residual * limitWeight * tWeight;
  gradient_jointParams_to_modelParams(
      referenceGrad, referenceParameterIndex, parameterTransform, gradient);

  return error;
}

template <typename T>
T computeHalfPlaneGradient(
    const LimitHalfPlane& data,
    const ModelParametersT<T>& params,
    T tWeight,
    T limitWeight,
    const ParameterSet& enabledParameters,
    Ref<Eigen::VectorX<T>> gradient) {
  if (!enabledParameters.test(data.param1) && !enabledParameters.test(data.param2)) {
    return T(0);
  }
  const Eigen::Vector2<T> p(params(data.param1), params(data.param2));
  const T residual = p.dot(data.normal.template cast<T>()) - data.offset;
  if (residual >= T(0)) {
    return T(0);
  }
  const T error = residual * residual * limitWeight * tWeight;
  if (enabledParameters.test(data.param1)) {
    gradient[data.param1] += T(2) * residual * T(data.normal[0]) * limitWeight * tWeight;
  }
  if (enabledParameters.test(data.param2)) {
    gradient[data.param2] += T(2) * residual * T(data.normal[1]) * limitWeight * tWeight;
  }
  return error;
}

template <typename T>
T computeEllipsoidGradient(
    const LimitEllipsoid& ct,
    const SkeletonStateT<T>& state,
    T tWeight,
    T limitWeight,
    const Eigen::VectorX<bool>& activeJointParams,
    const Skeleton& skeleton,
    const ParameterTransform& parameterTransform,
    Ref<Eigen::VectorX<T>> gradient) {
  const Eigen::Vector3<T> position =
      state.jointState[ct.parent].transform * ct.offset.template cast<T>();
  const Eigen::Vector3<T> localPosition =
      state.jointState[ct.ellipsoidParent].transform.inverse() * position;
  const Eigen::Vector3<T> ellipsoidPosition = ct.ellipsoidInv.template cast<T>() * localPosition;
  const Eigen::Vector3<T> normalizedPosition = ellipsoidPosition.normalized();
  const Eigen::Vector3<T> projectedPosition = ct.ellipsoid.template cast<T>() * normalizedPosition;
  const Eigen::Vector3<T> diff =
      position - state.jointState[ct.ellipsoidParent].transform * projectedPosition;
  const T wgt = T(2) * T(kPositionWeight) * limitWeight * tWeight;

  size_t jointIndex = ct.parent;
  while (jointIndex != ct.ellipsoidParent && jointIndex != kInvalidIndex) {
    MT_CHECK(jointIndex < skeleton.joints.size());
    const auto& jointState = state.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> posd = position - jointState.translation();

    for (size_t d = 0; d < 3; d++) {
      if (activeJointParams[paramIndex + d]) {
        const T val = diff.dot(jointState.getTranslationDerivative(d)) * wgt;
        gradient_jointParams_to_modelParams(val, paramIndex + d, parameterTransform, gradient);
      }
      if (activeJointParams[paramIndex + 3 + d]) {
        const T val = diff.dot(jointState.getRotationDerivative(d, posd)) * wgt;
        gradient_jointParams_to_modelParams(val, paramIndex + 3 + d, parameterTransform, gradient);
      }
    }
    if (activeJointParams[paramIndex + 6]) {
      const T val = diff.dot(jointState.getScaleDerivative(posd)) * wgt;
      gradient_jointParams_to_modelParams(val, paramIndex + 6, parameterTransform, gradient);
    }
    jointIndex = skeleton.joints[jointIndex].parent;
  }

  return diff.squaredNorm() * T(kPositionWeight) * tWeight * limitWeight;
}

template <typename T>
T computeMinMaxJacobian(
    const LimitMinMax& data,
    const ModelParametersT<T>& params,
    T tWeight,
    T limitWeight,
    T wgt,
    const ParameterSet& enabledParameters,
    Ref<Eigen::MatrixX<T>> jacobian,
    Ref<Eigen::VectorX<T>> residual,
    int row) {
  if (!enabledParameters.test(data.parameterIndex)) {
    return T(0);
  }
  if (params(data.parameterIndex) < data.limits[0]) {
    const T val = params(data.parameterIndex) - data.limits[0];
    jacobian(row, data.parameterIndex) = wgt;
    residual(row) = val * wgt;
    return val * val * limitWeight * tWeight;
  }
  if (params(data.parameterIndex) > data.limits[1]) {
    const T val = params(data.parameterIndex) - data.limits[1];
    jacobian(row, data.parameterIndex) = wgt;
    residual(row) = val * wgt;
    return val * val * limitWeight * tWeight;
  }
  return T(0);
}

template <typename T>
T computeMinMaxJointJacobian(
    const LimitMinMaxJoint& data,
    const SkeletonStateT<T>& state,
    T tWeight,
    T limitWeight,
    T wgt,
    const Eigen::VectorX<bool>& activeJointParams,
    const ParameterTransform& parameterTransform,
    Ref<Eigen::MatrixX<T>> jacobian,
    Ref<Eigen::VectorX<T>> residual,
    int row) {
  const size_t jointIndex = data.jointIndex * kParametersPerJoint + data.jointParameter;
  MT_CHECK(jointIndex < static_cast<size_t>(state.jointParameters.size()));
  if (!activeJointParams[jointIndex]) {
    return T(0);
  }
  if (state.jointParameters(jointIndex) < data.limits[0]) {
    const T val = state.jointParameters[jointIndex] - data.limits[0];
    jacobian_jointParams_to_modelParams(
        wgt, Eigen::Index(jointIndex), Eigen::Index(row), parameterTransform, jacobian);
    residual(row) = val * wgt;
    return val * val * limitWeight * tWeight;
  }
  if (state.jointParameters(jointIndex) > data.limits[1]) {
    const T val = state.jointParameters[jointIndex] - data.limits[1];
    jacobian_jointParams_to_modelParams(
        wgt, Eigen::Index(jointIndex), Eigen::Index(row), parameterTransform, jacobian);
    residual(row) = val * wgt;
    return val * val * limitWeight * tWeight;
  }
  return T(0);
}

template <typename T>
T computeLinearJacobian(
    const LimitLinear& data,
    const ModelParametersT<T>& params,
    T tWeight,
    T limitWeight,
    T wgt,
    const ParameterSet& enabledParameters,
    Ref<Eigen::MatrixX<T>> jacobian,
    Ref<Eigen::VectorX<T>> residual,
    int row) {
  MT_CHECK(data.referenceIndex < static_cast<size_t>(params.size()));
  MT_CHECK(data.targetIndex < static_cast<size_t>(params.size()));

  if ((!enabledParameters.test(data.targetIndex) && !enabledParameters.test(data.referenceIndex)) ||
      !isInRange(data, params(data.targetIndex))) {
    return T(0);
  }
  const T res = params(data.targetIndex) * data.scale - data.offset - params(data.referenceIndex);
  residual(row) = res * wgt;
  if (enabledParameters.test(data.targetIndex)) {
    jacobian(row, data.targetIndex) = T(data.scale) * wgt;
  }
  if (enabledParameters.test(data.referenceIndex)) {
    jacobian(row, data.referenceIndex) = -wgt;
  }
  return res * res * limitWeight * tWeight;
}

template <typename T>
T computeLinearJointJacobian(
    const LimitLinearJoint& data,
    const SkeletonStateT<T>& state,
    T tWeight,
    T limitWeight,
    T wgt,
    const Eigen::VectorX<bool>& activeJointParams,
    const ParameterTransform& parameterTransform,
    Ref<Eigen::MatrixX<T>> jacobian,
    Ref<Eigen::VectorX<T>> residual,
    int row) {
  const size_t referenceParameterIndex =
      data.referenceJointIndex * kParametersPerJoint + data.referenceJointParameter;
  const size_t targetParameterIndex =
      data.targetJointIndex * kParametersPerJoint + data.targetJointParameter;
  MT_CHECK(referenceParameterIndex < static_cast<size_t>(state.jointParameters.size()));
  MT_CHECK(targetParameterIndex < static_cast<size_t>(state.jointParameters.size()));

  if ((!activeJointParams[referenceParameterIndex] && !activeJointParams[targetParameterIndex]) ||
      !isInRange(data, state.jointParameters(targetParameterIndex))) {
    return T(0);
  }

  const T res = state.jointParameters(targetParameterIndex) * data.scale - data.offset -
      state.jointParameters(referenceParameterIndex);
  residual(row) = res * wgt;

  if (activeJointParams[targetParameterIndex]) {
    jacobian_jointParams_to_modelParams(
        T(data.scale) * wgt,
        Eigen::Index(targetParameterIndex),
        Eigen::Index(row),
        parameterTransform,
        jacobian);
  }
  if (activeJointParams[referenceParameterIndex]) {
    jacobian_jointParams_to_modelParams(
        -wgt,
        Eigen::Index(referenceParameterIndex),
        Eigen::Index(row),
        parameterTransform,
        jacobian);
  }
  return res * res * limitWeight * tWeight;
}

template <typename T>
T computeHalfPlaneJacobian(
    const LimitHalfPlane& data,
    const ModelParametersT<T>& params,
    T tWeight,
    T limitWeight,
    T wgt,
    const ParameterSet& enabledParameters,
    Ref<Eigen::MatrixX<T>> jacobian,
    Ref<Eigen::VectorX<T>> residual,
    int row) {
  MT_CHECK(data.param1 < static_cast<size_t>(params.size()));
  MT_CHECK(data.param2 < static_cast<size_t>(params.size()));

  if (!enabledParameters.test(data.param1) && !enabledParameters.test(data.param2)) {
    return T(0);
  }
  const Eigen::Vector2<T> p(params(data.param1), params(data.param2));
  const T res = p.dot(data.normal.template cast<T>()) - data.offset;
  if (res >= T(0)) {
    return T(0);
  }
  residual(row) = res * wgt;
  if (enabledParameters.test(data.param1)) {
    jacobian(row, data.param1) = T(data.normal[0]) * wgt;
  }
  if (enabledParameters.test(data.param2)) {
    jacobian(row, data.param2) = T(data.normal[1]) * wgt;
  }
  return res * res * limitWeight * tWeight;
}

template <typename T>
T computeEllipsoidJacobian(
    const LimitEllipsoid& ct,
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& state,
    T totalWeight,
    T limitWeight,
    const Eigen::VectorX<bool>& activeJointParams,
    const Skeleton& skeleton,
    const ParameterTransform& parameterTransform,
    Ref<Eigen::MatrixX<T>> jacobian,
    Ref<Eigen::VectorX<T>> residual,
    int row) {
  const Eigen::Vector3<T> position =
      state.jointState[ct.parent].transform * ct.offset.template cast<T>();
  const Eigen::Vector3<T> localPosition =
      state.jointState[ct.ellipsoidParent].transform.inverse() * position;
  const Eigen::Vector3<T> ellipsoidPosition = ct.ellipsoidInv.template cast<T>() * localPosition;
  const Eigen::Vector3<T> normalizedPosition = ellipsoidPosition.normalized();
  const Eigen::Vector3<T> projectedPosition = ct.ellipsoid.template cast<T>() * normalizedPosition;
  const Eigen::Vector3<T> diff =
      position - state.jointState[ct.ellipsoidParent].transform * projectedPosition;

  Eigen::Ref<Eigen::MatrixX<T>> jac = jacobian.block(row, 0, 3, params.size());
  Eigen::Ref<Eigen::VectorX<T>> res = residual.middleRows(row, 3);
  MT_CHECK(jac.cols() == static_cast<Eigen::Index>(parameterTransform.transform.cols()));
  MT_CHECK(jac.rows() == 3);

  const T jwgt = std::sqrt(totalWeight * T(kPositionWeight) * limitWeight);

  size_t jointIndex = ct.parent;
  while (jointIndex != ct.ellipsoidParent && jointIndex != kInvalidIndex) {
    MT_CHECK(jointIndex < skeleton.joints.size());
    const auto& jointState = state.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> posd = position - jointState.translation();

    for (size_t d = 0; d < 3; d++) {
      if (activeJointParams[paramIndex + d]) {
        const Eigen::Vector3<T> jc = jointState.getTranslationDerivative(d) * jwgt;
        for (auto index = parameterTransform.transform.outerIndexPtr()[paramIndex + d];
             index < parameterTransform.transform.outerIndexPtr()[paramIndex + d + 1];
             ++index) {
          jac.col(parameterTransform.transform.innerIndexPtr()[index]) +=
              jc * parameterTransform.transform.valuePtr()[index];
        }
      }
      if (activeJointParams[paramIndex + 3 + d]) {
        const Eigen::Vector3<T> jc = jointState.getRotationDerivative(d, posd) * jwgt;
        for (auto index = parameterTransform.transform.outerIndexPtr()[paramIndex + 3 + d];
             index < parameterTransform.transform.outerIndexPtr()[paramIndex + 3 + d + 1];
             ++index) {
          jac.col(parameterTransform.transform.innerIndexPtr()[index]) +=
              jc * parameterTransform.transform.valuePtr()[index];
        }
      }
    }
    if (activeJointParams[paramIndex + 6]) {
      const Eigen::Vector3<T> jc = jointState.getScaleDerivative(posd) * jwgt;
      for (auto index = parameterTransform.transform.outerIndexPtr()[paramIndex + 6];
           index < parameterTransform.transform.outerIndexPtr()[paramIndex + 6 + 1];
           ++index) {
        jac.col(parameterTransform.transform.innerIndexPtr()[index]) +=
            jc * parameterTransform.transform.valuePtr()[index];
      }
    }
    jointIndex = skeleton.joints[jointIndex].parent;
  }

  res = diff * jwgt;
  return jwgt * jwgt * diff.squaredNorm();
}

} // namespace

template <typename T>
LimitErrorFunctionT<T>::LimitErrorFunctionT(
    const Skeleton& skel,
    const ParameterTransform& pt,
    const ParameterLimits& pl)
    : SkeletonErrorFunctionT<T>(skel, pt), limits_(pl) {}

template <typename T>
LimitErrorFunctionT<T>::LimitErrorFunctionT(const Character& character)
    : LimitErrorFunctionT(
          character.skeleton,
          character.parameterTransform,
          character.parameterLimits) {}

template <typename T>
LimitErrorFunctionT<T>::LimitErrorFunctionT(const Character& character, const ParameterLimits& pl)
    : LimitErrorFunctionT(character.skeleton, character.parameterTransform, pl) {}

template <typename T>
double LimitErrorFunctionT<T>::getError(
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */) {
  MT_PROFILE_FUNCTION();

  MT_CHECK(
      state.jointParameters.size() ==
      gsl::narrow<Eigen::Index>(this->skeleton_.joints.size() * kParametersPerJoint));

  double error = 0.0;

  for (const auto& limit : limits_) {
    const T limitWeight = T(limit.weight);
    switch (limit.type) {
      case MinMax:
        error +=
            computeMinMaxError(limit.data.minMax, params, limitWeight, this->enabledParameters_);
        break;
      case MinMaxJoint:
        error += computeMinMaxJointError(
            limit.data.minMaxJoint, state, limitWeight, this->activeJointParams_);
        break;
      case MinMaxJointPassive:
        break;
      case Linear:
        error +=
            computeLinearError(limit.data.linear, params, limitWeight, this->enabledParameters_);
        break;
      case LinearJoint:
        error += computeLinearJointError(
            limit.data.linearJoint, state, limitWeight, this->activeJointParams_);
        break;
      case HalfPlane:
        error += computeHalfPlaneError(
            limit.data.halfPlane, params, limitWeight, this->enabledParameters_);
        break;
      case Ellipsoid:
        error += computeEllipsoidError(limit.data.ellipsoid, state, limitWeight);
        break;
      default:
        MT_THROW("Unknown parameter type for joint limit");
        break;
    }
  }

  return error * kLimitWeight * this->weight_;
}

template <typename T>
double LimitErrorFunctionT<T>::getGradient(
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */,
    Ref<Eigen::VectorX<T>> gradient) {
  MT_PROFILE_FUNCTION();

  const auto& parameterTransform = this->parameterTransform_;
  double error = 0.0;
  const T tWeight = kLimitWeight * this->weight_;

  for (const auto& limit : limits_) {
    const T limitWeight = T(limit.weight);
    switch (limit.type) {
      case MinMax:
        error += computeMinMaxGradient(
            limit.data.minMax, params, tWeight, limitWeight, this->enabledParameters_, gradient);
        break;
      case MinMaxJoint:
        error += computeMinMaxJointGradient(
            limit.data.minMaxJoint,
            state,
            tWeight,
            limitWeight,
            this->activeJointParams_,
            parameterTransform,
            gradient);
        break;
      case MinMaxJointPassive:
        break;
      case Linear:
        error += computeLinearGradient(
            limit.data.linear, params, tWeight, limitWeight, this->enabledParameters_, gradient);
        break;
      case LinearJoint:
        error += computeLinearJointGradient(
            limit.data.linearJoint,
            state,
            tWeight,
            limitWeight,
            this->activeJointParams_,
            parameterTransform,
            gradient);
        break;
      case HalfPlane:
        error += computeHalfPlaneGradient(
            limit.data.halfPlane, params, tWeight, limitWeight, this->enabledParameters_, gradient);
        break;
      case Ellipsoid:
        error += computeEllipsoidGradient(
            limit.data.ellipsoid,
            state,
            tWeight,
            limitWeight,
            this->activeJointParams_,
            this->skeleton_,
            parameterTransform,
            gradient);
        break;
      default:
        MT_THROW("Unknown parameter type for joint limit");
        break;
    }
  }

  return error;
}

template <typename T>
double LimitErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */,
    Ref<Eigen::MatrixX<T>> jacobian,
    Ref<Eigen::VectorX<T>> residual,
    int& usedRows) {
  MT_PROFILE_FUNCTION();

  const auto& parameterTransform = this->parameterTransform_;
  double error = 0.0;
  const T tWeight = kLimitWeight * this->weight_;

  jacobian.setZero();
  residual.setZero();

  int count = 0;
  for (const auto& limit : limits_) {
    const T limitWeight = T(limit.weight);
    const T wgt = std::sqrt(T(kLimitWeight) * this->weight_ * limitWeight);
    switch (limit.type) {
      case MinMax:
        error += computeMinMaxJacobian(
            limit.data.minMax,
            params,
            tWeight,
            limitWeight,
            wgt,
            this->enabledParameters_,
            jacobian,
            residual,
            count);
        count++;
        break;
      case MinMaxJoint:
        error += computeMinMaxJointJacobian(
            limit.data.minMaxJoint,
            state,
            tWeight,
            limitWeight,
            wgt,
            this->activeJointParams_,
            parameterTransform,
            jacobian,
            residual,
            count);
        count++;
        break;
      case MinMaxJointPassive:
        break;
      case Linear:
        error += computeLinearJacobian(
            limit.data.linear,
            params,
            tWeight,
            limitWeight,
            wgt,
            this->enabledParameters_,
            jacobian,
            residual,
            count);
        count++;
        break;
      case LinearJoint:
        error += computeLinearJointJacobian(
            limit.data.linearJoint,
            state,
            tWeight,
            limitWeight,
            wgt,
            this->activeJointParams_,
            parameterTransform,
            jacobian,
            residual,
            count);
        count++;
        break;
      case HalfPlane:
        error += computeHalfPlaneJacobian(
            limit.data.halfPlane,
            params,
            tWeight,
            limitWeight,
            wgt,
            this->enabledParameters_,
            jacobian,
            residual,
            count);
        count++;
        break;
      case Ellipsoid:
        error += computeEllipsoidJacobian(
            limit.data.ellipsoid,
            params,
            state,
            tWeight,
            limitWeight,
            this->activeJointParams_,
            this->skeleton_,
            parameterTransform,
            jacobian,
            residual,
            count);
        count += 3;
        break;
      default:
        MT_THROW("Unknown parameter type for joint limit");
        break;
    }
  }

  usedRows = count;
  return error;
}

template <typename T>
size_t LimitErrorFunctionT<T>::getJacobianSize() const {
  size_t count = 0;
  for (const auto& limit : limits_) {
    switch (limit.type) {
      case MinMax:
      case MinMaxJoint:
      case Linear:
      case LinearJoint:
      case HalfPlane:
        count++;
        break;
      case MinMaxJointPassive:
        break;
      case Ellipsoid:
        count += 3;
        break;
      default:
        MT_THROW("Unknown parameter type for joint limit");
        break;
    }
  }
  return count;
}

template <typename T>
void LimitErrorFunctionT<T>::setLimits(const ParameterLimits& lm) {
  limits_ = lm;
}

template <typename T>
void LimitErrorFunctionT<T>::setLimits(const Character& character) {
  limits_ = character.parameterLimits;
}

template class LimitErrorFunctionT<float>;
template class LimitErrorFunctionT<double>;

} // namespace momentum
