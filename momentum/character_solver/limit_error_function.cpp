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

// Each constraint contributes `w * loss(s)` where `s` is the raw squared residual and `w` is the
// product of the weighting terms. The helpers below are templated on whether the loss is a plain
// L2 loss:
//   - IsL2 == true:  the squared error is used directly. `invC2` (the loss scale, 1 for the default
//     c = 1) is folded into the surrounding weight by the caller, so these helpers do no
//     loss-function calls at all and match the original hand-optimized quadratic penalty.
//   - IsL2 == false: the squared error is transformed by GeneralizedLossT::value/deriv.

template <typename T, bool IsL2>
T computeMinMaxError(
    const LimitMinMax& data,
    const ModelParametersT<T>& params,
    T limitWeight,
    const GeneralizedLossT<T>& loss,
    const ParameterSet& enabledParameters) {
  if (!enabledParameters.test(data.parameterIndex)) {
    return T(0);
  }
  T error = T(0);
  if (params(data.parameterIndex) < data.limits[0]) {
    const T val = data.limits[0] - params(data.parameterIndex);
    const T sqrError = val * val;
    if constexpr (IsL2) {
      error = limitWeight * sqrError;
    } else {
      error = limitWeight * loss.value(sqrError);
    }
  }
  if (params(data.parameterIndex) > data.limits[1]) {
    const T val = data.limits[1] - params(data.parameterIndex);
    const T sqrError = val * val;
    if constexpr (IsL2) {
      error = limitWeight * sqrError;
    } else {
      error = limitWeight * loss.value(sqrError);
    }
  }
  return error;
}

template <typename T, bool IsL2>
T computeMinMaxJointError(
    const LimitMinMaxJoint& data,
    const SkeletonStateT<T>& state,
    T limitWeight,
    const GeneralizedLossT<T>& loss,
    const Eigen::VectorX<bool>& activeJointParams) {
  const size_t parameterIndex = data.jointIndex * kParametersPerJoint + data.jointParameter;
  MT_CHECK(parameterIndex < static_cast<size_t>(state.jointParameters.size()));
  if (!activeJointParams[parameterIndex]) {
    return T(0);
  }
  T error = T(0);
  if (state.jointParameters(parameterIndex) < data.limits[0]) {
    const T val = data.limits[0] - state.jointParameters[parameterIndex];
    const T sqrError = val * val;
    if constexpr (IsL2) {
      error = limitWeight * sqrError;
    } else {
      error = limitWeight * loss.value(sqrError);
    }
  }
  if (state.jointParameters(parameterIndex) > data.limits[1]) {
    const T val = data.limits[1] - state.jointParameters[parameterIndex];
    const T sqrError = val * val;
    if constexpr (IsL2) {
      error = limitWeight * sqrError;
    } else {
      error = limitWeight * loss.value(sqrError);
    }
  }
  return error;
}

template <typename T, bool IsL2>
T computeLinearError(
    const LimitLinear& data,
    const ModelParametersT<T>& params,
    T limitWeight,
    const GeneralizedLossT<T>& loss,
    const ParameterSet& enabledParameters) {
  MT_CHECK(data.referenceIndex < static_cast<size_t>(params.size()));
  MT_CHECK(data.targetIndex < static_cast<size_t>(params.size()));
  if ((!enabledParameters.test(data.targetIndex) && !enabledParameters.test(data.referenceIndex)) ||
      !isInRange(data, params(data.targetIndex))) {
    return T(0);
  }
  const T residual =
      params(data.targetIndex) * data.scale - data.offset - params(data.referenceIndex);
  const T sqrError = residual * residual;
  if constexpr (IsL2) {
    return limitWeight * sqrError;
  } else {
    return limitWeight * loss.value(sqrError);
  }
}

template <typename T, bool IsL2>
T computeLinearJointError(
    const LimitLinearJoint& data,
    const SkeletonStateT<T>& state,
    T limitWeight,
    const GeneralizedLossT<T>& loss,
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
  const T sqrError = residual * residual;
  if constexpr (IsL2) {
    return limitWeight * sqrError;
  } else {
    return limitWeight * loss.value(sqrError);
  }
}

template <typename T, bool IsL2>
T computeHalfPlaneError(
    const LimitHalfPlane& data,
    const ModelParametersT<T>& params,
    T limitWeight,
    const GeneralizedLossT<T>& loss,
    const ParameterSet& enabledParameters) {
  MT_CHECK(data.param1 < static_cast<size_t>(params.size()));
  MT_CHECK(data.param2 < static_cast<size_t>(params.size()));
  if (!enabledParameters.test(data.param1) && !enabledParameters.test(data.param2)) {
    return T(0);
  }
  const Eigen::Vector2<T> p(params(data.param1), params(data.param2));
  const T residual = p.dot(data.normal.template cast<T>()) - data.offset;
  if (residual < 0) {
    const T sqrError = residual * residual;
    if constexpr (IsL2) {
      return limitWeight * sqrError;
    } else {
      return limitWeight * loss.value(sqrError);
    }
  }
  return T(0);
}

template <typename T, bool IsL2>
T computeEllipsoidError(
    const LimitEllipsoid& ct,
    const SkeletonStateT<T>& state,
    T limitWeight,
    const GeneralizedLossT<T>& loss) {
  const Eigen::Vector3<T> position =
      state.jointState[ct.parent].transform * ct.offset.template cast<T>();
  const Eigen::Vector3<T> localPosition =
      state.jointState[ct.ellipsoidParent].transform.inverse() * position;
  const Eigen::Vector3<T> ellipsoidPosition = ct.ellipsoidInv.template cast<T>() * localPosition;
  const Eigen::Vector3<T> normalizedPosition = ellipsoidPosition.normalized();
  const Eigen::Vector3<T> projectedPosition = ct.ellipsoid.template cast<T>() * normalizedPosition;
  const Eigen::Vector3<T> diff =
      position - state.jointState[ct.ellipsoidParent].transform * projectedPosition;
  const T sqrError = diff.squaredNorm();
  if constexpr (IsL2) {
    return T(kPositionWeight) * limitWeight * sqrError;
  } else {
    return T(kPositionWeight) * limitWeight * loss.value(sqrError);
  }
}

template <typename T, bool IsL2>
T computeMinMaxGradient(
    const LimitMinMax& data,
    const ModelParametersT<T>& params,
    T tWeight,
    T limitWeight,
    const GeneralizedLossT<T>& loss,
    const ParameterSet& enabledParameters,
    Ref<Eigen::VectorX<T>> gradient) {
  if (!enabledParameters.test(data.parameterIndex)) {
    return T(0);
  }
  T error = T(0);
  if (params(data.parameterIndex) < data.limits[0]) {
    const T val = params(data.parameterIndex) - data.limits[0];
    const T sqrError = val * val;
    if constexpr (IsL2) {
      error = tWeight * limitWeight * sqrError;
      gradient[data.parameterIndex] += T(2) * val * tWeight * limitWeight;
    } else {
      error = tWeight * limitWeight * loss.value(sqrError);
      gradient[data.parameterIndex] += T(2) * val * tWeight * limitWeight * loss.deriv(sqrError);
    }
  }
  if (params(data.parameterIndex) > data.limits[1]) {
    const T val = params(data.parameterIndex) - data.limits[1];
    const T sqrError = val * val;
    if constexpr (IsL2) {
      error = tWeight * limitWeight * sqrError;
      gradient[data.parameterIndex] += T(2) * val * tWeight * limitWeight;
    } else {
      error = tWeight * limitWeight * loss.value(sqrError);
      gradient[data.parameterIndex] += T(2) * val * tWeight * limitWeight * loss.deriv(sqrError);
    }
  }
  return error;
}

template <typename T, bool IsL2>
T computeMinMaxJointGradient(
    const LimitMinMaxJoint& data,
    const SkeletonStateT<T>& state,
    T tWeight,
    T limitWeight,
    const GeneralizedLossT<T>& loss,
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
    const T sqrError = val * val;
    T jGrad;
    if constexpr (IsL2) {
      error = tWeight * limitWeight * sqrError;
      jGrad = T(2) * val * tWeight * limitWeight;
    } else {
      error = tWeight * limitWeight * loss.value(sqrError);
      jGrad = T(2) * val * tWeight * limitWeight * loss.deriv(sqrError);
    }
    gradient_jointParams_to_modelParams(jGrad, parameterIndex, parameterTransform, gradient);
  }
  if (state.jointParameters(parameterIndex) > data.limits[1]) {
    const T val = state.jointParameters[parameterIndex] - data.limits[1];
    const T sqrError = val * val;
    T jGrad;
    if constexpr (IsL2) {
      error = tWeight * limitWeight * sqrError;
      jGrad = T(2) * val * tWeight * limitWeight;
    } else {
      error = tWeight * limitWeight * loss.value(sqrError);
      jGrad = T(2) * val * tWeight * limitWeight * loss.deriv(sqrError);
    }
    gradient_jointParams_to_modelParams(jGrad, parameterIndex, parameterTransform, gradient);
  }
  return error;
}

template <typename T, bool IsL2>
T computeLinearGradient(
    const LimitLinear& data,
    const ModelParametersT<T>& params,
    T tWeight,
    T limitWeight,
    const GeneralizedLossT<T>& loss,
    const ParameterSet& enabledParameters,
    Ref<Eigen::VectorX<T>> gradient) {
  MT_CHECK(data.referenceIndex < static_cast<size_t>(params.size()));
  MT_CHECK(data.targetIndex < static_cast<size_t>(params.size()));
  if (!isInRange(data, params(data.targetIndex))) {
    return T(0);
  }
  const T residual =
      params(data.targetIndex) * data.scale - data.offset - params(data.referenceIndex);
  const T sqrError = residual * residual;
  T error;
  T gradScale;
  if constexpr (IsL2) {
    error = limitWeight * tWeight * sqrError;
    gradScale = T(2) * residual * limitWeight * tWeight;
  } else {
    error = limitWeight * tWeight * loss.value(sqrError);
    gradScale = T(2) * residual * limitWeight * tWeight * loss.deriv(sqrError);
  }
  if (enabledParameters.test(data.targetIndex)) {
    gradient[data.targetIndex] += gradScale * T(data.scale);
  }
  if (enabledParameters.test(data.referenceIndex)) {
    gradient[data.referenceIndex] -= gradScale;
  }
  return error;
}

template <typename T, bool IsL2>
T computeLinearJointGradient(
    const LimitLinearJoint& data,
    const SkeletonStateT<T>& state,
    T tWeight,
    T limitWeight,
    const GeneralizedLossT<T>& loss,
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
  const T sqrError = residual * residual;
  T error;
  T gradScale;
  if constexpr (IsL2) {
    error = limitWeight * tWeight * sqrError;
    gradScale = T(2) * residual * limitWeight * tWeight;
  } else {
    error = limitWeight * tWeight * loss.value(sqrError);
    gradScale = T(2) * residual * limitWeight * tWeight * loss.deriv(sqrError);
  }

  const T targetGrad = gradScale * T(data.scale);
  gradient_jointParams_to_modelParams(
      targetGrad, targetParameterIndex, parameterTransform, gradient);

  const T referenceGrad = -gradScale;
  gradient_jointParams_to_modelParams(
      referenceGrad, referenceParameterIndex, parameterTransform, gradient);

  return error;
}

template <typename T, bool IsL2>
T computeHalfPlaneGradient(
    const LimitHalfPlane& data,
    const ModelParametersT<T>& params,
    T tWeight,
    T limitWeight,
    const GeneralizedLossT<T>& loss,
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
  const T sqrError = residual * residual;
  T error;
  T gradScale;
  if constexpr (IsL2) {
    error = limitWeight * tWeight * sqrError;
    gradScale = T(2) * residual * limitWeight * tWeight;
  } else {
    error = limitWeight * tWeight * loss.value(sqrError);
    gradScale = T(2) * residual * limitWeight * tWeight * loss.deriv(sqrError);
  }
  if (enabledParameters.test(data.param1)) {
    gradient[data.param1] += gradScale * T(data.normal[0]);
  }
  if (enabledParameters.test(data.param2)) {
    gradient[data.param2] += gradScale * T(data.normal[1]);
  }
  return error;
}

template <typename T, bool IsL2>
T computeEllipsoidGradient(
    const LimitEllipsoid& ct,
    const SkeletonStateT<T>& state,
    T tWeight,
    T limitWeight,
    const GeneralizedLossT<T>& loss,
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
  const T sqrError = diff.squaredNorm();

  // wgt is the gradient scale: d(w * loss(s))/dq has the factor 2 * w * loss'(s), with
  // w = kPositionWeight * limitWeight * tWeight. For L2, loss'(s) is folded into tWeight by the
  // caller.
  T wgt;
  if constexpr (IsL2) {
    wgt = T(2) * T(kPositionWeight) * limitWeight * tWeight;
  } else {
    wgt = T(2) * T(kPositionWeight) * limitWeight * tWeight * loss.deriv(sqrError);
  }

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

  if constexpr (IsL2) {
    return T(kPositionWeight) * limitWeight * tWeight * sqrError;
  } else {
    return T(kPositionWeight) * limitWeight * tWeight * loss.value(sqrError);
  }
}

template <typename T, bool IsL2>
T computeMinMaxJacobian(
    const LimitMinMax& data,
    const ModelParametersT<T>& params,
    T tWeight,
    T limitWeight,
    T wgt,
    const GeneralizedLossT<T>& loss,
    const ParameterSet& enabledParameters,
    Ref<Eigen::MatrixX<T>> jacobian,
    Ref<Eigen::VectorX<T>> residual,
    int row) {
  if (!enabledParameters.test(data.parameterIndex)) {
    return T(0);
  }
  if (params(data.parameterIndex) < data.limits[0]) {
    const T val = params(data.parameterIndex) - data.limits[0];
    const T sqrError = val * val;
    if constexpr (IsL2) {
      jacobian(row, data.parameterIndex) = wgt;
      residual(row) = val * wgt;
      return tWeight * limitWeight * sqrError;
    } else {
      const T wgtLoss = std::sqrt(tWeight * limitWeight * loss.deriv(sqrError));
      jacobian(row, data.parameterIndex) = wgtLoss;
      residual(row) = val * wgtLoss;
      return tWeight * limitWeight * loss.value(sqrError);
    }
  }
  if (params(data.parameterIndex) > data.limits[1]) {
    const T val = params(data.parameterIndex) - data.limits[1];
    const T sqrError = val * val;
    if constexpr (IsL2) {
      jacobian(row, data.parameterIndex) = wgt;
      residual(row) = val * wgt;
      return tWeight * limitWeight * sqrError;
    } else {
      const T wgtLoss = std::sqrt(tWeight * limitWeight * loss.deriv(sqrError));
      jacobian(row, data.parameterIndex) = wgtLoss;
      residual(row) = val * wgtLoss;
      return tWeight * limitWeight * loss.value(sqrError);
    }
  }
  return T(0);
}

template <typename T, bool IsL2>
T computeMinMaxJointJacobian(
    const LimitMinMaxJoint& data,
    const SkeletonStateT<T>& state,
    T tWeight,
    T limitWeight,
    T wgt,
    const GeneralizedLossT<T>& loss,
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
    const T sqrError = val * val;
    T wgtLoss;
    T error;
    if constexpr (IsL2) {
      wgtLoss = wgt;
      error = tWeight * limitWeight * sqrError;
    } else {
      wgtLoss = std::sqrt(tWeight * limitWeight * loss.deriv(sqrError));
      error = tWeight * limitWeight * loss.value(sqrError);
    }
    jacobian_jointParams_to_modelParams(
        wgtLoss, Eigen::Index(jointIndex), Eigen::Index(row), parameterTransform, jacobian);
    residual(row) = val * wgtLoss;
    return error;
  }
  if (state.jointParameters(jointIndex) > data.limits[1]) {
    const T val = state.jointParameters[jointIndex] - data.limits[1];
    const T sqrError = val * val;
    T wgtLoss;
    T error;
    if constexpr (IsL2) {
      wgtLoss = wgt;
      error = tWeight * limitWeight * sqrError;
    } else {
      wgtLoss = std::sqrt(tWeight * limitWeight * loss.deriv(sqrError));
      error = tWeight * limitWeight * loss.value(sqrError);
    }
    jacobian_jointParams_to_modelParams(
        wgtLoss, Eigen::Index(jointIndex), Eigen::Index(row), parameterTransform, jacobian);
    residual(row) = val * wgtLoss;
    return error;
  }
  return T(0);
}

template <typename T, bool IsL2>
T computeLinearJacobian(
    const LimitLinear& data,
    const ModelParametersT<T>& params,
    T tWeight,
    T limitWeight,
    T wgt,
    const GeneralizedLossT<T>& loss,
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
  const T sqrError = res * res;
  T wgtLoss;
  T error;
  if constexpr (IsL2) {
    wgtLoss = wgt;
    error = tWeight * limitWeight * sqrError;
  } else {
    wgtLoss = std::sqrt(tWeight * limitWeight * loss.deriv(sqrError));
    error = tWeight * limitWeight * loss.value(sqrError);
  }
  residual(row) = res * wgtLoss;
  if (enabledParameters.test(data.targetIndex)) {
    jacobian(row, data.targetIndex) = T(data.scale) * wgtLoss;
  }
  if (enabledParameters.test(data.referenceIndex)) {
    jacobian(row, data.referenceIndex) = -wgtLoss;
  }
  return error;
}

template <typename T, bool IsL2>
T computeLinearJointJacobian(
    const LimitLinearJoint& data,
    const SkeletonStateT<T>& state,
    T tWeight,
    T limitWeight,
    T wgt,
    const GeneralizedLossT<T>& loss,
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
  const T sqrError = res * res;
  T wgtLoss;
  T error;
  if constexpr (IsL2) {
    wgtLoss = wgt;
    error = tWeight * limitWeight * sqrError;
  } else {
    wgtLoss = std::sqrt(tWeight * limitWeight * loss.deriv(sqrError));
    error = tWeight * limitWeight * loss.value(sqrError);
  }
  residual(row) = res * wgtLoss;

  if (activeJointParams[targetParameterIndex]) {
    jacobian_jointParams_to_modelParams(
        T(data.scale) * wgtLoss,
        Eigen::Index(targetParameterIndex),
        Eigen::Index(row),
        parameterTransform,
        jacobian);
  }
  if (activeJointParams[referenceParameterIndex]) {
    jacobian_jointParams_to_modelParams(
        -wgtLoss,
        Eigen::Index(referenceParameterIndex),
        Eigen::Index(row),
        parameterTransform,
        jacobian);
  }
  return error;
}

template <typename T, bool IsL2>
T computeHalfPlaneJacobian(
    const LimitHalfPlane& data,
    const ModelParametersT<T>& params,
    T tWeight,
    T limitWeight,
    T wgt,
    const GeneralizedLossT<T>& loss,
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
  const T sqrError = res * res;
  T wgtLoss;
  T error;
  if constexpr (IsL2) {
    wgtLoss = wgt;
    error = tWeight * limitWeight * sqrError;
  } else {
    wgtLoss = std::sqrt(tWeight * limitWeight * loss.deriv(sqrError));
    error = tWeight * limitWeight * loss.value(sqrError);
  }
  residual(row) = res * wgtLoss;
  if (enabledParameters.test(data.param1)) {
    jacobian(row, data.param1) = T(data.normal[0]) * wgtLoss;
  }
  if (enabledParameters.test(data.param2)) {
    jacobian(row, data.param2) = T(data.normal[1]) * wgtLoss;
  }
  return error;
}

template <typename T, bool IsL2>
T computeEllipsoidJacobian(
    const LimitEllipsoid& ct,
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& state,
    T totalWeight,
    T limitWeight,
    const GeneralizedLossT<T>& loss,
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
  const T sqrError = diff.squaredNorm();

  Eigen::Ref<Eigen::MatrixX<T>> jac = jacobian.block(row, 0, 3, params.size());
  Eigen::Ref<Eigen::VectorX<T>> res = residual.middleRows(row, 3);
  MT_CHECK(jac.cols() == static_cast<Eigen::Index>(parameterTransform.transform.cols()));
  MT_CHECK(jac.rows() == 3);

  // jwgt is the residual/Jacobian scale sqrt(w * loss'(s)), with w = totalWeight * kPositionWeight
  // * limitWeight. For L2, loss'(s) is folded into totalWeight by the caller.
  T jwgt;
  if constexpr (IsL2) {
    jwgt = std::sqrt(totalWeight * T(kPositionWeight) * limitWeight);
  } else {
    jwgt = std::sqrt(totalWeight * T(kPositionWeight) * limitWeight * loss.deriv(sqrError));
  }

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
  if constexpr (IsL2) {
    return totalWeight * T(kPositionWeight) * limitWeight * sqrError;
  } else {
    return totalWeight * T(kPositionWeight) * limitWeight * loss.value(sqrError);
  }
}

} // namespace

template <typename T>
LimitErrorFunctionT<T>::LimitErrorFunctionT(
    const Skeleton& skel,
    const ParameterTransform& pt,
    const ParameterLimits& pl,
    const T& lossAlpha,
    const T& lossC)
    : SkeletonErrorFunctionT<T>(skel, pt), limits_(pl), loss_(lossAlpha, lossC) {}

template <typename T>
LimitErrorFunctionT<T>::LimitErrorFunctionT(
    const Character& character,
    const T& lossAlpha,
    const T& lossC)
    : LimitErrorFunctionT(
          character.skeleton,
          character.parameterTransform,
          character.parameterLimits,
          lossAlpha,
          lossC) {}

template <typename T>
LimitErrorFunctionT<T>::LimitErrorFunctionT(
    const Character& character,
    const ParameterLimits& pl,
    const T& lossAlpha,
    const T& lossC)
    : LimitErrorFunctionT(character.skeleton, character.parameterTransform, pl, lossAlpha, lossC) {}

template <typename T>
template <bool IsL2>
[[nodiscard]] double LimitErrorFunctionT<T>::getErrorImpl(
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& state) const {
  double error = 0.0;

  for (const auto& limit : limits_) {
    const T limitWeight = T(limit.weight);
    switch (limit.type) {
      case MinMax:
        error += computeMinMaxError<T, IsL2>(
            limit.data.minMax, params, limitWeight, loss_, this->enabledParameters_);
        break;
      case MinMaxJoint:
        error += computeMinMaxJointError<T, IsL2>(
            limit.data.minMaxJoint, state, limitWeight, loss_, this->activeJointParams_);
        break;
      case MinMaxJointPassive:
        break;
      case Linear:
        error += computeLinearError<T, IsL2>(
            limit.data.linear, params, limitWeight, loss_, this->enabledParameters_);
        break;
      case LinearJoint:
        error += computeLinearJointError<T, IsL2>(
            limit.data.linearJoint, state, limitWeight, loss_, this->activeJointParams_);
        break;
      case HalfPlane:
        error += computeHalfPlaneError<T, IsL2>(
            limit.data.halfPlane, params, limitWeight, loss_, this->enabledParameters_);
        break;
      case Ellipsoid:
        error += computeEllipsoidError<T, IsL2>(limit.data.ellipsoid, state, limitWeight, loss_);
        break;
      case LimitTypeCount:
      default:
        MT_THROW("Unknown parameter type for joint limit");
        break;
    }
  }

  // For an L2 loss the per-limit terms above are raw squared residuals; the loss scale 1/c^2 is
  // applied once here. For a non-L2 loss it is already baked into loss_.value().
  if constexpr (IsL2) {
    return error * kLimitWeight * this->weight_ * loss_.invC2();
  } else {
    return error * kLimitWeight * this->weight_;
  }
}

template <typename T>
double LimitErrorFunctionT<T>::getError(
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */) {
  MT_PROFILE_FUNCTION();

  MT_CHECK(
      state.jointParameters.size() ==
      gsl::narrow<Eigen::Index>(this->skeleton_.joints.size() * kParametersPerJoint));

  return loss_.isL2() ? getErrorImpl<true>(params, state) : getErrorImpl<false>(params, state);
}

template <typename T>
template <bool IsL2>
[[nodiscard]] double LimitErrorFunctionT<T>::getGradientImpl(
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& state,
    Ref<Eigen::VectorX<T>> gradient) const {
  const auto& parameterTransform = this->parameterTransform_;
  double error = 0.0;

  // tWeight folds in the loss scale 1/c^2 for the L2 path so the per-constraint code below needs no
  // loss-function calls.
  T tWeight = kLimitWeight * this->weight_;
  if constexpr (IsL2) {
    tWeight *= loss_.invC2();
  }

  for (const auto& limit : limits_) {
    const T limitWeight = T(limit.weight);
    switch (limit.type) {
      case MinMax:
        error += computeMinMaxGradient<T, IsL2>(
            limit.data.minMax,
            params,
            tWeight,
            limitWeight,
            loss_,
            this->enabledParameters_,
            gradient);
        break;
      case MinMaxJoint:
        error += computeMinMaxJointGradient<T, IsL2>(
            limit.data.minMaxJoint,
            state,
            tWeight,
            limitWeight,
            loss_,
            this->activeJointParams_,
            parameterTransform,
            gradient);
        break;
      case MinMaxJointPassive:
        break;
      case Linear:
        error += computeLinearGradient<T, IsL2>(
            limit.data.linear,
            params,
            tWeight,
            limitWeight,
            loss_,
            this->enabledParameters_,
            gradient);
        break;
      case LinearJoint:
        error += computeLinearJointGradient<T, IsL2>(
            limit.data.linearJoint,
            state,
            tWeight,
            limitWeight,
            loss_,
            this->activeJointParams_,
            parameterTransform,
            gradient);
        break;
      case HalfPlane:
        error += computeHalfPlaneGradient<T, IsL2>(
            limit.data.halfPlane,
            params,
            tWeight,
            limitWeight,
            loss_,
            this->enabledParameters_,
            gradient);
        break;
      case Ellipsoid:
        error += computeEllipsoidGradient<T, IsL2>(
            limit.data.ellipsoid,
            state,
            tWeight,
            limitWeight,
            loss_,
            this->activeJointParams_,
            this->skeleton_,
            parameterTransform,
            gradient);
        break;
      case LimitTypeCount:
      default:
        MT_THROW("Unknown parameter type for joint limit");
        break;
    }
  }

  return error;
}

template <typename T>
double LimitErrorFunctionT<T>::getGradient(
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */,
    Ref<Eigen::VectorX<T>> gradient) {
  MT_PROFILE_FUNCTION();

  return loss_.isL2() ? getGradientImpl<true>(params, state, gradient)
                      : getGradientImpl<false>(params, state, gradient);
}

template <typename T>
template <bool IsL2>
double LimitErrorFunctionT<T>::getJacobianImpl(
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& state,
    Ref<Eigen::MatrixX<T>> jacobian,
    Ref<Eigen::VectorX<T>> residual,
    int& usedRows) const {
  const auto& parameterTransform = this->parameterTransform_;
  double error = 0.0;

  // tWeight folds in the loss scale 1/c^2 for the L2 path. The Jacobian weight sqrt(tWeight *
  // limitWeight) is then a constant per limit (it does not depend on the residual), so it is
  // computed once here rather than inside the per-constraint helpers. This matches the original
  // L2-only implementation; the non-L2 path cannot do this because its weight depends on the
  // per-constraint residual via loss'(s).
  T tWeight = kLimitWeight * this->weight_;
  if constexpr (IsL2) {
    tWeight *= loss_.invC2();
  }

  jacobian.setZero();
  residual.setZero();

  int count = 0;
  for (const auto& limit : limits_) {
    const T limitWeight = T(limit.weight);
    T wgt = T(0);
    if constexpr (IsL2) {
      wgt = std::sqrt(tWeight * limitWeight);
    }
    switch (limit.type) {
      case MinMax:
        error += computeMinMaxJacobian<T, IsL2>(
            limit.data.minMax,
            params,
            tWeight,
            limitWeight,
            wgt,
            loss_,
            this->enabledParameters_,
            jacobian,
            residual,
            count);
        count++;
        break;
      case MinMaxJoint:
        error += computeMinMaxJointJacobian<T, IsL2>(
            limit.data.minMaxJoint,
            state,
            tWeight,
            limitWeight,
            wgt,
            loss_,
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
        error += computeLinearJacobian<T, IsL2>(
            limit.data.linear,
            params,
            tWeight,
            limitWeight,
            wgt,
            loss_,
            this->enabledParameters_,
            jacobian,
            residual,
            count);
        count++;
        break;
      case LinearJoint:
        error += computeLinearJointJacobian<T, IsL2>(
            limit.data.linearJoint,
            state,
            tWeight,
            limitWeight,
            wgt,
            loss_,
            this->activeJointParams_,
            parameterTransform,
            jacobian,
            residual,
            count);
        count++;
        break;
      case HalfPlane:
        error += computeHalfPlaneJacobian<T, IsL2>(
            limit.data.halfPlane,
            params,
            tWeight,
            limitWeight,
            wgt,
            loss_,
            this->enabledParameters_,
            jacobian,
            residual,
            count);
        count++;
        break;
      case Ellipsoid:
        error += computeEllipsoidJacobian<T, IsL2>(
            limit.data.ellipsoid,
            params,
            state,
            tWeight,
            limitWeight,
            loss_,
            this->activeJointParams_,
            this->skeleton_,
            parameterTransform,
            jacobian,
            residual,
            count);
        count += 3;
        break;
      case LimitTypeCount:
      default:
        MT_THROW("Unknown parameter type for joint limit");
        break;
    }
  }

  usedRows = count;
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

  return loss_.isL2() ? getJacobianImpl<true>(params, state, jacobian, residual, usedRows)
                      : getJacobianImpl<false>(params, state, jacobian, residual, usedRows);
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
      case LimitTypeCount:
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
