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
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"

namespace momentum {

constexpr float kPositionWeight = 1e-4f;

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
    const SkeletonStateT<T>& state) {
  MT_PROFILE_FUNCTION();

  // check all is valid
  MT_CHECK(
      state.jointParameters.size() ==
      gsl::narrow<Eigen::Index>(this->skeleton_.joints.size() * kParametersPerJoint));

  // loop over all joints and check for limit violations
  double error = 0.0;

  // ---------------------------------------
  //  new joint error
  // ---------------------------------------
  for (const auto& limit : limits_) {
    switch (limit.type) {
      case MinMax: {
        const auto& data = limit.data.minMax;
        MT_CHECK(data.parameterIndex <= static_cast<size_t>(params.size()));
        if (this->enabledParameters_.test(data.parameterIndex)) {
          if (params(data.parameterIndex) < data.limits[0]) {
            const T val = data.limits[0] - params(data.parameterIndex);
            error += val * val * limit.weight;
          }
          if (params(data.parameterIndex) > data.limits[1]) {
            const T val = data.limits[1] - params(data.parameterIndex);
            error += val * val * limit.weight;
          }
        }
        break;
      }
      case MinMaxJoint: {
        const auto& data = limit.data.minMaxJoint;
        const size_t parameterIndex = data.jointIndex * kParametersPerJoint + data.jointParameter;
        MT_CHECK(parameterIndex < (size_t)state.jointParameters.size());
        if (this->activeJointParams_[parameterIndex]) {
          if (state.jointParameters(parameterIndex) < data.limits[0]) {
            const T val = data.limits[0] - state.jointParameters[parameterIndex];
            error += val * val * limit.weight;
          }
          if (state.jointParameters(parameterIndex) > data.limits[1]) {
            const T val = data.limits[1] - state.jointParameters[parameterIndex];
            error += val * val * limit.weight;
          }
        }
        break;
      }
      case MinMaxJointPassive: {
        break;
      }
      case Linear: {
        const auto& data = limit.data.linear;
        MT_CHECK(data.referenceIndex < static_cast<size_t>(params.size()));
        MT_CHECK(data.targetIndex < static_cast<size_t>(params.size()));

        if ((this->enabledParameters_.test(data.targetIndex) ||
             this->enabledParameters_.test(data.referenceIndex)) &&
            isInRange(data, params(data.targetIndex))) {
          const T residual =
              params(data.targetIndex) * data.scale - data.offset - params(data.referenceIndex);
          error += residual * residual * limit.weight;
        }
        break;
      }
      case LinearJoint: {
        const auto& data = limit.data.linearJoint;
        const size_t referenceParameterIndex =
            data.referenceJointIndex * kParametersPerJoint + data.referenceJointParameter;
        const size_t targetParameterIndex =
            data.targetJointIndex * kParametersPerJoint + data.targetJointParameter;
        MT_CHECK(referenceParameterIndex < (size_t)state.jointParameters.size());
        MT_CHECK(targetParameterIndex < (size_t)state.jointParameters.size());

        if ((this->activeJointParams_[referenceParameterIndex] ||
             this->activeJointParams_[targetParameterIndex]) &&
            isInRange(data, state.jointParameters[targetParameterIndex])) {
          const T residual = state.jointParameters(targetParameterIndex) * data.scale -
              data.offset - state.jointParameters(referenceParameterIndex);
          error += residual * residual * limit.weight;
        }
        break;
      }

      case HalfPlane: {
        const auto& data = limit.data.halfPlane;
        MT_CHECK(data.param1 < static_cast<size_t>(params.size()));
        MT_CHECK(data.param2 < static_cast<size_t>(params.size()));

        if ((this->enabledParameters_.test(data.param1) ||
             this->enabledParameters_.test(data.param2))) {
          const Eigen::Vector2<T> p(params(data.param1), params(data.param2));
          const T residual = p.dot(data.normal.template cast<T>()) - data.offset;
          if (residual < 0) {
            error += residual * residual * limit.weight;
          }
        }
        break;
      }
      case Ellipsoid: {
        const auto& ct = limit.data.ellipsoid;

        // get the constraint position in global space
        const Eigen::Vector3<T> position =
            state.jointState[ct.parent].transformation * ct.offset.template cast<T>();

        // get the constraint position in local ellipsoid space
        const Eigen::Vector3<T> localPosition =
            state.jointState[ct.ellipsoidParent].transformation.inverse() * position;

        // calculate constraint position in ellipsoid space
        const Eigen::Vector3<T> ellipsoidPosition =
            ct.ellipsoidInv.template cast<T>() * localPosition;

        // project onto closest surface point
        const Eigen::Vector3<T> normalizedPosition = ellipsoidPosition.normalized();

        // go back to ellipsoid frame
        const Eigen::Vector3<T> projectedPosition =
            ct.ellipsoid.template cast<T>() * normalizedPosition;

        // calculate the difference between projected position and actual position
        const Eigen::Vector3<T> diff =
            position - state.jointState[ct.ellipsoidParent].transformation * projectedPosition;

        error += diff.squaredNorm() * kPositionWeight * limit.weight;
        break;
      }
      default:
        // should never get here
        MT_THROW("Unknown parameter type for joint limit");
        break;
    }
  }

  // return error
  return error * kLimitWeight * this->weight_;
}

template <typename T>
double LimitErrorFunctionT<T>::getGradient(
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& state,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  MT_PROFILE_FUNCTION();

  const auto& parameterTransform = this->parameterTransform_;

  // loop over all joints and check for limit violations
  double error = 0.0;

  const T tWeight = kLimitWeight * this->weight_;

  // ---------------------------------------
  //  new joint error
  // ---------------------------------------
  for (const auto& limit : limits_) {
    switch (limit.type) {
      case MinMax: {
        const auto& data = limit.data.minMax;
        MT_CHECK(data.parameterIndex < static_cast<size_t>(params.size()));
        if (this->enabledParameters_.test(data.parameterIndex)) {
          if (params(data.parameterIndex) < data.limits[0]) {
            const T val = params(data.parameterIndex) - data.limits[0];
            error += val * val * tWeight * limit.weight;
            gradient[data.parameterIndex] += T(2) * val * tWeight * limit.weight;
          }
          if (params(data.parameterIndex) > data.limits[1]) {
            const T val = params(data.parameterIndex) - data.limits[1];
            error += val * val * tWeight * limit.weight;
            gradient[data.parameterIndex] += T(2) * val * tWeight * limit.weight;
          }
        }
        break;
      }
      case MinMaxJoint: {
        const auto& data = limit.data.minMaxJoint;
        const size_t parameterIndex = data.jointIndex * kParametersPerJoint + data.jointParameter;
        MT_CHECK(parameterIndex < (size_t)state.jointParameters.size());
        if (this->activeJointParams_[parameterIndex]) {
          if (state.jointParameters(parameterIndex) < data.limits[0]) {
            const T val = state.jointParameters[parameterIndex] - data.limits[0];
            error += val * val * tWeight * limit.weight;
            // explicitly multiply joint gradient with the parameter transform to generate parameter
            // space gradients
            const T jGrad = T(2) * val * tWeight * limit.weight;
            for (auto index = parameterTransform.transform.outerIndexPtr()[parameterIndex];
                 index < parameterTransform.transform.outerIndexPtr()[parameterIndex + 1];
                 ++index) {
              gradient[parameterTransform.transform.innerIndexPtr()[index]] +=
                  jGrad * parameterTransform.transform.valuePtr()[index];
            }
          }
          if (state.jointParameters(parameterIndex) > data.limits[1]) {
            const T val = state.jointParameters[parameterIndex] - data.limits[1];
            error += val * val * tWeight * limit.weight;
            // explicitly multiply joint gradient with the parameter transform to generate parameter
            // space gradients
            const T jGrad = T(2) * val * tWeight * limit.weight;
            for (auto index = parameterTransform.transform.outerIndexPtr()[parameterIndex];
                 index < parameterTransform.transform.outerIndexPtr()[parameterIndex + 1];
                 ++index) {
              gradient[parameterTransform.transform.innerIndexPtr()[index]] +=
                  jGrad * parameterTransform.transform.valuePtr()[index];
            }
          }
        }
        break;
      }
      case MinMaxJointPassive: {
        break;
      }
      case Linear: {
        const auto& data = limit.data.linear;
        MT_CHECK(data.referenceIndex < static_cast<size_t>(params.size()));
        MT_CHECK(data.targetIndex < static_cast<size_t>(params.size()));
        if (isInRange(data, params(data.targetIndex))) {
          const T residual =
              params(data.targetIndex) * data.scale - data.offset - params(data.referenceIndex);
          error += residual * residual * limit.weight * tWeight;

          if (this->enabledParameters_.test(data.targetIndex)) {
            gradient[data.targetIndex] += T(2) * residual * data.scale * limit.weight * tWeight;
          }
          if (this->enabledParameters_.test(data.referenceIndex)) {
            gradient[data.referenceIndex] -= T(2) * residual * limit.weight * tWeight;
          }
        }
        break;
      }
      case LinearJoint: {
        const auto& data = limit.data.linearJoint;
        const size_t referenceParameterIndex =
            data.referenceJointIndex * kParametersPerJoint + data.referenceJointParameter;
        const size_t targetParameterIndex =
            data.targetJointIndex * kParametersPerJoint + data.targetJointParameter;
        MT_CHECK(referenceParameterIndex < (size_t)state.jointParameters.size());
        MT_CHECK(targetParameterIndex < (size_t)state.jointParameters.size());

        if ((this->activeJointParams_[referenceParameterIndex] ||
             this->activeJointParams_[targetParameterIndex]) &&
            isInRange(data, state.jointParameters[targetParameterIndex])) {
          const T residual = state.jointParameters(targetParameterIndex) * data.scale -
              data.offset - state.jointParameters(referenceParameterIndex);
          error += residual * residual * limit.weight * tWeight;

          // explicitly multiply joint gradient with the parameter transform to generate parameter
          // space gradients
          const T targetGrad = T(2) * residual * data.scale * limit.weight * tWeight;
          for (auto index = parameterTransform.transform.outerIndexPtr()[targetParameterIndex];
               index < parameterTransform.transform.outerIndexPtr()[targetParameterIndex + 1];
               ++index) {
            gradient[parameterTransform.transform.innerIndexPtr()[index]] +=
                targetGrad * parameterTransform.transform.valuePtr()[index];
          }

          const T referenceGrad = T(-2) * residual * limit.weight * tWeight;
          for (auto index = parameterTransform.transform.outerIndexPtr()[referenceParameterIndex];
               index < parameterTransform.transform.outerIndexPtr()[referenceParameterIndex + 1];
               ++index) {
            gradient[parameterTransform.transform.innerIndexPtr()[index]] +=
                referenceGrad * parameterTransform.transform.valuePtr()[index];
          }
        }
        break;
      }
      case HalfPlane: {
        const auto& data = limit.data.halfPlane;
        if ((this->enabledParameters_.test(data.param1) ||
             this->enabledParameters_.test(data.param2))) {
          const Eigen::Vector2<T> p(params(data.param1), params(data.param2));
          const T residual = p.dot(data.normal.template cast<T>()) - data.offset;
          if (residual < 0.0f) {
            error += residual * residual * limit.weight * tWeight;

            if (this->enabledParameters_.test(data.param1)) {
              gradient[data.param1] += T(2) * residual * data.normal[0] * tWeight;
            }
            if (this->enabledParameters_.test(data.param2)) {
              gradient[data.param2] += T(2) * residual * data.normal[1] * tWeight;
            }
          }
        }
        break;
      }
      case Ellipsoid: {
        // NOTE: The gradient for these is currently simplified
        // It assumes the ellipsoid is static and doesn't move with the parent joint
        const auto& ct = limit.data.ellipsoid;

        // get the constraint position in global space
        const Eigen::Vector3<T> position =
            state.jointState[ct.parent].transformation * ct.offset.template cast<T>();

        // get the constraint position in local ellipsoid space
        const Eigen::Vector3<T> localPosition =
            state.jointState[ct.ellipsoidParent].transformation.inverse() * position;

        // calculate constraint position in ellipsoid space
        const Eigen::Vector3<T> ellipsoidPosition =
            ct.ellipsoidInv.template cast<T>() * localPosition;

        // project onto closest surface point
        const Eigen::Vector3<T> normalizedPosition = ellipsoidPosition.normalized();

        // go back to ellipsoid frame
        const Eigen::Vector3<T> projectedPosition =
            ct.ellipsoid.template cast<T>() * normalizedPosition;

        // calculate the difference between projected position and actual position
        const Eigen::Vector3<T> diff =
            position - state.jointState[ct.ellipsoidParent].transformation * projectedPosition;
        const T wgt = T(2) * kPositionWeight * limit.weight * tWeight;

        // loop over all joints the constraint is attached to and calculate gradient
        size_t jointIndex = ct.parent;
        while (jointIndex != ct.ellipsoidParent && jointIndex != kInvalidIndex) {
          // check for valid index
          MT_CHECK(jointIndex < this->skeleton_.joints.size());

          const auto& jointState = state.jointState[jointIndex];
          const size_t paramIndex = jointIndex * kParametersPerJoint;
          const Eigen::Vector3<T> posd = position - jointState.translation();

          // calculate derivatives based on active joints
          for (size_t d = 0; d < 3; d++) {
            if (this->activeJointParams_[paramIndex + d]) {
              // calculate joint gradient
              const T val = diff.dot(jointState.getTranslationDerivative(d)) * wgt;
              // explicitly multiply with the parameter transform to generate parameter space
              // gradients
              for (auto index = parameterTransform.transform.outerIndexPtr()[paramIndex + d];
                   index < parameterTransform.transform.outerIndexPtr()[paramIndex + d + 1];
                   ++index) {
                gradient[parameterTransform.transform.innerIndexPtr()[index]] +=
                    val * parameterTransform.transform.valuePtr()[index];
              }
            }
            if (this->activeJointParams_[paramIndex + 3 + d]) {
              // calculate joint gradient
              const T val = diff.dot(jointState.getRotationDerivative(d, posd)) * wgt;
              // explicitly multiply with the parameter transform to generate parameter space
              // gradients
              for (auto index = parameterTransform.transform.outerIndexPtr()[paramIndex + 3 + d];
                   index < parameterTransform.transform.outerIndexPtr()[paramIndex + d + 3 + 1];
                   ++index) {
                gradient[parameterTransform.transform.innerIndexPtr()[index]] +=
                    val * parameterTransform.transform.valuePtr()[index];
              }
            }
          }
          if (this->activeJointParams_[paramIndex + 6]) {
            // calculate joint gradient
            const T val = diff.dot(jointState.getScaleDerivative(posd)) * wgt;
            // explicitly multiply with the parameter transform to generate parameter space
            // gradients
            for (auto index = parameterTransform.transform.outerIndexPtr()[paramIndex + 6];
                 index < parameterTransform.transform.outerIndexPtr()[paramIndex + 6 + 1];
                 ++index) {
              gradient[parameterTransform.transform.innerIndexPtr()[index]] +=
                  val * parameterTransform.transform.valuePtr()[index];
            }
          }

          // go to the next joint
          jointIndex = this->skeleton_.joints[jointIndex].parent;
        }

        error += diff.squaredNorm() * kPositionWeight * tWeight * limit.weight;
        break;
      }
      default:
        // should never get here
        MT_THROW("Unknown parameter type for joint limit");
        break;
    }
  }

  // return error
  return error;
}

template <typename T>
double LimitErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& state,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    int& usedRows) {
  MT_PROFILE_FUNCTION();

  const auto& parameterTransform = this->parameterTransform_;

  // loop over all joints and check for limit violations
  double error = 0.0;

  const T tWeight = kLimitWeight * this->weight_;

  jacobian.setZero();
  residual.setZero();

  int count = 0;
  // ---------------------------------------
  //  new joint error
  // ---------------------------------------
  for (const auto& limit : limits_) {
    const T wgt = std::sqrt(kLimitWeight * this->weight_ * limit.weight);
    switch (limit.type) {
      case MinMax: {
        const auto& data = limit.data.minMax;
        MT_CHECK(data.parameterIndex < static_cast<size_t>(params.size()));
        if (this->enabledParameters_.test(data.parameterIndex)) {
          if (params(data.parameterIndex) < data.limits[0]) {
            const T val = params(data.parameterIndex) - data.limits[0];
            error += val * val * limit.weight * tWeight;
            jacobian(count, data.parameterIndex) = wgt;
            residual(count) = val * wgt;
          } else if (params(data.parameterIndex) > data.limits[1]) {
            const T val = params(data.parameterIndex) - data.limits[1];
            error += val * val * limit.weight * tWeight;
            jacobian(count, data.parameterIndex) = wgt;
            residual(count) = val * wgt;
          }
        }
        count++;
        break;
      }
      case MinMaxJoint: {
        // simple case, our jacobians are currently in joint space, just add them up
        const auto& data = limit.data.minMaxJoint;
        const size_t jointIndex = data.jointIndex * kParametersPerJoint + data.jointParameter;
        MT_CHECK(jointIndex < (size_t)state.jointParameters.size());
        if (this->activeJointParams_[jointIndex]) {
          if (state.jointParameters(jointIndex) < data.limits[0]) {
            const T val = state.jointParameters[jointIndex] - data.limits[0];
            error += val * val * limit.weight * tWeight;
            // explicitly multiply with the parameter transform to generate parameter space
            // gradients
            for (auto index = parameterTransform.transform.outerIndexPtr()[jointIndex];
                 index < parameterTransform.transform.outerIndexPtr()[jointIndex + 1];
                 ++index)
              jacobian(count, parameterTransform.transform.innerIndexPtr()[index]) +=
                  wgt * parameterTransform.transform.valuePtr()[index];
            residual(count) = val * wgt;
          } else if (state.jointParameters(jointIndex) > data.limits[1]) {
            const T val = state.jointParameters[jointIndex] - data.limits[1];
            error += val * val * limit.weight * tWeight;
            // explicitly multiply with the parameter transform to generate parameter space
            // gradients
            for (auto index = parameterTransform.transform.outerIndexPtr()[jointIndex];
                 index < parameterTransform.transform.outerIndexPtr()[jointIndex + 1];
                 ++index)
              jacobian(count, parameterTransform.transform.innerIndexPtr()[index]) +=
                  wgt * parameterTransform.transform.valuePtr()[index];
            residual(count) = val * wgt;
          }
        }
        count++;
        break;
      }
      case MinMaxJointPassive: {
        break;
      }
      case Linear: {
        const auto& data = limit.data.linear;
        MT_CHECK(data.referenceIndex < static_cast<size_t>(params.size()));
        MT_CHECK(data.targetIndex < static_cast<size_t>(params.size()));

        if ((this->enabledParameters_.test(data.targetIndex) ||
             this->enabledParameters_.test(data.referenceIndex)) &&
            isInRange(data, params(data.targetIndex))) {
          const T res =
              params(data.targetIndex) * data.scale - data.offset - params(data.referenceIndex);
          error += res * res * limit.weight * tWeight;
          residual(count) = res * wgt;

          if (this->enabledParameters_.test(data.targetIndex)) {
            jacobian(count, data.targetIndex) = data.scale * wgt;
          }

          if (this->enabledParameters_.test(data.referenceIndex)) {
            jacobian(count, data.referenceIndex) = -wgt;
          }
        }
        count++;
        break;
      }
      case LinearJoint: {
        const auto& data = limit.data.linearJoint;
        const size_t referenceParameterIndex =
            data.referenceJointIndex * kParametersPerJoint + data.referenceJointParameter;
        const size_t targetParameterIndex =
            data.targetJointIndex * kParametersPerJoint + data.targetJointParameter;
        MT_CHECK(referenceParameterIndex < (size_t)state.jointParameters.size());
        MT_CHECK(targetParameterIndex < (size_t)state.jointParameters.size());
        if ((this->activeJointParams_[referenceParameterIndex] ||
             this->activeJointParams_[targetParameterIndex]) &&
            isInRange(data, state.jointParameters(targetParameterIndex))) {
          const T res = state.jointParameters(targetParameterIndex) * data.scale - data.offset -
              state.jointParameters(referenceParameterIndex);
          error += res * res * limit.weight * tWeight;

          residual(count) = res * wgt;

          // explicitly multiply with the parameter transform to generate parameter space
          // gradients
          if (this->activeJointParams_[targetParameterIndex]) {
            for (auto index = parameterTransform.transform.outerIndexPtr()[targetParameterIndex];
                 index < parameterTransform.transform.outerIndexPtr()[targetParameterIndex + 1];
                 ++index) {
              jacobian(count, parameterTransform.transform.innerIndexPtr()[index]) +=
                  data.scale * wgt * parameterTransform.transform.valuePtr()[index];
            }
          }

          if (this->activeJointParams_[referenceParameterIndex]) {
            for (auto index = parameterTransform.transform.outerIndexPtr()[referenceParameterIndex];
                 index < parameterTransform.transform.outerIndexPtr()[referenceParameterIndex + 1];
                 ++index) {
              jacobian(count, parameterTransform.transform.innerIndexPtr()[index]) -=
                  wgt * parameterTransform.transform.valuePtr()[index];
            }
          }
        }
        count++;
        break;
      }
      case HalfPlane: {
        const auto& data = limit.data.halfPlane;
        MT_CHECK(data.param1 < static_cast<size_t>(params.size()));
        MT_CHECK(data.param2 < static_cast<size_t>(params.size()));

        if ((this->enabledParameters_.test(data.param1) ||
             this->enabledParameters_.test(data.param2))) {
          const Eigen::Vector2<T> p(params(data.param1), params(data.param2));
          const T res = p.dot(data.normal.template cast<T>()) - data.offset;
          if (res < 0) {
            error += res * res * limit.weight * tWeight;
            residual(count) = res * wgt;

            if (this->enabledParameters_.test(data.param1)) {
              jacobian(count, data.param1) = data.normal[0] * wgt;
            }

            if (this->enabledParameters_.test(data.param2)) {
              jacobian(count, data.param2) = data.normal[1] * wgt;
            }
          }
        }
        count++;
        break;
      }
      case Ellipsoid: {
        // NOTE: The jacobian for these is currently simplified
        // It assumes the ellipsoid is static and doesn't move with the parent joint
        const auto& ct = limit.data.ellipsoid;

        // get the constraint position in global space
        const Eigen::Vector3<T> position =
            state.jointState[ct.parent].transformation * ct.offset.template cast<T>();

        // get the constraint position in local ellipsoid space
        const Eigen::Vector3<T> localPosition =
            state.jointState[ct.ellipsoidParent].transformation.inverse() * position;

        // calculate constraint position in ellipsoid space
        const Eigen::Vector3<T> ellipsoidPosition =
            ct.ellipsoidInv.template cast<T>() * localPosition;

        // project onto closest surface point
        const Eigen::Vector3<T> normalizedPosition = ellipsoidPosition.normalized();

        // go back to ellipsoid frame
        const Eigen::Vector3<T> projectedPosition =
            ct.ellipsoid.template cast<T>() * normalizedPosition;

        // calculate the difference between projected position and actual position
        const Eigen::Vector3<T> diff =
            position - state.jointState[ct.ellipsoidParent].transformation * projectedPosition;

        // calculate offset in jacobian
        Eigen::Ref<Eigen::MatrixX<T>> jac = jacobian.block(count, 0, 3, params.size());
        Eigen::Ref<Eigen::VectorX<T>> res = residual.middleRows(count, 3);
        MT_CHECK(jac.cols() == static_cast<Eigen::Index>(parameterTransform.transform.cols()));
        MT_CHECK(jac.rows() == 3);

        // calculate the difference between target and position and error
        const T jwgt = std::sqrt(kLimitWeight * kPositionWeight * this->weight_ * limit.weight);

        // loop over all joints the constraint is attached to and calculate gradient
        size_t jointIndex = ct.parent;
        while (jointIndex != ct.ellipsoidParent && jointIndex != kInvalidIndex) {
          // check for valid index
          MT_CHECK(jointIndex < this->skeleton_.joints.size());

          const auto& jointState = state.jointState[jointIndex];
          const size_t paramIndex = jointIndex * kParametersPerJoint;
          const Eigen::Vector3<T> posd = position - jointState.translation();

          // calculate derivatives based on active joints
          for (size_t d = 0; d < 3; d++) {
            if (this->activeJointParams_[paramIndex + d]) {
              const Eigen::Vector3<T> jc = jointState.getTranslationDerivative(d) * jwgt;
              for (auto index = parameterTransform.transform.outerIndexPtr()[paramIndex + d];
                   index < parameterTransform.transform.outerIndexPtr()[paramIndex + d + 1];
                   ++index)
                jac.col(parameterTransform.transform.innerIndexPtr()[index]) +=
                    jc * parameterTransform.transform.valuePtr()[index];
            }
            if (this->activeJointParams_[paramIndex + 3 + d]) {
              const Eigen::Vector3<T> jc = jointState.getRotationDerivative(d, posd) * jwgt;
              for (auto index = parameterTransform.transform.outerIndexPtr()[paramIndex + d + 3];
                   index < parameterTransform.transform.outerIndexPtr()[paramIndex + d + 3 + 1];
                   ++index)
                jac.col(parameterTransform.transform.innerIndexPtr()[index]) +=
                    jc * parameterTransform.transform.valuePtr()[index];
            }
          }
          if (this->activeJointParams_[paramIndex + 6]) {
            const Eigen::Vector3<T> jc = jointState.getScaleDerivative(posd) * jwgt;
            for (auto index = parameterTransform.transform.outerIndexPtr()[paramIndex + 6];
                 index < parameterTransform.transform.outerIndexPtr()[paramIndex + 6 + 1];
                 ++index)
              jac.col(parameterTransform.transform.innerIndexPtr()[index]) +=
                  jc * parameterTransform.transform.valuePtr()[index];
          }

          // go to the next joint
          jointIndex = this->skeleton_.joints[jointIndex].parent;
        }

        res = diff * jwgt;
        error += jwgt * jwgt * diff.squaredNorm();
        count += 3;

        break;
      }
      default:
        // should never get here
        MT_THROW("Unknown parameter type for joint limit");
        break;
    }
  }

  usedRows = count;

  // return error
  return error;
}

template <typename T>
size_t LimitErrorFunctionT<T>::getJacobianSize() const {
  size_t count = 0;
  for (const auto& limit : limits_) {
    switch (limit.type) {
      case MinMax:
      case MinMaxJoint:
        count++;
        break;
      case MinMaxJointPassive:
        break;
      case Linear:
        count++;
        break;
      case LinearJoint:
        count++;
        break;
      case HalfPlane:
        count++;
        break;
      case Ellipsoid:
        count += 3;
        break;
      default:
        // should never get here
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
