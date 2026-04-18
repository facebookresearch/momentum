/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver_simd/simd_joint_to_joint_distance_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"

#include <dispenso/parallel_for.h>

#include <numeric>
#include <tuple>

namespace momentum {

void SimdJointToJointDistanceConstraints::clearConstraints() {
  joint1.clear();
  joint2.clear();
  offset1.clear();
  offset2.clear();
  targetDistance.clear();
  weight.clear();
}

void SimdJointToJointDistanceConstraints::addConstraint(
    size_t j1,
    const Vector3f& off1,
    size_t j2,
    const Vector3f& off2,
    float target,
    float w) {
  joint1.push_back(static_cast<uint32_t>(j1));
  joint2.push_back(static_cast<uint32_t>(j2));
  offset1.push_back(off1);
  offset2.push_back(off2);
  targetDistance.push_back(target);
  weight.push_back(w);
}

SimdJointToJointDistanceErrorFunction::SimdJointToJointDistanceErrorFunction(
    const Skeleton& skel,
    const ParameterTransform& pt,
    uint32_t maxThreads)
    : SkeletonErrorFunction(skel, pt), constraints_(nullptr), maxThreads_(maxThreads) {}

SimdJointToJointDistanceErrorFunction::SimdJointToJointDistanceErrorFunction(
    const Character& character,
    uint32_t maxThreads)
    : SimdJointToJointDistanceErrorFunction(
          character.skeleton,
          character.parameterTransform,
          maxThreads) {}

namespace {

/// Walk the chain from `startJoint` to root and accumulate the chain-rule gradient w.r.t. the
/// world point at `worldPos`. The `direction` is the unit vector from joint2's point to joint1's
/// point (or its negative when processing joint2's chain), and `wgt` is the scalar pre-multiplier
/// `2 * weight * distanceError`.
inline void walkChainGrad(
    size_t startJoint,
    const Vector3f& worldPos,
    const Vector3f& direction,
    float wgt,
    const SkeletonState& state,
    const Skeleton& skeleton,
    const ParameterTransform& parameterTransform,
    const VectorX<bool>& activeJointParams,
    VectorXd& grad_local) {
  size_t jointIndex = startJoint;
  while (jointIndex != kInvalidIndex) {
    // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
    const auto& jointState = state.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Vector3f posd = worldPos - jointState.translation();

    auto scatter = [&](float val, size_t srcParam) {
      for (auto idx = parameterTransform.transform.outerIndexPtr()[srcParam];
           idx < parameterTransform.transform.outerIndexPtr()[srcParam + 1];
           ++idx) {
        grad_local[parameterTransform.transform.innerIndexPtr()[idx]] +=
            static_cast<double>(val) * parameterTransform.transform.valuePtr()[idx];
      }
    };

    for (size_t d = 0; d < 3; ++d) {
      if (activeJointParams[paramIndex + d]) {
        const float val = direction.dot(jointState.getTranslationDerivative(d)) * wgt;
        scatter(val, paramIndex + d);
      }
      if (activeJointParams[paramIndex + 3 + d]) {
        const float val = direction.dot(jointState.getRotationDerivative(d, posd)) * wgt;
        scatter(val, paramIndex + 3 + d);
      }
    }
    if (activeJointParams[paramIndex + 6]) {
      const float val = direction.dot(jointState.getScaleDerivative(posd)) * wgt;
      scatter(val, paramIndex + 6);
    }

    // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
    jointIndex = skeleton.joints[jointIndex].parent;
  }
}

inline void walkChainJac(
    size_t startJoint,
    const Vector3f& worldPos,
    const Vector3f& direction,
    float wgt,
    int row,
    const SkeletonState& state,
    const Skeleton& skeleton,
    const ParameterTransform& parameterTransform,
    const VectorX<bool>& activeJointParams,
    Ref<MatrixXf> jacobian) {
  size_t jointIndex = startJoint;
  while (jointIndex != kInvalidIndex) {
    // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
    const auto& jointState = state.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Vector3f posd = worldPos - jointState.translation();

    auto scatter = [&](float val, size_t srcParam) {
      for (auto idx = parameterTransform.transform.outerIndexPtr()[srcParam];
           idx < parameterTransform.transform.outerIndexPtr()[srcParam + 1];
           ++idx) {
        jacobian(row, parameterTransform.transform.innerIndexPtr()[idx]) +=
            val * parameterTransform.transform.valuePtr()[idx];
      }
    };

    for (size_t d = 0; d < 3; ++d) {
      if (activeJointParams[paramIndex + d]) {
        const float val = direction.dot(jointState.getTranslationDerivative(d)) * wgt;
        scatter(val, paramIndex + d);
      }
      if (activeJointParams[paramIndex + 3 + d]) {
        const float val = direction.dot(jointState.getRotationDerivative(d, posd)) * wgt;
        scatter(val, paramIndex + 3 + d);
      }
    }
    if (activeJointParams[paramIndex + 6]) {
      const float val = direction.dot(jointState.getScaleDerivative(posd)) * wgt;
      scatter(val, paramIndex + 6);
    }

    // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
    jointIndex = skeleton.joints[jointIndex].parent;
  }
}

} // namespace

double SimdJointToJointDistanceErrorFunction::getError(
    const ModelParameters& /* params */,
    const SkeletonState& state,
    const MeshState& /* unused */) {
  if (constraints_ == nullptr || constraints_->getNumConstraints() == 0 ||
      state.jointState.empty()) {
    return 0.0f;
  }

  double error = 0.0;
  const size_t n = constraints_->getNumConstraints();
  for (size_t i = 0; i < n; ++i) {
    const auto& js1 = state.jointState[constraints_->joint1[i]];
    const auto& js2 = state.jointState[constraints_->joint2[i]];
    const Vector3f p1 = js1.transform * constraints_->offset1[i];
    const Vector3f p2 = js2.transform * constraints_->offset2[i];
    const float dist = (p1 - p2).norm();
    const float distErr = dist - constraints_->targetDistance[i];
    error += constraints_->weight[i] * distErr * distErr;
  }

  return static_cast<float>(error * kDistanceWeight * weight_);
}

double SimdJointToJointDistanceErrorFunction::getGradient(
    const ModelParameters& /* params */,
    const SkeletonState& state,
    const MeshState& /* unused */,
    Ref<VectorXf> gradient) {
  if (constraints_ == nullptr || constraints_->getNumConstraints() == 0 ||
      state.jointState.empty()) {
    return 0.0f;
  }

  std::vector<std::tuple<double, VectorXd>> ets_error_grad;

  auto dispensoOptions = dispenso::ParForOptions();
  dispensoOptions.maxThreads = maxThreads_;
  dispenso::parallel_for(
      ets_error_grad,
      [&]() -> std::tuple<double, VectorXd> {
        return {0.0, VectorXd::Zero(parameterTransform_.numAllModelParameters())};
      },
      0,
      constraints_->getNumConstraints(),
      [&](std::tuple<double, VectorXd>& acc, const size_t i) {
        double& error_local = std::get<0>(acc);
        VectorXd& grad_local = std::get<1>(acc);

        const auto& js1 = state.jointState[constraints_->joint1[i]];
        const auto& js2 = state.jointState[constraints_->joint2[i]];
        const Vector3f p1 = js1.transform * constraints_->offset1[i];
        const Vector3f p2 = js2.transform * constraints_->offset2[i];
        const Vector3f diff = p1 - p2;
        const float dist = diff.norm();
        if (dist < kDistanceEps) {
          return;
        }

        const float distErr = dist - constraints_->targetDistance[i];
        error_local += constraints_->weight[i] * distErr * distErr;

        const Vector3f direction = diff / dist;
        const float wgt = 2.0f * constraints_->weight[i] * distErr * kDistanceWeight;

        // joint1 chain gets +direction; joint2 chain gets -direction (since p1 - p2).
        walkChainGrad(
            constraints_->joint1[i],
            p1,
            direction,
            wgt,
            state,
            this->skeleton_,
            this->parameterTransform_,
            this->activeJointParams_,
            grad_local);
        walkChainGrad(
            constraints_->joint2[i],
            p2,
            -direction,
            wgt,
            state,
            this->skeleton_,
            this->parameterTransform_,
            this->activeJointParams_,
            grad_local);
      },
      dispensoOptions);

  double error = 0.0;
  if (!ets_error_grad.empty()) {
    ets_error_grad[0] = std::accumulate(
        ets_error_grad.begin() + 1,
        ets_error_grad.end(),
        ets_error_grad[0],
        [](const auto& a, const auto& b) -> std::tuple<double, VectorXd> {
          return {std::get<0>(a) + std::get<0>(b), std::get<1>(a) + std::get<1>(b)};
        });
    gradient += std::get<1>(ets_error_grad[0]).cast<float>() * weight_;
    error = std::get<0>(ets_error_grad[0]);
  }

  return static_cast<float>(kDistanceWeight * weight_ * error);
}

double SimdJointToJointDistanceErrorFunction::getJacobian(
    const ModelParameters& /* params */,
    const SkeletonState& state,
    const MeshState& /* unused */,
    Ref<MatrixXf> jacobian,
    Ref<VectorXf> residual,
    int& usedRows) {
  MT_PROFILE_FUNCTION();

  if (constraints_ == nullptr || constraints_->getNumConstraints() == 0 ||
      state.jointState.empty()) {
    usedRows = 0;
    return 0.0f;
  }

  std::vector<double> errors(constraints_->getNumConstraints(), 0.0);

  auto dispensoOptions = dispenso::ParForOptions();
  dispensoOptions.maxThreads = maxThreads_;
  dispenso::parallel_for(
      0,
      constraints_->getNumConstraints(),
      [&](const size_t i) {
        const auto& js1 = state.jointState[constraints_->joint1[i]];
        const auto& js2 = state.jointState[constraints_->joint2[i]];
        const Vector3f p1 = js1.transform * constraints_->offset1[i];
        const Vector3f p2 = js2.transform * constraints_->offset2[i];
        const Vector3f diff = p1 - p2;
        const float dist = diff.norm();
        if (dist < kDistanceEps) {
          residual(i) = 0.0f;
          return;
        }

        const float distErr = dist - constraints_->targetDistance[i];
        errors[i] = constraints_->weight[i] * distErr * distErr;

        const Vector3f direction = diff / dist;
        const float wgt = std::sqrt(constraints_->weight[i] * kDistanceWeight * weight_);
        residual(i) = wgt * distErr;

        walkChainJac(
            constraints_->joint1[i],
            p1,
            direction,
            wgt,
            static_cast<int>(i),
            state,
            this->skeleton_,
            this->parameterTransform_,
            this->activeJointParams_,
            jacobian);
        walkChainJac(
            constraints_->joint2[i],
            p2,
            -direction,
            wgt,
            static_cast<int>(i),
            state,
            this->skeleton_,
            this->parameterTransform_,
            this->activeJointParams_,
            jacobian);
      },
      dispensoOptions);

  usedRows = static_cast<int>(constraints_->getNumConstraints());
  const double totalError =
      std::accumulate(errors.begin(), errors.end(), 0.0) * kDistanceWeight * weight_;
  return static_cast<float>(totalError);
}

size_t SimdJointToJointDistanceErrorFunction::getJacobianSize() const {
  if (constraints_ == nullptr) {
    return 0;
  }
  return constraints_->getNumConstraints();
}

} // namespace momentum
