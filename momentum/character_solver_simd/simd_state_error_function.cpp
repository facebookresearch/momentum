/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver_simd/simd_state_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_solver/error_function_utils.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/math/utility.h"

#include <dispenso/parallel_for.h>

#include <numeric>
#include <tuple>

namespace momentum {

SimdStateErrorFunction::SimdStateErrorFunction(
    const Skeleton& skel,
    const ParameterTransform& pt,
    uint32_t maxThreads)
    : SkeletonErrorFunction(skel, pt), maxThreads_(maxThreads) {
  targetPositionWeights_ = Eigen::VectorXf::Ones(skel.joints.size());
  targetRotationWeights_ = Eigen::VectorXf::Ones(skel.joints.size());
}

SimdStateErrorFunction::SimdStateErrorFunction(const Character& character, uint32_t maxThreads)
    : SimdStateErrorFunction(character.skeleton, character.parameterTransform, maxThreads) {}

void SimdStateErrorFunction::setTargetState(const SkeletonState& target) {
  MT_CHECK(target.jointState.size() == this->skeleton_.joints.size());
  targetState_ = target.toTransforms();
}

void SimdStateErrorFunction::setTargetWeights(
    const Eigen::VectorXf& posWeight,
    const Eigen::VectorXf& rotWeight) {
  MT_CHECK(posWeight.size() == static_cast<Eigen::Index>(this->skeleton_.joints.size()));
  MT_CHECK(rotWeight.size() == static_cast<Eigen::Index>(this->skeleton_.joints.size()));
  targetPositionWeights_ = posWeight;
  targetRotationWeights_ = rotWeight;
}

namespace {

inline Eigen::Matrix<float, 9, 1>
matrixDiffJacobianCol(const Eigen::Quaternionf& rot, const Eigen::Vector3f& axis, float weight) {
  const Eigen::Matrix3f rotD = crossProductMatrix(axis) * rot * weight;
  return Eigen::Map<const Eigen::Matrix<float, 9, 1>>(rotD.data(), rotD.size());
}

inline float matrixDiffGradientCol(
    const Eigen::Quaternionf& rot,
    const Eigen::Quaternionf& target,
    const Eigen::Vector3f& axis) {
  const Eigen::Matrix3f rotD = crossProductMatrix(axis) * rot;
  const Eigen::Matrix3f rotDiff = rot.toRotationMatrix() - target.toRotationMatrix();
  return 2.0f * rotD.cwiseProduct(rotDiff).sum();
}

} // namespace

double SimdStateErrorFunction::getError(
    const ModelParameters& /* params */,
    const SkeletonState& state,
    const MeshState& /* unused */) {
  if (state.jointState.empty() || targetState_.size() != state.jointState.size()) {
    return 0.0f;
  }

  double error = 0.0;
  for (size_t i = 0; i < this->skeleton_.joints.size(); ++i) {
    const Eigen::Quaternionf& target = targetState_[i].rotation.normalized();
    const Eigen::Quaternionf& rot = state.jointState[i].rotation();
    const Eigen::Matrix3f rotDiff = rot.toRotationMatrix() - target.toRotationMatrix();
    const float rotationError = rotDiff.squaredNorm();

    error += rotationError * kOrientationWeight * rotWgt_ * targetRotationWeights_[i];

    const Eigen::Vector3f diff = state.jointState[i].translation() - targetState_[i].translation;
    error += diff.squaredNorm() * kPositionWeight * posWgt_ * targetPositionWeights_[i];
  }

  return error * weight_;
}

double SimdStateErrorFunction::getGradient(
    const ModelParameters& /* params */,
    const SkeletonState& state,
    const MeshState& /* unused */,
    Ref<VectorXf> gradient) {
  if (state.jointState.empty() || targetState_.size() != state.jointState.size()) {
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
      this->skeleton_.joints.size(),
      [&](std::tuple<double, VectorXd>& acc, const size_t i) {
        if (targetRotationWeights_[i] == 0 && targetPositionWeights_[i] == 0) {
          return;
        }
        double& error_local = std::get<0>(acc);
        VectorXd& grad_local = std::get<1>(acc);

        const Eigen::Quaternionf& target = targetState_[i].rotation.normalized();
        const Eigen::Quaternionf& rot = state.jointState[i].rotation();

        const Eigen::Matrix3f rotDiff = rot.toRotationMatrix() - target.toRotationMatrix();
        const float rotationError = rotDiff.squaredNorm();
        const float rwgt = kOrientationWeight * rotWgt_ * this->weight_ * targetRotationWeights_[i];
        error_local += rotationError * rwgt;

        const Eigen::Vector3f diff =
            state.jointState[i].translation() - targetState_[i].translation;
        const float pwgt = kPositionWeight * posWgt_ * this->weight_ * targetPositionWeights_[i];
        error_local += diff.squaredNorm() * pwgt;

        size_t jointIndex = i;
        while (jointIndex != kInvalidIndex) {
          const size_t paramIndex = jointIndex * kParametersPerJoint;
          const auto& jointState_j = state.jointState[jointIndex];
          const Eigen::Vector3f posd =
              state.jointState[i].translation() - jointState_j.translation();

          auto scatter = [&](float val, size_t srcParam) {
            for (auto idx = parameterTransform_.transform.outerIndexPtr()[srcParam];
                 idx < parameterTransform_.transform.outerIndexPtr()[srcParam + 1];
                 ++idx) {
              grad_local[parameterTransform_.transform.innerIndexPtr()[idx]] +=
                  static_cast<double>(val) * parameterTransform_.transform.valuePtr()[idx];
            }
          };

          for (size_t d = 0; d < 3; ++d) {
            if (this->activeJointParams_[paramIndex + d]) {
              const float val = 2.0f * diff.dot(jointState_j.getTranslationDerivative(d)) * pwgt;
              scatter(val, paramIndex + d);
            }
            if (this->activeJointParams_[paramIndex + 3 + d]) {
              const Eigen::Vector3f axis = jointState_j.rotationAxis.col(d);
              const float rotPart = matrixDiffGradientCol(rot, target, axis);
              const float val =
                  2.0f * diff.dot(jointState_j.getRotationDerivative(d, posd)) * pwgt +
                  rwgt * rotPart;
              scatter(val, paramIndex + 3 + d);
            }
          }
          if (this->activeJointParams_[paramIndex + 6]) {
            const float val = 2.0f * diff.dot(jointState_j.getScaleDerivative(posd)) * pwgt;
            scatter(val, paramIndex + 6);
          }

          jointIndex = this->skeleton_.joints[jointIndex].parent;
        }
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
    gradient += std::get<1>(ets_error_grad[0]).cast<float>();
    error = std::get<0>(ets_error_grad[0]);
  }

  return error;
}

double SimdStateErrorFunction::getJacobian(
    const ModelParameters& /* params */,
    const SkeletonState& state,
    const MeshState& /* unused */,
    Ref<MatrixXf> jacobian,
    Ref<VectorXf> residual,
    int& usedRows) {
  MT_PROFILE_FUNCTION();

  if (state.jointState.empty() || targetState_.size() != state.jointState.size()) {
    usedRows = 0;
    return 0.0f;
  }

  // Each active joint contributes a 12-element block (3 translation + 9 rotation matrix).
  // Compute per-joint base row offsets so the parallel loop can write disjoint regions.
  std::vector<int> rowOffsets(this->skeleton_.joints.size() + 1, 0);
  for (size_t i = 0; i < this->skeleton_.joints.size(); ++i) {
    const bool active = (targetRotationWeights_[i] != 0 || targetPositionWeights_[i] != 0);
    // NOLINTNEXTLINE(facebook-hte-LocalUncheckedArrayBounds)
    rowOffsets[i + 1] = rowOffsets[i] + (active ? static_cast<int>(kConstraintDim) : 0);
  }

  std::vector<double> errors(this->skeleton_.joints.size(), 0.0);

  auto dispensoOptions = dispenso::ParForOptions();
  dispensoOptions.maxThreads = maxThreads_;
  dispenso::parallel_for(
      0,
      this->skeleton_.joints.size(),
      [&](const size_t i) {
        if (targetRotationWeights_[i] == 0 && targetPositionWeights_[i] == 0) {
          return;
        }
        const int baseRow = rowOffsets[i];

        const Eigen::Vector3f transDiff =
            state.jointState[i].translation() - targetState_[i].translation;
        const Eigen::Quaternionf& target = targetState_[i].rotation.normalized();
        const Eigen::Quaternionf& rot = state.jointState[i].rotation();

        const float pwgt = kPositionWeight * posWgt_ * this->weight_ * targetPositionWeights_[i];
        const float rwgt = kOrientationWeight * rotWgt_ * this->weight_ * targetRotationWeights_[i];
        const float wgt = std::sqrt(pwgt);
        const float awgt = std::sqrt(rwgt);

        const Eigen::Matrix3f rotDiff = rot.toRotationMatrix() - target.toRotationMatrix();
        errors[i] = transDiff.squaredNorm() * pwgt + rotDiff.squaredNorm() * rwgt;

        residual.template segment<3>(baseRow).noalias() = transDiff * wgt;
        residual.template segment<9>(baseRow + 3).noalias() =
            Eigen::Map<const Eigen::Matrix<float, 9, 1>>(rotDiff.data()) * awgt;

        size_t jointIndex = i;
        while (jointIndex != kInvalidIndex) {
          const size_t paramIndex = jointIndex * kParametersPerJoint;
          const auto& jointState_j = state.jointState[jointIndex];
          const Eigen::Vector3f posd =
              state.jointState[i].translation() - jointState_j.translation();

          auto scatterTrans = [&](const Eigen::Vector3f& jc, size_t srcParam) {
            for (auto idx = parameterTransform_.transform.outerIndexPtr()[srcParam];
                 idx < parameterTransform_.transform.outerIndexPtr()[srcParam + 1];
                 ++idx) {
              const auto col = parameterTransform_.transform.innerIndexPtr()[idx];
              const float v = parameterTransform_.transform.valuePtr()[idx];
              jacobian.block<3, 1>(baseRow, col) += jc * v;
            }
          };
          auto scatterRot9 = [&](const Eigen::Matrix<float, 9, 1>& jc, size_t srcParam) {
            for (auto idx = parameterTransform_.transform.outerIndexPtr()[srcParam];
                 idx < parameterTransform_.transform.outerIndexPtr()[srcParam + 1];
                 ++idx) {
              const auto col = parameterTransform_.transform.innerIndexPtr()[idx];
              const float v = parameterTransform_.transform.valuePtr()[idx];
              jacobian.block<9, 1>(baseRow + 3, col) += jc * v;
            }
          };

          for (size_t d = 0; d < 3; ++d) {
            if (this->activeJointParams_[paramIndex + d]) {
              scatterTrans(jointState_j.getTranslationDerivative(d) * wgt, paramIndex + d);
            }
            if (this->activeJointParams_[paramIndex + 3 + d]) {
              const Eigen::Vector3f jcTrans = jointState_j.getRotationDerivative(d, posd) * wgt;
              scatterTrans(jcTrans, paramIndex + 3 + d);
              const Eigen::Vector3f axis = jointState_j.rotationAxis.col(d);
              scatterRot9(matrixDiffJacobianCol(rot, axis, awgt), paramIndex + 3 + d);
            }
          }
          if (this->activeJointParams_[paramIndex + 6]) {
            scatterTrans(jointState_j.getScaleDerivative(posd) * wgt, paramIndex + 6);
          }

          jointIndex = this->skeleton_.joints[jointIndex].parent;
        }
      },
      dispensoOptions);

  // NOLINTNEXTLINE(facebook-hte-LocalUncheckedArrayBounds)
  usedRows = rowOffsets.back();
  return std::accumulate(errors.begin(), errors.end(), 0.0);
}

size_t SimdStateErrorFunction::getJacobianSize() const {
  const auto nActive =
      (targetPositionWeights_.array() != 0 || targetRotationWeights_.array() != 0).count();
  return nActive * kConstraintDim;
}

} // namespace momentum
