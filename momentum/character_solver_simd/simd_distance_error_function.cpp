/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver_simd/simd_distance_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_solver_simd/simd_helpers.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/math/constants.h"
#include "momentum/simd/simd.h"

#include <dispenso/parallel_for.h>

#include <numeric>
#include <tuple>

namespace momentum {

SimdDistanceConstraints::SimdDistanceConstraints(const Skeleton* skel) {
  MT_CHECK(skel->joints.size() <= kMaxJoints);
  numJoints = static_cast<int>(skel->joints.size());
  const auto dataSize = kMaxConstraints * numJoints;
  MT_CHECK(dataSize % kSimdAlignment == 0);

  // 8 striped float blocks: offset (xyz), origin (xyz), target, weight.
  constexpr size_t nBlocks = 8;
  data = makeAlignedUnique<float, kSimdAlignment>(dataSize * nBlocks);
  float* dataPtr = data.get();
  std::fill_n(dataPtr, dataSize * nBlocks, 0.0f);

  offsetX = dataPtr + dataSize * 0;
  offsetY = dataPtr + dataSize * 1;
  offsetZ = dataPtr + dataSize * 2;
  originX = dataPtr + dataSize * 3;
  originY = dataPtr + dataSize * 4;
  originZ = dataPtr + dataSize * 5;
  targets = dataPtr + dataSize * 6;
  weights = dataPtr + dataSize * 7;

  clearConstraints();
}

SimdDistanceConstraints::~SimdDistanceConstraints() = default;

void SimdDistanceConstraints::addConstraint(
    const size_t jointIndex,
    const Vector3f& offset,
    const Vector3f& origin,
    const float target,
    const float targetWeight) {
  MT_CHECK(jointIndex < constraintCount.size());

  uint32_t index = 0;
  while (true) {
    index = constraintCount[jointIndex];
    if (index == kMaxConstraints) {
      return;
    }
    if (constraintCount[jointIndex].compare_exchange_weak(index, index + (uint32_t)1)) {
      break;
    }
  }

  const auto finalIndex = jointIndex * kMaxConstraints + index;
  offsetX[finalIndex] = offset.x();
  offsetY[finalIndex] = offset.y();
  offsetZ[finalIndex] = offset.z();
  originX[finalIndex] = origin.x();
  originY[finalIndex] = origin.y();
  originZ[finalIndex] = origin.z();
  targets[finalIndex] = target;
  weights[finalIndex] = targetWeight;
}

VectorXi SimdDistanceConstraints::getNumConstraints() const {
  VectorXi res = VectorXi::Zero(numJoints);
  for (int jointIndex = 0; jointIndex < numJoints; ++jointIndex) {
    res[jointIndex] = constraintCount[jointIndex];
  }
  return res;
}

void SimdDistanceConstraints::clearConstraints() {
  std::fill_n(weights, kMaxConstraints * numJoints, 0.0f);
  std::fill_n(constraintCount.begin(), numJoints, 0);
}

SimdDistanceErrorFunction::SimdDistanceErrorFunction(
    const Skeleton& skel,
    const ParameterTransform& pt,
    uint32_t maxThreads)
    : SkeletonErrorFunction(skel, pt), constraints_(nullptr), maxThreads_(maxThreads) {}

SimdDistanceErrorFunction::SimdDistanceErrorFunction(
    const Character& character,
    uint32_t maxThreads)
    : SimdDistanceErrorFunction(character.skeleton, character.parameterTransform, maxThreads) {}

double SimdDistanceErrorFunction::getError(
    const ModelParameters& /* params */,
    const SkeletonState& state,
    const MeshState& /* unused */) {
  if (constraints_ == nullptr) {
    return 0.0f;
  }

  MT_CHECK((size_t)constraints_->numJoints <= state.jointState.size());

  auto error = drjit::zeros<DoubleP>();
  for (int jointId = 0; jointId < constraints_->numJoints; ++jointId) {
    const size_t constraintCount = constraints_->constraintCount[jointId];
    // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
    const auto& jointState = state.jointState[jointId];

    for (uint32_t index = 0; index < constraintCount; index += kSimdPacketSize) {
      const auto offsetIdx = jointId * SimdDistanceConstraints::kMaxConstraints + index;

      const Vector3fP offset{
          drjit::load<FloatP>(&constraints_->offsetX[offsetIdx]),
          drjit::load<FloatP>(&constraints_->offsetY[offsetIdx]),
          drjit::load<FloatP>(&constraints_->offsetZ[offsetIdx])};
      const Vector3fP pos_world = jointState.transform * offset;

      const Vector3fP origin{
          drjit::load<FloatP>(&constraints_->originX[offsetIdx]),
          drjit::load<FloatP>(&constraints_->originY[offsetIdx]),
          drjit::load<FloatP>(&constraints_->originZ[offsetIdx])};
      const Vector3fP diff = pos_world - origin;

      // distance = ||diff||; residual = distance - target.
      const FloatP distance = drjit::sqrt(dot(diff, diff));
      const auto target = drjit::load<FloatP>(&constraints_->targets[offsetIdx]);
      const FloatP residual = distance - target;

      const auto weight = drjit::load<FloatP>(&constraints_->weights[offsetIdx]);
      error += weight * drjit::square(residual);
    }
  }

  return static_cast<float>(drjit::sum(error) * kDistanceWeight * weight_);
}

double SimdDistanceErrorFunction::getGradient(
    const ModelParameters& /* params */,
    const SkeletonState& state,
    const MeshState& /* unused */,
    Ref<VectorXf> gradient) {
  if (constraints_ == nullptr) {
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
      constraints_->numJoints,
      [&](std::tuple<double, VectorXd>& error_grad_local, const size_t jointId) {
        double& error_local = std::get<0>(error_grad_local);
        auto& grad_local = std::get<1>(error_grad_local);

        const size_t constraintCount = constraints_->constraintCount[jointId];
        const auto& jointState_cons = state.jointState[jointId];
        auto jointError = drjit::zeros<DoubleP>();

        for (uint32_t index = 0; index < constraintCount; index += kSimdPacketSize) {
          const auto offsetIdx = jointId * SimdDistanceConstraints::kMaxConstraints + index;

          const Vector3fP offset{
              drjit::load<FloatP>(&constraints_->offsetX[offsetIdx]),
              drjit::load<FloatP>(&constraints_->offsetY[offsetIdx]),
              drjit::load<FloatP>(&constraints_->offsetZ[offsetIdx])};
          const Vector3fP pos_world = jointState_cons.transform * offset;

          const Vector3fP origin{
              drjit::load<FloatP>(&constraints_->originX[offsetIdx]),
              drjit::load<FloatP>(&constraints_->originY[offsetIdx]),
              drjit::load<FloatP>(&constraints_->originZ[offsetIdx])};
          const Vector3fP diff = pos_world - origin;

          const FloatP distSq = dot(diff, diff);
          // Mask off lanes whose distance is below the singular-Jacobian threshold; their
          // contribution to error and gradient is silently dropped to match the scalar version.
          const auto valid = distSq > (kDistanceEps * kDistanceEps);
          const FloatP safeDistSq = drjit::select(valid, distSq, FloatP(1.0f));
          const FloatP distance = drjit::sqrt(safeDistSq);
          const FloatP invDistance = drjit::select(valid, 1.0f / distance, FloatP(0.0f));

          const auto target = drjit::load<FloatP>(&constraints_->targets[offsetIdx]);
          const FloatP residual = distance - target;
          const auto weight = drjit::load<FloatP>(&constraints_->weights[offsetIdx]);

          jointError +=
              weight * drjit::square(residual) * drjit::select(valid, FloatP(1.0f), FloatP(0.0f));

          // d(error)/d(p_world) = 2 * weight * residual * (diff / distance).
          const FloatP wgt = 2.0f * kDistanceWeight * weight * residual * invDistance;
          const Vector3fP dE_dpWorld{wgt * diff.x(), wgt * diff.y(), wgt * diff.z()};

          // Walk the joint chain accumulating chain-rule derivatives of pos_world wrt each joint
          // parameter, identical to SimdPositionErrorFunction's loop.
          size_t jointIndex = jointId;
          while (jointIndex != kInvalidIndex) {
            MT_CHECK(jointIndex < static_cast<size_t>(constraints_->numJoints));

            const auto& jointState = state.jointState[jointIndex];
            const size_t paramIndex = jointIndex * kParametersPerJoint;
            const Vector3fP posd = momentum::operator-(pos_world, jointState.translation());

            for (size_t d = 0; d < 3; ++d) {
              if (this->activeJointParams_[paramIndex + d]) {
                const FloatP val =
                    momentum::dot(dE_dpWorld, jointState.getTranslationDerivative(d));
                for (auto ptIndex = parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
                     ptIndex < parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
                     ++ptIndex) {
                  grad_local[parameterTransform_.transform.innerIndexPtr()[ptIndex]] +=
                      drjit::sum(val) * parameterTransform_.transform.valuePtr()[ptIndex];
                }
              }
              if (this->activeJointParams_[paramIndex + 3 + d]) {
                const FloatP val =
                    dot(dE_dpWorld, momentum::cross(jointState.rotationAxis.col(d), posd));
                for (auto ptIndex =
                         parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 3];
                     ptIndex <
                     parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 3 + 1];
                     ++ptIndex) {
                  grad_local[parameterTransform_.transform.innerIndexPtr()[ptIndex]] +=
                      drjit::sum(val) * parameterTransform_.transform.valuePtr()[ptIndex];
                }
              }
            }
            if (this->activeJointParams_[paramIndex + 6]) {
              const FloatP val = ln2<double>() * dot(dE_dpWorld, posd);
              for (auto ptIndex = parameterTransform_.transform.outerIndexPtr()[paramIndex + 6];
                   ptIndex < parameterTransform_.transform.outerIndexPtr()[paramIndex + 6 + 1];
                   ++ptIndex) {
                grad_local[parameterTransform_.transform.innerIndexPtr()[ptIndex]] +=
                    drjit::sum(val) * parameterTransform_.transform.valuePtr()[ptIndex];
              }
            }

            jointIndex = this->skeleton_.joints[jointIndex].parent;
          }
        }

        error_local += drjit::sum(jointError);
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

double SimdDistanceErrorFunction::getJacobian(
    const ModelParameters& /* params */,
    const SkeletonState& state,
    const MeshState& /* unused */,
    Ref<MatrixXf> jacobian,
    Ref<VectorXf> residual,
    int& usedRows) {
  MT_PROFILE_FUNCTION();

  if (constraints_ == nullptr) {
    return 0.0f;
  }

  MT_CHECK(jacobian.cols() == static_cast<Eigen::Index>(parameterTransform_.transform.cols()));
  MT_CHECK((size_t)constraints_->numJoints <= jacobianOffset_.size());

  std::vector<double> jointErrors(constraints_->numJoints);

  auto dispensoOptions = dispenso::ParForOptions();
  dispensoOptions.maxThreads = maxThreads_;
  dispenso::parallel_for(
      0,
      constraints_->numJoints,
      [&](const size_t jointId) {
        const size_t constraintCount = constraints_->constraintCount[jointId];
        const auto& jointState_cons = state.jointState[jointId];
        auto jointError = drjit::zeros<DoubleP>();

        for (uint32_t index = 0; index < constraintCount; index += kSimdPacketSize) {
          const auto offsetIdx = jointId * SimdDistanceConstraints::kMaxConstraints + index;
          const auto jacobianOffsetCur = jacobianOffset_[jointId] + index;

          const Vector3fP offset{
              drjit::load<FloatP>(&constraints_->offsetX[offsetIdx]),
              drjit::load<FloatP>(&constraints_->offsetY[offsetIdx]),
              drjit::load<FloatP>(&constraints_->offsetZ[offsetIdx])};
          const Vector3fP pos_world = jointState_cons.transform * offset;

          const Vector3fP origin{
              drjit::load<FloatP>(&constraints_->originX[offsetIdx]),
              drjit::load<FloatP>(&constraints_->originY[offsetIdx]),
              drjit::load<FloatP>(&constraints_->originZ[offsetIdx])};
          const Vector3fP diff = pos_world - origin;

          const FloatP distSq = dot(diff, diff);
          const auto valid = distSq > (kDistanceEps * kDistanceEps);
          const FloatP safeDistSq = drjit::select(valid, distSq, FloatP(1.0f));
          const FloatP distance = drjit::sqrt(safeDistSq);
          const FloatP invDistance = drjit::select(valid, 1.0f / distance, FloatP(0.0f));

          const auto target = drjit::load<FloatP>(&constraints_->targets[offsetIdx]);
          const FloatP residualP = distance - target;
          const auto weight = drjit::load<FloatP>(&constraints_->weights[offsetIdx]);

          jointError +=
              weight * drjit::square(residualP) * drjit::select(valid, FloatP(1.0f), FloatP(0.0f));

          // sqrt-weight scales residual so that residual^T * residual reproduces the loss.
          const FloatP wgt = drjit::sqrt(kDistanceWeight * weight_ * weight);
          const FloatP scaledResidual = wgt * residualP;
          drjit::store(residual.segment(jacobianOffsetCur, kSimdPacketSize).data(), scaledResidual);

          // d(residual)/d(p_world) = (diff / distance), then scaled by sqrt-weight.
          const FloatP dResidual = wgt * invDistance;
          const Vector3fP dRdP{dResidual * diff.x(), dResidual * diff.y(), dResidual * diff.z()};

          size_t jointIndex = jointId;
          while (jointIndex != kInvalidIndex) {
            MT_CHECK(jointIndex < static_cast<size_t>(constraints_->numJoints));

            const auto& jointState = state.jointState[jointIndex];
            const size_t paramIndex = jointIndex * kParametersPerJoint;
            const Vector3fP posd = momentum::operator-(pos_world, jointState.translation());

            for (size_t d = 0; d < 3; ++d) {
              if (this->activeJointParams_[paramIndex + d]) {
                const FloatP jc = momentum::dot(dRdP, jointState.getTranslationDerivative(d));
                jacobian_jointParams_to_modelParams(
                    jc,
                    paramIndex + d,
                    parameterTransform_,
                    jacobian.middleRows(jacobianOffsetCur, kSimdPacketSize));
              }
              if (this->activeJointParams_[paramIndex + 3 + d]) {
                const FloatP jc = dot(dRdP, momentum::cross(jointState.rotationAxis.col(d), posd));
                jacobian_jointParams_to_modelParams(
                    jc,
                    paramIndex + d + 3,
                    parameterTransform_,
                    jacobian.middleRows(jacobianOffsetCur, kSimdPacketSize));
              }
            }
            if (this->activeJointParams_[paramIndex + 6]) {
              const FloatP jc = ln2<double>() * dot(dRdP, posd);
              jacobian_jointParams_to_modelParams(
                  jc,
                  paramIndex + 6,
                  parameterTransform_,
                  jacobian.middleRows(jacobianOffsetCur, kSimdPacketSize));
            }

            jointIndex = this->skeleton_.joints[jointIndex].parent;
          }
        }

        jointErrors[jointId] += drjit::sum(jointError);
      },
      dispensoOptions);

  usedRows = gsl::narrow_cast<int>(jacobian.rows());

  const double error =
      kDistanceWeight * weight_ * std::accumulate(jointErrors.begin(), jointErrors.end(), 0.0);
  return static_cast<float>(error);
}

size_t SimdDistanceErrorFunction::getJacobianSize() const {
  if (constraints_ == nullptr) {
    return 0;
  }

  jacobianOffset_.resize(constraints_->numJoints);
  size_t num = 0;
  for (int i = 0; i < constraints_->numJoints; ++i) {
    jacobianOffset_[i] = num;
    const size_t count = constraints_->constraintCount[i];
    const auto numPackets = (count + kSimdPacketSize - 1) / kSimdPacketSize;
    num += numPackets * kSimdPacketSize;
  }
  return num;
}

} // namespace momentum
