/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver_simd/simd_fixed_axis_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_solver_simd/simd_helpers.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/simd/simd.h"

#include <dispenso/parallel_for.h>

#include <numeric>
#include <tuple>

namespace momentum {

SimdFixedAxisConstraints::SimdFixedAxisConstraints(const Skeleton* skel) {
  MT_CHECK(skel->joints.size() <= kMaxJoints);
  numJoints = static_cast<int>(skel->joints.size());
  const auto dataSize = kMaxConstraints * numJoints;
  MT_CHECK(dataSize % kSimdAlignment == 0);

  // 7 striped float blocks: localAxis (xyz), globalAxis (xyz), weight.
  constexpr size_t nBlocks = 7;
  data = makeAlignedUnique<float, kSimdAlignment>(dataSize * nBlocks);
  float* dataPtr = data.get();
  std::fill_n(dataPtr, dataSize * nBlocks, 0.0f);

  localAxisX = dataPtr + dataSize * 0;
  localAxisY = dataPtr + dataSize * 1;
  localAxisZ = dataPtr + dataSize * 2;
  globalAxisX = dataPtr + dataSize * 3;
  globalAxisY = dataPtr + dataSize * 4;
  globalAxisZ = dataPtr + dataSize * 5;
  weights = dataPtr + dataSize * 6;

  clearConstraints();
}

SimdFixedAxisConstraints::~SimdFixedAxisConstraints() = default;

void SimdFixedAxisConstraints::addConstraint(
    const size_t jointIndex,
    const Vector3f& localAxis,
    const Vector3f& globalAxis,
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
  // Match the scalar API which normalizes axes inside the constraint constructor.
  const Vector3f localNorm = localAxis.normalized();
  const Vector3f globalNorm = globalAxis.normalized();
  localAxisX[finalIndex] = localNorm.x();
  localAxisY[finalIndex] = localNorm.y();
  localAxisZ[finalIndex] = localNorm.z();
  globalAxisX[finalIndex] = globalNorm.x();
  globalAxisY[finalIndex] = globalNorm.y();
  globalAxisZ[finalIndex] = globalNorm.z();
  weights[finalIndex] = targetWeight;
}

VectorXi SimdFixedAxisConstraints::getNumConstraints() const {
  VectorXi res = VectorXi::Zero(numJoints);
  for (int jointIndex = 0; jointIndex < numJoints; ++jointIndex) {
    res[jointIndex] = constraintCount[jointIndex];
  }
  return res;
}

void SimdFixedAxisConstraints::clearConstraints() {
  std::fill_n(weights, kMaxConstraints * numJoints, 0.0f);
  std::fill_n(constraintCount.begin(), numJoints, 0);
}

SimdFixedAxisErrorFunction::SimdFixedAxisErrorFunction(
    const Skeleton& skel,
    const ParameterTransform& pt,
    uint32_t maxThreads)
    : SkeletonErrorFunction(skel, pt), constraints_(nullptr), maxThreads_(maxThreads) {}

SimdFixedAxisErrorFunction::SimdFixedAxisErrorFunction(
    const Character& character,
    uint32_t maxThreads)
    : SimdFixedAxisErrorFunction(character.skeleton, character.parameterTransform, maxThreads) {}

double SimdFixedAxisErrorFunction::getError(
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
    const Eigen::Matrix3f rotMat = jointState.rotation().toRotationMatrix();

    for (uint32_t index = 0; index < constraintCount; index += kSimdPacketSize) {
      const auto offsetIdx = jointId * SimdFixedAxisConstraints::kMaxConstraints + index;

      const Vector3fP localAxis{
          drjit::load<FloatP>(&constraints_->localAxisX[offsetIdx]),
          drjit::load<FloatP>(&constraints_->localAxisY[offsetIdx]),
          drjit::load<FloatP>(&constraints_->localAxisZ[offsetIdx])};
      const Vector3fP axisWorld = momentum::operator*(rotMat, localAxis);

      const Vector3fP globalAxis{
          drjit::load<FloatP>(&constraints_->globalAxisX[offsetIdx]),
          drjit::load<FloatP>(&constraints_->globalAxisY[offsetIdx]),
          drjit::load<FloatP>(&constraints_->globalAxisZ[offsetIdx])};
      const Vector3fP diff = axisWorld - globalAxis;

      const FloatP sqrErr = dot(diff, diff);
      const auto weight = drjit::load<FloatP>(&constraints_->weights[offsetIdx]);
      error += weight * sqrErr;
    }
  }

  return static_cast<float>(drjit::sum(error) * kFixedAxisWeight * weight_);
}

double SimdFixedAxisErrorFunction::getGradient(
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
        const Eigen::Matrix3f rotMat = jointState_cons.rotation().toRotationMatrix();
        auto jointError = drjit::zeros<DoubleP>();

        for (uint32_t index = 0; index < constraintCount; index += kSimdPacketSize) {
          const auto offsetIdx = jointId * SimdFixedAxisConstraints::kMaxConstraints + index;

          const Vector3fP localAxis{
              drjit::load<FloatP>(&constraints_->localAxisX[offsetIdx]),
              drjit::load<FloatP>(&constraints_->localAxisY[offsetIdx]),
              drjit::load<FloatP>(&constraints_->localAxisZ[offsetIdx])};
          const Vector3fP axisWorld = momentum::operator*(rotMat, localAxis);

          const Vector3fP globalAxis{
              drjit::load<FloatP>(&constraints_->globalAxisX[offsetIdx]),
              drjit::load<FloatP>(&constraints_->globalAxisY[offsetIdx]),
              drjit::load<FloatP>(&constraints_->globalAxisZ[offsetIdx])};
          const Vector3fP diff = axisWorld - globalAxis;

          const auto weight = drjit::load<FloatP>(&constraints_->weights[offsetIdx]);
          jointError += weight * dot(diff, diff);

          // For an axis (no translation/scale dependency), only rotation parameters contribute.
          // d(diff)/d(rotation_d) = axis x axisWorld, then dot with diff for the chain rule.
          const FloatP wgt = 2.0f * kFixedAxisWeight * weight;

          size_t jointIndex = jointId;
          while (jointIndex != kInvalidIndex) {
            MT_CHECK(jointIndex < static_cast<size_t>(constraints_->numJoints));

            const auto& jointState = state.jointState[jointIndex];
            const size_t paramIndex = jointIndex * kParametersPerJoint;

            for (size_t d = 0; d < 3; ++d) {
              if (this->activeJointParams_[paramIndex + 3 + d]) {
                // getRotationDerivative for an axis is axis.cross(axisWorld) (offset == axisWorld).
                const FloatP val =
                    wgt * dot(diff, momentum::cross(jointState.rotationAxis.col(d), axisWorld));
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

  return static_cast<float>(kFixedAxisWeight * weight_ * error);
}

double SimdFixedAxisErrorFunction::getJacobian(
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
        const Eigen::Matrix3f rotMat = jointState_cons.rotation().toRotationMatrix();
        auto jointError = drjit::zeros<DoubleP>();

        for (uint32_t index = 0, jindex = 0; index < constraintCount;
             index += kSimdPacketSize, jindex += kSimdPacketSize * kConstraintDim) {
          const auto offsetIdx = jointId * SimdFixedAxisConstraints::kMaxConstraints + index;
          const auto jacobianOffsetCur = jacobianOffset_[jointId] + jindex;

          const Vector3fP localAxis{
              drjit::load<FloatP>(&constraints_->localAxisX[offsetIdx]),
              drjit::load<FloatP>(&constraints_->localAxisY[offsetIdx]),
              drjit::load<FloatP>(&constraints_->localAxisZ[offsetIdx])};
          const Vector3fP axisWorld = momentum::operator*(rotMat, localAxis);

          const Vector3fP globalAxis{
              drjit::load<FloatP>(&constraints_->globalAxisX[offsetIdx]),
              drjit::load<FloatP>(&constraints_->globalAxisY[offsetIdx]),
              drjit::load<FloatP>(&constraints_->globalAxisZ[offsetIdx])};
          const Vector3fP diff = axisWorld - globalAxis;

          const auto weight = drjit::load<FloatP>(&constraints_->weights[offsetIdx]);
          jointError += weight * dot(diff, diff);

          const FloatP wgt = drjit::sqrt(kFixedAxisWeight * weight_ * weight);
          const Vector3fP res = wgt * diff;

          drjit::store(
              residual.segment(jacobianOffsetCur + kSimdPacketSize * 0, kSimdPacketSize).data(),
              res.x());
          drjit::store(
              residual.segment(jacobianOffsetCur + kSimdPacketSize * 1, kSimdPacketSize).data(),
              res.y());
          drjit::store(
              residual.segment(jacobianOffsetCur + kSimdPacketSize * 2, kSimdPacketSize).data(),
              res.z());

          size_t jointIndex = jointId;
          while (jointIndex != kInvalidIndex) {
            MT_CHECK(jointIndex < static_cast<size_t>(constraints_->numJoints));

            const auto& jointState = state.jointState[jointIndex];
            const size_t paramIndex = jointIndex * kParametersPerJoint;

            // Only rotation parameters contribute (axis-only constraint).
            for (size_t d = 0; d < 3; ++d) {
              if (this->activeJointParams_[paramIndex + 3 + d]) {
                const auto axisCrossAxisWorld =
                    momentum::cross(jointState.rotationAxis.col(d), axisWorld);
                const FloatP jcx = axisCrossAxisWorld.x() * wgt;
                const FloatP jcy = axisCrossAxisWorld.y() * wgt;
                const FloatP jcz = axisCrossAxisWorld.z() * wgt;
                jacobian_jointParams_to_modelParams(
                    jcx,
                    paramIndex + d + 3,
                    parameterTransform_,
                    jacobian.middleRows(jacobianOffsetCur + kSimdPacketSize * 0, kSimdPacketSize));
                jacobian_jointParams_to_modelParams(
                    jcy,
                    paramIndex + d + 3,
                    parameterTransform_,
                    jacobian.middleRows(jacobianOffsetCur + kSimdPacketSize * 1, kSimdPacketSize));
                jacobian_jointParams_to_modelParams(
                    jcz,
                    paramIndex + d + 3,
                    parameterTransform_,
                    jacobian.middleRows(jacobianOffsetCur + kSimdPacketSize * 2, kSimdPacketSize));
              }
            }

            jointIndex = this->skeleton_.joints[jointIndex].parent;
          }
        }

        jointErrors[jointId] += drjit::sum(jointError);
      },
      dispensoOptions);

  usedRows = gsl::narrow_cast<int>(jacobian.rows());

  const double error =
      kFixedAxisWeight * weight_ * std::accumulate(jointErrors.begin(), jointErrors.end(), 0.0);
  return static_cast<float>(error);
}

size_t SimdFixedAxisErrorFunction::getJacobianSize() const {
  if (constraints_ == nullptr) {
    return 0;
  }

  jacobianOffset_.resize(constraints_->numJoints);
  size_t num = 0;
  for (int i = 0; i < constraints_->numJoints; ++i) {
    jacobianOffset_[i] = num;
    const size_t count = constraints_->constraintCount[i];
    const auto numPackets = (count + kSimdPacketSize - 1) / kSimdPacketSize;
    num += numPackets * kSimdPacketSize * kConstraintDim;
  }
  return num;
}

} // namespace momentum
