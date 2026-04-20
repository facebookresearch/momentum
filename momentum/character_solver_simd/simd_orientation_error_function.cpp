/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver_simd/simd_orientation_error_function.h"

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

SimdOrientationConstraints::SimdOrientationConstraints(const Skeleton* skel) {
  MT_CHECK(skel->joints.size() <= kMaxJoints);
  numJoints = static_cast<int>(skel->joints.size());
  const auto dataSize = kMaxConstraints * numJoints;
  MT_CHECK(dataSize % kSimdAlignment == 0);

  // 19 striped float blocks: 9 (offset rotmat) + 9 (target rotmat) + 1 (weight).
  constexpr size_t nBlocks = 19;
  data = makeAlignedUnique<float, kSimdAlignment>(dataSize * nBlocks);
  float* dataPtr = data.get();
  std::fill_n(dataPtr, dataSize * nBlocks, 0.0f);

  offsetC0X = dataPtr + dataSize * 0;
  offsetC0Y = dataPtr + dataSize * 1;
  offsetC0Z = dataPtr + dataSize * 2;
  offsetC1X = dataPtr + dataSize * 3;
  offsetC1Y = dataPtr + dataSize * 4;
  offsetC1Z = dataPtr + dataSize * 5;
  offsetC2X = dataPtr + dataSize * 6;
  offsetC2Y = dataPtr + dataSize * 7;
  offsetC2Z = dataPtr + dataSize * 8;
  targetC0X = dataPtr + dataSize * 9;
  targetC0Y = dataPtr + dataSize * 10;
  targetC0Z = dataPtr + dataSize * 11;
  targetC1X = dataPtr + dataSize * 12;
  targetC1Y = dataPtr + dataSize * 13;
  targetC1Z = dataPtr + dataSize * 14;
  targetC2X = dataPtr + dataSize * 15;
  targetC2Y = dataPtr + dataSize * 16;
  targetC2Z = dataPtr + dataSize * 17;
  weights = dataPtr + dataSize * 18;

  clearConstraints();
}

SimdOrientationConstraints::~SimdOrientationConstraints() = default;

void SimdOrientationConstraints::addConstraint(
    const size_t jointIndex,
    const Eigen::Quaternionf& offset,
    const Eigen::Quaternionf& target,
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
  // Match scalar API: normalize then convert to rotation matrix once.
  const Eigen::Matrix3f offsetMat = offset.normalized().toRotationMatrix();
  const Eigen::Matrix3f targetMat = target.normalized().toRotationMatrix();

  offsetC0X[finalIndex] = offsetMat(0, 0);
  offsetC0Y[finalIndex] = offsetMat(1, 0);
  offsetC0Z[finalIndex] = offsetMat(2, 0);
  offsetC1X[finalIndex] = offsetMat(0, 1);
  offsetC1Y[finalIndex] = offsetMat(1, 1);
  offsetC1Z[finalIndex] = offsetMat(2, 1);
  offsetC2X[finalIndex] = offsetMat(0, 2);
  offsetC2Y[finalIndex] = offsetMat(1, 2);
  offsetC2Z[finalIndex] = offsetMat(2, 2);
  targetC0X[finalIndex] = targetMat(0, 0);
  targetC0Y[finalIndex] = targetMat(1, 0);
  targetC0Z[finalIndex] = targetMat(2, 0);
  targetC1X[finalIndex] = targetMat(0, 1);
  targetC1Y[finalIndex] = targetMat(1, 1);
  targetC1Z[finalIndex] = targetMat(2, 1);
  targetC2X[finalIndex] = targetMat(0, 2);
  targetC2Y[finalIndex] = targetMat(1, 2);
  targetC2Z[finalIndex] = targetMat(2, 2);
  weights[finalIndex] = targetWeight;
}

VectorXi SimdOrientationConstraints::getNumConstraints() const {
  VectorXi res = VectorXi::Zero(numJoints);
  for (int jointIndex = 0; jointIndex < numJoints; ++jointIndex) {
    res[jointIndex] = constraintCount[jointIndex];
  }
  return res;
}

void SimdOrientationConstraints::clearConstraints() {
  std::fill_n(weights, kMaxConstraints * numJoints, 0.0f);
  std::fill_n(constraintCount.begin(), numJoints, 0);
}

SimdOrientationErrorFunction::SimdOrientationErrorFunction(
    const Skeleton& skel,
    const ParameterTransform& pt,
    uint32_t maxThreads)
    : SkeletonErrorFunction(skel, pt), constraints_(nullptr), maxThreads_(maxThreads) {}

SimdOrientationErrorFunction::SimdOrientationErrorFunction(
    const Character& character,
    uint32_t maxThreads)
    : SimdOrientationErrorFunction(character.skeleton, character.parameterTransform, maxThreads) {}

namespace {

inline Vector3fP loadOffsetCol(const SimdOrientationConstraints& cs, size_t k, size_t off) {
  switch (k) {
    case 0:
      return Vector3fP{
          drjit::load<FloatP>(&cs.offsetC0X[off]),
          drjit::load<FloatP>(&cs.offsetC0Y[off]),
          drjit::load<FloatP>(&cs.offsetC0Z[off])};
    case 1:
      return Vector3fP{
          drjit::load<FloatP>(&cs.offsetC1X[off]),
          drjit::load<FloatP>(&cs.offsetC1Y[off]),
          drjit::load<FloatP>(&cs.offsetC1Z[off])};
    default:
      return Vector3fP{
          drjit::load<FloatP>(&cs.offsetC2X[off]),
          drjit::load<FloatP>(&cs.offsetC2Y[off]),
          drjit::load<FloatP>(&cs.offsetC2Z[off])};
  }
}

inline Vector3fP loadTargetCol(const SimdOrientationConstraints& cs, size_t k, size_t off) {
  switch (k) {
    case 0:
      return Vector3fP{
          drjit::load<FloatP>(&cs.targetC0X[off]),
          drjit::load<FloatP>(&cs.targetC0Y[off]),
          drjit::load<FloatP>(&cs.targetC0Z[off])};
    case 1:
      return Vector3fP{
          drjit::load<FloatP>(&cs.targetC1X[off]),
          drjit::load<FloatP>(&cs.targetC1Y[off]),
          drjit::load<FloatP>(&cs.targetC1Z[off])};
    default:
      return Vector3fP{
          drjit::load<FloatP>(&cs.targetC2X[off]),
          drjit::load<FloatP>(&cs.targetC2Y[off]),
          drjit::load<FloatP>(&cs.targetC2Z[off])};
  }
}

} // namespace

double SimdOrientationErrorFunction::getError(
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
      const auto offsetIdx = jointId * SimdOrientationConstraints::kMaxConstraints + index;
      const auto weight = drjit::load<FloatP>(&constraints_->weights[offsetIdx]);

      // Sum the squared norm of all 3 column-difference vectors.
      FloatP sqrErr(0.0f);
      for (size_t k = 0; k < 3; ++k) {
        const Vector3fP offCol = loadOffsetCol(*constraints_, k, offsetIdx);
        const Vector3fP rotCol = momentum::operator*(rotMat, offCol);
        const Vector3fP tgtCol = loadTargetCol(*constraints_, k, offsetIdx);
        const Vector3fP diff = rotCol - tgtCol;
        sqrErr += dot(diff, diff);
      }
      error += weight * sqrErr;
    }
  }

  return static_cast<float>(drjit::sum(error) * kOrientationWeight * weight_);
}

double SimdOrientationErrorFunction::getGradient(
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
          const auto offsetIdx = jointId * SimdOrientationConstraints::kMaxConstraints + index;
          const auto weight = drjit::load<FloatP>(&constraints_->weights[offsetIdx]);
          const FloatP wgt = 2.0f * kOrientationWeight * weight;

          // Each column k contributes a 3D residual diff_k = R*offset_k - target_k. Treat each
          // column as an independent axis-style constraint and sum gradients.
          // NOLINTNEXTLINE(modernize-avoid-c-arrays)
          Vector3fP rotCols[3];
          // NOLINTNEXTLINE(modernize-avoid-c-arrays)
          Vector3fP diffs[3];
          for (size_t k = 0; k < 3; ++k) {
            const Vector3fP offCol = loadOffsetCol(*constraints_, k, offsetIdx);
            rotCols[k] = momentum::operator*(rotMat, offCol);
            const Vector3fP tgtCol = loadTargetCol(*constraints_, k, offsetIdx);
            diffs[k] = rotCols[k] - tgtCol;
            jointError += weight * dot(diffs[k], diffs[k]);
          }

          // Walk the joint chain; only rotation parameters affect orientation residuals.
          size_t jointIndex = jointId;
          while (jointIndex != kInvalidIndex) {
            MT_CHECK(jointIndex < static_cast<size_t>(constraints_->numJoints));

            const auto& jointState = state.jointState[jointIndex];
            const size_t paramIndex = jointIndex * kParametersPerJoint;

            for (size_t d = 0; d < 3; ++d) {
              if (this->activeJointParams_[paramIndex + 3 + d]) {
                const Vector3f& axis = jointState.rotationAxis.col(d);
                FloatP val(0.0f);
                for (size_t k = 0; k < 3; ++k) {
                  val += dot(diffs[k], momentum::cross(axis, rotCols[k]));
                }
                val *= wgt;
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

  return static_cast<float>(kOrientationWeight * weight_ * error);
}

double SimdOrientationErrorFunction::getJacobian(
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
          const auto offsetIdx = jointId * SimdOrientationConstraints::kMaxConstraints + index;
          const auto jacobianOffsetCur = jacobianOffset_[jointId] + jindex;

          const auto weight = drjit::load<FloatP>(&constraints_->weights[offsetIdx]);
          const FloatP wgt = drjit::sqrt(kOrientationWeight * weight_ * weight);

          // For each column k: rotated offset -> diff -> scaled residual. Layout 9 residual rows
          // as [col0xyz packets, col1xyz packets, col2xyz packets] for cache locality.
          // NOLINTNEXTLINE(modernize-avoid-c-arrays)
          Vector3fP rotCols[3];
          // NOLINTNEXTLINE(modernize-avoid-c-arrays)
          Vector3fP diffs[3];
          for (size_t k = 0; k < 3; ++k) {
            const Vector3fP offCol = loadOffsetCol(*constraints_, k, offsetIdx);
            rotCols[k] = momentum::operator*(rotMat, offCol);
            const Vector3fP tgtCol = loadTargetCol(*constraints_, k, offsetIdx);
            diffs[k] = rotCols[k] - tgtCol;
            jointError += weight * dot(diffs[k], diffs[k]);

            const Vector3fP res{wgt * diffs[k].x(), wgt * diffs[k].y(), wgt * diffs[k].z()};
            const size_t baseRow = jacobianOffsetCur + k * 3 * kSimdPacketSize;
            drjit::store(
                residual.segment(baseRow + kSimdPacketSize * 0, kSimdPacketSize).data(), res.x());
            drjit::store(
                residual.segment(baseRow + kSimdPacketSize * 1, kSimdPacketSize).data(), res.y());
            drjit::store(
                residual.segment(baseRow + kSimdPacketSize * 2, kSimdPacketSize).data(), res.z());
          }

          size_t jointIndex = jointId;
          while (jointIndex != kInvalidIndex) {
            MT_CHECK(jointIndex < static_cast<size_t>(constraints_->numJoints));

            const auto& jointState = state.jointState[jointIndex];
            const size_t paramIndex = jointIndex * kParametersPerJoint;

            for (size_t d = 0; d < 3; ++d) {
              if (this->activeJointParams_[paramIndex + 3 + d]) {
                const Vector3f& axis = jointState.rotationAxis.col(d);
                for (size_t k = 0; k < 3; ++k) {
                  const Vector3fP axisCrossRotCol = momentum::cross(axis, rotCols[k]);
                  const FloatP jcx = axisCrossRotCol.x() * wgt;
                  const FloatP jcy = axisCrossRotCol.y() * wgt;
                  const FloatP jcz = axisCrossRotCol.z() * wgt;
                  const size_t baseRow = jacobianOffsetCur + k * 3 * kSimdPacketSize;
                  jacobian_jointParams_to_modelParams(
                      jcx,
                      paramIndex + d + 3,
                      parameterTransform_,
                      jacobian.middleRows(baseRow + kSimdPacketSize * 0, kSimdPacketSize));
                  jacobian_jointParams_to_modelParams(
                      jcy,
                      paramIndex + d + 3,
                      parameterTransform_,
                      jacobian.middleRows(baseRow + kSimdPacketSize * 1, kSimdPacketSize));
                  jacobian_jointParams_to_modelParams(
                      jcz,
                      paramIndex + d + 3,
                      parameterTransform_,
                      jacobian.middleRows(baseRow + kSimdPacketSize * 2, kSimdPacketSize));
                }
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
      kOrientationWeight * weight_ * std::accumulate(jointErrors.begin(), jointErrors.end(), 0.0);
  return static_cast<float>(error);
}

size_t SimdOrientationErrorFunction::getJacobianSize() const {
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
