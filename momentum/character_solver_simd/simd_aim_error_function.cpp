/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver_simd/simd_aim_error_function.h"

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

SimdAimConstraints::SimdAimConstraints(const Skeleton* skel) {
  MT_CHECK(skel->joints.size() <= kMaxJoints);
  numJoints = static_cast<int>(skel->joints.size());
  const auto dataSize = kMaxConstraints * numJoints;
  MT_CHECK(dataSize % kSimdAlignment == 0);

  // 10 striped float blocks: localPoint (xyz), localDir (xyz), globalTarget (xyz), weight.
  constexpr size_t nBlocks = 10;
  data = makeAlignedUnique<float, kSimdAlignment>(dataSize * nBlocks);
  float* dataPtr = data.get();
  std::fill_n(dataPtr, dataSize * nBlocks, 0.0f);

  localPointX = dataPtr + dataSize * 0;
  localPointY = dataPtr + dataSize * 1;
  localPointZ = dataPtr + dataSize * 2;
  localDirX = dataPtr + dataSize * 3;
  localDirY = dataPtr + dataSize * 4;
  localDirZ = dataPtr + dataSize * 5;
  targetX = dataPtr + dataSize * 6;
  targetY = dataPtr + dataSize * 7;
  targetZ = dataPtr + dataSize * 8;
  weights = dataPtr + dataSize * 9;

  clearConstraints();
}

SimdAimConstraints::~SimdAimConstraints() = default;

void SimdAimConstraints::addConstraint(
    const size_t jointIndex,
    const Vector3f& localPoint,
    const Vector3f& localDir,
    const Vector3f& globalTarget,
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
  // Match the scalar API which normalizes localDir inside the constraint constructor.
  const Vector3f dir = localDir.normalized();
  localPointX[finalIndex] = localPoint.x();
  localPointY[finalIndex] = localPoint.y();
  localPointZ[finalIndex] = localPoint.z();
  localDirX[finalIndex] = dir.x();
  localDirY[finalIndex] = dir.y();
  localDirZ[finalIndex] = dir.z();
  targetX[finalIndex] = globalTarget.x();
  targetY[finalIndex] = globalTarget.y();
  targetZ[finalIndex] = globalTarget.z();
  weights[finalIndex] = targetWeight;
}

VectorXi SimdAimConstraints::getNumConstraints() const {
  VectorXi res = VectorXi::Zero(numJoints);
  for (int jointIndex = 0; jointIndex < numJoints; ++jointIndex) {
    res[jointIndex] = constraintCount[jointIndex];
  }
  return res;
}

void SimdAimConstraints::clearConstraints() {
  std::fill_n(weights, kMaxConstraints * numJoints, 0.0f);
  std::fill_n(constraintCount.begin(), numJoints, 0);
}

SimdAimDistErrorFunction::SimdAimDistErrorFunction(
    const Skeleton& skel,
    const ParameterTransform& pt,
    uint32_t maxThreads)
    : SkeletonErrorFunction(skel, pt), constraints_(nullptr), maxThreads_(maxThreads) {}

SimdAimDistErrorFunction::SimdAimDistErrorFunction(const Character& character, uint32_t maxThreads)
    : SimdAimDistErrorFunction(character.skeleton, character.parameterTransform, maxThreads) {}

namespace {

/// Aim-distance residual evaluation, returns (point_world, dir_world, tgtVec, projLength, f).
struct AimEval {
  Vector3fP pointWorld;
  Vector3fP dirWorld;
  Vector3fP tgtVec; // globalTarget - pointWorld
  FloatP projLength; // dirWorld . tgtVec
  Vector3fP f; // projLength * dirWorld - tgtVec
};

template <typename JointStateT>
inline AimEval evaluateAim(
    const JointStateT& jointState,
    const Eigen::Matrix3f& rotMat,
    const SimdAimConstraints& cs,
    size_t offsetIdx) {
  AimEval ev;
  const Vector3fP localPoint{
      drjit::load<FloatP>(&cs.localPointX[offsetIdx]),
      drjit::load<FloatP>(&cs.localPointY[offsetIdx]),
      drjit::load<FloatP>(&cs.localPointZ[offsetIdx])};
  ev.pointWorld = jointState.transform * localPoint;

  const Vector3fP localDir{
      drjit::load<FloatP>(&cs.localDirX[offsetIdx]),
      drjit::load<FloatP>(&cs.localDirY[offsetIdx]),
      drjit::load<FloatP>(&cs.localDirZ[offsetIdx])};
  ev.dirWorld = momentum::operator*(rotMat, localDir);

  const Vector3fP target{
      drjit::load<FloatP>(&cs.targetX[offsetIdx]),
      drjit::load<FloatP>(&cs.targetY[offsetIdx]),
      drjit::load<FloatP>(&cs.targetZ[offsetIdx])};
  ev.tgtVec = target - ev.pointWorld;
  ev.projLength = dot(ev.dirWorld, ev.tgtVec);
  ev.f = Vector3fP{
      ev.projLength * ev.dirWorld.x() - ev.tgtVec.x(),
      ev.projLength * ev.dirWorld.y() - ev.tgtVec.y(),
      ev.projLength * ev.dirWorld.z() - ev.tgtVec.z()};
  return ev;
}

} // namespace

double SimdAimDistErrorFunction::getError(
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
      const auto offsetIdx = jointId * SimdAimConstraints::kMaxConstraints + index;
      const AimEval ev = evaluateAim(jointState, rotMat, *constraints_, offsetIdx);

      const auto weight = drjit::load<FloatP>(&constraints_->weights[offsetIdx]);
      error += weight * dot(ev.f, ev.f);
    }
  }

  return static_cast<float>(drjit::sum(error) * kAimWeight * weight_);
}

namespace {

/// Apply the chain rule from joint-parameter derivatives to model-parameter accumulators
/// for a 1D aim contribution; uses the shared sparse-transform helper.
inline void scatter1d(
    const FloatP& val,
    size_t paramIndex,
    const ParameterTransform& parameterTransform,
    VectorXd& grad_local) {
  for (auto ptIndex = parameterTransform.transform.outerIndexPtr()[paramIndex];
       ptIndex < parameterTransform.transform.outerIndexPtr()[paramIndex + 1];
       ++ptIndex) {
    grad_local[parameterTransform.transform.innerIndexPtr()[ptIndex]] +=
        drjit::sum(val) * parameterTransform.transform.valuePtr()[ptIndex];
  }
}

} // namespace

double SimdAimDistErrorFunction::getGradient(
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
          const auto offsetIdx = jointId * SimdAimConstraints::kMaxConstraints + index;
          const AimEval ev = evaluateAim(jointState_cons, rotMat, *constraints_, offsetIdx);

          const auto weight = drjit::load<FloatP>(&constraints_->weights[offsetIdx]);
          jointError += weight * dot(ev.f, ev.f);

          // Per the chain rule, d(error)/d(joint param) = 2*w * (dfdv_p^T * f).dot(d(point)/dq)
          // + (dfdv_d^T * f).dot(d(dir)/dq).
          // dfdv_p = I - dir·dir^T  (symmetric)
          // dfdv_d = projLength·I + dir·tgtVec^T
          // dfdv_p^T * f = f - dir * (dir·f)
          // dfdv_d^T * f = projLength·f + tgtVec * (dir·f)
          const FloatP wgt = 2.0f * kAimWeight * weight;
          const FloatP dirDotF = dot(ev.dirWorld, ev.f);
          const Vector3fP wRpoint{
              ev.f.x() - ev.dirWorld.x() * dirDotF,
              ev.f.y() - ev.dirWorld.y() * dirDotF,
              ev.f.z() - ev.dirWorld.z() * dirDotF};
          const Vector3fP wRdir{
              ev.projLength * ev.f.x() + ev.tgtVec.x() * dirDotF,
              ev.projLength * ev.f.y() + ev.tgtVec.y() * dirDotF,
              ev.projLength * ev.f.z() + ev.tgtVec.z() * dirDotF};

          size_t jointIndex = jointId;
          while (jointIndex != kInvalidIndex) {
            MT_CHECK(jointIndex < static_cast<size_t>(constraints_->numJoints));

            const auto& jointState = state.jointState[jointIndex];
            const size_t paramIndex = jointIndex * kParametersPerJoint;
            const Vector3fP posdPoint =
                momentum::operator-(ev.pointWorld, jointState.translation());

            for (size_t d = 0; d < 3; ++d) {
              // Translation: only point contributes.
              if (this->activeJointParams_[paramIndex + d]) {
                const FloatP val =
                    wgt * momentum::dot(wRpoint, jointState.getTranslationDerivative(d));
                scatter1d(val, paramIndex + d, parameterTransform_, grad_local);
              }
              // Rotation: both point (offset = posdPoint) and dir (offset = dirWorld) contribute.
              if (this->activeJointParams_[paramIndex + 3 + d]) {
                const Vector3f& axis = jointState.rotationAxis.col(d);
                const FloatP valPoint = wgt * dot(wRpoint, momentum::cross(axis, posdPoint));
                const FloatP valDir = wgt * dot(wRdir, momentum::cross(axis, ev.dirWorld));
                scatter1d(valPoint + valDir, paramIndex + d + 3, parameterTransform_, grad_local);
              }
            }
            // Scale: only point contributes.
            if (this->activeJointParams_[paramIndex + 6]) {
              const FloatP val = wgt * ln2<double>() * dot(wRpoint, posdPoint);
              scatter1d(val, paramIndex + 6, parameterTransform_, grad_local);
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

  return static_cast<float>(kAimWeight * weight_ * error);
}

double SimdAimDistErrorFunction::getJacobian(
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
          const auto offsetIdx = jointId * SimdAimConstraints::kMaxConstraints + index;
          const auto jacobianOffsetCur = jacobianOffset_[jointId] + jindex;

          const AimEval ev = evaluateAim(jointState_cons, rotMat, *constraints_, offsetIdx);

          const auto weight = drjit::load<FloatP>(&constraints_->weights[offsetIdx]);
          jointError += weight * dot(ev.f, ev.f);

          // sqrt-weight scaling so residual^T * residual reproduces the loss.
          const FloatP wgt = drjit::sqrt(kAimWeight * weight_ * weight);

          const Vector3fP res{wgt * ev.f.x(), wgt * ev.f.y(), wgt * ev.f.z()};
          drjit::store(
              residual.segment(jacobianOffsetCur + kSimdPacketSize * 0, kSimdPacketSize).data(),
              res.x());
          drjit::store(
              residual.segment(jacobianOffsetCur + kSimdPacketSize * 1, kSimdPacketSize).data(),
              res.y());
          drjit::store(
              residual.segment(jacobianOffsetCur + kSimdPacketSize * 2, kSimdPacketSize).data(),
              res.z());

          // d(f)/d(point) = I - dir*dir^T;  d(f)/d(dir) = projLength*I + dir*tgtVec^T.
          // For the Jacobian columns, we apply each of the two operators componentwise.
          size_t jointIndex = jointId;
          while (jointIndex != kInvalidIndex) {
            MT_CHECK(jointIndex < static_cast<size_t>(constraints_->numJoints));

            const auto& jointState = state.jointState[jointIndex];
            const size_t paramIndex = jointIndex * kParametersPerJoint;
            const Vector3fP posdPoint =
                momentum::operator-(ev.pointWorld, jointState.translation());

            auto applyDfdvPoint = [&](const Vector3fP& dPoint, Vector3fP& outRow) {
              // dfdv_p * dPoint = dPoint - dir * (dir·dPoint)
              const FloatP dirDotDp = dot(ev.dirWorld, dPoint);
              outRow = Vector3fP{
                  dPoint.x() - ev.dirWorld.x() * dirDotDp,
                  dPoint.y() - ev.dirWorld.y() * dirDotDp,
                  dPoint.z() - ev.dirWorld.z() * dirDotDp};
            };

            auto applyDfdvDir = [&](const Vector3fP& dDir, Vector3fP& outRow) {
              // dfdv_d * dDir = projLength * dDir + dir * (tgtVec·dDir)
              const FloatP tgtDotDd = dot(ev.tgtVec, dDir);
              outRow = Vector3fP{
                  ev.projLength * dDir.x() + ev.dirWorld.x() * tgtDotDd,
                  ev.projLength * dDir.y() + ev.dirWorld.y() * tgtDotDd,
                  ev.projLength * dDir.z() + ev.dirWorld.z() * tgtDotDd};
            };

            auto storeJacRow = [&](const Vector3fP& row, size_t srcParam) {
              const FloatP jcx = row.x() * wgt;
              const FloatP jcy = row.y() * wgt;
              const FloatP jcz = row.z() * wgt;
              jacobian_jointParams_to_modelParams(
                  jcx,
                  srcParam,
                  parameterTransform_,
                  jacobian.middleRows(jacobianOffsetCur + kSimdPacketSize * 0, kSimdPacketSize));
              jacobian_jointParams_to_modelParams(
                  jcy,
                  srcParam,
                  parameterTransform_,
                  jacobian.middleRows(jacobianOffsetCur + kSimdPacketSize * 1, kSimdPacketSize));
              jacobian_jointParams_to_modelParams(
                  jcz,
                  srcParam,
                  parameterTransform_,
                  jacobian.middleRows(jacobianOffsetCur + kSimdPacketSize * 2, kSimdPacketSize));
            };

            for (size_t d = 0; d < 3; ++d) {
              // Translation only via point.
              if (this->activeJointParams_[paramIndex + d]) {
                const Vector3f& axis = jointState.getTranslationDerivative(d);
                const Vector3fP dPoint{FloatP(axis.x()), FloatP(axis.y()), FloatP(axis.z())};
                Vector3fP row;
                applyDfdvPoint(dPoint, row);
                storeJacRow(row, paramIndex + d);
              }
              // Rotation via point + dir.
              if (this->activeJointParams_[paramIndex + 3 + d]) {
                const Vector3f& rotAxis = jointState.rotationAxis.col(d);
                Vector3fP rowPoint;
                Vector3fP rowDir;
                applyDfdvPoint(momentum::cross(rotAxis, posdPoint), rowPoint);
                applyDfdvDir(momentum::cross(rotAxis, ev.dirWorld), rowDir);
                Vector3fP row{
                    rowPoint.x() + rowDir.x(),
                    rowPoint.y() + rowDir.y(),
                    rowPoint.z() + rowDir.z()};
                storeJacRow(row, paramIndex + d + 3);
              }
            }
            if (this->activeJointParams_[paramIndex + 6]) {
              // Scale only via point: d(point)/d(scale) = ln2 * posdPoint.
              const Vector3fP dPoint{
                  FloatP(ln2<float>()) * posdPoint.x(),
                  FloatP(ln2<float>()) * posdPoint.y(),
                  FloatP(ln2<float>()) * posdPoint.z()};
              Vector3fP row;
              applyDfdvPoint(dPoint, row);
              storeJacRow(row, paramIndex + 6);
            }

            jointIndex = this->skeleton_.joints[jointIndex].parent;
          }
        }

        jointErrors[jointId] += drjit::sum(jointError);
      },
      dispensoOptions);

  usedRows = gsl::narrow_cast<int>(jacobian.rows());

  const double error =
      kAimWeight * weight_ * std::accumulate(jointErrors.begin(), jointErrors.end(), 0.0);
  return static_cast<float>(error);
}

size_t SimdAimDistErrorFunction::getJacobianSize() const {
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
