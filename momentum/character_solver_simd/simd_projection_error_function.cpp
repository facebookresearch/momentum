/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver_simd/simd_projection_error_function.h"

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

SimdProjectionConstraints::SimdProjectionConstraints(const Skeleton* skel) {
  MT_CHECK(skel->joints.size() <= kMaxJoints);
  numJoints = static_cast<int>(skel->joints.size());
  const auto dataSize = kMaxConstraints * numJoints;
  MT_CHECK(dataSize % kSimdAlignment == 0);

  // 18 striped float blocks: offset (3), projection (12), target (2), weight (1).
  constexpr size_t nBlocks = 18;
  data = makeAlignedUnique<float, kSimdAlignment>(dataSize * nBlocks);
  float* dataPtr = data.get();
  std::fill_n(dataPtr, dataSize * nBlocks, 0.0f);

  offsetX = dataPtr + dataSize * 0;
  offsetY = dataPtr + dataSize * 1;
  offsetZ = dataPtr + dataSize * 2;
  for (size_t i = 0; i < 12; ++i) {
    projection[i] = dataPtr + dataSize * (3 + i);
  }
  targetU = dataPtr + dataSize * 15;
  targetV = dataPtr + dataSize * 16;
  weights = dataPtr + dataSize * 17;

  clearConstraints();
}

SimdProjectionConstraints::~SimdProjectionConstraints() = default;

void SimdProjectionConstraints::addConstraint(
    const size_t jointIndex,
    const Vector3f& offset,
    const Eigen::Matrix<float, 3, 4>& proj,
    const Eigen::Vector2f& target,
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
  // Store projection in row-major order to match (row, col) iteration in the inner loop.
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 4; ++col) {
      projection[row * 4 + col][finalIndex] = proj(row, col);
    }
  }
  targetU[finalIndex] = target.x();
  targetV[finalIndex] = target.y();
  weights[finalIndex] = targetWeight;
}

VectorXi SimdProjectionConstraints::getNumConstraints() const {
  VectorXi res = VectorXi::Zero(numJoints);
  for (int jointIndex = 0; jointIndex < numJoints; ++jointIndex) {
    res[jointIndex] = constraintCount[jointIndex];
  }
  return res;
}

void SimdProjectionConstraints::clearConstraints() {
  std::fill_n(weights, kMaxConstraints * numJoints, 0.0f);
  std::fill_n(constraintCount.begin(), numJoints, 0);
}

SimdProjectionErrorFunction::SimdProjectionErrorFunction(
    const Skeleton& skel,
    const ParameterTransform& pt,
    float nearClip,
    uint32_t maxThreads)
    : SkeletonErrorFunction(skel, pt),
      constraints_(nullptr),
      nearClip_(nearClip),
      maxThreads_(maxThreads) {}

SimdProjectionErrorFunction::SimdProjectionErrorFunction(
    const Character& character,
    float nearClip,
    uint32_t maxThreads)
    : SimdProjectionErrorFunction(
          character.skeleton,
          character.parameterTransform,
          nearClip,
          maxThreads) {}

namespace {

/// Holds intermediate per-packet projection state; used by all three accessors.
struct ProjEval {
  Vector3fP pWorld;
  Vector3fP pProj; // (x, y, z) in projected camera space
  FloatP px2d; // x/z
  FloatP py2d; // y/z
  FloatP resU; // px2d - targetU
  FloatP resV; // py2d - targetV
  FloatP invZ;
  FloatP xz2; // x / (z*z)
  FloatP yz2; // y / (z*z)
  drjit::mask_t<FloatP> valid; // p_proj.z >= nearClip
};

template <typename JointStateT>
inline ProjEval evaluateProjection(
    const JointStateT& jointState,
    const SimdProjectionConstraints& cs,
    size_t off,
    float nearClip) {
  ProjEval ev;
  const Vector3fP offsetVec{
      drjit::load<FloatP>(&cs.offsetX[off]),
      drjit::load<FloatP>(&cs.offsetY[off]),
      drjit::load<FloatP>(&cs.offsetZ[off])};
  ev.pWorld = jointState.transform * offsetVec;

  // p_proj = projection * [pWorld; 1]
  // px = m00*x + m01*y + m02*z + m03; py = ...; pz = ...
  ev.pProj = Vector3fP{
      drjit::load<FloatP>(&cs.projection[0][off]) * ev.pWorld.x() +
          drjit::load<FloatP>(&cs.projection[1][off]) * ev.pWorld.y() +
          drjit::load<FloatP>(&cs.projection[2][off]) * ev.pWorld.z() +
          drjit::load<FloatP>(&cs.projection[3][off]),
      drjit::load<FloatP>(&cs.projection[4][off]) * ev.pWorld.x() +
          drjit::load<FloatP>(&cs.projection[5][off]) * ev.pWorld.y() +
          drjit::load<FloatP>(&cs.projection[6][off]) * ev.pWorld.z() +
          drjit::load<FloatP>(&cs.projection[7][off]),
      drjit::load<FloatP>(&cs.projection[8][off]) * ev.pWorld.x() +
          drjit::load<FloatP>(&cs.projection[9][off]) * ev.pWorld.y() +
          drjit::load<FloatP>(&cs.projection[10][off]) * ev.pWorld.z() +
          drjit::load<FloatP>(&cs.projection[11][off])};

  ev.valid = ev.pProj.z() >= FloatP(nearClip);
  // Use a safe denominator on invalid lanes to avoid 1/0; mask zero out their contributions later.
  const FloatP safeZ = drjit::select(ev.valid, ev.pProj.z(), FloatP(1.0f));
  ev.invZ = 1.0f / safeZ;
  const FloatP invZSqr = ev.invZ * ev.invZ;
  ev.px2d = ev.pProj.x() * ev.invZ;
  ev.py2d = ev.pProj.y() * ev.invZ;
  ev.xz2 = ev.pProj.x() * invZSqr;
  ev.yz2 = ev.pProj.y() * invZSqr;
  ev.resU = ev.px2d - drjit::load<FloatP>(&cs.targetU[off]);
  ev.resV = ev.py2d - drjit::load<FloatP>(&cs.targetV[off]);
  return ev;
}

} // namespace

double SimdProjectionErrorFunction::getError(
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
      const auto offsetIdx = jointId * SimdProjectionConstraints::kMaxConstraints + index;
      const ProjEval ev = evaluateProjection(jointState, *constraints_, offsetIdx, nearClip_);

      const auto weight = drjit::load<FloatP>(&constraints_->weights[offsetIdx]);
      const FloatP sqrErr = ev.resU * ev.resU + ev.resV * ev.resV;
      // Mask invalid lanes (behind near clip) to zero contribution.
      const FloatP maskedErr = drjit::select(ev.valid, weight * sqrErr, FloatP(0.0f));
      error += maskedErr;
    }
  }

  return static_cast<float>(drjit::sum(error) * kProjectionWeight * weight_);
}

namespace {

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

double SimdProjectionErrorFunction::getGradient(
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
          const auto offsetIdx = jointId * SimdProjectionConstraints::kMaxConstraints + index;
          const ProjEval ev =
              evaluateProjection(jointState_cons, *constraints_, offsetIdx, nearClip_);

          const auto weight = drjit::load<FloatP>(&constraints_->weights[offsetIdx]);
          const FloatP sqrErr = ev.resU * ev.resU + ev.resV * ev.resV;
          const FloatP zeroMask = drjit::select(ev.valid, FloatP(1.0f), FloatP(0.0f));
          jointError += weight * sqrErr * zeroMask;

          // Per-lane gradient contribution scale.
          const FloatP wgt = 2.0f * kProjectionWeight * weight * zeroMask;

          // Helper: compute (d_p_res . p_res) for a 3D world derivative dPoint.
          auto gradContrib = [&](const Vector3fP& dPoint) {
            // d_p_proj = projection.topLeftCorner<3,3>() * dPoint
            const FloatP dx =
                drjit::load<FloatP>(&constraints_->projection[0][offsetIdx]) * dPoint.x() +
                drjit::load<FloatP>(&constraints_->projection[1][offsetIdx]) * dPoint.y() +
                drjit::load<FloatP>(&constraints_->projection[2][offsetIdx]) * dPoint.z();
            const FloatP dy =
                drjit::load<FloatP>(&constraints_->projection[4][offsetIdx]) * dPoint.x() +
                drjit::load<FloatP>(&constraints_->projection[5][offsetIdx]) * dPoint.y() +
                drjit::load<FloatP>(&constraints_->projection[6][offsetIdx]) * dPoint.z();
            const FloatP dz =
                drjit::load<FloatP>(&constraints_->projection[8][offsetIdx]) * dPoint.x() +
                drjit::load<FloatP>(&constraints_->projection[9][offsetIdx]) * dPoint.y() +
                drjit::load<FloatP>(&constraints_->projection[10][offsetIdx]) * dPoint.z();

            // d_p_res_u = dx/z - x/z^2 * dz; d_p_res_v = dy/z - y/z^2 * dz.
            const FloatP dResU = dx * ev.invZ - ev.xz2 * dz;
            const FloatP dResV = dy * ev.invZ - ev.yz2 * dz;
            // Inner product with the residual.
            return dResU * ev.resU + dResV * ev.resV;
          };

          size_t jointIndex = jointId;
          while (jointIndex != kInvalidIndex) {
            MT_CHECK(jointIndex < static_cast<size_t>(constraints_->numJoints));

            const auto& jointState = state.jointState[jointIndex];
            const size_t paramIndex = jointIndex * kParametersPerJoint;
            const Vector3fP posd = momentum::operator-(ev.pWorld, jointState.translation());

            for (size_t d = 0; d < 3; ++d) {
              if (this->activeJointParams_[paramIndex + d]) {
                const Vector3f& axis = jointState.getTranslationDerivative(d);
                const Vector3fP dPoint{FloatP(axis.x()), FloatP(axis.y()), FloatP(axis.z())};
                scatter1d(
                    wgt * gradContrib(dPoint), paramIndex + d, parameterTransform_, grad_local);
              }
              if (this->activeJointParams_[paramIndex + 3 + d]) {
                const Vector3fP dPoint = momentum::cross(jointState.rotationAxis.col(d), posd);
                scatter1d(
                    wgt * gradContrib(dPoint), paramIndex + d + 3, parameterTransform_, grad_local);
              }
            }
            if (this->activeJointParams_[paramIndex + 6]) {
              const Vector3fP dPoint{
                  FloatP(ln2<float>()) * posd.x(),
                  FloatP(ln2<float>()) * posd.y(),
                  FloatP(ln2<float>()) * posd.z()};
              scatter1d(wgt * gradContrib(dPoint), paramIndex + 6, parameterTransform_, grad_local);
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

  return static_cast<float>(kProjectionWeight * weight_ * error);
}

double SimdProjectionErrorFunction::getJacobian(
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

        for (uint32_t index = 0, jindex = 0; index < constraintCount;
             index += kSimdPacketSize, jindex += kSimdPacketSize * kConstraintDim) {
          const auto offsetIdx = jointId * SimdProjectionConstraints::kMaxConstraints + index;
          const auto jacobianOffsetCur = jacobianOffset_[jointId] + jindex;
          const ProjEval ev =
              evaluateProjection(jointState_cons, *constraints_, offsetIdx, nearClip_);

          const auto weight = drjit::load<FloatP>(&constraints_->weights[offsetIdx]);
          const FloatP sqrErr = ev.resU * ev.resU + ev.resV * ev.resV;
          const FloatP zeroMask = drjit::select(ev.valid, FloatP(1.0f), FloatP(0.0f));
          jointError += weight * sqrErr * zeroMask;

          // sqrt-weight residual; masked to zero on invalid lanes.
          const FloatP wgt = drjit::sqrt(kProjectionWeight * weight_ * weight) * zeroMask;
          const FloatP scaledU = wgt * ev.resU;
          const FloatP scaledV = wgt * ev.resV;
          drjit::store(
              residual.segment(jacobianOffsetCur + kSimdPacketSize * 0, kSimdPacketSize).data(),
              scaledU);
          drjit::store(
              residual.segment(jacobianOffsetCur + kSimdPacketSize * 1, kSimdPacketSize).data(),
              scaledV);

          // For each derivative in world space, project to (d_res_u, d_res_v).
          auto compute2DDeriv = [&](const Vector3fP& dPoint, FloatP& outU, FloatP& outV) {
            const FloatP dx =
                drjit::load<FloatP>(&constraints_->projection[0][offsetIdx]) * dPoint.x() +
                drjit::load<FloatP>(&constraints_->projection[1][offsetIdx]) * dPoint.y() +
                drjit::load<FloatP>(&constraints_->projection[2][offsetIdx]) * dPoint.z();
            const FloatP dy =
                drjit::load<FloatP>(&constraints_->projection[4][offsetIdx]) * dPoint.x() +
                drjit::load<FloatP>(&constraints_->projection[5][offsetIdx]) * dPoint.y() +
                drjit::load<FloatP>(&constraints_->projection[6][offsetIdx]) * dPoint.z();
            const FloatP dz =
                drjit::load<FloatP>(&constraints_->projection[8][offsetIdx]) * dPoint.x() +
                drjit::load<FloatP>(&constraints_->projection[9][offsetIdx]) * dPoint.y() +
                drjit::load<FloatP>(&constraints_->projection[10][offsetIdx]) * dPoint.z();

            outU = (dx * ev.invZ - ev.xz2 * dz) * wgt;
            outV = (dy * ev.invZ - ev.yz2 * dz) * wgt;
          };

          size_t jointIndex = jointId;
          while (jointIndex != kInvalidIndex) {
            MT_CHECK(jointIndex < static_cast<size_t>(constraints_->numJoints));

            const auto& jointState = state.jointState[jointIndex];
            const size_t paramIndex = jointIndex * kParametersPerJoint;
            const Vector3fP posd = momentum::operator-(ev.pWorld, jointState.translation());

            auto storeRow = [&](FloatP jcU, FloatP jcV, size_t srcParam) {
              jacobian_jointParams_to_modelParams(
                  jcU,
                  srcParam,
                  parameterTransform_,
                  jacobian.middleRows(jacobianOffsetCur + kSimdPacketSize * 0, kSimdPacketSize));
              jacobian_jointParams_to_modelParams(
                  jcV,
                  srcParam,
                  parameterTransform_,
                  jacobian.middleRows(jacobianOffsetCur + kSimdPacketSize * 1, kSimdPacketSize));
            };

            for (size_t d = 0; d < 3; ++d) {
              if (this->activeJointParams_[paramIndex + d]) {
                const Vector3f& axis = jointState.getTranslationDerivative(d);
                const Vector3fP dPoint{FloatP(axis.x()), FloatP(axis.y()), FloatP(axis.z())};
                FloatP jcU;
                FloatP jcV;
                compute2DDeriv(dPoint, jcU, jcV);
                storeRow(jcU, jcV, paramIndex + d);
              }
              if (this->activeJointParams_[paramIndex + 3 + d]) {
                const Vector3fP dPoint = momentum::cross(jointState.rotationAxis.col(d), posd);
                FloatP jcU;
                FloatP jcV;
                compute2DDeriv(dPoint, jcU, jcV);
                storeRow(jcU, jcV, paramIndex + d + 3);
              }
            }
            if (this->activeJointParams_[paramIndex + 6]) {
              const Vector3fP dPoint{
                  FloatP(ln2<float>()) * posd.x(),
                  FloatP(ln2<float>()) * posd.y(),
                  FloatP(ln2<float>()) * posd.z()};
              FloatP jcU;
              FloatP jcV;
              compute2DDeriv(dPoint, jcU, jcV);
              storeRow(jcU, jcV, paramIndex + 6);
            }

            jointIndex = this->skeleton_.joints[jointIndex].parent;
          }
        }

        jointErrors[jointId] += drjit::sum(jointError);
      },
      dispensoOptions);

  usedRows = gsl::narrow_cast<int>(jacobian.rows());

  const double error =
      kProjectionWeight * weight_ * std::accumulate(jointErrors.begin(), jointErrors.end(), 0.0);
  return static_cast<float>(error);
}

size_t SimdProjectionErrorFunction::getJacobianSize() const {
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
