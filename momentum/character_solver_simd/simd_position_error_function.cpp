/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver_simd/simd_position_error_function.h"

#include "momentum/character_solver_simd/simd_helpers.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/math/constants.h"
#include "momentum/simd/simd.h"

#include <dispenso/parallel_for.h>

#include <numeric>
#include <tuple>

#ifndef __vectorcall
#define __vectorcall
#endif

namespace momentum {

SimdPositionConstraints::SimdPositionConstraints(const Skeleton* skel) {
  // resize all arrays to the number of joints
  MT_CHECK(skel->joints.size() <= kMaxJoints);
  numJoints = static_cast<int>(skel->joints.size());
  const auto dataSize = kMaxConstraints * numJoints;
  MT_CHECK(dataSize % kSimdAlignment == 0);

  // create memory for all floats
  constexpr size_t nBlocks = 7;
  data = makeAlignedUnique<float, kSimdAlignment>(dataSize * nBlocks);
  float* dataPtr = data.get();
  std::fill_n(dataPtr, dataSize * nBlocks, 0.0f);

  offsetX = dataPtr + dataSize * 0;
  offsetY = dataPtr + dataSize * 1;
  offsetZ = dataPtr + dataSize * 2;
  targetX = dataPtr + dataSize * 3;
  targetY = dataPtr + dataSize * 4;
  targetZ = dataPtr + dataSize * 5;
  weights = dataPtr + dataSize * 6;

  // makes sure the constraintCount is initialized to zero
  clearConstraints();
}

SimdPositionConstraints::~SimdPositionConstraints() = default;

void SimdPositionConstraints::addConstraint(
    const size_t jointIndex,
    const Vector3f& offset,
    const Vector3f& target,
    const float targetWeight) {
  MT_CHECK(jointIndex < constraintCount.size());

  // add the constraint to the corresponding arrays of the jointIndex if there's enough space
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
  targetX[finalIndex] = target.x();
  targetY[finalIndex] = target.y();
  targetZ[finalIndex] = target.z();
  weights[finalIndex] = targetWeight;
}

VectorXi SimdPositionConstraints::getNumConstraints() const {
  VectorXi res = VectorXi::Zero(numJoints);
  for (int jointIndex = 0; jointIndex < numJoints; ++jointIndex) {
    res[jointIndex] = constraintCount[jointIndex];
  }
  return res;
}

void SimdPositionConstraints::clearConstraints() {
  std::fill_n(weights, kMaxConstraints * numJoints, 0.0f);
  std::fill_n(constraintCount.begin(), numJoints, 0);
}

SimdPositionErrorFunction::SimdPositionErrorFunction(
    const Skeleton& skel,
    const ParameterTransform& pt,
    uint32_t maxThreads)
    : SkeletonErrorFunction(skel, pt), maxThreads_(maxThreads) {
  constraints_ = nullptr;
}

SimdPositionErrorFunction::SimdPositionErrorFunction(
    const Character& character,
    uint32_t maxThreads)
    : SimdPositionErrorFunction(character.skeleton, character.parameterTransform, maxThreads) {
  // Do nothing
}

double SimdPositionErrorFunction::getError(
    const ModelParameters& /* params */,
    const SkeletonState& state,
    const MeshState& /* unused */) {
  if (constraints_ == nullptr) {
    return 0.0f;
  }

  MT_CHECK((size_t)constraints_->numJoints <= state.jointState.size());

  // Loop over all joints, as these are our base units
  auto error = drjit::zeros<DoubleP>(); // use double to prevent rounding errors
  for (int jointId = 0; jointId < constraints_->numJoints; jointId++) {
    const size_t constraintCount = constraints_->constraintCount[jointId];
    MT_CHECK(jointId < static_cast<int>(state.jointState.size()));
    // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
    const auto& jointState = state.jointState[jointId];

    // Loop over all constraints in increments of kSimdPacketSize
    for (uint32_t index = 0; index < constraintCount; index += kSimdPacketSize) {
      const auto constraintOffsetIndex = jointId * SimdPositionConstraints::kMaxConstraints + index;

      // Transform offset by joint transform: pos = transform * offset
      const Vector3fP offset{
          drjit::load<FloatP>(&constraints_->offsetX[constraintOffsetIndex]),
          drjit::load<FloatP>(&constraints_->offsetY[constraintOffsetIndex]),
          drjit::load<FloatP>(&constraints_->offsetZ[constraintOffsetIndex])};
      const Vector3fP pos_world = jointState.transform * offset;

      // Calculate distance of point to target: dist = pos - target
      const Vector3fP target{
          drjit::load<FloatP>(&constraints_->targetX[constraintOffsetIndex]),
          drjit::load<FloatP>(&constraints_->targetY[constraintOffsetIndex]),
          drjit::load<FloatP>(&constraints_->targetZ[constraintOffsetIndex])};
      const Vector3fP diff = pos_world - target;

      // Calculate square norm: dist = diff.squaredNorm()
      const FloatP dist = dot(diff, diff);

      // Calculate error as squared distance: err = constraintWeight * dist
      const auto constraintWeight =
          drjit::load<FloatP>(&constraints_->weights[constraintOffsetIndex]);
      error += constraintWeight * dist;
    }
  }

  // Sum the SIMD error register for final result
  return static_cast<float>(drjit::sum(error) * kPositionWeight * weight_);
}

double SimdPositionErrorFunction::getGradient(
    const ModelParameters& /* params */,
    const SkeletonState& state,
    const MeshState& /* unused */,
    Ref<VectorXf> gradient) {
  if (constraints_ == nullptr) {
    return 0.0f;
  }

  // Storage for per-thread error and gradient
  std::vector<std::tuple<double, VectorXd>> ets_error_grad;

  // Loop over all joints, as these are our base units
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
        auto jointError = drjit::zeros<DoubleP>(); // use double to prevent rounding errors

        // Loop over all constraints in increments of kSimdPacketSize
        for (uint32_t index = 0; index < constraintCount; index += kSimdPacketSize) {
          const auto constraintOffsetIndex =
              jointId * SimdPositionConstraints::kMaxConstraints + index;

          // Transform offset by joint transform: pos = transform * offset
          const Vector3fP offset{
              drjit::load<FloatP>(&constraints_->offsetX[constraintOffsetIndex]),
              drjit::load<FloatP>(&constraints_->offsetY[constraintOffsetIndex]),
              drjit::load<FloatP>(&constraints_->offsetZ[constraintOffsetIndex])};
          const Vector3fP pos_world = jointState_cons.transform * offset;

          // Calculate distance of point to target: dist = pos - target
          const Vector3fP target{
              drjit::load<FloatP>(&constraints_->targetX[constraintOffsetIndex]),
              drjit::load<FloatP>(&constraints_->targetY[constraintOffsetIndex]),
              drjit::load<FloatP>(&constraints_->targetZ[constraintOffsetIndex])};
          const Vector3fP diff = pos_world - target;

          // Calculate square norm: dist = diff.squaredNorm()
          const FloatP dist = dot(diff, diff);

          // Calculate error as squared distance: err = constraintWeight * dist
          const auto constraintWeight =
              drjit::load<FloatP>(&constraints_->weights[constraintOffsetIndex]);
          jointError += constraintWeight * dist;

          // Pre-calculate values for the gradient: wgt = 2.0 * kPositionWeight * constraintWeight
          const FloatP wgt = 2.0f * (kPositionWeight * constraintWeight);

          // Loop over all joints the constraint is attached to and calculate gradient
          size_t jointIndex = jointId;
          while (jointIndex != kInvalidIndex) {
            // Check for valid index
            MT_CHECK(jointIndex < static_cast<size_t>(constraints_->numJoints));

            const auto& jointState = state.jointState[jointIndex];
            const size_t paramIndex = jointIndex * kParametersPerJoint;

            // Calculate difference between constraint position and joint center: posd = pos -
            // jointState.translation
            const Vector3fP posd = momentum::operator-(pos_world, jointState.translation());

            // Calculate derivatives based on active joints
            for (size_t d = 0; d < 3; ++d) {
              if (this->activeJointParams_[paramIndex + d]) {
                // Calculate gradient with respect to joint: grad = diff.dot(axis) * wgt
                const FloatP val =
                    momentum::dot(diff, jointState.getTranslationDerivative(d)) * wgt;
                // Explicitly multiply with the parameter transform to generate parameter space
                // gradients
                for (auto ptIndex = parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
                     ptIndex < parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
                     ++ptIndex) {
                  grad_local[parameterTransform_.transform.innerIndexPtr()[ptIndex]] +=
                      drjit::sum(val) * parameterTransform_.transform.valuePtr()[ptIndex];
                }
              }
              if (this->activeJointParams_[paramIndex + 3 + d]) {
                // Calculate gradient with respect to joint: grad = diff.dot(axis.cross(posd)) * wgt
                const FloatP val =
                    dot(diff, momentum::cross(jointState.rotationAxis.col(d), posd)) * wgt;
                // explicitly multiply with the parameter transform to generate parameter space
                // gradients
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
              // Calculate gradient with respect to joint: grad = diff.dot(posd * LN2) * wgt
              const FloatP val = ln2<double>() * dot(diff, posd) * wgt;
              // Explicitly multiply with the parameter transform to generate parameter space
              // gradients
              for (auto ptIndex = parameterTransform_.transform.outerIndexPtr()[paramIndex + 6];
                   ptIndex < parameterTransform_.transform.outerIndexPtr()[paramIndex + 6 + 1];
                   ++ptIndex) {
                grad_local[parameterTransform_.transform.innerIndexPtr()[ptIndex]] +=
                    drjit::sum(val) * parameterTransform_.transform.valuePtr()[ptIndex];
              }
            }

            // Go to the next joint
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

    // finalize the gradient
    gradient += std::get<1>(ets_error_grad[0]).cast<float>() * weight_;
    error = std::get<0>(ets_error_grad[0]);
  }

  // Sum the SIMD error register for final result
  return static_cast<float>(kPositionWeight * weight_ * error);
}

double SimdPositionErrorFunction::getJacobian(
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

  // Storage for joint errors
  std::vector<double> jointErrors(constraints_->numJoints);

  // Loop over all joints, as these are our base units
  auto dispensoOptions = dispenso::ParForOptions();
  dispensoOptions.maxThreads = maxThreads_;
  dispenso::parallel_for(
      0,
      constraints_->numJoints,
      [&](const size_t jointId) {
        const size_t constraintCount = constraints_->constraintCount[jointId];
        const auto& jointState_cons = state.jointState[jointId];
        // Precompute transform's linear (rotation*scale) and translation parts ONCE per joint
        // to avoid TransformT::operator*() recomputing toLinear() per constraint packet.
        const Eigen::Matrix3f jointLinear = jointState_cons.transform.toLinear();
        const Eigen::Vector3f jointTrans = jointState_cons.transform.translation;
        auto jointError = drjit::zeros<DoubleP>(); // use double to prevent rounding errors

        // Loop over all constraints in increments of kSimdPacketSize
        for (uint32_t index = 0, jindex = 0; index < constraintCount;
             index += kSimdPacketSize, jindex += kSimdPacketSize * kConstraintDim) {
          const auto constraintOffsetIndex =
              jointId * SimdPositionConstraints::kMaxConstraints + index;
          const auto jacobianOffsetCur = jacobianOffset_[jointId] + jindex;

          // Transform offset by joint transform: pos = transform * offset
          const Vector3fP offset{
              drjit::load<FloatP>(&constraints_->offsetX[constraintOffsetIndex]),
              drjit::load<FloatP>(&constraints_->offsetY[constraintOffsetIndex]),
              drjit::load<FloatP>(&constraints_->offsetZ[constraintOffsetIndex])};
          const Vector3fP pos_world =
              momentum::operator+(momentum::operator*(jointLinear, offset), jointTrans);

          // Calculate distance of point to target: dist = pos - target
          const Vector3fP target{
              drjit::load<FloatP>(&constraints_->targetX[constraintOffsetIndex]),
              drjit::load<FloatP>(&constraints_->targetY[constraintOffsetIndex]),
              drjit::load<FloatP>(&constraints_->targetZ[constraintOffsetIndex])};
          const Vector3fP diff = pos_world - target;

          // Calculate square norm: dist = diff.squaredNorm()
          const FloatP dist = dot(diff, diff);

          // Calculate error as squared distance: err = constraintWeight * dist
          const auto constraintWeight =
              drjit::load<FloatP>(&constraints_->weights[constraintOffsetIndex]);
          jointError += constraintWeight * dist;

          // Calculate square-root of weight: wgt = sqrt(kPositionWeight * weight *
          // constraintWeight)
          const FloatP wgt = drjit::sqrt(kPositionWeight * weight_ * constraintWeight);

          // Calculate residual: res = wgt * diff
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

          // Loop over all joints the constraint is attached to and calculate gradient
          size_t jointIndex = jointId;
          while (jointIndex != kInvalidIndex) {
            // Check for valid index
            MT_CHECK(jointIndex < static_cast<size_t>(constraints_->numJoints));

            const auto& jointState = state.jointState[jointIndex];
            const size_t paramIndex = jointIndex * kParametersPerJoint;

            // Calculate difference between constraint position and joint center: posd = pos -
            // jointState.translation
            const Vector3fP posd = momentum::operator-(pos_world, jointState.translation());

            // Calculate derivatives based on active joints
            for (size_t d = 0; d < 3; ++d) {
              if (this->activeJointParams_[paramIndex + d]) {
                // Calculate Jacobian with respect to joint: jac = axis * wgt
                const Vector3f& axis = jointState.getTranslationDerivative(d);
                const FloatP jcx = axis.x() * wgt;
                const FloatP jcy = axis.y() * wgt;
                const FloatP jcz = axis.z() * wgt;
                jacobian_jointParams_to_modelParams(
                    jcx,
                    paramIndex + d,
                    parameterTransform_,
                    jacobian.middleRows(jacobianOffsetCur + kSimdPacketSize * 0, kSimdPacketSize));
                jacobian_jointParams_to_modelParams(
                    jcy,
                    paramIndex + d,
                    parameterTransform_,
                    jacobian.middleRows(jacobianOffsetCur + kSimdPacketSize * 1, kSimdPacketSize));
                jacobian_jointParams_to_modelParams(
                    jcz,
                    paramIndex + d,
                    parameterTransform_,
                    jacobian.middleRows(jacobianOffsetCur + kSimdPacketSize * 2, kSimdPacketSize));
              }

              if (this->activeJointParams_[paramIndex + 3 + d]) {
                // Calculate Jacobian with respect to joint: jac = axis.cross(posd) * wgt;
                const auto axisCrossPosd = momentum::cross(jointState.rotationAxis.col(d), posd);
                const FloatP jcx = axisCrossPosd.x() * wgt;
                const FloatP jcy = axisCrossPosd.y() * wgt;
                const FloatP jcz = axisCrossPosd.z() * wgt;
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

            if (this->activeJointParams_[paramIndex + 6]) {
              // Calculate Jacobian with respect to joint: jac = posd * wgt * LN2
              const FloatP jcx = posd.x() * (ln2<double>() * wgt);
              const FloatP jcy = posd.y() * (ln2<double>() * wgt);
              const FloatP jcz = posd.z() * (ln2<double>() * wgt);
              jacobian_jointParams_to_modelParams(
                  jcx,
                  paramIndex + 6,
                  parameterTransform_,
                  jacobian.middleRows(jacobianOffsetCur + kSimdPacketSize * 0, kSimdPacketSize));
              jacobian_jointParams_to_modelParams(
                  jcy,
                  paramIndex + 6,
                  parameterTransform_,
                  jacobian.middleRows(jacobianOffsetCur + kSimdPacketSize * 1, kSimdPacketSize));
              jacobian_jointParams_to_modelParams(
                  jcz,
                  paramIndex + 6,
                  parameterTransform_,
                  jacobian.middleRows(jacobianOffsetCur + kSimdPacketSize * 2, kSimdPacketSize));
            }

            // Go to the next joint
            jointIndex = this->skeleton_.joints[jointIndex].parent;
          }
        }

        jointErrors[jointId] += drjit::sum(jointError);
      },
      dispensoOptions);

  usedRows = gsl::narrow_cast<int>(jacobian.rows());

  // Sum the SIMD error register for final result
  const double error =
      kPositionWeight * weight_ * std::accumulate(jointErrors.begin(), jointErrors.end(), 0.0);
  return static_cast<float>(error);
}

size_t SimdPositionErrorFunction::getJacobianSize() const {
  if (constraints_ == nullptr) {
    return 0;
  }

  jacobianOffset_.resize(constraints_->numJoints);
  size_t num = 0;
  for (int i = 0; i < constraints_->numJoints; ++i) {
    jacobianOffset_[i] = num;
    const size_t count = constraints_->constraintCount[i];
    const size_t residual = count % kSimdPacketSize;
    if (residual != 0) {
      num += (count + kSimdPacketSize - residual) * kConstraintDim;
    } else {
      num += count * kConstraintDim;
    }
  }
  return num;
}

void SimdPositionErrorFunction::setConstraints(const SimdPositionConstraints* cstrs) {
  constraints_ = cstrs;
}
} // namespace momentum
