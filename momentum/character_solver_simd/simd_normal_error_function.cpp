/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver_simd/simd_normal_error_function.h"

#include "momentum/character_solver_simd/simd_helpers.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/math/constants.h"
#include "momentum/simd/simd.h"

#include <dispenso/parallel_for.h>

#include <cstdlib>
#include <numeric>
#include <tuple>

#ifndef __vectorcall
#define __vectorcall
#endif

namespace momentum {

SimdNormalConstraints::SimdNormalConstraints(const Skeleton* skel) {
  // resize all arrays to the number of joints
  MT_CHECK(skel->joints.size() <= kMaxJoints);
  numJoints = static_cast<int>(skel->joints.size());
  const auto dataSize = kMaxConstraints * numJoints;
  MT_CHECK(dataSize % kSimdAlignment == 0);

  // create memory for all floats
  constexpr size_t nBlocks = 10;
  data = makeAlignedUnique<float, kSimdAlignment>(dataSize * nBlocks);
  float* dataPtr = data.get();
  std::fill_n(dataPtr, dataSize * nBlocks, 0.0f);

  offsetX = dataPtr + dataSize * 0;
  offsetY = dataPtr + dataSize * 1;
  offsetZ = dataPtr + dataSize * 2;
  normalX = dataPtr + dataSize * 3;
  normalY = dataPtr + dataSize * 4;
  normalZ = dataPtr + dataSize * 5;
  targetX = dataPtr + dataSize * 6;
  targetY = dataPtr + dataSize * 7;
  targetZ = dataPtr + dataSize * 8;
  weights = dataPtr + dataSize * 9;
}

SimdNormalConstraints::~SimdNormalConstraints() = default;

bool SimdNormalConstraints::addConstraint(
    const size_t jointIndex,
    const Vector3f& offset,
    const Vector3f& normal,
    const Vector3f& target,
    const float targetWeight) {
  MT_CHECK(jointIndex < constraintCount.size());

  // add the constraint to the corresponding arrays of the jointIndex if there's enough space
  uint32_t index = 0;
  while (true) {
    index = constraintCount[jointIndex];
    if (index == kMaxConstraints) {
      return false;
    }

    if (constraintCount[jointIndex].compare_exchange_weak(index, index + (uint32_t)1)) {
      break;
    }
  }

  const auto finalIndex = jointIndex * kMaxConstraints + index;
  offsetX[finalIndex] = offset.x();
  offsetY[finalIndex] = offset.y();
  offsetZ[finalIndex] = offset.z();
  normalX[finalIndex] = normal.x();
  normalY[finalIndex] = normal.y();
  normalZ[finalIndex] = normal.z();
  targetX[finalIndex] = target.x();
  targetY[finalIndex] = target.y();
  targetZ[finalIndex] = target.z();
  weights[finalIndex] = targetWeight;
  return true;
}

VectorXi SimdNormalConstraints::getNumConstraints() const {
  VectorXi res = VectorXi::Zero(numJoints);
  for (int jointIndex = 0; jointIndex < numJoints; ++jointIndex) {
    res[jointIndex] = constraintCount[jointIndex];
  }
  return res;
}

void SimdNormalConstraints::clearConstraints() {
  std::fill_n(weights, kMaxConstraints * numJoints, 0.0f);
  for (int i = 0; i < numJoints; i++) {
    constraintCount[i] = 0;
  }
}

SimdNormalErrorFunction::SimdNormalErrorFunction(
    const Skeleton& skel,
    const ParameterTransform& pt,
    uint32_t maxThreads)
    : SkeletonErrorFunction(skel, pt), maxThreads_(maxThreads) {
  constraints_ = nullptr;
}

SimdNormalErrorFunction::SimdNormalErrorFunction(const Character& character, uint32_t maxThreads)
    : SimdNormalErrorFunction(character.skeleton, character.parameterTransform, maxThreads) {}

double SimdNormalErrorFunction::getError(
    const ModelParameters& /* params */,
    const SkeletonState& state,
    const MeshState& /* meshState */) {
  if (constraints_ == nullptr) {
    return 0.0f;
  }

  MT_CHECK((size_t)constraints_->numJoints <= state.jointState.size());

  // loop over all joints, as these are our base units
  auto error = drjit::zeros<DoubleP>(); // use double to prevent rounding errors
  for (int jointId = 0; jointId < constraints_->numJoints; jointId++) {
    const size_t constraintCount = constraints_->constraintCount[jointId];
    MT_CHECK(jointId < static_cast<int>(state.jointState.size()));
    // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
    const auto& jointState = state.jointState[jointId];
    const Eigen::Matrix3f jointRotMat = jointState.rotation().toRotationMatrix();

    for (uint32_t index = 0; index < constraintCount; index += kSimdPacketSize) {
      const auto constraintOffsetIndex = jointId * SimdNormalConstraints::kMaxConstraints + index;
      const Vector3fP target{
          drjit::load<FloatP>(&constraints_->targetX[constraintOffsetIndex]),
          drjit::load<FloatP>(&constraints_->targetY[constraintOffsetIndex]),
          drjit::load<FloatP>(&constraints_->targetZ[constraintOffsetIndex])};

      const Vector3fP offset{
          drjit::load<FloatP>(&constraints_->offsetX[constraintOffsetIndex]),
          drjit::load<FloatP>(&constraints_->offsetY[constraintOffsetIndex]),
          drjit::load<FloatP>(&constraints_->offsetZ[constraintOffsetIndex])};
      const Vector3fP pos_world = jointState.transform * offset;

      const Vector3fP normal{
          drjit::load<FloatP>(&constraints_->normalX[constraintOffsetIndex]),
          drjit::load<FloatP>(&constraints_->normalY[constraintOffsetIndex]),
          drjit::load<FloatP>(&constraints_->normalZ[constraintOffsetIndex])};
      const Vector3fP normal_world = momentum::operator*(jointRotMat, normal);

      const FloatP dist = dot(pos_world - target, normal_world);

      const auto weight = drjit::load<FloatP>(&constraints_->weights[constraintOffsetIndex]);
      error += weight * drjit::square(dist);
    }
  }

  // sum the SIMD error register for final result
  return static_cast<float>(drjit::sum(error) * kPlaneWeight * weight_);
}

double SimdNormalErrorFunction::getGradient(
    const ModelParameters& /*params*/,
    const SkeletonState& state,
    const MeshState& /* meshState */,
    Ref<VectorXf> gradient) {
  if (constraints_ == nullptr) {
    return 0.0f;
  }

  auto error = drjit::zeros<DoubleP>(); // use double to prevent rounding errors
  for (int jointId = 0; jointId < constraints_->numJoints; jointId++) {
    const size_t constraintCount = constraints_->constraintCount[jointId];
    if (jointId >= static_cast<int>(state.jointState.size())) {
      break;
    }
    const Eigen::Matrix3f jointRotMat = state.jointState[jointId].rotation().toRotationMatrix();
    const auto& jointState_cons = state.jointState[jointId];
    for (uint32_t index = 0; index < constraintCount; index += kSimdPacketSize) {
      const auto constraintOffsetIndex = jointId * SimdNormalConstraints::kMaxConstraints + index;

      const Vector3fP offset{
          drjit::load<FloatP>(&constraints_->offsetX[constraintOffsetIndex]),
          drjit::load<FloatP>(&constraints_->offsetY[constraintOffsetIndex]),
          drjit::load<FloatP>(&constraints_->offsetZ[constraintOffsetIndex])};
      const Vector3fP pos_world = jointState_cons.transform * offset;

      const Vector3fP normal{
          drjit::load<FloatP>(&constraints_->normalX[constraintOffsetIndex]),
          drjit::load<FloatP>(&constraints_->normalY[constraintOffsetIndex]),
          drjit::load<FloatP>(&constraints_->normalZ[constraintOffsetIndex])};
      const Vector3fP normal_world = momentum::operator*(jointRotMat, normal);

      const Vector3fP target{
          drjit::load<FloatP>(&constraints_->targetX[constraintOffsetIndex]),
          drjit::load<FloatP>(&constraints_->targetY[constraintOffsetIndex]),
          drjit::load<FloatP>(&constraints_->targetZ[constraintOffsetIndex])};
      const FloatP dist = dot(pos_world - target, normal_world);

      const auto weight = drjit::load<FloatP>(&constraints_->weights[constraintOffsetIndex]);
      const FloatP wgt = weight * (2.0f * kPlaneWeight);

      error += weight * drjit::square(dist);

      // loop over all joints the constraint is attached to and calculate gradient
      size_t jointIndex = jointId;
      while (jointIndex != kInvalidIndex) {
        // check for valid index
        const auto& jointState = state.jointState[jointIndex];

        const size_t paramIndex = jointIndex * kParametersPerJoint;
        const Vector3fP posd = momentum::operator-(pos_world, jointState.translation());

        // calculate derivatives based on active joints
        for (size_t d = 0; d < 3; d++) {
          if (this->activeJointParams_[paramIndex + d]) {
            // calculate joint gradient
            const FloatP val =
                dist * momentum::dot(normal_world, jointState.getTranslationDerivative(d)) * wgt;
            // explicitly multiply with the parameter transform to generate parameter space
            // gradients
            for (auto ptIndex = parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
                 ptIndex < parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
                 ++ptIndex) {
              gradient[parameterTransform_.transform.innerIndexPtr()[ptIndex]] +=
                  drjit::sum(val) * parameterTransform_.transform.valuePtr()[ptIndex];
            }
          }
          if (this->activeJointParams_[paramIndex + 3 + d]) {
            // calculate joint gradient
            const auto crossProd =
                cross(normal_world, momentum::operator-(target, jointState.translation()));
            const FloatP val =
                -dist * momentum::dot(jointState.rotationAxis.col(d), crossProd) * wgt;
            // explicitly multiply with the parameter transform to generate parameter space
            // gradients
            for (auto ptIndex = parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 3];
                 ptIndex < parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 3 + 1];
                 ++ptIndex) {
              gradient[parameterTransform_.transform.innerIndexPtr()[ptIndex]] +=
                  drjit::sum(val) * parameterTransform_.transform.valuePtr()[ptIndex];
            }
          }
        }
        if (this->activeJointParams_[paramIndex + 6]) {
          // calculate joint gradient
          const FloatP val = dist * ln2() * dot(normal_world, posd) * wgt;
          // explicitly multiply with the parameter transform to generate parameter space gradients
          for (auto ptIndex = parameterTransform_.transform.outerIndexPtr()[paramIndex + 6];
               ptIndex < parameterTransform_.transform.outerIndexPtr()[paramIndex + 6 + 1];
               ++ptIndex) {
            gradient[parameterTransform_.transform.innerIndexPtr()[ptIndex]] +=
                drjit::sum(val) * parameterTransform_.transform.valuePtr()[ptIndex];
          }
        }

        // go to the next joint
        jointIndex = this->skeleton_.joints[jointIndex].parent;
      }
    }
  }

  return static_cast<float>(drjit::sum(error) * kPlaneWeight * weight_);
}

double SimdNormalErrorFunction::getJacobian(
    const ModelParameters& /*params*/,
    const SkeletonState& state,
    const MeshState& /* meshState */,
    Ref<MatrixXf> jacobian,
    Ref<VectorXf> residual,
    int& usedRows) {
  MT_PROFILE_FUNCTION();
  MT_CHECK(jacobian.cols() == static_cast<Eigen::Index>(parameterTransform_.transform.cols()));

  if (constraints_ == nullptr) {
    return 0.0f;
  }

  MT_CHECK((size_t)constraints_->numJoints <= jacobianOffset_.size());

  // storage for joint errors
  std::vector<double> jointErrors(constraints_->numJoints);

  // loop over all joints, as these are our base units
  auto dispensoOptions = dispenso::ParForOptions();
  dispensoOptions.maxThreads = maxThreads_;
  dispenso::parallel_for(
      0,
      constraints_->numJoints,
      [&](const size_t jointId) {
        const size_t constraintCount = constraints_->constraintCount[jointId];
        const auto& jointState_cons = state.jointState[jointId];
        // Precompute the linear (rotation*scale) and translation parts of the joint transform
        // ONCE per joint. The TransformT::operator*() recomputes toLinear() on every call,
        // which would otherwise rebuild the 3x3 matrix per constraint packet.
        const Eigen::Matrix3f jointLinear = jointState_cons.transform.toLinear();
        const Eigen::Vector3f jointTrans = jointState_cons.transform.translation;
        const Eigen::Matrix3f jointRotMat = jointState_cons.rotation().toRotationMatrix();
        auto jointError = drjit::zeros<DoubleP>(); // use double to prevent rounding errors

        for (uint32_t index = 0; index < constraintCount; index += kSimdPacketSize) {
          const auto constraintOffsetIndex =
              jointId * SimdNormalConstraints::kMaxConstraints + index;
          const auto jacobianOffsetCur = jacobianOffset_[jointId] + index;

          const Vector3fP offset{
              drjit::load<FloatP>(&constraints_->offsetX[constraintOffsetIndex]),
              drjit::load<FloatP>(&constraints_->offsetY[constraintOffsetIndex]),
              drjit::load<FloatP>(&constraints_->offsetZ[constraintOffsetIndex])};
          const Vector3fP pos_world =
              momentum::operator+(momentum::operator*(jointLinear, offset), jointTrans);

          const Vector3fP normal{
              drjit::load<FloatP>(&constraints_->normalX[constraintOffsetIndex]),
              drjit::load<FloatP>(&constraints_->normalY[constraintOffsetIndex]),
              drjit::load<FloatP>(&constraints_->normalZ[constraintOffsetIndex])};
          const Vector3fP normal_world = momentum::operator*(jointRotMat, normal);

          const Vector3fP target{
              drjit::load<FloatP>(&constraints_->targetX[constraintOffsetIndex]),
              drjit::load<FloatP>(&constraints_->targetY[constraintOffsetIndex]),
              drjit::load<FloatP>(&constraints_->targetZ[constraintOffsetIndex])};
          const FloatP dist = dot(pos_world - target, normal_world);

          const auto weight = drjit::load<FloatP>(&constraints_->weights[constraintOffsetIndex]);
          const FloatP wgt = drjit::sqrt(weight * kPlaneWeight * this->weight_);

          jointError += weight * drjit::square(dist);

          drjit::store(residual.segment(jacobianOffsetCur, kSimdPacketSize).data(), dist * wgt);

          // loop over all joints the constraint is attached to and calculate gradient
          size_t jointIndex = jointId;
          while (jointIndex != kInvalidIndex) {
            // check for valid index
            const auto& jointState = state.jointState[jointIndex];
            const size_t paramIndex = jointIndex * kParametersPerJoint;
            // Cache the chain joint's translation once: the loops below would otherwise recompute
            // it for every translation/rotation/scale derivative.
            const Eigen::Vector3f chainTrans = jointState.translation();
            // d-independent: cross product of normal_world with (target - chainTrans).
            // Hoisting out of the rotation d-loop saves up to 3x recomputation per chain step.
            const Vector3fP tgtd = momentum::operator-(target, chainTrans);
            const Vector3fP normCrossTgtd = cross(normal_world, tgtd);

            // calculate derivatives based on active joints
            for (size_t d = 0; d < 3; d++) {
              if (this->activeJointParams_[paramIndex + d]) {
                const FloatP jc =
                    momentum::dot(normal_world, jointState.getTranslationDerivative(d)) * wgt;
                jacobian_jointParams_to_modelParams(
                    jc,
                    paramIndex + d,
                    parameterTransform_,
                    jacobian.middleRows(jacobianOffsetCur, kSimdPacketSize));
              }
            }

            for (size_t d = 0; d < 3; ++d) {
              if (this->activeJointParams_[paramIndex + 3 + d]) {
                const Vector3f axis = jointState.rotationAxis.col(d);
                const FloatP jc = -momentum::dot(axis, normCrossTgtd) * wgt;
                jacobian_jointParams_to_modelParams(
                    jc,
                    paramIndex + d + 3,
                    parameterTransform_,
                    jacobian.middleRows(jacobianOffsetCur, kSimdPacketSize));
              }
            }

            if (this->activeJointParams_[paramIndex + 6]) {
              const Vector3fP posd = momentum::operator-(pos_world, chainTrans);
              const FloatP jc = dot(normal_world, posd) * (ln2() * wgt);
              jacobian_jointParams_to_modelParams(
                  jc,
                  paramIndex + 6,
                  parameterTransform_,
                  jacobian.middleRows(jacobianOffsetCur, kSimdPacketSize));
            }

            // go to the next joint
            jointIndex = this->skeleton_.joints[jointIndex].parent;
          }
        }

        jointErrors[jointId] += drjit::sum(jointError);
      },
      dispensoOptions);

  double error =
      kPlaneWeight * weight_ * std::accumulate(jointErrors.begin(), jointErrors.end(), 0.0);

  usedRows = gsl::narrow_cast<int>(jacobian.rows());

  // sum the SIMD error register for final result
  return static_cast<float>(error);
}

size_t SimdNormalErrorFunction::getJacobianSize() const {
  if (constraints_ == nullptr) {
    return 0;
  }

  jacobianOffset_.resize(constraints_->numJoints);
  size_t num = 0;
  for (int i = 0; i < constraints_->numJoints; i++) {
    jacobianOffset_[i] = num;
    const size_t count = constraints_->constraintCount[i];
    const auto numPackets = (count + kSimdPacketSize - 1) / kSimdPacketSize;
    num += numPackets * kSimdPacketSize;
  }
  return num;
}

} // namespace momentum
