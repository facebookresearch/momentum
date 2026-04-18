/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver_simd/simd_plane_error_function.h"

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

SimdPlaneConstraints::SimdPlaneConstraints(const Skeleton* skel) {
  // resize all arrays to the number of joints
  MT_CHECK(skel->joints.size() <= kMaxJoints);
  numJoints = static_cast<int>(skel->joints.size());
  const auto dataSize = kMaxConstraints * numJoints;
  MT_CHECK(dataSize % kSimdAlignment == 0);

  // create memory for all floats
  constexpr size_t nBlocks = 8;
  data = makeAlignedUnique<float, kSimdAlignment>(dataSize * nBlocks);
  float* dataPtr = data.get();
  std::fill_n(dataPtr, dataSize * nBlocks, 0.0f);

  offsetX = dataPtr + dataSize * 0;
  offsetY = dataPtr + dataSize * 1;
  offsetZ = dataPtr + dataSize * 2;
  normalX = dataPtr + dataSize * 3;
  normalY = dataPtr + dataSize * 4;
  normalZ = dataPtr + dataSize * 5;
  targets = dataPtr + dataSize * 6;
  weights = dataPtr + dataSize * 7;

  // makes sure the constraintCount is initialized to zero
  clearConstraints();
}

SimdPlaneConstraints::~SimdPlaneConstraints() = default;

void SimdPlaneConstraints::addConstraint(
    const size_t jointIndex,
    const Vector3f& offset,
    const Vector3f& targetNormal,
    const float targetOffset,
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
  normalX[finalIndex] = targetNormal.x();
  normalY[finalIndex] = targetNormal.y();
  normalZ[finalIndex] = targetNormal.z();
  targets[finalIndex] = targetOffset;
  weights[finalIndex] = targetWeight;
}

VectorXi SimdPlaneConstraints::getNumConstraints() const {
  auto res = VectorXi(numJoints);
  for (int jointIndex = 0; jointIndex < numJoints; ++jointIndex) {
    res[jointIndex] = constraintCount[jointIndex];
  }
  return res;
}

void SimdPlaneConstraints::clearConstraints() {
  std::fill_n(weights, kMaxConstraints * numJoints, 0.0f);
  std::fill_n(constraintCount.begin(), numJoints, 0);
}

SimdPlaneErrorFunction::SimdPlaneErrorFunction(
    const Skeleton& skel,
    const ParameterTransform& pt,
    uint32_t maxThreads)
    : SkeletonErrorFunction(skel, pt), maxThreads_(maxThreads) {
  constraints_ = nullptr;
}

SimdPlaneErrorFunction::SimdPlaneErrorFunction(const Character& character, uint32_t maxThreads)
    : SimdPlaneErrorFunction(character.skeleton, character.parameterTransform, maxThreads) {
  // Do nothing
}

double SimdPlaneErrorFunction::getError(
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
    const Eigen::Matrix3f jointRotMat = jointState.rotation().toRotationMatrix();

    // Loop over all constraints in increments of kSimdPacketSize
    for (uint32_t index = 0; index < constraintCount; index += kSimdPacketSize) {
      const auto constraintOffsetIndex = jointId * SimdPlaneConstraints::kMaxConstraints + index;

      // Transform offset by joint transform: pos = transform * offset
      const Vector3fP offset{
          drjit::load<FloatP>(&constraints_->offsetX[constraintOffsetIndex]),
          drjit::load<FloatP>(&constraints_->offsetY[constraintOffsetIndex]),
          drjit::load<FloatP>(&constraints_->offsetZ[constraintOffsetIndex])};
      const Vector3fP pos_world = jointState.transform * offset;

      // Calculate distance of point to plane: dist = pos.dot(normal) - target
      const Vector3fP normal{
          drjit::load<FloatP>(&constraints_->normalX[constraintOffsetIndex]),
          drjit::load<FloatP>(&constraints_->normalY[constraintOffsetIndex]),
          drjit::load<FloatP>(&constraints_->normalZ[constraintOffsetIndex])};
      const Vector3fP normal_world = momentum::operator*(jointRotMat, normal);
      const auto target = drjit::load<FloatP>(&constraints_->targets[constraintOffsetIndex]);
      const FloatP dist = dot(pos_world, normal_world) - target;

      // Calculate error as squared distance: err = constraintWeight * dist * dist
      const auto constraintWeight =
          drjit::load<FloatP>(&constraints_->weights[constraintOffsetIndex]);
      error += constraintWeight * drjit::square(dist);
    }
  }

  // Sum the SIMD error register for final result
  return static_cast<float>(drjit::sum(error) * kPlaneWeight * weight_);
}

double SimdPlaneErrorFunction::getGradient(
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
        const Eigen::Matrix3f jointRotMat = jointState_cons.rotation().toRotationMatrix();
        auto jointError = drjit::zeros<DoubleP>(); // use double to prevent rounding errors

        // Loop over all constraints in increments of kSimdPacketSize
        for (uint32_t index = 0; index < constraintCount; index += kSimdPacketSize) {
          const auto constraintOffsetIndex =
              jointId * SimdPlaneConstraints::kMaxConstraints + index;

          // Transform offset by joint transform: pos = transform * offset
          const Vector3fP offset{
              drjit::load<FloatP>(&constraints_->offsetX[constraintOffsetIndex]),
              drjit::load<FloatP>(&constraints_->offsetY[constraintOffsetIndex]),
              drjit::load<FloatP>(&constraints_->offsetZ[constraintOffsetIndex])};
          const Vector3fP pos_world = jointState_cons.transform * offset;

          // Calculate distance of point to plane: dist = pos.dot(normal) - target
          const Vector3fP normal{
              drjit::load<FloatP>(&constraints_->normalX[constraintOffsetIndex]),
              drjit::load<FloatP>(&constraints_->normalY[constraintOffsetIndex]),
              drjit::load<FloatP>(&constraints_->normalZ[constraintOffsetIndex])};
          const Vector3fP normal_world = momentum::operator*(jointRotMat, normal);
          const auto target = drjit::load<FloatP>(&constraints_->targets[constraintOffsetIndex]);
          const FloatP dist = dot(pos_world, normal_world) - target;

          // Calculate error as squared distance: err = constraintWeight * dist * dist
          const auto constraintWeight =
              drjit::load<FloatP>(&constraints_->weights[constraintOffsetIndex]);
          jointError += constraintWeight * drjit::square(dist);

          // Pre-calculate values for the gradient: wgt = 2.0 * kPlaneWeight * constraintWeight *
          // dist
          const FloatP wgt = 2.0f * kPlaneWeight * constraintWeight * dist;

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
            for (size_t d = 0; d < 3; d++) {
              if (this->activeJointParams_[paramIndex + d]) {
                // Calculate gradient with respect to joint: grad = normal.dot(axis) * wgt
                const FloatP val = dist *
                    momentum::dot(normal_world, jointState.getTranslationDerivative(d)) * wgt;
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
                // Calculate gradient with respect to joint: grad = normal.dot(axis.cross(posd)) *
                // wgt
                const FloatP val = dist *
                    dot(normal_world, momentum::cross(jointState.rotationAxis.col(d), posd)) * wgt;
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
              // Calculate gradient with respect to joint: grad = normal.dot(posd * log(2)) * wgt
              const FloatP val = dist * ln2<double>() * dot(normal_world, posd) * wgt;
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

  // Sum the joint errors for final result
  return static_cast<float>(kPlaneWeight * weight_ * error);
}

double SimdPlaneErrorFunction::getJacobian(
    const ModelParameters& /* params */,
    const SkeletonState& state,
    const MeshState& /* unused */,
    Ref<MatrixXf> jacobian,
    Ref<VectorXf> residual,
    int& usedRows) {
  MT_PROFILE_FUNCTION();
  MT_CHECK(jacobian.cols() == static_cast<Eigen::Index>(parameterTransform_.transform.cols()));

  if (constraints_ == nullptr) {
    return 0.0f;
  }

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
        // Precompute the linear (rotation*scale) and translation parts of the joint transform
        // ONCE per joint to avoid TransformT::operator*() recomputing toLinear() per packet.
        const Eigen::Matrix3f jointLinear = jointState_cons.transform.toLinear();
        const Eigen::Vector3f jointTrans = jointState_cons.transform.translation;
        const Eigen::Matrix3f jointRotMat = jointState_cons.rotation().toRotationMatrix();
        auto jointError = drjit::zeros<DoubleP>(); // use double to prevent rounding errors

        // Loop over all constraints in increments of kSimdPacketSize
        for (uint32_t index = 0; index < constraintCount; index += kSimdPacketSize) {
          const auto constraintOffsetIndex =
              jointId * SimdPlaneConstraints::kMaxConstraints + index;
          const auto jacobianOffsetCur = jacobianOffset_[jointId] + index;

          // Transform offset by joint transform: pos = transform * offset
          const Vector3fP offset{
              drjit::load<FloatP>(&constraints_->offsetX[constraintOffsetIndex]),
              drjit::load<FloatP>(&constraints_->offsetY[constraintOffsetIndex]),
              drjit::load<FloatP>(&constraints_->offsetZ[constraintOffsetIndex])};
          const Vector3fP pos_world =
              momentum::operator+(momentum::operator*(jointLinear, offset), jointTrans);

          // Calculate distance of point to plane: dist = pos.dot(normal) - target
          const Vector3fP normal{
              drjit::load<FloatP>(&constraints_->normalX[constraintOffsetIndex]),
              drjit::load<FloatP>(&constraints_->normalY[constraintOffsetIndex]),
              drjit::load<FloatP>(&constraints_->normalZ[constraintOffsetIndex])};
          const Vector3fP normal_world = momentum::operator*(jointRotMat, normal);
          const auto target = drjit::load<FloatP>(&constraints_->targets[constraintOffsetIndex]);
          const FloatP dist = dot(pos_world, normal_world) - target;

          // Calculate error as squared distance: err = constraintWeight * dist * dist
          const auto constraintWeight =
              drjit::load<FloatP>(&constraints_->weights[constraintOffsetIndex]);
          jointError += constraintWeight * drjit::square(dist);

          // Calculate square-root of weight: wgt = sqrt(kPlaneWeight * weight * constraintWeight)
          const FloatP wgt = drjit::sqrt(kPlaneWeight * weight_ * constraintWeight);

          // Calculate residual: res = wgt * dist
          drjit::store(residual.segment(jacobianOffsetCur, kSimdPacketSize).data(), dist * wgt);

          // Loop over all joints the constraint is attached to and calculate gradient
          size_t jointIndex = jointId;
          while (jointIndex != kInvalidIndex) {
            // Check for valid index
            MT_CHECK(jointIndex < static_cast<size_t>(constraints_->numJoints));

            const auto& jointState = state.jointState[jointIndex];
            const size_t paramIndex = jointIndex * kParametersPerJoint;
            // Cache translation once per chain joint.
            const Eigen::Vector3f chainTrans = jointState.translation();

            // Calculate difference between constraint position and joint center
            const Vector3fP posd = momentum::operator-(pos_world, chainTrans);
            // d-independent. For rotation Jacobian we use:
            //   normal.dot(axis x posd) = -dot(axis, normal x posd) = dot(posd x normal, axis)
            // Hoisting (posd x normal) out of the d-loop avoids recomputing the 3-element cross
            // product per axis.
            const Vector3fP posdCrossNorm = cross(posd, normal_world);

            // Calculate derivatives based on active joints
            for (size_t d = 0; d < 3; ++d) {
              if (this->activeJointParams_[paramIndex + d]) {
                // Jac translation: normal.dot(axis) * wgt
                const FloatP jc =
                    momentum::dot(normal_world, jointState.getTranslationDerivative(d)) * wgt;
                jacobian_jointParams_to_modelParams(
                    jc,
                    paramIndex + d,
                    parameterTransform_,
                    jacobian.middleRows(jacobianOffsetCur, kSimdPacketSize));
              }

              if (this->activeJointParams_[paramIndex + 3 + d]) {
                // Jac rotation: dot(posd x normal, axis_d) * wgt (using vector triple identity).
                const Vector3f axis = jointState.rotationAxis.col(d);
                const FloatP jc = momentum::dot(axis, posdCrossNorm) * wgt;
                jacobian_jointParams_to_modelParams(
                    jc,
                    paramIndex + d + 3,
                    parameterTransform_,
                    jacobian.middleRows(jacobianOffsetCur, kSimdPacketSize));
              }
            }

            if (this->activeJointParams_[paramIndex + 6]) {
              // Calculate Jacobian with respect to joint: jac = normal.dot(posd) * wgt * LN2
              const FloatP jc = dot(normal_world, posd) * (ln2<double>() * wgt);
              jacobian_jointParams_to_modelParams(
                  jc,
                  paramIndex + 6,
                  parameterTransform_,
                  jacobian.middleRows(jacobianOffsetCur, kSimdPacketSize));
            }

            // Go to the next joint
            jointIndex = this->skeleton_.joints[jointIndex].parent;
          }
        }

        jointErrors[jointId] += drjit::sum(jointError);
      },
      dispensoOptions);

  usedRows = gsl::narrow_cast<int>(jacobian.rows());

  // Sum the joint errors for final result
  const double error =
      kPlaneWeight * weight_ * std::accumulate(jointErrors.begin(), jointErrors.end(), 0.0);
  return static_cast<float>(error);
}

size_t SimdPlaneErrorFunction::getJacobianSize() const {
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

void SimdPlaneErrorFunction::setConstraints(const SimdPlaneConstraints* cstrs) {
  constraints_ = cstrs;
}
} // namespace momentum
