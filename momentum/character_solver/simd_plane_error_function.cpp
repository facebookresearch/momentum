/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/simd_plane_error_function.h"

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

SimdPlaneConstraints::~SimdPlaneConstraints() {
  // Do nothing
}

void SimdPlaneConstraints::addConstraint(
    const size_t jointIndex,
    const Vector3f& offset,
    const Vector3f& targetNormal,
    const float targetOffset,
    const float targetWeight) {
  MT_CHECK(jointIndex < constraintCount.size());

  // add the constraint to the corresponding arrays of the jointIndex if there's enough space
  auto index = constraintCount[jointIndex].fetch_add(1, std::memory_order_relaxed);
  if (index < kMaxConstraints) {
    const auto finalIndex = jointIndex * kMaxConstraints + index;
    offsetX[finalIndex] = offset.x();
    offsetY[finalIndex] = offset.y();
    offsetZ[finalIndex] = offset.z();
    normalX[finalIndex] = targetNormal.x();
    normalY[finalIndex] = targetNormal.y();
    normalZ[finalIndex] = targetNormal.z();
    targets[finalIndex] = targetOffset;
    weights[finalIndex] = targetWeight;
  } else
    constraintCount[jointIndex]--;
}

VectorXi SimdPlaneConstraints::getNumConstraints() const {
  VectorXi res = VectorXi(numJoints);
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
    size_t maxThreads)
    : SkeletonErrorFunction(skel, pt), maxThreads_(maxThreads) {
  constraints_ = nullptr;
}

SimdPlaneErrorFunction::SimdPlaneErrorFunction(const Character& character, size_t maxThreads)
    : SimdPlaneErrorFunction(character.skeleton, character.parameterTransform, maxThreads) {
  // Do nothing
}

double SimdPlaneErrorFunction::getError(
    const ModelParameters& /* params */,
    const SkeletonState& state) {
  if (constraints_ == nullptr) {
    return 0.0f;
  }

  MT_CHECK((size_t)constraints_->numJoints <= state.jointState.size());

  // Loop over all joints, as these are our base units
  auto error = drjit::zeros<DoubleP>(); // use double to prevent rounding errors
  for (int jointId = 0; jointId < constraints_->numJoints; jointId++) {
    const size_t constraintCount = constraints_->constraintCount[jointId];
    MT_CHECK(jointId < static_cast<int>(state.jointState.size()));
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
      const Vector3fP pos_world = jointState.transformation * offset;

      // Calculate distance of point to plane: dist = pos.dot(normal) - target
      const Vector3fP normal{
          drjit::load<FloatP>(&constraints_->normalX[constraintOffsetIndex]),
          drjit::load<FloatP>(&constraints_->normalY[constraintOffsetIndex]),
          drjit::load<FloatP>(&constraints_->normalZ[constraintOffsetIndex])};
      const Vector3fP normal_world = momentum::operator*(jointRotMat, normal);
      const auto target = drjit::load<FloatP>(&constraints_->targets[constraintOffsetIndex]);
      const FloatP dist = dot(pos_world, normal_world) - target;

      // Calculate error as squared distance: err = constraintWeight * dist * dist
      const FloatP constraintWeight =
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
    Ref<VectorXf> gradient) {
  if (constraints_ == nullptr) {
    return 0.0f;
  }

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
          const Vector3fP pos_world = jointState_cons.transformation * offset;

          // Calculate distance of point to plane: dist = pos.dot(normal) - target
          const Vector3fP normal{
              drjit::load<FloatP>(&constraints_->normalX[constraintOffsetIndex]),
              drjit::load<FloatP>(&constraints_->normalY[constraintOffsetIndex]),
              drjit::load<FloatP>(&constraints_->normalZ[constraintOffsetIndex])};
          const Vector3fP normal_world = momentum::operator*(jointRotMat, normal);
          const auto target = drjit::load<FloatP>(&constraints_->targets[constraintOffsetIndex]);
          const FloatP dist = dot(pos_world, normal_world) - target;

          // Calculate error as squared distance: err = constraintWeight * dist * dist
          const FloatP constraintWeight =
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
                  gradient[parameterTransform_.transform.innerIndexPtr()[ptIndex]] +=
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
                  gradient[parameterTransform_.transform.innerIndexPtr()[ptIndex]] +=
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
                gradient[parameterTransform_.transform.innerIndexPtr()[ptIndex]] +=
                    drjit::sum(val) * parameterTransform_.transform.valuePtr()[ptIndex];
              }
            }

            // Go to the next joint
            jointIndex = this->skeleton_.joints[jointIndex].parent;
          }
        }

        jointErrors[jointId] += drjit::sum(jointError);
      },
      dispensoOptions);

  // Sum the joint errors for final result
  const double error =
      kPlaneWeight * weight_ * std::accumulate(jointErrors.begin(), jointErrors.end(), 0.0);
  return static_cast<float>(error);
}

__vectorcall DRJIT_INLINE void jacobian_jointParams_to_modelParams(
    const FloatP& jacobian_jointParams,
    const Eigen::Index iJointParam,
    const ParameterTransform& parameterTransform_,
    Eigen::Ref<Eigen::MatrixX<float>> jacobian) {
  // explicitly multiply with the parameter transform to generate parameter space gradients
  for (auto index = parameterTransform_.transform.outerIndexPtr()[iJointParam];
       index < parameterTransform_.transform.outerIndexPtr()[iJointParam + 1];
       ++index) {
    const auto modelParamIdx = parameterTransform_.transform.innerIndexPtr()[index];
    float* jacPtr = jacobian.col(modelParamIdx).data();
    drjit::store(
        jacPtr,
        drjit::load<FloatP>(jacPtr) +
            parameterTransform_.transform.valuePtr()[index] * jacobian_jointParams);
  }
}

double SimdPlaneErrorFunction::getJacobian(
    const ModelParameters& /* params */,
    const SkeletonState& state,
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
          const Vector3fP pos_world = jointState_cons.transformation * offset;

          // Calculate distance of point to plane: dist = pos.dot(normal) - target
          const Vector3fP normal{
              drjit::load<FloatP>(&constraints_->normalX[constraintOffsetIndex]),
              drjit::load<FloatP>(&constraints_->normalY[constraintOffsetIndex]),
              drjit::load<FloatP>(&constraints_->normalZ[constraintOffsetIndex])};
          const Vector3fP normal_world = momentum::operator*(jointRotMat, normal);
          const auto target = drjit::load<FloatP>(&constraints_->targets[constraintOffsetIndex]);
          const FloatP dist = dot(pos_world, normal_world) - target;

          // Calculate error as squared distance: err = constraintWeight * dist * dist
          const FloatP constraintWeight =
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

            // Calculate difference between constraint position and joint center: posd = pos -
            // jointState.translation
            const Vector3fP posd = momentum::operator-(pos_world, jointState.translation());

            // Calculate derivatives based on active joints
            for (size_t d = 0; d < 3; ++d) {
              if (this->activeJointParams_[paramIndex + d]) {
                // Calculate Jacobian with respect to joint: jac = normal.dot(axis) * wgt
                const FloatP jc =
                    momentum::dot(normal_world, jointState.getTranslationDerivative(d)) * wgt;
                jacobian_jointParams_to_modelParams(
                    jc,
                    paramIndex + d,
                    parameterTransform_,
                    jacobian.middleRows(jacobianOffsetCur, kSimdPacketSize));
              }

              if (this->activeJointParams_[paramIndex + 3 + d]) {
                // Calculate Jacobian with respect to joint: jac = normal.dot(axis.cross(posd)) *
                // wgt;
                const FloatP jc =
                    dot(normal_world, momentum::cross(jointState.rotationAxis.col(d), posd)) * wgt;
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

#ifdef MOMENTUM_ENABLE_AVX

inline double __vectorcall sum8(const __m256 x) {
  // extract the higher and lower 4 floats
  const __m128 high = _mm256_extractf128_ps(x, 1);
  const __m128 low = _mm256_castps256_ps128(x);
  // convert them to 256 bit doubles and add them up
  const __m256d val = _mm256_add_pd(_mm256_cvtps_pd(high), _mm256_cvtps_pd(low));
  // sum the 4 doubles up and return the result
  const __m128d valupper = _mm256_extractf128_pd(val, 1);
  const __m128d vallower = _mm256_castpd256_pd128(val);
  _mm256_zeroupper();
  const __m128d valval = _mm_add_pd(valupper, vallower);
  const __m128d res = _mm_add_pd(_mm_permute_pd(valval, 1), valval);
  return _mm_cvtsd_f64(res);
}

SimdPlaneErrorFunctionAVX::SimdPlaneErrorFunctionAVX(
    const Skeleton& skel,
    const ParameterTransform& pt,
    size_t maxThreads)
    : SimdPlaneErrorFunction(skel, pt, maxThreads) {
  // Do nothing
}

SimdPlaneErrorFunctionAVX::SimdPlaneErrorFunctionAVX(const Character& character, size_t maxThreads)
    : SimdPlaneErrorFunction(character, maxThreads) {
  // Do nothing
}

double SimdPlaneErrorFunctionAVX::getError(
    const ModelParameters& /* params */,
    const SkeletonState& state) {
  // do all summations in double to prevent rounding errors
  double error = 0.0;
  if (constraints_ == nullptr) {
    return 0.0f;
  }

  // loop over all joints, as these are our base units
  for (int jointId = 0; jointId < constraints_->numJoints; jointId++) {
    // pre-load some joint specific values
    const auto& transformation = state.jointState[jointId].transformation;

    __m256 posx;
    __m256 posy;
    __m256 posz;
    __m256 dist;

    const auto jointOffset = jointId * SimdPlaneConstraints::kMaxConstraints;

    // loop over all constraints in increments of kAvxPacketSize
    const uint32_t count = constraints_->constraintCount[jointId];
    for (uint32_t index = 0; index < count; index += kAvxPacketSize) {
      // transform offset by joint transformation :           pos = transform * offset
      const __m256 valx = _mm256_loadu_ps(&constraints_->offsetX[jointOffset + index]);
      posx = _mm256_mul_ps(valx, _mm256_broadcast_ss(&transformation.data()[0]));
      posy = _mm256_mul_ps(valx, _mm256_broadcast_ss(&transformation.data()[1]));
      posz = _mm256_mul_ps(valx, _mm256_broadcast_ss(&transformation.data()[2]));
      const __m256 valy = _mm256_loadu_ps(&constraints_->offsetY[jointOffset + index]);
      posx = _mm256_fmadd_ps(valy, _mm256_broadcast_ss(&transformation.data()[4]), posx);
      posy = _mm256_fmadd_ps(valy, _mm256_broadcast_ss(&transformation.data()[5]), posy);
      posz = _mm256_fmadd_ps(valy, _mm256_broadcast_ss(&transformation.data()[6]), posz);
      const __m256 valz = _mm256_loadu_ps(&constraints_->offsetZ[jointOffset + index]);
      posx = _mm256_fmadd_ps(valz, _mm256_broadcast_ss(&transformation.data()[8]), posx);
      posy = _mm256_fmadd_ps(valz, _mm256_broadcast_ss(&transformation.data()[9]), posy);
      posz = _mm256_fmadd_ps(valz, _mm256_broadcast_ss(&transformation.data()[10]), posz);
      posx = _mm256_add_ps(_mm256_broadcast_ss(&transformation.data()[12]), posx);
      posy = _mm256_add_ps(_mm256_broadcast_ss(&transformation.data()[13]), posy);
      posz = _mm256_add_ps(_mm256_broadcast_ss(&transformation.data()[14]), posz);

      // calculate distance of point to plane :               dist = pos.dot(normal) - target
      const __m256 normalx = _mm256_loadu_ps(&constraints_->normalX[jointOffset + index]);
      dist = _mm256_mul_ps(posx, normalx);
      const __m256 normaly = _mm256_loadu_ps(&constraints_->normalY[jointOffset + index]);
      dist = _mm256_fmadd_ps(posy, normaly, dist);
      const __m256 normalz = _mm256_loadu_ps(&constraints_->normalZ[jointOffset + index]);
      dist = _mm256_fmadd_ps(posz, normalz, dist);
      const __m256 tgt = _mm256_loadu_ps(&constraints_->targets[jointOffset + index]);
      dist = _mm256_sub_ps(dist, tgt);

      // calculate error as squared distance :                err = weight * dist * dist
      const __m256 wdist =
          _mm256_mul_ps(dist, _mm256_loadu_ps(&constraints_->weights[jointOffset + index]));
      const __m256 err = _mm256_mul_ps(wdist, dist);

      // sum up the values and add to error
      error += sum8(err);
    }
  }

  // sum the AVX error register for final result
  return static_cast<float>(error * kPlaneWeight * weight_);
}

double SimdPlaneErrorFunctionAVX::getGradient(
    const ModelParameters& /* params */,
    const SkeletonState& state,
    Ref<VectorXf> gradient) {
  if (constraints_ == nullptr) {
    return 0.0f;
  }

  // storage for joint gradients
  std::vector<std::tuple<double, VectorXd>> ets_error_grad;

  // loop over all joints, as these are our base units
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

        // pre-load some joint specific values
        const auto& transformation = state.jointState[jointId].transformation;

        __m256 posx;
        __m256 posy;
        __m256 posz;
        __m256 dist;

        const auto jointOffset = jointId * SimdPlaneConstraints::kMaxConstraints;

        // loop over all constraints in increments of kAvxPacketSize
        const uint32_t count = constraints_->constraintCount[jointId];
        for (uint32_t index = 0; index < count; index += kAvxPacketSize) {
          // transform offset by joint transformation :           pos = transform * offset
          const __m256 valx = _mm256_loadu_ps(&constraints_->offsetX[jointOffset + index]);
          posx = _mm256_mul_ps(valx, _mm256_broadcast_ss(&transformation.data()[0]));
          posy = _mm256_mul_ps(valx, _mm256_broadcast_ss(&transformation.data()[1]));
          posz = _mm256_mul_ps(valx, _mm256_broadcast_ss(&transformation.data()[2]));
          const __m256 valy = _mm256_loadu_ps(&constraints_->offsetY[jointOffset + index]);
          posx = _mm256_fmadd_ps(valy, _mm256_broadcast_ss(&transformation.data()[4]), posx);
          posy = _mm256_fmadd_ps(valy, _mm256_broadcast_ss(&transformation.data()[5]), posy);
          posz = _mm256_fmadd_ps(valy, _mm256_broadcast_ss(&transformation.data()[6]), posz);
          const __m256 valz = _mm256_loadu_ps(&constraints_->offsetZ[jointOffset + index]);
          posx = _mm256_fmadd_ps(valz, _mm256_broadcast_ss(&transformation.data()[8]), posx);
          posy = _mm256_fmadd_ps(valz, _mm256_broadcast_ss(&transformation.data()[9]), posy);
          posz = _mm256_fmadd_ps(valz, _mm256_broadcast_ss(&transformation.data()[10]), posz);
          posx = _mm256_add_ps(_mm256_broadcast_ss(&transformation.data()[12]), posx);
          posy = _mm256_add_ps(_mm256_broadcast_ss(&transformation.data()[13]), posy);
          posz = _mm256_add_ps(_mm256_broadcast_ss(&transformation.data()[14]), posz);

          // calculate distance of point to plane :               dist = pos.dot(normal) - target
          const __m256 normalx = _mm256_loadu_ps(&constraints_->normalX[jointOffset + index]);
          dist = _mm256_mul_ps(posx, normalx);
          const __m256 normaly = _mm256_loadu_ps(&constraints_->normalY[jointOffset + index]);
          dist = _mm256_fmadd_ps(posy, normaly, dist);
          const __m256 normalz = _mm256_loadu_ps(&constraints_->normalZ[jointOffset + index]);
          dist = _mm256_fmadd_ps(posz, normalz, dist);
          const __m256 tgt = _mm256_loadu_ps(&constraints_->targets[jointOffset + index]);
          dist = _mm256_sub_ps(dist, tgt);

          // calculate squared distance :                         sqr = dist * dist
          const __m256 distSqrNorm = _mm256_mul_ps(dist, dist);

          // calculate combined element weight                    tweight = weight
          const __m256 tweight = _mm256_loadu_ps(&constraints_->weights[jointOffset + index]);

          // calculate error :                                    err = weight * sqr
          const __m256 err = _mm256_mul_ps(distSqrNorm, tweight);

          // sum up the values and add to error
          error_local += sum8(err);

          // pre-calculate values for the gradient :              wgt = 2.0 * kPlaneWeight * weight
          // * dist
          constexpr float kWeight = 2.0f * kPlaneWeight;
          const __m256 wt = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(kWeight), dist), tweight);

          // loop over all joints the constraint is attached to and calculate gradient
          size_t jointIndex = jointId;
          while (jointIndex != kInvalidIndex) {
            // check for valid index
            MT_CHECK(jointIndex < static_cast<size_t>(constraints_->numJoints));

            const auto& jointState = state.jointState[jointIndex];
            const size_t paramIndex = jointIndex * kParametersPerJoint;

            // calculate difference between constraint position and joint center :          posd =
            // pos - jointState.translation
            const __m256 posdx = _mm256_sub_ps(posx, _mm256_broadcast_ss(&jointState.x()));
            const __m256 posdy = _mm256_sub_ps(posy, _mm256_broadcast_ss(&jointState.y()));
            const __m256 posdz = _mm256_sub_ps(posz, _mm256_broadcast_ss(&jointState.z()));

            // calculate derivatives based on active joints
            for (size_t d = 0; d < 3; d++) {
              if (activeJointParams_[paramIndex + d]) {
                // calculate gradient with respect to joint :                           grad =
                // normal.dot(axis) * wgt;
                const Vector3f& axis = jointState.translationAxis.col(d);
                const __m256 res = _mm256_mul_ps(
                    _mm256_fmadd_ps(
                        normalz,
                        _mm256_broadcast_ss(&axis.z()),
                        _mm256_fmadd_ps(
                            normaly,
                            _mm256_broadcast_ss(&axis.y()),
                            _mm256_mul_ps(normalx, _mm256_broadcast_ss(&axis.x())))),
                    wt);
                // calculate joint gradient
                const double val = sum8(res);
                // explicitly multiply with the parameter transform to generate parameter space
                // gradients
                for (auto pIndex = parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
                     pIndex < parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
                     ++pIndex) {
                  grad_local[parameterTransform_.transform.innerIndexPtr()[pIndex]] +=
                      val * parameterTransform_.transform.valuePtr()[pIndex];
                }
              }
              if (activeJointParams_[paramIndex + 3 + d]) {
                // calculate gradient with respect to joint :                           grad =
                // normal.dot(axis.cross(posd)) * wgt;
                const Vector3f& axis = jointState.rotationAxis.col(d);
                const __m256 axisx = _mm256_broadcast_ss(&axis.x());
                const __m256 axisy = _mm256_broadcast_ss(&axis.y());
                const __m256 axisz = _mm256_broadcast_ss(&axis.z());
                const __m256 crossx = _mm256_fmsub_ps(axisy, posdz, _mm256_mul_ps(axisz, posdy));
                const __m256 crossy = _mm256_fmsub_ps(axisz, posdx, _mm256_mul_ps(axisx, posdz));
                const __m256 crossz = _mm256_fmsub_ps(axisx, posdy, _mm256_mul_ps(axisy, posdx));
                const __m256 res = _mm256_mul_ps(
                    _mm256_fmadd_ps(
                        normalz,
                        crossz,
                        _mm256_fmadd_ps(normaly, crossy, _mm256_mul_ps(normalx, crossx))),
                    wt);
                // calculate joint gradient
                const double val = sum8(res);
                // explicitly multiply with the parameter transform to generate parameter space
                // gradients
                for (auto pIndex =
                         parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d];
                     pIndex < parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 3 + 1];
                     ++pIndex) {
                  grad_local[parameterTransform_.transform.innerIndexPtr()[pIndex]] +=
                      val * parameterTransform_.transform.valuePtr()[pIndex];
                }
              }
            }
            if (activeJointParams_[paramIndex + 6]) {
              // calculate gradient with respect to joint:
              // grad = normal.dot(posd * LN2) * wgt;
              const __m256 res = _mm256_mul_ps(
                  _mm256_fmadd_ps(
                      normalz,
                      posdz,
                      _mm256_fmadd_ps(normaly, posdy, _mm256_mul_ps(normalx, posdx))),
                  wt);
              // calculate joint gradient
              const double val = sum8(res) * ln2<double>();
              // explicitly multiply with the parameter transform to generate parameter space
              // gradients
              for (auto pIndex = parameterTransform_.transform.outerIndexPtr()[paramIndex + 6];
                   pIndex < parameterTransform_.transform.outerIndexPtr()[paramIndex + 6 + 1];
                   ++pIndex) {
                grad_local[parameterTransform_.transform.innerIndexPtr()[pIndex]] +=
                    val * parameterTransform_.transform.valuePtr()[pIndex];
              }
            }

            // go to the next joint
            jointIndex = skeleton_.joints[jointIndex].parent;
          }
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

    // finalize the gradient
    gradient += std::get<1>(ets_error_grad[0]).cast<float>() * weight_;
    // sum the AVX error register for final result
    error = std::get<0>(ets_error_grad[0]);
  }

  return static_cast<float>(error * kPlaneWeight * weight_);
}

double SimdPlaneErrorFunctionAVX::getJacobian(
    const ModelParameters& /* params */,
    const SkeletonState& state,
    Ref<MatrixXf> jacobian,
    Ref<VectorXf> residual,
    int& usedRows) {
  MT_PROFILE_FUNCTION();
  MT_CHECK(jacobian.cols() == static_cast<Eigen::Index>(parameterTransform_.transform.cols()));

  if (constraints_ == nullptr) {
    return 0.0f;
  }

  // storage for joint errors
  std::vector<double> ets_error;

  // need to make sure we're actually at a 32 byte data offset at the first offset for AVX access
  const size_t addressOffset = computeOffset<kAvxAlignment>(jacobian);
  checkAlignment<kAvxAlignment>(jacobian, addressOffset);

  // calculate actually used number of rows
  const size_t maxRows = gsl::narrow_cast<size_t>(jacobian.rows());
  size_t num = 0;
  for (int i = 0; i < constraints_->numJoints; i++) {
    jacobianOffset_[i] = num;
    const size_t res = constraints_->constraintCount[i] % kAvxPacketSize;
    if (res != 0) {
      num += constraints_->constraintCount[i] + kAvxPacketSize - res;
    } else {
      num += constraints_->constraintCount[i];
    }
  }

  // loop over all joints, as these are our base units
  auto dispensoOptions = dispenso::ParForOptions();
  dispensoOptions.maxThreads = maxThreads_;
  dispenso::parallel_for(
      ets_error,
      []() { return 0.0; },
      0,
      constraints_->numJoints,
      [&](double& error_local, const size_t jointId) {
        // get initial offset
        const auto offset = jacobianOffset_[jointId] + addressOffset;

        // pre-load some joint specific values
        const auto& transformation = state.jointState[jointId].transformation;

        __m256 posx;
        __m256 posy;
        __m256 posz;
        __m256 dist;

        const auto jointOffset = jointId * SimdPlaneConstraints::kMaxConstraints;

        // loop over all constraints in increments of kAvxPacketSize
        const uint32_t count = constraints_->constraintCount[jointId];
        for (uint32_t index = 0; index < count; index += kAvxPacketSize) {
          // check if we're too much
          if (offset + index + kAvxPacketSize >= maxRows) {
            continue;
          }

          // transform offset by joint transformation: pos = transform * offset
          const __m256 valx = _mm256_loadu_ps(&constraints_->offsetX[jointOffset + index]);
          posx = _mm256_mul_ps(valx, _mm256_broadcast_ss(&transformation.data()[0]));
          posy = _mm256_mul_ps(valx, _mm256_broadcast_ss(&transformation.data()[1]));
          posz = _mm256_mul_ps(valx, _mm256_broadcast_ss(&transformation.data()[2]));
          const __m256 valy = _mm256_loadu_ps(&constraints_->offsetY[jointOffset + index]);
          posx = _mm256_fmadd_ps(valy, _mm256_broadcast_ss(&transformation.data()[4]), posx);
          posy = _mm256_fmadd_ps(valy, _mm256_broadcast_ss(&transformation.data()[5]), posy);
          posz = _mm256_fmadd_ps(valy, _mm256_broadcast_ss(&transformation.data()[6]), posz);
          const __m256 valz = _mm256_loadu_ps(&constraints_->offsetZ[jointOffset + index]);
          posx = _mm256_fmadd_ps(valz, _mm256_broadcast_ss(&transformation.data()[8]), posx);
          posy = _mm256_fmadd_ps(valz, _mm256_broadcast_ss(&transformation.data()[9]), posy);
          posz = _mm256_fmadd_ps(valz, _mm256_broadcast_ss(&transformation.data()[10]), posz);
          posx = _mm256_add_ps(_mm256_broadcast_ss(&transformation.data()[12]), posx);
          posy = _mm256_add_ps(_mm256_broadcast_ss(&transformation.data()[13]), posy);
          posz = _mm256_add_ps(_mm256_broadcast_ss(&transformation.data()[14]), posz);

          // calculate distance of point to plane: dist = pos.dot(normal) - target
          const __m256 normalx = _mm256_loadu_ps(&constraints_->normalX[jointOffset + index]);
          dist = _mm256_mul_ps(posx, normalx);
          const __m256 normaly = _mm256_loadu_ps(&constraints_->normalY[jointOffset + index]);
          dist = _mm256_fmadd_ps(posy, normaly, dist);
          const __m256 normalz = _mm256_loadu_ps(&constraints_->normalZ[jointOffset + index]);
          dist = _mm256_fmadd_ps(posz, normalz, dist);
          const __m256 tgt = _mm256_loadu_ps(&constraints_->targets[jointOffset + index]);
          dist = _mm256_sub_ps(dist, tgt);

          // calculate square-root of weight: wgt = sqrt(kPlaneWeight * weight * constraintWeight)
          const float w = kPlaneWeight * weight_;
          const __m256 wt = _mm256_mul_ps(
              _mm256_broadcast_ss(&w),
              _mm256_loadu_ps(&constraints_->weights[jointOffset + index]));
          const __m256 wgt = _mm256_sqrt_ps(wt);

          // calculate residual: res = wgt * dist
          const __m256 res = _mm256_mul_ps(wgt, dist);

          // calculate error as squared residual: err = res * res
          const __m256 err = _mm256_mul_ps(res, res);

          // sum up the values and add to error
          error_local += sum8(err);

          // loop over all joints the constraint is attached to and calculate gradient
          size_t jointIndex = jointId;
          while (jointIndex != kInvalidIndex) {
            // check for valid index
            MT_CHECK(jointIndex < static_cast<size_t>(constraints_->numJoints));

            const auto& jointState = state.jointState[jointIndex];
            const size_t paramIndex = jointIndex * kParametersPerJoint;

            // calculate difference between constraint position and joint center: posd = pos -
            // jointState.translation
            const __m256 posdx = _mm256_sub_ps(posx, _mm256_broadcast_ss(&jointState.x()));
            const __m256 posdy = _mm256_sub_ps(posy, _mm256_broadcast_ss(&jointState.y()));
            const __m256 posdz = _mm256_sub_ps(posz, _mm256_broadcast_ss(&jointState.z()));

            // calculate derivatives based on active joints
            for (size_t d = 0; d < 3; d++) {
              if (activeJointParams_[paramIndex + d]) {
                // calculate jacobian with respect to joint: jac = normal.dot(axis) * wgt;
                const Vector3f& axis = jointState.translationAxis.col(d);
                const __m256 axisx = _mm256_broadcast_ss(&axis.x());
                const __m256 axisy = _mm256_broadcast_ss(&axis.y());
                const __m256 axisz = _mm256_broadcast_ss(&axis.z());
                const __m256 jac = _mm256_mul_ps(
                    _mm256_fmadd_ps(
                        normalz,
                        axisz,
                        _mm256_fmadd_ps(normaly, axisy, _mm256_mul_ps(normalx, axisx))),
                    wgt);

                // explicitly multiply with the parameter transform to generate parameter space
                // jacobians
                for (auto pIndex = parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
                     pIndex < parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
                     ++pIndex) {
                  const float pw = parameterTransform_.transform.valuePtr()[pIndex];
                  const __m256 pjac = _mm256_mul_ps(_mm256_broadcast_ss(&pw), jac);
                  auto* const address = &jacobian(
                      offset + index, parameterTransform_.transform.innerIndexPtr()[pIndex]);
                  const __m256 prev = _mm256_loadu_ps(address);
                  _mm256_storeu_ps(address, _mm256_add_ps(prev, pjac));
                }
              }
              if (activeJointParams_[paramIndex + 3 + d]) {
                // calculate jacobian with respect to joint: jac = normal.dot(axis.cross(posd)) *
                // wgt;
                const Vector3f& axis = jointState.rotationAxis.col(d);
                const __m256 axisx = _mm256_broadcast_ss(&axis.x());
                const __m256 axisy = _mm256_broadcast_ss(&axis.y());
                const __m256 axisz = _mm256_broadcast_ss(&axis.z());
                const __m256 crossx = _mm256_fmsub_ps(axisy, posdz, _mm256_mul_ps(axisz, posdy));
                const __m256 crossy = _mm256_fmsub_ps(axisz, posdx, _mm256_mul_ps(axisx, posdz));
                const __m256 crossz = _mm256_fmsub_ps(axisx, posdy, _mm256_mul_ps(axisy, posdx));
                const __m256 jac = _mm256_mul_ps(
                    _mm256_fmadd_ps(
                        normalz,
                        crossz,
                        _mm256_fmadd_ps(normaly, crossy, _mm256_mul_ps(normalx, crossx))),
                    wgt);

                // explicitly multiply with the parameter transform to generate parameter space
                // jacobians
                for (auto pIndex =
                         parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 3];
                     pIndex < parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 3 + 1];
                     ++pIndex) {
                  const float pw = parameterTransform_.transform.valuePtr()[pIndex];
                  const __m256 pjac = _mm256_mul_ps(_mm256_broadcast_ss(&pw), jac);
                  auto* const address = &jacobian(
                      offset + index, parameterTransform_.transform.innerIndexPtr()[pIndex]);
                  const __m256 prev = _mm256_loadu_ps(address);
                  _mm256_storeu_ps(address, _mm256_add_ps(prev, pjac));
                }
              }
            }
            if (activeJointParams_[paramIndex + 6]) {
              // calculate jacobian with respect to joint: jac = normal.dot(posd) * wgt * LN2;
              constexpr double kLn2d = ln2<double>();
              const __m256 jac = _mm256_mul_ps(
                  _mm256_mul_ps(
                      _mm256_fmadd_ps(
                          normalz,
                          posdz,
                          _mm256_fmadd_ps(normaly, posdy, _mm256_mul_ps(normalx, posdx))),
                      wgt),
                  _mm256_set_ps(kLn2d, kLn2d, kLn2d, kLn2d, kLn2d, kLn2d, kLn2d, kLn2d));

              // explicitly multiply with the parameter transform to generate parameter space
              // jacobians
              for (auto pIndex = parameterTransform_.transform.outerIndexPtr()[paramIndex + 6];
                   pIndex < parameterTransform_.transform.outerIndexPtr()[paramIndex + 6 + 1];
                   ++pIndex) {
                const float pw = parameterTransform_.transform.valuePtr()[pIndex];
                const __m256 pjac = _mm256_mul_ps(_mm256_broadcast_ss(&pw), jac);
                auto* const address = &jacobian(
                    offset + index, parameterTransform_.transform.innerIndexPtr()[pIndex]);
                const __m256 prev = _mm256_loadu_ps(address);
                _mm256_storeu_ps(address, _mm256_add_ps(prev, pjac));
              }
            }

            // go to the next joint
            jointIndex = skeleton_.joints[jointIndex].parent;
          }

          // store the residual
          _mm256_storeu_ps(&residual(offset + index), res);
        }
      },
      dispensoOptions);

  double error = std::accumulate(ets_error.begin(), ets_error.end(), 0.0);

  if (num >= maxRows) {
    usedRows = gsl::narrow_cast<int>(maxRows - 1);
  } else {
    usedRows = gsl::narrow_cast<int>(num);
  }

  // sum the AVX error register for final result
  return static_cast<float>(error);
}

size_t SimdPlaneErrorFunctionAVX::getJacobianSize() const {
  if (constraints_ == nullptr) {
    return 0;
  }

  jacobianOffset_.resize(constraints_->numJoints);
  size_t num = 0;
  for (int i = 0; i < constraints_->numJoints; i++) {
    jacobianOffset_[i] = num;
    const size_t count = constraints_->constraintCount[i];
    const auto numPackets = (count + kAvxPacketSize - 1) / kAvxPacketSize;
    num += numPackets * kAvxPacketSize;
  }
  return num;
}

#endif // MOMENTUM_ENABLE_AVX

} // namespace momentum
