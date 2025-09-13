/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/simd_normal_error_function.h"

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
    size_t maxThreads)
    : SkeletonErrorFunction(skel, pt), maxThreads_(maxThreads) {
  constraints_ = nullptr;
}

SimdNormalErrorFunction::SimdNormalErrorFunction(const Character& character, size_t maxThreads)
    : SimdNormalErrorFunction(character.skeleton, character.parameterTransform, maxThreads) {}

double SimdNormalErrorFunction::getError(
    const ModelParameters& /* params */,
    const SkeletonState& state) {
  if (constraints_ == nullptr) {
    return 0.0f;
  }

  MT_CHECK((size_t)constraints_->numJoints <= state.jointState.size());

  // loop over all joints, as these are our base units
  auto error = drjit::zeros<DoubleP>(); // use double to prevent rounding errors
  for (int jointId = 0; jointId < constraints_->numJoints; jointId++) {
    const size_t constraintCount = constraints_->constraintCount[jointId];
    MT_CHECK(jointId < static_cast<int>(state.jointState.size()));
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
      const Vector3fP pos_world = jointState.transformation * offset;

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
    const ModelParameters& /* params */,
    const SkeletonState& state,
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
      const Vector3fP pos_world = jointState_cons.transformation * offset;

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

double SimdNormalErrorFunction::getJacobian(
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
          const Vector3fP pos_world = jointState_cons.transformation * offset;

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

            const Vector3fP tgtd = momentum::operator-(target, jointState.translation());
            for (size_t d = 0; d < 3; ++d) {
              if (this->activeJointParams_[paramIndex + 3 + d]) {
                const Vector3fP crossProd = cross(normal_world, tgtd);
                const Vector3f axis = jointState.rotationAxis.col(d);
                const FloatP jc = -momentum::dot(axis, crossProd) * wgt;
                jacobian_jointParams_to_modelParams(
                    jc,
                    paramIndex + d + 3,
                    parameterTransform_,
                    jacobian.middleRows(jacobianOffsetCur, kSimdPacketSize));
              }
            }

            if (this->activeJointParams_[paramIndex + 6]) {
              const Vector3fP posd = momentum::operator-(pos_world, jointState.translation());
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

double SimdNormalErrorFunctionAVX::getJacobian(
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

  // storage for joint errors
  std::vector<double> ets_error;

  // need to make sure we're actually at a 32 byte data offset at the first offset for AVX access
  const size_t addressOffset = computeOffset<kAvxAlignment>(jacobian);
  checkAlignment<kAvxAlignment>(jacobian, addressOffset);

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
        __m256 nmlx;
        __m256 nmly;
        __m256 nmlz;
        __m256 dist;

        const auto jointOffset = jointId * SimdNormalConstraints::kMaxConstraints;

        // loop over all constraints in increments of kAvxPacketSize
        const uint32_t count = constraints_->constraintCount[jointId];
        for (uint32_t index = 0; index < count; index += kAvxPacketSize) {
          // transform offset by joint transformation :           pos = transform * offset
          // also transform normal by transformation :            nml = (transform.linear() *
          // normal).normalized()
          const __m256 valx = _mm256_load_ps(&constraints_->offsetX[jointOffset + index]);
          const __m256 nx = _mm256_load_ps(&constraints_->normalX[jointOffset + index]);
          posx = _mm256_mul_ps(valx, _mm256_broadcast_ss(&transformation.data()[0]));
          posy = _mm256_mul_ps(valx, _mm256_broadcast_ss(&transformation.data()[1]));
          posz = _mm256_mul_ps(valx, _mm256_broadcast_ss(&transformation.data()[2]));
          nmlx = _mm256_mul_ps(nx, _mm256_broadcast_ss(&transformation.data()[0]));
          nmly = _mm256_mul_ps(nx, _mm256_broadcast_ss(&transformation.data()[1]));
          nmlz = _mm256_mul_ps(nx, _mm256_broadcast_ss(&transformation.data()[2]));
          const __m256 valy = _mm256_load_ps(&constraints_->offsetY[jointOffset + index]);
          const __m256 ny = _mm256_load_ps(&constraints_->normalY[jointOffset + index]);
          posx = _mm256_fmadd_ps(valy, _mm256_broadcast_ss(&transformation.data()[4]), posx);
          posy = _mm256_fmadd_ps(valy, _mm256_broadcast_ss(&transformation.data()[5]), posy);
          posz = _mm256_fmadd_ps(valy, _mm256_broadcast_ss(&transformation.data()[6]), posz);
          nmlx = _mm256_fmadd_ps(ny, _mm256_broadcast_ss(&transformation.data()[4]), nmlx);
          nmly = _mm256_fmadd_ps(ny, _mm256_broadcast_ss(&transformation.data()[5]), nmly);
          nmlz = _mm256_fmadd_ps(ny, _mm256_broadcast_ss(&transformation.data()[6]), nmlz);
          const __m256 valz = _mm256_load_ps(&constraints_->offsetZ[jointOffset + index]);
          const __m256 nz = _mm256_load_ps(&constraints_->normalZ[jointOffset + index]);
          posx = _mm256_fmadd_ps(valz, _mm256_broadcast_ss(&transformation.data()[8]), posx);
          posy = _mm256_fmadd_ps(valz, _mm256_broadcast_ss(&transformation.data()[9]), posy);
          posz = _mm256_fmadd_ps(valz, _mm256_broadcast_ss(&transformation.data()[10]), posz);
          nmlx = _mm256_fmadd_ps(nz, _mm256_broadcast_ss(&transformation.data()[8]), nmlx);
          nmly = _mm256_fmadd_ps(nz, _mm256_broadcast_ss(&transformation.data()[9]), nmly);
          nmlz = _mm256_fmadd_ps(nz, _mm256_broadcast_ss(&transformation.data()[10]), nmlz);
          posx = _mm256_add_ps(_mm256_broadcast_ss(&transformation.data()[12]), posx);
          posy = _mm256_add_ps(_mm256_broadcast_ss(&transformation.data()[13]), posy);
          posz = _mm256_add_ps(_mm256_broadcast_ss(&transformation.data()[14]), posz);

          __m256 norm = _mm256_mul_ps(nmlx, nmlx);
          norm = _mm256_fmadd_ps(nmly, nmly, norm);
          norm = _mm256_fmadd_ps(nmlz, nmlz, norm);
          norm = _mm256_rsqrt_ps(_mm256_max_ps(norm, _mm256_set1_ps(0.0001f)));
          nmlx = _mm256_mul_ps(nmlx, norm);
          nmly = _mm256_mul_ps(nmly, norm);
          nmlz = _mm256_mul_ps(nmlz, norm);

          // calculate distance of point to plane :               dist = normal.dot(pos) -
          // normal.dot(data->target)
          const __m256 tgtx = _mm256_load_ps(&constraints_->targetX[jointOffset + index]);
          const __m256 tgty = _mm256_load_ps(&constraints_->targetY[jointOffset + index]);
          const __m256 tgtz = _mm256_load_ps(&constraints_->targetZ[jointOffset + index]);
          dist = _mm256_mul_ps(posx, nmlx);
          dist = _mm256_fmadd_ps(posy, nmly, dist);
          dist = _mm256_fmadd_ps(posz, nmlz, dist);
          dist = _mm256_fnmadd_ps(tgtx, nmlx, dist);
          dist = _mm256_fnmadd_ps(tgty, nmly, dist);
          dist = _mm256_fnmadd_ps(tgtz, nmlz, dist);

          // calculate square-root of weight :                    wgt = sqrt(kPlaneWeight * weight
          // * constraintWeight)
          const float w = kPlaneWeight * weight_;
          const __m256 wt = _mm256_mul_ps(
              _mm256_broadcast_ss(&w), _mm256_load_ps(&constraints_->weights[jointOffset + index]));
          const __m256 wgt = _mm256_sqrt_ps(wt);

          // calculate residual :                                 res = wgt * dist
          const __m256 res = _mm256_mul_ps(wgt, dist);

          // calculate error as squared residual :                err = res * res
          const __m256 err = _mm256_mul_ps(res, res);

          // sum up the values and add to error
          error_local += sum8(err);

          // loop over all joints the constraint is attached to and calculate gradient
          size_t jointIndex = jointId;
          while (jointIndex != kInvalidIndex) {
            // check for valid index
            const auto& jointState = state.jointState[jointIndex];
            const size_t paramIndex = jointIndex * kParametersPerJoint;

            // calculate difference between constraint position and joint center :          posd =
            // pos - jointState.translation
            const __m256 posdx = _mm256_sub_ps(posx, _mm256_broadcast_ss(&jointState.x()));
            const __m256 posdy = _mm256_sub_ps(posy, _mm256_broadcast_ss(&jointState.y()));
            const __m256 posdz = _mm256_sub_ps(posz, _mm256_broadcast_ss(&jointState.z()));

            // calculate difference between target position and joint center :              tgtd =
            // tgt - jointState.translation
            const __m256 tgtdx = _mm256_sub_ps(tgtx, _mm256_broadcast_ss(&jointState.x()));
            const __m256 tgtdy = _mm256_sub_ps(tgty, _mm256_broadcast_ss(&jointState.y()));
            const __m256 tgtdz = _mm256_sub_ps(tgtz, _mm256_broadcast_ss(&jointState.z()));

            // calculate derivatives based on active joints
            for (size_t d = 0; d < 3; d++) {
              if (activeJointParams_[paramIndex + d]) {
                // calculate jacobian with respect to joint :                           jac =
                // normal.dot(axis) * wgt;
                const Vector3f& axis = jointState.translationAxis.col(d);
                const __m256 axisx = _mm256_broadcast_ss(&axis.x());
                const __m256 axisy = _mm256_broadcast_ss(&axis.y());
                const __m256 axisz = _mm256_broadcast_ss(&axis.z());
                const __m256 jac = _mm256_mul_ps(
                    _mm256_fmadd_ps(
                        nmlz, axisz, _mm256_fmadd_ps(nmly, axisy, _mm256_mul_ps(nmlx, axisx))),
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
                  MT_CHECK((uintptr_t)address % 32 == 0);
                  const __m256 prev = _mm256_load_ps(address);
                  _mm256_store_ps(address, _mm256_add_ps(prev, pjac));
                }
              }
              if (activeJointParams_[paramIndex + 3 + d]) {
                // calculate jacobian with respect to joint :                           jac =
                // axis.dot(tgtd.cross(normal)) * wgt;
                const __m256 crossx = _mm256_fmsub_ps(tgtdy, nmlz, _mm256_mul_ps(tgtdz, nmly));
                const __m256 crossy = _mm256_fmsub_ps(tgtdz, nmlx, _mm256_mul_ps(tgtdx, nmlz));
                const __m256 crossz = _mm256_fmsub_ps(tgtdx, nmly, _mm256_mul_ps(tgtdy, nmlx));
                const Vector3f& axis = jointState.rotationAxis.col(d);
                const __m256 axisx = _mm256_broadcast_ss(&axis.x());
                const __m256 axisy = _mm256_broadcast_ss(&axis.y());
                const __m256 axisz = _mm256_broadcast_ss(&axis.z());
                const __m256 jac = _mm256_mul_ps(
                    _mm256_fmadd_ps(
                        axisz,
                        crossz,
                        _mm256_fmadd_ps(axisy, crossy, _mm256_mul_ps(axisx, crossx))),
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
                  MT_CHECK((uintptr_t)address % 32 == 0);
                  const __m256 prev = _mm256_load_ps(address);
                  _mm256_store_ps(address, _mm256_add_ps(prev, pjac));
                }
              }
            }
            if (activeJointParams_[paramIndex + 6]) {
              // calculate jacobian with respect to joint :
              // jac = normal.dot(posd) * wgt * LN2;
              constexpr double kLn2d = ln2<double>();
              const __m256 jac = _mm256_mul_ps(
                  _mm256_mul_ps(
                      _mm256_fmadd_ps(
                          nmlz, posdz, _mm256_fmadd_ps(nmly, posdy, _mm256_mul_ps(nmlx, posdx))),
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
                MT_CHECK((uintptr_t)address % 32 == 0);
                const __m256 prev = _mm256_load_ps(address);
                _mm256_store_ps(address, _mm256_add_ps(prev, pjac));
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

  usedRows = gsl::narrow_cast<int>(jacobian.rows());

  // sum the AVX error register for final result
  return static_cast<float>(error);
}

size_t SimdNormalErrorFunctionAVX::getJacobianSize() const {
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
