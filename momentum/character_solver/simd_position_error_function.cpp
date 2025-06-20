/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/simd_position_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/math/constants.h"
#include "momentum/simd/simd.h"

#include <dispenso/parallel_for.h>

#include <numeric>

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
  auto index = constraintCount[jointIndex].fetch_add(1, std::memory_order_relaxed);
  if (index < kMaxConstraints) {
    const auto finalIndex = jointIndex * kMaxConstraints + index;
    offsetX[finalIndex] = offset.x();
    offsetY[finalIndex] = offset.y();
    offsetZ[finalIndex] = offset.z();
    targetX[finalIndex] = target.x();
    targetY[finalIndex] = target.y();
    targetZ[finalIndex] = target.z();
    weights[finalIndex] = targetWeight;
  } else {
    constraintCount[jointIndex]--;
  }
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
    size_t maxThreads)
    : SkeletonErrorFunction(skel, pt), maxThreads_(maxThreads) {
  constraints_ = nullptr;
}

SimdPositionErrorFunction::SimdPositionErrorFunction(const Character& character, size_t maxThreads)
    : SimdPositionErrorFunction(character.skeleton, character.parameterTransform, maxThreads) {
  // Do nothing
}

double SimdPositionErrorFunction::getError(
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

    // Loop over all constraints in increments of kSimdPacketSize
    for (uint32_t index = 0; index < constraintCount; index += kSimdPacketSize) {
      const auto constraintOffsetIndex = jointId * SimdPositionConstraints::kMaxConstraints + index;

      // Transform offset by joint transform: pos = transform * offset
      const Vector3fP offset{
          drjit::load<FloatP>(&constraints_->offsetX[constraintOffsetIndex]),
          drjit::load<FloatP>(&constraints_->offsetY[constraintOffsetIndex]),
          drjit::load<FloatP>(&constraints_->offsetZ[constraintOffsetIndex])};
      const Vector3fP pos_world = jointState.transformation * offset;

      // Calculate distance of point to target: dist = pos - target
      const Vector3fP target{
          drjit::load<FloatP>(&constraints_->targetX[constraintOffsetIndex]),
          drjit::load<FloatP>(&constraints_->targetY[constraintOffsetIndex]),
          drjit::load<FloatP>(&constraints_->targetZ[constraintOffsetIndex])};
      const Vector3fP diff = pos_world - target;

      // Calculate square norm: dist = diff.squaredNorm()
      const FloatP dist = dot(diff, diff);

      // Calculate error as squared distance: err = constraintWeight * dist
      const FloatP constraintWeight =
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
          const Vector3fP pos_world = jointState_cons.transformation * offset;

          // Calculate distance of point to target: dist = pos - target
          const Vector3fP target{
              drjit::load<FloatP>(&constraints_->targetX[constraintOffsetIndex]),
              drjit::load<FloatP>(&constraints_->targetY[constraintOffsetIndex]),
              drjit::load<FloatP>(&constraints_->targetZ[constraintOffsetIndex])};
          const Vector3fP diff = pos_world - target;

          // Calculate square norm: dist = diff.squaredNorm()
          const FloatP dist = dot(diff, diff);

          // Calculate error as squared distance: err = constraintWeight * dist
          const FloatP constraintWeight =
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
                  gradient[parameterTransform_.transform.innerIndexPtr()[ptIndex]] +=
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
                  gradient[parameterTransform_.transform.innerIndexPtr()[ptIndex]] +=
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

  // Sum the SIMD error register for final result
  const double error =
      kPositionWeight * weight_ * std::accumulate(jointErrors.begin(), jointErrors.end(), 0.0);
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

double SimdPositionErrorFunction::getJacobian(
    const ModelParameters& /* params */,
    const SkeletonState& state,
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
          const Vector3fP pos_world = jointState_cons.transformation * offset;

          // Calculate distance of point to target: dist = pos - target
          const Vector3fP target{
              drjit::load<FloatP>(&constraints_->targetX[constraintOffsetIndex]),
              drjit::load<FloatP>(&constraints_->targetY[constraintOffsetIndex]),
              drjit::load<FloatP>(&constraints_->targetZ[constraintOffsetIndex])};
          const Vector3fP diff = pos_world - target;

          // Calculate square norm: dist = diff.squaredNorm()
          const FloatP dist = dot(diff, diff);

          // Calculate error as squared distance: err = constraintWeight * dist
          const FloatP constraintWeight =
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

#ifdef MOMENTUM_ENABLE_AVX

namespace {

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

} // namespace

SimdPositionErrorFunctionAVX::SimdPositionErrorFunctionAVX(
    const Skeleton& skel,
    const ParameterTransform& pt,
    size_t maxThreads)
    : SimdPositionErrorFunction(skel, pt, maxThreads) {
  // Do nothing
}

SimdPositionErrorFunctionAVX::SimdPositionErrorFunctionAVX(
    const Character& character,
    size_t maxThreads)
    : SimdPositionErrorFunction(character, maxThreads) {
  // Do nothing
}

double SimdPositionErrorFunctionAVX::getError(
    const ModelParameters& /* params */,
    const SkeletonState& state) {
  double error = 0.0;

  if (constraints_ == nullptr) {
    return 0.0f;
  }

  MT_CHECK((size_t)constraints_->numJoints <= state.jointState.size());

  // loop over all joints, as these are our base units
  for (int jointId = 0; jointId < constraints_->numJoints; jointId++) {
    if (jointId >= static_cast<int>(state.jointState.size())) {
      break;
    }
    // pre-load some joint specific values
    const auto& transformation = state.jointState[jointId].transformation;

    __m256 posx;
    __m256 posy;
    __m256 posz;

    const auto jointOffset = jointId * SimdPositionConstraints::kMaxConstraints;

    // loop over all constraints in increments of 8
    const uint32_t count = constraints_->constraintCount[jointId];
    for (uint32_t index = 0; index < count; index += 8) {
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

      // calculate distance of point to target :              diff = (pos - target)
      const __m256 diffx =
          _mm256_sub_ps(posx, _mm256_loadu_ps(&constraints_->targetX[jointOffset + index]));
      const __m256 diffy =
          _mm256_sub_ps(posy, _mm256_loadu_ps(&constraints_->targetY[jointOffset + index]));
      const __m256 diffz =
          _mm256_sub_ps(posz, _mm256_loadu_ps(&constraints_->targetZ[jointOffset + index]));

      // calculate square norm :                              dist = diff.squaredNorm()
      const __m256 dist =
          _mm256_fmadd_ps(diffz, diffz, _mm256_fmadd_ps(diffy, diffy, _mm256_mul_ps(diffx, diffx)));

      // calculate error as squared distance :                err = weight * dist
      const __m256 wgt = _mm256_loadu_ps(&constraints_->weights[jointOffset + index]);
      const __m256 err = _mm256_mul_ps(wgt, dist);

      // sum up the values and add to error
      error += sum8(err);
    }
  }

  // return result
  return static_cast<float>(error * kPositionWeight * weight_);
}

double SimdPositionErrorFunctionAVX::getGradient(
    const ModelParameters& /* params */,
    const SkeletonState& state,
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

        // pre-load some joint specific values
        const auto& transformation = state.jointState[jointId].transformation;

        __m256 posx;
        __m256 posy;
        __m256 posz;

        const auto jointOffset = jointId * SimdPositionConstraints::kMaxConstraints;

        // loop over all constraints in increments of 8
        const uint32_t count = constraints_->constraintCount[jointId];
        for (uint32_t index = 0; index < count; index += 8) {
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

          // calculate distance of point to target :              diff = (pos - target)
          const __m256 diffx =
              _mm256_sub_ps(posx, _mm256_loadu_ps(&constraints_->targetX[jointOffset + index]));
          const __m256 diffy =
              _mm256_sub_ps(posy, _mm256_loadu_ps(&constraints_->targetY[jointOffset + index]));
          const __m256 diffz =
              _mm256_sub_ps(posz, _mm256_loadu_ps(&constraints_->targetZ[jointOffset + index]));

          // calculate square norm :                              dist = diff.squaredNorm()
          const __m256 dist = _mm256_fmadd_ps(
              diffz, diffz, _mm256_fmadd_ps(diffy, diffy, _mm256_mul_ps(diffx, diffx)));

          // calculate combined element weight                    wgt = weight
          const __m256 wgt = _mm256_loadu_ps(&constraints_->weights[jointOffset + index]);

          // calculate error as squared distance :                err = weight * dist
          const __m256 err = _mm256_mul_ps(wgt, dist);

          // sum up the values and add to error
          error_local += sum8(err);

          // pre-calculate values for the gradient :              wgt = weight * 2.0 *
          // kPositionWeight;
          constexpr float kWeight = 2.0 * kPositionWeight;
          const __m256 wt = _mm256_mul_ps(
              _mm256_set_ps(kWeight, kWeight, kWeight, kWeight, kWeight, kWeight, kWeight, kWeight),
              wgt);

          // loop over all joints the constraint is attached to and calculate gradient
          size_t jointIndex = jointId;
          while (jointIndex != kInvalidIndex) {
            // check for valid index
            MT_CHECK(jointIndex < static_cast<size_t>(constraints_->numJoints));

            const auto& jointState = state.jointState[jointIndex];
            const size_t paramIndex = jointIndex * kParametersPerJoint;

            // calculate difference between constraint position and joint center : posd = pos -
            // jointState.translation
            const __m256 posdx = _mm256_sub_ps(posx, _mm256_broadcast_ss(&jointState.x()));
            const __m256 posdy = _mm256_sub_ps(posy, _mm256_broadcast_ss(&jointState.y()));
            const __m256 posdz = _mm256_sub_ps(posz, _mm256_broadcast_ss(&jointState.z()));

            // calculate derivatives based on active joints
            for (size_t d = 0; d < 3; d++) {
              if (activeJointParams_[paramIndex + d]) {
                // calculate gradient with respect to joint : grad = diff.dot(axis) * wgt;
                const Vector3f& axis = jointState.translationAxis.col(d);
                __m256 res = _mm256_mul_ps(diffx, _mm256_broadcast_ss(&axis.x()));
                res = _mm256_fmadd_ps(diffy, _mm256_broadcast_ss(&axis.y()), res);
                res = _mm256_fmadd_ps(diffz, _mm256_broadcast_ss(&axis.z()), res);
                res = _mm256_mul_ps(res, wt);
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
                // calculate gradient with respect to joint : grad = diff.dot(axis.cross(posd)) *
                // wgt;
                const Vector3f& axis = jointState.rotationAxis.col(d);
                const __m256 axisx = _mm256_broadcast_ss(&axis.x());
                const __m256 axisy = _mm256_broadcast_ss(&axis.y());
                const __m256 axisz = _mm256_broadcast_ss(&axis.z());
                const __m256 crossx = _mm256_fmsub_ps(axisy, posdz, _mm256_mul_ps(axisz, posdy));
                const __m256 crossy = _mm256_fmsub_ps(axisz, posdx, _mm256_mul_ps(axisx, posdz));
                const __m256 crossz = _mm256_fmsub_ps(axisx, posdy, _mm256_mul_ps(axisy, posdx));
                __m256 res = _mm256_mul_ps(diffx, crossx);
                res = _mm256_fmadd_ps(diffy, crossy, res);
                res = _mm256_fmadd_ps(diffz, crossz, res);
                res = _mm256_mul_ps(res, wt);
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
              // calculate gradient with respect to joint : grad = diff.dot(posd * LN2) * wgt;
              __m256 res = _mm256_mul_ps(diffx, posdx);
              res = _mm256_fmadd_ps(diffy, posdy, res);
              res = _mm256_fmadd_ps(diffz, posdz, res);
              res = _mm256_mul_ps(res, wt);
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

  return static_cast<float>(error * kPositionWeight * weight_);
}

double SimdPositionErrorFunctionAVX::getJacobian(
    const ModelParameters& /* params */,
    const SkeletonState& state,
    Ref<MatrixXf> jacobian,
    Ref<VectorXf> residual,
    int& usedRows) {
  if (constraints_ == nullptr) {
    return 0.0f;
  }

  MT_CHECK(jacobian.cols() == static_cast<Eigen::Index>(parameterTransform_.transform.cols()));

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

        const auto jointOffset = jointId * SimdPositionConstraints::kMaxConstraints;

        // loop over all constraints in increments of kAvxPacketSize
        const uint32_t count = constraints_->constraintCount[jointId];
        for (uint32_t index = 0, jindex = 0; index < count;
             index += kAvxPacketSize, jindex += kAvxPacketSize * kConstraintDim) {
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

          // calculate distance of point to target :              diff = (pos - target)
          const __m256 diffx =
              _mm256_sub_ps(posx, _mm256_loadu_ps(&constraints_->targetX[jointOffset + index]));
          const __m256 diffy =
              _mm256_sub_ps(posy, _mm256_loadu_ps(&constraints_->targetY[jointOffset + index]));
          const __m256 diffz =
              _mm256_sub_ps(posz, _mm256_loadu_ps(&constraints_->targetZ[jointOffset + index]));

          // calculate square-root of weight :                    wgt = sqrt(kPositionWeight *
          // weight
          // * constraintWeight)
          const float w = kPositionWeight * weight_;
          const __m256 wt = _mm256_mul_ps(
              _mm256_broadcast_ss(&w),
              _mm256_loadu_ps(&constraints_->weights[jointOffset + index]));
          const __m256 wgt = _mm256_sqrt_ps(wt);

          // calculate residual :                                 res = wgt * diff
          const __m256 resx = _mm256_mul_ps(wgt, diffx);
          const __m256 resy = _mm256_mul_ps(wgt, diffy);
          const __m256 resz = _mm256_mul_ps(wgt, diffz);

          // calculate error as squared residual :                err = res.squaredNorm()
          const __m256 err =
              _mm256_fmadd_ps(resz, resz, _mm256_fmadd_ps(resy, resy, _mm256_mul_ps(resx, resx)));

          // sum up the values and add to error
          error_local += sum8(err);

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
                // calculate jacobian with respect to joint: jac = axis * wgt;
                const Vector3f& axis = jointState.translationAxis.col(d);
                const __m256 jacx = _mm256_mul_ps(_mm256_broadcast_ss(&axis.x()), wgt);
                const __m256 jacy = _mm256_mul_ps(_mm256_broadcast_ss(&axis.y()), wgt);
                const __m256 jacz = _mm256_mul_ps(_mm256_broadcast_ss(&axis.z()), wgt);

                // explicitly multiply with the parameter transform to generate parameter space
                // jacobians
                for (auto pIndex = parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
                     pIndex < parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
                     ++pIndex) {
                  const float pw = parameterTransform_.transform.valuePtr()[pIndex];

                  const __m256 pjacx = _mm256_mul_ps(_mm256_broadcast_ss(&pw), jacx);
                  auto* const addressx = &jacobian(
                      offset + jindex + kAvxPacketSize * 0,
                      parameterTransform_.transform.innerIndexPtr()[pIndex]);
                  const __m256 prevx = _mm256_loadu_ps(addressx);
                  _mm256_storeu_ps(addressx, _mm256_add_ps(prevx, pjacx));

                  const __m256 pjacy = _mm256_mul_ps(_mm256_broadcast_ss(&pw), jacy);
                  auto* const addressy = &jacobian(
                      offset + jindex + kAvxPacketSize * 1,
                      parameterTransform_.transform.innerIndexPtr()[pIndex]);
                  const __m256 prevy = _mm256_loadu_ps(addressy);
                  _mm256_storeu_ps(addressy, _mm256_add_ps(prevy, pjacy));

                  const __m256 pjacz = _mm256_mul_ps(_mm256_broadcast_ss(&pw), jacz);
                  auto* const addressz = &jacobian(
                      offset + jindex + kAvxPacketSize * 2,
                      parameterTransform_.transform.innerIndexPtr()[pIndex]);
                  const __m256 prevz = _mm256_loadu_ps(addressz);
                  _mm256_storeu_ps(addressz, _mm256_add_ps(prevz, pjacz));
                }
              }
              if (activeJointParams_[paramIndex + 3 + d]) {
                // calculate jacobian with respect to joint: jac = axis.cross(posd) * wgt;
                const Vector3f& axis = jointState.rotationAxis.col(d);
                const __m256 axisx = _mm256_broadcast_ss(&axis.x());
                const __m256 axisy = _mm256_broadcast_ss(&axis.y());
                const __m256 axisz = _mm256_broadcast_ss(&axis.z());
                const __m256 jacx =
                    _mm256_mul_ps(_mm256_fmsub_ps(axisy, posdz, _mm256_mul_ps(axisz, posdy)), wgt);
                const __m256 jacy =
                    _mm256_mul_ps(_mm256_fmsub_ps(axisz, posdx, _mm256_mul_ps(axisx, posdz)), wgt);
                const __m256 jacz =
                    _mm256_mul_ps(_mm256_fmsub_ps(axisx, posdy, _mm256_mul_ps(axisy, posdx)), wgt);

                // explicitly multiply with the parameter transform to generate parameter space
                // jacobians
                for (auto pIndex =
                         parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 3];
                     pIndex < parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 3 + 1];
                     ++pIndex) {
                  const float pw = parameterTransform_.transform.valuePtr()[pIndex];

                  const __m256 pjacx = _mm256_mul_ps(_mm256_broadcast_ss(&pw), jacx);
                  auto* const addressx = &jacobian(
                      offset + jindex + kAvxPacketSize * 0,
                      parameterTransform_.transform.innerIndexPtr()[pIndex]);
                  const __m256 prevx = _mm256_loadu_ps(addressx);
                  _mm256_storeu_ps(addressx, _mm256_add_ps(prevx, pjacx));

                  const __m256 pjacy = _mm256_mul_ps(_mm256_broadcast_ss(&pw), jacy);
                  auto* const addressy = &jacobian(
                      offset + jindex + kAvxPacketSize * 1,
                      parameterTransform_.transform.innerIndexPtr()[pIndex]);
                  const __m256 prevy = _mm256_loadu_ps(addressy);
                  _mm256_storeu_ps(addressy, _mm256_add_ps(prevy, pjacy));

                  const __m256 pjacz = _mm256_mul_ps(_mm256_broadcast_ss(&pw), jacz);
                  auto* const addressz = &jacobian(
                      offset + jindex + kAvxPacketSize * 2,
                      parameterTransform_.transform.innerIndexPtr()[pIndex]);
                  const __m256 prevz = _mm256_loadu_ps(addressz);
                  _mm256_storeu_ps(addressz, _mm256_add_ps(prevz, pjacz));
                }
              }
            }
            if (activeJointParams_[paramIndex + 6]) {
              // calculate jacobian with respect to joint: jac = (posd * LN2) * wgt;
              const float l = ln2<float>();

              const __m256 jacx = _mm256_mul_ps(_mm256_mul_ps(posdx, wgt), _mm256_broadcast_ss(&l));
              const __m256 jacy = _mm256_mul_ps(_mm256_mul_ps(posdy, wgt), _mm256_broadcast_ss(&l));
              const __m256 jacz = _mm256_mul_ps(_mm256_mul_ps(posdz, wgt), _mm256_broadcast_ss(&l));

              // explicitly multiply with the parameter transform to generate parameter space
              // jacobians
              for (auto pIndex = parameterTransform_.transform.outerIndexPtr()[paramIndex + 6];
                   pIndex < parameterTransform_.transform.outerIndexPtr()[paramIndex + 6 + 1];
                   ++pIndex) {
                const float pw = parameterTransform_.transform.valuePtr()[pIndex];

                const __m256 pjacx = _mm256_mul_ps(_mm256_broadcast_ss(&pw), jacx);
                auto* const addressx = &jacobian(
                    offset + jindex + kAvxPacketSize * 0,
                    parameterTransform_.transform.innerIndexPtr()[pIndex]);
                const __m256 prevx = _mm256_loadu_ps(addressx);
                _mm256_storeu_ps(addressx, _mm256_add_ps(prevx, pjacx));

                const __m256 pjacy = _mm256_mul_ps(_mm256_broadcast_ss(&pw), jacy);
                auto* const addressy = &jacobian(
                    offset + jindex + kAvxPacketSize * 1,
                    parameterTransform_.transform.innerIndexPtr()[pIndex]);
                const __m256 prevy = _mm256_loadu_ps(addressy);
                _mm256_storeu_ps(addressy, _mm256_add_ps(prevy, pjacy));

                const __m256 pjacz = _mm256_mul_ps(_mm256_broadcast_ss(&pw), jacz);
                auto* const addressz = &jacobian(
                    offset + jindex + kAvxPacketSize * 2,
                    parameterTransform_.transform.innerIndexPtr()[pIndex]);
                const __m256 prevz = _mm256_loadu_ps(addressz);
                _mm256_storeu_ps(addressz, _mm256_add_ps(prevz, pjacz));
              }
            }

            // go to the next joint
            jointIndex = skeleton_.joints[jointIndex].parent;
          }

          // store the residual
          _mm256_storeu_ps(&residual(offset + jindex + kAvxPacketSize * 0), resx);
          _mm256_storeu_ps(&residual(offset + jindex + kAvxPacketSize * 1), resy);
          _mm256_storeu_ps(&residual(offset + jindex + kAvxPacketSize * 2), resz);
        }
      },
      dispensoOptions);

  const double error = std::accumulate(ets_error.begin(), ets_error.end(), 0.0);

  usedRows = gsl::narrow_cast<int>(jacobian.rows());

  // sum the AVX error register for final result
  return static_cast<float>(error);
}

size_t SimdPositionErrorFunctionAVX::getJacobianSize() const {
  if (constraints_ == nullptr) {
    return 0;
  }

  jacobianOffset_.resize(constraints_->numJoints);
  size_t num = 0;
  for (int i = 0; i < constraints_->numJoints; ++i) {
    jacobianOffset_[i] = num;
    const size_t count = constraints_->constraintCount[i];
    const size_t residual = count % kAvxPacketSize;
    if (residual != 0) {
      num += (count + kAvxPacketSize - residual) * kConstraintDim;
    } else {
      num += count * kConstraintDim;
    }
  }
  return num;
}

#endif // MOMENTUM_ENABLE_AVX

} // namespace momentum
