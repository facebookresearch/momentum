/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/parameter_transform.h>
#include <momentum/simd/simd.h>

#include <Eigen/Core>

#ifdef MOMENTUM_ENABLE_AVX
#include <immintrin.h>
#endif

#ifndef __vectorcall
#define __vectorcall
#endif

namespace momentum {

/// Accumulates a Jacobian column block (one packet wide) into the model-parameter Jacobian by
/// expanding through the parameter transform sparse matrix.
///
/// The SIMD error functions compute Jacobian entries in joint-parameter space, then must spread
/// each entry into the (sparse) model-parameter columns it influences. This helper isolates that
/// spread, which is otherwise duplicated across every SIMD error function.
///
/// @param jacobian_jointParams The packet of Jacobian values w.r.t. one joint parameter.
/// @param iJointParam Index of the joint parameter producing the gradient.
/// @param parameterTransform_ The character's parameter transform (joint -> model parameters).
/// @param jacobian Output: model-parameter Jacobian rows (one packet wide).
// Naming matches the legacy per-file copies for drop-in replacement; should be renamed to
// camelCase per the project style guide in a follow-up cleanup.
__vectorcall DRJIT_INLINE void jacobian_jointParams_to_modelParams(
    const FloatP& jacobian_jointParams,
    const Eigen::Index iJointParam,
    const ParameterTransform& parameterTransform_,
    Eigen::Ref<Eigen::MatrixXf> jacobian) {
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

#ifdef MOMENTUM_ENABLE_AVX

/// Horizontal sum of 8 floats in an AVX-256 register, accumulating in double precision.
///
/// The conversion to double prevents the catastrophic cancellation that can occur when summing
/// many small squared residuals at single precision.
inline double __vectorcall sum8(const __m256 x) {
  const __m128 high = _mm256_extractf128_ps(x, 1);
  const __m128 low = _mm256_castps256_ps128(x);
  const __m256d val = _mm256_add_pd(_mm256_cvtps_pd(high), _mm256_cvtps_pd(low));
  const __m128d valupper = _mm256_extractf128_pd(val, 1);
  const __m128d vallower = _mm256_castpd256_pd128(val);
  _mm256_zeroupper();
  const __m128d valval = _mm_add_pd(valupper, vallower);
  const __m128d res = _mm_add_pd(_mm_permute_pd(valval, 1), valval);
  return _mm_cvtsd_f64(res);
}

#endif // MOMENTUM_ENABLE_AVX

} // namespace momentum
